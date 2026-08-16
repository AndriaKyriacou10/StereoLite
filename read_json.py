import json
import pandas as pd
import math
import statistics

import core.stereo_datasets as datasets
import numpy as np

# with open('./eval_results_video/per_scene_results_LAS_stabilizer_sintel_clean_TEST1.json') as f:
#     data = json.load(f)

# vals = [d['r_raw'] for d in data
#         if d['n_pairwise'] > 0 and not math.isnan(d['r_raw'])]

# print(f"scenes used: {len(vals)} / {len(data)}")
# print(f"median r_raw: {statistics.median(vals):.4f}")
# print(f"mean   r_raw: {statistics.mean(vals):.4f}")
# print(f"IQR:    {statistics.quantiles(vals, n=4)[0]:.4f} – "
#       f"{statistics.quantiles(vals, n=4)[2]:.4f}")

las_pth = 'LAS/per_scene_results_LAS_stabilizer_things.csv'
raft_pth = 'Raft_Stereo/per_scene_results_raftstereo_stabilizer_things.csv'
df = pd.read_csv(f'./eval_results_video/{raft_pth}')

group_idx = {'A':0, 'B':150, 'C':300}
def scene_id_to_idx(scene_id):
    group = scene_id.split('/')[0]
    video = scene_id.split('/')[1]
    return group_idx[group] + int(video)

def to_np(t):
    return t.squeeze().cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)

df["delta_epe"] = df["epe_stabilized"] - df["epe_raw"]
df["delta_tepe"] = df["tepe_stabilized"] - df["tepe_raw"]

corr = df["delta_epe"].corr(df["delta_tepe"])  # Pearson by default
print(corr)

outliers = df.nlargest(10, "delta_tepe")["scene_id"]
df_no_outliers = df[~df["scene_id"].isin(outliers)]
print(df_no_outliers["delta_epe"].corr(df_no_outliers["delta_tepe"]))

# 2. Rank-based version — robust to a few extreme magnitudes
print(df["delta_epe"].corr(df["delta_tepe"], method="spearman"))

def scene_disp_range(disp_gt_frames, valid_frames):
    ranges = []
    for disp, valid in zip(disp_gt_frames, valid_frames):
        d, v = to_np(disp), to_np(valid).astype(bool)
        vals = d[v]
        if vals.size > 0:
            ranges.append(np.percentile(vals, 95) - np.percentile(vals, 5))
    return float(np.mean(ranges)) if ranges else float('nan')

def lr_occlusion_mask(disp):
    H, W = disp.shape
    occ = np.zeros((H, W), dtype=bool)
    xs = np.arange(W)
    for y in range(H):
        d = disp[y]
        tx = np.round(xs - d).astype(int)
        in_frame = (tx >= 0) & (tx < W)
        occ[y, ~in_frame] = True
        winner = np.full(W, -1.0)
        np.maximum.at(winner, tx[in_frame], d[in_frame])
        occ[y, in_frame] = d[in_frame] < winner[tx[in_frame]]
    return occ

def scene_occlusion_pct(disp_gt_frames, valid_frames):
    pcts = []
    for disp, valid in zip(disp_gt_frames, valid_frames):
        d, v = to_np(disp), to_np(valid).astype(bool)
        occ = lr_occlusion_mask(d)
        pcts.append(occ[v].mean() if v.any() else np.nan)
    return float(np.nanmean(pcts)) if pcts else float('nan')

def scene_motion_magnitude(disp_gt_frames, valid_frames):
    deltas = []
    for t in range(len(disp_gt_frames) - 1):
        d0, v0 = to_np(disp_gt_frames[t]), to_np(valid_frames[t]).astype(bool)
        d1, v1 = to_np(disp_gt_frames[t+1]), to_np(valid_frames[t+1]).astype(bool)
        v = v0 & v1
        if v.any():
            deltas.append(np.abs((d0 - d1)[v]).mean())
    return float(np.mean(deltas)) if deltas else float('nan')

flyingthings = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'], max_disp=192)
rows = []
scene_to_idx = {flyingthings[i]['scene_id']: i for i in range(len(flyingthings))}
for scene_id in df["scene_id"]:
    idx = scene_to_idx[scene_id]
    print(f"{scene_id} idx:{idx}")
    data = flyingthings[idx]
    disp_gt_frames, valid_frames = data['disp'], data['valid']
    rows.append({
        'scene_id': scene_id,
        'disp_range': scene_disp_range(disp_gt_frames, valid_frames),
        'occlusion_pct': scene_occlusion_pct(disp_gt_frames, valid_frames),
        'motion_magnitude': scene_motion_magnitude(disp_gt_frames, valid_frames),
    })

features_df = pd.DataFrame(rows)
df = df.merge(features_df, on='scene_id')
df.sort_values(by='delta_tepe', ascending=False, inplace=True)
df.to_csv('./eval_results_video/Raft_Stereo/stb_things.csv', index=False)