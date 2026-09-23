import pandas as pd
import numpy as np

def _length_weighted_mean(values, lengths):
    num, denom = 0.0, 0.0
    for v, l in zip(values, lengths):
        if not np.isnan(v):
            num += v * l
            denom += l
    return num / denom if denom > 0 else float('nan')


las_pth_sintel = 'LAS/per_scene_results_LAS_stabilizer_sintel_clean_disp_masked.csv'
las_pth_things = 'LAS/per_scene_results_LAS_stabilizer_things.csv'

raft_pth_things = 'Raft_Stereo/per_scene_results_raftstereo_stabilizer_things.csv'
raft_pth_sintel = 'Raft_Stereo/per_scene_results_raftstereo_stabilizer_sintel_clean.csv'

df = pd.read_csv(f'./eval_results_video/{raft_pth_things}')


df['delta_tepe'] = df['tepe_stabilized'] - df['tepe_raw']   # >0 = stabilizer hurt this scene
df['delta_epe']  = df['epe_stabilized']  - df['epe_raw']

# contribution to the length-weighted pooled mean (matches your aggregate_eval_results logic)
df['contrib_tepe'] = df['delta_tepe'] * df['seq_length']
total_pooled_delta = df['contrib_tepe'].sum() / df['seq_length'].sum()
# sanity check: this should equal pooled_tepe_stb - pooled_tepe_raw from your overall_results.json

# rank scenes by how much they push the pooled mean
outliers = df.sort_values('contrib_tepe', ascending=False).reset_index(drop=True)
print(outliers[['scene_id', 'seq_length', 'delta_tepe', 'contrib_tepe']].head(15))

# fraction of total degradation coming from top-K scenes
topk = outliers.head(10)
print(f"Top 10 scenes account for {topk['contrib_tepe'].sum() / df[df['contrib_tepe']>0]['contrib_tepe'].sum():.1%} of total positive contribution")

for k in [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    trimmed = outliers.iloc[k:]
    pooled_raw = _length_weighted_mean(trimmed['tepe_raw'].tolist(), trimmed['seq_length'].tolist())
    pooled_stb = _length_weighted_mean(trimmed['tepe_stabilized'].tolist(), trimmed['seq_length'].tolist())
    delta = pooled_stb - pooled_raw
    verdict = 'DEGRADES' if delta > 0 else 'IMPROVES'
    print(f"k={k:>3} | scenes left={len(trimmed):>3} | pooled_raw={pooled_raw:.4f} | pooled_stb={pooled_stb:.4f} | delta={delta:+.4f} | {verdict}")
    
    
'''What percentage of scenes individually improved from stabilization'''
valid = df.dropna(subset=['delta_tepe'])

print(len(valid), "valid scenes")

print(f"Scenes that improved: {(valid['delta_tepe'] < 0).mean():.1%}")

print(f"NaN scenes: {df['delta_tepe'].isna().sum()} ")

print(df[df['delta_tepe'].isna()]['scene_id'])