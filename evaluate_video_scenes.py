import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from core.liteanystereo import original_LAS
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

import matplotlib.animation as animation

def compute_epe(disp_pred, disp_gt, valid_mask):
    disp_pred = disp_pred.squeeze()   # -> (H, W), regardless of leading 1s
    disp_gt = disp_gt.squeeze()       # -> (H, W)
    valid_mask = valid_mask.squeeze().bool()  # -> (H, W)
    
    if valid_mask.sum() == 0:
        return float('nan')  # or skip this frame entirely upstream
    
    disp_pred_valid = disp_pred[valid_mask]
    disp_gt_valid = disp_gt[valid_mask]
    
    abs_epe = torch.abs(disp_pred_valid.float() - disp_gt_valid.float())
    epe = torch.mean(abs_epe).item()
    
    
    return epe

def compute_tepe(disp_t, disp_t1, flow, visualize=False):
    # Ensure disp_t and disp_t1 are (1, 1, H, W)
    while disp_t.dim() < 4:
        disp_t = disp_t.unsqueeze(0)
    while disp_t1.dim() < 4:
        disp_t1 = disp_t1.unsqueeze(0)

    assert disp_t.shape == disp_t1.shape, f"Shape mismatch: {disp_t.shape} vs {disp_t1.shape}"
    
    H ,W = flow.shape[1:]
    
    # Location of each pixel in frame t
    grid_x, grid_y = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # Location of each pixel in frame t+1 according to the flow
    sample_x = grid_x + flow[0]
    sample_y = grid_y + flow[1]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    sample_x_norm = 2.0 * sample_x/(W-1) - 1.0
    sample_y_norm = 2.0 * sample_y/(H-1) - 1.0
    
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1).unsqueeze(0)
    
    # Warp disp_pred_{t+1} back into frame t's coordinate frame
    warped_disp_t1 = F.grid_sample(disp_t1, grid, mode='bilinear', align_corners=True)
    
    # out-of-bounds mask
    valid = (sample_x >= 0) & (sample_x <= W - 1) & (sample_y >= 0) & (sample_y <= H - 1)

    tepe_map = torch.abs(disp_t - warped_disp_t1).squeeze()
    tepe_masked = tepe_map[valid]
    tepe_scalar = tepe_masked.mean().item() if valid.any() else float('nan')
    
    if visualize:
        tepe_map_vis = tepe_map.clone()
        tepe_map_vis[~valid] = float('nan')
        
        warped_vis = warped_disp_t1.squeeze().clone()
        warped_vis[~valid] = float('nan')
        
        return tepe_scalar, {'disp_t': disp_t.squeeze(), 'warped_disp_t1': warped_vis, 'tepe_map': tepe_map_vis, 'valid': valid}
    
    return tepe_scalar
    
@torch.no_grad()
def validate_video_scenes(model, visualize = False, animate_scene_ids=None, out_dir='./visualize_video'):
    animate_scene_ids = animate_scene_ids or []
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = datasets.SceneFlowVideo(root_dir='./data/datasets/SceneFlow', mode='TEST', subsets=['flyingthings'])
    scene_results = []
    
    if visualize:
        logging.info(f"Visualizing results for scenes: {animate_scene_ids}")
        len_data = len(animate_scene_ids)
    else:
        len_data = len(dataset)

    for scene_idx in range(len_data):
        scene_dict = dataset[scene_idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames, flow_frames = scene_dict['scene_id'], scene_dict['left'], scene_dict['right'], scene_dict['disp'], scene_dict['valid'], scene_dict['flow'] 

        print(scene_id)
        break
        disp_predictions = []
        epe_list = []
        for t in range(len(left_imgs)):
            img1 = left_imgs[t].unsqueeze(0).to(device)
            img2 = right_imgs[t].unsqueeze(0).to(device)

            padder = InputPadder(img1.shape, divis_by=32)
            img1, img2 = padder.pad(img1, img2)
            
            disp_pred = model(img1, img2, test_mode=True)
            disp_pred = padder.unpad(disp_pred.float()).squeeze(0).cpu()
            disp_predictions.append(disp_pred)
            
            epe_t =  compute_epe(disp_pred, disp_gt_frames[t], valid_frames[t])
            if np.isnan(epe_t):
                logging.warning(f"Scene {scene_idx} - Frame:{t} : no valid pixels. EPE is NaN")
            epe_list.append(epe_t)
        
        tepe_list = []
        tepe_maps = []
        
        for t in range(len(disp_predictions) - 1):
            flow_t = flow_frames[t] # flow from t -> t+1 frame
            disp_t = disp_predictions[t] # disparity of scene at frame t
            disp_t1 = disp_predictions[t+1] # disparity of scene at frame t+1
            
            if visualize:
                tepe_t, tepe_map = compute_tepe(disp_t, disp_t1, flow_t, visualize)
                tepe_maps.append(tepe_map)
            else:
                tepe_t = compute_tepe(disp_t, disp_t1, flow_t, visualize)
            tepe_list.append(tepe_t)
        
        scene_results.append({
            'scene_id': scene_id,
            'spatial_epes': epe_list,
            'temporal_epes': tepe_list,
            'spatial_epe_mean': float(np.nanmean(epe_list)) if not all(np.isnan(epe_list)) else None,
            'temporal_epe_mean': float(np.mean(tepe_list)) if tepe_list else None,
        })
        
        if visualize:
            if scene_id in animate_scene_ids:
                animate_tepe(
                    disp_predictions, flow_frames,
                    save_path=os.path.join(out_dir, f"{scene_id}_tepe.gif"),
                    scene_id=scene_id
                )

    
    scene_means = [s['spatial_epe_mean'] for s in scene_results if s['spatial_epe_mean'] is not None]
    overall_spatial_epe_mom = float(np.mean(scene_means)) if scene_means else None       # mean of means

    all_spatial = [e for s in scene_results for e in s['spatial_epes']]
    overall_spatial_epe_pooled = float(np.nanmean(all_spatial)) if not all(np.isnan(all_spatial)) else None    # pooled
    
    scene_tepe = [s['temporal_epe_mean'] for s in scene_results if s['temporal_epe_mean'] is not None]
    overall_temporal_epe_mom = float(np.mean(scene_tepe))
    
    all_temporal = [e for s in scene_results for e in s['temporal_epes']]
    overall_temporal_epe_pooled = float(np.mean(all_temporal))

    print(f"Spatial EPE (mean-of-scene-means): {overall_spatial_epe_mom:.4f}")
    print(f"Spatial EPE (pooled): {overall_spatial_epe_pooled:.4f}")
    print(f"Temporal EPE (mean-of-scene-means): {overall_temporal_epe_mom:.4f}")
    print(f"Temporal EPE (pooled): {overall_temporal_epe_pooled:.4f}")

    return {'per_scene': scene_results, 'spatial_epe': overall_spatial_epe_pooled, 'temporal_epe': overall_temporal_epe_pooled}

def animate_tepe(disp_predictions, flow_frames, save_path, scene_id="", fps=5):
    """
    disp_predictions: list of (H, W) or (1,1,H,W) tensors, per-frame disparity predictions for one scene
    flow_frames: list of (2, H, W) tensors, optical flow t -> t+1 (length = len(disp_predictions) - 1)
    """
    n_pairs = len(disp_predictions) - 1
    assert n_pairs == len(flow_frames), (n_pairs, len(flow_frames))

    # Precompute all frame-pair data up front so we can fix color scales
    disp_t_list, warped_list, tepe_list = [], [], []
    for t in range(n_pairs):
        disp_t = disp_predictions[t]
        disp_t1 = disp_predictions[t + 1]
        flow_t = flow_frames[t]

        _, maps = compute_tepe(disp_t, disp_t1, flow_t, visualize=True)
        disp_t_list.append(maps['disp_t'].numpy())
        warped_list.append(maps['warped_disp_t1'].numpy())
        tepe_list.append(maps['tepe_map'].numpy())

    # Fixed scales across the whole scene — critical, don't let matplotlib auto-scale per frame
    disp_stack = np.stack(disp_t_list + warped_list)
    disp_vmin = np.nanpercentile(disp_stack, 1)
    disp_vmax = np.nanpercentile(disp_stack, 99)

    tepe_stack = np.stack(tepe_list)
    tepe_vmax = np.nanpercentile(tepe_stack, 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Disparity @ t", "Disparity @ t+1 (warped to t)", "TEPE = |diff|"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    im0 = axes[0].imshow(disp_t_list[0], cmap='magma', vmin=disp_vmin, vmax=disp_vmax)
    im1 = axes[1].imshow(warped_list[0], cmap='magma', vmin=disp_vmin, vmax=disp_vmax)
    im2 = axes[2].imshow(tepe_list[0], cmap='inferno', vmin=0, vmax=tepe_vmax)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    suptitle = fig.suptitle(f"Scene {scene_id} | frame 0→1")

    def update(i):
        im0.set_data(disp_t_list[i])
        im1.set_data(warped_list[i])
        im2.set_data(tepe_list[i])
        suptitle.set_text(f"Scene {scene_id} | frame {i}→{i+1}")
        return [im0, im1, im2]

    ani = animation.FuncAnimation(fig, update, frames=n_pairs, interval=1000 / fps, blit=False)
    ani.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved animation to {save_path}")
    
def test():
    H, W = 20, 30
    disp_t = torch.rand(1, 1, H, W) * 50
    disp_t1 = torch.rand(1, 1, H, W) * 50
    flow_zero = torch.zeros(2, H, W)   # no displacement anywhere

    tepe = compute_tepe(disp_t, disp_t1, flow_zero)
    expected = torch.abs(disp_t - disp_t1).mean().item()

    print(tepe, expected)   # should match closely (small bilinear-edge differences possible, but very close)
    
    # Synthetic test 2: known constant flow shifts everything by exactly 1 pixel
    disp_t1_shifted = torch.roll(disp_t1, shifts=1, dims=3)  # shift 1 px right
    flow_const = torch.zeros(2, H, W)
    flow_const[0] = 1.0   # u = 1 everywhere

    tepe = compute_tepe(disp_t1, disp_t1_shifted, flow_const)
    print(f"Test 2: tepe = {tepe}")
    
    # after warping back by u=1, warped_disp_t1 should recover disp_t1 almost exactly
    # so tepe should be near 0
    
    # Synthetic test 3: out-of-bounds flow gets correctly excluded
    flow_huge = torch.zeros(2, H, W)
    flow_huge[0, 0, 0] = 1000.0   # sends this one pixel way out of frame

    tepe = compute_tepe(disp_t, disp_t1, flow_huge)
    print(f"Test 3: tepe = {tepe}")
    # should not error, should not silently include a garbage 0-padded value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help="restore checkpoint", default='./checkpoints/LiteAnyStereo.pth')
    parser.add_argument('--visualize', action='store_true', help="visualize results")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = original_LAS()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.ckpt is not None:
        assert args.ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.ckpt, map_location=device)

        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    
    model.to(device)
    model.eval()
    
    # dataset = datasets.SceneFlowVideo(root_dir='./data/datasets/SceneFlow', mode='TEST', subsets=['flyingthings'])
    # print(f"Loaded {len(dataset)} video scenes for evaluation.")
    # scene = dataset[0]
    # print(scene['scene_id'], len(scene['left']), len(scene['flow']))
    
    validate_video_scenes(model, visualize = args.visualize, animate_scene_ids=['A/0006', 'A/0122', 'A/0143', 'B/0000', 'C/0082'])