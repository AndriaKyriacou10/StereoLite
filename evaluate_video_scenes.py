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

def compute_tepe(disp_t, disp_t1, flow):
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

    tepe_map = torch.abs(disp_t - warped_disp_t1)
    tepe_masked = tepe_map.squeeze()[valid]
    tepe_scalar = tepe_masked.mean().item() if valid.any() else float('nan')
    
    return tepe_scalar
    
@torch.no_grad()
def validate_video_scenes(model):
    dataset = datasets.SceneFlowVideo(root_dir='./data/datasets/SceneFlow', mode='TEST', subsets=['flyingthings'])
    scene_results = []
    for scene_idx in range(len(dataset)):
        scene_dict = dataset[scene_idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames, flow_frames = scene_dict['scene_id'], scene_dict['left'], scene_dict['right'], scene_dict['disp'], scene_dict['valid'], scene_dict['flow'] 

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
            epe_list.append(epe_t)
        
        tepe_list = []
        for t in range(len(disp_predictions) - 1):
            flow_t = flow_frames[t] # flow from t -> t+1 frame
            disp_t = disp_predictions[t] # disparity of scene at frame t
            disp_t1 = disp_predictions[t+1] # disparity of scene at frame t+1
            
            tepe_t = compute_tepe(disp_t, disp_t1, flow_t)
            tepe_list.append(tepe_t)
        
        scene_results.append({
            'scene_id': scene_id,
            'spatial_epes': epe_list,
            'temporal_epes': tepe_list,
            'spatial_epe_mean': float(np.mean(epe_list)),
            'temporal_epe_mean': float(np.mean(tepe_list)) if tepe_list else None,
        })
    
    scene_means = [s['spatial_epe_mean'] for s in scene_results]
    overall_spatial_epe_mom = float(np.mean(scene_means))        # mean of means

    all_spatial = [e for s in scene_results for e in s['spatial_epes']]
    overall_spatial_epe_pooled = float(np.mean(all_spatial))     # pooled
    
    scene_tepe = [s['temporal_epe_mean'] for s in scene_results if s['temporal_epe_mean'] is not None]
    overall_temporal_epe_mom = float(np.mean(scene_tepe))
    
    all_temporal = [e for s in scene_results for e in s['temporal_epes']]
    overall_temporal_epe_pooled = float(np.mean(all_temporal))

    print(f"Spatial EPE (mean-of-scene-means): {overall_spatial_epe_mom:.4f}")
    print(f"Spatial EPE (pooled): {overall_spatial_epe_pooled:.4f}")
    print(f"Temporal EPE (mean-of-scene-means): {overall_temporal_epe_mom:.4f}")
    print(f"Temporal EPE (pooled): {overall_temporal_epe_pooled:.4f}")

    return {'per_scene': scene_results, 'spatial_epe': overall_spatial_epe_pooled, 'temporal_epe': overall_temporal_epe_pooled}
            
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
    
    dataset = datasets.SceneFlowVideo(root_dir='./data/datasets/SceneFlow', mode='TEST', subsets=['flyingthings'])
    scene = dataset[0]
    print(scene['scene_id'], len(scene['left']), len(scene['flow']))
    
    #validate_video_scenes(model)