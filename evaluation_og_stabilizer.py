from ast import arg
from collections import defaultdict
import os
import argparse
import time
import logging
import numpy as np
from sympy import fraction
import torch
import torch.nn.functional as F
from tqdm import tqdm
from core.liteanystereo import original_LAS
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime


from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer


def load_models(ckpt_model, ckpt_stb, device):
    stereo_model = original_LAS()
    checkpoint = torch.load(ckpt_model, map_location=device)
    target_model = stereo_model.module if hasattr(stereo_model, 'module') else stereo_model
    target_model.load_state_dict(checkpoint, strict=True)
    stereo_model.to(device)
    stereo_model.eval()
    for p in stereo_model.parameters():
        p.requires_grad_(False)

    stb_model = BiDAStabilizer()
    stb_checkpoint = torch.load(ckpt_stb, map_location=device)
    stb_model.load_state_dict(stb_checkpoint, strict=True)
    stb_model.to(device)
    stb_model.eval()
    for p in stb_model.parameters():
        p.requires_grad_(False)

    return stereo_model, stb_model


def run_las_on_video(model, device, left_imgs, right_imgs):    
    disp_predictions = []
    
    for frame in range(len(left_imgs)):
        img1 = left_imgs[frame].unsqueeze(0).to(device)
        img2 = right_imgs[frame].unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        
        with torch.no_grad():
            disp_pred = model(img1, img2, test_mode=True)
        
        disp_pred = padder.unpad(disp_pred.float()).squeeze(0)
        disp_predictions.append(disp_pred)
    return disp_predictions

def run_stabilizer_on_video(stabilizer:BiDAStabilizer, left_imgs, disp_predictions, device, kernel_size=50):
    video = torch.stack(left_imgs, dim=0).to(device) # T C H W
    disp_predictions = torch.stack(disp_predictions, dim=0).to(device) # T 1 H W
    
    with torch.no_grad():
        disp_stabilized = stabilizer.forward_batch(video, -disp_predictions, kernel_size).squeeze(1)
        # logging.info(f"disp_stabilized min: {disp_stabilized.min()}, max: {disp_stabilized.max()}")
    return disp_stabilized.abs()

def compute_epe(disp, disp_gt, valid_mask):
    error = (valid_mask * (disp - disp_gt)**2).sum(dim=0).sqrt()
    
    nonzero = torch.count_nonzero(error)
        
    return (error.sum(), nonzero)

def compute_tepe(disp_t, disp_t1, disp_gt_t, disp_gt_t1, valid_t, valid_t1):
    delta_mask = valid_t * valid_t1
    disp_delta = disp_t - disp_t1
    gt_delta = disp_gt_t - disp_gt_t1
    
    error = (delta_mask * (disp_delta - gt_delta)**2).sum(dim=0).sqrt()
    
    nonzero = torch.count_nonzero(error)
    
    return (error.sum(), nonzero)

def measure_metrics_per_scene(disp_raw, disp_stabilized, disp_gt, valid_mask, device):
    
    def _get_value(error, nonzero, clamp_thr=1e-5):
        ''' Return the tepe/epe value per scene, if nonzero is 0, return nan '''
        if nonzero == 0:
            return float('nan')
        else:
            return (error / torch.clamp(nonzero, min=clamp_thr)).item()
        
    disp_gt = [d.to(device) for d in disp_gt]
    valid_mask = [v.unsqueeze(0).to(device) for v in valid_mask] # H W -> 1 H W
    
    # total_tepe['stb'] : sum the error of all frames to get a SCENE-LEVEL error
    
    total_tepe = {'stb': 0, 'raw': 0, 'nonzero_stb':0, 'nonzero_raw':0}
    
    for  t in range(len(disp_raw)-1):
        result = compute_tepe(disp_stabilized[t], disp_stabilized[t+1], disp_gt[t], disp_gt[t+1], valid_mask[t], valid_mask[t+1])
        total_tepe['stb'] += result[0]
        total_tepe['nonzero_stb'] += result[1]
     
        
        tepe_raw = compute_tepe(disp_raw[t], disp_raw[t+1], disp_gt[t], disp_gt[t+1], valid_mask[t], valid_mask[t+1])
        total_tepe['raw'] += tepe_raw[0]
        total_tepe['nonzero_raw'] += tepe_raw[1]
    
    total_epe = {'stb': 0, 'raw': 0, 'nonzero_stb':0, 'nonzero_raw':0}
    
    for t in range(len(disp_raw)):
        epe_stb = compute_epe(disp_stabilized[t], disp_gt[t], valid_mask[t])
        total_epe['stb'] += epe_stb[0]
        total_epe['nonzero_stb'] += epe_stb[1]
        
        epe_raw = compute_epe(disp_raw[t], disp_gt[t], valid_mask[t])
        total_epe['raw'] += epe_raw[0]
        total_epe['nonzero_raw'] += epe_raw[1]

    tepe_stb = _get_value(total_tepe['stb'], total_tepe['nonzero_stb'])
    tepe_raw = _get_value(total_tepe['raw'], total_tepe['nonzero_raw'])
    epe_stb = _get_value(total_epe['stb'], total_epe['nonzero_stb'])
    epe_raw = _get_value(total_epe['raw'], total_epe['nonzero_raw'])
    
    return {
        'tepe_stabilized': tepe_stb,
        'tepe_raw': tepe_raw,
        'epe_stabilized': epe_stb,
        'epe_raw': epe_raw
    }

def evaluate(stereo_model, stb_model, device):
    dataset = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'])
    
    results = []
    
    for scene_idx in range(len(dataset)):
        data = dataset[scene_idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames = data['scene_id'], data['left'], data['right'], data['disp'], data['valid']
        disp_preds = run_las_on_video(stereo_model, device, left_imgs, right_imgs)
        disp_stabilized = run_stabilizer_on_video(stb_model, left_imgs, disp_preds, device)
        
        result_scene = measure_metrics_per_scene(disp_preds, disp_stabilized, disp_gt_frames, valid_frames, device)
        logging.info(f"type: {type(result_scene['epe_raw'])}")
        results.append({
            'scene_id':str(scene_id),
            'epe_stabilized': result_scene['epe_stabilized'],
            'epe_raw': result_scene['epe_raw'],
            'tepe_stabilized': result_scene['tepe_stabilized'],
            'tepe_raw': result_scene['tepe_raw']
        })
        
        
    epe_raw_scenes = [s['epe_raw'] for s in results]
    epe_stb_scenes = [s['epe_stabilized'] for s in results]
    tepe_raw_scenes = [s['tepe_raw'] for s in results]
    tepe_stb_scenes = [s['tepe_stabilized'] for s in results]

    n_nan_epe_raw = sum(np.isnan(v) for v in epe_raw_scenes)
    n_nan_tepe_raw = sum(np.isnan(v) for v in tepe_raw_scenes)  
    n_nan_epe_stb = sum(np.isnan(v) for v in epe_stb_scenes)
    n_nan_tepe_stb = sum(np.isnan(v) for v in tepe_stb_scenes)
    
    
    logging.info(f"NaN scenes — epe_raw: {n_nan_epe_raw}/{len(results)}, tepe_raw: {n_nan_tepe_raw}/{len(results)}, epe_stb: {n_nan_epe_stb}/{len(results)}, tepe_stb: {n_nan_tepe_stb}/{len(results)}")
    
    overall = {
        'epe_raw': float(np.nanmean(epe_raw_scenes)),
        'epe_stabilized': float(np.nanmean(epe_stb_scenes)),
        'tepe_raw': float(np.nanmean(tepe_raw_scenes)),
        'tepe_stabilized': float(np.nanmean(tepe_stb_scenes)),
    }
    
    return results, overall



def build_scanline_image(disp_stabilized, left_imgs, disp_raw, row_idx=None, save_path=None):
    ''' Build a scanline image from a sequence of disparity maps '''    
    def _get_scanline(disp_sequence, row_idx):
        if isinstance(disp_sequence, list):
            disp_sequence = torch.stack(disp_sequence, dim=0) # T 1 H W
        pass

        if row_idx is None:
            row_idx = disp_sequence.shape[-2] // 2
        
        scanline = disp_sequence[:, 0, row_idx, :].cpu().numpy() # T W
        return scanline
    
    def _img_scanline(left_imgs, row_idx):
        left_imgs = torch.stack(left_imgs, dim=0) # T C H W
        
        if row_idx is None:
            row_idx = left_imgs.shape[-2] // 2
        
        img_scanline = left_imgs[:, :, row_idx, :].permute(0, 2, 1).cpu().numpy() / 255.0 # T W C
        return img_scanline
    
    scanline_stabilized = _get_scanline(disp_stabilized, row_idx)
    scanline_raw = _get_scanline(disp_raw, row_idx)
    
    left_img = _img_scanline(left_imgs, row_idx)
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 6))
    axes[0].imshow(left_img); axes[0].set_title("Left Frames"); axes[0].axis('off')

    vmax = np.percentile(np.concatenate([scanline_stabilized, scanline_raw]), 99)
    im1 = axes[1].imshow(scanline_stabilized, cmap='inferno', vmin=0, vmax=vmax);  axes[1].set_title('Disparity Stabilized'); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)    
    
    im2 = axes[2].imshow(scanline_raw, cmap='inferno', vmin=0, vmax=vmax);  axes[2].set_title('Disparity Original'); axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)  
      
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_model', help="restore stereo model checkpoint", default='./checkpoints/LiteAnyStereo.pth')
    parser.add_argument('--ckpt_stb', help="restore stabilizer model checkpoint", required=True)
    parser.add_argument('--visualize', action='store_true', help="visualize results")
    parser.add_argument('--output_dir', default='./eval_results_video')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_path = os.path.join(args.output_dir, f'evaluation_{current_time}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stereo_model, stb_model = load_models(args.ckpt_model, args.ckpt_stb, device)
    
    results, overall = evaluate(stereo_model, stb_model, device)

    logging.info("===== Evaluation Results =====")
    for k, v in overall.items():
        logging.info(f"{k:20s}: {v:.4f}")
        
    overall_path = os.path.join(args.output_dir, f'overall_results_{current_time}.json')
    with open(overall_path, 'w') as f:
        json.dump(overall, f, indent=2)

    per_scene_path = os.path.join(args.output_dir, f'per_scene_results_{current_time}.json')
    with open(per_scene_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    csv_path = os.path.join(args.output_dir, f'per_scene_results_{current_time}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # VISUALIZE
    if args.visualize:
        scene_id='A/0120'
        group_idx = {'A':0, 'B':150, 'C':300}
        group = scene_id.split('/')[0]
        video = scene_id.split('/')[1]
        idx = group_idx[group] + int(video)
        
        data = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'])[idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames = data['scene_id'], data['left'], data['right'], data['disp'], data['valid']
        disp_preds = run_las_on_video(stereo_model, device, left_imgs, right_imgs)
        disp_stabilized = run_stabilizer_on_video(stb_model, left_imgs, disp_preds, device)
        
        save_path = os.path.join(args.output_dir, f'visualize_scanline.png')
        build_scanline_image(disp_stabilized, left_imgs, disp_preds, row_idx=None, save_path=save_path)