from ast import arg
from collections import defaultdict
import os
import sys
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
import cv2

from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer


def load_models(name, ckpt_model, ckpt_stb, device):
    if name == "LAS_stabilizer":
        stereo_model = original_LAS(fnet_pretrained=True)
        import bidastabilizer_integration.video_datasets as datasets
        logging.info(f"Stereo model: LAS")
    elif name == "raftstereo_stabilizer":
        from bidastabilizer_integration.models.raft_stereo_model import RAFTStereoModel
        stereo_model = RAFTStereoModel().model
        import bidastabilizer_integration.video_datasets2 as datasets
        logging.info(f"Stereo model: RAFT-Stereo")
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    if ckpt_model is not None:
        # Restore LAS / RAFT-Stereo checkpoint
        assert ckpt_model.endswith(".pth") or ckpt_model.endswith(
            ".pt"
        )
        logging.info("Loading checkpoint...")
        print("Loading checkpoint", ckpt_model)

        strict = True

        state_dict = torch.load(ckpt_model, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        stereo_model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Done loading stero model checkpoint")
    
    # checkpoint = torch.load(ckpt_model, map_location=device)
    # target_model = stereo_model.module if hasattr(stereo_model, 'module') else stereo_model
    # target_model.load_state_dict(checkpoint, strict=True)
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


def run_stereo_model(name, model, device, left_imgs, right_imgs):    
    disp_predictions = []
    
    for frame in range(len(left_imgs)):
        img1 = left_imgs[frame].unsqueeze(0).to(device)
        img2 = right_imgs[frame].unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        
        with torch.no_grad():
            if name == "raftstereo_stabilizer":
                _, disp_pred = model(img1, img2, test_mode=True)
                disp_pred = -disp_pred # convert to positive since GT is positive
            elif name == "LAS_stabilizer":
                disp_pred = model(img1, img2, test_mode=True)
    
        disp_pred = padder.unpad(disp_pred.float()).squeeze(0)
        disp_predictions.append(disp_pred)
    return disp_predictions

def run_stabilizer_on_video(name, stabilizer:BiDAStabilizer, left_imgs, disp_predictions, device = torch.cuda, kernel_size=50):
    video = torch.stack(left_imgs, dim=0).to(device) # T C H W
    
    disp_predictions = torch.stack(disp_predictions, dim=0).to(device) # T 1 H W
    disp_predictions = -disp_predictions # Stabilizer expects negative disparity
    
    logging.info(f"Disparity from {name}: min: {disp_predictions.min()}, max: {disp_predictions.max()}")
    
    with torch.no_grad():
        disp_stabilized = stabilizer.forward_batch(video, disp_predictions, kernel_size).squeeze(1)
        logging.info(f"disp_stabilized min: {disp_stabilized.min()}, max: {disp_stabilized.max()}")
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


def measure_error_persistence(disp_raw, disp_stabilized, disp_gt, valid_mask, device):
    """
    Test 1: is the backbone's error temporally jittery or temporally persistent?

    Because TEPE's integrand is exactly (e_t - e_t+1) where e_t = d_t - gt_t,
    TEPE measures the frame-to-frame CHANGE in error. Comparing it against the
    error magnitude tells us how much of the error is temporally uncorrelated,
    i.e. how much flicker a temporal refiner could possibly remove.

        rho = TEPE / EPE   ~1.4 if errors are independent across frames,
                           -> 0 as errors become perfectly persistent
        r   = corr(e_t, e_t+1)   ~0 = fresh error each frame (jitter)
                                 ~1 = same mistake every frame (systematic)

    Both are computed on the pairwise-valid mask with a valid-pixel denominator,
    so that rho is a self-consistent ratio. This deliberately does NOT reuse
    _get_value(), which divides by count_nonzero(error) and would give EPE and
    TEPE different, data-dependent denominators.
    """
    nan = float('nan')
    keys = ('n', 'sum_absE', 'sum_absD', 'sum_a', 'sum_b',
            'sum_aa', 'sum_bb', 'sum_ab')

    disp_gt = [d.to(device) for d in disp_gt]
    valid_mask = [v.to(device) if v.dim() == 3 else v.unsqueeze(0).to(device)
                  for v in valid_mask]                      # H W -> 1 H W

    stats = {k: dict.fromkeys(keys, 0.0) for k in ('raw', 'stb')}

    for t in range(len(disp_raw) - 1):
        m = (valid_mask[t] * valid_mask[t + 1]).float()
        n = m.sum()
        if n == 0:
            continue

        for key, dmap in (('raw', disp_raw), ('stb', disp_stabilized)):
            e_t  = (dmap[t]     - disp_gt[t]).float()     * m
            e_t1 = (dmap[t + 1] - disp_gt[t + 1]).float() * m
            d    = e_t - e_t1

            s = stats[key]
            s['n']        += n.item()
            s['sum_absE'] += e_t.abs().sum().item()
            s['sum_absD'] += d.abs().sum().item()
            s['sum_a']    += e_t.sum().item()
            s['sum_b']    += e_t1.sum().item()
            s['sum_aa']   += (e_t ** 2).sum().item()
            s['sum_bb']   += (e_t1 ** 2).sum().item()
            s['sum_ab']   += (e_t * e_t1).sum().item()

    def _reduce(s):
        n = s['n']
        if n == 0:
            return {'epe_pairwise': nan, 'tepe_pairwise': nan, 'rho': nan, 'r': nan}
        epe  = s['sum_absE'] / n
        tepe = s['sum_absD'] / n
        mu_a, mu_b = s['sum_a'] / n, s['sum_b'] / n
        var_a = max(s['sum_aa'] / n - mu_a ** 2, 0.0)
        var_b = max(s['sum_bb'] / n - mu_b ** 2, 0.0)
        denom = (var_a * var_b) ** 0.5
        return {
            'epe_pairwise':  epe,
            'tepe_pairwise': tepe,
            'rho': tepe / epe if epe > 1e-12 else nan,
            'r':   (s['sum_ab'] / n - mu_a * mu_b) / denom if denom > 1e-12 else nan,
        }

    out = {'n_pairwise': stats['raw']['n']}
    for key, tag in (('raw', 'raw'), ('stb', 'stb')):
        for name, val in _reduce(stats[key]).items():
            out[f'{name}_{tag}'] = val
    return out

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

def evaluate(name, stereo_model, stb_model, device, dataset_name):
    if dataset_name == 'things':
        max_disp = 192 if name == "LAS_stabilizer" else 192
        dataset = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'], max_disp=max_disp)
    elif dataset_name.split('_')[0] == 'sintel':
        dstype = dataset_name.split('_')[1]
        max_disp = 192 if name == "LAS_stabilizer" else None
        dataset = datasets.SintelStereoVideo(mode='training', dstype=dstype, max_disp = max_disp)
    results = []
    
    def _length_weighted_mean(values, lengths):
        ''' matches repo's aggregate_eval_results: sum(val_i * length_i) / sum(length_i) '''
        num, denom = 0.0, 0.0
        for v, l in zip(values, lengths):
            if not np.isnan(v):
                num += v * l
                denom += l
        return num / denom if denom > 0 else float('nan') 
    
    for scene_idx in range(len(dataset)):
        data = dataset[scene_idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames = data['scene_id'], data['left'], data['right'], data['disp'], data['valid']
        disp_preds = run_stereo_model(name, stereo_model, device, left_imgs, right_imgs)
        disp_stabilized = run_stabilizer_on_video(name, stb_model, left_imgs, disp_preds, device=device)
        
        result_scene = measure_metrics_per_scene(disp_preds, disp_stabilized, disp_gt_frames, valid_frames, device)
        result_scene.update(measure_error_persistence(disp_preds, disp_stabilized, disp_gt_frames, valid_frames, device))
        # logging.info(f"type: {type(result_scene['epe_raw'])}")
        results.append({
            'scene_id':str(scene_id),
            'seq_length': len(disp_preds),
            'epe_stabilized': result_scene['epe_stabilized'],
            'epe_raw': result_scene['epe_raw'],
            'tepe_stabilized': result_scene['tepe_stabilized'],
            'tepe_raw': result_scene['tepe_raw'],
            # Test 1 
            # 'n_pairwise': result_scene['n_pairwise'],
            # 'epe_pairwise_raw': result_scene['epe_pairwise_raw'],
            # 'tepe_pairwise_raw': result_scene['tepe_pairwise_raw'],
            # 'rho_raw': result_scene['rho_raw'],
            # 'r_raw': result_scene['r_raw'],
            # 'epe_pairwise_stb': result_scene['epe_pairwise_stb'],
            # 'tepe_pairwise_stb': result_scene['tepe_pairwise_stb'],
            # 'rho_stb': result_scene['rho_stb'],
            # 'r_stb': result_scene['r_stb'],
        })
        
        
    epe_raw_scenes = [s['epe_raw'] for s in results]
    epe_stb_scenes = [s['epe_stabilized'] for s in results]
    tepe_raw_scenes = [s['tepe_raw'] for s in results]
    tepe_stb_scenes = [s['tepe_stabilized'] for s in results]
    seq_lengths = [s['seq_length'] for s in results]

    n_nan_epe_raw = sum(np.isnan(v) for v in epe_raw_scenes)
    n_nan_tepe_raw = sum(np.isnan(v) for v in tepe_raw_scenes)  
    n_nan_epe_stb = sum(np.isnan(v) for v in epe_stb_scenes)
    n_nan_tepe_stb = sum(np.isnan(v) for v in tepe_stb_scenes)
    
    logging.info(f"NaN scenes — epe_raw: {n_nan_epe_raw}/{len(results)}, tepe_raw: {n_nan_tepe_raw}/{len(results)}, epe_stb: {n_nan_epe_stb}/{len(results)}, tepe_stb: {n_nan_tepe_stb}/{len(results)}")
    
    epe_raw = _length_weighted_mean(epe_raw_scenes, seq_lengths)
    tepe_raw = _length_weighted_mean(tepe_raw_scenes, seq_lengths)
    epe_stb = _length_weighted_mean(epe_stb_scenes, seq_lengths)
    tepe_stb = _length_weighted_mean(tepe_stb_scenes, seq_lengths)  
    
    # overall = {
    #     'epe_raw': float(np.nanmean(epe_raw_scenes)),
    #     'epe_stabilized': float(np.nanmean(epe_stb_scenes)),
    #     'tepe_raw': float(np.nanmean(tepe_raw_scenes)),
    #     'tepe_stabilized': float(np.nanmean(tepe_stb_scenes)),
    # }
    overall = {
        'epe_raw': epe_raw,
        'epe_stabilized': epe_stb,
        'tepe_raw': tepe_raw,
        'tepe_stabilized': tepe_stb,
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    axes[0].imshow(left_img, aspect='auto', interpolation='nearest'); axes[0].set_title("Left Frames"); axes[0].axis('off')

    vmax = np.percentile(np.concatenate([scanline_stabilized, scanline_raw]), 99)
    im1 = axes[1].imshow(scanline_stabilized, cmap='inferno', vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
    axes[1].set_title('Disparity Stabilized'); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)    
    
    im2 = axes[2].imshow(scanline_raw, cmap='inferno', vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
    axes[2].set_title('Disparity Original'); axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)  
      
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

def _select_interesting_frames(disp_raw, disp_stabilized, disp_gt, valid_mask, max_frames=4, device=torch.cuda):
    '''
    Pick a small, informative set of frame indices for the Fig-9-style grid:
    first frame, the frame with the worst raw error, the frame where
    stabilization does the most damage relative to raw, and the last frame.
    Ties this directly to the metrics already being reported, rather than
    an arbitrary/even spacing across the clip.
    '''
    if isinstance(disp_raw, list):
        disp_raw = torch.stack(disp_raw, dim=0)
    if isinstance(disp_stabilized, list):
        disp_stabilized = torch.stack(disp_stabilized, dim=0)

    T = disp_raw.shape[0]
    per_frame_epe_raw, per_frame_epe_stb = [], []
    for t in range(T):
        disp_gt[t] = disp_gt[t].to(device)
        valid_mask[t] = valid_mask[t].unsqueeze(0).to(device)
        e_raw, n_raw = compute_epe(disp_raw[t], disp_gt[t], valid_mask[t])
        e_stb, n_stb = compute_epe(disp_stabilized[t], disp_gt[t], valid_mask[t])
        per_frame_epe_raw.append((e_raw / max(n_raw, 1)).item())
        per_frame_epe_stb.append((e_stb / max(n_stb, 1)).item())

    per_frame_epe_raw = np.array(per_frame_epe_raw)
    per_frame_epe_stb = np.array(per_frame_epe_stb)
    degradation = per_frame_epe_stb - per_frame_epe_raw # if large positive, then stabilization made things worse

    candidates = [0, int(np.argmax(per_frame_epe_raw)), int(np.argmax(degradation)), T - 1]
    frame_idxs = sorted(set(candidates))[:max_frames]
    return frame_idxs, per_frame_epe_raw, per_frame_epe_stb


def build_figure9_grid(disp_raw, disp_stabilized, left_imgs, disp_gt, valid_mask,
                        frame_idxs=None, save_path=None, device=torch.cuda):
    '''
    Fig-9-style qualitative grid: rows = selected frames, columns = 
    [Left frame | Raw disparity | Stabilized disparity].
    Raw and stabilized share one vmax (scene-wide, 99th percentile) so
    the comparison is honest -- independently auto-scaled colorbars would
    hide real differences between the two.
    '''
    if isinstance(disp_raw, list):
        disp_raw = torch.stack(disp_raw, dim=0)
    if isinstance(disp_stabilized, list):
        disp_stabilized = torch.stack(disp_stabilized, dim=0)
    if isinstance(left_imgs, list):
        left_imgs = torch.stack(left_imgs, dim=0)

    if frame_idxs is None:
        frame_idxs, _, _ = _select_interesting_frames(disp_raw, disp_stabilized, disp_gt, valid_mask, device=device)

    vmax = np.percentile(
        torch.cat([disp_raw[frame_idxs], disp_stabilized[frame_idxs]]).cpu().numpy(), 99
    )

    n_rows = len(frame_idxs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['Left Frame', 'Raw Disparity', 'Stabilized Disparity']
    for row, t in enumerate(frame_idxs):
        img = left_imgs[t].permute(1, 2, 0).cpu().numpy() / 255.0
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f'frame {t}', fontsize=10)

        im_raw = axes[row, 1].imshow(disp_raw[t, 0].cpu().numpy(), cmap='inferno', vmin=0, vmax=vmax)
        im_stb = axes[row, 2].imshow(disp_stabilized[t, 0].cpu().numpy(), cmap='inferno', vmin=0, vmax=vmax)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            for ax, title in zip(axes[row], col_titles):
                ax.set_title(title)

    plt.colorbar(im_stb, ax=axes[:, 2], fraction=0.025, pad=0.02, label='disparity (px)')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def build_diff_maps(disp_raw, disp_stabilized, disp_gt, frame_idx, save_path=None):
    '''
    Diagnostic figure for ONE frame: |stab-raw|, |raw-gt|, |stab-gt|, full
    2D, each with its OWN percentile-scaled colorbar -- not the disparity
    range. Sharing one scale with the disparity magnitude is what made the
    original scanline attempt unreadable; this fixes that on purpose.
    '''
    if isinstance(disp_raw, list):
        disp_raw = torch.stack(disp_raw, dim=0)
    if isinstance(disp_stabilized, list):
        disp_stabilized = torch.stack(disp_stabilized, dim=0)

    raw_t = disp_raw[frame_idx, 0].cpu().numpy()
    stb_t = disp_stabilized[frame_idx, 0].cpu().numpy()
    gt_t = disp_gt[frame_idx][0].cpu().numpy()

    diff_stb_raw = np.abs(stb_t - raw_t)
    diff_raw_gt = np.abs(raw_t - gt_t)
    diff_stb_gt = np.abs(stb_t - gt_t)

    panels = [
        (diff_stb_raw, '|Stabilized - Raw|'),
        (diff_raw_gt, '|Raw - GT|'),
        (diff_stb_gt, '|Stabilized - GT|'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (data_map, title) in zip(axes, panels):
        vmax = np.percentile(data_map, 99)  # independent scale per panel, on purpose
        im = ax.imshow(data_map, cmap='inferno', vmin=0, vmax=vmax)
        ax.set_title(f'{title}  (frame {frame_idx})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='|Δ disparity| (px)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)




def check_flow(stabilizer:BiDAStabilizer, device):
    scene_id='A/0120'
    group_idx = {'A':0, 'B':150, 'C':300}
    group = scene_id.split('/')[0]
    video = scene_id.split('/')[1]
    idx = group_idx[group] + int(video)
    
    scene = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'])[idx]

    left_imgs = scene['left']   # list of (3,H,W) tensors, [0,255] range
    gt_flow   = scene['flow']   # list of (2,H,W) tensors -- "into future", i.e. t -> t+1

    frame_idx = 0
    img1 = left_imgs[frame_idx].unsqueeze(0).to(device)
    img2 = left_imgs[frame_idx + 1].unsqueeze(0).to(device)
    gt   = gt_flow[frame_idx].to(device)
    
    with torch.no_grad():
        pred_flow = stabilizer.raft.forward_fullres(img1, img2)

    print("pred_flow shape:", pred_flow.shape)
    print("gt flow shape:  ", gt.shape)
    
    pred_flow = pred_flow.squeeze()
    
    flow_epe = torch.sqrt(((pred_flow - gt) ** 2).sum(dim=0)).mean()
    print(f"\nFlow EPE vs GT: {flow_epe.item():.3f} px")
    print(f"pred -- mean: {pred_flow.mean():.3f}  std: {pred_flow.std():.3f}  "
        f"min: {pred_flow.min():.3f}  max: {pred_flow.max():.3f}")
    print(f"gt   -- mean: {gt.mean():.3f}  std: {gt.std():.3f}  "
        f"min: {gt.min():.3f}  max: {gt.max():.3f}")

    def flow_to_color(flow):
        u, v = flow[0], flow[1]
        mag, ang = cv2.cartToPolar(u, v)
        hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(mag * 4, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(left_imgs[frame_idx].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[0].set_title('Frame t')
    axes[1].imshow(left_imgs[frame_idx + 1].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axes[1].set_title('Frame t+1')
    axes[2].imshow(flow_to_color(pred_flow.cpu().numpy()))
    axes[2].set_title('Predicted flow (internal RAFT)')
    axes[3].imshow(flow_to_color(gt.cpu().numpy()))
    axes[3].set_title('GT flow')
    for ax in axes:
        ax.axis('off')
    plt.savefig('flow_sanity_check.png', dpi=120, bbox_inches='tight')

def get_scene_by_name(dataset, scene_name):
    for i, s in enumerate(dataset.scenes):
        if s['scene_id'] == scene_name:
            return i
    raise ValueError(f"Scene '{scene_name}' not found. Available: {[s['scene_id'] for s in dataset.scenes]}")

    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="[raftstereo_stabilizer, igevstereo_stabilizer, LAS_stabilizer]")
    parser.add_argument('--ckpt_model', help="restore stereo model checkpoint", default='./checkpoints/LiteAnyStereo.pth')
    parser.add_argument('--ckpt_stb', help="restore stabilizer model checkpoint", required=True)
    parser.add_argument('--visualize', action='store_true', help="visualize results")
    parser.add_argument('--output_dir', default='./eval_results_video')
    parser.add_argument('--dataset', choices=['things'] + [f'sintel_{dstype}' for dstype in ['clean', 'final']], default='things', help="dataset for evaluation")
    args = parser.parse_args()
    

    
    
    
    
    
    
    
    
    
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    stereo_model, stb_model = load_models(args.name, args.ckpt_model, args.ckpt_stb, device)
    
    # VISUALIZE
    if args.visualize:
        if args.dataset == 'things':
            scene_id='A/0149'
            group_idx = {'A':0, 'B':150, 'C':300}
            group = scene_id.split('/')[0]
            video = scene_id.split('/')[1]
            idx = group_idx[group] + int(video)
            dataset = datasets.SceneFlowVideo(mode="TEST", subsets=['flyingthings'])
        elif args.dataset.split('_')[0] == 'sintel':
            dstype = args.dataset.split('_')[1]
            dataset = datasets.SintelStereoVideo(mode='training', dstype=dstype)
            scene_name = 'shaman_3'  # Example scene name for Sintel dataset
            idx = get_scene_by_name(dataset, scene_name)

        
        data = dataset[idx]
        scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames = data['scene_id'], data['left'], data['right'], data['disp'], data['valid']
        disp_preds = run_stereo_model(args.name, stereo_model, device, left_imgs, right_imgs)
        disp_stabilized = run_stabilizer_on_video(args.name, stb_model, left_imgs, disp_preds, device=device)
    
        save_path = os.path.join(args.output_dir, f'visualize_scanline.png')
        # build_scanline_image(disp_stabilized, left_imgs, disp_preds, row_idx=None, save_path=save_path)

        
                
        frame_idxs, epe_raw_per_frame, epe_stb_per_frame = _select_interesting_frames(
            disp_preds, disp_stabilized, disp_gt_frames, valid_frames, device=device
        )

        T = len(disp_preds)
        worst = int(np.argmax(epe_stb_per_frame - epe_raw_per_frame))
        start = max(0, min(worst - 2, T - 5))      # clamp so the window stays in range
        frame_idxs = list(range(start, start + 5))
        
        build_figure9_grid(
            disp_preds, disp_stabilized, left_imgs, disp_gt_frames, valid_frames,
            frame_idxs=frame_idxs,
            save_path=os.path.join(args.output_dir, f'{args.name}_{args.dataset}_visualize_fig9_grid_{scene_name}.png')
        )
    
        worst_frame = frame_idxs[int(np.argmax(epe_stb_per_frame[frame_idxs] - epe_raw_per_frame[frame_idxs]))]
        build_diff_maps(
            disp_preds, disp_stabilized, disp_gt_frames, frame_idx=worst_frame,
            save_path=os.path.join(args.output_dir, f'{args.name}_{args.dataset}_visualize_diff_maps_{scene_name}.png')
        )
        sys.exit(0)

    results, overall = evaluate(args.name, stereo_model, stb_model, device, args.dataset)

    logging.info("===== Evaluation Results =====")
    for k, v in overall.items():
        logging.info(f"{k:20s}: {v:.4f}")
        
    overall_path = os.path.join(args.output_dir, f'overall_results_{args.name}_{args.dataset}_192_disp.json')
    with open(overall_path, 'w') as f:
        json.dump(overall, f, indent=2)

    per_scene_path = os.path.join(args.output_dir, f'per_scene_results_{args.name}_{args.dataset}_192_disp.json')
    with open(per_scene_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    csv_path = os.path.join(args.output_dir, f'per_scene_results_{args.name}_{args.dataset}_192_disp.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    