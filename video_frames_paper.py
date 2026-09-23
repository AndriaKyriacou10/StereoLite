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
import core.video_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
import cv2
from evaluation_og_stabilizer import run_stereo_model, run_stabilizer_on_video

from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer

CMAP = "inferno"
PCT = 99.0
MAX_DISP = 192
CBAR_DPI = 300
PCT_DIFF = 90.0

def _to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return np.squeeze(np.asarray(x, dtype=np.float32))

# Resize Images to max width
def _resize_rgb(rgb, max_width):
    """Downsample the COLOURISED image, never the raw disparity -- area-
    averaging disparity across a depth discontinuity invents values no model
    predicted."""
    if max_width is None or rgb.shape[1] <= max_width:
        return rgb
    h, w = rgb.shape[:2]
    new_h = int(round(h * max_width / w))
    return np.asarray(Image.fromarray(rgb).resize((max_width, new_h), Image.LANCZOS))
    
def colour_invalid(arr, vmin, vmax, cmap=CMAP, valid=None, bad=(255, 255, 255)):
    """2D float array -> uint8 RGB. Invalid pixels get a flat colour so a GT
    hole never reads as a disparity value."""
    arr = _to_numpy(arr)
    rgba = plt.get_cmap(cmap)(Normalize(vmin=vmin, vmax=vmax, clip=True)(arr))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    if valid is not None:
        rgb[~_to_numpy(valid).astype(bool)] = np.asarray(bad, dtype=np.uint8)
    return rgb

def save_rgb(img, path, max_width=None):
    """Left image. Accepts CHW or HWC, [0,1] or [0,255]."""
    arr = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 3: # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(_resize_rgb(arr, max_width)).save(path)
    
def save_colorbar(vmin, vmax, path, cmap=CMAP, label="disparity (px)"):
    """Vector -- this is the part that actually benefits from a pdf wrapper,
    it's text and lines, not a bitmap."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.34, 2.2))
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
                cax=ax, label=label)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=CBAR_DPI)
    plt.close(fig)

def _get_idx(dataset, scene):
    for i, s in enumerate(dataset.scenes):
        if s['scene_id'] == scene:
                return i
    raise ValueError(f"Scene '{scene}' not found. Available: {[s['scene_id'] for s in dataset.scenes]}")
        

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

def load_dataset(dataset):
    if dataset.startswith("sintel"):
        dstype = dataset.split("_")[1]
        val_dataset = datasets.SintelStereoVideo(dstype=dstype, max_disp=MAX_DISP)
    elif dataset == "things":
        val_dataset = datasets.SceneFlowVideo(mode='TEST', subsets=['flyingthings'], max_disp=MAX_DISP)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return val_dataset

def save_panel(arr, path, vmin, vmax, cmap=CMAP, valid=None, max_width=None):
    rgb = _resize_rgb(colour_invalid(arr, vmin, vmax, cmap, valid), max_width)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(rgb).save(path)

def save_video_frame_panels(disp_raw, disp_stabilized, left_imgs, disp_gt, frame_idxs, scene, out_dir, max_width=None, gt_tag="gt"):
    """
    Writes left/raw/stabilized as three separate PNGs per frame in
    frame_idxs, plus one shared colorbar for the whole scene.
 
    disp_raw, disp_stabilized: [T,1,H,W] tensor or list of [1,H,W] tensors.
    left_imgs: [T,3,H,W] tensor or list of [3,H,W] tensors, [0,255] range.
    frame_idxs: which frames to write out -- every one gets all three panels.
    scene: filename prefix, e.g. "southken_scene03".
    out_dir: written directly here (not namespaced per-dataset like the
        image-index script, since "scene" already disambiguates).
 
    Returns (vmin, vmax) actually used, in case you want it for a caption or
    to reuse the exact same scale in a follow-up call (e.g. an error map).
    """
    if isinstance(disp_raw, list):
        disp_raw = torch.stack(disp_raw, dim=0)
    if isinstance(disp_stabilized, list):
        disp_stabilized = torch.stack(disp_stabilized, dim=0)
    if isinstance(left_imgs, list):
        left_imgs = torch.stack(left_imgs, dim=0)
    if isinstance(disp_gt, list):
        disp_gt = torch.stack(disp_gt, dim=0)
    if isinstance(disp_gt, list):
        disp_gt = torch.stack(disp_gt, dim=0)
 
    vmin = 0.0
    pooled = torch.cat([disp_raw, disp_stabilized]).clamp(min=0, max=MAX_DISP)
    # vmax = float(np.percentile(pooled.cpu().numpy(), PCT))
    sel = torch.as_tensor(frame_idxs, dtype=torch.long)
    pooled = torch.cat([disp_raw[sel], disp_stabilized[sel]]).clamp(min=0, max=MAX_DISP)
    # vmax = float(np.percentile(pooled.cpu().numpy(), PCT))
    
    sel = torch.as_tensor(list(frame_idxs), dtype=torch.long)
    pooled = torch.cat([disp_raw[sel], disp_stabilized[sel]]).clamp(min=0, max=MAX_DISP)
    vmax = float(np.percentile(pooled.cpu().numpy(), PCT))
 
    os.makedirs(out_dir, exist_ok=True)
    for t in frame_idxs:
        stem = f"{scene}_{t:03d}"
        save_rgb(left_imgs[t], f"{out_dir}/{stem}_left.png", max_width=max_width)
        save_panel(disp_raw[t, 0], f"{out_dir}/raw/{stem}_raw.png", vmin, vmax, max_width=max_width)
        save_panel(disp_stabilized[t, 0], f"{out_dir}/stb/{stem}_stb.png", vmin, vmax, max_width=max_width)
        save_panel(disp_gt[t, 0], f"{out_dir}/{gt_tag}/{stem}_{gt_tag}.png", vmin, vmax, max_width=max_width)
        stem = f"{scene}_{t:03d}"
        save_rgb(left_imgs[t], f"{out_dir}/{stem}_left.png", max_width=max_width)
        save_panel(disp_raw[t, 0], f"{out_dir}/raw/{stem}_raw.png", vmin, vmax, max_width=max_width)
        save_panel(disp_stabilized[t, 0], f"{out_dir}/stb/{stem}_stb.png", vmin, vmax, max_width=max_width)
        save_panel(disp_gt[t, 0], f"{out_dir}/{gt_tag}/{stem}_{gt_tag}.png", vmin, vmax, max_width=max_width)
        print(f"[{scene} frame {t}] saved left/raw/stb/GT  (vmax {vmax:.1f}px)")
 
    save_colorbar(vmin, vmax, os.path.join(out_dir, f"{scene}_cbar.pdf"))
def _temporal_diff(disp, t):
    """|D_t - D_{t-1}| for a [T,1,H,W] tensor, as a 2D float array."""
    return _to_numpy((disp[t, 0] - disp[t - 1, 0]).abs())


def save_temporal_diff_panels(disp_raw, disp_stabilized, frame_idxs, scene,
                              out_dir, max_width=None, pct=PCT_DIFF, vmax=None):
    """
    Writes |D_t - D_{t-1}| for raw and stabilised as separate PNGs, on a
    SHARED colour scale, plus one colourbar.

    Flicker is a between-frame quantity, so consecutive disparity maps look
    identical on a poster and prove nothing. These do show it: temporal noise
    reads as bright speckle, and the stabilised row should be visibly darker.

    The shared vmax is the whole point -- autoscaling each panel makes both
    rows equally bright and destroys the comparison. It's taken from the raw
    differences so the stabilised row is darker because it IS darker.
    """
    if isinstance(disp_raw, list):
        disp_raw = torch.stack(disp_raw, dim=0)
    if isinstance(disp_stabilized, list):
        disp_stabilized = torch.stack(disp_stabilized, dim=0)

    disp_raw = disp_raw.clamp(min=0, max=MAX_DISP)
    disp_stabilized = disp_stabilized.clamp(min=0, max=MAX_DISP)

    idxs = [t for t in frame_idxs if t >= 1]          # t=0 has no predecessor
    if not idxs:
        raise ValueError("temporal differences need at least one index >= 1")

    if vmax is None:
        pooled = np.stack([_temporal_diff(disp_raw, t) for t in idxs])
        vmax = float(np.percentile(pooled, pct))
    vmin = 0.0

    os.makedirs(out_dir, exist_ok=True)
    for t in idxs:
        stem = os.path.join(out_dir, f"{scene}_{t:03d}")
        save_panel(_temporal_diff(disp_raw, t), f"{stem}_diff_raw.png",
                   vmin, vmax, max_width=max_width)
        save_panel(_temporal_diff(disp_stabilized, t), f"{stem}_diff_stb.png",
                   vmin, vmax, max_width=max_width)
        print(f"[{scene} frame {t}] saved diff raw/stb  (vmax {vmax:.2f}px)")

    save_colorbar(vmin, vmax, os.path.join(out_dir, f"{scene}_diff_cbar.pdf"),
                  label=r"$|D_t - D_{t-1}|$  (px)")

    # sanity check before you build the panel: if these are close, the figure
    # won't show anything and you need a different scene
    m_raw = float(np.mean([_temporal_diff(disp_raw, t).mean() for t in idxs]))
    m_stb = float(np.mean([_temporal_diff(disp_stabilized, t).mean() for t in idxs]))
    print(f"[{scene}] mean |dD/dt|  raw {m_raw:.3f}px  stb {m_stb:.3f}px  "
          f"({100.0 * (m_stb - m_raw) / max(m_raw, 1e-9):+.1f}%)")

    return vmin, vmax

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="[raftstereo_stabilizer, igevstereo_stabilizer, LAS_stabilizer]")
    parser.add_argument('--ckpt_model', help="restore stereo model checkpoint", default='./checkpoints/LiteAnyStereo.pth')
    parser.add_argument('--ckpt_stb', help="restore stabilizer model checkpoint", required=True)
    parser.add_argument('--out_dir', default='./video_frames')
    parser.add_argument('--dataset', choices=['things', 'southken'] + [f'sintel_{dstype}' for dstype in ['clean', 'final']], default='things', help="dataset for evaluation")
    parser.add_argument('--scene_name', default=None, help="specific scene name for Sintel dataset (e.g., bamboo_2)")
    parser.add_argument('--start_frame', type=int, default=0, help="start frame index for video")
    parser.add_argument('--iter_name', default='iter35k', help="specific scene name for Sintel dataset (e.g., bamboo_2)")
    parser.add_argument('--end_frame', type=int, default=None, help="end frame index for video")
    parser.add_argument('--kernel_size', type=int, default=50, help="kernel size for stabilizer")
    parser.add_argument('--max_width', type=int, default=1500, help="max width for output images")
    parser.add_argument('--end_frame', type=int, default=None, help="end frame index for video")
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    
    stereo_model, stb_model = load_models(args.name, args.ckpt_model, args.ckpt_stb, device)
    dataset = load_dataset(args.dataset)
    
    idx = _get_idx(dataset, args.scene_name)
    print(idx)
    data = dataset[idx]
    scene_id, left_imgs, right_imgs, disp_gt_frames, valid_frames = data['scene_id'], data['left'], data['right'], data['disp'], data['valid']
    
    disp_preds = run_stereo_model(args.name, stereo_model, device, left_imgs, right_imgs)
    disp_stabilized = run_stabilizer_on_video(args.name, stb_model, left_imgs, disp_preds, device=device, kernel_size=args.kernel_size)
    
    end_frame = args.end_frame if args.end_frame is not None else 20
    
    if end_frame > len(left_imgs):
        end_frame = len(left_imgs)
        print(f"End frame exceeds available frames. Setting end_frame to {end_frame}.")
    
    frame_idxs = list(range(args.start_frame, end_frame))
    disp_stabilized = run_stabilizer_on_video(args.name, stb_model, left_imgs, disp_preds, device=device, kernel_size=args.kernel_size)
    
    end_frame = args.end_frame if args.end_frame is not None else 20
    
    if end_frame > len(left_imgs):
        end_frame = len(left_imgs)
        print(f"End frame exceeds available frames. Setting end_frame to {end_frame}.")
    
    frame_idxs = list(range(args.start_frame, end_frame))
    gt_tag = 'pseudo_gt' if args.dataset == 'southken' else 'gt'
    
    if args.max_width is None:
        res = 'full'
    else:
        res = f"maxwidth_{args.max_width}"
    
    out_dir = f"{args.out_dir}/{args.name}/{args.dataset}/{scene_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    for t in frame_idxs:
        r, s = disp_raw[t,0], disp_stabilized[t,0]
        print(f"t={t}  raw {r.min():.1f}/{r.median():.1f}/{r.max():.1f}  "
            f"stb {s.min():.1f}/{s.median():.1f}/{s.max():.1f}  vmax={vmax:.1f}")
    
    save_video_frame_panels(
    disp_preds, disp_stabilized, left_imgs, disp_gt_frames,
    frame_idxs=frame_idxs,
    scene=args.scene_name,
    out_dir=out_dir,
    max_width=args.max_width, 
    max_width=None, 
    gt_tag=gt_tag
    )
    