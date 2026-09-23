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
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime
import cv2

from bidastabilizer_integration.models.bidastabilizer import BiDAStabilizer

def load_model(name, ckpt_model, device):
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

    return stereo_model


@torch.no_grad()
def persistence_stats(disp_raw, disp_gt, valid_mask):
    num_r = den_t = den_t1 = 0.0
    sum_abs_e = sum_abs_de = 0.0
    n_e = n_de = 0

    for t in range(len(disp_raw) - 1):
        
        e_t  = (disp_raw[t]   - disp_gt[t]).squeeze()
        e_t1 = (disp_raw[t+1] - disp_gt[t+1]).squeeze()
        m = (valid_mask[t].squeeze().bool() & valid_mask[t+1].squeeze().bool())
        if not m.any():
            continue
        a, b = e_t[m], e_t1[m]

        sum_abs_e  += a.abs().sum().item();       n_e  += m.sum().item()
        sum_abs_de += (a - b).abs().sum().item(); n_de += m.sum().item()

        a_c, b_c = a - a.mean(), b - b.mean()
        num_r  += (a_c * b_c).sum().item()
        den_t  += (a_c ** 2).sum().item()
        den_t1 += (b_c ** 2).sum().item()

    if n_e == 0:
        return {"epe": float('nan'), "tepe": float('nan'),
                "rho": float('nan'), "r": float('nan')}
    
    epe  = sum_abs_e  / max(n_e, 1)
    tepe = sum_abs_de / max(n_de, 1)
    return {"epe": epe, "tepe": tepe,
            "rho": tepe / max(epe, 1e-8),
            "r": num_r / max((den_t * den_t1) ** 0.5, 1e-8) if den_t * den_t1 > 0 else float('nan')}

def run_stereo_model(name, model, left_imgs, right_imgs, device):
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


if __name__ == "__main__":
    def _maxdisp(s):
        return None if s.lower() in ("none", "null", "") else int(s)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="LAS_stabilizer", help="model name")
    parser.add_argument("--dataset", type=str, default="things", choices = ['things', 'sintel_clean', 'sintel_final'], help="dataset name")
    parser.add_argument("--ckpt_model", type=str, default=None, help="path to the checkpoint of the stereo model")
    parser.add_argument("--ckpt_stb", type=str, default=None, help="path to the checkpoint of the stabilizer model")
    parser.add_argument("--max_disp", type=_maxdisp, default=192)
    parser.add_argument("--output_dir", type=str, default="./output", help="path to save the output results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_path = os.path.join(args.output_dir, f'stats_stabilizer_{current_time}_{args.name}_{args.dataset}_{args.max_disp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    stereo_model = load_model(args.name, args.ckpt_model, device)

    if args.dataset == "things":
        dataset = datasets.SceneFlowVideo(mode="TEST", subsets=["flyingthings"],
                                      max_disp=args.max_disp)
    else:
        dataset = datasets.SintelStereoVideo(mode="training",
                                         dstype=args.dataset.split("_")[1],
                                         max_disp=args.max_disp)
    logging.info(f"Dataset: {args.dataset} | Number of scenes: {len(dataset)} | Max disp: {args.max_disp}")
    
    # Evaluate
    rows = []
    for scene_idx in range(len(dataset)):
        data = dataset[scene_idx]
        disp_predictions = run_stereo_model(args.name, stereo_model, data['left'], data['right'], device)
        
        disp_gt = [d.to(device) for d in data['disp']]
        valid   = [v.to(device) for v in data['valid']]

        stats = persistence_stats(disp_predictions, disp_gt, valid)

        row = {'scene_id': str(data['scene_id']),
            'seq_length': len(disp_predictions),
            **stats}
        rows.append(row)

        logging.info(
            f"[{scene_idx+1}/{len(dataset)}] {row['scene_id']:<24} "
            f"EPE {row['epe']:7.3f} | TEPE {row['tepe']:7.3f} | "
            f"rho {row['rho']:5.3f} | r {row['r']:6.3f}"
    )
    def _length_weighted_mean(values, lengths):
        num = denom = 0.0
        for v, l in zip(values, lengths):
            if not np.isnan(v):
                num += v * l
                denom += l
        return num / denom if denom > 0 else float('nan')

    lengths = [r['seq_length'] for r in rows]
    agg = {k: _length_weighted_mean([r[k] for r in rows], lengths)
        for k in ('epe', 'tepe', 'rho', 'r')}
    
    tag = f"{args.name}_{args.dataset}_maxdisp{args.max_disp}"
    with open(f"{args.output_dir}/stats_{tag}_{current_time}.json", "w") as f:
        json.dump({'args': vars(args), 'aggregate': agg,
                'pooled_rho': agg['tepe'] / max(agg['epe'], 1e-8),
                'per_scene': rows}, f, indent=2)

    with open(f"{args.output_dir}/stats_{tag}_{current_time}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)