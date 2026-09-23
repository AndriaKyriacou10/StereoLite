from ast import arg

import torch
import matplotlib.pyplot as plt
from core.liteanystereo import CustomLiteAnyStereo
from core.stereo_datasets import TrainingDataset, SceneFlowDataset, ETH3D, Middlebury
import numpy as np
import argparse
from core.utils.utils import InputPadder
import os
import logging
from PIL import Image

def read_dataset(dataset, idx):
    if isinstance(dataset, ETH3D):
        img1, img2, _, _, disp_gt, valid_mask, occ_file = dataset[idx]
        occ_mask = Image.open(occ_file)
        occ_mask = np.ascontiguousarray(occ_mask).flatten()
        occ_tensor = torch.from_numpy(occ_mask == 255).bool()
        val = (valid_mask.flatten() >= 0.5) & occ_tensor 
        threshold = 1.0
        
    else:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        if isinstance(dataset, Middlebury):
            disp_gt_flattened = disp_gt.flatten()
            val = (valid_mask.flatten() >= 0.5) & (disp_gt_flattened < 192)
            threshold = 2.0
        else:
            val = valid_mask.flatten() >= 0.5
            threshold = 1.0
    return img1, img2, disp_gt, valid_mask, val, threshold



@torch.no_grad()
def get_predictions(model, device, dataset, compute_cost_volume=False, max_iters=16):
    model.eval()
    sums_epe = torch.zeros(max_iters, device=device)
    sums_bad = torch.zeros(max_iters, device=device)
    counts = torch.zeros(max_iters, device=device)
    for idx in range(len(dataset)):
        img1, img2, disp_gt, valid_mask, val, threshold = read_dataset(dataset, idx)
        
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device)
        val = val.to(device)

        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        if val.sum().item() == 0: 
            print(f"[{idx}] zero valid pixels — skipping")
            continue
        
        if isinstance(dataset, SceneFlowDataset) and (torch.isnan(disp_gt).any() or torch.isinf(disp_gt).any()):
            print(f"[{idx}] disp_gt has NaN/Inf — skipping")
            continue
        
        disp_preds = model(img1, img2, test_mode=False, compute_cost_volume=compute_cost_volume, iterations = max_iters)

        if torch.stack([p.isnan().any() for p in disp_preds]).any():
            continue
        
        for i, pred in enumerate(disp_preds):
            pred = torch.clamp(padder.unpad(pred), min=0).squeeze()
            assert pred.shape == disp_gt.squeeze().shape
            epe_flat = (pred - disp_gt.squeeze()).abs().flatten()
            sums_epe[i] += epe_flat[val].mean().item()
            sums_bad[i] += (epe_flat > threshold)[val].float().mean().item()
            counts[i] += 1
    return {'epe': (sums_epe / counts).tolist(), 'bad': (100 * sums_bad / counts).tolist(), 'n_images': int(counts[0].item())}


def plot_epe_iters(epe_list, max_iters, dataset_name, cv_flag=False):
    save_path = f'./gru_iterations_test_layer1/epe_vs_iterations_{dataset_name}_{"cv" if cv_flag else "no_cv"}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_iters + 1), epe_list, marker='o')
    plt.title(f'EPE vs. GRU Iterations {dataset_name}')
    plt.xlabel('Number of GRU Iterations')
    plt.ylabel('End-Point Error (EPE)')
    plt.xticks(range(1, max_iters + 1))
    plt.grid()
    plt.savefig(f"{save_path}.png", dpi=400)
    plt.savefig(f"{save_path}.eps")
    logging.info(f"Plot saved as {save_path}.png and {save_path}.eps")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--dataset', type=str, choices=['eth3d', 'sceneflow'] +[f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--max_iters', type=int, default=16, help='Maximum number of GRU iterations')
    parser.add_argument('--cost_volume', action='store_true')
    parser.add_argument('--layer2', action='store_true', help='Use ContextNet with a second layer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLiteAnyStereo(layer2=args.layer2).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    
    if args.dataset == 'eth3d':
        dataset = ETH3D(condition='test', return_occ=True, train_frac=0.5)
    elif args.dataset == 'sceneflow':
        dataset = SceneFlowDataset(mode='TEST', subsets=['flyingthings'])
    elif args.dataset.startswith('middlebury'):
        resolution = args.dataset.split('_')[1]
        dataset = Middlebury(split='MiddEval3', resolution=resolution)
    
    predictions = get_predictions(model, device, dataset, compute_cost_volume=args.cost_volume, max_iters=args.max_iters)
    
    print(f"Results for dataset {args.dataset} with {'cost volume' if args.cost_volume else 'no cost volume'}:")
    print(f"Number of images evaluated: {predictions['n_images']}")
    for i in range(args.max_iters):
        marker = "  <-- trained @ 8" if i == 7 else ""
        print(f"Iteration {i+1}: EPE = {predictions['epe'][i]:.4f}, Bad = {predictions['bad'][i]:.2f}%{marker}")
    plot_epe_iters(predictions['epe'], args.max_iters, args.dataset, args.cost_volume)
    