import torch
import matplotlib.pyplot as plt
from core.liteanystereo import CustomLiteAnyStereo
from core.training_datasets import TrainingDataset, SceneFlowDataset, ETH3D, Middlebury
import numpy as np
import random
import argparse
from core.utils.utils import InputPadder
import os

def calculate_error(disp_pred, disp_gt, valid_mask):
    # 1. Force everything down to a clean 2D spatial layout [H, W]
    # This strips away any accidental [1, 1, H, W] or [H, 1, W] dimensions
    gt_2d = disp_gt.squeeze()
    pred_2d = disp_pred.squeeze()
    mask_2d = valid_mask.squeeze().bool()
    
    # 2. Calculate squared error safely on pure 2D maps
    error = (gt_2d - pred_2d) ** 2
    
    # 3. Mask out the invalid regions
    error[~mask_2d] = 0.0
    
    return error
     
@torch.no_grad()
def get_prediction(model, device, dataset, idx, compute_cost_volume=False):
    
    if isinstance(dataset, ETH3D):
        img1, img2, _, _, disp_gt, valid_mask, occ_file = dataset[idx]
    else:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
    
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    disp_gt = disp_gt.to(device)
    valid_mask = valid_mask.to(device)
    
    padder = InputPadder(img1.shape, divis_by=32)
    img1, img2 = padder.pad(img1, img2)
    print(img1.shape)
    disp_pred = model(img1, img2, test_mode = True, compute_cost_volume=compute_cost_volume)
    
    disp_pred = padder.unpad(disp_pred)
    error = calculate_error(disp_pred, disp_gt, valid_mask)
    
    img1 = img1/255.0
    
    left_img_np = (img1.detach().cpu().squeeze().permute(1, 2, 0).numpy()).clip(0, 1)
    
    disp_pred_np = disp_pred.detach().cpu().squeeze().numpy()
    error_np = error.detach().cpu().squeeze().numpy()
    
    negative = np.sum(disp_pred_np < 0)
    print(f"Negative Disparities: {negative} | Total Pixels: {disp_pred_np.size}")
    
    print(f"Min Disp:{disp_pred.min()} | Max Disp:{disp_pred.max()} | Percenile: {np.percentile(disp_pred_np, [5, 25, 50, 75, 95])}")
    return left_img_np, disp_pred_np, error_np
    
def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for CustomLiteAnyStereo')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/phase2_best_model.pth', help='Path to the model weights')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to process')
    parser.add_argument('--out_dir', type=str, default='./inference_images', help='Directory to save inference results')
    parser.add_argument('--compute_cost_volume', action='store_true',
                         help='Run with cost volume (teacher-equivalent) instead of the lite student path')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed for random sampling when --indices is not given')
    parser.add_argument('--dataset', type=str, default='sceneflow', choices=['sceneflow', 'eth3d', 'middlebury'], help='Dataset for evaluation')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                         help='Specific dataset indices to run, e.g. --indices 3237 1295 2273')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLiteAnyStereo().to(device)
    out_dir = None
    
    # Load Weights
    weights = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(weights['model_state'])
    model.eval()
    
    print("Initializing dataset...")
    if args.dataset == 'sceneflow':
        val_dataset = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
        out_dir = f"{args.out_dir}/sceneflow"
    elif args.dataset == 'eth3d':
        val_dataset = ETH3D(condition='test', return_occ=True)
        out_dir = f"{args.out_dir}/eth3d"
    elif args.dataset == 'middlebury':
        val_dataset = Middlebury(split='MiddEval3', resolution='F')
        out_dir = f"{args.out_dir}/middlebury"
        
    dataset_length = len(val_dataset)
    
    if args.indices is not None:
        indices = args.indices
    else:
        random.seed(args.seed)
        indices = [random.randint(0, len(val_dataset) - 1) for _ in range(args.num_samples)]
    
    for i, idx in enumerate(indices):
        
        print(f"--- Processing pair {i + 1}/10 (Dataset Index: {idx}) ---")
        left_img, disp_pred, error_map = get_prediction(model, device, val_dataset, idx, compute_cost_volume=args.compute_cost_volume)
        
        plt.figure(figsize=(18, 5)) 
        # plt.imshow(disp_pred < 0, cmap='gray')
        # save_path = f"{out_dir}/negative_disparity_map_{idx}.png"
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        plt.subplot(1, 3, 1)
        plt.title(f"Left RGB (Index: {idx})")
        plt.imshow(left_img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Predicted Disparity")
        plt.imshow(disp_pred, cmap='magma') 
        plt.colorbar(fraction=0.046, pad=0.04) 
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Squared Error (>0 indicates failure)")
        plt.imshow(error_map, cmap='hot', vmin=0, vmax=50) 
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.tight_layout() 
        
        # Save dynamically named file so they don't overwrite
        tag = 'cv' if args.compute_cost_volume else 'nocv'
        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/inference_analysis_{idx}_{tag}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"All {len(indices)} inferences completed and saved")