import torch

import os
import argparse
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from core.liteanystereo import LiteAnyStereo
import core.stereo_datasets as datasets


def filter_missing_files(dataset, name="Dataset"):
    """
    Purges dataset lists of any samples missing physical image or disparity files.
    Prevents the dataloader from falling back to index 0 and overfitting.
    """
    valid_images = []
    valid_disps = []
    missing_count = 0
    
    for i in range(len(dataset.image_list)):
        img1, img2 = dataset.image_list[i]
        disp = dataset.disparity_list[i]
        
        if os.path.exists(img1) and os.path.exists(img2) and os.path.exists(disp):
            valid_images.append([img1, img2])
            valid_disps.append(disp)
        else:
            missing_count += 1
            
    # Overwrite the broken lists with the clean ones
    dataset.image_list = valid_images
    dataset.disparity_list = valid_disps
    
    print(f"[{name}] Cleaned! Retained {len(valid_images)} valid pairs. Purged {missing_count} missing files.")
    return dataset

class CropAugmentor:
    def __init__(self, crop_size=(320, 736), max_disp=192, min_valid_pixels = 5000):
        self.crop_size = crop_size
        self.max_disp = max_disp
        self.min_valid_pixels = min_valid_pixels
    def __call__(self, img1, img2, flow, valid=None):
        th, tw = self.crop_size 
        h, w, _ = img1.shape
        
        # Pad if image is smaller than crop size
        if h < th or w < tw:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)
            img1 = np.pad(img1, ((0, pad_h), (0, pad_w), (0,0)), mode='edge')
            img2 = np.pad(img2, ((0, pad_h), (0, pad_w), (0,0)), mode='edge')
            # Flow -> Ground Truth Displacement Map    
            flow = np.pad(flow, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            if valid is not None:
                valid = np.pad(valid, ((0, pad_h), (0, pad_w)), mode='constant')
            h, w = img1.shape[:2]
        
        # Find a crop with meaningful amount of valid data
        # e.g avoid crops of solid coloured walls
        max_attempts = 10
        for _ in range(max_attempts):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th) 
            
            crop_flow = flow[y1:y1+th, x1:x1+tw]
            crop_valid = valid[y1:y1+th, x1:x1+tw] if valid is not None else None
            
            if crop_valid is not None:
                disp = crop_flow[..., 0] # select the first channel i.e horizontal disparity (discard vertical disparity)
                valid_mask = (crop_valid >= 0.5) & (disp < self.max_disp) & (disp > 0)

                # Count how many pixels pass the test
                if np.sum(valid_mask) > self.min_valid_pixels:
                    break 
            else:
                break
        img1 = img1[y1:y1+th, x1:x1+tw]
        img2 = img2[y1:y1+th, x1:x1+tw]

        if valid is not None:
            return img1, img2, crop_flow, crop_valid
        return img1, img2, crop_flow

def l1_loss(disp_preds, disp_gt, valid, max_disp=192):
    """
    Calculate L1 loss by comparing predicted disparity with ground truth disparity, only over valid pixels.

    """
    loss = 0.0
    weights = [1.0, 0.3]  # Weight for final disp and intermediate disp
    
    disp_gt = disp_gt[:, 0, :, :]
    
    valid = (valid >= 0.5) & (disp_gt < max_disp) & (disp_gt > 0)
    valid_mask = valid.float()
    
   # Loop through predictions (final and intermediate)
    for i, disp_pr in enumerate(disp_preds):
        disp_pr = disp_pr.squeeze(1) # [B, H, W] : Remove channel dimension
        diff = F.l1_loss(disp_pr[valid], disp_gt[valid], reduction='mean')
        if not torch.isnan(diff):
            loss += weights[i] * diff
            
    return loss


def main():
    num_epochs = 250
    physical_batch_size = 4        
    effective_batch_size = 64      # Target batch size to simulate multi-GPUy
    
    accumulation_steps = effective_batch_size // physical_batch_size
    peak_lr = 2e-4
    max_disp = 192
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    cudnn.benchmark = True
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    
    logging.info(f"Initializing LiteAnyStereo model")
    
    augmentor = CropAugmentor(crop_size=(320, 736))
    
    tmp_dir = os.environ.get('TMPDIR', './data/datasets')
    root_path = f"{tmp_dir}/Middlebury"
    logging.info(f"Using root path: {root_path}")
    
    logging.info("Loading Middlebury datasets...")
    train_datasets = []
    for split in ['2005', '2006', '2021']:
        logging.info(f"Loading Middlebury {split} dataset")
        dataset = datasets.Middlebury(aug_params=None, root = root_path, split=split)
        dataset.augmentor = augmentor
        dataset = filter_missing_files(dataset, name=f"Middlebury {split}")
        train_datasets.append(dataset)

    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=physical_batch_size, 
        shuffle=True, 
        num_workers=8, 
        drop_last=True, 
        pin_memory=True
    )   
    
    model = LiteAnyStereo().to(device)
    
    checkpoint_path = './checkpoints/LiteAnyStereo_MIX_Stage2.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint, strict=True)
    
    logging.info("Freeze backbone...")
    for param in model.fnet.parameters():
        param.requires_grad = False
    model.fnet.eval() # Prevent BatchNorm from updating running stats
    
    
    # --- OPTIMIZER & ONE-CYCLE SCHEDULER ---
    # Pass ONLY the unfrozen parameters (cost aggregation modules)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=peak_lr, weight_decay=1e-4)

    # Calculate true total steps based on accumulation
    steps_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=peak_lr, 
        total_steps=total_steps,
        pct_start=0.05, 
        cycle_momentum=False
    )
    
    # --- TRAINING LOOP ---
    logging.info("Starting training loop...")
    
    for epoch in range(num_epochs):
        model.train()
        # Keep all BatchNorm layers in eval mode to prevent running stats corruption
        # from small batch sizes (stem_2, FPNLayer use BatchNorm2d)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(train_loader):
            _, img1, img2, gt_disp, valid_mask = data
            
            # Use non_blocking=True for faster data transfers with pin_memory=True
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            gt_disp = gt_disp.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            
            # Forward Pass
            disp_preds = model(img1, img2, max_disp=max_disp, test_mode=False, kd_mode=False)
            
            # Compute scaled loss
            loss = l1_loss(disp_preds, gt_disp, valid_mask)
            scaled_loss = loss / accumulation_steps
            
            # Backward pass
            scaled_loss.backward()

            # Optimizer Step (Gradient Accumulation)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            
            if batch_idx % 16 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = f"checkpoints/liteanystereo_middlebury_ep{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    print("Fine-tuning complete. The model is ready for validation.")

if __name__ == '__main__':
    main()
            
            