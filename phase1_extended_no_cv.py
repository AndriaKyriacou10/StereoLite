
from calendar import c
import enum

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from core.stereolite import StereoLite
import os
import time
from core.stereo_datasets import fetch_training_dataloader, fetch_testing_dataloader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from core.utils.utils import InputPadder, CustomLogger
import logging

def sequence_loss(disp_preds, disp_gt, valid_mask, gamma=0.8):
    total_loss = 0.0
    num_predictions = len(disp_preds)
    
    # Extract only valid ground truth pixels
    valid_gt = disp_gt[valid_mask.bool().unsqueeze(1)]
    
    for i, pred in enumerate(disp_preds):
        # Extract only valid prediction pixels
        valid_pred = pred[valid_mask.bool().unsqueeze(1)]
        
        # Base L1 Loss (Mean Absolute Error)
        l1_error = F.l1_loss(valid_pred, valid_gt, reduction='mean')
        
        # Exponential Weighting: 0.8 ** (N - 1 - i)
        weight = gamma ** (num_predictions - 1 - i)
        total_loss += weight * l1_error
        
    return total_loss

def calculate_metrics(final_pred, disparity_gt, valid_mask, logger=None, is_eval=False, padder=None):
    
    if padder is not None:
        final_pred = padder.unpad(final_pred)
    else:
        H, W = valid_mask.shape[-2:]
        final_pred = final_pred[..., :H, :W]
    
    valid_pred = final_pred[valid_mask.bool().unsqueeze(1)]
    valid_gt = disparity_gt[valid_mask.bool().unsqueeze(1)]
    
    # A 1D tensor containing the absolute error of every valid pixel
    abs_err = torch.abs(valid_pred.float() - valid_gt.float())
    
    
    epe = torch.mean(abs_err)
    epe2 = abs_err.mean().item()
    
    bad = {
        "bad1": (torch.mean((abs_err > 1).float()) * 100.0).item(),
        "bad2": (torch.mean((abs_err > 2).float()) * 100.0).item(),
        "bad3": (torch.mean((abs_err > 3).float()) * 100.0).item()
    }
    
    return epe.item(), bad

def evaluate(model:StereoLite, val_loader, device, logger=None, cv=True):
    #  Set to validation mode
    model.eval()

    total_epe = total_bad1 = total_bad2 = total_bad3 = 0.0
    n_batches = 0
    with torch.no_grad():
        for data in val_loader:
                img1, img2, _, _, disp_gt, valid_mask = data
                img1, img2 = img1.to(device), img2.to(device)
                disp_gt, valid_mask = disp_gt.to(device), valid_mask.to(device)

                padder = InputPadder(img1.shape, divis_by=32)
                img1, img2 = padder.pad(img1, img2)

                if not valid_mask.any():
                    continue

                disp_pred = model(img1, img2, test_mode=True, compute_cost_volume=cv)
                epe, bad = calculate_metrics(disp_pred, disp_gt, valid_mask, logger, True, padder=padder)

                total_epe += epe
                total_bad1 += bad['bad1']
                total_bad2 += bad['bad2']
                total_bad3 += bad['bad3']
                n_batches += 1

    return total_epe / n_batches, {'bad1': total_bad1/n_batches, 'bad2': total_bad2/n_batches, 'bad3': total_bad3/n_batches}


def phase1_extended_no_cv(ckpt, resume_ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = fetch_training_dataloader(is_phase_2=False, datasets=['sceneflow', 'eth3d', 'middlebury'])
    val_loaders = fetch_testing_dataloader(datasets=['sceneflow', 'eth3d', 'middlebury'])
    
    p1_model = StereoLite().to(device)
    
    total_steps = 450000
    
    
    if resume_ckpt:
        logging.info(f"Resuming training from checkpoint: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location=device)
    else:
        checkpoint = torch.load(ckpt, map_location=device)
    
    if resume_ckpt:
        step = checkpoint.get('step', 0)
        best_epe = checkpoint.get('best_epe', float('inf'))
        logging.info(f"Resuming from step: {step}")
    else:
        step = 0
        best_epe = float('inf')
        
    p1_model.load_state_dict(checkpoint['model_state'])
    
    optimizer = torch.optim.AdamW(p1_model.parameters(), lr=0.00002)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # for g in optimizer.param_groups:
    #     g['lr'] = 1.4e-4

    scaler = torch.amp.GradScaler('cuda')
    
    logging.info("Initializing TensorBoard Logger...")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = CustomLogger(log_dir=f'./runs/phase1_extended_one_cycleLR_{current_time}', flush_freq=100)

    w = p1_model.context_net.model.conv1.weight.data
    logger.log_batch({'ctxnet_w_left_norm': w[:, :3].norm().item(), 'ctxnet_w_right_norm': w[:, 3:].norm().item()})

    keep_training = True
   
    # total_steps += 50000 # Extra steps for extended training until convergence
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 0.0002, total_steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear'
    )
    
    if resume_ckpt:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scheduler.step()
    
    logging.info(f"Starting Phase 1 Extended Training | Total Steps: {total_steps} | Starting Step: {step} | Best EPE: {best_epe:.4f}")
        
    save_dir = './checkpoints_extended_run'
    os.makedirs(save_dir, exist_ok=True)
    
    val_freq = len(train_loader)
    
    def _run_validation(best_epe, model, val_loaders, device, logger, step, optimizer, save_dir, current_time, scheduler):
        per_dataset_epe = {}
        for name, val_loader in val_loaders.items():
            val_epe, _ = evaluate(model, val_loader, device, logger, cv=False)
            per_dataset_epe[name] = val_epe
            logger.writer.add_text('Validation - Dataset', f"Step {step+1}: Dataset: {name} Validation EPE = {val_epe:.4f}", step+1)

        val_epe = sum(per_dataset_epe.values()) / len(per_dataset_epe)
        logger.log_batch({'val_epe': val_epe})
        logger.writer.add_text('Phase 1 Extended No CV: Validation Summary', f"Step {step+1}: Validation EPE = {val_epe:.4f}", step+1)
        logging.info(f"Phase 1 Extended No CV | Step {step+1} | Validation EPE: {val_epe:.2f}") 
        
        checkpoint = {
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        tag = 'latest'
        if val_epe < best_epe:
            tag = 'BEST'
            best_epe = val_epe
            logging.info(f"--> Saved new best model: (EPE: {best_epe:.4f})")
        
        checkpoint['best_epe'] = best_epe
        
        if tag == 'latest':
            torch.save(checkpoint, f"{save_dir}/phase1_extended_oneCycleLR_Resume_{tag}.pth")
        else:
            torch.save(checkpoint, f"{save_dir}/phase1_extended_{tag}_oneCycleLR_Resume_{current_time}.pth")
        return val_epe, best_epe
        
    
    while keep_training:
        for idx, data in enumerate(train_loader):
            _, _, aug1, aug2, disp_gt, valid_mask = data
            
            img1 = aug1.to(device)
            img2 = aug2.to(device)
            disp_gt = disp_gt.to(device)
            valid_mask = valid_mask.to(device)
            
            p1_model.train()
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type):
                disp_preds = p1_model(img1, img2, test_mode=False, compute_cost_volume=False)
                loss = sequence_loss(disp_preds, disp_gt, valid_mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            scheduler.step()

            final_pred = disp_preds[-1].detach() # Detach so it doesn't drain VRAM
            epe, bad_metrics = calculate_metrics(final_pred, disp_gt, valid_mask)
            
            w = p1_model.context_net.model.conv1.weight.data
            w_left_norm = w[:, :3].norm().item()
            w_right_norm = w[:, 3:].norm().item()
            
            if logger:
                current_metrics = {
                    'loss': loss.item(),
                    'epe': epe,
                    'bad_1': bad_metrics['bad1'],
                    'bad_2': bad_metrics['bad2'],
                    'bad_3': bad_metrics['bad3'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'ctxnet_w_left_norm': w_left_norm,
                    'ctxnet_w_right_norm': w_right_norm,
                    'w_ratio': w_right_norm / w_left_norm
                }
            
                logger.log_batch(current_metrics)
            
                text_string =  f"Step {logger.global_step} | Loss: {loss.item():.4f} | EPE: {epe:.4f} | Bad-1: {bad_metrics['bad1']:.2f}% | Bad-2: {bad_metrics['bad2']:.2f}% | Bad-3: {bad_metrics['bad3']:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
                logger.writer.add_text('Training Metrics', text_string, logger.global_step)

            if step % val_freq == 0:
                logging.info(text_string)
                val_epe, best_epe = _run_validation(best_epe, p1_model, val_loaders, device, logger, step, optimizer, save_dir, current_time, scheduler)
            step += 1
            if step >= total_steps:
                keep_training = False
                break
    val_epe, best_epe = _run_validation(best_epe, p1_model, val_loaders, device, logger, step, optimizer, save_dir, current_time, scheduler)
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LiteAnyStereo Model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/phase1_best_model.pth', help='Path to the teacher model checkpoint for Phase 2 training')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    
    log_path = './phase1_extended_no_cv_training_RESUME.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    if args.resume:
        resume_ckpt = './checkpoints_extended_run/phase1_extended_oneCycleLRlatest.pth'
    else:
        resume_ckpt = None
    phase1_extended_no_cv(args.ckpt, resume_ckpt)