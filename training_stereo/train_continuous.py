'''
Instead of two phases, train the model continuously, and at a specific training step CUT OFF the COST VOLUME. 
For the remaining training step, the model will be trained without the COST VOLUME.
Similar to the ReCoVEr paper, training approach
'''

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
import json

def fetch_optimizer(model, args):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.lr, total_steps=args.total_steps, pct_start=args.pct, cycle_momentum=False, anneal_strategy='linear'
        )
    
    return optimizer, scheduler


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

def run_validation(best_epe, model, val_loaders, device, logger, step, optimizer, save_dir, current_time, scheduler, cv=False, log_dir=None):
    per_dataset_epe = {}
    run = 'With CV' if cv else 'No CV'
    
    for name, val_loader in val_loaders.items():
        val_epe, _ = evaluate(model, val_loader, device, logger, cv=cv)
        per_dataset_epe[name] = val_epe
        val_text = f"Step {step+1}: Dataset: {name} Validation EPE = {val_epe:.4f}"
        logger.writer.add_text('Validation - Dataset', val_text, step+1)
        logging.info(val_text)

    val_epe = sum(per_dataset_epe.values()) / len(per_dataset_epe)
    logger.log_batch({'val_epe': val_epe})
    logger.writer.add_text(f'{run}: Validation Summary', f"Step {step+1}: Validation EPE = {val_epe:.4f}", step+1)
    logging.info(f"{run} | Step {step+1} | Validation EPE: {val_epe:.2f}") 
    
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'log_dir': log_dir
    }
    
    tag = 'latest'
    cv_tag = 'cv' if cv else 'no_cv'
    if val_epe < best_epe:
        tag = 'best'
        best_epe = val_epe
        logging.info(f"--> Saved new best model: (EPE: {best_epe:.4f})")
    
    checkpoint[f'best_epe_{cv_tag}'] = best_epe
    
    if tag == 'best':
        torch.save(checkpoint, f"{save_dir}/train_continuous_{tag}_{cv_tag}_{current_time}.pth")
    torch.save(checkpoint, f"{save_dir}/train_continuous_latest_{cv_tag}.pth")
    return val_epe, best_epe
    
def train(args):
    with open(args.save_dir + "/meta.json", "w") as file:
        json.dump(vars(args), file, sort_keys=True, indent=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = StereoLite(scale_right=args.scale_right).to(device)
    
    train_loader = fetch_training_dataloader(is_phase_2=False, datasets=['sceneflow', 'middlebury', 'eth3d'])
    val_loader = fetch_testing_dataloader(datasets=['sceneflow', 'middlebury', 'eth3d'])
    
    logging.info(f'Loaded Training Dataloader | Image Pairs = {len(train_loader)}')
    
    optimizer, scheduler = fetch_optimizer(model, args)
    
    if args.resume_ckpt:
        logging.info(f"Resuming training from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        step = checkpoint.get('step', 0)
        best_epe_cv = checkpoint.get('best_epe_cv', float('inf'))
        best_epe_no_cv = checkpoint.get('best_epe_no_cv', float('inf'))

        if step < args.cutoff_step and best_epe_cv == float('inf'):
            logging.warning("Best EPE with CV not found in checkpoint. Initializing to infinity.")
        elif step >= args.cutoff_step and best_epe_no_cv == float('inf'):
            logging.warning("Best EPE without CV not found in checkpoint. Initializing to infinity.")
                
        model.load_state_dict(checkpoint['model_state'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scheduler.step()
        
        epochs = step // len(train_loader)
        logging.info(f"Resuming from step: {step}")
        
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logger = CustomLogger(log_dir=checkpoint.get('log_dir', f'./runs/train_continuous_{current_time}'), flush_freq=100, step = step)
        
    else:
        step = 0
        best_epe_cv = float('inf')
        best_epe_no_cv = float('inf')
        epochs = 0
        
        logging.info("Initializing TensorBoard Logger...")
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logger = CustomLogger(log_dir=f'./runs/train_continuous_{current_time}_cutoff_90k', flush_freq=100, step = step)
    
    
    logging.info(f"Starting training from step: {step}")
    
    scaler = torch.amp.GradScaler('cuda')
    

    w = model.context_net.model.conv1.weight.data
    logger.log_batch({'ctxnet_w_left_norm': w[:, :3].norm().item(), 'ctxnet_w_right_norm': w[:, 3:].norm().item()})
    
    val_freq = len(train_loader)
    
    keep_training = True
    
    cut_off_step = args.cutoff_step 

    while keep_training:
        epochs += 1
        start_time = time.time()
        
        for idx, data in enumerate(train_loader):
            model.train()
            _, _, aug1, aug2, disp_gt, valid_mask = data
            
            img1 = aug1.to(device)
            img2 = aug2.to(device)
            disp_gt = disp_gt.to(device)
            valid_mask = valid_mask.to(device)
            
            
            if step < cut_off_step:
                compute_cost_volume = True
                best_epe = best_epe_cv
            else:
                compute_cost_volume = False
                best_epe = best_epe_no_cv
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type):
                disp_preds = model(img1, img2, test_mode=False, compute_cost_volume = compute_cost_volume)
                loss = sequence_loss(disp_preds, disp_gt, valid_mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            final_pred = disp_preds[-1].detach() # Detach so it doesn't drain VRAM
            epe, bad_metrics = calculate_metrics(final_pred, disp_gt, valid_mask)
            
            w = model.context_net.model.conv1.weight.data
            w_left_norm = w[:, :3].norm().item()
            w_right_norm = w[:, 3:].norm().item()
            
            if logger:
                current_metrics = {
                    'cv_active': 1.0 if step < cut_off_step else 0.0,
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
                val_epe, best_epe = run_validation(best_epe, model, val_loader, device, logger, step, optimizer, save_dir=args.save_dir, current_time=current_time, scheduler=scheduler, cv=compute_cost_volume, log_dir=logger.log_dir)

                if compute_cost_volume:
                    best_epe_cv = best_epe
                else:
                    best_epe_no_cv = best_epe
            
            step += 1
            if step >= args.total_steps:
                keep_training = False
                break
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Epoch {epochs} completed in {duration:.2f} seconds.")
            
    val_epe, best_epe = run_validation(best_epe, model, val_loader, device, logger, step, optimizer, save_dir=args.save_dir, current_time=current_time, scheduler=scheduler, cv=compute_cost_volume)
    best_epe_no_cv = best_epe
    logging.info(f"Training Completed. Best EPE with CV: {best_epe_cv:.4f}, Best EPE without CV: {best_epe_no_cv:.4f}")
    logger.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=450000)
    parser.add_argument('--cutoff_step', type=int, default=90000, help='Step at which to cut off the cost volume')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--scale_right', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--pct', type=float, default=0.01)
    parser.add_argument('--save_dir', default='./continuous_training/checkpoints_continuous')
    parser.add_argument('--resume_ckpt', default=None, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_path = './continuous_training/continuous_training(RECOVER).log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    train(args)