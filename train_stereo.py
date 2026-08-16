import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from core.liteanystereo import CustomLiteAnyStereo
import os
import time
from core.training_datasets import fetch_training_dataloader, fetch_testing_dataloader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from core.utils.utils import InputPadder, CustomLogger

def fetch_optimizer(model, learning_rate=0.0002, weight_decay=0.00001, total_steps=100000):
    """Sets up the AdamW optimizer and OneCycle learning rate scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate, total_steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler

def sequence_loss(disparity_preds, disparity_gt, valid_mask, gamma=0.8):
    """Calculates the exponentially weighted L1 loss over all GRU iterations."""
    total_loss = 0.0
    num_predictions = len(disparity_preds)
    
    # Extract only valid ground truth pixels
    valid_gt = disparity_gt[valid_mask.bool().unsqueeze(1)]
    
    for i, pred in enumerate(disparity_preds):
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

def evaluate(model:CustomLiteAnyStereo, val_loader, device, logger=None, cv=True):
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

def train_one_epoch_phase1(model:CustomLiteAnyStereo, optimizer, scheduler, dataloader, device, scaler, logger=None):
    model.train()
    
    
    total_epoch_loss = 0.0
    
    for batch_idx, data in enumerate(dataloader):
        _, _, aug_img1, aug_img2, disp_gt, valid_mask = data
        
        img1 = aug_img1.to(device)
        img2 = aug_img2.to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            disp_preds = model(img1, img2, compute_cost_volume=True)
            loss = sequence_loss(disp_preds, disp_gt, valid_mask)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step Learning Rate of Scheduler
        scheduler.step()
        
        total_epoch_loss += loss.item()
        
        
        final_pred = disp_preds[-1].detach() # Detach so it doesn't drain VRAM
        epe, bad_metrics = calculate_metrics(final_pred, disp_gt, valid_mask)
        
        w = model.context_net.model.conv1.weight.data
        w_left_norm = w[:, :3].norm().item()
        w_right_norm = w[:, 3:].norm().item()
        
        if logger:
            current_metrics = {
                'loss': loss.item(),
                'epe': epe,
                'bad_1': bad_metrics['bad1'],
                'bad_2': bad_metrics['bad2'],
                'bad_3': bad_metrics['bad3'],
                'learning_rate': scheduler.get_last_lr()[0],
                'ctxnet_w_left_norm': w_left_norm,
                'ctxnet_w_right_norm': w_right_norm,
                'w_ratio': w_right_norm / w_left_norm
                        
            }
            
            logger.log_batch(current_metrics)
            
            text_string =  f"Step {logger.global_step} | Loss: {loss.item():.4f} | EPE: {epe:.4f} | Bad-1: {bad_metrics['bad1']:.2f}% | Bad-2: {bad_metrics['bad2']:.2f}% | Bad-3: {bad_metrics['bad3']:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}"
            logger.writer.add_text('Training Metrics', text_string, logger.global_step)
            
    return total_epoch_loss / len(dataloader)

def train_one_epoch_phase2(teacher_model:CustomLiteAnyStereo, student_model:CustomLiteAnyStereo, optimizer, scheduler, dataloader, device, scaler, logger=None):
    student_model.train()
    
    epoch_loss = 0.0
    
    for batch_idx, data in enumerate(dataloader):
        clean_img1, clean_img2, aug_img1, aug_img2, disp_gt, valid_mask = data
        
        clean_img1 = clean_img1.to(device)
        clean_img2 = clean_img2.to(device)
        aug_img1 = aug_img1.to(device)
        aug_img2 = aug_img2.to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device)
        
        optimizer.zero_grad()
        
        # TEACHER
        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                teacher_preds = teacher_model(clean_img1, clean_img2, test_mode=False, compute_cost_volume=True)
                teacher_final_pred = teacher_preds[-1].detach()  # Detach to save VRAM

        # STUDENT
        with torch.autocast(device_type=device.type):
            student_preds = student_model(aug_img1, aug_img2, test_mode=False, compute_cost_volume=False)
            loss = sequence_loss(student_preds, teacher_final_pred, valid_mask)
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        epoch_loss += loss.item()
        
        # Calculate Metrics + Log
        epe, bad_metrics = calculate_metrics(student_preds[-1].detach(), disp_gt, valid_mask)
        
        # Context Network Weights
        w = student_model.context_net.model.conv1.weight.data
        w_left_norm = w[:, :3].norm().item()
        w_right_norm = w[:, 3:].norm().item()
        
        if logger:
            current_metrics = {
                'loss': loss.item(),
                'epe': epe,
                'bad_1': bad_metrics['bad1'],
                'bad_2': bad_metrics['bad2'],
                'bad_3': bad_metrics['bad3'],
                'learning_rate': scheduler.get_last_lr()[0],
                'ctxnet_w_left_norm': w_left_norm,
                'ctxnet_w_right_norm': w_right_norm,
                'w_ratio': w_right_norm / w_left_norm
                
            }
            logger.log_batch(current_metrics)
            text_string =  f"Phase 2 | Step {logger.global_step} | Loss: {loss.item():.4f} | EPE: {epe:.4f} | Bad-1: {bad_metrics['bad1']:.2f}% | Bad-2: {bad_metrics['bad2']:.2f}% | Bad-3: {bad_metrics['bad3']:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}"
            logger.writer.add_text('Training Metrics', text_string, logger.global_step)
    return epoch_loss / len(dataloader)

def phase1_training(epochs, scale_right_weight):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RUN = 'scale_right_' + str(scale_right_weight)
    
    # Best EPE Model Checkpoint Directory
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Inrermediate Checkpoints Directory
    # save_inter = 'checkpoints_inter'
    # os.makedirs(save_inter, exist_ok=True)
    
    ckpt_save_path=f'scale_right_{scale_right_weight}'
    os.makedirs(ckpt_save_path, exist_ok=True)
    
    # Load Dataset
    print("Loading Dataset...")
    train_loader = fetch_training_dataloader(is_phase_2=False, datasets=['sceneflow', 'eth3d', 'middlebury'])
    val_loaders = fetch_testing_dataloader(datasets=['sceneflow', 'eth3d', 'middlebury'], return_occ=False)
    
    # Load Model
    model = CustomLiteAnyStereo(scale_right=scale_right_weight).to(device)
    
    w = model.context_net.model.conv1.weight.data
    w_left_norm = w[:, :3].norm().item()
    w_right_norm = w[:, 3:].norm().item()
            
    total_steps = epochs * len(train_loader)
    optimizer, scheduler = fetch_optimizer(model, learning_rate=0.0002, total_steps=total_steps)
    
    print("Initializing TensorBoard Logger...")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = CustomLogger(log_dir=f"./runs/phase1_training_{RUN}_{current_time}", flush_freq=100)
    
    logger.log_batch({'ctxnet_w_left_norm': w_left_norm, 'ctxnet_w_right_norm': w_right_norm})
    
    
    print('Starting Training Loop...')
    scaler = torch.amp.GradScaler('cuda')
    
    best_epe = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        avg_loss = train_one_epoch_phase1(model, optimizer, scheduler, train_loader, device, scaler, logger)
        end_time = time.time()
        epoch_duration = end_time - start_time
        text_string = f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s"
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")
        
        logger.writer.add_text('Epoch Summary', text_string, epoch+1)
        
        # VALIDATION
        per_dataset_epe = {}
        for name, val_loader in val_loaders.items():
            val_epe, _ = evaluate(model, val_loader, device, logger)
            per_dataset_epe[name] = val_epe
            logger.writer.add_text('Validation - Dataset', f"Epoch {epoch+1}: Dataset: {name} Validation EPE = {val_epe:.4f}", epoch+1)
        
        val_epe = sum(per_dataset_epe.values()) / len(per_dataset_epe)
        logger.log_batch({'val_epe': val_epe})
        logger.writer.add_text('Validation Summary', f"Epoch {epoch+1}: Validation EPE = {val_epe:.4f}", epoch+1)
        
        print(text_string)        
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        
        if epoch % 5 == 0:
            torch.save(checkpoint, f"{ckpt_save_path}/phase1_epoch_{epoch+1}_{RUN}.pth")
        
        if val_epe < best_epe:
            best_epe = val_epe
            torch.save(checkpoint, f"{ckpt_save_path}/phase1_best_model_{RUN}_{current_time}.pth")
            print(f"--> Saved new best model: (EPE: {best_epe:.4f})")
    
    logger.close()  
    print("Phase 1 Training Complete!")
    

def phase2_training(epochs, scale_right_weight, val_freq = 2500, teacher_ckpt='./checkpoints/phase1_best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './checkpoints'
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = fetch_training_dataloader(datasets=['sceneflow', 'eth3d', 'middlebury'], is_phase_2=True)
    
    val_loaders = fetch_testing_dataloader(datasets=['sceneflow', 'eth3d', 'middlebury'], return_occ=False)
    
    teacher_model = CustomLiteAnyStereo(scale_right=scale_right_weight).to(device)
    student_model = CustomLiteAnyStereo(scale_right=scale_right_weight).to(device)
    
    phase1_checkpoint = torch.load(teacher_ckpt, map_location=device)
    teacher_model.load_state_dict(phase1_checkpoint['model_state'])
    student_model.load_state_dict(phase1_checkpoint['model_state'])
    
    
    
    # FREEZE Teacher Model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    total_steps = epochs * len(train_loader)
    
    optimizer, scheduler = fetch_optimizer(student_model, learning_rate=0.0001, total_steps=total_steps)
    
    print("Initializing TensorBoard Logger...")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = CustomLogger(log_dir=f"./runs/phase2_training_{current_time}", flush_freq=100)
    
    w = student_model.context_net.model.conv1.weight.data
    
    logger.log_batch({'ctxnet_w_left_norm': w[:, :3].norm().item(), 'ctxnet_w_right_norm': w[:, 3:].norm().item()})
    
    scaler = torch.amp.GradScaler('cuda')
    
    global_step = 0
    best_epe = float('inf')
    
    student_model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        avg_loss = train_one_epoch_phase2(teacher_model, student_model, optimizer, scheduler, train_loader, device, scaler, logger)
        end_time = time.time()
        
        epoch_duration = end_time - start_time
        text_string = f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s"
        logger.writer.add_text('Phase 2: Epoch Summary', text_string, epoch+1)
        
        print(text_string)     
        global_step += 1
        
        # val_epe, _ = evaluate(student_model, val_loader, device, logger, cv=False)
        # logger.writer.add_text("Phase 2: Validation Summary", f"Epoch {epoch + 1} | Validation EPE={val_epe:.2f}", epoch + 1)     
        # print(f"Phase 2 | Epoch {epoch+1} | Validation EPE: {val_epe:.2f}") 
        
        # VALIDATION
        per_dataset_epe = {}
        for name, val_loader in val_loaders.items():
            val_epe, _ = evaluate(student_model, val_loader, device, logger, cv=False)
            per_dataset_epe[name] = val_epe
            logger.writer.add_text('Validation - Dataset', f"Epoch {epoch+1}: Dataset: {name} Validation EPE = {val_epe:.4f}", epoch+1)
        
        val_epe = sum(per_dataset_epe.values()) / len(per_dataset_epe)
        logger.log_batch({'val_epe': val_epe})
        logger.writer.add_text('Phase 2: Validation Summary', f"Epoch {epoch+1}: Validation EPE = {val_epe:.4f}", epoch+1)
        print(f"Phase 2 | Epoch {epoch+1} | Validation EPE: {val_epe:.2f}") 
    
        checkpoint = {
            'epoch': epoch,
            'model_state': student_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        if val_epe < best_epe:
            best_epe = val_epe
            torch.save(checkpoint, f"{save_dir}/phase2_epoch_{epoch+1}_best_model_RUN2_{current_time}.pth")
            print(f"--> Saved new best model: (EPE: {best_epe:.4f})")
        
        student_model.train()  # Ensure the model is back in training mode after evaluation
    
    logger.close()
    print("Phase 2 Training Complete!")
    
    
def main(epochs, scale_right_weight, teacher_ckpt=None, phase=1):
    if phase == 1:
        print("Starting Phase 1 Training...")
        phase1_training(epochs, scale_right_weight)
    elif phase == 2:
        print('Staring Phase 2 Training...')
        phase2_training(epochs, scale_right_weight, teacher_ckpt=teacher_ckpt)
    print("Training Complete!")
    

    

def test_lr(teacher_ckpt):  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = fetch_training_dataloader(is_phase_2=True, datasets=['sceneflow', 'eth3d', 'middlebury'])
    
    teacher_model = CustomLiteAnyStereo().to(device)
    student_model = CustomLiteAnyStereo().to(device)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.00001)
        
    teacher_checkpoint = torch.load(teacher_ckpt, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state'])
    
    scaler = torch.amp.GradScaler('cuda')
    
    # FREEZE Teacher Model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    student_ckpt = './checkpoints/phase2_best_model_RUN2_contextnet.pth'
    student_checkpoint = torch.load(student_ckpt, map_location=device)
    student_model.load_state_dict(student_checkpoint['model_state'])
    optimizer.load_state_dict(student_checkpoint['optimizer_state'])
    
    for g in optimizer.param_groups:
        g['lr'] = 1e-5
    
    print("Initializing TensorBoard Logger...")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = CustomLogger(log_dir=f"./runs/finetune_phase2{current_time}", flush_freq=100)
    
    w = student_model.context_net.model.conv1.weight.data
    
    logger.log_batch({'ctxnet_w_left_norm': w[:, :3].norm().item(), 'ctxnet_w_right_norm': w[:, 3:].norm().item()})
    
    
    student_model.train()
    for epoch in range(51, 62, 1):
        epoch_loss = 0.0
        for idx, data in enumerate(train_loader):
            clean_img1, clean_img2, aug_img1, aug_img2, disp_gt, valid_mask = data
            clean_img1 = clean_img1.to(device)
            clean_img2 = clean_img2.to(device)
            aug_img1 = aug_img1.to(device)
            aug_img2 = aug_img2.to(device)
            disp_gt = disp_gt.to(device)
            valid_mask = valid_mask.to(device)
            
            optimizer.zero_grad()
            
            # TEACHER
            with torch.no_grad():
                with torch.autocast(device_type=device.type):
                    teacher_preds = teacher_model(clean_img1, clean_img2, test_mode=False, compute_cost_volume=True)
                    teacher_final_pred = teacher_preds[-1].detach()  # Detach to save VRAM

            # STUDENT
            with torch.autocast(device_type=device.type):
                student_preds = student_model(aug_img1, aug_img2, test_mode=False, compute_cost_volume=False)
                loss = sequence_loss(student_preds, teacher_final_pred, valid_mask)
            
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            # Calculate Metrics + Log
            epe, bad_metrics = calculate_metrics(student_preds[-1].detach(), disp_gt, valid_mask)
            
            # Context Network Weights
            w = student_model.context_net.model.conv1.weight.data
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
                text_string =  f"Finetune | Step {logger.global_step} | Loss: {loss.item():.4f} | EPE: {epe:.4f} | Bad-1: {bad_metrics['bad1']:.2f}% | Bad-2: {bad_metrics['bad2']:.2f}% | Bad-3: {bad_metrics['bad3']:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
                logger.writer.add_text('Training Metrics', text_string, logger.global_step)
        avg_loss = epoch_loss / len(train_loader)
        text_string = f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}"
        logger.writer.add_text('Finetune: Epoch Summary', text_string, epoch+1)
        
        print(text_string)                    
        
        
        
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LiteAnyStereo Model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--teacher_ckpt', type=str, default='./checkpoints/phase1_best_model.pth', help='Path to the teacher model checkpoint for Phase 2 training')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='Phase of training: 1 for Phase 1, 2 for Phase 2')
    parser.add_argument('--scale_right_weight', type=float, default=0.3, help='Scaling factor for the right image weights in the context network')
    args = parser.parse_args()
        
    main(args.epochs,  args.scale_right_weight, args.teacher_ckpt, args.phase)
    
    # test_lr(teacher_ckpt)  # Call the test_lr function after training is complete