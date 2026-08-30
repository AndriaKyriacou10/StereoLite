import os
import json
import time
import logging
import argparse
from datetime import datetime
 
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
 
from core.liteanystereo import CustomLiteAnyStereo
from core.training_datasets import SceneFlowDataset, ETH3D, Middlebury, fetch_testing_dataloader, fetch_training_dataloader
from core.augmentor import StereoAugmentor
from core.utils.utils import CustomLogger
 
from train_continuous_two_cycle import sequence_loss, calculate_metrics, evaluate

def save_ckpt(path, model, optimizer, step, best_epe, per_ds, args):
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_epe': best_epe,
        'per_dataset_epe': per_ds,
        'finetune_args': vars(args),
        'source_ckpt': args.ckpt,
    }, path)
 
 
def validate(model, val_loaders, device, cv, logger, step):
    per_ds = {}
    for name, loader in val_loaders.items():
        epe, _ = evaluate(model, loader, device, logger, cv=cv)
        per_ds[name] = epe
        logging.info(f"  step {step}: {name} EPE {epe:.4f}")
    mean_epe = sum(per_ds.values()) / len(per_ds)
    logging.info(f"  step {step}: MEAN EPE {mean_epe:.4f}")
    return mean_epe, per_ds



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    tag = 'cv' if args.compute_cost_volume else 'no_cv'
 
    # the flag must match how the checkpoint was trained; a mismatch runs a model
    # against a configuration it never saw and produces meaningless numbers
    low = args.ckpt.lower()
    if 'no_cv' in low and args.compute_cost_volume:
        logging.warning("ckpt name says no_cv but --compute_cost_volume was passed")
    if 'no_cv' not in low and '_cv' in low and not args.compute_cost_volume:
        logging.warning("ckpt name says cv but --compute_cost_volume was NOT passed")
 
    with open(os.path.join(args.save_dir, f'meta_{tag}.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
 
    model = CustomLiteAnyStereo(scale_right=args.scale_right).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    logging.info(f"loaded {args.ckpt} (step {ckpt.get('step', 'n/a')}), cost_volume={args.compute_cost_volume}")
 
    train_loader = fetch_training_dataloader(
        is_phase_2=False,
        datasets=['sceneflow', 'middlebury', 'eth3d'],
        middlebury_splits=args.middlebury_splits,
        sf_fraction=args.sceneflow_frac,
        num_samples=args.steps * args.batch_size
        )
    
    val_loaders = fetch_testing_dataloader(datasets=['sceneflow', 'middlebury', 'eth3d'])
 
    # fresh optimizer, constant LR: adaptation, not resumed training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
 
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logger = CustomLogger(log_dir=f'./runs/finetune_real_{tag}_{current_time}',
                          flush_freq=100, step=0)
 
    # baseline before any updates -- the number every later result is measured against
    logging.info("=== pre-finetune baseline ===")
    base_epe, base_per_ds = validate(model, val_loaders, device,
                                     args.compute_cost_volume, logger, 0)
    best_epe = base_epe
    save_ckpt(os.path.join(args.save_dir, f'finetune_{tag}_best.pth'),
              model, optimizer, 0, best_epe, base_per_ds, args)
 
    step = 0
    t0 = time.time()
    for data in train_loader:
        model.train()
        _, _, aug1, aug2, disp_gt, valid_mask = data
        img1, img2 = aug1.to(device), aug2.to(device)
        disp_gt, valid_mask = disp_gt.to(device), valid_mask.to(device)
 
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            preds = model(img1, img2, test_mode=False,
                          compute_cost_volume=args.compute_cost_volume)
            loss = sequence_loss(preds, disp_gt, valid_mask)
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        epe, bad = calculate_metrics(preds[-1].detach(), disp_gt, valid_mask)
 
        w = model.context_net.model.conv1.weight.data
        wl, wr = w[:, :3].norm().item(), w[:, 3:].norm().item()
 
        logger.log_batch({'loss': loss.item(), 'epe': epe,
                          'bad_1': bad['bad1'], 'bad_2': bad['bad2'], 'bad_3': bad['bad3'],
                          'learning_rate': optimizer.param_groups[0]['lr'],
                          'ctxnet_w_left_norm': wl, 'ctxnet_w_right_norm': wr,
                          'w_ratio': wr / wl})
 
        step += 1
 
        if step % args.val_freq == 0 or step == args.steps:
            logging.info(f"=== step {step}/{args.steps} ({time.time()-t0:.0f}s) ===")
            mean_epe, per_ds = validate(model, val_loaders, device,
                                        args.compute_cost_volume, logger, step)
            logger.log_batch({'val_epe': mean_epe})
 
            save_ckpt(os.path.join(args.save_dir, f'finetune_{tag}_latest.pth'),
                      model, optimizer, step, best_epe, per_ds, args)
            if mean_epe < best_epe:
                best_epe = mean_epe
                save_ckpt(os.path.join(args.save_dir, f'finetune_{tag}_best.pth'),
                          model, optimizer, step, best_epe, per_ds, args)
                logging.info(f" --> Saved new best model: EPE=({best_epe:.4f})")
 
        if step >= args.steps:
            break
 
    logging.info("\n=== Summary ===")
    for name in base_per_ds:
        logging.info(f"{name:>12}: {base_per_ds[name]:8.4f} -> {per_ds[name]:8.4f} "
                     f"({100*(per_ds[name]/base_per_ds[name]-1):+.1f}%)")
    logger.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='checkpoint to fine-tune from')
    parser.add_argument('--compute_cost_volume', action='store_true',
                   help='MUST match how the checkpoint was trained')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5,
                   help='constant; adaptation LR, ~10x below the 2e-4 training peak')
    parser.add_argument('--sceneflow_frac', type=float, default=0.2,
                   help='fraction of samples drawn from SceneFlow, to limit forgetting')
    parser.add_argument('--middlebury_splits', nargs='+',
                   default=['2005', '2006', '2014', '2021'])
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scale_right', type=float, default=1.0)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--save_dir', default='./finetune_real/checkpoints')
    args = parser.parse_args()
 
    os.makedirs(args.save_dir, exist_ok=True)
    cv = 'cv' if args.compute_cost_volume else 'no_cv'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.save_dir, f'finetune_real_{cv}.log')),
                  logging.StreamHandler()])
    main(args)
    
    # ckpt = torch.load('./finetune_real/finetune_cv_best.pth', map_location='cpu')
    # print(ckpt['step'], ckpt['per_dataset_epe'])