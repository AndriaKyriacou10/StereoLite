import argparse
import logging
from sympy import to_cnf
import torch
import numpy as np
from core.liteanystereo import CustomLiteAnyStereo, original_LAS
from core.training_datasets import ETH3D, fetch_testing_dataloader, Middlebury, SceneFlowDataset
from core.utils.utils import InputPadder
from PIL import Image

@torch.no_grad()
def validate_flyingthings(model, cost_volume=False, layer2=False):
    model.eval()

    val_dataset = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    out_list, epe_list = [], []

    for idx in range(len(val_dataset)):
        img1, img2, _, _, disp_gt, valid_mask = val_dataset[idx]

        if torch.isnan(disp_gt).any() or torch.isinf(disp_gt).any():
            logging.warning(f"[{idx}] disp_gt has NaN/Inf — skipping")
            continue

        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)

        if isinstance(model, CustomLiteAnyStereo):
            disp_pred = model(img1, img2, test_mode=True, compute_cost_volume=cost_volume)
        else:
            disp_pred = model(img1, img2, test_mode=True)

        if torch.isnan(disp_pred).any():
            logging.warning(f"[{idx}] model produced NaN output — skipping")
            continue

        disp_pred = padder.unpad(disp_pred).squeeze().cpu()
        assert disp_pred.shape == disp_gt.squeeze().shape

        epe_map = torch.abs(disp_pred - disp_gt.squeeze())
        epe_flattened = epe_map.flatten()

        val = valid_mask.flatten() >= 0.5
        n_valid = val.sum().item()

        if n_valid == 0:
            logging.warning(f"[{idx}] zero valid pixels (max disp_gt={disp_gt.max().item():.1f}) — skipping")
            continue

        outliers = (epe_flattened > 1.0)
        image_out = outliers[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()

        logging.info(f"FlyingThings3D {idx+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Bad1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)
    
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation FlyingThings3D: EPE {epe}, Bad1 {d1}")
    return {'flyingthings-epe': epe, 'flyingthings-d1': d1}

@torch.no_grad()
def validate_eth3d(model, cost_volume = False, layer2 = False):
    model.eval()
    
    val_dataset = ETH3D(condition = 'test', return_occ = True, train_frac=0.5)
    out_list, epe_list = [], []
    
    for idx in range(len(val_dataset)):
        img1, img2, _, _, disp_gt, valid_mask, occ_file = val_dataset[idx]
        
        # Add batch dimension
        img1 = img1.unsqueeze(0).to(device) 
        img2 = img2.unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        
        if isinstance(model, CustomLiteAnyStereo):
            disp_pred = model(img1, img2, test_mode = True, compute_cost_volume = cost_volume)
        else:
            disp_pred = model(img1, img2, test_mode = True)
            
        disp_pred = padder.unpad(disp_pred).squeeze().cpu()
        
        assert disp_pred.shape == disp_gt.squeeze().shape
        
        epe_map = torch.abs(disp_pred - disp_gt.squeeze())

        # 2. Flatten the error and ground truth tensors to 1D lists of pixels
        epe_flattened = epe_map.flatten()
        
        occ_mask = Image.open(occ_file)
        occ_mask = np.ascontiguousarray(occ_mask).flatten()
        occ_tensor = torch.from_numpy(occ_mask == 255).bool()
        
        val = (valid_mask.flatten() >= 0.5) & occ_tensor 
        
        # Bad1 error
        outliers = (epe_flattened > 1.0)
        image_out = outliers[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()

        logging.info(f"ETH3D {idx+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Bad1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation ETH3D: EPE {epe}, Bad1 {d1}")
    return {'eth3d-epe': epe, 'eth3d-d1': d1}

@torch.no_grad()
def validate_middlebury(model, split='MiddEval3', resolution='F', cost_volume = False, layer2 = False):
    model.eval()

    
    val_dataset = Middlebury(split=split, resolution=resolution)
    
    i = 0
    out_list, epe_list = [], []
    for idx in range(len(val_dataset)):
        img1, img2, _, _, disp_gt, valid_mask = val_dataset[idx]

        # Add batch dimension
        img1 = img1.unsqueeze(0).to(device) 
        img2 = img2.unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape, divis_by=32)
        img1, img2 = padder.pad(img1, img2)
        
        if isinstance(model, CustomLiteAnyStereo):
            disp_pred = model(img1, img2, test_mode = True, compute_cost_volume = cost_volume)
        else:
            disp_pred = model(img1, img2, test_mode = True)
        
        disp_pred = padder.unpad(disp_pred).squeeze().cpu()
        
        epe_map = torch.abs(disp_pred - disp_gt.squeeze())

        # 2. Flatten the error and ground truth tensors to 1D lists of pixels
        epe_flattened = epe_map.flatten()
        disp_gt_flattened = disp_gt.flatten()

        # 3. Apply the exact logic filter combination from the original repo
        # (Ensuring it's safe for stereo by removing the multi-channel optical flow index)
        val_mask = (valid_mask.flatten() >= 0.5) & (disp_gt_flattened < 192)

        # 4. Extract metrics cleanly
        image_epe = epe_flattened[val_mask].mean().item()
        epe_list.append(image_epe)
        
        # 5. Extract the standard Middlebury "Bad 2.0" / D1 outlier metric (Outliers > 2 pixels)
        outliers = (epe_flattened > 2.0)
        image_bad2 = outliers[val_mask].float().mean().item()
        out_list.append(image_bad2)
        
        logging.info(f"Middlebury Iter {idx+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} Bad2 {round(image_bad2,4)}")
        print(f"Valid pixels: {val_mask.sum().item()} / {val_mask.numel()} ({100*val_mask.float().mean().item():.1f}%)")
        
    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}_{resolution}_192: EPE {epe}, Bad2 {d1}")
    return {f'middlebury{split}_{resolution}-epe': epe, f'middlebury{split}-d1': d1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='Restore Checkpoint', default='./checkpoints/phase2_best_model.pth')
    parser.add_argument('--dataset', help='dataset for evaluation', choices=['eth3d', 'sceneflow'] +[f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--cost_volume', action='store_true', help='Compute cost volume during inference')
    parser.add_argument('--model', choices=['Custom', 'Original'], default='Custom', help='Model type to use for evaluation')
    parser.add_argument('--layer2', action='store_true', help='Use ContextNet with a second layer')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'Custom':
    
        model = CustomLiteAnyStereo(layer2=args.layer2)
        if args.ckpt is not None:
            logging.info("Loading checkpoint...")
            weights = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(weights['model_state'])
    else:
        model = original_LAS()
        if args.ckpt is not None:
            assert args.ckpt.endswith(".pth")
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(args.ckpt, map_location=device)

            target_model = model.module if hasattr(model, 'module') else model
            target_model.load_state_dict(checkpoint, strict=True)
            logging.info(f"Done loading checkpoint")
    

    model.to(device)
    model.eval()   
    
    if args.dataset == 'eth3d':
        validate_eth3d(model, cost_volume = args.cost_volume, layer2 = args.layer2)
    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, cost_volume = args.cost_volume, resolution = args.dataset[-1], layer2 = args.layer2)
    elif args.dataset == 'sceneflow':
        validate_flyingthings(model, cost_volume = args.cost_volume, layer2 = args.layer2)