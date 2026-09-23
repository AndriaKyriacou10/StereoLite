import argparse
import torch
import numpy as np
import torch.nn.functional as F

from core.liteanystereo import CustomLiteAnyStereo
from core.stereo_datasets import SceneFlowDataset
from core.utils.utils import InputPadder

def downsample_disp_and_valid(disp_gt, valid_mask, padder, scale_factor=4):
    '''
    init_disp is at 1/4 resolution.
    To compare with ground disparity, need to downsample ground truth by a scale of 4 to match init_disp
    '''
    while disp_gt.ndim > 2:
        disp_gt = disp_gt.squeeze(0)
    while valid_mask.ndim > 2:
        valid_mask = valid_mask.squeeze(0)

    disp_gt = disp_gt.unsqueeze(0).unsqueeze(0)       # -> [1,1,H,W]
    valid_mask = valid_mask.float().unsqueeze(0).unsqueeze(0)  # -> [1,1,H,W]
    
   
    # pad GT the same way img1/img2 were padded → now 544 x 960
    disp_gt_p, valid_mask_p = padder.pad(disp_gt, valid_mask)

    disp_down = F.avg_pool2d(disp_gt_p, kernel_size=scale_factor, stride=scale_factor) / scale_factor
    valid_down = F.avg_pool2d(valid_mask_p, kernel_size=scale_factor, stride=scale_factor)
    valid_down = (valid_down > 0.99)   # only keep fully-valid 4x4 blocks — avoids averaging across occlusion boundaries

    return disp_down.squeeze(), valid_down.squeeze()

@torch.no_grad()
def get_init_disp(model, img1, img2, zero_right=False):
    left = (2 * (img1 / 255.0) - 1.0).contiguous()
    right = (2 * (img2 / 255.0) - 1.0).contiguous()
    if zero_right:
        right = torch.zeros_like(right)
    init_disp, _ = model.context_net(left, right)
    return init_disp

@torch.no_grad()
def run(model, device, dataset, num_samples):
    model.eval()
    
    indices = range(min(num_samples, len(dataset)))
    epe_stereo, epe_mono = [], []
    for idx in indices:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device).bool()
        
        padder = InputPadder(img1.shape, divis_by=32) # 540x960 -> 544x960
        img1_p, img2_p = padder.pad(img1, img2) 
        
        init_disp_stereo = get_init_disp(model, img1_p, img2_p, zero_right=False) #136x240
        init_disp_mono = get_init_disp(model, img1_p, img2_p, zero_right=True) # 136x240
        
        disp_gt_q, valid_mask_q = downsample_disp_and_valid(disp_gt, valid_mask, padder)

        assert init_disp_stereo.squeeze().shape == disp_gt_q.shape

        epe_stereo.append(torch.abs(init_disp_stereo.squeeze()[valid_mask_q] - disp_gt_q[valid_mask_q]).mean().item())
        epe_mono.append(torch.abs(init_disp_mono.squeeze()[valid_mask_q]   - disp_gt_q[valid_mask_q]).mean().item())
    
        if idx % 50 == 0:
            print(f"[{idx}/{len(indices)}] stereo EPE {epe_stereo[-1]:.3f} | mono EPE {epe_mono[-1]:.3f}")
    
    epe_stereo, epe_mono = np.array(epe_stereo), np.array(epe_mono)
    print("\n--- Summary ---")
    print(f"Mean EPE (real right image): {epe_stereo.mean():.4f}")
    print(f"Mean EPE (right image zeroed): {epe_mono.mean():.4f}")
    print(f"Relative degradation: {(epe_mono.mean() - epe_stereo.mean()) / epe_stereo.mean() * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    # parser.add_argument('--compute_cost_volume', action='store_true')
    parser.add_argument('--num_samples', type=int, default=500)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLiteAnyStereo().to(device)
    weights = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(weights['model_state'])
    
    dataset = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    run(model, device, dataset, num_samples=args.num_samples)