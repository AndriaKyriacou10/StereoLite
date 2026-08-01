import argparse
import torch
import numpy as np
from core.liteanystereo import CustomLiteAnyStereo
from core.training_datasets import SceneFlowDataset
from core.utils.utils import InputPadder
from core.submodule import build_correlation_volume
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_error_vis(left_img, disp_pred, disp_gt, init_disp, valid_mask, save_path):
    
    left_img = left_img.cpu().squeeze().permute(1, 2, 0).numpy() / 255.0
    disp_pred = disp_pred.cpu().squeeze().numpy()
    disp_gt = disp_gt.cpu().squeeze().numpy()
    init_disp = init_disp.cpu().squeeze().numpy()
    vm = valid_mask.squeeze().cpu().bool().numpy()
    error = np.abs(disp_pred - disp_gt)
    error[~vm] = 0.0
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(left_img); axes[0].set_title('Left Image'); axes[0].axis('off')
    
    vmax = np.percentile(disp_gt[vm], 99)  # shared scale for GT/pred, robust to outliers
    im1 = axes[1].imshow(disp_gt, cmap='magma', vmin=0, vmax=vmax); axes[1].set_title("GT Disparity"); axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(disp_pred, cmap='magma', vmin=0, vmax=vmax); axes[2].set_title("Predicted Disparity"); axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(error, cmap='hot', vmin=0, vmax=np.percentile(error[vm], 95))
    axes[3].set_title("Abs Error"); axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    error_init = np.abs(init_disp - disp_pred)
    error_init[~vm] = 0.0
    im4 = axes[4].imshow(error_init, cmap='hot', vmin=0, vmax=np.percentile(error_init[vm], 95))
    axes[4].set_title('Init Disp Error'); axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def visualize(model, dataset, device, samples, compute_cost_volume):
    model.eval()
    def _normalize(x):
        return (2 * (x / 255.0) - 1.0).contiguous()
    
    indices_all = np.array(range(min(samples, len(dataset))))
    indices = indices_all[indices_all % 50 == 0]
    for idx in indices:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape, divis_by=32)
        img1_p, img2_p = padder.pad(img1, img2)
        
        init_disp, _ = model.context_net(_normalize(img1_p), _normalize(img2_p))
        init_disp_up = F.interpolate(init_disp, scale_factor=4, mode='bilinear', align_corners=False) * 4.0
        init_disp_up = padder.unpad(init_disp_up)
        
        disp_pred = model(img1_p, img2_p, test_mode=True, compute_cost_volume = compute_cost_volume)
        disp_pred = padder.unpad(disp_pred)
        save_error_vis(img1, disp_pred, disp_gt, init_disp_up, valid_mask, save_path=f'./context_net_tests/sample_{idx}.png')
        print(f'Saved sample {idx}')

@torch.no_grad()
def get_disp_isolated(model, img1, img2, compute_cost_volume, zero_context_right=False):
    left = (2 * (img1 / 255.0) - 1.0).contiguous()
    right = (2 * (img2 / 255.0) - 1.0).contiguous()

    ctx_right_input = torch.zeros_like(right) if zero_context_right else right
    init_disp, context_features = model.context_net(left, ctx_right_input)

    if compute_cost_volume:
        features_left = model.fnet(left)
        features_right = model.fnet(right)   # always the REAL right image to compute accurate COST VOLUME
        cost_volume = build_correlation_volume(features_left[0], features_right[0], 192 // 4)
        cv_3d = model.cost_stem_3d(cost_volume[:, None]).squeeze(1)
        cv = model.cost_agg_2d(cv_3d, features_left)
    else:
        cv = torch.zeros((left.shape[0], 48, left.shape[2] // 4, left.shape[3] // 4),
                          device=left.device, dtype=left.dtype)

    disp = init_disp
    hidden_state = context_features
    for _ in range(8):
        gru_input = torch.cat((disp, cv, context_features), dim=1)
        hidden_state = model.conv_gru(hidden_state, gru_input)
        delta = model.disp_head(hidden_state)
        disp = disp + delta
        mask = model.mask_head(hidden_state)
        disp_up = model.upsample_disp(disp, mask)
    return disp_up

@torch.no_grad()
def run(model, device, dataset, n_samples, compute_cost_volume):
    model.eval()
    epe_stereo, epe_mono = [], []

    indices = range(min(n_samples, len(dataset)))
    for idx in indices:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device).bool()

        padder = InputPadder(img1.shape, divis_by=32)
        img1_p, img2_p = padder.pad(img1, img2)
        
        pred_stereo = get_disp_isolated(model, img1_p, img2_p, compute_cost_volume=compute_cost_volume, zero_context_right=False)
        pred_mono   = get_disp_isolated(model, img1_p, img2_p, compute_cost_volume=compute_cost_volume, zero_context_right=True)

        pred_stereo = padder.unpad(pred_stereo).squeeze()
        pred_mono   = padder.unpad(pred_mono).squeeze()

        vm = valid_mask.squeeze()
        gt = disp_gt.squeeze()

        epe_stereo.append(torch.abs(pred_stereo[vm] - gt[vm]).mean().item())
        epe_mono.append(torch.abs(pred_mono[vm] - gt[vm]).mean().item())

        if idx % 50 == 0:
            print(f"[{idx}/{len(indices)}] stereo EPE {epe_stereo[-1]:.3f} | mono EPE {epe_mono[-1]:.3f}")

    epe_stereo, epe_mono = np.array(epe_stereo), np.array(epe_mono)
    print("\n--- Summary ---")
    print(f"Mean EPE (real right image): {epe_stereo.mean():.4f}")
    print(f"Mean EPE (right image zeroed): {epe_mono.mean():.4f}")
    print(f"Relative degradation: {(epe_mono.mean() - epe_stereo.mean()) / epe_stereo.mean() * 100:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--compute_cost_volume', action='store_true')
    parser.add_argument('--mode', default='run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomLiteAnyStereo().to(device)
    weights = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(weights['model_state'])

    dataset = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    
    if args.mode.lower() == 'run': 
        run(model, device, dataset, args.n_samples, args.compute_cost_volume)
    else:
        visualize(model, dataset, device, args.n_samples, args.compute_cost_volume)