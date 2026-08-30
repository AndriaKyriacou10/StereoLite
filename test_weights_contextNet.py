import argparse
import torch
import numpy as np
from core.liteanystereo import CustomLiteAnyStereo
from core.training_datasets import SceneFlowDataset, ETH3D, Middlebury
from core.utils.utils import InputPadder
from core.submodule import build_correlation_volume
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
RF_LIMIT = 29.0  # ContextNet theoretical receptive field radius, full-res px

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
def binned_error(model, device, dataset, n_samples, compute_cost_volume,
                 name='sceneflow', max_disp=192, bin_width=4, min_count=500,
                 save_dir='./context_net_tests/two_cycle'):
    model.eval()
    cv = "cv" if compute_cost_volume else "no_cv"
    
    def _normalize(x):
        return (2 * (x / 255.0) - 1.0).contiguous()
 
    bins = np.arange(0, max_disp + bin_width, bin_width)
    nb = len(bins) - 1
 
    # streaming accumulators -- constant memory, and every valid pixel counts.
    # subsampling would preserve FT3D's low-disparity skew and starve exactly the
    # high-disparity bins where the knee should appear.
    acc = {k: {'abs': np.zeros(nb), 'rel': np.zeros(nb), 'n': np.zeros(nb, dtype=np.int64)}
           for k in ('init', 'final')}
    
    
    img_means, img_stds = [], []
    n_tot = 0
    img_means, img_stds = [], []
    all_init, all_gt = [], []
    
    for idx in range(min(n_samples, len(dataset))):
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device).bool()
 
        padder = InputPadder(img1.shape, divis_by=32)
        img1_p, img2_p = padder.pad(img1, img2)
 
        # init_disp is 1/4 res AND in 1/4-res disparity units -- upsample_disp applies
        # `scale * disp` internally, so the x4 here is required, not cosmetic.
        init_disp, _ = model.context_net(_normalize(img1_p), _normalize(img2_p))
        init_up = F.interpolate(init_disp, scale_factor=4, mode='bilinear',
                                align_corners=False) * 4.0
        init_up = padder.unpad(init_up).squeeze()
 
        final_disp = model(img1_p, img2_p, test_mode=True, compute_cost_volume=compute_cost_volume)
        final_disp = padder.unpad(final_disp).squeeze()
 
        gt = disp_gt.squeeze()
        vm = valid_mask.squeeze() & (disp_gt.squeeze() > 0) & (disp_gt.squeeze() < max_disp)
        if not vm.any():
            continue
        
        init_valid = init_up[vm].float().cpu().numpy()
        gt_valid = gt[vm].float().cpu().numpy()
        
        img_means.append(init_valid.mean())
        img_stds.append(init_valid.std())
        
        if init_valid.size > 5000:
            sel = np.random.choice(init_valid.size, 5000, replace=False)
            all_init.append(init_valid[sel])
            all_gt.append(gt_valid[sel])
        else:
            all_init.append(init_valid)
            all_gt.append(gt_valid)
        
        g = gt[vm].float().cpu().numpy()
        bi = np.digitize(g, bins) - 1 # bucket that GT pixel falls into
        ok = (bi >= 0) & (bi < nb)
        bi, g = bi[ok], g[ok]
 
        for key, pred in (('init', init_up), ('final', final_disp)):
            e = torch.abs(pred[vm] - gt[vm]).float().cpu().numpy()[ok]
            acc[key]['abs'] += np.bincount(bi, weights=e, minlength=nb)
            acc[key]['rel'] += np.bincount(bi, weights=e / g, minlength=nb)
            acc[key]['n'] += np.bincount(bi, minlength=nb)
 
        if idx % 50 == 0:
            print(f"[{idx}/{min(n_samples, len(dataset))}] {name}")
 
    centers = (bins[:-1] + bins[1:]) / 2.0
    keep = acc['init']['n'] >= min_count  # sparse bins give wild means at exactly the
                                          # high-disparity end you're inspecting
    out = {}
    for key in ('init', 'final'):
        n = np.maximum(acc[key]['n'], 1)
        out[key] = {'abs': acc[key]['abs'] / n, 'rel': acc[key]['rel'] / n}
 
    counts = acc['init']['n']
 
    print(f"\n{'bin':>10} {'n':>10} {'init |e|':>10} {'init rel':>9} {'final |e|':>10}")
    for i in np.where(keep)[0]:
        print(f"{bins[i]:>4.0f}-{bins[i+1]:>4.0f} {counts[i]:>10,} "
              f"{out['init']['abs'][i]:>10.3f} {out['init']['rel'][i]:>9.3f} "
              f"{out['final']['abs'][i]:>10.3f}")
 
    plot_binned(centers[keep], out, keep, counts,
                f"{save_dir}/{name}_binned_init_error_{cv}.png", name)
    np.savez(f"{save_dir}/{name}_binned_init_error_{cv}.npz",
             centers=centers, keep=keep, counts=counts,
             init_abs=out['init']['abs'], init_rel=out['init']['rel'],
             final_abs=out['final']['abs'], final_rel=out['final']['rel'])
    
    
    img_means, img_stds = np.array(img_means), np.array(img_stds)
    all_init, all_gt = np.concatenate(all_init), np.concatenate(all_gt)
    r = np.corrcoef(all_init, all_gt)[0, 1]


    print(f"init_disp global mean      : {all_init.mean():.4f} px")
    print(f"global std                 : {all_init.std():.4f} px")
    print(f"within-image std (mean)    : {img_stds.mean():.4f} px")
    print(f"std of per-image means     : {img_means.std():.4f} px")
    print(f"per-image mean range       : {img_means.min():.3f} .. {img_means.max():.3f}")
    print(f"pooled Pearson r vs GT     : {r:.4f}")
    
    return centers, out, counts

def plot_binned(centers, out, keep, counts, save_path, name):
    """Stacked panels sharing x, so the 29px line sits at the same place in all three.
    Separate panels rather than overlay: px error, a dimensionless ratio, and counts
    spanning orders of magnitude cannot share a y-axis."""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 9),
                           gridspec_kw={'height_ratios': [3, 2, 1.2]})
 
    ax[0].plot(centers, out['init']['abs'][keep], 'o-', color='#2a78d6',
               label='init_disp (ContextNet)')
    ax[0].plot(centers, out['final']['abs'][keep], 's--', color='#1baf7a',
               label='final disp (after GRU)')
    ax[0].set_ylabel('mean |pred - GT|  (px)')
    ax[0].legend(fontsize=9)
 
    ax[1].plot(centers, out['init']['rel'][keep], 'o-', color='#eb6834')
    ax[1].set_ylabel('relative error')
 
    all_centers = centers
    ax[2].bar(all_centers, counts[keep], width=3.5, color='#888780')
    ax[2].set_yscale('log')
    ax[2].set_ylabel('pixels')
    ax[2].set_xlabel('GT disparity (px)')
 
    for a in ax:
        a.axvline(RF_LIMIT, color='r', ls='--', lw=1.2)
        a.grid(alpha=0.3)
    ax[0].text(RF_LIMIT + 2, ax[0].get_ylim()[1] * 0.9,
               f'RF limit (±{RF_LIMIT:.0f}px)', color='r', fontsize=9)
    ax[0].set_title(f'{name}: init_disp error vs GT disparity')
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"wrote {save_path}")
    
@torch.no_grad()
def visualize(model, dataset, device, samples, compute_cost_volume, name="sceneflow"):
    model.eval()
    def _normalize(x):
        return (2 * (x / 255.0) - 1.0).contiguous()
    
    random.seed(42)
    vis_samples = 10
    indices = [random.randint(0, len(dataset) - 1) for _ in range(vis_samples)]
    cv = "cv" if compute_cost_volume else "no_cv"
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
        save_error_vis(img1, disp_pred, disp_gt, init_disp_up, valid_mask, save_path=f'./context_net_tests/two_cycle/{name}/sample_{idx}_{cv}.png')
        
        
        print(f'Saved sample {idx}')

@torch.no_grad()
def get_disp_isolated(model, img1, img2, compute_cost_volume, zero_context_right=False, zero_context_left=False):
    left = (2 * (img1 / 255.0) - 1.0).contiguous()
    right = (2 * (img2 / 255.0) - 1.0).contiguous()

    ctx_left_input = torch.zeros_like(left) if zero_context_left else left
    ctx_right_input = torch.zeros_like(right) if zero_context_right else right
    init_disp, context_features = model.context_net(ctx_left_input, ctx_right_input)

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
    CONDITIONS = {
        'no_right': dict(zero_context_right=True, zero_context_left=False),
        'no_left': dict(zero_context_right=False, zero_context_left=True),
        'both_zero': dict(zero_context_right=True, zero_context_left=True),
        'base': dict(zero_context_right=False, zero_context_left=False)
    }
    epe_all = defaultdict(list)
    for idx in indices:
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        disp_gt = disp_gt.to(device)
        valid_mask = valid_mask.to(device).bool()

        padder = InputPadder(img1.shape, divis_by=32)
        img1_p, img2_p = padder.pad(img1, img2)
        
        for condition, kwargs in CONDITIONS.items():
            pred_disp = get_disp_isolated(model, img1_p, img2_p, compute_cost_volume=compute_cost_volume, **kwargs)
            pred_disp = padder.unpad(pred_disp).squeeze()
            vm = valid_mask.squeeze() 
            epe = torch.abs(pred_disp[vm] - disp_gt.squeeze()[vm]).mean().item()
            epe_all[condition].append(epe)
        
    base = np.array(epe_all['base'])
    for name in CONDITIONS:
        epe_arr = np.array(epe_all[name])
        delta = epe_arr - base
        print(f"{name:10s} EPE {epe_arr.mean():7.4f}  Δ {delta.mean():+7.4f}  "
          f"worse on {(delta > 0).sum()}/{len(delta)} samples")
    
    cnxNet = 'context_net.model.conv1.weight'
    ckpt_no_cv_path = './continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_best_no_cv_Aug15_14-32-26.pth'
    ckpt_cv_path = './continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_best_cv_Aug15_14-32-26.pth'
    with_cv = torch.load(ckpt_cv_path, map_location='cpu')['model_state'][cnxNet]
    no_cv = torch.load(ckpt_no_cv_path, map_location='cpu')['model_state'][cnxNet]
    
    for tag, sl in [('left', slice(0, 3)), ('right', slice(3, 6))]:
        x, y = with_cv[:, sl].flatten(1), no_cv[:, sl].flatten(1)
        cos = F.cosine_similarity(x, y, dim=1)
        print(f"{tag:5s}  cos mean {cos.mean():.3f}  min {cos.min():.3f}"
          f"norm {x.norm(dim=1).mean():.2f} -> {y.norm(dim=1).mean():.2f}")
    # epe_stereo, epe_mono = np.array(epe_stereo), np.array(epe_mono)
    # print("\n--- Summary ---")
    # print(f"Mean EPE (real right image): {epe_stereo.mean():.4f}")
    # print(f"Mean EPE (right image zeroed): {epe_mono.mean():.4f}")
    # print(f"Relative degradation: {(epe_mono.mean() - epe_stereo.mean()) / epe_stereo.mean() * 100:.1f}%")

@torch.no_grad()
def per_scene(model, device, dataset, compute_cost_volume, max_disp=192, sat_thresh=0.9):
    rows = []
    cv = "cv" if compute_cost_volume else "no_cv"
    for idx in range(len(dataset)):
        img1, img2, _, _, disp_gt, valid_mask = dataset[idx]
        img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
        disp_gt, valid_mask = disp_gt.to(device), valid_mask.to(device).bool()

        padder = InputPadder(img1.shape, divis_by=32)
        pred = padder.unpad(model(*padder.pad(img1, img2), test_mode=True,
                                  compute_cost_volume=compute_cost_volume)).squeeze()

        vm = valid_mask.squeeze() & (disp_gt.squeeze() > 0) & (disp_gt.squeeze() < max_disp)
        p, g = pred[vm], disp_gt.squeeze()[vm]

        rows.append({
            'idx': idx,
            'epe': torch.abs(p - g).mean().item(),
            'sat': (p > sat_thresh * max_disp).float().mean().item() * 100,
            'gt_max': g.max().item(),
        })
    plot_per_scene(rows, save_path=f'./context_net_tests/two_cycle/per_scene_{dataset.__class__.__name__}_{cv}.png')

def plot_per_scene(rows, save_path):
    rows.sort(key=lambda r: r['epe'])
    names = [r.get('name', str(r['idx'])) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(rows)), [r['epe'] for r in rows], color='#2a78d6')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('EPE (px)')
    ax2 = ax.twinx()
    ax2.plot(range(len(rows)), [r['sat'] for r in rows], 'o--', color='#eb6834')
    ax2.set_ylabel('% pixels near max_disp', color='#eb6834')
    plt.savefig(save_path, dpi=140)
    
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

    dataset_SF = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    dataset_eth = ETH3D(condition='test')
    dataset_mb = Middlebury(split='MiddEval3', resolution='H')
    if args.mode.lower() == 'run': 
        run(model, device, dataset_SF, args.n_samples, args.compute_cost_volume)
    elif args.mode.lower() == 'binned':
        binned_error(model, device, dataset_SF, args.n_samples, args.compute_cost_volume, name="sceneflow")
        binned_error(model, device, dataset_eth, args.n_samples, args.compute_cost_volume, name="eth3d", bin_width=2)
        binned_error(model, device, dataset_mb, args.n_samples, args.compute_cost_volume, name="mb")
    else:
        visualize(model, dataset_SF, device, args.n_samples, args.compute_cost_volume, name="sceneflow")
        visualize(model, dataset_eth, device, args.n_samples, args.compute_cost_volume, name="eth3d")
        visualize(model, dataset_mb, device, args.n_samples, args.compute_cost_volume, name="middlebury")
        
    if args.mode.lower() == 'per_scene':
        per_scene(model, device, dataset_mb, args.compute_cost_volume)