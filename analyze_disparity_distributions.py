'''
Diagnostic: predicted vs ground-truth disparity distributions, per dataset, for the
CV and no-CV checkpoints of the continuous training run.

Motivation
----------
Validation EPE tells us the no-CV model is wrong on Middlebury and ETH3D, but not HOW.
Three distinct failures all produce a large EPE and imply different things about RQ1:

  (1) scale / range error      -> predictions structurally sensible but in the wrong
                                  numeric range (monocular prior locked to SceneFlow's
                                  camera geometry). Shows as mass on a line with
                                  slope != 1 or a non-zero intercept.
  (2) collapse to a constant   -> predictions cluster around one value regardless of
                                  scene. Shows as a horizontal band.
  (3) right range, wrong pixels-> marginal distribution matches GT but per-pixel
                                  assignment is scrambled. Shows as a diffuse cloud
                                  with no diagonal structure. This one FALSIFIES the
                                  scale story.

A marginal histogram separates (1) from (2) but is blind to (3) -- a perfect histogram
is compatible with a terrible EPE. The 2D joint density of pred vs GT with the y = x
diagonal drawn on top discriminates all three, so that is the primary figure; the
marginals are produced as a secondary, easier-to-read view.

Outputs
-------
  joint_density_pred_vs_gt.png   2 x N grid (rows = CV / no-CV, cols = datasets)
  marginal_histograms.png        2 x N grid of pred vs GT marginals
  disparity_distribution_stats.json
  disparity_distribution_stats.csv

Usage
-----
  python analyze_disparity_distributions.py \
      --ckpt_cv    ./continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_best_cv_<time>.pth \
      --ckpt_no_cv ./continuous_training/checkpoints_continuous_two_cycle/train_continuous_two_cycle_latest_no_cv.pth \
      --datasets sceneflow middlebury eth3d \
      --out_dir ./continuous_training/dist_analysis
'''

import os
import json
import argparse
import logging

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')  # no display on the cluster
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from core.liteanystereo import CustomLiteAnyStereo
from core.training_datasets import fetch_testing_dataloader
from core.utils.utils import InputPadder


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

def load_model(ckpt_path, device, scale_right=1.0):
    """Builds CustomLiteAnyStereo and loads a training checkpoint saved by
    train_continuous*.py (state lives under the 'model_state' key)."""
    try:
        model = CustomLiteAnyStereo(scale_right=scale_right)
    except TypeError:
        # older signature without scale_right
        model = CustomLiteAnyStereo()

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)

    step = ckpt.get('step', None)
    logging.info(f"Loaded {ckpt_path}" + (f" (step {step})" if step is not None else ""))

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------------------

@torch.no_grad()
def collect_pairs(model, val_loader, device, cv, max_disp=192, per_batch_cap=20000, seed=0):
    """Runs the model over a validation loader and returns two flat float arrays of
    matched (gt, pred) disparities over valid pixels.

    Masking mirrors calculate_metrics() in train_continuous.py -- unpad FIRST so that
    prediction and GT stay pixel-aligned, then apply the valid mask. The additional
    (gt < max_disp) cap matches the convention in evaluate_other_datasets.py: the model
    cannot represent disparities beyond max_disp, so including them would manufacture
    error that has nothing to do with the CV ablation.
    """
    rng = np.random.default_rng(seed)
    preds_all, gts_all = [], []
    n_seen = 0

    for idx, data in enumerate(val_loader):
        img1, img2, _, _, disp_gt, valid_mask = data
        img1, img2 = img1.to(device), img2.to(device)
        disp_gt, valid_mask = disp_gt.to(device), valid_mask.to(device)

        if not valid_mask.any():
            continue

        padder = InputPadder(img1.shape, divis_by=32)
        img1p, img2p = padder.pad(img1, img2)

        with torch.autocast(device_type=device.type):
            pred = model(img1p, img2p, test_mode=True, compute_cost_volume=cv)

        pred = padder.unpad(pred).float()

        m = valid_mask.bool().unsqueeze(1)
        m = m & (disp_gt < max_disp) & (disp_gt > 0)

        if not m.any():
            continue

        p = pred[m].cpu().numpy()
        g = disp_gt[m].float().cpu().numpy()

        # subsample per batch: full val sets across 3 datasets x 2 models is tens of
        # millions of pixels, and the density plot saturates long before that.
        if len(p) > per_batch_cap:
            sel = rng.choice(len(p), per_batch_cap, replace=False)
            p, g = p[sel], g[sel]

        preds_all.append(p)
        gts_all.append(g)
        n_seen += 1

    if not preds_all:
        return np.array([]), np.array([])

    preds = np.concatenate(preds_all)
    gts = np.concatenate(gts_all)
    logging.info(f"  collected {len(preds):,} pixel pairs over {n_seen} batches")
    return gts, preds


# --------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------

def compute_stats(gts, preds):
    """Quantitative companion to the figure.

    slope / intercept  : least-squares fit of pred ~ gt. slope != 1 or intercept != 0
                         is the signature of systematic mis-scaling (failure mode 1).
    pearson_r          : how much per-pixel structure survives, independent of scale.
                         High r with bad slope  -> scale error.
                         Low r with good range  -> scrambled correspondence (mode 3).
    pred_std           : near-zero relative to gt_std indicates collapse (mode 2).
    """
    if len(gts) == 0:
        return {k: float('nan') for k in
                ['slope', 'intercept', 'pearson_r', 'epe', 'gt_median', 'pred_median',
                 'gt_std', 'pred_std', 'gt_p99', 'pred_p99', 'n_pixels']}

    slope, intercept = np.polyfit(gts, preds, 1)
    r = float(np.corrcoef(gts, preds)[0, 1])

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'pearson_r': r,
        'epe': float(np.mean(np.abs(preds - gts))),
        'gt_median': float(np.median(gts)),
        'pred_median': float(np.median(preds)),
        'gt_std': float(np.std(gts)),
        'pred_std': float(np.std(preds)),
        'gt_p99': float(np.percentile(gts, 99)),
        'pred_p99': float(np.percentile(preds, 99)),
        'n_pixels': int(len(gts)),
    }


# --------------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------------

def plot_joint_grid(results, datasets, out_path, gridsize=60):
    """2 x N hexbin grid. Rows = model (CV, no-CV), cols = dataset.

    Axis limits are shared DOWN each column (same dataset, both models) rather than
    across the whole figure: ETH3D and Middlebury disparity ranges differ by enough
    that a single global limit would squash ETH3D into a corner. The limit is taken as
    the max of GT and prediction p99 across both models, so that predictions
    overshooting the GT range stay visible instead of being clipped out of frame.
    """
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9.5), squeeze=False)

    for j, ds in enumerate(datasets):
        lim = 1.0
        for tag in ['cv', 'no_cv']:
            g, p = results[tag][ds]['gts'], results[tag][ds]['preds']
            if len(g):
                lim = max(lim, np.percentile(g, 99), np.percentile(p, 99))
        lim = float(np.ceil(lim * 1.05))

        for i, (tag, label) in enumerate([('cv', 'With CV'), ('no_cv', 'No CV')]):
            ax = axes[i][j]
            g, p = results[tag][ds]['gts'], results[tag][ds]['preds']

            if len(g) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue

            hb = ax.hexbin(g, p, gridsize=gridsize, bins='log', cmap='viridis',
                           extent=(0, lim, 0, lim), mincnt=1)
            ax.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='y = x')

            s = results[tag][ds]['stats']
            xs = np.array([0, lim])
            ax.plot(xs, s['slope'] * xs + s['intercept'], color='white', linewidth=1.2,
                    linestyle='-', label=f"fit: {s['slope']:.2f}x + {s['intercept']:.1f}")

            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel('GT disparity (px)')
            ax.set_ylabel('Predicted disparity (px)')
            ax.set_title(f"{ds} - {label}\nEPE {s['epe']:.2f} | r {s['pearson_r']:.3f}",
                         fontsize=11)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.7)
            fig.colorbar(hb, ax=ax, label='pixels (log)')

    fig.suptitle('Predicted vs GT disparity: joint density', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"wrote {out_path}")


def plot_marginals(results, datasets, out_path, nbins=120):
    """2 x N marginal histograms. Secondary view -- easier to read at a glance, but
    cannot distinguish a matching distribution from correct per-pixel assignment."""
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for j, ds in enumerate(datasets):
        lim = 1.0
        for tag in ['cv', 'no_cv']:
            g, p = results[tag][ds]['gts'], results[tag][ds]['preds']
            if len(g):
                lim = max(lim, np.percentile(g, 99), np.percentile(p, 99))
        lim = float(np.ceil(lim * 1.05))
        bins = np.linspace(0, lim, nbins)

        for i, (tag, label) in enumerate([('cv', 'With CV'), ('no_cv', 'No CV')]):
            ax = axes[i][j]
            g, p = results[tag][ds]['gts'], results[tag][ds]['preds']
            if len(g) == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue

            ax.hist(g, bins=bins, alpha=0.55, label='GT', density=True, color='#2a78d6')
            ax.hist(p, bins=bins, alpha=0.55, label='Predicted', density=True, color='#eb6834')
            ax.set_yscale('log')
            ax.set_xlabel('disparity (px)')
            ax.set_ylabel('density (log)')
            ax.set_title(f"{ds} - {label}", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle('Marginal disparity distributions', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"wrote {out_path}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    val_loaders = fetch_testing_dataloader(datasets=args.datasets)

    results = {'cv': {}, 'no_cv': {}}

    for tag, ckpt, cv_flag in [('cv', args.ckpt_cv, True),
                               ('no_cv', args.ckpt_no_cv, False)]:
        logging.info(f"=== {tag} checkpoint (compute_cost_volume={cv_flag}) ===")
        model = load_model(ckpt, device, scale_right=args.scale_right)

        for ds in args.datasets:
            logging.info(f" {ds}...")
            gts, preds = collect_pairs(
                model, val_loaders[ds], device, cv=cv_flag,
                max_disp=args.max_disp, per_batch_cap=args.per_batch_cap, seed=args.seed
            )
            stats = compute_stats(gts, preds)
            results[tag][ds] = {'gts': gts, 'preds': preds, 'stats': stats}
            logging.info(
                f"  EPE {stats['epe']:.3f} | slope {stats['slope']:.3f} | "
                f"intercept {stats['intercept']:.3f} | r {stats['pearson_r']:.3f} | "
                f"pred_std/gt_std {stats['pred_std']/max(stats['gt_std'],1e-9):.3f}"
            )

        del model
        torch.cuda.empty_cache()

    plot_joint_grid(results, args.datasets,
                    os.path.join(args.out_dir, 'joint_density_pred_vs_gt.png'))
    plot_marginals(results, args.datasets,
                   os.path.join(args.out_dir, 'marginal_histograms.png'))

    # tables
    flat = {tag: {ds: results[tag][ds]['stats'] for ds in args.datasets}
            for tag in ['cv', 'no_cv']}
    json_path = os.path.join(args.out_dir, 'disparity_distribution_stats.json')
    with open(json_path, 'w') as f:
        json.dump(flat, f, indent=4)
    logging.info(f"wrote {json_path}")

    csv_path = os.path.join(args.out_dir, 'disparity_distribution_stats.csv')
    keys = ['epe', 'slope', 'intercept', 'pearson_r', 'gt_median', 'pred_median',
            'gt_std', 'pred_std', 'gt_p99', 'pred_p99', 'n_pixels']
    with open(csv_path, 'w') as f:
        f.write('model,dataset,' + ','.join(keys) + '\n')
        for tag in ['cv', 'no_cv']:
            for ds in args.datasets:
                s = flat[tag][ds]
                f.write(f"{tag},{ds}," + ','.join(f"{s[k]}" for k in keys) + '\n')
    logging.info(f"wrote {csv_path}")

    # readable summary
    print('\n' + '=' * 92)
    print(f"{'model':>7} {'dataset':>12} {'EPE':>8} {'slope':>8} {'intcpt':>8} "
          f"{'r':>7} {'predSD/gtSD':>12} {'gt_med':>8} {'pred_med':>9}")
    print('=' * 92)
    for tag in ['cv', 'no_cv']:
        for ds in args.datasets:
            s = flat[tag][ds]
            ratio = s['pred_std'] / max(s['gt_std'], 1e-9)
            print(f"{tag:>7} {ds:>12} {s['epe']:>8.3f} {s['slope']:>8.3f} "
                  f"{s['intercept']:>8.3f} {s['pearson_r']:>7.3f} {ratio:>12.3f} "
                  f"{s['gt_median']:>8.2f} {s['pred_median']:>9.2f}")
    print('=' * 92)
    print("\nreading the table:")
    print("  slope ~1, intercept ~0, high r   -> calibrated")
    print("  slope !=1 or intercept !=0, high r -> systematic scale/offset error")
    print("  predSD/gtSD ~0                   -> collapse toward a constant")
    print("  low r despite sane range         -> scrambled correspondence, NOT scale")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_cv', required=True,
                        help='checkpoint from the CV phase (best_cv or latest_cv)')
    parser.add_argument('--ckpt_no_cv', required=True,
                        help='checkpoint from the no-CV phase (latest_no_cv)')
    parser.add_argument('--datasets', nargs='+',
                        default=['sceneflow', 'middlebury', 'eth3d'])
    parser.add_argument('--max_disp', type=int, default=192,
                        help='GT disparities >= this are excluded (model cannot represent them)')
    parser.add_argument('--per_batch_cap', type=int, default=20000,
                        help='max pixels sampled per batch')
    parser.add_argument('--scale_right', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./continuous_training/dist_analysis')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(
                os.path.dirname(args.out_dir) or '.', 'dist_analysis.log')),
            logging.StreamHandler()
        ]
    )
    main(args)
