from core.training_datasets import Middlebury, ETH3D, SceneFlowDataset
from core.stereo_datasets import SceneFlowVideo, SintelStereoVideo, SouthKenSV
from core.utils import frame_utils
import numpy as np
from PIL import Image


def test_eth3d():
    n_checks = 20
    train_dataset = ETH3D(augmentor=None, condition='train', train_frac=0.5, is_phase_2=False)
    test_dataset = ETH3D(augmentor=None, condition='test', train_frac=0.5, is_phase_2=False)
    print(f"ETH3D TRAINING dataset has {len(train_dataset.left_img_paths)} samples.")
    print(f"ETH3D TESTING dataset has {len(test_dataset.left_img_paths)} samples.")
    return
    idxs = range(0, len(dataset.left_img_paths), max(1, len(dataset.left_img_paths)//n_checks))
    for i in idxs:
        l, r, d = dataset.left_img_paths[i], dataset.right_img_paths[i], dataset.disp_paths[i]
        scene_dir = os.path.basename(os.path.dirname(d))
        # print(f"{scene_dir in l} | {l}")
        assert scene_dir in l and scene_dir in r and scene_dir in d, (l, r, d)
        assert os.path.exists(l) and os.path.exists(r) and os.path.exists(d), (l, r, d)
    print(f"Checked {len(list(idxs))} triples — all aligned and exist.")

def test_middlebury():
    n_checks = 100
    dataset1 = Middlebury(split='2005', augmentor=None, is_phase_2=False)
    dataset2 = Middlebury(split='2006', augmentor=None, is_phase_2=False)
    dataset3 = Middlebury(split='2021', augmentor=None, is_phase_2=False)
    dataset4 = Middlebury(split='2014', augmentor=None, is_phase_2=False)
    data_samples = dataset1.left_img_paths + dataset2.left_img_paths + dataset3.left_img_paths + dataset4.left_img_paths
    
    # idxs = range(0, len(dataset.left_img_paths), max(1, len(dataset.left_img_paths)//n_checks))
    print(len(data_samples))
    
    for split in ['2005', '2006', '2021', '2014', 'MiddEval3']:
        ds = Middlebury(split=split)
        over = tot = 0
        for path in sorted(set(ds.disp_paths)):      
            disp, valid = frame_utils.readDispMiddlebury(path)
            mask = valid & np.isfinite(disp)
            over += int((disp[mask] >= 192).sum())
            tot  += int(mask.sum())
        print(split, len(ds))
        print(f"{split}: {100*over/tot:.3f}% of valid pixels >= 192  ({over}/{tot})")
        
    scene = '/rds/general/user/kk1525/home/IRP/data/datasets/Middlebury/2006/Lampshade1'
    
    L = np.array(Image.open(f'{scene}/view1.png').convert('L')).astype(np.float32)
    R = np.array(Image.open(f'{scene}/view5.png').convert('L')).astype(np.float32)
    d, v = frame_utils.readDispMiddlebury(f'{scene}/disp1.png')

    H, W = d.shape
    print(f"image {H}x{W} | disp max {d[v].max():.0f} | max/W = {d[v].max()/W:.2f}")

    ys, xs = np.nonzero(v)
    for name, s in [("raw", 1.0), ("div3", 1/3)]:
        xr = np.round(xs - d[ys, xs] * s).astype(int)
        ok = xr >= 0
        print(f"{name:5s}: photometric error {np.abs(L[ys[ok],xs[ok]] - R[ys[ok],xr[ok]]).mean():.2f}")

def test_things():
    train_set = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TRAIN', subsets=['flyingthings', 'driving', 'monkaa'])
    driving_set = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TRAIN', subsets=['driving'])
    monkaa_set = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TRAIN', subsets=['monkaa'])
    test_set = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    print(f"SceneFlow TRAIN has {len(train_set.left_img_paths)} samples")
    print(f"SceneFlow DRIVING has {len(driving_set.left_img_paths)} samples")
    print(f"SceneFlow MONKAA has {len(monkaa_set.left_img_paths)} samples")
    print(f"FlyingThings TEST has {len(test_set.left_img_paths)} samples")


def test_scene_flow_video():
    dataset = SceneFlowVideo(mode='TEST', subsets=['flyingthings'])
    print(f"Total scenes: {len(dataset)}")
    for i in range(len(dataset)):
        scene = dataset[i]
        print(f"Scene {i}: {scene['scene_id']}, Frames: {len(scene['left'])}, Disparity maps: {len(scene['disp'])}, Optical flows: {len(scene['flow'])}")
        
def test_valid_pixels(dataset_name, dstype=" "):
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    if dataset_name == 'sintel':
        dstype = 'clean'  # run this once for 'clean', once for 'final'
        dataset = SintelStereoVideo(mode='training', dstype=dstype)
    elif dataset_name == 'flyingthings':
        dataset = SceneFlowVideo(mode = 'TEST', subsets=['flyingthings'])
        all_vals = []
        for scene in dataset.scenes:
            for disp_path in scene['disp']:
                disp = frame_utils.readPFM(disp_path)
                all_vals.append(disp.flatten())
        all_vals = np.concatenate(all_vals)
        print(f"max disparity: {all_vals.max():.1f}")
        print(f"% pixels > 192: {100*(all_vals > 192).mean():.2f}%")
        sys.exit(0)
        
    all_valid_disps = []
    per_scene_stats = []

    for i in range(len(dataset)):
        scene = dataset[i]
        scene_vals = []
        for disp, valid in zip(scene['disp'], scene['valid']):
            disp_np = disp.squeeze(0).numpy()          # (H, W)
            valid_np = valid.numpy().astype(bool)       # (H, W)
            vals = disp_np[valid_np]                    # only ground-truth-valid pixels
            scene_vals.append(vals)
            all_valid_disps.append(vals)

        scene_all = np.concatenate(scene_vals)
        per_scene_stats.append({
            'scene_id': scene['scene_id'],
            'max_disp': scene_all.max(),
            'mean_disp': scene_all.mean(),
            'pct_over_192': 100 * (scene_all > 192).mean(),
        })

    all_valid_disps = np.concatenate(all_valid_disps)

    print(f"[{dstype}] total valid pixels: {len(all_valid_disps):,}")
    print(f"[{dstype}] max disparity: {all_valid_disps.max():.1f}")
    print(f"[{dstype}] % of pixels with disparity > 192: {100*(all_valid_disps>192).mean():.2f}%")

    per_scene_stats.sort(key=lambda s: s['pct_over_192'], reverse=True)
    print(f"\nTop 10 scenes by %% pixels exceeding max_disp=192 [{dstype}]:")
    for s in per_scene_stats[:10]:
        print(f"  {s['scene_id']:15s}  max={s['max_disp']:6.1f}  mean={s['mean_disp']:5.1f}  %>192={s['pct_over_192']:5.2f}%")

    plt.figure(figsize=(8, 5))
    plt.hist(all_valid_disps, bins=100, range=(0, max(500, all_valid_disps.max())))
    plt.axvline(192, color='red', linestyle='--', label='LAS1 max_disp=192')
    plt.yscale('log')  # most pixels sit low; the problem tail is small and would be invisible on a linear axis
    plt.xlabel('GT disparity (px)')
    plt.ylabel('pixel count (log scale)')
    plt.legend()
    plt.title(f'{dataset_name} {dstype} — GT disparity distribution')
    plt.savefig(f'{dataset_name}_{dstype}_disp_histogram.png', dpi=120)
    
def test_disp():
    dataset = SceneFlowVideo(mode="TEST", subsets=['flyingthings'])
    # no max_disp needed here -- .scenes just holds file paths, doesn't touch _fetch_disparity at all

    n_inf, n_nan, n_total = 0, 0, 0
    worst_scenes = []

    for scene in dataset.scenes:
        for disp_path in scene['disp']:
            disp = frame_utils.readPFM(disp_path)
            inf_count = np.isinf(disp).sum()
            nan_count = np.isnan(disp).sum()
            n_inf += inf_count
            n_nan += nan_count
            n_total += disp.size
            if inf_count > 0 or nan_count > 0:
                worst_scenes.append((scene['scene_id'], disp_path, inf_count, nan_count))

    print(f"total pixels: {n_total:,}")
    print(f"inf pixels:   {n_inf:,} ({100*n_inf/n_total:.4f}%)")
    print(f"nan pixels:   {n_nan:,} ({100*n_nan/n_total:.4f}%)")
    print(f"scenes affected: {len(worst_scenes)} / {len(dataset.scenes)}")    

def test_southken():
    import matplotlib.pyplot as plt
    dataset = SouthKenSV(pseudo_gt_dir='./data/datasets/SouthKensington/indoor/pseudo_gt', max_disp=192, border=0)

    t = 0
    data = dataset[24]
    print(len(data['left']), len(data['disp']))

    img1 = data['left'][t] / 255.0
    d = data['disp'][t][0].numpy()
    v = data['valid'][t].numpy()
    print(f"disp {d.min():.2f}..{d.max():.2f}  valid {v.mean():.3f}")

    vmax = np.percentile(d, 99)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img1.permute(1, 2, 0))
    ax[0].set_title(f"Scene {data['scene_id']} | Frame {t}")
    im_disp = ax[1].imshow(d, cmap='inferno', vmin=0, vmax=vmax)
    plt.colorbar(im_disp, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Disparity Map (Pseudo GT)")
    plt.savefig("southken_test.png", dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    dataset = SouthKenSV(pseudo_gt_dir='./data/datasets/SouthKensington/indoor/pseudo_gt', max_disp=192, border=0)
    scene = dataset[1]
    print(f"Scene ID: {scene['scene_id']}")