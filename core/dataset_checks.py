from core.training_datasets import Middlebury, ETH3D, SceneFlowDataset
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
    
    for split in ['2005', '2006', '2021', '2014']:
        ds = Middlebury(split=split)
        over = tot = 0
        for path in sorted(set(ds.disp_paths)):      
            disp, valid = frame_utils.readDispMiddlebury(path)
            mask = valid & np.isfinite(disp)
            over += int((disp[mask] >= 192).sum())
            tot  += int(mask.sum())
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
    test_set = SceneFlowDataset(augmentor=None, is_phase_2=False, mode='TEST', subsets=['flyingthings'])
    print(f"SceneFlow TRAIN has {len(train_set.left_img_paths)} samples")
    print(f"FlyingThings TEST has {len(test_set.left_img_paths)} samples")
    
if __name__ == "__main__":
    test_middlebury()