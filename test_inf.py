import numpy as np
from core.training_datasets import SceneFlowDataset, ETH3D, Middlebury

def scan_for_bad_disp(dataset, name, n=None):
    n = len(dataset)
    bad = 0
    for i in range(n):
        _, _, _, _, disp, valid = dataset[i]
        d = disp.numpy().squeeze()      # (1,H,W) -> (H,W)
        v = valid.numpy().astype(bool)  # (H,W)
        if d.shape != v.shape:
            print(f"{name} idx {i}: shape mismatch disp={d.shape} valid={v.shape}")
            continue
        if not np.isfinite(d[v]).all():
            bad += 1
            print(f"{name} idx {i}: non-finite disp inside 'valid' region, "
                  f"max={np.nanmax(d[v]):.2f}, has_inf={np.isinf(d[v]).any()}")
    print(f"{name}: {bad}/{n} samples with non-finite valid GT")
# scan_for_bad_disp(Middlebury(split='2005', augmentor=None), "Middlebury2005", n=None)
# scan_for_bad_disp(Middlebury(split='2006', augmentor=None), "Middlebury2006", n=None)
# scan_for_bad_disp(Middlebury(split='2021', augmentor=None), "Middlebury2021", n=None)
scan_for_bad_disp(Middlebury(split='MiddEval3', augmentor=None), "MiddEval3", n=None)

# scan_for_bad_disp(ETH3D(augmentor=None, condition='train'), "ETH3D", n=None)

