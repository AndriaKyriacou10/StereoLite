"""
Composite already-rendered disparity PNGs into a qualitative comparison grid.

Layout follows the convention used in the stereo literature: rows are scenes,
columns are configurations, column headers only, no axes, minimal spacing.

Two details this handles that a naive imshow grid does not:

* PER-ROW ASPECT RATIOS. ETH3D, Middlebury and FlyingThings3D images have
  different aspect ratios, so a uniform grid would either stretch them or leave
  uneven gaps. Row heights are derived from each row's actual image aspect.

* WHITE BORDERS. matplotlib saves usually carry padding even with
  bbox_inches='tight'. CROP_BORDERS trims near-white margins so panels abut
  cleanly. Disable it if your renders are already tight, or if a genuinely
  white scene region touches the frame edge.

Colourbars are off by default. The comparison of interest is within a row, and
the reference layout omits them; enabling them costs roughly 8% of the width per
row and is only worth it if absolute disparity values matter to the argument.

Usage:
    python make_qualitative_grid.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PATH = "./video_frames/LAS_stabilizer/sintel_clean/ambush_2"  # where the rendered PNGs live
PATH = "./video_frames/video045/full"  # where the rendered PNGs live
PATH_diff = "./temporal_diff/southken/video045/full"  # where the rendered PNGs live

CROP_BORDERS = True      # trim near-white margins left by matplotlib saves
SHOW_COLORBARS = False  # per-row colourbars in a narrow final column
SHOW_ROW_LABELS = True   # rotated dataset names down the left edge
FIG_WIDTH_IN = 7.5       # PDF is included at \textwidth, so this sets aspect
                         # and effective DPI rather than printed size
GAP = 0.006              # gap between panels, as a fraction of figure width

# Columns are ordered CV, No-CV, GT so the two predictions sit adjacent and can
# be compared directly, with ground truth as the reference on the right.
COLUMNS = ['ETH3D', 'Middlebury', 'FlyingThings3D']

# ETH3D first: it has the largest CV/No-CV gap, so it establishes what to look
# for in the rows beneath.
start_idx = 172
end_idx = 175

scene = PATH.split('/')[-2]
OUT = f'./video_frames/{scene}_panels'  # output PDF/PNG path, no extension
OUT = f'./video_frames/{scene}_diff_panels'  # output PDF/PNG path, no extension


COLUMNS = [f"Frame {i}" for i in range(start_idx, end_idx + 1)]

ROWS = [
    ('Left RGB', [f"{PATH}/{scene}_{i:03d}_left.png" for i in range(start_idx, end_idx + 1)], None),
    
    ('Pseudo GT', [f"{PATH}/pseudo_gt/{scene}_{i:03d}_pseudo_gt.png" for i in range(start_idx, end_idx + 1)], None),
        
    ('Raw', [f"{PATH}/raw/{scene}_{i:03d}_raw.png" for i in range(start_idx, end_idx + 1)], None),
    
    ('Stab.', [f"{PATH}/stb/{scene}_{i:03d}_stb.png" for i in range(start_idx, end_idx + 1)], None),
]

ROWS = [
    ('Left RGB', [f"{PATH}/{scene}_{i:03d}_left.png" for i in range(start_idx, end_idx + 1)], None),
    
    ('$D_t - D_{t-1}$ Raw', [f"{PATH_diff}/raw/{scene}_{i:03d}_diff_raw.png" for i in range(start_idx, end_idx + 1)], None),
    
    ('$D_t - D_{t-1}$ Stb.', [f"{PATH_diff}/stb/{scene}_{i:03d}_diff_stb.png" for i in range(start_idx, end_idx + 1)], None),
]
# To add a left-image column, prepend its path to each row's list and prepend
# 'Left Image' to COLUMNS. At four columns each panel is ~25% narrower, so
# consider a sidewaysfigure in that case.


# ---------------------------------------------------------------------------

def crop_white(img, tol=0.98):
    """Trim near-white rows and columns from the edges.

    Operates on the mean over colour channels, ignoring alpha. Returns the
    original if cropping would remove everything, which happens on genuinely
    white images.
    """
    if img.ndim == 3:
        grey = img[..., :3].mean(axis=2)
    else:
        grey = img
    if grey.max() > 1.5:          # uint8 rather than float
        grey = grey / 255.0

    mask = grey < tol
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing: {path}")
    img = mpimg.imread(path)
    return crop_white(img) if CROP_BORDERS else img


def main():
    n_rows, n_cols = len(ROWS), len(COLUMNS)

    # load everything first so row aspects can be computed before laying out
    loaded, cbars, aspects = [], [], []
    for name, paths, cbar_path in ROWS:
        imgs = [load(p) for p in paths]
        loaded.append(imgs)
        h, w = imgs[0].shape[:2]
        aspects.append(h / w)
        if SHOW_COLORBARS:
            cbars.append(load(cbar_path) if os.path.exists(cbar_path) else None)
        else:
            cbars.append(None)
        print(f"{name:>16}: {w}x{h}  aspect {h/w:.3f}")

    label_w = 0.035 if SHOW_ROW_LABELS else 0.0
    cbar_w = 0.06 if SHOW_COLORBARS else 0.0
    # gaps sit between panels only, not at the outer edges
    panel_w = (1.0 - label_w - cbar_w - GAP * (n_cols - 1)) / n_cols

    # each row's height follows its own image aspect, so nothing is stretched
    row_h = [panel_w * a for a in aspects]
    header_h = 0.05
    total_h = sum(row_h) + header_h + GAP * (n_rows - 1)

    fig_h = FIG_WIDTH_IN * total_h
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h))

    y = 1.0 - header_h
    for r, ((name, _, _), imgs) in enumerate(zip(ROWS, loaded)):
        h = row_h[r] / total_h
        y -= h
        if r > 0:
            y -= GAP / total_h

        if SHOW_ROW_LABELS:
            fig.text(label_w / 2, y + h / 2, name, rotation=90,
                     va='center', ha='center', fontsize=10)

        for c, img in enumerate(imgs):
            ax = fig.add_axes([label_w + c * (panel_w + GAP), y, panel_w, h])
            ax.imshow(img, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=10, pad=6)

        if SHOW_COLORBARS and cbars[r] is not None:
            ax = fig.add_axes([label_w + n_cols * (panel_w + GAP), y, cbar_w, h])
            ax.imshow(cbars[r], aspect='auto')
            ax.axis('off')

    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    fig.savefig(f'{OUT}.eps', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{OUT}.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"\nwrote {OUT}.pdf and {OUT}.png")
    print(f"figure aspect {FIG_WIDTH_IN:.2f} x {fig_h:.2f} in")


if __name__ == '__main__':
    main()