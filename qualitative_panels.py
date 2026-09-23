import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PATH = './gru_iterations_L1vsL2'  # where the rendered PNGs live
OUT = './gru_iterations_L1vsL2/gru_test_all'  # output PDF/PNG path, no extension

CROP_BORDERS = True      # trim near-white margins left by matplotlib saves
SHOW_COLORBARS = False  # per-row colourbars in a narrow final column
SHOW_ROW_LABELS = False   # rotated dataset names down the left edge
FIG_WIDTH_IN = 7.5       # PDF is included at \textwidth, so this sets aspect
                         # and effective DPI rather than printed size
GAP = 0.006              # gap between panels, as a fraction of figure width

# Columns are ordered CV, No-CV, GT so the two predictions sit adjacent and can
# be compared directly, with ground truth as the reference on the right.
COLUMNS = ['ETH3D', 'Middlebury', 'FlyingThings3D']

# ETH3D first: it has the largest CV/No-CV gap, so it establishes what to look
# for in the rows beneath.
# ROWS = [
#     ('ETH3D', [
#         f'{PATH}/eth3d/cv/sample_2_left.png',
#         f'{PATH}/eth3d/cv/sample_2_disp_cv.png',
#         f'{PATH}/eth3d/nocv/sample_2_disp_nocv.png',
#         f'{PATH}/eth3d/nocv/sample_2_gt.png',
#     ], f'{PATH}/eth3d/nocv/sample_2_cbar.png'),

#     ('Middlebury', [
#         f'{PATH}/middlebury_H/cv/sample_4_left.png',
#         f'{PATH}/middlebury_H/cv/sample_4_disp_cv.png',
#         f'{PATH}/middlebury_H/nocv/sample_4_disp_nocv.png',
#         f'{PATH}/middlebury_H/nocv/sample_4_gt.png',
#     ], f'{PATH}/middlebury_H/nocv/sample_4_cbar.png'),

#     ('FlyingThings3D', [
#         f'{PATH}/sceneflow/cv/sample_712_left.png',
#         f'{PATH}/sceneflow/cv/sample_712_disp_cv.png',
#         f'{PATH}/sceneflow/nocv/sample_712_disp_nocv.png',
#         f'{PATH}/sceneflow/nocv/sample_712_gt.png',
#     ], f'{PATH}/sceneflow/nocv/sample_712_cbar.png'),
# ]

ROWS = [
    ('', [
        f'{PATH}/epe_vs_iterations_eth3d_comparison.png',
        f'{PATH}/epe_vs_iterations_middlebury_H_comparison.png',
        f'{PATH}/epe_vs_iterations_sceneflow_comparison.png',
    ], None),
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
                     va='center', ha='center', fontsize=9, fontweight='bold')

        for c, img in enumerate(imgs):
            ax = fig.add_axes([label_w + c * (panel_w + GAP), y, panel_w, h])
            ax.imshow(img, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=8, fontweight='bold', pad=6)

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