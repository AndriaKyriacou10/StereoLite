# Reconstructing Reality: Stereo Vision in Action 
### Cost Volume Removal and Temporal Stabilisation

Code for my MSc Applied Machine Learning thesis (Individual Research Project) at Imperial College London, supervised by Prof. Krystian Mikolajczyk (MatchLab). 

![RQ1 Results](/assets/qualitative_cv_nocv.png)

The project builds on [LiteAnyStereo](https://arxiv.org/abs/2511.16555) (Jing et al., 2025) and asks two independent research questions:

| | Research question | Model |
|---|---|---|
| **RQ1** | Can the cost volume be removed from a LiteAnyStereo-derived architecture while retaining useful accuracy, and what architectural compensations are needed? | **StereoLite** <br> (no cost volume) |
| **RQ2** | Can BiDAStabilizer be applied post-hoc to LiteAnyStereo to improve temporal consistency without retraining the base model? | **LAS + BiDA** (`original_LAS` frozen + BiDAStabilizer) |

RQ1 and RQ2 use different models and are not compared against each other.

---

## Results summary

### RQ1 — Cost-volume removal (StereoLite)

| Model | ETH3D EPE ↓ | Middlebury(H) EPE ↓ | FlyingThings3D EPE ↓ | Latency ↓ | Params ↓ |
|---|---|---|---|---|---|
| Original LAS | 0.32 | 0.94 | – | 22.5 ms | 7.6 M |
| StereoLite (with CV) | 0.602 | 1.487 | 0.801 | 38.9 ms | 10.7 M |
| **StereoLite (no CV)** | 1.322 | 7.152 | 1.388 | **19.24 ms** | **3.1 M** |

Efficiency measured at 384×1248 on an NVIDIA A100.

### RQ2 — Post-hoc temporal stabilisation (LAS + BiDA, 25k iterations, disp ≤ 192)

| Backbone | Dataset | ΔEPE | ΔTEPE |
|---|---|---|---|
| LiteAnyStereo | FlyingThings3D | −4.0% | −7.0% |
| LiteAnyStereo | Sintel Clean | +0.9% | −5.5% |
| RAFT-Stereo | FlyingThings3D | +9.6% | +10.6% |
| RAFT-Stereo | Sintel Clean | −0.2% | −8.1% |

Pipeline cost at 720×1280 (fp32, A100-40GB, kernel_size=50):<br>17.23 M params | 225.99 ms per frame | 18.5 GB.

<video width="1280" height="720" controls>
  <source src="assets/side_by_side.mp4" type="video/mp4">
</video>

---

## Architecture

**Base (LiteAnyStereo).** MobileNetV2 FPN features (1/4–1/32) → correlation cost volume (D_max = 192) → hybrid 3D→2D aggregation → soft-argmax regression → convex upsampling.
![StereoLite Architecture](/assets/CustomLiteAnyStereo_fig.svg
)

**StereoLite (RQ1).** Adds a ContextNet (ResNet-34 on 6-channel stereo input; `layer1` + `layer2` merged through an FPN to 1/4 resolution) and ConvGRU iterative refinement with context features re-injected every iteration. In the no-CV configuration the cost volume is replaced with a zero tensor of the same shape, so the GRU input dimensionality is unchanged.

**LAS + BiDA (RQ2).** BiDAStabilizer trained on top of frozen `original_LAS` predictions, with SEA-RAFT providing optical flow (as in the published BiDA code).
![LAS + BiDA](/assets/stb.png)

---
## Repository structure

```
.
├── core/
|   ├── stereolite.py                               # StereoLite
│   ├── liteanystereo.py                            # original_LAS
│   ├── fnet.py                                     # MobileNetV2 FPN feature extractor
│   ├── aggregation.py                              # 2D cost aggregation (ConvNeXt blocks, AttentionModule2D)
│   ├── submodule.py                                # Conv blocks, cost-volume builders, disparity regression
│   ├── context_network.py                          # ResNet-34 ContextNet
│   ├── conv_gru.py                                 # ConvGRU refinement cell
│   ├── raft_stereo.py                              # RAFT-Stereo (comparison backbone for RQ2)
│   ├── stereo_datasets.py                          # SceneFlow, Middlebury, ETH3D for RQ1
│   ├── video_datasets.py                           # Video datasets (SceneFlow, Sintel, SouthKensington SV)
│   ├── augmentor.py                                # Stereo augmentations
│   ├── frame_utils.py                              # PFM / flow / disparity readers
│   └── utils/
├── bidastabilizer_integration/                     # BiDAStabilizer model, losses, training, video datasets
│
├── utility/
|   ├── is_stb_training.py                          # RQ2: diagnostic pass over stabiliser training batches
|   ├── read_tb.py                                  # Read TensorBoard runs
|   ├── read_json.py                                # Read .json from RQ2
|   ├── test_weights_contextNet.py                  # RQ1 ablations: binned error, right-image zeroing
|   └── diagnose_stabiliser.py                      # RQ2 Ablation
|
├── training_stereo/
|   ├── train_continuous_two_cycle.py               # Continuous training with cost-volume cutoff using TWO learning rate schedulers (Chosen Training Setup) 
|   ├── train_continuous.py                         # Continuous training with cost-volume cutoff using ONE learning rate scheduler
|   └── train_stereo.py                             # Two-phase training (phase 1 + distillation phase 2)
|
├── evaluation_STEREO/
|   └── evaluate_stereolite.py                      # RQ1: ETH3D / Middlebury / SceneFlow evaluation
|
├── evaluate_stabiliser/
|   └── evaluation_og_stabilizer.py                 # RQ2: EPE / TEPE evaluation
|
├── example_results/                                # RQ1 visualisation examples
|
└── generate_report_figures_video/                  # RQ2 Visualisation Scripts per frame
```

---
## Setup

Tested with Python 3.12, PyTorch 2.12, CUDA 12.6 on NVIDIA A100 GPUs (Imperial CX3 HPC).

```bash
conda create -n py312 python=3.12
conda activate py312

# Install PyTorch for your CUDA version first (see pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

## Data

Datasets are expected under `./data/datasets/`:

```
data/datasets/
├── SceneFlow/          # FlyingThings3D, Driving, Monkaa (frames_cleanpass + disparity)
├── Middlebury/         # 2005, 2006, 2014, 2021/data, MiddEval3
├── ETH3D/              # two_view_training, two_view_training_gt
├── Sintel/             # RQ2 only
└── SouthKensington/    # RQ2 only (qualitative)
```

ETH3D uses a scene-disjoint train/eval split with `seed=42` (13 train / 14 eval scenes).

## Checkpoints 
You can find the checkpoints here: <u> [link](https://drive.google.com/drive/folders/1Ot6DxDaAVL5Yi6AtFTlGDIrvJse9NJQP?usp=drive_link) </u>

Place checkpoints in `./checkpoints/`:

| File | Description |
|---|---|
| `LiteAnyStereo.pth` | Official LiteAnyStereo weights (`original_LAS`) |
| `stereoLite_FINAL_no_cv.pth` | StereoLite - No Cost Volume |
| `stereoLite_FINAL_cv.pth` | StereoLite - Cost Volume |
| `LAS_stabilizer_final.pth` | BiDAStabilizer trained on LAS (25k iterations) |

---

## Usage

### RQ1 — StereoLite

**Train**
```bash
# Continuous schedule: cost volume on until --cutoff_step, then removed
python -m  training_stereo.train_continuous_two_cycle --total_steps 340000 --cutoff_step 90000 --lr 2e-4 [--layer2]\
    --save_dir ./checkpoints/continuous_training
```

**Evaluate**

Run the full RQ1 evaluation (CV and no-CV models on Middlebury H, ETH3D and SceneFlow):

```bash
bash evaluation_STEREO/eval_FINAL.sh
```

Edit `CKPT_CV` and `CKPT_NO_CV` at the top of the script to point to your checkpoints.

To evaluate a single checkpoint on one dataset:

```bash
python -m evaluation_STEREO.evaluate_stereoLite --dataset eth3d --ckpt <ckpt> --layer2 [--cost_volume]
```

| Flag | Meaning |
|---|---|
| `--dataset` | `eth3d`, `sceneflow`, `middlebury_F` / `middlebury_H` / `middlebury_Q` |
| `--layer2` | Use the ContextNet with the layer2 FPN extension (StereoLite) |
| `--cost_volume` | Compute the cost volume (omit for the no-CV model) |
| `--model` | `Custom` (default) or `Original` for the official LiteAnyStereo |

To evaluate the other training methods:
```bash
bash evaluation_STEREO/eval_one_cycle.sh
```
```bash
bash evaluation_STEREO/eval_two_phase.sh
```
**Ablations**
```bash
python -m utility.test_weights_contextNet --ckpt <ckpt> --n_samples 500 [--compute_cost_volume] --mode [run, binned, visualize] [--layer2]
```

### RQ2 — LAS + BiDA

**Train the stabiliser**
```bash
python bidastabilizer_integration/train_bidastabilizer.py --name LAS_stabilizer --batch_size 8 \
--spatial_scale -0.2 0.4 --image_size 192 384 --saturation_range 0 1.4 --num_steps 9000  \
--restore_ckpt checkpoints/LiteAnyStereo.pth --ckpt_path checkpoints/<ckpt>/ \
--sample_len 8 --lr 0.0002 --train_iters 22 --valid_iters 32    \
--num_workers 8 --save_freq 100 --train_datasets things monkaa driving \
```

**Evaluate**
```bash
bash evaluate_stabiliser/evaluate_video_stb.sh
```
## Acknowledgements

This code builds on:

- **LiteAnyStereo** — Jing et al., *Lite Any Stereo: Efficient Zero-Shot Stereo Matching*, 2025 [[Repo]](https://github.com/TomTomTommi/LiteAnyStereo/tree/main)
- **BiDAStabilizer / BiDAVideo** — Jing et al., *Match Stereo Videos via Bidirectional Alignment*, 2024 [[Repo]](https://github.com/TomTomTommi/bidavideo)
- **ReCoVEr** — S. Kiefhaber, S.Roth, and S.Schaub-Meyer, *Removing Cost Volumes from Optical Flow Estimators*, 2025 [[Repo]](https://github.com/visinf/ReCoVEr/tree/main)

Please see the original repositories for their licences.

## Citation

```bibtex
@mastersthesis{andria2026stereo,
  title  = {Reconstructing Reality: Stereo Vision in Action - Cost Volume Removal and Temporal Stabilisation},
  author = {Andria Kyriacou},
  school = {Imperial College London},
  year   = {2026}
}
```
