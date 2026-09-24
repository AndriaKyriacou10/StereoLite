# Reconstructing Reality: Stereo Vision in Action 
### Cost Volume Removal and Temporal Stabilisation

Code for my MSc Applied Machine Learning thesis (Individual Research Project) at Imperial College London, supervised by Prof. Krystian Mikolajczyk (MatchLab). 

The project builds on [LiteAnyStereo](https://arxiv.org/abs/) (Jing et al., 2025) and asks two independent research questions:

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

---

## Architecture

**Base (LiteAnyStereo).** MobileNetV2 FPN features (1/4–1/32) → correlation cost volume (D_max = 192) → hybrid 3D→2D aggregation → soft-argmax regression → convex upsampling.
![StereoLite Architecture](/assets/CustomLiteAnyStereo_fig.svg
)

**StereoLite (RQ1).** Adds a ContextNet (ResNet-34 on 6-channel stereo input; `layer1` + `layer2` merged through an FPN to 1/4 resolution) and ConvGRU iterative refinement with context features re-injected every iteration. In the no-CV configuration the cost volume is replaced with a zero tensor of the same shape, so the GRU input dimensionality is unchanged.

**LAS + BiDA (RQ2).** BiDAStabilizer trained on top of frozen `original_LAS` predictions, with SEA-RAFT providing optical flow (as in the published BiDA code).
![LAS + BiDA](/assets/stb.png)

---