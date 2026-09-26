# The v2 experiment matrix

The matrix has two parts, and conflating them is the mistake this file exists to
prevent:

* **The core ablation** — A, B, B', B'', C, D0, D1, D2, the W sweep and the K
  pair. All off-axis, each differing from its comparator by exactly one term, so
  a difference is attributable to that term.
* **The modality comparison** — arm **G**, which applies arm A's configuration
  unchanged to in-line Gabor holograms. It is not part of the per-cell-loss
  ablation and must never be placed in a single-modality table: two acquisition
  geometries cannot share a column. Its numbers live in
  `runs/v2_baseline_modality_comparison.json` and figures 5, 13 and 14.

Both parts were trained in the completed study, and `run_v2.sh`'s default arm
list contains D2 and G for that reason.

Every arm extends `config/base.yaml` and shares:

* **the angular-spectrum front end OFF** (`model.frontend.kind: none`) — the
  network is fed the raw hologram. The front end is implemented and can be
  enabled, but no arm of this study used it.
* no drug-condition head (unstable at +/-9.5 points, outside the question)
* `phase_mask_contrast` and `dry_mass_consistency` permanently off; see
  `config/base.yaml` for the measurement behind each

## The arms, and the one question each answers

| arm | manuscript name | file | the question it answers alone |
|---|---|---|---|
| A   | End-to-End Neural Baseline | `a_baseline.yaml`         | what conventional losses alone achieve |
| B   | +IPP (per-cell)  | `b_cell_ipp.yaml`         | does a PER-CELL mass constraint help |
| B'  | +IPP (image)     | `b1_image_volume.yaml`    | ... or would an IMAGE-LEVEL one have done |
| B'' | +Area            | `b2_cell_area.yaml`       | does constraining area as well as mass help |
| C   | +BGA             | `c_cell_ipp_bga.yaml`     | does boundary-gradient alignment help on THIS morphology |
| D0  | +Amplitude       | `d0_amplitude.yaml`       | can the amplitude head be supervised at all |
| D1  | +Fwd (fixed z)   | `d_forward_amplitude.yaml`| what the forward-model term does (DIAGNOSTIC) |
| D2  | +Fwd (free z)    | `d2_learned_z.yaml`       | is the propagation distance recoverable when z is free |
| W   | +IPP (per-cell), w = 0.1 / 0.3 / 3.0 | `w_ipp_*.yaml` | the weight sweep for B (`w_ipp_10` is B itself) |
| KA / KB | Compact Baseline / Compact +IPP | `k_compact_*.yaml` | what the compact decoder costs |
| L   | LoRA (not trained) | `l_lora.yaml`           | what LoRA adaptation of the encoder costs (DEFINED, NOT TRAINED) |

Outside the ablation, one modality arm:

| arm | manuscript name | file | the question it answers alone |
|---|---|---|---|
| G   | In-Line Neural Configuration | `g_baseline_gabor.yaml` | arm A's configuration on in-line Gabor holograms |

A, B and B' were trained at seeds 42, 1337 and 2024; every other arm once, at
seed 42. The results are in `runs/benchmark_results/` and the README.

## Why B' exists, and why it is not optional

B raises `cell_integrated_phase`. If B beats A, the natural claim is "a
measurement-aware loss helps". That claim is not established by B alone,
because an image-level phase-volume term would also be a measurement-aware
loss. B' runs the image-level term at B's weight (w = 1.0; its gradient
magnitude was not separately matched), and the comparison B vs B' is the one
that addresses the word PER-CELL, which is the actual contribution. Without B', A vs B supports only the weaker claim.

## Why the weight sweep exists

`scripts/check_gradient_path.py` measures the ratio of the per-cell term's
gradient to the segmentation loss's gradient on the same parameters, AT WEIGHT
1.0. Measured over 30 batches of 512 px crops: median 0.302, range 0.18-0.67
(`runs/gradient_path_b_cell_ipp_512.csv`). The objective adds `w * term`, so
the effective ratio is `w * 0.302`:

| w    | effective ratio | segmentation : term |
|------|-----------------|---------------------|
| 0.1  | 0.030           | 33 : 1              |
| 0.3  | 0.091           | 11 : 1              |
| 1.0  | 0.302           | 3.3 : 1             |
| 3.0  | 0.906           | 1.1 : 1             |

At 0.1 the term is swamped and a null result would say nothing about the
hypothesis, only about the weight. B therefore runs at 1.0 and the sweep
brackets it. Re-run the script if `data.train_crop` changes: the ratio roughly
doubles at 256 px, because the per-cell mean divides by the number of cells
while the segmentation loss does not.

## Why D1 is labelled a diagnostic

`scripts/amplitude_sensitivity.py`, off-axis, z = 33.77 um, crops and
augmentation disabled:

| phase variant | per_field surface | global surface |
|---|---|---|
| x 0.9 (mild, 10% bias)   | **-0.0005** | **-0.0027** |
| x 0.5 (coarse, wrecked)  | +0.053      | +0.0016     |

A degraded phase must score WORSE. The mild degradation scores BETTER in both
modes, so near the truth the residual's gradient points the wrong way and the
term cannot refine a good reconstruction. Under the deployable global surface
even the coarse discrimination collapses by a factor of about 35. D1 is
therefore reported as a measurement of the term's behaviour, not as a loss
ablation, and its weight stays low.

The final measurement (`runs/z_calibration.json`, 32 validation fields, the
training border) agrees: a 10% rescaled phase changes the residual by -5e-5,
within the 0.01 tolerance, and scores lower than the true phase on 22 of 32
fields. D1 and D2 were trained at `forward_model: 0.02` and are reported as
diagnostics.
