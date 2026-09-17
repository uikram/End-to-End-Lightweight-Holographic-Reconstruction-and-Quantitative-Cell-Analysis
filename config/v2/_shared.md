# The v2 experiment matrix

Every arm extends `config/base.yaml` and shares:

* off-axis only (the in-line arm is deferred to study 2)
* no drug-condition head (unstable at +/-9.5 points, outside the question)
* `phase_mask_contrast` and `dry_mass_consistency` permanently off; see
  `config/base.yaml` for the measurement behind each

## The arms, and the one question each answers

| arm | file | the question it answers alone |
|---|---|---|
| A   | `a_baseline.yaml`         | what conventional losses alone achieve |
| B   | `b_cell_ipp.yaml`         | does a PER-CELL mass constraint help |
| B'  | `b1_image_volume.yaml`    | ... or would an IMAGE-LEVEL one have done |
| B'' | `b2_cell_area.yaml`       | does constraining area as well as mass help |
| C   | `c_cell_ipp_bga.yaml`     | does boundary-gradient alignment help on THIS morphology |
| D0  | `d0_amplitude.yaml`       | can the amplitude head be supervised at all |
| D1  | `d_forward_amplitude.yaml`| what the forward-model term does (DIAGNOSTIC) |
| W   | `w_ipp_*.yaml`            | the weight sweep for B |
| K   | `k_compact_*.yaml`        | what the compact decoder costs |

## Why B' exists, and why it is not optional

B raises `cell_integrated_phase`. If B beats A, the natural claim is "a
measurement-aware loss helps". That claim is not established by B alone,
because an image-level phase-volume term would also be a measurement-aware
loss. B' runs the image-level term at a weight matched by gradient magnitude,
so the comparison B vs B' isolates the word PER-CELL, which is the actual
contribution. Without B', A vs B supports only the weaker claim.

## Why the weight sweep exists

`scripts/check_gradient_path.py` measures the ratio of the per-cell term's
gradient to the segmentation loss's gradient on the same parameters, AT WEIGHT
1.0. Measured: median 0.25-0.29 on 512 px crops. The objective adds
`w * term`, so the effective ratio is `w * 0.27`:

| w    | effective ratio | segmentation : term |
|------|-----------------|---------------------|
| 0.1  | 0.027           | 37 : 1              |
| 0.3  | 0.081           | 12 : 1              |
| 1.0  | 0.27            | 3.7 : 1             |
| 3.0  | 0.81            | 1.2 : 1             |

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
