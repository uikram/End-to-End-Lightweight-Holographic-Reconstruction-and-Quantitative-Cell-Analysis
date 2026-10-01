# The configuration matrix

Every file in `config/v2/` extends `config/base.yaml` and differs from its comparator by exactly one term,
one output or one capacity choice, so a difference is attributable to that change. The In-Line Neural
Configuration is the only in-line configuration: it applies the baseline unchanged to in-line Gabor
holograms (SNU_01–SNU_50 excluded) and must not be placed in a single-geometry table.

All configurations share: the raw hologram as network input (`model.frontend.kind: none`); no drug-condition
head; `phase_mask_contrast`, `dry_mass_consistency` and `projected_area_consistency` at weight 0; crop
origin (−6, −2); propagation distance 33.77 µm.

| Manuscript name | file | change relative to comparator | comparator | training runs (seeds) |
|---|---|---|---|---|
| End-to-End Neural Baseline | `a_baseline.yaml` | L_phase + L_seg only | – | 3 (42, 1337, 2024) |
| In-Line Neural Configuration | `g_baseline_gabor.yaml` | in-line holograms | Baseline | 1 (42) |
| +IPP (per-cell) | `b_cell_ipp.yaml` | + L_IPP^cell, w = 1.0 | Baseline | 3 (42, 1337, 2024) |
| +IPP (image) | `b1_image_volume.yaml` | + L_IPP^img, w = 1.0 (gradient not matched to the per-cell term) | Baseline | 3 (42, 1337, 2024) |
| +Area | `b2_cell_area.yaml` | + L_area | +IPP (per-cell) | 1 (42) |
| +BGA | `c_cell_ipp_bga.yaml` | + L_BGA, w = 0.05 | +IPP (per-cell) | 1 (42) |
| +IPP (per-cell), w = 0.1 / 0.3 / 3.0 | `w_ipp_01.yaml`, `w_ipp_03.yaml`, `w_ipp_30.yaml` | L_IPP^cell weight | Baseline | 1 (42) each |
| (repeat check of +IPP (per-cell)) | `w_ipp_10.yaml` | identical to `b_cell_ipp.yaml`; separate seed-42 run, not a fourth seed | – | 1 (42) |
| +Amplitude | `d0_amplitude.yaml` | amplitude head + L_amp, w = 0.1 | +IPP (per-cell) | 1 (42) |
| +Fwd (fixed z) | `d_forward_amplitude.yaml` | + L_fwd, w = 0.02 | +Amplitude | 1 (42) |
| +Fwd (free z) | `d2_learned_z.yaml` | z trainable | +Fwd (fixed z) | 1 (42) |
| Compact Baseline | `k_compact_a.yaml` | shared 1×1 projection to 256 channels | Baseline | 1 (42) |
| Compact +IPP | `k_compact_b.yaml` | compact decoder with L_IPP^cell | +IPP (per-cell) | 1 (42) |
| (not used in the manuscript) | `l_lora.yaml` | LoRA-adapted encoder; defined, not trained | – | – |

The loss weight key of L_IPP^img is `image_integrated_phase` (older configs and stored resolved configs use the
deprecated alias `phase_volume`). Comparisons in which either side has fewer than three training runs are not
estimable under the between-seed criterion (README §7).

## Loss weight and gradient ratio

`scripts/check_gradient_path.py` measures the ratio of the gradient of the per-cell term to that of the
segmentation loss, on the same parameters, at weight 1.0. Result on 30 batches of 512 × 512 crops
(`runs/gradient_path_b_cell_ipp_512.csv`): median 0.373, range 0.243–0.655. The objective adds `w · term`,
so the effective ratio is `w × 0.373`:

| w | effective ratio | segmentation : term |
|---|---|---|
| 0.1 | 0.037 | 26.8 : 1 |
| 0.3 | 0.112 | 8.9 : 1 |
| 1.0 | 0.373 | 2.7 : 1 |
| 3.0 | 1.120 | 1 : 1.1 |

The ratio depends on the crop size (it was higher at 256 px, with about 7 cells per crop); re-run the script if
`data.train_crop` changes.

## Forward-model probes

`scripts/calibrate_z.py` and `scripts/amplitude_sensitivity.py` (off-axis, z = 33.77 µm, crops and augmentation
disabled) give the residual when the reference phase is scaled; values and field counts are in
`results_for_manuscript/amplitude/amplitude.json` and in Table 11 of the manuscript. Under the implemented
operator, scaling the reference phase by 0.9 raises the off-axis residual by 0.00035 on 32 validation fields
(18 of 32 fields), below the configured tolerance of 0.01, so the residual does not resolve the phase scale near
the reference. The +Fwd configurations are therefore reported as a measurement of the term's behaviour.
