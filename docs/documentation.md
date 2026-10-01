# HoloQPI — Technical Documentation

Formulation, parameter reference and protocol of the code that produced the results of
`HoloQPI_4.2/`. Terminology follows the manuscript: integrated phase S_k, L_IPP^cell,
L_IPP^img, measurement-aware objectives, forward-model consistency, dry mass, matched
cells, field totals. Values quoted here come from `config/` and from
`results_for_manuscript/`; the README gives the overview.

- [1. Scope](#1-scope)
- [2. Data and splits](#2-data-and-splits)
- [3. Physical model and calibration](#3-physical-model-and-calibration)
- [4. Segmentation labels](#4-segmentation-labels)
- [5. Network](#5-network)
- [6. Joint objective](#6-joint-objective)
- [7. Evaluation metrics](#7-evaluation-metrics)
- [8. Experimental protocol and statistics](#8-experimental-protocol-and-statistics)
- [9. Verified diagnostics](#9-verified-diagnostics)
- [10. Configuration reference](#10-configuration-reference)
- [11. Extending the code](#11-extending-the-code)
- [12. Limitations](#12-limitations)

---

## 1. Scope

One network maps a raw hologram to a quantitative phase map and a cell segmentation (and
optionally a transmitted-amplitude map). A fixed measurement chain computes projected
area, circularity, integrated phase and dry mass per cell. Prediction quality is judged at
three levels kept separate: network outputs (phase, amplitude, segmentation overlap), cell
detection and geometry (recall, precision, area, circularity) and the measurements
(integrated phase, dry mass). The reference masks are derived from the reference phase.

## 2. Data and splits

800 fields of view: three cell lines (NCI-H1299, SNU-475, T-24) × five conditions
(control, blebbistatin, FCCP, staurosporine, rotenone); NCI 250, SNU 300, T24 250 fields.
Each field: off-axis hologram, in-line Gabor hologram (1024×1024, 8-bit), 900×900 float32
phase map in radians (supplied aberration-corrected, background-subtracted, unwrapped),
membrane-stain fluorescence image. The condition labels the dish, so it is used only to
stratify the split; it is not a network input or target.

Splits (`data/splits.json`, SHA-256 recorded in every run's provenance), stratified by cell
line × condition: 560 / 127 / 113 (train / validation / test), identical for every off-axis
configuration and seed. The 113 test fields contain 3186 reference cells.

In-line: the frames of SNU_01–SNU_50 do not show the same field as the off-axis hologram
and the reference phase (band-pass registration against a derangement control; see the
manuscript, Sec. 4.1) and are excluded from the in-line modality only
(`data.exclude`, 39 + 5 + 6 fields), giving 521 / 122 / 107. The 107 in-line test fields
(3055 reference cells) define the common-field evaluation of Table 5. The registration
statistics quoted in the manuscript come from a diagnostics archive that is not in this
repository (`analysis/v42/AUDIT.md`, §6).

Holograms are cropped 1024 → 900 px without resampling; the origin is moved by (−6, −2) px
in (y, x) from the centred crop (`data.crop_offset_px`), determined by registering the
classical off-axis reconstruction of each field to its reference phase (validation median
offset of the centred crop +5.4 / +1.1 px; test residual at (−6, −2): −0.1 / 0.0 px).

## 3. Physical model and calibration

Constants (`config/base.yaml`, `optics`): λ = 0.666 µm, dx = dy = 0.284871 µm,
α = 0.2 mL/g (literature range 0.173–0.215 mL/g), propagation distance 33.77 µm.

Per segmented instance Ω_k (N_k pixels, Crofton perimeter P_k):

```
A_k = N_k dx dy          C_k = min(1, 4π N_k / P_k²)
S_k = Σ_{p∈Ω_k} φ_p dx dy   [rad·µm²]          m_k = λ / (2π α) · S_k   [pg]
```

Dry mass is a fixed multiple of the integrated phase, so the relative integrated-phase
error equals the relative dry-mass error; reported once, as dry-mass MAPE. (Stored metric
keys named `optical_volume_*` carry the same numbers as `dry_mass_*` for relative
statistics and are not used in the manuscript.) `scripts/selftest.py` checks the chain
against analytic synthetic cells.

Propagation: angular-spectrum operator P_z (evanescent components attenuated); in-line
intensity |P_z U|², off-axis intensity |R|² + |P_z U|² + R*P_z U + R P_z U*. The aberration
surface (order 5) removed from the supplied phase is added back in the forward-model term
as a single global surface (per-coefficient median of per-field fits on the training
split); per-field surfaces are fitted against the reference phase and exist only for
diagnostics.

## 4. Segmentation labels

No manual annotation. Reference masks: Gaussian smoothing (σ = 4 px), per-field Otsu
threshold, binary closing (2 px), hole filling, border-object removal, area filter 30–6000
µm²; instances by distance-transform watershed (minimum peak distance 15 px). The labels are
a deterministic function of the reference phase (silver standard). The closing step leaves a
two-pixel empty rim, so border-object removal removes nothing as configured; edge-truncated
objects remain in the reference. A membrane-derived mask set is available for a check that
does not depend on the phase (`scripts/prepare_membrane.py`); it is provisional.

## 5. Network

MobileNetV2 encoder (ImageNet initialisation, first convolution rebuilt for one channel),
two U-Net-style decoders (widths 256/128/64/32): a phase decoder (phase head, optional
amplitude head = 1 + 0.3 tanh, zero-initialised) and a segmentation decoder (two-class head).
Input z-scored per field, padded to a multiple of 32; outputs at stride 2, bilinearly
upsampled to 900×900. 9,598,099 parameters, 45.85 GMAC at 900×900. Compact variant: shared
1×1 projection 1280 → 256 channels before both decoders, 3,360,403 parameters. No classical
reconstruction in the inference path (`model.frontend.kind: none`). The drug-condition head is
disabled (`model.classifier_enabled: false`).

## 6. Joint objective

`holoqpi/losses/composite.py` (`JointMeasurementLoss`)

```
L = L_phase + L_seg + w_IPPcell L_IPP^cell + w_IPPimg L_IPP^img + w_area L_area + w_BGA L_BGA
    + w_amp L_amp + w_fwd L_fwd
```

All weights except those of L_phase and L_seg are zero in the baseline.

* `L_phase` = ‖φ̂−φ‖₁ + 0.5‖|∇φ̂|−|∇φ|‖₁ + 0.2(1−SSIM); `L_seg` = Dice + CE (class weights 0.5, 1.0).
* **Measurement-aware objectives** (couple predicted phase to the predicted foreground
  probability f; no model of image formation):
  * `L_IPP^cell` (key `cell_integrated_phase`): mean over reference instances k of
    |Σ_{Ω_k} f φ̂ − Σ_{Ω_k} M φ| / (|Σ_{Ω_k} M φ| + ε), instances with reference integral ≥ 50 rad·px,
    relative error capped at 10, domains = watershed instances of the reference mask on the full field.
  * `L_IPP^img` (key `image_integrated_phase`; deprecated alias `phase_volume`): the same
    relative error with sums over the whole crop.
  * `L_area` (key `cell_projected_area`): per-instance relative error of Σ f against Σ M, instances ≥ 370 px.
  * `L_BGA` (key `boundary_gradient_alignment`): ℓ₁ distance between max-normalised |∇f| and |∇φ̂|.
* `L_amp` (key `amplitude`): ℓ₁ to the reconstruction-derived amplitude reference.
* **Forward-model consistency** `L_fwd` (key `forward_model`): the predicted field
  Â e^{i(φ̂+Ψ)} is propagated and expanded into the intensity components of the recorded
  geometry; radiometric constants are removed by a per-image ridge least-squares fit; the loss is
  the unexplained fraction of the hologram variance after discarding a 79 px border. z is fixed
  at 33.77 µm or, in one configuration, trainable (learning rate 100× the base rate, no weight decay).
  The configured tolerance for a usable margin is 0.01 (`loss.forward_model.discrimination_tolerance`).
* Legacy terms, weight 0 in every manuscript configuration and retained so that old configs
  load: `phase_mask_contrast` (class `LegacyPhaseMaskContrast`), `dry_mass_consistency`,
  `projected_area_consistency`. Old Python names are resolved through
  `holoqpi/losses/_legacy.py` with a deprecation warning.

Gradient path (`scripts/check_gradient_path.py`, `runs/gradient_path_b_cell_ipp_512.csv`, 30
batches of 512×512 crops): at unit weight the gradient of L_IPP^cell into the segmentation
decoder has a median magnitude **0.373** (range **0.243–0.655**) times that of L_seg on the
same parameters. The effective ratio is w × this value.

## 7. Evaluation metrics

| Family | Metrics |
|---|---|
| Phase | MAE (field, inside reference cells), Pearson r, SSIM, bias |
| Segmentation / detection | Dice, AJI, boundary F1 (2 px), recall, precision (IoU ≥ 0.5 matching) |
| Per-cell (matched cells) | MAPE, per-cell Pearson r, bootstrap CI (2000 resamples over fields) of area, circularity, dry mass; relative Bland–Altman on per-field medians |
| Coverage | coverage-adjusted MAPE = r_det·MAPE + (1 − r_det) |
| Field totals | sums over all predicted and all reference cells: MAPE, bias, Pearson r (area, dry mass; none for circularity) |
| Amplitude | MAE, RMSE, bias, Pearson r, MAE inside cells, MAE of A = 1 |
| Forward model | residual of the prediction and of the reference phase, ratio |
| Decomposition | domain factor S(Ω̂,φ)/S(Ω,φ), phase factor S(Ω̂,φ̂)/S(Ω̂,φ); geometric mean and median \|log\| at matched-cell (all/interior/edge) and field level (`scripts/decompose_mass_error.py`) |

Edge cell: reference instance with a pixel within 3 px of the field boundary
(`edge_margin_px: 4` with strict inequality in `decompose_mass_error._edge_labels`).

## 8. Experimental protocol and statistics

* Training: 60 epochs, seeds 42 / 1337 / 2024 for the baseline, +IPP (per-cell) and +IPP (image),
  seed 42 for all other configurations (`config/v2/`, `run_v2.sh`, `BENCHMARKING.md`).
* Checkpoint selection on validation: 0.35 Dice + 0.25 r_φ + 0.25 (1 − MAPE_mass) + 0.15 F1_det.
* Resolution criterion: |Δ| > 2·sqrt((s₁² + s₂²)/2) with s the between-seed sample SD (ddof = 1)
  of each configuration, and at least three training runs on both sides. Otherwise: "Not estimable
  from the available runs" (no zero or borrowed variance). No hypothesis test is performed.
* Classical pipelines (`scripts/conventional_baseline.py`): off-axis first-order sideband
  isolation (radius 130 px, DC exclusion 60 px), conjugate selection by skewness, back-propagation
  to 33.77 µm; in-line back-propagation of √I followed by 20 Gerchberg–Saxton iterations; both then
  unwrapped, order-3 background polynomial removed, reference mask rule applied. Deterministic,
  run once.
* Common-field evaluation: `--tag common` with `data.exclude.modalities` extended to off-axis
  (107 fields).
* Result package: `analysis/v42/compile_results.py` → `results_for_manuscript/`; checks:
  `analysis/v42/check_numbers.py`.

## 9. Verified diagnostics

* Forward model (`scripts/calibrate_z.py`, `runs/z_calibration.json`; scope: the implemented
  operator): in-line pooled residual minimum at the grid point 33.68 µm nearest the 33.77 µm
  recording distance, on all four scanned fields; off-axis per-field best distances −96.2, +96.2,
  33.7, 33.7 µm (two at the scan limits): the off-axis residual does not constrain z. Phase-scale
  probes: scaling the reference phase by 0.9 / 0.5 changes the off-axis residual by +0.00035 /
  +0.0080 on 32 validation fields (below the 0.01 tolerance) and the in-line residual by +0.0043 /
  +0.0622.
* Free z: 34.10 µm after epoch 1, 33.41 µm after 60 epochs; checkpoint (epoch 44) scored at
  33.408 µm.
* Boundary sensitivity (`scripts/error_propagation.py`) and synthetic chain check
  (`scripts/synthetic_validation.py`): see `results_for_manuscript/boundary_sensitivity/`.

## 10. Configuration reference

`config/base.yaml` holds every value; each file in `config/v2/` extends it and overrides only
what that configuration changes (`config/v2/_shared.md`).

| Section | Governs |
|---|---|
| `project` | seed, determinism |
| `paths`, `formats` | data directories, phase-file header layout |
| `optics` | λ, α, pixel pitches, aberration and conjugate settings |
| `data` | modality, crop, `exclude`, normalisation, batching, splits, augmentation |
| `membrane` | membrane-channel registration constants |
| `mask_generation` | label construction |
| `model` | encoder, decoder widths, heads, front end, LoRA |
| `loss` | weights and each term's parameters (`loss.physics` holds the shared parameters of the mask-coupling terms; the name is retained for stored configs) |
| `training` | epochs, optimiser, schedule, AMP, checkpoint selection |
| `evaluation` | matching, measurement bounds, classical-baseline settings |
| `deploy` | ONNX export, benchmark protocol (50 warm-up, 500 timed passes, batch 1, 900 px) |

Inline overrides use dotted paths: `python main.py train --config config/v2/g_baseline_gabor.yaml --set training.epochs=100`.
Augmentation is restricted to flips and quarter-turns, applied identically to hologram, phase and mask.

## 11. Extending the code

| Task | Where |
|---|---|
| New encoder | `_ENCODERS` in `holoqpi/models/encoders.py` (five feature maps, `out_channels`) |
| New loss term | `holoqpi/losses/terms.py`, wire into `composite.py`, add the weight to the YAML |
| New metric | module in `holoqpi/metrics/`; the evaluator merges every family's `compute()` |
| Manual annotations | `paths.manual_mask_dir` |

Re-run `python scripts/selftest.py` after any change to the measurement chain or the loss.

## 12. Limitations

Labels are derived from the reference phase and are not independent annotations; the amplitude
reference is a classical reconstruction, not a measurement; the dataset is one instrument with
three cell lines; replicated configurations have three training runs and all others one (so no
between-seed comparison involving them is estimable); recall at the field boundary is low
(0.212 for edge instances against 0.912 for interior ones, baseline); benchmarking is on one
workstation GPU.
