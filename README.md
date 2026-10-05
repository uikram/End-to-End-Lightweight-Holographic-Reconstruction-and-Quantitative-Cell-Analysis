# Joint Phase Reconstruction and Cell Segmentation from Raw Holograms for Per-Cell Dry-Mass Measurement

Code, configurations, stored results and analysis scripts of the study. Every number reported
in the paper is traceable to a file under `runs/` or `config/` through `results_for_manuscript/`.
The manuscript itself is not part of this repository.

## 1. Study overview

Per-cell dry mass in quantitative phase imaging is the phase integrated over a
segmented cell, so it depends on the reconstructed phase and on the cell boundary
together. One network (a MobileNetV2 encoder with two U-Net-style decoders,
9.6 M parameters) maps a raw digital hologram to a quantitative phase map and a
cell segmentation, and optionally to a transmitted-amplitude map. A fixed
measurement chain computes per-cell quantities from these outputs, and every
prediction is scored with phase and segmentation metrics and with the paired
per-cell measurement error. Segmentation labels are derived from the reference
phase (no manual annotation). The study compares the network with one classical
reconstruction pipeline per geometry, tests measurement-aware integrated-phase
objectives and a forward-model consistency term, and reports model-free
boundary-sensitivity analyses.

## 2. Pipeline

```
raw hologram ──► lightweight network ──┬─► quantitative phase (rad)
 (off-axis or                          ├─► 2-class segmentation ──► instances (watershed)
  in-line Gabor)                       └─► transmitted amplitude (+Amplitude / +Fwd only)
                                                      │
                          per-cell measurements: projected area A_k, circularity C_k,
                          integrated phase S_k, dry mass m_k = λ/(2πα) · S_k
```

No classical reconstruction enters the inference path.

## 3. Dataset

800 fields of view of three human cancer cell lines (NCI-H1299, SNU-475, T-24),
each under a control condition or one of four drugs. Each field has an off-axis
hologram and an in-line Gabor hologram (1024×1024, 8-bit), a 900×900 float32
quantitative phase map (rad, supplied pre-corrected) and a membrane-stain
fluorescence image. λ = 0.666 µm, pixel pitch 0.284871 µm, supplied propagation
distance 33.77 µm, α = 0.2 mL/g. Holograms are cropped 1024 → 900 px without
resampling, with the crop origin moved by (−6, −2) px (`data.crop_offset_px`).

```
data/
├── manifest.csv, splits.json      in the repository
├── off_axis_hologram/  <stem>_holo.tif
├── in_line_gabor_hologram/ <stem>_gabor.tif
├── phase/ <stem>_phase.bin        900×900 float32 + 23-byte header
└── mask/  <stem>_mask.png         written by `python main.py prepare`
```
(image data are not in the repository.)

## 4. Splits

| Set | Fields | Reference cells |
|---|---|---|
| Off-axis train / validation / test | 560 / 127 / 113 | test: 3186 |
| In-line train / validation / test | 521 / 122 / 107 | test: 3055 |
| Common test set (both geometries) | 107 | 3055 |

The in-line frames of SNU_01–SNU_50 do not show the same field of view as their
off-axis holograms; they are excluded from the in-line modality before training
(39 train + 5 validation + 6 test fields), `data.exclude` in `config/base.yaml`.
Off-axis configurations keep all 800 fields.

## 5. Configurations

Each differs from its comparator by one term, output or capacity choice
(`config/v2/`). **Training runs** are counted as trained models.

| Configuration | Config file | Training runs (seeds) | Comparator |
|---|---|---|---|
| End-to-End Neural Baseline | `a_baseline.yaml` | **3** (42, 1337, 2024) | – |
| +IPP (per-cell) | `b_cell_ipp.yaml` | **3** (42, 1337, 2024) | Baseline |
| +IPP (image) | `b1_image_volume.yaml` | **3** (42, 1337, 2024) | Baseline |
| In-Line Neural Configuration | `g_baseline_gabor.yaml` | 1 (42) | Baseline |
| +Area, +BGA | `b2_cell_area.yaml`, `c_cell_ipp_bga.yaml` | 1 (42) each | +IPP (per-cell) |
| +IPP (per-cell), w = 0.1 / 0.3 / 3.0 | `w_ipp_01/03/30.yaml` | 1 (42) each | Baseline |
| +Amplitude | `d0_amplitude.yaml` | 1 (42) | +IPP (per-cell) |
| +Fwd (fixed z), +Fwd (free z) | `d_forward_amplitude.yaml`, `d2_learned_z.yaml` | 1 (42) each | +Amplitude, +Fwd (fixed z) |
| Compact Baseline, Compact +IPP | `k_compact_a.yaml`, `k_compact_b.yaml` | 1 (42) each | Baseline, +IPP (per-cell) |
| Classical off-axis / in-line | `scripts/conventional_baseline.py` | deterministic, run once | – |

Single-run configurations do not support a between-seed comparison. `w_ipp_10.yaml`
is a separate seed-42 training of the +IPP (per-cell) configuration (a repeat
check, not a fourth seed). `config/v2/l_lora.yaml` is an optional LoRA variant
not used in the paper.

Loss (Methods of the paper): `L = L_phase + L_seg + w·L_IPP^cell + w·L_IPP^img +
w·L_area + w·L_BGA + w·L_amp + w·L_fwd`; weights other than the first two are zero
in the baseline. Measurement-aware objectives: L_IPP^cell, L_IPP^img, L_area, L_BGA
(config keys `cell_integrated_phase`, `image_integrated_phase`,
`cell_projected_area`, `boundary_gradient_alignment`). Forward-model consistency:
`forward_model`. Older keys (`phase_volume`, `phase_mask_contrast`) are legacy aliases.

## 6. Measurement definitions

Cell domains are instances of the mask after a distance-transform watershed
(minimum peak distance 15 px, area 30–6000 µm²), applied identically to
predicted and reference masks.

```
A_k = N_k dx dy        C_k = min(1, 4π N_k / P_k²)      S_k = Σ φ dx dy      m_k = λ/(2πα) S_k
```

Predicted and reference instances are paired by greedy descending IoU (≥ 0.5).
Because λ/(2πα) multiplies predicted and reference alike, the relative integrated-phase
error equals the relative dry-mass error; it is reported once, as dry-mass MAPE.

## 7. Evaluation metrics

* Phase: MAE (field and inside reference cells), Pearson r, SSIM.
* Segmentation and detection: Dice, AJI, boundary F1 (2 px), recall, precision.
* Per-cell measurement (matched cells only): MAPE and Pearson r of area,
  circularity and dry mass; coverage-adjusted MAPE `r_det·MAPE + (1 − r_det)`.
* Field totals: sums over all predicted and all reference cells (circularity is
  non-additive, so it has no field total).
* Mass-error decomposition: domain factor `S(Ω̂,φ)/S(Ω,φ)` × phase factor
  `S(Ω̂,φ̂)/S(Ω̂,φ)`; geometric mean and median |log| spread.
* Recovered in-cell phase contrast (paper Table S3): median over fields of the mean phase
  inside the reference cells minus the mean outside.
* Resolution criterion: a difference is resolved if |Δ| > 2·sqrt((s₁²+s₂²)/2)
  (between-seed SD, ddof = 1) **and** both configurations have ≥ 3 training
  runs; otherwise the assessment is "Not estimable from the available runs".

## 8. Reproducibility

```bash
conda env create -f environment.yml && conda activate qpi_extended     # Python 3.11
python main.py prepare   --config config/base.yaml                      # masks, splits, manifest
python scripts/selftest.py                                              # must end "All checks passed."
python main.py train     --config config/v2/a_baseline.yaml --seed 42
python main.py evaluate  --config config/v2/a_baseline.yaml --seed 42
bash run_v2.sh                                                          # whole study (stages, seeds, benchmark; logs in logs/)
```

Supporting analyses (each documented in its docstring): `scripts/conventional_baseline.py`
(classical pipelines), `scripts/decompose_mass_error.py`, `scripts/calibrate_z.py`,
`scripts/amplitude_sensitivity.py`, `scripts/error_propagation.py`,
`scripts/synthetic_validation.py`, `scripts/check_gradient_path.py`,
`scripts/register_holograms.py` (off-axis/in-line pairing; writes `runs/diagnostics/`),
`scripts/measure_crop_offset.py`, `scripts/summarise_common_fields.py`,
`scripts/neural_phase_contrast.py`. `BENCHMARKING.md` describes the stages of `run_v2.sh`,
the seed replication and the benchmark protocol.
Checkpoints (`best_model.pt`) are not in the repository; their SHA-256 values are
recorded in `results_for_manuscript/metadata/configurations.json`.

Rebuild the result package and the figures from the stored results (no GPU needed):

```bash
python analysis/compile_results.py                          # runs/ + config/ -> results_for_manuscript/ (526 consistency checks)
python results_for_manuscript/figures/scripts/make_all.py   # Figs. 2, 4, 5, 6, 7, S1, S2 -> figures/regenerated/
```

Fig. 3 and the image panels of Fig. 1 run the trained network
(`results_for_manuscript/figures/scripts/fig3_qualitative.py`, `fig1_panels.py`), so they need
the checkpoints and the image data.

## 9. Paper / result correspondence

`results_for_manuscript/RESULTS_INDEX.md` maps every table and figure of the paper (main text
and supplementary) to its JSON file, source file, source key, training runs, seeds and field
set. `results_for_manuscript/README.md` describes each file of the package.

## 10. Computational benchmarking

900×900 input, batch 1, one NVIDIA RTX A5000, PyTorch 2.6 and ONNX Runtime (opset 17),
FP32 and FP16, 50 warm-up and 500 timed passes. Baseline (9,598,099 parameters,
45.85 GMAC; mean ± SD over three benchmark sessions, one per training seed):
12.28 ± 0.08 ms (PyTorch FP32), 9.26 ± 0.05 ms (PyTorch FP16), 8.39 ± 0.06 ms (ONNX FP16).
Compact Baseline (3,360,403 parameters, one session): 11.44 ms (PyTorch FP32), 7.89 ms
(ONNX FP16). Values: `results_for_manuscript/benchmarking/benchmarking.json`. These are measurements on
one workstation GPU; no deployment on other hardware was evaluated.

## 11. Repository structure

```
main.py                      prepare | train | evaluate | compare | benchmark | export
run_v2.sh                    the whole study in stages (training, seeds, evaluation, benchmark)
holoqpi/                     data, models, losses, physics, metrics, engine, deploy
config/base.yaml, config/v2/ shared settings; one file per configuration (config/v2/_shared.md)
data/                        manifest.csv, splits.json (image data are not in the repository)
scripts/                     analyses, diagnostics, collectors, selftest (see §8)
runs/                        stored experiment outputs (source of truth): one folder per training run,
                             benchmark_results/, common_fields/, diagnostics/, provenance/ (split logs)
analysis/compile_results.py  builds results_for_manuscript/ from runs/ and config/
results_for_manuscript/      result package: JSON per table, RESULTS_INDEX.md, figures/ (data, scripts, output)
test/                        evaluation notebook for the trained checkpoints
docs/documentation.md        technical documentation
BENCHMARKING.md              replication and benchmarking protocol
```

Checkpoints (`*.pt`), ONNX exports, image data and run logs (`logs/`) are not tracked.
