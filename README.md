# HoloQPI — End-to-End Lightweight Holographic Analysis

A single lightweight network that takes a **raw hologram** and directly produces a
**quantitative phase image** and a **cell segmentation map**, from which projected
area, circularity, optical volume and dry mass are computed per cell.

The study compares two imaging configurations on identical data, splits and
schedules:

| | input |
|---|---|
| **Off-axis** | `data/off_axis_hologram/<stem>_holo.tif` |
| **In-line Gabor** | `data/in_line_gabor_hologram/<stem>_gabor.tif` |

```
                        ┌──────────────► phase decoder ──► quantitative phase (rad)
raw hologram ──► shared │
                encoder ├──────────────► segmentation decoder ──► cell map
                        │
                        └──────────────► classifier ──► drug condition (5-way)
                                                │
                                                ▼
                        projected area · circularity · optical volume · dry mass
```

Everything numeric — wavelength, refraction increment, pixel pitches, loss
weights, thresholds, schedules — lives in `config/*.yaml`. The Python sources
contain no hard-coded physical or hyper-parameter values.

---

## Install

**Python 3.10 or newer is required.** The sources use PEP 604 annotations and
PyTorch 2.x; on an older interpreter they fail to parse, which appears as a
`SyntaxError` on the first function definition rather than a useful message.

```bash
conda env create -f environment.yml
conda activate qpi_extended
python -V                      # expect 3.11.x
```

Or into an existing Python >= 3.10 environment:

```bash
pip install -r requirements.txt
```

Confirm the GPU is visible before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If pip resolves a CUDA build the driver cannot run, install torch first from
<https://pytorch.org/get-started/locally/> and then `pip install -r requirements.txt`.

## Data layout

```
data/
├── off_axis_hologram/      <stem>_holo.tif    1024 x 1024, 8-bit, LZW
├── in_line_gabor_hologram/ <stem>_gabor.tif   1024 x 1024, 8-bit, LZW
├── phase/                  <stem>_phase.bin   900 x 900 float32 + 23-byte header
└── mask/                   <stem>_mask.png    written by `prepare`
```

Stems encode the labels, e.g. `NCI_01`, `NCI_Blebbistatin_5uM_07`,
`SNU_staurosporine_23`, `T24_rotenone_500nM_50`. Parsing is case-insensitive and
the concentration token is optional, since the delivered data is inconsistent
about both.

Holograms are **centre-cropped** 1024 → 900 to reach the phase sampling grid; the
reconstruction crops the camera frame rather than resampling it.

## Run

```bash
# 1. masks, stratified splits, manifest, calibration report
python main.py prepare   --config config/base.yaml

# 2. verify the measurement chain and the loss terms
python scripts/selftest.py --config config/base.yaml

# 3. train one arm
python main.py train     --config config/off_axis.yaml
python main.py train     --config config/gabor.yaml

# 4. the central comparison: both arms, one table
python main.py compare   --config config/base.yaml

# 5. efficiency and deployment
python main.py benchmark --config config/base.yaml
python main.py export    --config config/off_axis.yaml --precision fp16
```

Any configuration key can be overridden inline:

```bash
python main.py train --config config/gabor.yaml --set training.epochs=100 data.batch_size=8
```

## Outputs

```
runs/<experiment>_<modality>/
├── best_model.pt            checkpoint selected on the composite metric
├── resolved_config.yaml     the exact configuration this run used
├── history.json             per-epoch losses and validation metrics
├── metrics_test.json        every metric family
├── confusion_test.json      5-way drug-condition confusion matrix
└── per_cell_test.csv        one row per matched cell, predicted vs reference

runs/<experiment>_modality_comparison.csv     off-axis vs Gabor, side by side
runs/<experiment>_hardware_benchmark_*.csv    params, GMACs, latency, FPS, VRAM
```

## Ablations

Each is a config file, so the command stays a one-liner:

```bash
python main.py compare --config config/ablation/no_physics.yaml
python main.py compare --config config/ablation/no_measurement.yaml
python main.py compare --config config/ablation/classification_only.yaml
```

`no_physics` removes every physics and measurement term; `no_measurement` keeps
the previous study's three terms and drops only the two this study adds;
`classification_only` trains the condition head alone, as a control for the
off-axis / Gabor classification gap.

## Running the whole study

For the **v2 study** (the current one), use `run_v2.sh`:

```bash
CLEAN=1 nohup bash run_v2.sh > v2.out 2>&1 &   # everything, in dependency order
bash run_v2.sh --stage 5                       # one stage
ARMS="A B B1" bash run_v2.sh                   # only these arms
SEEDS="1337 2024" bash run_v2.sh               # replicate the main arms
NO_TRAIN=1 bash run_v2.sh                      # reuse checkpoints, re-evaluate only
QUICK=1 bash run_v2.sh                         # tiny settings, plumbing only
```

Ten stages under `logs/v2_<timestamp>/`. Two of them produce results that need
no trained model at all — stage 5's boundary-error propagation and synthetic
ground-truth validation — so those survive any training failure. Stage 4 sets
the per-cell loss weight from a measured gradient ratio and decides whether the
forward-model term is usable; read it before stage 6. `config/v2/_shared.md`
describes the thirteen arms and the one question each answers.

`CLEAN=1` matters: the pixel pitch and refraction increment were corrected on
2026-09-10, so any checkpoint or metric written before then reports areas 34.4%
low and masses 24.3% low, and is not comparable with anything written after.

For the **v1 study** (already reported), `run_study.sh` is unchanged:

```bash
bash run_study.sh                 # everything, in dependency order
bash run_study.sh --stage 5       # one stage
NO_TRAIN=1 bash run_study.sh      # reuse checkpoints, re-evaluate only
QUICK=1 bash run_study.sh         # tiny settings, to prove the plumbing works
```

Eleven stages, each logged separately under `logs/study_<timestamp>/`, continuing
past failures. Three logs to read first: `02_calibrate_z.log` (is z
identifiable? gates the physics ablation), `07_conventional_baseline.log` (is the
classical reconstruction valid?), and `03_audit_labels.log` (does the label
threshold drift with condition?).

## Diagnostics

```bash
python scripts/selftest.py --config config/base.yaml       # measurement chain and losses
python scripts/diagnose_bias.py --config config/base.yaml  # boundary error or phase error?
python scripts/audit_labels.py --config config/base.yaml   # are the silver labels sound?
python scripts/calibrate_z.py --config config/base.yaml    # recover the propagation distance
python scripts/conventional_baseline.py --config config/base.yaml   # the classical floor
python scripts/estimate_aberration.py --config config/base.yaml     # the removed surface
python scripts/prepare_amplitude.py --config config/base.yaml       # the amplitude reference
python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml --batches 30
python scripts/amplitude_sensitivity.py --config config/base.yaml --split test
python scripts/error_propagation.py --config config/base.yaml       # boundary -> measurement
python scripts/synthetic_validation.py --config config/base.yaml    # the pipeline's own floor
```

The last four were added on 2026-09-10 and each answers a question that was
previously settled by argument instead of measurement:

* `check_gradient_path.py` — the per-cell term's gradient into the segmentation
  decoder, as a median over batches, plus its **cosine** with the segmentation
  gradient. It prints the weight arithmetic explicitly (`effective = w x ratio`)
  because the ratio is measured at weight 1.0 and was once read as if it were
  the effective ratio at any weight.
* `amplitude_sensitivity.py` — whether the forward-model residual responds to
  the amplitude, and whether it responds to the phase *in the right direction*
  at both a mild (x0.9) and a coarse (x0.5) degradation. On this data the mild
  response has the wrong sign, which a coarse-only test does not reveal.
* `error_propagation.py` — the exchange rate between a segmentation error and a
  measurement error. No model involved: reference masks, reference phase, and a
  boundary moved a known number of pixels.
* `synthetic_validation.py` — the measurement chain against exact analytic
  ground truth (spherical caps, closed-form area and integrated phase), which
  is the only way to separate pipeline error from model error.

`calibrate_z` recovers the sample-to-sensor distance the forward-model loss
needs, by matching each hologram against its reference phase in both directions.
It reports whether z is identifiable and **refuses to return a number when it is
not**, rather than handing back a confident-looking guess. It also reports how
sensitive each geometry's hologram is to phase at that distance, which is what
decides whether the forward-model term can teach that arm anything at all.

`conventional_baseline` runs the textbook reconstruction for both geometries
through the same evaluator as the network. Check `reconstruction_valid` in its
output before quoting any number from it.

`audit_labels` answers the two questions the phase-derived masks raise: whether
the per-image Otsu level drifts with drug condition (which would confound the
classification result), and how much of the segmentation head is already implied
by the phase head (which explains why the physics terms are inert). Add
`--skip-redundancy` to run the threshold half alone — it needs no checkpoint and
no GPU.

## Figures

```bash
python scripts/make_figures.py --config config/base.yaml           # all eighteen
python scripts/make_figures.py --config config/base.yaml --list    # what each one shows
python scripts/make_figures.py --config config/base.yaml --only 2 4 5
```

Writes `figures/fig<NN>_<name>.png` and `.pdf`. Most read files the earlier steps
already produced, so they regenerate in seconds; a figure whose inputs are missing
prints its reason and is skipped without stopping the rest. What each figure is
for is tabulated in [`docs/documentation.md` §10a](docs/documentation.md).

## Segmentation labels

The delivered dataset has no manual cell annotations. `prepare` derives binary
masks from the ground-truth phase (smoothing → Otsu → morphological cleanup →
area filtering). These are **silver-standard** labels and must be described as
such in any write-up.

When manual annotations arrive, point `paths.manual_mask_dir` at them; nothing
else changes.

---

Mathematical formulation, parameter reference and experimental protocol:
**[`docs/documentation.md`](docs/documentation.md)**.
