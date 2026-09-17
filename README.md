# HoloQPI — End-to-End Lightweight Holographic Analysis

A single lightweight network that takes a **raw hologram** and directly produces a
**quantitative phase image**, a **transmitted-amplitude map** and a **cell
segmentation map**, from which projected area, circularity, optical volume and dry
mass are computed per cell. No classical reconstruction step appears anywhere in
the inference path.

The study compares two imaging configurations on identical data, splits and
schedules:

| | input |
|---|---|
| **Off-axis** | `data/off_axis_hologram/<stem>_holo.tif` |
| **In-line Gabor** | `data/in_line_gabor_hologram/<stem>_gabor.tif` |

```
                        ┌──────────────► phase decoder ──► quantitative phase (rad)
raw hologram ──► shared ├──────────────► amplitude head ──► transmitted amplitude
                encoder ├──────────────► segmentation decoder ──► cell map
                        └──────────────► classifier ──► drug condition (disabled)
                                                │
                                                ▼
                        projected area · circularity · optical volume · dry mass
```

The **amplitude head** is off by default (`model.amplitude.enabled`) and switched
on in arms D0, D1 and D2; with it off the specimen is treated as purely
refractive (A = 1), the standard thin-phase-object assumption.

The **condition classifier** is disabled throughout the v2 study
(`model.classifier_enabled: false`). It was unstable at ±9.5 accuracy points
between seeds, which is outside the question this study asks. The head remains in
the code and can be re-enabled.

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

# 2. verify the measurement chain, the loss terms and the metrics
#    143 checks; it is the gate on everything below and must end
#    with "All checks passed."
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
├── last_model.pt            final epoch; nothing reads it, kept for restarts
├── resolved_config.yaml     the exact configuration this run used
├── history.json             per-epoch losses, validation metrics, and the
│                            learned propagation distance when z is free
├── metrics_test.json        every metric family, including the amplitude rows
│                            for an arm whose amplitude head is on
├── confusion_test.json      condition confusion matrix, when that head is on
├── per_cell_test.csv        one row per matched cell, predicted vs reference
├── unmatched_test.csv       missed reference cells and false positives, with
│                            their areas and masses — the input to figure 15
└── <experiment>_<modality>_fp32.onnx

runs/RESULTS.md                               every table, assembled — read this first
runs/results_table.csv                        the same tables, machine-readable
runs/seed_aggregate.json                      the significance verdicts
runs/z_calibration.json                       is the propagation distance identifiable?
runs/<experiment>_modality_comparison.csv     off-axis vs Gabor, side by side
runs/<experiment>_hardware_benchmark_*.csv    params, GMACs, latency, FPS, VRAM
runs/error_propagation_summary.csv            boundary error -> measurement error
runs/synthetic_validation_fields.csv          the pipeline's own floor
```

**`runs/RESULTS.md` is the document to read and to paste from.** It is assembled
by `scripts/collect_results.py` from the files above; every number in it is read
from a file and none is typed. It carries Table 1 (the ablation), Table 2 (the
differences with the significance rule applied), Table 3 (measurement quality),
Table 3b (the physics-aware forward model and the learned-z trajectory),
**Table 3c (the amplitude output)**, Table 4 (efficiency), and the two results
that need no trained model.

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

`CLEAN=1` matters when an earlier result set is present: results written under a
different objective are not comparable with what follows, and the driver below
archives rather than overwrites them.

### The two drivers that run it end to end

`run_v2.sh` runs one stage or one arm at a time. For a full unattended run, the
two drivers below sequence every stage in dependency order, use GPU 2 and GPU 3
only, and are safe to re-run: each records the phase it finished in
`.run_state/` or `.finish_state/`, so an interrupted run picks up where it
stopped.

```bash
nohup bash RUN_ON_SERVER.sh   > run.out 2>&1 &     # the whole study, ~38 h
bash RUN_ON_SERVER.sh --status                     # where it is

nohup bash FINISH_ON_SERVER.sh > finish_gaps.out 2>&1 &   # scoring only, ~1 h
bash FINISH_ON_SERVER.sh --status
```

`RUN_ON_SERVER.sh` archives any existing `runs/` and `figures/` to
`*_before_audit_<timestamp>`, gates on the self-test, regenerates the masks with
provenance, then runs stages 1, 2, 3 and 5, trains on both GPUs in parallel, and
finishes with the classical baseline, the benchmark, the figures and the tables.
`FINISH_ON_SERVER.sh` re-scores and rebuilds the reporting without retraining
anything.

**Delete `.run_state/` and `.finish_state/` before reusing a folder.** The
drivers read them as "already done" and will skip every phase while appearing to
succeed.

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
python scripts/make_figures.py --config config/base.yaml           # all twenty
python scripts/make_figures.py --config config/base.yaml --list    # what each one shows
python scripts/make_figures.py --config config/base.yaml --only 2 4 5
```

Writes `figures/fig<NN>_<name>.png` and `.pdf`. Twenty figures are registered and
**eighteen build** on the current study: figure 7 needs the v1 ablation results
and figure 11 needs the condition head, which is disabled. Most read files the
earlier steps already produced, so they regenerate in seconds; a figure whose
inputs are missing prints its reason and is skipped without stopping the rest.

`figures/_manifest.json` records, per figure, which config, experiment, modality
and split produced it — the filenames carry none of that, and stage 9 builds the
set from three different configs. It is **merged** across invocations, and a
figure that was skipped has its entry removed, so the manifest never vouches for
a file left over from an earlier run.

What each figure is for is tabulated in
[`docs/documentation.md` §10a](docs/documentation.md).

## Segmentation labels

The delivered dataset has no manual cell annotations. `prepare` derives binary
masks from the ground-truth phase (smoothing → Otsu → morphological cleanup →
area filtering). These are **silver-standard** labels and must be described as
such in any write-up.

When manual annotations arrive, point `paths.manual_mask_dir` at them; nothing
else changes.

## The amplitude reference

`scripts/prepare_amplitude.py` writes the amplitude target as the modulus of a
classical off-axis reconstruction, normalised so the mask background reads 1. It
is **not a measurement**: it carries the sideband filter's lost high frequencies,
residual twin-image structure and any illumination vignetting. Every amplitude
number the framework reports is therefore agreement with one reconstruction
algorithm's output and must be described in those words.

Because of that, the amplitude metric is reported against two comparators: the
reference itself, and the thin-phase-object assumption A = 1 that the rest of the
study runs on (`amplitude_mae_over_unity`). Below 1.0 the head is closer to the
reference than that assumption; a head that has collapsed to a constant scores
exactly 1.000 with zero predicted spread, which is the failure this comparator
exists to expose.

## Moving files between machines

Two whole-folder copies between a workstation and the compute server have
damaged this project's source tree, once leaving 16 of 39 modules in place and
producing `ModuleNotFoundError: No module named 'holoqpi.analysis'`. The rules
that follow:

* Off the server: `tar -czf` the named result files and move one archive.
* Onto the server: extract into an empty folder first, then copy.
* Never whole-folder paste in either direction.
* `data/` and `weights/` are multi-gigabyte inputs, not clutter. Replace only
  code, `runs/` and `figures/`.
* Verify before running anything:

```bash
find holoqpi -name "*.py" | wc -l      # must be 39
python -c "import holoqpi.analysis.cells, holoqpi.deploy.benchmark, holoqpi.metrics.amplitude"
```

---

Mathematical formulation, parameter reference and experimental protocol:
**[`docs/documentation.md`](docs/documentation.md)**.

Results: **`runs/RESULTS.md`**. Operational runbook:
[`docs/RUNBOOK.md`](docs/RUNBOOK.md). Change records, most recent first:
[`docs/gap_closure_2026-09-17.md`](docs/gap_closure_2026-09-17.md),
[`docs/code_audit_2026-09-16.md`](docs/code_audit_2026-09-16.md),
[`docs/v2_code_changes.md`](docs/v2_code_changes.md).
