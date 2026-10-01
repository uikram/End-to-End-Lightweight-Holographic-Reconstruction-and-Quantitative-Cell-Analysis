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

Every experiment-level constant — wavelength, refraction increment, pixel
pitches, loss weights, thresholds, schedules — lives in `config/*.yaml`. The
Python sources contain no hard-coded physical or hyper-parameter values; what
remains in code is algorithmic constants and numerical safeguards (stability
limits, epsilons, tensor shapes), each documented at its definition.

Current values: λ = 0.666 µm, α = 0.2 mL/g, isotropic pixel pitch 0.284871 µm.
Two of these were corrected on 2026-09-10 and **absolute** areas and masses from
before that date are not comparable; see
[`docs/documentation.md` §3](docs/documentation.md). Relative results are
invariant to both, which the self-test proves.

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

# 3. train and evaluate one arm (every arm is a file in config/v2/)
python main.py train     --config config/v2/a_baseline.yaml
python main.py evaluate  --config config/v2/a_baseline.yaml

# 4. off-axis against in-line: arm A and arm G, one table
python main.py compare   --config config/v2/a_baseline.yaml --modalities off_axis gabor

# 5. efficiency and deployment
python main.py benchmark --config config/v2/a_baseline.yaml --modalities off_axis
python main.py export    --config config/v2/k_compact_a.yaml --precision fp16
```

Any configuration key can be overridden inline:

```bash
python main.py train --config config/v2/g_baseline_gabor.yaml --set training.epochs=100 data.batch_size=8
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
runs/seed_aggregate.json                      between-seed spread and resolution verdicts
runs/benchmark_results/results_*.json         every benchmark over all seeds: mean, SD, n, 95% CI
runs/z_calibration.json                       is the propagation distance identifiable?
runs/<experiment>_modality_comparison.csv     off-axis vs Gabor, side by side
runs/<experiment>_hardware_benchmark_*.csv    params, GMACs, latency, FPS, VRAM
runs/error_propagation_summary.csv            boundary error -> measurement error
runs/synthetic_validation_fields.csv          the pipeline's own floor
```

**Two result documents, one for each question.**
`runs/benchmark_results/` is the one to report from: one JSON per arm and per
benchmark, pooling every seed (mean, sample SD, n, SEM, 95% CI) with the
checkpoint hash and provenance of every run, plus `results_comparisons.json`
(differences of means with the resolution rule applied) and
`benchmark_summary.csv`. The manuscript's tables are generated from it.
`runs/RESULTS.md` is the per-run view, assembled by `scripts/collect_results.py`:
its per-arm tables show the **seed-42** run of every arm, and its Table 2 uses
the seed means. Every number in both is read from a file and none is typed.

## Results at a glance

Off-axis test split, 113 fields, 3186 reference cells. Arms A, B and B′ are
mean ± SD over seeds 42, 1337 and 2024; every other arm is one run (seed 42).
Source: `runs/benchmark_results/`.

| | Classical | A baseline | B + per-cell IPP | B′ + image-level | KA compact |
|---|---|---|---|---|---|
| Dry-mass MAPE (matched cells) ↓ | 0.2245 | **0.1763 ± 0.0024** | 0.1915 ± 0.0044 | 0.1935 ± 0.0029 | 0.1783 |
| Dry-mass MAPE, coverage-adjusted ↓ | — | 0.5124 ± 0.0009 | 0.5179 ± 0.0030 | 0.5291 ± 0.0017 | 0.5136 |
| Projected-area MAPE ↓ | 0.1567 | 0.1546 ± 0.0010 | 0.1630 ± 0.0019 | 0.1643 ± 0.0021 | 0.1533 |
| Phase MAE [rad] ↓ | 0.2114 | 0.1593 ± 0.0008 | 0.1697 ± 0.0022 | 0.1666 ± 0.0006 | 0.1599 |
| Dice ↑ | 0.7885 | 0.8312 ± 0.0014 | 0.8245 ± 0.0015 | 0.8242 ± 0.0007 | 0.8321 |
| Boundary F1 ↑ | 0.2463 | 0.4574 ± 0.0018 | 0.4353 ± 0.0056 | 0.4223 ± 0.0047 | 0.4598 |
| Detection recall ↑ | 0.5763 | 0.5920 ± 0.0011 | 0.5964 ± 0.0020 | 0.5839 ± 0.0024 | 0.5920 |

* **The measurement-aware terms do not help.** B vs A: +0.0152 dry-mass MAPE
  against a 2×SD threshold of 0.0070 (resolved, worse); B′ vs A: +0.0172 against
  0.0052 (resolved, worse). Every single-seed coupling arm (B″, C, D0, D1, D2,
  and the weight sweep 0.1/0.3/3.0) lands between 0.1828 and 0.2102 —
  all worse than A, none resolvable at one run.
* **In-line (Gabor):** the classical pipeline fails (in-cell phase contrast
  −0.067 rad, 116 of 3186 cells matched); the learned arm G reaches dry-mass
  MAPE 0.2308, Dice 0.7758, phase r 0.8098 over 1755 matched cells.
* **Amplitude head:** MAE 0.0637 (D1) against 0.1469 for the thin-phase
  assumption A = 1 (ratio 0.434). The reference is a reconstruction, not a
  measurement.
* **No model needed:** a 1-px boundary shift costs +6.9 % area and +4.8 %
  dry mass (Dice 0.974); dry mass is 1.41× less sensitive to boundary error than
  area. The measurement chain's own floor on analytic fields is 0.036 ± 0.004 %
  per cell (mass) and 0.224 ± 0.018 % (area), over three synthetic seeds.
* **Cost (RTX A5000, 900×900, batch 1):** arm A 12.32 ± 0.11 ms PyTorch FP32,
  8.42 ± 0.06 ms ONNX FP16 (119 FPS), 36.77 MB → 18.38 MB weights; compact KA
  7.98 ms ONNX FP16 (125 FPS), 6.48 MB. Workstation figures, not an edge claim.
* **Not bit-exact:** evaluation uses `cudnn.benchmark`, so re-evaluating a
  checkpoint can move a metric by up to about 0.002.

## Running the whole study

For the **v2 study** (the current one), use `run_v2.sh`:

```bash
CLEAN=1 nohup bash run_v2.sh > v2.out 2>&1 &   # everything, in dependency order
bash run_v2.sh --stage 5                       # one stage
ARMS="A B B1" bash run_v2.sh                   # only these arms
bash run_v2.sh --stage 11                      # every arm at every seed (BENCHMARKING.md)
NO_TRAIN=1 bash run_v2.sh                      # reuse checkpoints, re-evaluate only
QUICK=1 bash run_v2.sh                         # tiny settings, plumbing only
```

Fourteen stages under `logs/v2_<timestamp>_gpu<N>/`. Stages 11–14 re-run every
benchmark on the trained models and collect the results with statistics; the commands,
seeds, output files and statistical method are in **`BENCHMARKING.md`**. Two of them produce results that need
no trained model at all — stage 5's boundary-error propagation and synthetic
ground-truth validation — so those survive any training failure. Stage 4 sets
the per-cell loss weight from a measured gradient ratio and decides whether the
forward-model term is usable; read it before stage 6. `config/v2/_shared.md`
describes the thirteen arms and the one question each answers.

`CLEAN=1` matters when an earlier result set is present: results written under a
different objective are not comparable with what follows, and the driver
archives rather than overwrites them.

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
**eighteen build** on the current study: figure 7 needs the v1 ablation results,
which are not part of this version, and figure 11 needs the condition head,
which is disabled. Most read files the
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

Results: **`runs/RESULTS.md`**, and every benchmark with its statistics in
**`runs/benchmark_results/`** ([`BENCHMARKING.md`](BENCHMARKING.md)).

To remove old results and clutter from a copy of this folder: `bash cleanup.sh`
on the server, `CLEANUP.cmd` on Windows.
