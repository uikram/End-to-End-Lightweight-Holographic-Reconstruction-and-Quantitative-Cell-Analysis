# BENCHMARKING.md — repeated runs, seeds and statistics

Every benchmark reported in the paper is re-run on the trained models and
collected into one JSON file per benchmark with **mean, standard deviation, n,
every seed's value and a 95% confidence interval** where there is more than one
trained model. This file is the procedure.

- [1. What is repeated, and what counts as a replicate](#1-what-is-repeated-and-what-counts-as-a-replicate)
- [2. Seeds](#2-seeds)
- [3. Run everything — exact commands](#3-run-everything--exact-commands)
- [4. Run one benchmark by hand](#4-run-one-benchmark-by-hand)
- [5. Where results are stored](#5-where-results-are-stored)
- [6. The collector and its JSON files](#6-the-collector-and-its-json-files)
- [7. JSON fields](#7-json-fields)
- [8. Statistics](#8-statistics)
- [9. Metrics collected](#9-metrics-collected)
- [10. Latest-run handling: what the collector refuses](#10-latest-run-handling-what-the-collector-refuses)
- [11. Reproducibility and determinism](#11-reproducibility-and-determinism)
- [12. Building a manuscript table from the JSON](#12-building-a-manuscript-table-from-the-json)

---

## 1. What is repeated, and what counts as a replicate

The project has two benchmark commands, and they are repeated differently because
their randomness lives in different places.

| Benchmark | Command | Where the randomness is | Independent replicate |
|---|---|---|---|
| Accuracy and measurement (phase, segmentation, detection, per-cell area / circularity / optical volume / dry mass, amplitude, forward model) | `main.py train` then `main.py evaluate` | **Training**: decoder initialisation, batch order, random crops, flips and rotations | one model **trained** at that seed, evaluated once |
| Latency, throughput, memory, parameters | `main.py benchmark` | the benchmark session (timing noise) | one separate benchmark **process** per seed, timing that seed's checkpoint |
| Measurement-chain floor (synthetic ground truth) | `scripts/synthetic_validation.py` | the randomly generated synthetic cells | one set of synthetic fields per seed |
| Classical baseline, boundary-error propagation, z scan | `conventional_baseline.py`, `error_propagation.py`, `calibrate_z.py` | none | deterministic: run once, n = 1 |

**Evaluation itself is deterministic.** Scoring the same checkpoint five times
reproduces the same numbers, so it is not replication. Repeating an accuracy
benchmark therefore means training the configuration again at another seed.

**Images and cells are not replicates.** The 113 test fields and ~3,000 cells are
the *same* in every run. Inside one run the evaluator summarises them into one
value per metric (its own within-run bootstrap intervals, over fields, are kept
as metrics in their own right — `*_ci_lower`, `*_ci_upper`). Across runs, only
those per-run values are aggregated, so **n is the number of seeds**, never the
number of images or cells.

## 2. Seeds

The models that already exist, and nothing else — **no model is retrained**:

| arms | trained models (seeds) | n |
|---|---|---|
| A, B, B′ | 42, 1337, 2024 | 3 |
| B″, C, D0, D1, D2, G, W01, W03, W30, KA, KB | 42 | 1 |

This is set in `config/base.yaml`: `evaluation.seed_replication.seeds: [1337, 2024]`
plus `project.seed` (42), with the single-model arms listed in `reduced_arms`
(`reduced_runs: 1`). W10 is arm B (same objective, same seed) and is reported once
as arm B. L (LoRA) is not part of the study.

Arms with n = 1 are reported with their value and `n: 1`; they have no standard
deviation or interval, and the JSON says so rather than inventing one.
Comparisons between arms can only be *resolved* where both arms have n ≥ 2
(A, B and B′).

**To add seeds later**, add them to `seeds`, remove arms from `reduced_arms`, and
run stage 11: it trains only the models that are missing (about 1.2–1.6 h each).

## 3. Run everything — exact commands

On the server, in the project folder. GPUs 2 and 3. Nothing here trains a model.

**Step 1.** Check the plan. Every cell should read `trained` or `evaluated`;
`-` means not planned.

```bash
python scripts/collect_benchmark_results.py plan
```

**Step 2.** Re-evaluate every model (adds provenance; about 1–2 h in total).

```bash
CUDA_VISIBLE_DEVICES=2 ARMS="A B B1 B2 C D0 D1 D2" nohup bash run_v2.sh --stage 11 > eval_gpu2.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 ARMS="G W01 W03 W30 KA KB" nohup bash run_v2.sh --stage 11 > eval_gpu3.out 2>&1 &
```

Finished when `plan` shows every model as `evaluated`.

**Step 3.** Re-run the three deterministic analyses so they carry provenance.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/conventional_baseline.py --config config/base.yaml
python scripts/error_propagation.py --config config/base.yaml --max-shift 5
CUDA_VISIBLE_DEVICES=2 python scripts/calibrate_z.py --config config/base.yaml --steps 41
```

**Step 4.** Hardware benchmark — only when nothing else is using either GPU.

```bash
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 12
```

**Step 5.** Synthetic ground-truth floor, one run per seed (CPU, minutes).

```bash
bash run_v2.sh --stage 13
```

**Step 6.** Collect.

```bash
bash run_v2.sh --stage 14
```

This writes `runs/benchmark_results/` and regenerates `runs/seed_aggregate.json`
and `runs/RESULTS.md`. The end of its log says `all benchmarks complete` or lists
exactly what is missing.

Each invocation logs to `logs/v2_<timestamp>_gpu<N>/`; `SUMMARY.txt` there lists
every step with `OK` or `FAILED`.

### Environment variables for stages 11–14

| variable | effect |
|---|---|
| `ARMS="..."` | only these arms |
| `SEEDS="..."` | only these seeds |
| `REEVALUATE=1` | re-evaluate models already evaluated (11), redo synthetic seeds (13) |
| `REBENCH=1` | re-time models already benchmarked (12) |
| `FORCE_RETRAIN=1` | retrain a model the collector reports as `mismatch` — overwrites it |

## 4. Run one benchmark by hand

Seed 42 lives in the arm's own directory; every other seed needs its own
`experiment_name` so it gets its own directory. The pattern is `v2_<ARM>_seed<N>`.

```bash
# arm A, seed 1337: evaluate and benchmark the existing model
CUDA_VISIBLE_DEVICES=2 python main.py evaluate  --config config/v2/a_baseline.yaml --seed 1337 --set experiment_name=v2_A_seed1337
CUDA_VISIBLE_DEVICES=2 python main.py benchmark --config config/v2/a_baseline.yaml --seed 1337 --set experiment_name=v2_A_seed1337 --modalities off_axis

# only for a NEW seed: train it first, the same way
CUDA_VISIBLE_DEVICES=2 python main.py train     --config config/v2/a_baseline.yaml --seed 777 --set experiment_name=v2_A_seed777

# arm A, seed 42 (the primary run): no experiment_name needed
CUDA_VISIBLE_DEVICES=2 python main.py evaluate  --config config/v2/a_baseline.yaml --seed 42
```

`python scripts/collect_benchmark_results.py status --arm A --seed 1337` prints the
directory and experiment name for any arm and seed, and whether it still needs
training or evaluation.

`--seed N` is shorthand for `--set project.seed=N` and works on every
`main.py` command. `main.py train` refuses to write into a directory that holds a
run trained with a different seed, so a forgotten `experiment_name` cannot
overwrite the primary checkpoint.

## 5. Where results are stored

```
runs/
├── v2_baseline_off_axis/                 arm A, seed 42
├── v2_A_seed1337_off_axis/               arm A, seed 1337  (same layout for every arm and seed)
│   ├── best_model.pt
│   ├── resolved_config.yaml              the seed and settings it was trained with
│   ├── history.json                      every epoch; proves the run finished
│   ├── metrics_test.json                 all evaluator metrics
│   ├── metrics_test.provenance.json      which checkpoint (SHA-256), seed and settings produced them
│   ├── metrics_test_membrane.json        the same, scored against the independent membrane labels
│   ├── metrics_test_membrane.provenance.json
│   ├── per_cell_test.csv, unmatched_test.csv
│   └── hardware_benchmark_cuda.json      stage 12, hardware arms only: one row per runtime × precision
├── v2_baseline_gabor/                    arm G (in-line holograms)
├── synthetic_validation_seeds/seed1337/synthetic_validation_summary.json
├── conventional_off_axis/metrics_test.json + .provenance.json
├── error_propagation_summary.csv + .provenance.json
├── z_calibration.json + .provenance.json
└── benchmark_results/                    written by the collector (§6)
```

Seeded hardware results go into the run directory, not into the
`runs/*_hardware_benchmark_*.json` files at the top level, so they cannot leak
into `RESULTS.md` Table 4.

## 6. The collector and its JSON files

```bash
python scripts/collect_benchmark_results.py                   # collect everything (stage 14 runs this)
python scripts/collect_benchmark_results.py plan              # arm × seed table of what exists
python scripts/collect_benchmark_results.py --arms A B B1     # only some arms
python scripts/collect_benchmark_results.py --allow-partial   # also write benchmarks still missing seeds
```

It writes to `runs/benchmark_results/`, deleting the previous `results_*.json`
first so no stale file survives:

| file | contents |
|---|---|
| `results_arm_<ARM>.json` | accuracy and measurement metrics, one per arm: A, B, B1, B2, C, D0, D1, D2, G, W01, W03, W30, KA, KB |
| `results_arm_<ARM>_membrane.json` | the same checkpoints scored against the independent membrane labels |
| `results_hardware_arm_<ARM>.json` | latency / throughput / memory / parameters: A, B, D0, KA, KB |
| `results_synthetic_validation.json` | the measurement chain's floor, one run per seed |
| `results_conventional_off_axis.json`, `results_conventional_gabor.json` | classical baseline, deterministic |
| `results_error_propagation.json` | boundary displacement → measurement error, deterministic |
| `results_z_calibration.json` | z identifiability scan, deterministic |
| `results_comparisons.json` | every arm against its comparator (and G against A): difference of means, pooled between-seed SD, 2×SD verdict |
| `benchmark_index.json` | list of all files, seeds per arm, what was not written and why, ignored directories |
| `benchmark_summary.csv` | one row per benchmark × metric: n, mean, std, sem, ci95, min, max — the quickest input for a table |

It exits 0 when every planned benchmark is complete and 3 otherwise, printing
each missing or refused run with the reason and the command that fixes it.
Benchmarks that are not complete are **not written** unless you pass
`--allow-partial`, in which case they carry `"complete": false` and list
`seeds_missing`.

## 7. JSON fields

Annotated (the `//` comments are not part of the file):

```jsonc
{
  "benchmark": "arm_A",
  "kind": "evaluation",                     // evaluation | hardware | measurement_chain_floor | deterministic
  "arm": "A", "label": "A  baseline", "config": "config/v2/a_baseline.yaml",
  "experiment_name": "v2_baseline", "modality": "off_axis", "split": "test",
  "labels": "phase-derived (Otsu)",
  "replication": "independent_runs",        // or "deterministic"
  "unit_of_replication": "one training run per seed, evaluated once",
  "n_runs": 5,
  "seeds": [42, 1337, 2024, 123, 456],      // seeds actually collected
  "seeds_planned": [...], "seeds_missing": [],
  "complete": true,
  "warnings": [],
  "runs": [
    {
      "seed": 42,
      "run_dir": "runs/v2_baseline_off_axis",
      "experiment_name": "v2_baseline",
      "checkpoint_sha256": "…",             // the checkpoint these metrics came from
      "checkpoint_epoch": 41,               // best epoch
      "provenance": "verified",             // or "legacy" (only with --allow-legacy)
      "evaluated_at": 1790000000.0,
      "training": {"epochs_completed": 60, "wall_hours": 1.4},
      "metrics": {"dry_mass_mape": 0.1767, "seg_dice": 0.8328, "...": "..."},
      "non_numeric": {}                     // any string/boolean fields of the source file
    }
  ],
  "statistics": {
    "dry_mass_mape": {
      "mean": 0.0, "std": 0.0, "n": 5, "sem": 0.0,
      "ci95": [0.0, 0.0], "min": 0.0, "max": 0.0,
      "n_missing": 0
    }
  },
  "ci_level": 0.95,
  "generated_at": "2026-09-22 10:00:00"
}
```

(Values above are placeholders showing the shape only.)

Optional keys inside a statistics entry: `constant_across_runs: true` when every
run gave the same value (parameter counts, image counts); `seeds_without_metric`
and `seeds_non_finite` when a metric is absent from, or NaN in, some runs — the
value is then excluded from that metric's statistics and **n says so**; nothing
is dropped silently.

Hardware files name their metrics `<runtime>_<precision>.<field>`, e.g.
`onnx_fp16.latency_mean_ms`, and keep the architecture fields (`params_total`,
`gmacs`, `input_size`, …) un-prefixed. Their `non_numeric` block holds the GPU
name and software versions, which must agree across seeds.

Deterministic files have `n_runs: 1`, `seeds: []`, and `std`, `sem` and `ci95`
all `null`: repeating them reproduces the same number.

All files are strict JSON: NaN and infinity are written as `null`.

## 8. Statistics

For each metric, over the n seeds with a finite value x₁…xₙ:

| field | definition |
|---|---|
| `mean` | x̄ = Σxᵢ / n |
| `std` | sample standard deviation, s = √(Σ(xᵢ − x̄)² / (n − 1)) |
| `sem` | s / √n |
| `ci95` | x̄ ± t₀.₉₇₅,ₙ₋₁ · s / √n (Student t) |
| `min`, `max` | extremes over seeds |

**Why Student t and not ±1.96·SEM.** With five runs the normal approximation
understates the interval: t₀.₉₇₅,₄ = 2.776, not 1.96. With three runs it is
4.303. The interval describes uncertainty in the **mean over training runs** of
that configuration, i.e. how well the reported number would reproduce if the
whole training procedure were repeated.

**The within-run intervals are something else.** Keys such as
`dry_mass_mape_ci_lower` are the evaluator's bootstrap over the 113 test fields of
*one* run: uncertainty from which fields happened to be in the test set, with the
model fixed. They are collected as metrics (so their mean over seeds is reported)
but are not the between-run interval.

**Comparisons** (`results_comparisons.json`) use the project's existing rule: a
difference of means is `resolved` only if it exceeds `resolve_factor` (2) × the
pooled between-seed SD, √(((n₁−1)s₁² + (n₂−1)s₂²)/(n₁+n₂−2)), computed by the same
function `aggregate_seeds.py` and `collect_results.py` use. This is a resolution
criterion, not a significance test, and no p-values are produced.

`collect_results.py` (→ `RESULTS.md`) now computes its between-seed spread from
**all** planned seeds including seed 42 — it previously left the primary run out —
and, once both arms of a comparison are replicated, reports the difference of
means rather than of the seed-42 values. The 2×SD thresholds in `RESULTS.md`
Table 2 will therefore change when it is regenerated.

## 9. Metrics collected

Every numeric field the project's own code writes is collected under the name it
already has. Nothing is recomputed or renamed.

**Per arm (from `main.py evaluate`, 111 fields; the amplitude rows only for D0, D1, D2):**

| family | keys |
|---|---|
| reconstruction | `phase_mae_rad`, `phase_rmse_rad`, `phase_bias_rad`, `phase_psnr_db`, `phase_ssim`, `phase_pearson_r`, `phase_mae_rad_in_cell`, `phase_bias_rad_in_cell`, `phase_mae_rad_background`, `phase_n_images` |
| segmentation | `seg_dice`, `seg_dice_macro`, `seg_iou`, `seg_aji`, `seg_boundary_f1`, `seg_instance_count_mape`, `seg_n_images` |
| detection | `detection_precision`, `detection_recall`, `detection_f1`, `coverage`, `cells_detected`, `cells_reference`, `cells_matched`, `cells_missed`, `cells_false_positive`, `cell_count_ratio`, `missed_median_area_um2`, `false_positive_median_area_um2` |
| per-cell measurement, for each of `area`, `circularity`, `optical_volume`, `dry_mass` | `_mape`, `_mae`, `_mape_ci_lower/upper`, `_mape_coverage_adjusted`, `_cell_pearson_r` (+ `_ci_lower/upper`), `_image_pearson_r`, `_relative_bias`, `_median_relative_bias`, `_loa_lower`, `_loa_upper`, `_n_cells`, `_bootstrap_fields`, `_bootstrap_resamples`; and for area, optical volume and dry mass `_field_total_bias`, `_field_total_mape`, `_field_total_pearson_r`, `_n_fields` |
| forward model | `forward_residual`, `forward_residual_reference`, `forward_residual_ratio`, `forward_residual_n`, `forward_distance_um` |
| amplitude (D0, D1, D2) | `amplitude_mae`, `amplitude_unity_mae`, `amplitude_mae_over_unity`, `amplitude_mae_in_cell`, `amplitude_bias`, `amplitude_pearson_r`, `amplitude_pred_mean`, `amplitude_pred_sd`, `amplitude_reference_mean` |

The integrated phase of a cell is its **optical volume**, V = Σφ·dx·dy; the
`optical_volume_*` keys are the integrated-phase errors. Dry mass is V times the
fixed scalar λ/(2πα), so its relative errors equal the optical-volume ones.

**Per hardware arm (from `main.py benchmark`),** for each of pytorch/onnx × fp32/fp16:
`latency_mean_ms`, `latency_p50_ms`, `latency_p99_ms`, `latency_std_ms`,
`latency_p99_over_p50`, `fps`, `timed_runs`, `foreign_gpu_memory_mb`, and for
PyTorch on CUDA `weights_mb`, `peak_activation_mb`, `peak_allocated_mb`,
`process_peak_allocated_mb`, `process_peak_reserved_mb`; plus `params_total`,
`params_trainable`, `params_frozen`, `params_trainable_fraction`, `gmacs`,
`input_size`.

**Synthetic floor:** `cells_placed`, `cells_measured`, `cells_paired`, signed and
absolute relative errors of area and mass (mean, median, 95th percentile), and the
field-total mass error — the numbers `synthetic_validation.py` prints.

**Deterministic analyses:** every numeric field of the classical baseline's
metrics file (same 111-field schema), every column of
`error_propagation_summary.csv` per pixel shift, and the scalar fields of
`z_calibration.json` per modality.

## 10. Latest-run handling: what the collector refuses

Runs in this project are not timestamped. Each (arm, seed) has exactly one
directory, fixed by its experiment name, and a rerun overwrites it — so the file
in that directory is the latest by construction. The risk is not an older
*directory* but an older *file* inside a current one, or a directory that is not
what its name says. The collector checks for each run, and reports the reason if
any check fails:

1. `resolved_config.yaml` records the expected seed and experiment name.
2. Its model, loss, training, data, optics and label settings equal the arm's
   config today — a QUICK smoke test, an older objective or older constants fail.
3. `history.json` holds every epoch — a killed run fails.
4. The metrics file has a provenance file whose checkpoint SHA-256 equals the
   `best_model.pt` on disk, whose training seed matches, and whose measurement
   settings and split-file hash equal today's — a metrics file left over from
   another checkpoint, another seed or other settings fails.
5. All runs of one benchmark scored the same number of images and reference
   cells.
6. Hardware runs share GPU model and PyTorch version, have a stable latency
   distribution (p99/p50 ≤ 1.25) and ran ONNX on the GPU.
7. Deterministic analyses carry provenance matching today's settings and split.
8. Directories for seeds or arms outside the plan (e.g. an old seed-7 smoke test)
   are listed under `ignored_directories` and never pooled.

`--allow-legacy` accepts a metrics file without provenance if it is newer than its
checkpoint, marking it `"provenance": "legacy"`; `--allow-unstable` accepts an
unstable timing. Neither should be needed after step 2 of §3, which re-evaluates
every existing run and writes its provenance.

## 11. Reproducibility and determinism

`seed_everything(seed)` (`holoqpi/utils.py`) seeds Python `random`, NumPy's global
generator, the PyTorch CPU and CUDA generators, and sets `PYTHONHASHSEED`. Through
them the seed controls:

| source of randomness | how it is seeded |
|---|---|
| decoder initialisation | torch generator, when the model is built |
| batch order | DataLoader shuffle, from the torch generator |
| DataLoader workers | `_seed_worker` gives each worker a stream derived from torch's per-worker seed |
| random crops, flips, 90° rotations | `random.Random(project.seed + n_samples)` in the dataset and the augmentation |
| synthetic input for timing, MAC count | torch / NumPy generators (`main.py benchmark` now seeds them) |
| synthetic cells | `np.random.default_rng(project.seed)` |
| within-run bootstrap over fields | `project.seed` |

**Fixed for every seed:** the dataset, the train/val/test split
(`data/splits.json`, written once by `main.py prepare`; stages 11–14 never rerun
it, and every provenance file records its SHA-256 so a regenerated split would be
refused), the labels, the ImageNet encoder weights, every setting in the config.

**Determinism setting.** `project.deterministic: false` for every reported run,
as in the original study. cuDNN then chooses the fastest convolution algorithm
(`cudnn.benchmark = True`), so rerunning the *same* seed reproduces a result
closely but not bit-for-bit. The seeds are replicates of the training procedure,
and this is the variation they are meant to capture. Setting
`project.deterministic: true` additionally enables
`torch.use_deterministic_algorithms` and a fixed cuBLAS workspace; it is slower
and **must not be mixed** with runs made without it — the collector's protocol
check refuses such a mixture. Every provenance file and hardware row records the
settings actually used (`cudnn_benchmark`, `cudnn_deterministic`,
`deterministic_algorithms`, PyTorch and CUDA versions).

## 12. Building a manuscript table from the JSON

```python
import json
from pathlib import Path

results = Path("runs/benchmark_results")
for arm in ["A", "B", "B1"]:
    s = json.loads((results / f"results_arm_{arm}.json").read_text())["statistics"]
    m = s["dry_mass_mape"]
    print(f"{arm} & ${m['mean']:.4f} \\pm {m['std']:.4f}$ & "
          f"[{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] & {m['n']} \\\\")
```

`benchmark_summary.csv` has the same numbers as one flat table for a
spreadsheet.
