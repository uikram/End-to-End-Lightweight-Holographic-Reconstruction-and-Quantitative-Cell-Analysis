# results_for_manuscript

The paper's result package: every reported number, with its source. `RESULTS_INDEX.md` maps each
table and figure of the paper to a file in this folder, its source under `runs/`, the source key,
the number of training runs, the seeds and the field set.

Built by `python analysis/compile_results.py` from `runs/` (including `runs/provenance/`),
`config/` and `data/`. The compiler runs 526 consistency checks and writes them to
`metadata/consistency_checks.json`. Every numeric entry carries
`value, units, source_file, source_key, field_set, n_runs, seeds, derivation (direct|derived), derived_from, calculation`.

**Numbering.** JSON keys and figure file names use the original numbering (`table_4`, `figure_7`, ...).
`RESULTS_INDEX.md` gives the label each one has in the final paper (main text: Tables 1-5,
Figs. 1-7; supplementary: Tables S1-S10, Figs. S1-S2).

**Training runs** (n = trained models): End-to-End Neural Baseline, +IPP (per-cell) and +IPP (image)
have 3 (seeds 42, 1337, 2024). Every other neural configuration, including the In-Line Neural
Configuration, has 1 (seed 42). The classical pipelines are deterministic (one run). Fields and
cells are given as N, never as n.

| File | Contents | Paper item | Main sources | Runs / field set | Direct or derived |
|---|---|---|---|---|---|
| `summary/manuscript_summary.json` | index, design, consistency-check result | - | all | - | - |
| `metadata/configurations.json` | training runs, seeds, comparator, loss weights, architecture, geometry, checkpoint SHA-256 | Table S2 | `runs/*/resolved_config.yaml`, provenance, collector JSON | see file | direct |
| `metadata/splits.json` | split sizes | Table S1 | `data/splits.json`, `runs/provenance/train_G.log` | - | direct |
| `metadata/consistency_checks.json` | the compiler's consistency checks | - | - | - | - |
| `baseline/baseline.json` | baseline mean ± SD; baseline vs classical; measurement agreement | Tables 2, 3 | `runs/v2_{baseline,A_seed*}/metrics_test.json`, `runs/conventional_off_axis` | 3 runs; 113 fields; 3186 cells | mean/SD derived; matched-count difference descriptive |
| `classical/classical.json` | classical off-axis and in-line metrics (113 and 107 fields) | Tables 2, S3 | `runs/conventional_*`, `runs/common_fields/conventional_*` | deterministic | direct |
| `inline/inline.json` | 107 common fields, in-line split 521/122/107, exclusion statement | Table S3 | `metrics_test_common.json`, `runs/common_fields/`, `runs/provenance/` | in-line 1 run; off-axis 3 runs; 107 fields; 3055 cells | direct / mean-SD derived |
| `ipp/ipp.json` | ablation; replicated comparisons (delta, pooled SD, 2× pooled SD, runs, evaluability, assessment); gradient-path check | Tables 5, S5, S6 | `runs/*/metrics_test.json`, `runs/gradient_path_b_cell_ipp_512.csv` | per row | derived (resolution); gradient direct |
| `amplitude/amplitude.json` | amplitude output, forward residual, phase-scale probes with field counts, configured tolerance | Table S7 | `metrics_test.json`, `z_calibration.json`, `amplitude_sensitivity_off_axis_*.csv` | 1 run each (A, B: 3) | direct / derived counts |
| `forward_model/forward_model.json` | recording distance, z scans, learned z | Table S7, Fig. 6 | `z_calibration.json`, `RESULTS.md`, resolved configs | 1 run | direct |
| `decomposition/decomposition.json` | domain/phase/total factors (geometric mean, median \|log\|), recall, comparisons | Table S4 | `runs/diagnostics/decomposition_*.csv` (recomputed) | A, B, B1: 3 runs; classical 1 | derived |
| `boundary_sensitivity/boundary.json` | edge/interior recall and MAPE, false positives; boundary displacement and synthetic floor | Tables 4, S8, Fig. 7 | decomposition CSVs, `unmatched_test.csv`, `error_propagation_summary.csv`, synthetic JSON | A, B, B1: 3 runs; classical 1 | derived |
| `benchmarking/benchmarking.json` | latency mean/p50/p99, p99/p50, FPS, parameters, GMACs, memory, protocol | Tables S9, S10, Fig. S2 | `runs/benchmark_results/results_hardware_arm_*.json`, `config/base.yaml` | one benchmark session per training seed | direct (collector means) |
| `metadata/gradient_registration_crop.json` | gradient ratio (median 0.373, 0.243-0.655), registration statistics (recomputed from `registration/hologram_registration.csv`), crop offset | Methods | see file | - | direct |
| `registration/` | `registration_summary.json`, `hologram_registration.csv`, `classical_phase_shift.csv`, `hologram_inventory.csv` (copies of `runs/diagnostics/`, written by `scripts/register_holograms.py`) | Methods (in-line pairing) | `runs/diagnostics/` | 800 fields | direct |
| `figures/` | `data/` (sources copied from `runs/`), `scripts/`, `regenerated/` (script output), `figure_index.json` | Figs. 1-7, S1, S2 | `runs/` | - | direct |

**Figures.** `python results_for_manuscript/figures/scripts/make_all.py` regenerates every figure
that is drawn from stored results (Figs. 2, 4, 5, 6, 7, S1, S2) into `figures/regenerated/`.
Fig. 3 (`scripts/fig3_qualitative.py`) and the image panels of Fig. 1 (`scripts/fig1_panels.py`)
run the trained network, so they need the checkpoints and the image data. Neither is in the
repository. A different matplotlib or font version can shift the layout slightly; the plotted
data stay the same.

**Assessment wording** (replicated comparisons): "Exceeds 2× pooled SD", "Within 2× pooled SD",
or "Not estimable from the available runs" (n = 1 on either side; no substitute variance).
