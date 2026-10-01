# results_for_manuscript

Built by `python analysis/v42/compile_results.py` from `runs/`, `logs/`, `config/`, `data/` (never from the manuscript). Verified by `python analysis/v42/check_numbers.py` (`analysis/v42/check_numbers.csv`). Every numeric entry carries `value, units, source_file, source_key, field_set, n_runs, seeds, derivation (direct|derived), derived_from, calculation`.

Training runs (n = trained models): baseline, +IPP (per-cell), +IPP (image) = 3 (seeds 42, 1337, 2024); every other neural configuration, including the In-Line Neural Configuration, = 1 (seed 42); classical pipelines are deterministic (single run). `N`/fields/cells are never called n.

| File | Contents | Table / figure | Main sources | Runs / field set | Direct or derived |
|---|---|---|---|---|---|
| `summary/manuscript_summary.json` | index, design, consistency-check result | — | all | — | — |
| `metadata/configurations.json` | training runs, seeds, comparator, loss weights, architecture, geometry, checkpoints | Table 3 | `runs/*/resolved_config.yaml`, provenance, collector JSON | see file | direct |
| `baseline/baseline.json` | baseline mean ± SD; Table 4 (vs classical); Table 6 | 4, 6 | `runs/v2_{baseline,A_seed*}/metrics_test.json`, `runs/conventional_off_axis` | 3 runs; 113 fields; 3186 cells | mean/SD derived; Δ derived; matched-count difference descriptive |
| `classical/classical.json` | classical off-axis/in-line metrics (113 and 107 fields) | 4, 5 | `runs/conventional_*`, `runs/common_fields/conventional_*` | deterministic | direct |
| `inline/inline.json` | Table 5 (107 common fields), in-line split 521/122/107, exclusion statement | 5 | `metrics_test_common.json`, `common_fields/`, `train_G.log` | in-line 1 run; off-axis 3 runs; 107 fields; 3055 cells | direct / mean-SD derived; neural recovered contrast `N/A` until `scripts/neural_phase_contrast.py` is run |
| `ipp/ipp.json` | Table 9 (ablation), Table 10 (delta, pooled SD, 2× pooled SD, runs, evaluability, assessment), gradient-path check | 9, 10, §3 | `runs/*/metrics_test.json`, `runs/gradient_path_b_cell_ipp_512.csv` | per row | derived (resolution), gradient direct from CSV |
| `amplitude/amplitude.json` | Table 11a–c (amplitude, forward residual, phase-scale probes with field counts), configured tolerance | 11 | `metrics_test.json`, `z_calibration.json`, `amplitude_sensitivity_off_axis_*.csv` | 1 run each (A, B: 3) | direct / derived counts |
| `forward_model/forward_model.json` | recording distance, z scans, learned z | 11, 7, §5.5 | `z_calibration.json`, `RESULTS.md`, resolved configs | 1 run | direct |
| `decomposition/decomposition.json` | Table 7: per-run and per-config domain/phase/total GM and median \|log\|, recall, comparisons | 7 | `runs/diagnostics/decomposition_*.csv` (recomputed) | A, B, B1: 3 runs; classical 1 | derived |
| `boundary_sensitivity/boundary.json` | Table 8 (edge/interior, false positives); Table 12 boundary displacement and synthetic floor | 8, 12, Fig. 8 | decomposition CSVs, `unmatched_test.csv`, `error_propagation_summary.csv`, synthetic JSON | A, B, B1: 3 runs; classical 1 | derived |
| `benchmarking/benchmarking.json` | latency mean/p50/p99, p99/p50, fps, params, GMACs, memory, protocol | 13, Fig. 9 | `runs/benchmark_results/results_hardware_arm_*.json`, `config/base.yaml` | benchmark sessions per training seed | direct (collector means) |
| `metadata/gradient_registration_crop.json` | gradient ratio (0.373, 0.243–0.655), registration values (**unverified**, archive absent), crop offset | §3, §4.1 | see file | — | direct / author-supplied |
| `metadata/splits.json`, `metadata/consistency_checks.json` | split sizes, 510 consistency checks | — | — | — | — |
| `figures/` | figure source data (`data/`), figure scripts 7 and 8, `figure_index.json` | Figs. 2, 4–9 | `runs/` | — | direct |

Assessment wording: "Exceeds 2× pooled SD", "Within 2× pooled SD", "Not estimable from the available runs" (n = 1 on either side; no substitute variance).
