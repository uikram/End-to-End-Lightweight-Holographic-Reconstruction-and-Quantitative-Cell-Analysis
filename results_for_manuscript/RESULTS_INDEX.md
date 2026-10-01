# RESULTS_INDEX — manuscript table/figure → package file → source → key → runs → seeds → field set

Built from `analysis/v42/compile_results.py`. "Runs" = training runs. Source paths are relative to the repository root;
`runs/<A>` = `runs/v2_baseline_off_axis`, `runs/v2_A_seed1337_off_axis`, `runs/v2_A_seed2024_off_axis` (baseline);
`<B>` = `v2_cell_ipp_off_axis`, `v2_B_seed1337_off_axis`, `v2_B_seed2024_off_axis`;
`<B1>` = `v2_image_volume_off_axis`, `v2_B1_seed1337_off_axis`, `v2_B1_seed2024_off_axis`.

| Manuscript item | Package file (path inside) | Source file(s) | Source key | Runs | Seeds | Field set |
|---|---|---|---|---|---|---|
| Table 1 (dataset composition) | – (data composition) | `data/manifest.csv`, `data/splits.json` | – | – | – | 800 fields |
| Table 2 / Sec. 4.1 splits | `metadata/splits.json`, `inline/inline.json::split` | `data/splits.json`; `logs/v2_20260930_074055_gpu3/train_G.log`, `logs/corrected_20260930_054214/common_G.log` | split lines | – | – | 560/127/113; 521/122/107 |
| Table 3 | `metadata/configurations.json` | `runs/*/resolved_config.yaml`, `*/metrics_test.provenance.json`, `runs/benchmark_results/results_arm_*.json` | `seed`, `loss.weights`, `model`, `optics`, `checkpoint_sha256` | 3 or 1 | 42 (1337, 2024) | – |
| Table 4 | `baseline/baseline.json::table_4`; `classical/classical.json` | `<A>/metrics_test.json`; `runs/conventional_off_axis/metrics_test.json` | `phase_mae_rad`, `phase_mae_rad_in_cell`, `phase_pearson_r`, `phase_ssim`, `seg_dice`, `seg_aji`, `seg_boundary_f1`, `detection_recall`, `detection_precision`, `area_mape`, `dry_mass_mape`, `dry_mass_cell_pearson_r`, `dry_mass_field_total_mape`, `cells_matched` | neural 3; classical 1 (deterministic) | 42, 1337, 2024 | 113 off-axis test fields; 3186 cells |
| Table 5 | `inline/inline.json::table_5` | `<A>/metrics_test_common.json`; `runs/v2_baseline_gabor/metrics_test_common.json`; `runs/common_fields/conventional_{off_axis,gabor}/metrics_test.json`; `runs/common_fields/neural_phase_contrast.json` | as Table 4; `recovered_phase_contrast_rad`; `*.median_contrast_rad` | in-line 1; off-axis 3; classical 1 | 42; 42, 1337, 2024 | 107 common test fields; 3055 cells |
| Table 6 | `baseline/baseline.json::table_6` | `<A>/metrics_test.json` | `{area,circularity,dry_mass}_{mape,mape_ci_lower,mape_ci_upper,mape_coverage_adjusted,cell_pearson_r,relative_bias,loa_lower,loa_upper,field_total_*}` (circularity field total: N/A) | 3 | 42, 1337, 2024 | 113 fields |
| Table 7 | `decomposition/decomposition.json` | `runs/diagnostics/decomposition_{cells,fields,summary,by_config}.csv` | `domain`, `phase`, `total`, `status`, `edge` | A, B, B1: 3; classical 1 | 42, 1337, 2024 | 113 fields |
| Table 8 | `boundary_sensitivity/boundary.json::table_8` | `runs/diagnostics/decomposition_*.csv`; `<A>,<B>,<B1>/unmatched_test.csv`; `runs/conventional_off_axis/unmatched_test.csv` | edge recall/MAPE; false positives (all, ≤ 15 px) | A, B, B1: 3; classical 1 | 42, 1337, 2024 | 113 fields; 1452 edge / 1734 interior cells |
| Table 9 | `ipp/ipp.json::table_9` | `runs/v2_*/metrics_test.json` (all configurations) | `dry_mass_mape`, `dry_mass_mape_coverage_adjusted`, `dry_mass_field_total_mape`, `area_mape`, `detection_recall`, `detection_precision`, `seg_dice`, `phase_mae_rad` | 3 or 1 | see Table 3 | 113 fields |
| Table 10 | `ipp/ipp.json::table_10` | as Table 9; cross-check `runs/benchmark_results/results_comparisons.json` | delta, pooled SD, 2× pooled SD, runs per side, evaluability, assessment | 3 or 1 | see Table 3 | 113 fields |
| Table 11a–b | `amplitude/amplitude.json::table_11a, table_11b` | `runs/v2_{amplitude,forward_amplitude,learned_z}_off_axis/metrics_test.json`; `<A>`, `<B>` | `amplitude_*`, `forward_residual*` | 1 (A, B: 3) | 42 (A, B: 42, 1337, 2024) | 113 fields |
| Table 11c | `amplitude/amplitude.json::table_11c` | `runs/z_calibration.json`; `runs/amplitude_sensitivity_off_axis_{global,per_field}.csv` | `discrimination.margins`, `win_rates` × `images`; per-field residual counts | – | – | 32 validation fields; 113 / 112 test fields |
| Sec. 5.5, Fig. 7 | `forward_model/forward_model.json` | `runs/z_calibration.json`; `runs/RESULTS.md`; `runs/v2_learned_z_off_axis/metrics_test.json`; `config/base.yaml` | `curve`, `per_image_best_z_um`, `forward_distance_um` | 1 | 42 | 4 scan fields per geometry; 32 validation fields |
| Sec. 3.5.2 gradient ratio | `ipp/ipp.json::gradient_path` | `runs/gradient_path_b_cell_ipp_512.csv` | `ratio_at_weight_1` (median 0.373; 0.243–0.655) | – | 42 | 30 training batches |
| Sec. 4.1 registration | `metadata/gradient_registration_crop.json::registration` | `registration/` (copies of `runs/diagnostics/`: `registration_summary.json`, `hologram_registration.csv`, `classical_phase_shift.csv`, `hologram_inventory.csv`; regenerated with `scripts/register_holograms.py`; `verified: true`, 15 claims recomputed) | – | – | – | 800 fields |
| Sec. 4.1 crop offset | `metadata/gradient_registration_crop.json::crop` | `runs/diagnostics/crop_offset_test_dy-6_dx-2.json`; `config/base.yaml` | `data.crop_offset_px` | – | – | 127 validation / 113 test fields |
| Sec. 4.2 membrane check | `metadata/gradient_registration_crop.json::membrane_registration` | `config/base.yaml` (`membrane.*`) | `scale`, `offset_y`, `offset_x` | – | – | 13 fields tested |
| Table 12 / Fig. 8 | `boundary_sensitivity/boundary.json::table_12_*`, `median_mass_over_area_ratio` | `runs/error_propagation_summary.csv`; `runs/benchmark_results/results_synthetic_validation.json` | `mass_relative_error`, `area_relative_error`, synthetic metrics | – | 3 synthetic sets | 113 fields; 3 sets × 12 fields |
| Table 13 / Fig. 9 | `benchmarking/benchmarking.json` | `runs/benchmark_results/results_hardware_arm_{A,B,D0,KA,KB}.json`; `config/base.yaml` (`deploy.benchmark`) | `statistics.<runtime>.<metric>.mean` | 3 sessions (baseline, +IPP) or 1 | 42, 1337, 2024 | 900×900, batch 1 |
| Figs. 2, 4–9 | `figures/data/`, `figures/scripts/`, `figures/regenerated/` | copies of the files above | – | – | – | see `analysis/v42/FIGURE_STATUS.md` |
| Figs. 1, 3 | not regenerated here (checkpoints needed) | `manuscript_images/` | – | – | 42 | NCI_06, NCI_08 |

## Source results retained outside the package (not manuscript-facing)

`runs/` is kept whole as the traceable source. Files in it that no table or figure uses (retained, not deleted):
`runs/v2_*_hardware_benchmark_cuda.{csv,json}` and `runs/v2_*/hardware_benchmark_cuda.*` (earlier benchmark stage; the manuscript uses the collector files),
`runs/*/metrics_test_membrane.json` for arms other than the baseline, `runs/v2_ipp_w10_off_axis` (repeat check, quoted in Table 9's note),
`runs/aberration_fit_order5.csv`, `runs/label_audit_*` (Sec. 4.2 Otsu statistics only), `runs/seed_aggregate.*`, `runs/results_table.csv`, `runs/RESULTS.md` (collector text; its learned-z line is used by Fig. 7).
Large per-field CSVs of the decomposition are also in `results_for_manuscript/figures/data/diagnostics/` as figure sources.
