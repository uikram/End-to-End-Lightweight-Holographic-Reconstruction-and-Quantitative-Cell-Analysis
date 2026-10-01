| Quantity | v4.1 | 4.2 | Source (file; key in analysis/v42/values.json) |
|---|---|---|---|
| Off-axis baseline dry-mass MAPE (3 seeds) | 0.1763 ± 0.0024 | 0.1742 ± 0.0045 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_mape` |
| Seed-42 bootstrap CI, dry-mass MAPE | 0.1666–0.1867 | 0.1650–0.1849 | `runs/v2_baseline_off_axis/metrics_test.json`; `run.A.s42.dry_mass_mape_ci_lower` |
| Baseline per-cell dry-mass r | 0.9359 | 0.9363 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_cell_pearson_r` |
| Baseline area MAPE | 0.1546 | 0.1568 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_mape` |
| Baseline area r | 0.8979 | 0.8986 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_cell_pearson_r` |
| Baseline circularity MAPE | 0.0637 | 0.0571 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.circularity_mape` |
| Baseline circularity r | 0.8713 | 0.8861 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.circularity_cell_pearson_r` |
| Baseline coverage-adjusted dry-mass MAPE | 0.5124 | 0.5102 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_mape_coverage_adjusted` |
| Baseline recall | 0.5920 ± 0.0011 | 0.5931 ± 0.0027 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.detection_recall` |
| Baseline precision | 0.8550 | 0.8488 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.detection_precision` |
| Baseline matched cells | 1886 ± 4 | 1890 ± 9 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.cells_matched` |
| Baseline phase MAE [rad] | 0.1593 | 0.1568 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_mae_rad` |
| Baseline in-cell phase MAE [rad] | 0.2756 | 0.2689 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_mae_rad_in_cell` |
| Baseline phase r | 0.8701 | 0.8741 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_pearson_r` |
| Baseline SSIM | 0.6275 | 0.6386 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_ssim` |
| Baseline Dice | 0.8312 | 0.8337 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_dice` |
| Baseline AJI | 0.6696 | 0.6710 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_aji` |
| Baseline boundary F1 | 0.4574 | 0.4687 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_boundary_f1` |
| Baseline BA bias dry mass | -0.0434 | -0.0372 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_relative_bias` |
| Baseline BA LoA dry mass | [-0.252, +0.165] | [-0.247, +0.172] | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_loa_lower` |
| Baseline BA bias area | +0.0455 | 0.0649 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_relative_bias` |
| Baseline BA LoA area | [-0.172, +0.263] | [-0.162, +0.292] | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_loa_lower` |
| Baseline field-total bias, dry mass | -0.1632 | -0.1560 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_bias` |
| Baseline field-total bias, area | -0.1078 | -0.0901 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_field_total_bias` |
| Baseline field-total r, dry mass | 0.9879 | 0.9883 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_pearson_r` |
| Baseline field-total dry-mass MAPE (new row) | -- | 0.1631 ± 0.0141 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_mape` |
| Classical off-axis dry-mass MAPE | 0.2245 | 0.2203 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_mape` |
| Classical off-axis area MAPE | 0.1567 | 0.1578 | `runs/conventional_baseline_test.json`; `classical113.off_axis.area_mape` |
| Classical off-axis phase MAE | 0.2114 | 0.1968 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_mae_rad` |
| Classical off-axis in-cell phase MAE | 0.3655 | 0.3112 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_mae_rad_in_cell` |
| Classical off-axis phase r | 0.7580 | 0.7851 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_pearson_r` |
| Classical off-axis SSIM | 0.5022 | 0.5299 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_ssim` |
| Classical off-axis Dice | 0.7885 | 0.8253 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_dice` |
| Classical off-axis AJI | 0.5931 | 0.6387 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_aji` |
| Classical off-axis boundary F1 | 0.2463 | 0.3954 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_boundary_f1` |
| Classical off-axis recall | 0.5763 | 0.5869 | `runs/conventional_baseline_test.json`; `classical113.off_axis.detection_recall` |
| Classical off-axis precision | 0.7773 | 0.7910 | `runs/conventional_baseline_test.json`; `classical113.off_axis.detection_precision` |
| Classical off-axis cell r | 0.8904 | 0.8937 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_cell_pearson_r` |
| Classical off-axis matched cells | 1836 | 1870 | `runs/conventional_baseline_test.json`; `classical113.off_axis.cells_matched` |
| Classical off-axis field-total dry-mass MAPE (new) | -- | 0.1094 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_field_total_mape` |
| Classical off-axis in-cell phase contrast | +0.902 (113) | 0.922 | `runs/common_fields/conventional_baseline_test.json`; `classical107.off_axis.recovered_phase_contrast_rad` |
| Classical in-line phase r | -0.1359 (113 fields) | -0.2075 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.phase_pearson_r` |
| Classical in-line in-cell contrast | -0.067 | -0.084 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.recovered_phase_contrast_rad` |
| Classical in-line Dice | 0.1627 | 0.1583 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.seg_dice` |
| Classical in-line matched cells | 116 of 3186 | 106 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.cells_matched` |
| In-line neural phase r | 0.8098 (113 fields) | 0.8667 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.phase_pearson_r` |
| In-line neural Dice | 0.7758 | 0.8148 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.seg_dice` |
| In-line neural recall | 0.5508 | 0.5885 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.detection_recall` |
| In-line neural dry-mass MAPE | 0.2308 | 0.2129 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.dry_mass_mape` |
| In-line neural matched cells | 1755 of 3186 | 1798 | `runs/v2_baseline_gabor/metrics_test_common.json`; `common_run.G.s42.cells_matched` |
| +IPP (per-cell) dry-mass MAPE | 0.1915 | 0.1883 | `runs/v2_cell_ipp_off_axis/metrics_test.json (+ seeds)`; `mean.B.dry_mass_mape` |
| +IPP (image) dry-mass MAPE | 0.1935 | 0.1887 | `runs/v2_image_volume_off_axis/metrics_test.json (+ seeds)`; `mean.B1.dry_mass_mape` |
| Δ +IPP (per-cell) vs baseline | +0.0152 (2×SD 0.0070) | +0.0141 (2×SD 0.0097) | `runs/*/metrics_test.json`; `cmp.B_vs_A.dry_mass_mape.diff` |
| Δ +IPP (image) vs baseline | +0.0172 (2×SD 0.0052) | +0.0145 (2×SD 0.0066) | `runs/*/metrics_test.json`; `cmp.B1_vs_A.dry_mass_mape.diff` |
| Δ +IPP (image) vs +IPP (per-cell) | 0.0020 (2×SD 0.0074) | +0.0004 (2×SD 0.0076) | `runs/*/metrics_test.json`; `cmp.B1_vs_B.dry_mass_mape.diff` |
| +IPP (per-cell) precision | 0.7995 | 0.8077 | `runs/v2_cell_ipp_off_axis/metrics_test.json (+ seeds)`; `mean.B.detection_precision` |
| +Area dry-mass MAPE | 0.1917 | 0.1850 | `runs/v2_cell_ipp_area_off_axis/metrics_test.json`; `mean.B2.dry_mass_mape` |
| +BGA dry-mass MAPE | 0.1978 | 0.1921 | `runs/v2_cell_ipp_bga_off_axis/metrics_test.json`; `mean.C.dry_mass_mape` |
| w=0.1 dry-mass MAPE | 0.1853 | 0.1754 | `runs/v2_ipp_w01_off_axis/metrics_test.json`; `mean.W01.dry_mass_mape` |
| w=0.3 dry-mass MAPE | 0.1828 | 0.1881 | `runs/v2_ipp_w03_off_axis/metrics_test.json`; `mean.W03.dry_mass_mape` |
| w=3.0 dry-mass MAPE | 0.2102 | 0.1963 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.dry_mass_mape` |
| w=3.0 Dice | 0.8101 | 0.8164 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.seg_dice` |
| w=3.0 precision | 0.7252 | 0.7855 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.detection_precision` |
| w=3.0 phase MAE | 0.1851 | 0.1813 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.phase_mae_rad` |
| +Amplitude dry-mass MAPE | 0.2015 | 0.1857 | `runs/v2_amplitude_off_axis/metrics_test.json`; `mean.D0.dry_mass_mape` |
| Δ +Amplitude vs +IPP (per-cell) | +0.0100 | -0.0026 | `runs/*/metrics_test.json`; `cmp.D0_vs_B.dry_mass_mape.diff` |
| Compact Baseline dry-mass MAPE | 0.1783 | 0.1740 | `runs/v2_compact_baseline_off_axis/metrics_test.json`; `mean.KA.dry_mass_mape` |
| Δ Compact vs baseline | +0.0019 | -0.0002 | `runs/*/metrics_test.json`; `cmp.KA_vs_A.dry_mass_mape.diff` |
| Baseline 2×SD dry-mass MAPE | 0.0047 | 0.0089 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `sd.A.dry_mass_mape` |
| Edge recall (baseline) | 0.208 | 0.212 | `runs/diagnostics/decomposition_by_config.csv`; `decomp.A.cell.edge.recall_mean` |
| Interior recall (baseline) | 0.914 | 0.912 | `runs/diagnostics/decomposition_by_config.csv`; `decomp.A.cell.interior.recall_mean` |
| Edge dry-mass MAPE (baseline) | 0.298 | 0.305 | `derived`; `locmean.A.edge.mape` |
| Interior dry-mass MAPE (baseline) | 0.153 | 0.149 | `derived`; `locmean.A.interior.mape` |
| False positives per run (baseline) | 320 | 337 | `derived`; `locmean.A.fp_all` |
| False positives within 15 px (baseline) | 141 | 136 | `derived`; `locmean.A.fp_near` |
| False positives per run (+IPP per-cell) | 477 | 456 | `derived`; `locmean.B.fp_all` |
| Amplitude MAE (+Amplitude) | 0.0656 | 0.0651 | `runs/v2_amplitude_off_axis/metrics_test.json`; `amp.D0.amplitude_mae` |
| Amplitude MAE (+Fwd fixed z) | 0.0637 | 0.0665 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_mae` |
| Amplitude r (+Fwd fixed z) | 0.9034 | 0.9064 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_pearson_r` |
| Amplitude ratio to A=1 (+Fwd fixed z) | 0.4337 | 0.4526 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_mae_over_unity` |
| Reference amplitude mean | 0.9977 | 0.9985 | `runs/v2_amplitude_off_axis/metrics_test.json`; `amp.D0.amplitude_reference_mean` |
| Off-axis reference residual (test) | 0.923 | 0.8914 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.forward_residual_reference` |
| Baseline residual ratio | 0.9969 | 1.0039 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.forward_residual_ratio` |
| +Fwd (fixed z) residual ratio | 0.9915 | 1.0068 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `mean.D1.forward_residual_ratio` |
| +Fwd (free z) residual ratio | 0.9928 | 1.0068 | `runs/v2_learned_z_off_axis/metrics_test.json`; `mean.D2.forward_residual_ratio` |
| +Fwd (fixed z) in-cell phase MAE | 0.3011 | 0.2893 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `mean.D1.phase_mae_rad_in_cell` |
| +Fwd (free z) in-cell phase MAE | 0.2903 | 0.2886 | `runs/v2_learned_z_off_axis/metrics_test.json`; `mean.D2.phase_mae_rad_in_cell` |
| 0.9x probe, 32 validation fields, off-axis | -5e-5 (22 of 32 lower) | 0.00035 | `runs/z_calibration.json`; `disc.off_axis.margin.scaled_0.9` |
| Best phase scale, off-axis (median) | 0.68 (IQR 0.80) | 0.975 | `runs/z_calibration.json`; `scale.off_axis.best_scale_median` |
| 0.9x probe, test, global surface | -8e-5 | 0.00027 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_global.log`; `probe.global.phase_scaled_0.9.change` |
| 0.9x probe, test, per-field surfaces | -6e-4 | 0.0011 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_scaled_0.9.change` |
| Per-field-surface residual | 0.43 | 0.391 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_reference` |
| 0.5x probe, test, global | +0.003 | 0.0070 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_global.log`; `probe.global.phase_scaled_0.5.change` |
| 0.5x probe, test, per-field | +0.025 | 0.0359 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_scaled_0.5.change` |
| Learned z after epoch 1 [um] | 34.12 | 34.10 | `runs/RESULTS.md`; `learnedz.initial_after_epoch1` |
| Learned z final [um] | 33.39 | 33.41 | `runs/RESULTS.md`; `learnedz.final` |
| Learned z excursion [um] | 0.958 | 0.884 | `runs/RESULTS.md`; `learnedz.excursion` |
| Learned z last-10 spread [um] | 0.026 | 0.018 | `runs/RESULTS.md`; `learnedz.last10_spread` |
| Learned z at selected checkpoint [um] | 33.35 | 33.41 | `runs/v2_learned_z_off_axis/metrics_test.json`; `learnedz.scored` |
| z scan IQR off-axis [um] | 156.4 | 48.1 | `runs/z_calibration.json`; `z.off_axis.per_image_iqr_um` |
| z scan in-line median [um] | 50.5 | 33.68 | `runs/z_calibration.json`; `z.gabor.per_image_median_z_um` |
| z scan IQR in-line [um] | 6.0 | 0.0 | `runs/z_calibration.json`; `z.gabor.per_image_iqr_um` |
| Gradient ratio median | 0.302 | 0.373 | `runs/gradient_path_b_cell_ipp_512.csv`; `grad.median` |
| Gradient ratio range | 0.18-0.67 | 0.243–0.655 | `runs/gradient_path_b_cell_ipp_512.csv`; `grad.min` |
| Baseline latency PyTorch FP32 [ms] | 12.32 ± 0.11 | 12.31 ± 0.12 | `runs/benchmark_results/results_hardware_arm_A.json`; `hw.A.mean.pytorch_fp32.latency_mean_ms` |
| Baseline latency ONNX FP16 [ms] | 8.42 ± 0.06 | 8.42 ± 0.07 | `runs/benchmark_results/results_hardware_arm_A.json`; `hw.A.mean.onnx_fp16.latency_mean_ms` |
| Compact latency ONNX FP16 [ms] | 7.98 | 7.97 | `runs/benchmark_results/results_hardware_arm_KA.json`; `hw.KA.mean.onnx_fp16.latency_mean_ms` |
| Membrane Dice (baseline) | 0.496 | 0.502 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_dice` |
| Membrane AJI | 0.235 | 0.236 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_aji` |
| Membrane boundary F1 | 0.062 | 0.064 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_boundary_f1` |
| Membrane recall | 0.107 | 0.114 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.detection_recall` |
