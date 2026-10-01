"""Write the number table of CHANGELOG.md: v4.1 value -> 4.2 value -> source.

    python analysis/v42/make_changelog_numbers.py > analysis/v42/changelog_numbers.md
Old values are those printed in HoloQPI_Manuscript_V4.1 (main.tex and tables).
"""
import json
from pathlib import Path

V = json.loads((Path(__file__).resolve().parent / "values.json").read_text())
ROWS = [  # quantity, v4.1 printed, key, format
 ("Off-axis baseline dry-mass MAPE (3 seeds)", "0.1763 ± 0.0024", "mean.A.dry_mass_mape", "sd.A.dry_mass_mape", 4),
 ("Seed-42 bootstrap CI, dry-mass MAPE", "0.1666–0.1867", "run.A.s42.dry_mass_mape_ci_lower", "run.A.s42.dry_mass_mape_ci_upper", "ci"),
 ("Baseline per-cell dry-mass r", "0.9359", "mean.A.dry_mass_cell_pearson_r", None, 4),
 ("Baseline area MAPE", "0.1546", "mean.A.area_mape", None, 4),
 ("Baseline area r", "0.8979", "mean.A.area_cell_pearson_r", None, 4),
 ("Baseline circularity MAPE", "0.0637", "mean.A.circularity_mape", None, 4),
 ("Baseline circularity r", "0.8713", "mean.A.circularity_cell_pearson_r", None, 4),
 ("Baseline coverage-adjusted dry-mass MAPE", "0.5124", "mean.A.dry_mass_mape_coverage_adjusted", None, 4),
 ("Baseline recall", "0.5920 ± 0.0011", "mean.A.detection_recall", "sd.A.detection_recall", 4),
 ("Baseline precision", "0.8550", "mean.A.detection_precision", None, 4),
 ("Baseline matched cells", "1886 ± 4", "mean.A.cells_matched", "sd.A.cells_matched", 0),
 ("Baseline phase MAE [rad]", "0.1593", "mean.A.phase_mae_rad", None, 4),
 ("Baseline in-cell phase MAE [rad]", "0.2756", "mean.A.phase_mae_rad_in_cell", None, 4),
 ("Baseline phase r", "0.8701", "mean.A.phase_pearson_r", None, 4),
 ("Baseline SSIM", "0.6275", "mean.A.phase_ssim", None, 4),
 ("Baseline Dice", "0.8312", "mean.A.seg_dice", None, 4),
 ("Baseline AJI", "0.6696", "mean.A.seg_aji", None, 4),
 ("Baseline boundary F1", "0.4574", "mean.A.seg_boundary_f1", None, 4),
 ("Baseline BA bias dry mass", "-0.0434", "mean.A.dry_mass_relative_bias", None, 4),
 ("Baseline BA LoA dry mass", "[-0.252, +0.165]", "mean.A.dry_mass_loa_lower", "mean.A.dry_mass_loa_upper", "loa"),
 ("Baseline BA bias area", "+0.0455", "mean.A.area_relative_bias", None, 4),
 ("Baseline BA LoA area", "[-0.172, +0.263]", "mean.A.area_loa_lower", "mean.A.area_loa_upper", "loa"),
 ("Baseline field-total bias, dry mass", "-0.1632", "mean.A.dry_mass_field_total_bias", None, 4),
 ("Baseline field-total bias, area", "-0.1078", "mean.A.area_field_total_bias", None, 4),
 ("Baseline field-total r, dry mass", "0.9879", "mean.A.dry_mass_field_total_pearson_r", None, 4),
 ("Baseline field-total dry-mass MAPE (new row)", "--", "mean.A.dry_mass_field_total_mape", "sd.A.dry_mass_field_total_mape", 4),
 ("Classical off-axis dry-mass MAPE", "0.2245", "classical113.off_axis.dry_mass_mape", None, 4),
 ("Classical off-axis area MAPE", "0.1567", "classical113.off_axis.area_mape", None, 4),
 ("Classical off-axis phase MAE", "0.2114", "classical113.off_axis.phase_mae_rad", None, 4),
 ("Classical off-axis in-cell phase MAE", "0.3655", "classical113.off_axis.phase_mae_rad_in_cell", None, 4),
 ("Classical off-axis phase r", "0.7580", "classical113.off_axis.phase_pearson_r", None, 4),
 ("Classical off-axis SSIM", "0.5022", "classical113.off_axis.phase_ssim", None, 4),
 ("Classical off-axis Dice", "0.7885", "classical113.off_axis.seg_dice", None, 4),
 ("Classical off-axis AJI", "0.5931", "classical113.off_axis.seg_aji", None, 4),
 ("Classical off-axis boundary F1", "0.2463", "classical113.off_axis.seg_boundary_f1", None, 4),
 ("Classical off-axis recall", "0.5763", "classical113.off_axis.detection_recall", None, 4),
 ("Classical off-axis precision", "0.7773", "classical113.off_axis.detection_precision", None, 4),
 ("Classical off-axis cell r", "0.8904", "classical113.off_axis.dry_mass_cell_pearson_r", None, 4),
 ("Classical off-axis matched cells", "1836", "classical113.off_axis.cells_matched", None, 0),
 ("Classical off-axis field-total dry-mass MAPE (new)", "--", "classical113.off_axis.dry_mass_field_total_mape", None, 4),
 ("Classical off-axis in-cell phase contrast", "+0.902 (113)", "classical107.off_axis.recovered_phase_contrast_rad", None, 3),
 ("Classical in-line phase r", "-0.1359 (113 fields)", "classical107.gabor.phase_pearson_r", None, 4),
 ("Classical in-line in-cell contrast", "-0.067", "classical107.gabor.recovered_phase_contrast_rad", None, 3),
 ("Classical in-line Dice", "0.1627", "classical107.gabor.seg_dice", None, 4),
 ("Classical in-line matched cells", "116 of 3186", "classical107.gabor.cells_matched", None, 0),
 ("In-line neural phase r", "0.8098 (113 fields)", "common.G_s42.phase_pearson_r", None, 4),
 ("In-line neural Dice", "0.7758", "common.G_s42.seg_dice", None, 4),
 ("In-line neural recall", "0.5508", "common.G_s42.detection_recall", None, 4),
 ("In-line neural dry-mass MAPE", "0.2308", "common.G_s42.dry_mass_mape", None, 4),
 ("In-line neural matched cells", "1755 of 3186", "common_run.G.s42.cells_matched", None, 0),
 ("+IPP (per-cell) dry-mass MAPE", "0.1915", "mean.B.dry_mass_mape", None, 4),
 ("+IPP (image) dry-mass MAPE", "0.1935", "mean.B1.dry_mass_mape", None, 4),
 ("Δ +IPP (per-cell) vs baseline", "+0.0152 (2×SD 0.0070)", "cmp.B_vs_A.dry_mass_mape.diff", "cmp.B_vs_A.dry_mass_mape.thr", "cmp"),
 ("Δ +IPP (image) vs baseline", "+0.0172 (2×SD 0.0052)", "cmp.B1_vs_A.dry_mass_mape.diff", "cmp.B1_vs_A.dry_mass_mape.thr", "cmp"),
 ("Δ +IPP (image) vs +IPP (per-cell)", "0.0020 (2×SD 0.0074)", "cmp.B1_vs_B.dry_mass_mape.diff", "cmp.B1_vs_B.dry_mass_mape.thr", "cmp"),
 ("+IPP (per-cell) precision", "0.7995", "mean.B.detection_precision", None, 4),
 ("+Area dry-mass MAPE", "0.1917", "mean.B2.dry_mass_mape", None, 4),
 ("+BGA dry-mass MAPE", "0.1978", "mean.C.dry_mass_mape", None, 4),
 ("w=0.1 dry-mass MAPE", "0.1853", "mean.W01.dry_mass_mape", None, 4),
 ("w=0.3 dry-mass MAPE", "0.1828", "mean.W03.dry_mass_mape", None, 4),
 ("w=3.0 dry-mass MAPE", "0.2102", "mean.W30.dry_mass_mape", None, 4),
 ("w=3.0 Dice", "0.8101", "mean.W30.seg_dice", None, 4),
 ("w=3.0 precision", "0.7252", "mean.W30.detection_precision", None, 4),
 ("w=3.0 phase MAE", "0.1851", "mean.W30.phase_mae_rad", None, 4),
 ("+Amplitude dry-mass MAPE", "0.2015", "mean.D0.dry_mass_mape", None, 4),
 ("Δ +Amplitude vs +IPP (per-cell)", "+0.0100", "cmp.D0_vs_B.dry_mass_mape.diff", None, 4),
 ("Compact Baseline dry-mass MAPE", "0.1783", "mean.KA.dry_mass_mape", None, 4),
 ("Δ Compact vs baseline", "+0.0019", "cmp.KA_vs_A.dry_mass_mape.diff", None, 4),
 ("Baseline 2×SD dry-mass MAPE", "0.0047", "sd.A.dry_mass_mape", None, "2sd"),
 ("Edge recall (baseline)", "0.208", "decomp.A.cell.edge.recall_mean", None, 3),
 ("Interior recall (baseline)", "0.914", "decomp.A.cell.interior.recall_mean", None, 3),
 ("Edge dry-mass MAPE (baseline)", "0.298", "locmean.A.edge.mape", None, 3),
 ("Interior dry-mass MAPE (baseline)", "0.153", "locmean.A.interior.mape", None, 3),
 ("False positives per run (baseline)", "320", "locmean.A.fp_all", None, 0),
 ("False positives within 15 px (baseline)", "141", "locmean.A.fp_near", None, 0),
 ("False positives per run (+IPP per-cell)", "477", "locmean.B.fp_all", None, 0),
 ("Amplitude MAE (+Amplitude)", "0.0656", "amp.D0.amplitude_mae", None, 4),
 ("Amplitude MAE (+Fwd fixed z)", "0.0637", "amp.D1.amplitude_mae", None, 4),
 ("Amplitude r (+Fwd fixed z)", "0.9034", "amp.D1.amplitude_pearson_r", None, 4),
 ("Amplitude ratio to A=1 (+Fwd fixed z)", "0.4337", "amp.D1.amplitude_mae_over_unity", None, 4),
 ("Reference amplitude mean", "0.9977", "amp.D0.amplitude_reference_mean", None, 4),
 ("Off-axis reference residual (test)", "0.923", "mean.A.forward_residual_reference", None, 4),
 ("Baseline residual ratio", "0.9969", "mean.A.forward_residual_ratio", None, 4),
 ("+Fwd (fixed z) residual ratio", "0.9915", "mean.D1.forward_residual_ratio", None, 4),
 ("+Fwd (free z) residual ratio", "0.9928", "mean.D2.forward_residual_ratio", None, 4),
 ("+Fwd (fixed z) in-cell phase MAE", "0.3011", "mean.D1.phase_mae_rad_in_cell", None, 4),
 ("+Fwd (free z) in-cell phase MAE", "0.2903", "mean.D2.phase_mae_rad_in_cell", None, 4),
 ("0.9x probe, 32 validation fields, off-axis", "-5e-5 (22 of 32 lower)", "disc.off_axis.margin.scaled_0.9", None, 5),
 ("Best phase scale, off-axis (median)", "0.68 (IQR 0.80)", "scale.off_axis.best_scale_median", None, 3),
 ("0.9x probe, test, global surface", "-8e-5", "probe.global.phase_scaled_0.9.change", None, 5),
 ("0.9x probe, test, per-field surfaces", "-6e-4", "probe.per_field.phase_scaled_0.9.change", None, 4),
 ("Per-field-surface residual", "0.43", "probe.per_field.phase_reference", None, 3),
 ("0.5x probe, test, global", "+0.003", "probe.global.phase_scaled_0.5.change", None, 4),
 ("0.5x probe, test, per-field", "+0.025", "probe.per_field.phase_scaled_0.5.change", None, 4),
 ("Learned z after epoch 1 [um]", "34.12", "learnedz.initial_after_epoch1", None, 2),
 ("Learned z final [um]", "33.39", "learnedz.final", None, 2),
 ("Learned z excursion [um]", "0.958", "learnedz.excursion", None, 3),
 ("Learned z last-10 spread [um]", "0.026", "learnedz.last10_spread", None, 3),
 ("Learned z at selected checkpoint [um]", "33.35", "learnedz.scored", None, 2),
 ("z scan IQR off-axis [um]", "156.4", "z.off_axis.per_image_iqr_um", None, 1),
 ("z scan in-line median [um]", "50.5", "z.gabor.per_image_median_z_um", None, 2),
 ("z scan IQR in-line [um]", "6.0", "z.gabor.per_image_iqr_um", None, 1),
 ("Gradient ratio median", "0.302", "grad.median", None, 3),
 ("Gradient ratio range", "0.18-0.67", "grad.min", "grad.max", "range"),
 ("Baseline latency PyTorch FP32 [ms]", "12.32 ± 0.11", "hw.A.mean.pytorch_fp32.latency_mean_ms", "hw.A.sd.pytorch_fp32.latency_mean_ms", 2),
 ("Baseline latency ONNX FP16 [ms]", "8.42 ± 0.06", "hw.A.mean.onnx_fp16.latency_mean_ms", "hw.A.sd.onnx_fp16.latency_mean_ms", 2),
 ("Compact latency ONNX FP16 [ms]", "7.98", "hw.KA.mean.onnx_fp16.latency_mean_ms", None, 2),
 ("Membrane Dice (baseline)", "0.496", "membrane.A.mean.seg_dice", None, 3),
 ("Membrane AJI", "0.235", "membrane.A.mean.seg_aji", None, 3),
 ("Membrane boundary F1", "0.062", "membrane.A.mean.seg_boundary_f1", None, 3),
 ("Membrane recall", "0.107", "membrane.A.mean.detection_recall", None, 3),
]


def fmt(k, d):
    return f"{V[k]['value']:.{d}f}"


print("| Quantity | v4.1 | 4.2 | Source (file; key in analysis/v42/values.json) |")
print("|---|---|---|---|")
for q, old, k, k2, d in ROWS:
    if d == "ci":
        new = f"{fmt(k, 4)}–{fmt(k2, 4)}"
    elif d == "loa":
        new = f"[{V[k]['value']:+.3f}, {V[k2]['value']:+.3f}]"
    elif d == "cmp":
        new = f"{V[k]['value']:+.4f} (2×SD {V[k2]['value']:.4f})"
    elif d == "range":
        new = f"{fmt(k, 3)}–{fmt(k2, 3)}"
    elif d == "2sd":
        new = f"{2 * V[k]['value']:.4f}"
    else:
        new = fmt(k, d) + (f" ± {fmt(k2, d)}" if k2 else "")
    src = V[k]["source"].split(";")[0]
    if len(V[k]["source"].split(";")) > 1:
        src += " (+ seeds)"
    print(f"| {q} | {old} | {new} | `{src}`; `{k}` |")
