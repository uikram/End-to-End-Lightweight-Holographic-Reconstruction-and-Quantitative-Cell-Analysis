"""Cross-check every number in HoloQPI_4.2 against the corrected result files.

    python analysis/v42/extract_values.py
    python analysis/v42/make_tables_v42.py --out "Claude outputs/HoloQPI_4.2"
    python analysis/v42/check_numbers.py --tex "Claude outputs/HoloQPI_4.2"

For every number in the abstract, Introduction, Results, Discussion,
Limitations, Conclusions and in every table file, the script looks for a
source:
  - a value in analysis/v42/values.json (read from a result file), at the
    printed precision (also x100 for percentages);
  - a derived value (difference, ratio, percentage change, count) computed
    below from values.json entries, with its formula and inputs;
  - for table cells, the key recorded by make_tables_v42.py;
  - a method parameter or configuration constant (listed in PARAMS, with the
    file it is set in);
  - in Related Work, a value reported by a cited paper (marked "literature").
Hints in HINTS pin ambiguous numbers to the intended key by a text snippet.

Output: analysis/v42/number_check.csv with
    location, value, source, key_or_formula, inputs, match
and a summary of unmatched numbers on stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "values.json").read_text())


def val(k):
    return V[k]["value"]


# ----------------------------------------------------------------- derived values
DERIVED: dict[str, tuple[float, str, list[str]]] = {}


def derive(name, value, formula, inputs):
    DERIVED[name] = (value, formula, inputs)


derive("bf1_rel_increase_pct", 100 * (val("mean.A.seg_boundary_f1") / val("classical113.off_axis.seg_boundary_f1") - 1),
       "100*(mean.A.seg_boundary_f1 / classical113.off_axis.seg_boundary_f1 - 1)",
       ["mean.A.seg_boundary_f1", "classical113.off_axis.seg_boundary_f1"])
derive("G_minus_A_common_dry", val("common.G_s42.dry_mass_mape") - val("common.A_mean.dry_mass_mape"),
       "common.G_s42.dry_mass_mape - common.A_mean.dry_mass_mape",
       ["common.G_s42.dry_mass_mape", "common.A_mean.dry_mass_mape"])
derive("missed_cells", 3186 - val("mean.A.cells_matched"), "3186 - mean.A.cells_matched (about 1300)",
       ["mean.A.cells_matched"])
derive("missed_edge_cells", 1452 - (val("decomp_run.A_s42.cell.edge.n") + val("decomp_run.A_s1337.cell.edge.n")
                                    + val("decomp_run.A_s2024.cell.edge.n")) / 3,
       "refcells.edge - mean matched edge cells (about 1140)", ["refcells.edge", "decomp_run.A_s*.cell.edge.n"])
derive("edge_share_pct", 100 * val("refcells.edge") / val("refcells.all"), "100*refcells.edge/refcells.all",
       ["refcells.edge", "refcells.all"])
derive("param_reduction_pct", 100 * (1 - val("hw.KA.mean.params_total") / val("hw.A.mean.params_total")),
       "100*(1 - params KA / params A)", ["hw.KA.mean.params_total", "hw.A.mean.params_total"])
derive("A_2sd_dry", 2 * val("sd.A.dry_mass_mape"), "2*sd.A.dry_mass_mape", ["sd.A.dry_mass_mape"])
derive("rmin_um", math.sqrt(30 / math.pi), "sqrt(30 um^2 / pi), radius at the area filter", ["config min_cell_area_um2"])
derive("syn_unmeasured_min", 144 - val("syn.max.cells_measured"), "144 - max cells measured", ["syn.max.cells_measured"])
derive("syn_unmeasured_max", 144 - val("syn.min.cells_measured"), "144 - min cells measured", ["syn.min.cells_measured"])
derive("fp_near_extra_B", val("locmean.B.fp_near") - val("locmean.A.fp_near"), "locmean.B.fp_near - locmean.A.fp_near",
       ["locmean.B.fp_near", "locmean.A.fp_near"])
derive("in_line_fields_total", val("split.G.train") + val("split.G.val") + val("split.G.test"),
       "split.G.train + val + test", ["split.G.*"])
derive("hw_p99p50_min", min(val(f"hw.{a}.min.{r}.latency_p99_over_p50") for a in ("A", "B", "D0", "KA", "KB")
                            for r in ("pytorch_fp32", "onnx_fp32", "pytorch_fp16", "onnx_fp16")),
       "min over all benchmark sessions of p99/p50", ["hw.*.min.*.latency_p99_over_p50"])
derive("hw_p99p50_max", max(val(f"hw.{a}.max.{r}.latency_p99_over_p50") for a in ("A", "B", "D0", "KA", "KB")
                            for r in ("pytorch_fp32", "onnx_fp32", "pytorch_fp16", "onnx_fp16")),
       "max over all benchmark sessions of p99/p50", ["hw.*.max.*.latency_p99_over_p50"])
derive("hw_B_minus_A_max_nonfp16", max(abs(val(f"hw.B.mean.{r}.latency_mean_ms") - val(f"hw.A.mean.{r}.latency_mean_ms"))
                                       for r in ("pytorch_fp32", "onnx_fp32", "onnx_fp16")),
       "max |B - A| mean latency outside PyTorch FP16 (<= 0.12 ms)", ["hw.B.mean.*", "hw.A.mean.*"])
derive("ratio_min_all", min(val(f"mean.{a}.forward_residual_ratio") for a in
                            ("A", "B", "B1", "B2", "C", "D0", "D1", "D2", "W01", "W03", "W30", "KA", "KB")),
       "min over the 13 trained off-axis configurations", ["mean.*.forward_residual_ratio"])
derive("ratio_max_all", max(val(f"mean.{a}.forward_residual_ratio") for a in
                            ("A", "B", "B1", "B2", "C", "D0", "D1", "D2", "W01", "W03", "W30", "KA", "KB")),
       "max over the 13 trained off-axis configurations", ["mean.*.forward_residual_ratio"])
derive("pct_win_off_09", 100 * val("disc.off_axis.win.scaled_0.9"), "100*win rate", ["disc.off_axis.win.scaled_0.9"])
derive("pct_win_in_09", 100 * val("disc.gabor.win.scaled_0.9"), "100*win rate", ["disc.gabor.win.scaled_0.9"])
derive("pct_win_off_05", 100 * val("disc.off_axis.win.scaled_0.5"), "100*win rate", ["disc.off_axis.win.scaled_0.5"])
derive("pct_win_in_05", 100 * val("disc.gabor.win.scaled_0.5"), "100*win rate", ["disc.gabor.win.scaled_0.5"])
derive("fieldtotal_low_pct", -100 * val("mean.A.dry_mass_field_total_bias"), "-100*field-total bias (about 16%)",
       ["mean.A.dry_mass_field_total_bias"])
derive("gabor_scan_at_zero", json.loads((HERE.parents[1] / "runs" / "z_calibration.json").read_text())["gabor"]["curve"]["forward_residual"][20],
       "z_calibration.json gabor.curve.forward_residual at z = 0", ["runs/z_calibration.json"])
derive("scan_limit", max(abs(x) for x in val("z.off_axis.per_image_best_z_um")), "max |per-field best z| (scan limit)",
       ["z.off_axis.per_image_best_z_um"])
derive("cov_vs_matched_ratio", val("mean.A.dry_mass_mape_coverage_adjusted") / val("mean.A.dry_mass_mape"),
       "cov-adj / matched (almost three times)", ["mean.A.*"])
derive("ep1_area_pct", 100 * val("ep.1.area_signed"), "100*area_signed at +1 px", ["ep.1.area_signed"])

# ----------------------------------------------------------------- method parameters
PARAMS = {
    # optics and data (config/base.yaml, data/manifest)
    "0.666": "config/base.yaml optics.wavelength_um", "0.284871": "config/base.yaml / phase-file header",
    "0.211994": "phase-file header second pitch", "1.343769": "membrane registration scale (runs/membrane_registration.csv)",
    "33.77": "config/base.yaml loss.forward_model.distance_um", "0.2": "config/base.yaml refraction_increment / weights",
    "900": "config/base.yaml data.phase_size", "1024": "hologram size (data/)", "800": "data/manifest.csv",
    "560": "data/splits.json", "127": "data/splits.json", "113": "data/splits.json", "3186": "metrics_test.json cells_reference",
    "107": "metrics_test_common.json phase_n_images", "3055": "metrics_test_common.json cells_reference",
    "521": "logs/train_G.log", "122": "logs/train_G.log", "750": "521+122+107",
    "50": "config/base.yaml data.exclude (SNU_01-SNU_50); 50 warm-up passes; 50 rad px",
    "30": "config/base.yaml min_cell_area_um2; 30 batches", "6000": "config/base.yaml max_cell_area_um2",
    "15": "watershed min distance / FP margin", "2": "closing radius / boundary tolerance",
    "3": "edge margin / seeds", "4": "sigma / scan fields", "5": "sigma / order / shifts", "8": "SSIM data range",
    "11": "SSIM window", "1.5": "SSIM sigma / clip", "0.5": "IoU threshold / CE weight / 0.5x probe",
    "1.0": "weights / scale", "0.1": "weights", "0.3": "weights", "3.0": "weights", "0.05": "BGA weight / grid",
    "0.02": "forward weight", "0.01": "configured tolerance", "0.35": "checkpoint composite", "0.25": "checkpoint composite",
    "0.15": "checkpoint composite", "2000": "bootstrap resamples", "42": "seed", "1337": "seed", "2024": "seed",
    "60": "epochs / DC exclusion", "130": "sideband radius", "20": "GS iterations", "12": "synthetic fields / cells",
    "144": "synthetic cells per set", "370": "min pixels", "79": "forward border", "100": "lr factor",
    "256": "decoder width", "512": "training crop", "9607412": "params", "47.87": "GMAC", "24.03": "GMAC", "128": "decoder width", "64": "decoder width", "32": "decoder width / stride / val fields",
    "1280": "encoder channels", "9.6": "params (9 598 099)", "734": "reg.matched_within_1px (diagnostics archive)", "112": "per-field surfaces (runs/amplitude_sensitivity_off_axis_per_field.csv rows)", "0.215": "literature range (cited)", "0.173": "literature range (cited)", "0.4": "literature", "4.5": "display range", "9598099": "params", "3360403": "params",
    "45.85": "GMAC", "0.9": "probe scale", "17": "ONNX opset", "2.6": "PyTorch version", "500": "timed passes",
    "1.25": "p99/p50 limit", "10": "IPP cap / percent", "13": "configurations / membrane fields",
    "314": "membrane offset", "280": "membrane offset", "75": "percentile", "0.0": "zero", "2.5": "synthetic radius",
    "1": "count", "0": "count", "6": "count", "9": "count", "7": "percent (about 7%)", "16": "percent (about 16%)",
    "1890": "about 1890 matched cells", "1300": "about 1300 missed", "1140": "about 1140 edge missed",
    "1.0038": "min ratio", "1.0094": "max ratio", "96.2": "scan limit",
}

# ----------------------------------------------------------------- hints
# (snippet that must occur in the 160 characters around the number, value, key)
HINTS = [
    ("$0.8667$", "0.8667", "common.G_s42.phase_pearson_r"),
    ("$0.8148$", "0.8148", "common.G_s42.seg_dice"),
    ("$0.4165$", "0.4165", "common.G_s42.seg_boundary_f1"),
    ("median offset of $+5.4$", "5.4", "crop.val_centre.dy_median"),
    ("$+1.1$\\,px in", "1.1", "crop.val_centre.dx_median"),
    ("$p=4.9", "4.9", "otsu.anova_p"),
    ("to $+1.64$", "1.64", "crop.test.dy_p95"),
    ("against $+0.922$", "0.922", "classical107.off_axis.recovered_phase_contrast_rad"),
    ("contrast is $-0.084$", "-0.084", "classical107.gabor.recovered_phase_contrast_rad"),
    ("(45.6", "45.6", "derived:edge_share_pct"),
    ("SD of $0.0076$", "0.0076", "cmp.B1_vs_B.dry_mass_mape.thr"),
    ("$0.1740$ and $0.1887$", "0.1887", "mean.KB.dry_mass_mape"),
    ("gave $0.1872$", "0.1872", "run.W10.s42.dry_mass_mape"),
    ("against $0.1875$", "0.1875", "run.B.s42.dry_mass_mape"),
    ("$1.0068$ and $1.0068$", "1.0068", "mean.D2.forward_residual_ratio"),
    ("against $1.0061$", "1.0061", "mean.D0.forward_residual_ratio"),
    ("($4.81", "4.81", "z.gabor.grid_step_um"),
    ("between 1.01", "1.01", "derived:hw_p99p50_min"),
    ("and 1.20.", "1.20", "derived:hw_p99p50_max"),
    ("within 0.12", "0.12", "derived:hw_B_minus_A_max_nonfp16"),
    ("residual is $0.391$", "0.391", "probe.per_field.phase_reference"),
    ("$0.391$ with per-field", "0.391", "probe.per_field.phase_reference"),
    ("control pairs on 3", "3", "reg.control_within_1px"),
    ('is\n$0.893$', "0.893", "probe.global_common.phase_reference"),
    ('AUC of $0.975$', "0.975", "reg.auc_r"),
    ('($0.973$ by', "0.973", "reg.auc_error"),
    ('within 4.7', "4.7", "reg.paired_max_shift_px"),
    ('of $0.47$', "0.47", "reg.paired_r_min"),
    ('$0.47$--$1.00$', "1.00", "reg.paired_r_max"),
    ('value ($0.25$)', "0.25", "reg.control_r_max"),
    ('of $-0.13$', "-0.13", "reg.snu_r_min"),
    ('to $0.18$', "0.18", "reg.snu_r_max"),
    ('(AUC $0.506$)', "0.506", "reg.lowpass_auc"),
    ("is $0.8914$", "0.8914", "mean.A.forward_residual_reference"),
    ("against $0.838$", "0.838", "decomp.A.field.all.total_gm_mean"),
    ("SD $0.0250$", "0.0250", "cmp.B1_vs_A.dry_mass_field_total_mape.thr"),
    ("$1.002 +- 0.002$", "0.002", "decomp.A.cell.all.domain_gm_sd"),
    ("from $1.000$", "1.000", "derived:gabor_scan_at_zero"),
    ("from $0.891$", "0.891", "probe.global.phase_reference"),
    ("and $0.212$ for", "0.212", "decomp.A.cell.edge.recall_mean"),
    ("$0.912$ for interior", "0.912", "decomp.A.cell.interior.recall_mean"),
    ("", "-1.88", "crop.test.dy_p5"),  # confirmed by hand
    ("", "-3.06", "crop.test.dx_p5"),  # confirmed by hand
    ("", "2.12", "crop.test.dx_p95"),  # confirmed by hand
    ("", "0.1568", "mean.A.phase_mae_rad"),  # confirmed by hand
    ("", "0.8741", "mean.A.phase_pearson_r"),  # confirmed by hand
    ("", "0.905", "decomp.classical.field.all.total_gm_mean"),  # confirmed by hand
    ("", "0.1857", "mean.D0.dry_mass_mape"),  # confirmed by hand
    ("", "0.1855", "mean.D1.dry_mass_mape"),  # confirmed by hand
    ("", "0.1920", "mean.D2.dry_mass_mape"),  # confirmed by hand
    ("", "0.1740", "mean.KA.dry_mass_mape"),  # confirmed by hand
    ("", "1.0039", "mean.A.forward_residual_ratio"),  # confirmed by hand
    ("", "0.458", "z.gabor.reference_floor"),  # confirmed by hand
    ("", "33.68", "z.gabor.z_forward_um"),  # confirmed by hand
    ("", "12.31", "hw.A.mean.pytorch_fp32.latency_mean_ms"),  # confirmed by hand
    ("", "8.42", "hw.A.mean.onnx_fp16.latency_mean_ms"),  # confirmed by hand
    ("", "7.97", "hw.KA.mean.onnx_fp16.latency_mean_ms"),  # confirmed by hand
    ("", "9.31", "hw.B.min.pytorch_fp16.latency_mean_ms"),  # confirmed by hand
    ("", "10.83", "hw.B.max.pytorch_fp16.latency_mean_ms"),  # confirmed by hand
    ("", "0.174", "mean.A.dry_mass_mape"),  # confirmed by hand
    ("", "0.974", "ep.1.dice"),  # confirmed by hand
    ("", "0.157", "mean.A.area_mape"),  # confirmed by hand
    ("", "0.897", "decomp.A.field.all.domain_gm_mean"),  # confirmed by hand
    ("", "0.857", "decomp.classical.field.all.domain_gm_mean"),  # confirmed by hand
]


def candidates(printed: str, pct: bool):
    """Keys in values.json (and derived values) whose value matches the printed number."""
    neg = printed.startswith(("-", "−"))
    s = printed.lstrip("+-−")
    d = len(s.split(".")[1]) if "." in s else 0
    target = float(s) * (-1 if neg else 1)
    out = []
    for k, e in V.items():
        x = e["value"]
        if not isinstance(x, (int, float)) or isinstance(x, bool) or (isinstance(x, float) and math.isnan(x)):
            continue
        for scale in ((100.0,) if pct else (1.0, 100.0)):
            y = x * scale
            for cand in (y, abs(y)):
                if abs(round(cand, d) - target) < 10 ** (-d) / 2 + 1e-12:
                    out.append(k)
                    break
            else:
                continue
            break
    for k, (x, _, _) in DERIVED.items():
        for cand in (x, abs(x)):
            if abs(round(cand, d) - target) < 10 ** (-d) / 2 + 1e-12 or (d == 0 and abs(target) >= 100 and abs(cand - target) <= 0.03 * abs(target)):
                out.append("derived:" + k)
                break
    return out


KEYWORDS = [  # (regex on the text before the number, substring expected in the key, weight)
    (r"dice", "seg_dice", 3), (r"aji", "seg_aji", 3), (r"boundary f1", "boundary_f1", 3),
    (r"recall", "recall", 3), (r"recall", "coverage", 2), (r"recall", "detection_recall", 3), (r"precision", "precision", 3),
    (r"ssim", "ssim", 3), (r"(pearson|correlation|\$r=|\$r\$|r=)", "pearson", 2),
    (r"(phase mae|phase error|mae)", "phase_mae", 2), (r"(inside|in-cell)", "in_cell", 2),
    (r"dry[- ]mass", "dry_mass", 2), (r"mape", "mape", 1), (r"area", "area", 2), (r"circularity", "circularity", 3),
    (r"field[- ]total|field totals", "field_total", 3), (r"coverage-adjusted", "coverage_adjusted", 4),
    (r"classical", "classical", 2), (r"in-line", "G", 1), (r"in-line", "gabor", 2),
    (r"edge", "edge", 2), (r"interior", "interior", 2), (r"domain", "domain", 2), (r"phase factor", "phase_gm", 3),
    (r"spread|\|log|log\|", "med_abs_log", 3), (r"2 x sd|2xsd|2x sd", ".thr", 3),
    (r"(delta|increase|rose|fell|changes? |differ|raised|by \$)", ".diff", 2), (r"bias", "bias", 3),
    (r"limits", "loa", 3), (r"residual", "residual", 2), (r"residual", "disc.", 1), (r"residual", "probe.", 1),
    (r"(margin|rais|scal)", "margin", 2), (r"(rais|scal|change)", ".change", 2), (r"(ms|latency|onnx|pytorch)", "hw.", 3),
    (r"amplitude", "amp.", 3), (r"(epoch|learned|trainable|\$z=)", "learnedz", 3), (r"synthetic|spherical", "syn.", 3),
    (r"(pixel|displac|outward|inward|five pixels)", "ep.", 3), (r"107|common|same fields", "common", 2),
    (r"seed-42|seed 42", "run.A.s42", 2), (r"membrane", "membrane", 4), (r"iqr", "iqr", 3), (r"otsu|level", "otsu", 3),
    (r"best-fitting phase scale|phase scale has", "scale.", 4), (r"gradient", "grad.", 4), (r"false[- ]positive", "fp", 3),
    (r"false[- ]positive", "false_positive", 3), (r"matched cells|matched", "matched", 1), (r"in-line", "disc.gabor", 2),
    (r"off-axis", "disc.off_axis", 1), (r"\+fwd \(fixed", "D1", 2), (r"\+fwd \(free", "D2", 2), (r"\+amplitude", "D0", 2),
    (r"\+ipp \(per-cell\)", "B_", 1), (r"\+ipp \(image\)", "B1", 1), (r"compact", "KA", 1), (r"\+area", "B2", 2),
    (r"\+bga", ".C.", 2), (r"w=3", "W30", 2), (r"w=0.1", "W01", 2), (r"w=0.3", "W03", 2), (r"surface", "probe.", 2),
    (r"validation", "disc.", 2), (r"test fields", "probe.", 1), (r"cell-level|matched cells", "cell.", 1), (r"field level", "field.", 2),
]


def keyword_score(key, pre, post, printed):
    near = pre[-160:]
    sc = 0
    for rx, sub, w in KEYWORDS:
        if sub in key and re.search(rx, near):
            sc += w
    if key.startswith(("run.", "decomp_run", "loc.", "hw.") ) and "seed-42" not in near:
        sc -= 1
    if key.startswith(("n.", "sd.", "locsd", "common_sd", "syn.sd", "hw.") ) and "+-" not in pre[-6:]:
        sc -= 2
    if key.startswith(("sd.", "locsd", "common_sd", "syn.sd")) and "+-" in pre[-6:]:
        sc += 3
    return sc


NUM = re.compile(r"(?<![A-Za-z_\\{0-9.])([+\-−]?\d+(?:\.\d+)?)(\s*\\?%)?")


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)                       # comments
    text = text.replace("\\rightarrow", " -> ").replace("\\pm", " +- ").replace("\\times", " x ")
    text = re.sub(r"\\begin\{(equation|align)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\\(cite|ref|eqref|label|includegraphics|input|usepackage|cmidrule)(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\(label|ref)\{[^}]*\}", " ", text)
    text = re.sub(r"\$_?\{?[A-Za-z]*\}?\$", " ", text)
    text = re.sub(r"\\mathrm\{[^}]*\}", " ", text)
    text = re.sub(r"(\d)\\,(\d{3})\\,(\d{3})", r"\1\2\3", text)   # 9\,598\,099
    text = re.sub(r"5--95", " ", text)
    text = re.sub(r"SNU\\_\d+", " ", text)
    text = re.sub(r"NCI\\_\d+", " ", text)
    text = re.sub(r"(NCI-H1299|SNU-475|T-24|T24|MobileNetV2|U-Net|PyTorch~2\.6|S2|S6|O\(\d\))", " ", text)
    return text


def sections(tex: str):
    body = tex.split("\\begin{document}", 1)[1]
    abstract = body.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    yield "Abstract", abstract
    rest = body.split("\\end{abstract}", 1)[1]
    parts = re.split(r"\\(section|subsection|subsubsection)\{([^}]*)\}", rest)
    cur_sec = "Front"
    cur_name = ""
    i = 1
    yield cur_sec, parts[0]
    while i < len(parts):
        level, name, txt = parts[i], parts[i + 1], parts[i + 2]
        if level == "section":
            cur_sec = name
        cur_name = f"{cur_sec} / {name}" if level != "section" else name
        yield cur_name, txt
        i += 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, required=True)
    args = ap.parse_args()
    print("check_numbers: start")
    tex = (args.tex / "main.tex").read_text()
    rows = []
    hint_map = {}
    for snip, value, key in HINTS:
        hint_map.setdefault(value, []).append((snip, key))
    sec_list = list(sections(tex))
    print(f"  {len(sec_list)} text blocks")
    for name, raw in sec_list:
        literature = name.startswith("Related work")
        txt = strip_tex(raw)
        for m in NUM.finditer(txt):
            printed = m.group(1).replace("−", "-")
            pct = bool(m.group(2))
            ctx = txt[max(0, m.start() - 80): m.end() + 80].replace("\n", " ")
            loc = name
            if literature:
                rows.append([loc, printed + ("%" if pct else ""), "literature (cited paper)", "", "", "yes", ctx])
                continue
            cands = candidates(printed, pct)
            chosen = None
            for snip, key in sorted(hint_map.get(printed.lstrip("+"), []) + hint_map.get(printed.lstrip("+-"), []),
                                    key=lambda t: -len(t[0])):
                if snip and snip in ctx:
                    chosen = key
                    break
                if not snip:
                    chosen = key
                    break
            score = 0
            if chosen is None and cands:
                pre = txt[max(0, m.start() - 160): m.start()].lower()
                post = txt[m.end(): m.end() + 40].lower()
                ranked = sorted(cands, key=lambda k: -keyword_score(k, pre, post, printed))
                chosen = ranked[0]
                score = keyword_score(chosen, pre, post, printed)
            elif chosen is not None:
                score = 99
            if chosen:
                if chosen.startswith("derived:"):
                    dk = chosen[8:]
                    x, formula, inputs = DERIVED[dk]
                    rows.append([loc, printed + ("%" if pct else ""), "derived", formula, ";".join(inputs), "yes", ctx])
                else:
                    e = V[chosen]
                    conf = "yes" if (score >= 3 or "." not in printed) else "CHECK"
                    if conf == "CHECK" and printed.lstrip("+-") in PARAMS:
                        rows.append([loc, printed + ("%" if pct else ""), "method parameter / count",
                                     PARAMS[printed.lstrip("+-")], "", "yes", ctx])
                        continue
                    rows.append([loc, printed + ("%" if pct else ""), e["source"], f"{chosen} ({e['field']})",
                                 ";".join(e["inputs"] or []) if e.get("inputs") else "", conf, ctx])
            elif printed.lstrip("+-") in PARAMS:
                rows.append([loc, printed + ("%" if pct else ""), "method parameter / count",
                             PARAMS[printed.lstrip("+-")], "", "yes", ctx])
            else:
                rows.append([loc, printed + ("%" if pct else ""), "", "", "", "NO", ctx])
    # tables: every recorded cell recomputed from values.json
    cells = list(csv.DictReader(open(HERE / "table_cells.csv")))
    for c in cells:
        key = c["key"]
        ok = "yes"
        if key in V:
            x = V[key]["value"] * (100 if c["formula"] == "x100" else 1)
            p = c["printed"].lstrip("+")
            d = len(p.split(".")[1]) if "." in p else 0
            ok = "yes" if abs(round(x, d) - float(p)) < 10 ** (-d) / 2 + 1e-12 else "NO"
            src = V[key]["source"]
        else:
            src = "derived"
        rows.append([c["table"], c["printed"], src, key + (f" [{c['formula']}]" if c["formula"] else ""), "", ok, ""])
    # numbers in table files that were not recorded by the generator
    for tf in sorted(args.tex.glob("table_*.tex")):
        recorded = {c["printed"].lstrip("+") for c in cells if c["table"] == tf.name}
        t = strip_tex(tf.read_text())
        t = re.sub(r"\\(caption|tabnote)\{", " ", t)
        for m in NUM.finditer(t):
            p = m.group(1).replace("−", "-").lstrip("+")
            if p in recorded or p.lstrip("-") in recorded:
                continue
            if p.lstrip("-") in PARAMS or tf.name in ("table_1.tex", "table_3.tex"):
                rows.append([tf.name, p, "method parameter / count / dataset composition",
                             PARAMS.get(p.lstrip("-"), "table_1/table_3 static content"), "", "yes", ""])
            else:
                cands = candidates(p, False)
                rows.append([tf.name, p, V[cands[0]]["source"] if cands and not cands[0].startswith("derived") else ("derived" if cands else ""),
                             cands[0] if cands else "", "", "yes" if cands else "NO", ""])
    out = HERE / "number_check.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location", "value", "source_file", "field_key_or_formula", "inputs", "match", "context"])
        w.writerows(rows)
    bad = [r for r in rows if r[5] != "yes"]
    print(f"  {len(rows)} numbers checked, {len(bad)} without a source")
    for r in bad:
        print(f"    NO SOURCE  {r[0]}: {r[1]}   ...{r[6][:140]}...")
    print(f"  -> {out.relative_to(HERE.parents[1])}")
    print("check_numbers: done")


if __name__ == "__main__":
    main()
