#!/usr/bin/env python
"""Write the numeric part of analysis/v42/MANUSCRIPT_SYNC.md from results_for_manuscript/ (no hand-typed numbers).

python analysis/v42/make_sync.py  -> regenerates the tables of analysis/v42/MANUSCRIPT_SYNC.md

"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results_for_manuscript"
J = lambda p: json.loads((R / p).read_text())
out = []


def fmt(e, d=4):
    if isinstance(e, dict) and "value" in e:
        v = e["value"]
        if not isinstance(v, (int, float)):
            return str(v)
        s = f"{v:.{d}f}" if abs(v) < 1000 and not float(v).is_integer() else f"{v:.0f}"
        if e.get("sd") is not None and e.get("n_runs", 1) >= 2:
            s += f" ± {e['sd']:.{d}f}"
        return s
    return str(e)


def src(e):
    if not isinstance(e, dict) or "source_file" not in e:
        return ""
    f = e["source_file"].split(";")
    f = f[0].strip() + (f" (+{len(f)-1} more)" if len(f) > 1 else "")
    return f"`{f}` :: `{e['source_key']}` ({e['derivation']})"


def table(title, header, rows):
    out.append(f"\n### {title}\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")


cfg = J("metadata/configurations.json")
table("Table 3 — configurations (training runs, seeds, comparator)",
      ["Configuration", "Training runs", "Seeds", "Comparator", "Loss weights (non-zero)", "Source"],
      [[c["label"], c["training_runs"], ", ".join(map(str, c["seeds"])), c["comparator"] or "--",
        ", ".join(f"{k}={v:g}" for k, v in c["loss_weights_nonzero"].items()), "`metadata/configurations.json`"]
       for k, c in cfg.items() if k in ("A","G","B","B1","B2","C","W01","W03","W30","D0","D1","D2","KA","KB")])

b = J("baseline/baseline.json")
rows = []
for k, v in b["table_4"].items():
    d = v["delta_neural_minus_classical"]
    rows.append([k, fmt(v["classical"]), fmt(v["neural"]), v.get("descriptive_difference_text") or fmt(d),
                 "descriptive only (not a resolution test)" if k == "cells_matched" else "no resolution test (deterministic comparator)"])
table("Table 4 — Baseline (mean ± SD, 3 seeds) vs classical off-axis; 113 test fields, 3186 reference cells",
      ["Metric key", "Classical (n=1, deterministic)", "Neural (3 training runs)", "Δ neural − classical", "Note"], rows)
out.append("\nSource for every row: `baseline/baseline.json::table_4.<key>` (neural: `runs/v2_{baseline,A_seed1337,A_seed2024}*/metrics_test.json`; classical: `runs/conventional_off_axis/metrics_test.json`).")

i = J("inline/inline.json")["table_5"]
rows = []
for k, v in i.items():
    if k == "recovered_in_cell_phase_contrast":
        c = v
        rows.append(["recovered in-cell phase contrast [rad]", fmt(c["in_line_neural"]) if "value" in c["in_line_neural"] else "N/A (see AUDIT §7)",
                     fmt(c["off_axis_neural"]) if "value" in c["off_axis_neural"] else "N/A (see AUDIT §7)",
                     fmt(c["classical_off_axis"], 3), fmt(c["classical_in_line"], 3)])
        continue
    rows.append([k, fmt(v["in_line_neural"]), fmt(v["off_axis_neural"]), fmt(v["classical_off_axis"]), fmt(v["classical_in_line"])])
table("Table 5 — 107 common fields, 3055 reference cells (n = training runs: in-line neural 1, off-axis neural 3, classical deterministic)",
      ["Metric key", "In-line neural (n=1, seed 42)", "Off-axis neural (n=3)", "Classical off-axis", "Classical in-line"], rows)

t6 = b["table_6"]
rows = []
for q, r in t6.items():
    m, ba, ft = r["matched_cells"], r["per_field_bland_altman"], r["field_total"]
    rows.append([q, fmt(m["mape"]) if m["mape"] else "N/A", fmt(m["coverage_adjusted_mape"]) if m["coverage_adjusted_mape"] else "N/A",
                 fmt(m["pearson_r_per_cell"]), fmt(ba["relative_bias"]), f"[{ba['loa_lower']['value']:+.3f}, {ba['loa_upper']['value']:+.3f}]",
                 fmt(ft["bias"]), fmt(ft["mape"]), fmt(ft["pearson_r"])])
table("Table 6 — measurement agreement (baseline, 3 seeds; matched cells vs field totals); circularity field totals are N/A",
      ["Quantity", "Matched MAPE", "Cov.-adj. MAPE", "Pearson r (cells)", "BA bias (per field)", "LoA (mean over seeds)", "Field-total bias", "Field-total MAPE", "Field-total r"], rows)

d = J("decomposition/decomposition.json")
rows = []
for cfgk in ("A", "B", "B1", "classical"):
    bc = d["by_config"][cfgk]
    for lvl in ("cell.all", "cell.interior", "cell.edge", "field.all"):
        x = bc[lvl]
        f = lambda k: (f"{x[k]['mean']:.3f}" + (f" ± {x[k]['sd']:.3f}" if x[k]["sd"] is not None else ""))
        n = x["n_matched"]["mean"] if "n_matched" in x else x["n_fields"]["mean"]
        rows.append([cfgk, bc["training_runs"], lvl, f"{n:.0f}", f("recall") if "recall" in x else "--", f("domain_gm"), f("phase_gm"), f("total_gm"),
                     f"{x['domain_med_abs_log']['mean']:.3f}", f"{x['phase_med_abs_log']['mean']:.3f}", f"{x['total_med_abs_log']['mean']:.3f}"])
table("Table 7 — mass-error decomposition (columns: training runs / N matched cells (or fields) kept apart)",
      ["Config", "Training runs", "Level.group", "N (matched cells or fields)", "Recall", "Domain GM", "Phase GM", "Total GM", "|log| domain", "|log| phase", "|log| total"], rows)
out.append("\nSource: `decomposition/decomposition.json::by_config` (recomputed from `runs/diagnostics/decomposition_{cells,fields}.csv`; agrees with `decomposition_summary.csv`). Derived.")

bd = J("boundary_sensitivity/boundary.json")["table_8"]
rows = []
for k, v in bd.items():
    g = lambda m, dd=3: f"{v[m]['mean']:.{dd}f}" + (f" ± {v[m]['sd']:.{dd}f}" if v[m]["sd"] is not None else "")
    rows.append([v["label"], v["training_runs"], g("edge_recall"), g("interior_recall"), g("edge_mape"), g("interior_mape"), g("fp_all", 0), g("fp_near_boundary", 0)])
table("Table 8 — edge / interior stratification, false positives (all, within 15 px of the border)",
      ["Configuration", "Training runs", "Edge recall", "Interior recall", "Edge dry-mass MAPE", "Interior dry-mass MAPE", "FP all", "FP near boundary"], rows)

ab = J("ipp/ipp.json")["table_9"]
rows = []
for k, v in ab.items():
    if k == "W10_sweep_point":
        continue
    rows.append([v["label"], v["training_runs"]] + [fmt(v[m]) if v["training_runs"] >= 3 else f"{v[m]['value']:.4f}" for m in
                                                       ("dry_mass_mape", "dry_mass_mape_coverage_adjusted", "dry_mass_field_total_mape", "area_mape", "detection_recall", "detection_precision", "seg_dice", "phase_mae_rad")])
table("Table 9 — ablation (mean ± SD for 3 training runs; point estimate, no SD, for 1 run)",
      ["Configuration", "Training runs", "Matched MAPE", "Cov.-adj. MAPE", "Field-total MAPE", "Area MAPE", "Recall", "Precision", "Dice", "Phase MAE [rad]"], rows)

rows = []
for r in J("ipp/ipp.json")["table_10"]:
    if r["metric"] in ("dry_mass_mape", "dry_mass_field_total_mape"):
        rows.append([r["configuration"], r["comparator"], r["metric"], f"{r['delta']:+.4f}", f"{r['pooled_sd']:.4f}" if r["pooled_sd"] is not None else "n/e",
                     f"{r['two_x_pooled_sd']:.4f}" if r["two_x_pooled_sd"] is not None else "n/e", f"{r['n_runs_config']}/{r['n_runs_comparator']}", str(r["criterion_evaluable"]), r["assessment"]])
table("Table 10 — resolution quantities (n/e = not estimable)",
      ["Configuration", "Comparator", "Metric", "Δ (config − comparator)", "Pooled SD", "2× pooled SD", "Runs (config/comparator)", "Evaluable", "Assessment"], rows)

a = J("amplitude/amplitude.json")
rows = []
for k in ("amplitude_mae", "amplitude_unity_mae", "ratio_mae_over_unity_mae", "amplitude_mae_in_cell", "amplitude_bias", "amplitude_pearson_r", "amplitude_pred_mean", "amplitude_pred_sd", "amplitude_reference_mean"):
    rows.append([k] + [fmt(a["table_11a"][x][k]) for x in ("D0", "D1", "D2")])
table("Table 11a — amplitude output (1 training run each, 113 test fields)", ["Quantity", "+Amplitude", "+Fwd (fixed z)", "+Fwd (free z)"], rows)
rows = []
for k in ("forward_residual", "forward_residual_reference", "forward_residual_ratio", "phase_mae_rad_in_cell"):
    rows.append([k] + [fmt(a["table_11b"][x][k]) for x in ("A", "B", "D0", "D1", "D2")])
table("Table 11b — forward-model residual (A, B: mean ± SD of 3 runs)", ["Quantity", "Baseline", "+IPP (per-cell)", "+Amplitude", "+Fwd (fixed z)", "+Fwd (free z)"], rows)
rows = []
for k, v in a["table_11c"].items():
    s9, s5 = v["scale_0.9"], v["scale_0.5"]
    rows.append([k, fmt(v["reference_residual"]), fmt(s9["margin"], 5), f"{s9['fields_favouring_reference_phase']['value']}/{s9['of']}",
                 fmt(s5["margin"], 4), f"{s5['fields_favouring_reference_phase']['value']}/{s5['of']}"])
table("Table 11c — phase-scale probes; 'Fields favouring reference phase' = fields where the scaled phase gives the HIGHER residual",
      ["Geometry and fields", "Residual (reference phase)", "Margin 0.9×", "Fields favouring reference 0.9×", "Margin 0.5×", "Fields favouring reference 0.5×"], rows)
ms = ROOT / "analysis/v42/MANUSCRIPT_SYNC.md"
head = ms.read_text().split("\n### Table 3")[0]      # the written change list is kept; the tables are regenerated
ms.write_text(head + "\n" + "\n".join(out) + "\n")
print("regenerated the tables in analysis/v42/MANUSCRIPT_SYNC.md")
