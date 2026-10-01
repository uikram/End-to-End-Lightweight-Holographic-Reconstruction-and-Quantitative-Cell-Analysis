#!/usr/bin/env python
"""Bring HoloQPI_4.2/table_*.tex in line with results_for_manuscript/ (idempotent).

* Table 3: "Training runs (seeds)" column from metadata/configurations.json
* Table 4: descriptive matched-cell difference
* Table 5: recovered in-cell phase contrast (inline.json)
* Table 6: N/A for circularity field totals
* Table 7: N (cells/fields) kept apart from n (training runs)
* Table 10(b): Comparison | Delta | 2x pooled SD | Assessment (ipp.json::table_10)
* Table 11(c): "Fields favouring reference phase" with counts (amplitude.json::table_11c)

Numbers are formatted from the JSON files, not typed. Run from anywhere:
    python analysis/v42/update_tables.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "HoloQPI_4.2"
R = ROOT / "results_for_manuscript"
J = lambda p: json.loads((R / p).read_text())


def edit(name, pairs):
    p = MS / name
    t = p.read_text()
    for old, new in pairs:
        if new in t:
            continue
        assert old in t, f"{name}: pattern not found: {old[:70]!r}"
        t = t.replace(old, new)
    p.write_text(t)


cfg = J("metadata/configurations.json")
inline = J("inline/inline.json")["table_5"]["recovered_in_cell_phase_contrast"]
base = J("baseline/baseline.json")
cm = base["table_4"]["cells_matched"]
assert cm["descriptive_difference_text"] == "+20 \u00b1 9"

# ---- Table 3
seeds = {a: f"{c['training_runs']} ({', '.join(map(str, c['seeds']))})" for a, c in cfg.items() if isinstance(c, dict) and "training_runs" in c}
t3 = (MS / "table_3.tex").read_text()
if "Training runs (seeds)" not in t3:
    t3 = t3.replace("Seeds: number of training runs.", "Training runs (seeds): number of trained models and their seeds.")
    t3 = t3.replace("p{3.6cm}c@{}}", "p{3.6cm}>{\\centering\\arraybackslash}p{2.7cm}@{}}")
    t3 = t3.replace("Comparator & Seeds \\\\", "Comparator & Training runs (seeds) \\\\")
    rows = t3.split("\n")
    out = []
    order = [("End-to-End Neural Baseline & off-axis", "A"), ("In-Line Neural Configuration", "G"), ("+IPP (per-cell) & $+", "B"),
             ("+IPP (image)", "B1"), ("+Area", "B2"), ("+BGA", "C"), ("+IPP (per-cell), $w=0.1/0.3/3.0$", None),
             ("+Amplitude", "D0"), ("+Fwd (fixed", "D1"), ("+Fwd (free", "D2"), ("Compact Baseline", "KA"), ("Compact +IPP", "KB")]
    for row in rows:
        for key, arm in order:
            if row.startswith(key) and row.rstrip().endswith("\\\\"):
                val = (seeds[arm] if arm else "1 (42)")
                row = re.sub(r"& [0-9]+ \\\\$", f"& {val} \\\\\\\\", row.rstrip())
                break
        out.append(row)
    (MS / "table_3.tex").write_text("\n".join(out))

# ---- Table 4
edit("table_4.tex", [
    ("Cells matched (of 3186) & 1870 & $1890 \\pm 9$ & -- \\\\",
     "Cells matched (of 3186) & 1870 & $1890 \\pm 9$ & $+20 \\pm 9^{\\dagger}$ \\\\"),
    ("\\bottomrule\n\\end{tabular*}\n\\end{table}",
     "\\bottomrule\n\\end{tabular*}\n\\tabnote{$^{\\dagger}$Descriptive difference of the matched-cell count (neural mean minus classical; "
     "the SD is the between-seed SD of the neural count). A matched-cell count is not an error metric, and the resolution criterion of "
     "Sec.~\\ref{sec:stats} is not applied to it. The classical pipeline is deterministic, so no difference in this table is assessed "
     "against a between-seed spread.}\n\\end{table}"),
])

# ---- Table 5
f3 = lambda e: f"{e['value']:+.3f}"
neu = inline["off_axis_neural"]
edit("table_5.tex", [
    ("Recovered in-cell phase contrast [rad] & -- & -- & $+0.922$ & $-0.084$ \\\\",
     f"Recovered in-cell phase contrast [rad]$^{{\\ddagger}}$ & ${f3(inline['in_line_neural'])}$ & ${neu['value']:+.3f} \\pm {neu['sd']:.3f}$ & "
     f"${f3(inline['classical_off_axis'])}$ & ${f3(inline['classical_in_line'])}$ \\\\"),
    ("\\tabnote{$^{\\ast}$Computed over the 106 cells matched by the classical in-line pipeline; not comparable with the other columns.}",
     "\\tabnote{$^{\\ast}$Computed over the 106 cells matched by the classical in-line pipeline; not comparable with the other columns. "
     "$^{\\ddagger}$Median over the 107 fields of the mean phase inside the reference cell masks minus the mean phase outside them, "
     "computed identically for all four columns; the reference phase gives "
     f"${inline['reference_median_contrast_rad']:+.3f}$\\,rad. The differences between the single in-line run and the three off-axis runs "
     "are not resolvable under the criterion of Sec.~\\ref{sec:stats}.}"),
])

edit("table_5.tex", [("classical pipelines: one deterministic run each.}", "classical pipelines: one deterministic run each. $n$: number of training runs.}")])

# ---- Table 6
edit("table_6.tex", [
    ("Circularity & $+0.0217 \\pm 0.0057$ & $[-0.054, +0.097]$ & -- & -- & -- \\\\",
     "Circularity & $+0.0217 \\pm 0.0057$ & $[-0.054, +0.097]$ & N/A & N/A & N/A \\\\"),
    ("run. Limits of agreement: means over seeds.", "run. Limits of agreement: means over seeds. Field totals are not defined for circularity, which is non-additive."),
])

# ---- Table 7
edit("table_7.tex", [
    ("Level / group & $n$ & Recall", "Level / group & $N$ & Recall"),
    ("\\quad Field (foreground) & 113 fields & -- &", "\\quad Field (foreground) & 113 fields & N/A &"),
    ("\\tabnote{$n$: matched cells per run (baseline: mean over seeds) or fields.",
     "\\tabnote{$N$: matched cells per run (baseline: mean over seeds) or fields; $n$ in the group labels is the number of training runs."),
])

# ---- Table 10(b): regenerated from ipp.json
rows = {(r["configuration"], r["comparator"], r["metric"]): r for r in J("ipp/ipp.json")["table_10"]}
ORDER = [("+IPP (per-cell)", "End-to-End Neural Baseline", "+IPP (per-cell) vs Baseline"),
         ("+IPP (image)", "End-to-End Neural Baseline", "+IPP (image) vs Baseline"),
         ("+IPP (image)", "+IPP (per-cell)", "+IPP (image) vs +IPP (per-cell)"),
         ("+Area", "+IPP (per-cell)", "+Area vs +IPP (per-cell)"), ("+BGA", "+IPP (per-cell)", "+BGA vs +IPP (per-cell)"),
         ("+IPP (per-cell), w=0.1", "End-to-End Neural Baseline", "$w=0.1$ vs Baseline"),
         ("+IPP (per-cell), w=0.3", "End-to-End Neural Baseline", "$w=0.3$ vs Baseline"),
         ("+IPP (per-cell), w=3.0", "End-to-End Neural Baseline", "$w=3.0$ vs Baseline"),
         ("+Amplitude", "+IPP (per-cell)", "+Amplitude vs +IPP (per-cell)"),
         ("+Fwd (fixed z)", "+Amplitude", "+Fwd (fixed $z$) vs +Amplitude"),
         ("+Fwd (free z)", "+Fwd (fixed z)", "+Fwd (free $z$) vs +Fwd (fixed $z$)"),
         ("Compact Baseline", "End-to-End Neural Baseline", "Compact Baseline vs Baseline"),
         ("Compact +IPP", "+IPP (per-cell)", "Compact +IPP vs +IPP (per-cell)")]
TXT = {"Exceeds 2x pooled SD": "Exceeds $2\\times$ pooled SD", "Within 2x pooled SD": "Within $2\\times$ pooled SD",
       "Not estimable from the available runs": "Not estimable from the available runs"}


def block(metric):
    out = []
    first_single = False
    for conf, comp, name in ORDER:
        r = rows[(conf, comp, metric)]
        if r["n_runs_config"] == 1 and not first_single:
            out.append("\\midrule\n\\multicolumn{4}{@{}l}{\\emph{At least one configuration with a single training run}} \\\\")
            first_single = True
        d = f"{r['delta']:+.4f}"
        if r["assessment"].startswith("Exceeds"):
            d = f"\\mathbf{{{d}}}"
        two = f"{r['two_x_pooled_sd']:.4f}" if r["two_x_pooled_sd"] is not None else "N/A"
        out.append(f"{name} & ${d}$ & {two} & {TXT[r['assessment']]} \\\\")
    return "\n".join(out)


t10 = (MS / "table_10.tex").read_text()
head = t10[:t10.index("\\vspace{8pt}\n\\textbf{(b) Differences}")]
head = head.replace("(b) Differences in matched-cell and\nfield-total dry-mass MAPE, configuration minus comparator, with the resolution\nthreshold $2\\times$ the pooled between-seed SD.",
                    "(b) Differences in matched-cell and\nfield-total dry-mass MAPE, configuration minus comparator, with $2\\times$ the pooled between-seed SD\nand the assessment.")
new = head + """\\vspace{8pt}
\\textbf{(b) Differences}\\\\[2pt]
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrl@{}}
\\toprule
Comparison & $\\Delta$ & $2\\times$ pooled SD & Assessment \\\\
\\midrule
\\multicolumn{4}{@{}l}{\\emph{Matched-cell dry-mass MAPE; both configurations with three training runs}} \\\\
""" + "\n".join(block("dry_mass_mape").split("\n")[:3]) + """
""" + "\n".join(block("dry_mass_mape").split("\n")[3:]) + """
\\midrule
\\multicolumn{4}{@{}l}{\\emph{Field-total dry-mass MAPE; both configurations with three training runs}} \\\\
""" + block("dry_mass_field_total_mape") + """
\\bottomrule
\\end{tabular*}
\\tabnote{$\\Delta$: configuration minus comparator; positive is worse. Bold: $|\\Delta| > 2\\sqrt{(s_1^2+s_2^2)/2}$ with $s$ the
between-seed SD of each configuration, both configurations having at least three training runs. Not estimable: fewer than three
training runs on one side, so no between-seed SD exists; no other variance is substituted. Differences are computed from unrounded means.}
\\end{table}
"""
(MS / "table_10.tex").write_text(new)

# ---- Table 11(c): regenerated from amplitude.json
c = J("amplitude/amplitude.json")["table_11c"]
NAMES = [("Off-axis, 32 validation fields", "Off-axis, 32 validation fields"),
         ("In-line, 32 validation fields", "In-line, 32 validation fields"),
         ("Off-axis, 113 test fields, global surface", "Off-axis, 113 test fields, global surface"),
         ("Off-axis, 112 test fields, per-field surfaces", "Off-axis, 112 test fields, per-field surfaces$^{\\ast}$")]
rowsc = []
for key, lab in NAMES:
    v = c[key]
    s9, s5 = v["scale_0.9"], v["scale_0.5"]
    rowsc.append(f"{lab} & {v['reference_residual']['value']:.4f} & ${s9['margin']['value']:+.5f}$ & "
                 f"{s9['fields_favouring_reference_phase']['value']}/{s9['of']} & ${s5['margin']['value']:+.4f}$ & "
                 f"{s5['fields_favouring_reference_phase']['value']}/{s5['of']} \\\\")
same = c["Off-axis, 112 test fields, global surface (same fields as per-field row)"]
t11 = (MS / "table_11.tex").read_text()
i = t11.index("\\textbf{(c) Phase-scale probes}")
j = t11.index("\\end{table}")
new11 = t11[:i] + """\\textbf{(c) Phase-scale probes}\\\\[2pt]
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l>{\\raggedleft\\arraybackslash}p{1.5cm}rp{2.4cm}rp{2.4cm}@{}}
\\toprule
 & & \\multicolumn{2}{c}{$0.9\\times$ phase} & \\multicolumn{2}{c}{$0.5\\times$ phase} \\\\
\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}
Geometry and fields & Residual & Margin & Fields favouring reference phase & Margin & Fields favouring reference phase \\\\
\\midrule
""" + "\n".join(rowsc) + """
\\bottomrule
\\end{tabular*}
\\tabnote{Fields favouring reference phase: number of fields on which the scaled phase gives the higher residual than the reference
phase (a count of fields, not a separate test). Margin: scaled minus reference residual, averaged over the fields. Configured tolerance for
a usable margin: 0.01. Residual levels in (b) for configurations with an amplitude output use the predicted amplitude in place of
$A=1$ and are not comparable with the others; only the ratio is. $^{\\ast}$Per-field aberration surfaces are fitted against each
field's reference phase and are not available at inference. One test field (T24\\_Staurosporine\\_100nM\\_10) has
no per-field surface: its fit was rejected because the fitted surface exceeded the 45\\,rad peak-to-valley limit (405\\,rad).
With the global surface on the same 112 fields the residual is """ + f"{same['reference_residual']['value']:.4f}" + """, the margins are
""" + f"${same['scale_0.9']['margin']['value']:+.5f}$ and ${same['scale_0.5']['margin']['value']:+.4f}$ and the counts {same['scale_0.9']['fields_favouring_reference_phase']['value']}/{same['scale_0.9']['of']} and {same['scale_0.5']['fields_favouring_reference_phase']['value']}/{same['scale_0.5']['of']}." + "}\n" + t11[j:]
(MS / "table_11.tex").write_text(new11)
print("tables updated from results_for_manuscript/")
