"""Write the HoloQPI_4.2 tables from analysis/v42/values.json.

    python analysis/v42/extract_values.py      # first
    python analysis/v42/make_tables_v42.py --out "Claude outputs/HoloQPI_4.2"

Every number printed in a table is looked up in values.json by key (or computed
from keys that are named in the call) and recorded in
analysis/v42/table_cells.csv: table file, printed string, key or formula.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "values.json").read_text())
CELLS: list[dict] = []
CUR = {"table": None}


def v(key):
    return V[key]["value"]


def rec(s, key, formula=None):
    CELLS.append({"table": CUR["table"], "printed": s, "key": key, "formula": formula or ""})
    return s


def f(key, d=4, sign=False, pct=False, scale=1.0, math_mode=True):
    x = v(key) * (100 if pct else 1) * scale
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace("-", "-")
    rec(s, key, "x100" if pct else None)
    return f"${s}$" if (math_mode and (sign or x < 0)) else s


def fpm(mkey, skey, d=4, sign=False):
    m = f(mkey, d, sign, math_mode=False)
    s = f(skey, d, math_mode=False)
    return f"${m} \\pm {s}$"


def fint(key):
    s = f"{v(key):.0f}"
    rec(s, key)
    return s


def fdiff(k1, k2, d=4):
    x = v(k1) - v(k2)
    s = f"{x:+.{d}f}"
    rec(s, f"{k1} - {k2}", "difference")
    return f"${s}$"


# File names follow the printed table numbers of main.tex (order of appearance).
RENAME = {"table_12.tex": "table_5.tex", "table_7.tex": "table_6.tex", "table_13.tex": "table_7.tex",
          "table_11.tex": "table_8.tex", "table_5.tex": "table_9.tex", "table_6.tex": "table_10.tex",
          "table_8.tex": "table_11.tex", "table_9.tex": "table_12.tex", "table_10.tex": "table_13.tex"}


def write(out: Path, name: str, body: str):
    final = RENAME.get(name, name)
    for c in CELLS:
        if c["table"] == name:
            c["table"] = final
    (out / final).write_text(body.replace(f"% Table {name[6:-4]}:", f"% Table {final[6:-4]}:", 1))
    print(f"  wrote {final}")


# ------------------------------------------------------------------ tables
def table_2(out):
    CUR["table"] = "table_2.tex"
    a_tr, a_va, a_te = fint("split.A.train"), fint("split.A.val"), fint("split.A.test")
    g_tr, g_va, g_te = fint("split.G.train"), fint("split.G.val"), fint("split.G.test")
    cells, gcells = fint("split.A.test_cells"), fint("split.G.test_cells")
    dy, dx = fint("config.crop_offset_dy"), fint("config.crop_offset_dx")
    body = rf"""% Table 2: acquisition, calibration, split and training settings.
\begin{{table}}[!htbp]
\centering
\caption{{Acquisition, calibration, data split and training settings.}}
\label{{tab:setup}}
\footnotesize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}ll@{{}}}}
\toprule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Acquisition and calibration}}}} \\
\midrule
Hologram                    & $1024\times1024$, 8-bit TIFF \\
Quantitative phase (reference) & $900\times900$, float32, radians \\
Hologram-to-phase mapping   & crop $1024\rightarrow900$, origin $({dy}, {dx})$\,px from the centred crop \\
Illumination wavelength $\lambda$ & $0.666\,\mu$m \\
Pixel pitch $dx=dy$         & $0.284871\,\mu$m \\
Refraction increment $\alpha$ & $0.2\,$mL/g \\
Propagation distance $z$    & $33.77\,\mu$m (supplied) \\
Aberration surface (forward model) & order-5 polynomial, median over training fields \\
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Splits}}}} \\
\midrule
Off-axis fields (train / val / test) & {a_tr} / {a_va} / {a_te} \\
In-line fields (train / val / test)  & {g_tr} / {g_va} / {g_te} (SNU\_01--SNU\_50 excluded) \\
Stratification              & cell line $\times$ drug condition \\
Reference cells in test     & {cells} off-axis; {gcells} in-line \\
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Network}}}} \\
\midrule
Encoder                     & MobileNetV2, ImageNet-initialised \\
Decoders                    & two U-Net decoders, 256/128/64/32 \\
Heads                       & phase, amplitude (optional), segmentation (2 classes) \\
Network input               & raw hologram, z-scored per field \\
Output grid                 & stride 2, bilinearly upsampled \\
Parameters, standard        & 9\,598\,099 (45.85\,GMAC) \\
Parameters, with amplitude  & 9\,607\,412 (47.87\,GMAC) \\
Parameters, compact         & 3\,360\,403 (24.03\,GMAC) \\
\midrule
\multicolumn{{2}}{{@{{}}l}}{{\textbf{{Optimisation}}}} \\
\midrule
Epochs                      & 60 \\
Optimiser                   & AdamW, $\beta=(0.9,0.999)$ \\
Learning rate               & $3\times10^{{-4}}$, 2 warm-up epochs, cosine to $10^{{-6}}$ \\
Learning rate for $z$ (+Fwd, free $z$)& $100\times$ base \\
Weight decay                & $1\times10^{{-4}}$ \\
Batch size / train crop     & 4 / $512\times512$ \\
Precision                   & mixed, FP16 \\
Gradient clipping           & 1.0 (global norm) \\
Augmentation                & flips, $90^\circ$ rotations \\
Checkpoint criterion        & best validation composite (Sec.~\ref{{sec:stats}}), every epoch, no early stopping \\
Seeds                       & 42 (primary), 1337, 2024 \\
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    for k, s in (("hw.A.mean.params_total", "9598099"), ("hw.D0.mean.params_total", "9607412"),
                 ("hw.KA.mean.params_total", "3360403")):
        assert f"{v(k):.0f}" == s, k
        rec(s, k)
    for k, s in (("hw.A.mean.gmacs", "45.85"), ("hw.D0.mean.gmacs", "47.87"), ("hw.KA.mean.gmacs", "24.03")):
        assert f"{v(k):.2f}" == s, k
        rec(s, k)
    write(out, "table_2.tex", body)


def table_3(out):
    CUR["table"] = "table_3.tex"
    body = r"""% Table 3: experimental configurations and the single change each makes.
\begin{table}[!htbp]
\centering
\caption{Experimental configurations. Each differs from its comparator by one
change; weights not listed are zero. Seeds: number of training runs.}
\label{tab:arms}
\footnotesize
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}p{3.6cm}c@{}}
\toprule
Configuration & Change relative to comparator & Comparator & Seeds \\
\midrule
\multicolumn{4}{@{}l}{\emph{Main pipeline and hologram mode}} \\
End-to-End Neural Baseline & off-axis; reconstruction + segmentation losses only & -- & 3 \\
In-Line Neural Configuration & the baseline's objective on in-line Gabor holograms; SNU\_01--SNU\_50 excluded & End-to-End Neural Baseline & 1 \\
\midrule
\multicolumn{4}{@{}l}{\emph{Measurement-aware objectives}} \\
+IPP (per-cell) & $+\,\mathcal{L}_\mathrm{IPP}^\mathrm{cell}$, $w=1.0$ & End-to-End Neural Baseline & 3 \\
+IPP (image) & $+\,\mathcal{L}_\mathrm{IPP}^\mathrm{img}$, $w=1.0$ & End-to-End Neural Baseline & 3 \\
+Area & $+\,\mathcal{L}_\mathrm{area}$, per-cell, $w=1.0$ & +IPP (per-cell) & 1 \\
+BGA & $+\,\mathcal{L}_\mathrm{BGA}$, $w=0.05$ & +IPP (per-cell) & 1 \\
+IPP (per-cell), $w=0.1/0.3/3.0$ & $\mathcal{L}_\mathrm{IPP}^\mathrm{cell}$ weight $0.1$, $0.3$, $3.0$ & End-to-End Neural Baseline & 1 \\
\midrule
\multicolumn{4}{@{}l}{\emph{Amplitude output and forward-model consistency}} \\
+Amplitude & $+$ amplitude head, $+\,\mathcal{L}_\mathrm{amp}$, $w=0.1$ & +IPP (per-cell) & 1 \\
+Fwd (fixed $z$) & $+\,\mathcal{L}_\mathrm{fwd}$, $w=0.02$, $z$ fixed & +Amplitude & 1 \\
+Fwd (free $z$) & $z$ becomes a trainable parameter & +Fwd (fixed $z$) & 1 \\
\midrule
\multicolumn{4}{@{}l}{\emph{Model capacity}} \\
Compact Baseline & shared $1\times1$ projection to 256 channels before the decoders (3.36\,M parameters) & End-to-End Neural Baseline & 1 \\
Compact +IPP & compact decoder, with $\mathcal{L}_\mathrm{IPP}^\mathrm{cell}$ & +IPP (per-cell) & 1 \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(out, "table_3.tex", body)


def table_4(out):
    """Off-axis neural vs classical, 113 fields."""
    CUR["table"] = "table_4.tex"
    rows = [
        ("Phase MAE [rad] $\\downarrow$", "phase_mae_rad"),
        ("Phase MAE inside cells [rad] $\\downarrow$", "phase_mae_rad_in_cell"),
        ("Phase Pearson $r$ $\\uparrow$", "phase_pearson_r"),
        ("Phase SSIM $\\uparrow$", "phase_ssim"),
        None,
        ("Dice $\\uparrow$", "seg_dice"),
        ("AJI $\\uparrow$", "seg_aji"),
        ("Boundary F1 $\\uparrow$", "seg_boundary_f1"),
        ("Detection recall $\\uparrow$", "detection_recall"),
        ("Detection precision $\\uparrow$", "detection_precision"),
        None,
        ("Projected-area MAPE $\\downarrow$", "area_mape"),
        ("Dry-mass MAPE $\\downarrow$", "dry_mass_mape"),
        ("Dry-mass Pearson $r$, per cell $\\uparrow$", "dry_mass_cell_pearson_r"),
        ("Field-total dry-mass MAPE $\\downarrow$", "dry_mass_field_total_mape"),
    ]
    lines = []
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        lab, k = r
        lines.append(f"{lab} & {f(f'classical113.off_axis.{k}')} & {fpm(f'mean.A.{k}', f'sd.A.{k}')} & "
                     f"{fdiff(f'mean.A.{k}', f'classical113.off_axis.{k}')} \\\\")
    lines.append(f"Cells matched (of {fint('split.A.test_cells')}) & {fint('classical113.off_axis.cells_matched')} & "
                 f"${f('mean.A.cells_matched', 0, math_mode=False)} \\pm {f('sd.A.cells_matched', 0, math_mode=False)}$ & -- \\\\")
    body = r"""% Table 4: End-to-End Neural Baseline against the tested classical pipeline, off-axis, 113 test fields.
\begin{table}[!htbp]
\centering
\caption{End-to-End Neural Baseline and the tested classical pipeline on the
off-axis test set (113 fields, 3186 reference cells). Neural values are mean
$\pm$ SD over three training seeds; the classical pipeline is deterministic and
was run once. $\Delta$: neural minus classical.}
\label{tab:learned-vs-classical}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Quantity & Classical & Neural ($n=3$) & $\Delta$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    write(out, "table_4.tex", body)


def table_12(out):
    """Common 107 fields: both geometries, neural and classical."""
    CUR["table"] = "table_12.tex"
    cols = [("G", "common.G_s42", "common_run.G.s42"), ("A", "common.A_mean", None),
            ("Coff", "common.classical_off_axis", "classical107.off_axis"),
            ("Cin", "common.classical_gabor", "classical107.gabor")]
    rows = [("Phase MAE [rad] $\\downarrow$", "phase_mae_rad"),
            ("Phase MAE inside cells [rad] $\\downarrow$", "phase_mae_rad_in_cell"),
            ("Phase Pearson $r$ $\\uparrow$", "phase_pearson_r"),
            ("Phase SSIM $\\uparrow$", "phase_ssim"), None,
            ("Dice $\\uparrow$", "seg_dice"), ("AJI $\\uparrow$", "seg_aji"),
            ("Boundary F1 $\\uparrow$", "seg_boundary_f1"),
            ("Detection recall $\\uparrow$", "detection_recall"),
            ("Detection precision $\\uparrow$", "detection_precision"), None,
            ("Projected-area MAPE $\\downarrow$", "area_mape"),
            ("Dry-mass MAPE $\\downarrow$", "dry_mass_mape"),
            ("Field-total dry-mass MAPE $\\downarrow$", "dry_mass_field_total_mape")]
    lines = [f"Recovered in-cell phase contrast [rad] & -- & -- & {f('classical107.off_axis.recovered_phase_contrast_rad', 3, sign=True)} & "
             f"{f('classical107.gabor.recovered_phase_contrast_rad', 3, sign=True)} \\\\", r"\midrule"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        lab, k = r
        cells = [f(f"common.G_s42.{k}"), fpm(f"common.A_mean.{k}", f"common.A_sd.{k}"),
                 f(f"common.classical_off_axis.{k}"), f(f"common.classical_gabor.{k}")]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(f"Dry-mass Pearson $r$, per cell $\\uparrow$ & {f('common_run.G.s42.dry_mass_cell_pearson_r')} & "
                 f"{fpm('common_mean.A.dry_mass_cell_pearson_r', 'common_sd.A.dry_mass_cell_pearson_r')} & "
                 f"{f('classical107.off_axis.dry_mass_cell_pearson_r')} & {f('classical107.gabor.dry_mass_cell_pearson_r')}$^{{\\ast}}$ \\\\")
    lines.append(f"Cells matched (of {fint('split.G.test_cells')}) & {fint('common_run.G.s42.cells_matched')} & "
                 f"${f('common_mean.A.cells_matched', 0, math_mode=False)} \\pm {f('common_sd.A.cells_matched', 0, math_mode=False)}$ & "
                 f"{fint('classical107.off_axis.cells_matched')} & {fint('classical107.gabor.cells_matched')} \\\\")
    body = r"""% Table 12: both geometries on the 107 test fields whose in-line hologram is correctly paired.
\begin{table}[!htbp]
\centering
\caption{Off-axis and in-line pipelines on the 107 test fields common to both
geometries (3055 reference cells; SNU\_01--SNU\_50 excluded). In-line neural:
one training run; off-axis neural: mean $\pm$ SD over three seeds, the same
models as in Table~\ref{tab:learned-vs-classical} re-scored on these fields;
classical pipelines: one deterministic run each.}
\label{tab:common}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{Neural} & \multicolumn{2}{c}{Classical} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Quantity & In-line ($n=1$) & Off-axis ($n=3$) & Off-axis & In-line \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{$^{\ast}$Computed over the """ + fint("classical107.gabor.cells_matched") + r""" cells matched by the classical in-line pipeline; not comparable with the other columns.}
\end{table}
"""
    write(out, "table_12.tex", body)


def table_7(out):
    CUR["table"] = "table_7.tex"
    a_rows = []
    for lab, q in (("Projected area", "area"), ("Circularity", "circularity"), ("Dry mass", "dry_mass")):
        a_rows.append(f"{lab} & {fpm(f'mean.A.{q}_mape', f'sd.A.{q}_mape')} & "
                      f"{f(f'run.A.s42.{q}_mape_ci_lower', math_mode=False)}--{f(f'run.A.s42.{q}_mape_ci_upper', math_mode=False)} & "
                      f"{fpm(f'mean.A.{q}_mape_coverage_adjusted', f'sd.A.{q}_mape_coverage_adjusted')} & "
                      f"{fpm(f'mean.A.{q}_cell_pearson_r', f'sd.A.{q}_cell_pearson_r')} \\\\")
    b_rows = []
    for lab, q in (("Projected area", "area"), ("Circularity", "circularity"), ("Dry mass", "dry_mass")):
        lo = f(f"mean.A.{q}_loa_lower", 3, sign=True, math_mode=False)
        hi = f(f"mean.A.{q}_loa_upper", 3, sign=True, math_mode=False)
        if q == "circularity":
            ft = "-- & -- & --"
        else:
            ft = (f"{fpm(f'mean.A.{q}_field_total_bias', f'sd.A.{q}_field_total_bias', sign=True)} & "
                  f"{fpm(f'mean.A.{q}_field_total_mape', f'sd.A.{q}_field_total_mape')} & "
                  f"{fpm(f'mean.A.{q}_field_total_pearson_r', f'sd.A.{q}_field_total_pearson_r')}")
        b_rows.append(f"{lab} & {fpm(f'mean.A.{q}_relative_bias', f'sd.A.{q}_relative_bias', sign=True)} & "
                      f"$[{lo}, {hi}]$ & {ft} \\\\")
    body = r"""% Table 7: per-cell and per-field measurement agreement of the End-to-End Neural Baseline.
\begin{table}[!htbp]
\centering
\caption{Measurement agreement of the End-to-End Neural Baseline on the
off-axis test set (113 fields, 3186 reference cells). Mean $\pm$ SD over three
training seeds unless stated. (a) Matched cells. (b) Relative Bland--Altman
statistics on per-field medians of matched cells, and field totals over all
predicted and all reference cells.}
\label{tab:measurement}
\footnotesize
\textbf{(a) Matched cells}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
Quantity & MAPE & 95\% CI (seed 42) & Cov.-adj. MAPE & Pearson $r$ \\
\midrule
""" + "\n".join(a_rows) + r"""
\bottomrule
\end{tabular*}

\vspace{8pt}
\textbf{(b) Per field and field totals}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
 & \multicolumn{2}{c}{Bland--Altman (per field)} & \multicolumn{3}{c}{Field total} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}
Quantity & Bias & Limits of agreement & Bias & MAPE & Pearson $r$ \\
\midrule
""" + "\n".join(b_rows) + r"""
\bottomrule
\end{tabular*}
\tabnote{CI: percentile bootstrap over fields (2000 resamples) of the seed-42
run. Limits of agreement: means over seeds. Dry mass and integrated phase $S_k$
have identical relative statistics (Sec.~\ref{sec:eval}).}
\end{table}
"""
    write(out, "table_7.tex", body)


def table_13(out):
    """Mass-error decomposition."""
    CUR["table"] = "table_13.tex"
    lines = []
    for cfg, lab in (("A", "End-to-End Neural Baseline ($n=3$)"), ("classical", "Classical pipeline ($n=1$)")):
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        for lvl, grp, glab in (("cell", "all", "Matched cells, all"), ("cell", "interior", "Matched cells, interior"),
                               ("cell", "edge", "Matched cells, edge"), ("field", "all", "Field (foreground)")):
            base = f"decomp.{cfg}.{lvl}.{grp}"
            if lvl == "cell":
                if cfg == "A":
                    nkey = [f"decomp_run.{r}.cell.{grp}.n" for r in ("A_s42", "A_s1337", "A_s2024")]
                    nval = sum(v(k) for k in nkey) / 3
                    n = f"{nval:.0f}"
                    rec(n, " + ".join(nkey), "mean of matched cells per run")
                else:
                    n = fint(f"decomp_run.classical.cell.{grp}.n")
                rc = f(f"{base}.recall_mean", 3)
            else:
                n = "113 fields"
                rec("113", "split.A.test")
                rc = "--"
            gm = []
            for k in ("domain_gm", "phase_gm", "total_gm"):
                if cfg == "A":
                    gm.append(fpm(f"{base}.{k}_mean", f"{base}.{k}_sd", 3))
                else:
                    gm.append(f(f"{base}.{k}_mean", 3))
            sp = [f(f"{base}.{k}_mean", 3) for k in ("domain_med_abs_log", "phase_med_abs_log", "total_med_abs_log")]
            lines.append(f"\\quad {glab} & {n} & {rc} & " + " & ".join(gm + sp) + r" \\")
        if cfg == "A":
            lines.append(r"\midrule")
    body = r"""% Table 13: decomposition of the dry-mass ratio into a domain factor and a phase factor.
\begin{table}[!htbp]
\centering
\caption{Decomposition of the predicted-to-reference dry-mass ratio into a
domain factor and a phase factor (Eq.~\eqref{eq:decomp}) on the off-axis test
set (113 fields). GM: geometric mean of the factor (mean $\pm$ SD over three
seeds for the baseline); $|\log|$: median absolute log ratio (spread). Edge:
reference instance with a pixel within 3\,px of the field boundary.}
\label{tab:decomp}
\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrcccccccc@{}}
\toprule
 & & & \multicolumn{3}{c}{GM} & \multicolumn{3}{c}{$|\log|$} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Level / group & $n$ & Recall & Domain & Phase & Total & Domain & Phase & Total \\
\midrule
""".replace("lrcccccccc", "lrccccccc") + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{$n$: matched cells per run (baseline: mean over seeds) or fields.
Matched cells are recomputed by the decomposition script and differ from the
stored evaluation by at most three cells per run; the re-scoring gives a per-run
mean of 1889 matched cells, against $1890\pm9$ from the stored evaluation
(Table~\ref{tab:learned-vs-classical}). Field level: the whole
predicted and reference foreground, so undetected and spurious cells enter the
domain factor.}
\end{table}
"""
    write(out, "table_13.tex", body)


def table_11(out):
    CUR["table"] = "table_11.tex"
    lines = []
    for cfg, lab in (("A", "End-to-End Neural Baseline"), ("B", "+IPP (per-cell)"), ("B1", "+IPP (image)")):
        runs = [f"{cfg}_s{s}" for s in (42, 1337, 2024)]
        rec_e = fpm(f"decomp.{cfg}.cell.edge.recall_mean", f"decomp.{cfg}.cell.edge.recall_sd", 3)
        rec_i = fpm(f"decomp.{cfg}.cell.interior.recall_mean", f"decomp.{cfg}.cell.interior.recall_sd", 3)
        me = fpm(f"locmean.{cfg}.edge.mape", f"locsd.{cfg}.edge.mape", 3)
        mi = fpm(f"locmean.{cfg}.interior.mape", f"locsd.{cfg}.interior.mape", 3)
        fa = f"${f(f'locmean.{cfg}.fp_all', 0, math_mode=False)} \\pm {f(f'locsd.{cfg}.fp_all', 0, math_mode=False)}$"
        fn = f"${f(f'locmean.{cfg}.fp_near', 0, math_mode=False)} \\pm {f(f'locsd.{cfg}.fp_near', 0, math_mode=False)}$"
        lines.append(f"{lab} & {rec_e} & {rec_i} & {me} & {mi} & {fa} & {fn} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Classical pipeline ($n=1$) & {f('decomp.classical.cell.edge.recall_mean', 3)} & "
                 f"{f('decomp.classical.cell.interior.recall_mean', 3)} & {f('locmean.classical.edge.mape', 3)} & "
                 f"{f('locmean.classical.interior.mape', 3)} & {fint('loc.classical.fp_all')} & {fint('loc.classical.fp_near')} \\\\")
    body = r"""% Table 11: detection and matched-cell error by location of the reference instance.
\begin{table}[!htbp]
\centering
\caption{Detection recall, matched-cell dry-mass MAPE and false positives by
location of the reference instance on the off-axis test set (113 fields; """ + fint("refcells.edge") + r""" edge
and """ + f"{v('refcells.all') - v('refcells.edge'):.0f}" + r""" interior reference instances). Neural
configurations: mean $\pm$ SD over three seeds.}
\label{tab:edge}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Recall} & \multicolumn{2}{c}{Dry-mass MAPE} & \multicolumn{2}{c}{False positives per run} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Configuration & Edge & Interior & Edge & Interior & All & Within 15\,px \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{Edge: reference instance with a pixel within 3\,px of the field
boundary. Within 15\,px: false-positive centroid within 15\,px of the field
boundary. Recall and MAPE are recomputed by the decomposition script (matched
cells within three of the stored evaluation per run).}
\end{table}
"""
    rec(f"{v('refcells.all') - v('refcells.edge'):.0f}", "refcells.all - refcells.edge", "difference")
    write(out, "table_11.tex", body)


ABL = [("End-to-End Neural Baseline", "A"), None,
       ("+IPP (per-cell)", "B"), ("+IPP (image)", "B1"), ("+Area", "B2"), ("+BGA", "C"), None,
       ("+IPP (per-cell), $w=0.1$", "W01"), ("+IPP (per-cell), $w=0.3$", "W03"),
       ("+IPP (per-cell), $w=1.0$", "B"), ("+IPP (per-cell), $w=3.0$", "W30"), None,
       ("+Amplitude", "D0"), ("+Fwd (fixed $z$)", "D1"), ("+Fwd (free $z$)", "D2"), None,
       ("Compact Baseline", "KA"), ("Compact +IPP", "KB")]


def table_5(out):
    CUR["table"] = "table_5.tex"
    cols = ["dry_mass_mape", "dry_mass_mape_coverage_adjusted", "dry_mass_field_total_mape", "area_mape",
            "detection_recall", "detection_precision", "seg_dice", "phase_mae_rad"]
    lines = []
    for r in ABL:
        if r is None:
            lines.append(r"\midrule")
            continue
        lab, a = r
        n = fint(f"n.{a}.dry_mass_mape")
        lines.append(f"{lab} & {n} & " + " & ".join(f(f"mean.{a}.{c}") for c in cols) + r" \\")
    w10 = f("run.W10.s42.dry_mass_mape")
    b42 = f("run.B.s42.dry_mass_mape")
    body = r"""% Table 5: ablation over objective terms and decoder capacity (off-axis, 113 test fields).
\begin{table}[!htbp]
\centering
\caption{Ablation over objective terms and decoder capacity on the off-axis
test set (113 fields, 3186 reference cells). $n$: training runs; $n=3$
entries are means over seeds 42, 1337 and 2024, $n=1$ entries are seed-42 runs.}
\label{tab:ablation}
\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrrrrr@{}}
\toprule
 & & \multicolumn{3}{c}{Dry-mass MAPE $\downarrow$} & Area & Recall & Precision & Dice & Phase \\
\cmidrule(lr){3-5}
Configuration & $n$ & matched & cov.-adj. & field total & MAPE $\downarrow$ & $\uparrow$ & $\uparrow$ & $\uparrow$ & MAE [rad] $\downarrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{Differences from a single run ($n=1$) are not resolvable under the
between-seed criterion (Sec.~\ref{sec:stats}). Field totals sum all predicted
and all reference cells and are dominated by the coverage deficit: the
field-total dry-mass bias is negative in every configuration (Table~\ref{tab:seeds}a
for the replicated ones). The $w=1.0$ row repeats +IPP (per-cell); a separate
seed-42 training of the identical configuration gave a matched-cell MAPE of
""" + w10 + r""" (seed-42 run of +IPP (per-cell): """ + b42 + r""").}
\end{table}
"""
    # field-total bias range for the note
    xs = [v(f"mean.{a}.dry_mass_field_total_bias") for a in ("A", "B", "B1", "B2", "C", "D0", "D1", "D2", "W01", "W03", "W30", "KA", "KB")]
    assert max(xs) < 0, "field-total bias not negative everywhere"
    write(out, "table_5.tex", body)


def table_6(out):
    CUR["table"] = "table_6.tex"
    mets = [("Dry-mass MAPE", "dry_mass_mape"), ("Field-total dry-mass MAPE", "dry_mass_field_total_mape"),
            ("Field-total dry-mass bias", "dry_mass_field_total_bias"),
            ("Area MAPE", "area_mape"), ("Phase MAE [rad]", "phase_mae_rad"),
            ("Phase MAE inside cells [rad]", "phase_mae_rad_in_cell"), ("Phase $r$", "phase_pearson_r"),
            ("Dice", "seg_dice"), ("AJI", "seg_aji"), ("Boundary F1", "seg_boundary_f1"),
            ("Detection recall", "detection_recall"), ("Detection precision", "detection_precision"),
            ("False positives per run", "cells_false_positive")]
    a_lines = []
    for lab, k in mets:
        if k == "cells_false_positive":
            cells = [f"${f(f'mean.{a}.{k}', 0, math_mode=False)} \\pm {f(f'sd.{a}.{k}', 0, math_mode=False)}$" for a in ("A", "B", "B1")]
        else:
            cells = [fpm(f"mean.{a}.{k}", f"sd.{a}.{k}", sign=(k == "dry_mass_field_total_bias")) for a in ("A", "B", "B1")]
        a_lines.append(f"{lab} & " + " & ".join(cells) + r" \\")

    def bcell(a, b, k):
        key = f"cmp.{a}_vs_{b}.{k}"
        d = v(key + ".diff")
        ds = f"{d:+.4f}"
        rec(ds, key + ".diff")
        verdict = v(key + ".verdict")
        if key + ".thr" in V:
            t = f"{v(key + '.thr'):.4f}"
            rec(t, key + ".thr")
            ds_tex = f"$\\mathbf{{{ds}}}$" if verdict == "resolved" else f"${ds}$"
            return f"{ds_tex} & {t} & {verdict}"
        return f"${ds}$ & -- & {verdict}"

    rows3 = [("+IPP (per-cell) vs Baseline", "B", "A"), ("+IPP (image) vs Baseline", "B1", "A"),
             ("+IPP (image) vs +IPP (per-cell)", "B1", "B")]
    rows1 = [("+Area vs +IPP (per-cell)", "B2", "B"), ("+BGA vs +IPP (per-cell)", "C", "B"),
             ("$w=0.1$ vs Baseline", "W01", "A"), ("$w=0.3$ vs Baseline", "W03", "A"),
             ("$w=3.0$ vs Baseline", "W30", "A"), ("+Amplitude vs +IPP (per-cell)", "D0", "B"),
             ("+Fwd (fixed $z$) vs +Amplitude", "D1", "D0"), ("+Fwd (free $z$) vs +Fwd (fixed $z$)", "D2", "D1"),
             ("Compact Baseline vs Baseline", "KA", "A"), ("Compact +IPP vs +IPP (per-cell)", "KB", "B")]
    b_lines = [r"\multicolumn{7}{@{}l}{\emph{Both configurations trained with three seeds}} \\"]
    for lab, a, b in rows3:
        b_lines.append(f"{lab} & {bcell(a, b, 'dry_mass_mape')} & {bcell(a, b, 'dry_mass_field_total_mape')} \\\\")
    b_lines.append(r"\midrule")
    b_lines.append(r"\multicolumn{7}{@{}l}{\emph{One configuration trained with one seed}} \\")
    for lab, a, b in rows1:
        b_lines.append(f"{lab} & {bcell(a, b, 'dry_mass_mape')} & {bcell(a, b, 'dry_mass_field_total_mape')} \\\\")
    body = r"""% Table 6: between-seed variability and resolution of differences.
\begin{table}[!htbp]
\centering
\caption{Replicated configurations on the off-axis test set (113 fields). (a)
Mean $\pm$ SD over seeds 42, 1337 and 2024. (b) Differences in matched-cell and
field-total dry-mass MAPE, configuration minus comparator, with the resolution
threshold $2\times$ the pooled between-seed SD.}
\label{tab:seeds}
\footnotesize
\textbf{(a) Mean $\pm$ SD over three seeds}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
Metric & Baseline & +IPP (per-cell) & +IPP (image) \\
\midrule
""" + "\n".join(a_lines) + r"""
\bottomrule
\end{tabular*}

\vspace{8pt}
\textbf{(b) Differences}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrlrrl@{}}
\toprule
 & \multicolumn{3}{c}{Matched-cell dry-mass MAPE} & \multicolumn{3}{c}{Field-total dry-mass MAPE} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Comparison & $\Delta$ & $2\times$SD & Verdict & $\Delta$ & $2\times$SD & Verdict \\
\midrule
""" + "\n".join(b_lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{$\Delta$: positive is worse. Bold: resolved, $|\Delta| >
2\sqrt{(s_1^2+s_2^2)/2}$ with $s$ the between-seed SD of each configuration.
Not resolvable: fewer than three runs on one side, so no between-seed SD exists.
Differences are computed from unrounded means.}
\end{table}
"""
    write(out, "table_6.tex", body)


def table_8(out):
    CUR["table"] = "table_8.tex"
    amp = [("MAE vs the reference", "amplitude_mae"), ("\\quad the same MAE for $A=1$", "amplitude_unity_mae"),
           ("\\quad ratio ($<1$: closer than $A=1$)", "amplitude_mae_over_unity"),
           ("MAE inside cells", "amplitude_mae_in_cell"), ("Bias", "amplitude_bias"),
           ("Pearson $r$", "amplitude_pearson_r"), ("Predicted amplitude, mean", "amplitude_pred_mean"),
           ("Predicted amplitude, SD", "amplitude_pred_sd"), ("Reference amplitude, mean", "amplitude_reference_mean")]
    a_lines = [f"{lab} & " + " & ".join(f(f"amp.{a}.{k}", sign=(k == "amplitude_bias")) for a in ("D0", "D1", "D2")) + r" \\"
               for lab, k in amp]
    b_lines = []
    for lab, k in (("Residual, prediction", "forward_residual"), ("Residual, reference phase", "forward_residual_reference"),
                   ("Ratio (1.0 = reference phase)", "forward_residual_ratio"),
                   ("In-cell phase MAE [rad]", "phase_mae_rad_in_cell")):
        cells = []
        for a in ("A", "B", "D0", "D1", "D2"):
            if f"sd.{a}.{k}" in V and not math.isnan(v(f"sd.{a}.{k}")):
                cells.append(fpm(f"mean.{a}.{k}", f"sd.{a}.{k}"))
            else:
                cells.append(f(f"mean.{a}.{k}"))
        b_lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    c_lines = []
    for lab, g in (("Off-axis", "off_axis"), ("In-line", "gabor")):
        def wins(k):
            n = v(f"disc.{g}.images")
            w = round(v(k) * n)
            rec(f"{w}", f"{k} x disc.{g}.images", "win rate times fields")
            return f"{w}/{n:.0f}"
        c_lines.append(f"{lab}, {fint(f'disc.{g}.images')} validation fields & {f(f'disc.{g}.floor')} & "
                       f"{f(f'disc.{g}.margin.scaled_0.9', 5, sign=True)} & {wins(f'disc.{g}.win.scaled_0.9')} & "
                       f"{f(f'disc.{g}.margin.scaled_0.5', 4, sign=True)} & {wins(f'disc.{g}.win.scaled_0.5')} \\\\")
    for lab, m in (("global surface", "global"), ("per-field surfaces$^{\\ast}$", "per_field")):
        c_lines.append(f"Off-axis, {fint(f'probe.{m}.images')} test fields, {lab} & {f(f'probe.{m}.phase_reference')} & "
                       f"{f(f'probe.{m}.phase_scaled_0.9.change', 5, sign=True)} & -- & "
                       f"{f(f'probe.{m}.phase_scaled_0.5.change', sign=True)} & -- \\\\")
    body = r"""% Table 8: amplitude output, forward-model residual and phase-scale probes.
\begin{table}[!htbp]
\centering
\caption{(a) Amplitude output against the reconstruction-derived amplitude
reference (113 off-axis test fields; one run each). (b) Forward-model residual
on the 113 off-axis test fields (Baseline and +IPP (per-cell): mean $\pm$ SD
over three seeds; others one run). (c) Change in the residual when the
reference phase is scaled by 0.9 or 0.5 (margin: scaled minus reference;
positive means the reference phase has the lower residual).}
\label{tab:amplitude}
\footnotesize
\textbf{(a) Amplitude output}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}
\toprule
Quantity & +Amplitude & +Fwd (fixed $z$) & +Fwd (free $z$) \\
\midrule
""" + "\n".join(a_lines) + r"""
\bottomrule
\end{tabular*}

\vspace{8pt}
\textbf{(b) Forward-model residual}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Quantity & Baseline & +IPP (per-cell) & +Amplitude & +Fwd (fixed $z$) & +Fwd (free $z$) \\
\midrule
""" + "\n".join(b_lines) + r"""
\bottomrule
\end{tabular*}

\vspace{8pt}
\textbf{(c) Phase-scale probes}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{$0.9\times$ phase} & \multicolumn{2}{c}{$0.5\times$ phase} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Geometry and fields & Residual & Margin & Fields worse & Margin & Fields worse \\
\midrule
""" + "\n".join(c_lines) + r"""
\bottomrule
\end{tabular*}
\tabnote{Fields worse: number of fields on which the scaled phase gives the higher
residual. Configured tolerance for a usable margin: """ + f("config.discrimination_tolerance", 2) + r""". Residual levels in (b)
for configurations with an amplitude output use the predicted amplitude in place of
$A=1$ and are not comparable with the others; only the ratio is. $^{\ast}$Per-field
aberration surfaces are fitted against each field's reference phase and are not
available at inference. One test field (""" + v("probe.per_field.missing.99.stem").replace("_", "\\_") + r""") has
no per-field surface: its fit was rejected because the fitted surface exceeded
the 45\,rad peak-to-valley limit (""" + f("probe.per_field.missing.99.surface_pv_rad", 0) + r"""\,rad). With the global
surface on the same 112 fields the residual is """ + f("probe.global_common.phase_reference") + r""".}
\end{table}
"""
    write(out, "table_8.tex", body)


def table_9(out):
    CUR["table"] = "table_9.tex"
    lines = []
    for s in range(-5, 6):
        sh = f"{s:+d}" if s else "0"
        if s == 0:
            lines.append(r"$0$ & $+0.000$ & 1.0000 & $+0.0000$ & $+0.0000$ & -- \\")
            continue
        lines.append(f"${sh}$ & {f(f'ep.{s}.shift_um', 3, sign=True)} & {f(f'ep.{s}.dice')} & "
                     f"{f(f'ep.{s}.area_signed', sign=True)} & {f(f'ep.{s}.mass_signed', sign=True)} & "
                     f"{f(f'ep.{s}.mass_over_area', 3)} \\\\")
    mm = f"${f('syn.mean.mass_abs_relative_error_mean', 3, pct=True, math_mode=False)} \\pm {f('syn.sd.mass_abs_relative_error_mean', 3, pct=True, math_mode=False)}$\\,\\%"
    ma = f"${f('syn.mean.area_abs_relative_error_mean', 3, pct=True, math_mode=False)} \\pm {f('syn.sd.area_abs_relative_error_mean', 3, pct=True, math_mode=False)}$\\,\\%"
    mt = f"${f('syn.mean.field_total_mass_error_mean', 3, pct=True, math_mode=False)} \\pm {f('syn.sd.field_total_mass_error_mean', 3, pct=True, math_mode=False)}$\\,\\%"
    body = r"""% Table 9: model-free boundary-displacement analysis and synthetic-cell check.
\begin{table}[!htbp]
\centering
\caption{(a) Change in the measurements produced by a uniform displacement of
the reference boundaries (113 off-axis test fields, reference phase and masks
only). (b) Error of the measurement chain on synthetic fields with analytic
mass and area (mean $\pm$ SD over three independently generated sets).}
\label{tab:floors}
\footnotesize
\textbf{(a) Boundary displacement}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrr@{}}
\toprule
Shift [px] & Shift [$\mu$m] & Dice & Area error & Mass error & Mass/area ratio \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}

\vspace{8pt}
\textbf{(b) Measurement chain on analytic fields}\\[2pt]
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr@{}}
\toprule
Quantity & Error \\
\midrule
Per-cell dry mass, mean $|$error$|$       & """ + mm + r""" \\
Per-cell projected area, mean $|$error$|$ & """ + ma + r""" \\
Field-total dry mass, mean bias          & """ + mt + r""" \\
\bottomrule
\end{tabular*}
\end{table}
"""
    write(out, "table_9.tex", body)


def table_10(out):
    CUR["table"] = "table_10.tex"
    rt = [("PyTorch FP32", "pytorch_fp32"), ("ONNX FP32", "onnx_fp32"), ("PyTorch FP16", "pytorch_fp16"), ("ONNX FP16", "onnx_fp16")]
    blocks = []
    for a, lab, sub in (("A", "End-to-End Neural Baseline, $n=3$", "9\\,598\\,099 par., 45.85\\,GMAC"),
                        ("D0", "+Amplitude, $n=1$", "9\\,607\\,412 par., 47.87\\,GMAC"),
                        ("KA", "Compact Baseline, $n=1$", "3\\,360\\,403 par., 24.03\\,GMAC")):
        rows = []
        for i, (rl, r) in enumerate(rt):
            def cell(fld, d):
                k = f"hw.{a}.mean.{r}.{fld}"
                if k not in V:
                    return "--"
                sk = f"hw.{a}.sd.{r}.{fld}"
                if sk in V and fld not in ("weights_mb", "peak_allocated_mb"):
                    return f"${f(k, d, math_mode=False)} \\pm {f(sk, d, math_mode=False)}$"
                return f(k, d)
            first = lab if i == 0 else (sub if i == 1 else "")
            rows.append(f"{first} & {rl} & {cell('latency_mean_ms', 2)} & {cell('latency_p99_over_p50', 2)} & "
                        f"{cell('fps', 1)} & {cell('weights_mb', 2)} & {cell('peak_allocated_mb', 1)} \\\\")
        blocks.append("\n".join(rows))
    body = r"""% Table 10: inference latency, throughput and memory on one workstation GPU.
\begin{table}[!htbp]
\centering
\caption{Computational benchmarking at $900\times900$, batch 1, on an NVIDIA
RTX~A5000. Baseline entries are mean $\pm$ SD over three benchmark sessions,
one per trained seed; the +Amplitude and Compact Baseline rows are each a
single benchmark session.}
\label{tab:efficiency}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccccc@{}}
\toprule
Configuration & Runtime & \shortstack{Latency\\{[ms]}} & $p_{99}/p_{50}$ & FPS & \shortstack{Weights\\{[MB]}} & \shortstack{Peak alloc.\\{[MB]}} \\
\midrule
""" + "\n\\midrule\n".join(blocks) + r"""
\bottomrule
\end{tabular*}
\tabnote{Latency: mean over 500 timed passes after 50 warm-up passes. Weights
and peak allocated memory are reported for PyTorch only.}
\end{table}
"""
    write(out, "table_10.tex", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print("make_tables_v42: start")
    for fn in (table_2, table_3, table_4, table_5, table_6, table_7, table_8, table_9, table_10,
               table_11, table_12, table_13):
        fn(args.out)
    with open(HERE / "table_cells.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "printed", "key", "formula"])
        w.writeheader()
        w.writerows(CELLS)
    print(f"  {len(CELLS)} printed cells recorded -> analysis/v42/table_cells.csv")
    print("make_tables_v42: done")


if __name__ == "__main__":
    main()
