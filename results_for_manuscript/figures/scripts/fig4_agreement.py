"""Figure 6 - per-cell measurement agreement on the test split.

WHAT IT SHOWS
    For every IoU-matched cell of the test split: predicted against reference
    dry mass (a) and projected area (c), with the corresponding Bland-Altman
    relative-difference plots (b, d).

WHY IT IS INCLUDED
    Correlation and agreement are different claims, and only the second one
    matters for a measurement. Two methods can correlate at r = 0.99 and still
    disagree by 30% on every cell. The limits of agreement are the number a
    biologist needs in order to decide whether the method resolves the mass
    differences their experiment is about, which is exactly the
    measurement-readiness claim this paper makes.

    The relative form is used because dry mass is a fixed scalar multiple of the
    integrated phase, so a relative error in the phase integral is by
    construction the relative error in dry mass.

    UNIT OF ANALYSIS. Cells segmented from the same field share one acquisition
    scale and are therefore not independent observations. The agreement panels
    accordingly use the FIELD as the unit of analysis, each field summarised by
    the median of its matched cells - the same convention the project's own
    evaluator uses, so the bias and limits here reproduce the reported table.
    The correlation panels keep the per-cell scatter, which is what the reported
    per-cell Pearson r describes.

INPUTS
    data/per_cell_test_A.csv   one row per matched cell, arm A, test split
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

rows = M.load_csv("per_cell_test_A.csv")
print(f"  matched cells: {len(rows)}")

QUANTITIES = [
    ("dry_mass_pg", "dry mass", "pg", M.C["blue"]),
    ("area_um2", "projected area", r"$\mu$m$^2$", M.C["orange"]),
]

fig, axes = plt.subplots(2, 2, figsize=(M.COL2, 4.3))

for r_i, (key, name, unit, colour) in enumerate(QUANTITIES):
    pred = M.column(rows, f"{key}_pred")
    ref = M.column(rows, f"{key}_ref")
    ok = np.isfinite(pred) & np.isfinite(ref) & (ref != 0)
    pred, ref = pred[ok], ref[ok]
    rel = (pred - ref) / ref

    # ---- correlation panel ------------------------------------------------
    ax = axes[r_i][0]
    lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    pad = (lim[1] - lim[0]) * 0.04
    lim = [lim[0] - pad, lim[1] + pad]
    ax.plot(lim, lim, color=M.INK2, lw=0.8, zorder=1, label="identity")
    ax.scatter(ref, pred, s=3.2, color=colour, alpha=0.40, linewidth=0, zorder=2)
    r = float(np.corrcoef(ref, pred)[0, 1])
    mape = float(np.mean(np.abs(rel)))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"reference {name} [{unit}]")
    ax.set_ylabel(f"predicted {name} [{unit}]")
    ax.set_title(f"({'ac'[r_i]})  {name.capitalize()}: agreement", pad=4)
    ax.text(0.04, 0.96, f"$r$ = {r:.3f}\nMAPE = {mape:.3f}\n$n$ = {len(ref)}",
            transform=ax.transAxes, fontsize=6.4, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, linewidth=0,
                      boxstyle="round,pad=0.2"))
    ax.legend(fontsize=6.2, loc="lower right")
    M.despine(ax)

    # ---- Bland-Altman panel, per FIELD -----------------------------------
    ax = axes[r_i][1]
    by_field: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            pv, rv = float(row[f"{key}_pred"]), float(row[f"{key}_ref"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(pv) and np.isfinite(rv) and rv != 0:
            by_field.setdefault(row["stem"], []).append((pv, rv))
    fp = np.array([np.median([v for v, _ in vals]) for vals in by_field.values()])
    fr = np.array([np.median([v for _, v in vals]) for vals in by_field.values()])
    # The relative difference is taken against the MEAN of the pair, which is
    # the standard relative Bland-Altman construction and the one the project's
    # evaluator uses. Dividing by the reference instead shifts the bias by a few
    # thousandths and the figure would no longer reproduce the reported table.
    mean_val = (fp + fr) / 2.0
    keep = np.abs(mean_val) > 0
    rel = (fp[keep] - fr[keep]) / mean_val[keep]
    mean_val = mean_val[keep]
    bias, sd = float(np.mean(rel)), float(np.std(rel, ddof=1))
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    ax.axhline(0, color=M.INK2, lw=0.7, zorder=1)
    ax.scatter(mean_val, rel, s=9, color=colour, alpha=0.65, linewidth=0,
               zorder=2)
    ax.axhline(bias, color=M.C["purple"], lw=1.1, zorder=3,
               label=f"bias {bias:+.3f}")
    for y in loa:
        ax.axhline(y, color=M.C["purple"], lw=0.9, ls=(0, (4, 2)), zorder=3)
    ax.plot([], [], color=M.C["purple"], lw=0.9, ls=(0, (4, 2)),
            label=f"95% LoA {loa[0]:+.3f}, {loa[1]:+.3f}")
    ax.set_xlabel(f"field median, mean of predicted and reference [{unit}]")
    ax.set_ylabel("relative difference")
    ax.set_title(f"({'bd'[r_i]})  {name.capitalize()}: Bland-Altman, per field",
                 pad=4)
    ax.legend(fontsize=6.2, loc="upper right")
    M.despine(ax)
    print(f"  {name}: per-cell r={r:.4f} MAPE={mape:.4f} (n={len(ok[ok])})")
    print(f"      per-field bias={bias:+.4f} LoA=[{loa[0]:+.4f}, {loa[1]:+.4f}]"
          f"  (n={len(rel)} fields)")

fig.tight_layout()
M.save(fig, "figure_6.png")
