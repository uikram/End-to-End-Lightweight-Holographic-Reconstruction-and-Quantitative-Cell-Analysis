"""Figure 4 - the measurement-aware ablation, against seed noise.

WHAT IT SHOWS
    Dry-mass MAPE (a) and segmentation Dice (b) for every trained arm, with the
    between-seed standard deviation drawn as an error bar on the three arms that
    were replicated at seeds 42, 1337 and 2024. The shaded band is the baseline
    plus or minus twice its pooled between-seed SD, which is the study's
    resolution criterion.

WHY IT IS INCLUDED
    The band is the whole point. Without it a reader ranks bars that are inside
    seed noise. With it, the figure shows that adding measurement-aware terms
    moves the primary metric outside the band in the WRONG direction, and that
    (the compact-decoder variants are listed in Table S6, not plotted).

INPUTS
    data/benchmark_results/results_arm_<ARM>.json   mean, SD and n per arm
    (A, B, B' are means over three seeds; every other arm is its single run)
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

# Order follows the paper's grouping, not the alphabet. The first element is
# the internal code used by the result files; the second is the paper's name.
ARMS = [("A", "End-to-End\nNeural Baseline"), ("B", "+IPP\n(per-cell)"),
        ("B1", "+IPP\n(image)"), ("B2", "+Area"), ("C", "+BGA"),
        ("D0", "+Amplitude"), ("D1", "+Fwd\n(fixed z)"), ("D2", "+Fwd\n(free z)")]

METRIC = {"mass MAPE": "dry_mass_mape", "Dice": "seg_dice"}


def panel(ax, column, label, title, lower_is_better):
    codes = [a for a, _ in ARMS]
    stats = [M.arm(a, METRIC[column]) for a in codes]
    vals = [st["mean"] for st in stats]
    errs = [st["sd"] for st in stats]
    nseed = [st["n"] for st in stats]

    base = vals[0]
    base_sd = errs[0]
    # The resolution criterion: 2 x the pooled between-seed SD. With equal n the
    # pooled SD of two arms of the same spread is that spread, so the band is
    # +/- 2 SD about the baseline.
    ax.axhspan(base - 2 * base_sd, base + 2 * base_sd,
               color=M.C["blue"], alpha=0.13, lw=0, zorder=0,
               label=r"baseline $\pm\,2\times$ between-seed SD")
    ax.axhline(base, color=M.C["blue"], lw=0.8, zorder=1)

    colours = []
    for a in codes:
        if a == "A":
            colours.append(M.C["blue"])
        elif a.startswith("K"):
            colours.append(M.C["green"])
        elif a in ("D0", "D1", "D2"):
            colours.append(M.C["purple"])
        else:
            colours.append(M.C["orange"])

    x = np.arange(len(codes))
    # A DOT PLOT, not bars. The interesting differences here are a few parts in
    # a thousand, so the y-axis cannot start at zero -- and a bar whose baseline
    # is not zero encodes its length dishonestly. A dot encodes position only.
    rep = [i for i, n in enumerate(nseed) if n > 1]
    ax.errorbar(x[rep], [vals[i] for i in rep], yerr=[errs[i] for i in rep],
                fmt="none", ecolor=M.INK, elinewidth=0.9, capsize=2.4, zorder=3)
    # A faint stem to the baseline keeps each point readable against the band
    # without implying magnitude-from-zero.
    for xi, v, c in zip(x, vals, colours):
        ax.plot([xi, xi], [base, v], color=c, lw=0.8, alpha=0.55, zorder=2)
    ax.scatter(x, vals, s=34, c=colours, edgecolor="white", linewidth=0.7,
               zorder=4)

    span = max(vals) - min(vals)
    for xi, v, e, n in zip(x, vals, errs, nseed):
        # Label on the far side of the baseline, so it never sits on the stem
        # or inside the error bar.
        half = e if n > 1 else 0.0
        if v >= base:
            ax.text(xi, v + half + span * 0.075, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=5.8, color=M.INK)
        else:
            ax.text(xi, v - half - span * 0.075, f"{v:.3f}", ha="center",
                    va="top", fontsize=5.8, color=M.INK)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in ARMS], fontsize=5.8)
    arrow = r"$\downarrow$" if lower_is_better else r"$\uparrow$"
    ax.set_ylabel(f"{label}  {arrow}")
    ax.set_title(title, pad=4)
    ax.grid(axis="x", visible=False)
    M.despine(ax)
    lo = min(v - e for v, e in zip(vals, errs))
    hi = max(v + e for v, e in zip(vals, errs))
    pad = (hi - lo) * 0.34
    ax.set_ylim(lo - pad * 0.8, hi + pad * 1.25)
    return nseed


fig, axes = plt.subplots(2, 1, figsize=(M.COL2, 4.1), sharex=True)
n1 = panel(axes[0], "mass MAPE", "Dry mass MAPE", "(a)  Primary measurement metric", True)
panel(axes[1], "Dice", "Segmentation Dice", "(b)  Segmentation overlap", False)

handles, labels = axes[0].get_legend_handles_labels()
extra = [plt.Line2D([], [], marker="o", linestyle="none", markersize=5,
                    markerfacecolor=c, markeredgecolor="white")
         for c in (M.C["blue"], M.C["orange"], M.C["purple"])]
fig.legend(handles + extra,
           labels + ["End-to-End Neural Baseline", "measurement-aware objectives",
                     "amplitude output / forward-model term"],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.085),
           frameon=False)
axes[1].set_xlabel("configuration  (error bars: between-seed SD, n = 3 where shown)",
                   labelpad=2)
fig.tight_layout()
print("figure_4: ablation against seed noise")
print(f"  configurations with >1 seed: {[a for (a, _), n in zip(ARMS, n1) if n > 1]}")
M.save(fig, "figure_4.png")
