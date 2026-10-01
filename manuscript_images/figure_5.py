"""Figure 5 - weight sweep on the per-cell integrated-phase term.

WHAT IT SHOWS
    Three metrics against the weight applied to the per-cell integrated-phase
    loss, w in {0.1, 0.3, 1.0, 3.0}. The horizontal line and band are the
    baseline (w = 0) and its +/- 2 x between-seed SD interval.

WHY IT IS INCLUDED
    It separates two very different conclusions: "the term hurts" and "this
    particular weight hurts". Dice and phase MAE degrade monotonically as w
    rises, which makes the effect a dose response rather than a tuning accident.
    That is the strongest form the negative result can take.

    w = 1.0 is arm B's objective exactly, at the same seed, so the study does
    not train it twice; the point plotted at w = 1.0 is arm B.

INPUTS
    data/benchmark_results/results_arm_{W01,W03,B,W30,A}.json
    (A and B are means over three seeds, with their SD; W01/W03/W30 one run each)
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

WEIGHTS = [0.1, 0.3, 1.0, 3.0]
ARMS = ["W01", "W03", "B", "W30"]          # w = 1.0 is arm B itself

PANELS = [
    ("Segmentation Dice",   "seg_dice",       False),
    ("Phase MAE [rad]",     "phase_mae_rad",  True),
    ("Dry mass MAPE",       "dry_mass_mape",  True),
]

fig, axes = plt.subplots(1, 3, figsize=(M.COL2, 2.25))

for ax, (label, key, lower_better) in zip(axes, PANELS):
    stats = [M.arm(a, key) for a in ARMS]
    vals = [st["mean"] for st in stats]
    base_st = M.arm("A", key)
    base, sd = base_st["mean"], base_st["sd"]

    ax.axhspan(base - 2 * sd, base + 2 * sd, color=M.C["blue"], alpha=0.13,
               lw=0, zorder=0)
    ax.axhline(base, color=M.C["blue"], lw=0.9, zorder=1,
               label="End-to-End Neural Baseline, $w=0$")
    ax.plot(WEIGHTS, vals, "-o", color=M.C["orange"], zorder=3,
            markeredgecolor="white", markeredgewidth=0.7,
            label="+IPP (per-cell) at weight $w$")
    # Arm B has three seeds: show its between-seed SD like the baseline's.
    ax.errorbar([1.0], [vals[2]], yerr=[stats[2]["sd"]], fmt="none",
                ecolor=M.INK, elinewidth=0.9, capsize=2.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(WEIGHTS)
    ax.set_xticklabels([f"{w:g}" for w in WEIGHTS])
    ax.minorticks_off()
    arrow = r"$\downarrow$" if lower_better else r"$\uparrow$"
    ax.set_ylabel(f"{label}  {arrow}")
    ax.set_xlabel("per-cell loss weight $w$")
    M.despine(ax)

    # Mark which point is arm B, since it is borrowed rather than retrained.
    ax.annotate("+IPP (per-cell)", xy=(1.0, vals[2]), xytext=(-7, 0),
                textcoords="offset points", ha="right", va="center", fontsize=6,
                color=M.INK2)
    span = max(max(vals), base) - min(min(vals), base)
    ax.set_ylim(min(min(vals), base - 2 * sd) - span * 0.22,
                max(max(vals), base + 2 * sd) + span * 0.22)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.4,
           bbox_to_anchor=(0.5, -0.06), frameon=False)
fig.tight_layout()
print("figure_5: per-cell loss weight sweep")
M.save(fig, "figure_5.png")
