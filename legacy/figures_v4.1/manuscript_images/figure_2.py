"""Figure 2 - the learned framework against the classical baseline, both geometries.

WHAT IT SHOWS
    Ten matched test metrics for the end-to-end network and for the classical
    reconstruct-then-segment pipeline, in the off-axis geometry (left) and the
    in-line Gabor geometry (right). Both pipelines are scored by the same
    evaluator on the same 113 test fields, so a difference is a difference in
    reconstruction and nothing else.

WHY IT IS THE FIRST FIGURE
    It carries the paper's primary claim, and the right-hand panel carries the
    strongest single result: the classical pipeline fails on in-line holograms
    while the network does not.

INPUTS
    data/benchmark_results/results_conventional_{off_axis,gabor}.json   classical
    data/benchmark_results/results_arm_A.json    learned, off-axis (mean of 3 seeds)
    data/benchmark_results/results_arm_G.json    learned, in-line (one run)

NOTE ON ORIENTATION
    Every metric is plotted so that LONGER IS BETTER. Error metrics are shown as
    (1 - error) and labelled accordingly, because a reader should not have to
    reconcile a table in which some bars improve upward and others downward.
    The raw values are printed at the end of each bar.
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

def _means(name):
    return {k: v["mean"] for k, v in M.bench(name)["statistics"].items()}


CLASSICAL = {"off_axis": _means("conventional_off_axis"),
             "gabor": _means("conventional_gabor")}
# Arm A is the mean over its three seeds, the same value as Table 4.
LEARNED = {"off_axis": _means("arm_A"), "gabor": _means("arm_G")}

# (key, display label, "higher" or "lower" is better)
METRICS = [
    ("phase_pearson_r",   r"Phase Pearson $r$  $\uparrow$",   "higher"),
    ("phase_ssim",        r"Phase SSIM  $\uparrow$",           "higher"),
    ("phase_mae_rad",     r"Phase MAE [rad]  $\downarrow$",    "lower"),
    ("seg_dice",          r"Segmentation Dice  $\uparrow$",    "higher"),
    ("seg_aji",           r"Segmentation AJI  $\uparrow$",     "higher"),
    ("seg_boundary_f1",   r"Boundary F1  $\uparrow$",          "higher"),
    ("detection_f1",      r"Detection F1  $\uparrow$",         "higher"),
    ("area_mape",         r"Area MAPE  $\downarrow$",          "lower"),
    ("dry_mass_mape",     r"Dry mass MAPE  $\downarrow$",      "lower"),
    ("circularity_mape",  r"Circularity MAPE  $\downarrow$",   "lower"),
]

def plot_panel(ax, modality, title):
    labels, cls_plot, lrn_plot, cls_txt, lrn_txt = [], [], [], [], []
    cls_clipped = []
    for key, label, better in METRICS:
        c = CLASSICAL[modality].get(key)
        l = LEARNED[modality].get(key)
        if c is None or l is None:
            continue
        labels.append(label)
        # Map to a common "longer is better" axis in [0, 1].
        if better == "lower":
            cv, lv = 1.0 - c, 1.0 - l
        else:
            cv, lv = c, l
        cls_clipped.append(cv < 0)
        cls_plot.append(max(0.0, cv)); lrn_plot.append(max(0.0, lv))
        cls_txt.append(f"{c:.3f}"); lrn_txt.append(f"{l:.3f}")

    y = np.arange(len(labels))
    h = 0.36
    # 2 px surface gap between adjacent fills is achieved by the 0.04 offset pad.
    ax.barh(y + h / 2 + 0.02, lrn_plot, height=h, color=M.C["blue"],
            edgecolor="white", linewidth=0.6, label="Neural (End-to-End Neural Baseline / In-Line Neural Configuration)")
    ax.barh(y - h / 2 - 0.02, cls_plot, height=h, color=M.C["orange"],
            edgecolor="white", linewidth=0.6, label="Classical Pipeline")

    for yi, v, t in zip(y + h / 2 + 0.02, lrn_plot, lrn_txt):
        ax.text(min(v, 1.0) + 0.012, yi, t, va="center", ha="left",
                fontsize=6.2, color=M.INK)
    for yi, v, t, clipped in zip(y - h / 2 - 0.02, cls_plot, cls_txt, cls_clipped):
        # A clipped bar has zero length, so its label would sit on top of the
        # marker drawn at the axis; offset it further right in that case.
        ax.text(min(v, 1.0) + (0.030 if clipped else 0.012), yi, t,
                va="center", ha="left", fontsize=6.2, color=M.INK2)
        if clipped:
            # A negative value cannot be drawn on a longer-is-better axis; mark
            # it so the empty bar is not read as a missing measurement.
            ax.plot([0.004], [yi], marker="<", markersize=3.4,
                    color=M.C["orange"], clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score, oriented so that longer is better")
    ax.set_title(title, pad=4)
    ax.grid(axis="y", visible=False)
    M.despine(ax)

fig, axes = plt.subplots(1, 2, figsize=(M.COL2, 3.15), sharex=True)

plot_panel(axes[0], "off_axis", "(a)  Off-axis holography")
plot_panel(axes[1], "gabor",    "(b)  In-line (Gabor) holography")
axes[1].set_yticklabels([])
axes[1].set_ylabel("")

valid = M.bench("conventional_gabor")["runs"][0].get("non_numeric", {}).get("reconstruction_valid")
print(f"  in-line classical reconstruction_valid = {valid}"
      "  (stated in the caption, not annotated on the bars)")

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.045), frameon=False)
fig.tight_layout()
print("figure_2: learned vs classical, both geometries")
M.save(fig, "figure_2.png")
