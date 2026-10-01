"""Figure 7 - the physics-aware terms as diagnostics.

WHAT IT SHOWS
    (a) The hologram forward-model residual for every arm, divided by the
        residual obtained from the GROUND-TRUTH phase. A ratio below 1 means the
        predicted field explains the recorded hologram better than the true
        field does, which is the signature of an anti-discriminative objective.
    (b) The per-epoch trajectory of the propagation distance in the arm that
        makes z a free parameter.
    (c) Per-field agreement of the distance recovered by an independent scan, in
        the two acquisition geometries.

WHY IT IS INCLUDED
    Together these three panels support the paper's second negative result and
    keep it correctly scoped. Panel (a) shows the residual cannot be used as a
    training signal on this data. Panels (b) and (c) show why a stable learned z
    must not be reported as a measured distance: the off-axis residual carries
    almost no information about z, so stability near the initialisation reflects
    a weak gradient rather than recovery of the physical distance.

INPUTS
    data/benchmark_results/results_arm_<ARM>.json   residual ratio per arm
                                    (A, B, B' = mean of three seeds)
    data/history_v2_learned_z.json  per-epoch z
    data/z_calibration.json         independent scan, both geometries
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

fig, axes = plt.subplots(1, 3, figsize=(M.COL2, 2.35))

# ---- (a) residual ratio per arm -----------------------------------------
ax = axes[0]
# Every trained off-axis arm once (w = 1.0 is arm B itself, so no W10).
arms = ["A", "B", "B1", "B2", "C", "D0", "D1", "D2", "W01", "W03", "W30", "KA", "KB"]
ratios = [M.arm(a, "forward_residual_ratio")["mean"] for a in arms]
order = np.argsort(ratios)
arms = [arms[i] for i in order]
ratios = [ratios[i] for i in order]

# Only the two +Fwd configurations optimise the forward-model term; +Amplitude
# (D0) predicts amplitude but does not.
colours = [M.C["purple"] if a in ("D1", "D2") else M.C["blue"]
           for a in arms]
x = np.arange(len(arms))
ax.axhline(1.0, color=M.C["orange"], lw=1.0, zorder=1,
           label="ground-truth phase")
ax.scatter(x, ratios, s=26, c=colours, edgecolor="white", linewidth=0.6,
           zorder=3)
ax.set_xticks(x)
NAME = {"A": "Baseline", "B": "+IPP (per-cell)", "B1": "+IPP (image)",
        "B2": "+Area", "C": "+BGA", "D0": "+Amplitude", "D1": "+Fwd (fixed z)",
        "D2": "+Fwd (free z)", "W01": "+IPP, w=0.1", "W03": "+IPP, w=0.3",
        "W30": "+IPP, w=3.0", "KA": "Compact Baseline", "KB": "Compact +IPP"}
ax.set_xticklabels([NAME[a] for a in arms], fontsize=5.4, rotation=90)
ax.set_ylabel("residual / reference residual")
ax.set_title("(a)  Forward-model residual", pad=4)
ax.set_ylim(min(ratios) - 0.004, 1.004)
ax.grid(axis="x", visible=False)
M.despine(ax)
# Identity is not carried by colour alone: the two groups are also separated in
# the legend.
opt = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                 markerfacecolor=M.C["purple"], markeredgecolor="white")
notopt = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                    markerfacecolor=M.C["blue"], markeredgecolor="white")
gt = plt.Line2D([], [], color=M.C["orange"], lw=1.0)
ax.legend([gt, opt, notopt],
          ["ground-truth phase", "optimises the forward term", "does not"],
          fontsize=5.6, loc="lower right", handlelength=1.4,
          borderaxespad=0.2, labelspacing=0.25)
ax.annotate("below 1: the predicted phase explains\n"
            "the hologram better than the truth does",
            xy=(0.03, 0.985), xycoords="axes fraction", fontsize=5.6,
            color=M.INK2, va="top", ha="left")
print(f"  residual ratios: {min(ratios):.4f} to {max(ratios):.4f}; "
      f"{sum(1 for r in ratios if r < 1.0)}/{len(ratios)} below 1.0")

# ---- (b) learned-z trajectory -------------------------------------------
ax = axes[1]
z = M.learned_z_trajectory()
epochs = np.arange(1, len(z) + 1)
supplied = M.load_json("z_calibration.json")["off_axis"]["discrimination"]["distance_um"]
ax.axhline(supplied, color=M.C["orange"], lw=1.0,
           label=f"supplied / initial $z$ = {supplied:g} " + r"$\mu$m")
ax.plot(epochs, z, color=M.C["purple"], lw=1.3)
ax.scatter([1, len(z)], [z[0], z[-1]], s=22, color=M.C["purple"],
           edgecolor="white", linewidth=0.6, zorder=3)
ax.annotate(f"{z[0]:.2f}", xy=(1, z[0]), xytext=(5, 3),
            textcoords="offset points", fontsize=6, color=M.INK)
ax.annotate(f"{z[-1]:.2f}", xy=(len(z), z[-1]), xytext=(-4, -10),
            textcoords="offset points", fontsize=6, color=M.INK, ha="right")
ax.set_xlabel("epoch")
ax.set_ylabel(r"propagation distance $z$ [$\mu$m]")
ax.set_title("(b)  Learned $z$, +Fwd (free $z$)", pad=4)
lo_z, hi_z = float(np.nanmin(z)), float(np.nanmax(z))
ax.set_ylim(lo_z - 0.08, hi_z + 0.32)          # headroom so the legend clears the curve
ax.legend(fontsize=6.0, loc="upper right")
M.despine(ax)
print(f"  learned z: {z[0]:.3f} -> {z[-1]:.3f} um, "
      f"excursion {np.nanmax(z)-np.nanmin(z):.3f} um")

# ---- (c) identifiability by geometry ------------------------------------
ax = axes[2]
zc = M.load_json("z_calibration.json")
geoms = [("off_axis", "Off-axis"), ("gabor", "In-line\n(Gabor)")]
med = [zc[g]["per_image_median_z_um"] for g, _ in geoms]
iqr = [zc[g]["per_image_iqr_um"] for g, _ in geoms]
ident = [zc[g]["identifiable"] for g, _ in geoms]
colours = [M.C["green"] if i else M.C["orange"] for i in ident]

x = np.arange(len(geoms))
ax.errorbar(x, med, yerr=[i / 2 for i in iqr], fmt="none", ecolor=M.INK,
            elinewidth=1.0, capsize=4, zorder=2)
ax.scatter(x, med, s=40, c=colours, edgecolor="white", linewidth=0.7, zorder=3)
for xi, m, i, ok in zip(x, med, iqr, ident):
    ax.text(xi + 0.12, m, f"IQR {i:.1f}" + r" $\mu$m", fontsize=6.2,
            va="center", color=M.INK)
    ax.text(xi, ax.get_ylim()[0], "", ha="center")
ax.axhline(supplied, color=M.NEUTRAL, lw=0.8, ls=(0, (4, 2)))
ax.set_xticks(x)
ax.set_xticklabels([f"{lbl}\n{'identifiable' if ok else 'not identifiable'}"
                    for (_, lbl), ok in zip(geoms, ident)], fontsize=6.2)
ax.set_ylabel(r"recovered $z$ [$\mu$m]")
ax.set_title("(c)  Is $z$ recoverable?", pad=4)
ax.set_xlim(-0.5, 1.7)
ax.grid(axis="x", visible=False)
M.despine(ax)
print(f"  identifiable: off_axis={ident[0]}, gabor={ident[1]}")

fig.tight_layout()
M.save(fig, "figure_7.png")
