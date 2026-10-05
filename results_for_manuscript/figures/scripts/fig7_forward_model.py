"""Forward-model figure (Fig. 6 of the paper; key figure_7 in figure_index.json): consistency on the corrected runs.

(a) Forward-model residual of every trained off-axis configuration on the 113
    test fields, divided by the residual obtained with the reference phase in
    place of the prediction (1.0 = same residual as the reference phase).
(b) Per-epoch propagation distance of +Fwd (free z).
(c) Residual computed from the reference phase as a function of the
    propagation distance (four validation fields per geometry), with the
    per-field best distances marked and the recording distance dashed.

Inputs (all under <repository>/runs, or $HOLOQPI_RUNS):
    benchmark_results/results_arm_<ARM>.json   residual ratio per configuration
    RESULTS.md                                 learned-z trajectory
    z_calibration.json                         residual scans, both geometries

Run:  python fig7_forward_model.py      -> ../regenerated/figure_7.png
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

print("figure 7: start")
M.style()
fig, axes = plt.subplots(1, 3, figsize=(M.COL2, 2.45),
                         gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})

# ---- (a) residual ratio per configuration --------------------------------
print("  panel a: residual ratio per configuration")
ax = axes[0]
arms = ["A", "B", "B1", "B2", "C", "D0", "D1", "D2", "W01", "W03", "W30", "KA", "KB"]
ratios = [M.arm(a, "forward_residual_ratio")["mean"] for a in arms]
order = np.argsort(ratios)
arms = [arms[i] for i in order]
ratios = [ratios[i] for i in order]
colours = [M.C["purple"] if a in ("D1", "D2") else M.C["blue"] for a in arms]
x = np.arange(len(arms))
ax.axhline(1.0, color=M.C["orange"], lw=1.0, zorder=1)
ax.scatter(x, ratios, s=26, c=colours, edgecolor="white", linewidth=0.6, zorder=3)
NAME = {"A": "Baseline", "B": "+IPP (per-cell)", "B1": "+IPP (image)",
        "B2": "+Area", "C": "+BGA", "D0": "+Amplitude", "D1": "+Fwd (fixed z)",
        "D2": "+Fwd (free z)", "W01": "+IPP, w=0.1", "W03": "+IPP, w=0.3",
        "W30": "+IPP, w=3.0", "KA": "Compact Baseline", "KB": "Compact +IPP"}
ax.set_xticks(x)
ax.set_xticklabels([NAME[a] for a in arms], fontsize=5.4, rotation=90)
ax.set_ylabel("residual ratio")
ax.set_title("(a)  Forward-model residual", pad=4)
lo_r, hi_r = min(min(ratios), 1.0), max(max(ratios), 1.0)
span = max(hi_r - lo_r, 1e-3)
ax.set_ylim(lo_r - 0.55 * span, hi_r + 0.15 * span)
ax.grid(axis="x", visible=False)
M.despine(ax)
ref = plt.Line2D([], [], color=M.C["orange"], lw=1.0)
opt = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                 markerfacecolor=M.C["purple"], markeredgecolor="white")
notopt = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                    markerfacecolor=M.C["blue"], markeredgecolor="white")
ax.legend([ref, opt, notopt],
          ["reference phase", r"trained with $\mathcal{L}_\mathrm{fwd}$", r"trained without"],
          fontsize=5.6, loc="lower right", handlelength=1.4, borderaxespad=0.2,
          labelspacing=0.25)
print(f"    ratios {min(ratios):.4f} to {max(ratios):.4f}")

# ---- (b) learned-z trajectory ---------------------------------------------
print("  panel b: learned z trajectory")
ax = axes[1]
z = M.learned_z_trajectory()
epochs = np.arange(1, len(z) + 1)
zc = M.load_json("z_calibration.json")
supplied = zc["off_axis"]["discrimination"]["distance_um"]
ax.axhline(supplied, color=M.C["orange"], lw=1.0,
           label=f"recording distance {supplied:g} " + r"$\mu$m")
ax.plot(epochs, z, color=M.C["purple"], lw=1.3)
ax.scatter([1, len(z)], [z[0], z[-1]], s=22, color=M.C["purple"],
           edgecolor="white", linewidth=0.6, zorder=3)
ax.annotate(f"{z[0]:.2f}", xy=(1, z[0]), xytext=(5, 3), textcoords="offset points",
            fontsize=6, color=M.INK)
ax.annotate(f"{z[-1]:.2f}", xy=(len(z), z[-1]), xytext=(-4, -10),
            textcoords="offset points", fontsize=6, color=M.INK, ha="right")
ax.set_xlabel("epoch")
ax.set_ylabel(r"propagation distance $z$ [$\mu$m]")
ax.set_title("(b)  Learned $z$, +Fwd (free $z$)", pad=4)
ax.set_ylim(float(np.nanmin(z)) - 0.08, float(np.nanmax(z)) + 0.32)
ax.legend(fontsize=6.0, loc="upper right")
M.despine(ax)
print(f"    z {z[0]:.3f} -> {z[-1]:.3f} um")

# ---- (c) residual scan over z ----------------------------------------------
print("  panel c: residual scan over z, both geometries")
ax = axes[2]
for geom, label, colour, marker in (("off_axis", "off-axis", M.C["blue"], "o"),
                                    ("gabor", "in-line (Gabor)", M.C["green"], "s")):
    c = zc[geom]["curve"]
    d = np.asarray(c["distances_um"])
    r = np.asarray(c["forward_residual"])
    ax.plot(d, r, color=colour, lw=1.2, marker=marker, markersize=2.2,
            label=f"{label}, {c['images']} fields")
    best = np.asarray(c["per_image_best_z_um"])
    # per-field best distances as ticks above the curves (one row per geometry)
    ax.scatter(best, np.full_like(best, 1.075 if geom == "gabor" else 1.11),
               marker="|", s=60, color=colour, linewidths=1.2, zorder=3)
    print(f"    {geom}: per-field best z {np.round(best, 2).tolist()}")
ax.axvline(supplied, color=M.NEUTRAL, lw=0.8, ls=(0, (4, 2)))
ax.text(supplied + 2, 0.40, f"{supplied:g} " + r"$\mu$m", fontsize=5.8, color=M.INK2,
        va="bottom")
ax.set_xlabel(r"propagation distance $z$ [$\mu$m]")
ax.set_ylabel("residual, reference phase")
ax.set_title("(c)  Residual scan over $z$", pad=4)
ax.set_ylim(0.38, 1.14)
ax.text(-88, 1.13, "per-field best $z$", fontsize=5.4, color=M.INK2, va="top")
ax.legend(fontsize=5.8, loc="lower left", bbox_to_anchor=(0.0, 0.0),
          handlelength=1.6, borderaxespad=0.2)
M.despine(ax)

fig.tight_layout()
M.save(fig, "figure_7.png")
print("figure 7: done")
