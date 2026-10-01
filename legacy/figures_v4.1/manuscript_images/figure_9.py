"""Figure 9 - efficiency, and what the compact decoder costs.

WHAT IT SHOWS
    (a) Measured latency against model size for every runtime and precision
        combination, full decoder versus compact decoder.
    (b) Throughput for the same combinations.
    (c) The accuracy cost of the compact decoder against the between-seed
        interval, which is the only fair yardstick for a difference this small.

WHY IT IS INCLUDED
    Edge suitability is a trade-off, not a score, so the accuracy axis has to
    appear beside the cost axes. Panel (c) is the one that licenses the
    recommendation: the compact model's accuracy penalty is smaller than the
    spread the same configuration produces when only the random seed changes.

    These are workstation measurements. They establish the relative ordering of
    the configurations and the budgets an embedded target must meet; they are
    not a demonstration of embedded deployment.

INPUTS
    data/benchmark_results/results_hardware_arm_{A,B,D0,KA,KB}.json
        timing; A and B are means over one session per seed (n = 3)
    data/benchmark_results/results_arm_{A,KA}.json   accuracy, panel (c)
"""
import matplotlib.pyplot as plt
import numpy as np

import mstyle as M

M.style()

COMBOS = [("pytorch", "fp32"), ("onnx", "fp32"), ("pytorch", "fp16"), ("onnx", "fp16")]
data = []
for code in ("A", "B", "D0", "KA", "KB"):
    st = M.bench(f"hardware_arm_{code}")["statistics"]
    for rt, pr in COMBOS:
        k = f"{rt}_{pr}."
        data.append({
            "experiment": code, "runtime": rt, "precision": pr,
            "params": st["params_total"]["mean"], "gmacs": st["gmacs"]["mean"],
            "p50": st[k + "latency_p50_ms"]["mean"],
            "p99": st[k + "latency_p99_ms"]["mean"],
            "fps": st[k + "fps"]["mean"],
            "ratio_max": st[k + "latency_p99_over_p50"]["max"],
        })

# The two decoder configurations, identified by parameter count rather than by
# name, so a renamed experiment does not silently drop out of the figure.
FULL = [d for d in data if d["params"] > 6e6]
COMPACT = [d for d in data if d["params"] <= 6e6]
print(f"  timing rows: {len(data)}  (full {len(FULL)}, compact {len(COMPACT)})")
print(f"  worst p99/p50 over every session: {max(d['ratio_max'] for d in data):.2f}")

MARKERS = {("pytorch", "fp32"): "o", ("pytorch", "fp16"): "s",
           ("onnx", "fp32"): "^", ("onnx", "fp16"): "D"}

fig, axes = plt.subplots(1, 3, figsize=(M.COL2, 2.35))

# ---- (a) latency vs size -------------------------------------------------
ax = axes[0]
for group, colour, label in ((FULL, M.C["blue"], "standard decoder"),
                             (COMPACT, M.C["green"], "compact decoder")):
    for d in group:
        ax.scatter(d["params"] / 1e6, d["p50"], s=30, color=colour,
                   marker=MARKERS[(d["runtime"], d["precision"])],
                   edgecolor="white", linewidth=0.6, zorder=3)
        # p99 as a whisker: the tail is what a real-time budget must respect.
        ax.plot([d["params"] / 1e6] * 2, [d["p50"], d["p99"]], color=colour,
                lw=0.8, alpha=0.6, zorder=2)
ax.set_xlabel("parameters [M]")
ax.set_ylabel(r"latency $p_{50}$ [ms]  $\downarrow$")
ax.set_title("(a)  Latency vs model size", pad=4)
ax.set_xlim(2.4, 10.7)
M.despine(ax)

# ---- (b) throughput ------------------------------------------------------
ax = axes[1]
combos = [("pytorch", "fp32"), ("onnx", "fp32"), ("pytorch", "fp16"), ("onnx", "fp16")]
labels = ["PyTorch\nfp32", "ONNX\nfp32", "PyTorch\nfp16", "ONNX\nfp16"]
x = np.arange(len(combos))
w = 0.36
# Same configurations as Table 10: End-to-End Neural Baseline and Compact Baseline.
BAR_A = [d for d in data if d["experiment"] == "A"]
BAR_KA = [d for d in data if d["experiment"] == "KA"]
for off, group, colour, name in ((-w / 2 - 0.015, BAR_A, M.C["blue"], "End-to-End Neural Baseline"),
                                 (+w / 2 + 0.015, BAR_KA, M.C["green"], "Compact Baseline")):
    vals = []
    for combo in combos:
        match = [d["fps"] for d in group
                 if (d["runtime"], d["precision"]) == combo]
        vals.append(float(np.mean(match)) if match else np.nan)
    ax.bar(x + off, vals, width=w, color=colour, edgecolor="white",
           linewidth=0.6, label=name)
    for xi, v in zip(x + off, vals):
        if np.isfinite(v):
            ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=5.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.2)
ax.set_ylabel(r"throughput [fps]  $\uparrow$")
ax.set_title("(b)  Throughput", pad=4)
ax.set_ylim(0, 152)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=6.2, loc="upper left")
M.despine(ax)

# ---- (c) the accuracy cost, against seed noise ---------------------------
ax = axes[2]
a_mass = M.arm("A", "dry_mass_mape")["mean"]      # mean of three seeds
ka_mass = M.arm("KA", "dry_mass_mape")["mean"]    # one run
sd = M.arm("A", "dry_mass_mape")["sd"]
delta = ka_mass - a_mass

ax.axhspan(-2 * sd, 2 * sd, color=M.C["blue"], alpha=0.15, lw=0,
           label=r"$\pm\,2\times$ between-seed SD")
ax.axhline(0, color=M.INK2, lw=0.8)
ax.bar([0], [delta], width=0.34, color=M.C["green"], edgecolor="white",
       linewidth=0.6, zorder=3)
ax.text(0, delta + 0.0004, f"+{delta:.4f}", ha="center", va="bottom",
        fontsize=6.8, color=M.INK)
ax.set_xlim(-0.55, 0.55)
ax.set_xticks([0])
ax.set_xticklabels(["Compact Baseline\n$-$ End-to-End\nNeural Baseline"], fontsize=6.2)
ax.set_ylabel("change in dry mass MAPE")
ax.set_title("(c)  Cost of the compact decoder", pad=4)
lim = max(2.4 * sd, abs(delta) * 1.9)
ax.set_ylim(-lim, lim)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=6.2, loc="lower center")
M.despine(ax)

fig.tight_layout()
print(f"  compact minus full dry-mass MAPE: {delta:+.4f}  "
      f"(2 x between-seed SD = {2*sd:.4f})")
M.save(fig, "figure_9.png")
