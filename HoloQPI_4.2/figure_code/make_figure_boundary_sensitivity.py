"""Manuscript figure: model-free sensitivity of the measurements.

(a) Mean absolute relative error in projected area and dry mass (columns
    area_relative_error, mass_relative_error) when every reference boundary is
    displaced by a known number of pixels (upper axis in um).
(b) Ratio of dry-mass error to area error, with its median.
(c) Error of the measurement chain on synthetic fields with analytic mass and
    area: mean over three independently generated sets, error bars = SD.

All three panel titles are placed at the same height: every title uses the
same padding above the axes, and the padding is large enough to clear the
micrometre axis drawn above panel (a).

Inputs (paths relative to the repository root):
    runs/error_propagation_summary.csv                       (scripts/error_propagation.py)
    runs/synthetic_validation_seeds/seed*/synthetic_validation_summary.json
                                                              (scripts/synthetic_validation.py)
Output:
    figure_8.png (and .pdf) in the output directory given by --out.

Usage:
    python make_figure_boundary_sensitivity.py --root <repo root> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PIXEL_UM = 0.284871          # pixel pitch dx = dy, config: optics.pixel_pitch_x_um
C_AREA = "#D55E00"           # projected area
C_MASS = "#0072B2"           # dry mass
C_RATIO = "#882255"          # mass/area ratio
C_FIELD = "#009E73"          # field-total bar
TITLE_PAD = 40               # points above each axes; same for all panels


def read_propagation(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    order = np.argsort(cols["shift_px"])
    return {k: v[order] for k, v in cols.items()}


def read_synthetic(root: Path) -> dict[str, tuple[float, float]]:
    keys = {
        "per-cell\ndry mass": "mass_abs_relative_error_mean",
        "per-cell\nprojected area": "area_abs_relative_error_mean",
        "field-total\ndry mass": "field_total_mass_abs_error_mean",
    }
    files = sorted(root.glob("seed*/synthetic_validation_summary.json"))
    if len(files) == 3:
        runs = [json.loads(f.read_text()) for f in files]
    else:  # same three sets, as collected by scripts/collect_benchmark_results.py
        coll = root.parent / "benchmark_results" / "results_synthetic_validation.json"
        runs = [r["metrics"] for r in json.loads(coll.read_text())["runs"]]
        if len(runs) != 3:
            raise SystemExit(f"expected 3 synthetic sets, found {len(runs)}")
    out = {}
    for label, key in keys.items():
        values = 100 * np.array([r[key] for r in runs])
        out[label] = (float(values.mean()), float(values.std(ddof=1)))
    return out


def style() -> None:
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.8,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = ap.parse_args()

    prop = read_propagation(args.root / "runs" / "error_propagation_summary.csv")
    synth = read_synthetic(args.root / "runs" / "synthetic_validation_seeds")

    style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2),
                             gridspec_kw={"wspace": 0.38})
    shift = prop["shift_px"]

    # (a) boundary error propagation
    ax = axes[0]
    ax.axvline(0.0, color="0.55", lw=1.0)
    ax.plot(shift, 100 * prop["area_relative_error"], "o-", color=C_AREA, lw=2.2,
            ms=7, mec="white", label="projected area")
    ax.plot(shift, 100 * prop["mass_relative_error"], "s-", color=C_MASS, lw=2.2,
            ms=7, mec="white", label="dry mass")
    ax.set_xlabel("boundary displacement [px]")
    ax.set_ylabel("relative error [%]")
    ax.legend(loc="upper center")
    top = ax.secondary_xaxis("top", functions=(lambda v: v * PIXEL_UM,
                                               lambda v: v / PIXEL_UM))
    top.set_xlabel(r"displacement [$\mu$m]")
    top.set_xticks(np.arange(-1.5, 1.51, 0.5))
    ax.set_title("(a)  Boundary error propagation", pad=TITLE_PAD)

    # (b) exchange rate
    ax = axes[1]
    finite = np.isfinite(prop["mass_over_area"])
    median = float(np.median(prop["mass_over_area"][finite]))
    ax.axhline(1.0, color="0.55", lw=1.5)
    ax.text(-5.2, 1.015, "equal sensitivity", color="0.35", va="bottom")
    ax.axhline(median, color=C_RATIO, lw=1.5, ls="--")
    ax.text(-5.2, median - 0.015, f"median {median:.3f}", color=C_RATIO, va="top")
    ax.plot(shift[finite], prop["mass_over_area"][finite], "o-", color=C_RATIO,
            lw=2.2, ms=7, mec="white")
    ax.set_ylim(0.35, 1.12)
    ax.set_xlabel("boundary displacement [px]")
    ax.set_ylabel("mass error / area error")
    ax.set_title("(b)  Exchange rate", pad=TITLE_PAD)

    # (c) measurement-chain floor
    ax = axes[2]
    labels = list(synth)
    means = [synth[k][0] for k in labels]
    sds = [synth[k][1] for k in labels]
    bars = ax.bar(labels, means, yerr=sds, capsize=6, width=0.6,
                  color=[C_MASS, C_AREA, C_FIELD],
                  error_kw={"elinewidth": 1.6, "capthick": 1.6})
    for bar, m, s in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, (m + s) * 1.25, f"{m:.3f}%",
                ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(min(means) / 2.0, max(means) * 5.0)
    ax.set_ylabel("mean |error| vs analytic value [%]")
    ax.grid(axis="x", visible=False)
    ax.set_title("(c)  Measurement-chain floor", pad=TITLE_PAD)

    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "figure_8.png")
    fig.savefig(args.out / "figure_8.pdf")
    print("wrote", args.out / "figure_8.png")


if __name__ == "__main__":
    main()
