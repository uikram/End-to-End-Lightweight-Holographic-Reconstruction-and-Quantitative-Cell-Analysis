"""Research figures for the off-axis versus in-line Gabor study.

Every figure here answers a question the paper has to answer. None of them are
decorative restatements of a table. The rationale for each, and whether it is
essential or supplementary, is documented in docs/documentation.md under
"Figures"; the one-line version is repeated in each function's docstring.

Most figures are built from files that `compare`, `diagnose_bias`,
`audit_labels` and `benchmark` have already written, so they cost nothing to
regenerate. The three that need a forward pass (qualitative panel, phase error
structure) load the checkpoint and are skipped automatically when it is absent.

    python scripts/make_figures.py --config config/base.yaml
    python scripts/make_figures.py --config config/base.yaml --only 2 4 5
    python scripts/make_figures.py --config config/base.yaml --list

Output: figures/fig<NN>_<name>.png and .pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from holoqpi.config import load_config, parse_overrides
from holoqpi.utils import (
    get_logger,
    pooled_between_seed_sd,
    resolve_device,
    run_directory,
    write_json,
)

LOGGER = get_logger(__name__)

_SEED_RUN = re.compile(r"_(?:seed|s)\d+$")


def comparison_files(output_root: Path) -> list[Path]:
    """Modality-comparison JSONs that are ABLATION ARMS, not seed replicates.

    ``glob("*_modality_comparison.json")`` also matches the seed-replication
    runs, which run_study.sh names ``<arm>_s<seed>`` and run_v2.sh names
    ``v2_<arm>_seed<seed>``. Figures 7 and 13 plotted each of those as a
    SEPARATE ABLATION ROW, so pure seed noise appeared as an ablation effect with
    nothing in the figure to say so -- which is the exact opposite of what the
    seed replication exists to establish. The replicates belong to
    scripts/aggregate_seeds.py, which pools them.
    """
    return [
        path for path in sorted(output_root.glob("*_modality_comparison.json"))
        if not _SEED_RUN.search(path.name.replace("_modality_comparison.json", ""))
    ]


# One colour per modality, used everywhere, so a reader can follow an arm across
# figures without re-reading a legend.
COLOUR = {"off_axis": "#1b6ca8", "gabor": "#d1495b"}
LABEL = {"off_axis": "Off-axis", "gabor": "In-line (Gabor)"}
GREY = "#4a4a4a"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
    })


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
#: Figures that actually wrote a file during this run, with their provenance.
#: `main` appended a figure number to `built` whenever its function RETURNED,
#: and every figure returns normally when it skips for a missing input -- so a
#: run that produced nothing still reported a list of "built" figures. The
#: summary then printed a directory listing of fig*.png, which also included
#: PNGs left over from previous runs. Recording the writes themselves is the
#: only thing that distinguishes what this run made from what was already there.
_WRITTEN: dict[int, dict] = {}


def save(fig, out_dir: Path, number: int, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"fig{number:02d}_{name}"
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    LOGGER.info("wrote %s.png / .pdf", stem)
    _WRITTEN[number] = {"name": name, "stem": str(stem)}
    return stem


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def run_dirs(cfg, modalities) -> dict[str, Path]:
    return {
        m: run_directory(cfg.paths.output_root, cfg.experiment_name, m) for m in modalities
    }


def ordered(args, mapping: dict) -> list:
    """Modalities in the order the user asked for, so every figure agrees.

    ``sorted()`` would put gabor before off_axis, which silently flips the
    left-right arrangement between figures and makes them harder to read
    together.
    """
    keys = [m for m in args.modalities if m in mapping]
    return [(k, mapping[k]) for k in keys] + [
        (k, v) for k, v in sorted(mapping.items()) if k not in keys
    ]


def skip(number: int, name: str, reason: str) -> None:
    LOGGER.warning("figure %02d (%s) skipped: %s", number, name, reason)


def _percentile_limits(*arrays, low=1.0, high=99.0, pad=0.08):
    values = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [low, high])
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span


# ===========================================================================
# FIGURE 1 -- qualitative reconstruction panel                    [ESSENTIAL]
# ===========================================================================
def figure_qualitative(cfg, args, out_dir: Path) -> None:
    """What the two hologram types look like and what the model recovers.

    Essential. The entire study is a comparison of two encodings of the same
    field, and no table conveys what distinguishes them: an off-axis hologram
    carries the object in a separated sideband, an in-line Gabor hologram
    carries it superposed with its twin image. This panel shows the raw input,
    the recovered phase, the signed phase error and the segmentation boundaries
    for both arms on the same field of view, so the reader can see the failure
    mode the numbers later quantify.
    """
    import torch

    from holoqpi.data import build_dataloaders
    from holoqpi.engine import load_checkpoint
    from holoqpi.models import build_model

    device = resolve_device(args.device)
    modalities = list(args.modalities)
    panels = {}

    for modality in modalities:
        modality_cfg = cfg.merged({"data": {"modality": modality}})
        directory = run_directory(
            modality_cfg.paths.output_root, modality_cfg.experiment_name, modality
        )
        checkpoint = directory / "best_model.pt"
        if not checkpoint.is_file():
            skip(1, "qualitative", f"no checkpoint at {checkpoint}")
            return

        model = build_model(modality_cfg)
        load_checkpoint(model, checkpoint, device)
        model.to(device).eval()

        loader = build_dataloaders(modality_cfg, splits_to_build=(args.split,))[args.split]
        with torch.no_grad():
            batch = next(iter(loader))
            outputs = model(batch["hologram"].to(device))
        index = min(args.example_index, batch["hologram"].shape[0] - 1)
        panels[modality] = {
            "hologram": batch["hologram"][index, 0].numpy(),
            "phase_gt": batch["phase"][index, 0].numpy(),
            "phase_pred": outputs["phase"][index, 0].float().cpu().numpy(),
            "mask_gt": batch["mask"][index].numpy() > 0,
            "mask_pred": outputs["segmentation"][index].argmax(0).cpu().numpy() > 0,
            "stem": batch["stem"][index],
        }

    phase_lo, phase_hi = _percentile_limits(
        *[p["phase_gt"] for p in panels.values()], low=0.5, high=99.5, pad=0.0
    )
    error_span = max(
        np.percentile(np.abs(p["phase_pred"] - p["phase_gt"]), 99) for p in panels.values()
    )

    fig, axes = plt.subplots(
        len(modalities), 5, figsize=(13.5, 3.0 * len(modalities)), squeeze=False
    )
    titles = ["Raw hologram", "Reference phase (rad)", "Reconstructed phase (rad)",
              "Phase error (rad)", "Segmentation"]

    for row, modality in enumerate(modalities):
        panel = panels[modality]
        error = panel["phase_pred"] - panel["phase_gt"]

        images = [
            (panel["hologram"], "gray", None, None),
            (panel["phase_gt"], "viridis", phase_lo, phase_hi),
            (panel["phase_pred"], "viridis", phase_lo, phase_hi),
            (error, "RdBu_r", -error_span, error_span),
        ]
        for col, (data, cmap, vmin, vmax) in enumerate(images):
            ax = axes[row][col]
            handle = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row == 0:
                ax.set_title(titles[col])
            bar = fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
            bar.ax.tick_params(labelsize=6)

        ax = axes[row][4]
        ax.imshow(panel["phase_gt"], cmap="gray", vmin=phase_lo, vmax=phase_hi)
        ax.contour(panel["mask_gt"], levels=[0.5], colors="#2ecc71", linewidths=0.8)
        ax.contour(panel["mask_pred"], levels=[0.5], colors="#e67e22", linewidths=0.8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if row == 0:
            ax.set_title(titles[4])
            ax.text(0.5, -0.06, "green = reference,  orange = predicted",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)

        axes[row][0].set_ylabel(LABEL[modality], fontsize=10, labelpad=8,
                                color=COLOUR[modality], fontweight="bold")

    fig.suptitle(f"Reconstruction and segmentation on the same field "
                 f"({panels[modalities[0]]['stem']})", y=1.01)
    save(fig, out_dir, 1, "qualitative_panel")


# ===========================================================================
# FIGURE 2 -- Bland-Altman agreement                              [ESSENTIAL]
# ===========================================================================
def figure_bland_altman(cfg, args, out_dir: Path) -> None:
    """Whether the measurements are usable, not merely correlated.

    Essential. The brief asks for measurement-ready phase, and correlation does
    not establish that: two methods can correlate at r = 0.99 and still disagree
    by 30% on every cell. Bland-Altman is the standard test of agreement between
    a candidate instrument and a reference, and it exposes what a scatter plot
    hides -- a constant offset, and error that grows with cell size. The limits
    of agreement are the number a biologist needs in order to decide whether the
    method can resolve the mass differences their experiment is about.
    """
    directories = run_dirs(cfg, args.modalities)
    quantities = [
        ("dry_mass_pg", "Dry mass", "pg"),
        ("area_um2", "Projected area", "um^2"),
    ]

    available = {
        m: read_csv_rows(d / f"per_cell_{args.split}.csv") for m, d in directories.items()
    }
    available = {m: rows for m, rows in available.items() if rows}
    if not available:
        skip(2, "bland_altman", "no per_cell CSV found")
        return

    fig, axes = plt.subplots(len(quantities), len(available),
                             figsize=(5.2 * len(available), 3.6 * len(quantities)),
                             squeeze=False)

    for row, (key, title, unit) in enumerate(quantities):
        for col, (modality, rows) in enumerate(ordered(args, available)):
            ax = axes[row][col]
            predicted = column(rows, f"{key}_pred")
            reference = column(rows, f"{key}_ref")
            valid = np.isfinite(predicted) & np.isfinite(reference)
            predicted, reference = predicted[valid], reference[valid]
            if predicted.size == 0:
                ax.set_visible(False)
                continue

            mean_value = (predicted + reference) / 2.0
            relative = 100.0 * (predicted - reference) / mean_value
            bias = float(np.mean(relative))
            spread = float(np.std(relative, ddof=1))
            lower, upper = bias - 1.96 * spread, bias + 1.96 * spread

            ax.scatter(mean_value, relative, s=5, alpha=0.25,
                       color=COLOUR.get(modality, GREY), edgecolors="none", rasterized=True)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(bias, color=COLOUR.get(modality, GREY), lw=1.3)
            for limit in (lower, upper):
                ax.axhline(limit, color=COLOUR.get(modality, GREY), lw=1.0, ls="--")
            ax.fill_between(
                [np.min(mean_value), np.max(mean_value)], lower, upper,
                color=COLOUR.get(modality, GREY), alpha=0.06,
            )
            ax.text(0.98, 0.03,
                    f"bias {bias:+.1f}%\n95% LoA [{lower:+.0f}, {upper:+.0f}]%\nn = {predicted.size}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))
            ax.set_ylim(*_percentile_limits(relative, low=0.5, high=99.5, pad=0.15))
            ax.set_xlabel(f"Mean of predicted and reference ({unit})")
            if col == 0:
                ax.set_ylabel(f"{title}\nrelative difference (%)")
            if row == 0:
                ax.set_title(LABEL.get(modality, modality))

    fig.suptitle("Bland-Altman agreement per matched cell", y=1.0)
    save(fig, out_dir, 2, "bland_altman")


# ===========================================================================
# FIGURE 3 -- where the dry-mass error comes from                 [ESSENTIAL]
# ===========================================================================
def figure_error_decomposition(cfg, args, out_dir: Path) -> None:
    """Boundary error or reconstruction error: which one costs the mass.

    Essential. A dry-mass error is a number; this figure is the reason for it.
    Mass is the phase integral over the segmented domain, so the ratio to the
    reference factorises exactly into a domain term (true phase, predicted
    boundary) and a phase term (predicted phase, predicted boundary). Plotting
    the two against each other places every image relative to the unit point and
    to the iso-mass diagonal, and it settles the only actionable question the
    error raises: whether to work on the segmentation head or the phase head.
    """
    directories = run_dirs(cfg, args.modalities)
    data = {
        m: read_csv_rows(d / f"bias_diagnosis_{args.split}.csv")
        for m, d in directories.items()
    }
    data = {m: rows for m, rows in data.items() if rows}
    if not data:
        skip(3, "error_decomposition", "no bias_diagnosis CSV (run scripts/diagnose_bias.py)")
        return

    fig = plt.figure(figsize=(11.5, 4.2))
    grid = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)
    scatter_ax = fig.add_subplot(grid[0])
    bar_ax = fig.add_subplot(grid[1])
    hist_ax = fig.add_subplot(grid[2])

    for modality, rows in ordered(args, data):
        domain = column(rows, "domain_factor")
        phase = column(rows, "phase_factor")
        valid = np.isfinite(domain) & np.isfinite(phase)
        scatter_ax.scatter(domain[valid], phase[valid], s=16, alpha=0.55,
                           color=COLOUR.get(modality, GREY), edgecolors="none",
                           label=LABEL.get(modality, modality))

    limits = _percentile_limits(
        *[column(r, "domain_factor") for r in data.values()],
        *[column(r, "phase_factor") for r in data.values()], low=1, high=99, pad=0.12,
    )
    span = np.linspace(max(limits[0], 1e-3), limits[1], 100)
    for level, style_ in ((1.0, "-"), (0.9, ":"), (1.1, ":")):
        scatter_ax.plot(span, level / span, color="k", lw=0.7, ls=style_, alpha=0.5)
    scatter_ax.axvline(1.0, color="k", lw=0.6, alpha=0.4)
    scatter_ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    scatter_ax.set_xlim(*limits); scatter_ax.set_ylim(*limits)
    scatter_ax.set_xlabel("Domain factor  (true phase, predicted boundary)")
    scatter_ax.set_ylabel("Phase factor  (predicted phase, predicted boundary)")
    scatter_ax.set_title("Mass ratio factorised\nsolid line = correct mass")
    scatter_ax.legend(loc="upper right")

    # Geometric means: the only summary for which domain x phase = total exactly.
    def geometric(values):
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.exp(np.mean(np.log(values)))) if values.size else np.nan

    names = [m for m, _ in ordered(args, data)]
    width = 0.35
    positions = np.arange(3)
    # One offset per modality, computed rather than hardcoded to two. With a
    # fixed 2-tuple a third modality appeared in this figure's other two panels
    # and silently vanished from this one, and a single modality sat off-centre.
    offsets = (
        np.zeros(1) if len(names) == 1
        else np.linspace(-width / 2, width / 2, len(names))
    )
    for offset, modality in zip(offsets, names):
        rows = data[modality]
        factors = [
            geometric(column(rows, "domain_factor")),
            geometric(column(rows, "phase_factor")),
            geometric(column(rows, "mass_ratio")),
        ]
        bar_ax.bar(positions + offset, [100 * (f - 1) for f in factors], width,
                   color=COLOUR.get(modality, GREY), label=LABEL.get(modality, modality))
    bar_ax.axhline(0, color="k", lw=0.8)
    bar_ax.set_xticks(positions)
    bar_ax.set_xticklabels(["domain\n(boundary)", "phase\n(recon)", "total\n(mass)"])
    bar_ax.set_ylabel("Contribution to mass error (%)")
    bar_ax.set_title("Attribution, geometric mean")
    bar_ax.legend()

    for modality, rows in ordered(args, data):
        intra = column(rows, "intra_cell_phase_bias_rad")
        background = column(rows, "background_phase_bias_rad")
        for values, style_, tag in ((intra, "-", "inside cells"),
                                    (background, "--", "background")):
            values = values[np.isfinite(values)]
            if values.size < 2:
                continue
            hist_ax.hist(values, bins=25, histtype="step", ls=style_, density=True,
                         color=COLOUR.get(modality, GREY),
                         label=f"{LABEL.get(modality, modality)}, {tag}")
    hist_ax.axvline(0, color="k", lw=0.8)
    hist_ax.set_xlabel("Phase bias (rad)")
    hist_ax.set_ylabel("Density")
    hist_ax.set_title("Where the phase error sits\nmass integrates only the inside")
    hist_ax.legend(fontsize=7)

    save(fig, out_dir, 3, "error_decomposition")


# ===========================================================================
# FIGURE 4 -- the detection gap                                   [ESSENTIAL]
# ===========================================================================
def figure_detection(cfg, args, out_dir: Path) -> None:
    """Which cells never reach the measurement, and why that biases the field.

    Essential. Every per-cell measurement statistic is computed over IoU-matched
    cells, so it says nothing about cells that were missed -- and missed cells
    are the dominant error in whole-field mass. This figure closes that gap: the
    detection cascade shows how many reference cells survive to a match, and the
    size distributions show whether the misses are small, faint cells (a
    threshold effect) or large ones (a merging effect). It converts "61% recall"
    into a statement about which biology the method loses.
    """
    directories = run_dirs(cfg, args.modalities)
    matched = {m: read_csv_rows(d / f"per_cell_{args.split}.csv") for m, d in directories.items()}
    unmatched = {m: read_csv_rows(d / f"unmatched_{args.split}.csv") for m, d in directories.items()}
    metrics = {m: read_json(d / f"metrics_{args.split}.json") for m, d in directories.items()}
    present = [m for m in args.modalities if matched.get(m)]
    if not present:
        skip(4, "detection", "no per_cell CSV found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    cascade_ax, size_ax, iou_ax = axes

    width = 0.35
    positions = np.arange(3)
    for offset, modality in zip(np.linspace(-width / 2, width / 2, len(present)), present):
        stats = metrics.get(modality) or {}
        counts = [stats.get("cells_reference", np.nan),
                  stats.get("cells_detected", np.nan),
                  stats.get("cells_matched", len(matched[modality]))]
        bars = cascade_ax.bar(positions + offset, counts, width,
                              color=COLOUR.get(modality, GREY),
                              label=LABEL.get(modality, modality))
        for rect, value in zip(bars, counts):
            if np.isfinite(value):
                cascade_ax.text(rect.get_x() + rect.get_width() / 2, value,
                                f"{int(value)}", ha="center", va="bottom", fontsize=7)
        recall = stats.get("detection_recall")
        precision = stats.get("detection_precision")
        if recall is not None and precision is not None:
            cascade_ax.text(0.02, 0.03 + 0.075 * present.index(modality),
                            f"{LABEL.get(modality, modality)}: recall {recall:.0%}, "
                            f"precision {precision:.0%}",
                            transform=cascade_ax.transAxes, fontsize=7.5,
                            color=COLOUR.get(modality, GREY), va="bottom")
    cascade_ax.set_xticks(positions)
    cascade_ax.set_xticklabels(["reference\ncells", "detected\ncells", "matched\npairs"])
    cascade_ax.set_ylabel("Cells in the test split")
    cascade_ax.margins(y=0.18)
    cascade_ax.set_title("Detection cascade")
    cascade_ax.legend(loc="upper right")

    # Are the missed cells systematically different from the matched ones?
    for modality in present:
        matched_area = column(matched[modality], "area_um2_ref")
        rows = unmatched.get(modality) or []
        missed_area = np.asarray(
            [float(r["area_um2"]) for r in rows
             if r.get("kind") == "missed_reference" and r.get("area_um2")], dtype=float
        )
        bins = np.logspace(np.log10(max(np.nanmin(matched_area), 1.0)),
                           np.log10(np.nanmax(matched_area)), 30)
        size_ax.hist(matched_area[np.isfinite(matched_area)], bins=bins, density=True,
                     histtype="step", color=COLOUR.get(modality, GREY),
                     label=f"{LABEL.get(modality, modality)}: matched")
        if missed_area.size:
            size_ax.hist(missed_area, bins=bins, density=True, histtype="stepfilled",
                         alpha=0.25, color=COLOUR.get(modality, GREY),
                         label=f"{LABEL.get(modality, modality)}: missed")
    size_ax.set_xscale("log")
    size_ax.set_xlabel("Reference cell area (um^2)")
    size_ax.set_ylabel("Density")
    size_ax.set_title("Size of missed vs matched cells")
    size_ax.legend(fontsize=7)

    for modality in present:
        ious = column(matched[modality], "match_iou")
        ious = ious[np.isfinite(ious)]
        if ious.size:
            iou_ax.hist(ious, bins=np.linspace(0, 1, 41), histtype="step", density=True,
                        color=COLOUR.get(modality, GREY), label=LABEL.get(modality, modality))
    threshold = cfg.evaluation.measurement.match_iou_threshold
    iou_ax.axvline(threshold, color="k", ls="--", lw=0.9)
    iou_ax.text(threshold, iou_ax.get_ylim()[1] * 0.55, f"  matching\n  threshold {threshold}",
                fontsize=7, va="center")
    iou_ax.set_xlabel("IoU of matched pairs")
    iou_ax.set_ylabel("Density")
    iou_ax.set_title("Quality of the matches that were made")
    iou_ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, 4, "detection_gap")


# ===========================================================================
# FIGURE 5 -- the head-to-head comparison                         [ESSENTIAL]
# ===========================================================================
def figure_modality_summary(cfg, args, out_dir: Path) -> None:
    """The central contribution, on one page.

    Essential. The brief names seven axes on which the two hologram types are to
    be compared. This figure reports all seven side by side with a consistent
    orientation -- every bar is drawn so that longer is better -- so a reader can
    see at once which modality wins where, instead of reconciling a table in
    which some metrics improve upward and others downward.
    """
    output_root = Path(cfg.paths.output_root)
    payload = read_json(output_root / f"{cfg.experiment_name}_modality_comparison.json")
    if not payload:
        skip(5, "modality_summary", "no *_modality_comparison.json")
        return

    # (metric key, display name, lower_is_better)
    axes_spec = [
        ("phase_pearson_r", "Phase correlation r", False),
        ("phase_ssim", "Phase SSIM", False),
        ("phase_mae_rad", "Phase MAE (rad)", True),
        ("phase_mae_rad_in_cell", "Phase MAE in cells (rad)", True),
        ("seg_dice", "Segmentation Dice", False),
        ("seg_aji", "Instance AJI", False),
        ("seg_boundary_f1", "Boundary F1", False),
        ("detection_f1", "Detection F1", False),
        ("cls_accuracy", "Condition accuracy", False),
        ("cls_macro_f1", "Condition macro F1", False),
        ("area_mape", "Area MAPE", True),
        ("optical_volume_mape", "Optical volume MAPE", True),
        ("dry_mass_mape", "Dry mass MAPE", True),
    ]
    # `payload[m] or {}` everywhere below, because a comparison JSON can carry
    # `"gabor": null` for an arm that was never evaluated -- and a bare
    # `.get` on None is an AttributeError, not a skip. Figures 7 and 13 already
    # guarded this; figure 5 did not.
    modalities = [m for m in args.modalities if isinstance(payload.get(m), dict)]
    if len(modalities) < 1:
        skip(5, "modality_summary",
             "comparison JSON holds no populated entry for any requested modality")
        return

    usable = [
        spec for spec in axes_spec
        if any(isinstance(payload[m].get(spec[0]), (int, float)) for m in modalities)
    ]
    if not usable:
        skip(5, "modality_summary", "no requested metric is present for either modality")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(usable) + 2.0),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    raw_ax, delta_ax = axes

    positions = np.arange(len(usable))[::-1]
    height = 0.36
    for offset, modality in zip(np.linspace(-height / 2, height / 2, len(modalities)), modalities):
        values, labels = [], []
        for key, name, lower_better in usable:
            value = payload[modality].get(key)
            values.append(float(value) if isinstance(value, (int, float)) else np.nan)
            labels.append(name + (" (lower better)" if lower_better else ""))
        # Normalise each metric to the better of the two arms so the bars share
        # one axis without pretending the units are comparable.
        normalised = []
        for (key, _, lower_better), value in zip(usable, values):
            others = [payload[m].get(key) for m in modalities]
            others = [float(o) for o in others if isinstance(o, (int, float)) and np.isfinite(o)]
            if not others or not np.isfinite(value):
                normalised.append(np.nan)
            elif lower_better:
                normalised.append(min(others) / value if value else np.nan)
            else:
                normalised.append(value / max(others) if max(others) else np.nan)
        raw_ax.barh(positions + offset, normalised, height,
                    color=COLOUR.get(modality, GREY), label=LABEL.get(modality, modality))
        for y, value, norm in zip(positions + offset, values, normalised):
            if np.isfinite(value) and np.isfinite(norm):
                raw_ax.text(norm + 0.01, y, f"{value:.3f}", va="center", fontsize=6.8)

    raw_ax.set_yticks(positions)
    raw_ax.set_yticklabels([name for _, name, _ in usable])
    raw_ax.set_xlim(0, 1.22)
    raw_ax.set_xlabel("Relative to the better arm (1.0 = best; longer is better)")
    raw_ax.set_title("Off-axis versus in-line Gabor, all reported axes\n"
                     "annotations are the raw values")
    raw_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)

    if len(modalities) == 2:
        first, second = modalities
        deltas, colours = [], []
        for key, _, lower_better in usable:
            a, b = payload[first].get(key), payload[second].get(key)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or b == 0:
                deltas.append(np.nan); colours.append(GREY); continue
            relative = 100.0 * (a - b) / abs(b)
            if lower_better:
                relative = -relative                 # positive always favours `first`
            deltas.append(relative)
            colours.append(COLOUR[first] if relative > 0 else COLOUR[second])
        delta_ax.barh(positions, deltas, 0.6, color=colours)
        delta_ax.axvline(0, color="k", lw=0.9)
        delta_ax.set_yticks(positions); delta_ax.set_yticklabels([])
        delta_ax.set_xlabel(f"% advantage of {LABEL[first]} over {LABEL[second]}")
        delta_ax.set_title("Signed advantage\n(positive favours off-axis)")
    else:
        delta_ax.set_visible(False)

    fig.tight_layout()
    save(fig, out_dir, 5, "modality_comparison")


# ===========================================================================
# FIGURE 6 -- what the objective actually optimised               [ESSENTIAL]
# ===========================================================================
def figure_loss_components(cfg, args, out_dir: Path) -> None:
    """Why the physics terms could not have changed the outcome.

    Essential, because the paper reports a negative result and this is its
    evidence. The composite objective is a weighted sum, and the share each term
    holds over training decides what the optimiser was in a position to trade.
    Plotting the shares shows the physics-consistency group collapsing to a few
    percent of the total, and the phase-mask contrast hinge reaching exactly zero
    early -- which is what a null ablation looks like from the inside, and it
    turns "removing the terms changed nothing" from a surprise into a prediction.
    """
    directories = run_dirs(cfg, args.modalities)
    histories = {
        m: read_json(d / "history.json") for m, d in directories.items()
        if (d / "history.json").is_file()
    }
    histories = {m: h for m, h in histories.items() if h and len(h) > 1}
    if not histories:
        skip(6, "loss_components",
             "no history.json with more than one epoch (a composition plot needs a trajectory)")
        return

    physics = ["phase_mask_contrast", "boundary_gradient_alignment", "phase_volume",
               "dry_mass_consistency", "projected_area_consistency"]
    primary = ["phase", "segmentation", "classification"]

    fig, axes = plt.subplots(2, len(histories), figsize=(6.0 * len(histories), 7.0),
                             squeeze=False)

    for col, (modality, history) in enumerate(ordered(args, histories)):
        epochs = np.array([entry["epoch"] for entry in history], dtype=float)
        keys = [k for k in primary + physics if f"train_{k}" in history[0]]
        weights = cfg.loss.weights
        shares = {}
        for key in keys:
            weight = float(weights.get(key, 1.0))
            shares[key] = weight * np.array(
                [entry.get(f"train_{key}", np.nan) for entry in history], dtype=float
            )
        stack = np.vstack([shares[k] for k in keys])
        total = np.nansum(stack, axis=0)
        fraction = 100.0 * stack / np.where(total > 0, total, np.nan)

        top = axes[0][col]
        top.stackplot(epochs, fraction, labels=[k.replace("_", " ") for k in keys],
                      colors=plt.cm.tab20(np.linspace(0, 1, len(keys))), alpha=0.9)
        top.set_ylim(0, 100)
        top.set_xlim(epochs.min(), epochs.max())
        top.set_ylabel("Share of the weighted objective (%)")
        top.set_title(f"{LABEL.get(modality, modality)}: composition of the loss")
        top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=6.5)

        physics_share = 100.0 * np.nansum(
            [shares[k] for k in keys if k in physics], axis=0
        ) / np.where(total > 0, total, np.nan)
        top.text(0.02, 0.05,
                 f"physics terms at the final epoch: {physics_share[-1]:.1f}% of the objective",
                 transform=top.transAxes, fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.5))

        bottom = axes[1][col]
        for key in keys:
            if key in physics:
                bottom.plot(epochs, shares[key], label=key.replace("_", " "))
        bottom.set_yscale("symlog", linthresh=1e-4)
        bottom.set_xlabel("Epoch")
        bottom.set_ylabel("Weighted term value")
        bottom.set_title("Physics-consistency terms alone (symlog)")
        bottom.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, 6, "loss_composition")


# ===========================================================================
# FIGURE 7 -- ablations                                           [ESSENTIAL]
# ===========================================================================
def figure_ablations(cfg, args, out_dir: Path) -> None:
    """What each part of the objective is worth, stated honestly.

    Essential. An ablation table invites the reader to hunt for the largest
    number; a signed-delta plot with a visible zero line shows immediately that
    most of the deltas straddle it. That is the paper's claim -- the physics
    terms are inert in this setup -- and it should be legible at a glance rather
    than argued from four decimal places.
    """
    output_root = Path(cfg.paths.output_root)
    files = comparison_files(output_root)
    if len(files) < 2:
        skip(7, "ablations", f"need >= 2 comparison JSONs in {output_root}, found {len(files)}")
        return

    baseline_name = cfg.experiment_name
    payloads = {path.name.replace("_modality_comparison.json", ""): read_json(path)
                for path in files}
    if baseline_name not in payloads:
        skip(7, "ablations", f"baseline '{baseline_name}' not among {list(payloads)}")
        return

    metrics = [
        ("seg_dice", "Segmentation Dice", False),
        ("detection_f1", "Detection F1", False),
        ("phase_mae_rad", "Phase MAE", True),
        ("dry_mass_mape", "Dry mass MAPE", True),
        ("area_mape", "Area MAPE", True),
        ("cls_accuracy", "Condition accuracy", False),
    ]
    variants = [name for name in sorted(payloads) if name != baseline_name]

    fig, axes = plt.subplots(1, len(args.modalities),
                             figsize=(6.2 * len(args.modalities), 0.45 * len(metrics) * len(variants) + 2),
                             squeeze=False)

    for col, modality in enumerate(args.modalities):
        ax = axes[0][col]
        rows, labels, colours = [], [], []
        for variant in variants:
            for key, name, lower_better in metrics:
                base = (payloads[baseline_name].get(modality) or {}).get(key)
                other = (payloads[variant].get(modality) or {}).get(key)
                if not (isinstance(base, (int, float)) and isinstance(other, (int, float))) or not base:
                    continue
                delta = 100.0 * (other - base) / abs(base)
                if lower_better:
                    delta = -delta               # positive always means the variant is better
                rows.append(delta)
                labels.append(f"{variant.replace('_', ' ')}  |  {name}")
                colours.append("#2a9d8f" if delta > 0 else "#e76f51")
        if not rows:
            ax.set_visible(False)
            continue
        positions = np.arange(len(rows))[::-1]
        ax.barh(positions, rows, 0.65, color=colours)
        ax.axvline(0, color="k", lw=1.0)
        ax.axvspan(-2, 2, color="0.85", alpha=0.5, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("% change relative to the full objective\n(positive = the ablation is better)")
        ax.set_title(f"{LABEL.get(modality, modality)}\ngrey band = +/- 2%, within run-to-run noise")

    fig.tight_layout()
    save(fig, out_dir, 7, "ablation_deltas")


# ===========================================================================
# FIGURE 8 -- accuracy against cost                               [ESSENTIAL]
# ===========================================================================
def figure_efficiency(cfg, args, out_dir: Path) -> None:
    """Whether the thing can actually be deployed, and at what accuracy.

    Essential. Edge suitability is one of the axes the brief asks for, and it is
    a trade-off rather than a score: latency alone says nothing without the
    accuracy it buys. Plotting the configurations in the latency-accuracy plane
    makes the operating points explicit. The runtime marker matters as much as
    the position: an ONNX row timed on CPU cannot be read against a PyTorch row
    timed on the GPU, so rows flagged as not comparable are drawn hollow.
    """
    output_root = Path(cfg.paths.output_root)
    files = sorted(output_root.glob(f"{cfg.experiment_name}_hardware_benchmark_*.csv"))
    if not files:
        skip(8, "efficiency", f"no {cfg.experiment_name}_hardware_benchmark_*.csv")
        return
    rows = [r for path in files for r in read_csv_rows(path)]
    if not rows:
        skip(8, "efficiency", "benchmark CSV is empty")
        return

    metrics = {
        m: read_json(d / f"metrics_{args.split}.json")
        for m, d in run_dirs(cfg, args.modalities).items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    pareto_ax, latency_ax, size_ax = axes

    markers = {"pytorch": "o", "onnx": "s"}
    for row in rows:
        modality = row.get("modality", "")
        runtime = row.get("runtime", "")
        precision = row.get("precision", "")
        try:
            latency = float(row["latency_p50_ms"])
        except (KeyError, ValueError):
            continue
        stats = metrics.get(modality) or {}
        dice = stats.get("seg_dice")
        # COMPARABILITY IS RUNTIME-AWARE, and unknown provenance is not trusted.
        #
        # `comparable_to_pytorch_row` is written only by benchmark_onnx, so a
        # PyTorch row never carries it -- it IS the reference row. An ONNX row
        # is comparable only when it says so; defaulting a missing flag to
        # "True", which is what this did, drew a CSV written before the column
        # existed as fully comparable, the permissive direction on a figure
        # whose own title warns against that comparison.
        if runtime == "pytorch":
            comparable = True
        else:
            flag = row.get("comparable_to_pytorch_row")
            comparable = flag is not None and str(flag).lower() not in ("false", "")
        # A timing measured on a contended device is not a property of the model.
        # Marked here so a contaminated point cannot be read off this figure as
        # an efficiency result. Rows written before the flag existed are treated
        # as unknown rather than stable.
        stability = row.get("timing_stable")
        unstable = stability is not None and str(stability).lower() == "false"
        if dice is None:
            continue
        pareto_ax.scatter(
            latency, float(dice), s=90, marker=markers.get(runtime, "^"),
            facecolor=COLOUR.get(modality, GREY) if comparable else "none",
            edgecolor="#c44e52" if unstable else COLOUR.get(modality, GREY),
            linewidth=2.4 if unstable else 1.4, zorder=3,
        )
        pareto_ax.annotate(
            f"{runtime}/{precision}" + (" UNSTABLE" if unstable else ""),
            (latency, float(dice)),
            textcoords="offset points", xytext=(6, 4), fontsize=6.8,
            color="#c44e52" if unstable else "0.2",
        )
    pareto_ax.set_xscale("log")
    pareto_ax.set_xlabel("Median latency per field (ms, log scale)")
    pareto_ax.set_ylabel("Segmentation Dice")
    pareto_ax.set_title("Accuracy against inference cost\nhollow = not comparable to the "
                        "PyTorch row; red edge = contended device")

    labels = [f"{r.get('modality','')[:8]}\n{r.get('runtime','')}/{r.get('precision','')}"
              for r in rows]
    p50 = column(rows, "latency_p50_ms")
    p99 = column(rows, "latency_p99_ms")
    positions = np.arange(len(rows))
    colours = [COLOUR.get(r.get("modality", ""), GREY) for r in rows]
    latency_ax.bar(positions, p50, 0.6, color=colours, label="median")
    latency_ax.errorbar(positions, p50, yerr=np.maximum(p99 - p50, 0), fmt="none",
                        ecolor="k", elinewidth=0.9, capsize=3, label="to p99")
    latency_ax.set_xticks(positions)
    latency_ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    latency_ax.set_ylabel("Latency (ms)")
    latency_ax.set_title("Median and tail latency\ntail matters for a real-time pipeline")
    latency_ax.legend(fontsize=7)

    params = column(rows, "params_total") / 1e6
    gmacs = column(rows, "gmacs")
    if np.isfinite(gmacs).any():
        size_ax.scatter(gmacs, p50, s=80, color=colours, zorder=3)
        for x, y, label in zip(gmacs, p50, labels):
            if np.isfinite(x):
                size_ax.annotate(label.replace("\n", " "), (x, y), fontsize=6.5,
                                 textcoords="offset points", xytext=(5, 3))
        size_ax.set_xlabel("Multiply-accumulate cost (GMACs)")
    else:
        size_ax.scatter(params, p50, s=80, color=colours, zorder=3)
        size_ax.set_xlabel("Parameters (millions)")
    size_ax.set_ylabel("Median latency (ms)")
    size_ax.set_title("Compute cost against measured latency\ndeparture from a line = memory bound")

    fig.tight_layout()
    save(fig, out_dir, 8, "efficiency_tradeoff")


# ===========================================================================
# FIGURE 9 -- structure of the phase error                          [OPTIONAL]
# ===========================================================================
def figure_phase_error_structure(cfg, args, out_dir: Path) -> None:
    """Whether the residual error is the kind that survives integration.

    Supplementary but valuable. Dry mass is an integral, so it is insensitive to
    zero-mean high-frequency error and very sensitive to a slowly varying offset
    inside cells. A single MAE cannot distinguish the two. The error-versus-truth
    curve reveals amplitude-dependent bias -- systematic under-recovery of the
    densest material -- and the radial power spectrum shows the spatial scale the
    error lives on. Together they justify, or refuse, the claim that the phase is
    measurement-ready rather than merely close on average.
    """
    import torch

    from holoqpi.data import build_dataloaders
    from holoqpi.engine import load_checkpoint
    from holoqpi.models import build_model

    device = resolve_device(args.device)
    collected = {}

    for modality in args.modalities:
        modality_cfg = cfg.merged({"data": {"modality": modality}})
        directory = run_directory(
            modality_cfg.paths.output_root, modality_cfg.experiment_name, modality
        )
        if not (directory / "best_model.pt").is_file():
            continue
        model = build_model(modality_cfg)
        load_checkpoint(model, directory / "best_model.pt", device)
        model.to(device).eval()

        loader = build_dataloaders(modality_cfg, splits_to_build=(args.split,))[args.split]
        truth, error, spectra = [], [], []
        with torch.no_grad():
            for seen, batch in enumerate(loader):
                outputs = model(batch["hologram"].to(device))
                predicted = outputs["phase"].squeeze(1).float().cpu().numpy()
                reference = batch["phase"].squeeze(1).numpy()
                mask = batch["mask"].numpy() > 0
                for i in range(predicted.shape[0]):
                    residual = predicted[i] - reference[i]
                    inside = mask[i]
                    if inside.any():
                        truth.append(reference[i][inside])
                        error.append(residual[inside])
                    spectra.append(_radial_spectrum(residual))
                if seen + 1 >= args.spectrum_batches:
                    break
        if truth:
            collected[modality] = {
                "truth": np.concatenate(truth),
                "error": np.concatenate(error),
                "spectrum": np.mean(np.vstack(spectra), axis=0),
            }

    if not collected:
        skip(9, "phase_error_structure", "no checkpoints available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    binned_ax, spectrum_ax = axes

    for modality, data in ordered(args, collected):
        truth, error = data["truth"], data["error"]
        edges = np.percentile(truth, np.linspace(0, 100, 26))
        edges = np.unique(edges)
        centres, medians, low, high = [], [], [], []
        for start, end in zip(edges[:-1], edges[1:]):
            selection = (truth >= start) & (truth < end)
            if selection.sum() < 20:
                continue
            centres.append(0.5 * (start + end))
            values = error[selection]
            medians.append(np.median(values))
            low.append(np.percentile(values, 25))
            high.append(np.percentile(values, 75))
        colour = COLOUR.get(modality, GREY)
        binned_ax.plot(centres, medians, color=colour, label=LABEL.get(modality, modality))
        binned_ax.fill_between(centres, low, high, color=colour, alpha=0.18)
    binned_ax.axhline(0, color="k", lw=0.9)
    binned_ax.set_xlabel("Reference phase inside cells (rad)")
    binned_ax.set_ylabel("Reconstruction error (rad)\nmedian, IQR shaded")
    binned_ax.set_title("Amplitude-dependent bias\na downward slope = dense material under-recovered")
    binned_ax.legend()

    for modality, data in ordered(args, collected):
        spectrum = data["spectrum"]
        frequency = np.arange(1, spectrum.size + 1) / (2.0 * spectrum.size)
        spectrum_ax.loglog(frequency, spectrum, color=COLOUR.get(modality, GREY),
                           label=LABEL.get(modality, modality))
    spectrum_ax.set_xlabel("Spatial frequency (cycles / pixel)")
    spectrum_ax.set_ylabel("Error power (radially averaged)")
    spectrum_ax.set_title("Spatial scale of the error\nlow-frequency power is what dry mass inherits")
    spectrum_ax.legend()

    fig.tight_layout()
    save(fig, out_dir, 9, "phase_error_structure")


def _radial_spectrum(image: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.fftshift(np.fft.fft2(image - image.mean()))) ** 2
    height, width = power.shape
    y, x = np.indices((height, width))
    radius = np.hypot(y - height / 2, x - width / 2).astype(int)
    bins = min(height, width) // 2
    total = np.bincount(radius.ravel(), power.ravel(), minlength=bins + 1)[:bins]
    counts = np.bincount(radius.ravel(), minlength=bins + 1)[:bins]
    return total / np.maximum(counts, 1)


# ===========================================================================
# FIGURE 10 -- label provenance                                   [IMPORTANT]
# ===========================================================================
def figure_label_audit(cfg, args, out_dir: Path) -> None:
    """The two properties of the silver-standard labels a reviewer will probe.

    Important. The masks are Otsu thresholds of the same phase map the network
    regresses, and the paper has to say what follows. The left panel shows how
    much the threshold -- the operational definition of "cell" -- moves between
    images and whether it moves with drug condition, which is the route by which
    a label artefact could masquerade as a classification result. The right panel
    shows that thresholding the *predicted* phase reproduces the segmentation
    head, which is the mechanism behind the inert physics terms.
    """
    output_root = Path(cfg.paths.output_root)
    thresholds = read_csv_rows(output_root / "label_audit_thresholds.csv")
    redundancy = read_json(output_root / "label_audit_redundancy.json")
    if not thresholds and not redundancy:
        skip(10, "label_audit", "run scripts/audit_labels.py first")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    threshold_ax, redundancy_ax = axes

    if thresholds:
        conditions = sorted({row["condition"] for row in thresholds})
        grouped = [
            [float(row["otsu_level_rad"]) for row in thresholds if row["condition"] == c]
            for c in conditions
        ]
        parts = threshold_ax.violinplot(grouped, showmedians=True, widths=0.8)
        for body in parts["bodies"]:
            body.set_facecolor("#1b6ca8"); body.set_alpha(0.35)
        for index, values in enumerate(grouped, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.035, len(values))
            threshold_ax.scatter(index + jitter, values, s=8, color=GREY, alpha=0.5,
                                 edgecolors="none", zorder=3)
        everything = np.concatenate([np.asarray(g) for g in grouped])
        threshold_ax.axhline(everything.mean(), color="k", ls="--", lw=0.9)
        threshold_ax.set_xticks(range(1, len(conditions) + 1))
        threshold_ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=7.5)
        threshold_ax.set_ylabel("Otsu threshold (rad)")
        summary = read_json(output_root / "label_audit_thresholds.json") or {}
        note = f"CV = {summary.get('level_cv', float('nan')):.0%}"
        if "anova_p" in summary:
            note += f"   ANOVA p = {summary['anova_p']:.3g}"
        threshold_ax.set_title("Per-image label threshold by condition\n" + note)
    else:
        threshold_ax.set_visible(False)

    if redundancy:
        names = [r["modality"] for r in redundancy]
        positions = np.arange(len(names))
        series = [
            ("dice_head_vs_ground_truth", "seg head vs GT mask", "#8d99ae"),
            ("dice_threshold_of_predicted_phase_vs_ground_truth",
             "threshold(pred phase) vs GT mask", "#457b9d"),
            ("dice_head_vs_threshold_of_predicted_phase",
             "seg head vs threshold(pred phase)", "#e63946"),
        ]
        width = 0.26
        for offset, (key, label, colour) in zip(np.linspace(-width, width, 3), series):
            values = [r[key] for r in redundancy]
            bars = redundancy_ax.bar(positions + offset, values, width, color=colour, label=label)
            for rect, value in zip(bars, values):
                redundancy_ax.text(rect.get_x() + rect.get_width() / 2, value + 0.01,
                                   f"{value:.2f}", ha="center", fontsize=6.8)
        redundancy_ax.set_xticks(positions)
        redundancy_ax.set_xticklabels([LABEL.get(n, n) for n in names])
        redundancy_ax.set_ylim(0, 1.12)
        redundancy_ax.set_ylabel("Dice")
        redundancy_ax.set_title("Are the two heads independent?\n"
                                "a tall red bar means the mask is implied by the phase")
        redundancy_ax.legend(fontsize=7, loc="upper center",
                             bbox_to_anchor=(0.5, -0.12), ncol=1)
    else:
        redundancy_ax.set_visible(False)

    fig.tight_layout()
    save(fig, out_dir, 10, "label_audit")


# ===========================================================================
# FIGURE 11 -- classification structure                           [IMPORTANT]
# ===========================================================================
def figure_confusion(cfg, args, out_dir: Path) -> None:
    """Which drug conditions the model can actually tell apart.

    Important. Off-axis reconstructs phase better yet classifies condition worse,
    and an accuracy number cannot explain that. The confusion matrices show
    whether the gap comes from one collapsed class or from diffuse confusion
    everywhere -- a distinction that decides whether the effect is biological
    (some conditions are genuinely similar in morphology) or an artefact of the
    acquisition batches.
    """
    directories = run_dirs(cfg, args.modalities)
    matrices = {
        m: read_json(d / f"confusion_{args.split}.json") for m, d in directories.items()
    }
    matrices = {m: v for m, v in matrices.items() if v}
    if not matrices:
        skip(11, "confusion", f"no confusion_{args.split}.json")
        return

    names = list(cfg.labels.conditions)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.0 * len(matrices), 4.4), squeeze=False)

    for col, (modality, payload) in enumerate(ordered(args, matrices)):
        matrix = np.asarray(payload["matrix"] if isinstance(payload, dict) else payload, dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalised = matrix / np.where(row_totals > 0, row_totals, 1)

        ax = axes[0][col]
        handle = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{normalised[i, j]:.2f}\n({int(matrix[i, j])})",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if normalised[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if col == 0 else [], fontsize=7)
        ax.set_xlabel("Predicted")
        if col == 0:
            ax.set_ylabel("True")
        accuracy = np.trace(matrix) / matrix.sum() if matrix.sum() else np.nan
        ax.set_title(f"{LABEL.get(modality, modality)}   accuracy {accuracy:.3f}")
        ax.grid(False)
        fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.03).set_label("row-normalised", fontsize=7)

    fig.suptitle("Drug-condition confusion on the test split", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, 11, "confusion_matrices")


# ===========================================================================
# FIGURE 12 -- convergence                                        [IMPORTANT]
# ===========================================================================
def figure_convergence(cfg, args, out_dir: Path) -> None:
    """Whether the schedule was long enough for every head.

    Important, and the reason is specific: the heads converge at different times.
    Segmentation and phase plateau early while condition accuracy is still
    climbing at the last epoch, so a schedule chosen by watching Dice alone would
    under-train the classifier and understate one arm of the comparison. This
    figure is the evidence that the reported numbers were read at a defensible
    point, and it marks the epoch each checkpoint was selected at.
    """
    directories = run_dirs(cfg, args.modalities)
    histories = {
        m: read_json(d / "history.json") for m, d in directories.items()
        if (d / "history.json").is_file()
    }
    if not histories:
        skip(12, "convergence", "no history.json")
        return

    tracked = [
        ("val_seg_dice", "Segmentation Dice", False),
        ("val_phase_mae_rad", "Phase MAE (rad)", True),
        ("val_cls_accuracy", "Condition accuracy", False),
        ("val_dry_mass_mape", "Dry mass MAPE", True),
    ]
    present = [spec for spec in tracked
               if any(spec[0] in h[0] for h in histories.values())]
    if not present:
        skip(12, "convergence", "history.json holds none of the tracked metrics")
        return

    fig, axes = plt.subplots(1, len(present), figsize=(3.5 * len(present), 3.4), squeeze=False)

    for col, (key, name, lower_better) in enumerate(present):
        ax = axes[0][col]
        for modality, history in ordered(args, histories):
            epochs = [e["epoch"] for e in history if key in e]
            values = [e[key] for e in history if key in e]
            if not values:
                continue
            colour = COLOUR.get(modality, GREY)
            ax.plot(epochs, values, color=colour, label=LABEL.get(modality, modality))
            best = int(np.argmin(values) if lower_better else np.argmax(values))
            ax.scatter([epochs[best]], [values[best]], s=34, color=colour, zorder=4)
            ax.annotate(f"ep {epochs[best]}", (epochs[best], values[best]), fontsize=6.5,
                        textcoords="offset points", xytext=(4, -9), color=colour)
        ax.set_xlabel("Epoch")
        ax.set_title(name + ("  (lower better)" if lower_better else ""))
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Validation trajectories; markers show each metric's own best epoch", y=1.04)
    fig.tight_layout()
    save(fig, out_dir, 12, "convergence")



# ===========================================================================
# FIGURE 13 -- forward-model consistency                          [ESSENTIAL]
# ===========================================================================
def figure_forward_model(cfg, args, out_dir: Path) -> None:
    """Whether the reconstruction is consistent with the measurement it came from.

    Essential once the forward-model term exists, because it is the only figure
    that scores a reconstruction without a reference. The residual compares the
    hologram synthesised from the predicted field with the hologram actually
    recorded, so it asks a question ground-truth phase cannot answer and one that
    transfers to data where no ground truth exists.

    Two comparisons carry the meaning. Against the *reference* residual -- the
    same quantity computed from the ground-truth phase -- which is the floor set
    by everything the forward model cannot explain (amplitude, aberration, an
    imperfect z) rather than by any error the network made. And across ablation
    arms, which shows what optimising the term buys, since arms that never
    optimised it are still scored on it.
    """
    output_root = Path(cfg.paths.output_root)
    files = comparison_files(output_root)
    if not files:
        skip(13, "forward_model", "no *_modality_comparison.json")
        return

    payloads = {p.name.replace("_modality_comparison.json", ""): read_json(p) for p in files}
    experiments = [
        name for name in sorted(payloads)
        if any(isinstance((payloads[name].get(m) or {}).get("forward_residual"), (int, float))
               for m in args.modalities)
    ]
    if not experiments:
        skip(13, "forward_model",
             "no run reports forward_residual; set loss.forward_model.distance_um "
             "and re-evaluate")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    bar_ax, floor_ax = axes

    positions = np.arange(len(experiments))
    width = 0.36
    for offset, modality in zip(np.linspace(-width / 2, width / 2, len(args.modalities)),
                                args.modalities):
        values = [
            (payloads[name].get(modality) or {}).get("forward_residual", np.nan)
            for name in experiments
        ]
        bar_ax.bar(positions + offset, values, width,
                   color=COLOUR.get(modality, GREY), label=LABEL.get(modality, modality))
    bar_ax.set_xticks(positions)
    bar_ax.set_xticklabels([n.replace("_", "\n") for n in experiments], fontsize=7)
    bar_ax.set_ylabel("Forward-model residual  (1 - correlation)")
    bar_ax.set_title("Consistency with the recorded hologram\nlower is more physically plausible")
    bar_ax.legend()

    # Prediction against the floor the reference phase itself achieves.
    for modality in args.modalities:
        predicted, reference, labels = [], [], []
        for name in experiments:
            entry = payloads[name].get(modality) or {}
            if isinstance(entry.get("forward_residual_reference"), (int, float)):
                predicted.append(entry["forward_residual"])
                reference.append(entry["forward_residual_reference"])
                labels.append(name)
        if not predicted:
            continue
        floor_ax.scatter(reference, predicted, s=70, color=COLOUR.get(modality, GREY),
                         label=LABEL.get(modality, modality), zorder=3)
        for x, y, name in zip(reference, predicted, labels):
            floor_ax.annotate(name.replace("_", " "), (x, y), fontsize=6.5,
                              textcoords="offset points", xytext=(6, 3))
    limits = floor_ax.get_xlim()
    if limits[1] > limits[0]:
        span = np.linspace(*limits, 10)
        floor_ax.plot(span, span, color="k", lw=0.9, ls="--")
        floor_ax.text(span[-1], span[-1], "  prediction equals\n  the reference floor",
                      fontsize=7, va="center")
    floor_ax.set_xlabel("Residual from the reference phase (the achievable floor)")
    floor_ax.set_ylabel("Residual from the predicted phase")
    floor_ax.set_title("How much of the achievable consistency is reached")
    floor_ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, 13, "forward_model_consistency")


# ===========================================================================
# FIGURE 16 -- what the forward model can and cannot tell        [ESSENTIAL]
# ===========================================================================
def figure_forward_model_diagnostics(cfg, args, out_dir: Path) -> None:
    """The quantitative statement of what physics consistency is worth here.

    Three panels, and together they are the honest answer to the question the
    reference literature raises and this dataset frustrates.

    LEFT, the z scan. The in-line residual has a real minimum because an in-line
    hologram encodes phase only through defocus; the off-axis residual is flat
    because its carrier encodes phase at any distance. The minimum's agreement
    with the distance the acquiring group supplied is the study's independent
    check that the forward operator is physically correct.

    CENTRE, the discrimination test. Six phase variants scored against the
    residual, as a margin over the true phase. A term whose bars are all positive
    is correctly signed; how far above the tolerance they sit is whether it can
    teach anything.

    RIGHT, the calibration probe, which is the sharpest of the three. The
    reference phase is multiplied by s and the residual minimised over s. A
    minimum at s = 1 means the operator reproduces the recorded hologram at the
    phase that was delivered -- the operational definition of measurement-ready.
    A minimum away from 1 is a quantified disagreement about magnitude, and the
    two geometries differ here in a way that is physical rather than incidental.
    """
    payload_path = Path(cfg.paths.output_root) / "z_calibration.json"
    if not payload_path.is_file():
        skip(16, "forward_model_diagnostics",
             "no runs/z_calibration.json; run scripts/calibrate_z.py")
        return
    payload = read_json(payload_path)
    arms = [m for m in args.modalities if m in payload]
    if not arms:
        skip(16, "forward_model_diagnostics", "no modality present in z_calibration.json")
        return

    fig, (scan_ax, margin_ax, scale_ax) = plt.subplots(1, 3, figsize=(15.0, 4.3))

    for modality in arms:
        curve = (payload[modality] or {}).get("curve") or {}
        if not curve.get("distances_um"):
            continue
        distances = np.asarray(curve["distances_um"], dtype=float)
        residual = np.asarray(curve["forward_residual"], dtype=float)
        colour = COLOUR.get(modality, GREY)
        scan_ax.plot(distances, residual, color=colour, lw=1.6,
                     label=LABEL.get(modality, modality))
        scan_ax.axvline(distances[int(np.argmin(residual))], color=colour, lw=0.9, ls=":")
    supplied = cfg.loss.forward_model.distance_um
    if supplied is not None:
        scan_ax.axvline(float(supplied), color="k", lw=1.1, ls="--")
        scan_ax.annotate(f"z = {float(supplied):.2f} um\nfrom the acquiring group",
                         xy=(float(supplied), scan_ax.get_ylim()[1]),
                         xytext=(6, -12), textcoords="offset points", fontsize=7,
                         va="top")
    scan_ax.set_xlabel("Propagation distance z  (um)")
    scan_ax.set_ylabel("Forward-model residual")
    scan_ax.set_title("Is z identifiable from the data?\nin-line yes, off-axis flat by construction")
    scan_ax.legend(fontsize=7)

    variants = ["scaled_0.9", "noisy_0.3", "scaled_0.5", "zero", "mirrored"]
    positions = np.arange(len(variants))
    width = 0.8 / max(len(arms), 1)
    tolerance = None
    for index, modality in enumerate(arms):
        entry = (payload[modality] or {}).get("discrimination") or {}
        margins = entry.get("margins") or {}
        if not margins:
            continue
        tolerance = entry.get("tolerance", tolerance)
        offset = (index - (len(arms) - 1) / 2) * width
        margin_ax.bar(positions + offset,
                      [margins.get(v, np.nan) for v in variants], width,
                      color=COLOUR.get(modality, GREY),
                      label=LABEL.get(modality, modality))
    if tolerance is not None:
        margin_ax.axhline(tolerance, color="k", lw=0.9, ls="--")
        margin_ax.axhline(0.0, color="k", lw=0.8)
        margin_ax.text(len(variants) - 0.5, tolerance, "  tolerance", fontsize=7,
                       va="bottom", ha="right")
    margin_ax.set_xticks(positions)
    margin_ax.set_xticklabels([v.replace("_", " ") for v in variants], fontsize=7.5,
                              rotation=20, ha="right")
    margin_ax.set_ylabel("Residual margin over the true phase")
    margin_ax.set_title("Does a degraded phase score worse?\nabove zero is correctly signed")
    margin_ax.legend(fontsize=7)

    for modality in arms:
        entry = (payload[modality] or {}).get("scale_response") or {}
        if not entry.get("scales"):
            continue
        scales = np.asarray(entry["scales"], dtype=float)
        residual = np.asarray(entry["residual"], dtype=float)
        colour = COLOUR.get(modality, GREY)
        # Each arm's residual has its own offset; what is being compared is where
        # the minimum sits, so each curve is shown relative to its own minimum.
        scale_ax.plot(scales, residual - residual.min(), color=colour, lw=1.6,
                      label=f"{LABEL.get(modality, modality)}  "
                            f"(min at x{entry['best_scale_median']:.2f})")
        scale_ax.axvline(entry["best_scale_median"], color=colour, lw=0.9, ls=":")
    scale_ax.axvline(1.0, color="k", lw=1.1, ls="--")
    scale_ax.text(1.0, scale_ax.get_ylim()[1], " delivered phase", fontsize=7, va="top")
    scale_ax.set_xlabel("Reference phase multiplied by")
    scale_ax.set_ylabel("Residual above each arm's own minimum")
    scale_ax.set_title("Does the operator agree about MAGNITUDE?\na minimum at 1.0 is measurement-ready")
    scale_ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, 16, "forward_model_diagnostics")


# ===========================================================================
# FIGURE 14 -- learned against classical                          [ESSENTIAL]
# ===========================================================================
def figure_conventional_baseline(cfg, args, out_dir: Path) -> None:
    """What the network buys over the textbook reconstruction.

    Essential, and the figure a reviewer looks for first. Every number the study
    reports is uninterpretable without a floor: a Dice of 0.83 is excellent or
    unremarkable depending entirely on what the classical pipeline achieves on
    the same images. Because both are scored by the same evaluator, the same
    instance labelling and the same measurement chain, the gap between them is
    attributable to reconstruction and to nothing else.

    The expected asymmetry is the study's motivation made quantitative: for
    off-axis the classical method is strong, because the sideband separates
    analytically, so the network has to justify itself. For in-line the twin
    image is inseparable and the classical method should fail, which is where
    learning earns its place.
    """
    output_root = Path(cfg.paths.output_root)
    baseline = read_json(output_root / f"conventional_baseline_{args.split}.json")
    learned = read_json(output_root / f"{cfg.experiment_name}_modality_comparison.json")
    if not baseline or not learned:
        skip(14, "conventional_baseline",
             "run scripts/conventional_baseline.py and main.py compare first")
        return

    metrics = [
        ("phase_pearson_r", "Phase correlation r", False),
        ("phase_mae_rad_in_cell", "In-cell phase MAE", True),
        ("seg_dice", "Dice", False),
        ("seg_aji", "Instance AJI", False),
        ("detection_f1", "Detection F1", False),
        ("area_mape", "Area MAPE", True),
        ("dry_mass_mape", "Dry mass MAPE", True),
    ]
    modalities = [m for m in args.modalities if m in baseline and m in learned]
    if not modalities:
        skip(14, "conventional_baseline", "no modality present in both tables")
        return

    fig, axes = plt.subplots(1, len(modalities), figsize=(6.0 * len(modalities), 4.4),
                             squeeze=False)

    for column, modality in enumerate(modalities):
        ax = axes[0][column]
        valid = baseline[modality].get("reconstruction_valid", True)
        rows, labels, clipped = [], [], []
        for key, name, lower_better in metrics:
            a = learned[modality].get(key)
            b = baseline[modality].get(key)
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            # A relative advantage is meaningless when the baseline is ~0: the
            # ratio explodes and one bar swallows the axis. Report those as a
            # capped bar and say so, rather than plotting 10^13 percent.
            floor = 0.02 * max(abs(a), 1e-9)
            if abs(b) < floor:
                advantage, was_clipped = 200.0 if a > b else -200.0, True
            else:
                advantage = 100.0 * (a - b) / abs(b)
                if lower_better:
                    advantage = -advantage
                was_clipped = abs(advantage) > 200.0
                advantage = float(np.clip(advantage, -200.0, 200.0))
            rows.append(advantage)
            clipped.append(was_clipped)
            labels.append(f"{name}\n{a:.3f} vs {b:.3f}")
        if not rows:
            ax.set_visible(False)
            continue

        positions = np.arange(len(rows))[::-1]
        bars = ax.barh(positions, rows, 0.6,
                       color=[COLOUR.get(modality, GREY) if v > 0 else "#8d99ae"
                              for v in rows])
        for rect, value, was_clipped in zip(bars, rows, clipped):
            if was_clipped:
                ax.text(value, rect.get_y() + rect.get_height() / 2,
                        "  capped" if value > 0 else "capped  ",
                        va="center", ha="left" if value > 0 else "right", fontsize=6.5)
        ax.set_xlim(-215, 215)
        ax.axvline(0, color="k", lw=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("% advantage of the network over classical reconstruction")
        title = f"{LABEL.get(modality, modality)}   (learned vs classical)"
        if not valid:
            title += "\nCLASSICAL RECONSTRUCTION FAILED - not a valid baseline"
            ax.set_facecolor("#00000008")
        ax.set_title(title, fontsize=9.5)

    fig.tight_layout()
    save(fig, out_dir, 14, "learned_vs_classical")


# ===========================================================================
# FIGURE 15 -- which cells are lost                               [IMPORTANT]
# ===========================================================================
def figure_recall_by_size(cfg, args, out_dir: Path) -> None:
    """Detection recall as a function of cell size.

    Important, and it pre-empts the strongest objection to the measurement
    results: that dry-mass agreement holds because the model preferentially
    finds large, easy cells. It does. This figure states by how much, and in
    which size range the method should not be trusted, which is exactly the
    information a biologist needs before applying it to a population whose size
    distribution shifts under treatment.
    """
    directories = run_dirs(cfg, args.modalities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    curve_ax, mass_ax = axes
    drawn = False

    for modality in args.modalities:
        matched = read_csv_rows(directories[modality] / f"per_cell_{args.split}.csv")
        unmatched = read_csv_rows(directories[modality] / f"unmatched_{args.split}.csv")
        if not matched:
            continue
        found = column(matched, "area_um2_ref")
        missed = np.asarray(
            [float(r["area_um2"]) for r in unmatched
             if r.get("kind") == "missed_reference" and r.get("area_um2")], dtype=float
        )
        if missed.size == 0:
            continue
        drawn = True

        every = np.concatenate([found[np.isfinite(found)], missed])
        edges = np.percentile(every, np.linspace(0, 100, 13))
        edges = np.unique(edges)
        centres, recall, counts = [], [], []
        for low, high in zip(edges[:-1], edges[1:]):
            in_found = int(((found >= low) & (found < high)).sum())
            in_missed = int(((missed >= low) & (missed < high)).sum())
            if in_found + in_missed < 10:
                continue
            centres.append(np.sqrt(low * high))         # geometric centre, log axis
            recall.append(in_found / (in_found + in_missed))
            counts.append(in_found + in_missed)

        colour = COLOUR.get(modality, GREY)
        curve_ax.plot(centres, recall, marker="o", ms=4, color=colour,
                      label=LABEL.get(modality, modality))
        # Binomial standard error, so a bin with few cells is visibly uncertain.
        error = [np.sqrt(r * (1 - r) / n) for r, n in zip(recall, counts)]
        curve_ax.fill_between(centres, np.array(recall) - 1.96 * np.array(error),
                              np.array(recall) + 1.96 * np.array(error),
                              color=colour, alpha=0.15)

        # What fraction of total reference mass the misses carry, by size.
        found_mass = column(matched, "dry_mass_pg_ref")
        missed_mass = np.asarray(
            [float(r["dry_mass_pg"]) for r in unmatched
             if r.get("kind") == "missed_reference" and r.get("dry_mass_pg")], dtype=float
        )
        if missed_mass.size:
            mass_ax.hist(
                [found_mass[np.isfinite(found_mass)], missed_mass],
                bins=np.logspace(
                    np.log10(max(np.nanmin(found_mass), 1e-2)),
                    np.log10(np.nanmax(found_mass)), 26),
                stacked=True, density=False,
                color=[colour, "#00000000"], histtype="stepfilled", alpha=0.35,
                label=[f"{LABEL.get(modality, modality)}: found", "missed"],
            )
            mass_ax.hist(missed_mass, bins=np.logspace(
                np.log10(max(np.nanmin(found_mass), 1e-2)),
                np.log10(np.nanmax(found_mass)), 26),
                histtype="step", color=colour, ls="--", lw=1.3)

    if not drawn:
        skip(15, "recall_by_size", "no unmatched_<split>.csv; re-run the evaluation")
        plt.close(fig)
        return

    curve_ax.axhline(1.0, color="k", lw=0.8, ls=":")
    curve_ax.set_xscale("log")
    curve_ax.set_ylim(0, 1.05)
    curve_ax.set_xlabel("Reference cell area (um^2, log scale)")
    curve_ax.set_ylabel("Detection recall")
    curve_ax.set_title("Recall by cell size\nshaded band = 95% binomial interval")
    curve_ax.legend(fontsize=7, loc="lower right")

    mass_ax.set_xscale("log")
    mass_ax.set_xlabel("Reference dry mass per cell (pg, log scale)")
    mass_ax.set_ylabel("Cells")
    mass_ax.set_title("Where the missed mass sits\ndashed = missed cells")
    mass_ax.legend(fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, 15, "recall_by_cell_size")


# ---------------------------------------------------------------------------
# ===========================================================================
# FIGURE 17 -- boundary error to measurement error                 [ESSENTIAL]
# ===========================================================================
def figure_error_propagation(cfg, args, out_dir: Path) -> None:
    """The exchange rate between a segmentation error and a measurement error.

    THE ONE FIGURE IN THIS SET THAT NEEDS NO MODEL. It is built from the
    reference masks and the reference phase, by moving the boundary a known
    number of pixels and measuring what happens to area and to dry mass. So it
    is a property of the specimen and the optics rather than of any method,
    which is what makes it a calibration curve other people can use instead of
    a score only this project can quote.

    The two curves must be plotted separately and never averaged. Dilation adds
    an annulus outside the cell, where the phase is near zero, so mass
    saturates while area keeps growing. Erosion removes an annulus just inside
    the membrane, where the cell is thinnest but not empty, so mass tracks area
    more closely. The gap between the two is the sign of the segmenter's bias,
    which is exactly what a Bland-Altman plot of the final result is about.

    Input: runs/error_propagation_summary.csv, from scripts/error_propagation.py
    """
    number, name = 17, "error_propagation"
    rows = read_csv_rows(Path(cfg.paths.output_root) / "error_propagation_summary.csv")
    if not rows:
        skip(number, name, "runs/error_propagation_summary.csv missing; "
                           "run scripts/error_propagation.py")
        return

    shift = column(rows, "shift_px")
    shift_um = column(rows, "shift_um")
    dice = column(rows, "dice")
    area = column(rows, "area_signed")
    mass = column(rows, "mass_signed")
    ratio = column(rows, "mass_over_area")
    order = np.argsort(shift)
    shift, shift_um, dice = shift[order], shift_um[order], dice[order]
    area, mass, ratio = area[order], mass[order], ratio[order]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    ax = axes[0]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.plot(shift, 100 * area, "o-", color="#c44e52", label="projected area")
    ax.plot(shift, 100 * mass, "s-", color="#4c72b0", label="dry mass")
    ax.set_xlabel("boundary displacement (pixels)")
    ax.set_ylabel("relative error (%)")
    # Extra pad: this panel carries a secondary axis above it, and the title
    # would otherwise sit on top of the micrometre scale.
    ax.set_title("what a boundary error costs", pad=30)
    ax.legend(loc="upper left")
    # The physical scale, so the x axis is not only in pixels.
    secondary = ax.secondary_xaxis(
        "top",
        functions=(lambda v: v * float(cfg.optics.pixel_pitch_x_um),
                   lambda v: v / float(cfg.optics.pixel_pitch_x_um)),
    )
    secondary.set_xlabel("micrometres")

    # Against Dice, because that is the number a segmentation paper reports and
    # therefore the number a reader arrives with.
    ax = axes[1]
    outward, inward = shift > 0, shift < 0
    ax.plot(dice[outward], 100 * np.abs(mass[outward]), "s-", color="#4c72b0",
            label="dilated (mass)")
    ax.plot(dice[inward], 100 * np.abs(mass[inward]), "s--", color="#4c72b0",
            label="eroded (mass)")
    ax.plot(dice[outward], 100 * np.abs(area[outward]), "o-", color="#c44e52",
            label="dilated (area)")
    ax.plot(dice[inward], 100 * np.abs(area[inward]), "o--", color="#c44e52",
            label="eroded (area)")
    ax.invert_xaxis()
    ax.set_xlabel("Dice against the reference mask")
    ax.set_ylabel("|relative error| (%)")
    ax.set_title("the exchange rate a reader can use")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[2]
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    finite = np.isfinite(ratio)
    ax.plot(shift[finite], ratio[finite], "D-", color="#55a868")
    ax.set_xlabel("boundary displacement (pixels)")
    ax.set_ylabel("mass error / area error")
    ax.set_title("mass is the more robust measurement")
    # `np.nanmax` of an empty slice raises, and `finite` is empty whenever every
    # mass/area ratio is non-finite -- which is what a run with no matched cells
    # produces. The figure should still draw its other panels.
    ceiling = (
        float(np.nanmax(ratio[finite])) * 1.1 if bool(np.any(finite)) else 1.15
    )
    ax.set_ylim(0.0, max(1.15, ceiling))
    ax.text(0.03, 0.06,
            "below 1: the boundary sits where\nthe cell is thinnest, so the\n"
            "pixels a boundary error moves\ncarry little phase",
            transform=ax.transAxes, fontsize=7.5, va="bottom", color="0.3")

    fig.suptitle(
        "Segmentation error propagated into quantitative measurement\n"
        "(reference masks and reference phase only; no model involved)",
        y=1.13, fontsize=10,
    )
    save(fig, out_dir, number, name)


# ===========================================================================
# FIGURE 18 -- the pipeline's own error floor                      [ESSENTIAL]
# ===========================================================================
def figure_synthetic_floor(cfg, args, out_dir: Path) -> None:
    """What the measurement chain gets wrong when the answer is known exactly.

    Every reported measurement error is the sum of model error and pipeline
    error, and on real data the two cannot be separated because no real cell's
    true mass is known. On synthetic spherical caps it can be: area is pi r^2
    and integrated phase is phi_0 (2/3) pi r^2, both exact.

    Two floors come out of this, and they are different numbers for different
    claims. The PER-CELL floor is discretisation, and it falls with cell size.
    The FIELD-TOTAL floor is the area filter discarding small cells, and it is
    an order of magnitude larger. Quoting the first where the second applies
    would understate a population mean's bias by that whole factor.

    Input: runs/synthetic_validation_cells.csv and _fields.csv, from
    scripts/synthetic_validation.py
    """
    number, name = 18, "synthetic_floor"
    root = Path(cfg.paths.output_root)
    cells = read_csv_rows(root / "synthetic_validation_cells.csv")
    fields = read_csv_rows(root / "synthetic_validation_fields.csv")
    if not cells:
        skip(number, name, "runs/synthetic_validation_cells.csv missing; "
                           "run scripts/synthetic_validation.py")
        return

    radius = column(cells, "radius_um")
    area_error = 100 * column(cells, "area_relative_error")
    mass_error = 100 * column(cells, "mass_relative_error")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    ax = axes[0]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.scatter(radius, area_error, s=14, alpha=0.7, color="#c44e52", label="area")
    ax.scatter(radius, mass_error, s=14, alpha=0.7, color="#4c72b0", label="dry mass")
    ax.set_xlabel("cell radius (um)")
    ax.set_ylabel("pipeline error (%)")
    ax.set_title("error against exact ground truth")
    ax.legend()

    # Binned, because the scatter's message is a trend and the trend is the
    # thing that sets a minimum measurable cell size.
    ax = axes[1]
    edges = np.percentile(radius[np.isfinite(radius)], [0, 20, 40, 60, 80, 100])
    centres, area_band, mass_band = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        band = (radius >= low) & (radius <= high)
        if band.sum() == 0:
            continue
        centres.append(0.5 * (low + high))
        area_band.append(np.nanmean(np.abs(area_error[band])))
        mass_band.append(np.nanmean(np.abs(mass_error[band])))
    ax.plot(centres, area_band, "o-", color="#c44e52", label="area")
    ax.plot(centres, mass_band, "s-", color="#4c72b0", label="dry mass")
    ax.set_yscale("log")
    ax.set_xlabel("cell radius (um)")
    ax.set_ylabel("mean |pipeline error| (%)")
    ax.set_title("discretisation falls with cell size")
    ax.legend()
    floor_um2 = float(cfg.evaluation.measurement.min_cell_area_um2)
    floor_radius = float(np.sqrt(floor_um2 / np.pi))
    ax.axvline(floor_radius, color="0.4", ls="--", lw=0.9)
    # Bottom of the axes, not the top: the top is where the area curve starts.
    ax.text(floor_radius, ax.get_ylim()[0],
            f"  min_cell_area = {floor_um2:.0f} um^2\n  ({floor_radius:.2f} um radius)",
            fontsize=7.5, va="bottom", color="0.3")

    # The two floors side by side, which is the point of the figure.
    ax = axes[2]
    per_cell = float(np.nanmean(np.abs(mass_error)))
    total = (
        100 * float(np.nanmean(column(fields, "mass_total_relative_error")))
        if fields else np.nan
    )
    # TWO DIFFERENT STATISTICS, so they are labelled as two.
    #
    # The left bar is a mean ABSOLUTE error -- a scatter measure -- and the right
    # one is the magnitude of a MEAN, which is a bias. Both were labelled
    # "|bias| (%)", and the caption tells the reader to quote one or the other,
    # so the mislabelling was load-bearing.
    labels = ["per cell\nmean |error|\n(discretisation)",
              "field total\nmean bias\n(area filter)"]
    values = [per_cell, abs(total)]
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452"], width=0.55)
    ax.set_ylabel("error magnitude (%)")
    ax.set_title("two floors, two different claims")
    ax.set_yscale("log")
    finite_values = [v for v in values if np.isfinite(v) and v > 0]
    if finite_values:
        # Headroom for the value labels, and a floor low enough that the
        # smaller bar is visible on a log scale rather than clipped to nothing.
        ax.set_ylim(min(finite_values) / 6.0, max(finite_values) * 3.0)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18,
                    f"{value:.2f}%", ha="center", fontsize=9)
    # Below the axes: inside them it sat on top of the bars.
    ax.text(0.5, -0.30,
            "quote the left number for 'the mass of this cell',\n"
            "the right one for 'the mass on this field'",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color="0.3")

    fig.suptitle(
        "Measurement pipeline validated against exact analytic ground truth\n"
        "(synthetic spherical caps; no model, no reconstruction)",
        y=1.09, fontsize=10,
    )
    save(fig, out_dir, number, name)


# ===========================================================================
# FIGURE 19 -- the v2 ablation, with the significance rule drawn  [ESSENTIAL]
# ===========================================================================
def figure_v2_ablation(cfg, args, out_dir: Path) -> None:
    """What each term is worth, against the noise it has to clear.

    THE PAPER'S CENTRAL FIGURE, and the one that did not exist. Figure 7 reads
    `*_modality_comparison.json`, which only `main.py compare` writes for the
    v1 off-axis-versus-Gabor study; v2 is off-axis only, so figure 7 can never
    build for it and the v2 ablation had no figure at all.

    This reads `runs/<arm>/metrics_test.json` directly -- the same source
    `scripts/collect_results.py` uses, so the figure and the table cannot
    disagree -- and plots each arm's signed delta against ITS OWN comparator,
    because each pair differs by exactly one term.

    THE SHADED BAND IS THE POINT. It is +/- 2x the pooled between-seed standard
    deviation of the same metric, which is the threshold a difference has to
    clear to count in this project. A bar inside the band is seed noise. Drawing
    the band rather than printing a p-value means a reader cannot mistake a
    difference that did not clear it for one that did -- and the v1 round
    resolved 0 of 54 comparisons by this rule, so the band is usually wider than
    the bars. That is the honest picture and it should be legible at a glance.

    When no seed replication exists the band cannot be computed, and the figure
    says so across the axes instead of drawing a band of zero width, which
    would read as "every difference is significant".

    Input: runs/<arm>/metrics_test.json for each arm, plus runs/v2_<arm>_seed*
    """
    number, name = 19, "v2_ablation"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from collect_results import ARMS, experiment_name, load_metrics, seed_spread
    except Exception as exc:
        skip(number, name, f"cannot import collect_results: {exc}")
        return

    root = Path(cfg.paths.output_root)
    modality = args.modalities[0] if args.modalities else cfg.data.modality

    found, comparator_of, labels = {}, {}, {}
    for code, config_path, label, comparator in ARMS:
        experiment = experiment_name(Path(config_path))
        if experiment is None:
            continue
        metrics = load_metrics(root, experiment, modality, args.split)
        if metrics is not None:
            found[code] = metrics
            comparator_of[code] = comparator
            labels[code] = label
    if len(found) < 2:
        skip(number, name,
             f"need >= 2 trained arms in {root}; found {len(found)}. Run stage 6.")
        return

    metrics_wanted = [
        ("dry_mass_mape",    "Dry mass MAPE",   True),
        ("area_mape",        "Area MAPE",       True),
        ("detection_recall", "Detection recall", False),
        ("seg_dice",         "Segmentation Dice", False),
        ("phase_mae_rad",    "Phase MAE",       True),
    ]
    # Keep only metrics at least one arm actually reported, so an untrained
    # study does not produce five empty panels.
    def _reported(code: str, key: str) -> bool:
        """Did this arm report a finite value for this metric?

        `x or np.nan` was the old test, and `0.0 or np.nan` is `np.nan` -- so a
        metric that is genuinely 0.0 on every arm was treated as unreported and
        its whole panel was dropped. That is live in this project: an earlier
        seed aggregate carried cls_accuracy = 0.0 on every arm, which is a
        result and not a missing value.
        """
        value = found[code].get(key)
        return isinstance(value, (int, float)) and np.isfinite(value)

    metrics_wanted = [
        entry for entry in metrics_wanted
        if any(_reported(c, entry[0]) for c in found)
    ]
    if not metrics_wanted:
        skip(number, name,
             "no arm reports any of the headline metrics. The usual cause is that "
             "no cell was detected, so no per-cell error exists -- check "
             "detection_recall before reading anything into the arms.")
        return

    pairs = [(c, comparator_of[c]) for c in found
             if comparator_of.get(c) and comparator_of[c] in found]
    if not pairs:
        skip(number, name, "no arm has its comparator trained; nothing to difference")
        return

    fig, axes = plt.subplots(
        1, len(metrics_wanted),
        figsize=(3.0 * len(metrics_wanted) + 1.5, 0.42 * len(pairs) + 2.4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    positions = np.arange(len(pairs))
    # From configuration, so the band drawn is the rule the project states.
    resolve_factor = float(cfg.evaluation.seed_replication.resolve_factor)
    any_band = False

    for axis, (key, title, lower_is_better) in zip(axes, metrics_wanted):
        deltas, bands = [], []
        for code, comparator in pairs:
            value = found[code].get(key)
            reference = found[comparator].get(key)
            if value is None or reference is None:
                deltas.append(np.nan)
                bands.append(np.nan)
                continue
            deltas.append(float(value) - float(reference))
            a = seed_spread(root, code, modality, args.split, key)
            b = seed_spread(root, comparator, modality, args.split, key)
            # THE SHARED IMPLEMENTATION of the project's significance rule.
            # This used to compute sqrt(0.5 (v_a + v_b)) while
            # scripts/aggregate_seeds.py computed sqrt(v_a + v_b) for the same
            # rule -- a factor of sqrt(2) apart, so one document could call a
            # comparison resolved and the other could call it noise.
            pooled, reason = pooled_between_seed_sd(a, b)
            bands.append(resolve_factor * pooled if reason is None else np.nan)

        deltas = np.asarray(deltas, float)
        bands = np.asarray(bands, float)

        # Green where the delta is an improvement, red where it is a
        # regression, grey where it is inside the seed band and therefore not a
        # difference at all.
        colours = []
        for delta, band in zip(deltas, bands):
            if not np.isfinite(delta):
                colours.append("0.85")
            elif np.isfinite(band) and abs(delta) <= band:
                colours.append("0.72")
            else:
                improved = (delta < 0) if lower_is_better else (delta > 0)
                colours.append("#55a868" if improved else "#c44e52")

        axis.barh(positions, np.nan_to_num(deltas), color=colours, height=0.62)
        axis.axvline(0.0, color="0.25", lw=1.0)

        # EACH BAR'S OWN BAND, drawn as a marker on that bar's row.
        #
        # A single shaded span of the WIDEST band was drawn before while each bar
        # was COLOURED against its own, so a grey bar ("inside the band") could
        # sit outside the shading and a coloured one inside it -- the plot
        # contradicting its own caption. The bands differ per comparison because
        # the arms have different numbers of seed replicates.
        for position, band in zip(positions, bands):
            if np.isfinite(band) and band > 0:
                axis.plot([-band, band], [position, position],
                          color="#4c72b0", lw=6.0, alpha=0.22,
                          solid_capstyle="butt", zorder=0)
                any_band = True

        axis.set_yticks(positions)
        axis.set_yticklabels([f"{c} vs {k}" for c, k in pairs])
        axis.invert_yaxis()
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel("delta (lower is better)" if lower_is_better
                        else "delta (higher is better)")
        finite = deltas[np.isfinite(deltas)]
        # The axis has to hold the widest band as well as the largest delta, or
        # a band would be drawn off the edge of its own panel.
        finite_bands = bands[np.isfinite(bands)]
        span = max(float(np.abs(finite).max()) if finite.size else 1.0,
                   float(finite_bands.max()) if finite_bands.size else 0.0)
        # A panel where every arm scored identically would otherwise be drawn on
        # a 1e-9 axis, which looks like structure at the limit of float
        # precision instead of what it is: no variation at all. Byte-identical
        # metrics across arms have appeared in this project before and meant the
        # model had collapsed, so this case gets said out loud.
        if span < 1e-6:
            axis.set_xlim(-1.0, 1.0)
            axis.set_xticks([-1, 0, 1])
            axis.text(0.5, 0.5, "every arm identical\n(no variation to plot)",
                      transform=axis.transAxes, ha="center", va="center",
                      fontsize=8.5, color="#c44e52")
        else:
            axis.set_xlim(-1.25 * span, 1.25 * span)

    if any_band:
        caption = ("Shaded band: +/- 2x the pooled between-seed SD. A bar inside it "
                   "is seed noise, not a difference. Grey bars are inside the band.")
    else:
        caption = ("NO SEED REPLICATION: the significance band cannot be computed, so "
                   "NO bar here can be called a difference. Run "
                   'SEEDS="1337 2024" bash run_v2.sh.')
    fig.suptitle("HoloQPI v2 ablation: each arm against its own comparator, "
                 "one term apart\n" + caption, y=1.02, fontsize=9.5)
    save(fig, out_dir, number, name)


# ===========================================================================
# FIGURE 20 -- the membrane channel against the phase labels     [ESSENTIAL]
# ===========================================================================
def figure_membrane(cfg, args, out_dir: Path) -> None:
    """Independent labels, and what they say about the circular ones.

    Every mask in this study is `Otsu(gaussian * phase_reference)`, so the
    segmentation metrics measure agreement with a threshold of the network's own
    input. The membrane fluorescence channel is an independent measurement of
    where the cell is, and this figure is the evidence that it is now usable:
    registered onto the phase grid by a single global transform.

    READ THE THIRD PANEL CAREFULLY. It shows where the two label sources
    disagree, and the disagreement is not symmetric. The phase threshold keeps
    the thick perinuclear body and drops the thin spread periphery, because a
    phase threshold can only see where the cell is optically thick. That is a
    systematic bias in projected area, and -- by exactly the mechanism figure 17
    measures -- a much smaller one in dry mass.

    The membrane mask is PROVISIONAL and the figure says so. Membrane stain
    marks the perimeter rather than the interior, so a threshold returns rings
    and filling them fails wherever a lamellipodium leaves one open. Manual
    annotation on 25-40 fields remains the gold standard; this makes producing
    it cheap, because the annotator traces on an already-aligned image.

    Input: data/membrane_aligned/<stem>_membrane.npy from
    scripts/prepare_membrane.py, plus the phase and the Otsu mask.
    """
    number, name = 20, "membrane_labels"
    root = Path(cfg.paths.data_root)
    aligned = root / cfg.membrane.output_dir
    if not aligned.is_dir():
        skip(number, name, f"{aligned} missing; run scripts/prepare_membrane.py")
        return

    from holoqpi.data import io as data_io

    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        skip(number, name, "no manifest; run `python main.py prepare`")
        return
    with open(manifest, newline="") as handle:
        stems = [row["stem"] for row in csv.DictReader(handle)]
    stems = [s for s in stems
             if (aligned / f"{s}{cfg.membrane.suffix}").is_file()][:3]
    if not stems:
        skip(number, name, f"no aligned membrane arrays in {aligned}")
        return

    from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter

    size = cfg.data.phase_size
    fig, axes = plt.subplots(len(stems), 4, figsize=(14.5, 3.7 * len(stems)))
    axes = np.atleast_2d(axes)
    agreement = []

    for row, stem in enumerate(stems):
        phase = data_io.read_phase_bin(
            data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
        ).phase.astype(np.float32)
        otsu = data_io.align_to_phase_grid(
            data_io.read_mask(data_io.mask_path(root, cfg, stem)), size, cfg.data.align
        ) > 0
        membrane = np.load(aligned / f"{stem}{cfg.membrane.suffix}")
        valid_path = aligned / f"{stem}{cfg.membrane.valid_suffix}"
        valid = (np.load(valid_path) if valid_path.is_file()
                 else np.ones_like(otsu, dtype=bool))

        smooth = gaussian_filter(membrane, 2.0)
        threshold = float(np.percentile(smooth[valid], cfg.membrane.threshold_percentile))
        derived = binary_fill_holes(
            binary_opening((smooth > threshold) & valid, np.ones((3, 3)))
        )
        otsu = otsu & valid
        intersection = float((derived & otsu).sum())
        agreement.append(2 * intersection / (derived.sum() + otsu.sum() + 1e-9))

        axes[row, 0].imshow(phase, cmap="viridis")
        axes[row, 0].set_title(f"{stem}\nquantitative phase", fontsize=9)

        axes[row, 1].imshow(np.log1p(np.clip(membrane, 0, None)), cmap="magma")
        axes[row, 1].set_title("membrane, warped to the phase grid", fontsize=9)

        # Red/green rather than a difference image: which source claims a pixel
        # is the question, and a signed difference hides that.
        overlay = np.zeros((size, size, 3))
        overlay[..., 0] = derived
        overlay[..., 1] = otsu
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f"red = membrane   green = phase Otsu\nyellow = agree   "
            f"Dice {agreement[-1]:.3f}", fontsize=9,
        )

        axes[row, 3].imshow(phase, cmap="gray")
        axes[row, 3].contour(derived, [0.5], colors="#c44e52", linewidths=0.7)
        axes[row, 3].contour(otsu, [0.5], colors="#55a868", linewidths=0.7)
        if not valid.all():
            axes[row, 3].contour(valid, [0.5], colors="#4c72b0",
                                 linewidths=0.8, linestyles="dashed")
        axes[row, 3].set_title("contours on the phase\n(blue dashed = illuminated window)",
                               fontsize=9)
        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle(
        "Independent membrane labels registered onto the phase grid "
        f"(scale {cfg.membrane.scale:.4f} = 0.284871/0.211994, "
        f"offset {cfg.membrane.offset_y},{cfg.membrane.offset_x})\n"
        f"median Dice against the phase-threshold labels {np.median(agreement):.3f}. "
        "The membrane mask is PROVISIONAL: stain marks the perimeter, so a "
        "threshold returns rings and filling them leaks where they open.",
        y=1.0, fontsize=9.5,
    )
    save(fig, out_dir, number, name)


FIGURES = {
    1: ("qualitative_panel", "ESSENTIAL", figure_qualitative),
    2: ("bland_altman", "ESSENTIAL", figure_bland_altman),
    3: ("error_decomposition", "ESSENTIAL", figure_error_decomposition),
    4: ("detection_gap", "ESSENTIAL", figure_detection),
    5: ("modality_comparison", "ESSENTIAL", figure_modality_summary),
    6: ("loss_composition", "ESSENTIAL", figure_loss_components),
    7: ("ablation_deltas", "ESSENTIAL", figure_ablations),
    8: ("efficiency_tradeoff", "ESSENTIAL", figure_efficiency),
    9: ("phase_error_structure", "OPTIONAL", figure_phase_error_structure),
    10: ("label_audit", "IMPORTANT", figure_label_audit),
    11: ("confusion_matrices", "IMPORTANT", figure_confusion),
    12: ("convergence", "IMPORTANT", figure_convergence),
    13: ("forward_model_consistency", "ESSENTIAL", figure_forward_model),
    14: ("learned_vs_classical", "ESSENTIAL", figure_conventional_baseline),
    15: ("recall_by_cell_size", "IMPORTANT", figure_recall_by_size),
    16: ("forward_model_diagnostics", "ESSENTIAL", figure_forward_model_diagnostics),
    17: ("error_propagation", "ESSENTIAL", figure_error_propagation),
    18: ("synthetic_floor", "ESSENTIAL", figure_synthetic_floor),
    19: ("v2_ablation", "ESSENTIAL", figure_v2_ablation),
    20: ("membrane_labels", "ESSENTIAL", figure_membrane),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modalities", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        help="figure numbers to build; default is all")
    parser.add_argument("--list", action="store_true", help="list the figures and exit")
    parser.add_argument("--example-index", type=int, default=0,
                        help="which image of the first evaluation batch to draw in figure 1")
    parser.add_argument("--spectrum-batches", type=int, default=4,
                        help="batches to average for the figure 9 spectrum")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    if args.list:
        for number, (name, tier, function) in sorted(FIGURES.items()):
            first_line = (function.__doc__ or "").strip().splitlines()[0]
            print(f"  {number:>2}  [{tier:<9}] {name:<24} {first_line}")
        return 0

    cfg = load_config(args.config, parse_overrides(args.set))
    style()
    out_dir = Path(args.out_dir)
    wanted = sorted(args.only) if args.only else sorted(FIGURES)

    _WRITTEN.clear()
    skipped, failed = [], []
    for number in wanted:
        if number not in FIGURES:
            LOGGER.warning("no figure %s", number)
            continue
        name, tier, function = FIGURES[number]
        LOGGER.info("--- figure %02d (%s, %s) ---", number, name, tier)
        try:
            function(cfg, args, out_dir)
            if number not in _WRITTEN:
                # Returned without writing, i.e. it took a `skip(...)` branch
                # for a missing input. That is not the same as built.
                skipped.append(number)
        except Exception as exc:                 # one bad figure must not stop the rest
            LOGGER.warning("figure %02d (%s) failed: %s: %s",
                           number, name, type(exc).__name__, exc)
            failed.append(number)

    built = sorted(_WRITTEN)
    # A MANIFEST, because the filenames carry no experiment, modality or split.
    # `figures/fig19_v2_ablation.png` built from one config overwrites the
    # identically named file built from another, and nothing in the directory
    # said which config produced what. This is the record.
    #
    # MERGED, NOT OVERWRITTEN. Stage 9 calls this script three times -- global
    # figures from config/base.yaml, per-arm figures from one arm's config, and
    # the modality pair -- and each writes to the same directory. A whole-file
    # write meant the last invocation's manifest was the only one left: after the
    # 2026-09-17 run the file recorded three figures, 5, 13 and 14, while
    # seventeen sat in the directory beside it. The manifest existed precisely so
    # nobody had to guess which config produced which figure, and it was
    # answering that question for three of them.
    entry = {
        "config": str(Path(args.config)),
        "experiment_name": cfg.experiment_name,
        "modalities": list(args.modalities),
        "split": args.split,
    }
    manifest_path = out_dir / "_manifest.json"
    existing = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception as exc:
            # A corrupt manifest is replaced rather than allowed to block the
            # record of this run, and the loss is said out loud.
            LOGGER.warning("could not read the existing figure manifest (%s); "
                           "it will be replaced", exc)
            existing = {}
    if not isinstance(existing.get("figures"), dict):
        existing = {"figures": {}}

    figures = existing["figures"]
    for number in built:
        figures[str(number)] = {"name": _WRITTEN[number]["name"], **entry}
    # A figure this run tried and skipped or failed loses its old entry: the file
    # on disk is then from an earlier run and the manifest must not vouch for it.
    for number in list(skipped) + list(failed):
        figures.pop(str(number), None)

    manifest = {
        "figures": dict(sorted(figures.items(), key=lambda kv: int(kv[0]))),
        "last_run": {
            **entry,
            "built": [f"{n:02d}" for n in built],
            "skipped_missing_input": skipped,
            "failed": failed,
        },
    }
    try:
        write_json(manifest, manifest_path)
    except Exception as exc:
        LOGGER.warning("could not write the figure manifest: %s", exc)

    print(f"\nfigures written to {out_dir.resolve()}")
    print(f"  from {args.config} (experiment {cfg.experiment_name}, "
          f"{'+'.join(args.modalities)}, {args.split} split)")
    if built:
        print(f"\n{len(built)} figure(s) written by THIS run:")
        for number in built:
            print(f"  fig{number:02d}_{_WRITTEN[number]['name']}.png / .pdf")
    else:
        print("\n  none: no figure wrote a file in this run.")
    if skipped:
        print(f"\n{len(skipped)} figure(s) skipped for a missing input: {skipped}")
        print("Each printed its reason above. Any file for these in the directory is")
        print("from an EARLIER run and does not describe this one.")
    if failed:
        print(f"\n{len(failed)} figure(s) could not be built: {failed}")
        print("Each prints its reason above; the usual cause is a missing input file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
