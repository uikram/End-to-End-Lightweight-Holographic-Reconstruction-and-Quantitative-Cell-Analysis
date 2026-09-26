"""Audit the two assumptions the silver-standard labels rest on.

The segmentation targets are not manual annotations. They are produced by
Otsu-thresholding the same ground-truth phase map that the phase head regresses.
Two consequences follow, and neither is visible in any training curve, so both
are measured here and reported in the paper rather than assumed away.

1. THRESHOLD STABILITY.  Otsu picks a level per image from that image's own
   histogram, so "cell" is defined slightly differently in every field of view.
   If the level also drifts systematically with drug condition -- because a drug
   changes confluence, background offset or contrast -- then measured morphology
   differs between conditions partly because the label definition differs, and
   the classification result is confounded. This script reports the level per
   image, its spread, and whether it separates by condition.

2. TASK REDUNDANCY.  Because the mask is a deterministic function of the phase,
   the two supervised heads are not independent: a perfect phase reconstruction
   determines the mask exactly. This predicts that the segmentation head learns
   little the phase head does not already encode, and it is the mechanism behind
   the null result for the physics-consistency loss terms. The prediction is
   testable: apply the mask-generation function to the *predicted* phase and
   measure how closely the result agrees with what the segmentation head
   produced. High agreement confirms the redundancy.

    python scripts/audit_labels.py --config config/base.yaml
    python scripts/audit_labels.py --config config/base.yaml --skip-redundancy
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.utils import get_logger, resolve_device, run_directory, write_csv, write_json

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. threshold stability
# ---------------------------------------------------------------------------
def audit_thresholds(cfg) -> dict:
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    from holoqpi.data import io as data_io
    from holoqpi.data.metadata import LabelSchema

    mask_cfg = cfg.mask_generation
    root = Path(cfg.paths.data_root)
    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found. Run `python main.py prepare` first.")

    stems = [row["stem"] for row in csv.DictReader(open(manifest))]
    schema = LabelSchema(cfg)

    rows: list[dict] = []
    by_condition: dict[str, list[float]] = defaultdict(list)

    for position, stem in enumerate(stems, start=1):
        record = data_io.read_phase_bin(
            data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
        )
        smoothed = ndimage.gaussian_filter(
            record.phase.astype(np.float64), mask_cfg.smoothing_sigma_px
        )
        # THE METHOD THE LABELS WERE ACTUALLY MADE WITH.
        #
        # This hardcoded Otsu while holoqpi/data/masks.py branches on
        # mask_generation.threshold_method -- and this script's own verdict
        # RECOMMENDS switching to `fixed`. Under that setting the audit was
        # reporting per-image variability, a CV and an ANOVA p-value for a
        # threshold that is a constant and has none.
        if mask_cfg.threshold_method == "otsu":
            level = float(threshold_otsu(smoothed)) * mask_cfg.otsu_scale
        elif mask_cfg.threshold_method == "fixed":
            level = float(mask_cfg.fixed_threshold_rad)
        else:
            raise ValueError(
                f"unknown mask_generation.threshold_method "
                f"{mask_cfg.threshold_method!r}"
            )
        meta = schema.parse(stem)
        rows.append({
            "stem": stem,
            "cell_line": meta.cell_line,
            "condition": meta.condition,
            "otsu_level_rad": level,
            # The RAW threshold's foreground fraction, before binary_closing,
            # fill_holes, clear_border and the area filter -- which is not the
            # written mask's fraction, and the difference is not small: raw
            # Otsu output carries 4-11 border-touching components per field
            # that closing then removes entirely. Named accordingly rather than
            # labelled as if it described the mask.
            "threshold_foreground_fraction": float((smoothed > level).mean()),
            "phase_p50_rad": float(np.percentile(record.phase, 50)),
            "phase_p99_rad": float(np.percentile(record.phase, 99)),
        })
        by_condition[meta.condition].append(level)
        if position % 100 == 0:
            LOGGER.info("thresholds: %d/%d", position, len(stems))

    levels = np.array([r["otsu_level_rad"] for r in rows])
    groups = [np.array(v) for _, v in sorted(by_condition.items())]

    summary = {
        "n_images": len(rows),
        "level_mean_rad": float(levels.mean()),
        "level_sd_rad": float(levels.std(ddof=1)) if levels.size > 1 else 0.0,
        "level_min_rad": float(levels.min()),
        "level_max_rad": float(levels.max()),
        "level_cv": float(levels.std(ddof=1) / levels.mean()) if levels.mean() else float("nan"),
        # UNWEIGHTED descriptive spreads, and named so. A condition with three
        # fields counts as much as one with three hundred in both, so they must
        # not be read as a variance ratio -- the f_oneway and kruskal results
        # below are the size-aware tests and are the ones to quote.
        "between_condition_sd_unweighted": float(np.std([g.mean() for g in groups], ddof=1))
        if len(groups) > 1 else 0.0,
        "within_condition_sd_unweighted": float(
            np.sqrt(np.mean([g.var(ddof=1) for g in groups if g.size > 1]))
        ) if any(g.size > 1 for g in groups) else 0.0,
        # The size-weighted pooled within-condition SD, which is the one that
        # can be compared against the between-condition spread.
        "within_condition_sd_pooled": float(np.sqrt(
            sum((g.size - 1) * g.var(ddof=1) for g in groups if g.size > 1)
            / max(sum(g.size - 1 for g in groups if g.size > 1), 1)
        )) if any(g.size > 1 for g in groups) else 0.0,
        "threshold_method": mask_cfg.threshold_method,
        "smoothing_sigma_px": mask_cfg.smoothing_sigma_px,
        "otsu_scale": mask_cfg.otsu_scale,
        "labels_read_from": str(cfg.paths.manual_mask_dir or cfg.paths.mask_dir),
    }

    # Does the label definition separate by condition more than by chance?
    try:
        from scipy.stats import f_oneway, kruskal
        usable = [g for g in groups if g.size >= 2]
        if len(usable) >= 2:
            summary["anova_F"], summary["anova_p"] = (float(v) for v in f_oneway(*usable))
            summary["kruskal_H"], summary["kruskal_p"] = (float(v) for v in kruskal(*usable))
    except Exception as exc:                                   # scipy missing a test
        LOGGER.warning("condition test unavailable: %s", exc)

    summary["per_condition"] = {
        condition: {
            "n": len(values),
            "mean_rad": float(np.mean(values)),
            "sd_rad": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for condition, values in sorted(by_condition.items())
    }
    return {"summary": summary, "rows": rows}


def report_thresholds(result: dict) -> None:
    summary = result["summary"]
    print("\n=== 1. Otsu threshold stability ===")
    print(f"  {summary['n_images']} images")
    print(f"  level  {summary['level_mean_rad']:.4f} +/- {summary['level_sd_rad']:.4f} rad "
          f"[{summary['level_min_rad']:.4f}, {summary['level_max_rad']:.4f}]  "
          f"CV = {summary['level_cv']:.1%}")
    print(f"  between-condition SD {summary['between_condition_sd_unweighted']:.4f} rad "
          f"(unweighted)   within-condition SD "
          f"{summary['within_condition_sd_pooled']:.4f} rad (pooled)")
    for condition, stats in summary["per_condition"].items():
        print(f"    {condition:<22} n={stats['n']:<4} {stats['mean_rad']:.4f} "
              f"+/- {stats['sd_rad']:.4f} rad")
    if "anova_p" in summary:
        print(f"  one-way ANOVA across conditions  F={summary['anova_F']:.3f}  "
              f"p={summary['anova_p']:.4g}")
        print(f"  Kruskal-Wallis                   H={summary['kruskal_H']:.3f}  "
              f"p={summary['kruskal_p']:.4g}")
        print("\n  -> ", end="")
        if summary["anova_p"] < 0.05:
            print("the threshold DOES separate by condition. The label definition is\n"
                  "     partly confounded with the class label; report this, and consider\n"
                  "     mask_generation.threshold_method: fixed with fixed_threshold_rad\n"
                  f"     set near {summary['level_mean_rad']:.3f} as a sensitivity check.")
        else:
            print("no detectable condition dependence in the threshold. The label\n"
                  "     definition varies image to image but not systematically by class,\n"
                  "     so the classification result is not confounded through this route.")


# ---------------------------------------------------------------------------
# 2. head redundancy
# ---------------------------------------------------------------------------
def audit_redundancy(cfg, modality: str, device, split: str) -> dict | None:
    """How much of the segmentation head is reproducible from the phase head?"""
    import torch

    from holoqpi.data import build_dataloaders
    from holoqpi.data.masks import build_mask
    from holoqpi.analysis.cells import calibration_from_config
    from holoqpi.engine import load_checkpoint
    from holoqpi.models import build_model

    cfg = cfg.merged({"data": {"modality": modality}})
    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, modality)
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.is_file():
        LOGGER.warning("no checkpoint at %s; skipping %s", checkpoint, modality)
        return None

    model = build_model(cfg)
    load_checkpoint(model, checkpoint, device)
    model.to(device).eval()

    pixel_area = calibration_from_config(cfg).pixel_area_um2
    loader = build_dataloaders(cfg, splits_to_build=(split,))[split]

    head_vs_derived, head_vs_gt, derived_vs_gt = [], [], []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["hologram"].to(device))
            phase_pred = outputs["phase"].squeeze(1).float().cpu().numpy()
            head = outputs["segmentation"].argmax(dim=1).cpu().numpy() > 0
            gt = batch["mask"].numpy() > 0

            for i in range(phase_pred.shape[0]):
                # The same function that produced the training labels, applied to
                # the reconstruction instead of to the ground truth.
                derived = build_mask(phase_pred[i], cfg.mask_generation, pixel_area) > 0
                head_vs_derived.append(_dice(head[i], derived))
                head_vs_gt.append(_dice(head[i], gt[i]))
                derived_vs_gt.append(_dice(derived, gt[i]))

    if not head_vs_gt:
        LOGGER.warning("no images read for %s / %s; skipping", modality, split)
        return None

    return {
        "modality": modality,
        # The provenance this payload was missing. It carried modality and an
        # image count and nothing about WHICH split or WHICH arm produced it, so
        # a run at --split val overwrote a run at --split test with no trace, and
        # figure 10 plots these bars with no way to say what they describe.
        "split": split,
        "experiment": cfg.experiment_name,
        "labels_read_from": str(cfg.paths.manual_mask_dir or cfg.paths.mask_dir),
        "images": len(head_vs_gt),
        "dice_head_vs_threshold_of_predicted_phase": float(np.mean(head_vs_derived)),
        "dice_head_vs_ground_truth": float(np.mean(head_vs_gt)),
        "dice_threshold_of_predicted_phase_vs_ground_truth": float(np.mean(derived_vs_gt)),
    }


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = float(a.sum() + b.sum())
    return 2.0 * float((a & b).sum()) / total if total > 0 else 1.0


def report_redundancy(results: list[dict]) -> None:
    print("\n=== 2. Segmentation / phase head redundancy ===")
    print("  Dice(seg head, threshold(predicted phase)) measures how much of the\n"
          "  segmentation output is already implied by the reconstruction.\n")
    print(f"  {'modality':<12} {'head vs GT':>12} {'derived vs GT':>15} {'head vs derived':>17}")
    for result in results:
        print(f"  {result['modality']:<12} "
              f"{result['dice_head_vs_ground_truth']:>12.4f} "
              f"{result['dice_threshold_of_predicted_phase_vs_ground_truth']:>15.4f} "
              f"{result['dice_head_vs_threshold_of_predicted_phase']:>17.4f}")

    best = max(r["dice_head_vs_threshold_of_predicted_phase"] for r in results)
    print("\n  -> ", end="")
    if best > 0.85:
        print("the two heads are largely redundant. Thresholding the predicted\n"
              "     phase reproduces the segmentation head, which is the expected\n"
              "     consequence of deriving the masks from the phase and is the\n"
              "     mechanism behind the null result for the physics-consistency\n"
              "     terms: those terms constrain a quantity already determined by\n"
              "     the two primary losses. State this in the paper.")
    else:
        print("the segmentation head carries information beyond a threshold of the\n"
              "     predicted phase, so the two tasks are not simply redundant.")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-redundancy", action="store_true",
                        help="threshold audit only; needs no checkpoint and no GPU")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    output_root = Path(cfg.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # THE LABELS MAY NOT BE PHASE-DERIVED AT ALL. paths.manual_mask_dir
    # overrides mask_dir in the loader, and run_v2.sh evaluates one arm against
    # the membrane masks -- for which this whole section's premise ("the labels
    # rest on a per-image Otsu threshold") is simply false. Said out loud
    # instead of reported as though it applied.
    if cfg.paths.manual_mask_dir:
        print(f"\nNOTE: paths.manual_mask_dir is {cfg.paths.manual_mask_dir!r}, so the "
              f"labels in use are\n  NOT phase-derived and section 1 below does not "
              f"describe them. It still reports\n  what a phase threshold WOULD do, "
              f"which is the comparison, not the label audit.")

    thresholds = audit_thresholds(cfg)
    report_thresholds(thresholds)
    # Scoped by split and label source, so a second run cannot silently replace
    # the first. The split was not even recorded in the payload before, and
    # figure 10 reads these files by their fixed names.
    tag = f"_{args.split}" + (
        f"_{Path(str(cfg.paths.manual_mask_dir)).name}" if cfg.paths.manual_mask_dir else ""
    )
    write_csv(thresholds["rows"], output_root / f"label_audit_thresholds{tag}.csv")
    write_json(thresholds["summary"], output_root / f"label_audit_thresholds{tag}.json")
    # Also written under the unsuffixed name that figure 10 reads, so the figure
    # keeps working while the scoped copy preserves the provenance.
    write_csv(thresholds["rows"], output_root / "label_audit_thresholds.csv")
    write_json(thresholds["summary"], output_root / "label_audit_thresholds.json")
    print(f"\n  per-image detail -> {output_root / f'label_audit_thresholds{tag}.csv'}")

    if not args.skip_redundancy:
        device = resolve_device(args.device)
        results = [
            r for r in (audit_redundancy(cfg, m, device, args.split) for m in args.modality)
            if r is not None
        ]
        if results:
            report_redundancy(results)
            write_json(results, output_root / "label_audit_redundancy.json")
            print(f"\n  detail -> {output_root / 'label_audit_redundancy.json'}")
        else:
            print("\n=== 2. Segmentation / phase head redundancy ===")
            print("  no checkpoints found; skipped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
