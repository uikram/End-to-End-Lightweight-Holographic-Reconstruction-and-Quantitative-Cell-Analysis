"""Attribute the dry-mass error to its two possible sources.

Dry mass is the phase integral over the segmented domain, so an error in it can
come from the domain or from the phase:

    m_pred / m_ref  =  [ S(Omega_pred, phi_gt) / S(Omega_ref, phi_gt) ]   <- domain
                     x [ S(Omega_pred, phi_pred) / S(Omega_pred, phi_gt) ] <- phase

The first factor holds the phase fixed and varies only the boundary; the second
holds the boundary fixed and varies only the reconstruction. Their product is the
observed mass ratio, so whichever departs further from 1.0 is where the bias
lives, and that decides whether to work on the segmentation head or the phase
head.

A global phase bias near zero does not settle this on its own: the reconstruction
can be high inside cells and low in the background and still average to nothing,
while dry mass integrates only the inside. The intra-cell figures below separate
that case.

    python scripts/diagnose_bias.py --config config/base.yaml --modality off_axis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.engine import load_checkpoint
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device, run_name, write_csv

LOGGER = get_logger(__name__)


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean of a ratio, which makes the decomposition exact.

    The attribution only means something if the two factors reproduce the total,
    and that constrains the choice of summary. Neither the median of the ratios
    nor the median of their logarithms composes: the median is not linear, so
    median(log d) + median(log f) generally differs from median(log d + log f).
    The arithmetic mean of logarithms is linear, so the geometric mean satisfies
    domain x phase = total identically.
    """
    positive = [np.log(v) for v in values if np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(positive))) if positive else float("nan")


def _median_ratio(values: list[float]) -> float:
    """Median ratio, reported alongside as an outlier-resistant cross-check."""
    finite = [v for v in values if np.isfinite(v) and v > 0]
    return float(np.median(finite)) if finite else float("nan")


def diagnose(cfg, modality: str, device: torch.device, split: str) -> dict | None:
    cfg = cfg.merged({"data": {"modality": modality}})
    # run_name, not run_directory: run_directory CREATES the path, and it was
    # called BEFORE the checkpoint test below -- so every invocation against an
    # untrained arm left a new empty runs/<experiment>_<modality>/ behind, and
    # collect_results and make_figures then saw an arm directory that had never
    # been trained. The directory is only needed once there is something to
    # write into it.
    run_dir = Path(cfg.paths.output_root) / run_name(cfg.experiment_name, modality)
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.is_file():
        # One missing arm should not abort the other: report and skip.
        LOGGER.warning("no checkpoint at %s; skipping %s", checkpoint, modality)
        return None

    model = build_model(cfg)
    load_checkpoint(model, checkpoint, device)
    model.to(device).eval()

    loader = build_dataloaders(cfg, splits_to_build=(split,))[split]

    domain, phase_factor, total = [], [], []
    area_ratio, intra_bias, background_bias = [], [], []
    unphysical: list[str] = []
    # Counted separately from `unphysical`. An image with an empty GT or
    # predicted mask was skipped and tallied NOWHERE, so a split in which every
    # prediction was empty reported images = 0 and unphysical = 0 -- and the
    # report then blamed inverted contrast, which was the wrong diagnosis for a
    # detector that simply found nothing.
    empty_prediction: list[str] = []
    empty_reference: list[str] = []
    rows = []

    with torch.no_grad():
        for batch in loader:
            hologram = batch["hologram"].to(device)
            outputs = model(hologram)

            phase_pred = outputs["phase"].squeeze(1).float().cpu().numpy()
            mask_pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
            phase_gt = batch["phase"].squeeze(1).numpy()
            mask_gt = batch["mask"].numpy()

            for i in range(phase_pred.shape[0]):
                pp, mp = phase_pred[i], mask_pred[i] > 0
                pg, mg = phase_gt[i], mask_gt[i] > 0
                stem = batch["stem"][i] if "stem" in batch else f"image_{i}"
                if not mg.any():
                    empty_reference.append(stem)
                    continue
                if not mp.any():
                    empty_prediction.append(stem)
                    continue

                s_ref = float(pg[mg].sum())
                s_domain = float(pg[mp].sum())      # true phase, predicted boundary
                s_pred = float(pp[mp].sum())        # predicted phase, predicted boundary

                # A negative integrated phase is unphysical for a cell: it means
                # the prediction has inverted contrast. Such images are counted
                # and excluded rather than folded into a ratio.
                if min(s_ref, s_domain, s_pred) <= 0.0:
                    unphysical.append(batch["stem"][i])
                    continue

                d = s_domain / s_ref
                f = s_pred / s_domain
                domain.append(d)
                phase_factor.append(f)
                total.append(s_pred / s_ref)
                area_ratio.append(float(mp.sum()) / float(mg.sum()))
                intra_bias.append(float(pp[mg].mean() - pg[mg].mean()))
                background_bias.append(float(pp[~mg].mean() - pg[~mg].mean()))

                rows.append({
                    "stem": batch["stem"][i],
                    "domain_factor": d,
                    "phase_factor": f,
                    "mass_ratio": s_pred / s_ref,
                    "area_ratio": area_ratio[-1],
                    "intra_cell_phase_bias_rad": intra_bias[-1],
                    "background_phase_bias_rad": background_bias[-1],
                })

    domain_factor = _geometric_mean(domain)
    phase_factor = _geometric_mean(phase_factor)

    return {
        "modality": modality,
        "images": len(total),
        "unphysical": len(unphysical),
        "empty_prediction": len(empty_prediction),
        "empty_reference": len(empty_reference),
        "domain_factor": domain_factor,
        "phase_factor": phase_factor,
        "composed_ratio": domain_factor * phase_factor,
        "mass_ratio": _geometric_mean(total),
        "mass_ratio_median": _median_ratio(total),
        "area_ratio": _geometric_mean(area_ratio),
        "intra_cell_phase_bias_rad": _median(intra_bias),
        "background_phase_bias_rad": _median(background_bias),
        "_rows": rows,
        "_run_dir": run_dir,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['modality']}  ({result['images']} images) ===")
    if result["unphysical"]:
        print(f"  {result['unphysical']} images excluded: non-positive phase integral")
    if result.get("empty_prediction"):
        print(f"  {result['empty_prediction']} images excluded: the model predicted "
              f"no foreground at all")
    if result.get("empty_reference"):
        print(f"  {result['empty_reference']} images excluded: the reference mask is "
              f"empty")
    if result["images"] == 0:
        # EACH CAUSE NAMED, because they call for different actions and this
        # used to blame the wrong one. The message asserted inverted contrast
        # unconditionally, so a detector that found nothing -- which leaves
        # `unphysical` at zero -- was reported as a phase-sign problem.
        if result.get("empty_prediction") and not result["unphysical"]:
            print("  nothing to report: the model predicted NO foreground on any image, "
                  "so there is\n  no predicted domain to integrate over. This is a "
                  "detection failure, not a phase\n  problem -- check seg_dice and "
                  "detection_recall in the arm's metrics first.")
        elif result.get("empty_reference") and not result["unphysical"]:
            print("  nothing to report: every REFERENCE mask is empty, so there is "
                  "nothing to compare\n  against. Check that data/mask was generated "
                  "for this split.")
        elif result["unphysical"]:
            print("  nothing to report: every image had a non-positive integrated phase, "
                  "which means\n  the reconstruction has inverted contrast. Check the "
                  "checkpoint before reading further.")
        else:
            print("  nothing to report: no image reached the measurement. The split may "
                  "be empty.")
        return
    print(f"  mass ratio  pred/ref        {result['mass_ratio']:+.4f}"
          f"   ({100*(result['mass_ratio']-1):+.1f}% mass error)")
    print(f"    from the DOMAIN (boundary) {result['domain_factor']:+.4f}"
          f"   ({100*(result['domain_factor']-1):+.1f}%)")
    print(f"    from the PHASE  (recon)    {result['phase_factor']:+.4f}"
          f"   ({100*(result['phase_factor']-1):+.1f}%)")
    print(f"    domain x phase             {result['composed_ratio']:+.4f}"
          f"   (equals the mass ratio by construction)")
    print(f"  median mass ratio            {result['mass_ratio_median']:+.4f}"
          f"   (outlier-resistant cross-check)")
    print(f"  predicted / reference area   {result['area_ratio']:+.4f}"
          f"   ({100*(result['area_ratio']-1):+.1f}% area error)")
    print(f"  phase bias inside cells      {result['intra_cell_phase_bias_rad']:+.4f} rad")
    print(f"  phase bias in background     {result['background_phase_bias_rad']:+.4f} rad")

    domain_off = abs(result["domain_factor"] - 1.0)
    phase_off = abs(result["phase_factor"] - 1.0)
    print("\n  -> ", end="")
    if domain_off > 2 * phase_off:
        print("the BOUNDARY dominates. Work on the segmentation side: raise\n"
              "     loss.weights.projected_area_consistency or boundary_gradient_alignment.")
    elif phase_off > 2 * domain_off:
        print("the RECONSTRUCTION dominates. Work on the phase side: raise\n"
              "     loss.weights.phase or loss.phase.l1.")
    else:
        print("both contribute comparably; neither head alone explains the error.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)

    missing = 0
    for modality in args.modality:
        result = diagnose(cfg, modality, device, args.split)
        if result is None:
            missing += 1
            continue
        report(result)
        destination = result["_run_dir"] / f"bias_diagnosis_{args.split}.csv"
        # Created only now, when there is something to put in it.
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_csv(result["_rows"], destination)
        print(f"  per-image detail -> {destination}")

    if missing == len(args.modality):
        # Exit 3, not 1. run_v2.sh reads a traceback-free non-zero exit as a
        # reported finding, so "no checkpoint exists" was being logged as a
        # successful diagnostic -- and figure 3 then skipped for the same root
        # cause in a different log, with neither counting as a failure. 3 is the
        # reserved hard-error code the runner treats as FAILED.
        print("\nHARD ERROR: no checkpoint was found for any requested modality "
              f"({', '.join(args.modality)}).")
        print("  Train the arm first, or point --config at an arm that has been "
              "trained.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
