"""How much measurement error does one pixel of boundary error cause?

WHY THIS EXISTS
---------------
Every quantitative result in this project is an integral over a segmented
domain. Projected area is the domain's size and dry mass is the phase summed
inside it, so a boundary that is one pixel too generous inflates both. The
question a reviewer will ask, and the question that decides how good the
segmentation has to be, is the exchange rate: what does a Dice of 0.95 cost in
picograms?

That question is answered WITHOUT A MODEL. Take the reference mask and the
reference phase, move the boundary by a known amount, and measure what happens
to the numbers. No network is involved, nothing is trained, and the result is a
property of the specimen and the optics rather than of any method -- which is
what makes it a calibration curve other people can use rather than a score.

WHAT IT MEASURES, AND WHY THE TWO CURVES DIFFER
-----------------------------------------------
Dilating by k pixels adds an annulus of area 4kr-ish around a cell of radius r,
so the AREA error grows roughly linearly in k and is purely geometric.

The MASS error cannot grow as fast, and the reason is physical rather than
statistical: the annulus added by dilation lies OUTSIDE the cell, where the
phase is near zero, while the annulus removed by erosion lies just inside the
membrane, where a cell is thinnest. Both boundaries are in low-phase territory.
So mass is systematically less sensitive to the boundary than area is, and the
ratio of the two slopes is a number this instrument and this specimen fix.

If that ratio is well below one, the headline claim is available: dry mass is
robust to segmentation error in a way projected area is not, so a
measurement-grade mass does not require a measurement-grade boundary. If it is
near one, it is not available, and the honest statement is that mass inherits
the boundary error directly.

THE ASYMMETRY MATTERS TOO
-------------------------
Dilation and erosion are reported separately and never averaged. A segmenter
biased outwards and one biased inwards by the same Dice do not make the same
measurement error, because the phase profile is not symmetric about the
boundary. A single "sensitivity" number would hide the sign of the bias, which
is exactly the quantity a Bland-Altman plot of the final result is about.

    python scripts/error_propagation.py --config config/base.yaml
    python scripts/error_propagation.py --config config/base.yaml --max-shift 7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.analysis.cells import calibration_from_config, match_cells, measure_cells
from holoqpi.config import load_config, parse_overrides
from holoqpi.data import io as data_io
from holoqpi.data.masks import split_instances
from holoqpi.metrics.segmentation import SegmentationMetrics
from holoqpi.utils import get_logger, write_csv

LOGGER = get_logger(__name__)


def shift_boundary(binary: np.ndarray, shift: int) -> np.ndarray:
    """Dilate for positive shift, erode for negative, by that many pixels.

    A disc structuring element rather than the default cross, so the boundary
    moves by the requested distance in every direction instead of by sqrt(2)
    times less on the diagonals. On a cell of radius 15 px that difference is
    about 8% of the annulus, which is the same order as the effect being
    measured.
    """
    from scipy import ndimage

    if shift == 0:
        return binary.astype(bool)
    radius = abs(int(shift))
    size = 2 * radius + 1
    grid = np.arange(size) - radius
    element = (grid[:, None] ** 2 + grid[None, :] ** 2) <= radius ** 2 + 1e-9
    operation = ndimage.binary_dilation if shift > 0 else ndimage.binary_erosion
    return operation(binary.astype(bool), structure=element)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--max-shift", type=int, default=5,
                        help="largest boundary displacement in pixels (default 5)")
    parser.add_argument("--limit", type=int, default=None, help="first N fields only")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    calibration = calibration_from_config(cfg)
    measurement_cfg = cfg.evaluation.measurement
    method = cfg.evaluation.segmentation.instance_from
    distance = cfg.mask_generation.watershed_min_distance_px
    root = Path(cfg.paths.data_root)

    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found. Run `python main.py prepare` first.")
    stems = [row["stem"] for row in csv.DictReader(open(manifest))]
    if args.limit:
        stems = stems[: args.limit]

    shifts = list(range(-abs(args.max_shift), abs(args.max_shift) + 1))
    pitch = calibration.pitch_x_um
    LOGGER.info(
        "propagating boundary shifts %d..%d px (%.3f..%.3f um) over %d fields",
        shifts[0], shifts[-1], shifts[0] * pitch, shifts[-1] * pitch, len(stems),
    )

    rows: list[dict] = []
    for position, stem in enumerate(stems, start=1):
        record = data_io.read_phase_bin(
            data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
        )
        phase = record.phase.astype(np.float32)
        mask = data_io.read_mask(data_io.mask_path(root, cfg, stem))
        mask = data_io.align_to_phase_grid(mask, cfg.data.phase_size, cfg.data.align)
        truth = (mask > 0).astype(np.uint8)
        if not truth.any():
            continue

        truth_labels = split_instances(truth, method, distance)
        reference = measure_cells(
            phase, truth, calibration, measurement_cfg, method, distance,
            labels=truth_labels,
        )
        if not reference:
            continue

        for shift in shifts:
            moved = shift_boundary(truth, shift).astype(np.uint8)
            if not moved.any():
                # Eroding a field out of existence is a real outcome at large
                # negative shifts and is recorded as a total loss rather than
                # skipped, which would flatter the curve.
                rows.append({
                    "stem": stem, "shift_px": shift, "shift_um": shift * pitch,
                    "cells": 0, "dice": 0.0,
                    "area_relative_error": 1.0, "mass_relative_error": 1.0,
                    "area_signed": -1.0, "mass_signed": -1.0,
                })
                continue

            moved_labels = split_instances(moved, method, distance)
            moved_cells = measure_cells(
                phase, moved, calibration, measurement_cfg, method, distance,
                labels=moved_labels,
            )
            pairs = match_cells(
                moved_cells, reference, moved_labels, truth_labels,
                measurement_cfg.match_iou_threshold,
            )

            # Dice on the same pair, so the exchange rate is expressed against
            # the metric a segmentation paper reports rather than against a
            # pixel count nobody quotes.
            segmentation = SegmentationMetrics(
                num_classes=cfg.model.segmentation_classes,
                boundary_tolerance=cfg.evaluation.segmentation.boundary_tolerance_px,
                instance_method=method,
                watershed_min_distance=distance,
            )
            segmentation.update(
                moved.astype(np.int64)[None],
                truth.astype(np.int64)[None],
                prediction_instances=[moved_labels],
                target_instances=[truth_labels],
            )
            dice = segmentation.compute().get("seg_dice", float("nan"))

            if not pairs:
                rows.append({
                    "stem": stem, "shift_px": shift, "shift_um": shift * pitch,
                    "cells": 0, "dice": dice,
                    "area_relative_error": float("nan"),
                    "mass_relative_error": float("nan"),
                    "area_signed": float("nan"), "mass_signed": float("nan"),
                })
                continue

            area = np.array([[p["area_um2"], r["area_um2"]] for p, r in pairs])
            mass = np.array([[p["dry_mass_pg"], r["dry_mass_pg"]] for p, r in pairs])
            usable_mass = np.abs(mass[:, 1]) > 0
            rows.append({
                "stem": stem,
                "shift_px": shift,
                "shift_um": shift * pitch,
                "cells": len(pairs),
                "dice": dice,
                # Unsigned, for "how wrong", and signed, for "in which
                # direction". Both, because the second is the bias.
                "area_relative_error": float(
                    np.mean(np.abs(area[:, 0] - area[:, 1]) / area[:, 1])
                ),
                "mass_relative_error": float(
                    np.mean(np.abs(mass[usable_mass, 0] - mass[usable_mass, 1])
                            / mass[usable_mass, 1])
                ) if usable_mass.any() else float("nan"),
                "area_signed": float(np.mean((area[:, 0] - area[:, 1]) / area[:, 1])),
                "mass_signed": float(
                    np.mean((mass[usable_mass, 0] - mass[usable_mass, 1])
                            / mass[usable_mass, 1])
                ) if usable_mass.any() else float("nan"),
            })

        if position % 20 == 0 or position == len(stems):
            LOGGER.info("  %d/%d", position, len(stems))

    if not rows:
        raise SystemExit("no fields with reference cells; nothing to propagate")

    report = Path(cfg.paths.output_root) / "error_propagation.csv"
    write_csv(rows, report)

    print(f"\n=== boundary error -> measurement error, {len(stems)} fields ===")
    print(f"  pixel pitch {pitch:.6f} um, so one pixel of boundary is "
          f"{pitch:.3f} um of radius")
    print(f"\n  {'shift':>7}{'um':>8}{'Dice':>8}{'|dA/A|':>10}{'|dm/m|':>10}"
          f"{'dA/A':>10}{'dm/m':>10}{'mass/area':>11}")

    summary: list[dict] = []
    for shift in shifts:
        group = [r for r in rows if r["shift_px"] == shift]
        if not group:
            continue

        def column(key: str) -> float:
            values = np.array([r[key] for r in group], dtype=float)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else float("nan")

        area = column("area_relative_error")
        mass = column("mass_relative_error")
        ratio = mass / area if area > 0 else float("nan")
        summary.append({
            "shift_px": shift, "shift_um": shift * pitch, "dice": column("dice"),
            "area_relative_error": area, "mass_relative_error": mass,
            "area_signed": column("area_signed"), "mass_signed": column("mass_signed"),
            "mass_over_area": ratio,
        })
        print(f"  {shift:>7d}{shift * pitch:>8.3f}{column('dice'):>8.4f}"
              f"{area:>10.4f}{mass:>10.4f}"
              f"{column('area_signed'):>+10.4f}{column('mass_signed'):>+10.4f}"
              f"{ratio:>11.3f}")

    write_csv(summary, Path(cfg.paths.output_root) / "error_propagation_summary.csv")

    # The exchange rate at one pixel, which is the number to quote.
    one_out = next((s for s in summary if s["shift_px"] == 1), None)
    one_in = next((s for s in summary if s["shift_px"] == -1), None)
    ratios = np.array(
        [s["mass_over_area"] for s in summary
         if s["shift_px"] != 0 and np.isfinite(s["mass_over_area"])],
        dtype=float,
    )

    if one_out and one_in:
        print(f"\n  ONE PIXEL of boundary error, which is {pitch:.3f} um:")
        print(f"    dilated  by 1 px   Dice {one_out['dice']:.4f}   "
              f"area {one_out['area_signed']:+.2%}   mass {one_out['mass_signed']:+.2%}")
        print(f"    eroded   by 1 px   Dice {one_in['dice']:.4f}   "
              f"area {one_in['area_signed']:+.2%}   mass {one_in['mass_signed']:+.2%}")
        asymmetry = abs(one_out["mass_signed"]) - abs(one_in["mass_signed"])
        print(f"    asymmetry in mass  {asymmetry:+.2%}   "
              f"(outward minus inward, in magnitude)")

    if ratios.size:
        print(f"\n  mass error / area error   median {np.median(ratios):.3f}"
              f"   min {ratios.min():.3f}   max {ratios.max():.3f}")

    print(f"\n  per-field detail -> {report}")
    print(f"  summary -> {Path(cfg.paths.output_root) / 'error_propagation_summary.csv'}")

    print("\n  -> ", end="")
    if not ratios.size:
        print("no usable ratio: too few cells survived pairing at any shift. Check\n"
              "     the mask and the IoU threshold before interpreting anything.")
        return 1
    median_ratio = float(np.median(ratios))
    if median_ratio < 0.7:
        print(f"DRY MASS IS MORE ROBUST TO SEGMENTATION ERROR THAN AREA IS, by a\n"
              f"     factor of {1 / median_ratio:.2f} on this specimen. The boundary sits where\n"
              f"     the cell is thinnest, so the pixels a boundary error adds or removes\n"
              f"     carry little phase. A measurement-grade mass therefore does not\n"
              f"     require a measurement-grade boundary -- and this curve says how much\n"
              f"     boundary error each mass tolerance buys. Report both curves and the\n"
              f"     dilation/erosion asymmetry; do not average them.")
        return 0
    if median_ratio > 1.3:
        print(f"dry mass is MORE sensitive to the boundary than area is (ratio\n"
              f"     {median_ratio:.2f}). That happens when the phase peaks near the rim rather\n"
              f"     than the centre. Check the phase profile before reporting: it is a\n"
              f"     real possibility for a rounded or a rimmed cell, but it is also what\n"
              f"     a misregistration between mask and phase would look like.")
        return 0
    print(f"mass and area inherit the boundary error at about the same rate (ratio\n"
          f"     {median_ratio:.2f}). The robustness claim is NOT available on this data, and\n"
          f"     the honest statement is that dry mass carries the segmentation error\n"
          f"     directly. Quote the exchange rate at one pixel and move on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
