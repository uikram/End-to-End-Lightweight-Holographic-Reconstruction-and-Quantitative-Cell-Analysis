"""Does the measurement pipeline return the right number when the answer is known?

WHY THIS EXISTS
---------------
Every quantitative result in this project passes through the same chain:
watershed instance labelling, an area filter, a per-cell sum of phase, and a
scalar calibration into picograms. If that chain has an error of its own, it is
added to every reported measurement error and there is no way to tell the two
apart -- a 4% dry-mass error means something quite different if the pipeline's
own floor is 0.5% than if it is 3%.

On real data the floor cannot be measured, because the true mass of a real cell
is not known. On synthetic data it can, exactly, because the phase profile is
chosen and its integral is available in closed form.

THE CONSTRUCTION
----------------
Each cell is a spherical cap of radius r and peak phase phi_0:

    phi(d) = phi_0 * sqrt(1 - (d/r)^2)     for d <= r

whose integrals are exact and elementary:

    area              A = pi r^2
    integrated phase  V = phi_0 * (2/3) * pi * r^2
    dry mass          m = lambda / (2 pi alpha) * V

So for every synthetic cell there is a number the pipeline must reproduce, and
the difference is pipeline error and nothing else: no network, no
reconstruction, no labels, no reference phase.

WHAT AN ERROR HERE WOULD MEAN
-----------------------------
DISCRETISATION. A disc of radius r rasterised on a pixel grid does not have area
exactly pi r^2. That error falls as 1/r, so it is reported against cell size and
not as a single number -- and it sets the smallest cell worth measuring, which
is a configuration value (``min_cell_area_um2``) currently chosen on
biological grounds alone.

INSTANCE SPLITTING. Watershed can merge two touching cells or split one. Either
corrupts the per-cell measurement while leaving the total almost intact, so the
per-cell and summed errors are both reported: a large gap between them is a
labelling failure rather than an integration failure.

THE CALIBRATION ITSELF. lambda / (2 pi alpha) and the pixel area are applied
here exactly as in production, from the same config, so a wrong wavelength,
refraction increment or pixel pitch shows up as a constant multiplicative bias
in the mass column while the area column stays clean. That is a specific and
recognisable signature, and it is the one that the pitch and alpha corrections
of 2026-09-10 were about.

THE SECOND TEST: DOES THE LOSS MEASURE WHAT IT CLAIMS
-----------------------------------------------------
With exact ground truth available, the per-cell integrated-phase term can be
checked against the analytic relative error rather than against another
implementation of itself. A boundary shifted by a known amount produces a known
change in each cell's mass; the term must return that number. The unit tests in
``scripts/selftest.py`` check the same mechanism on flat squares, where the
answer is trivially 0.2; this checks it on a curved profile at realistic cell
sizes, where an off-by-one in the domain would actually show.

    python scripts/synthetic_validation.py --config config/base.yaml
    python scripts/synthetic_validation.py --config config/base.yaml --fields 20
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.analysis.cells import calibration_from_config, match_cells, measure_cells
from holoqpi.config import load_config, parse_overrides
from holoqpi.data.masks import split_instances
from holoqpi.utils import get_logger, write_csv

LOGGER = get_logger(__name__)


def synthesise(
    size: int,
    radii_px: list[float],
    peak_phase: list[float],
    generator: np.random.Generator,
    margin: int = 12,
    separation: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Place spherical-cap cells at non-overlapping positions.

    ``separation`` keeps a gap between cells so that instance labelling is not
    itself under test here: merging is a real failure mode but it is a
    SEPARATE one, and mixing it in would leave an integration error and a
    labelling error indistinguishable. The touching case is exercised
    deliberately by ``--touching``.
    """
    phase = np.zeros((size, size), dtype=np.float64)
    mask = np.zeros((size, size), dtype=np.uint8)
    truth: list[dict] = []
    centres: list[tuple[float, float, float]] = []

    y_grid = np.arange(size, dtype=np.float64)[:, None]
    x_grid = np.arange(size, dtype=np.float64)[None, :]

    for radius, phi0 in zip(radii_px, peak_phase):
        for _ in range(200):
            centre_y = float(generator.uniform(margin + radius, size - margin - radius))
            centre_x = float(generator.uniform(margin + radius, size - margin - radius))
            if all(
                math.hypot(centre_y - y, centre_x - x) > radius + r + separation
                for y, x, r in centres
            ):
                centres.append((centre_y, centre_x, radius))
                break
        else:
            continue

        distance = np.hypot(y_grid - centre_y, x_grid - centre_x)
        inside = distance <= radius
        profile = np.zeros_like(phase)
        # sqrt(1 - (d/r)^2), clipped at zero so the boundary pixel is exact.
        profile[inside] = phi0 * np.sqrt(
            np.clip(1.0 - (distance[inside] / radius) ** 2, 0.0, None)
        )
        phase += profile
        mask[inside] = 1
        truth.append({
            "centre_y": centre_y, "centre_x": centre_x,
            "radius_px": radius, "peak_phase_rad": phi0,
            "area_px_exact": math.pi * radius ** 2,
            "phase_sum_exact": phi0 * (2.0 / 3.0) * math.pi * radius ** 2,
        })

    return phase.astype(np.float32), mask.astype(np.int64), truth


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--fields", type=int, default=8)
    parser.add_argument("--cells-per-field", type=int, default=12)
    parser.add_argument("--size", type=int, default=None,
                        help="field size in pixels; default data.phase_size")
    parser.add_argument("--radius-um", type=float, nargs=2, default=[2.5, 8.0],
                        help="cell radius range in micrometres")
    parser.add_argument("--peak-phase", type=float, nargs=2, default=[0.8, 3.0],
                        help="peak phase range in radians")
    parser.add_argument("--touching", action="store_true",
                        help="place cells with no separation, to exercise watershed")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    calibration = calibration_from_config(cfg)
    measurement_cfg = cfg.evaluation.measurement
    method = cfg.evaluation.segmentation.instance_from
    distance = cfg.mask_generation.watershed_min_distance_px
    size = args.size or cfg.data.phase_size
    generator = np.random.default_rng(cfg.project.seed)

    pitch = calibration.pitch_x_um
    pixel_area = calibration.pixel_area_um2
    pg_per_radian_pixel = calibration.picogram_per_radian_pixel

    print(f"\n  {args.fields} synthetic fields of {size} px, "
          f"{args.cells_per_field} cells each")
    print(f"  radius {args.radius_um[0]:.1f}-{args.radius_um[1]:.1f} um "
          f"= {args.radius_um[0] / pitch:.1f}-{args.radius_um[1] / pitch:.1f} px   "
          f"peak phase {args.peak_phase[0]:.1f}-{args.peak_phase[1]:.1f} rad")
    print(f"  calibration: pitch {pitch:.6f} um, pixel area {pixel_area:.6f} um^2, "
          f"{pg_per_radian_pixel:.6e} pg per rad-pixel")
    print(f"  instance labelling: {method}, watershed distance {distance} px")

    rows: list[dict] = []
    field_rows: list[dict] = []

    for field in range(args.fields):
        radii = list(
            generator.uniform(args.radius_um[0], args.radius_um[1], args.cells_per_field)
            / pitch
        )
        peaks = list(
            generator.uniform(args.peak_phase[0], args.peak_phase[1], args.cells_per_field)
        )
        phase, mask, truth = synthesise(
            size, radii, peaks, generator,
            separation=0.0 if args.touching else 4.0,
        )
        if not truth:
            continue

        labels = split_instances((mask > 0).astype(np.uint8), method, distance)
        measured = measure_cells(
            phase, mask.astype(np.int64), calibration, measurement_cfg, method,
            distance, labels=labels,
        )

        # Pair each measured cell to the truth cell whose centre it contains.
        # Centre containment rather than IoU, because the truth here is a list
        # of analytic cells and not a label image, and containment cannot pair
        # a cell with a neighbour.
        for record in measured:
            best = None
            for index, cell in enumerate(truth):
                separation_px = math.hypot(
                    record["centroid_y"] - cell["centre_y"],
                    record["centroid_x"] - cell["centre_x"],
                )
                if separation_px <= cell["radius_px"] and (
                    best is None or separation_px < best[0]
                ):
                    best = (separation_px, index)
            if best is None:
                continue
            cell = truth[best[1]]

            area_exact = cell["area_px_exact"] * pixel_area
            mass_exact = cell["phase_sum_exact"] * pg_per_radian_pixel
            rows.append({
                "field": field,
                "radius_um": cell["radius_px"] * pitch,
                "peak_phase_rad": cell["peak_phase_rad"],
                "area_exact_um2": area_exact,
                "area_measured_um2": record["area_um2"],
                "area_relative_error": (record["area_um2"] - area_exact) / area_exact,
                "mass_exact_pg": mass_exact,
                "mass_measured_pg": record["dry_mass_pg"],
                "mass_relative_error": (record["dry_mass_pg"] - mass_exact) / mass_exact,
                "centroid_offset_px": best[0],
            })

        # The summed view. Instance labelling cannot change a total, so a
        # per-cell error much larger than the summed one is a labelling
        # failure and not an integration failure.
        total_exact = sum(c["phase_sum_exact"] for c in truth) * pg_per_radian_pixel
        total_measured = sum(r["dry_mass_pg"] for r in measured)
        field_rows.append({
            "field": field,
            "cells_placed": len(truth),
            "cells_measured": len(measured),
            "mass_total_exact_pg": total_exact,
            "mass_total_measured_pg": total_measured,
            "mass_total_relative_error": (
                (total_measured - total_exact) / total_exact if total_exact else float("nan")
            ),
        })

    if not rows:
        raise SystemExit("no cells were paired; check the radius range against min_cell_area_um2")

    write_csv(rows, Path(cfg.paths.output_root) / "synthetic_validation_cells.csv")
    write_csv(field_rows, Path(cfg.paths.output_root) / "synthetic_validation_fields.csv")

    area_error = np.array([r["area_relative_error"] for r in rows])
    mass_error = np.array([r["mass_relative_error"] for r in rows])
    radius = np.array([r["radius_um"] for r in rows])
    total_error = np.array([f["mass_total_relative_error"] for f in field_rows])
    total_error = total_error[np.isfinite(total_error)]

    placed = sum(f["cells_placed"] for f in field_rows)
    found = sum(f["cells_measured"] for f in field_rows)

    print(f"\n=== pipeline error against exact ground truth ===")
    print(f"  cells placed {placed}   measured {found}   paired {len(rows)}")
    print(f"  area  signed  mean {area_error.mean():+.4%}   "
          f"median {np.median(area_error):+.4%}   "
          f"p95 |.| {np.percentile(np.abs(area_error), 95):.4%}")
    print(f"  mass  signed  mean {mass_error.mean():+.4%}   "
          f"median {np.median(mass_error):+.4%}   "
          f"p95 |.| {np.percentile(np.abs(mass_error), 95):.4%}")
    if total_error.size:
        print(f"  field-summed mass  mean {total_error.mean():+.4%}   "
              f"median {np.median(total_error):+.4%}")

    # THE GAP BETWEEN THE TWO IS THE AREA FILTER, and it is easy to miss.
    # Per-cell errors are computed over cells that were MEASURED, so a cell
    # rejected by min_cell_area_um2 never enters them. The field total is over
    # the cells that were PLACED, so it does. A per-cell error of 0.05% next to
    # a summed error of -2% is not an inconsistency: it is the filter, and it
    # says exactly what fraction of the specimen's mass the filter discards.
    lost = placed - found
    if lost:
        floor_um2 = cfg.evaluation.measurement.min_cell_area_um2
        floor_radius = math.sqrt(floor_um2 / math.pi)
        print(f"\n  {lost} of {placed} cells were never measured. "
              f"min_cell_area_um2 = {floor_um2:.1f} um^2 rejects anything below "
              f"{floor_radius:.2f} um of radius,")
        print(f"  and the synthesised radii start at {args.radius_um[0]:.2f} um. "
              f"That is the whole of the gap between a per-cell error of "
              f"{np.abs(mass_error).mean():.2%}")
        print(f"  and a field-summed error of {total_error.mean():+.2%}: the filter "
              f"discards small cells, so the")
        print(f"  TOTAL mass of a field is biased low by that much while every cell "
              f"it keeps is exact.")
        print(f"  Report the filter's cost whenever a field total or a population "
              f"mean is reported.")

    # Discretisation falls with cell size, so the floor is reported by size.
    print(f"\n  {'radius (um)':>14}{'cells':>7}{'|dA/A|':>10}{'|dm/m|':>10}")
    edges = np.percentile(radius, [0, 25, 50, 75, 100])
    for low, high in zip(edges[:-1], edges[1:]):
        band = (radius >= low) & (radius <= high)
        if band.sum() == 0:
            continue
        print(f"  {low:>6.2f}-{high:<7.2f}{int(band.sum()):>7}"
              f"{np.abs(area_error[band]).mean():>10.4%}"
              f"{np.abs(mass_error[band]).mean():>10.4%}")

    # ---- the loss term, against the analytic answer ---------------------
    import torch

    from holoqpi.losses.terms import CellIntegratedPhase, CellProjectedArea

    phase, mask, truth = synthesise(
        size, [6.0 / pitch] * 6, [2.0] * 6, np.random.default_rng(cfg.project.seed)
    )
    labels = split_instances((mask > 0).astype(np.uint8), method, distance)
    instances = torch.from_numpy(labels.astype(np.int64))[None]
    phase_t = torch.from_numpy(phase)[None, None].float()
    foreground = torch.from_numpy((mask > 0).astype(np.float32))[None, None]

    integrated = CellIntegratedPhase(
        cfg.loss.physics.volume_epsilon, cfg.loss.physics.cell_min_reference_rad, None
    )
    area_term = CellProjectedArea(
        cfg.loss.physics.volume_epsilon, cfg.loss.physics.min_foreground_pixels, None
    )

    exact_zero = float(integrated(foreground, phase_t, foreground, phase_t, instances))
    scaled = float(integrated(foreground, phase_t * 1.07, foreground, phase_t, instances))
    area_zero = float(area_term(foreground, foreground, instances))

    print(f"\n=== the loss term against the analytic answer, 6 cells of "
          f"{6.0:.1f} um radius ===")
    print(f"  exact prediction            {exact_zero:.3e}   (must be 0)")
    print(f"  phase scaled by 1.07        {scaled:.6f}   (must be 0.070000)")
    print(f"  area term, exact foreground {area_zero:.3e}   (must be 0)")

    ok = (
        abs(exact_zero) < 1e-6
        and abs(scaled - 0.07) < 1e-4
        and abs(area_zero) < 1e-6
    )

    print("\n  -> ", end="")
    if not ok:
        print("THE LOSS TERM DOES NOT RETURN THE ANALYTIC RELATIVE ERROR on a curved\n"
              "     profile, even though it does on the flat squares in selftest.py. Fix\n"
              "     that before running any experiment that depends on it.")
        return 2
    floor = float(np.mean(np.abs(mass_error)))
    print(f"the measurement pipeline reproduces exact ground truth to "
          f"{floor:.2%} in dry\n"
          f"     mass on the cells it keeps, and the loss term returns the analytic\n"
          f"     relative error. That {floor:.2%} is the FLOOR under every PER-CELL\n"
          f"     measurement error this project reports: a measured error of 4% is\n"
          f"     {4.0 / max(100 * floor, 1e-9):.0f}x the floor, so it is model error and not "
          f"pipeline error. The\n"
          f"     floor grows for small cells -- see the table by radius.")
    if lost and total_error.size:
        print(f"\n     But the FIELD TOTAL has a separate and much larger bias, "
              f"{total_error.mean():+.2%},\n"
              f"     entirely from the area filter discarding small cells. The two floors\n"
              f"     are different numbers for different claims: quote {floor:.2%} for "
              f"'the mass of\n"
              f"     this cell' and {total_error.mean():+.2%} for 'the mass on this field'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
