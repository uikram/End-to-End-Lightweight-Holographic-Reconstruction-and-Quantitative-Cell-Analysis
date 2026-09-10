"""Recover the aberration surface that was removed from the reference phase.

WHY THIS EXISTS
---------------
The delivered phase maps are not raw reconstructions. The acquiring group
applied aberration correction, background surface subtraction and phase
unwrapping before handing them over, so

    phi_reference  =  phi_raw_reconstruction  -  Psi

where Psi is a smooth surface produced by the objective's curvature and the
reference beam's tilt. The forward-model consistency term has to reproduce the
hologram that was actually *recorded*, and that hologram still contains Psi.
Propagating the reference phase alone therefore cannot match it, at any
distance -- which is precisely what the z calibration was reporting when it
found no identifiable minimum and said individual images disagreed.

There is a second, sharper reason this matters. A quadratic phase across the
field is exactly what defocus looks like. Psi and z are therefore partially
degenerate: with Psi missing, the search tries to absorb a fixed optical
aberration into a propagation distance, and no single z can do it. Restoring Psi
removes that degeneracy and makes z identifiable in the first place.

HOW IT IS RECOVERED
-------------------
For each field, reconstruct the off-axis hologram classically (sideband
isolation, no aberration removal), unwrap, and fit a low-order polynomial to the
difference from the delivered reference phase. On this dataset a second-order
surface explains 94-99% of that difference, which is what a curvature term plus
a tilt should look like and is strong evidence the model is right.

The off-axis arm is used for the estimate because it can be demodulated without
knowing z. The aberration belongs to the optics and to the processing, both
shared by the single acquisition that produced each matched pair, so the same
surface applies to the in-line arm.

THIS IS NOT LEAKAGE
-------------------
A second-order surface has six coefficients. Six numbers cannot encode cell
positions, boundaries or phase values; they can only describe a smooth bowl and
a tilt. The fit is reported per image with its residual so this can be checked
rather than asserted: if a surface ever explained substantially more than the
smooth component, that would show up as an implausibly high R-squared.

    python scripts/estimate_aberration.py --config config/base.yaml
    python scripts/estimate_aberration.py --config config/base.yaml --order 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import io as data_io
from holoqpi.physics import (
    detrend_polynomial,
    fit_polynomial_surface,
    reconstruct_off_axis,
    resolve_conjugate,
    unwrap_phase_2d,
)
from holoqpi.utils import get_logger, write_csv

LOGGER = get_logger(__name__)


def fit_surface(difference: torch.Tensor, order: int) -> tuple[np.ndarray, float]:
    """Least-squares polynomial fit; returns coefficients and R-squared."""
    coefficients, _, r_squared = fit_polynomial_surface(difference, order)
    return coefficients.cpu().numpy(), r_squared


def agreement(raw: torch.Tensor, reference: torch.Tensor, order: int) -> float:
    """Correlation of the two phase maps after the smooth surface is removed.

    This is the direct test of the model this script assumes: that the delivered
    phase differs from a classical reconstruction of the same hologram by a
    smooth surface and nothing else. Detrending removes that surface from both,
    so what remains is the cells, and they must agree. On this dataset the
    correlation is 0.87-0.95 once the sideband is resolved and about -0.93 when
    it is not, so it separates the two cases with no ambiguity at all and makes
    a poor field visible instead of silently corrupting the stored surface.
    """
    left = detrend_polynomial(raw, order)
    right = detrend_polynomial(reference, order)
    left = left - left.mean()
    right = right - right.mean()
    return float((left * right).mean() / (left.std() * right.std() + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--order", type=int, default=None,
                        help="default: optics.aberration.order from the config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="first N images only")
    parser.add_argument("--tolerance-factor", type=float, default=None,
                        help="default: optics.aberration.span_tolerance_factor")
    parser.add_argument("--min-agreement", type=float, default=None,
                        help="default: optics.aberration.min_agreement")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    order = args.order if args.order is not None else cfg.optics.aberration.order
    tolerance_factor = (
        args.tolerance_factor if args.tolerance_factor is not None
        else cfg.optics.aberration.span_tolerance_factor
    )
    min_agreement = (
        args.min_agreement if args.min_agreement is not None
        else cfg.optics.aberration.min_agreement
    )
    device = torch.device(args.device)
    optics = cfg.optics
    conjugate = optics.conjugate
    root = Path(cfg.paths.data_root)

    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found. Run `python main.py prepare` first.")
    stems = [row["stem"] for row in csv.DictReader(open(manifest))]
    if args.limit:
        stems = stems[: args.limit]

    LOGGER.info("fitting order-%d aberration surfaces over %d fields", order, len(stems))

    coefficients: dict[str, list[float]] = {}
    rows: list[dict] = []

    for position, stem in enumerate(stems, start=1):
        record = data_io.read_phase_bin(
            data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
        )
        reference = torch.from_numpy(record.phase.astype(np.float32)).to(device)

        hologram = data_io.read_hologram(
            data_io.hologram_path(root, cfg, stem, "off_axis")
        )
        hologram = data_io.align_to_phase_grid(hologram, cfg.data.phase_size, cfg.data.align)
        hologram = torch.from_numpy(hologram.astype(np.float32)).to(device).view(1, 1, *reference.shape)

        field = reconstruct_off_axis(
            hologram, optics.wavelength_um,
            optics.pixel_pitch_x_um, optics.pixel_pitch_y_um,
            0.0, cfg.model.frontend.sideband_radius_px, cfg.model.frontend.dc_exclusion_px,
        )
        # WITHOUT THIS THE STORED SURFACE IS THE WRONG SIGN ON MOST FIELDS.
        # The sideband a spectral argmax returns is arbitrary, and taking the
        # conjugate one negates the reconstructed phase. The difference from the
        # delivered map is then not an aberration surface at all but roughly
        # -(2 * reference + surface), which a fifth-order polynomial still fits
        # to R-squared 0.99 because the cells are a small part of the variance.
        # The forward-model term then adds a surface of the opposite handedness
        # to the predicted phase, and its residual falls as the phase is removed
        # -- which is exactly the anti-discriminative behaviour the off-axis arm
        # was showing. Measured here: 11 of 13 fields need the flip.
        field, flipped, skewness = resolve_conjugate(
            field, conjugate.detrend_order, conjugate.min_skewness
        ) if conjugate.enabled else (field, [False], [float("nan")])
        raw = unwrap_phase_2d(torch.angle(field))[0, 0]

        difference = raw - reference
        vector, r_squared = fit_surface(difference, order)
        coefficients[stem] = [float(v) for v in vector]
        rows.append({
            "stem": stem,
            "r_squared": r_squared,
            "surface_peak_to_valley_rad": float(difference.max() - difference.min()),
            "conjugated": bool(flipped[0]),
            "skewness": float(skewness[0]),
            "agreement": agreement(raw, reference, order),
        })

        if position % 50 == 0 or position == len(stems):
            LOGGER.info("  %d/%d", position, len(stems))

    quality = np.array([r["r_squared"] for r in rows], dtype=float)
    spans = np.array([r["surface_peak_to_valley_rad"] for r in rows], dtype=float)

    # R-SQUARED IS NOT A VALID QUALITY GATE HERE, and using it as one would be a
    # trap. When the phase unwrapping fails -- which happens on fields with many
    # wraps, so disproportionately on the blebbistatin conditions where cells
    # round up and thicken -- it produces a huge smooth ramp. A fifth-order
    # polynomial fits that ramp essentially perfectly, so a catastrophic failure
    # scores R-squared = 1.000, better than a good fit to a real aberration.
    #
    # The valid check is physical: an objective's curvature plus a reference
    # tilt is tens of radians across the field, not hundreds. Fields whose
    # surface is far outside the population are unwrapping failures and are
    # marked unusable, so the forward-model term skips them rather than being
    # handed a several-hundred-radian phase error.
    median_span = float(np.median(spans))
    limit = max(tolerance_factor * median_span, median_span + 6.0 * float(
        np.median(np.abs(spans - median_span))) + 1e-9)

    # The second gate is the one that actually tests the model. After the smooth
    # surface is removed from both maps, what is left is the cells, and a
    # correct surface leaves them correlated. A field that fails this has
    # something wrong the polynomial cannot express -- an unresolved sideband, a
    # broken unwrap, a misregistration -- and no coefficient vector will fix it.
    agreements = np.array([r["agreement"] for r in rows], dtype=float)
    valid = (spans <= limit) & (agreements >= min_agreement)
    for row, ok, span, score in zip(rows, valid, spans, agreements):
        row["valid"] = bool(ok)
        if ok:
            row["reason"] = ""
        elif span > limit:
            row["reason"] = f"surface {span:.0f} rad exceeds {limit:.0f} rad"
        else:
            row["reason"] = f"detrended agreement {score:+.2f} below {min_agreement:.2f}"

    quality = quality[np.isfinite(quality)]

    payload = {
        "order": order,
        "grid": [cfg.data.phase_size, cfg.data.phase_size],
        "note": (
            "Coefficients of a polynomial in x, y normalised to [-1, 1] over the "
            "full phase grid, ordered as (x**i)*(y**j) for i in 0..order, "
            "j in 0..order-i. Add this surface to a predicted phase before "
            "propagating, to restore what the delivered reference phase had "
            "removed."
        ),
        "coefficients": coefficients,
        "valid": {row["stem"]: bool(row["valid"]) for row in rows},
        # Recorded so the reconstruction can be reproduced exactly, and so the
        # fraction of fields whose sideband needed flipping is reportable.
        "conjugated": {row["stem"]: bool(row["conjugated"]) for row in rows},
        "median_span_rad": median_span,
        "span_limit_rad": float(limit),
        "min_agreement": float(min_agreement),
    }
    destination = root / cfg.paths.aberration_file
    destination.write_text(json.dumps(payload))
    write_csv(rows, Path(cfg.paths.output_root) / "aberration_fit.csv")

    print(f"\n=== aberration surfaces, order {order} ===")
    print(f"  fields fitted        {len(rows)}")
    print(f"  R^2  median {np.median(quality):.4f}   "
          f"p10 {np.percentile(quality, 10):.4f}   min {quality.min():.4f}")
    print(f"  surface peak-to-valley  median {median_span:.1f} rad   "
          f"p95 {np.percentile(spans, 95):.1f} rad   max {spans.max():.1f} rad")
    conjugated = int(sum(r["conjugated"] for r in rows))
    print(f"  sideband conjugated  {conjugated} / {len(rows)} fields"
          f"   (median |skewness| {np.median(np.abs([r['skewness'] for r in rows])):.2f})")
    print(f"  detrended agreement  median {np.median(agreements):+.3f}   "
          f"min {agreements.min():+.3f}")
    print(f"  usable fields        {int(valid.sum())} / {len(rows)}"
          f"   (span limit {limit:.0f} rad, agreement >= {min_agreement:.2f})")
    if (~valid).any():
        by_span = int(((spans > limit)).sum())
        by_agreement = int(((agreements < min_agreement) & (spans <= limit)).sum())
        print(f"  rejected: {by_span} for surface magnitude (unwrapping failures), "
              f"{by_agreement} for disagreeing with the delivered phase")
        for row in [r for r in rows if not r["valid"]][:6]:
            print(f"    {row['stem']:<32} {row['surface_peak_to_valley_rad']:8.0f} rad"
                  f"   R^2 {row['r_squared']:.3f}   {row['reason']}")
        if int((~valid).sum()) > 6:
            print(f"    ... and {int((~valid).sum()) - 6} more")
        print("    Note their R^2: a perfect fit to a broken unwrap still scores 1.000,")
        print("    which is exactly why the gate is the surface magnitude and the")
        print("    detrended agreement, and not R^2.")
    print(f"  surfaces -> {destination}")
    print(f"  per-image quality -> {Path(cfg.paths.output_root) / 'aberration_fit.csv'}")

    print("\n  -> ", end="")
    if np.median(quality) > 0.85:
        print("the difference between the raw reconstruction and the delivered\n"
              "     phase IS a smooth surface, as the acquiring group described. The\n"
              "     forward model can now restore it, and z should become identifiable:\n"
              "     re-run scripts/calibrate_z.py.")
    else:
        print("the difference is NOT well described by a smooth surface. Either the\n"
              "     processing did more than aberration removal, or the classical\n"
              "     reconstruction used here is failing. Check\n"
              "     scripts/conventional_baseline.py before relying on this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
