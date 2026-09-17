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

PER-FIELD SURFACES ARE NOT DEPLOYABLE, AND A GLOBAL ONE IS
----------------------------------------------------------
Read the recovery procedure again: each field's surface is fitted from the
difference between that field's reconstruction and that field's DELIVERED
REFERENCE PHASE -- which is the network's own training target.

That has two consequences, and the second is the serious one.

1. It cannot be reproduced at inference. Given a new hologram and no reference
   phase, there is no way to obtain that field's surface. A forward model that
   depends on it is therefore not deployable, whatever it scores in training.

2. It makes the forward-model discrimination test partly circular. The surface
   was fitted on the assumption that the reference phase is correct, so the
   reference phase is guaranteed a favourable residual, and the comparison
   against deliberately degraded phases is no longer clean.

Neither is fixed by arguing about how many coefficients there are. A 5th-order
2-D polynomial has 21 of them, which is far too few to encode cell positions --
so this is not leakage of the answer -- but the surface is still target-derived,
and that is enough to disqualify it from a reported result.

The fix is to treat the aberration as what it physically is: a fixed property of
the objective and the reference tilt, shared by every acquisition. This script
therefore also writes a GLOBAL surface, the per-coefficient MEDIAN over the
fields of one split (train by default) that passed the per-field quality gates.
The median rather than the mean, so a single bad fit cannot drag the
calibration. That surface is applied to every field including val and test, is
obtainable from a one-off calibration in a real deployment, and is what
``optics.aberration.mode: global`` selects.

WHAT THE GLOBAL SURFACE COSTS, MEASURED
---------------------------------------
One global surface does not reproduce the per-field surfaces on this data. Run
this script and it reports, on the 13 local fields:

    global surface peak-to-valley                    15.21 rad
    per-field departure from it, all terms   median   10.62 rad   (69.8%)
    the same departure, degree >= 2 only     median    3.91 rad   (25.7%)
    per-field tilt, degree == 1 only         median   13.98 rad

So the CURVATURE is close to static, at a quarter of the surface's magnitude,
which is what a fixed objective should look like. The degree <= 1 part is not,
and it is the larger share.

A HYBRID WAS TRIED AND THE DATA REJECTED IT
-------------------------------------------
A ramp is the one component that ought to be obtainable without the reference
phase, for a specific mechanical reason. ``reconstruct_off_axis`` centres the
sideband with ``torch.roll``, which shifts by WHOLE FFT bins only, so a carrier
that does not land on a bin leaves a residual ramp of up to pi radians across
the field. ``estimate_carrier`` refines the same carrier to a small fraction of
a bin from the hologram's spectrum alone. If the fitted tilt were that
remainder, it could be predicted at inference and the surface would be
deployable: global curvature plus a per-field carrier-derived ramp. That is
what ``optics.aberration.mode: hybrid`` builds.

It does not work here. Predicting the ramp from the carrier explains 10.9% of
the fitted degree-1 tilt and leaves a median 12.46 rad, against 13.98 rad
before. Field by field the two are uncorrelated in sign as well as magnitude.
The tilt is therefore something else -- reference-beam drift between
acquisitions, or a ramp introduced by the 2-D unwrapping itself -- and this
script cannot separate those without information it does not have.

The consequence has to be stated rather than worked around: on this
instrument, the deployable aberration model is the global surface, and it
leaves an unmodelled per-field ramp of order 10 rad in the forward model. That
makes the forward-model term a DIAGNOSTIC on this data, not a deployable
physical constraint, and the experiment built on it must be reported that way.

All three modes are written to the same file. ``per_field`` is retained only so
this comparison can be made; it is target-derived and must not be used for a
reported result. ``hybrid`` is retained because a rejected hypothesis with a
number attached is worth more than a deleted one.

    python scripts/estimate_aberration.py --config config/base.yaml
    python scripts/estimate_aberration.py --config config/base.yaml --order 3
    python scripts/estimate_aberration.py --config config/base.yaml --global-from val
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import io as data_io
from holoqpi.physics import (
    detrend_polynomial,
    estimate_carrier,
    fit_polynomial_surface,
    reconstruct_off_axis,
    resolve_conjugate,
    unwrap_phase_2d,
)
from holoqpi.utils import get_logger, write_csv

LOGGER = get_logger(__name__)


def monomial_degrees(order: int) -> np.ndarray:
    """Total degree of each coefficient, in the order the fit and render use."""
    return np.array(
        [i + j for i in range(order + 1) for j in range(order + 1 - i)], dtype=int
    )


def monomial_index(order: int, i: int, j: int) -> int:
    """Position of (x**i)*(y**j) in that same ordering."""
    index = 0
    for a in range(order + 1):
        for b in range(order + 1 - a):
            if (a, b) == (i, j):
                return index
            index += 1
    raise ValueError(f"(x**{i})*(y**{j}) is not part of an order-{order} basis")


def demodulation_ramp(
    hologram: torch.Tensor, order: int, dc_exclusion_px: int, flipped: bool
) -> np.ndarray:
    """Coefficients of the phase ramp that integer-bin demodulation leaves behind.

    ``reconstruct_off_axis`` centres the sideband with ``torch.roll``, which
    moves the spectrum by whole bins only. The fringe frequency is not an
    integer number of bins, so the demodulated field keeps a residual carrier

        delta f  =  f_true  -  (integer bins actually rolled) / N

    and that residual is a linear phase ramp of 2 pi . delta f . N radians
    across the field -- up to pi for a half-bin miss, on any grid size.

    ``f_true`` comes from ``estimate_carrier``, which reads the fringe frequency
    off the hologram's own spectrum and refines it to a small fraction of a bin.
    NOTHING HERE TOUCHES THE REFERENCE PHASE, which is the whole point: this
    ramp is reproducible at inference, so a surface built from it is deployable.

    Returns a full coefficient vector that is zero except for the constant and
    the two first-degree terms, expressed in the same normalised x, y in
    [-1, 1] basis as the fitted surfaces.
    """
    _, _, height, width = hologram.shape

    # The same argmax the reconstruction used, so the bins compared are the
    # bins that were actually rolled.
    spectrum = torch.fft.fftshift(torch.fft.fft2(hologram.float()), dim=(-2, -1))
    centre_y, centre_x = height // 2, width // 2
    grid_y = torch.arange(height).view(-1, 1)
    grid_x = torch.arange(width).view(1, -1)
    from_dc = torch.hypot((grid_y - centre_y).float(), (grid_x - centre_x).float())
    magnitude = spectrum[0, 0].abs().clone()
    magnitude[from_dc < dc_exclusion_px] = 0.0
    flat = int(torch.argmax(magnitude))
    bin_y = flat // width - centre_y
    bin_x = flat % width - centre_x

    carrier_y, carrier_x = estimate_carrier(hologram, dc_exclusion_px)
    fine_y = float(carrier_y[0])
    fine_x = float(carrier_x[0])
    # estimate_carrier always returns the upper half-plane peak while the
    # reconstruction's argmax takes either one. When they disagree the two are
    # conjugates, so negating the fine estimate puts both on the same sideband.
    if (bin_y < 0) or (bin_y == 0 and bin_x < 0):
        fine_y, fine_x = -fine_y, -fine_x

    delta_y = fine_y - bin_y / height
    delta_x = fine_x - bin_x / width

    # 2 pi (delta_y . y_px + delta_x . x_px) on the normalised grid.
    #
    # np.linspace(-1, 1, height) maps row k to v = -1 + 2k/(height - 1), so
    # y_px = (height - 1)(v + 1)/2 -- NOT height (v + 1)/2, which is what this
    # used and what the old comment asserted. The error is one part in 900 here,
    # immaterial to the conclusion this feeds (the carrier explains only 10.9%
    # of the fitted tilt), but the grid convention has to match
    # holoqpi/physics/surface.py:polynomial_basis and
    # holoqpi/data/io.py:render_aberration exactly, because those two are what
    # the coefficients are later evaluated on.
    coefficient_v = math.pi * delta_y * (height - 1)
    coefficient_u = math.pi * delta_x * (width - 1)
    sign = -1.0 if flipped else 1.0

    vector = np.zeros(int(monomial_degrees(order).size), dtype=float)
    vector[monomial_index(order, 0, 0)] = sign * (coefficient_v + coefficient_u)
    vector[monomial_index(order, 0, 1)] = sign * coefficient_v
    vector[monomial_index(order, 1, 0)] = sign * coefficient_u
    return vector


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
    parser.add_argument("--global-from", default=None,
                        choices=["train", "val", "test", "all"],
                        help="split whose valid fields form the global surface; "
                             "default: optics.aberration.global_from_split")
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
    ramps: dict[str, np.ndarray] = {}
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
        # Hologram-only prediction of the piston and tilt this reconstruction
        # introduced, for the hybrid surface and for the test of it below.
        ramps[stem] = demodulation_ramp(
            hologram, order, cfg.model.frontend.dc_exclusion_px, bool(flipped[0])
        )
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
    if quality.size == 0:
        # np.median / np.percentile / .min() on an empty array all raise, and the
        # raise happened AFTER the JSON had been written -- so the file was on
        # disk and the run looked half-successful. Every R-squared is NaN when
        # every fit had zero variance, which means the difference field was
        # constant and no surface was recovered at all.
        raise RuntimeError(
            f"no fit produced a finite R-squared over {len(rows)} fields. The "
            "difference between the classical reconstruction and the delivered "
            "phase is constant, so no aberration surface exists to fit. Check the "
            "sideband radius, the conjugate resolution and the phase files before "
            "anything downstream uses this."
        )

    # ------------------------------------------------------------------
    # The global surface: the per-coefficient median over the valid fields of
    # one split. Restricting it to a split -- train by default -- keeps val and
    # test out of the calibration, so the surface applied at evaluation time was
    # not informed by the fields it is evaluated on.
    # ------------------------------------------------------------------
    global_split = args.global_from or cfg.optics.aberration.global_from_split
    splits_path = root / cfg.paths.splits_file
    if global_split == "all" or not splits_path.is_file():
        calibration_stems = [row["stem"] for row in rows if row["valid"]]
        global_source = "all valid fields"
    else:
        from holoqpi.data.splits import load_splits

        allowed = set(load_splits(splits_path).get(global_split, []))
        calibration_stems = [
            row["stem"] for row in rows if row["valid"] and row["stem"] in allowed
        ]
        global_source = f"{global_split} split"
        if not calibration_stems:
            LOGGER.warning(
                "no valid fields in the %s split; falling back to all valid fields for "
                "the global surface", global_split,
            )
            calibration_stems = [row["stem"] for row in rows if row["valid"]]
            global_source = "all valid fields (fallback)"

    if calibration_stems:
        stack = np.array([coefficients[stem] for stem in calibration_stems], dtype=float)
        global_vector = np.median(stack, axis=0)
        # Spread across the calibration fields. A genuinely fixed optical
        # aberration should give a tight distribution; a wide one means the
        # surface is absorbing something field-specific and the global
        # approximation is poor.
        global_spread = np.median(np.abs(stack - global_vector), axis=0)
        coefficients["__global__"] = [float(v) for v in global_vector]

        # The hybrid surface, one per field: the global curvature with its
        # piston and tilt replaced by this field's own, taken from the carrier.
        # The global surface's own degree <= 1 terms are dropped rather than
        # added to, because the ramp prediction is absolute and not a departure.
        degrees_all = monomial_degrees(order)[: global_vector.size]
        curvature_only = global_vector.copy()
        curvature_only[degrees_all <= 1] = 0.0
        for stem in ramps:
            coefficients[f"hybrid::{stem}"] = [
                float(v) for v in curvature_only + ramps[stem][: global_vector.size]
            ]
    else:
        global_vector = None
        global_spread = None

    payload = {
        "order": order,
        "grid": [cfg.data.phase_size, cfg.data.phase_size],
        "global_split": global_split,
        "global_source": global_source,
        "global_fields": len(calibration_stems),
        "global_coefficient_mad": (
            [float(v) for v in global_spread] if global_spread is not None else None
        ),
        "note": (
            "Coefficients of a polynomial in x, y normalised to [-1, 1] over the "
            "full phase grid, ordered as (x**i)*(y**j) for i in 0..order, "
            "j in 0..order-i. Add this surface to a predicted phase before "
            "propagating, to restore what the delivered reference phase had "
            "removed."
        ),
        "coefficients": coefficients,
        "valid": {
            **{row["stem"]: bool(row["valid"]) for row in rows},
            # The global surface is valid whenever it could be computed at all.
            "__global__": bool(calibration_stems),
            # A hybrid surface is valid wherever the global one is, because its
            # per-field part comes from the hologram and cannot fail a gate
            # that tests agreement with the reference phase.
            **{
                f"hybrid::{row['stem']}": bool(calibration_stems) for row in rows
            },
        },
        # Recorded so the reconstruction can be reproduced exactly, and so the
        # fraction of fields whose sideband needed flipping is reportable.
        "conjugated": {row["stem"]: bool(row["conjugated"]) for row in rows},
        "median_span_rad": median_span,
        "span_limit_rad": float(limit),
        "min_agreement": float(min_agreement),
        # THE RECONSTRUCTION SETTINGS THIS SURFACE WAS FITTED UNDER.
        #
        # The surface is added back to the predicted phase before propagation by
        # every forward-model residual in the study, so a surface fitted under a
        # different sideband radius or a different pixel pitch is quietly wrong
        # in every one of them. None of these were recorded, so a surface could
        # be loaded under settings it was never fitted for with nothing on disk
        # to say so. The propagation distance is included because it is
        # deliberately 0 here -- the surface therefore absorbs whatever defocus
        # exists at the real z, which is a documented approximation rather than
        # an accident, and it should be visible.
        "fitted_with": {
            "sideband_radius_px": cfg.model.frontend.sideband_radius_px,
            "dc_exclusion_px": cfg.model.frontend.dc_exclusion_px,
            "reconstruction_distance_um": 0.0,
            "wavelength_um": cfg.optics.wavelength_um,
            "pixel_pitch_x_um": cfg.optics.pixel_pitch_x_um,
            "pixel_pitch_y_um": cfg.optics.pixel_pitch_y_um,
            "phase_size": cfg.data.phase_size,
            "align": cfg.data.align,
            "conjugate_detrend_order": cfg.optics.conjugate.detrend_order,
            "conjugate_min_skewness": cfg.optics.conjugate.min_skewness,
        },
    }
    destination = root / cfg.paths.aberration_file
    destination.write_text(json.dumps(payload))
    # Keyed by order, so re-running at a different order does not overwrite the
    # diagnostics of the one that is actually in use.
    write_csv(rows, Path(cfg.paths.output_root) / f"aberration_fit_order{order}.csv")

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
    if global_vector is not None:
        # How well one surface stands in for all of them. Peak-to-valley of the
        # residual after the global surface is removed from each field's own
        # surface: small means the aberration really is fixed.
        y = np.linspace(-1.0, 1.0, cfg.data.phase_size)[:, None]
        x = np.linspace(-1.0, 1.0, cfg.data.phase_size)[None, :]

        def render(vector):
            surface = np.zeros((cfg.data.phase_size, cfg.data.phase_size))
            index = 0
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    if index >= len(vector):
                        break
                    surface += vector[index] * (x ** i) * (y ** j)
                    index += 1
            return surface

        degrees = monomial_degrees(order)[: len(global_vector)]

        reference_surface = render(global_vector)
        full, curvature, tilt, hybrid_residual = [], [], [], []
        for stem in calibration_stems:
            departure = np.asarray(coefficients[stem], dtype=float) - global_vector
            full.append(float(np.ptp(render(departure))))
            # Same departure with the piston and tilt terms removed. A constant
            # offset is absorbed by the radiometric fit anyway, and a linear
            # ramp is what integer-bin demodulation leaves behind and the
            # forward model's sub-bin carrier already removes -- so if the
            # departure is mostly degree <= 1 it is an artefact of THIS script's
            # classical reconstruction and not field-to-field optical variation.
            # What genuinely has to be static is the curvature.
            high = departure.copy()
            high[degrees <= 1] = 0.0
            curvature.append(float(np.ptp(render(high))))

            # The degree <= 1 part on its own, and what is left of it once the
            # hologram-only ramp prediction is subtracted. The second number is
            # the test: if the ramp really is demodulation remainder, predicting
            # it from the carrier collapses this.
            # DEGREE 1 ONLY on both sides of this comparison. The piston is the
            # unwrap's arbitrary additive constant and is absorbed downstream,
            # so scoring it would be meaningless; the curvature is a different
            # question, already answered two lines up. Mixing either in would
            # make the number look like a verdict on the ramp when it is not.
            low = np.asarray(coefficients[stem], dtype=float).copy()
            low[degrees != 1] = 0.0
            tilt.append(float(np.ptp(render(low))))
            residual = low - ramps[stem][: global_vector.size]
            residual[degrees != 1] = 0.0
            hybrid_residual.append(float(np.ptp(render(residual))))

        magnitude = max(np.ptp(reference_surface), 1e-9)
        print(f"\n=== global surface ({global_source}, {len(calibration_stems)} fields) ===")
        print(f"  peak-to-valley of the global surface        {np.ptp(reference_surface):.2f} rad")
        print(f"  per-field departure, all terms              median {np.median(full):.2f} rad"
              f"   p95 {np.percentile(full, 95):.2f}   max {max(full):.2f}")
        print(f"  per-field tilt, degree == 1 only            median {np.median(tilt):.2f} rad"
              f"   p95 {np.percentile(tilt, 95):.2f}   max {max(tilt):.2f}")
        print(f"  per-field departure, degree >= 2 only       median {np.median(curvature):.2f} rad"
              f"   p95 {np.percentile(curvature, 95):.2f}   max {max(curvature):.2f}")
        print(f"  as a fraction of the surface's magnitude    all {100 * np.median(full) / magnitude:.1f}%"
              f"   curvature-only {100 * np.median(curvature) / magnitude:.1f}%")

        # NOT A DECOMPOSITION, and it used to be labelled as one. `full` and
        # `curvature` are peak-to-valley spans of two rendered surfaces, and a
        # peak-to-valley is not additive across degree groups -- the two extrema
        # need not fall at the same pixel -- so this ratio can exceed 1 or go
        # negative and cannot be read as "the share that is piston+tilt". It is
        # kept because it does order the two magnitudes usefully, under a name
        # that says what it is, and the verdict below is worded to match.
        curvature_share = np.median(curvature) / max(np.median(full), 1e-9)
        recoverable = 1.0 - curvature_share
        print(f"  curvature span as a fraction of the total   "
              f"{100 * curvature_share:.1f}%   (spans, not an additive split)")

        # ---- the test of the hybrid ----------------------------------
        print(f"\n=== hybrid surface: global curvature + carrier-derived ramp ===")
        print("  the ramp is predicted from the hologram's carrier alone, with no")
        print("  reference phase, so this is a genuine out-of-sample prediction of")
        print("  the fitted surface's tilt.")
        print(f"  fitted tilt to be predicted                 median {np.median(tilt):.2f} rad")
        print(f"  left over after the prediction              median {np.median(hybrid_residual):.2f} rad"
              f"   p95 {np.percentile(hybrid_residual, 95):.2f}   max {max(hybrid_residual):.2f}")
        explained = 1.0 - np.median(hybrid_residual) / max(np.median(tilt), 1e-9)
        print(f"  share of the tilt the carrier explains      {100 * explained:.1f}%")
        print(f"  residual as a fraction of the surface       {100 * np.median(hybrid_residual) / magnitude:.1f}%")

        print("\n  -> ", end="")
        if np.median(full) / magnitude < 0.15:
            print("one surface describes the whole instrument well. Use\n"
                  "     optics.aberration.mode: global for anything reported; the cost of\n"
                  "     dropping the target-derived per-field fit is negligible.")
        elif explained > 0.5 and np.median(hybrid_residual) / magnitude < 0.4:
            print("the per-field variation is mostly demodulation remainder, and\n"
                  "     predicting it from the carrier removes most of it. Use\n"
                  "     optics.aberration.mode: hybrid for anything reported: it needs no\n"
                  "     reference phase, so it is deployable, and it reproduces the fitted\n"
                  "     surfaces far better than one global surface does.")
        elif recoverable > 0.6:
            print("most of the variation is degree <= 1, but the carrier does not\n"
                  "     predict it well. Piston is still meaningless and is still absorbed\n"
                  "     by the radiometric fit, so report the GLOBAL surface and state that\n"
                  "     the residual ramp is an unmodelled term of the forward model on\n"
                  "     this instrument. Do not use per_field for a reported result.")
        else:
            print("the aberration is NOT static, and the variation is mostly CURVATURE,\n"
                  "     which cannot be recovered from the hologram without the reference\n"
                  "     phase. So no deployable static calibration reproduces the\n"
                  "     per-field surfaces on this data. Report the global result -- it is\n"
                  "     the only honest one -- and state that the forward-model term is a\n"
                  "     diagnostic rather than a deployable constraint on this instrument.")

    print(f"\n  surfaces -> {destination}")
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
