"""Systematic uncertainty on absolute dry mass from the refraction increment.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
Dry mass is

    m = lambda / (2 pi alpha) * sum(phi) * dx * dy        [pg]

and alpha, the specific refraction increment, is not measured for this specimen.
It is taken from the literature at 0.185 mL/g.

The instinct is to sweep alpha and see how much of the reported dry-mass error it
explains. That instinct is WRONG, and this script exists partly to make the
reason explicit and to keep anyone from spending GPU time on it.

    The SAME alpha multiplies the predicted mass and the reference mass.

Every relative quantity the study reports -- dry_mass_mape, the predicted/
reference mass ratio, the relative bias, the Bland-Altman limits expressed as
fractions -- therefore has alpha cancel out of it exactly. A sweep would return a
flat line. None of the -19.2% mass error can be attributed to alpha, and neither
can the -9.2% phase-side component of it. scripts/selftest.py asserts this
invariance so that a later "improvement" to alpha cannot silently be justified by
a change in MAPE that is arithmetically impossible.

Where alpha DOES matter is the absolute number of picograms. Any statement of the
form "the mean cell dry mass is X pg" inherits alpha's uncertainty in full, as a
systematic scale factor:

    m(alpha) = m(alpha_0) * alpha_0 / alpha

This script propagates the literature range through the measured per-cell masses
and reports the band that belongs beside every absolute mass in the paper.

The range comes from the configuration, not from this file. Sources for the
default:
  Zhao, Brown & Schuck, Biophys. J. 2011 -- protein dn/dc mean 0.190 mL/g
    (SD 0.003), individual proteins spanning 0.173-0.215.
  Chaumet, Bon, Maire, Sentenac & Baffou, Light Sci. Appl. 2024 -- alpha for
    biological media "approximately constant, ranging from 0.18 to 0.21".

    python scripts/mass_uncertainty.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.analysis.cells import calibration_from_config
from holoqpi.config import load_config, parse_overrides
from holoqpi.utils import get_logger, write_json

LOGGER = get_logger(__name__)


def per_cell_masses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and reference dry mass in pg from a per-cell CSV."""
    predicted, reference = [], []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            try:
                predicted.append(float(row["dry_mass_pg_pred"]))
                reference.append(float(row["dry_mass_pg_ref"]))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--experiment", default=None,
                        help="default: experiment_name from the config")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    experiment = args.experiment or cfg.experiment_name
    output_root = Path(cfg.paths.output_root)

    alpha = cfg.optics.refraction_increment_ml_per_g
    low, high = cfg.optics.refraction_increment_range_ml_per_g
    calibration = calibration_from_config(cfg)

    # m scales as 1/alpha, so the smallest alpha gives the largest mass.
    upper_factor = alpha / float(low)
    lower_factor = alpha / float(high)

    print(f"\n=== absolute dry-mass scale, alpha = {alpha:.3f} mL/g ===")
    print(f"  prefactor lambda/(2 pi alpha)   "
          f"{calibration.picogram_per_radian_um2:.6f} pg/(rad*um^2)")
    print(f"  literature range                {low:.3f} - {high:.3f} mL/g")
    print(f"  systematic band on ANY absolute mass   "
          f"{(upper_factor - 1) * 100:+.1f}% / {(lower_factor - 1) * 100:+.1f}%")
    print("\n  Relative metrics (dry_mass_mape, mass ratio, relative bias) are")
    print("  INVARIANT to alpha: the same constant multiplies the predicted and")
    print("  the reference mass and cancels. Do not sweep alpha hoping to move")
    print("  them, and do not attribute any part of the mass error to alpha.")

    payload: dict = {
        "alpha_ml_per_g": float(alpha),
        "alpha_range_ml_per_g": [float(low), float(high)],
        "picogram_per_radian_um2": calibration.picogram_per_radian_um2,
        "systematic_fraction_upper": float(upper_factor - 1.0),
        "systematic_fraction_lower": float(lower_factor - 1.0),
        "modalities": {},
    }

    for modality in args.modality:
        run_dir = output_root / f"{experiment}_{modality}"
        candidates = sorted(run_dir.glob(f"per_cell_{args.split}*.csv"))
        if not candidates:
            LOGGER.warning("no per-cell CSV under %s; skipping %s", run_dir, modality)
            continue
        predicted, reference = per_cell_masses(candidates[0])
        if predicted.size == 0:
            LOGGER.warning("%s holds no usable rows", candidates[0])
            continue

        mean_reference = float(reference.mean())
        mape = float(np.mean(np.abs(predicted - reference) / np.abs(reference)))
        # The same computation with a different alpha, to demonstrate rather than
        # assert the invariance.
        scaled = float(np.mean(
            np.abs(predicted * lower_factor - reference * lower_factor)
            / np.abs(reference * lower_factor)
        ))

        print(f"\n  --- {modality} ({predicted.size} matched cells) ---")
        print(f"    mean reference dry mass   {mean_reference:8.2f} pg"
              f"   [{mean_reference * lower_factor:.2f}, "
              f"{mean_reference * upper_factor:.2f}] pg from alpha alone")
        print(f"    mean predicted dry mass   {float(predicted.mean()):8.2f} pg")
        print(f"    dry-mass MAPE at alpha = {alpha:.3f}    {mape:.6f}")
        print(f"    dry-mass MAPE at alpha = {high:.3f}    {scaled:.6f}"
              f"   <- identical, as it must be")

        payload["modalities"][modality] = {
            "matched_cells": int(predicted.size),
            "mean_reference_pg": mean_reference,
            "mean_predicted_pg": float(predicted.mean()),
            "reference_pg_low": mean_reference * lower_factor,
            "reference_pg_high": mean_reference * upper_factor,
            "dry_mass_mape": mape,
            "dry_mass_mape_alpha_high": scaled,
        }

    destination = Path(args.out) if args.out else output_root / "mass_uncertainty.json"
    write_json(payload, destination)
    print(f"\n  -> {destination}")
    print("\n  For the paper: quote absolute dry mass as  m +/- (systematic, alpha)")
    print("  using the band above, and quote MAPE without an alpha caveat, because")
    print("  it does not have one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
