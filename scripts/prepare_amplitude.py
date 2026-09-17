"""Write a classical-reconstruction amplitude reference for every field.

WHY THIS EXISTS
---------------
The network has a phase head and a segmentation head, both supervised against
delivered targets. The forward-model term also needs the transmitted AMPLITUDE
at the sample plane, because a hologram is formed by a complex field and not by
a phase alone. Until now that amplitude was assumed to be unity everywhere --
the thin-phase-object approximation -- which is defensible for a transparent
cell but is still an assumption the forward model cannot check.

This script produces a reference the amplitude head can be trained against, so
the assumption becomes a measurable rather than a fixture.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
It is the modulus of the classically reconstructed off-axis field: isolate one
first-order sideband, resolve the conjugate ambiguity with the same physical
prior the rest of the pipeline uses, take |U|, and normalise so the clear
background reads 1.

IT IS NOT A MEASURED GROUND TRUTH. It is one algorithm's estimate, and it
carries that algorithm's errors: sideband filtering removes high spatial
frequencies, the twin image is suppressed rather than absent, and any residual
tilt or vignetting in the illumination survives into it. So the amplitude term
is a REGULARISER towards a physically plausible modulus, not a fidelity term
against a truth, and it must be described that way in anything reported.
``loss.weights.amplitude`` is 0.0 by default for exactly that reason: it enters
only in the experiment that is about it.

NORMALISATION
-------------
The absolute scale of |U| is not observable. It is set by the illumination
power and the reference-to-object ratio, and the forward-model term fits both
out as radiometric coefficients. So the reference is divided by a robust
estimate of its own background -- the median over the pixels the reference mask
calls background, falling back to the field median where no mask exists -- and
what remains is transmittance relative to the clear medium, which is the
quantity that means something. Values are clipped into ``[0, clip_max]``
afterwards, because a sideband-filtered reconstruction can ring above 1 at a
cell edge and an amplitude above about 1.5 is ringing rather than transmission.

    python scripts/prepare_amplitude.py --config config/base.yaml
    python scripts/prepare_amplitude.py --config config/base.yaml --limit 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import io as data_io
from holoqpi.physics import reconstruct_off_axis, resolve_conjugate
from holoqpi.utils import (
    assert_provenance,
    get_logger,
    write_csv,
    write_provenance,
)

LOGGER = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="first N fields only")
    parser.add_argument("--clip-max", type=float, default=None,
                        help="upper clip on normalised amplitude; "
                             "default: amplitude_reference.clip_max")
    parser.add_argument("--overwrite", action="store_true",
                        help="rewrite fields whose amplitude file already exists")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = torch.device(args.device)
    optics = cfg.optics
    # From configuration, not from a CLI default: see the amplitude_reference
    # block in config/base.yaml.
    reference_cfg = cfg.amplitude_reference
    clip_max = float(
        args.clip_max if args.clip_max is not None else reference_cfg.clip_max
    )
    min_background_fraction = float(reference_cfg.min_background_fraction)
    conjugate = optics.conjugate
    root = Path(cfg.paths.data_root)

    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found. Run `python main.py prepare` first.")
    stems = [row["stem"] for row in csv.DictReader(open(manifest))]
    if args.limit:
        stems = stems[: args.limit]

    destination = root / cfg.paths.amplitude_dir
    destination.mkdir(parents=True, exist_ok=True)
    suffix = cfg.formats.amplitude.suffix

    # THE PARAMETERS THAT DEFINE THIS RECONSTRUCTION, recorded beside it.
    #
    # Existing files are skipped rather than rebuilt, and the only consumer-side
    # check is the array's shape (holoqpi/data/dataset.py), which none of these
    # settings changes. So altering the sideband radius or the clip and
    # re-running silently reused arrays from the previous configuration, and the
    # amplitude term in arms D0, D1 and D2 trained against them.
    parameters = {
        "sideband_radius_px": cfg.model.frontend.sideband_radius_px,
        "dc_exclusion_px": cfg.model.frontend.dc_exclusion_px,
        "clip_max": clip_max,
        "phase_size": cfg.data.phase_size,
        "align": cfg.data.align,
        "conjugate_enabled": bool(conjugate.enabled),
        "conjugate_detrend_order": conjugate.detrend_order,
        "conjugate_min_skewness": conjugate.min_skewness,
        "wavelength_um": optics.wavelength_um,
        "pixel_pitch_x_um": optics.pixel_pitch_x_um,
        "pixel_pitch_y_um": optics.pixel_pitch_y_um,
    }
    if not args.overwrite:
        assert_provenance(
            destination, parameters,
            "Regenerate with:  python scripts/prepare_amplitude.py "
            "--config <cfg> --overwrite",
        )

    LOGGER.info("writing amplitude references for %d fields to %s", len(stems), destination)

    rows: list[dict] = []
    written = 0
    for position, stem in enumerate(stems, start=1):
        target = destination / f"{stem}{suffix}"
        if target.is_file() and not args.overwrite:
            rows.append({"stem": stem, "skipped": True})
            continue

        hologram = data_io.read_hologram(
            data_io.hologram_path(root, cfg, stem, "off_axis")
        )
        hologram = data_io.align_to_phase_grid(
            hologram, cfg.data.phase_size, cfg.data.align
        )
        height, width = hologram.shape
        tensor = torch.from_numpy(hologram.astype(np.float32)).to(device).view(
            1, 1, height, width
        )

        field = reconstruct_off_axis(
            tensor, optics.wavelength_um,
            optics.pixel_pitch_x_um, optics.pixel_pitch_y_um,
            0.0, cfg.model.frontend.sideband_radius_px, cfg.model.frontend.dc_exclusion_px,
        )
        # The conjugate choice does not change |U| at all -- conjugation
        # preserves the modulus -- but it is applied anyway so that this script
        # and the aberration estimate reconstruct the SAME field, and any later
        # comparison between them is of one reconstruction rather than two.
        if conjugate.enabled:
            field, _, _ = resolve_conjugate(
                field, conjugate.detrend_order, conjugate.min_skewness
            )
        amplitude = field.abs()[0, 0].cpu().numpy().astype(np.float32)

        # Background normalisation. The reference mask is the best available
        # statement of which pixels are clear medium; without it the field
        # median stands in, which is sound here because cells occupy a small
        # minority of every field.
        mask_path = data_io.mask_path(root, cfg, stem)
        background = None
        if mask_path.is_file():
            mask = data_io.read_mask(mask_path)
            mask = data_io.align_to_phase_grid(mask, cfg.data.phase_size, cfg.data.align)
            clear = mask == 0
            if clear.sum() > min_background_fraction * clear.size:
                background = float(np.median(amplitude[clear]))
        source = "mask background" if background is not None else "field median"
        if background is None or not np.isfinite(background) or background <= 0:
            background = float(np.median(amplitude))
            source = "field median"
        if not np.isfinite(background) or background <= 0:
            raise ValueError(f"{stem}: reconstructed amplitude has no usable background")

        normalised = np.clip(amplitude / background, 0.0, clip_max)
        np.save(target, normalised.astype(np.float32))
        written += 1

        rows.append({
            "stem": stem,
            "skipped": False,
            "background_source": source,
            "background_raw": background,
            "median": float(np.median(normalised)),
            "p01": float(np.percentile(normalised, 1)),
            "p99": float(np.percentile(normalised, 99)),
            "clipped_fraction": float(np.mean(amplitude / background > clip_max)),
        })

        if position % 50 == 0 or position == len(stems):
            LOGGER.info("  %d/%d", position, len(stems))

    # Only when this run actually wrote something, and never from a --limit
    # run: stamping the current settings onto files produced by an unknown
    # earlier configuration would bless exactly the staleness the guard exists
    # to catch, and a 4-field diagnostic must not certify 800 files.
    if written and args.limit is None:
        write_provenance(destination, parameters)
    elif written:
        LOGGER.info("--limit run: no provenance written for %s", destination)
    else:
        LOGGER.warning(
            "every amplitude reference already existed, so none was written and "
            "no provenance was recorded. %s cannot be attributed to a "
            "configuration; re-run with --overwrite if there is any doubt.",
            destination,
        )

    # A --limit run must not overwrite the full report: with a fixed name a
    # four-field diagnostic replaced the 800-field table, and the CSV carried no
    # column saying which it was.
    report = Path(cfg.paths.output_root) / (
        "amplitude_reference.csv" if args.limit is None
        else f"amplitude_reference_first{args.limit}.csv"
    )
    for row in rows:
        row["clip_max"] = clip_max
        row["sideband_radius_px"] = cfg.model.frontend.sideband_radius_px
    write_csv(rows, report)

    fresh = [r for r in rows if not r["skipped"]]
    print(f"\n=== amplitude reference ===")
    print(f"  fields written        {written} / {len(stems)}"
          f"   ({len(stems) - written} already present)")
    if fresh:
        medians = np.array([r["median"] for r in fresh])
        low = np.array([r["p01"] for r in fresh])
        clipped = np.array([r["clipped_fraction"] for r in fresh])
        from_mask = sum(r["background_source"] == "mask background" for r in fresh)
        print(f"  background from mask  {from_mask} / {len(fresh)} fields")
        print(f"  background median     median {np.median(medians):.4f}   "
              f"min {medians.min():.4f}   max {medians.max():.4f}")
        print(f"  1st percentile        median {np.median(low):.4f}   min {low.min():.4f}")
        print(f"  clipped above {clip_max:.2f}    median "
              f"{100 * np.median(clipped):.3f}%   max {100 * clipped.max():.3f}%")
        print("\n  -> ", end="")
        if abs(np.median(medians) - 1.0) < 0.05 and np.median(clipped) < 0.02:
            print("the reference behaves like a transmittance: background at 1 and\n"
                  "     little ringing. Set data.provide_amplitude: true and\n"
                  "     loss.weights.amplitude > 0 to use it -- as a regulariser towards a\n"
                  "     plausible modulus, NOT as a fidelity term against a truth.")
        else:
            print("the normalisation is not behaving. A median far from 1 means the\n"
                  "     background estimate is picking up cells or vignetting; heavy\n"
                  "     clipping means the sideband filter is ringing. Check\n"
                  "     scripts/conventional_baseline.py on one field before using this.")
    print(f"\n  arrays -> {destination}/<stem>{suffix}")
    print(f"  per-field statistics -> {report}")
    print("\n  REMINDER: this is a classical reconstruction, not a measurement. "
          "Report it as a reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
