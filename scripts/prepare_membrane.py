"""Register the membrane fluorescence channel onto the phase grid.

WHY THIS EXISTS
---------------
Every segmentation label in this project is `Otsu(gaussian * phase_reference)`,
so segmentation is scored against a threshold of its own input. That
circularity is the single largest weakness in the study, and no amount of
training fixes it: a better Dice against a phase threshold is a better
agreement with a phase threshold.

The membrane channel is the way out, because it is an INDEPENDENT measurement
of where the cell is. This script does the part that has to be right before any
of that is usable: putting the two channels on the same pixel grid.

THE GEOMETRY, MEASURED
----------------------
The membrane TIFFs are 2048 x 2048 uint16. The phase is 900 x 900, itself a
centre crop of a 1024 x 1024 hologram. They are NOT the same field of view and
not the same sampling, so no crop or resize alone aligns them.

Registration by scale-and-translation search over the 13 locally available
fields, with both channels detrended first (the phase carries tens of radians
of smooth aberration and the membrane a bright illumination envelope; leaving
either in lets the search lock onto the envelopes instead of the cells):

    scale   1.32 - 1.34 membrane px per phase px, winning on 13/13 fields
            (mean cross-correlation 0.277, against 0.11 for every "same field
            of view" hypothesis, so the alternatives are excluded, not merely
            less good)
    offset  dy = 314 +/- 3.0 px,  dx = 280 +/- 5.7 px   -- CONSTANT across fields

The scale that the data prefers is indistinguishable from

    0.284871 / 0.211994  =  1.343769

and 0.211994 um is exactly the second pitch value in the phase .bin header --
the one the acquiring group said to disregard. It was never a wrong y-pitch for
the phase: it is the FLUORESCENCE camera's pixel pitch, and it was sitting in
the header all along. So the transform uses that ratio rather than a fitted
number, because a constant derived from two stated pitches is reproducible on
the full 800 fields and a per-field fit is not.

At that scale the membrane's 2048 px span 434 um while the phase spans 256 um,
which is why the illuminated block sits inside a larger dark sensor area.

WHAT THIS SCRIPT DOES AND DOES NOT CLAIM
----------------------------------------
IT DOES establish the transform, warp every membrane field onto the phase grid,
and write a validity window marking where the fluorescence illumination
actually reaches. That part is verified and reusable.

IT DOES NOT hand you trustworthy labels, and the reason is biological.
Membrane stain marks the PERIMETER, not the interior: thresholding it returns
rings, and filling those rings only works where they close. Spread cells with
lamellipodia leave them open, so a threshold-plus-fill mask leaks into
background. Measured here by sweeping the threshold and asking what the newly
included pixels are worth optically:

    threshold pct   foreground   Dice vs Otsu   mean phase of the added pixels
         60            41.3%         0.521            -0.187   <- background
         70            30.9%         0.519            -0.039   <- background
         75            25.9%         0.542            +0.083   <- signal
         80            20.8%         0.545            +0.246

The marginal phase crosses zero between the 70th and 75th percentile: looser
than that and the pixels being added carry NEGATIVE optical path, which a real
cell cannot. That is a falsification test, and a threshold mask fails it before
it reaches the footprint the images plainly show.

So ``--write-masks`` exists, it reports that diagnostic per field, and its
output is labelled provisional. The gold standard remains manual annotation on
25-40 fields, for EVALUATION only -- which is now far cheaper to produce,
because the annotator can trace on a membrane image already aligned to the
phase grid.

INDEPENDENCE IS THE WHOLE POINT, SO GUARD IT
--------------------------------------------
A membrane mask that uses the phase to decide its shape is not independent and
reintroduces exactly the circularity it was meant to break. The mask here is
derived from the membrane ALONE. The phase enters only through the
marginal-phase diagnostic, which calibrates ONE SCALAR and is reported rather
than applied silently. That is a weak coupling, not circularity, and it has to
be stated in any writeup that uses these labels.

    python scripts/prepare_membrane.py --config config/base.yaml --estimate
    python scripts/prepare_membrane.py --config config/base.yaml
    python scripts/prepare_membrane.py --config config/base.yaml --write-masks
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import io as data_io
from holoqpi.utils import get_logger, write_csv

LOGGER = get_logger(__name__)


def read_membrane(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit(
            "tifffile is required to read the membrane TIFFs: pip install tifffile"
        ) from exc
    return tifffile.imread(str(path)).astype(np.float32)


def detrended(image: np.ndarray, sigma: float, envelope: float) -> np.ndarray:
    """Log-compress, smooth, and remove the slow illumination envelope.

    The envelope has to go before any correlation: it is brighter than the
    cells and shaped like the phase map's own aberration bowl, so a scale search
    on the raw images can match the two envelopes and report a confident
    alignment that has nothing to do with the specimen.
    """
    from scipy.ndimage import gaussian_filter

    compressed = gaussian_filter(np.log1p(np.clip(image, 0.0, None)), sigma)
    return compressed - gaussian_filter(compressed, envelope)


def warp_to_phase(
    membrane: np.ndarray, scale: float, offset_y: int, offset_x: int, size: int
) -> tuple[np.ndarray, bool]:
    """Resample by 1/scale, then take the (offset_y, offset_x) window of `size`.

    Returns the window and whether it fell entirely inside the resampled image;
    a window that runs off the edge is padded with the image's own background
    level and flagged, rather than being silently clipped to a smaller array.
    """
    from scipy.ndimage import zoom

    resampled = zoom(membrane, 1.0 / scale, order=1)
    height, width = resampled.shape
    window = np.full((size, size), float(np.median(resampled)), dtype=np.float32)

    y0, x0 = int(offset_y), int(offset_x)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(height, y0 + size), min(width, x0 + size)
    if sy1 <= sy0 or sx1 <= sx0:
        return window, False
    window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = resampled[sy0:sy1, sx0:sx1]
    complete = (y0 >= 0 and x0 >= 0 and y0 + size <= height and x0 + size <= width)
    return window, complete


def illumination_window(window: np.ndarray, sigma: float, factor: float) -> np.ndarray:
    """Where the fluorescence excitation actually reaches.

    Outside it the membrane channel carries no information, so a label derived
    there would be an absence of signal reported as an absence of cells. Every
    metric computed against these labels must be restricted to this window, and
    the fraction it covers is reported so that restriction is visible.
    """
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(window, sigma)
    floor = float(np.percentile(smooth, 2))
    return smooth > floor * factor


def estimate_registration(cfg, stems: list[str], root: Path, membrane_dir: Path,
                          scale: float, search: int) -> dict:
    """Cross-correlate each field to find the translation at a fixed scale.

    The scale is NOT refitted here. It comes from the two stated pixel pitches,
    and a fit over thirteen fields could not distinguish 1.32 from 1.3438
    anyway (mean correlation 0.277 against 0.266). Fitting it per dataset would
    trade a reproducible constant for a number that moves with whichever fields
    happen to be available.
    """
    import torch
    from scipy.ndimage import gaussian_filter, zoom

    from holoqpi.physics import detrend_polynomial

    size = cfg.data.phase_size
    coarse = 4

    def peak(a, b, step):
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        n = max(a.shape[0] + b.shape[0], a.shape[1] + b.shape[1])
        n = 1 << int(np.ceil(np.log2(n)))
        spectrum = np.fft.rfft2(a, (n, n)) * np.conj(np.fft.rfft2(b, (n, n)))
        correlation = np.fft.irfft2(spectrum, (n, n))
        index = int(np.argmax(correlation))
        dy, dx = divmod(index, n)
        if dy > n // 2:
            dy -= n
        if dx > n // 2:
            dx -= n
        return float(correlation.max() / a.size), dy * step, dx * step

    results: dict[str, dict] = {}
    for position, stem in enumerate(stems, start=1):
        record = data_io.read_phase_bin(
            data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
        )
        phase = detrend_polynomial(
            torch.from_numpy(record.phase.astype(np.float32)), 3
        ).numpy().astype(np.float32)
        membrane = detrended(read_membrane(membrane_dir / f"{stem}_membrane.tif"), 5.0, 60.0)

        _, dy, dx = peak(
            zoom(membrane, 1.0 / (scale * coarse), order=1),
            zoom(gaussian_filter(phase, 5.0), 1.0 / coarse, order=1),
            coarse,
        )
        # Refine on the full grid: the coarse pass is only accurate to a few
        # pixels, and a few pixels of the boundary is several per cent of a
        # cell's area (see scripts/error_propagation.py).
        resampled = zoom(membrane, 1.0 / scale, order=1)
        target = gaussian_filter(phase, 3.0)
        target = (target - target.mean()) / (target.std() + 1e-9)
        best = None
        for step_y in range(-search, search + 1, 2):
            for step_x in range(-search, search + 1, 2):
                y0, x0 = dy + step_y, dx + step_x
                if y0 < 0 or x0 < 0:
                    continue
                if y0 + size > resampled.shape[0] or x0 + size > resampled.shape[1]:
                    continue
                patch = resampled[y0:y0 + size, x0:x0 + size]
                patch = (patch - patch.mean()) / (patch.std() + 1e-9)
                score = float((patch * target).mean())
                if best is None or score > best[0]:
                    best = (score, y0, x0)
        if best is None:
            LOGGER.warning("%s: no valid window during refinement; keeping coarse", stem)
            best = (float("nan"), dy, dx)
        results[stem] = {"correlation": best[0], "offset_y": int(best[1]),
                         "offset_x": int(best[2])}
        if position % 25 == 0 or position == len(stems):
            LOGGER.info("  %d/%d", position, len(stems))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--estimate", action="store_true",
                        help="re-measure the offsets by cross-correlation and stop")
    parser.add_argument("--search", type=int, default=8,
                        help="refinement half-window in phase pixels (--estimate)")
    parser.add_argument("--write-masks", action="store_true",
                        help="also write PROVISIONAL threshold masks; read the docstring")
    parser.add_argument("--threshold-pct", type=float, default=None,
                        help="default: membrane.threshold_percentile")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    membrane_cfg = cfg.membrane
    root = Path(cfg.paths.data_root)
    membrane_dir = root / membrane_cfg.source_dir
    if not membrane_dir.is_dir():
        raise SystemExit(
            f"{membrane_dir} not found. Point membrane.source_dir at the folder of "
            f"<stem>_membrane.tif files."
        )

    manifest = root / cfg.paths.manifest_file
    if not manifest.is_file():
        raise SystemExit(f"{manifest} not found. Run `python main.py prepare` first.")
    stems = [row["stem"] for row in csv.DictReader(open(manifest))]
    if args.limit:
        stems = stems[: args.limit]
    available = [s for s in stems if (membrane_dir / f"{s}_membrane.tif").is_file()]
    if len(available) < len(stems):
        LOGGER.warning(
            "%d of %d fields have no membrane file and are skipped",
            len(stems) - len(available), len(stems),
        )
    stems = available
    if not stems:
        raise SystemExit(f"no <stem>_membrane.tif files found in {membrane_dir}")

    scale = float(membrane_cfg.scale)
    size = cfg.data.phase_size

    # ---- --estimate: measure the offsets and report their spread ------
    if args.estimate:
        LOGGER.info("estimating registration over %d fields at scale %.6f",
                    len(stems), scale)
        found = estimate_registration(cfg, stems, root, membrane_dir, scale, args.search)
        rows = [{"stem": s, **v} for s, v in found.items()]
        write_csv(rows, Path(cfg.paths.output_root) / "membrane_registration.csv")

        ys = np.array([v["offset_y"] for v in found.values()], float)
        xs = np.array([v["offset_x"] for v in found.values()], float)
        cs = np.array([v["correlation"] for v in found.values()], float)
        cs = cs[np.isfinite(cs)]
        print(f"\n=== membrane registration, {len(found)} fields, scale {scale:.6f} ===")
        print(f"  offset_y  median {np.median(ys):.0f}  sd {ys.std():.1f}  "
              f"range {ys.min():.0f}..{ys.max():.0f}")
        print(f"  offset_x  median {np.median(xs):.0f}  sd {xs.std():.1f}  "
              f"range {xs.min():.0f}..{xs.max():.0f}")
        if cs.size:
            print(f"  correlation  median {np.median(cs):.3f}  min {cs.min():.3f}")
        print(f"  config currently says offset_y={membrane_cfg.offset_y}, "
              f"offset_x={membrane_cfg.offset_x}")
        print("\n  -> ", end="")
        if ys.std() < 6 and xs.std() < 6:
            print(f"the offset is CONSTANT across fields (sd < 6 px), so one global\n"
                  f"     transform serves the whole dataset. Set\n"
                  f"     membrane.offset_y: {int(np.median(ys))}\n"
                  f"     membrane.offset_x: {int(np.median(xs))}\n"
                  f"     and do not store per-field offsets: a per-field fit needs the\n"
                  f"     phase, which is what these labels exist to be independent of.")
        else:
            print("the offset VARIES between fields, so the two cameras were not in a\n"
                  "     fixed relationship and a single transform will not do. Per-field\n"
                  "     offsets would need the phase to fit them, which destroys the\n"
                  "     independence that makes these labels worth having. Stop and\n"
                  "     resolve the acquisition geometry before using them.")
        print(f"\n  per-field detail -> "
              f"{Path(cfg.paths.output_root) / 'membrane_registration.csv'}")
        return 0

    # ---- warp every field onto the phase grid -------------------------
    destination = root / membrane_cfg.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    mask_destination = root / membrane_cfg.mask_dir
    if args.write_masks:
        mask_destination.mkdir(parents=True, exist_ok=True)

    percentile = (
        args.threshold_pct if args.threshold_pct is not None
        else float(membrane_cfg.threshold_percentile)
    )
    LOGGER.info(
        "warping %d membrane fields onto the %d px phase grid "
        "(scale %.6f, offset %d,%d)",
        len(stems), size, scale, membrane_cfg.offset_y, membrane_cfg.offset_x,
    )

    from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter

    rows: list[dict] = []
    for position, stem in enumerate(stems, start=1):
        membrane = read_membrane(membrane_dir / f"{stem}_membrane.tif")
        window, complete = warp_to_phase(
            membrane, scale, membrane_cfg.offset_y, membrane_cfg.offset_x, size
        )
        valid = illumination_window(
            window, membrane_cfg.illumination_sigma, membrane_cfg.illumination_factor
        )
        np.save(destination / f"{stem}{membrane_cfg.suffix}", window.astype(np.float32))
        np.save(destination / f"{stem}{membrane_cfg.valid_suffix}", valid)

        row = {
            "stem": stem,
            "window_complete": complete,
            "valid_fraction": float(valid.mean()),
            "median_intensity": float(np.median(window)),
        }

        if args.write_masks:
            smooth = gaussian_filter(window, 2.0)
            threshold = float(np.percentile(smooth[valid], percentile))
            mask = binary_fill_holes(
                binary_opening((smooth > threshold) & valid, np.ones((3, 3)))
            )
            data_io.write_mask(
                mask.astype(np.uint8),
                mask_destination / f"{stem}{cfg.formats.mask.suffix}",
            )
            row["threshold_percentile"] = percentile
            row["foreground_fraction"] = float(mask[valid].mean())

            # THE FALSIFICATION TEST. A cell cannot carry negative optical path,
            # so if the pixels this threshold includes beyond a tighter one have
            # negative mean phase, the mask has leaked into background. Reported
            # per field rather than enforced, because enforcing it would let the
            # phase decide the mask's shape.
            record = data_io.read_phase_bin(
                data_io.phase_path(root, cfg, stem), cfg.formats.phase_binary
            )
            phase = record.phase.astype(np.float32)
            tighter = binary_fill_holes(
                binary_opening(
                    (smooth > float(np.percentile(smooth[valid], min(99.0, percentile + 5))))
                    & valid,
                    np.ones((3, 3)),
                )
            )
            added = mask & ~tighter
            row["marginal_phase_rad"] = (
                float(phase[added].mean()) if added.sum() > 50 else float("nan")
            )
            row["mask_mean_phase_rad"] = (
                float(phase[mask].mean()) if mask.any() else float("nan")
            )

        rows.append(row)
        if position % 50 == 0 or position == len(stems):
            LOGGER.info("  %d/%d", position, len(stems))

    report = Path(cfg.paths.output_root) / "membrane_prepare.csv"
    write_csv(rows, report)

    incomplete = sum(1 for r in rows if not r["window_complete"])
    valid_fraction = np.array([r["valid_fraction"] for r in rows])
    print(f"\n=== membrane warped onto the phase grid ===")
    print(f"  fields written        {len(rows)}")
    print(f"  scale                 {scale:.6f} membrane px per phase px")
    print(f"  offset                y {membrane_cfg.offset_y}, x {membrane_cfg.offset_x}")
    print(f"  windows off the edge  {incomplete}")
    print(f"  illuminated fraction  median {np.median(valid_fraction):.3f}   "
          f"min {valid_fraction.min():.3f}")
    print(f"  arrays -> {destination}/<stem>{membrane_cfg.suffix}  (+ valid mask)")

    if args.write_masks:
        foreground = np.array([r["foreground_fraction"] for r in rows])
        marginal = np.array([r.get("marginal_phase_rad", np.nan) for r in rows])
        marginal = marginal[np.isfinite(marginal)]
        print(f"\n  PROVISIONAL masks -> {mask_destination}")
        print(f"  threshold             {percentile:.0f}th percentile inside the window")
        print(f"  foreground            median {np.median(foreground):.3f}   "
              f"min {foreground.min():.3f}   max {foreground.max():.3f}")
        if marginal.size:
            print(f"  marginal phase        median {np.median(marginal):+.3f} rad   "
                  f"(fraction negative {float((marginal < 0).mean()):.2f})")
            print("\n  -> ", end="")
            if np.median(marginal) < 0:
                print(f"THE MASK HAS LEAKED INTO BACKGROUND. The pixels this threshold\n"
                      f"     adds carry a mean phase of {np.median(marginal):+.3f} rad, and a cell\n"
                      f"     cannot have negative optical path. Raise\n"
                      f"     membrane.threshold_percentile until this turns positive, and do\n"
                      f"     not use these masks for a reported number until it does.")
            else:
                print(f"the added pixels carry positive optical path "
                      f"({np.median(marginal):+.3f} rad), so the\n"
                      f"     threshold is inside real signal. These masks are still\n"
                      f"     PROVISIONAL: membrane stain marks the perimeter, so a threshold\n"
                      f"     returns rings and filling them fails wherever a lamellipodium\n"
                      f"     leaves one open. Use them as a cross-check, and get manual\n"
                      f"     labels on 25-40 fields for anything reported.")
    else:
        print("\n  -> the channels are now on one grid. Next, either")
        print("     --write-masks for a provisional threshold mask (read the caveat), or")
        print("     annotate 25-40 fields by hand for evaluation, which is the gold")
        print("     standard and is now much cheaper: the annotator traces on a membrane")
        print("     image already aligned to the phase.")
    print(f"\n  per-field detail -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
