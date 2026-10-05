"""Check how the off-axis and Gabor holograms of a field relate to each other.

Three questions, answered from the files alone:

1. INVENTORY. Where the two hologram sets and the phase files are, whether the
   two hologram types have the same size and dtype, and whether any TIFF carries
   acquisition metadata (date/time, software, description, ImageJ/OME blocks).
   File modification times are listed too, but they record when the files were
   copied, not when they were acquired.

2. REGISTRATION. For every field, both holograms are low-pass filtered
   (Gaussian, lowpass_sigma_px) to suppress the off-axis carrier fringes (the
   "bandpass" variant also removes the slow illumination background and applies
   a Hann window), and the translation between them is measured with
   skimage.registration.phase_cross_correlation (upsample_factor). The CONTROL
   pairs each Gabor frame with a DIFFERENT field's off-axis frame (a random
   derangement, control_seed). The method separates matched from unmatched
   pairs only if the matched shifts are tight and the matched correlation is
   clearly above the control's. Besides the registration error, the Pearson r of
   the overlapping region after applying the (integer-rounded) shift is reported.

3. CLASSICAL RECONSTRUCTION vs REFERENCE PHASE. On classical_phase_split, the
   classical off-axis reconstruction (scripts/conventional_baseline.py's own
   predictor, hologram centre-cropped exactly as in evaluation) is registered
   against the supplied reference phase.

    python scripts/register_holograms.py
    python scripts/register_holograms.py --device cuda:0      # step 3 on the GPU
    python scripts/register_holograms.py --skip-classical

Parameters: config/diagnostics.yaml (registration block).
Outputs, under <paths.output_root>/<output_dir>/:
    hologram_inventory.csv, hologram_registration.csv,
    classical_phase_shift.csv, registration_summary.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data.io import hologram_path, phase_path, read_hologram, read_phase_header
from holoqpi.utils import get_logger, resolve_device, write_csv

LOGGER = get_logger(__name__)

# TIFF tags that would carry acquisition information if the writer stored any.
METADATA_TAGS = ("DateTime", "DateTimeOriginal", "Software", "ImageDescription", "Artist",
                 "HostComputer", "DocumentName", "Make", "Model", "PageName")


def _stems(cfg) -> list[str]:
    manifest = Path(cfg.paths.data_root) / cfg.paths.manifest_file
    with manifest.open(newline="") as handle:
        return [row["stem"] for row in csv.DictReader(handle)]


def _inventory(cfg, stems: list[str]) -> tuple[list[dict], dict]:
    import tifffile

    data_root = Path(cfg.paths.data_root)
    rows = []
    for stem in stems:
        for modality in cfg.paths.hologram_dirs:
            path = hologram_path(data_root, cfg, stem, modality)
            row = {"stem": stem, "modality": modality, "path": str(path),
                   "exists": path.is_file()}
            if path.is_file():
                with tifffile.TiffFile(str(path)) as tif:
                    page = tif.pages[0]
                    row.update(
                        pages=len(tif.pages), shape="x".join(map(str, page.shape)),
                        dtype=str(page.dtype), compression=str(page.compression),
                        bytes=path.stat().st_size,
                        mtime_utc=dt.datetime.fromtimestamp(
                            path.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
                        tag_names=";".join(sorted(t.name for t in page.tags.values())),
                        imagej=bool(tif.imagej_metadata), ome=bool(tif.ome_metadata),
                    )
                    for tag in METADATA_TAGS:
                        if tag in page.tags:
                            row[f"tag_{tag}"] = str(page.tags[tag].value)[:200]
            rows.append(row)
        phase = phase_path(data_root, cfg, stem)
        row = {"stem": stem, "modality": "phase", "path": str(phase), "exists": phase.is_file()}
        if phase.is_file():
            header = read_phase_header(phase, cfg.formats.phase_binary)
            row.update(shape=f"{header.height}x{header.width}", dtype="float32",
                       bytes=phase.stat().st_size,
                       header_pitch_um=f"{header.pitch_x_um}/{header.pitch_y_um}")
        rows.append(row)

    summary = {}
    for modality in list(cfg.paths.hologram_dirs) + ["phase"]:
        chosen = [r for r in rows if r["modality"] == modality]
        present = [r for r in chosen if r["exists"]]
        entry = {
            "directory": str(data_root / (cfg.paths.hologram_dirs[modality] if modality != "phase"
                                          else cfg.paths.phase_dir)),
            "files_expected": len(chosen), "files_present": len(present),
            "shapes": sorted({r.get("shape", "?") for r in present}),
            "dtypes": sorted({r.get("dtype", "?") for r in present}),
        }
        if modality != "phase":
            entry["compression"] = sorted({r.get("compression", "?") for r in present})
            entry["tag_sets"] = sorted({r.get("tag_names", "") for r in present})
            entry["files_with_acquisition_tags"] = {
                tag: sum(1 for r in present if f"tag_{tag}" in r) for tag in METADATA_TAGS}
            entry["imagej_or_ome_metadata"] = sum(1 for r in present if r["imagej"] or r["ome"])
            times = sorted(r["mtime_utc"] for r in present)
            entry["mtime_range_utc"] = [times[0], times[-1]] if times else []
        else:
            entry["header_pitches_um"] = sorted({r.get("header_pitch_um", "?") for r in present})
        summary[modality] = entry
    return rows, summary


def _prepare(image: np.ndarray, variant) -> np.ndarray:
    """Low-pass, optionally remove the slow background, standardise, optionally window."""
    from scipy.ndimage import gaussian_filter

    image = image.astype(np.float64)
    smooth = gaussian_filter(image, float(variant.lowpass_sigma_px))
    if variant.background_sigma_px:
        smooth = smooth - gaussian_filter(image, float(variant.background_sigma_px))
    std = smooth.std()
    smooth = (smooth - smooth.mean()) / (std if std > 0 else 1.0)
    if variant.window == "hann":
        smooth = smooth * np.outer(np.hanning(smooth.shape[0]), np.hanning(smooth.shape[1]))
    elif variant.window != "none":
        raise ValueError(f"unknown window {variant.window!r}; use none or hann")
    return smooth.astype(np.float32)


def _overlap_r(a: np.ndarray, b: np.ndarray, shift) -> float:
    """Pearson r of the overlap after moving ``b`` by the integer-rounded shift."""
    dy, dx = (int(round(s)) for s in shift)
    h, w = a.shape
    ya, yb = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xa, xb = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    pa, pb = a[ya, xa].ravel(), b[yb, xb].ravel()
    if pa.size < 2 or pa.std() == 0 or pb.std() == 0:
        return float("nan")
    return float(np.corrcoef(pa, pb)[0, 1])


def _register(a, b, settings) -> dict:
    from skimage.registration import phase_cross_correlation

    shift, error, _ = phase_cross_correlation(
        a, b, upsample_factor=int(settings.upsample_factor),
        normalization=settings.normalization,
    )
    return {"dy": float(shift[0]), "dx": float(shift[1]), "error": float(error),
            "r_after_shift": _overlap_r(a, b, shift)}


def _describe(values) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not values.size:
        return {}
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "mean": float(values.mean()), "sd": float(values.std()),
            "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
            "min": float(values.min()), "max": float(values.max())}


def _auc(positive, negative) -> float:
    """Probability that a matched pair scores higher than an unmatched one."""
    positive = np.asarray([v for v in positive if np.isfinite(v)])
    negative = np.asarray([v for v in negative if np.isfinite(v)])
    if not positive.size or not negative.size:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def _classical_shift(cfg, settings, device) -> list[dict]:
    import torch

    from holoqpi.data import build_dataloaders
    from scripts.conventional_baseline import build_predictor

    split = settings.classical_phase_split
    cfg = cfg.merged({"data": {"modality": "off_axis"}})
    conv = cfg.evaluation.conventional_baseline
    predict, _ = build_predictor(cfg, "off_axis", device,
                                 float(cfg.loss.forward_model.distance_um or 0.0),
                                 conv.gs_iterations, conv.aberration_order)
    loader = build_dataloaders(cfg, splits_to_build=(split,))[split]
    rows = []
    with torch.no_grad():
        for batch in loader:
            phase = predict(batch)["phase"].squeeze(1).float().cpu().numpy()
            reference = batch["phase"].squeeze(1).float().numpy()
            for i in range(phase.shape[0]):
                a = reference[i] - reference[i].mean()
                b = phase[i] - phase[i].mean()
                row = {"stem": batch["stem"][i], "split": split}
                row.update(_register(a, b, settings))
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--diagnostics", default="config/diagnostics.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    diag = load_config(args.diagnostics)
    settings = diag.registration
    destination = Path(cfg.paths.output_root) / diag.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg.paths.data_root)

    stems = _stems(cfg)
    inventory, inventory_summary = _inventory(cfg, stems)
    write_csv(inventory, destination / "hologram_inventory.csv")

    paired = [s for s in stems
              if hologram_path(data_root, cfg, s, "off_axis").is_file()
              and hologram_path(data_root, cfg, s, "gabor").is_file()]
    permutation = _derangement(len(paired), int(settings.control_seed))
    LOGGER.info("registering %d fields, variants %s, upsample %d", len(paired),
                [v.name for v in settings.variants], int(settings.upsample_factor))
    # Kept as uint8 (their native dtype, checked in the inventory) so that 1600
    # frames need about 1.7 GB of memory rather than 6.7 GB as float32.
    raw_off = {s: read_hologram(hologram_path(data_root, cfg, s, "off_axis")).astype(np.uint8)
               for s in paired}
    raw_gabor = {s: read_hologram(hologram_path(data_root, cfg, s, "gabor")).astype(np.uint8)
                 for s in paired}

    rows = []
    for variant in settings.variants:
        off_axis = {s: _prepare(raw_off[s], variant) for s in paired}
        for index, stem in enumerate(paired):
            gabor = _prepare(raw_gabor[stem], variant)
            for pairing, other in (("matched", stem), ("control", paired[permutation[index]])):
                row = {"variant": variant.name, "stem": stem, "pairing": pairing,
                       "off_axis_stem": other}
                row.update(_register(off_axis[other], gabor, settings))
                rows.append(row)
    write_csv(rows, destination / "hologram_registration.csv")

    summary = {"inventory": inventory_summary, "settings": settings.to_dict(),
               "registration": {}}
    for variant in settings.variants:
        block = {}
        mine = [r for r in rows if r["variant"] == variant.name]
        for pairing in ("matched", "control"):
            chosen = [r for r in mine if r["pairing"] == pairing]
            magnitude = [float(np.hypot(r["dy"], r["dx"])) for r in chosen]
            block[pairing] = {
                "dy": _describe([r["dy"] for r in chosen]),
                "dx": _describe([r["dx"] for r in chosen]),
                "shift_magnitude": _describe(magnitude),
                "fraction_within_1px": float(np.mean(np.array(magnitude) <= 1.0)) if magnitude else None,
                "fraction_within_5px": float(np.mean(np.array(magnitude) <= 5.0)) if magnitude else None,
                "error": _describe([r["error"] for r in chosen]),
                "r_after_shift": _describe([r["r_after_shift"] for r in chosen]),
            }
        matched = [r for r in mine if r["pairing"] == "matched"]
        control = [r for r in mine if r["pairing"] == "control"]
        block["separation"] = {
            "auc_r_after_shift": _auc([r["r_after_shift"] for r in matched],
                                      [r["r_after_shift"] for r in control]),
            "auc_minus_error": _auc([-r["error"] for r in matched],
                                    [-r["error"] for r in control]),
        }
        counts = {}
        for r in matched:
            key = (round(r["dy"]), round(r["dx"]))
            counts[key] = counts.get(key, 0) + 1
        block["matched_integer_shift_counts_top10"] = [
            {"dy": k[0], "dx": k[1], "fields": v}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:10]]
        block["matched_largest_shifts"] = [
            {"stem": r["stem"], "dy": round(r["dy"], 2), "dx": round(r["dx"], 2),
             "r": round(r["r_after_shift"], 3)}
            for r in sorted(matched, key=lambda r: -np.hypot(r["dy"], r["dx"]))[:10]]
        summary["registration"][variant.name] = block

    if not args.skip_classical:
        classical = _classical_shift(cfg, settings, resolve_device(args.device))
        write_csv(classical, destination / "classical_phase_shift.csv")
        summary["classical_vs_reference"] = {
            "split": settings.classical_phase_split,
            "dy": _describe([r["dy"] for r in classical]),
            "dx": _describe([r["dx"] for r in classical]),
            "error": _describe([r["error"] for r in classical]),
            "r_after_shift": _describe([r["r_after_shift"] for r in classical]),
        }

    (destination / "registration_summary.json").write_text(json.dumps(summary, indent=2,
                                                                      default=float))
    _print(summary)
    print(f"\nWritten to {destination}/")
    return 0


def _print(summary: dict) -> None:
    print("\n=== 1. Inventory ===")
    for modality, e in summary["inventory"].items():
        print(f"  {modality:<9} {e['files_present']}/{e['files_expected']} files in {e['directory']}")
        print(f"            shapes {e['shapes']}  dtypes {e['dtypes']}")
        if "compression" in e:
            print(f"            compression {e['compression']}")
            print(f"            distinct TIFF tag sets: {len(e['tag_sets'])}")
            for tags in e["tag_sets"]:
                print(f"              {tags}")
            print(f"            files with acquisition tags: {e['files_with_acquisition_tags']}")
            print(f"            files with ImageJ/OME metadata: {e['imagej_or_ome_metadata']}")
            print(f"            file mtimes (copy times, UTC): {e['mtime_range_utc']}")
        else:
            print(f"            header pitches (um): {e['header_pitches_um']}")

    def line(name, d):
        if not d:
            return f"    {name:<16} (none)"
        return (f"    {name:<16} median {d['median']:+8.3f}  IQR [{d['p25']:+.3f}, {d['p75']:+.3f}]"
                f"  5-95% [{d['p05']:+.3f}, {d['p95']:+.3f}]  n={d['n']}")

    for name, reg in summary["registration"].items():
        print(f"\n=== 2. Off-axis vs Gabor registration, variant '{name}' ===")
        for pairing in ("matched", "control"):
            e = reg[pairing]
            print(f"  {pairing}:")
            for key in ("dy", "dx", "shift_magnitude", "error", "r_after_shift"):
                print(line(key, e[key]))
            print(f"    within 1 px: {e['fraction_within_1px']:.3f}   within 5 px: "
                  f"{e['fraction_within_5px']:.3f}")
        print(f"  separation AUC (r after shift) {reg['separation']['auc_r_after_shift']:.3f}   "
              f"(-error) {reg['separation']['auc_minus_error']:.3f}   (0.5 = no separation)")
        print(f"  most common matched integer shifts: {reg['matched_integer_shift_counts_top10']}")
        print(f"  largest matched shifts: {reg['matched_largest_shifts']}")

    if "classical_vs_reference" in summary:
        e = summary["classical_vs_reference"]
        print(f"\n=== 3. Classical off-axis reconstruction vs reference phase ({e['split']}) ===")
        for key in ("dy", "dx", "error", "r_after_shift"):
            print(line(key, e[key]))


if __name__ == "__main__":
    sys.exit(main())
