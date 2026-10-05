"""Measure the offset between the hologram crop and the supplied phase grid.

The classical off-axis reconstruction (scripts/conventional_baseline.py's own
predictor, at the settings in config/base.yaml) is computed from the hologram
exactly as the dataset crops it -- the current data.crop_offset_px included --
and registered against the supplied reference phase with
skimage.registration.phase_cross_correlation. The returned (dy, dx) is the
shift that moves the classical map onto the reference, so a residual of zero
means the crop and the phase grid coincide.

Sideband filtering and angular-spectrum propagation do not translate the image,
so a consistent shift is a property of the crop. Moving the crop origin by the
negative of the rounded median shift aligns the two; the tool reports that value
and, with --write, sets it in the config.

    python scripts/measure_crop_offset.py --split test                    # measure only
    python scripts/measure_crop_offset.py --split val --search --write    # choose the offset on the
                                                                          # validation fields and set it
    python scripts/measure_crop_offset.py --split test --require-aligned  # confirm on the test fields

--search tries the 3x3 whole-pixel neighbourhood of the one-step estimate,
because single fields scatter by about +-1 px and a one-step estimate can land
one pixel off; it recommends the offset whose median residual is smallest.

Parameters: config/diagnostics.yaml (crop_offset block).
Outputs, under <paths.output_root>/<output_dir>/:
    crop_offset_<split>_dy<dy>_dx<dx>.csv    one row per field
    crop_offset_<split>_dy<dy>_dx<dx>.json   the summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.data import io as data_io
from holoqpi.utils import get_logger, resolve_device, write_csv

LOGGER = get_logger(__name__)
SPLITS = ("train", "val", "test")


def _group(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def _describe(values) -> dict:
    values = np.asarray(values, dtype=float)
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "median": float(q[2]), "p25": float(q[1]),
            "p75": float(q[3]), "p05": float(q[0]), "p95": float(q[4]),
            "mean": float(values.mean())}


def _write_offset(config_path: Path, offset: tuple[int, int]) -> str:
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*crop_offset_px:\s*)\[[^\]\n]*\]", re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SystemExit(f"{config_path}: expected exactly one 'crop_offset_px: [..]' line, "
                         f"found {len(matches)}. Run apply_fixes.py first.")
    new_text = pattern.sub(lambda m: f"{m.group(1)}[{offset[0]}, {offset[1]}]", text, count=1)
    config_path.write_text(new_text, encoding="utf-8")
    return f"[{offset[0]}, {offset[1]}]"


def _measure(base, offset, splits, predict, settings, device) -> list[dict]:
    """Register the classical reconstruction against the reference on every field."""
    from skimage.registration import phase_cross_correlation

    # Every field once, whole, in its evaluation form: no random crop, no
    # augmentation, nothing dropped, whatever the split.
    cfg = base.merged({"data": {
        "modality": "off_axis", "train_crop": None, "eval_size": None,
        "augmentation": {"enabled": False}, "drop_last": False,
        "batch_size": 1, "eval_batch_size": 1, "provide_amplitude": False,
        "crop_offset_px": [int(offset[0]), int(offset[1])],
    }})
    loaders = build_dataloaders(cfg, splits_to_build=splits)
    rows = []
    with torch.no_grad():
        for split in splits:
            for batch in loaders[split]:
                phase = predict(batch)["phase"].squeeze(1).float().cpu().numpy()
                reference = batch["phase"].squeeze(1).float().numpy()
                for i in range(phase.shape[0]):
                    a = reference[i] - reference[i].mean()
                    b = phase[i] - phase[i].mean()
                    shift, error, _ = phase_cross_correlation(
                        a, b, upsample_factor=int(settings.upsample_factor),
                        normalization=settings.normalization)
                    rows.append({"stem": batch["stem"][i], "split": split,
                                 "group": _group(batch["stem"][i]),
                                 "offset_dy": int(offset[0]), "offset_dx": int(offset[1]),
                                 "dy": float(shift[0]), "dx": float(shift[1]),
                                 "error": float(error)})
    LOGGER.info("offset %s: %d fields registered", list(offset), len(rows))
    return rows


def _medians(rows) -> tuple[float, float]:
    return (float(np.median([r["dy"] for r in rows])),
            float(np.median([r["dx"] for r in rows])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--diagnostics", default="config/diagnostics.yaml")
    parser.add_argument("--split", default="test", choices=list(SPLITS) + ["all"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--search", action="store_true",
                        help="also try the 3x3 whole-pixel neighbourhood of the estimate "
                             "and recommend the offset with the smallest median residual")
    parser.add_argument("--write", action="store_true",
                        help="set data.crop_offset_px in --config to the recommended value")
    parser.add_argument("--require-aligned", action="store_true",
                        help="exit 4 unless the median residual is within tolerance")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    base = load_config(args.config, parse_overrides(args.set))
    diag = load_config(args.diagnostics)
    settings = diag.crop_offset
    device = resolve_device(args.device)
    current = data_io.hologram_crop_offset(base.data)
    splits = SPLITS if args.split == "all" else (args.split,)
    tolerance = float(settings.tolerance_px)

    from scripts.conventional_baseline import build_predictor

    cfg = base.merged({"data": {"modality": "off_axis"}})
    conv = cfg.evaluation.conventional_baseline
    distance = cfg.loss.forward_model.distance_um
    predict, _ = build_predictor(cfg, "off_axis", device, float(distance or 0.0),
                                 conv.gs_iterations, conv.aberration_order)
    LOGGER.info("classical reconstruction at z = %s um; current crop offset %s",
                distance, list(current))

    rows = _measure(base, current, splits, predict, settings, device)
    median = _medians(rows)
    aligned = abs(median[0]) <= tolerance and abs(median[1]) <= tolerance
    estimate = (current[0] - int(np.round(median[0])), current[1] - int(np.round(median[1])))

    candidates = {tuple(current): median}
    all_rows = list(rows)
    if args.search:
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                offset = (estimate[0] + ddy, estimate[1] + ddx)
                if offset in candidates:
                    continue
                trial = _measure(base, offset, splits, predict, settings, device)
                candidates[offset] = _medians(trial)
                all_rows += trial
        recommended = min(candidates, key=lambda o: (np.hypot(*candidates[o]), abs(o[0]) + abs(o[1])))
    else:
        recommended = estimate if not aligned else tuple(current)

    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    group_medians = {g: {"n": len(v), "dy": float(np.median([r["dy"] for r in v])),
                         "dx": float(np.median([r["dx"] for r in v]))}
                     for g, v in sorted(groups.items())}
    dy = np.array([r["dy"] for r in rows])
    dx = np.array([r["dx"] for r in rows])
    summary = {
        "split": args.split, "fields": len(rows), "current_offset": list(current),
        "residual_dy": _describe(dy), "residual_dx": _describe(dx),
        "tolerance_px": tolerance, "aligned": aligned,
        "estimate": list(estimate), "recommended_offset": list(recommended),
        "candidates": {f"{o[0]},{o[1]}": {"median_dy": m[0], "median_dx": m[1]}
                       for o, m in sorted(candidates.items())},
        "by_group": group_medians,
    }
    destination = Path(base.paths.output_root) / diag.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    name = f"crop_offset_{args.split}_dy{current[0]}_dx{current[1]}"
    write_csv(all_rows, destination / f"{name}.csv")
    (destination / f"{name}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Crop offset, {args.split} split, {len(rows)} fields ===")
    print(f"  current data.crop_offset_px   {list(current)}")
    for label, d in (("residual dy", summary["residual_dy"]), ("residual dx", summary["residual_dx"])):
        print(f"  {label}  median {d['median']:+.2f} px   IQR [{d['p25']:+.2f}, {d['p75']:+.2f}]"
              f"   5-95% [{d['p05']:+.2f}, {d['p95']:+.2f}]")
    print("  per group (median dy, dx):")
    for g, v in group_medians.items():
        print(f"    {g:<26} n={v['n']:>3}  {v['dy']:+.2f}  {v['dx']:+.2f}")
    if len(candidates) > 1:
        print("  candidate offsets (median residual dy, dx):")
        for o, m in sorted(candidates.items()):
            mark = "   <- recommended" if o == tuple(recommended) else ""
            print(f"    [{o[0]:>3}, {o[1]:>3}]   {m[0]:+.2f}  {m[1]:+.2f}{mark}")
    if aligned:
        print(f"\n  ALIGNED at the current offset: both medians within {tolerance} px.")
    else:
        print(f"\n  OFFSET PRESENT at the current setting.")
    print(f"  Recommended data.crop_offset_px: {list(recommended)}")
    print(f"  written to {destination}/{name}.csv and .json")

    if args.write:
        if tuple(recommended) == tuple(current):
            print("\n  --write: the recommended offset is the current one; config unchanged.")
        else:
            value = _write_offset(Path(args.config), recommended)
            print(f"\n  WROTE data.crop_offset_px: {value} to {args.config}")
            print("  Confirm on another split with --require-aligned.")

    if args.require_aligned and not aligned:
        print("\nNOT ALIGNED -- stopping (exit 4).")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
