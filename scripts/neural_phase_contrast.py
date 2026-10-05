"""Recovered in-cell phase contrast of the trained neural models.

Table 5 reports a "recovered in-cell phase contrast" for the two classical
pipelines (stored in ``recovered_phase_contrast_rad`` by
scripts/conventional_baseline.py). This script computes the SAME quantity for
the neural models so that all four columns of the table are defined identically:

    per field f:   c_f = mean(phase[pixels inside the reference cell mask])
                         - mean(phase[pixels outside it])
    reported:      median over the scored fields of c_f

(the definition in scripts/conventional_baseline.py, ``predict()``: fields with
an empty reference mask or no background are skipped).

The neural phase map is the model's ``phase`` output in the same units (rad) that
the classical reconstruction is scored in. The reference phase gives the same
statistic for the reference itself (``reference_median_contrast_rad``), which is
the number the recovered contrast should be read against.

Fields: the 107 test fields whose in-line frame is correctly paired, i.e. the
same data.exclude override that produced metrics_test_common.json
(``data.exclude.modalities = [gabor, off_axis]``).

Run on the server (needs the checkpoints and the data), one GPU:

    CUDA_VISIBLE_DEVICES=2 python scripts/neural_phase_contrast.py \\
        --config config/base.yaml --device cuda

Writes runs/common_fields/neural_phase_contrast.json, which
analysis/compile_results.py reads. ``--with-classical`` additionally re-runs
the classical off-axis predictor through the same function and checks it
reproduces the stored 0.9216 rad (a check that this script's definition equals
the classical one).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.engine import load_checkpoint
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device

LOGGER = get_logger(__name__)

# label -> run directory under paths.output_root (training seeds in parentheses)
RUNS = {
    "off_axis_neural": {
        "42": "v2_baseline_off_axis",
        "1337": "v2_A_seed1337_off_axis",
        "2024": "v2_A_seed2024_off_axis",
    },
    "in_line_neural": {"42": "v2_baseline_gabor"},
}
COMMON_EXCLUDE = {"data": {"exclude": {"modalities": ["gabor", "off_axis"]}}}


def field_contrast(phase: np.ndarray, reference_mask: np.ndarray) -> float | None:
    inside = reference_mask > 0
    if not inside.any() or inside.all():
        return None
    return float(phase[inside].mean() - phase[~inside].mean())


def score(predict, loader, device) -> dict:
    pred_c, ref_c, stems = [], [], []
    with torch.no_grad():
        for batch in loader:
            phase = predict(batch)["phase"].squeeze(1).float().cpu().numpy()
            reference = batch["phase"].squeeze(1).numpy()
            mask = batch["mask"].numpy()
            for i in range(phase.shape[0]):
                c = field_contrast(phase[i], mask[i])
                if c is None:
                    continue
                pred_c.append(c)
                ref_c.append(field_contrast(reference[i], mask[i]))
                stems.append(batch["stem"][i] if "stem" in batch else f"image_{len(stems)}")
    return {"median_contrast_rad": float(np.median(pred_c)),
            "reference_median_contrast_rad": float(np.median(ref_c)),
            "n_fields": len(pred_c), "per_field_contrast_rad": dict(zip(stems, pred_c))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-classical", action="store_true")
    parser.add_argument("--output", default="common_fields/neural_phase_contrast.json",
                        help="relative to paths.output_root")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    base = load_config(args.config, overrides)
    device = resolve_device(args.device)
    root = Path(base.paths.output_root)
    result: dict = {"definition": __doc__.split("Fields:")[0].strip(), "split": args.split,
                    "fields": "107 common test fields (data.exclude.modalities = [gabor, off_axis])"}
    reference_medians = []

    for label, runs in RUNS.items():
        if label == "in_line_neural":
            result[label] = {}
        else:
            result[label] = {}
        for seed, directory in runs.items():
            run_dir = root / directory
            checkpoint, resolved = run_dir / "best_model.pt", run_dir / "resolved_config.yaml"
            if not checkpoint.is_file():
                print(f"HARD ERROR: {checkpoint} not found (checkpoints are not in the repository)")
                return 3
            cfg = load_config(resolved, overrides).merged(COMMON_EXCLUDE)
            model = build_model(cfg)
            load_checkpoint(model, checkpoint, device)
            model.to(device).eval()
            loader = build_dataloaders(cfg, splits_to_build=(args.split,))[args.split]
            r = score(lambda b, m=model: m(b["hologram"].to(device)), loader, device)
            LOGGER.info("%s seed %s: %d fields, median contrast %+.4f rad (reference %+.4f rad)",
                        label, seed, r["n_fields"], r["median_contrast_rad"],
                        r["reference_median_contrast_rad"])
            if label == "in_line_neural":
                result[label] = {**r, "seed": int(seed)}
            else:
                result[label][seed] = r
            reference_medians.append(r["reference_median_contrast_rad"])
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result["reference_median_contrast_rad"] = float(np.median(reference_medians))
    if args.with_classical:
        from scripts.conventional_baseline import build_predictor

        cfg = base.merged({"data": {"modality": "off_axis"}}).merged(COMMON_EXCLUDE)
        conv = cfg.evaluation.conventional_baseline
        predict, _ = build_predictor(cfg, "off_axis", device,
                                     float(cfg.loss.forward_model.distance_um or 0.0),
                                     conv.gs_iterations, conv.aberration_order)
        loader = build_dataloaders(cfg, splits_to_build=(args.split,))[args.split]
        r = score(predict, loader, device)
        stored = json.loads((root / "common_fields" / "conventional_off_axis" /
                             "metrics_test.json").read_text())["recovered_phase_contrast_rad"]
        result["classical_off_axis_recheck"] = {"recomputed": r["median_contrast_rad"], "stored": stored,
                                                "agree": abs(r["median_contrast_rad"] - stored) < 1e-6}
        LOGGER.info("classical re-check: %.6f vs stored %.6f", r["median_contrast_rad"], stored)

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
