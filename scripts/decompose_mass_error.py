"""Split the dry-mass error into a boundary part and a phase part, per run and per cell.

Dry mass is the phase integral over a cell's domain, so the ratio of predicted to
reference mass factorises exactly:

    m_pred / m_ref = [ S(Omega_pred, phi_ref) / S(Omega_ref, phi_ref) ]    domain factor
                   x [ S(Omega_pred, phi_pred) / S(Omega_pred, phi_ref) ]  phase factor

with S(Omega, phi) the sum of phi over Omega. The domain factor holds the phase at
the reference and moves only the boundary; the phase factor holds the boundary at
the prediction and changes only the phase.

Two levels are reported for every run.

FIELD LEVEL, identical to scripts/diagnose_bias.py: Omega is the whole semantic
foreground of a field (argmax mask), before instance splitting or any area
filter. Missed and invented cells therefore enter the domain factor.

CELL LEVEL, the evaluator's own chain: watershed instances, the 30-6000 um^2
filter of evaluation.measurement, and greedy IoU >= match_iou_threshold matching
(holoqpi.analysis.cells.measure_cells / match_cells, called exactly as
holoqpi/engine/evaluator.py calls them). For each matched pair, Omega_ref is the
reference instance and Omega_pred the matched predicted instance. Cells are split
into EDGE (any reference pixel within edge_margin_px of the field border) and
INTERIOR.

Summaries are the geometric mean of each factor, which composes exactly
(domain x phase = total), and the median |log ratio|, which measures spread
regardless of sign. A pair or field whose integrals are not all positive is
counted and excluded, as diagnose_bias.py does.

Two self-checks are printed per run and must agree with the run's stored
metrics_test.json: the matched-cell dry-mass MAPE and cell count, and the
dry-mass field-total MAPE (which also confirms which cells the field totals sum
over). A disagreement means the run was not reproduced and its rows should not
be used.

    python scripts/decompose_mass_error.py --device cuda:0
    python scripts/decompose_mass_error.py --device cuda:0 --only A_s42 --no-classical

Parameters: config/diagnostics.yaml (runs, edge margin, classical on/off).
Outputs, under <paths.output_root>/<output_dir>/:
    decomposition_cells.csv      one row per matched cell per run
    decomposition_fields.csv     one row per field per run (field level)
    decomposition_summary.csv    one row per run x level x group
    decomposition_by_config.csv  mean and SD over seeds per configuration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from holoqpi.analysis.cells import calibration_from_config, match_cells, measure_cells
from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.data.masks import split_instances
from holoqpi.engine import load_checkpoint
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device, write_csv

LOGGER = get_logger(__name__)

GROUPS = ("all", "edge", "interior")


# ----------------------------------------------------------------------------- summaries
def _log_ratios(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log(values[np.isfinite(values) & (values > 0)])


def _geometric_mean(values) -> float:
    logs = _log_ratios(values)
    return float(np.exp(logs.mean())) if logs.size else float("nan")


def _median_abs_log(values) -> float:
    logs = _log_ratios(values)
    return float(np.median(np.abs(logs))) if logs.size else float("nan")


def _summarise(domain, phase, total) -> dict:
    return {
        "n": int(len(total)),
        "domain_gm": _geometric_mean(domain),
        "phase_gm": _geometric_mean(phase),
        "total_gm": _geometric_mean(total),
        "domain_med_abs_log": _median_abs_log(domain),
        "phase_med_abs_log": _median_abs_log(phase),
        "total_med_abs_log": _median_abs_log(total),
    }


# ----------------------------------------------------------------------------- one field
def _edge_labels(labels: np.ndarray, margin: int) -> set[int]:
    """Instance labels with any pixel within ``margin`` px of the array border."""
    from scipy import ndimage

    height, width = labels.shape
    edge = set()
    for index, box in enumerate(ndimage.find_objects(labels)):
        if box is None:
            continue
        rows, cols = box
        if (rows.start < margin or cols.start < margin
                or rows.stop > height - margin or cols.stop > width - margin):
            edge.add(index + 1)
    return edge


def _analyse_field(phase_pred, mask_pred, phase_ref, mask_ref, stem, cfg, calibration,
                   margin, run_label):
    evaluation = cfg.evaluation
    method = evaluation.segmentation.instance_from
    distance = cfg.mask_generation.watershed_min_distance_px
    measurement = evaluation.measurement

    mp, mg = mask_pred > 0, mask_ref > 0
    field_row = {"run": run_label, "stem": stem, "status": "ok"}

    # ---- field level, as scripts/diagnose_bias.py
    if not mg.any():
        field_row["status"] = "empty_reference"
    elif not mp.any():
        field_row["status"] = "empty_prediction"
    else:
        s_ref = float(phase_ref[mg].sum())
        s_dom = float(phase_ref[mp].sum())
        s_pred = float(phase_pred[mp].sum())
        field_row.update(S_ref=s_ref, S_domain=s_dom, S_pred=s_pred)
        if min(s_ref, s_dom, s_pred) <= 0:
            field_row["status"] = "non_positive"
        else:
            field_row.update(domain=s_dom / s_ref, phase=s_pred / s_dom, total=s_pred / s_ref)

    # ---- cell level, as holoqpi/engine/evaluator.py
    pred_labels = split_instances(mp.astype(np.uint8), method, distance)
    ref_labels = split_instances(mg.astype(np.uint8), method, distance)
    predicted = measure_cells(phase_pred, mask_pred, calibration, measurement, method,
                              distance, labels=pred_labels)
    reference = measure_cells(phase_ref, mask_ref, calibration, measurement, method,
                              distance, labels=ref_labels)
    pairs = match_cells(predicted, reference, pred_labels, ref_labels,
                        measurement.match_iou_threshold)

    # Field totals over every accepted cell on each side, as MeasurementMetrics does.
    field_row["field_total_mass_pred"] = float(sum(c["dry_mass_pg"] for c in predicted))
    field_row["field_total_mass_ref"] = float(sum(c["dry_mass_pg"] for c in reference))
    field_row["n_ref_cells"] = len(reference)
    field_row["n_pred_cells"] = len(predicted)
    field_row["n_matched"] = len(pairs)

    size = int(max(pred_labels.max(), ref_labels.max())) + 1
    s_pred_by = np.bincount(pred_labels.ravel(), weights=phase_pred.ravel().astype(np.float64),
                            minlength=size)
    s_dom_by = np.bincount(pred_labels.ravel(), weights=phase_ref.ravel().astype(np.float64),
                           minlength=size)
    s_ref_by = np.bincount(ref_labels.ravel(), weights=phase_ref.ravel().astype(np.float64),
                           minlength=size)
    edge = _edge_labels(ref_labels, margin)
    field_row["n_ref_edge"] = sum(1 for c in reference if c["label"] in edge)

    cells = []
    for p, r in pairs:
        s_ref, s_dom, s_pred = s_ref_by[r["label"]], s_dom_by[p["label"]], s_pred_by[p["label"]]
        row = {
            "run": run_label, "stem": stem,
            "ref_label": r["label"], "pred_label": p["label"],
            "edge": int(r["label"] in edge), "match_iou": p.get("match_iou"),
            "area_um2_ref": r["area_um2"], "area_um2_pred": p["area_um2"],
            "dry_mass_pg_ref": r["dry_mass_pg"], "dry_mass_pg_pred": p["dry_mass_pg"],
            "S_ref": s_ref, "S_domain": s_dom, "S_pred": s_pred, "status": "ok",
        }
        if min(s_ref, s_dom, s_pred) <= 0:
            row["status"] = "non_positive"
        else:
            row.update(domain=s_dom / s_ref, phase=s_pred / s_dom, total=s_pred / s_ref)
        cells.append(row)
    return field_row, cells


# ----------------------------------------------------------------------------- one run
def _run(label, loader, predict, cfg, margin, stored_metrics=None):
    calibration = calibration_from_config(cfg)
    fields, cells = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = predict(batch)
            phase_pred = outputs["phase"].squeeze(1).float().cpu().numpy()
            mask_pred = outputs["segmentation"].argmax(dim=1).cpu().numpy()
            phase_ref = batch["phase"].squeeze(1).float().numpy()
            mask_ref = batch["mask"].numpy()
            for i in range(phase_pred.shape[0]):
                f, c = _analyse_field(phase_pred[i], mask_pred[i], phase_ref[i], mask_ref[i],
                                      batch["stem"][i], cfg, calibration, margin, label)
                fields.append(f)
                cells.extend(c)
        LOGGER.info("%s: %d fields, %d matched cells", label, len(fields), len(cells))

    summary = []
    ok_fields = [f for f in fields if f["status"] == "ok"]
    row = {"run": label, "level": "field", "group": "all",
           "excluded": len(fields) - len(ok_fields)}
    row.update(_summarise([f["domain"] for f in ok_fields], [f["phase"] for f in ok_fields],
                          [f["total"] for f in ok_fields]))
    summary.append(row)

    for group in GROUPS:
        chosen = [c for c in cells if group == "all" or c["edge"] == (group == "edge")]
        ok = [c for c in chosen if c["status"] == "ok"]
        row = {"run": label, "level": "cell", "group": group, "excluded": len(chosen) - len(ok)}
        row.update(_summarise([c["domain"] for c in ok], [c["phase"] for c in ok],
                              [c["total"] for c in ok]))
        n_ref = sum(f.get("n_ref_edge" if group == "edge" else "n_ref_cells", 0) for f in fields)
        if group == "interior":
            n_ref = sum(f.get("n_ref_cells", 0) - f.get("n_ref_edge", 0) for f in fields)
        row["n_reference_cells"] = n_ref
        row["recall"] = len(chosen) / n_ref if n_ref else float("nan")
        summary.append(row)

    # ---- self-checks against the stored evaluation
    mass_p = np.array([c["dry_mass_pg_pred"] for c in cells], dtype=float)
    mass_r = np.array([c["dry_mass_pg_ref"] for c in cells], dtype=float)
    valid = np.abs(mass_r) > 0
    mape = float(np.mean(np.abs(mass_p[valid] - mass_r[valid]) / np.abs(mass_r[valid])))
    tot_p = np.array([f["field_total_mass_pred"] for f in fields], dtype=float)
    tot_r = np.array([f["field_total_mass_ref"] for f in fields], dtype=float)
    usable = tot_r > 0
    field_total_mape = float(np.mean(np.abs(tot_p[usable] - tot_r[usable]) / tot_r[usable]))
    check = {"run": label, "cells_matched": len(cells), "dry_mass_mape": mape,
             "dry_mass_field_total_mape": field_total_mape}
    if stored_metrics:
        check.update(
            stored_cells_matched=stored_metrics.get("dry_mass_n_cells"),
            stored_dry_mass_mape=stored_metrics.get("dry_mass_mape"),
            stored_dry_mass_field_total_mape=stored_metrics.get("dry_mass_field_total_mape"),
        )
    return fields, cells, summary, check


# ----------------------------------------------------------------------------- printing
def _print_summary(summary: list[dict]) -> None:
    header = (f"{'run':<10} {'level':<5} {'group':<8} {'n':>5} {'excl':>4} "
              f"{'domain':>7} {'phase':>7} {'total':>7}  {'|log|d':>7} {'|log|p':>7} "
              f"{'|log|t':>7} {'recall':>6}")
    print(header)
    print("-" * len(header))
    for r in summary:
        recall = r.get("recall")
        print(f"{r['run']:<10} {r['level']:<5} {r['group']:<8} {r['n']:>5} {r['excluded']:>4} "
              f"{r['domain_gm']:>7.4f} {r['phase_gm']:>7.4f} {r['total_gm']:>7.4f}  "
              f"{r['domain_med_abs_log']:>7.4f} {r['phase_med_abs_log']:>7.4f} "
              f"{r['total_med_abs_log']:>7.4f} "
              f"{(f'{recall:.3f}' if recall is not None and np.isfinite(recall) else ''):>6}")


def _by_config(summary: list[dict]) -> list[dict]:
    """Mean and SD over seeds, grouping runs by the label prefix before '_s'."""
    keys = ("domain_gm", "phase_gm", "total_gm", "domain_med_abs_log", "phase_med_abs_log",
            "total_med_abs_log", "recall")
    grouped: dict[tuple, list[dict]] = {}
    for r in summary:
        config = r["run"].split("_s")[0]
        grouped.setdefault((config, r["level"], r["group"]), []).append(r)
    out = []
    for (config, level, group), rows in grouped.items():
        entry = {"config": config, "level": level, "group": group, "runs": len(rows)}
        for key in keys:
            values = np.array([r.get(key, np.nan) for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            entry[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
            entry[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        out.append(entry)
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--diagnostics", default="config/diagnostics.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--only", nargs="*", default=None, metavar="LABEL",
                        help="run labels from diagnostics.yaml to process (default: all)")
    parser.add_argument("--no-classical", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides applied to base.yaml and to every run's config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    base = load_config(args.config, overrides)
    diag = load_config(args.diagnostics)
    settings = diag.decomposition
    split, margin = settings.split, int(settings.edge_margin_px)
    device = resolve_device(args.device)

    output_root = Path(base.paths.output_root)
    destination = output_root / diag.output_dir
    destination.mkdir(parents=True, exist_ok=True)

    all_fields, all_cells, all_summary, checks = [], [], [], []

    for label, run_dir_name in settings.runs.items():
        if args.only and label not in args.only:
            continue
        run_dir = output_root / run_dir_name
        checkpoint, resolved = run_dir / "best_model.pt", run_dir / "resolved_config.yaml"
        if not checkpoint.is_file() or not resolved.is_file():
            LOGGER.warning("%s: missing best_model.pt or resolved_config.yaml in %s; skipped",
                           label, run_dir)
            continue
        cfg = load_config(resolved, overrides)
        model = build_model(cfg)
        load_checkpoint(model, checkpoint, device)
        model.to(device).eval()
        loader = build_dataloaders(cfg, splits_to_build=(split,))[split]
        stored = run_dir / f"metrics_{split}.json"
        stored_metrics = json.loads(stored.read_text()) if stored.is_file() else None

        def predict(batch, model=model):
            return model(batch["hologram"].to(device))

        f, c, s, chk = _run(label, loader, predict, cfg, margin, stored_metrics)
        all_fields += f
        all_cells += c
        all_summary += s
        checks.append(chk)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if settings.include_classical and not args.no_classical and (
            not args.only or "classical" in args.only):
        from scripts.conventional_baseline import build_predictor

        cfg = base.merged({"data": {"modality": "off_axis"}})
        conv = cfg.evaluation.conventional_baseline
        distance = cfg.loss.forward_model.distance_um
        predict, _ = build_predictor(cfg, "off_axis", device, float(distance or 0.0),
                                     conv.gs_iterations, conv.aberration_order)
        loader = build_dataloaders(cfg, splits_to_build=(split,))[split]
        stored = output_root / "conventional_off_axis" / f"metrics_{split}.json"
        stored_metrics = json.loads(stored.read_text()) if stored.is_file() else None
        f, c, s, chk = _run("classical", loader, predict, cfg, margin, stored_metrics)
        all_fields += f
        all_cells += c
        all_summary += s
        checks.append(chk)

    if not all_summary:
        print("HARD ERROR: no run was processed.")
        return 3

    write_csv(all_cells, destination / "decomposition_cells.csv")
    write_csv(all_fields, destination / "decomposition_fields.csv")
    write_csv(all_summary, destination / "decomposition_summary.csv")
    by_config = _by_config(all_summary)
    write_csv(by_config, destination / "decomposition_by_config.csv")

    print("\n=== Self-check against stored metrics (must agree) ===")
    for chk in checks:
        print(f"  {chk['run']:<10} matched {chk['cells_matched']:>5} "
              f"(stored {chk.get('stored_cells_matched')})   "
              f"dry-mass MAPE {chk['dry_mass_mape']:.4f} "
              f"(stored {chk.get('stored_dry_mass_mape')})   "
              f"field-total MAPE {chk['dry_mass_field_total_mape']:.4f} "
              f"(stored {chk.get('stored_dry_mass_field_total_mape')})")

    print(f"\n=== Decomposition ({split} split; edge = within {margin} px of the border) ===")
    print("domain/phase/total: geometric mean of the ratio (domain x phase = total)")
    print("|log|: median |log ratio| (spread, sign-free)\n")
    _print_summary(all_summary)

    print("\n=== Mean over seeds per configuration (geometric means) ===")
    for e in by_config:
        print(f"  {e['config']:<9} {e['level']:<5} {e['group']:<8} runs={e['runs']}  "
              f"domain {e['domain_gm_mean']:.4f}±{e['domain_gm_sd']:.4f}  "
              f"phase {e['phase_gm_mean']:.4f}±{e['phase_gm_sd']:.4f}  "
              f"total {e['total_gm_mean']:.4f}±{e['total_gm_sd']:.4f}")
    print(f"\nWritten to {destination}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
