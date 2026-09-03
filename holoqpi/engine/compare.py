"""Off-axis versus in-line Gabor comparison.

The central contribution the brief asks for. Both arms share the split file, the
architecture, the objective, the schedule and the seed; the modality is the only
free variable, so any difference in the reported numbers is attributable to the
hologram type rather than to the experimental setup.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config, save_config
from ..data import build_dataloaders
from ..models import build_model
from ..utils import get_logger, resolve_device, run_directory, seed_everything, write_csv, write_json
from .evaluator import Evaluator, save_per_cell
from .trainer import Trainer, load_checkpoint

LOGGER = get_logger(__name__)

# Metrics carried into the head-to-head table, in the order the brief lists them.
COMPARISON_METRICS = [
    # 1. quantitative phase reconstruction accuracy
    "phase_mae_rad",
    "phase_rmse_rad",
    "phase_psnr_db",
    "phase_ssim",
    "phase_pearson_r",
    "phase_mae_rad_in_cell",        # where the measurement is actually taken
    "phase_bias_rad_in_cell",
    "phase_mae_rad_background",
    # 2. segmentation performance
    "seg_dice",
    "seg_iou",
    "seg_aji",
    "seg_boundary_f1",
    # 3. morphology classification accuracy
    "cls_accuracy",
    "cls_macro_f1",
    # 4-6. projected area, optical volume, dry mass
    "area_mape",
    "area_cell_pearson_r",
    "area_image_pearson_r",
    "area_relative_bias",
    "optical_volume_mape",
    "dry_mass_mape",
    "dry_mass_cell_pearson_r",
    "dry_mass_image_pearson_r",
    "dry_mass_relative_bias",
    "dry_mass_loa_lower",
    "dry_mass_loa_upper",
    # detection: how many cells reached the measurement at all
    "cells_reference",
    "cells_detected",
    "cells_matched",
    "detection_recall",
    "detection_precision",
    "detection_f1",
]


def run_single_modality(cfg: Config, modality: str, train: bool = True) -> dict:
    """Train (optionally) and evaluate one arm; returns its test metrics."""
    cfg = cfg.merged({"data": {"modality": modality}})
    device = resolve_device(cfg.get("device", "auto"))
    seed_everything(cfg.project.seed, cfg.project.deterministic)

    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, modality)
    save_config(cfg, run_dir / "resolved_config.yaml")

    LOGGER.info("=" * 70)
    LOGGER.info("modality=%s  device=%s  run_dir=%s", modality, device, run_dir)
    LOGGER.info("=" * 70)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg)

    if train:
        trainer = Trainer(model, cfg, loaders, device, run_dir)
        trainer.train()

    checkpoint_path = run_dir / "best_model.pt"
    if checkpoint_path.is_file():
        load_checkpoint(model, checkpoint_path, device)
    else:
        LOGGER.warning("no checkpoint at %s; evaluating current weights", checkpoint_path)
    model.to(device)

    split = "test" if "test" in loaders else "val"
    evaluator = Evaluator(cfg, device)
    evaluation = evaluator.run(
        model, loaders[split], collect_per_cell=cfg.evaluation.save_per_cell_csv
    )

    write_json(evaluation["metrics"], run_dir / f"metrics_{split}.json")
    write_json(evaluation["confusion_matrix"], run_dir / f"confusion_{split}.json")
    if cfg.evaluation.save_per_cell_csv:
        save_per_cell(evaluation["per_cell"], run_dir / f"per_cell_{split}.csv")
        save_per_cell(evaluation["unmatched"], run_dir / f"unmatched_{split}.csv")

    LOGGER.info("modality=%s %s metrics written to %s", modality, split, run_dir)
    return evaluation["metrics"]


def compare_modalities(cfg: Config, modalities: list[str], train: bool = True) -> dict:
    """Run every arm and write the head-to-head comparison table."""
    results: dict[str, dict] = {}
    for modality in modalities:
        results[modality] = run_single_modality(cfg, modality, train=train)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_root = Path(cfg.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in COMPARISON_METRICS:
        row: dict = {"metric": metric}
        for modality in modalities:
            row[modality] = results[modality].get(metric)
        if len(modalities) == 2:
            first, second = (row[m] for m in modalities)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                row["difference"] = float(first) - float(second)
        rows.append(row)

    table_path = output_root / f"{cfg.experiment_name}_modality_comparison.csv"
    write_csv(rows, table_path)
    write_json(results, output_root / f"{cfg.experiment_name}_modality_comparison.json")

    LOGGER.info("comparison table written to %s", table_path)
    return {"per_modality": results, "table": rows}
