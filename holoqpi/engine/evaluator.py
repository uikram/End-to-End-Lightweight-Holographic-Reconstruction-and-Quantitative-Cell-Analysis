"""Evaluation: metrics for every head plus the per-cell measurement chain.

The evaluator runs the full pipeline the study reports on, so its output is the
same whether it is called for validation during training or for the final
test-set numbers of one arm of the modality comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..analysis.cells import calibration_from_config, match_cells, measure_cells
from ..config import Config
from ..data.masks import split_instances
from ..metrics import (
    ClassificationMetrics,
    ForwardModelMetrics,
    MeasurementMetrics,
    PhaseMetrics,
    SegmentationMetrics,
)
from ..utils import get_logger, write_csv

LOGGER = get_logger(__name__)


class Evaluator:
    """Scores a model on one dataloader."""

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        evaluation = cfg.evaluation
        self.calibration = calibration_from_config(cfg)
        self.measurement_cfg = evaluation.measurement
        self.instance_method = evaluation.segmentation.instance_from
        self.watershed_distance = cfg.mask_generation.watershed_min_distance_px

        self.phase_metrics = PhaseMetrics(evaluation.phase.psnr_data_range)
        self.segmentation_metrics = SegmentationMetrics(
            num_classes=cfg.model.segmentation_classes,
            boundary_tolerance=evaluation.segmentation.boundary_tolerance_px,
            instance_method=self.instance_method,
            watershed_min_distance=self.watershed_distance,
        )
        self.classification_metrics = ClassificationMetrics(list(cfg.labels.conditions))
        self.measurement_metrics = MeasurementMetrics(
            report_bland_altman=self.measurement_cfg.report_bland_altman
        )
        self.forward_metrics = (
            ForwardModelMetrics(cfg.loss.forward_model, cfg.optics, cfg.data.modality)
            if evaluation.forward_model.report_residual else None
        )

    def reset(self) -> None:
        self.phase_metrics.reset()
        self.segmentation_metrics.reset()
        self.classification_metrics.reset()
        self.measurement_metrics.reset()
        if self.forward_metrics is not None:
            self.forward_metrics.reset()

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module | None,
        loader,
        collect_per_cell: bool = False,
        loss_fn=None,
        predict_fn=None,
    ) -> dict:
        """Score a model, or any predictor, on one dataloader.

        ``predict_fn`` replaces the model call with an arbitrary callable taking
        the batch and returning the same output dictionary. The conventional
        reconstruction baseline uses it, which is what makes the classical and
        the learned pipeline comparable: they are scored by the same code, the
        same instance labelling and the same measurement chain, so a difference
        between them is a difference in reconstruction and nothing else.
        """
        if model is not None:
            model.eval()
        self.reset()

        per_cell_rows: list[dict] = []
        loss_totals: dict[str, float] = {}
        batches = 0

        for batch in loader:
            hologram = batch["hologram"].to(self.device, non_blocking=True)
            raw_hologram = batch.get("hologram_raw")
            if raw_hologram is not None:
                raw_hologram = raw_hologram.to(self.device, non_blocking=True)
            phase_target = batch["phase"].to(self.device, non_blocking=True)
            mask_target = batch["mask"].to(self.device, non_blocking=True)
            condition_target = batch["condition"].to(self.device, non_blocking=True)

            outputs = predict_fn(batch) if predict_fn is not None else model(hologram)

            if loss_fn is not None:
                moved = {
                    "phase": phase_target,
                    "mask": mask_target,
                    "condition": condition_target,
                    "hologram": hologram,
<<<<<<< Updated upstream
=======
                    "hologram_raw": raw_hologram,
>>>>>>> Stashed changes
                }
                _, components = loss_fn(outputs, moved)
                for key, value in components.items():
                    loss_totals[key] = loss_totals.get(key, 0.0) + value
                batches += 1

            phase_prediction = outputs["phase"].squeeze(1).float().cpu().numpy()
            phase_reference = phase_target.squeeze(1).float().cpu().numpy()
            mask_prediction = outputs["segmentation"].argmax(dim=1).cpu().numpy()
            mask_reference = mask_target.cpu().numpy()
            condition_prediction = outputs["condition"].argmax(dim=1).cpu().numpy()
            condition_reference = condition_target.cpu().numpy()

            # Label instances once per image and thread the result through every
            # consumer. Watershed on a full field costs more than the forward
            # pass, and it was previously repeated six times per image.
            predicted_instances = [
                split_instances(
                    (m > 0).astype(np.uint8), self.instance_method, self.watershed_distance
                )
                for m in mask_prediction
            ]
            reference_instances = [
                split_instances(
                    (m > 0).astype(np.uint8), self.instance_method, self.watershed_distance
                )
                for m in mask_reference
            ]

            self.phase_metrics.update(phase_prediction, phase_reference, mask=mask_reference)
            if self.forward_metrics is not None:
                self.forward_metrics.update(
                    outputs["phase"].float(), outputs.get("amplitude"),
<<<<<<< Updated upstream
                    hologram, reference_phase=phase_target.float(),
=======
                    raw_hologram if raw_hologram is not None else hologram,
                    reference_phase=phase_target.float(),
>>>>>>> Stashed changes
                )
            self.segmentation_metrics.update(
                mask_prediction, mask_reference,
                prediction_instances=predicted_instances,
                target_instances=reference_instances,
            )
            self.classification_metrics.update(condition_prediction, condition_reference)

            for index in range(phase_prediction.shape[0]):
                rows = self._measure_pair(
                    phase_prediction[index],
                    mask_prediction[index],
                    phase_reference[index],
                    mask_reference[index],
                    predicted_labels=predicted_instances[index],
                    reference_labels=reference_instances[index],
                    stem=batch["stem"][index],
                    condition_true=int(condition_reference[index]),
                    condition_pred=int(condition_prediction[index]),
                    collect=collect_per_cell,
                )
                per_cell_rows.extend(rows)

        results = {}
        results.update(self.phase_metrics.compute())
        results.update(self.segmentation_metrics.compute())
        results.update(self.classification_metrics.compute())
        results.update(self.measurement_metrics.compute())
        if self.forward_metrics is not None:
            results.update(self.forward_metrics.compute())

        if batches:
            for key, value in loss_totals.items():
                results[f"loss_{key}"] = value / batches

        return {
            "metrics": results,
            "confusion_matrix": self.classification_metrics.confusion_matrix,
            "per_cell": per_cell_rows,
            "unmatched": self.measurement_metrics.unmatched_rows() if collect_per_cell else [],
        }

    def _measure_pair(
        self,
        phase_prediction: np.ndarray,
        mask_prediction: np.ndarray,
        phase_reference: np.ndarray,
        mask_reference: np.ndarray,
        predicted_labels: np.ndarray,
        reference_labels: np.ndarray,
        stem: str,
        condition_true: int,
        condition_pred: int,
        collect: bool,
    ) -> list[dict]:
        predicted_cells = measure_cells(
            phase_prediction, mask_prediction, self.calibration,
            self.measurement_cfg, self.instance_method, self.watershed_distance,
            labels=predicted_labels,
        )
        reference_cells = measure_cells(
            phase_reference, mask_reference, self.calibration,
            self.measurement_cfg, self.instance_method, self.watershed_distance,
            labels=reference_labels,
        )

        pairs = match_cells(
            predicted_cells, reference_cells, predicted_labels, reference_labels,
            self.measurement_cfg.match_iou_threshold,
        )
        # Stamp the field of view onto every record before the metric object
        # accumulates it, so an unmatched cell can be traced back to its image.
        for record in predicted_cells:
            record["stem"] = stem
        for record in reference_cells:
            record["stem"] = stem

        self.measurement_metrics.update_pairs(pairs)
        self.measurement_metrics.update_image(pairs, predicted_cells, reference_cells)

        if not collect:
            return []

        condition_names = list(self.cfg.labels.conditions)
        rows = []
        for predicted, reference in pairs:
            rows.append(
                {
                    "stem": stem,
                    "condition_true": condition_names[condition_true],
                    "condition_pred": condition_names[condition_pred],
                    "area_um2_pred": predicted["area_um2"],
                    "area_um2_ref": reference["area_um2"],
                    "circularity_pred": predicted["circularity"],
                    "circularity_ref": reference["circularity"],
                    "optical_volume_pred": predicted["optical_volume_rad_um2"],
                    "optical_volume_ref": reference["optical_volume_rad_um2"],
                    "dry_mass_pg_pred": predicted["dry_mass_pg"],
                    "dry_mass_pg_ref": reference["dry_mass_pg"],
                    "mean_phase_pred": predicted["mean_phase_rad"],
                    "mean_phase_ref": reference["mean_phase_rad"],
                    "match_iou": predicted.get("match_iou", float("nan")),
                }
            )
        return rows


def composite_score(metrics: dict, weights: Config) -> float:
    """Single scalar for checkpoint selection.

    Combines region overlap, phase agreement, measurement accuracy and detection
    completeness so that a model cannot be selected for excelling at one head
    while failing the quantity the study reports.

    Detection F1 carries weight because the measurement terms are computed over
    IoU-matched cells only: a model that detects a handful of large, easy cells
    and misses the rest scores an excellent dry-mass MAPE. Without a detection
    term the selection rule actively prefers that model.
    """
    dice = float(metrics.get("seg_dice", 0.0))
    pearson = float(metrics.get("phase_pearson_r", 0.0))
    mape = metrics.get("dry_mass_mape", None)
    mass_accuracy = (
        float(np.clip(1.0 - mape, 0.0, 1.0)) if mape is not None and np.isfinite(mape) else 0.0
    )
    detection = metrics.get("detection_f1", None)
    detection = float(detection) if detection is not None and np.isfinite(detection) else 0.0

    return (
        weights.seg_dice * dice
        + weights.phase_pearson * max(pearson, 0.0)
        + weights.dry_mass_accuracy * mass_accuracy
        + float(weights.get("detection_f1", 0.0)) * detection
    )


def save_per_cell(rows: list[dict], destination: str | Path) -> Path | None:
    if not rows:
        LOGGER.warning("no matched cells to write to %s", destination)
        return None
    return write_csv(rows, destination)
