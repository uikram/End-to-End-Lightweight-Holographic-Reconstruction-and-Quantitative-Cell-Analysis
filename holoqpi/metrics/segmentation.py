"""Segmentation accuracy: region overlap, instance separation, contour fidelity.

Instance separation matters here beyond the usual reasons. Every downstream
quantity in this study is *per cell*, so a prediction that covers the right
pixels but merges two touching cells into one component yields two wrong
measurements even though its Dice score is excellent.
"""

from __future__ import annotations

import numpy as np


class SegmentationMetrics:
    """Accumulates pixel-, instance- and boundary-level statistics."""

    def __init__(self, num_classes: int, boundary_tolerance: int,
                 instance_method: str, watershed_min_distance: int):
        self.num_classes = num_classes
        self.boundary_tolerance = boundary_tolerance
        self.instance_method = instance_method
        self.watershed_min_distance = watershed_min_distance
        self.reset()

    def reset(self) -> None:
        self._tp = np.zeros(self.num_classes)
        self._fp = np.zeros(self.num_classes)
        self._fn = np.zeros(self.num_classes)
        self._aji_intersection = 0.0
        self._aji_union = 0.0
        self._boundary_tp = 0.0
        self._boundary_fp = 0.0
        self._boundary_fn = 0.0
        self._instance_count_error: list[float] = []
        self._n = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        prediction_instances: list | None = None,
        target_instances: list | None = None,
    ) -> None:
        """``prediction`` and ``target`` are (B, H, W) integer class maps.

        Instance labellings may be supplied when the caller has already computed
        them; watershed dominates the cost of an evaluation pass.
        """
        from ..data.masks import split_instances

        for index, (predicted, actual) in enumerate(
            zip(np.atleast_3d(prediction), np.atleast_3d(target))
        ):
            for class_index in range(self.num_classes):
                predicted_class = predicted == class_index
                actual_class = actual == class_index
                self._tp[class_index] += np.logical_and(predicted_class, actual_class).sum()
                self._fp[class_index] += np.logical_and(predicted_class, ~actual_class).sum()
                self._fn[class_index] += np.logical_and(~predicted_class, actual_class).sum()

            predicted_fg = (predicted > 0).astype(np.uint8)
            actual_fg = (actual > 0).astype(np.uint8)

            if prediction_instances is not None:
                predicted_instances = prediction_instances[index]
            else:
                predicted_instances = split_instances(
                    predicted_fg, self.instance_method, self.watershed_min_distance
                )
            if target_instances is not None:
                actual_instances = target_instances[index]
            else:
                actual_instances = split_instances(
                    actual_fg, self.instance_method, self.watershed_min_distance
                )

            intersection, union = aggregated_jaccard(predicted_instances, actual_instances)
            self._aji_intersection += intersection
            self._aji_union += union

            tp, fp, fn = boundary_counts(predicted_fg, actual_fg, self.boundary_tolerance)
            self._boundary_tp += tp
            self._boundary_fp += fp
            self._boundary_fn += fn

            n_actual = int(actual_instances.max())
            if n_actual > 0:
                n_predicted = int(predicted_instances.max())
                self._instance_count_error.append(abs(n_predicted - n_actual) / n_actual)

            self._n += 1

    def compute(self) -> dict:
        if self._n == 0:
            return {}
        eps = 1e-8

        dice = (2 * self._tp + eps) / (2 * self._tp + self._fp + self._fn + eps)
        iou = (self._tp + eps) / (self._tp + self._fp + self._fn + eps)

        precision = (self._boundary_tp + eps) / (self._boundary_tp + self._boundary_fp + eps)
        recall = (self._boundary_tp + eps) / (self._boundary_tp + self._boundary_fn + eps)

        results = {
            # Foreground is the class the study measures; index 0 is background.
            "seg_dice": float(dice[1]) if self.num_classes > 1 else float(dice[0]),
            "seg_iou": float(iou[1]) if self.num_classes > 1 else float(iou[0]),
            "seg_dice_macro": float(np.mean(dice[1:])) if self.num_classes > 1 else float(dice[0]),
            "seg_aji": float((self._aji_intersection + eps) / (self._aji_union + eps)),
            "seg_boundary_f1": float(2 * precision * recall / (precision + recall + eps)),
            "seg_n_images": self._n,
        }
        if self._instance_count_error:
            results["seg_instance_count_mape"] = float(np.mean(self._instance_count_error))
        return results


def aggregated_jaccard(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Aggregated Jaccard Index numerator and denominator for one image pair.

    Intersections for every (target, prediction) pair are obtained in a single
    bincount over combined labels, which keeps the cost bounded when a model
    emits thousands of speckle components.
    """
    predicted_ids = np.unique(prediction[prediction > 0])
    target_ids = np.unique(target[target > 0])

    if target_ids.size == 0 and predicted_ids.size == 0:
        return 1.0, 1.0
    if target_ids.size == 0:
        return 0.0, float((prediction > 0).sum())
    if predicted_ids.size == 0:
        return 0.0, float((target > 0).sum())

    stride = int(prediction.max()) + 1
    predicted_areas = np.bincount(prediction.ravel(), minlength=stride)
    target_areas = np.bincount(target.ravel(), minlength=int(target.max()) + 1)

    overlap = (prediction > 0) & (target > 0)
    pair_counts = np.bincount((target[overlap] * stride + prediction[overlap]).ravel())

    total_intersection = 0.0
    total_union = 0.0
    matched: set[int] = set()

    for target_id in target_ids:
        best_iou = 0.0
        best = None
        for predicted_id in predicted_ids:
            index = target_id * stride + predicted_id
            if index >= pair_counts.size:
                continue
            intersection = float(pair_counts[index])
            if intersection <= 0:
                continue
            union = float(target_areas[target_id] + predicted_areas[predicted_id] - intersection)
            iou = intersection / union
            if iou > best_iou:
                best_iou, best = iou, (predicted_id, intersection, union)

        if best is None:
            total_union += float(target_areas[target_id])
        else:
            predicted_id, intersection, union = best
            total_intersection += intersection
            total_union += union
            matched.add(int(predicted_id))

    for predicted_id in predicted_ids:
        if int(predicted_id) not in matched:
            total_union += float(predicted_areas[predicted_id])

    return total_intersection, total_union


def boundary_counts(prediction: np.ndarray, target: np.ndarray,
                    tolerance: int) -> tuple[float, float, float]:
    """Boundary-F1 counts with a fixed pixel tolerance."""
    from scipy.ndimage import binary_dilation, binary_erosion

    def contour(mask: np.ndarray) -> np.ndarray:
        return mask.astype(bool) & ~binary_erosion(mask.astype(bool))

    predicted_contour = contour(prediction)
    target_contour = contour(target)

    if not predicted_contour.any() and not target_contour.any():
        return 1.0, 0.0, 0.0

    predicted_zone = binary_dilation(predicted_contour, iterations=tolerance)
    target_zone = binary_dilation(target_contour, iterations=tolerance)

    tp = float(np.logical_and(predicted_contour, target_zone).sum())
    fp = float(np.logical_and(predicted_contour, ~target_zone).sum())
    fn = float(np.logical_and(target_contour, ~predicted_zone).sum())
    return tp, fp, fn
