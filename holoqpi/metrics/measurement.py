"""Agreement between predicted and reference cellular measurements.

This is the metric family the professor's brief is ultimately about: whether a
reconstruction is *measurement-ready*. Two views are accumulated.

* per cell   -- cells paired between prediction and reference by IoU, giving
                relative errors on area, optical volume and dry mass
* per image  -- the median cell of each field of view, giving the Pearson
                agreement and Bland-Altman bias that a reviewer expects for a
                quantitative instrument
"""

from __future__ import annotations

import numpy as np

_QUANTITIES = ("area_um2", "optical_volume_rad_um2", "dry_mass_pg", "circularity")
_SHORT_NAME = {
    "area_um2": "area",
    "optical_volume_rad_um2": "optical_volume",
    "dry_mass_pg": "dry_mass",
    "circularity": "circularity",
}


class MeasurementMetrics:
    """Accumulates paired cell measurements and per-image summaries."""

    def __init__(self, report_bland_altman: bool = True):
        self.report_bland_altman = report_bland_altman
        self.reset()

    def reset(self) -> None:
        self._paired: dict[str, list[tuple[float, float]]] = {q: [] for q in _QUANTITIES}
        self._per_image: dict[str, list[tuple[float, float]]] = {q: [] for q in _QUANTITIES}
        self._predicted_counts: list[int] = []
        self._reference_counts: list[int] = []
        self._matched_counts: list[int] = []
        self._missed: list[dict] = []
        self._false_positives: list[dict] = []

    def update_pairs(self, pairs: list[tuple[dict, dict]]) -> None:
        for predicted, reference in pairs:
            for quantity in _QUANTITIES:
                p, r = predicted.get(quantity), reference.get(quantity)
                if p is None or r is None or not np.isfinite(p) or not np.isfinite(r):
                    continue
                self._paired[quantity].append((float(p), float(r)))

    def update_image(
        self,
        pairs: list[tuple[dict, dict]],
        predicted_cells: list[dict],
        reference_cells: list[dict],
    ) -> None:
        """Accumulate one field of view.

        The per-image summaries are taken over the IoU-matched pairs, the same
        population the per-cell errors use. Taking them over every detected and
        every reference cell instead would compare two different populations:
        cells the model missed are absent from one side and merged detections
        are absent from the other, so the difference between the medians would
        report the detection gap rather than the measurement accuracy, and it
        would do so under a name that says bias.

        How many cells were missed or invented is a real and important result,
        but it is reported separately by the detection counts below.
        """
        self._predicted_counts.append(len(predicted_cells))
        self._reference_counts.append(len(reference_cells))
        self._matched_counts.append(len(pairs))

        # Keep the cells that failed to pair. A missed reference cell and an
        # invented predicted cell are the two failure modes the paired metrics
        # cannot see, and their size distribution says whether the detector is
        # losing small cells, merged clusters, or something else.
        for record in reference_cells:
            if not np.isfinite(record.get("match_iou", float("nan"))):
                self._missed.append(record)
        for record in predicted_cells:
            if not np.isfinite(record.get("match_iou", float("nan"))):
                self._false_positives.append(record)

        if not pairs:
            return

        matched_predicted = [p for p, _ in pairs]
        matched_reference = [r for _, r in pairs]
        for quantity in _QUANTITIES:
            p = _median(matched_predicted, quantity)
            r = _median(matched_reference, quantity)
            if np.isfinite(p) and np.isfinite(r):
                self._per_image[quantity].append((p, r))

    def compute(self) -> dict:
        results: dict = {}

        for quantity in _QUANTITIES:
            short = _SHORT_NAME[quantity]

            pairs = self._paired[quantity]
            if pairs:
                predicted = np.array([p for p, _ in pairs])
                reference = np.array([r for _, r in pairs])
                results[f"{short}_mape"] = _mape(predicted, reference)
                results[f"{short}_mae"] = float(np.abs(predicted - reference).mean())
                results[f"{short}_cell_pearson_r"] = _pearson(predicted, reference)
                results[f"{short}_n_cells"] = int(predicted.size)

            per_image = self._per_image[quantity]
            if per_image:
                predicted = np.array([p for p, _ in per_image])
                reference = np.array([r for _, r in per_image])
                results[f"{short}_image_pearson_r"] = _pearson(predicted, reference)
                if self.report_bland_altman:
                    agreement = _bland_altman(predicted, reference)
                    results[f"{short}_relative_bias"] = agreement["bias"]
                    results[f"{short}_median_relative_bias"] = agreement["median_bias"]
                    results[f"{short}_loa_lower"] = agreement["loa_lower"]
                    results[f"{short}_loa_upper"] = agreement["loa_upper"]

        if self._reference_counts:
            predicted_total = int(np.sum(self._predicted_counts))
            reference_total = int(np.sum(self._reference_counts))
            matched_total = int(np.sum(self._matched_counts))
            results["cells_detected"] = predicted_total
            results["cells_reference"] = reference_total
            results["cells_matched"] = matched_total
            results["cell_count_ratio"] = (
                predicted_total / reference_total if reference_total else float("nan")
            )
            # Detection quality at the configured IoU threshold. cell_count_ratio
            # alone cannot express this: a model that misses half the cells and
            # invents an equal number of false positives scores a ratio of 1.0.
            recall = matched_total / reference_total if reference_total else float("nan")
            precision = matched_total / predicted_total if predicted_total else float("nan")
            results["detection_recall"] = recall
            results["detection_precision"] = precision
            results["detection_f1"] = (
                2 * precision * recall / (precision + recall)
                if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
                else float("nan")
            )
            results["cells_missed"] = len(self._missed)
            results["cells_false_positive"] = len(self._false_positives)
            if self._missed:
                results["missed_median_area_um2"] = _median(self._missed, "area_um2")
            if self._false_positives:
                results["false_positive_median_area_um2"] = _median(
                    self._false_positives, "area_um2"
                )

        return results

    def unmatched_rows(self) -> list[dict]:
        """Missed reference cells and false-positive detections, tagged by kind."""
        rows = []
        for record in self._missed:
            rows.append({"kind": "missed_reference", **record})
        for record in self._false_positives:
            rows.append({"kind": "false_positive", **record})
        return rows


def _median(cells: list[dict], quantity: str) -> float:
    values = [c[quantity] for c in cells if quantity in c and np.isfinite(c[quantity])]
    return float(np.median(values)) if values else float("nan")


def _mape(predicted: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.abs(reference)
    valid = denominator > 0
    if not valid.any():
        return float("nan")
    return float(
        np.mean(np.abs(predicted[valid] - reference[valid]) / denominator[valid])
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def _bland_altman(predicted: np.ndarray, reference: np.ndarray) -> dict:
    """Relative Bland-Altman agreement, dimensionless.

    The bias and the limits of agreement must describe the same centre, or the
    reported interval is not centred on the reported bias and the pair cannot be
    read as a Bland-Altman result. The standard construction is used: mean
    relative difference, limits at mean +/- 1.96 SD. The median is reported
    alongside as a robust cross-check rather than substituted for the mean.
    """
    empty = {"bias": float("nan"), "median_bias": float("nan"),
             "loa_lower": float("nan"), "loa_upper": float("nan"), "n": 0}
    mean_value = (predicted + reference) / 2.0
    valid = np.abs(mean_value) > 0
    if not valid.any():
        return empty

    relative = (predicted[valid] - reference[valid]) / mean_value[valid]
    mean_relative = float(np.mean(relative))
    spread = float(np.std(relative, ddof=1)) if relative.size > 1 else 0.0
    return {
        "bias": mean_relative,
        "median_bias": float(np.median(relative)),
        "loa_lower": mean_relative - 1.96 * spread,
        "loa_upper": mean_relative + 1.96 * spread,
        "n": int(relative.size),
    }
