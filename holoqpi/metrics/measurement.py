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

    def update_pairs(self, pairs: list[tuple[dict, dict]]) -> None:
        for predicted, reference in pairs:
            for quantity in _QUANTITIES:
                p, r = predicted.get(quantity), reference.get(quantity)
                if p is None or r is None or not np.isfinite(p) or not np.isfinite(r):
                    continue
                self._paired[quantity].append((float(p), float(r)))

    def update_image(self, predicted_cells: list[dict], reference_cells: list[dict]) -> None:
        self._predicted_counts.append(len(predicted_cells))
        self._reference_counts.append(len(reference_cells))
        if not predicted_cells or not reference_cells:
            return
        for quantity in _QUANTITIES:
            p = _median(predicted_cells, quantity)
            r = _median(reference_cells, quantity)
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
                    bias, limits = _bland_altman(predicted, reference)
                    results[f"{short}_median_relative_bias"] = bias
                    results[f"{short}_loa_lower"] = limits[0]
                    results[f"{short}_loa_upper"] = limits[1]

        if self._reference_counts:
            predicted_total = int(np.sum(self._predicted_counts))
            reference_total = int(np.sum(self._reference_counts))
            results["cells_detected"] = predicted_total
            results["cells_reference"] = reference_total
            results["cell_count_ratio"] = (
                predicted_total / reference_total if reference_total else float("nan")
            )

        return results


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


def _bland_altman(predicted: np.ndarray, reference: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Median relative bias and 95% limits of agreement, both dimensionless."""
    mean_value = (predicted + reference) / 2.0
    valid = np.abs(mean_value) > 0
    if not valid.any():
        return float("nan"), (float("nan"), float("nan"))

    relative = (predicted[valid] - reference[valid]) / mean_value[valid]
    bias = float(np.median(relative))
    spread = float(np.std(relative, ddof=1)) if relative.size > 1 else 0.0
    mean_relative = float(np.mean(relative))
    return bias, (mean_relative - 1.96 * spread, mean_relative + 1.96 * spread)
