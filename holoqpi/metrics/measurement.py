"""Agreement between predicted and reference cellular measurements.

This is the metric family the professor's brief is ultimately about: whether a
reconstruction is *measurement-ready*. Four views are accumulated.

* per cell   -- cells paired between prediction and reference by IoU, giving
                relative errors on area, optical volume and dry mass
* per image  -- the median cell of each field of view, giving the Pearson
                agreement and Bland-Altman bias that a reviewer expects for a
                quantitative instrument
* coverage-adjusted -- the same per-cell error with the cells the model never
                found counted in, because the matched-cell error alone rewards
                a detector for being selective
* field-bootstrapped -- an interval on the per-cell statistics obtained by
                resampling FIELDS, because cells within a field are not
                independent samples

WHY THE LAST TWO ARE NOT OPTIONAL EXTRAS
----------------------------------------
COVERAGE. Every per-cell error here is computed over IoU-matched pairs. A model
that detects the easiest 40% of cells and measures those perfectly reports an
excellent MAPE, and a model that detects everything and measures it well
reports a worse one. Ranking on the matched-cell error therefore rewards
missing cells. The coverage-adjusted variant closes that hole by counting a
missed reference cell as an error of 1.0 -- its whole mass went unreported --
so the number describes the population rather than a self-selected subset. It
is reported ALONGSIDE the matched error and the recall, never instead of them,
because "not measured" and "measured as zero" are different statements and the
convention has to be visible.

INDEPENDENCE. Cells in one field share an illumination, a focus, an aberration
surface and a segmentation threshold, so their errors are correlated. A
confidence interval built from n cells treated as independent is too narrow by
whatever that correlation is worth, and with roughly thirty cells per field the
factor is not small. Resampling FIELDS with replacement respects the structure
and is the honest primary interval; the cell count is still reported so the
reader can see how few fields it rests on.
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

    def __init__(
        self,
        report_bland_altman: bool = True,
        report_coverage_adjusted: bool = True,
        field_bootstrap_resamples: int = 0,
        bootstrap_seed: int = 0,
    ):
        self.report_bland_altman = report_bland_altman
        self.report_coverage_adjusted = report_coverage_adjusted
        self.field_bootstrap_resamples = int(field_bootstrap_resamples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.reset()

    def reset(self) -> None:
        # Pairs are grouped BY FIELD rather than pooled, because the bootstrap
        # resamples fields. ``_pending`` collects the current field's pairs
        # between ``update_pairs`` and ``update_image``, which the evaluator
        # always calls in that order; ``compute`` flushes anything left over so
        # a caller that only ever calls ``update_pairs`` still gets its data.
        self._fields: list[dict[str, list[tuple[float, float]]]] = []
        self._pending: dict[str, list[tuple[float, float]]] = {q: [] for q in _QUANTITIES}
        self._per_image: dict[str, list[tuple[float, float]]] = {q: [] for q in _QUANTITIES}
        self._predicted_counts: list[int] = []
        self._reference_counts: list[int] = []
        self._matched_counts: list[int] = []
        self._field_totals: list[dict[str, float]] = []
        self._missed: list[dict] = []
        self._false_positives: list[dict] = []

    def update_pairs(self, pairs: list[tuple[dict, dict]]) -> None:
        for predicted, reference in pairs:
            for quantity in _QUANTITIES:
                p, r = predicted.get(quantity), reference.get(quantity)
                if p is None or r is None or not np.isfinite(p) or not np.isfinite(r):
                    continue
                self._pending[quantity].append((float(p), float(r)))

    def _close_field(self) -> None:
        if any(self._pending[q] for q in _QUANTITIES):
            self._fields.append({q: list(self._pending[q]) for q in _QUANTITIES})
        self._pending = {q: [] for q in _QUANTITIES}

    @property
    def _paired(self) -> dict[str, list[tuple[float, float]]]:
        """Every pair, pooled across fields. Derived, never the storage."""
        pooled: dict[str, list[tuple[float, float]]] = {q: [] for q in _QUANTITIES}
        for field in self._fields:
            for quantity in _QUANTITIES:
                pooled[quantity].extend(field[quantity])
        for quantity in _QUANTITIES:
            pooled[quantity].extend(self._pending[quantity])
        return pooled

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
        self._close_field()

        # Field totals over ALL cells on each side, not only the matched ones.
        # This is the one measurement view that needs no pairing at all: the
        # total dry mass on a field is what a storage-lesion or drug-response
        # study integrates, and a missed cell lowers it while a false positive
        # raises it. So it includes the detection failures by construction
        # instead of excluding them the way every paired metric must.
        self._field_totals.append({
            f"{_SHORT_NAME[quantity]}_predicted": _total(predicted_cells, quantity)
            for quantity in ("area_um2", "optical_volume_rad_um2", "dry_mass_pg")
        } | {
            f"{_SHORT_NAME[quantity]}_reference": _total(reference_cells, quantity)
            for quantity in ("area_um2", "optical_volume_rad_um2", "dry_mass_pg")
        })

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
        self._close_field()
        pooled = self._paired

        for quantity in _QUANTITIES:
            short = _SHORT_NAME[quantity]

            pairs = pooled[quantity]
            if pairs:
                predicted = np.array([p for p, _ in pairs])
                reference = np.array([r for _, r in pairs])
                results[f"{short}_mape"] = _mape(predicted, reference)
                results[f"{short}_mae"] = float(np.abs(predicted - reference).mean())
                results[f"{short}_cell_pearson_r"] = _pearson(predicted, reference)
                results[f"{short}_n_cells"] = int(predicted.size)

                # Interval from resampling FIELDS, not cells. Reported for the
                # two statistics a result would actually be ranked on.
                if self.field_bootstrap_resamples > 0 and len(self._fields) > 1:
                    interval = self._bootstrap(quantity)
                    for name, value in interval.items():
                        results[f"{short}_{name}"] = value

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

            # ---- coverage-adjusted per-cell error ----------------------
            # A missed reference cell contributes a relative error of exactly
            # 1.0: nothing was reported for it, so the whole of its area or
            # mass is unaccounted for. Mixing that with the matched-cell error
            # in proportion to recall gives an error over the FULL reference
            # population, which is the quantity a reader assumes they are being
            # shown. Note it is an upper bound on a missed cell's contribution,
            # not a measurement of one, and it says nothing about false
            # positives -- those inflate a total rather than a per-cell error,
            # and the field totals below are where they show up.
            if self.report_coverage_adjusted and np.isfinite(recall):
                results["coverage"] = recall
                for quantity in _QUANTITIES:
                    short = _SHORT_NAME[quantity]
                    matched = results.get(f"{short}_mape")
                    if matched is None or not np.isfinite(matched):
                        continue
                    results[f"{short}_mape_coverage_adjusted"] = (
                        recall * matched + (1.0 - recall) * 1.0
                    )

        # ---- field totals, which need no pairing -----------------------
        if self._field_totals:
            for quantity in ("area_um2", "optical_volume_rad_um2", "dry_mass_pg"):
                short = _SHORT_NAME[quantity]
                predicted = np.array(
                    [f[f"{short}_predicted"] for f in self._field_totals], dtype=float
                )
                reference = np.array(
                    [f[f"{short}_reference"] for f in self._field_totals], dtype=float
                )
                usable = np.isfinite(predicted) & np.isfinite(reference) & (reference > 0)
                if not usable.any():
                    continue
                relative = (predicted[usable] - reference[usable]) / reference[usable]
                results[f"{short}_field_total_bias"] = float(np.mean(relative))
                results[f"{short}_field_total_mape"] = float(np.mean(np.abs(relative)))
                results[f"{short}_field_total_pearson_r"] = _pearson(
                    predicted[usable], reference[usable]
                )
                results[f"{short}_n_fields"] = int(usable.sum())

        return results

    def _bootstrap(self, quantity: str) -> dict:
        """Percentile interval on the per-cell statistics, resampling fields.

        Fields are drawn with replacement and their cells concatenated, so a
        field that happens to be drawn twice contributes all of its cells
        twice. That is the point: it reproduces the sampling unit of the study,
        which is the field of view and not the cell.
        """
        available = [field for field in self._fields if field[quantity]]
        if len(available) < 2:
            return {}
        generator = np.random.default_rng(self.bootstrap_seed)
        mapes, correlations = [], []
        for _ in range(self.field_bootstrap_resamples):
            picks = generator.integers(0, len(available), size=len(available))
            pairs: list[tuple[float, float]] = []
            for index in picks:
                pairs.extend(available[index][quantity])
            if len(pairs) < 2:
                continue
            predicted = np.array([p for p, _ in pairs])
            reference = np.array([r for _, r in pairs])
            mapes.append(_mape(predicted, reference))
            correlations.append(_pearson(predicted, reference))

        mapes = np.array([v for v in mapes if np.isfinite(v)])
        correlations = np.array([v for v in correlations if np.isfinite(v)])
        interval: dict = {"bootstrap_fields": len(available),
                          "bootstrap_resamples": self.field_bootstrap_resamples}
        if mapes.size:
            interval["mape_ci_lower"] = float(np.percentile(mapes, 2.5))
            interval["mape_ci_upper"] = float(np.percentile(mapes, 97.5))
        if correlations.size:
            interval["cell_pearson_r_ci_lower"] = float(np.percentile(correlations, 2.5))
            interval["cell_pearson_r_ci_upper"] = float(np.percentile(correlations, 97.5))
        return interval

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


def _total(cells: list[dict], quantity: str) -> float:
    """Sum over every cell on one side of one field. Zero when there are none.

    Zero rather than NaN, because a field where the model found nothing has a
    predicted total of zero and that is a real, reportable failure -- turning it
    into a missing value would quietly drop the worst fields from the result.
    """
    values = [c[quantity] for c in cells if quantity in c and np.isfinite(c[quantity])]
    return float(np.sum(values)) if values else 0.0


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
