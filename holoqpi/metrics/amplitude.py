"""Agreement between the predicted transmitted amplitude and its reference.

WHAT THIS IS NOT. The reference is the modulus of a classical off-axis
reconstruction written by ``scripts/prepare_amplitude.py``, normalised so the
mask background reads 1. It is not a measurement: it carries the sideband
filter's lost high frequencies, residual twin-image structure and any
illumination vignetting. Every number produced here is therefore *agreement
with one reconstruction algorithm's output*, and must be reported in those
words. That is the same caveat the D0 config states for the amplitude loss, and
it applies with equal force to a metric.

WHY IT EXISTS ANYWAY. The framework's stated output is phase **and amplitude**
and segmentation. Before this module the predicted amplitude entered exactly
one number -- the forward-model residual -- where it is entangled with the
phase, the propagation distance and the aberration surface, so an amplitude
head could have been producing a constant field without any table saying so.
The first two rows below, ``amplitude_mae`` against ``amplitude_unity_mae``,
are what settle that: if the head has learned nothing it predicts 1 everywhere
and the two are equal.

The unity comparator is the honest baseline here, not zero. A = 1 everywhere is
the thin-phase-object assumption the rest of the study runs on, so beating it is
the whole claim the amplitude head has to support.
"""

from __future__ import annotations

import numpy as np

from .phase import _as_batch, _pearson


class AmplitudeMetrics:
    """Accumulates predicted-vs-reference amplitude statistics over a split."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._absolute_error: list[float] = []
        self._squared_error: list[float] = []
        self._bias: list[float] = []
        self._pearson: list[float] = []
        self._unity_absolute_error: list[float] = []
        self._in_cell_mae: list[float] = []
        self._in_cell_bias: list[float] = []
        self._prediction_mean: list[float] = []
        self._prediction_sd: list[float] = []
        self._reference_mean: list[float] = []
        self._n = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """``prediction``, ``target`` and optional ``mask`` are (B, H, W) arrays.

        The in-cell split matters for the same reason it does for the phase: the
        specimen is a fifth of the field, so a head that predicts the background
        modulus perfectly and the cells not at all still scores well field-wide.
        """
        prediction = _as_batch(prediction)
        target = _as_batch(target)
        masks = _as_batch(mask) if mask is not None else [None] * len(prediction)

        for predicted, actual, cell_mask in zip(prediction, target, masks):
            difference = predicted - actual

            if cell_mask is not None:
                inside = cell_mask > 0
                if inside.any():
                    self._in_cell_mae.append(float(np.abs(difference[inside]).mean()))
                    self._in_cell_bias.append(float(difference[inside].mean()))

            self._absolute_error.append(float(np.abs(difference).mean()))
            self._squared_error.append(float((difference ** 2).mean()))
            self._bias.append(float(difference.mean()))
            self._pearson.append(_pearson(predicted.ravel(), actual.ravel()))
            self._unity_absolute_error.append(float(np.abs(1.0 - actual).mean()))
            # The spread of the prediction, reported because a head that has
            # collapsed to a constant is the failure mode this metric exists to
            # expose, and a near-zero sd says so more directly than an MAE.
            self._prediction_mean.append(float(predicted.mean()))
            self._prediction_sd.append(float(predicted.std()))
            self._reference_mean.append(float(actual.mean()))
            self._n += 1

    def compute(self) -> dict:
        if self._n == 0:
            return {}
        results = {
            "amplitude_mae": float(np.mean(self._absolute_error)),
            "amplitude_rmse": float(np.sqrt(np.mean(self._squared_error))),
            "amplitude_bias": float(np.mean(self._bias)),
            "amplitude_pearson_r": float(np.mean(self._pearson)),
            "amplitude_unity_mae": float(np.mean(self._unity_absolute_error)),
            "amplitude_pred_mean": float(np.mean(self._prediction_mean)),
            "amplitude_pred_sd": float(np.mean(self._prediction_sd)),
            "amplitude_reference_mean": float(np.mean(self._reference_mean)),
            "amplitude_n_images": self._n,
        }
        # The ratio is the row to read: below 1 the head is closer to the
        # reference than the thin-phase assumption is, at or above 1 it is not,
        # and no amplitude claim survives the second case.
        unity = results["amplitude_unity_mae"]
        results["amplitude_mae_over_unity"] = (
            results["amplitude_mae"] / unity if unity > 0 else float("nan")
        )
        if self._in_cell_mae:
            results["amplitude_mae_in_cell"] = float(np.mean(self._in_cell_mae))
            results["amplitude_bias_in_cell"] = float(np.mean(self._in_cell_bias))
        return results
