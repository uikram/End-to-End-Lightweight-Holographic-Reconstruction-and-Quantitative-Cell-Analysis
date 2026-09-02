"""Quantitative phase reconstruction accuracy.

Errors are reported in radians rather than as normalised image-quality scores,
because the downstream measurement inherits them in physical units: a phase bias
of delta radians integrated over a cell propagates directly into its dry mass.
"""

from __future__ import annotations

import numpy as np


class PhaseMetrics:
    """Accumulates phase-reconstruction statistics over a split."""

    def __init__(self, data_range: float):
        self.data_range = data_range
        self.reset()

    def reset(self) -> None:
        self._absolute_error: list[float] = []
        self._squared_error: list[float] = []
        self._psnr: list[float] = []
        self._ssim: list[float] = []
        self._pearson: list[float] = []
        self._bias: list[float] = []
        self._n = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """``prediction`` and ``target`` are (B, H, W) arrays in radians."""
        for predicted, actual in zip(np.atleast_3d(prediction), np.atleast_3d(target)):
            difference = predicted - actual
            mae = float(np.abs(difference).mean())
            mse = float((difference ** 2).mean())

            self._absolute_error.append(mae)
            self._squared_error.append(mse)
            self._bias.append(float(difference.mean()))
            self._psnr.append(
                float(10.0 * np.log10((self.data_range ** 2) / mse)) if mse > 0 else float("inf")
            )
            self._ssim.append(_ssim(predicted, actual, self.data_range))
            self._pearson.append(_pearson(predicted.ravel(), actual.ravel()))
            self._n += 1

    def compute(self) -> dict:
        if self._n == 0:
            return {}
        finite_psnr = [v for v in self._psnr if np.isfinite(v)]
        return {
            "phase_mae_rad": float(np.mean(self._absolute_error)),
            "phase_rmse_rad": float(np.sqrt(np.mean(self._squared_error))),
            "phase_bias_rad": float(np.mean(self._bias)),
            "phase_psnr_db": float(np.mean(finite_psnr)) if finite_psnr else float("nan"),
            "phase_ssim": float(np.mean(self._ssim)),
            "phase_pearson_r": float(np.mean(self._pearson)),
            "phase_n_images": self._n,
        }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def _ssim(prediction: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """Global SSIM with a uniform 7x7 window (scikit-image free)."""
    from scipy.ndimage import uniform_filter

    window = 7
    mu_p = uniform_filter(prediction, window)
    mu_t = uniform_filter(target, window)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p = uniform_filter(prediction * prediction, window) - mu_p_sq
    sigma_t = uniform_filter(target * target, window) - mu_t_sq
    sigma_pt = uniform_filter(prediction * target, window) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    )
    return float(ssim_map.mean())
