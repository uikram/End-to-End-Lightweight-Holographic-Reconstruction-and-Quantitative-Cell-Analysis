"""Scalar diffraction physics shared by the loss, the baselines and calibration."""

from .propagation import (
    angular_spectrum_kernel,
    diffraction_radius_px,
    estimate_carrier,
    form_hologram,
    max_reflect_pad,
    pad_reflect,
    propagate,
    reconstruct_gabor,
    reconstruct_off_axis,
    reference_wave,
    unpad,
    unwrap_phase_2d,
)
from .surface import (
    detrend_polynomial,
    fit_polynomial_surface,
    phase_skewness,
    polynomial_basis,
    resolve_conjugate,
)

__all__ = [
    "angular_spectrum_kernel",
    "detrend_polynomial",
    "diffraction_radius_px",
    "estimate_carrier",
    "fit_polynomial_surface",
    "form_hologram",
    "max_reflect_pad",
    "pad_reflect",
    "phase_skewness",
    "polynomial_basis",
    "propagate",
    "reconstruct_gabor",
    "reconstruct_off_axis",
    "reference_wave",
    "resolve_conjugate",
    "unpad",
    "unwrap_phase_2d",
]
