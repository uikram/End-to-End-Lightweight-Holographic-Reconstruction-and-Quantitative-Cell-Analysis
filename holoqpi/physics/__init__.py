"""Scalar diffraction physics shared by the loss, the baselines and calibration."""

from .propagation import (
    angular_spectrum_kernel,
    diffraction_radius_px,
    estimate_carrier,
    form_hologram,
<<<<<<< Updated upstream
=======
    max_reflect_pad,
    pad_reflect,
>>>>>>> Stashed changes
    propagate,
    reconstruct_gabor,
    reconstruct_off_axis,
    reference_wave,
<<<<<<< Updated upstream
=======
    unpad,
>>>>>>> Stashed changes
    unwrap_phase_2d,
)

__all__ = [
    "angular_spectrum_kernel",
    "diffraction_radius_px",
    "estimate_carrier",
    "form_hologram",
<<<<<<< Updated upstream
=======
    "max_reflect_pad",
    "pad_reflect",
>>>>>>> Stashed changes
    "propagate",
    "reconstruct_gabor",
    "reconstruct_off_axis",
    "reference_wave",
<<<<<<< Updated upstream
=======
    "unpad",
>>>>>>> Stashed changes
    "unwrap_phase_2d",
]
