"""Individual terms of the joint objective.

Two families are represented. The supervised terms compare each head against its
own target. The physics terms couple the heads to one another and to the
measurements the study exists to produce: they are the mechanism by which the
network is pushed to emit a *measurement-ready* phase map rather than a
visually plausible one.

Unless stated otherwise, each term returns a per-sample tensor of shape (B,) so
the caller can mask out fields of view that contain no cells.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Phase reconstruction
# ---------------------------------------------------------------------------
def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.outer(kernel).view(1, 1, size, size)


def structural_similarity(
    prediction: torch.Tensor, target: torch.Tensor,
    window: int, sigma: float, data_range: float,
) -> torch.Tensor:
    """Per-sample mean SSIM; implemented locally to avoid an extra dependency."""
    kernel = _gaussian_window(window, sigma, prediction.device, prediction.dtype)
    pad = window // 2

    mu_p = F.conv2d(prediction, kernel, padding=pad)
    mu_t = F.conv2d(target, kernel, padding=pad)
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p = F.conv2d(prediction * prediction, kernel, padding=pad) - mu_p_sq
    sigma_t = F.conv2d(target * target, kernel, padding=pad) - mu_t_sq
    sigma_pt = F.conv2d(prediction * target, kernel, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    )
    return ssim_map.mean(dim=[1, 2, 3])


def spatial_gradient(x: torch.Tensor) -> torch.Tensor:
    """Forward-difference gradient magnitude, zero-padded to preserve shape."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)


class PhaseReconstructionLoss(nn.Module):
    """L1 + gradient + (1 - SSIM) on the predicted quantitative phase."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_l1 = cfg.l1
        self.w_gradient = cfg.gradient
        self.w_ssim = cfg.ssim
        self.window = cfg.ssim_window
        self.sigma = cfg.ssim_sigma
        self.data_range = cfg.ssim_data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_l1 * (prediction - target).abs().mean(dim=[1, 2, 3])

        if self.w_gradient > 0:
            gradient_error = (spatial_gradient(prediction) - spatial_gradient(target)).abs()
            loss = loss + self.w_gradient * gradient_error.mean(dim=[1, 2, 3])

        if self.w_ssim > 0:
            ssim = structural_similarity(
                prediction, target, self.window, self.sigma, self.data_range
            )
            loss = loss + self.w_ssim * (1.0 - ssim)

        return loss


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
class SegmentationLoss(nn.Module):
    """Class-weighted Dice plus cross-entropy over the segmentation classes."""

    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.w_dice = cfg.dice
        self.w_ce = cfg.cross_entropy
        self.smooth = cfg.dice_smooth

        weights = torch.tensor(list(cfg.class_weights), dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(
                f"loss.segmentation.class_weights has {weights.numel()} entries "
                f"but the model predicts {num_classes} classes"
            )
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)

        if self.w_ce > 0:
            cross_entropy = F.cross_entropy(
                logits, target, weight=self.class_weights.to(logits.dtype), reduction="none"
            )
            loss = loss + self.w_ce * cross_entropy.mean(dim=[1, 2])

        if self.w_dice > 0:
            probabilities = torch.softmax(logits, dim=1)
            dice_total = torch.zeros_like(loss)
            weight_total = 0.0
            for class_index in range(self.num_classes):
                predicted = probabilities[:, class_index]
                actual = (target == class_index).to(probabilities.dtype)
                intersection = (predicted * actual).sum(dim=[1, 2])
                union = predicted.sum(dim=[1, 2]) + actual.sum(dim=[1, 2])
                class_dice = 1.0 - (2 * intersection + self.smooth) / (union + self.smooth)
                weight = float(self.class_weights[class_index])
                dice_total = dice_total + weight * class_dice
                weight_total += weight
            loss = loss + self.w_dice * dice_total / max(weight_total, 1e-8)

        return loss


# ---------------------------------------------------------------------------
# Physics-aware coupling terms
# ---------------------------------------------------------------------------
class PhaseMaskContrast(nn.Module):
    """The segmented interior must carry more optical path than its surround."""

    def __init__(self, margin: float, collapse_warn_ratio: float):
        super().__init__()
        self.margin = margin
        self.collapse_warn_ratio = collapse_warn_ratio
        self._warnings_emitted = 0

    def forward(self, foreground: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        foreground = foreground.squeeze(1)
        phase = phase.squeeze(1)

        cell_area = foreground.sum(dim=[1, 2]).clamp(min=1.0)
        background_area = (1.0 - foreground).sum(dim=[1, 2]).clamp(min=1.0)

        mean_cell = (foreground * phase).sum(dim=[1, 2]) / cell_area
        mean_background = ((1.0 - foreground) * phase).sum(dim=[1, 2]) / background_area

        if self.training and self._warnings_emitted < 3:
            ratio = float(foreground.mean().detach())
            if ratio > self.collapse_warn_ratio:
                self._warnings_emitted += 1
                print(
                    f"[PhaseMaskContrast] foreground fraction {ratio:.1%} exceeds "
                    f"{self.collapse_warn_ratio:.0%}: the mask may be collapsing to "
                    f"all-cell ({self._warnings_emitted}/3)"
                )

        return F.relu(mean_background - mean_cell + self.margin)


class BoundaryGradientAlignment(nn.Module):
    """Drive the mask edge onto the ridge of steepest optical path change.

    Both gradient fields are max-normalised per sample first: a probability map
    and a phase map in radians are otherwise on incomparable scales.
    """

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, foreground: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        mask_gradient = spatial_gradient(foreground)
        phase_gradient = spatial_gradient(phase)

        mask_gradient = mask_gradient / (
            mask_gradient.amax(dim=[1, 2, 3], keepdim=True) + self.epsilon
        )
        phase_gradient = phase_gradient / (
            phase_gradient.amax(dim=[1, 2, 3], keepdim=True) + self.epsilon
        )
        return (mask_gradient - phase_gradient).abs().mean(dim=[1, 2, 3])


class PhaseVolumePreservation(nn.Module):
    """Conserve the phase integral enclosed by the predicted boundary.

    Formulated as a relative error, both because dry mass inherits exactly this
    relative error and because an absolute integral over a full field of view
    reaches magnitudes that destabilise mixed-precision training.
    """

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        foreground: torch.Tensor,
        phase: torch.Tensor,
        target_foreground: torch.Tensor,
        target_phase: torch.Tensor,
    ) -> torch.Tensor:
        predicted = (foreground.squeeze(1) * phase.squeeze(1)).sum(dim=[1, 2])
        reference = (target_foreground.squeeze(1) * target_phase.squeeze(1)).sum(dim=[1, 2])
        return (predicted - reference).abs() / (reference.abs() + self.epsilon)


class DryMassConsistency(nn.Module):
    """End-to-end measurement agreement, from raw hologram to picograms.

    Dry mass is a fixed scalar multiple of the enclosed phase integral, so the
    calibration constant cancels in this relative form. The term is kept
    separate from phase-volume preservation because it is evaluated against the
    *reference* mask and phase jointly, closing the loop on the quantity the
    study reports rather than on either head alone.
    """

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        foreground: torch.Tensor,
        phase: torch.Tensor,
        target_foreground: torch.Tensor,
        target_phase: torch.Tensor,
    ) -> torch.Tensor:
        predicted = (foreground.squeeze(1) * phase.squeeze(1)).sum(dim=[1, 2])
        reference = (target_foreground.squeeze(1) * target_phase.squeeze(1)).sum(dim=[1, 2])
        return F.smooth_l1_loss(
            predicted / (reference.abs() + self.epsilon),
            torch.ones_like(predicted),
            reduction="none",
            beta=1.0,
        )


class ProjectedAreaConsistency(nn.Module):
    """Keep the segmented footprint calibrated in absolute pixel count."""

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, foreground: torch.Tensor, target_foreground: torch.Tensor) -> torch.Tensor:
        predicted = foreground.squeeze(1).sum(dim=[1, 2])
        reference = target_foreground.squeeze(1).sum(dim=[1, 2])
        return (predicted - reference).abs() / (reference + self.epsilon)


class ForwardModelConsistency(nn.Module):
    """Data fidelity against the hologram that was actually recorded.

    This is the constraint the reference literature means by physics
<<<<<<< Updated upstream
    consistency, and the one the rest of this objective was missing. Every other
    term relates the network's two outputs to each other or to their targets;
    this one propagates the predicted field back to the sensor and asks whether
    it could have produced the measurement:

        L = || standardise( |P_z(A_hat exp(i phi_hat))|-formed hologram )
              - standardise(I_measured) ||

    The raw hologram enters the network as input and, without this term, never
    appears in the loss again. Adding it introduces evidence no other term
    exploits, which is precisely why it is not redundant with the phase
    reconstruction loss the way the mask-phase coupling terms are.

    Three details matter for it to mean anything.

    STANDARDISATION.  Illumination brightness, camera gain and exposure differ
    between a synthesised and a recorded hologram for reasons unrelated to the
    field. Both sides are reduced to zero mean and unit variance so the residual
    measures structure, not scale.

    BORDER EXCLUSION.  The FFT treats the array as periodic, so on a training
    crop the light that should have arrived from outside instead wraps around
    from the opposite edge. The field is reflection-padded before propagation and
    a margin of the same width is dropped from the residual.

    GEOMETRY.  In-line and off-axis form intensity differently, and the
    difference is physical rather than a convention: at zero defocus a pure phase
    object produces no in-line intensity contrast at all, so this term has no
    gradient for the Gabor arm unless z is large enough. Off-axis carries the
    phase in the fringe modulation and stays informative at any distance. Run
    ``scripts/calibrate_z.py`` to see the sensitivity of both arms before
    trusting this term.
=======
    consistency (Huang et al., Nat. Mach. Intell. 2023; Galande et al.,
    J. Biomed. Opt.; Lee et al., APL Mach. Learn. 4, 026106), and the one the
    rest of this objective was missing. Every other term relates the network's
    two outputs to each other or to their targets; this one propagates the
    predicted field to the sensor and asks whether it could have produced the
    measurement.

    THE RADIOMETRIC MODEL
    ---------------------
    A sensor does not record |U|^2. It records

        I_measured  =  gain * (physical intensity)  +  offset  +  noise

    with gain set by illumination power, exposure and camera response, and
    offset by the black level and stray light. None of those are known, and none
    of them carry information about the phase. Comparing a synthesised intensity
    with a measured one directly would therefore be dominated by three unknown
    scalars.

    They are removed the honest way: by *fitting* them, per image, in closed
    form, as part of the observation model. Writing the off-axis intensity in
    its three physical components,

        I  =  |R|^2  +  |U|^2  +  2 Re(R* U)
           =  c0 . 1  +  c1 . |U|^2  +  c2 . Re(R* U)

    the unknown reference power, object gain and reference-to-object amplitude
    ratio are exactly the three coefficients c0, c1, c2. Solving the 3 x 1
    least-squares problem against the measured intensity marginalises all of
    them out at once, and what remains in the residual is structure the phase
    must explain. For in-line the reference is not separate, so the model is
    two-parameter, I = c0 + c1 |U|^2.

    This replaces an earlier z-score of both sides. Standardising is the same
    two-parameter fit written implicitly, but it hides the calibration inside a
    preprocessing step instead of exposing it, cannot absorb the off-axis
    reference ratio at all, and leaves no way to check whether the fitted gain
    is physically sensible. The coefficients are returned so they can be logged:
    a gain that drifts or changes sign is a modelling error announcing itself.

    THE MEASUREMENT IT COMPARES AGAINST
    -----------------------------------
    ``hologram`` must be the RAW intensity, not the normalised network input.
    The dataset carries both (``hologram_raw`` and ``hologram``) precisely so
    that the network can have a well-conditioned input without the physics term
    being handed a redefined observation model.

    OTHER DETAILS THAT MATTER
    -------------------------
    BORDER EXCLUSION. The FFT treats the array as periodic, so on a training
    crop the light that should have arrived from outside instead wraps around
    from the opposite edge. The field is reflection-padded before propagation
    and a margin of the same width is dropped from the residual.

    GEOMETRY. In-line and off-axis form intensity differently, and the
    difference is physical rather than conventional: at zero defocus a pure
    phase object produces no in-line intensity contrast at all, so this term has
    no gradient for the Gabor arm unless z is large enough. Off-axis carries the
    phase in the fringe modulation and stays informative at any distance. Run
    ``scripts/calibrate_z.py`` to see the sensitivity of both arms before
    trusting this term.

    DISTANCE. ``learn_distance`` makes z an ``nn.Parameter`` initialised at
    ``distance_um``, so it can be refined by gradient descent when the grid
    search leaves it only approximately determined. The propagation kernel is
    differentiable in z, so this is a real refinement and not a placeholder.
>>>>>>> Stashed changes
    """

    def __init__(self, cfg, optics):
        super().__init__()
        self.wavelength_um = float(optics.wavelength_um)
        self.pitch_x_um = float(optics.pixel_pitch_x_um)
        self.pitch_y_um = float(optics.pixel_pitch_y_um)

<<<<<<< Updated upstream
        self.distance_um = cfg.distance_um
        self.pad_px = cfg.pad_px
        self.border_px = cfg.border_px
        self.feature_um = float(cfg.feature_um)
        self.reference_ratio = float(cfg.reference_ratio)
        self.dc_exclusion_px = int(cfg.dc_exclusion_px)
        self.criterion = cfg.criterion
        if self.criterion not in ("l1", "l2", "correlation"):
            raise ValueError(f"unknown forward-model criterion {self.criterion!r}")
        self._warned = False

    # -- helpers ----------------------------------------------------------
=======
        self.pad_px = cfg.pad_px
        self.border_px = cfg.border_px
        self.feature_um = float(cfg.feature_um)
        self.dc_exclusion_px = cfg.dc_exclusion_px
        self.dc_exclusion_frac = float(cfg.dc_exclusion_frac)
        self.fit_radiometry = bool(cfg.fit_radiometry)
        self.criterion = cfg.criterion
        if self.criterion not in ("l1", "l2", "correlation"):
            raise ValueError(f"unknown forward-model criterion {self.criterion!r}")

        self.learn_distance = bool(cfg.learn_distance)
        distance = cfg.distance_um
        if distance is None:
            self.distance = None
        elif self.learn_distance:
            self.distance = nn.Parameter(torch.tensor(float(distance)))
        else:
            self.register_buffer("distance", torch.tensor(float(distance)))
        self._warned = False

    # -- helpers ----------------------------------------------------------
    @property
    def distance_um(self) -> float | None:
        return None if self.distance is None else float(self.distance.detach())

>>>>>>> Stashed changes
    def required_pad(self) -> int:
        """Diffraction spread over z, in pixels: the context the field needs."""
        if self.pad_px is not None:
            return int(self.pad_px)
<<<<<<< Updated upstream
        if not self.distance_um:
            return 0
        spread_um = abs(self.wavelength_um * float(self.distance_um)) / self.feature_um
=======
        distance = self.distance_um
        if not distance:
            return 0
        spread_um = abs(self.wavelength_um * distance) / self.feature_um
>>>>>>> Stashed changes
        return int(math.ceil(spread_um / min(self.pitch_x_um, self.pitch_y_um)))

    def _pad_for(self, size: int) -> int:
        """Pad actually applied, with a one-time warning when it falls short.

        Light scattered inside the crop travels lambda*z/feature micrometres
        sideways before reaching the sensor, and light from outside travels the
        same distance inward. Neither is available on a crop smaller than that
        spread, so the synthesised hologram is then missing real contributions
        and the residual partly measures the missing context rather than the
        reconstruction. Padding is capped at half the array to keep the FFT
        affordable; the shortfall is reported once, so it is a known
        approximation rather than a silent one.
        """
        required = self.required_pad()
        affordable = max(0, size // 2 - 1)
        applied = min(required, affordable)
        if required > applied and not self._warned:
            self._warned = True
            LOGGER.warning(
                "forward model at z = %.1f um needs %d px of diffraction context but a "
                "%d px field allows only %d. The residual is approximate on this crop. "
                "Train on larger crops (data.train_crop), lower "
                "loss.forward_model.distance_um, or set loss.forward_model.pad_px "
                "explicitly to accept the approximation deliberately.",
<<<<<<< Updated upstream
                float(self.distance_um), required, size, applied,
=======
                self.distance_um, required, size, applied,
>>>>>>> Stashed changes
            )
        return applied

    @staticmethod
<<<<<<< Updated upstream
    def _standardise(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        mean = flat.mean(dim=1).view(-1, 1, 1, 1)
        std = flat.std(dim=1).view(-1, 1, 1, 1).clamp(min=1e-6)
        return (x - mean) / std
=======
    def _fit_components(components: torch.Tensor, measured: torch.Tensor):
        """Least-squares fit of the radiometric coefficients, per image.

        ``components`` is (B, K, N) and ``measured`` is (B, N). Returns the
        fitted prediction (B, N) and the coefficients (B, K). Solved in closed
        form with a small ridge term, so it is differentiable and cannot blow up
        when two components are nearly collinear (which happens when |U|^2 is
        almost constant, i.e. exactly at the in-line degenerate case).
        """
        gram = components @ components.transpose(1, 2)
        rhs = (components * measured.unsqueeze(1)).sum(dim=2, keepdim=True)
        ridge = 1e-6 * torch.diag_embed(
            torch.diagonal(gram, dim1=1, dim2=2).clamp(min=1e-12)
        )
        coefficients = torch.linalg.solve(gram + ridge, rhs)
        fitted = (coefficients.transpose(1, 2) @ components).squeeze(1)
        return fitted, coefficients.squeeze(2)
>>>>>>> Stashed changes

    def forward(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor,
        hologram: torch.Tensor,
        modality: str,
<<<<<<< Updated upstream
    ) -> torch.Tensor:
        from ..physics import estimate_carrier, form_hologram

        if self.distance_um is None:
=======
        return_coefficients: bool = False,
    ) -> torch.Tensor:
        from ..physics import (
            estimate_carrier, pad_reflect, propagate, reference_wave, unpad,
        )

        if self.distance is None:
>>>>>>> Stashed changes
            raise ValueError(
                "loss.forward_model.distance_um is null. Run scripts/calibrate_z.py "
                "or obtain the acquisition distance before enabling this term."
            )

        pad = self._pad_for(min(phase.shape[-2], phase.shape[-1]))
<<<<<<< Updated upstream
        carrier = (
            estimate_carrier(hologram, self.dc_exclusion_px)
            if modality == "off_axis" else None
        )

        synthetic = form_hologram(
            phase, amplitude, modality,
            self.wavelength_um, self.pitch_x_um, self.pitch_y_um,
            float(self.distance_um), carrier=carrier,
            reference_ratio=self.reference_ratio, pad=pad,
        )

        border = int(self.border_px) if self.border_px is not None else pad
        if border > 0 and min(synthetic.shape[-2:]) > 2 * border + 8:
            synthetic = synthetic[..., border:-border, border:-border]
            hologram = hologram[..., border:-border, border:-border]

        predicted = self._standardise(synthetic)
        measured = self._standardise(hologram)

        if self.criterion == "l1":
            return (predicted - measured).abs().flatten(1).mean(dim=1)
        if self.criterion == "l2":
            return ((predicted - measured) ** 2).flatten(1).mean(dim=1)
        # correlation: 1 - r, bounded and insensitive to residual scale error
        return 1.0 - (predicted * measured).flatten(1).mean(dim=1)
=======

        field = torch.polar(amplitude.clamp(min=0.0).float(), phase.float())
        field = pad_reflect(field, pad)
        propagated = propagate(
            field, self.wavelength_um, self.pitch_x_um, self.pitch_y_um, self.distance
        )
        propagated = unpad(propagated, pad)

        # Physical components of the recorded intensity, in the order of the
        # interference expansion. Their coefficients are the unknown
        # radiometric constants, fitted below rather than assumed.
        object_intensity = propagated.abs() ** 2
        parts = [torch.ones_like(object_intensity), object_intensity]
        if modality == "off_axis":
            carrier_y, carrier_x = estimate_carrier(
                hologram, self.dc_exclusion_px, self.dc_exclusion_frac
            )
            reference = reference_wave(
                propagated.shape[-2], propagated.shape[-1], carrier_y, carrier_x,
                device=propagated.device, dtype=torch.float32,
            )
            # BOTH conjugate cross-terms enter as separate fitted components.
            # The two first-order sidebands have equal magnitude, so any rule
            # for picking one is a coin flip that flips the sign of the fringe
            # term and destroys the residual. Carrying both and letting the
            # least-squares fit weight them removes the ambiguity entirely
            # rather than resolving it by guess.
            parts.append(2.0 * (reference.conj() * propagated).real)
            parts.append(2.0 * (reference * propagated).real)
        elif modality != "gabor":
            raise ValueError(f"unknown modality {modality!r}")

        border = int(self.border_px) if self.border_px is not None else pad
        if border > 0 and min(object_intensity.shape[-2:]) > 2 * border + 8:
            parts = [p[..., border:-border, border:-border] for p in parts]
            hologram = hologram[..., border:-border, border:-border]

        batch = hologram.shape[0]
        measured = hologram.reshape(batch, -1).float()
        components = torch.stack([p.reshape(batch, -1) for p in parts], dim=1)

        if self.fit_radiometry:
            predicted, coefficients = self._fit_components(components, measured)
        else:
            predicted = components.sum(dim=1)
            coefficients = torch.ones(batch, components.shape[1], device=measured.device)

        # Scale the residual by the measurement's own spread, so the term is
        # dimensionless and does not depend on camera units.
        scale = measured.std(dim=1).clamp(min=1e-6)
        residual = (predicted - measured) / scale.unsqueeze(1)

        if self.criterion == "l1":
            value = residual.abs().mean(dim=1)
        elif self.criterion == "l2":
            value = (residual ** 2).mean(dim=1)
        else:
            # Fraction of the measurement's variance the model fails to explain.
            # Zero when the fit is perfect, one when it explains nothing.
            value = (residual ** 2).mean(dim=1)

        return (value, coefficients) if return_coefficients else value
>>>>>>> Stashed changes
