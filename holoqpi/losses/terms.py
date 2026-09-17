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

# Warnings about insufficient diffraction context are deduplicated across every
# instance of the loss. A distance sweep builds one term per candidate distance,
# and without this the calibration log fills with a hundred copies of the same
# sentence, burying everything else in it.
_WARNED_PAD: set = set()


def _accumulation_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """float32 unless something wider is already in play.

    WHY EVERY PER-CELL AND PER-IMAGE INTEGRAL IN THIS FILE IS PINNED EXPLICITLY.

    These terms sum over thousands of pixels, and this dataset's per-cell
    quantities are large:

        projected area      median  5964 px      max  9717 px
        integrated phase    median  5399 rad     max  9429 rad

    In float16 (11-bit mantissa) the spacing at 4096 is 4, so a sequential
    accumulation of values near unity STALLS -- 4096 + 1 rounds back to 4096 --
    and an integral saturates instead of growing. An image-level integral is
    worse: a 512 px crop at this dataset's ~19% coverage sums to roughly 70,000,
    past float16's largest finite value of 65504, giving ``inf`` and then
    ``inf/inf``. ``scatter_add`` is the acute case because it accumulates in the
    output tensor's own dtype rather than in a wider one.

    WHAT WAS AND WAS NOT ACTUALLY HAPPENING, because the distinction matters
    for what may be said about the runs already recorded.

    Training runs under autocast, and CUDA autocast places ``softmax`` and
    ``sum`` in its FP32 list (``AT_FORALL_FP32_SET_OPT_DTYPE`` in
    ``ATen/autocast_mode.h``: prod, softmax, log_softmax, cumprod, cumsum,
    linalg norms, sum). So on the GPU the foreground probability map came back
    float32, the products with it were float32, and these integrals were ALREADY
    accumulating in float32. The saturation above did not occur on the study's
    runs, and no recorded result is affected by this change.

    There is an independent proof of that, which does not rely on reading the
    policy table: before this change ``CellIntegratedPhase`` built its
    accumulator from ``foreground * phase`` while the reference map was
    ``(mask > 0).to(phase.dtype) * phase_target``, whose float32 target promotes
    the product to float32. If ``foreground`` had been float16 the two would have
    disagreed and ``scatter_add`` would have raised
    "Expected self.dtype to be equal to src.dtype" on the first step. Arm B
    trained for sixty epochs with that term at weight 1.0, so ``foreground`` was
    float32.

    IT IS STILL PINNED, for three reasons. The CPU autocast policy is a
    different and smaller list -- ``softmax`` is NOT on it -- so the same code
    raised under CPU autocast, which is reachable by any future run that enables
    mixed precision off the GPU and is what ``selftest.test_autocast_integrals``
    reproduces. The policy is a PyTorch implementation detail that has changed
    between releases. And the correctness of the study's central measurement
    should not rest on an op being on a version-specific promotion list. float32
    costs nothing measurable here: these are reductions, not convolutions, and
    the gradient flows back into the half-precision graph unchanged.
    """
    for tensor in tensors:
        if tensor is not None and tensor.dtype in (torch.float64, torch.complex128):
            return torch.float64
    return torch.float32


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
            # float32 for the same reason as every other integral in this file
            # (see _accumulation_dtype). These sums are over the WHOLE crop:
            # 512 x 512 = 262,144 pixels, so a background-class sum reaches
            # ~131,000 and overflows float16 outright. CUDA autocast promotes
            # softmax and sum to float32 so this did not bite on the GPU, but
            # this is the one term with weight 1.0 in EVERY arm and its
            # correctness should not depend on that policy.
            dtype = _accumulation_dtype(logits)
            probabilities = torch.softmax(logits, dim=1).to(dtype)
            dice_total = torch.zeros_like(loss, dtype=dtype)
            weight_total = 0.0
            for class_index in range(self.num_classes):
                predicted = probabilities[:, class_index]
                actual = (target == class_index).to(dtype)
                intersection = (predicted * actual).sum(dim=[1, 2])
                union = predicted.sum(dim=[1, 2]) + actual.sum(dim=[1, 2])
                class_dice = 1.0 - (2 * intersection + self.smooth) / (union + self.smooth)
                weight = float(self.class_weights[class_index])
                dice_total = dice_total + weight * class_dice
                weight_total += weight
            loss = loss + self.w_dice * (
                dice_total / max(weight_total, 1e-8)
            ).to(loss.dtype)

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
        # Pixel-count and phase sums over a whole crop, so float32: see
        # _accumulation_dtype.
        dtype = _accumulation_dtype(foreground, phase)
        foreground = foreground.squeeze(1).to(dtype)
        phase = phase.squeeze(1).to(dtype)

        cell_area = foreground.sum(dim=[1, 2]).clamp(min=1.0)
        background_area = (1.0 - foreground).sum(dim=[1, 2]).clamp(min=1.0)

        mean_cell = (foreground * phase).sum(dim=[1, 2]) / cell_area
        mean_background = ((1.0 - foreground) * phase).sum(dim=[1, 2]) / background_area

        if self.training and self._warnings_emitted < 3:
            ratio = float(foreground.mean().detach())
            if ratio > self.collapse_warn_ratio:
                self._warnings_emitted += 1
                # Through the logger, so it lands in the run's train.log next to
                # the epoch it belongs to rather than only on a terminal nobody
                # is watching by the time it fires.
                LOGGER.warning(
                    "PhaseMaskContrast: foreground fraction %.1f%% exceeds %.0f%%; "
                    "the mask may be collapsing to all-cell (%d/3)",
                    100.0 * ratio, 100.0 * self.collapse_warn_ratio,
                    self._warnings_emitted,
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
        # float32: a 512 px crop's phase integral reaches ~70,000, past
        # float16's maximum finite value. See _accumulation_dtype.
        dtype = _accumulation_dtype(foreground, phase, target_phase)
        predicted = (foreground.squeeze(1).to(dtype)
                     * phase.squeeze(1).to(dtype)).sum(dim=[1, 2])
        reference = (target_foreground.squeeze(1).to(dtype)
                     * target_phase.squeeze(1).to(dtype)).sum(dim=[1, 2])
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
        # float32, for the same overflow reason as PhaseVolumePreservation.
        dtype = _accumulation_dtype(foreground, phase, target_phase)
        predicted = (foreground.squeeze(1).to(dtype)
                     * phase.squeeze(1).to(dtype)).sum(dim=[1, 2])
        reference = (target_foreground.squeeze(1).to(dtype)
                     * target_phase.squeeze(1).to(dtype)).sum(dim=[1, 2])
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
        # float32: a foreground pixel count over a 512 px crop reaches ~50,000,
        # which float16 can represent but not accumulate. See
        # _accumulation_dtype.
        dtype = _accumulation_dtype(foreground, target_foreground)
        predicted = foreground.squeeze(1).to(dtype).sum(dim=[1, 2])
        reference = target_foreground.squeeze(1).to(dtype).sum(dim=[1, 2])
        return (predicted - reference).abs() / (reference + self.epsilon)


class CellIntegratedPhase(nn.Module):
    """Integrated-Phase Preservation, evaluated PER CELL rather than per image.

    This is the central term of the v2 study, and the reason it exists is a
    cancellation the image-level version cannot see.

    ``PhaseVolumePreservation`` compares one number per image: the phase integral
    inside the predicted mask against the integral inside the reference mask. A
    field in which one cell is over-measured by 20% and another under-measured by
    20% scores a perfect zero. Dry mass is reported per cell, so that is exactly
    the error the study cares about and exactly the error an image-level sum
    hides. Restricting the integral to individual cellular regions removes the
    cancellation and is the literal reading of "preserve the integrated
    quantitative phase within individual cellular regions".

    Regions come from the REFERENCE instance labelling, computed once per field
    in the dataloader. Using reference rather than predicted regions is
    deliberate: the domains must not move while the loss is being minimised, or
    the term could be satisfied by redrawing the boundaries rather than by
    correcting the measurement.

    Within each region the term compares the predicted measurement -- the
    predicted phase integrated over the predicted foreground -- against the
    reference measurement. Both heads therefore receive gradient, which is the
    coupling the objective is trying to establish.

    Cells whose reference integral is below ``min_reference`` are skipped: the
    relative error of a near-zero denominator is meaningless and would dominate
    the batch. The per-cell errors are averaged over CELLS, not over pixels, so a
    field of many small cells is not outweighed by one large one.
    """

    def __init__(self, epsilon: float, min_reference: float, max_relative_error: float | None):
        super().__init__()
        self.epsilon = epsilon
        self.min_reference = min_reference
        self.max_relative_error = max_relative_error

    def forward(
        self,
        foreground: torch.Tensor,
        phase: torch.Tensor,
        target_foreground: torch.Tensor,
        target_phase: torch.Tensor,
        instances: torch.Tensor,
    ) -> torch.Tensor:
        # PINNED TO float32, AND THIS IS LOAD-BEARING.
        #
        # scatter_add accumulates in the tensor's own dtype. Under autocast both
        # operands arrive as float16, and this dataset's per-cell phase integral
        # has a median of 5399 rad -- past the point where float16's spacing
        # exceeds the per-pixel increment, so the accumulation stalls and every
        # large cell reports the same saturated integral. The relative error then
        # measures the arithmetic rather than the reconstruction. Two of the
        # three integrals here used to be float16 and one float32, which also
        # made the comparison between them systematically biased.
        # See _accumulation_dtype for the measured numbers.
        dtype = _accumulation_dtype(foreground, phase, target_foreground, target_phase)
        foreground = foreground.squeeze(1).to(dtype)
        phase = phase.squeeze(1).to(dtype)
        target_foreground = target_foreground.squeeze(1).to(dtype)
        target_phase = target_phase.squeeze(1).to(dtype)
        if instances.dim() == 4:
            instances = instances.squeeze(1)

        batch = foreground.shape[0]
        predicted_map = foreground * phase
        reference_map = target_foreground * target_phase

        losses = []
        for item in range(batch):
            labels = instances[item].reshape(-1).long()
            count = int(labels.max().item())
            if count < 1:
                losses.append(predicted_map.new_zeros(()))
                continue

            # Scatter-add into one bin per cell. Bin 0 collects background and is
            # discarded; it is kept in the tensor only so the label values index
            # directly without an offset.
            bins = predicted_map.new_zeros(count + 1)
            predicted = bins.scatter_add(0, labels, predicted_map[item].reshape(-1))
            reference = bins.scatter_add(0, labels, reference_map[item].reshape(-1))
            predicted, reference = predicted[1:], reference[1:]

            usable = reference.abs() >= self.min_reference
            if not bool(usable.any()):
                losses.append(predicted_map.new_zeros(()))
                continue

            error = (predicted[usable] - reference[usable]).abs() / (
                reference[usable].abs() + self.epsilon
            )
            if self.max_relative_error is not None:
                error = error.clamp(max=float(self.max_relative_error))
            losses.append(error.mean())

        return torch.stack(losses)


class CellProjectedArea(nn.Module):
    """Per-cell projected-area preservation: the symmetric partner of IPP.

    ``CellIntegratedPhase`` constrains one of the two quantities a per-cell
    measurement depends on. This constrains the other.

    Dry mass is
        m_i = (lambda / 2 pi alpha) * sum_{p in Omega_i} phi(p) * dx * dy
    and projected area is
        A_i = |Omega_i| * dx * dy
    so the segmented footprint is a reported measurement in its own right and
    also the domain over which the phase integral is taken. After the v2 change
    switched off the image-level ``projected_area_consistency``, nothing
    constrained it at all.

    Formulated exactly like the integrated-phase term -- a relative error over
    the REFERENCE instance domains, averaged over cells rather than pixels -- so
    the two are directly comparable and share the same scatter-add machinery.
    Using reference domains matters for the same reason: a term evaluated on
    predicted domains could be satisfied by moving the boundary rather than by
    getting the footprint right.

    Note what this does and does not do. Inside Omega_i^GT it pushes the soft
    foreground toward one, so it sharpens confidence within known cells. It
    cannot suppress false-positive area OUTSIDE a reference cell, because those
    pixels are never summed. That is the segmentation loss's job.
    """

    def __init__(self, epsilon: float, min_reference: float, max_relative_error: float | None):
        super().__init__()
        self.epsilon = epsilon
        self.min_reference = min_reference
        self.max_relative_error = max_relative_error

    def forward(
        self,
        foreground: torch.Tensor,
        target_foreground: torch.Tensor,
        instances: torch.Tensor,
    ) -> torch.Tensor:
        # float32, for the reason given in CellIntegratedPhase: a per-cell
        # footprint has a median of 5964 px on this dataset, and a float16
        # accumulator stops growing at 4096 when the increments are near unity,
        # so both the predicted and the reference area would saturate at the
        # same value and the relative error would read close to zero however
        # wrong the boundary was. See _accumulation_dtype.
        dtype = _accumulation_dtype(foreground, target_foreground)
        foreground = foreground.squeeze(1).to(dtype)
        target_foreground = target_foreground.squeeze(1).to(dtype)
        if instances.dim() == 4:
            instances = instances.squeeze(1)

        losses = []
        for item in range(foreground.shape[0]):
            labels = instances[item].reshape(-1).long()
            count = int(labels.max().item())
            if count < 1:
                losses.append(foreground.new_zeros(()))
                continue

            bins = foreground.new_zeros(count + 1)
            predicted = bins.scatter_add(0, labels, foreground[item].reshape(-1))
            reference = bins.scatter_add(0, labels, target_foreground[item].reshape(-1))
            predicted, reference = predicted[1:], reference[1:]

            # min_reference is in PIXELS here, not radians: it is the smallest
            # footprint worth measuring, so the same floor the mask generator
            # uses for its area filter.
            usable = reference >= self.min_reference
            if not bool(usable.any()):
                losses.append(foreground.new_zeros(()))
                continue

            error = (predicted[usable] - reference[usable]).abs() / (
                reference[usable] + self.epsilon
            )
            if self.max_relative_error is not None:
                error = error.clamp(max=float(self.max_relative_error))
            losses.append(error.mean())

        return torch.stack(losses)


class AmplitudeReconstructionLoss(nn.Module):
    """L1 against the classical-reconstruction amplitude REFERENCE.

    The professor's brief asks for phase AND amplitude. Until now the amplitude
    head had no target at all: it was zero-initialised to emit exactly 1.0 and
    was trained only by the forward-model residual, which is itself off by
    default. So the framework did not deliver one of its three stated outputs.

    The target here is ``|reconstruct_off_axis(H)|`` precomputed by
    ``scripts/prepare_amplitude.py``. It is a REFERENCE, not ground truth: it is
    a classical reconstruction carrying its own errors, and no independent
    amplitude measurement exists for this dataset. Any write-up must say so.

    The weight is deliberately low for that reason. OAH-Net (Biomed. Opt.
    Express 16(3):894, 2025) weights amplitude at 0.1 against phase in its
    combined L1, on a dataset where amplitude ground truth came from the
    instrument's own software; with a reconstructed pseudo-target the case for
    keeping it modest is stronger, not weaker.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_l1 = cfg.l1

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_l1 * (prediction - target).abs().mean(dim=[1, 2, 3])


class ForwardModelConsistency(nn.Module):
    """Data fidelity against the hologram that was actually recorded.

    This is the constraint the reference literature means by physics
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
    """

    def __init__(self, cfg, optics):
        super().__init__()
        self.wavelength_um = float(optics.wavelength_um)
        self.pitch_x_um = float(optics.pixel_pitch_x_um)
        self.pitch_y_um = float(optics.pixel_pitch_y_um)

        self.pad_px = cfg.pad_px
        self.border_px = cfg.border_px
        self.feature_um = float(cfg.feature_um)
        self.dc_exclusion_px = cfg.dc_exclusion_px
        self.dc_exclusion_frac = float(cfg.dc_exclusion_frac)
        self.fit_radiometry = bool(cfg.fit_radiometry)
        # A distance sweep already knows, and reports once, how many of its
        # candidate distances exceed what the field can model. Repeating the
        # warning per distance buries the result it is attached to.
        self.warn_on_short_pad = bool(getattr(cfg, "warn_on_short_pad", True))
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
        # THE PAD AND THE BORDER ARE FROZEN AT THE CONFIGURED DISTANCE, NEVER
        # TAKEN FROM THE LIVE PARAMETER.
        #
        # required_pad() derives a diffraction margin from z, and border_px
        # defaults to that margin, so the residual is computed over the field
        # minus that border. If the margin tracked a learnable z, the objective's
        # comparison DOMAIN would change every time z moved: at z = 33.77 um the
        # border is 79 px and the residual is scored on a 742 px window, while at
        # z = -5 um it is 12 px and a 776 px window. Gradient descent could then
        # lower the residual by shrinking |z| purely because that changes which
        # pixels are being compared, which is not a statement about the
        # propagation distance at all -- and the two residuals are not on the
        # same scale, so the trajectory could not be read either.
        #
        # Freezing it makes the residual a smooth function of z over one fixed
        # domain, which is the only form in which "learn z" means anything. For a
        # fixed-z arm the behaviour is identical to before, because the live value
        # never changes.
        self._reference_distance_um = None if distance is None else abs(float(distance))

    # -- helpers ----------------------------------------------------------
    @property
    def distance_um(self) -> float | None:
        """The distance the operator currently propagates by, tracking z."""
        return None if self.distance is None else float(self.distance.detach())

    def required_pad(self) -> int:
        """Diffraction spread in pixels, at the CONFIGURED distance.

        Deliberately independent of the live value of z; see the note in
        ``__init__``. A distance sweep builds one term per candidate distance and
        so still gets one pad per candidate, which is why a sweep must pass an
        explicit ``pad_px`` if it wants a single domain across the scan.
        """
        if self.pad_px is not None:
            return int(self.pad_px)
        distance = self._reference_distance_um
        if not distance:
            return 0
        spread_um = abs(self.wavelength_um * distance) / self.feature_um
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
        signature = (round(float(self._reference_distance_um or 0.0), 1), size)
        if required > applied and self.warn_on_short_pad and signature not in _WARNED_PAD:
            _WARNED_PAD.add(signature)
            LOGGER.warning(
                "forward model at z = %.1f um needs %d px of diffraction context but a "
                "%d px field allows only %d. The residual is approximate on this crop. "
                "Train on larger crops (data.train_crop), lower "
                "loss.forward_model.distance_um, or set loss.forward_model.pad_px "
                "explicitly to accept the approximation deliberately.",
                self._reference_distance_um, required, size, applied,
            )
        # A SEPARATE CHECK ON THE LIVE DISTANCE, because the pad is deliberately
        # frozen at the configured one. If a learned z drifts far from where the
        # pad was sized, the residual becomes badly under-padded and nothing
        # above would say so -- the warning keys on the reference distance, which
        # by construction never moves. config/base.yaml's note on
        # physics_parameter_lr_scale says such an excursion must be reported, so
        # it is reported here rather than left to whoever reads history.json.
        live = self.distance_um
        reference = self._reference_distance_um
        if (
            self.learn_distance and live is not None and reference
            and abs(abs(live) - reference) > max(2.0, 0.25 * reference)
        ):
            signature = ("drift", round(float(live), 0), size)
            if signature not in _WARNED_PAD:
                _WARNED_PAD.add(signature)
                LOGGER.warning(
                    "the learned propagation distance has moved to %.2f um, far from "
                    "the %.2f um the diffraction pad was sized for. The pad is held "
                    "fixed on purpose -- a residual whose domain moves with z cannot "
                    "be compared across epochs -- but the synthesised field is now "
                    "under-padded, so the residual is approximate. Report the "
                    "trajectory, and consider lowering "
                    "training.physics_parameter_lr_scale.",
                    live, reference,
                )
        return applied

    @staticmethod
    def _fit_components(components: torch.Tensor, measured: torch.Tensor):
        """Least-squares fit of the radiometric coefficients, per image.

        ``components`` is (B, K, N) and ``measured`` is (B, N). Returns the
        fitted prediction (B, N) and the coefficients (B, K). Solved in closed
        form with a small ridge term, so it is differentiable and cannot blow up
        when two components are nearly collinear (which happens when |U|^2 is
        almost constant, i.e. exactly at the in-line degenerate case).

        AUTOCAST IS DISABLED FOR THIS WHOLE REGION, and it has to be, for two
        separate reasons.

        The first one crashed experiment D1 twice. ``gram`` is built by a
        matmul, and **autocast re-casts a matmul's operands to float16 even
        when they are already float32** -- that is what autocast is for. So
        ``gram`` came back Half while ``rhs``, which is elementwise multiply
        plus sum and therefore left alone, stayed Float, and
        ``torch.linalg.solve`` raised "Expected A and B to have the same dtype,
        but found A of type Half and B of type Float". Casting the inputs with
        ``.float()`` does not fix it, because the cast is undone by the very
        next operation. Only leaving the autocast region does.

        The second would have been worse if the first had not fired: float16
        has an epsilon of about 1e-3, so a ridge of 1e-6 relative to the Gram
        diagonal is far below the representable resolution and the
        regularisation this function depends on would silently do nothing.

        Verified by construction: inside ``torch.autocast`` a float32 matmul
        returns reduced precision while the elementwise branch does not; with
        the region disabled both come back float32.
        """
        dtype = components.dtype
        with torch.autocast(device_type=components.device.type, enabled=False):
            comp = components.float()
            meas = measured.float()
            gram = comp @ comp.transpose(1, 2)
            rhs = (comp * meas.unsqueeze(1)).sum(dim=2, keepdim=True)
            ridge = 1e-6 * torch.diag_embed(
                torch.diagonal(gram, dim1=1, dim2=2).clamp(min=1e-12)
            )
            coefficients = torch.linalg.solve(gram + ridge, rhs)
            fitted = (coefficients.transpose(1, 2) @ comp).squeeze(1)
        # Back to the surrounding graph's dtype so the residual is computed in
        # the same precision as every other loss term.
        return fitted.to(dtype), coefficients.squeeze(2).to(dtype)

    def forward(
        self,
        phase: torch.Tensor,
        amplitude: torch.Tensor,
        hologram: torch.Tensor,
        modality: str,
        aberration: torch.Tensor | None = None,
        return_coefficients: bool = False,
    ) -> torch.Tensor:
        from ..physics import (
            estimate_carrier, pad_reflect, propagate, reference_wave, unpad,
        )

        if self.distance is None:
            raise ValueError(
                "loss.forward_model.distance_um is null. Run scripts/calibrate_z.py "
                "or obtain the acquisition distance before enabling this term."
            )

        pad = self._pad_for(min(phase.shape[-2], phase.shape[-1]))

        # Restore what the delivered phase had removed. The reference maps were
        # aberration-corrected and background-subtracted before delivery, but
        # the recorded hologram still contains that surface, so propagating the
        # phase alone cannot reproduce the measurement at any distance. Omitting
        # this does not merely add error: a quadratic aberration is degenerate
        # with defocus, so the search tries to absorb a fixed optical term into
        # z and no single distance fits.
        total_phase = phase.float()
        if aberration is not None:
            total_phase = total_phase + aberration.float()
        field = torch.polar(amplitude.clamp(min=0.0).float(), total_phase)
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
            #
            # AND BOTH QUADRATURES OF EACH, which is not optional either.
            # Propagation multiplies the field by exp(i 2 pi z / lambda): a
            # global piston phase that is not measurable and that cycles
            # completely every half wavelength of z. With only the real parts
            # the fit cannot absorb it, so the residual becomes a function of
            # where z happens to fall modulo lambda/2. Measured on this data
            # before the quadratures were added, the off-axis residual swung
            # between 0.13 and 0.99 -- nearly its whole range -- over a 0.33 um
            # change in z, and any reported value was a lottery. Carrying the
            # imaginary parts spans Re(R* U exp(i phi)) for every phi, because
            # Re(R* U e^{i phi}) = cos(phi) Re(R* U) - sin(phi) Im(R* U), so the
            # fit removes the piston instead of being defeated by it.
            #
            # The in-line arm needs none of this: |U|^2 cancels any global
            # phase already, which is why its residual is smooth in z while the
            # off-axis one was not. That contrast is the control for this fix.
            parts.append(2.0 * (reference.conj() * propagated).real)
            parts.append(2.0 * (reference * propagated).real)
            parts.append(-2.0 * (reference.conj() * propagated).imag)
            parts.append(-2.0 * (reference * propagated).imag)
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
            # Mean squared residual in units of the measurement's own variance,
            # i.e. the fraction of that variance the model fails to explain.
            # Zero on a perfect fit, one when it explains nothing.
            value = (residual ** 2).mean(dim=1)
        else:
            # 1 - Pearson r between the synthesised and the measured intensity.
            #
            # This branch used to be a verbatim copy of the l2 one, so selecting
            # `correlation` silently got `l2` while the docstring described
            # something else. The two are genuinely different: l2 is sensitive to
            # a residual the radiometric fit could not absorb at all, whereas
            # this is invariant to any remaining affine error and responds only
            # to structure. Kept selectable because that invariance is sometimes
            # what a diagnostic wants; `l2` remains the default and is what every
            # arm in this study uses.
            centred_prediction = predicted - predicted.mean(dim=1, keepdim=True)
            centred_measurement = measured - measured.mean(dim=1, keepdim=True)
            numerator = (centred_prediction * centred_measurement).sum(dim=1)
            denominator = (
                centred_prediction.pow(2).sum(dim=1).sqrt()
                * centred_measurement.pow(2).sum(dim=1).sqrt()
            ).clamp(min=1e-12)
            value = 1.0 - numerator / denominator

        return (value, coefficients) if return_coefficients else value
