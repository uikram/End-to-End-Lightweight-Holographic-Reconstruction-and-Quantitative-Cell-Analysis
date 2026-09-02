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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


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
