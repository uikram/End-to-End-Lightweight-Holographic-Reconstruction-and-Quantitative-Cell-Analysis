"""The joint physics-aware objective.

    L = w_phase * L_phase
      + w_seg   * L_seg
      + w_cls   * L_cls
      + w_pmc   * L_PMC + w_bga * L_BGA + w_pv * L_PV
      + w_mass  * L_mass + w_area * L_area

The first three terms supervise each head against its own target. The remaining
five are the extension of the previous study's physics-aware loss to the
end-to-end setting: they act on the *predicted* phase and the *predicted* mask
together, so the network cannot satisfy them by getting either output right in
isolation. That coupling is what makes the reconstruction measurement-ready.

Every weight is read from ``loss.weights`` in the configuration; setting one to
zero removes its term, which is how the component-wise ablation is run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .terms import (
    BoundaryGradientAlignment,
    DryMassConsistency,
    PhaseMaskContrast,
    PhaseReconstructionLoss,
    ProjectedAreaConsistency,
    PhaseVolumePreservation,
    SegmentationLoss,
)

_PHYSICS_KEYS = (
    "phase_mask_contrast",
    "boundary_gradient_alignment",
    "phase_volume",
    "dry_mass_consistency",
    "projected_area_consistency",
)


class JointPhysicsAwareLoss(nn.Module):
    """Combines reconstruction, segmentation, classification and physics."""

    def __init__(self, cfg: Config):
        super().__init__()
        loss_cfg = cfg.loss
        weights = loss_cfg.weights
        physics_cfg = loss_cfg.physics

        self.weights = {key: float(weights[key]) for key in weights}
        self.skip_empty_targets = physics_cfg.skip_empty_targets
        self.min_foreground_pixels = physics_cfg.min_foreground_pixels
        self.max_relative_error = physics_cfg.max_relative_error
        self.bga_reference = physics_cfg.bga_reference
        if self.bga_reference not in ("pred", "gt"):
            raise ValueError(f"loss.physics.bga_reference must be pred or gt, got {self.bga_reference!r}")

        self.phase_loss = PhaseReconstructionLoss(loss_cfg.phase)
        self.segmentation_loss = SegmentationLoss(
            loss_cfg.segmentation, cfg.model.segmentation_classes
        )

        classification_weights = loss_cfg.classification.class_weights
        self.register_buffer(
            "condition_weights",
            torch.tensor(list(classification_weights), dtype=torch.float32)
            if classification_weights else None,
        )
        self.label_smoothing = loss_cfg.classification.label_smoothing

        self.pmc = PhaseMaskContrast(physics_cfg.pmc_margin, physics_cfg.pmc_collapse_warn_ratio)
        self.bga = BoundaryGradientAlignment(physics_cfg.bga_epsilon)
        self.phase_volume = PhaseVolumePreservation(physics_cfg.volume_epsilon)
        self.dry_mass = DryMassConsistency(physics_cfg.volume_epsilon)
        self.projected_area = ProjectedAreaConsistency(physics_cfg.volume_epsilon)

    @property
    def active_physics_terms(self) -> list[str]:
        return [key for key in _PHYSICS_KEYS if self.weights.get(key, 0.0) != 0.0]

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        phase_pred = outputs["phase"]
        seg_logits = outputs["segmentation"]
        condition_logits = outputs["condition"]

        phase_target = batch["phase"]
        mask_target = batch["mask"]
        condition_target = batch["condition"]

        components: dict[str, float] = {}
        batch_size = phase_pred.shape[0]
        total = torch.zeros(batch_size, device=phase_pred.device, dtype=phase_pred.dtype)

        # -- supervised terms ---------------------------------------------
        phase_term = self.phase_loss(phase_pred, phase_target)
        total = total + self.weights["phase"] * phase_term
        components["phase"] = float(phase_term.mean().detach())

        segmentation_term = self.segmentation_loss(seg_logits, mask_target)
        total = total + self.weights["segmentation"] * segmentation_term
        components["segmentation"] = float(segmentation_term.mean().detach())

        classification_term = F.cross_entropy(
            condition_logits,
            condition_target,
            weight=self.condition_weights.to(condition_logits.dtype)
            if self.condition_weights is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        total = total + self.weights["classification"] * classification_term
        components["classification"] = float(classification_term.mean().detach())

        # -- physics coupling ---------------------------------------------
        foreground = 1.0 - torch.softmax(seg_logits, dim=1)[:, 0:1]
        target_foreground = (mask_target > 0).to(phase_pred.dtype).unsqueeze(1)

        # The measurement terms are relative errors, so they divide by the
        # reference integral of the crop. A crop holding a sliver of one cell --
        # a field edge, or the gap between cells -- carries no measurable
        # quantity, and dividing by its near-zero reference produces a spike
        # large enough to dominate the epoch. Such crops are excluded here on the
        # same footing as empty ones.
        foreground_pixels = target_foreground.sum(dim=[1, 2, 3])
        has_cells = (foreground_pixels >= self.min_foreground_pixels).to(phase_pred.dtype)
        if not self.skip_empty_targets:
            has_cells = torch.ones_like(has_cells)
        denominator = has_cells.sum().clamp(min=1.0)

        raw_terms: dict[str, torch.Tensor] = {}

        if self.weights.get("phase_mask_contrast", 0.0):
            raw_terms["phase_mask_contrast"] = self.pmc(foreground, phase_pred)

        if self.weights.get("boundary_gradient_alignment", 0.0):
            reference_phase = phase_pred if self.bga_reference == "pred" else phase_target
            raw_terms["boundary_gradient_alignment"] = self.bga(foreground, reference_phase)

        if self.weights.get("phase_volume", 0.0):
            raw_terms["phase_volume"] = self.phase_volume(
                foreground, phase_pred, target_foreground, phase_target
            )

        if self.weights.get("dry_mass_consistency", 0.0):
            raw_terms["dry_mass_consistency"] = self.dry_mass(
                foreground, phase_pred, target_foreground, phase_target
            )

        if self.weights.get("projected_area_consistency", 0.0):
            raw_terms["projected_area_consistency"] = self.projected_area(
                foreground, target_foreground
            )

        for name, term in raw_terms.items():
            # Second line of defence: cap any single relative error so one
            # pathological crop cannot outweigh the rest of the batch. The cap
            # sits far above the values a converging model produces, so it never
            # binds during normal training.
            if self.max_relative_error is not None:
                term = term.clamp(max=self.max_relative_error)
            masked = (term * has_cells).sum() / denominator
            total = total + self.weights[name] * masked
            components[name] = float(masked.detach())

        loss = total.mean()
        components["total"] = float(loss.detach())
        return loss, components


def build_loss(cfg: Config) -> JointPhysicsAwareLoss:
    return JointPhysicsAwareLoss(cfg)
