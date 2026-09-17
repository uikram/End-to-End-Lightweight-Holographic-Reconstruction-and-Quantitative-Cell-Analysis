"""The joint physics-aware objective.

    L = w_phase * L_phase
      + w_seg   * L_seg
      + w_cls   * L_cls
      + w_fwd   * L_forward_model                       <- reads the raw hologram
      + w_pmc   * L_PMC + w_bga * L_BGA + w_pv * L_PV    <- mask <-> phase
      + w_mass  * L_mass + w_area * L_area               <- measurement

The first three terms supervise each head against its own target.

The physics group splits into two families that behave very differently, and
keeping them separate is the point of the ablation.

L_forward_model propagates the predicted field to the sensor and compares it
with the hologram that was actually recorded. It is the only term that reads the
measurement, so it is the only one that introduces information the supervised
losses have not already consumed. This is what the reference literature means by
physics consistency (Huang et al., Nat. Mach. Intell. 2023; Galande et al.,
J. Biomed. Opt.; Lee et al., APL Mach. Learn. 4, 026106).

The remaining five couple the predicted mask to the predicted phase. They are
the previous study's physics-aware loss carried into the end-to-end setting,
where its information source has changed: in that study the phase was a measured
input and only the boundary was learned, so the coupling supplied the mask head
with knowledge of the optical field. Here both operands are network outputs and
both are directly supervised, and the mask target is itself a threshold of the
phase target, so the coupled quantity is already determined. Expect them to be
close to inert, and see docs/documentation.md for the measurement that shows it.

Every weight is read from ``loss.weights`` in the configuration; setting one to
zero removes its term, which is how the component-wise ablation is run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)
from .terms import (
    AmplitudeReconstructionLoss,
    BoundaryGradientAlignment,
    CellIntegratedPhase,
    CellProjectedArea,
    DryMassConsistency,
    ForwardModelConsistency,
    PhaseMaskContrast,
    PhaseReconstructionLoss,
    ProjectedAreaConsistency,
    PhaseVolumePreservation,
    SegmentationLoss,
)

_PHYSICS_KEYS = (
    "forward_model",
    "cell_integrated_phase",
    "cell_projected_area",
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
        self.cell_integrated_phase = CellIntegratedPhase(
            physics_cfg.volume_epsilon,
            physics_cfg.cell_min_reference_rad,
            physics_cfg.max_relative_error,
        )
        self.cell_projected_area = CellProjectedArea(
            physics_cfg.volume_epsilon,
            # A footprint floor in PIXELS, reusing the same "one cell" threshold
            # the gated terms use, so the two per-cell terms skip the same cells.
            float(physics_cfg.min_foreground_pixels),
            physics_cfg.max_relative_error,
        )
        self.amplitude_loss = AmplitudeReconstructionLoss(loss_cfg.amplitude)
        self.dry_mass = DryMassConsistency(physics_cfg.volume_epsilon)
        self.projected_area = ProjectedAreaConsistency(physics_cfg.volume_epsilon)

        # The only term that reads the raw hologram. Built unconditionally so a
        # misconfiguration surfaces at construction rather than at epoch 1.
        self.forward_model = ForwardModelConsistency(loss_cfg.forward_model, cfg.optics)
        self.modality = cfg.data.modality
        self._warned_normalised = False

    @property
    def active_physics_terms(self) -> list[str]:
        return [key for key in _PHYSICS_KEYS if self.weights.get(key, 0.0) != 0.0]

    @property
    def active_terms(self) -> dict[str, float]:
        """Every term with a non-zero weight, and its weight.

        Logged at the start of training, because this one line is what tells
        someone reading a log six weeks later which arm the checkpoint belongs
        to. ``active_physics_terms`` lists only the physics group, so an arm
        configured with an amplitude weight and nothing else appeared as
        'none' -- which is exactly the case where the log is load-bearing.
        """
        return {
            key: float(weight)
            for key, weight in sorted(self.weights.items())
            if float(weight) != 0.0
        }

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        phase_pred = outputs["phase"]
        seg_logits = outputs["segmentation"]
        condition_logits = outputs.get("condition")

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

        # Amplitude, against the classical-reconstruction reference. Skipped
        # rather than zeroed when either the head or the target is absent, so a
        # missing precompute is visible in the logged breakdown instead of
        # looking like a satisfied constraint.
        if self.weights.get("amplitude", 0.0):
            amplitude_pred = outputs.get("amplitude")
            amplitude_target = batch.get("amplitude")
            if amplitude_pred is None:
                raise KeyError(
                    "loss.weights.amplitude is non-zero but the model has no amplitude "
                    "head. Set model.amplitude.enabled: true."
                )
            if amplitude_target is None:
                raise KeyError(
                    "loss.weights.amplitude is non-zero but the batch carries no "
                    "'amplitude' target. Set data.provide_amplitude: true and run "
                    "`python scripts/prepare_amplitude.py --config <cfg>` first."
                )
            amplitude_term = self.amplitude_loss(amplitude_pred, amplitude_target)
            total = total + self.weights["amplitude"] * amplitude_term
            components["amplitude"] = float(amplitude_term.mean().detach())

        # Absent when model.classifier_enabled is false, in which case the term
        # is skipped rather than contributing a zero that would still appear in
        # the logged breakdown as if the head existed.
        if condition_logits is not None and self.weights.get("classification", 0.0):
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

        # -- forward-model consistency ------------------------------------
        # Unlike every other physics term this one is not gated by the
        # foreground count: it constrains the whole field, including the
        # background, and a crop without cells still has to be consistent with
        # the hologram it came from.
        if self.weights.get("forward_model", 0.0):
            # The RAW measurement, not the normalised network input. Falling
            # back to the normalised one would silently change the observation
            # model this term exists to test, so the fallback is announced.
            hologram = batch.get("hologram_raw")
            if hologram is None:
                hologram = batch.get("hologram")
                if hologram is None:
                    raise KeyError(
                        "loss.weights.forward_model is non-zero but the batch carries "
                        "neither 'hologram_raw' nor 'hologram'."
                    )
                if not self._warned_normalised:
                    self._warned_normalised = True
                    LOGGER.warning(
                        "forward-model term is comparing against the NORMALISED hologram "
                        "because the batch carries no 'hologram_raw'. The fitted "
                        "radiometric coefficients then describe the normalisation rather "
                        "than the camera. Rebuild the dataloaders so the raw intensity "
                        "reaches the loss."
                    )
            amplitude = outputs.get("amplitude")
            if amplitude is None:
                amplitude = torch.ones_like(phase_pred)
            forward_term, coefficients = self.forward_model(
                phase_pred, amplitude, hologram, self.modality,
                aberration=batch.get("aberration"),
                return_coefficients=True,
            )
            # Fields whose aberration surface could not be recovered are
            # excluded rather than down-weighted: without the surface the
            # forward model is not approximately wrong, it is wrong by more
            # than the signal, and averaging that in would corrupt the batch.
            usable = batch.get("aberration_valid")
            if usable is not None:
                # DEVICE as well as dtype. `.to(dtype)` alone leaves the tensor
                # where it was, and the evaluator hands this key straight off
                # the CPU batch while the trainer moves it, so on the GPU the
                # multiply below raised "Expected all tensors to be on the same
                # device" at the end of D1's first epoch -- in validation, not
                # in training, which is why it looked like a training bug. The
                # per-cell terms further down already move at the point of use
                # (`instances.to(phase_pred.device)`); this now matches them, so
                # the branch is correct whichever caller reaches it.
                weight = usable.to(
                    device=forward_term.device, dtype=forward_term.dtype
                ).reshape(-1)
                forward_term = forward_term * weight
                divisor = weight.sum().clamp(min=1.0)
                components["forward_model_fields_used"] = float(weight.sum())
            else:
                divisor = torch.tensor(
                    float(forward_term.numel()), device=forward_term.device
                ).clamp(min=1.0)

            # AVERAGED OVER USABLE FIELDS, ONCE, AND THE SAME WAY IT IS LOGGED.
            #
            # `total` is a per-sample vector that the caller reduces with
            # .mean(), i.e. divides by the batch size. Adding the per-sample
            # `forward_term` to it therefore contributed sum/B, while the value
            # recorded in `components` was sum/n_usable -- so the weight the
            # objective actually applied drifted from the logged one by
            # n_usable/B, and it drifted BATCH BY BATCH as the number of fields
            # with a recoverable aberration surface changed. Reducing to a scalar
            # here makes the contribution exactly w * sum/n_usable (a scalar
            # broadcast across the vector survives .mean() unchanged), which is
            # how every other physics term below is already handled.
            forward_mean = forward_term.sum() / divisor
            total = total + self.weights["forward_model"] * forward_mean
            components["forward_model"] = float(forward_mean.detach())
            # Logged so the calibration is inspectable. A gain that changes sign
            # or drifts across epochs means the forward model is wrong, and that
            # would otherwise be invisible inside the residual.
            fitted = coefficients.mean(dim=0).detach()
            # Six for off-axis: the constant, the object intensity, and both
            # quadratures of each conjugate cross-term. The two quadratures of
            # one sideband are the in-phase and out-of-phase parts of the same
            # fringe; what is physically meaningful is their magnitude, since
            # their ratio is the unmeasurable global piston phase.
            names = ("offset", "object_gain",
                     "fringe_gain", "fringe_gain_conj",
                     "fringe_quadrature", "fringe_quadrature_conj")
            for index, name in enumerate(names):
                if index < fitted.numel():
                    components[f"forward_{name}"] = float(fitted[index])
            if fitted.numel() >= 6:
                components["forward_fringe_magnitude"] = float(
                    (fitted[2] ** 2 + fitted[4] ** 2).sqrt()
                )
            if self.forward_model.learn_distance:
                components["forward_distance_um"] = float(
                    self.forward_model.distance.detach()
                )

        if self.weights.get("phase_mask_contrast", 0.0):
            raw_terms["phase_mask_contrast"] = self.pmc(foreground, phase_pred)

        if self.weights.get("boundary_gradient_alignment", 0.0):
            reference_phase = phase_pred if self.bga_reference == "pred" else phase_target
            raw_terms["boundary_gradient_alignment"] = self.bga(foreground, reference_phase)

        if self.weights.get("phase_volume", 0.0):
            raw_terms["phase_volume"] = self.phase_volume(
                foreground, phase_pred, target_foreground, phase_target
            )

        # Per-cell Integrated-Phase Preservation. Needs the reference instance
        # labelling, which the dataloader carries alongside the mask; without it
        # the term cannot be evaluated and is skipped rather than silently
        # degrading to the image-level version.
        if self.weights.get("cell_integrated_phase", 0.0):
            instances = batch.get("instances")
            if instances is None:
                raise KeyError(
                    "loss.weights.cell_integrated_phase is non-zero but the batch "
                    "carries no 'instances' tensor. Set data.provide_instances: true."
                )
            raw_terms["cell_integrated_phase"] = self.cell_integrated_phase(
                foreground, phase_pred, target_foreground, phase_target,
                instances.to(phase_pred.device),
            )

        # Per-cell projected area. Same reference domains as the integrated-phase
        # term, so the two constrain the two factors of the same measurement.
        if self.weights.get("cell_projected_area", 0.0):
            instances = batch.get("instances")
            if instances is None:
                raise KeyError(
                    "loss.weights.cell_projected_area is non-zero but the batch "
                    "carries no 'instances' tensor. Set data.provide_instances: true."
                )
            raw_terms["cell_projected_area"] = self.cell_projected_area(
                foreground, target_foreground, instances.to(phase_pred.device),
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
