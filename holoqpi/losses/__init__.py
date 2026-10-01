"""Joint objective (measurement-aware terms, forward-model consistency) for hologram reconstruction and cell analysis."""

from .composite import JointMeasurementLoss, JointPhysicsAwareLoss, build_loss
from .terms import (
    BoundaryGradientAlignment,
    DryMassConsistency,
    ImageIntegratedPhase,
    LegacyPhaseMaskContrast,
    PhaseMaskContrast,
    PhaseReconstructionLoss,
    PhaseVolumePreservation,
    ProjectedAreaConsistency,
    SegmentationLoss,
    spatial_gradient,
    structural_similarity,
)

__all__ = [
    "JointMeasurementLoss",
    "JointPhysicsAwareLoss",  # backward-compatible alias
    "build_loss",
    "PhaseReconstructionLoss",
    "SegmentationLoss",
    "LegacyPhaseMaskContrast",
    "PhaseMaskContrast",  # backward-compatible alias
    "ImageIntegratedPhase",
    "BoundaryGradientAlignment",
    "PhaseVolumePreservation",  # backward-compatible alias
    "DryMassConsistency",
    "ProjectedAreaConsistency",
    "spatial_gradient",
    "structural_similarity",
]
