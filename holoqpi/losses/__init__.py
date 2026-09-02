"""Joint physics-aware objective for hologram reconstruction and cell analysis."""

from .composite import JointPhysicsAwareLoss, build_loss
from .terms import (
    BoundaryGradientAlignment,
    DryMassConsistency,
    PhaseMaskContrast,
    PhaseReconstructionLoss,
    PhaseVolumePreservation,
    ProjectedAreaConsistency,
    SegmentationLoss,
    spatial_gradient,
    structural_similarity,
)

__all__ = [
    "JointPhysicsAwareLoss",
    "build_loss",
    "PhaseReconstructionLoss",
    "SegmentationLoss",
    "PhaseMaskContrast",
    "BoundaryGradientAlignment",
    "PhaseVolumePreservation",
    "DryMassConsistency",
    "ProjectedAreaConsistency",
    "spatial_gradient",
    "structural_similarity",
]
