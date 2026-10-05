"""Joint objective (measurement-aware terms, forward-model consistency) for hologram reconstruction and cell analysis."""

from .composite import JointMeasurementLoss, build_loss
from .terms import (
    BoundaryGradientAlignment,
    DryMassConsistency,
    ImageIntegratedPhase,
    LegacyPhaseMaskContrast,
    PhaseReconstructionLoss,
    ProjectedAreaConsistency,
    SegmentationLoss,
    spatial_gradient,
    structural_similarity,
)

__all__ = [
    "JointMeasurementLoss",
    "build_loss",
    "PhaseReconstructionLoss",
    "SegmentationLoss",
    "LegacyPhaseMaskContrast",
    "ImageIntegratedPhase",
    "BoundaryGradientAlignment",
    "DryMassConsistency",
    "ProjectedAreaConsistency",
    "spatial_gradient",
    "structural_similarity",
]


def __getattr__(name):  # deprecated names (see _legacy.py): importable with a warning, not exported
    from ._legacy import resolve
    return resolve(name, __name__)
