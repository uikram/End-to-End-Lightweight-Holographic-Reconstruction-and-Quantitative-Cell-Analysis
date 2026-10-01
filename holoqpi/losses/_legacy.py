"""Deprecated names kept only so that old notebooks and scripts still import.

Current names: ``JointMeasurementLoss``, ``ImageIntegratedPhase`` (L_IPP^img),
``LegacyPhaseMaskContrast`` (weight 0 in every manuscript configuration).
Nothing in the package or in the manuscript pipeline uses the old names.
"""
import warnings

_MAP = {
    "JointPhysicsAwareLoss": ("composite", "JointMeasurementLoss"),
    "PhaseVolumePreservation": ("terms", "ImageIntegratedPhase"),
    "PhaseMaskContrast": ("terms", "LegacyPhaseMaskContrast"),
}


def resolve(name: str, module: str):
    if name in _MAP:
        sub, new = _MAP[name]
        warnings.warn(f"{name} is deprecated; use {new}", DeprecationWarning, stacklevel=3)
        import importlib
        return getattr(importlib.import_module(f"holoqpi.losses.{sub}"), new)
    raise AttributeError(f"module {module!r} has no attribute {name!r}")
