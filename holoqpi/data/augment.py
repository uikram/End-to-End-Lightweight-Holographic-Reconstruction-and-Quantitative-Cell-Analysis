"""Geometry-only augmentation.

A hologram encodes optical path length in the position and contrast of its
fringes. Brightness, contrast, noise and blur augmentation would therefore
change the quantity the network is being asked to measure, so only isometries
of the sampling grid are permitted. Every transform is applied identically to
the hologram, the phase target and the mask.
"""

from __future__ import annotations

import random

import numpy as np

from ..config import Config


class GeometricAugmentation:
    """Random flips and quarter-turns shared across all three arrays."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.enabled = cfg.enabled
        self.p_horizontal = cfg.horizontal_flip
        self.p_vertical = cfg.vertical_flip
        self.p_rot90 = cfg.rot90
        self._rng = random.Random(seed)

    def __call__(
        self, hologram: np.ndarray, phase: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.enabled:
            return hologram, phase, mask

        arrays = [hologram, phase, mask]

        if self._rng.random() < self.p_horizontal:
            arrays = [np.flip(a, axis=-1) for a in arrays]
        if self._rng.random() < self.p_vertical:
            arrays = [np.flip(a, axis=-2) for a in arrays]
        if self._rng.random() < self.p_rot90:
            turns = self._rng.choice((1, 2, 3))
            arrays = [np.rot90(a, k=turns, axes=(-2, -1)) for a in arrays]

        return tuple(np.ascontiguousarray(a) for a in arrays)


def random_crop(
    arrays: list[np.ndarray], size: int, rng: random.Random
) -> list[np.ndarray]:
    """Take the same random square window from every array."""
    height, width = arrays[0].shape[-2:]
    if size >= min(height, width):
        return arrays
    top = rng.randint(0, height - size)
    left = rng.randint(0, width - size)
    return [a[..., top:top + size, left:left + size] for a in arrays]


def center_crop(arrays: list[np.ndarray], size: int) -> list[np.ndarray]:
    """Take the same centred square window from every array."""
    height, width = arrays[0].shape[-2:]
    if size >= min(height, width):
        return arrays
    top = (height - size) // 2
    left = (width - size) // 2
    return [a[..., top:top + size, left:left + size] for a in arrays]
