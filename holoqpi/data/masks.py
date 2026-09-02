"""Phase-derived segmentation masks.

The delivered dataset carries no manual cell annotations, so binary masks are
derived from the ground-truth quantitative phase: cells are the connected
regions whose optical path length rises measurably above the surrounding medium.
These are silver-standard labels and must be described as such in any write-up.

Setting ``paths.manual_mask_dir`` makes the loader read that directory instead,
with no other change anywhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)


def build_mask(phase: np.ndarray, cfg: Config, pixel_area_um2: float) -> np.ndarray:
    """Derive a binary cell mask (0 background, 1 cell) from a phase map."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smoothed = ndimage.gaussian_filter(phase.astype(np.float64), cfg.smoothing_sigma_px)

    if cfg.threshold_method == "otsu":
        level = float(threshold_otsu(smoothed)) * cfg.otsu_scale
    elif cfg.threshold_method == "fixed":
        level = float(cfg.fixed_threshold_rad)
    else:
        raise ValueError(f"unknown threshold_method {cfg.threshold_method!r}")

    binary = smoothed > level

    if cfg.binary_closing_px > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        binary = ndimage.binary_closing(binary, structure, iterations=cfg.binary_closing_px)

    if cfg.fill_holes:
        binary = ndimage.binary_fill_holes(binary)

    min_px = max(int(round(cfg.min_object_area_um2 / pixel_area_um2)), 1)
    max_px = int(round(cfg.max_object_area_um2 / pixel_area_um2))
    binary = filter_by_area(binary, min_px, max_px)

    return binary.astype(np.uint8)


def filter_by_area(binary: np.ndarray, min_px: int, max_px: int) -> np.ndarray:
    """Drop components outside a plausible single-cell footprint.

    The lower bound removes speckle, the upper bound removes debris and confluent
    sheets that no per-cell measurement could describe. Implemented directly on a
    label bincount so the behaviour does not track scikit-image API changes.
    """
    from scipy import ndimage

    labels, count = ndimage.label(binary)
    if count == 0:
        return binary

    sizes = np.bincount(labels.ravel())
    rejected = np.where((sizes < min_px) | (sizes > max_px))[0]
    rejected = rejected[rejected != 0]
    if rejected.size:
        binary = binary & ~np.isin(labels, rejected)
    return binary


def split_instances(binary: np.ndarray, method: str, min_distance_px: int) -> np.ndarray:
    """Label individual cells within a binary foreground.

    ``watershed`` separates touching cells using the distance transform, which
    matters for instance metrics and for per-cell measurements. Both the
    prediction and the reference are labelled with this same function so the
    comparison stays fair.
    """
    from scipy import ndimage

    binary = binary.astype(bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=np.int32)

    if method == "connected_components":
        labels, _ = ndimage.label(binary)
        return labels.astype(np.int32)

    if method != "watershed":
        raise ValueError(f"unknown instance method {method!r}")

    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    distance = ndimage.distance_transform_edt(binary)
    coordinates = peak_local_max(
        distance, min_distance=min_distance_px, labels=binary, exclude_border=False
    )
    markers = np.zeros(distance.shape, dtype=np.int32)
    if coordinates.size:
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    else:
        markers, _ = ndimage.label(binary)

    return watershed(-distance, markers, mask=binary).astype(np.int32)


def generate_masks(
    data_root: Path,
    cfg: Config,
    stems: list[str],
    pixel_area_um2: float,
) -> dict:
    """Write a mask for every stem; returns coverage statistics."""
    from . import io as data_io

    mask_cfg = cfg.mask_generation
    destination = data_root / cfg.paths.mask_dir
    destination.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    coverage: list[float] = []

    for position, stem in enumerate(stems, start=1):
        target = destination / f"{stem}{cfg.formats.mask.suffix}"
        if target.is_file() and not mask_cfg.overwrite_existing:
            skipped += 1
            continue

        record = data_io.read_phase_bin(
            data_io.phase_path(data_root, cfg, stem), cfg.formats.phase_binary
        )
        mask = build_mask(record.phase, mask_cfg, pixel_area_um2)
        data_io.write_mask(mask, target)

        coverage.append(float(mask.mean()))
        written += 1

        if position % 100 == 0:
            LOGGER.info("masks: %d/%d processed", position, len(stems))

    stats = {
        "written": written,
        "skipped_existing": skipped,
        "mean_foreground_fraction": float(np.mean(coverage)) if coverage else None,
        "min_foreground_fraction": float(np.min(coverage)) if coverage else None,
        "max_foreground_fraction": float(np.max(coverage)) if coverage else None,
        "directory": str(destination),
    }
    LOGGER.info(
        "masks written=%d skipped=%d mean foreground=%s",
        written, skipped,
        f"{stats['mean_foreground_fraction']:.3f}" if coverage else "n/a",
    )
    return stats
