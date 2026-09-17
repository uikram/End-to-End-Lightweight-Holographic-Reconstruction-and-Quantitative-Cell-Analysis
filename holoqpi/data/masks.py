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

    # TRUNCATED-OBJECT EXCLUSION, made explicit rather than incidental.
    #
    # A cell crossing the field of view has a physically incomplete measurement:
    # its area and its integrated phase are both truncated by the sensor, not by
    # the specimen. Excluding such objects is standard practice and is justified
    # on its own terms, independently of any artefact argument.
    #
    # MEASURED, so the buffer is chosen rather than guessed:
    #   buffer 0 (touching the edge)  315 -> 315 cells,  0.0% lost, fg 19.13%
    #   buffer 4                      315 -> 236 cells, 25.1% lost, fg 14.16%
    #   buffer 8                      315 -> 232 cells, 26.3% lost, fg 13.87%
    #   buffer 34                     315 -> 204 cells, 35.2% lost, fg 12.20%
    #
    # Buffer 0 costs nothing today because binary_closing has already removed
    # everything that touches the edge (see the note below). It is kept anyway
    # so the guarantee is stated in code rather than inherited from a side
    # effect: change binary_closing_px to 0 and this step becomes load-bearing.
    #
    # A larger buffer is NOT justified here. The background phase does deviate
    # from its plateau out to about 34 px from the edge (median over 13 fields
    # with cells masked out: -0.50 rad at 2 px against a -0.167 rad plateau,
    # settling within 3 sd at 34 px), but using that as a buffer would discard a
    # third of all cells to suppress an artefact that demonstrably never reaches
    # these masks.
    #
    # ON THE MECHANISM, because it has been got wrong twice. Raw Otsu output
    # carries 4-11 border-touching components per field, the largest 309-1479
    # um^2, comfortably inside the area window. Measured stage by stage in the
    # order this function executes, they drop to ZERO immediately after
    # binary_closing, in 5 of 5 fields tested, before fill_holes and before the
    # area filter -- which runs last, not first. Closing is extensive on an
    # infinite domain, but scipy applies border_value=0 to its erosion step, so
    # at an array edge it is NOT extensive: a 12-px blob touching the top row
    # reduces to 4 px, and a 1-px rim vanishes entirely. That border convention,
    # not the area filter, is what removes them.
    if cfg.border_buffer_px is not None:
        from skimage.segmentation import clear_border

        labels, count = ndimage.label(binary)
        if count:
            binary = clear_border(labels, buffer_size=int(cfg.border_buffer_px)) > 0

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

    from ..utils import assert_provenance, read_provenance, write_provenance

    mask_cfg = cfg.mask_generation
    destination = data_root / cfg.paths.mask_dir
    destination.mkdir(parents=True, exist_ok=True)

    # EVERY PARAMETER THAT CHANGES A MASK, recorded beside the masks.
    #
    # These files are both the segmentation target and the domain for every
    # per-cell measurement, and existing ones are skipped rather than rebuilt --
    # so without this, changing a threshold and re-running left the whole study
    # measuring against masks from the previous configuration, with nothing on
    # disk to say so. The pixel area is included because it converts the two
    # area limits into pixel counts.
    parameters = {
        "smoothing_sigma_px": mask_cfg.smoothing_sigma_px,
        "threshold_method": mask_cfg.threshold_method,
        "fixed_threshold_rad": mask_cfg.fixed_threshold_rad,
        "otsu_scale": mask_cfg.otsu_scale,
        "min_object_area_um2": mask_cfg.min_object_area_um2,
        "max_object_area_um2": mask_cfg.max_object_area_um2,
        "fill_holes": mask_cfg.fill_holes,
        "binary_closing_px": mask_cfg.binary_closing_px,
        "border_buffer_px": mask_cfg.border_buffer_px,
        "pixel_area_um2": round(float(pixel_area_um2), 9),
    }
    if not mask_cfg.overwrite_existing:
        assert_provenance(
            destination, parameters,
            "Regenerate with:  python main.py prepare --config <cfg> --force-masks",
        )

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

    # ONLY WHEN THIS RUN ACTUALLY WROTE SOMETHING.
    #
    # Writing it unconditionally would stamp the current configuration onto files
    # produced by an unknown earlier one: the first run after this guard was
    # added finds no provenance file, correctly skips the check, skips all 800
    # existing masks, and would then certify them as matching -- after which the
    # guard passes forever and blesses exactly the staleness it exists to catch.
    if written:
        write_provenance(destination, parameters)
    elif skipped and read_provenance(destination) is None:
        # Only when there is genuinely no record. Once a --force-masks run has
        # written the provenance, a later skip-everything run is fine: the guard
        # above has already confirmed the parameters match, so repeating the
        # warning would be telling the user to fix something that is fixed.
        LOGGER.warning(
            "every mask already existed, so none was written and no provenance "
            "was recorded. %s cannot be attributed to a configuration; "
            "regenerate once with --force-masks so the settings that produced it "
            "are recorded.", destination,
        )

    stats = {
        "written": written,
        "skipped_existing": skipped,
        "parameters": parameters,
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
