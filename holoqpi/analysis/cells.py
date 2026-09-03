"""Per-cell quantitative measurement from a phase map and a cell mask.

For each connected cell region Omega:

    projected area   A     = N_Omega * dx * dy                      [um^2]
    circularity      C     = 4 * pi * A / P^2                       [-]
    optical volume   V_phi = sum(phi_i) * dx * dy                   [rad * um^2]
    dry mass         m     = lambda / (2 * pi * alpha) * V_phi      [pg]

Dry mass is a fixed scalar multiple of the enclosed phase integral, which is why
the objective constrains that integral directly: a relative error in V_phi is
exactly the relative error in m.

Every constant is supplied by ``optics`` in the configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..config import Config


@dataclass(frozen=True)
class Calibration:
    """Optical constants converting integrated phase into picograms."""

    wavelength_um: float
    refraction_increment: float
    pitch_x_um: float
    pitch_y_um: float

    @property
    def pixel_area_um2(self) -> float:
        return self.pitch_x_um * self.pitch_y_um

    @property
    def picogram_per_radian_um2(self) -> float:
        """lambda / (2 * pi * alpha), with alpha in mL/g == um^3/pg."""
        return self.wavelength_um / (2.0 * math.pi * self.refraction_increment)

    @property
    def picogram_per_radian_pixel(self) -> float:
        return self.pixel_area_um2 * self.picogram_per_radian_um2

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            pixel_area_um2=self.pixel_area_um2,
            picogram_per_radian_um2=self.picogram_per_radian_um2,
            picogram_per_radian_pixel=self.picogram_per_radian_pixel,
        )
        return payload


def calibration_from_config(cfg: Config) -> Calibration:
    optics = cfg.optics
    return Calibration(
        wavelength_um=optics.wavelength_um,
        refraction_increment=optics.refraction_increment_ml_per_g,
        pitch_x_um=optics.pixel_pitch_x_um,
        pitch_y_um=optics.pixel_pitch_y_um,
    )


def _perimeter(binary: np.ndarray) -> float:
    """Crofton perimeter where scikit-image is available, else a contour count."""
    try:
        from skimage.measure import perimeter_crofton

        return float(perimeter_crofton(binary))
    except Exception:
        from scipy.ndimage import binary_erosion

        return float((binary & ~binary_erosion(binary)).sum())


def measure_cells(
    phase: np.ndarray,
    mask: np.ndarray,
    calibration: Calibration,
    cfg: Config,
    instance_method: str,
    watershed_min_distance: int,
    labels: np.ndarray | None = None,
) -> list[dict]:
    """Extract one measurement record per accepted cell.

    ``phase`` is in radians and ``mask`` is a binary or class-index map at the
    same resolution. Pass ``labels`` when instance labelling has already been
    done for this mask: watershed is the dominant cost of an evaluation pass and
    there is no reason to repeat it.
    """
    from scipy import ndimage

    from ..data.masks import split_instances

    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return []

    if labels is None:
        labels = split_instances(binary, instance_method, watershed_min_distance)
    pixel_area = calibration.pixel_area_um2
    pg_per_radian_pixel = calibration.picogram_per_radian_pixel

    minimum_area = cfg.min_cell_area_um2
    maximum_area = cfg.max_cell_area_um2

    # Work inside each component's bounding box rather than over the whole field.
    # A full-frame boolean pass per cell costs ~40x more on a 900 px image with
    # forty cells, and this runs twice per image during every validation epoch.
    boxes = ndimage.find_objects(labels)
    areas = np.bincount(labels.ravel())

    records: list[dict] = []
    for index, box in enumerate(boxes):
        label = index + 1
        if box is None:
            continue

        pixel_count = int(areas[label]) if label < areas.size else 0
        if pixel_count == 0:
            continue

        area_um2 = pixel_count * pixel_area
        if not (minimum_area <= area_um2 <= maximum_area):
            continue

        window = labels[box] == label
        phase_window = phase[box][window]

        phase_sum = float(phase_window.sum())
        perimeter_px = _perimeter(window)
        circularity = (
            4.0 * math.pi * pixel_count / (perimeter_px ** 2) if perimeter_px > 0 else float("nan")
        )

        offsets = np.argwhere(window)
        centroid_y = float(offsets[:, 0].mean()) + box[0].start
        centroid_x = float(offsets[:, 1].mean()) + box[1].start

        records.append(
            {
                "label": label,
                "area_px": pixel_count,
                "area_um2": area_um2,
                "perimeter_px": perimeter_px,
                "circularity": min(circularity, 1.0) if np.isfinite(circularity) else float("nan"),
                "mean_phase_rad": float(phase_window.mean()),
                "max_phase_rad": float(phase_window.max()),
                "optical_volume_rad_um2": phase_sum * pixel_area,
                "dry_mass_pg": phase_sum * pg_per_radian_pixel,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
            }
        )

    return records


def match_cells(
    predicted: list[dict],
    reference: list[dict],
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    iou_threshold: float,
) -> list[tuple[dict, dict]]:
    """Pair predicted cells with reference cells by greedy best IoU.

    Every record is stamped with ``match_iou`` (NaN when the cell found no
    partner), so the caller can separate matched from unmatched cells without
    repeating the overlap computation. Unmatched reference cells are misses and
    unmatched predicted cells are false positives; both are needed to report
    detection recall and precision rather than only the accuracy of the cells
    that happened to pair.
    """
    for record in predicted:
        record["match_iou"] = float("nan")
    for record in reference:
        record["match_iou"] = float("nan")

    if not predicted or not reference:
        return []

    predicted_areas = {r["label"]: r["area_px"] for r in predicted}
    reference_areas = {r["label"]: r["area_px"] for r in reference}
    predicted_by_label = {r["label"]: r for r in predicted}
    reference_by_label = {r["label"]: r for r in reference}

    stride = int(predicted_labels.max()) + 1
    overlap = (predicted_labels > 0) & (reference_labels > 0)
    if not overlap.any():
        return []
    pair_counts = np.bincount(
        (reference_labels[overlap] * stride + predicted_labels[overlap]).ravel()
    )

    candidates: list[tuple[float, int, int]] = []
    for reference_label in reference_by_label:
        for predicted_label in predicted_by_label:
            index = reference_label * stride + predicted_label
            if index >= pair_counts.size:
                continue
            intersection = float(pair_counts[index])
            if intersection <= 0:
                continue
            union = (
                reference_areas[reference_label] + predicted_areas[predicted_label] - intersection
            )
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, reference_label, predicted_label))

    candidates.sort(reverse=True)
    used_reference: set[int] = set()
    used_predicted: set[int] = set()
    pairs: list[tuple[dict, dict]] = []

    for iou, reference_label, predicted_label in candidates:
        if reference_label in used_reference or predicted_label in used_predicted:
            continue
        used_reference.add(reference_label)
        used_predicted.add(predicted_label)
        predicted_by_label[predicted_label]["match_iou"] = float(iou)
        reference_by_label[reference_label]["match_iou"] = float(iou)
        pairs.append((predicted_by_label[predicted_label], reference_by_label[reference_label]))

    return pairs
