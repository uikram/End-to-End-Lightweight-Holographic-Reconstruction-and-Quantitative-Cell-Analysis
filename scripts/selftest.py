"""Self-test for the measurement chain and the loss terms.

Verifies the parts of the pipeline whose correctness cannot be judged by
inspecting a training curve:

* dry mass reduces to the analytic value on a synthetic cell of known phase
* an identical prediction scores perfectly on every metric family
* an eroded prediction degrades every metric in the expected direction
* each loss term is finite and differentiable

Run:  python scripts/selftest.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.analysis.cells import calibration_from_config, match_cells, measure_cells
from holoqpi.config import load_config
from holoqpi.data.masks import split_instances
from holoqpi.losses import build_loss
from holoqpi.metrics import MeasurementMetrics, PhaseMetrics, SegmentationMetrics

_PASS, _FAIL = "  PASS", "  FAIL"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{_PASS if condition else _FAIL}  {name}{f'  [{detail}]' if detail else ''}")
    if not condition:
        _failures.append(name)


def synthetic_field(size: int = 256, radius: int = 30, amplitude: float = 2.0):
    """Two disc-shaped cells of known area and uniform phase."""
    phase = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    for centre_y, centre_x in ((70, 70), (180, 180)):
        disc = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2 <= radius ** 2
        phase[disc] = amplitude
        mask[disc] = 1
    return phase, mask


def test_calibration(cfg) -> None:
    print("\n[1] calibration and dry mass")
    calibration = calibration_from_config(cfg)

    expected_prefactor = cfg.optics.wavelength_um / (
        2 * math.pi * cfg.optics.refraction_increment_ml_per_g
    )
    check(
        "prefactor equals lambda / (2 pi alpha)",
        abs(calibration.picogram_per_radian_um2 - expected_prefactor) < 1e-12,
        f"{calibration.picogram_per_radian_um2:.6f} pg/(rad*um^2)",
    )

    radius, amplitude = 30, 2.0
    phase, mask = synthetic_field(radius=radius, amplitude=amplitude)
    cells = measure_cells(
        phase, mask, calibration, cfg.evaluation.measurement,
        cfg.evaluation.segmentation.instance_from,
        cfg.mask_generation.watershed_min_distance_px,
    )
    check("two synthetic cells detected", len(cells) == 2, f"found {len(cells)}")
    if not cells:
        return

    pixel_count = cells[0]["area_px"]
    analytic_area = pixel_count * calibration.pixel_area_um2
    analytic_mass = pixel_count * amplitude * calibration.picogram_per_radian_pixel

    check(
        "area equals pixel count times dx*dy",
        abs(cells[0]["area_um2"] - analytic_area) < 1e-6,
        f"{cells[0]['area_um2']:.3f} um^2",
    )
    check(
        "dry mass equals the analytic integral",
        abs(cells[0]["dry_mass_pg"] - analytic_mass) < 1e-4,
        f"{cells[0]['dry_mass_pg']:.4f} pg",
    )
    check(
        "circularity of a disc is near unity",
        0.85 <= cells[0]["circularity"] <= 1.0,
        f"{cells[0]['circularity']:.3f}",
    )


def test_metrics_identity(cfg) -> None:
    print("\n[2] metrics on an identical prediction")
    phase, mask = synthetic_field()

    phase_metrics = PhaseMetrics(cfg.evaluation.phase.psnr_data_range)
    phase_metrics.update(phase[None], phase[None])
    phase_results = phase_metrics.compute()
    check("phase MAE is zero", phase_results["phase_mae_rad"] < 1e-6)
    check("phase Pearson r is one", abs(phase_results["phase_pearson_r"] - 1.0) < 1e-4)

    segmentation_metrics = SegmentationMetrics(
        num_classes=cfg.model.segmentation_classes,
        boundary_tolerance=cfg.evaluation.segmentation.boundary_tolerance_px,
        instance_method=cfg.evaluation.segmentation.instance_from,
        watershed_min_distance=cfg.mask_generation.watershed_min_distance_px,
    )
    segmentation_metrics.update(mask[None], mask[None])
    segmentation_results = segmentation_metrics.compute()
    check("Dice is one", abs(segmentation_results["seg_dice"] - 1.0) < 1e-4)
    check("AJI is one", abs(segmentation_results["seg_aji"] - 1.0) < 1e-4)
    check("boundary F1 is one", abs(segmentation_results["seg_boundary_f1"] - 1.0) < 1e-4)


def test_measurement_agreement(cfg) -> None:
    print("\n[3] measurement agreement, identical then eroded")
    from scipy.ndimage import binary_erosion

    calibration = calibration_from_config(cfg)
    phase, mask = synthetic_field()
    method = cfg.evaluation.segmentation.instance_from
    distance = cfg.mask_generation.watershed_min_distance_px

    def measure(m):
        return measure_cells(phase, m, calibration, cfg.evaluation.measurement, method, distance)

    reference = measure(mask)

    metrics = MeasurementMetrics()
    labels = split_instances((mask > 0).astype(np.uint8), method, distance)
    pairs = match_cells(reference, reference, labels, labels,
                        cfg.evaluation.measurement.match_iou_threshold)
    metrics.update_pairs(pairs)
    metrics.update_image(reference, reference)
    identical = metrics.compute()

    check("all cells pair with themselves", len(pairs) == len(reference),
          f"{len(pairs)}/{len(reference)}")
    check("dry-mass MAPE is zero", identical["dry_mass_mape"] < 1e-9)
    check("area MAPE is zero", identical["area_mape"] < 1e-9)

    eroded = binary_erosion(mask > 0, iterations=3).astype(np.int64)
    eroded_cells = measure(eroded)
    eroded_labels = split_instances((eroded > 0).astype(np.uint8), method, distance)

    degraded = MeasurementMetrics()
    degraded.update_pairs(
        match_cells(eroded_cells, reference, eroded_labels, labels,
                    cfg.evaluation.measurement.match_iou_threshold)
    )
    degraded.update_image(eroded_cells, reference)
    eroded_results = degraded.compute()

    check("eroded prediction loses area", eroded_results["area_mape"] > 0.01,
          f"MAPE {eroded_results['area_mape']:.3f}")
    check("eroded prediction loses dry mass", eroded_results["dry_mass_mape"] > 0.01,
          f"MAPE {eroded_results['dry_mass_mape']:.3f}")
    check(
        "area and mass degrade together for a uniform cell",
        abs(eroded_results["area_mape"] - eroded_results["dry_mass_mape"]) < 1e-6,
        "relative errors coincide as the physics predicts",
    )


def test_loss_terms(cfg) -> None:
    print("\n[4] loss terms: finite, differentiable, correctly signed")
    torch.manual_seed(cfg.project.seed)

    batch, size = 2, 128
    phase, mask = synthetic_field(size=size)
    phase_target = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)
    mask_target = torch.from_numpy(mask).unsqueeze(0).repeat(batch, 1, 1)

    criterion = build_loss(cfg)

    perfect_logits = torch.stack(
        [(mask_target == 0).float() * 10.0, (mask_target == 1).float() * 10.0], dim=1
    ).requires_grad_(True)
    perfect = {
        "phase": phase_target.clone().requires_grad_(True),
        "segmentation": perfect_logits,
        "condition": torch.zeros(batch, cfg.model.condition_classes, requires_grad=True),
    }
    targets = {
        "phase": phase_target,
        "mask": mask_target,
        "condition": torch.zeros(batch, dtype=torch.long),
    }

    loss, components = criterion(perfect, targets)
    check("total loss is finite", math.isfinite(components["total"]),
          f"{components['total']:.4f}")
    for name in ("phase", "segmentation", "phase_volume", "dry_mass_consistency",
                 "projected_area_consistency"):
        check(f"component '{name}' is finite", math.isfinite(components[name]),
              f"{components[name]:.5f}")

    check("phase term vanishes on an exact reconstruction", components["phase"] < 1e-4)
    check("phase-volume term vanishes on an exact prediction",
          components["phase_volume"] < 1e-2, f"{components['phase_volume']:.5f}")

    loss.backward()
    check("gradient reaches the phase head", perfect["phase"].grad is not None
          and torch.isfinite(perfect["phase"].grad).all())

    noisy = {
        "phase": phase_target + torch.randn_like(phase_target) * 0.5,
        "segmentation": torch.randn(batch, cfg.model.segmentation_classes, size, size),
        "condition": torch.randn(batch, cfg.model.condition_classes),
    }
    _, degraded = criterion(noisy, targets)
    check("a worse prediction scores a higher total loss",
          degraded["total"] > components["total"],
          f"{degraded['total']:.4f} > {components['total']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"self-test using {args.config}")

    test_calibration(cfg)
    test_metrics_identity(cfg)
    test_measurement_agreement(cfg)
    test_loss_terms(cfg)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
