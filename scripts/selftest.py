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
    metrics.update_image(pairs, reference, reference)
    identical = metrics.compute()

    check("all cells pair with themselves", len(pairs) == len(reference),
          f"{len(pairs)}/{len(reference)}")
    check("dry-mass MAPE is zero", identical["dry_mass_mape"] < 1e-9)
    check("area MAPE is zero", identical["area_mape"] < 1e-9)
    check("detection recall is one on an identical prediction",
          abs(identical["detection_recall"] - 1.0) < 1e-9,
          f"{identical['detection_recall']:.4f}")
    check("detection precision is one on an identical prediction",
          abs(identical["detection_precision"] - 1.0) < 1e-9)
    check("nothing is recorded as missed or invented",
          identical["cells_missed"] == 0 and identical["cells_false_positive"] == 0)
    check("Bland-Altman bias lies inside its own limits of agreement",
          identical["dry_mass_loa_lower"] - 1e-9
          <= identical["dry_mass_relative_bias"]
          <= identical["dry_mass_loa_upper"] + 1e-9,
          f"bias {identical['dry_mass_relative_bias']:+.4f} in "
          f"[{identical['dry_mass_loa_lower']:+.4f}, {identical['dry_mass_loa_upper']:+.4f}]")

    eroded = binary_erosion(mask > 0, iterations=3).astype(np.int64)
    eroded_cells = measure(eroded)
    eroded_labels = split_instances((eroded > 0).astype(np.uint8), method, distance)

    degraded = MeasurementMetrics()
    eroded_pairs = match_cells(eroded_cells, reference, eroded_labels, labels,
                               cfg.evaluation.measurement.match_iou_threshold)
    degraded.update_pairs(eroded_pairs)
    degraded.update_image(eroded_pairs, eroded_cells, reference)
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

    # A detector that finds nothing must score zero recall, not a flattering
    # measurement error over an empty matched set.
    empty = MeasurementMetrics()
    blank = np.zeros_like(mask)
    blank_cells = measure(blank)
    blank_pairs = match_cells(blank_cells, reference,
                              split_instances((blank > 0).astype(np.uint8), method, distance),
                              labels, cfg.evaluation.measurement.match_iou_threshold)
    empty.update_pairs(blank_pairs)
    empty.update_image(blank_pairs, blank_cells, reference)
    nothing = empty.compute()
    check("an empty prediction scores zero detection recall",
          nothing["detection_recall"] == 0.0, f"{nothing['detection_recall']:.3f}")
    check("every reference cell is counted as missed",
          nothing["cells_missed"] == len(reference),
          f"{nothing['cells_missed']}/{len(reference)}")


def test_phase_masked_errors(cfg) -> None:
    print("\n[5] phase error inside cells versus background")
    phase, mask = synthetic_field()

    # A prediction wrong only inside the cells must show that in the in-cell
    # figures and nowhere else, or the field-wide numbers hide the failure that
    # matters to dry mass.
    prediction = phase.copy()
    prediction[mask > 0] += 0.5

    metrics = PhaseMetrics(cfg.evaluation.phase.psnr_data_range)
    metrics.update(prediction[None], phase[None], mask=mask[None])
    results = metrics.compute()

    check("in-cell bias recovers the injected offset",
          abs(results["phase_bias_rad_in_cell"] - 0.5) < 1e-5,
          f"{results['phase_bias_rad_in_cell']:.5f} rad")
    check("background is reported as clean",
          results["phase_mae_rad_background"] < 1e-9,
          f"{results['phase_mae_rad_background']:.2e} rad")
    check("the field-wide MAE understates the in-cell error",
          results["phase_mae_rad"] < results["phase_mae_rad_in_cell"],
          f"{results['phase_mae_rad']:.4f} vs {results['phase_mae_rad_in_cell']:.4f} rad")

    # And a 2-D array must be treated as one image, not as a stack of rows.
    single = PhaseMetrics(cfg.evaluation.phase.psnr_data_range)
    single.update(prediction, phase)
    check("a 2-D array counts as one image", single.compute()["phase_n_images"] == 1,
          f"{single.compute()['phase_n_images']}")


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


def test_physics(cfg) -> None:
    print("\n[6] forward model: propagation, hologram formation, consistency loss")
    from holoqpi.physics import form_hologram, propagate, unwrap_phase_2d
    from holoqpi.losses.terms import ForwardModelConsistency

    optics = cfg.optics
    wavelength = optics.wavelength_um
    dx, dy = optics.pixel_pitch_x_um, optics.pixel_pitch_y_um

    size = 128
    # A band-limited field, deliberately. A hard-edged disc carries energy above
    # the propagating cutoff, and those evanescent components genuinely do not
    # survive propagation, so a round trip could not return them and the test
    # would be measuring physics rather than the implementation.
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    smooth = 1.5 * np.exp(
        -(((grid_y - size / 2) ** 2 + (grid_x - size / 2) ** 2) / (2 * 14.0 ** 2))
    ).astype(np.float32)
    phase_t = torch.from_numpy(smooth).view(1, 1, size, size)
    amplitude_t = torch.ones_like(phase_t)

    # Propagating forward and back must return the original field: the angular
    # spectrum kernel is unitary on the propagating components, so a round trip
    # is the sharpest available check that the transfer function is correct.
    field = torch.polar(amplitude_t, phase_t)
    there = propagate(field, wavelength, dx, dy, 120.0)
    back = propagate(there, wavelength, dx, dy, -120.0)
    error = (back - field).abs().max()
    check("propagation round trip returns the field", float(error) < 1e-3,
          f"max |error| {float(error):.2e}")

    energy_ratio = float(there.abs().pow(2).sum() / field.abs().pow(2).sum())
    check("propagation conserves energy", abs(energy_ratio - 1.0) < 1e-3,
          f"ratio {energy_ratio:.6f}")

    identity = propagate(field, wavelength, dx, dy, 0.0)
    check("propagating by zero is the identity",
          float((identity - field).abs().max()) < 1e-5,
          f"max |error| {float((identity - field).abs().max()):.2e}")

    # The physical asymmetry the study is about. At zero defocus a pure phase
    # object produces no in-line intensity contrast, while the off-axis carrier
    # still records it. This is why the forward-model term can teach the
    # off-axis arm at any distance and the in-line arm only away from focus.
    carrier = (torch.tensor([0.12]), torch.tensor([0.10]))
    inline_at_zero = form_hologram(phase_t, amplitude_t, "gabor",
                                   wavelength, dx, dy, 0.0)
    check("in-line intensity is flat at zero defocus",
          float(inline_at_zero.std()) < 1e-5, f"std {float(inline_at_zero.std()):.2e}")

    inline_defocused = form_hologram(phase_t, amplitude_t, "gabor",
                                     wavelength, dx, dy, 200.0)
    check("in-line intensity gains contrast with defocus",
          float(inline_defocused.std()) > 1e-2,
          f"std {float(inline_defocused.std()):.4f}")

    off_axis_at_zero = form_hologram(phase_t, amplitude_t, "off_axis",
                                     wavelength, dx, dy, 0.0, carrier=carrier)
    check("off-axis carries phase at zero defocus",
          float(off_axis_at_zero.std()) > 1e-2,
          f"std {float(off_axis_at_zero.std()):.4f}")

    # The loss must vanish when the phase is exactly right and rise when it is not.
    overrides = {"distance_um": 200.0, "criterion": "correlation", "feature_um": 1.0,
                 "pad_px": 0, "border_px": 8, "reference_ratio": 1.0,
                 "dc_exclusion_px": 20}
    forward_cfg = type("Cfg", (), overrides)()
    term = ForwardModelConsistency(forward_cfg, optics)

    measured = form_hologram(phase_t, amplitude_t, "gabor",
                             wavelength, dx, dy, 200.0)
    exact = term(phase_t, amplitude_t, measured, "gabor")
    wrong = term(phase_t * 0.3, amplitude_t, measured, "gabor")
    check("forward-model loss vanishes on the true phase",
          float(exact.mean()) < 1e-4, f"{float(exact.mean()):.3e}")
    check("forward-model loss rises on a wrong phase",
          float(wrong.mean()) > float(exact.mean()) + 1e-3,
          f"{float(wrong.mean()):.4f} > {float(exact.mean()):.4f}")

    gradient_probe = phase_t.clone().requires_grad_(True)
    term(gradient_probe, amplitude_t, measured, "gabor").sum().backward()
    check("forward-model loss is differentiable w.r.t. phase",
          gradient_probe.grad is not None
          and torch.isfinite(gradient_probe.grad).all()
          and float(gradient_probe.grad.abs().sum()) > 0)

    # Unwrapping must undo a wrap it did not create.
    ramp = torch.linspace(0, 6 * math.pi, size).view(1, 1, 1, size).expand(1, 1, size, size)
    wrapped = torch.atan2(torch.sin(ramp), torch.cos(ramp))
    unwrapped = unwrap_phase_2d(wrapped)
    aligned = unwrapped - unwrapped.mean() + ramp.mean()
    correlation = float(
        torch.corrcoef(torch.stack([aligned.flatten(), ramp.flatten()]))[0, 1]
    )
    check("phase unwrapping recovers a multi-turn ramp", correlation > 0.99,
          f"r = {correlation:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"self-test using {args.config}")

    test_calibration(cfg)
    test_metrics_identity(cfg)
    test_measurement_agreement(cfg)
    test_phase_masked_errors(cfg)
    test_loss_terms(cfg)
    test_physics(cfg)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
