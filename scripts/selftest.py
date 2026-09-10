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

    # ALPHA CANCELS FROM EVERY RELATIVE QUANTITY, and this is pinned here so that
    # nobody later tunes the refraction increment hoping to improve dry_mass_mape.
    # The same constant lambda/(2 pi alpha) multiplies the predicted mass and the
    # reference mass, so it survives in absolute picograms and vanishes from any
    # ratio. A sweep of alpha against MAPE would return a flat line by
    # arithmetic, not by coincidence.
    from dataclasses import replace

    low, high = cfg.optics.refraction_increment_range_ml_per_g
    shifted = replace(calibration, refraction_increment=float(high))
    shifted_cells = measure_cells(
        phase, mask, shifted, cfg.evaluation.measurement,
        cfg.evaluation.segmentation.instance_from,
        cfg.mask_generation.watershed_min_distance_px,
    )
    expected_scale = calibration.refraction_increment / float(high)
    absolute_ratio = shifted_cells[0]["dry_mass_pg"] / cells[0]["dry_mass_pg"]
    check(
        "absolute dry mass scales as 1/alpha",
        abs(absolute_ratio - expected_scale) < 1e-6,
        f"alpha {calibration.refraction_increment:.3f} -> {float(high):.3f} "
        f"scales mass by {absolute_ratio:.4f}",
    )
    # A relative error computed entirely at the shifted alpha must equal the one
    # computed at the configured alpha, for any prediction whatsoever.
    predicted = np.array([c["dry_mass_pg"] for c in cells]) * 0.8
    reference = np.array([c["dry_mass_pg"] for c in cells])
    mape_base = float(np.mean(np.abs(predicted - reference) / reference))
    predicted_shifted = predicted * expected_scale
    reference_shifted = reference * expected_scale
    mape_shifted = float(np.mean(
        np.abs(predicted_shifted - reference_shifted) / reference_shifted
    ))
    check(
        "relative dry-mass error is invariant to alpha",
        abs(mape_base - mape_shifted) < 1e-12,
        f"MAPE {mape_base:.6f} at both alphas -- alpha cannot explain mass error",
    )
    check(
        "the configured alpha lies inside the literature range",
        float(low) <= calibration.refraction_increment <= float(high),
        f"{calibration.refraction_increment:.3f} in [{float(low):.3f}, {float(high):.3f}]",
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
    # Iterate over whatever the configured objective actually produced rather
    # than a fixed list: the v2 defaults switch several v1 terms off, and a
    # hard-coded name would turn a deliberate configuration change into a test
    # failure.
    for name, value in sorted(components.items()):
        if name == "total":
            continue
        check(f"component '{name}' is finite", math.isfinite(value), f"{value:.5f}")
    active = [n for n in components if n != "total"]
    check("the configured objective produced at least the supervised terms",
          {"phase", "segmentation"} <= set(active),
          f"active: {', '.join(sorted(active))}")

    check("phase term vanishes on an exact reconstruction", components["phase"] < 1e-4)
    for name in ("phase_volume", "cell_integrated_phase"):
        if name in components:
            check(f"'{name}' vanishes on an exact prediction",
                  components[name] < 1e-2, f"{components[name]:.5f}")

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
    from holoqpi.physics import (
        detrend_polynomial,
        form_hologram,
        phase_skewness,
        propagate,
        resolve_conjugate,
        unwrap_phase_2d,
    )
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
    forward_cfg = type("Cfg", (), {
        "distance_um": 200.0, "learn_distance": False, "criterion": "l2",
        "fit_radiometry": True, "feature_um": 1.0, "pad_px": 0, "border_px": 8,
        "dc_exclusion_frac": 0.13, "dc_exclusion_px": None,
    })()
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

    # ---- the four checks the forward model must pass to be trusted ----
    # Recovery must be exact through the FULL operator, and must not depend on
    # camera gain or black level: those are fitted, not assumed. A term that
    # changed when the exposure changed would be measuring the camera.
    from holoqpi.losses.terms import ForwardModelConsistency

    forward_cfg = type("Cfg", (), {
        "distance_um": 200.0, "learn_distance": False, "criterion": "l2",
        "fit_radiometry": True, "feature_um": 1.0, "pad_px": 0, "border_px": 8,
        "dc_exclusion_frac": 0.13, "dc_exclusion_px": None,
    })()
    consistency = ForwardModelConsistency(forward_cfg, optics)
    carrier_pair = (torch.tensor([0.11]), torch.tensor([0.09]))

    for geometry, carrier_arg in (("gabor", None), ("off_axis", carrier_pair)):
        truth = form_hologram(phase_t, amplitude_t, geometry,
                              wavelength, dx, dy, 200.0, carrier=carrier_arg)
        plain = 1.0 * truth
        scaled = 37.0 * truth + 120.0          # arbitrary gain and black level

        exact = float(consistency(phase_t, amplitude_t, plain, geometry).mean())
        exact_scaled = float(consistency(phase_t, amplitude_t, scaled, geometry).mean())
        degraded = float(consistency(phase_t * 0.5, amplitude_t, scaled, geometry).mean())
        empty = float(consistency(torch.zeros_like(phase_t), amplitude_t, scaled, geometry).mean())

        tolerance = 1e-6 if geometry == "gabor" else 2e-2
        check(f"{geometry}: residual vanishes on the true field",
              exact < tolerance, f"{exact:.2e}")
        check(f"{geometry}: residual is invariant to camera gain and offset",
              abs(exact - exact_scaled) < max(tolerance, 1e-6), 
              f"{exact:.2e} vs {exact_scaled:.2e}")
        check(f"{geometry}: residual grows as the phase degrades",
              exact < degraded < empty,
              f"{exact:.4f} < {degraded:.4f} < {empty:.4f}")

        probe = (phase_t * 0.5).clone().requires_grad_(True)
        consistency(probe, amplitude_t, scaled, geometry).sum().backward()
        check(f"{geometry}: gradient is finite and non-zero",
              probe.grad is not None and torch.isfinite(probe.grad).all()
              and float(probe.grad.abs().sum()) > 0)

    # z must be recoverable by gradient descent, or `learn_distance` is a lie.
    learnable_cfg = type("Cfg", (), {
        "distance_um": 140.0, "learn_distance": True, "criterion": "l2",
        "fit_radiometry": True, "feature_um": 1.0, "pad_px": 0, "border_px": 8,
        "dc_exclusion_frac": 0.13, "dc_exclusion_px": None,
    })()
    refinable = ForwardModelConsistency(learnable_cfg, optics)
    target = form_hologram(phase_t, amplitude_t, "gabor", wavelength, dx, dy, 200.0)
    optimiser = torch.optim.Adam(refinable.parameters(), lr=4.0)
    for _ in range(80):
        optimiser.zero_grad()
        refinable(phase_t, amplitude_t, target, "gabor").mean().backward()
        optimiser.step()
    recovered = float(refinable.distance.detach())
    check("z is recoverable by gradient descent from a wrong start",
          abs(recovered - 200.0) < 5.0, f"140.0 -> {recovered:.2f} um (true 200.0)")

    # THE CANCELLATION THE PER-CELL TERM EXISTS TO CATCH.
    # Two cells, one over-measured by 20% and one under-measured by 20%. The
    # image-level phase-volume term sums them and scores zero; the per-cell term
    # must not. If this check ever fails, experiment B is measuring nothing.
    from holoqpi.losses.terms import CellIntegratedPhase, PhaseVolumePreservation

    field = torch.zeros(1, 1, 64, 64)
    labels = torch.zeros(1, 64, 64, dtype=torch.long)
    field[0, 0, 8:24, 8:24] = 1.0
    labels[0, 8:24, 8:24] = 1
    field[0, 0, 40:56, 40:56] = 1.0
    labels[0, 40:56, 40:56] = 2

    truth = field.clone()
    skewed = field.clone()
    skewed[0, 0, 8:24, 8:24] *= 1.2          # +20% on cell 1
    skewed[0, 0, 40:56, 40:56] *= 0.8        # -20% on cell 2
    ones = torch.ones_like(field)

    image_level = PhaseVolumePreservation(1e-4)(ones, skewed, ones, truth)
    per_cell = CellIntegratedPhase(1e-4, 1.0, None)(ones, skewed, ones, truth, labels)
    check("image-level phase volume is blind to +20%/-20% cancelling cells",
          float(image_level) < 1e-6, f"loss {float(image_level):.2e}")
    check("per-cell integrated phase catches it",
          abs(float(per_cell) - 0.2) < 1e-4, f"loss {float(per_cell):.4f} (expected 0.2000)")

    exact = CellIntegratedPhase(1e-4, 1.0, None)(ones, truth, ones, truth, labels)
    check("per-cell integrated phase vanishes on an exact prediction",
          float(exact) < 1e-6, f"{float(exact):.2e}")

    graded = CellIntegratedPhase(1e-4, 1.0, None)(
        ones, truth * 1.1, ones, truth, labels)
    check("per-cell integrated phase scales with the error",
          abs(float(graded) - 0.1) < 1e-4, f"10% error -> {float(graded):.4f}")

    tiny = CellIntegratedPhase(1e-4, 1e9, None)(ones, skewed, ones, truth, labels)
    check("cells below the reference floor are skipped",
          float(tiny) == 0.0, "all cells excluded -> zero, not a division blow-up")

    probe = (truth * 1.1).clone().requires_grad_(True)
    CellIntegratedPhase(1e-4, 1.0, None)(ones, probe, ones, truth, labels).mean().backward()
    check("per-cell integrated phase is differentiable w.r.t. phase",
          probe.grad is not None and torch.isfinite(probe.grad).all()
          and float(probe.grad.abs().sum()) > 0)

    # The off-axis residual must not depend on the global piston phase, which
    # is unmeasurable and cycles completely every half wavelength of z. Without
    # the quadrature components this test swings over most of the residual's
    # range for a sub-micrometre change in distance.
    piston_cfg = type("Cfg", (), {
        "distance_um": 40.0, "learn_distance": False, "criterion": "l2",
        "fit_radiometry": True, "feature_um": 1.0, "pad_px": 0, "border_px": 8,
        "dc_exclusion_frac": 0.13, "dc_exclusion_px": None,
        "warn_on_short_pad": False,
    })()
    carrier_y = torch.full((1,), 0.11)
    carrier_x = torch.full((1,), 0.09)
    measured = form_hologram(phase_t, amplitude_t, "off_axis", wavelength, dx, dy,
                             40.0, carrier=(carrier_y, carrier_x))
    swings = []
    for step in range(5):
        shifted = type("Cfg", (), {**piston_cfg.__class__.__dict__,
                                   "distance_um": 40.0 + step * wavelength / 4})()
        swings.append(float(ForwardModelConsistency(shifted, optics)(
            phase_t, amplitude_t, measured, "off_axis").mean()))
    swing = max(swings) - min(swings)
    check("off_axis: residual is invariant to the unmeasurable piston phase",
          swing < 0.02, f"range {swing:.5f} over one wavelength of z")

    # The conjugate sideband must be resolved by the physical prior, and it must
    # be resolved on the DETRENDED phase. A synthetic field of Gaussian "cells"
    # on a strong quadratic bowl reproduces the situation exactly: the bowl's own
    # skewness is larger than the cells', so a test that skips detrending reads
    # the aberration instead of the specimen and answers backwards.
    grid = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    cells = torch.zeros(size, size)
    for cy, cx in ((-0.4, -0.3), (0.2, 0.5), (0.5, -0.6)):
        cells = cells + 3.0 * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.01)
    bowl = 18.0 * (xx ** 2 + yy ** 2) + 4.0 * xx - 3.0 * yy
    truth = (cells + bowl).view(1, 1, size, size)
    upright = torch.polar(torch.ones_like(truth), truth)

    fixed, flipped, skews = resolve_conjugate(upright, detrend_order=2, min_skewness=0.0)
    check("conjugate rule leaves a correctly signed field alone",
          flipped == [False] and skews[0] > 0, f"skewness {skews[0]:+.2f}")

    fixed, flipped, _ = resolve_conjugate(upright.conj(), detrend_order=2, min_skewness=0.0)
    recovered = unwrap_phase_2d(torch.angle(fixed))
    reference = unwrap_phase_2d(torch.angle(upright))
    agreement = float(torch.corrcoef(torch.stack([
        detrend_polynomial(recovered[0, 0], 2).flatten(),
        detrend_polynomial(reference[0, 0], 2).flatten(),
    ]))[0, 1])
    check("conjugate rule recovers a conjugated field", flipped == [True] and agreement > 0.99,
          f"flipped, detrended r = {agreement:.4f}")

    # A surface whose own skewness opposes the cells' defeats a test that does
    # not detrend, which is what the real data does: measured on this dataset,
    # the raw unwrapped skewness picks the wrong sideband on 13 of 13 fields.
    domed = (cells - 18.0 * (xx ** 2 + yy ** 2) + 4.0 * xx).view(1, 1, size, size)
    naive = domed[0, 0] - domed[0, 0].mean()
    raw_skew = float(((naive / naive.std()) ** 3).mean())
    detrended_skew = phase_skewness(domed[0, 0], 2)
    check("a surface's own skewness answers backwards without detrending",
          raw_skew < 0 < detrended_skew,
          f"raw {raw_skew:+.2f} vs detrended {detrended_skew:+.2f}")

    try:
        phase_skewness(truth[0, 0], 1)
        check("detrend order below 2 is refused", False, "no error raised")
    except ValueError:
        check("detrend order below 2 is refused", True)

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
