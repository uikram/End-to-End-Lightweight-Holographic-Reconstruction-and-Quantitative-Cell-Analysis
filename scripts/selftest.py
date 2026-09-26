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

    # ---- the hole coverage adjustment exists to close -------------------
    # A detector that reports only the cells it is most confident about gets a
    # flattering matched-cell MAPE. Here half the reference cells are dropped
    # and the survivors are measured EXACTLY, so the matched MAPE stays at zero
    # while the coverage-adjusted one must rise to the fraction that was lost.
    half = MeasurementMetrics()
    kept = reference[: max(1, len(reference) // 2)]
    kept_labels = np.where(np.isin(labels, [c["label"] for c in kept]), labels, 0)
    kept_cells = measure((kept_labels > 0).astype(np.int64))
    kept_pairs = match_cells(kept_cells, reference, kept_labels, labels,
                             cfg.evaluation.measurement.match_iou_threshold)
    half.update_pairs(kept_pairs)
    half.update_image(kept_pairs, kept_cells, reference)
    selective = half.compute()
    coverage = selective["coverage"]
    check("a selective detector still scores a near-zero matched MAPE",
          selective["dry_mass_mape"] < 1e-6,
          f"matched MAPE {selective['dry_mass_mape']:.2e} at coverage {coverage:.2f}")
    check("coverage adjustment charges it for the cells it never reported",
          abs(selective["dry_mass_mape_coverage_adjusted"] - (1.0 - coverage)) < 1e-6,
          f"adjusted {selective['dry_mass_mape_coverage_adjusted']:.4f} "
          f"= 1 - coverage = {1.0 - coverage:.4f}")
    check("the field total falls by the mass of the dropped cells",
          selective["dry_mass_field_total_bias"] < -1e-3,
          f"bias {selective['dry_mass_field_total_bias']:+.4f}")
    check("an identical prediction has no field-total bias",
          abs(identical["dry_mass_field_total_bias"]) < 1e-9)

    # ---- the field bootstrap -------------------------------------------
    # One field cannot be resampled, so the interval must be absent rather
    # than a zero-width one masquerading as certainty. With several fields it
    # must appear and must bracket the point estimate.
    single = MeasurementMetrics(field_bootstrap_resamples=200)
    single.update_pairs(eroded_pairs)
    single.update_image(eroded_pairs, eroded_cells, reference)
    one_field = single.compute()
    check("no bootstrap interval is reported from a single field",
          "dry_mass_mape_ci_lower" not in one_field)

    several = MeasurementMetrics(field_bootstrap_resamples=200)
    for _ in range(5):
        several.update_pairs(eroded_pairs)
        several.update_image(eroded_pairs, eroded_cells, reference)
    bootstrapped = several.compute()
    check("the field bootstrap reports an interval once there are fields to draw",
          "dry_mass_mape_ci_lower" in bootstrapped,
          f"{bootstrapped.get('dry_mass_bootstrap_fields')} fields")
    check("the interval brackets the point estimate",
          bootstrapped["dry_mass_mape_ci_lower"] - 1e-9
          <= bootstrapped["dry_mass_mape"]
          <= bootstrapped["dry_mass_mape_ci_upper"] + 1e-9,
          f"{bootstrapped['dry_mass_mape_ci_lower']:.4f} <= "
          f"{bootstrapped['dry_mass_mape']:.4f} <= "
          f"{bootstrapped['dry_mass_mape_ci_upper']:.4f}")
    check("pooling fields does not change the point estimate",
          abs(bootstrapped["dry_mass_mape"] - eroded_results["dry_mass_mape"]) < 1e-9,
          "grouping by field is bookkeeping, not a different statistic")


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


def test_amplitude_metric(cfg) -> None:
    """The amplitude metric must expose a collapsed head rather than flatter it.

    The failure this guards against is specific: an amplitude head that has
    learned nothing predicts 1 everywhere, which is the thin-phase-object
    assumption, and a bare MAE against a reference whose background is also 1
    looks respectable for it. The ratio against that assumption is the row that
    settles it, so the ratio has to be exactly 1 for a collapsed head and below
    1 for a head carrying real structure.
    """
    print("\n[5b] amplitude metric against the thin-phase assumption")
    from holoqpi.metrics import AmplitudeMetrics

    _, mask = synthetic_field()
    # A reference that departs from unity only inside the cells, which is what a
    # transmitting specimen does.
    reference = np.ones_like(mask, dtype=np.float64)
    reference[mask > 0] = 0.8

    collapsed = AmplitudeMetrics()
    collapsed.update(np.ones_like(reference)[None], reference[None], mask=mask[None])
    collapsed_results = collapsed.compute()
    check("a collapsed head scores exactly the A = 1 assumption",
          abs(collapsed_results["amplitude_mae_over_unity"] - 1.0) < 1e-12,
          f"ratio {collapsed_results['amplitude_mae_over_unity']:.6f}")
    check("and its predicted spread is reported as zero",
          collapsed_results["amplitude_pred_sd"] < 1e-12,
          f"sd {collapsed_results['amplitude_pred_sd']:.2e}")

    perfect = AmplitudeMetrics()
    perfect.update(reference[None], reference[None], mask=mask[None])
    perfect_results = perfect.compute()
    check("a correct head beats the A = 1 assumption",
          perfect_results["amplitude_mae_over_unity"] < 1e-12,
          f"ratio {perfect_results['amplitude_mae_over_unity']:.2e}")
    check("the in-cell error is reported separately",
          "amplitude_mae_in_cell" in perfect_results
          and perfect_results["amplitude_mae_in_cell"] < 1e-12,
          f"{perfect_results.get('amplitude_mae_in_cell')}")

    # A head biased high inside cells must be worse than assuming unity there,
    # so the sign of the comparison cannot be the other way round.
    biased = AmplitudeMetrics()
    prediction = np.ones_like(reference)
    prediction[mask > 0] = 1.4
    biased.update(prediction[None], reference[None], mask=mask[None])
    biased_results = biased.compute()
    check("a head biased the wrong way scores above 1.0",
          biased_results["amplitude_mae_over_unity"] > 1.0,
          f"ratio {biased_results['amplitude_mae_over_unity']:.4f}")
    check("an empty accumulator reports nothing rather than zeros",
          AmplitudeMetrics().compute() == {}, "{}")


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

    # THE SAME ARGUMENT FOR AREA, WHICH THE PHASE TERM CANNOT MAKE.
    # cell_integrated_phase constrains the mass of each cell, and mass is the
    # product of an area and a phase. A boundary pulled inwards while the phase
    # inside is scaled up leaves the product unchanged, so the phase term alone
    # cannot pin the morphology -- and projected area and circularity are
    # reported outputs in their own right. This term constrains the area
    # directly, per cell, over the SAME reference bins.
    from holoqpi.losses.terms import AmplitudeReconstructionLoss, CellProjectedArea

    area_term = CellProjectedArea(1e-4, 1.0, None)
    exact_area = area_term(ones, ones, labels)
    check("per-cell area vanishes when the foreground is exact",
          float(exact_area) < 1e-6, f"{float(exact_area):.2e}")

    # Half of each cell's foreground removed: 50% area error on both cells, and
    # unlike the mass term this cannot be hidden by compensating phase.
    thinned = ones.clone()
    thinned[0, 0, 8:16, 8:24] = 0.0
    thinned[0, 0, 40:48, 40:56] = 0.0
    halved = area_term(thinned, ones, labels)
    check("per-cell area catches a boundary that lost half of every cell",
          abs(float(halved) - 0.5) < 1e-4, f"loss {float(halved):.4f} (expected 0.5000)")

    # One cell wrong by 25% and the other exact averages to 0.125 -- the mean is
    # over CELLS, so a single bad cell costs a fixed share of the loss no matter
    # how large the field is. An image-level area term would divide the same
    # error by the whole field's foreground and all but lose it.
    lopsided = ones.clone()
    lopsided[0, 0, 8:12, 8:24] = 0.0          # -25% on cell 1, cell 2 untouched
    one_bad = area_term(lopsided, ones, labels)
    check("per-cell area averages over cells, so one bad cell is not diluted",
          abs(float(one_bad) - 0.125) < 1e-4,
          f"loss {float(one_bad):.4f} (expected 0.1250 = 0.25 / 2 cells)")

    small = CellProjectedArea(1e-4, 1e9, None)(thinned, ones, labels)
    check("cells below the reference pixel floor are skipped by the area term",
          float(small) == 0.0, "all cells excluded -> zero, not a division blow-up")

    area_probe = thinned.clone().requires_grad_(True)
    area_term(area_probe, ones, labels).mean().backward()
    check("per-cell area is differentiable w.r.t. the foreground probability",
          area_probe.grad is not None and torch.isfinite(area_probe.grad).all()
          and float(area_probe.grad.abs().sum()) > 0)

    amplitude_term = AmplitudeReconstructionLoss(type("Cfg", (), {"l1": 1.0})())
    check("amplitude loss vanishes on an exact prediction",
          float(amplitude_term(ones, ones).mean()) < 1e-9)
    check("amplitude loss is the mean absolute error",
          abs(float(amplitude_term(ones * 0.9, ones).mean()) - 0.1) < 1e-6,
          f"{float(amplitude_term(ones * 0.9, ones).mean()):.4f} (expected 0.1000)")

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


def test_mixed_precision(cfg) -> None:
    """Does the radiometric fit survive an autocast region?

    This is here because experiment D1 died at epoch 1 twice for the same
    reason. The Gram matrix in ForwardModelConsistency._fit_components is built
    by a matmul, and autocast re-casts a matmul's operands to float16 EVEN WHEN
    THEY ARE ALREADY float32 -- so `gram` came back Half while the right-hand
    side, built from elementwise ops that autocast leaves alone, stayed Float,
    and torch.linalg.solve refused the mismatch. Casting the inputs does not
    help; the region has to leave autocast.

    Training runs with training.mixed_precision true, so nothing else in the
    test suite exercises that path. These checks do.
    """
    print("\n--- mixed precision (the D1 crash) ---")
    from holoqpi.losses.terms import ForwardModelConsistency

    batch, terms, pixels = 2, 6, 64
    measured = torch.randn(batch, pixels, dtype=torch.float32)

    # First establish the hazard is real, so this test cannot pass vacuously if
    # a future torch stops down-casting.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        probe = torch.randn(batch, terms, pixels)
        gram_dtype = (probe.float() @ probe.float().transpose(1, 2)).dtype
    check("autocast does down-cast a float32 matmul (the hazard is real)",
          gram_dtype != torch.float32, f"gram came back {gram_dtype}")

    for dtype in (torch.float16, torch.bfloat16):
        components = torch.randn(batch, terms, pixels,
                                 dtype=torch.float16, requires_grad=True)
        failure = ""
        try:
            with torch.autocast(device_type="cpu", dtype=dtype):
                fitted, coefficients = ForwardModelConsistency._fit_components(
                    components, measured
                )
            residual = (fitted - measured).pow(2).mean()
            residual.backward()
            finite = bool(torch.isfinite(components.grad).all())
            nonzero = float(components.grad.abs().sum()) > 0.0
        except RuntimeError as exc:
            failure, finite, nonzero = str(exc)[:70], False, False
        check(f"radiometric fit runs under autocast {dtype}",
              not failure, failure)
        check(f"its gradient is finite and non-zero under autocast {dtype}",
              finite and nonzero)

    # Outside autocast the behaviour must be untouched.
    plain = torch.randn(batch, terms, pixels)
    fitted, coefficients = ForwardModelConsistency._fit_components(plain, measured)
    check("outside autocast the fit still returns float32",
          fitted.dtype == torch.float32 and coefficients.dtype == torch.float32)

    # The ridge only does its job if it is representable, which is the second
    # reason the solve is pinned to float32.
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        collinear = torch.ones(1, 3, 32, dtype=torch.float16)
        collinear[0, 1] += 1e-4
        _, coefficients = ForwardModelConsistency._fit_components(
            collinear, torch.randn(1, 32)
        )
    check("near-collinear components still give finite coefficients",
          bool(torch.isfinite(coefficients).all()))


def test_autocast_integrals(cfg) -> None:
    """Do the measurement terms survive autocast at this dataset's magnitudes?

    Nothing else in this file exercised that. Every other test here runs in
    float32 while training runs under ``training.mixed_precision``, so the terms
    were only ever checked at a precision they never see, on toy fields two
    orders of magnitude smaller than the real ones.

    THE HAZARD. ``scatter_add`` accumulates in the tensor's own dtype, and the
    per-cell integrals on this dataset are large (median 5964 px of area,
    5399 rad of integrated phase). A half-precision accumulator's spacing at
    4096 already exceeds the per-pixel increment, so the running total stops
    growing and every large cell reports the same saturated integral; an
    image-level integral overflows to ``inf`` outright on a 512 px crop.

    WHAT THIS DID AND DID NOT CATCH. On CUDA, autocast promotes ``softmax`` and
    ``sum`` to float32, so the GPU runs were already accumulating in float32 and
    no recorded result was affected -- see the long note on
    ``holoqpi.losses.terms._accumulation_dtype`` for the proof. The CPU autocast
    policy is a different, smaller list that does not include ``softmax``, and
    under it the same code RAISED. These checks pin the behaviour explicitly on
    both, so the study's central measurement no longer depends on which ops a
    given PyTorch release happens to promote.

    The magnitudes below are the measured ones, not convenient ones, and the
    first check establishes that a low-precision accumulator really does fail
    here so the rest cannot pass vacuously.
    """
    print("\n[7] measurement integrals under autocast, at real magnitudes")
    from holoqpi.losses.terms import (
        CellIntegratedPhase,
        CellProjectedArea,
        PhaseVolumePreservation,
    )

    size = 128
    # One cell of 6084 px carrying ~1 rad, i.e. an integral of ~6084 -- the
    # median cell on this dataset is 5964 px and 5399 rad.
    foreground = torch.zeros(1, 1, size, size)
    labels = torch.zeros(1, size, size, dtype=torch.long)
    foreground[0, 0, 25:103, 25:103] = 1.0
    labels[0, 25:103, 25:103] = 1
    phase = foreground.clone()
    cell_pixels = int(foreground.sum())

    # The hazard, demonstrated rather than asserted: a low-precision
    # accumulation of the same data must NOT reproduce the true sum.
    flat = foreground.reshape(-1)
    index = labels.reshape(-1)
    low = torch.zeros(2, dtype=torch.bfloat16).scatter_add(
        0, index, flat.to(torch.bfloat16)
    )
    check(
        "a half-precision accumulator does lose this integral (the hazard is real)",
        abs(float(low[1]) - cell_pixels) > 1.0,
        f"{cell_pixels} px accumulated as {float(low[1]):.0f}",
    )

    area_term = CellProjectedArea(1e-4, 1.0, None)
    phase_term = CellIntegratedPhase(1e-4, 1.0, None)
    volume_term = PhaseVolumePreservation(1e-4)

    # A 10% shortfall in both quantities, so the correct answer is exactly 0.1
    # and any accumulation error shows up directly in the number.
    thinned = foreground.clone()
    thinned[0, 0, 25:33, 25:103] = 0.0            # removes 8 of 78 rows
    expected_area = 1.0 - float(thinned.sum()) / cell_pixels

    reference_area = float(area_term(thinned, foreground, labels))
    reference_phase = float(phase_term(foreground, phase * 0.9, foreground, phase, labels))
    reference_volume = float(volume_term(foreground, phase * 0.9, foreground, phase))

    for dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cpu", dtype=dtype):
            # Cast the inputs the way the model's heads would hand them over.
            low_foreground = foreground.to(dtype)
            low_phase = phase.to(dtype)
            low_thinned = thinned.to(dtype)

            area = float(area_term(low_thinned, low_foreground, labels))
            integrated = float(
                phase_term(low_foreground, low_phase * 0.9, low_foreground, phase, labels)
            )
            volume = float(
                volume_term(low_foreground, low_phase * 0.9, low_foreground, phase)
            )

        check(
            f"per-cell area is exact under autocast {dtype}",
            abs(area - expected_area) < 2e-3,
            f"{area:.5f} vs {expected_area:.5f} in float32 {reference_area:.5f}",
        )
        check(
            f"per-cell integrated phase is exact under autocast {dtype}",
            abs(integrated - 0.1) < 2e-3 and abs(integrated - reference_phase) < 2e-3,
            f"{integrated:.5f} (expected 0.1000)",
        )
        check(
            f"image-level phase volume is finite and exact under autocast {dtype}",
            math.isfinite(volume) and abs(volume - reference_volume) < 2e-3,
            f"{volume:.5f} (expected {reference_volume:.5f})",
        )

    # A 512 px crop at this dataset's ~19% foreground holds ~50,000 cell pixels
    # and an integral near 70,000, which exceeds float16's largest finite value.
    # The term must not return inf.
    big = torch.zeros(1, 1, 512, 512)
    big[0, 0, :230, :230] = 1.0                   # 52,900 px
    big_phase = big * 1.4                         # integral ~74,000
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        overflowing = float(
            volume_term(big.half(), (big_phase * 0.9).half(), big, big_phase)
        )
    check(
        "a full-crop phase integral does not overflow under autocast float16",
        math.isfinite(overflowing) and abs(overflowing - 0.1) < 2e-3,
        f"{overflowing:.5f} on an integral of {float(big_phase.sum()):.0f} rad",
    )


def test_composite_under_autocast(cfg) -> None:
    """The whole objective, for every arm, inside an autocast region.

    The dtype crash that killed experiment D1 twice was only reachable by
    running an arm, because ``config/base.yaml`` switches every physics term off
    and the self-test only ever loaded base.yaml. So the configurations the study
    actually trains were never constructed here at all. These checks build each
    one and push a batch through its full objective under autocast, which is the
    combination that fails.
    """
    print("\n[8] every arm's objective, constructed and run under autocast")
    import torch.nn.functional as F

    from holoqpi.losses import build_loss

    arm_configs = sorted(Path("config/v2").glob("*.yaml"))
    check("the v2 arm configurations are present", bool(arm_configs),
          f"{len(arm_configs)} found under config/v2/")

    batch, size = 2, 96
    phase_map, mask = synthetic_field(size=size, radius=30, amplitude=2.0)
    phase_target = torch.from_numpy(phase_map).unsqueeze(0).unsqueeze(0).repeat(
        batch, 1, 1, 1
    )
    mask_target = torch.from_numpy(mask).unsqueeze(0).repeat(batch, 1, 1)
    instances = torch.from_numpy(
        split_instances((mask > 0).astype(np.uint8),
                        cfg.evaluation.segmentation.instance_from,
                        cfg.mask_generation.watershed_min_distance_px).astype(np.int64)
    ).unsqueeze(0).repeat(batch, 1, 1)

    for path in arm_configs:
        arm = load_config(str(path))
        # Floors that gate the physics terms are set for a 512 px crop; this
        # batch is 96 px, so lower them or every term is skipped and the arm is
        # not exercised at all.
        arm = arm.merged({
            "loss": {"physics": {"min_foreground_pixels": 1,
                                 "cell_min_reference_rad": 1.0}}
        })
        try:
            criterion = build_loss(arm)
        except Exception as exc:
            check(f"{path.name}: objective builds", False, f"{type(exc).__name__}: {exc}")
            continue

        targets = {
            "phase": phase_target,
            "mask": mask_target,
            "condition": torch.zeros(batch, dtype=torch.long),
            "instances": instances,
            "hologram_raw": torch.rand(batch, 1, size, size) * 200.0 + 20.0,
            "aberration": torch.zeros(batch, 1, size, size),
            "aberration_valid": torch.ones(batch, dtype=torch.bool),
        }
        if arm.data.provide_amplitude:
            targets["amplitude"] = torch.ones(batch, 1, size, size)

        failure = ""
        finite = False
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # Through convolutions, so the outputs carry the reduced dtype
                # the real heads would produce rather than a float32 literal.
                phase_out = F.conv2d(
                    phase_target, torch.ones(1, 1, 1, 1), padding=0
                ).requires_grad_(True)
                seg_out = F.conv2d(
                    torch.stack([(mask_target == 0).float(),
                                 (mask_target == 1).float()], dim=1) * 8.0,
                    torch.eye(2).view(2, 2, 1, 1),
                ).requires_grad_(True)
                outputs = {"phase": phase_out, "segmentation": seg_out}
                if arm.model.amplitude.enabled:
                    outputs["amplitude"] = torch.ones(
                        batch, 1, size, size, requires_grad=True
                    )
                if arm.model.classifier_enabled:
                    outputs["condition"] = torch.zeros(
                        batch, arm.model.condition_classes, requires_grad=True
                    )
                loss, components = criterion(outputs, targets)
            total = float(loss.detach())
            loss.backward()
            finite = (
                math.isfinite(total)
                and all(math.isfinite(v) for v in components.values())
                and phase_out.grad is not None
                and bool(torch.isfinite(phase_out.grad).all())
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"[:110]

        check(f"{path.name}: objective runs under autocast", not failure, failure)
        if not failure:
            check(f"{path.name}: loss and every component are finite", finite)


def test_learned_distance_domain(cfg) -> None:
    """Does the residual's comparison domain stay fixed while z moves?

    It did not. ``required_pad`` derived the diffraction margin from the live
    value of z, and ``border_px`` defaults to that margin, so every step z took
    changed WHICH PIXELS the residual was computed over. Gradient descent could
    then lower the residual by shrinking |z|, purely because a smaller border
    keeps a different set of pixels, and two residuals from different epochs were
    not on the same scale. "Learn z" only means something over one fixed domain.
    """
    print("\n[9] the learned-distance arm's residual domain")
    from holoqpi.losses.terms import ForwardModelConsistency

    settings = type("Cfg", (), {
        "distance_um": 33.77, "learn_distance": True, "criterion": "l2",
        "fit_radiometry": True, "feature_um": 1.0, "pad_px": None, "border_px": None,
        "dc_exclusion_frac": 0.13, "dc_exclusion_px": None, "warn_on_short_pad": False,
    })()
    term = ForwardModelConsistency(settings, cfg.optics)

    at_start = term.required_pad()
    with torch.no_grad():
        term.distance.fill_(-4.95)                 # where z actually ended up
    after_move = term.required_pad()
    with torch.no_grad():
        term.distance.fill_(120.0)
    after_big_move = term.required_pad()

    check("the diffraction pad is set by the configured distance, not the live one",
          at_start == after_move == after_big_move and at_start > 0,
          f"pad {at_start} px at z = 33.77, {after_move} at -4.95, "
          f"{after_big_move} at 120.0")
    check("and it is the value the configured distance implies",
          at_start == int(math.ceil(
              abs(cfg.optics.wavelength_um * 33.77) / 1.0
              / min(cfg.optics.pixel_pitch_x_um, cfg.optics.pixel_pitch_y_um))),
          f"{at_start} px")

    # An explicit pad_px must still win, because a z sweep needs one domain
    # across every candidate distance.
    explicit = type("Cfg", (), {**settings.__class__.__dict__, "pad_px": 64})()
    check("an explicit pad_px overrides the derived value",
          ForwardModelConsistency(explicit, cfg.optics).required_pad() == 64)

    # The criterion options must be genuinely different. `correlation` used to be
    # a verbatim copy of `l2`, so selecting it silently got l2.
    from holoqpi.physics import form_hologram

    grid_y, grid_x = np.mgrid[0:128, 0:128]
    smooth = 1.5 * np.exp(
        -(((grid_y - 64) ** 2 + (grid_x - 64) ** 2) / (2 * 14.0 ** 2))
    ).astype(np.float32)
    phase_t = torch.from_numpy(smooth).view(1, 1, 128, 128)
    amplitude_t = torch.ones_like(phase_t)
    measured = form_hologram(phase_t, amplitude_t, "gabor",
                             cfg.optics.wavelength_um, cfg.optics.pixel_pitch_x_um,
                             cfg.optics.pixel_pitch_y_um, 200.0)

    values = {}
    for name in ("l1", "l2", "correlation"):
        options = type("Cfg", (), {
            "distance_um": 200.0, "learn_distance": False, "criterion": name,
            "fit_radiometry": True, "feature_um": 1.0, "pad_px": 0, "border_px": 8,
            "dc_exclusion_frac": 0.13, "dc_exclusion_px": None,
            "warn_on_short_pad": False,
        })()
        built = ForwardModelConsistency(options, cfg.optics)
        values[name] = (
            float(built(phase_t, amplitude_t, measured, "gabor").mean()),
            float(built(phase_t * 0.3, amplitude_t, measured, "gabor").mean()),
        )

    for name, (exact, wrong) in values.items():
        check(f"criterion '{name}' vanishes on the true phase and rises on a wrong one",
              exact < 1e-3 and wrong > exact,
              f"exact {exact:.2e}, degraded {wrong:.4f}")
    check("'correlation' is not silently the same function as 'l2'",
          abs(values["correlation"][1] - values["l2"][1]) > 1e-6,
          f"correlation {values['correlation'][1]:.4f} vs l2 {values['l2'][1]:.4f} "
          f"on the same degraded phase")


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
    test_amplitude_metric(cfg)
    test_loss_terms(cfg)
    test_physics(cfg)
    test_mixed_precision(cfg)
    test_autocast_integrals(cfg)
    test_composite_under_autocast(cfg)
    test_learned_distance_domain(cfg)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED {len(_failures)} check(s): {_failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
