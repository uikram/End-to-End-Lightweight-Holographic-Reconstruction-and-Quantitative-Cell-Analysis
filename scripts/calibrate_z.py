"""Recover the sample-to-sensor distance z from the data.

The forward-model consistency loss needs z, and z is not recorded anywhere: the
phase `.bin` header carries width, height and the two pixel pitches, and nothing
else. Asking the group that acquired the data is the right first move, but the
number is also recoverable, because the dataset already contains both halves of
the propagation: a raw hologram and the reference phase reconstructed from it.

Two estimators are computed for every candidate distance and each modality.

1. FORWARD.  Synthesise the hologram the reference phase would produce at
   distance z and score it with the *same* ForwardModelConsistency term the
   training loss uses -- including its per-image radiometric fit, so unknown
   camera gain and black level cannot masquerade as a distance. This is the
   criterion that matters, because it is literally the residual training
   minimises.

2. INVERSE.  Back-propagate the measured hologram by z and correlate the
   resulting phase with the reference phase. A cruder criterion: it assumes the
   hologram amplitude is sqrt(I) with zero phase, which is only reasonable
   in-line and ignores the twin image entirely. Reported as a weak cross-check,
   never as a veto.

IDENTIFIABILITY is decided by whether *individual images agree*, not by whether
the two estimators agree. Each image is scanned separately and the spread of its
own best z is reported: if eighty independent fields all place the minimum within
a few micrometres of each other, z is determined, whatever the weaker estimator
says. This replaces an earlier rule that required both estimators to agree, which
let the crude one veto the good one.

A NOTE ON THE TWO GEOMETRIES, which the results here will show plainly. An
off-axis hologram encodes phase in its carrier fringes at *any* distance, so its
residual is nearly flat in z and z is only weakly identifiable from it -- that is
a physical property of the geometry, not a failure of the search. An in-line
hologram encodes phase only through defocus, so its residual has a real minimum.
The same specimen was recorded at the same distance in both, so **use the Gabor
arm to determine z and apply it to both**.

    python scripts/calibrate_z.py --config config/base.yaml
    python scripts/calibrate_z.py --config config/base.yaml --z-range -400 400 --steps 81
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.physics import estimate_carrier, form_hologram, propagate
from holoqpi.utils import get_logger, resolve_device

LOGGER = get_logger(__name__)


def _standardise(x: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance per image.

    Illumination brightness, camera gain and exposure differ between the
    synthetic and the measured hologram for reasons that have nothing to do with
    z. Standardising both removes that nuisance affine factor so the comparison
    is about structure only.
    """
    flat = x.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).view(-1, 1, 1, 1).clamp(min=1e-8)
    return (x - mean) / std


def _correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _standardise(a), _standardise(b)
    return (a * b).flatten(1).mean(dim=1)


def usable_z_range_um(cfg, field_px: int, feature_um: float = 1.0) -> float:
    """Largest |z| whose diffraction still fits inside the field.

    The FFT is periodic, so light spreading further than the array wraps around
    and the synthesised hologram becomes meaningless. Light scattered by a
    feature of size ``feature_um`` spreads laterally by about
    lambda * z / feature_um; requiring that to stay within a quarter of the
    field bounds z. Using the Nyquist frequency instead of a real feature size
    would be far more conservative than the optics warrant, because cells do not
    carry structure at the pixel scale.
    """
    dx = min(cfg.optics.pixel_pitch_x_um, cfg.optics.pixel_pitch_y_um)
    budget_um = 0.25 * field_px * dx
    return budget_um * feature_um / cfg.optics.wavelength_um


def phase_sensitivity(cfg, modality: str, distance_um: float, phase, carrier) -> float:
    """How strongly the synthesised hologram responds to a change in phase.

    This decides whether the forward-model loss can teach the phase head
    anything at all, and the answer is not the same for the two geometries. At
    z = 0 a pure phase object produces no intensity contrast in-line -- the
    intensity is |A exp(i phi)|^2 = A^2, with phi absent -- so the Gabor forward
    model is degenerate and its gradient with respect to phase vanishes. The
    off-axis model stays informative at any distance, because the phase
    modulates the reference fringes directly. In-line holography needs defocus
    to encode phase; off-axis does not. A sensitivity near zero means the term
    will contribute nothing for that arm and should be reported, not trained.
    """
    optics = cfg.optics
    dx, dy = optics.pixel_pitch_x_um, optics.pixel_pitch_y_um
    amplitude = torch.ones_like(phase)
    base = form_hologram(phase, amplitude, modality, optics.wavelength_um, dx, dy,
                         distance_um, carrier=carrier)

    # Perturb the phase CONTRAST, not the absolute phase. A uniform phase offset
    # is unobservable in an in-line intensity by construction, so probing with a
    # constant would report zero sensitivity at every distance and say nothing
    # about whether the term can teach the phase head the cell structure.
    epsilon = 0.05
    mean_phase = phase.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
    perturbed = phase + epsilon * (phase - mean_phase)
    predicted = form_hologram(perturbed, amplitude, modality, optics.wavelength_um,
                              dx, dy, distance_um, carrier=carrier)

    # Normalise by mean intensity, not by the synthetic hologram's own contrast:
    # at z = 0 an in-line hologram of a pure phase object is flat, and dividing
    # by its (near-zero) standard deviation would report a huge sensitivity for
    # exactly the case that carries no phase information at all.
    change = (predicted - base).flatten(1).std(dim=1)
    scale = base.flatten(1).mean(dim=1).abs().clamp(min=1e-8)
    return float((change / scale).mean() / epsilon)


def _forward_term(cfg, distance_um: float, border_px: int | None = None):
    """The training residual, instantiated at one distance.

    ``border_px`` overrides the margin dropped from the residual. A sweep MUST
    pass a fixed value. Left to its default the border follows the pad, the pad
    follows z, and every distance is then scored on a different number of
    pixels: at 66 um the residual covers 29% of a 900 px field and at 34 um it
    covers 58%. Fewer, more central pixels are easier for four free radiometric
    coefficients to fit, so the residual falls with |z| for a reason that has
    nothing to do with focus, and the minimum runs to the edge of the scan.
    """
    from holoqpi.losses.terms import ForwardModelConsistency

    forward = cfg.loss.forward_model
    settings = type("Cfg", (), {
        "distance_um": float(distance_um),
        "learn_distance": False,
        "criterion": forward.criterion,
        "fit_radiometry": True,
        "feature_um": forward.feature_um,
        "pad_px": forward.pad_px,
        "border_px": forward.border_px if border_px is None else int(border_px),
        "dc_exclusion_frac": forward.dc_exclusion_frac,
        "dc_exclusion_px": forward.dc_exclusion_px,
        # The sweep reports the modellable range once, up front.
        "warn_on_short_pad": False,
    })()
    return ForwardModelConsistency(settings, cfg.optics)



def discrimination(cfg, modality: str, device, distance_um: float, split: str,
                   batches: int, tolerance: float, win_threshold: float,
                   border_px: int | None = None) -> dict:
    """Does the residual rise when the phase is degraded away from the truth?

    This, not the size of the residual, decides whether the forward-model term
    can be trained. The operator here is approximate -- the acquiring group's
    reconstruction runs inside closed acquisition software, so the exact
    demodulation, filtering and aberration model are unavailable and cannot be
    inverted -- which puts a floor under the residual that no prediction can
    beat. A floor is only a constant offset, and a constant does not affect a
    gradient.

    What does matter is the sign of the slope. If a degraded phase scores WORSE
    than the true phase, minimising the residual moves the prediction toward the
    truth and the term is a useful regulariser despite its floor. If a degraded
    phase scores BETTER, the term is anti-discriminative: training on it would
    actively drive the reconstruction away from the truth, and it must not be
    used at any weight.

    Six degradations, from mild to total. A usable term ranks them all above the
    reference, and the margin on the most severe of them is reported as the
    discrimination.
    """
    from holoqpi.data import build_dataloaders

    modality_cfg = cfg.merged({"data": {"modality": modality}})
    loader = build_dataloaders(modality_cfg, splits_to_build=(split,))[split]
    term = _forward_term(cfg, distance_um, border_px).to(device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.project.seed)

    totals: dict[str, list[float]] = {}
    seen = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            hologram = batch.get("hologram_raw", batch["hologram"]).to(device).float()
            phase = batch["phase"].to(device).float()
            surface = batch.get("aberration")
            if surface is not None:
                surface = surface.to(device).float()
            # Fields whose surface could not be recovered would dominate every
            # statistic here; drop them, as the loss does.
            usable = batch.get("aberration_valid")
            if usable is not None and not bool(usable.all()):
                keep = usable.to(torch.bool).reshape(-1)
                if not bool(keep.any()):
                    continue
                hologram, phase = hologram[keep], phase[keep]
                if surface is not None:
                    surface = surface[keep]
            amplitude = torch.ones_like(phase)
            seen += hologram.shape[0]

            noise = torch.randn(phase.shape, generator=generator).to(device)
            variants = {
                "reference": phase,
                "scaled_0.9": phase * 0.9,
                "scaled_0.5": phase * 0.5,
                "noisy_0.3": phase + 0.3 * noise,
                "mirrored": phase.flip(-1),
                "zero": torch.zeros_like(phase),
            }
            for name, candidate in variants.items():
                value = term(candidate, amplitude, hologram, modality, aberration=surface)
                totals.setdefault(name, []).extend(
                    value.detach().float().cpu().reshape(-1).tolist()
                )

    if seen == 0:
        return {"usable": False, "reason": "no images"}

    scores = {name: float(np.mean(values)) for name, values in totals.items()}
    floor = scores["reference"]

    # A MEAN OVER A HANDFUL OF FIELDS IS NOT A VERDICT. The residual varies more
    # between fields than between a true phase and a mildly degraded one, so a
    # four-image average is dominated by which four fields were read. The
    # decision is therefore made on a per-image sign test: in what fraction of
    # fields does the degraded phase score worse than the truth? Under no
    # discrimination that is one half, and it is insensitive to the field-to-
    # field spread that swamps the mean.
    reference = np.asarray(totals["reference"], dtype=float)
    margins, win_rates = {}, {}
    for name, values in totals.items():
        if name == "reference":
            continue
        difference = np.asarray(values, dtype=float) - reference
        margins[name] = float(difference.mean())
        win_rates[name] = float((difference > 0).mean())

    # Near and far degradations answer different questions. The far ones
    # (mirrored, zero) ask whether the term can tell a reconstruction from
    # nonsense at all. The near ones (0.9x, mild noise) ask whether its gradient
    # still points the right way once the prediction is already close, which is
    # where training actually spends its time and is the stricter test.
    near_keys, far_keys = ("scaled_0.9", "noisy_0.3"), ("mirrored", "zero")
    near = min(margins[k] for k in near_keys)
    far = min(margins[k] for k in far_keys)
    near_win = min(win_rates[k] for k in near_keys)
    far_win = min(win_rates[k] for k in far_keys)

    # ANTI-DISCRIMINATIVE MEANS THE SIGN IS WRONG, NOT THAT THE MARGIN IS SMALL.
    # An earlier version of this test fell through to that label whenever a
    # margin failed to clear the tolerance, and so reported "a degraded phase
    # scores BETTER than the truth" for data in which every degraded phase in
    # fact scored worse. A term whose margins are all positive but tiny is
    # uninformative, which is a different finding and has a different remedy.
    if near < -tolerance or far < -tolerance:
        verdict = "anti_discriminative"
    elif near > tolerance and far > tolerance and near_win >= win_threshold:
        verdict = "usable"
    elif far > tolerance and far_win >= win_threshold:
        verdict = "marginal"
    else:
        verdict = "uninformative"

    return {
        "usable": verdict == "usable",
        "verdict": verdict,
        "floor": floor,
        "images": int(seen),
        "near_margin": near,
        "far_margin": far,
        "near_win_rate": near_win,
        "far_win_rate": far_win,
        "tolerance": float(tolerance),
        "win_threshold": float(win_threshold),
        "scores": scores,
        "margins": margins,
        "win_rates": win_rates,
        "distance_um": float(distance_um),
    }


def phase_scale_response(cfg, modality: str, device, distance_um: float, split: str,
                         batches: int, border_px: int | None = None) -> dict:
    """At what multiple of the reference phase is the residual minimised?

    The discrimination test asks whether the residual rises when the phase is
    degraded. This asks a sharper question, and one the paper can quote: does the
    forward model agree with the delivered phase about its MAGNITUDE, not merely
    its structure? Scaling the reference phase by s and minimising over s gives a
    calibration factor. s = 1 means the operator reproduces the measured hologram
    at the phase the acquiring group reported, which is what measurement-ready
    means. s far from 1 is a quantified disagreement, not a vague caveat.

    The two geometries answer differently, and the difference is physical rather
    than numerical: an off-axis carrier records phase linearly, so its residual
    has a well near the true scale, while an in-line intensity carries phase only
    through defocus with the twin image superposed on the object, so the model
    accounts for a fraction of the measured modulation and its well sits low.
    """
    forward = cfg.loss.forward_model
    scales = np.linspace(forward.scale_probe_min, forward.scale_probe_max,
                         forward.scale_probe_steps)
    modality_cfg = cfg.merged({"data": {"modality": modality}})
    loader = build_dataloaders(modality_cfg, splits_to_build=(split,))[split]
    term = _forward_term(cfg, distance_um, border_px).to(device)

    curve = np.zeros(len(scales), dtype=np.float64)
    best_per_field: list[float] = []
    seen = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            hologram = batch.get("hologram_raw", batch["hologram"]).to(device).float()
            phase = batch["phase"].to(device).float()
            surface = batch.get("aberration")
            if surface is not None:
                surface = surface.to(device).float()
            usable = batch.get("aberration_valid")
            if usable is not None and not bool(usable.all()):
                keep = usable.to(torch.bool).reshape(-1)
                if not bool(keep.any()):
                    continue
                hologram, phase = hologram[keep], phase[keep]
                if surface is not None:
                    surface = surface[keep]
            amplitude = torch.ones_like(phase)
            seen += hologram.shape[0]

            per_field = np.zeros((hologram.shape[0], len(scales)), dtype=np.float64)
            for position, scale in enumerate(scales):
                value = term(float(scale) * phase, amplitude, hologram, modality,
                             aberration=surface)
                per_field[:, position] = value.detach().float().cpu().numpy()
                curve[position] += float(value.sum())
            best_per_field.extend(scales[per_field.argmin(axis=1)].tolist())

    if seen == 0:
        return {"reason": "no images"}

    votes = np.asarray(best_per_field, dtype=float)
    return {
        "images": int(seen),
        "scales": scales.tolist(),
        "residual": (curve / seen).tolist(),
        "best_scale_pooled": float(scales[int(np.argmin(curve))]),
        "best_scale_median": float(np.median(votes)),
        "best_scale_iqr": float(np.percentile(votes, 75) - np.percentile(votes, 25)),
        "residual_at_unit_scale": float(
            (curve / seen)[int(np.argmin(np.abs(scales - 1.0)))]
        ),
        "distance_um": float(distance_um),
    }


def report_scale_response(modality: str, result: dict) -> None:
    print(f"\n--- {modality}: does the operator agree about phase MAGNITUDE? ---")
    if "scales" not in result:
        print(f"  inconclusive ({result.get('reason', 'unknown')})")
        return
    scales = np.asarray(result["scales"])
    residual = np.asarray(result["residual"])
    marks = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    shown = [(s, residual[int(np.argmin(np.abs(scales - s)))])
             for s in marks if scales.min() <= s <= scales.max()]
    print("    phase x  " + "".join(f"{s:>9.2f}" for s, _ in shown))
    print("    residual " + "".join(f"{v:>9.4f}" for _, v in shown))
    best = result["best_scale_median"]
    print(f"  residual minimised at phase x {best:.2f}"
          f"   (IQR {result['best_scale_iqr']:.2f} over {result['images']} fields,"
          f" pooled {result['best_scale_pooled']:.2f})")
    print("  -> ", end="")
    if abs(best - 1.0) <= 0.15:
        print(f"the forward model agrees with the delivered phase about its\n"
              f"     magnitude to within {abs(best - 1.0):.0%}. The reconstruction this arm\n"
              f"     is scored against is quantitatively consistent with the recorded\n"
              f"     hologram, which is what measurement-ready has to mean.")
    else:
        direction = "less" if best < 1.0 else "more"
        print(f"the operator reproduces the hologram best with {direction} phase than\n"
              f"     was delivered, by {abs(best - 1.0):.0%}. For an in-line geometry that is\n"
              f"     expected rather than anomalous: the twin image is superposed on the\n"
              f"     object, so a single-term forward model accounts for only part of the\n"
              f"     measured modulation. Report the factor; it is the quantitative form\n"
              f"     of the geometry's disadvantage.")


def report_discrimination(modality: str, result: dict) -> None:
    print(f"\n--- {modality}: can the forward-model term be trained? ---")
    if "scores" not in result:
        print(f"  inconclusive ({result.get('reason', 'unknown')})")
        return
    print(f"  {result['images']} fields at z = {result['distance_um']:.3f} um")
    print(f"    {'variant':<12}{'residual':>10}{'margin':>10}{'worse than truth':>19}")
    for name, value in result["scores"].items():
        if name == "reference":
            print(f"    {name:<12}{value:>10.4f}{'':>10}{'<- floor':>19}")
        else:
            print(f"    {name:<12}{value:>10.4f}{result['margins'][name]:>+10.4f}"
                  f"{result['win_rates'][name]:>18.0%}")
    print(f"  near-truth  margin {result['near_margin']:+.4f}"
          f"   worse in {result['near_win_rate']:.0%} of fields")
    print(f"  far-from-truth   margin {result['far_margin']:+.4f}"
          f"   worse in {result['far_win_rate']:.0%} of fields")
    print(f"  (tolerance {result['tolerance']:.3f}, "
          f"sign-test threshold {result['win_threshold']:.0%}; chance is 50%)")
    print("  -> ", end="")
    if result["verdict"] == "usable":
        print("USABLE. Degraded phases score worse than the truth both near it and\n"
              "     far from it, in most individual fields, so the gradient points toward\n"
              "     the reference even though the operator is approximate. The floor is a\n"
              "     constant offset from an acquisition chain that cannot be inverted;\n"
              "     report it with the result.")
    elif result["verdict"] == "marginal":
        print("MARGINAL. The term separates a real reconstruction from nonsense, but\n"
              "     near the truth its slope is within noise, so it cannot refine a\n"
              "     prediction that is already close. Reportable as a diagnostic; not\n"
              "     trained by default. Raise loss.weights.forward_model deliberately if\n"
              "     you want to test it, and check the phase metrics do not degrade.")
    elif result["verdict"] == "uninformative":
        print("UNINFORMATIVE. Every degraded phase scores worse than the truth, so the\n"
              "     term is correctly signed and nothing about it is harmful -- but the\n"
              "     margins are inside the tolerance, so what it measures is dominated by\n"
              "     the part of the acquisition the forward operator cannot reproduce\n"
              "     rather than by the reconstruction. Not worth training; worth reporting\n"
              "     as the quantitative form of the non-invertibility limitation.")
    else:
        print("ANTI-DISCRIMINATIVE, which is worse than useless: a degraded phase\n"
              "     scores BETTER than the truth, so minimising this term would drive the\n"
              "     reconstruction away from the reference. Not trained on this arm at\n"
              "     any weight. The asymmetry between geometries is itself a result.")

def fixed_border_px(cfg, distances: np.ndarray) -> int:
    """The margin every distance in a scan must share.

    Sized for the largest |z| scanned, so the widest diffraction spread is
    excluded at every distance and all residuals are computed on identical
    pixels. Without this the scan is comparing different images.
    """
    forward = cfg.loss.forward_model
    if forward.border_px is not None:
        return int(forward.border_px)
    spread_um = abs(cfg.optics.wavelength_um * float(np.abs(distances).max()))
    pitch = min(cfg.optics.pixel_pitch_x_um, cfg.optics.pixel_pitch_y_um)
    required = int(math.ceil(spread_um / forward.feature_um / pitch))

    # A border wider than the field would leave nothing to score, and the term
    # would silently fall back to using every pixel -- which is the bias this
    # function exists to remove, reappearing without a warning. Cap it at a
    # third of the field and say so, because a scan that needs a wider border
    # than that is scanning distances the field cannot model in the first place.
    field_px = cfg.data.eval_size or cfg.data.phase_size
    cap = int(field_px // 3)
    if required > cap:
        LOGGER.warning(
            "a scan over |z| up to %.0f um needs a %d px border, more than a %d px "
            "field can give; capping at %d px. Distances beyond about %.0f um are "
            "not modellable here, so narrow --z-range instead of trusting these.",
            float(np.abs(distances).max()), required, field_px, cap,
            cap * pitch * forward.feature_um / cfg.optics.wavelength_um,
        )
        return cap
    return required


def scan(cfg, modality: str, device, distances: np.ndarray, batches: int,
         split: str = "val", border_px: int | None = None) -> dict:
    optics = cfg.optics
    wavelength = optics.wavelength_um
    dx, dy = optics.pixel_pitch_x_um, optics.pixel_pitch_y_um

    # Evaluation splits are unaugmented and uncropped. Both matter here: a random
    # crop breaks the propagation (the diffracted light comes from outside it),
    # and a 90-degree rotation swaps the two pixel pitches, which are not equal.
    modality_cfg = cfg.merged({"data": {"modality": modality}})
    loader = build_dataloaders(modality_cfg, splits_to_build=(split,))[split]

    forward_scores = np.zeros((len(distances),), dtype=np.float64)
    inverse_scores = np.zeros((len(distances),), dtype=np.float64)
    sensitivity_probe: list = []
    per_image_best: list[float] = []
    seen = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            # The RAW measurement, and the aberration surface the delivered
            # phase had removed. Without the surface no distance can fit, because
            # a quadratic aberration is degenerate with defocus.
            hologram = batch.get("hologram_raw", batch["hologram"]).to(device).float()
            phase = batch["phase"].to(device).float()
            surface = batch.get("aberration")
            if surface is not None:
                surface = surface.to(device).float()
            amplitude = torch.ones_like(phase)
            carrier = estimate_carrier(hologram) if modality == "off_axis" else None
            seen += hologram.shape[0]
            if index == 0:
                sensitivity_probe.append((phase, carrier))

            per_image = np.zeros((hologram.shape[0], len(distances)), dtype=np.float64)
            for position, distance in enumerate(distances):
                # The training residual itself, so calibration and optimisation
                # cannot disagree about what "consistent" means.
                term = _forward_term(cfg, float(distance), border_px).to(device)
                value = term(phase, amplitude, hologram, modality,
                             aberration=surface)
                per_image[:, position] = value.detach().cpu().numpy()
                forward_scores[position] += float(value.sum())

                field = torch.complex(hologram.clamp(min=0).sqrt(),
                                      torch.zeros_like(hologram))
                back = propagate(field, wavelength, dx, dy, -float(distance))
                recovered = torch.angle(back)
                inverse_scores[position] += float(
                    _correlation(recovered, phase).abs().sum()
                )
            per_image_best.extend(distances[per_image.argmin(axis=1)].tolist())

    if seen == 0:
        raise SystemExit("no images were read; check the split file and data root")

    forward_scores /= seen
    inverse_scores /= seen

    # Sensitivity at the best distance, and at zero for contrast.
    best = float(distances[int(np.argmin(forward_scores))])
    # The residual the GROUND-TRUTH phase achieves at the best distance is the
    # floor this term can ever reach. Everything the forward operator fails to
    # explain -- an unknown acquisition geometry, processing applied to the
    # delivered phase, an amplitude the pure-phase assumption discards -- lands
    # here, and none of it is an error the network made. If the reference phase
    # cannot explain the hologram, neither can any prediction, and optimising
    # the term would be fitting our ignorance of the acquisition rather than
    # improving the reconstruction.
    floor = float(min(forward_scores))    # diagnostic only; see discrimination()
    phase, carrier = sensitivity_probe[0]
    sensitivity = {
        "at_best_z": phase_sensitivity(cfg, modality, best, phase, carrier),
        "at_zero_z": phase_sensitivity(cfg, modality, 0.0, phase, carrier),
    }

    return {
        "modality": modality,
        "images": seen,
        "reference_floor": floor,
        "distances_um": distances.tolist(),
        "forward_residual": forward_scores.tolist(),
        "inverse_correlation": inverse_scores.tolist(),
        "per_image_best_z_um": per_image_best,
        "phase_sensitivity": sensitivity,
    }


def summarise(result: dict) -> dict:
    distances = np.asarray(result["distances_um"])
    forward = np.asarray(result["forward_residual"])
    inverse = np.asarray(result["inverse_correlation"])
    step = float(abs(distances[1] - distances[0])) if distances.size > 1 else 0.0

    # The forward criterion is a residual, so the best z is its MINIMUM.
    best = int(np.argmin(forward))
    spread = float(forward.std()) or 1e-12
    depth = float((np.median(forward) - forward[best]) / spread)

    z_inverse = float(distances[int(np.argmax(inverse))])
    inverse_spread = float(inverse.std()) or 1e-12
    inverse_prominence = float((inverse.max() - np.median(inverse)) / inverse_spread)

    # The real identifiability test: do individual images independently agree?
    votes = np.asarray(result.get("per_image_best_z_um") or [], dtype=float)
    if votes.size >= 3:
        vote_median = float(np.median(votes))
        vote_spread = float(np.percentile(votes, 75) - np.percentile(votes, 25))
        # Consistent if the middle half of the images fall within a few grid
        # steps of one another, and away from the edge of the scanned range.
        consistent = bool(vote_spread <= max(3.0 * step, 1e-9))
        at_edge = bool(
            abs(vote_median - distances[0]) < step or abs(vote_median - distances[-1]) < step
        )
    else:
        vote_median, vote_spread, consistent, at_edge = float("nan"), float("nan"), False, False

    identifiable = bool(depth > 2.0 and consistent and not at_edge)
    return {
        "reference_floor": result.get("reference_floor"),
        "z_forward_um": float(distances[best]),
        "forward_residual_min": float(forward[best]),
        "forward_well_depth": depth,
        "z_inverse_um": z_inverse,
        "inverse_prominence": inverse_prominence,
        "per_image_median_z_um": vote_median,
        "per_image_iqr_um": vote_spread,
        "images_agree": consistent,
        "peak_at_scan_edge": at_edge,
        "grid_step_um": step,
        "identifiable": identifiable,
        "recommended_z_um": vote_median if identifiable else None,
    }


def report(result: dict, summary: dict) -> None:
    print(f"\n=== {result['modality']}  ({result['images']} images) ===")
    print(f"  forward  (training residual, minimised)   z = {summary['z_forward_um']:+9.3f} um"
          f"   residual = {summary['forward_residual_min']:.5f}"
          f"   well depth {summary['forward_well_depth']:.1f}")
    print(f"  per-image agreement                       median "
          f"{summary['per_image_median_z_um']:+9.3f} um   IQR "
          f"{summary['per_image_iqr_um']:.3f} um   consistent: {summary['images_agree']}")
    print(f"  inverse  (weak cross-check)               z = {summary['z_inverse_um']:+9.3f} um"
          f"   prominence {summary['inverse_prominence']:.1f}")
    print(f"  grid step {summary['grid_step_um']:.3f} um"
          + ("   PEAK AT THE EDGE OF THE SCANNED RANGE" if summary["peak_at_scan_edge"] else ""))
    floor = result.get("reference_floor")
    if floor is not None:
        print(f"  residual using the REFERENCE phase       {floor:.4f}"
              f"   (the floor set by the approximate operator)")

    sensitivity = result.get("phase_sensitivity", {})
    if sensitivity:
        print(f"  forward-model sensitivity to phase   "
              f"at best z {sensitivity['at_best_z']:.4f} /rad   "
              f"at z=0 {sensitivity['at_zero_z']:.4f} /rad")
        if sensitivity["at_best_z"] < 1e-3:
            print("     WARNING: the synthesised hologram barely responds to phase at this\n"
                  "     distance, so the forward-model term has almost no gradient for this\n"
                  "     arm. For in-line Gabor at small z this is expected and physical:\n"
                  "     |A exp(i phi)|^2 does not contain phi. Defocus is what encodes it.")

    print("\n  -> ", end="")
    if summary["identifiable"]:
        print(f"z IS identifiable at {summary['recommended_z_um']:+.3f} um "
              f"(images agree to {summary['per_image_iqr_um']:.2f} um).\n"
              f"     Set loss.forward_model.distance_um to this value in config/base.yaml.")
    elif summary["peak_at_scan_edge"]:
        print("the minimum sits at the EDGE of the scanned range, so the true z is\n"
              "     probably outside it. Re-run with a wider --z-range before concluding.")
    elif summary["forward_well_depth"] <= 2.0:
        print("the residual is FLAT in z, so z is not identifiable from this arm.\n"
              "     For off-axis this is expected and physical: the carrier encodes phase\n"
              "     at any distance, so the hologram barely changes with z. Take z from the\n"
              "     Gabor arm instead -- it is the same specimen at the same distance.")
    else:
        print("the residual has a minimum but individual images DISAGREE about where.\n"
              "     Something varies between fields that should not: check data.align, and\n"
              "     check that the reference phase was reconstructed with one fixed z\n"
              "     rather than refocused per image. Do not guess a value.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--z-range", nargs=2, type=float, default=None,
                        metavar=("MIN_UM", "MAX_UM"),
                        help="default: the largest range the field size supports")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="use an unaugmented, uncropped split (default val)")
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--batches", type=int, default=4,
                        help="batches to average the z scan over")
    parser.add_argument("--discrimination-batches", type=int, default=None,
                        help="batches for the discrimination test; default: "
                             "loss.forward_model.discrimination_batches")
    parser.add_argument("--feature-um", type=float, default=1.0,
                        help="smallest object feature, used to bound the z range")
    parser.add_argument("--refine", action="store_true",
                        help="second pass on a 20x finer grid around the coarse peak")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/z_calibration.json")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)

    field_px = cfg.data.eval_size or cfg.data.phase_size
    limit = usable_z_range_um(cfg, field_px, args.feature_um)
    z_range = args.z_range or [-limit, limit]
    if max(abs(z_range[0]), abs(z_range[1])) > limit * 1.05:
        LOGGER.warning(
            "requested |z| up to %.1f um, but on a %d px grid diffraction wraps "
            "around beyond %.1f um; scores outside that are not physical",
            max(abs(z_range[0]), abs(z_range[1])), field_px, limit,
        )
    probe = np.linspace(z_range[0], z_range[1], args.steps)
    beyond = int((np.abs(probe) > limit).sum())
    print(f"field {field_px} px -> physically usable |z| <= {limit:.1f} um; "
          f"scanning [{z_range[0]:+.1f}, {z_range[1]:+.1f}] um in {args.steps} steps")
    if beyond:
        print(f"  note: {beyond} of {args.steps} distances exceed what a {field_px} px field "
              f"can model without\n        the diffraction wrapping around. Their residuals "
              f"are approximate, and a minimum\n        found only out there should not be "
              f"trusted. This is a property of the field size,\n        not of the data.")

    # One margin for the whole sweep, so every distance is scored on identical
    # pixels. See _forward_term for what happens without it.
    distances = np.linspace(z_range[0], z_range[1], args.steps)
    border = fixed_border_px(cfg, distances)
    print(f"  residual scored on a fixed {field_px - 2 * border} px window at every "
          f"distance (border {border} px, sized for |z| = "
          f"{np.abs(distances).max():.1f} um)")

    forward_cfg = cfg.loss.forward_model
    discrimination_batches = (
        args.discrimination_batches if args.discrimination_batches is not None
        else forward_cfg.discrimination_batches
    )

    payload = {}
    for modality in args.modality:
        result = scan(cfg, modality, device, distances, args.batches, args.split,
                      border_px=border)
        summary = summarise(result)

        if args.refine and summary["identifiable"]:
            span = 2.0 * summary["grid_step_um"]
            centre = summary["recommended_z_um"]
            if centre is None:
                centre = summary["z_forward_um"]
            fine = np.linspace(centre - span, centre + span, 41)
            LOGGER.info("refining %s around %+.3f um", modality, centre)
            result = scan(cfg, modality, device, fine, args.batches, args.split,
                          border_px=border)
            summary = summarise(result)

        report(result, summary)

        # The decisive test. The residual's floor is set by an acquisition chain
        # that cannot be inverted; its slope is what training actually follows.
        #
        # A distance supplied by the acquiring group always wins over anything
        # found here. The scan minimises an APPROXIMATE operator, so its minimum
        # is partly the distance that best fits the operator's own error; a
        # measured distance is not.
        configured = cfg.loss.forward_model.distance_um
        if configured is not None:
            best = float(configured)
            print(f"\n  validating the supplied distance z = {best:.3f} um "
                  f"(the scan above is diagnostic only)")
            # Independent corroboration is worth stating plainly. The scan
            # minimises an approximate operator and does not decide z, but if
            # its minimum lands within a grid step of the number the acquiring
            # group supplied, two unrelated routes agree and that is a result to
            # report rather than a coincidence to leave in a log file.
            step = summary["grid_step_um"] or 0.0
            offset = abs(summary["z_forward_um"] - best)
            if step and offset <= step:
                print(f"     the scan's own minimum ({summary['z_forward_um']:+.3f} um) "
                      f"agrees to within one grid step ({step:.3f} um)")
            elif step:
                print(f"     note: the scan's minimum ({summary['z_forward_um']:+.3f} um) "
                      f"is {offset:.1f} um away, {offset / step:.1f} grid steps")
        else:
            best = summary["recommended_z_um"]
            if best is None:
                best = summary["z_forward_um"]
        verdict = discrimination(cfg, modality, device, float(best), args.split,
                                 discrimination_batches,
                                 forward_cfg.discrimination_tolerance,
                                 forward_cfg.discrimination_win_rate,
                                 border_px=border)
        report_discrimination(modality, verdict)

        scale = phase_scale_response(cfg, modality, device, float(best), args.split,
                                     discrimination_batches, border_px=border)
        report_scale_response(modality, scale)
        payload[modality] = {**summary, "forward_model_usable": verdict["usable"],
                             "forward_model_verdict": verdict["verdict"],
                             "discrimination": verdict, "scale_response": scale,
                             "curve": result}

    # The same specimen was recorded at the same distance in both geometries, but
    # only the in-line arm constrains z (see the module docstring). If Gabor
    # identified it and off-axis did not, say so explicitly rather than leaving
    # the reader to conclude the calibration failed.
    gabor = payload.get("gabor") or {}
    off_axis = payload.get("off_axis") or {}
    if gabor.get("identifiable") and not off_axis.get("identifiable"):
        print(f"\n  ** Use z = {gabor['recommended_z_um']:+.3f} um for BOTH arms. **")
        print("     The in-line arm determines it; the off-axis residual is flat in z")
        print("     because its carrier encodes phase without needing defocus. Same")
        print("     specimen, same acquisition, same distance.")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\ncurves and estimates -> {destination}")

    verdicts = {m: (payload[m] or {}).get("forward_model_verdict", "unknown")
                for m in payload}
    print("\n" + "=" * 66)
    print(" FORWARD-MODEL VERDICT")
    print("=" * 66)
    for arm, verdict in verdicts.items():
        print(f"  {arm:<12} {verdict}")

    if all(v == "uninformative" for v in verdicts.values()):
        print("\nEvery arm is UNINFORMATIVE: every degraded phase scores worse than the")
        print("truth, so the term is correctly signed and would do no harm, but its")
        print("margins sit inside the tolerance. What it measures is dominated by the")
        print("part of the acquisition chain the forward operator cannot reproduce")
        print("rather than by the reconstruction. Leave loss.weights.forward_model at")
        print("0.0 and report these margins as the quantitative form of the")
        print("non-invertibility limitation -- that is a result, not a gap.")
        return 0

    marginal = [m for m, v in verdicts.items() if v == "marginal"]
    if marginal and not any(v == "usable" for v in verdicts.values()):
        print("\nEvery arm is MARGINAL: the term tells a reconstruction from nonsense")
        print("but cannot refine one that is already close. That is a reportable")
        print("diagnostic and a legitimate negative result about physics-consistency")
        print("losses under a non-invertible acquisition pipeline. It is not trained")
        print("by default. To test it anyway:")
        print("    python main.py compare --config config/ablation/physics_forward_only.yaml \\")
        print("        --set loss.weights.forward_model=0.05")
        print("and check that the phase metrics do not degrade against the base run.")
        return 0

    usable_arms = [m for m in payload if (payload[m] or {}).get("forward_model_usable")]
    if usable_arms:
        print(f"\nForward-model term is usable for: {', '.join(usable_arms)}")
        print("The residual floor is set by the acquisition chain, which the acquiring")
        print("group confirms cannot be inverted exactly (the reconstruction runs inside")
        print("closed software and no uncorrected phase is saved). Report the floor with")
        print("the result; what training follows is the slope, not the offset.")
        blocked = [m for m in payload if m not in usable_arms]
        if blocked:
            print(f"NOT usable for: {', '.join(blocked)} -- a degraded phase scores better")
            print("than the truth there, so the term would push the reconstruction away")
            print("from the reference. Those arms are excluded, and that asymmetry is")
            print("itself a result worth reporting.")
        return 0

    if not any((payload[m] or {}).get("forward_model_usable", True) for m in payload):
        print("\nThe forward operator does not reproduce these holograms even from the")
        print("ground-truth phase, in either geometry. This is not a tuning problem and")
        print("no distance will fix it. The delivered phase has been demodulated,")
        print("filtered, aberration-corrected and unwrapped by a pipeline we are")
        print("reverse-engineering, and the accumulated approximation dominates the")
        print("residual.")
        print("\nWhat unblocks it, in order:")
        print("  1. The acquiring group's reconstruction code or its exact parameters:")
        print("     demodulation and sideband filter, aberration model, unwrapping, and")
        print("     the per-sample distance. With those the operator is exact.")
        print("  2. Failing that, a handful of RAW unprocessed phase reconstructions")
        print("     (no aberration removal, no background subtraction) with their")
        print("     holograms, which is enough to validate the operator.")
        print("\nUntil then leave loss.weights.forward_model at 0.0. The term is")
        print("implemented and verified against synthetic data where the operator is")
        print("known (scripts/selftest.py section 6); what is missing is this")
        print("dataset's operator, not the code.")
        return 2

    if not any(payload[m]["identifiable"] for m in payload):
        print("\nNo modality produced an identifiable z.")
        print("Next steps, in order:")
        print("  1. Ask the acquiring group for the reconstruction distance. This is one")
        print("     email and it settles the question outright.")
        print("  2. Re-run with a wider range: --z-range -600 600 --steps 121 --feature-um 2")
        print("     A minimum at the edge of the scan means the range was too narrow.")
        print("  3. If z is genuinely unknown, train with")
        print("        loss.forward_model.learn_distance: true")
        print("     and an initial distance_um from the Gabor forward minimum above. The")
        print("     propagation kernel is differentiable in z, so it is refined alongside")
        print("     the weights; report the converged value in the paper.")
        print("  4. Until one of those, leave loss.weights.forward_model at 0.0. Training")
        print("     the term with a wrong z is worse than not training it: the residual")
        print("     then measures the error in z rather than in the reconstruction.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
