"""Recover the sample-to-sensor distance z from the data.

The forward-model consistency loss needs z, and z is not recorded anywhere: the
phase `.bin` header carries width, height and the two pixel pitches, and nothing
else. Asking the group that acquired the data is the right first move, but the
number is also recoverable, because the dataset already contains both halves of
the propagation: a raw hologram and the reference phase reconstructed from it.

Two estimators are computed for every candidate distance and each modality.

<<<<<<< Updated upstream
1. FORWARD.  Synthesise the hologram that the reference phase would produce at
   distance z and correlate it with the measured hologram. This is the criterion
   that matters, because it is exactly the residual the training loss minimises.

2. INVERSE.  Back-propagate the measured hologram by z and correlate the
   resulting phase with the reference phase. Independent of the forward
   estimator's amplitude assumptions, so agreement between the two is evidence
   that the recovered z is physical rather than an artefact of one criterion.

Both are scanned over a coarse grid and then refined. A sharp, single-peaked
agreement curve means z is identifiable; a flat curve means it is not, and the
report says so instead of returning a number that looks confident.
=======
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
>>>>>>> Stashed changes

    python scripts/calibrate_z.py --config config/base.yaml
    python scripts/calibrate_z.py --config config/base.yaml --z-range -400 400 --steps 81
"""

from __future__ import annotations

import argparse
import json
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


<<<<<<< Updated upstream
=======
def _forward_term(cfg, distance_um: float):
    """The training residual, instantiated at one distance."""
    from holoqpi.losses.terms import ForwardModelConsistency

    forward = cfg.loss.forward_model
    settings = type("Cfg", (), {
        "distance_um": float(distance_um),
        "learn_distance": False,
        "criterion": forward.criterion,
        "fit_radiometry": True,
        "feature_um": forward.feature_um,
        "pad_px": forward.pad_px,
        "border_px": forward.border_px,
        "dc_exclusion_frac": forward.dc_exclusion_frac,
        "dc_exclusion_px": forward.dc_exclusion_px,
    })()
    return ForwardModelConsistency(settings, cfg.optics)


>>>>>>> Stashed changes
def scan(cfg, modality: str, device, distances: np.ndarray, batches: int,
         split: str = "val") -> dict:
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
<<<<<<< Updated upstream
=======
    per_image_best: list[float] = []
>>>>>>> Stashed changes
    seen = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            hologram = batch["hologram"].to(device).float()
            phase = batch["phase"].to(device).float()
            amplitude = torch.ones_like(phase)
            carrier = estimate_carrier(hologram) if modality == "off_axis" else None
            seen += hologram.shape[0]
            if index == 0:
                sensitivity_probe.append((phase, carrier))

<<<<<<< Updated upstream
            for position, distance in enumerate(distances):
                synthetic = form_hologram(
                    phase, amplitude, modality, wavelength, dx, dy,
                    float(distance), carrier=carrier,
                )
                forward_scores[position] += float(
                    _correlation(synthetic, hologram).abs().sum()
                )
=======
            per_image = np.zeros((hologram.shape[0], len(distances)), dtype=np.float64)
            for position, distance in enumerate(distances):
                # The training residual itself, so calibration and optimisation
                # cannot disagree about what "consistent" means.
                term = _forward_term(cfg, float(distance)).to(device)
                value = term(phase, amplitude, hologram, modality)
                per_image[:, position] = value.detach().cpu().numpy()
                forward_scores[position] += float(value.sum())
>>>>>>> Stashed changes

                field = torch.complex(hologram.clamp(min=0).sqrt(),
                                      torch.zeros_like(hologram))
                back = propagate(field, wavelength, dx, dy, -float(distance))
                recovered = torch.angle(back)
                inverse_scores[position] += float(
                    _correlation(recovered, phase).abs().sum()
                )
<<<<<<< Updated upstream
=======
            per_image_best.extend(distances[per_image.argmin(axis=1)].tolist())
>>>>>>> Stashed changes

    if seen == 0:
        raise SystemExit("no images were read; check the split file and data root")

    forward_scores /= seen
    inverse_scores /= seen

    # Sensitivity at the best distance, and at zero for contrast.
<<<<<<< Updated upstream
    best = float(distances[int(np.argmax(forward_scores))])
=======
    best = float(distances[int(np.argmin(forward_scores))])
>>>>>>> Stashed changes
    phase, carrier = sensitivity_probe[0]
    sensitivity = {
        "at_best_z": phase_sensitivity(cfg, modality, best, phase, carrier),
        "at_zero_z": phase_sensitivity(cfg, modality, 0.0, phase, carrier),
    }

    return {
        "modality": modality,
        "images": seen,
        "distances_um": distances.tolist(),
<<<<<<< Updated upstream
        "forward_correlation": forward_scores.tolist(),
        "inverse_correlation": inverse_scores.tolist(),
=======
        "forward_residual": forward_scores.tolist(),
        "inverse_correlation": inverse_scores.tolist(),
        "per_image_best_z_um": per_image_best,
>>>>>>> Stashed changes
        "phase_sensitivity": sensitivity,
    }


def summarise(result: dict) -> dict:
    distances = np.asarray(result["distances_um"])
<<<<<<< Updated upstream
    forward = np.asarray(result["forward_correlation"])
    inverse = np.asarray(result["inverse_correlation"])

    def peak(scores):
        best = int(np.argmax(scores))
        # Identifiability: how far the peak stands above the typical value,
        # in units of the curve's own spread. A flat curve scores near zero.
        spread = float(scores.std()) or 1e-12
        prominence = float((scores[best] - np.median(scores)) / spread)
        return float(distances[best]), float(scores[best]), prominence

    z_forward, s_forward, p_forward = peak(forward)
    z_inverse, s_inverse, p_inverse = peak(inverse)

    step = float(abs(distances[1] - distances[0])) if distances.size > 1 else 0.0
    agree = abs(z_forward - z_inverse) <= max(2.0 * step, 0.05 * abs(z_forward) + 1e-9)

    return {
        "z_forward_um": z_forward, "forward_peak": s_forward, "forward_prominence": p_forward,
        "z_inverse_um": z_inverse, "inverse_peak": s_inverse, "inverse_prominence": p_inverse,
        "estimators_agree": bool(agree),
        "grid_step_um": step,
        "identifiable": bool(p_forward > 2.0 and p_inverse > 2.0 and agree),
        "recommended_z_um": z_forward if agree else None,
=======
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
>>>>>>> Stashed changes
    }


def report(result: dict, summary: dict) -> None:
    print(f"\n=== {result['modality']}  ({result['images']} images) ===")
<<<<<<< Updated upstream
    print(f"  forward  (synthesise hologram from reference phase)  "
          f"z = {summary['z_forward_um']:+9.3f} um   r = {summary['forward_peak']:.4f}   "
          f"prominence {summary['forward_prominence']:.1f}")
    print(f"  inverse  (back-propagate hologram to reference phase) "
          f"z = {summary['z_inverse_um']:+9.3f} um   r = {summary['inverse_peak']:.4f}   "
          f"prominence {summary['inverse_prominence']:.1f}")
    print(f"  grid step {summary['grid_step_um']:.3f} um   "
          f"estimators agree: {summary['estimators_agree']}")
=======
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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        print(f"z is identifiable at {summary['recommended_z_um']:+.3f} um.\n"
              f"     Set optics.propagation_distance_um to this value in config/base.yaml,\n"
              f"     or per modality in config/off_axis.yaml and config/gabor.yaml.")
    elif summary["forward_prominence"] <= 2.0 and summary["inverse_prominence"] <= 2.0:
        print("both agreement curves are FLAT: z is not identifiable from this data.\n"
              "     The most likely explanation is that the reference phase was produced\n"
              "     with the hologram already numerically refocused, so the effective\n"
              "     distance is zero. Try a finer grid near zero before concluding, then\n"
              "     ask the acquiring group. Do not guess a value.")
    else:
        print("the two estimators DISAGREE, so neither should be trusted yet.\n"
              "     Widen or refine the grid, and check that the alignment between\n"
              "     hologram and phase (data.align) is correct before reading further.")
=======
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
>>>>>>> Stashed changes


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
                        help="training batches to average over")
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
    print(f"field {field_px} px -> physically usable |z| <= {limit:.1f} um; "
          f"scanning [{z_range[0]:+.1f}, {z_range[1]:+.1f}] um in {args.steps} steps")

    payload = {}
    for modality in args.modality:
        distances = np.linspace(z_range[0], z_range[1], args.steps)
        result = scan(cfg, modality, device, distances, args.batches, args.split)
        summary = summarise(result)

        if args.refine and summary["estimators_agree"]:
            span = 2.0 * summary["grid_step_um"]
            centre = summary["recommended_z_um"]
            fine = np.linspace(centre - span, centre + span, 41)
            LOGGER.info("refining %s around %+.3f um", modality, centre)
            result = scan(cfg, modality, device, fine, args.batches, args.split)
            summary = summarise(result)

        report(result, summary)
        payload[modality] = {**summary, "curve": result}

<<<<<<< Updated upstream
=======
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

>>>>>>> Stashed changes
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\ncurves and estimates -> {destination}")

    if not any(payload[m]["identifiable"] for m in payload):
<<<<<<< Updated upstream
        print("\nNo modality produced an identifiable z. The forward-model loss cannot\n"
              "be trained meaningfully until this is resolved; leave\n"
              "loss.weights.forward_model at 0.0 and ask for the acquisition distance.")
=======
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
>>>>>>> Stashed changes
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
