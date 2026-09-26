"""Does the forward model care about the amplitude, or only about the phase?

WHY THIS EXISTS
---------------
The forward-model term propagates a complex field, and a complex field has two
parts. Until now one of them was a fixture: the amplitude was set to 1
everywhere, the thin-phase-object assumption. The network has since been given
an amplitude head and ``scripts/prepare_amplitude.py`` writes a reference for
it, but adding a head and a loss term is only justified if the amplitude
actually changes the measurement it is supposed to help explain.

THE QUESTION, STATED SO IT CAN BE ANSWERED WRONG
------------------------------------------------
Two numbers decide it, and one is useless without the other.

SENSITIVITY. How much the residual moves when the amplitude changes from unity
to the classical reference, and as it is perturbed away from that reference.
If it barely moves, the amplitude is not part of this measurement and the
amplitude head is a parameter cost with no physical return -- which is a
result, and a publishable one, not a failure.

THE SCALE TO JUDGE IT AGAINST. A residual change of 0.02 means nothing on its
own. The same term's response to DEGRADING THE PHASE is the yardstick: if
scaling the phase by 0.9 moves the residual by 0.15 and switching the amplitude
from unity to the reference moves it by 0.01, then the forward model is a phase
constraint with an amplitude rounding error, and the amplitude weight should be
zero. If they are comparable, the unit-amplitude assumption was costing the
term real accuracy and the head earns its place.

So this script reports the amplitude response, the phase response, and their
ratio, on the same fields with the same radiometric fit and the same aberration
surface.

WHAT IT CANNOT TELL YOU
-----------------------
The reference amplitude is a classical reconstruction, not a measurement (see
``scripts/prepare_amplitude.py``). If the residual falls when it is used, that
says the reference is a better explanation of the hologram than unity is -- it
does not say the reference is correct. Both are models. The honest reading of a
fall is "the amplitude is a material part of the forward model here", and the
honest reading of no change is "it is not".

    python scripts/prepare_amplitude.py --config config/base.yaml   # first
    python scripts/amplitude_sensitivity.py --config config/base.yaml
    python scripts/amplitude_sensitivity.py --config config/base.yaml --modality gabor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.utils import get_logger, resolve_device, write_csv

LOGGER = get_logger(__name__)


def forward_term(cfg, distance_um: float, border_px: int | None):
    """The training term itself, at a fixed distance.

    ``border_px = None`` is the right default and means "whatever the term's own
    rule gives", which at this dataset's z is 79 px. The previous default was a
    hardcoded 64, which is SMALLER than the diffraction pad the term applies, so
    a 15-pixel ring of wrap-around-contaminated pixels stayed inside the
    residual at every variant. The comparison between variants was still
    internally valid -- the same pixels every time -- but the absolute residual
    was contaminated and was not on the same scale as the one training reports,
    which is the number the `ratio < 0.1` verdict is implicitly compared against.

    ``fit_radiometry`` comes from the config rather than being hardcoded true, so
    this diagnostic cannot end up characterising a different observation model
    from the one the objective uses.
    """
    from holoqpi.losses.terms import ForwardModelConsistency

    forward = cfg.loss.forward_model
    settings = type("Cfg", (), {
        "distance_um": float(distance_um),
        "learn_distance": False,
        "criterion": forward.criterion,
        "fit_radiometry": bool(forward.fit_radiometry),
        "feature_um": forward.feature_um,
        "pad_px": forward.pad_px,
        "border_px": forward.border_px if border_px is None else int(border_px),
        "dc_exclusion_frac": forward.dc_exclusion_frac,
        "dc_exclusion_px": forward.dc_exclusion_px,
        "warn_on_short_pad": False,
    })()
    return ForwardModelConsistency(settings, cfg.optics)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--modality", default=None, help="default: data.modality")
    parser.add_argument("--distance-um", type=float, default=None,
                        help="default: loss.forward_model.distance_um")
    parser.add_argument(
        "--border-px", type=int, default=None,
        help="margin dropped from the residual. Default: the term's own "
             "diffraction pad, so the residual is on the same scale as the one "
             "training reports. A smaller value leaves wrap-around in.",
    )
    parser.add_argument("--batches", type=int, default=10 ** 6,
                        help="batches to read; the default reads the whole split, "
                             "which is what makes the numbers comparable")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)
    modality = args.modality or cfg.data.modality
    distance = (
        args.distance_um if args.distance_um is not None
        else cfg.loss.forward_model.distance_um
    )
    if distance is None:
        # Exit code 3, not 1: see the note at the "no usable images" branch.
        # Exit 1 is a real verdict of this script, so a misconfiguration must not
        # share it.
        print("HARD ERROR: loss.forward_model.distance_um is null and --distance-um "
              "was not given.\n  Run scripts/calibrate_z.py first.")
        return 3

    # The amplitude reference has to be loaded, whatever the config says, or
    # there is nothing to compare unity against.
    probe = cfg.merged({
        "data": {"modality": modality, "provide_amplitude": True},
    })
    loader = build_dataloaders(probe, splits_to_build=(args.split,))[args.split]
    term = forward_term(cfg, distance, args.border_px).to(device)
    applied_border = (
        int(args.border_px) if args.border_px is not None
        else (int(cfg.loss.forward_model.border_px)
              if cfg.loss.forward_model.border_px is not None
              else term.required_pad())
    )

    print(f"\n  modality {modality}   z {distance:.1f} um   "
          f"border {applied_border} px   criterion {cfg.loss.forward_model.criterion}")
    print(f"  aberration mode {cfg.optics.aberration.mode}   "
          f"crop {cfg.data.train_crop}   augmentation "
          f"{'on' if cfg.data.augmentation.enabled else 'off'}")
    if cfg.data.train_crop or cfg.data.augmentation.enabled:
        print("  NOTE: random crops or augmentation are on, so two runs see different\n"
              "  pixels and the residuals are not comparable between runs. Pass\n"
              "  `--set data.train_crop=null data.augmentation.enabled=false` to compare\n"
              "  configurations against each other.")

    rows: list[dict] = []
    seen = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.batches:
                break
            hologram = batch.get("hologram_raw", batch["hologram"]).to(device).float()
            phase = batch["phase"].to(device).float()
            reference_amplitude = batch["amplitude"].to(device).float()
            surface = batch.get("aberration")
            if surface is not None:
                surface = surface.to(device).float()

            usable = batch.get("aberration_valid")
            if usable is not None and not bool(usable.all()):
                keep = usable.to(torch.bool).reshape(-1)
                if not bool(keep.any()):
                    continue
                hologram, phase = hologram[keep], phase[keep]
                reference_amplitude = reference_amplitude[keep]
                if surface is not None:
                    surface = surface[keep]

            unity = torch.ones_like(phase)
            seen += hologram.shape[0]

            # AMPLITUDE VARIANTS, phase held at the reference throughout.
            # The blends interpolate between the two models rather than jumping
            # between them, so a monotone trend can be distinguished from a
            # coincidence at the endpoints.
            amplitude_variants = {
                "amp_unity": unity,
                "amp_blend_0.25": unity + 0.25 * (reference_amplitude - unity),
                "amp_blend_0.50": unity + 0.50 * (reference_amplitude - unity),
                "amp_reference": reference_amplitude,
                # Past the reference, to see whether the reference is a minimum
                # or merely a point on a slope. If 1.5x overshoots and scores
                # worse, the reference is genuinely near the best explanation.
                "amp_over_1.5": unity + 1.5 * (reference_amplitude - unity),
                "amp_inverted": 2.0 * unity - reference_amplitude,
            }
            # PHASE VARIANTS, amplitude held at unity: the yardstick.
            phase_variants = {
                "phase_reference": phase,
                "phase_scaled_0.9": phase * 0.9,
                "phase_scaled_0.5": phase * 0.5,
            }

            # PER IMAGE, not per batch. The loader shuffles, so which images
            # land in a batch changes between runs, and batches at the end of a
            # split are short. Averaging batch means would then weight images
            # unequally and by an amount that depends on the shuffle, which is
            # enough to move the mild phase response across zero -- measured on
            # this data at 9 fields. One row per image removes that entirely.
            values: dict[str, torch.Tensor] = {}
            for name, candidate in amplitude_variants.items():
                values[name] = term(
                    phase, candidate.clamp(min=0.0), hologram, modality, aberration=surface
                ).detach().float().reshape(-1).cpu()
            for name, candidate in phase_variants.items():
                values[name] = term(
                    candidate, unity, hologram, modality, aberration=surface
                ).detach().float().reshape(-1).cpu()

            for item in range(hologram.shape[0]):
                rows.append(
                    {"batch": index, "item": item}
                    | {name: float(values[name][item]) for name in values}
                )
            print(f"    batch {index:<3} images {hologram.shape[0]:<3} "
                  f"unity {float(values['amp_unity'].mean()):.6f}   "
                  f"reference {float(values['amp_reference'].mean()):.6f}   "
                  f"phase x0.9 {float(values['phase_scaled_0.9'].mean()):.6f}")

    if not rows:
        # Hard input error, not a verdict. run_v2.sh reads a traceback-free
        # non-zero exit as "the script ran and is reporting a finding", so a
        # SystemExit here was being counted as a successful diagnostic. Code 3
        # is reserved for a misconfiguration and the runner treats it as FAILED.
        print("\nHARD ERROR: no usable images -- every field's aberration surface was "
              "rejected.\n  Re-run scripts/estimate_aberration.py, or check "
              "optics.aberration.mode.")
        return 3

    # KEYED BY ABERRATION MODE AS WELL AS MODALITY.
    #
    # run_v2.sh runs this script twice, once per aberration mode, and says in its
    # own comment that "the difference between them IS the result". Both runs use
    # the same modality, so with a modality-only filename the second overwrote
    # the first and half of that result was destroyed -- silently, because only
    # the .log files were distinguished by mode. The rows carried no mode column
    # either, so the survivor could not even be attributed.
    mode = cfg.optics.aberration.mode
    for row in rows:
        row["aberration_mode"] = mode
        row["border_px"] = applied_border
        row["distance_um"] = float(distance)
    report = (
        Path(cfg.paths.output_root)
        / f"amplitude_sensitivity_{modality}_{mode}.csv"
    )
    write_csv(rows, report)

    def mean(name: str) -> float:
        return float(np.mean([r[name] for r in rows]))

    baseline = mean("amp_unity")
    print(f"\n=== residual over {seen} images ===")
    print(f"  {'variant':<20}{'residual':>12}{'change vs unity':>18}")
    order = ["amp_unity", "amp_blend_0.25", "amp_blend_0.50", "amp_reference",
             "amp_over_1.5", "amp_inverted",
             "phase_reference", "phase_scaled_0.9", "phase_scaled_0.5"]
    for name in order:
        value = mean(name)
        marker = "" if name == "amp_unity" else f"{value - baseline:+.6f}"
        print(f"  {name:<20}{value:>12.6f}{marker:>18}")

    print("\n  READ THE DIFFERENCES, NOT THE LEVELS. The train loader shuffles and drops")
    print("  the short final batch, so which fields are seen changes between runs and")
    print("  the absolute residual moves with them -- measured here, 0.44 to 0.51 on")
    print("  the same configuration. Every difference in the right-hand column is")
    print("  computed within a run on identical pixels and is reproducible to the third")
    print("  decimal. Use --split test for a fixed set.")

    amplitude_response = abs(mean("amp_reference") - baseline)
    # SIGNED, and they must stay signed. A degraded phase is supposed to score
    # WORSE, so both differences are supposed to be positive. Taking absolute
    # values would turn the one failure mode that invalidates everything else
    # on this page into a perfectly healthy-looking yardstick.
    #
    # TWO SCALES, because on this data they disagree. A 0.5x phase is a wrecked
    # reconstruction and a 0.9x phase is a good one with a 10% bias -- which is
    # the regime training is in once it has nearly converged, and the only
    # regime in which a refinement term is doing anything. A term that catches
    # the first and not the second cannot refine, however good its coarse
    # ranking looks.
    mild = mean("phase_scaled_0.9") - mean("phase_reference")
    coarse = mean("phase_scaled_0.5") - mean("phase_reference")
    ratio = amplitude_response / abs(mild) if mild != 0 else float("inf")
    # `1 / ratio` is printed in the verdict below, and ratio is exactly zero
    # whenever the amplitude reference moves the residual not at all -- which is
    # what an amplitude reference of identically 1.0 would do.
    inverse_ratio = 1.0 / ratio if ratio > 0 else float("inf")

    print(f"\n  amplitude response  unity -> reference        {amplitude_response:.6f}")
    print(f"  phase response      reference -> x0.9         {mild:+.6f}"
          f"   (must be POSITIVE)")
    print(f"  phase response      reference -> x0.5         {coarse:+.6f}"
          f"   (must be POSITIVE)")
    print(f"  ratio, amplitude / |mild phase|               {ratio:.4f}")
    print(f"  per-batch detail -> {report}")

    improves = mean("amp_reference") < baseline

    print("\n  -> ", end="")
    if mild < 0 and coarse > 0:
        print(f"the term ranks a WRECKED phase correctly (+{coarse:.6f} at 0.5x) but a\n"
              f"     NEARLY CORRECT one backwards ({mild:+.6f} at 0.9x). So it can tell\n"
              f"     reconstruction from noise and cannot refine a good reconstruction --\n"
              f"     and refining is what a loss term does once training has converged.\n"
              f"     Near the truth its gradient points the wrong way, so it must not be\n"
              f"     used as a loss at any weight. Report it as a DIAGNOSTIC with this\n"
              f"     measurement attached, and note that a coarse-only discrimination\n"
              f"     test would have passed it. The amplitude numbers above still stand\n"
              f"     on their own, but they have no local phase yardstick to be scaled\n"
              f"     against.")
        return 2
    if mild < 0 and coarse <= 0:
        print(f"STOP. The term is ANTI-DISCRIMINATIVE at BOTH scales: a degraded phase\n"
              f"     scores better than the truth ({mild:+.6f} at 0.9x, {coarse:+.6f} at\n"
              f"     0.5x). Minimising this residual would actively drive the\n"
              f"     reconstruction away from the truth. Do not use it, as a loss or as a\n"
              f"     metric. Check the conjugate sideband resolution and the aberration\n"
              f"     surface before anything else.")
        return 2
    if mild == 0:
        print("the term does not respond to the PHASE at all, so nothing here can be\n"
              "     interpreted. Check scripts/calibrate_z.py: at this z and this\n"
              "     geometry the forward model has no gradient.")
        return 2
    if coarse <= 0:
        # THE FOURTH SIGN COMBINATION, which had no branch at all and fell
        # through to the amplitude verdict below. mild > 0 with coarse <= 0 means
        # the term ranks a nearly-correct phase correctly but a WRECKED one
        # backwards, so its residual is not monotone in reconstruction quality.
        # Without this branch the script could print "the unit-amplitude
        # assumption is MEASURED to be adequate" and exit 0, having already
        # printed a negative coarse response under a heading that says the value
        # must be positive.
        print(f"the term ranks a nearly correct phase correctly ({mild:+.6f} at 0.9x)\n"
              f"     but a WRECKED one BACKWARDS ({coarse:+.6f} at 0.5x), so its residual\n"
              f"     is not monotone in reconstruction quality. Something in the operator\n"
              f"     is wrong rather than merely weak -- check the conjugate sideband\n"
              f"     resolution, the aberration surface and the carrier estimate before\n"
              f"     reading the amplitude numbers above, which have no trustworthy phase\n"
              f"     yardstick to be scaled against.")
        return 2
    if ratio < 0.1:
        print(f"the amplitude is NOT a material part of this forward model: it moves\n"
              f"     the residual {inverse_ratio:.0f}x less than a 10% phase error does. Keep\n"
              f"     loss.weights.amplitude at 0 and report the unit-amplitude assumption\n"
              f"     as MEASURED to be adequate here rather than merely assumed. The\n"
              f"     amplitude head is then a parameter cost with no return, and dropping\n"
              f"     it is the honest recommendation.")
        return 0
    if improves:
        print(f"the amplitude matters, and the classical reference explains the\n"
              f"     hologram BETTER than unity does ({amplitude_response:.6f} lower, "
              f"{ratio:.2f}x the\n"
              f"     phase response). The unit-amplitude assumption was costing the term\n"
              f"     accuracy. Run experiment D0 with loss.weights.amplitude > 0. Say that\n"
              f"     the reference is a reconstruction, not a measurement.")
        return 0
    print(f"the amplitude matters, but the classical reference explains the hologram\n"
          f"     WORSE than unity does. That is evidence against the reference, not for\n"
          f"     the assumption: it means the reconstruction's modulus carries errors\n"
          f"     larger than the amplitude structure it is trying to capture. Do not\n"
          f"     train on it. Report the number and keep the unit amplitude.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
