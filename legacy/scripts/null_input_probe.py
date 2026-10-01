"""Does the model invent cells when given an input that contains none?

WHY THIS EXISTS
---------------
The in-line (Gabor) arm performs far better than the optics predict. At the
acquisition distance the specimen is nearly in focus, an in-line intensity of a
pure phase object is |A exp(i phi)|^2 = A^2 and contains no phi at all, and the
classical pipeline duly fails: in-cell contrast -0.073 rad, correlation -0.155.
The learned model nonetheless reaches Dice 0.761 and phase MAE 0.184 rad on the
same holograms.

There are two explanations and they have opposite consequences for the paper.
Either the network extracts a real, weak signal that classical single-step
back-propagation cannot reach -- which is a genuine and interesting result -- or
it has learned a shape prior and is drawing plausible cells largely regardless of
the input, which would make the in-line numbers uncitable as reconstruction.

This probe distinguishes them. Three inputs that contain no cell:

    zero            a constant field. Because the network is fed z-scored
                    holograms, a genuinely flat field standardises to exactly
                    zeros -- so "flat background" is not separately expressible
                    and is not offered as a tier.
    noise           Gaussian noise matched to the real hologram's mean and
                    standard deviation.
    shuffled_real   a REAL hologram with its pixels randomly permuted. Every
                    statistic except spatial arrangement is that of a genuine
                    measurement, so anything the model draws on it is drawn
                    from a prior.

Counting uses a FIXED phase threshold taken from the label rule, not Otsu. Otsu
is adaptive: it splits whatever histogram it is handed and can never report "no
structure", so on a noise-driven output it manufactures blobs. The first version
of this script counted with Otsu and duly reported 57 "cells" on pure noise
against 26.6 on real holograms -- more structure from noise than from data, which
is a broken metric rather than a finding. The Otsu count is still printed beside
the fixed-threshold count so the discrepancy stays visible.

One phase map per tier is saved for visual confirmation, but the count is the
criterion.

    python scripts/null_input_probe.py --config config/base.yaml --modality gabor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.analysis.cells import calibration_from_config
from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.data.masks import build_mask
from holoqpi.engine import load_checkpoint
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device, run_directory, write_json

LOGGER = get_logger(__name__)


def count_cells(phase: np.ndarray, cfg, pixel_area: float,
                level: float | None) -> tuple[int, float, int]:
    """Cell-sized components in a phase map.

    Returns (fixed-threshold count, fraction above threshold, Otsu count).

    THE FIXED THRESHOLD IS THE ONE THAT COUNTS, and getting this wrong inverted
    the verdict on the first run of this probe. Otsu is ADAPTIVE: it splits
    whatever histogram it is given, so it can never report "no structure". Handed
    a noise-driven output it manufactures blobs, and the probe then reports more
    cells on pure noise (57) than on real holograms (26.6) -- which is a broken
    metric, not a shape prior.

    A fixed level taken from the real data asks the question that was intended:
    does the output contain phase that would be CALLED a cell by the same rule
    the labels use? Cells raise the optical path, so the level is positive; an
    output sitting at negative phase contains no cell by definition.

    The Otsu count is still returned, and reported alongside, so the difference
    between the two is visible rather than hidden.
    """
    from scipy import ndimage

    otsu_mask = build_mask(phase, cfg.mask_generation, pixel_area)
    _, otsu_count = ndimage.label(otsu_mask > 0)

    if level is None:
        return int(otsu_count), float(otsu_mask.mean()), int(otsu_count)

    fixed_cfg = cfg.mask_generation.merged({
        "threshold_method": "fixed", "fixed_threshold_rad": float(level)
    })
    mask = build_mask(phase, fixed_cfg, pixel_area)
    _, count = ndimage.label(mask > 0)
    return int(count), float(mask.mean()), int(otsu_count)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modality", default="gabor", choices=["gabor", "off_axis"])
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fields", type=int, default=8,
                        help="real fields used for the reference count and statistics")
    parser.add_argument("--threshold-rad", type=float, default=None,
                        help="fixed phase level counted as cell; default: "
                             "mask_generation.fixed_threshold_rad")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set)).merged(
        {"data": {"modality": args.modality}}
    )
    device = resolve_device(args.device)
    experiment = args.experiment or cfg.experiment_name
    pixel_area = calibration_from_config(cfg).pixel_area_um2

    run_dir = run_directory(cfg.paths.output_root, experiment, args.modality)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    loader = build_dataloaders(cfg, splits_to_build=("test",))["test"]

    # Real reference: the statistics the synthetic inputs must be matched to, and
    # the cell count a genuine input produces.
    # The level the study's own labels use, so "is this a cell" means the same
    # thing here as it does everywhere else in the project.
    level = args.threshold_rad
    if level is None:
        level = float(cfg.mask_generation.fixed_threshold_rad)
    real_counts, real_otsu, real_fraction, means, deviations = [], [], [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.fields:
                break
            hologram = batch["hologram"].to(device).float()
            raw = batch.get("hologram_raw")
            if raw is not None:
                means.append(float(raw.mean()))
                deviations.append(float(raw.std()))
            last_hologram = hologram
            phase = model(hologram)["phase"][0, 0].cpu().numpy()
            fixed, fraction, otsu = count_cells(phase, cfg, pixel_area, level)
            real_counts.append(fixed)
            real_fraction.append(fraction)
            real_otsu.append(otsu)

    shape = hologram.shape
    reference_mean = float(np.mean(means)) if means else 0.0
    reference_std = float(np.mean(deviations)) if deviations else 1.0
    generator = torch.Generator(device="cpu").manual_seed(cfg.project.seed)

    # The network is fed NORMALISED holograms, so each synthetic tier is built in
    # raw sensor units and then z-scored exactly as the dataloader would. A flat
    # field has zero variance, so it standardises to zeros rather than dividing
    # by zero.
    def standardise(x: torch.Tensor) -> torch.Tensor:
        spread = x.std()
        return (x - x.mean()) / spread if float(spread) > 1e-8 else torch.zeros_like(x)

    # THE NETWORK SEES Z-SCORED INPUT, which constrains what tiers can exist. A
    # genuinely flat hologram standardises to exactly zeros -- it IS the zero
    # tier, and cannot be tested separately. An earlier version of this script
    # built a near-constant field and standardised it, which rescaled its
    # negligible noise to unit variance and silently produced a second copy of
    # the noise tier; the two outputs were identical to three decimals and the
    # probe reported three tiers while testing two.
    #
    # The informative third tier is a real hologram with its SPATIAL structure
    # destroyed and its intensity histogram preserved exactly, by shuffling the
    # pixels. Anything the model draws on that is drawn from a prior, because
    # every statistic except spatial arrangement is that of a real measurement.
    real_batch = last_hologram.detach().cpu()
    flat_index = torch.randperm(real_batch[0, 0].numel(), generator=generator)
    shuffled = real_batch[0, 0].reshape(-1)[flat_index].reshape(1, 1, *shape[-2:])

    tiers = {
        "zero": torch.zeros(shape[:1] + (1,) + tuple(shape[-2:])),
        "noise": standardise(
            reference_mean + reference_std * torch.randn(1, 1, *shape[-2:],
                                                         generator=generator)
        ),
        "shuffled_real": standardise(shuffled),
    }

    print(f"\n=== null-input probe: {experiment} / {args.modality} ===")
    print(f"  checkpoint {checkpoint}")
    print(f"  real holograms: mean {reference_mean:.2f}, sd {reference_std:.2f} "
          f"(raw sensor units)")
    print(f"  cell threshold: phase > {level:+.3f} rad (fixed, from the label rule)")
    print(f"  REAL input -> {np.mean(real_counts):.1f} cells per field by that rule "
          f"(range {min(real_counts)}-{max(real_counts)}), "
          f"{np.mean(real_fraction):.2%} of pixels above it; "
          f"{np.mean(real_otsu):.1f} cells by adaptive Otsu")
    print(f"\n  {'input':<16}{'cells':>8}{'above thr':>12}"
          f"{'phase mean':>13}{'phase sd':>11}{'(otsu)':>9}")

    payload: dict = {
        "experiment": experiment, "modality": args.modality,
        "checkpoint": str(checkpoint),
        "threshold_rad": float(level),
        "real_cells_mean": float(np.mean(real_counts)),
        "real_fraction_above_threshold": float(np.mean(real_fraction)),
        "real_cells_otsu_mean": float(np.mean(real_otsu)),
        "real_cells_range": [int(min(real_counts)), int(max(real_counts))],
        "tiers": {},
    }
    figures = Path(cfg.paths.output_root) / "null_probe"
    figures.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for name, tensor in tiers.items():
            phase = model(tensor.to(device).float())["phase"][0, 0].cpu().numpy()
            count, foreground, otsu = count_cells(phase, cfg, pixel_area, level)
            print(f"  {name:<16}{count:>8}{foreground:>11.2%}"
                  f"{phase.mean():>13.4f}{phase.std():>11.4f}{otsu:>9}")
            payload["tiers"][name] = {
                "cells": count, "cells_otsu": otsu, "fraction_above_threshold": foreground,
                "phase_mean": float(phase.mean()), "phase_std": float(phase.std()),
            }
            np.save(figures / f"{experiment}_{args.modality}_{name}_phase.npy", phase)

    # If the fixed threshold finds almost nothing on REAL input either, it is
    # mis-set for this model's output scale and the comparison is meaningless in
    # both directions. Say so rather than returning a confident CLEAN.
    if np.mean(real_counts) < 5:
        print(f"\n  !! The fixed threshold {level:+.3f} rad finds only "
              f"{np.mean(real_counts):.1f} cells per field on REAL holograms, so it is\n"
              f"     too high for this model's output scale and the null comparison is\n"
              f"     uninformative. Re-run with --threshold-rad set near the model's own\n"
              f"     in-cell phase level before drawing any conclusion.")
        payload["verdict"] = "threshold_misset"
        write_json(payload, figures / f"{experiment}_{args.modality}_null_probe.json")
        return 2

    invented = max(entry["cells"] for entry in payload["tiers"].values())
    payload["max_cells_on_null_input"] = invented
    write_json(payload, figures / f"{experiment}_{args.modality}_null_probe.json")

    # A GRADED VERDICT, because the binary one was too crude for what this
    # measures. Three distinct situations, with different consequences:
    #
    #   unconditional prior  the ZERO tier produces cells. The model draws
    #                        regardless of input. This is the damning case and
    #                        the one that would invalidate a reconstruction claim.
    #   invents              null inputs produce structure comparable to real
    #                        input. Also disqualifying.
    #   out-of-distribution  zero is clean, but inputs with realistic contrast
    #   hallucination        and no meaningful content produce some structure at
    #                        a small fraction of the real rate. This is a
    #                        robustness caveat, NOT evidence of memorisation:
    #                        the model responds to contrast, and given contrast
    #                        without content it produces some blobs.
    #   clean                nothing above the threshold anywhere.
    #
    # The FRACTION of pixels above the threshold is the primary statistic rather
    # than the component count: it is scale-free, and a handful of large blobs
    # and a handful of small ones read very differently in a count while meaning
    # much the same thing.
    real_fraction_mean = float(np.mean(real_fraction))
    zero_cells = payload["tiers"]["zero"]["cells"]
    worst_fraction = max(e["fraction_above_threshold"] for e in payload["tiers"].values())
    fraction_ratio = worst_fraction / real_fraction_mean if real_fraction_mean > 0 else 0.0
    count_ratio = invented / np.mean(real_counts) if np.mean(real_counts) > 0 else 0.0

    print(f"\n  worst null tier reaches {worst_fraction:.2%} of pixels above threshold, "
          f"{fraction_ratio:.1%} of the real {real_fraction_mean:.2%}")
    print(f"  worst null tier produces {invented} cells, "
          f"{count_ratio:.1%} of the real {np.mean(real_counts):.1f}")
    print(f"  zero-input tier produces {zero_cells} cells")

    print("\n  -> ", end="")
    if zero_cells > max(1, 0.05 * np.mean(real_counts)):
        print(f"UNCONDITIONAL SHAPE PRIOR. The model draws {zero_cells} cells from a\n"
              f"     CONSTANT input. It is not reconstructing; it is generating. The\n"
              f"     {args.modality} result cannot be cited as phase retrieval.")
        payload_note = "unconditional_prior"
    elif fraction_ratio >= 0.25:
        print(f"INVENTS CELLS. Null inputs reach {fraction_ratio:.0%} of the real\n"
              f"     above-threshold area. Too close to the real rate to attribute to\n"
              f"     out-of-distribution behaviour. Do not cite the {args.modality}\n"
              f"     result as evidence of phase retrieval.")
        payload_note = "invents"
    elif fraction_ratio >= 0.05:
        print(f"OUT-OF-DISTRIBUTION HALLUCINATION, not memorisation. A constant input\n"
              f"     produces {zero_cells} cells, so there is no unconditional shape prior:\n"
              f"     the model does respond to its input. But inputs with realistic\n"
              f"     contrast and no content still produce {fraction_ratio:.0%} of the real\n"
              f"     above-threshold area. Report this as a robustness limitation and\n"
              f"     cite the control; it does NOT by itself invalidate the\n"
              f"     {args.modality} reconstruction result.")
        payload_note = "ood_hallucination"
    else:
        print(f"CLEAN. Null inputs reach only {fraction_ratio:.1%} of the real\n"
              f"     above-threshold area, and a constant input produces {zero_cells} cells.\n"
              f"     The {args.modality} result is not shape-prior memorisation. The\n"
              f"     mechanism is still unestablished and should be hedged as such.")
        payload_note = "clean"
    payload["fraction_ratio"] = float(fraction_ratio)
    payload["count_ratio"] = float(count_ratio)
    payload["verdict"] = payload_note
    write_json(payload, figures / f"{experiment}_{args.modality}_null_probe.json")
    print(f"\n  phase maps and verdict -> {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
