"""Does the per-cell measurement term actually reach the segmentation decoder?

WHY THIS EXISTS
---------------
If it does not, experiments B and B' are measuring nothing, and that would only
become apparent after a full training run.

The test must ISOLATE the term. Backward on the summed objective is useless here
because the Dice and cross-entropy terms produce a large segmentation gradient on
their own, so the decoder's ``.grad`` is non-zero whether or not the measurement
term contributed anything at all.

Four things this script does deliberately:

* fp32, not the production mixed-precision setting. Under AMP an unscaled
  gradient can underflow to exactly zero and be misread as "disconnected".
* ``torch.autograd.grad(..., allow_unused=True)`` rather than ``.backward()``
  then reading ``.grad``. A parameter genuinely absent from the graph returns
  ``None`` explicitly, which is a stronger statement than a zero.
* Reports the segmentation loss gradient on the same parameters and the same
  batch as a reference scale, because "non-zero" means nothing without one.
* Averages over SEVERAL batches and reports the median and the spread. The
  number of cells, their size and how many touch the border all vary batch to
  batch, and a ratio read off a single batch is not reproducible enough to set
  a loss weight from.

TWO NUMBERS, NOT ONE
--------------------
MAGNITUDE RATIO answers "will it be heard": sum |grad of the measurement term|
divided by sum |grad of the segmentation loss|, on the same parameters.

COSINE SIMILARITY answers "does it say anything new": the normalised inner
product of the two gradient vectors over those same parameters. Near +1 and the
term is asking for what Dice already asks for, so it can only reinforce, and a
better result from it is not evidence of the measurement constraining anything.
Near 0 and it is orthogonal -- new information, which is the case worth
running. Negative and it actively opposes the segmentation loss, which is
informative too but means the weight decides which one wins.

The magnitude ratio alone was misread once in this project, so read the next
section before using it.

READ THE WEIGHT ARITHMETIC, NOT THE RATIO
-----------------------------------------
The ratio printed here is measured at WEIGHT 1.0. The composite objective adds
``w * term``, so the gradient the optimiser actually sees is

    effective ratio  =  w  *  (the ratio printed here)

which means a printed ratio of 0.299 and a weight of 0.3 give 0.090, an 11:1
advantage to the segmentation loss -- not the near-parity the bare ratio might
suggest. The table at the end does this arithmetic explicitly for a set of
candidate weights and states the weight that reaches parity, so the mistake
cannot be repeated by reading the wrong line.

RUN IT AT THE CROP SIZE YOU WILL TRAIN AT
-----------------------------------------
The ratio is not a property of the term alone. Measured here: median 0.54 on
256 px crops with about 7 cells, and 0.25 to 0.29 on 512 px crops with about 20
-- roughly a factor of two, in the direction the formulation predicts, because
the per-cell mean divides by the number of cells while the segmentation loss is
a per-pixel mean that does not. A weight chosen from one crop size is the wrong
weight at another, so run this with the config the experiment will actually
use and re-run it if ``data.train_crop`` changes.

    python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml
    python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml --batches 30
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
from holoqpi.losses.terms import CellIntegratedPhase, SegmentationLoss
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device, write_csv

LOGGER = get_logger(__name__)


def flatten(grads, parameters) -> tuple[torch.Tensor, int]:
    """Concatenate a gradient tuple into one vector, counting the missing ones.

    A parameter absent from the graph contributes zeros of its own shape rather
    than being dropped, so the vectors from two different losses stay aligned
    element for element and their inner product is meaningful.
    """
    pieces = []
    disconnected = 0
    for grad, parameter in zip(grads, parameters):
        if grad is None:
            disconnected += 1
            pieces.append(torch.zeros(parameter.numel(), device=parameter.device))
        else:
            pieces.append(grad.detach().reshape(-1))
    return torch.cat(pieces), disconnected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/v2/b_cell_ipp.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batches", type=int, default=8,
                        help="batches to average over (default 8; use 30 on the server)")
    parser.add_argument("--device", default="cpu",
                        help="cpu keeps this exactly fp32; cuda is fine too")
    parser.add_argument("--weights", type=float, nargs="*",
                        default=[0.1, 0.3, 1.0, 3.0],
                        help="candidate loss weights for the arithmetic table")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)

    loader = build_dataloaders(cfg, splits_to_build=(args.split,))[args.split]

    model = build_model(cfg).to(device).float()
    model.train()

    # The parameters under test: the segmentation decoder and its head. If the
    # measurement term is to shape the boundary at all, it must reach these.
    named = [(n, p) for n, p in model.named_parameters()
             if ("segmentation_decoder" in n or "segmentation_head" in n) and p.requires_grad]
    if not named:
        named = [(n, p) for n, p in model.named_parameters()
                 if "segmentation" in n and p.requires_grad]
    names = [n for n, _ in named]
    parameters = [p for _, p in named]
    phase_parameters = [p for n, p in model.named_parameters()
                        if ("phase_decoder" in n or "phase_head" in n) and p.requires_grad]
    # Both lists are indexed and differentiated against below, so an empty one
    # is an IndexError and a torch error rather than a message. A model with no
    # matching parameter is a configuration problem worth naming.
    if not parameters:
        raise RuntimeError(
            "no trainable segmentation parameters matched 'segmentation_decoder', "
            "'segmentation_head' or 'segmentation'. Check model.share_decoder and "
            "model.lora.always_trainable_patterns."
        )
    if not phase_parameters:
        raise RuntimeError(
            "no trainable phase parameters matched 'phase_decoder' or 'phase_head'. "
            "Check model.share_decoder."
        )
    print(f"\n  probing {len(parameters)} segmentation parameters "
          f"({names[0]} ... {names[-1]})")

    physics = cfg.loss.physics
    term = CellIntegratedPhase(
        physics.volume_epsilon, physics.cell_min_reference_rad, physics.max_relative_error
    )
    segmentation_loss = SegmentationLoss(cfg.loss.segmentation, cfg.model.segmentation_classes)

    rows: list[dict] = []
    iterator = iter(loader)
    for index in range(max(1, args.batches)):
        try:
            batch = next(iterator)
        except StopIteration:
            # A short split is not an error, it is a smaller sample. Say so once
            # rather than silently reporting a median over fewer batches.
            LOGGER.info("split exhausted after %d batches", index)
            break

        hologram = batch["hologram"].to(device).float()
        phase_target = batch["phase"].to(device).float()
        mask_target = batch["mask"].to(device).long()
        instances = batch["instances"].to(device).long()

        outputs = model(hologram)
        seg_logits = outputs["segmentation"]
        phase_pred = outputs["phase"]

        foreground = 1.0 - torch.softmax(seg_logits, dim=1)[:, 0:1]
        target_foreground = (mask_target > 0).to(phase_pred.dtype).unsqueeze(1)

        # REDUCED EXACTLY AS JointPhysicsAwareLoss REDUCES IT.
        #
        # The objective gates crops whose reference foreground is below
        # loss.physics.min_foreground_pixels and then divides by the number of
        # crops that SURVIVED:
        #
        #     has_cells = (foreground_pixels >= min_foreground_pixels)
        #     masked    = (term * has_cells).sum() / has_cells.sum()
        #
        # This script used a plain .mean() over the whole batch, which includes
        # the gated crops -- and CellIntegratedPhase returns exactly zero for a
        # crop it skips. So every below-threshold crop in a batch pulled the
        # measured gradient down by a factor of n_surviving/B, and the ratio this
        # script exists to report was biased low by however often that happened.
        # That ratio is what set loss.weights.cell_integrated_phase for the
        # central arm of the study, and the script's own docstring warns that
        # "the magnitude ratio alone was misread once in this project".
        raw = term(foreground, phase_pred, target_foreground, phase_target, instances)
        if cfg.loss.physics.max_relative_error is not None:
            raw = raw.clamp(max=float(cfg.loss.physics.max_relative_error))
        foreground_pixels = target_foreground.sum(dim=[1, 2, 3])
        if cfg.loss.physics.skip_empty_targets:
            has_cells = (
                foreground_pixels >= cfg.loss.physics.min_foreground_pixels
            ).to(phase_pred.dtype)
        else:
            has_cells = torch.ones_like(foreground_pixels)
        gated = int(has_cells.numel() - int(has_cells.sum()))
        measurement = (raw * has_cells).sum() / has_cells.sum().clamp(min=1.0)
        reference_loss = segmentation_loss(seg_logits, mask_target).mean()

        measurement_grads = torch.autograd.grad(
            measurement, parameters, retain_graph=True, allow_unused=True
        )
        reference_grads = torch.autograd.grad(
            reference_loss, parameters, retain_graph=True, allow_unused=True
        )
        phase_grads = torch.autograd.grad(
            measurement, phase_parameters, retain_graph=False, allow_unused=True
        )

        measurement_vector, disconnected = flatten(measurement_grads, parameters)
        reference_vector, _ = flatten(reference_grads, parameters)
        phase_vector, _ = flatten(phase_grads, phase_parameters)

        m_sum = float(measurement_vector.abs().sum())
        r_sum = float(reference_vector.abs().sum())
        denominator = float(measurement_vector.norm()) * float(reference_vector.norm())
        cosine = (
            float((measurement_vector * reference_vector).sum()) / denominator
            if denominator > 0 else float("nan")
        )

        rows.append({
            "batch": index,
            # Recorded so the CSV can be attributed to the configuration that
            # produced it. Without these the file was unidentifiable even to the
            # run that wrote it, and the crop size in particular changes the
            # answer -- the ratio is a function of how many cells a crop holds.
            "config": str(Path(args.config)),
            "train_crop": cfg.data.train_crop,
            "batch_size": cfg.data.batch_size,
            "min_foreground_pixels": cfg.loss.physics.min_foreground_pixels,
            "crops_gated": gated,
            "cells": int(instances.max()),
            "measurement_loss": float(measurement.detach()),
            "segmentation_loss": float(reference_loss.detach()),
            "measurement_grad_sum": m_sum,
            "segmentation_grad_sum": r_sum,
            "ratio_at_weight_1": m_sum / r_sum if r_sum > 0 else float("nan"),
            "cosine": cosine,
            "phase_grad_sum": float(phase_vector.abs().sum()),
            "disconnected": disconnected,
        })
        print(f"    batch {index:<3} cells {rows[-1]['cells']:<4} "
              f"|g_meas| {m_sum:11.4e}   |g_seg| {r_sum:11.4e}   "
              f"ratio {rows[-1]['ratio_at_weight_1']:9.3e}   cos {cosine:+.3f}")

    if not rows:
        raise RuntimeError(
            f"no batches were read from split {args.split!r}. Check "
            f"{cfg.paths.data_root}/{cfg.paths.splits_file}."
        )

    # KEYED BY THE CONFIG THAT PRODUCED IT. A single fixed name was truncated by
    # whichever arm ran last, and run_v2.sh plus b2_cell_area.yaml's own comment
    # both invite running this for more than one arm.
    report = (
        Path(cfg.paths.output_root)
        / f"gradient_path_{Path(args.config).stem}_{cfg.data.train_crop}.csv"
    )
    write_csv(rows, report)

    ratios = np.array([r["ratio_at_weight_1"] for r in rows], dtype=float)
    cosines = np.array([r["cosine"] for r in rows], dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    cosines = cosines[np.isfinite(cosines)]
    phase_sum = float(np.median([r["phase_grad_sum"] for r in rows]))
    disconnected = int(max(r["disconnected"] for r in rows))
    total_gated = int(sum(r["crops_gated"] for r in rows))
    total_crops = int(sum(1 for _ in rows)) * int(cfg.data.batch_size)
    if total_gated:
        print(f"\n  {total_gated} of {total_crops} crops were below "
              f"min_foreground_pixels={cfg.loss.physics.min_foreground_pixels} and "
              f"are excluded here exactly as the objective excludes them")

    print(f"\n=== over {len(rows)} batches ===")
    if ratios.size:
        print(f"  magnitude ratio at weight 1.0   median {np.median(ratios):.4e}   "
              f"min {ratios.min():.4e}   max {ratios.max():.4e}")
    if cosines.size:
        print(f"  cosine with the seg gradient    median {np.median(cosines):+.4f}   "
              f"min {cosines.min():+.4f}   max {cosines.max():+.4f}")
    print(f"  gradient into the phase decoder median {phase_sum:.4e}"
          f"   (sanity: must be non-zero)")
    print(f"  parameters never in the graph    {disconnected}/{len(parameters)}")
    print(f"  per-batch detail -> {report}")

    median_ratio = float(np.median(ratios)) if ratios.size else 0.0

    print("\n=== the weight arithmetic ===")
    print("  the objective adds w * term, so the gradient the optimiser sees is")
    print("  w times the ratio above. THIS is the number to choose a weight from.")
    print(f"\n    {'weight w':>10}   {'effective ratio':>16}   {'seg : term':>14}")
    for weight in args.weights:
        effective = weight * median_ratio
        if effective <= 0:
            balance = "seg only"
        elif effective >= 1.0:
            balance = f"1 : {effective:.2f}"
        else:
            balance = f"{1 / effective:.2f} : 1"
        print(f"    {weight:>10.3f}   {effective:>16.4f}   {balance:>14}")
    if median_ratio > 0:
        print(f"\n  parity (effective ratio 1.0) is at w = {1 / median_ratio:.3f}")

    print("\n  -> ", end="")
    if disconnected == len(parameters) or median_ratio == 0.0:
        print("DISCONNECTED. The term does not reach the segmentation decoder at all.\n"
              "     Experiments B and B' would measure nothing. Fix before training.")
        return 2
    if phase_sum <= 0:
        print("no gradient reaches the phase decoder, which should be impossible for\n"
              "     this term. Investigate before trusting anything above.")
        return 2
    if cosines.size and np.median(cosines) > 0.9:
        print(f"CONNECTED, but at cosine {np.median(cosines):+.2f} this term is asking for\n"
              "     almost exactly what the segmentation loss already asks for. It can\n"
              "     reinforce but it is not adding a constraint, so a better result from\n"
              "     it is NOT evidence that the measurement shaped the boundary. Say so.")
        return 1
    # THE THRESHOLD, AND WHY IT IS 0.1 AND NOT 1e-3.
    #
    # "Comparable in scale" has to mean within an order of magnitude, or the
    # phrase is not true. The old bound was 1e-3, so a term 10x to 1000x weaker
    # than the segmentation loss fell through to the final message and was
    # reported as "CONNECTED, comparable in scale" -- and that band is exactly
    # where this project's own measurements landed: the docstring quotes ratios
    # of 0.25 to 0.54 and works through 0.299 x 0.3 = 0.090 as "an 11:1 advantage
    # to the segmentation loss", i.e. not comparable. The verdict contradicted
    # the worked example beside it.
    #
    # 0.1 is the order-of-magnitude reading, and either way the effective-ratio
    # table above is what the weight should be taken from -- the verdict only
    # decides whether the default weight of 1.0 is defensible unexamined.
    _COMPARABLE = 0.1
    if median_ratio < _COMPARABLE:
        print(f"CONNECTED but {1 / median_ratio:.1f}x weaker than the segmentation loss at\n"
              f"     weight 1.0, so it is not comparable in scale and will be swamped\n"
              f"     unless the weight is raised deliberately. Take the weight from the\n"
              f"     table above -- parity is at w = {1 / median_ratio:.3f} -- not a round\n"
              f"     number, and report the ratio alongside it.")
        return 1
    print(f"CONNECTED, comparable in scale (ratio {median_ratio:.3f}, within an order of\n"
          f"     magnitude of the segmentation loss) and not collinear with it. The term\n"
          f"     can shape the boundary and is adding information. Choose the weight from\n"
          f"     the table above to set the intended balance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
