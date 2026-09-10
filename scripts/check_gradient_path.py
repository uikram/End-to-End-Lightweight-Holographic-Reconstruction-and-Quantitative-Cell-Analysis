"""Does the per-cell measurement term actually reach the segmentation decoder?

WHY THIS EXISTS
---------------
If it does not, experiments B and B' are measuring nothing, and that would only
become apparent after a full training run.

The test must ISOLATE the term. Backward on the summed objective is useless here
because the Dice and cross-entropy terms produce a large segmentation gradient on
their own, so the decoder's ``.grad`` is non-zero whether or not the measurement
term contributed anything at all.

Three things this script does deliberately:

* fp32, not the production mixed-precision setting. Under AMP an unscaled
  gradient can underflow to exactly zero and be misread as "disconnected".
* ``torch.autograd.grad(..., allow_unused=True)`` rather than ``.backward()``
  then reading ``.grad``. A parameter genuinely absent from the graph returns
  ``None`` explicitly, which is a stronger statement than a zero.
* Reports the segmentation loss gradient on the same parameters and the same
  batch as a reference scale, because "non-zero" means nothing without one.

    python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.data import build_dataloaders
from holoqpi.losses.terms import CellIntegratedPhase, SegmentationLoss
from holoqpi.models import build_model
from holoqpi.utils import get_logger, resolve_device

LOGGER = get_logger(__name__)


def summarise(name: str, grads, parameters) -> tuple[float, float, int]:
    total = 0.0
    largest = 0.0
    disconnected = 0
    for grad, parameter in zip(grads, parameters):
        if grad is None:
            disconnected += 1
            continue
        total += float(grad.abs().sum())
        largest = max(largest, float(grad.abs().max()))
    print(f"    {name:<34} sum |grad| {total:12.6e}   max |grad| {largest:12.6e}"
          f"   disconnected {disconnected}/{len(parameters)}")
    return total, largest, disconnected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/v2/b_cell_ipp.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cpu",
                        help="cpu keeps this exactly fp32; cuda is fine too")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)

    loader = build_dataloaders(cfg, splits_to_build=(args.split,))[args.split]
    batch = next(iter(loader))

    model = build_model(cfg).to(device).float()
    model.train()

    hologram = batch["hologram"].to(device).float()
    phase_target = batch["phase"].to(device).float()
    mask_target = batch["mask"].to(device).long()
    instances = batch["instances"].to(device).long()

    outputs = model(hologram)
    seg_logits = outputs["segmentation"]
    phase_pred = outputs["phase"]

    foreground = 1.0 - torch.softmax(seg_logits, dim=1)[:, 0:1]
    target_foreground = (mask_target > 0).to(phase_pred.dtype).unsqueeze(1)

    # The parameters under test: the segmentation decoder and its head. If the
    # measurement term is to shape the boundary at all, it must reach these.
    named = [(n, p) for n, p in model.named_parameters()
             if ("segmentation_decoder" in n or "segmentation_head" in n) and p.requires_grad]
    if not named:
        named = [(n, p) for n, p in model.named_parameters()
                 if "segmentation" in n and p.requires_grad]
    names = [n for n, _ in named]
    parameters = [p for _, p in named]
    print(f"\n  probing {len(parameters)} segmentation parameters "
          f"({names[0]} ... {names[-1]})")
    phase_parameters = [p for n, p in model.named_parameters()
                        if ("phase_decoder" in n or "phase_head" in n) and p.requires_grad]

    physics = cfg.loss.physics
    term = CellIntegratedPhase(
        physics.volume_epsilon, physics.cell_min_reference_rad, physics.max_relative_error
    )
    measurement = term(foreground, phase_pred, target_foreground, phase_target, instances).mean()

    segmentation_loss = SegmentationLoss(cfg.loss.segmentation, cfg.model.segmentation_classes)
    reference_loss = segmentation_loss(seg_logits, mask_target).mean()

    print(f"\n  batch {hologram.shape[0]} x {tuple(hologram.shape[-2:])}   "
          f"cells in batch {int(instances.max())}   "
          f"per-cell loss {float(measurement):.6f}   seg loss {float(reference_loss):.6f}")

    print("\n  gradient into the SEGMENTATION decoder + head")
    measurement_grads = torch.autograd.grad(
        measurement, parameters, retain_graph=True, allow_unused=True
    )
    m_sum, m_max, m_none = summarise("cell_integrated_phase alone", measurement_grads, parameters)
    reference_grads = torch.autograd.grad(
        reference_loss, parameters, retain_graph=True, allow_unused=True
    )
    r_sum, r_max, _ = summarise("segmentation (Dice+CE) alone", reference_grads, parameters)

    print("\n  gradient into the PHASE decoder + head (sanity: this must be non-zero)")
    phase_grads = torch.autograd.grad(
        measurement, phase_parameters, retain_graph=True, allow_unused=True
    )
    p_sum, _, _ = summarise("cell_integrated_phase alone", phase_grads, phase_parameters)

    ratio = m_sum / r_sum if r_sum > 0 else float("nan")
    print(f"\n  ratio of segmentation-path gradient, measurement / segmentation: {ratio:.3e}")

    print("\n  -> ", end="")
    if m_none == len(parameters) or m_sum == 0.0:
        print("DISCONNECTED. The term does not reach the segmentation decoder at all.\n"
              "     Experiments B and B' would measure nothing. Fix before training.")
        return 2
    if ratio < 1e-3:
        print(f"REACHES the decoder but {1/ratio:.0f}x weaker than the segmentation\n"
              f"     loss on the same batch. It will be swamped unless the weight is\n"
              f"     raised deliberately. Report the ratio and choose the weight from it\n"
              f"     rather than from a round number.")
        return 1
    print("CONNECTED and comparable in scale to the segmentation loss. The term can\n"
          "     shape the boundary. Choose its weight to set the intended balance.")
    if p_sum <= 0:
        print("\n  WARNING: no gradient reaches the phase decoder, which should be\n"
              "  impossible for this term. Investigate before trusting anything above.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
