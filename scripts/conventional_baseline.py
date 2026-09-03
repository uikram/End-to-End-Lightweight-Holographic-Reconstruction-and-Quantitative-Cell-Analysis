"""Classical reconstruction, scored exactly like the network.

Without this the study cannot answer the first question a reviewer will ask:
what does the network buy over the textbook pipeline? A Dice of 0.83 and a
dry-mass MAPE of 0.19 mean nothing until something else has been measured on the
same test split with the same instance labelling and the same measurement chain.

Two classical pipelines, one per geometry, each the standard method for it:

    off-axis   isolate one first-order sideband in the Fourier plane, shift the
               carrier to the origin, back-propagate, take the argument. The
               twin image and the DC term are removed analytically, which is the
               whole advantage of the geometry.

    in-line    back-propagate the recorded intensity to the sample plane and
               take the argument. The twin image stays superposed on the object,
               which is the failure this geometry is known for. Optional
               Gerchberg-Saxton iterations with a non-negative absorption
               constraint give the classical method its fairest chance before
               the comparison is drawn.

Both are followed by the *same* phase-derived mask generation the network's
targets came from, so the segmentation comparison is like for like.

Two corrections are applied before scoring, and both are necessary for the
baseline to be a fair opponent rather than a strawman.

    SIDEBAND CHOICE the two first-order sidebands are conjugates, and taking
                    the wrong one returns the negated phase. Which peak is
                    numerically larger is arbitrary, so the choice is made on
                    physical grounds: cells add optical path, so the correct
                    reconstruction is right-skewed in phase.

    UNWRAPPING      reconstructed phase is known modulo 2 pi. A cell thicker
                    than one wavelength of optical path wraps, and dry mass is
                    an integral of the unwrapped phase.

    ABERRATION      the objective imposes a curvature term and the reference
                    beam a tilt; together they dominate the recovered phase. A
                    polynomial surface is fitted to the background and
                    subtracted, which is the numerical aberration compensation a
                    DHM operator performs. On this dataset order 3 is needed
                    before the recovered in-cell contrast matches the reference;
                    a plane fit alone leaves the reconstruction unusable.

    python scripts/conventional_baseline.py --config config/base.yaml
    python scripts/conventional_baseline.py --config config/base.yaml --gs-iterations 20
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
from holoqpi.engine import Evaluator, save_per_cell
from holoqpi.physics import reconstruct_gabor, reconstruct_off_axis, unwrap_phase_2d
from holoqpi.utils import get_logger, resolve_device, write_csv, write_json

LOGGER = get_logger(__name__)


def _polynomial_design(height: int, width: int, order: int, device) -> torch.Tensor:
    """Design matrix for a 2-D polynomial surface of the given total order."""
    y = torch.linspace(-1, 1, height, device=device).view(-1, 1).expand(height, width)
    x = torch.linspace(-1, 1, width, device=device).view(1, -1).expand(height, width)
    columns = [(x ** i) * (y ** j)
               for i in range(order + 1) for j in range(order + 1 - i)]
    return torch.stack(columns, dim=-1).reshape(-1, len(columns))


def remove_aberration(phase: torch.Tensor, mask: torch.Tensor, order: int) -> torch.Tensor:
    """Fit a polynomial surface to the background and subtract it.

    A classical reconstruction carries far more than a constant offset. The
    microscope objective imposes a curvature term, and the reference beam adds a
    tilt; together they dominate the recovered phase and would swamp the cells
    entirely. Removing them is not cosmetic tidying but the standard numerical
    aberration compensation a DHM operator performs, and omitting it would make
    this baseline a strawman.

    A first-order fit removes only the tilt. On this dataset the curvature is
    strong enough that order 3 is needed before the recovered in-cell phase
    contrast matches the reference; the order is exposed as a parameter so that
    choice is visible rather than buried.
    """
    batch, _, height, width = phase.shape
    design = _polynomial_design(height, width, order, phase.device)

    corrected = []
    for item in range(batch):
        values = phase[item, 0].reshape(-1)
        background = ~mask[item].reshape(-1).bool()
        if background.sum() < design.shape[1] * 4:
            corrected.append(phase[item, 0] - values.median())
            continue
        solution = torch.linalg.lstsq(
            design[background], values[background].unsqueeze(1)
        ).solution
        corrected.append(phase[item, 0] - (design @ solution).reshape(height, width))
    return torch.stack(corrected).unsqueeze(1)


def select_conjugate(field: torch.Tensor) -> torch.Tensor:
    """Choose between the two conjugate sidebands on physical grounds.

    An off-axis hologram carries the object in two conjugate first-order
    sidebands. Picking the wrong one returns the conjugate field and therefore
    the negated phase, which looks like a plausible reconstruction and scores as
    an anti-correlation. Which peak is numerically larger is arbitrary, so the
    choice cannot be left to the spectrum.

    The disambiguating fact is physical: cells are optically denser than their
    medium, so they add optical path. A field of cells on a flat background is
    therefore right-skewed in phase. Skewness needs no mask and no threshold, so
    it can be applied before segmentation exists.
    """
    selected = []
    for item in range(field.shape[0]):
        angle = torch.angle(field[item])
        centred = angle - angle.mean()
        deviation = centred.std().clamp(min=1e-8)
        skew = float((centred ** 3).mean() / deviation ** 3)
        selected.append(field[item] if skew >= 0 else field[item].conj())
    return torch.stack(selected)


def build_predictor(cfg, modality: str, device, distance_um: float,
                    gs_iterations: int, aberration_order: int):
    """A predict_fn for the Evaluator that reconstructs classically.

    Returns the callable and a diagnostics dictionary it fills in as it runs.
    """
    diagnostics: dict[str, list] = {"contrast": []}
    optics = cfg.optics
    pixel_area = calibration_from_config(cfg).pixel_area_um2
    frontend = cfg.model.frontend
    num_classes = cfg.model.segmentation_classes
    num_conditions = cfg.model.condition_classes

    def predict(batch: dict) -> dict:
        hologram = batch["hologram"].to(device).float()

        if modality == "off_axis":
            field = reconstruct_off_axis(
                hologram, optics.wavelength_um,
                optics.pixel_pitch_x_um, optics.pixel_pitch_y_um,
                distance_um, frontend.sideband_radius_px, frontend.dc_exclusion_px,
            )
        else:
            field = reconstruct_gabor(
                hologram, optics.wavelength_um,
                optics.pixel_pitch_x_um, optics.pixel_pitch_y_um,
                distance_um, iterations=gs_iterations,
            )

        field = select_conjugate(field)
        phase = unwrap_phase_2d(torch.angle(field))

        # First pass gives a rough mask, used only to mark which pixels count as
        # background for the aberration fit; the mask is rebuilt afterwards on
        # the corrected phase, so a poor first guess costs accuracy in the fit
        # but does not propagate into the reported segmentation.
        rough = torch.from_numpy(
            np.stack([
                build_mask(p, cfg.mask_generation, pixel_area)
                for p in phase[:, 0].cpu().numpy()
            ])
        ).to(device)
        phase = remove_aberration(phase, rough, aberration_order)

        masks = np.stack([
            build_mask(p, cfg.mask_generation, pixel_area)
            for p in phase[:, 0].cpu().numpy()
        ])
        mask = torch.from_numpy(masks).to(device).long()

        # The Evaluator reads argmax over the class axis, so present the mask as
        # saturated logits. The classical pipeline has no classifier; uniform
        # logits make that explicit rather than pretending to a prediction.
        logits = torch.stack(
            [(mask == index).float() * 10.0 for index in range(num_classes)], dim=1
        )
        condition = torch.zeros(hologram.shape[0], num_conditions, device=device)

        # Sanity signal, accumulated so the caller can tell a genuinely poor
        # classical result from a broken pipeline. Cells must end up optically
        # denser than the medium; if they do not, the reconstruction has failed
        # (wrong conjugate, wrong z, or aberration left in) and the metrics that
        # follow describe the failure rather than the method.
        reference_mask = batch["mask"].to(device) > 0
        for item in range(phase.shape[0]):
            inside = reference_mask[item]
            if inside.any() and (~inside).any():
                diagnostics["contrast"].append(
                    float(phase[item, 0][inside].mean() - phase[item, 0][~inside].mean())
                )
        return {"phase": phase, "segmentation": logits, "condition": condition}

    return predict, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--distance-um", type=float, default=None,
                        help="default: loss.forward_model.distance_um, else 0")
    parser.add_argument("--gs-iterations", type=int, default=0,
                        help="Gerchberg-Saxton iterations for the in-line arm")
    parser.add_argument("--aberration-order", type=int, default=3,
                        help="polynomial order for numerical aberration removal")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = resolve_device(args.device)
    output_root = Path(cfg.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    distance = args.distance_um
    if distance is None:
        distance = cfg.loss.forward_model.distance_um
    if distance is None:
        LOGGER.warning(
            "no propagation distance given and loss.forward_model.distance_um is null; "
            "reconstructing at z = 0. For the in-line arm that is degenerate -- a pure "
            "phase object produces no in-line contrast at zero defocus -- so the Gabor "
            "baseline will look far worse than the method deserves. Run "
            "scripts/calibrate_z.py or pass --distance-um."
        )
        distance = 0.0

    results = {}
    for modality in args.modality:
        modality_cfg = cfg.merged({"data": {"modality": modality}})
        loader = build_dataloaders(modality_cfg, splits_to_build=(args.split,))[args.split]
        evaluator = Evaluator(modality_cfg, device)
        predictor, diagnostics = build_predictor(
            modality_cfg, modality, device, float(distance),
            args.gs_iterations, args.aberration_order,
        )

        LOGGER.info("conventional %s reconstruction at z = %.2f um (GS iterations %d)",
                    modality, distance, args.gs_iterations)
        evaluation = evaluator.run(
            None, loader, collect_per_cell=cfg.evaluation.save_per_cell_csv,
            predict_fn=predictor,
        )
        metrics = evaluation["metrics"]
        # The classical pipeline predicts no condition; drop the meaningless
        # classification entries rather than reporting chance accuracy as a result.
        for key in [k for k in metrics if k.startswith("cls_")]:
            metrics.pop(key)
        metrics["reconstruction"] = "conventional"
        metrics["distance_um"] = float(distance)
        metrics["gs_iterations"] = int(args.gs_iterations)
        metrics["aberration_order"] = int(args.aberration_order)

        contrast = float(np.median(diagnostics["contrast"])) if diagnostics["contrast"] else float("nan")
        metrics["recovered_phase_contrast_rad"] = contrast
        metrics["reconstruction_valid"] = bool(contrast > 0.1)
        results[modality] = metrics

        destination = output_root / f"conventional_{modality}"
        destination.mkdir(parents=True, exist_ok=True)
        write_json(metrics, destination / f"metrics_{args.split}.json")
        if cfg.evaluation.save_per_cell_csv:
            save_per_cell(evaluation["per_cell"], destination / f"per_cell_{args.split}.csv")
            save_per_cell(evaluation["unmatched"], destination / f"unmatched_{args.split}.csv")

        print(f"\n=== conventional {modality} ===")
        print(f"  {'recovered in-cell phase contrast':<28} {contrast:+.4f} rad")
        if not metrics["reconstruction_valid"]:
            print("  !! The reconstruction did NOT recover positive cell contrast, so the\n"
                  "     metrics below describe a failed reconstruction, not the classical\n"
                  "     method's real performance. Do not report them as a baseline.\n"
                  "     Most likely causes, in order: the propagation distance is wrong\n"
                  "     (run scripts/calibrate_z.py, or ask for z); the aberration order is\n"
                  "     too low for this objective (try --aberration-order 4 or 5); for the\n"
                  "     in-line arm, z = 0 is degenerate and no phase can be recovered.")
        for key in ("phase_mae_rad", "phase_mae_rad_in_cell", "phase_pearson_r",
                    "seg_dice", "seg_aji", "seg_boundary_f1",
                    "detection_recall", "detection_f1",
                    "area_mape", "dry_mass_mape", "dry_mass_relative_bias"):
            if key in metrics:
                print(f"  {key:<28} {metrics[key]:.4f}")

    rows = []
    for key in sorted({k for m in results.values() for k in m}):
        row = {"metric": key}
        row.update({m: results[m].get(key) for m in results})
        rows.append(row)
    table = output_root / f"conventional_baseline_{args.split}.csv"
    write_csv(rows, table)
    write_json(results, table.with_suffix(".json"))
    print(f"\nbaseline table -> {table}")
    print("Compare against runs/*_modality_comparison.csv; the two are scored by the\n"
          "same evaluator, so the difference is attributable to reconstruction alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
