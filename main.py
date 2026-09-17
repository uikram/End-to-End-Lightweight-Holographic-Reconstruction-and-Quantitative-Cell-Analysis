#!/usr/bin/env python
"""Command-line entry point.

    python main.py prepare   --config config/base.yaml
    python main.py train     --config config/off_axis.yaml
    python main.py evaluate  --config config/off_axis.yaml
    python main.py compare   --config config/base.yaml
    python main.py benchmark --config config/base.yaml
    python main.py export    --config config/off_axis.yaml

Any configuration value can be overridden inline, for example:

    python main.py train --config config/gabor.yaml --set training.epochs=5 data.batch_size=2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from holoqpi.config import load_config, parse_overrides, save_config
from holoqpi.utils import add_file_logging, get_logger, resolve_device, seed_everything

LOGGER = get_logger("holoqpi")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to a YAML configuration file")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted configuration overrides, e.g. training.epochs=5",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end lightweight holographic reconstruction and cell analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build masks, splits and the manifest")
    _add_common(prepare)
    prepare.add_argument("--force-masks", action="store_true", help="regenerate existing masks")

    train = subparsers.add_parser("train", help="train one modality")
    _add_common(train)
    train.add_argument(
        "--init-from", default=None,
        help="checkpoint to initialise weights from before fine-tuning. Every "
             "experiment condition must start from the SAME pretrained "
             "checkpoint, or a difference between conditions could be "
             "pretraining noise rather than the effect of the loss under test.",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate a trained checkpoint")
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", default=None, help="defaults to the run's best_model.pt")
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument(
        "--tag", default=None,
        help="suffix for the output files, so a second label source (e.g. "
             "paths.manual_mask_dir=membrane_mask) does not overwrite the first",
    )
    evaluate.add_argument(
        "--allow-untrained", action="store_true",
        help="evaluate even when no checkpoint is found (scores random weights)",
    )

    compare = subparsers.add_parser("compare", help="run the off-axis / Gabor comparison")
    _add_common(compare)
    compare.add_argument("--init-from", default=None,
                         help="see `train --init-from`; applies to every arm")
    compare.add_argument("--modalities", nargs="+", default=["off_axis", "gabor"])
    # TRAINING IS OPT-IN HERE, AND IT DID NOT USED TO BE.
    #
    # `compare` exists for two jobs: training a matched pair of arms, and pairing
    # two arms that are ALREADY trained so that the modality-comparison JSON and
    # figures 5, 13 and 14 can be written. The second is overwhelmingly the
    # common one -- run_v2.sh trains the arms in stage 6 and then needs the
    # pairing -- and with training on by default the documented invocation
    # `python main.py compare --config config/v2/a_baseline.yaml` silently
    # retrained both arms for 60 epochs each and overwrote the two checkpoints
    # the study was about to report. A default that can destroy a day of GPU time
    # and two results is the wrong default.
    compare.add_argument(
        "--train", action="store_true",
        help="train each arm before evaluating it. Off by default: without it, "
             "compare scores the checkpoints already in the run directories and "
             "writes only the comparison table.",
    )
    compare.add_argument(
        "--no-train", action="store_true",
        help="force evaluation of existing checkpoints. This is already the "
             "default; the flag is kept so older scripts and notes keep "
             "working, and it overrides --train if both are given.",
    )

    benchmark = subparsers.add_parser("benchmark", help="profile latency, memory and size")
    _add_common(benchmark)
    benchmark.add_argument("--modalities", nargs="+", default=["off_axis", "gabor"])
    benchmark.add_argument("--checkpoint", default=None)

    export = subparsers.add_parser("export", help="export a checkpoint to ONNX")
    _add_common(export)
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--output", default=None)
    export.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    export.add_argument(
        "--allow-untrained", action="store_true",
        help="export even when no checkpoint is found (produces a random-weight graph)",
    )

    return parser


def _load(args) -> "object":
    overrides = parse_overrides(args.set)
    if getattr(args, "device", None) and args.device != "auto":
        overrides.setdefault("device", args.device)
    return load_config(args.config, overrides)


# ---------------------------------------------------------------------------
def command_prepare(args) -> int:
    from scripts.prepare_data import prepare

    cfg = _load(args)
    prepare(cfg, force_masks=args.force_masks)
    return 0


def command_train(args) -> int:
    from holoqpi.data import build_dataloaders
    from holoqpi.engine import Trainer
    from holoqpi.models import build_model
    from holoqpi.utils import run_directory

    cfg = _load(args)
    seed_everything(cfg.project.seed, cfg.project.deterministic)
    device = resolve_device(args.device)

    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, cfg.data.modality)
    add_file_logging(LOGGER, run_dir / "train.log")
    save_config(cfg, run_dir / "resolved_config.yaml")

    LOGGER.info("training %s on %s (device=%s)", cfg.experiment_name, cfg.data.modality, device)

    loaders = build_dataloaders(cfg, splits_to_build=("train", "val"))
    model = build_model(cfg)

    initial = getattr(args, "init_from", None)
    if initial:
        from holoqpi.engine.trainer import load_checkpoint

        LOGGER.info("initialising weights from %s", initial)
        load_checkpoint(model, initial, device, strict=False)
        # Recorded in the run directory so a result can always be traced back to
        # the checkpoint it started from.
        (run_dir / "initialised_from.txt").write_text(str(Path(initial).resolve()) + "\n")

    Trainer(model, cfg, loaders, device, run_dir).train()
    return 0


def command_evaluate(args) -> int:
    from holoqpi.data import build_dataloaders
    from holoqpi.engine import Evaluator, apply_learned_physics, load_checkpoint, save_per_cell
    from holoqpi.models import build_model
    from holoqpi.utils import run_directory, write_json

    cfg = _load(args)
    seed_everything(cfg.project.seed, cfg.project.deterministic)
    device = resolve_device(args.device)

    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, cfg.data.modality)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"

    # Before anything reads cfg.loss.forward_model: a learned propagation
    # distance lives in the checkpoint, not in the config, and the forward-model
    # metric has to be built at the value this run actually trained to.
    cfg = apply_learned_physics(cfg, checkpoint)

    model = build_model(cfg)
    if checkpoint.is_file():
        load_checkpoint(model, checkpoint, device)
    elif args.allow_untrained:
        LOGGER.warning("checkpoint %s not found; scoring UNTRAINED weights as requested",
                       checkpoint)
    else:
        # Metrics from random weights look like metrics. Refuse rather than
        # write a plausible-looking metrics_test.json nobody would question.
        available = sorted(
            str(q.parent) for q in Path(cfg.paths.output_root).glob("*/best_model.pt")
        )
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            f"Checkpoints found under {cfg.paths.output_root}: "
            f"{available if available else 'none'}\n"
            f"Pass --allow-untrained only if scoring random weights is genuinely intended."
        )
    model.to(device)

    loaders = build_dataloaders(cfg, splits_to_build=(args.split,))
    evaluation = Evaluator(cfg, device).run(
        model, loaders[args.split], collect_per_cell=cfg.evaluation.save_per_cell_csv
    )

    # A tag keeps a second evaluation of the SAME checkpoint from overwriting the
    # first. That is not hypothetical: evaluating against the membrane labels
    # (paths.manual_mask_dir=membrane_mask) reuses the run directory, and without
    # a tag it silently replaces the Otsu-label metrics with the membrane-label
    # ones under an identical filename. Two label sources are two results.
    tag = f"_{args.tag}" if args.tag else ""
    write_json(evaluation["metrics"], run_dir / f"metrics_{args.split}{tag}.json")
    write_json(evaluation["confusion_matrix"], run_dir / f"confusion_{args.split}{tag}.json")
    if cfg.evaluation.save_per_cell_csv:
        save_per_cell(evaluation["per_cell"], run_dir / f"per_cell_{args.split}{tag}.csv")
        # The unmatched rows -- missed reference cells and false positives -- are
        # the input to figure 15 (recall against cell size). This line used to
        # sit inside the `if args.tag` block below while its filename carried no
        # tag, so an untagged evaluation wrote no file at all and a tagged one
        # wrote the membrane result under the Otsu result's name. It belongs
        # here, beside per_cell, and it carries the same tag.
        save_per_cell(evaluation["unmatched"], run_dir / f"unmatched_{args.split}{tag}.csv")
    if args.tag:
        LOGGER.info("labels from %s -> metrics_%s%s.json",
                    cfg.paths.manual_mask_dir or cfg.paths.mask_dir, args.split, tag)

    for key, value in sorted(evaluation["metrics"].items()):
        LOGGER.info("  %-34s %s", key, f"{value:.4f}" if isinstance(value, float) else value)
    return 0


def command_compare(args) -> int:
    from holoqpi.engine import compare_modalities

    cfg = _load(args)
    train = bool(args.train) and not bool(args.no_train)
    if not train:
        LOGGER.info(
            "compare: scoring existing checkpoints (pass --train to train each arm first)"
        )
    compare_modalities(cfg, args.modalities, train=train,
                       init_from=getattr(args, 'init_from', None))
    return 0


def command_benchmark(args) -> int:
    from holoqpi.deploy import profile_model
    from holoqpi.engine import load_checkpoint
    from holoqpi.models import build_model
    from holoqpi.utils import run_directory, write_csv, write_json

    cfg = _load(args)
    device = resolve_device(args.device)
    output_root = Path(cfg.paths.output_root)
    onnx_dir = output_root / "onnx"

    rows: list[dict] = []
    for modality in args.modalities:
        modality_cfg = cfg.merged({"data": {"modality": modality}})
        model = build_model(modality_cfg)

        run_dir = run_directory(
            modality_cfg.paths.output_root, modality_cfg.experiment_name, modality
        )
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
        if checkpoint.is_file():
            load_checkpoint(model, checkpoint, device, strict=False)
        else:
            LOGGER.warning("no checkpoint for %s; profiling the untrained graph", modality)

        rows.extend(profile_model(model, modality_cfg, device, modality, onnx_dir=onnx_dir))

    destination = output_root / f"{cfg.experiment_name}_hardware_benchmark_{device.type}.csv"
    write_csv(rows, destination)
    write_json(rows, destination.with_suffix(".json"))
    LOGGER.info("benchmark written to %s", destination)
    return 0


def command_export(args) -> int:
    from holoqpi.deploy import export_onnx, verify_onnx
    from holoqpi.engine import load_checkpoint
    from holoqpi.models import build_model
    from holoqpi.utils import run_directory, run_name

    cfg = _load(args)
    device = resolve_device(args.device)

    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, cfg.data.modality)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"

    model = build_model(cfg)
    if checkpoint.is_file():
        load_checkpoint(model, checkpoint, device)
    elif args.allow_untrained:
        LOGGER.warning("checkpoint %s not found; exporting UNTRAINED weights as requested",
                       checkpoint)
    else:
        # Silently exporting random weights produces a graph that traces, verifies
        # and benchmarks exactly like a trained one. Refuse, and say where the
        # checkpoints actually are: the run directory is named after the config,
        # so exporting with off_axis.yaml looks in runs/off_axis while a compare
        # run driven by base.yaml wrote runs/base_off_axis.
        available = sorted(
            str(p.parent) for p in Path(cfg.paths.output_root).glob("*/best_model.pt")
        )
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            f"Checkpoints found under {cfg.paths.output_root}: "
            f"{available if available else 'none'}\n"
            f"Export from the run that trained the model, e.g.\n"
            f"    python main.py export --config config/base.yaml "
            f"--set data.modality={cfg.data.modality} --precision {args.precision}\n"
            f"or pass --allow-untrained if a random-weight graph is genuinely what you want."
        )

    destination = Path(args.output) if args.output else (
        run_dir / f"{run_name(cfg.experiment_name, cfg.data.modality)}_{args.precision}.onnx"
    )
    export_onnx(
        model, cfg, destination,
        input_size=cfg.deploy.benchmark.input_size,
        device=device, precision=args.precision,
    )
    verify_onnx(destination)
    return 0


_COMMANDS = {
    "prepare": command_prepare,
    "train": command_train,
    "evaluate": command_evaluate,
    "compare": command_compare,
    "benchmark": command_benchmark,
    "export": command_export,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
