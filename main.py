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

    evaluate = subparsers.add_parser("evaluate", help="evaluate a trained checkpoint")
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", default=None, help="defaults to the run's best_model.pt")
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument(
        "--allow-untrained", action="store_true",
        help="evaluate even when no checkpoint is found (scores random weights)",
    )

    compare = subparsers.add_parser("compare", help="run the off-axis / Gabor comparison")
    _add_common(compare)
    compare.add_argument("--modalities", nargs="+", default=["off_axis", "gabor"])
    compare.add_argument(
        "--no-train", action="store_true",
        help="evaluate existing checkpoints instead of training first",
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
    Trainer(model, cfg, loaders, device, run_dir).train()
    return 0


def command_evaluate(args) -> int:
    from holoqpi.data import build_dataloaders
    from holoqpi.engine import Evaluator, load_checkpoint, save_per_cell
    from holoqpi.models import build_model
    from holoqpi.utils import run_directory, write_json

    cfg = _load(args)
    seed_everything(cfg.project.seed, cfg.project.deterministic)
    device = resolve_device(args.device)

    run_dir = run_directory(cfg.paths.output_root, cfg.experiment_name, cfg.data.modality)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"

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

    write_json(evaluation["metrics"], run_dir / f"metrics_{args.split}.json")
    write_json(evaluation["confusion_matrix"], run_dir / f"confusion_{args.split}.json")
    if cfg.evaluation.save_per_cell_csv:
        save_per_cell(evaluation["per_cell"], run_dir / f"per_cell_{args.split}.csv")
        save_per_cell(evaluation["unmatched"], run_dir / f"unmatched_{args.split}.csv")

    for key, value in sorted(evaluation["metrics"].items()):
        LOGGER.info("  %-34s %s", key, f"{value:.4f}" if isinstance(value, float) else value)
    return 0


def command_compare(args) -> int:
    from holoqpi.engine import compare_modalities

    cfg = _load(args)
    compare_modalities(cfg, args.modalities, train=not args.no_train)
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
