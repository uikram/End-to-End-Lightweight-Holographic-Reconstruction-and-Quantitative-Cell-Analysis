"""Seeding, device selection, logging and small run-management helpers."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_logging(logger: logging.Logger, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(handler)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype_from_name(name: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported amp dtype {name!r}; choose from {sorted(mapping)}")
    return mapping[name]


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


def run_name(experiment_name: str, modality: str) -> str:
    """Identifier for one arm of the comparison.

    off_axis.yaml is already named after its modality, so avoid "off_axis_off_axis".
    """
    if experiment_name.endswith(modality):
        return experiment_name
    return f"{experiment_name}_{modality}"


def run_directory(output_root: str | Path, experiment_name: str, modality: str) -> Path:
    path = Path(output_root) / run_name(experiment_name, modality)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Any, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list, destination: str | Path, fieldnames: list | None = None) -> Path | None:
    """Write dict rows to CSV, taking the union of their keys as the header.

    Rows can legitimately differ in shape: an ONNX benchmark row carries a
    provider column a PyTorch row has no equivalent of. Deriving the header from
    the first row alone would raise partway through and leave a truncated file.
    """
    import csv

    destination = Path(destination)
    if not rows:
        return None

    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def format_metrics(metrics: Mapping[str, Any], precision: int = 4) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        elif isinstance(value, (int, str)):
            parts.append(f"{key}={value}")
    return "  ".join(parts)
