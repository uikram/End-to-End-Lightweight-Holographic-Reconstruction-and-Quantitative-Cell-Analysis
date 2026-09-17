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


def pooled_between_seed_sd(values_a, values_b) -> tuple[float, str | None]:
    """The project's significance yardstick, in one place.

    Returns ``(pooled_sd, reason_it_cannot_be_used)``, where the reason is
    ``None`` when the value is usable.

    ``sqrt(((n_a - 1) s_a^2 + (n_b - 1) s_b^2) / (n_a + n_b - 2))`` -- the
    textbook pooled standard deviation of two seed populations. A difference
    counts as resolved when it exceeds
    ``evaluation.seed_replication.resolve_factor`` times this.

    WHY IT LIVES HERE. The same rule was implemented three times with two
    different formulas and one hardcoded constant:

        scripts/aggregate_seeds.py    sqrt(s_a^2 + s_b^2)
        scripts/collect_results.py    sqrt(0.5 (v_a + v_b))
        scripts/make_figures.py       sqrt(0.5 (v_a + v_b)) for the band,
                                      and a flat +/-2% in figure 7's shading

    The first is larger than the second by sqrt(2) for equal n, so the SAME
    comparison could be called resolved by one document and unresolved by
    another -- and the flat 2% had no relation to any measurement at all.

    Two cases are refused rather than answered:

    * fewer than two runs on either arm, because a spread of one run is not a
      spread;
    * a pooled SD of exactly zero, which means the metric did not move between
      seeds at all. Calling a large difference "within noise" when the noise is
      measurably nil inverts the verdict, and zero-variance metrics have already
      appeared in this project's results.
    """
    a = np.asarray([v for v in np.asarray(values_a, dtype=float).ravel()
                    if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in np.asarray(values_b, dtype=float).ravel()
                    if np.isfinite(v)], dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan"), f"needs 2+ runs per arm (have {a.size} and {b.size})"
    variance = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (
        a.size + b.size - 2
    )
    sd = float(np.sqrt(variance))
    if not np.isfinite(sd):
        return float("nan"), "spread is not finite"
    if sd == 0.0:
        return 0.0, "between-seed spread is exactly zero -- check the runs differ"
    return sd, None


_PROVENANCE_NAME = "_provenance.json"


def provenance_path(directory: str | Path) -> Path:
    return Path(directory) / _PROVENANCE_NAME


def read_provenance(directory: str | Path) -> dict | None:
    """The parameters a generated directory was produced with, if recorded."""
    path = provenance_path(directory)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def write_provenance(directory: str | Path, parameters: Mapping[str, Any]) -> Path:
    """Record the parameters a generated directory was produced with."""
    return write_json(dict(parameters), provenance_path(directory))


def assert_provenance(
    directory: str | Path, parameters: Mapping[str, Any], regenerate_hint: str
) -> None:
    """Refuse to reuse a generated directory built with different parameters.

    THE FAILURE THIS PREVENTS, which has already happened in this project.

    Both generators that fill a directory -- the phase-derived masks and the
    classical amplitude reference -- skip any file that already exists. Neither
    recorded what produced it, and the only consumer-side check was the array's
    SHAPE, which a parameter change does not alter. So changing
    ``mask_generation.smoothing_sigma_px``, ``otsu_scale``,
    ``border_buffer_px`` or ``model.frontend.sideband_radius_px`` and re-running
    silently reused every old file, and the run then trained and measured
    against artefacts from a different configuration. For the masks that is
    especially bad: they are simultaneously the segmentation target and the
    reference for every per-cell measurement.

    Raising is the right response rather than warning. A warning in a stage log
    is exactly what was missed before, and regenerating is one flag away.
    """
    stored = read_provenance(directory)
    if stored is None:
        return
    current = dict(parameters)
    differences = {
        key: (stored.get(key), current[key])
        for key in sorted(set(stored) | set(current))
        if stored.get(key) != current.get(key)
    }
    if not differences:
        return
    lines = "\n".join(
        f"    {key}: produced with {was!r}, config now says {now!r}"
        for key, (was, now) in differences.items()
    )
    raise ValueError(
        f"{directory} was generated with different parameters and would be "
        f"reused as-is:\n{lines}\n"
        f"  Existing files are skipped rather than rebuilt, so the run would "
        f"train and measure against artefacts from another configuration.\n"
        f"  {regenerate_hint}"
    )


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
