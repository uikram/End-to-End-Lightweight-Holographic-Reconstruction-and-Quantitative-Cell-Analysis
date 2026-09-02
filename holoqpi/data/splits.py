"""Deterministic, stratified train/validation/test partitioning.

The split is written once to disk and reused by every experiment, so that the
off-axis and Gabor arms of the comparison see exactly the same images.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from ..config import Config
from ..utils import get_logger, read_json, write_json
from .metadata import SampleMetadata

LOGGER = get_logger(__name__)

_SPLIT_NAMES = ("train", "val", "test")


def build_splits(samples: list[SampleMetadata], cfg: Config, seed: int) -> dict:
    """Partition stems, keeping every stratum proportionally represented."""
    split_cfg = cfg.data.split
    fractions = {name: float(split_cfg[name]) for name in _SPLIT_NAMES}

    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")

    keys = list(split_cfg.stratify_by)
    strata: dict[str, list[SampleMetadata]] = defaultdict(list)
    for sample in samples:
        strata["|".join(str(getattr(sample, key)) for key in keys)].append(sample)

    rng = random.Random(seed)
    assignment: dict[str, list[str]] = {name: [] for name in _SPLIT_NAMES}
    undersized: list[str] = []

    total_samples = len(samples)
    targets = {name: fractions[name] * total_samples for name in _SPLIT_NAMES}

    for stratum, members in sorted(strata.items()):
        if len(members) < split_cfg.min_per_stratum:
            undersized.append(f"{stratum} (n={len(members)})")

        ordered = sorted(members, key=lambda s: s.stem)
        rng.shuffle(ordered)
        count = len(ordered)

        if count >= len(_SPLIT_NAMES):
            n_train = int(round(count * fractions["train"]))
            n_val = int(round(count * fractions["val"]))
            # Guarantee at least one image per split whenever the stratum allows it.
            n_train = min(max(n_train, 1), count - 2)
            n_val = min(max(n_val, 1), count - n_train - 1)

            assignment["train"].extend(s.stem for s in ordered[:n_train])
            assignment["val"].extend(s.stem for s in ordered[n_train:n_train + n_val])
            assignment["test"].extend(s.stem for s in ordered[n_train + n_val:])
        else:
            # A stratum too small to divide is assigned whole, to whichever split
            # is furthest below its global target. Without this, many tiny strata
            # all round the same way and a split can end up empty.
            for sample in ordered:
                deficits = {
                    name: targets[name] - len(assignment[name]) for name in _SPLIT_NAMES
                }
                chosen = max(_SPLIT_NAMES, key=lambda name: deficits[name])
                assignment[chosen].append(sample.stem)

    _warn_on_empty(assignment, total_samples)

    if undersized:
        LOGGER.warning(
            "%d strata hold fewer than min_per_stratum=%d images: %s",
            len(undersized), split_cfg.min_per_stratum, undersized[:5],
        )

    for name in _SPLIT_NAMES:
        assignment[name] = sorted(assignment[name])

    _assert_disjoint(assignment)

    LOGGER.info(
        "splits: train=%d val=%d test=%d (stratified by %s, seed=%d)",
        len(assignment["train"]), len(assignment["val"]), len(assignment["test"]),
        "+".join(keys), seed,
    )
    return {"seed": seed, "stratify_by": keys, "splits": assignment}


def _warn_on_empty(assignment: dict, total_samples: int) -> None:
    empty = [name for name in _SPLIT_NAMES if not assignment[name]]
    if empty and total_samples >= len(_SPLIT_NAMES):
        LOGGER.warning(
            "splits %s are empty with %d samples available; the dataset has too few "
            "images per stratum to divide three ways",
            empty, total_samples,
        )


def _assert_disjoint(assignment: dict) -> None:
    seen: dict[str, str] = {}
    for name, stems in assignment.items():
        for stem in stems:
            if stem in seen:
                raise RuntimeError(f"{stem} appears in both {seen[stem]} and {name}")
            seen[stem] = name


def save_splits(payload: dict, path: str | Path) -> Path:
    return write_json(payload, path)


def load_splits(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}. Run `python main.py prepare --config <cfg>` first."
        )
    payload = read_json(path)
    return payload["splits"]
