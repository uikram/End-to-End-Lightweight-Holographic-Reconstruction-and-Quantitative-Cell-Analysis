"""Sample discovery and label parsing.

Stems follow ``<cell line>[_<drug>[_<concentration>]]_<index>``, for example::

    NCI_01                      NCI, control
    NCI_Blebbistatin_5uM_07     NCI, blebbistatin
    SNU_staurosporine_23        SNU, staurosporine     (no concentration token)
    T24_rotenone_500nM_50       T24, rotenone

Capitalisation and the presence of a concentration token are inconsistent across
cell lines in the delivered data, so matching is case-insensitive and treats the
concentration as optional. Every alias is declared in ``labels`` in the YAML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from ..config import Config
from ..utils import get_logger

LOGGER = get_logger(__name__)

_TRAILING_INDEX = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class SampleMetadata:
    stem: str
    cell_line: str
    condition: str
    index: int
    cell_line_id: int
    condition_id: int

    @property
    def group(self) -> str:
        """Stratification key: one cell line under one drug condition."""
        return f"{self.cell_line}|{self.condition}"

    def as_row(self) -> dict:
        row = asdict(self)
        row["group"] = self.group
        return row


class LabelSchema:
    """Maps stems onto cell-line and drug-condition labels."""

    def __init__(self, cfg: Config):
        labels = cfg.labels
        self.cell_lines: list[str] = list(labels.cell_lines)
        self.conditions: list[str] = list(labels.conditions)
        self.segmentation_classes: list[str] = list(labels.segmentation_classes)

        self._cell_line_ids = {name: i for i, name in enumerate(self.cell_lines)}
        self._condition_ids = {name: i for i, name in enumerate(self.conditions)}

        # alias token -> canonical condition name
        self._alias_lookup: dict[str, str] = {}
        for canonical, aliases in labels.condition_aliases.items():
            if canonical not in self._condition_ids:
                raise ValueError(f"condition alias group {canonical!r} is not in labels.conditions")
            for alias in aliases:
                if alias:
                    self._alias_lookup[alias.lower()] = canonical
            self._alias_lookup[canonical.lower()] = canonical

        self._default_condition = self._resolve_default(labels)

    def _resolve_default(self, labels: Config) -> str:
        """The condition assigned when a stem carries no drug token."""
        for canonical, aliases in labels.condition_aliases.items():
            if any(alias == "" for alias in aliases):
                return canonical
        return self.conditions[0]

    def parse(self, stem: str) -> SampleMetadata:
        match = _TRAILING_INDEX.search(stem)
        if match is None:
            raise ValueError(f"stem {stem!r} does not end in an image index")
        index = int(match.group(1))
        body = stem[: match.start()]

        tokens = body.split("_")
        line_token = tokens[0]
        cell_line = self._match_cell_line(line_token)
        if cell_line is None:
            raise ValueError(f"stem {stem!r} does not begin with a known cell line")

        condition = self._default_condition
        for token in tokens[1:]:
            candidate = self._alias_lookup.get(token.lower())
            if candidate is not None:
                condition = candidate
                break

        return SampleMetadata(
            stem=stem,
            cell_line=cell_line,
            condition=condition,
            index=index,
            cell_line_id=self._cell_line_ids[cell_line],
            condition_id=self._condition_ids[condition],
        )

    def _match_cell_line(self, token: str) -> str | None:
        for name in self.cell_lines:
            if token.lower() == name.lower():
                return name
        return None


def discover_samples(data_root: str | Path, cfg: Config) -> list[SampleMetadata]:
    """List stems for which the phase map and every hologram modality exist."""
    data_root = Path(data_root)
    schema = LabelSchema(cfg)

    phase_dir = data_root / cfg.paths.phase_dir
    phase_suffix = cfg.formats.phase_binary.suffix
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"phase directory not found: {phase_dir}")

    modalities = list(cfg.paths.hologram_dirs.keys())
    samples: list[SampleMetadata] = []
    skipped: dict[str, list[str]] = {}

    for phase_file in sorted(phase_dir.glob(f"*{phase_suffix}")):
        stem = phase_file.name[: -len(phase_suffix)]

        missing = [
            modality
            for modality in modalities
            if not (
                data_root
                / cfg.paths.hologram_dirs[modality]
                / f"{stem}{cfg.formats.hologram.suffixes[modality]}"
            ).is_file()
        ]
        if missing:
            skipped.setdefault(",".join(missing), []).append(stem)
            continue

        try:
            samples.append(schema.parse(stem))
        except ValueError as exc:
            skipped.setdefault("unparsable", []).append(f"{stem} ({exc})")

    for reason, stems in skipped.items():
        LOGGER.warning("skipped %d samples (%s), first: %s", len(stems), reason, stems[:3])

    if not samples:
        raise RuntimeError(
            f"no complete samples under {data_root}. Expected "
            f"{cfg.paths.phase_dir}/<stem>{phase_suffix} alongside a hologram in "
            f"each of {modalities}."
        )

    LOGGER.info("discovered %d complete samples across %d modalities", len(samples), len(modalities))
    return samples


def summarise(samples: list[SampleMetadata]) -> dict:
    """Counts per cell line, per condition and per stratification group."""
    from collections import Counter

    return {
        "total": len(samples),
        "by_cell_line": dict(Counter(s.cell_line for s in samples)),
        "by_condition": dict(Counter(s.condition for s in samples)),
        "by_group": dict(Counter(s.group for s in samples)),
    }
