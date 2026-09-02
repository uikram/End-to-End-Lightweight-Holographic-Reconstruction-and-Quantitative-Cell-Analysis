"""Dataset and dataloader construction.

One sample couples a raw hologram of the selected modality with the two
supervision targets (quantitative phase, cell mask) and the image-level drug
condition. Selecting the modality is the only difference between the two arms of
the comparison, which keeps the experiment controlled.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..utils import get_logger
from . import io as data_io
from .augment import GeometricAugmentation, center_crop, random_crop
from .metadata import LabelSchema, SampleMetadata
from .splits import load_splits

LOGGER = get_logger(__name__)


class HologramQPIDataset(Dataset):
    """Raw hologram -> (quantitative phase, cell mask, condition label)."""

    def __init__(
        self,
        data_root: str | Path,
        cfg: Config,
        stems: list[str],
        split: str,
        augment: bool,
    ):
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.split = split
        self.modality = cfg.data.modality
        self.schema = LabelSchema(cfg)
        self.samples: list[SampleMetadata] = [self.schema.parse(stem) for stem in stems]

        data_cfg = cfg.data
        self.phase_size = data_cfg.phase_size
        self.align = data_cfg.align
        self.normalisation = data_cfg.hologram_normalisation
        self.phase_format = cfg.formats.phase_binary

        crop = data_cfg.train_crop if split == "train" else data_cfg.eval_size
        self.crop_size = crop
        self.augment = GeometricAugmentation(data_cfg.augmentation, seed=cfg.project.seed) if augment else None
        self._rng = random.Random(cfg.project.seed + len(stems))

        self._cache: dict[int, tuple] | None = {} if data_cfg.cache_in_memory else None

        self._verify_first_sample()
        LOGGER.info(
            "%s split: %d samples | modality=%s | crop=%s | augment=%s",
            split, len(self.samples), self.modality, crop, bool(augment),
        )

    def _verify_first_sample(self) -> None:
        if not self.samples:
            raise RuntimeError(f"{self.split} split is empty")
        stem = self.samples[0].stem
        for path in (
            data_io.phase_path(self.data_root, self.cfg, stem),
            data_io.hologram_path(self.data_root, self.cfg, stem, self.modality),
            data_io.mask_path(self.data_root, self.cfg, stem),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing. Run `python main.py prepare --config <cfg>` "
                    "to generate masks and splits before training."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_arrays(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is not None and index in self._cache:
            return self._cache[index]

        stem = self.samples[index].stem

        record = data_io.read_phase_bin(
            data_io.phase_path(self.data_root, self.cfg, stem), self.phase_format
        )
        phase = record.phase

        hologram = data_io.read_hologram(
            data_io.hologram_path(self.data_root, self.cfg, stem, self.modality)
        )
        hologram = data_io.align_to_phase_grid(hologram, self.phase_size, self.align)
        hologram = data_io.normalise_hologram(hologram, self.normalisation)

        mask = data_io.read_mask(data_io.mask_path(self.data_root, self.cfg, stem))

        if phase.shape != hologram.shape or mask.shape != phase.shape:
            raise ValueError(
                f"{stem}: shape mismatch after alignment "
                f"(hologram {hologram.shape}, phase {phase.shape}, mask {mask.shape})"
            )

        arrays = (hologram.astype(np.float32), phase.astype(np.float32), mask.astype(np.int64))
        if self._cache is not None:
            self._cache[index] = arrays
        return arrays

    def __getitem__(self, index: int) -> dict:
        meta = self.samples[index]
        hologram, phase, mask = self._load_arrays(index)

        if self.crop_size:
            if self.split == "train":
                hologram, phase, mask = random_crop([hologram, phase, mask], self.crop_size, self._rng)
            else:
                hologram, phase, mask = center_crop([hologram, phase, mask], self.crop_size)

        if self.augment is not None:
            hologram, phase, mask = self.augment(hologram, phase, mask)

        return {
            "hologram": torch.from_numpy(np.ascontiguousarray(hologram)).unsqueeze(0).float(),
            "phase": torch.from_numpy(np.ascontiguousarray(phase)).unsqueeze(0).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "condition": torch.tensor(meta.condition_id, dtype=torch.long),
            "cell_line": torch.tensor(meta.cell_line_id, dtype=torch.long),
            "stem": meta.stem,
        }


def build_dataloaders(
    cfg: Config, splits_to_build: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, DataLoader]:
    """Construct the requested dataloaders from the persisted split file."""
    data_root = Path(cfg.paths.data_root)
    splits = load_splits(data_root / cfg.paths.splits_file)
    data_cfg = cfg.data

    loaders: dict[str, DataLoader] = {}
    for split in splits_to_build:
        stems = splits.get(split, [])
        if not stems:
            LOGGER.warning("split %r is empty; skipping", split)
            continue

        is_train = split == "train"
        dataset = HologramQPIDataset(
            data_root=data_root,
            cfg=cfg,
            stems=stems,
            split=split,
            augment=is_train and data_cfg.augmentation.enabled,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=data_cfg.batch_size if is_train else data_cfg.eval_batch_size,
            shuffle=is_train,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory and torch.cuda.is_available(),
            drop_last=data_cfg.drop_last and is_train,
            persistent_workers=data_cfg.num_workers > 0,
        )
    return loaders
