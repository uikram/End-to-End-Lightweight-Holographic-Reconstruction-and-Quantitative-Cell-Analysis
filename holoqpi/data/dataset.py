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
from .masks import split_instances
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

        # The surface the acquiring group subtracted from the delivered phase.
        # The forward-model term has to add it back before propagating, because
        # the recorded hologram still contains it.
        self.aberration = (
            data_io.load_aberration(self.data_root / cfg.paths.aberration_file)
            if cfg.optics.aberration.enabled else None
        )
        # WHICH surface each field gets.
        #
        # per_field looks up the field's own fitted surface. That surface was
        # obtained from the difference between the field's classical
        # reconstruction and its DELIVERED REFERENCE PHASE, i.e. the network's
        # target, so it could not be reproduced at inference on a new hologram --
        # and it makes the forward-model discrimination test partly circular,
        # because the surface was fitted assuming the reference phase is right.
        #
        # global looks up a single instrument-wide surface, fitted from the
        # TRAIN split only and applied to every field. That is what an optical
        # aberration is, and it is obtainable from a one-off calibration.
        #
        # hybrid is the global curvature plus a per-field ramp predicted from
        # the hologram's carrier. It is equally deployable, but on this data it
        # was MEASURED not to help: the carrier explains 10.9% of the fitted
        # tilt. It is selectable so that result can be reproduced, not because
        # it is recommended. See scripts/estimate_aberration.py.
        self.aberration_mode = cfg.optics.aberration.mode
        if self.aberration_mode not in ("global", "per_field", "hybrid"):
            raise ValueError(
                f"optics.aberration.mode must be global, per_field or hybrid, "
                f"got {self.aberration_mode!r}"
            )
        self.aberration_key = {"global": "__global__"}.get(self.aberration_mode)
        probe = (
            self.aberration_key if self.aberration_key is not None
            else f"hybrid::{stems[0]}" if self.aberration_mode == "hybrid" and stems
            else None
        )
        if (
            cfg.optics.aberration.enabled
            and probe is not None
            and self.aberration is not None
            and probe not in self.aberration.get("coefficients", {})
        ):
            LOGGER.warning(
                "optics.aberration.mode is %r but %s carries no %r surface. Every field "
                "will fall back to the pure-phase assumption and the forward-model term "
                "will be skipped. Re-run "
                "`python scripts/estimate_aberration.py --config <cfg>` to write it.",
                self.aberration_mode, self.data_root / cfg.paths.aberration_file, probe,
            )
        if cfg.optics.aberration.enabled and self.aberration is None:
            LOGGER.warning(
                "optics.aberration.enabled is true but %s is missing. The forward-model "
                "term will treat the delivered phase as a raw reconstruction, which it "
                "is not, and no propagation distance will fit. Run "
                "`python scripts/estimate_aberration.py --config <cfg>` first.",
                self.data_root / cfg.paths.aberration_file,
            )

        self.provide_instances = bool(data_cfg.provide_instances)
        # Classical-reconstruction amplitude, a REFERENCE and not a measurement.
        self.provide_amplitude = bool(data_cfg.provide_amplitude)
        self.amplitude_dir = cfg.paths.amplitude_dir
        self.amplitude_suffix = cfg.formats.amplitude.suffix
        self.instance_method = cfg.evaluation.segmentation.instance_from
        self.watershed_min_distance = cfg.mask_generation.watershed_min_distance_px

        crop = data_cfg.train_crop if split == "train" else data_cfg.eval_size
        self.crop_size = crop
        self._base_seed = cfg.project.seed + len(stems)
        self.augment = (
            GeometricAugmentation(data_cfg.augmentation, seed=self._base_seed) if augment else None
        )
        self._rng = random.Random(self._base_seed)

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

    def _load_arrays(self, index: int) -> tuple[np.ndarray, ...]:
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
        # Two representations of the same measurement, kept separately on
        # purpose. The network needs a normalised input; the forward-model loss
        # needs the intensity the sensor actually recorded. Normalising once and
        # using the result for both would silently redefine the observation
        # model the physics term is supposed to test.
        hologram_raw = hologram.copy()
        hologram = data_io.normalise_hologram(hologram, self.normalisation)

        mask = data_io.read_mask(data_io.mask_path(self.data_root, self.cfg, stem))

        # Reference instance labelling, computed once per field. The per-cell
        # integrated-phase term integrates over these regions, so they must be
        # the SAME regions throughout training -- deriving them from the
        # prediction would let the term be satisfied by moving boundaries rather
        # than by correcting the measurement.
        #
        # Connected components rather than watershed: watershed on a 900 px
        # field costs more than a forward pass and would be repeated every
        # epoch. Two touching cells are then integrated as one region, which
        # makes the constraint coarser but never wrong -- a merged region is
        # still a legitimate domain over which the integral must be preserved.
        # Evaluation still uses watershed, so the reported per-cell metrics are
        # unaffected by this choice.
        if self.provide_instances:
            instances = split_instances(
                (mask > 0).astype(np.uint8), self.instance_method,
                self.watershed_min_distance,
            ).astype(np.int32)
        else:
            instances = np.zeros_like(mask, dtype=np.int32)
        surface_key = (
            self.aberration_key
            or (f"hybrid::{stem}" if self.aberration_mode == "hybrid" else stem)
        )
        surface, surface_valid = data_io.render_aberration(
            self.aberration, surface_key, phase.shape[0], phase.shape[1]
        )

        if self.provide_amplitude:
            amplitude = self._read_amplitude(stem, phase.shape)
        else:
            # Unit transmittance: the thin-phase-object assumption. Carried as an
            # array anyway so the sample dict has a stable shape.
            amplitude = np.ones_like(phase, dtype=np.float32)

        if phase.shape != hologram.shape or mask.shape != phase.shape:
            raise ValueError(
                f"{stem}: shape mismatch after alignment "
                f"(hologram {hologram.shape}, phase {phase.shape}, mask {mask.shape})"
            )

        arrays = (hologram.astype(np.float32), phase.astype(np.float32),
                  mask.astype(np.int64), hologram_raw.astype(np.float32),
                  surface.astype(np.float32), instances,
                  amplitude.astype(np.float32), bool(surface_valid))
        if self._cache is not None:
            self._cache[index] = arrays
        return arrays

    def _read_amplitude(self, stem: str, shape: tuple[int, int]) -> np.ndarray:
        """Load the precomputed classical-reconstruction amplitude for one field."""
        path = self.data_root / self.amplitude_dir / f"{stem}{self.amplitude_suffix}"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing but data.provide_amplitude is true. Run "
                "`python scripts/prepare_amplitude.py --config <cfg>` first."
            )
        amplitude = np.load(path).astype(np.float32)
        if amplitude.shape != tuple(shape):
            raise ValueError(
                f"{stem}: amplitude reference is {amplitude.shape} but the phase is "
                f"{tuple(shape)}. Regenerate the amplitude with the same phase_size."
            )
        return amplitude

    def __getitem__(self, index: int) -> dict:
        meta = self.samples[index]
        (hologram, phase, mask, hologram_raw, surface, instances, amplitude,
         surface_valid) = self._load_arrays(index)

        # The aberration surface travels with the other arrays through cropping
        # and augmentation. Rendering it on the full field and then cropping it
        # keeps it registered to the hologram; rendering it on the crop's own
        # coordinates would silently re-centre the bowl on every crop.
        stack = [hologram, phase, mask, hologram_raw, surface, instances, amplitude]
        if self.crop_size:
            stack = (random_crop(stack, self.crop_size, self._rng)
                     if self.split == "train"
                     else center_crop(stack, self.crop_size))
        if self.augment is not None:
            stack = self.augment(*stack)
        hologram, phase, mask, hologram_raw, surface, instances, amplitude = stack
        # Cropping can leave gaps in the label sequence; the loss scatters into
        # one bin per label value, so they must stay dense and start at 1.
        instances = _relabel_dense(instances)

        return {
            "hologram": torch.from_numpy(np.ascontiguousarray(hologram)).unsqueeze(0).float(),
            "hologram_raw": torch.from_numpy(
                np.ascontiguousarray(hologram_raw)
            ).unsqueeze(0).float(),
            "aberration": torch.from_numpy(
                np.ascontiguousarray(surface)
            ).unsqueeze(0).float(),
            # False where the surface could not be recovered, so the
            # forward-model term can skip the field instead of being handed a
            # phase error larger than the signal.
            "aberration_valid": torch.tensor(bool(surface_valid)),
            "phase": torch.from_numpy(np.ascontiguousarray(phase)).unsqueeze(0).float(),
            # Named 'amplitude' but it is a classical reconstruction, not a
            # measurement. See scripts/prepare_amplitude.py.
            "amplitude": torch.from_numpy(
                np.ascontiguousarray(amplitude)
            ).unsqueeze(0).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "instances": torch.from_numpy(np.ascontiguousarray(instances)).long(),
            "condition": torch.tensor(meta.condition_id, dtype=torch.long),
            "cell_line": torch.tensor(meta.cell_line_id, dtype=torch.long),
            "stem": meta.stem,
        }


def _relabel_dense(instances: np.ndarray) -> np.ndarray:
    """Renumber instance labels to 1..N with no gaps.

    A random crop can remove whole cells and leave the surviving labels sparse
    (say 3, 7, 12). The loss allocates one accumulator bin per label value, so a
    sparse set would allocate bins for cells that are no longer present and
    average the per-cell error over the wrong denominator.
    """
    present = np.unique(instances)
    present = present[present != 0]
    if present.size == 0:
        return np.zeros_like(instances, dtype=np.int32)
    lookup = np.zeros(int(instances.max()) + 1, dtype=np.int32)
    lookup[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return lookup[instances]


def _seed_worker(worker_id: int) -> None:
    """Give each dataloader worker its own random stream.

    A ``DataLoader`` with ``num_workers=N`` forks N copies of the dataset, and
    each copy carries an identical ``random.Random`` instance. Every worker then
    replays the same sequence of crop offsets and the same flip/rotation
    decisions, so the augmentation an epoch actually sees has only 1/N of the
    intended diversity. Re-seeding per worker from torch's own per-worker seed
    keeps the run reproducible while making the streams independent.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    base = int(torch.initial_seed() % (2 ** 31))
    if hasattr(dataset, "_rng"):
        dataset._rng = random.Random(base)
    if getattr(dataset, "augment", None) is not None:
        dataset.augment._rng = random.Random(base + 1)


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
            worker_init_fn=_seed_worker if data_cfg.num_workers > 0 else None,
        )
    return loaders
