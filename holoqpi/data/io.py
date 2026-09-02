"""Readers for the three on-disk representations used by this project.

* quantitative phase  -- headered float32 ``.bin`` produced by the DHM
  reconstruction software (ground truth for the phase head)
* holograms           -- LZW-compressed 8-bit TIFF, one directory per modality
* segmentation masks  -- 8-bit PNG written by ``prepare``

The layout of the phase header is described entirely by ``formats.phase_binary``
in the configuration, so a change of acquisition software is a YAML edit.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config

_BYTE_ORDER = {"little": "<", "big": ">"}
_DTYPES = {"float32": "f4", "float64": "f8", "uint16": "u2"}


@dataclass(frozen=True)
class PhaseRecord:
    """A quantitative phase map together with the geometry from its header."""

    phase: np.ndarray          # (H, W) float32, radians
    width: int
    height: int
    pitch_x_um: float | None   # None when the header carries no pitch
    pitch_y_um: float | None


def read_phase_bin(path: str | Path, fmt: Config) -> PhaseRecord:
    """Decode one headered float32 phase file."""
    path = Path(path)
    raw = path.read_bytes()

    header_bytes = fmt.header_bytes
    if len(raw) <= header_bytes:
        raise ValueError(f"{path.name}: file shorter than its {header_bytes}-byte header")

    order = _BYTE_ORDER[fmt.byte_order]
    header = raw[:header_bytes]

    width = struct.unpack_from(f"{order}I", header, fmt.width_offset)[0]
    height = struct.unpack_from(f"{order}I", header, fmt.height_offset)[0]

    dtype = np.dtype(order + _DTYPES[fmt.dtype])
    expected = width * height * dtype.itemsize
    payload = raw[header_bytes:]
    if len(payload) != expected:
        raise ValueError(
            f"{path.name}: header declares {width}x{height} ({expected} bytes) "
            f"but the payload holds {len(payload)} bytes"
        )

    phase = np.frombuffer(payload, dtype=dtype).reshape(height, width).astype(np.float32)

    pitch_x = pitch_y = None
    px_off, py_off = fmt.pitch_x_offset, fmt.pitch_y_offset
    if px_off is not None and py_off is not None and max(px_off, py_off) + 4 <= header_bytes:
        # The header stores the pitches in metres; the project works in microns.
        pitch_x = float(struct.unpack_from(f"{order}f", header, px_off)[0]) * 1e6
        pitch_y = float(struct.unpack_from(f"{order}f", header, py_off)[0]) * 1e6

    return PhaseRecord(phase=phase, width=width, height=height,
                       pitch_x_um=pitch_x, pitch_y_um=pitch_y)


def read_hologram(path: str | Path) -> np.ndarray:
    """Read one hologram TIFF as float32 in its native intensity units."""
    array = None
    try:
        import tifffile

        array = tifffile.imread(str(path))
    except Exception:
        from PIL import Image

        with Image.open(path) as handle:
            array = np.array(handle)

    if array.ndim == 3:
        array = array[..., 0] if array.shape[-1] <= 4 else array[0]
    return array.astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a segmentation mask as an int64 label array."""
    from PIL import Image

    return np.array(Image.open(path)).astype(np.int64)


def write_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Persist a segmentation mask as 8-bit PNG."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)
    return path


def align_to_phase_grid(image: np.ndarray, target: int, mode: str) -> np.ndarray:
    """Bring a hologram onto the phase sampling grid.

    ``center_crop`` keeps the native sampling and discards the border, which is
    the correct operation for this dataset: the reconstruction crops the camera
    frame rather than resampling it. ``resize`` is provided for datasets whose
    reconstruction does rescale the field.
    """
    height, width = image.shape[:2]
    if (height, width) == (target, target):
        return image

    if mode == "center_crop":
        if height < target or width < target:
            raise ValueError(
                f"cannot centre-crop {height}x{width} to {target}x{target}; "
                "the source is smaller than the target"
            )
        top = (height - target) // 2
        left = (width - target) // 2
        return image[top:top + target, left:left + target]

    if mode == "resize":
        from scipy.ndimage import zoom

        return zoom(image, (target / height, target / width), order=1).astype(image.dtype)

    raise ValueError(f"unknown alignment mode {mode!r}; use center_crop or resize")


def normalise_hologram(image: np.ndarray, method: str) -> np.ndarray:
    """Scale a hologram for network input without touching its fringe structure."""
    image = image.astype(np.float32)
    if method == "none":
        return image
    if method == "unit":
        low, high = float(image.min()), float(image.max())
        return (image - low) / (high - low) if high > low else np.zeros_like(image)
    if method == "zscore":
        std = float(image.std())
        return (image - float(image.mean())) / (std if std > 0 else 1.0)
    raise ValueError(f"unknown hologram normalisation {method!r}")


def phase_path(data_root: Path, cfg: Config, stem: str) -> Path:
    return data_root / cfg.paths.phase_dir / f"{stem}{cfg.formats.phase_binary.suffix}"


def hologram_path(data_root: Path, cfg: Config, stem: str, modality: str) -> Path:
    directory = cfg.paths.hologram_dirs[modality]
    suffix = cfg.formats.hologram.suffixes[modality]
    return data_root / directory / f"{stem}{suffix}"


def mask_path(data_root: Path, cfg: Config, stem: str) -> Path:
    directory = cfg.paths.manual_mask_dir or cfg.paths.mask_dir
    return data_root / directory / f"{stem}{cfg.formats.mask.suffix}"
