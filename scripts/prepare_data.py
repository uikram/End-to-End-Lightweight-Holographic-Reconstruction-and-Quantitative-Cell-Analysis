"""One-off dataset preparation: manifest, phase-derived masks, splits.

Run this once after the data lands, and again whenever the mask-generation
parameters or the split fractions change:

    python main.py prepare --config config/base.yaml
"""

from __future__ import annotations

from pathlib import Path

from holoqpi.analysis.cells import calibration_from_config
from holoqpi.config import Config
from holoqpi.data import build_splits, discover_samples, save_splits, summarise
from holoqpi.data.io import phase_path, read_phase_bin
from holoqpi.data.masks import generate_masks
from holoqpi.utils import get_logger, write_csv, write_json

LOGGER = get_logger(__name__)


def check_header_geometry(data_root: Path, cfg: Config, stems: list[str], limit: int = 8) -> dict:
    """Compare the configured pixel pitch against what the phase headers declare.

    The pitch determines every area, volume and mass reported downstream, so a
    silent disagreement between the YAML and the acquisition would corrupt the
    whole measurement chain. This surfaces it before training starts.
    """
    optics = cfg.optics
    tolerance = optics.header_pitch_tolerance_um
    observed: list[tuple[float, float]] = []
    sizes: set[tuple[int, int]] = set()

    for stem in stems[:limit]:
        record = read_phase_bin(phase_path(data_root, cfg, stem), cfg.formats.phase_binary)
        sizes.add((record.height, record.width))
        if record.pitch_x_um is not None:
            observed.append((record.pitch_x_um, record.pitch_y_um))

    report = {
        "inspected": min(limit, len(stems)),
        "phase_shapes": sorted(sizes),
        "configured_pitch_um": [optics.pixel_pitch_x_um, optics.pixel_pitch_y_um],
    }

    if len(sizes) > 1:
        LOGGER.warning("phase files disagree on image size: %s", sorted(sizes))
    expected = (cfg.data.phase_size, cfg.data.phase_size)
    if sizes and expected not in sizes:
        LOGGER.warning(
            "data.phase_size=%d does not match the headers %s; update the config",
            cfg.data.phase_size, sorted(sizes),
        )

    if observed:
        mean_x = sum(p[0] for p in observed) / len(observed)
        mean_y = sum(p[1] for p in observed) / len(observed)
        report["header_pitch_um"] = [mean_x, mean_y]

        drift_x = abs(mean_x - optics.pixel_pitch_x_um)
        drift_y = abs(mean_y - optics.pixel_pitch_y_um)
        report["pitch_drift_um"] = [drift_x, drift_y]

        if optics.warn_on_header_pitch_mismatch and max(drift_x, drift_y) > tolerance:
            LOGGER.warning(
                "pixel pitch in the headers (%.6f, %.6f um) differs from optics.pixel_pitch_* "
                "(%.6f, %.6f um). Every area, optical volume and dry mass scales with these "
                "values: confirm which is correct before publishing numbers.",
                mean_x, mean_y, optics.pixel_pitch_x_um, optics.pixel_pitch_y_um,
            )
        if optics.trust_header_pitch:
            LOGGER.info("optics.trust_header_pitch is set; header values will be preferred")
    else:
        LOGGER.info("phase headers carry no pixel pitch; using the configured values")

    return report


def prepare(cfg: Config, force_masks: bool = False) -> dict:
    """Discover samples, write the manifest, generate masks and build splits."""
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"data_root {data_root} does not exist. Set paths.data_root in the config."
        )

    if force_masks:
        cfg = cfg.merged({"mask_generation": {"overwrite_existing": True}})

    LOGGER.info("scanning %s", data_root.resolve())
    samples = discover_samples(data_root, cfg)
    stems = [sample.stem for sample in samples]

    distribution = summarise(samples)
    LOGGER.info("cell lines: %s", distribution["by_cell_line"])
    LOGGER.info("conditions: %s", distribution["by_condition"])

    geometry = check_header_geometry(data_root, cfg, stems)

    calibration = calibration_from_config(cfg)
    LOGGER.info(
        "calibration: dx=%.6f dy=%.6f um -> %.6f um^2/px; 1 rad*px = %.6f pg",
        calibration.pitch_x_um, calibration.pitch_y_um,
        calibration.pixel_area_um2, calibration.picogram_per_radian_pixel,
    )

    mask_stats = generate_masks(data_root, cfg, stems, calibration.pixel_area_um2)

    split_payload = build_splits(samples, cfg, cfg.project.seed)
    save_splits(split_payload, data_root / cfg.paths.splits_file)

    manifest_rows = []
    assignment = {
        stem: name for name, group in split_payload["splits"].items() for stem in group
    }
    for sample in samples:
        row = sample.as_row()
        row["split"] = assignment.get(sample.stem, "unassigned")
        manifest_rows.append(row)
    write_csv(manifest_rows, data_root / cfg.paths.manifest_file)

    report = {
        "data_root": str(data_root.resolve()),
        "distribution": distribution,
        "geometry": geometry,
        "calibration": calibration.as_dict(),
        "masks": mask_stats,
        "splits": {name: len(group) for name, group in split_payload["splits"].items()},
        "mask_source": (
            "manual" if cfg.paths.manual_mask_dir else "phase-derived (silver standard)"
        ),
    }
    write_json(report, data_root / "prepare_report.json")

    LOGGER.info("preparation complete: %s", report["splits"])
    LOGGER.info("manifest -> %s", data_root / cfg.paths.manifest_file)
    return report
