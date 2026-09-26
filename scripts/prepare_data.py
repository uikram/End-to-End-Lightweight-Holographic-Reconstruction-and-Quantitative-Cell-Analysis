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
from holoqpi.data.io import phase_path, read_phase_header
from holoqpi.data.masks import generate_masks
from holoqpi.utils import get_logger, write_csv, write_json

LOGGER = get_logger(__name__)


def check_header_geometry(data_root: Path, cfg: Config, stems: list[str],
                          limit: int | None = None) -> dict:
    """Compare the configured pixel pitch against what the phase headers declare.

    The pitch determines every area, volume and mass reported downstream, so a
    silent disagreement between the YAML and the acquisition would corrupt the
    whole measurement chain. This surfaces it before training starts.

    EVERY file is inspected by default. It used to stop after eight and then
    phrase its warnings as statements about the dataset ("phase files disagree on
    image size"), so an odd file anywhere past the eighth passed `prepare` and
    reappeared much later as a shape error inside the dataloader. Reading only
    the 23-byte header and comparing the payload length against the file size
    makes the full pass cost nothing worth saving. ``limit`` is kept for a quick
    look on a slow filesystem.
    """
    optics = cfg.optics
    tolerance = optics.header_pitch_tolerance_um
    observed: list[tuple[float, float]] = []
    sizes: dict[tuple[int, int], list[str]] = {}
    unreadable: list[str] = []

    inspect = stems if limit is None else stems[:limit]
    for stem in inspect:
        try:
            record = read_phase_header(
                phase_path(data_root, cfg, stem), cfg.formats.phase_binary
            )
        except Exception as exc:
            unreadable.append(f"{stem} ({exc})")
            continue
        sizes.setdefault((record.height, record.width), []).append(stem)
        if record.pitch_x_um is not None:
            observed.append((record.pitch_x_um, record.pitch_y_um))

    report = {
        "inspected": len(inspect),
        "phase_shapes": sorted(sizes),
        "configured_pitch_um": [optics.pixel_pitch_x_um, optics.pixel_pitch_y_um],
        "unreadable": unreadable,
    }

    if unreadable:
        # A file whose declared geometry does not match its length is corrupt or
        # is in a different format, and it will fail inside the dataloader. Name
        # the files rather than letting the count speak for them.
        raise ValueError(
            f"{len(unreadable)} of {len(inspect)} phase files could not be read or "
            f"disagree with their own headers. First few: {unreadable[:5]}"
        )
    if len(sizes) > 1:
        # Named, with a few example stems per shape, because "the files disagree"
        # is not actionable on its own.
        detail = {shape: members[:3] for shape, members in sorted(sizes.items())}
        raise ValueError(
            f"phase files disagree on image size across {len(inspect)} files: "
            f"{detail}. The dataloader requires one geometry; fix or exclude the "
            f"odd files before training."
        )
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
            # REFUSED, RATHER THAN LOGGED AS IF IT HAPPENED.
            #
            # This used to print "header values will be preferred" and then do
            # nothing at all: `trust_header_pitch` appeared in exactly two places
            # in the codebase, both of them this branch, while
            # calibration_from_config unconditionally reads
            # optics.pixel_pitch_x_um / _y_um. Since the pitch multiplies every
            # projected area, optical volume and dry mass the project reports,
            # a flag that claims to change it and does not is the most dangerous
            # kind of no-op -- and on THIS dataset honouring the header would be
            # actively wrong, because the acquiring group stated that the
            # header's y value (0.211994 um) is the fluorescence camera's pitch
            # and must be disregarded.
            #
            # So it is an error rather than a silent lie. Implementing it would
            # mean threading a per-file pitch through the dataset and every
            # measurement, which nothing in this study needs.
            raise ValueError(
                "optics.trust_header_pitch is not implemented and must be false. "
                "The pixel pitch used for every measurement comes from "
                "optics.pixel_pitch_x_um / optics.pixel_pitch_y_um. On this "
                "dataset the header's y pitch is the FLUORESCENCE camera's "
                f"({mean_y:.6f} um) and the acquiring group confirmed it must be "
                "disregarded; see the pixel-pitch note in config/base.yaml. Set "
                "the pitch explicitly in the config instead."
            )
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
