"""Figure 1 panels - every image that goes into the overview figure, from real data.

WHAT IT MAKES
    One PNG per image slot in figure_1_overview.pptx, each named after the
    placeholder it fills (the placeholder box in the .pptx carries the same file
    name). Every map shows the same square window of the field (--view; the
    default avoids an illumination-edge artefact at the border of the reference
    phase) and is saved square, so it drops into the square placeholders
    without distortion. Colour bars are plain gradient strips; their tick values
    are editable text in the .pptx, taken from fig1_panels.json.

        p1_offaxis_hologram.png     raw off-axis hologram, cropped to the phase grid (data.crop_offset_px)
        p1_offaxis_zoom.png         40 x 40 px detail of the same hologram (the fringes)
        p1_offaxis_fft.png          log |FFT| of the hologram: DC term and the two sidebands
        p1_inline_hologram.png      raw in-line (Gabor) hologram of the same field
        p2_reference_phase.png      reference phase (training target), cividis
        p2_reference_labels.png     phase-derived reference instances (Otsu + watershed)
        p3_phase_pred.png           predicted phase, same colour range as the reference
        p3_amplitude_pred.png       predicted amplitude (+Amplitude configuration)
        p3_segmentation_pred.png    predicted cell instances
        p3_phase_colorbar.png       gradient strip for the phase maps (range in the json)
        p3_amplitude_colorbar.png   gradient strip for the amplitude map (range in the json)
        p4_measurement_overlay.png  predicted phase with cell contours and dry mass per cell
        fig1_panels.json            the numbers printed on the figure (ranges, counts)

    With --no-model only the p1_* and p2_* panels (real data, no network) are
    made; the p3_* and p4_* panels need the trained checkpoints.

RUN (server, needs the data and the checkpoints; pick your GPU with --device)
    from this folder:   python make_fig1_panels.py --project <project root> --device cuda:0
    (in the public repository <project root> is ../.. ; the study used --device cuda:2)

    Then open figure_1_overview.pptx, right-click each grey placeholder box ->
    Change Picture -> From a File, and pick the PNG with the name written in
    that box. (Or delete the box and Insert -> Picture at the same size.)

INPUTS (all read from the project, nothing is copied)
    config/v2/a_baseline.yaml             data paths, optics, mask rules
    data/<off_axis|gabor|phase|mask>/     the chosen field
    runs/v2_baseline_off_axis/best_model.pt   End-to-End Neural Baseline (seed 42)
    runs/v2_amplitude_off_axis/best_model.pt  +Amplitude (for the amplitude panel)

The default field NCI_08 is a control-condition TEST field (never seen in
training) with 36 matched cells in the seed-42 evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def stretch(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Percentile contrast stretch to [0, 1] for display only."""
    lo, hi = np.percentile(image, [low, high])
    return np.clip((image - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def save_rgb(array: np.ndarray, path: Path, size: int | None = None) -> None:
    """Save an HxW (grey) or HxWx3/4 float array in [0, 1] as PNG, optionally resized."""
    from PIL import Image

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    img.save(path)
    print(f"  wrote {path.name}")


def colour_instances(labels: np.ndarray, background=(0.07, 0.07, 0.09)) -> np.ndarray:
    """Distinct colours per instance on a dark background (colour = identity only)."""
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    rgb[:] = background
    ids = np.unique(labels)
    ids = ids[ids > 0]
    palette = plt.get_cmap("tab20")
    rng = np.random.default_rng(7)          # fixed order so reruns look identical
    order = rng.permutation(len(ids))
    for k, label in zip(order, ids):
        rgb[labels == label] = palette(k % 20)[:3]
    return rgb


def colorbar_png(cmap, path: Path) -> None:
    """A plain vertical gradient strip (low at the bottom), no text.

    The tick labels are editable text boxes in the .pptx; their values are the
    display range written to fig1_panels.json.
    """
    strip = np.linspace(1.0, 0.0, 512)[:, None] * np.ones((1, 40))
    save_rgb(cmap(strip)[..., :3], path)


def fringe_window(hologram: np.ndarray, mask: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of a window centred on the largest cell (fringes bend there)."""
    from scipy import ndimage

    labels, n = ndimage.label(mask > 0)
    if n == 0:
        h, w = hologram.shape
        return h // 2 - size // 2, w // 2 - size // 2
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    cy, cx = ndimage.center_of_mass(labels == (int(np.argmax(sizes)) + 1))
    h, w = hologram.shape
    y = int(np.clip(cy - size // 2, 0, h - size))
    x = int(np.clip(cx - size // 2, 0, w - size))
    return y, x


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="project folder (contains holoqpi/)")
    parser.add_argument("--config", default="config/v2/a_baseline.yaml")
    parser.add_argument("--stem", default="NCI_08", help="field to draw (a test field)")
    parser.add_argument("--data-root", default=None, help="override paths.data_root")
    parser.add_argument("--checkpoint", default="runs/v2_baseline_off_axis/best_model.pt")
    parser.add_argument("--amplitude-config", default="config/v2/d0_amplitude.yaml")
    parser.add_argument("--amplitude-checkpoint",
                        default="runs/v2_amplitude_off_axis/best_model.pt")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--no-model", action="store_true",
                        help="make only the data panels (p1_*, p2_*); no checkpoint needed")
    parser.add_argument("--out", default=None,
                        help="output folder (default: fig1_panels/ beside this script)")
    parser.add_argument("--size", type=int, default=900, help="saved size of the square maps")
    parser.add_argument("--view", default="60,100,700",
                        help="y0,x0,side of the square region shown in every map panel. "
                             "The field border carries an illumination-edge artefact in the "
                             "reference phase (and hence in its Otsu labels); the default "
                             "window for NCI_08 keeps clear of it. Use 0,0,900 for the full field.")
    parser.add_argument("--zoom", type=int, default=40, help="fringe-detail window in px")
    parser.add_argument("--overlay-crop", type=int, default=420,
                        help="side of the square crop used for the measurement overlay")
    parser.add_argument("--label-cells", type=int, default=20,
                        help="maximum number of cells that get a dry-mass label on the overlay")
    parser.add_argument("--label-fontsize", type=float, default=11.0,
                        help="font size of the dry-mass labels on the overlay (border-cut cells use 7/8 of it)")
    parser.add_argument("--min-visible-px", type=int, default=1500,
                        help="smallest visible part (px) of a border-cut cell that still gets a label")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    sys.path.insert(0, str(project))
    from holoqpi.config import load_config
    from holoqpi.data import io as data_io
    from holoqpi.data.masks import split_instances
    from holoqpi.analysis.cells import calibration_from_config, measure_cells

    overrides = {"paths": {"data_root": args.data_root}} if args.data_root else None
    cfg = load_config(project / args.config, overrides)
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = project / data_root
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "fig1_panels"
    out.mkdir(parents=True, exist_ok=True)
    stem = args.stem
    size = cfg.data.phase_size
    info: dict = {"stem": stem, "config": args.config}
    print(f"field {stem}  ->  {out}")

    # ---- raw data --------------------------------------------------------
    holo_off_raw = data_io.read_hologram(data_io.hologram_path(data_root, cfg, stem, "off_axis"))
    holo_in_raw = data_io.read_hologram(data_io.hologram_path(data_root, cfg, stem, "gabor"))
    # The same whole-pixel crop as training and evaluation (data.crop_offset_px).
    crop_offset = data_io.hologram_crop_offset(cfg.data)
    info["crop_offset_px"] = list(crop_offset)
    holo_off = data_io.align_to_phase_grid(holo_off_raw, size, cfg.data.align, crop_offset)
    holo_in = data_io.align_to_phase_grid(holo_in_raw, size, cfg.data.align, crop_offset)
    phase_ref = data_io.read_phase_bin(data_io.phase_path(data_root, cfg, stem),
                                       cfg.formats.phase_binary).phase
    mask_ref = data_io.read_mask(data_io.mask_path(data_root, cfg, stem))
    info["hologram_raw_shape"] = list(holo_off_raw.shape)
    info["phase_shape"] = list(phase_ref.shape)
    vy, vx, vs = (int(v) for v in args.view.split(","))
    view = (slice(vy, vy + vs), slice(vx, vx + vs))
    info["view_yx_side"] = [vy, vx, vs]

    # ---- p1: inputs ------------------------------------------------------
    save_rgb(stretch(holo_off[view]), out / "p1_offaxis_hologram.png", args.size)
    save_rgb(stretch(holo_in[view]), out / "p1_inline_hologram.png", args.size)

    y, x = fringe_window(holo_off[view], mask_ref[view], args.zoom)
    y, x = y + vy, x + vx
    info["zoom_window_yx"] = [y, x, args.zoom]
    save_rgb(stretch(holo_off[y:y + args.zoom, x:x + args.zoom], 0.5, 99.5),
             out / "p1_offaxis_zoom.png", 480)

    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(holo_off - holo_off.mean()))))
    spec = stretch(spectrum, 50.0, 99.95) ** 1.4
    save_rgb(plt.get_cmap("magma")(spec)[..., :3], out / "p1_offaxis_fft.png", args.size)

    # ---- p2: reference phase and phase-derived labels ----------------------
    vmin, vmax = np.percentile(phase_ref[view], [0.5, 99.8])
    vmin, vmax = float(np.floor(vmin * 2) / 2), float(np.ceil(vmax * 2) / 2)
    info["phase_display_range_rad"] = [vmin, vmax]
    phase_cmap = plt.get_cmap("cividis")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    save_rgb(phase_cmap(norm(phase_ref[view]))[..., :3], out / "p2_reference_phase.png", args.size)

    method = cfg.evaluation.segmentation.instance_from
    distance = cfg.mask_generation.watershed_min_distance_px
    ref_labels = split_instances((mask_ref > 0).astype(np.uint8), method, distance)
    info["reference_cells"] = int(ref_labels.max())
    save_rgb(colour_instances(ref_labels[view]), out / "p2_reference_labels.png", args.size)
    colorbar_png(phase_cmap, out / "p3_phase_colorbar.png")

    if args.no_model:
        (out / "fig1_panels.json").write_text(json.dumps(info, indent=2))
        print("  wrote fig1_panels.json  (data panels only; rerun without --no-model "
              "on the server for the network panels)")
        return 0

    # ---- p3: network predictions ------------------------------------------
    import torch
    from holoqpi.models import build_model
    from holoqpi.engine import load_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x_in = data_io.normalise_hologram(holo_off, cfg.data.hologram_normalisation)
    x_in = torch.from_numpy(x_in[None, None].astype(np.float32)).to(device)

    def predict(config_path: str, checkpoint: str) -> dict:
        c = load_config(project / config_path, overrides)
        model = build_model(c)
        load_checkpoint(model, project / checkpoint, device)
        model.to(device).eval()
        with torch.no_grad():
            return {k: v.float().cpu().numpy() for k, v in model(x_in).items()}

    base = predict(args.config, args.checkpoint)
    phase_pred = base["phase"][0, 0]
    mask_pred = base["segmentation"][0].argmax(axis=0)
    pred_labels = split_instances((mask_pred > 0).astype(np.uint8), method, distance)
    save_rgb(phase_cmap(norm(phase_pred[view]))[..., :3], out / "p3_phase_pred.png", args.size)
    save_rgb(colour_instances(pred_labels[view]), out / "p3_segmentation_pred.png", args.size)
    info["predicted_cells"] = int(pred_labels.max())
    info["phase_mae_rad_this_field"] = float(np.abs(phase_pred - phase_ref).mean())

    amp_ckpt = project / args.amplitude_checkpoint
    if amp_ckpt.is_file():
        amp = predict(args.amplitude_config, args.amplitude_checkpoint)["amplitude"][0, 0]
        save_rgb(stretch(amp[view], 0.5, 99.5), out / "p3_amplitude_pred.png", args.size)
        lo, hi = np.percentile(amp[view], [0.5, 99.5])
        colorbar_png(plt.get_cmap("gray"), out / "p3_amplitude_colorbar.png")
        info["amplitude_display_range"] = [float(lo), float(hi)]
    else:
        print(f"  skipped amplitude panel: {amp_ckpt} not found")

    # ---- p4: per-cell measurement overlay ---------------------------------
    calibration = calibration_from_config(cfg)
    cells = measure_cells(phase_pred, mask_pred, calibration, cfg.evaluation.measurement,
                          method, distance, labels=pred_labels)
    info["measured_cells"] = len(cells)
    if cells:
        masses = np.array([c["dry_mass_pg"] for c in cells])
        info["dry_mass_pg_median"] = float(np.median(masses))
        info["dry_mass_pg_range"] = [float(masses.min()), float(masses.max())]

    # Crop around the densest group of measured cells so the labels are legible.
    crop = min(args.overlay_crop, size)
    in_view = [c for c in cells if vy <= c["centroid_y"] < vy + vs and vx <= c["centroid_x"] < vx + vs]
    if in_view:
        cy = float(np.median([c["centroid_y"] for c in in_view]))
        cx = float(np.median([c["centroid_x"] for c in in_view]))
    else:
        cy = cx = size / 2
    y0 = int(np.clip(cy - crop / 2, 0, size - crop))
    x0 = int(np.clip(cx - crop / 2, 0, size - crop))
    info["overlay_crop_yx"] = [y0, x0, crop]

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(phase_pred[y0:y0 + crop, x0:x0 + crop], cmap="gray", vmin=vmin, vmax=vmax)
    window = pred_labels[y0:y0 + crop, x0:x0 + crop]
    kept = {c["label"] for c in cells}
    for label in np.unique(window):
        if label == 0 or label not in kept:
            continue
        ax.contour(window == label, levels=[0.5], colors=["#F2A900"], linewidths=1.0)
    # Label every measured cell that has a visible part in the crop. The label sits at
    # the centre of the VISIBLE part, pulled inside the frame, so cells cut by the crop
    # border are labelled too (the value is the dry mass of the whole cell in the
    # field). Slivers smaller than --min-visible-px are skipped.
    by_label = {c["label"]: c for c in cells}
    placed = []
    for label in np.unique(window):
        if label == 0 or label not in by_label:
            continue
        yy, xx = np.nonzero(window == label)
        if yy.size < args.min_visible_px:
            continue
        placed.append((yy.size, label, float(xx.mean()), float(yy.mean()),
                       bool(yy.min() == 0 or xx.min() == 0
                            or yy.max() == crop - 1 or xx.max() == crop - 1)))
    placed.sort(reverse=True)
    for _, label, px, py, cut in placed[: args.label_cells]:
        # Cells cut by the crop border: anchor the label at the frame side that touches
        # the cell (text extends inwards over the visible part), not at the centroid.
        ha, x = "center", px
        if cut and px < 75:
            ha, x = "left", 6.0
        elif cut and px > crop - 75:
            ha, x = "right", crop - 6.0
        y = float(np.clip(py, 14, crop - 14))
        ax.text(x, y, f"{by_label[label]['dry_mass_pg']:.0f} pg",
                ha=ha, va="center", fontsize=args.label_fontsize * (0.875 if cut else 1.0), color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    ax.set_axis_off()
    fig.savefig(out / "p4_measurement_overlay.png")
    plt.close(fig)
    print("  wrote p4_measurement_overlay.png")

    (out / "fig1_panels.json").write_text(json.dumps(info, indent=2))
    print("  wrote fig1_panels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())