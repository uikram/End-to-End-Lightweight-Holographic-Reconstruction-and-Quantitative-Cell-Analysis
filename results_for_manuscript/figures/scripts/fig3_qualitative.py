"""Figure 3 - qualitative end-to-end output on held-out test fields.

WHAT IT SHOWS
    One row per test field: the recorded hologram the network is given, the
    predicted quantitative phase, the reference phase, the signed phase error,
    the predicted segmentation and the reference segmentation.

WHY IT IS INCLUDED
    No table conveys that the input is a fringe pattern bearing no resemblance
    to a cell, nor what the residual error looks like spatially. The signed
    error panel is the informative one: dry mass is an integral, so a slow
    offset inside a cell matters far more than zero-mean high-frequency error,
    and an absolute-error map hides the sign that distinguishes them.

REQUIRES THE RESEARCH REPOSITORY
    Unlike the other figure scripts, this one runs the trained network, so it
    needs the repository, a checkpoint and the raw data. Point --project at the
    project root. If the checkpoint or the fields are absent the script says so
    and exits without writing a partial figure.

    python results_for_manuscript/figures/scripts/fig3_qualitative.py \
        --config config/v2/a_baseline.yaml --split test      # -> ../regenerated/figure_3.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mstyle as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[3]),
                    help="project root (default: this repository)")
    ap.add_argument("--config", default="config/v2/a_baseline.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", type=int, default=2, help="fields to show")
    ap.add_argument("--out", default="figure_3.png")
    args = ap.parse_args()

    root = Path(args.project).resolve()
    if not (root / "holoqpi").is_dir():
        sys.exit(f"STOP: {root} does not contain holoqpi/. Pass --project.")
    sys.path.insert(0, str(root))
    import os
    os.chdir(root)

    import torch
    from holoqpi.config import load_config
    from holoqpi.data import io as data_io
    from holoqpi.engine import apply_learned_physics, load_checkpoint
    from holoqpi.models import build_model

    print("Loading configuration and checkpoint...")
    cfg = load_config(args.config)
    run_dir = Path(cfg.paths.output_root) / f"{cfg.experiment_name}_{cfg.data.modality}"
    ckpt = run_dir / "best_model.pt"
    if not ckpt.is_file():
        sys.exit(f"STOP: no checkpoint at {ckpt}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = apply_learned_physics(cfg, ckpt)
    model = build_model(cfg)
    load_checkpoint(model, ckpt, device)
    model.to(device).eval()
    print(f"  {cfg.experiment_name} on {device}")

    data_root = Path(cfg.paths.data_root)
    import json
    splits = json.loads((data_root / cfg.paths.splits_file).read_text())
    stems = splits.get("splits", splits).get(args.split, [])
    holo_dir = data_root / ("off_axis_hologram" if cfg.data.modality == "off_axis"
                            else "in_line_gabor_hologram")
    suffix = cfg.formats.hologram.suffixes.get(cfg.data.modality)
    present = [s for s in stems
               if (holo_dir / f"{s}{suffix}").is_file()
               and data_io.phase_path(data_root, cfg, s).is_file()]
    if not present:
        sys.exit(f"STOP: no '{args.split}' field is present under {data_root}.")
    chosen = present[: args.rows]
    print(f"  {len(present)} field(s) of the '{args.split}' split available; "
          f"showing {len(chosen)}")

    M.style()
    cols = ["Recorded hologram", "Predicted phase", "Reference phase",
            "Signed phase error", "Predicted mask", "Reference mask"]

    # Load everything first so the colour scales can be SHARED across rows.
    # Per-row scaling would make the rows incomparable, which defeats the point
    # of showing more than one field.
    print("Running inference...")
    fields = []
    for i, stem in enumerate(chosen):
        print(f"  [{i+1}/{len(chosen)}] {stem}")
        holo = data_io.read_hologram(holo_dir / f"{stem}{suffix}")
        holo = data_io.align_to_phase_grid(holo, cfg.data.phase_size, cfg.data.align,
                                            data_io.hologram_crop_offset(cfg.data))
        net_in = data_io.normalise_hologram(holo, cfg.data.hologram_normalisation)
        ref = data_io.read_phase_bin(data_io.phase_path(data_root, cfg, stem),
                                     cfg.formats.phase_binary).phase
        mask_path = data_io.mask_path(data_root, cfg, stem)
        ref_mask = data_io.read_mask(mask_path) if mask_path.is_file() else None
        with torch.no_grad():
            out = model(torch.from_numpy(net_in)[None, None].float().to(device))
        phase = out["phase"][0, 0].float().cpu().numpy()
        mask = out["segmentation"].argmax(1)[0].cpu().numpy()
        fields.append(dict(stem=stem, holo=holo, phase=phase, ref=ref,
                           mask=mask, ref_mask=ref_mask, err=phase - ref))

    allref = np.concatenate([f["ref"].ravel() for f in fields])
    lo, hi = np.percentile(allref, [1, 99.5])
    lim = float(np.percentile(np.abs(np.concatenate(
        [f["err"].ravel() for f in fields])), 99))
    print(f"  shared phase scale [{lo:.2f}, {hi:.2f}] rad; "
          f"error scale +/-{lim:.2f} rad")

    # A thin extra gridspec row carries the two shared colour bars, so no colour
    # bar steals width from a panel or overlaps its neighbour.
    panel_w = M.COL2 / 6.0
    nrow = len(fields)
    fig = plt.figure(figsize=(M.COL2, panel_w * nrow + 0.62))
    gs = fig.add_gridspec(nrow + 1, 6, height_ratios=[1] * nrow + [0.085],
                          hspace=0.05, wspace=0.05,
                          left=0.075, right=0.995, top=0.935, bottom=0.075)

    im_phase = im_err = None
    for i, f in enumerate(fields):
        panels = [
            (f["holo"], M.CMAP_INTENSITY, None, None),
            (f["phase"], M.CMAP_PHASE, lo, hi),
            (f["ref"], M.CMAP_PHASE, lo, hi),
            (f["err"], M.CMAP_SIGNED, -lim, lim),
            (f["mask"], "gray", 0, 1),
            (f["ref_mask"], "gray", 0, 1),
        ]
        for j, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            if img is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color=M.NEUTRAL)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                if j == 2:
                    im_phase = im
                if j == 3:
                    im_err = im
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_linewidth(0.4); side.set_color(M.GRID)
            if i == 0:
                ax.set_title(cols[j], fontsize=6.8, pad=3)
            if j == 0:
                ax.set_ylabel(f["stem"].replace("_", "\n"), fontsize=5.6,
                              rotation=0, ha="right", va="center", labelpad=4)
            if j == 3:
                # Inside the panel, so it cannot push the grid around.
                ax.text(0.03, 0.045,
                        f"MAE {np.abs(f['err']).mean():.3f} rad",
                        transform=ax.transAxes, fontsize=5.8, color=M.INK,
                        bbox=dict(facecolor="white", alpha=0.82, linewidth=0,
                                  boxstyle="round,pad=0.15"))

    for im, col, label in ((im_phase, 2, "quantitative phase [rad]"),
                           (im_err, 3, "phase error [rad]")):
        if im is None:
            continue
        cax = fig.add_subplot(gs[nrow, col])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.4, width=0.4, length=1.8, pad=1)
        cb.outline.set_linewidth(0.4)
        cb.set_label(label, fontsize=5.8, labelpad=1.5)

    M.save(fig, args.out)


if __name__ == "__main__":
    main()
