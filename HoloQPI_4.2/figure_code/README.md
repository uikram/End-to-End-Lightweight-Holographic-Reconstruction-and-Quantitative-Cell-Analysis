# Figure code for HoloQPI_4.2

| Figure | Built by |
|---|---|
| Fig. 1 | `fig1/build_fig1.js` (editable source; panels in `fig1/fig1_panels`, written on the server by `manuscript_images/make_fig1_panels.py` with crop offset [−6, −2]); then `soffice --convert-to pdf` and `pdftoppm -r 240` |
| Fig. 3 | `manuscript_images/figure_3.py` (needs checkpoints) |
| Figs. 2, 4–9 | `results_for_manuscript/figures/scripts/` — run `python results_for_manuscript/figures/scripts/make_all.py` (reads `results_for_manuscript/figures/data/`, writes `results_for_manuscript/figures/regenerated/`) |

See `analysis/v42/FIGURE_STATUS.md` for the status of each figure.
