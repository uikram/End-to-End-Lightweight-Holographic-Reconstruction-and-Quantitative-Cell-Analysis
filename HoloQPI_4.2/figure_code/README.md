# Figures of HoloQPI_4.2

| File in manuscript_images/ | Printed | Status in 4.2 | Built by |
|---|---|---|---|
| figure_1_overview.png | Fig. 1 | **Rebuilt.** Text corrected: integrated phase S_k, C_k formula, "forward model", no em dash. | `fig1/build_fig1.js` (panels in `fig1/fig1_panels`, written by `paper/manuscript_images/make_fig1_panels.py` on the server with crop offset [-6, -2]); then `soffice --convert-to pdf` and `pdftoppm -r 240` |
| figure_2.png | Fig. 2 | Verified (corrected runs; in-line panel on 107 fields) | `paper/manuscript_images/figure_2.py` |
| figure_3.png | Fig. 3 | Verified (server, corrected crop, seed-42 baseline) | `paper/manuscript_images/figure_3.py` (needs checkpoints) |
| figure_4.png | Fig. 4 | Verified (seed-42 run, n = 1898) | `paper/manuscript_images/figure_6.py` |
| figure_5.png | Fig. 5 | Verified | `paper/manuscript_images/figure_4.py` |
| figure_6.png | Fig. 6 | Verified (w = 1.0 point = +IPP (per-cell), n = 3) | `paper/manuscript_images/figure_5.py` |
| figure_7.png | Fig. 7 | **Regenerated.** Panel (c) is now the residual scan over z for both geometries; verdict words and "ground truth" removed. | `figure_7.py` (this folder) |
| figure_8.png | Fig. 8 | **Regenerated** (data unchanged; "analytic truth" → "analytic value") | `make_figure_boundary_sensitivity.py --root <repository> --out ../manuscript_images` |
| figure_9.png | Fig. 9 | Verified (Compact − Baseline −0.0002) | `paper/manuscript_images/figure_9.py` |

`figure_7.py` reads `<repository>/runs` (or `$HOLOQPI_RUNS`) through the `mstyle.py` in this folder.

Run:

    python figure_7.py
