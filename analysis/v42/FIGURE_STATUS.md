# FIGURE_STATUS

Manuscript figures are the PNGs in `HoloQPI_4.2/manuscript_images/` (not modified). "Regenerated" = rebuilt here from the repository result files into a scratch directory (not overwriting the manuscript files); "verified" = the plotted data were checked against stored results.

| Fig. | Manuscript file | Source script | Source data | Status | Issues |
|---|---|---|---|---|---|
| 1 | `figure_1_overview.png` | `HoloQPI_4.2/figure_code/fig1/build_fig1.js`, panels from `manuscript_images/make_fig1_panels.py` | `fig1_panels.json` (crop offset [−6, −2]) | Not regenerated | Panels need checkpoints and data (not in repo). Offset in `fig1_panels.json` equals `config/base.yaml`. |
| 2 | `figure_2.png` | `results_for_manuscript/figures/scripts/fig2_learned_vs_classical.py` | `runs/benchmark_results/results_arm_A.json`, `results_conventional_*.json` | Regenerated here (runs, same data); verified data | Output size differs by 1 px from the manuscript PNG, i.e. the authoring script in `paper/manuscript_images/` (outside the repo) differs slightly from the repo copy. Not pixel-identical; no numeric difference found. |
| 3 | `figure_3.png` | `manuscript_images/figure_3.py` | checkpoints, data | Identical to the repo PNG (same md5); not regenerated | Needs checkpoints. Field MAE labels were verified against `fig1_panels.json` (0.2959 rad NCI_08). |
| 4 (agreement) | `figure_4.png` | `…/scripts/fig4_agreement.py` | `data/per_cell_test_A_seed42.csv` (seed 42; n = 1898) | Regenerated here; verified (prints n = 1898, per-field n = 113) | 1-px size difference as above. The repo-root `manuscript_images/figure_4.png` is the **old ablation figure** (v4.1 numbering) and must not be used. |
| 5 (ablation) | `figure_5.png` | `…/scripts/fig5_ablation.py` | `results_arm_*.json` | Regenerated here; verified (replicated arms A, B, B1 only) | 489,975 differing pixels vs manuscript PNG (layout/size drift from the older script). |
| 6 (weight sweep) | `figure_6.png` | `…/scripts/fig6_weight_sweep.py` | `results_arm_{A,W01,W03,B,W30}.json` | Regenerated here; verified | w = 1.0 point is the three-seed +IPP (per-cell) (not the separate `w10` repeat). Pixel drift as above. |
| 7 (forward model) | `figure_7.png` | `…/scripts/fig7_forward_model.py` | `results_arm_*.json`, `RESULTS.md`, `z_calibration.json` | **Regenerated here and compared visually: identical content** (differences are font rendering only) | The archived `legacy/figures_v4.1/manuscript_images/figure_7.py/.png` is stale (title "physics-aware terms", "ground-truth", "negative result"); do not use. |
| 8 (boundary displacement) | `figure_8.png` | `…/scripts/fig8_boundary_sensitivity.py` | `runs/error_propagation_summary.csv`, `runs/synthetic_validation_seeds/` | Regenerated here; verified | 3-px size difference (matplotlib/font version). |
| 9 (efficiency) | `figure_9.png` | `…/scripts/fig9_efficiency.py` | `results_hardware_arm_*.json`, `results_arm_{A,KA,B,KB}.json` | Regenerated here; verified (compact − baseline = −0.0002, worst p99/p50 1.20) | 5-px size difference. |

All scripts above live in `results_for_manuscript/figures/scripts/` and read `results_for_manuscript/figures/data/` (copied from `runs/` by `compile_results.py`). `python results_for_manuscript/figures/scripts/make_all.py` regenerates figures 2, 4–9 into `results_for_manuscript/figures/regenerated/` (never over the manuscript files); tested, all eight regenerate. Figures 1 and 3 need checkpoints/data; their server-side inputs stay in `manuscript_images/`.

Archived (not used): the v4.1 scripts/PNGs and `assets/` are in `legacy/figures_v4.1/` (README inside). `figures/` (exploratory `scripts/make_figures.py` output) is kept but is not a manuscript figure set.

**Limits:** the regenerated PNGs are not byte-identical to `HoloQPI_4.2/manuscript_images/` (the authoring versions of the scripts for figures 2, 4, 5, 6, 9 are outside the repo, in `paper/manuscript_images/`; fonts/matplotlib version also differ). Plotted data are the same; no figure pixels were edited.
