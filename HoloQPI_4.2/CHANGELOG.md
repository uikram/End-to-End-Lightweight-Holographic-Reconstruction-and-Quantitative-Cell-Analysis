# CHANGELOG: HoloQPI_Manuscript_V4.1 → HoloQPI_4.2

> **Status note.** The regeneration commands in this changelog refer to the first-generation scripts (now in `legacy/analysis_v42_old/`). Current tooling: `python analysis/v42/compile_results.py`, `update_tables.py`, `check_numbers.py` (see `README.md`).


HoloQPI_Manuscript_V4.1 was not modified. All numbers are read from the corrected result files.

**Regenerate.** Run these from the repository root:

```
python analysis/v42/extract_values.py
python analysis/v42/make_tables_v42.py --out "Claude outputs/HoloQPI_4.2"
python analysis/v42/check_numbers.py   --tex "Claude outputs/HoloQPI_4.2"
```

**Corrections behind the new numbers.**
- The hologram crop origin is (−6, −2) px.
- The SNU_01–50 in-line frames are excluded from the in-line modality (521/122/107).
- All configurations were retrained and re-evaluated on the server on 2026-09-30.

## 1. Structure

- **Title:** "Joint Phase Reconstruction and Cell Segmentation from Raw Holograms for Per-Cell Dry-Mass Measurement".
- **Sections:** the order is now as specified.
  - 1 Introduction
  - 2 Related work (2.1–2.6 unchanged in title)
  - 3 Method (3.1–3.5, with 3.5.1–3.5.4 as subsubsections)
  - 4 Experiments (4.1–4.8; statistical analysis last)
  - 5 Results (5.1–5.7)
  - 6 Discussion (6.1–6.5)
  - 7 Limitations
  - 8 Conclusions
- **Moved text:**
  - "Dataset and reference phase" is now 4.1.
  - Labels are now 4.2.
  - Configurations are now 4.3.
  - The classical pipeline is now 4.4.
  - Evaluation is now 4.5.
  - The benchmark protocol (previously inside evaluation) is now 4.6.
  - The model-free analyses are now 4.7.
  - The seed/resolution rule and checkpoint criterion are now 4.8.
  - The efficiency results (previously inside 5.1) are now 5.7.
  - The implications paragraph (previously 6.6) is now in 6.1.
- **Labels:** all `\label`/`\ref` were updated, and `sec:stats` and `eq:decomp` are new. The compile has no undefined references or citations.
- **Introduction:** it now ends with four numbered contributions (`enumerate`). The model-free 1 px result is stated once as motivation.
- **File numbering:** table and figure files are renumbered to match their printed numbers.

| File in 4.2 | Printed | Content | Was (V4.1 file) |
|---|---|---|---|
| table_5.tex | Table 5 | Common 107 fields, both geometries (**new**) | — |
| table_6.tex | Table 6 | Measurement agreement | table_7.tex |
| table_7.tex | Table 7 | Mass-error decomposition (**new**) | — |
| table_8.tex | Table 8 | Edge/interior stratification | table_11.tex |
| table_9.tex | Table 9 | Ablation (+ field-total MAPE column) | table_5.tex |
| table_10.tex | Table 10 | Seeds and Δ/2×SD/verdicts | table_6.tex |
| table_11.tex | Table 11 | Amplitude, forward residual, phase-scale probes (part c new) | table_8.tex |
| table_12.tex | Table 12 | Boundary displacement, synthetic floor | table_9.tex |
| table_13.tex | Table 13 | Computational benchmarking | table_10.tex |
| figure_4.png | Fig. 4 | Agreement / Bland–Altman | figure_6.png |
| figure_5.png | Fig. 5 | Ablation | figure_4.png |
| figure_6.png | Fig. 6 | Weight sweep | figure_5.png |

## 2. Terminology (global)

- **Integrated phase:** V_k / "optical volume" → integrated phase S_k (rad µm²), and Eq. (mass) is now m_k = λ/(2πα) S_k. "Optical volume" remains only in Related Work 2.3, where it describes other papers.
- **Loss names:** `\Lpv`/L_PV → `\Lippi` (L_IPP^img), and the per-cell term → `\Lippc` (L_IPP^cell). Configuration names are unchanged.
- **Objective groups:** two categories, measurement-aware objectives and forward-model consistency (the only physics-based term). "Physics-based objective experiments" in Table 3 is now "Amplitude output and forward-model consistency".
- **Single reporting of dry-mass error:** Sec. 4.5 states that per-cell relative integrated-phase error equals dry-mass relative error. The Optical-volume rows were removed from the measurement table.
- **Phase-map format:** phase maps are stated to be float32 in radians (Sec. 4.1, Table 2).
- **"Bottleneck":** replaced by "projection" in Table 3 and Figure 1.
- **Benchmarking wording:** "computational benchmarking" (Sec. 4.6, 5.7, Table 13).

## 3. New disclosures

- **Crop origin (Sec. 4.1, Table 2, Limitations):** (−6, −2) px.
  - Validation centre-crop offset +5.4/+1.1 px.
  - 3×3 neighbourhood search.
  - Test median residual −0.1/0.0 px, with 5–95 % ranges.
  - The same crop is used everywhere.
- **Gabor exclusion (Sec. 4.1, 4.3, Tables 2–3):** band-pass registration method and control, the low-pass-only variant, and the 521/122/107 split. Cross-geometry comparisons use 107 fields / 3055 cells. Registration statistics are marked TODO-SOURCE (see DISCREPANCIES.md).
- **Forward-model tolerance:** 0.01, called "configured" (Sec. 3.5.4, 5.5, Table 11).
- **Sec. 4.1:** written from project files only. Cuche 1999 is cited for the off-axis reconstruction principle (added to references.bib). The supplied phase pipeline is described as not known to us, and Park et al.'s 750 samples are not mentioned.
- **IPP comparison:** B vs B1 is equal-coefficient with unmatched gradients. The gradient ratio is 0.373 (0.243–0.655), from `runs/gradient_path_b_cell_ipp_512.csv`.

## 4. Rewritten passages

| Section | What changed | Why |
|---|---|---|
| Abstract | Regenerated; 113-field values and 107-field single-run in-line values labelled in the same sentences; IPP, forward-model and 1 px results in the prescribed wording | Corrected results |
| 1 Intro, 2.4 | Park et al.: two segmentation strategies, 256×256 inputs, distributions and classifier accuracy (TODO-SOURCE); gap statement narrowed to the exact sentence | Prompt; Park SI |
| 3.5.2 | Formal property of L_IPP^cell (receding foreground with raised phase) kept as a property of the formulation | Prompt |
| 4.5 | Evaluation reorganised into reconstruction / segmentation and detection / per-cell / coverage and field totals / decomposition (Eq. 17) | New decomposition |
| 5.1 | Rebuilt around the corrected classical comparison: boundary F1 +19 % (was +86 %), area MAPE about equal, dry-mass lower for the network, field-total lower for classical. In-line now on 107 common fields (Table 5), with Park SI S2 cited as consistent (TODO-SOURCE). The "50.5 µm" sentence is removed. | Crop correction; Gabor exclusion |
| 5.2 | New decomposition paragraph and Table 7; compensatory error structure of the classical pipeline; the old caveat "the experiment does not separate the two contributions" is replaced by the measured split scoped to matched cells; edge stratification recomputed (Table 8, classical row added); median-area sentence removed (no source) | Diagnostics |
| 5.3 | Prescribed IPP wording with new Δ (+0.0141, +0.0145); field-total not resolved; decomposition results; +IPP (per-cell) field-level domain and false-positive increase mentioned (resolved); single runs "not resolvable"; repeat run W10 noted | Corrected results |
| 5.4 | Amplitude values regenerated in the corrected frame (ratio 0.443–0.453) | Corrected amplitude reference |
| 5.5 | Residuals 0.8914 (test) and 0.915/0.699 (validation); all ratios > 1; +Fwd no longer lower than the baseline; in-line z minimum at the recording distance (reversal of the v4.1 statement); off-axis z not constrained; probes with margins and field counts for both geometries | z_calibration.json |
| 5.7 | Efficiency from the corrected benchmark sessions | Benchmark JSONs |
| 6.2 | The +86 % argument replaced by the corrected Dice/area/dry-mass contrast and the decomposition; the scale argument kept | Crop correction |
| 6.3 | Field-total reversal and compensatory structure added | Diagnostics |
| 6.4 | Untested label/edge explanations kept only as untested possibilities; formal route "not observed" | Prompt |
| 6.5 | Rewritten: in-line residual identifies z, off-axis does not; sensitivity below tolerance in both; the carrier explanation kept as an interpretation | z_calibration.json |
| 7 | Updated list (in-line single run on 107 fields after 50 excluded; empirical crop origin with per-field scatter; labels; single-run ablations; one instrument; α); obsolete in-line distance statement removed | Prompt |
| 8 | Regenerated; no latency sentence first | Prompt |
| Fig. 1–9 captions | State what is shown, conditions and field set (113 / 107); no verdict words | Prompt |

## 5. Pre-correction text intentionally kept

- **Related Work statements and literature numbers.** These include "optical volume" in 2.3, the 96.97 M and 89.47 ms of Park et al., and the α literature values. They describe other papers and do not depend on the correction.
- **Label-audit numbers.** Otsu level 0.406 ± 0.151 rad and ANOVA F = 15.26. They were re-read from the corrected-run `runs/label_audit_thresholds.json` and are unchanged, because the labels do not depend on the crop.
- **Membrane registration parameters.** These are a property of the membrane data, unaffected by the hologram crop.
- **Model-free boundary-displacement table and synthetic floor.** The files were verified to be identical in value (`runs/error_propagation_summary.csv`, `results_synthetic_validation.json`).
- **Grep hit "50.5".** The only hit is the substring of 0.5075 (+Area coverage-adjusted MAPE) in table_9.tex, not the old in-line distance.

## 5b. Second pass (2026-10-01)

1. **Sec. 4.1, registration.** Both TODO-SOURCE comments are replaced by the archive values:
   - matched pairs ≤1 px on 734/800, controls on 3/800;
   - AUC 0.975 (0.973 by error);
   - 750 paired fields within 4.7 px, 734 within 1 px, r 0.47–1.00 above the highest control value of 0.25;
   - SNU_01–50 r −0.13 to 0.18;
   - low-pass AUC 0.506.

   The diagnostics archive was added to the provenance comment at the top of main.tex.
2. **Park et al.** The three TODO-SOURCE comments were deleted. In the Introduction and Sec. 2.4, "agreement is reported as distributions and as classifier accuracy" is now "measurements from generated and ground-truth images are compared as distributions, and the generated measurements are used for cell-type and drug-response classification".
3. **Discussion 6.5.** The comparison is now on a common set: on the 112 test fields with a per-field surface, the residual is 0.893 with the global surface and 0.391 with per-field surfaces. The previous text gave 0.891 on 113 fields against 0.391 on 112.
   - Sec. 5.5 states that per-field surfaces exist for 112 of the 113 fields.
   - The Table 11 note names the missing field (T24_Staurosporine_100nM_10, per-field fit rejected: surface 405 rad above the 45 rad peak-to-valley limit) and gives the 112-field global residual (0.8926).
   - New keys: `probe.global_common.*`, `probe.per_field.missing.*`.
4. **Table 7 note.** The re-scoring gives a per-run mean of 1889 matched cells, against 1890 ± 9 from the stored evaluation (Table 4).
5. **Table 13 caption.** The +Amplitude and Compact Baseline rows are each a single benchmark session.
6. **Fig. 1 panel (b).** The title "Lightweight end-to-end network" is now "End-to-end network"; the figure was rebuilt.

**Checks after the pass:**
- recompiled with no undefined references and no "??";
- `check_numbers.py`: 2027 numbers, 0 without a source;
- no TODO-SOURCE remains;
- banned-string grep: only the allowed hits ("optical volume" in Related Work 2.3, and "50.5" as a substring of 0.5075).

## 6. Changed numbers (v4.1 → 4.2)

Table values were regenerated in full from `values.json` by `make_tables_v42.py`. The cell-by-cell source of every printed table value is in `analysis/v42/table_cells.csv`, and its check is in `analysis/v42/number_check.csv`. The numbers that appear in the text are listed below.

| Quantity | v4.1 | 4.2 | Source (file; key in analysis/v42/values.json) |
|---|---|---|---|
| Off-axis baseline dry-mass MAPE (3 seeds) | 0.1763 ± 0.0024 | 0.1742 ± 0.0045 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_mape` |
| Seed-42 bootstrap CI, dry-mass MAPE | 0.1666–0.1867 | 0.1650–0.1849 | `runs/v2_baseline_off_axis/metrics_test.json`; `run.A.s42.dry_mass_mape_ci_lower` |
| Baseline per-cell dry-mass r | 0.9359 | 0.9363 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_cell_pearson_r` |
| Baseline area MAPE | 0.1546 | 0.1568 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_mape` |
| Baseline area r | 0.8979 | 0.8986 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_cell_pearson_r` |
| Baseline circularity MAPE | 0.0637 | 0.0571 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.circularity_mape` |
| Baseline circularity r | 0.8713 | 0.8861 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.circularity_cell_pearson_r` |
| Baseline coverage-adjusted dry-mass MAPE | 0.5124 | 0.5102 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_mape_coverage_adjusted` |
| Baseline recall | 0.5920 ± 0.0011 | 0.5931 ± 0.0027 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.detection_recall` |
| Baseline precision | 0.8550 | 0.8488 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.detection_precision` |
| Baseline matched cells | 1886 ± 4 | 1890 ± 9 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.cells_matched` |
| Baseline phase MAE [rad] | 0.1593 | 0.1568 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_mae_rad` |
| Baseline in-cell phase MAE [rad] | 0.2756 | 0.2689 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_mae_rad_in_cell` |
| Baseline phase r | 0.8701 | 0.8741 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_pearson_r` |
| Baseline SSIM | 0.6275 | 0.6386 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.phase_ssim` |
| Baseline Dice | 0.8312 | 0.8337 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_dice` |
| Baseline AJI | 0.6696 | 0.6710 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_aji` |
| Baseline boundary F1 | 0.4574 | 0.4687 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.seg_boundary_f1` |
| Baseline BA bias dry mass | -0.0434 | -0.0372 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_relative_bias` |
| Baseline BA LoA dry mass | [-0.252, +0.165] | [-0.247, +0.172] | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_loa_lower` |
| Baseline BA bias area | +0.0455 | 0.0649 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_relative_bias` |
| Baseline BA LoA area | [-0.172, +0.263] | [-0.162, +0.292] | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_loa_lower` |
| Baseline field-total bias, dry mass | -0.1632 | -0.1560 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_bias` |
| Baseline field-total bias, area | -0.1078 | -0.0901 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.area_field_total_bias` |
| Baseline field-total r, dry mass | 0.9879 | 0.9883 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_pearson_r` |
| Baseline field-total dry-mass MAPE (new row) | -- | 0.1631 ± 0.0141 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.dry_mass_field_total_mape` |
| Classical off-axis dry-mass MAPE | 0.2245 | 0.2203 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_mape` |
| Classical off-axis area MAPE | 0.1567 | 0.1578 | `runs/conventional_baseline_test.json`; `classical113.off_axis.area_mape` |
| Classical off-axis phase MAE | 0.2114 | 0.1968 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_mae_rad` |
| Classical off-axis in-cell phase MAE | 0.3655 | 0.3112 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_mae_rad_in_cell` |
| Classical off-axis phase r | 0.7580 | 0.7851 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_pearson_r` |
| Classical off-axis SSIM | 0.5022 | 0.5299 | `runs/conventional_baseline_test.json`; `classical113.off_axis.phase_ssim` |
| Classical off-axis Dice | 0.7885 | 0.8253 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_dice` |
| Classical off-axis AJI | 0.5931 | 0.6387 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_aji` |
| Classical off-axis boundary F1 | 0.2463 | 0.3954 | `runs/conventional_baseline_test.json`; `classical113.off_axis.seg_boundary_f1` |
| Classical off-axis recall | 0.5763 | 0.5869 | `runs/conventional_baseline_test.json`; `classical113.off_axis.detection_recall` |
| Classical off-axis precision | 0.7773 | 0.7910 | `runs/conventional_baseline_test.json`; `classical113.off_axis.detection_precision` |
| Classical off-axis cell r | 0.8904 | 0.8937 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_cell_pearson_r` |
| Classical off-axis matched cells | 1836 | 1870 | `runs/conventional_baseline_test.json`; `classical113.off_axis.cells_matched` |
| Classical off-axis field-total dry-mass MAPE (new) | -- | 0.1094 | `runs/conventional_baseline_test.json`; `classical113.off_axis.dry_mass_field_total_mape` |
| Classical off-axis in-cell phase contrast | +0.902 (113) | 0.922 | `runs/common_fields/conventional_baseline_test.json`; `classical107.off_axis.recovered_phase_contrast_rad` |
| Classical in-line phase r | -0.1359 (113 fields) | -0.2075 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.phase_pearson_r` |
| Classical in-line in-cell contrast | -0.067 | -0.084 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.recovered_phase_contrast_rad` |
| Classical in-line Dice | 0.1627 | 0.1583 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.seg_dice` |
| Classical in-line matched cells | 116 of 3186 | 106 | `runs/common_fields/conventional_baseline_test.json`; `classical107.gabor.cells_matched` |
| In-line neural phase r | 0.8098 (113 fields) | 0.8667 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.phase_pearson_r` |
| In-line neural Dice | 0.7758 | 0.8148 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.seg_dice` |
| In-line neural recall | 0.5508 | 0.5885 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.detection_recall` |
| In-line neural dry-mass MAPE | 0.2308 | 0.2129 | `runs/common_fields/common_fields_summary.csv`; `common.G_s42.dry_mass_mape` |
| In-line neural matched cells | 1755 of 3186 | 1798 | `runs/v2_baseline_gabor/metrics_test_common.json`; `common_run.G.s42.cells_matched` |
| +IPP (per-cell) dry-mass MAPE | 0.1915 | 0.1883 | `runs/v2_cell_ipp_off_axis/metrics_test.json (+ seeds)`; `mean.B.dry_mass_mape` |
| +IPP (image) dry-mass MAPE | 0.1935 | 0.1887 | `runs/v2_image_volume_off_axis/metrics_test.json (+ seeds)`; `mean.B1.dry_mass_mape` |
| Δ +IPP (per-cell) vs baseline | +0.0152 (2×SD 0.0070) | +0.0141 (2×SD 0.0097) | `runs/*/metrics_test.json`; `cmp.B_vs_A.dry_mass_mape.diff` |
| Δ +IPP (image) vs baseline | +0.0172 (2×SD 0.0052) | +0.0145 (2×SD 0.0066) | `runs/*/metrics_test.json`; `cmp.B1_vs_A.dry_mass_mape.diff` |
| Δ +IPP (image) vs +IPP (per-cell) | 0.0020 (2×SD 0.0074) | +0.0004 (2×SD 0.0076) | `runs/*/metrics_test.json`; `cmp.B1_vs_B.dry_mass_mape.diff` |
| +IPP (per-cell) precision | 0.7995 | 0.8077 | `runs/v2_cell_ipp_off_axis/metrics_test.json (+ seeds)`; `mean.B.detection_precision` |
| +Area dry-mass MAPE | 0.1917 | 0.1850 | `runs/v2_cell_ipp_area_off_axis/metrics_test.json`; `mean.B2.dry_mass_mape` |
| +BGA dry-mass MAPE | 0.1978 | 0.1921 | `runs/v2_cell_ipp_bga_off_axis/metrics_test.json`; `mean.C.dry_mass_mape` |
| w=0.1 dry-mass MAPE | 0.1853 | 0.1754 | `runs/v2_ipp_w01_off_axis/metrics_test.json`; `mean.W01.dry_mass_mape` |
| w=0.3 dry-mass MAPE | 0.1828 | 0.1881 | `runs/v2_ipp_w03_off_axis/metrics_test.json`; `mean.W03.dry_mass_mape` |
| w=3.0 dry-mass MAPE | 0.2102 | 0.1963 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.dry_mass_mape` |
| w=3.0 Dice | 0.8101 | 0.8164 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.seg_dice` |
| w=3.0 precision | 0.7252 | 0.7855 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.detection_precision` |
| w=3.0 phase MAE | 0.1851 | 0.1813 | `runs/v2_ipp_w30_off_axis/metrics_test.json`; `mean.W30.phase_mae_rad` |
| +Amplitude dry-mass MAPE | 0.2015 | 0.1857 | `runs/v2_amplitude_off_axis/metrics_test.json`; `mean.D0.dry_mass_mape` |
| Δ +Amplitude vs +IPP (per-cell) | +0.0100 | -0.0026 | `runs/*/metrics_test.json`; `cmp.D0_vs_B.dry_mass_mape.diff` |
| Compact Baseline dry-mass MAPE | 0.1783 | 0.1740 | `runs/v2_compact_baseline_off_axis/metrics_test.json`; `mean.KA.dry_mass_mape` |
| Δ Compact vs baseline | +0.0019 | -0.0002 | `runs/*/metrics_test.json`; `cmp.KA_vs_A.dry_mass_mape.diff` |
| Baseline 2×SD dry-mass MAPE | 0.0047 | 0.0089 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `sd.A.dry_mass_mape` |
| Edge recall (baseline) | 0.208 | 0.212 | `runs/diagnostics/decomposition_by_config.csv`; `decomp.A.cell.edge.recall_mean` |
| Interior recall (baseline) | 0.914 | 0.912 | `runs/diagnostics/decomposition_by_config.csv`; `decomp.A.cell.interior.recall_mean` |
| Edge dry-mass MAPE (baseline) | 0.298 | 0.305 | `derived`; `locmean.A.edge.mape` |
| Interior dry-mass MAPE (baseline) | 0.153 | 0.149 | `derived`; `locmean.A.interior.mape` |
| False positives per run (baseline) | 320 | 337 | `derived`; `locmean.A.fp_all` |
| False positives within 15 px (baseline) | 141 | 136 | `derived`; `locmean.A.fp_near` |
| False positives per run (+IPP per-cell) | 477 | 456 | `derived`; `locmean.B.fp_all` |
| Amplitude MAE (+Amplitude) | 0.0656 | 0.0651 | `runs/v2_amplitude_off_axis/metrics_test.json`; `amp.D0.amplitude_mae` |
| Amplitude MAE (+Fwd fixed z) | 0.0637 | 0.0665 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_mae` |
| Amplitude r (+Fwd fixed z) | 0.9034 | 0.9064 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_pearson_r` |
| Amplitude ratio to A=1 (+Fwd fixed z) | 0.4337 | 0.4526 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `amp.D1.amplitude_mae_over_unity` |
| Reference amplitude mean | 0.9977 | 0.9985 | `runs/v2_amplitude_off_axis/metrics_test.json`; `amp.D0.amplitude_reference_mean` |
| Off-axis reference residual (test) | 0.923 | 0.8914 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.forward_residual_reference` |
| Baseline residual ratio | 0.9969 | 1.0039 | `runs/v2_baseline_off_axis/metrics_test.json (+ seeds)`; `mean.A.forward_residual_ratio` |
| +Fwd (fixed z) residual ratio | 0.9915 | 1.0068 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `mean.D1.forward_residual_ratio` |
| +Fwd (free z) residual ratio | 0.9928 | 1.0068 | `runs/v2_learned_z_off_axis/metrics_test.json`; `mean.D2.forward_residual_ratio` |
| +Fwd (fixed z) in-cell phase MAE | 0.3011 | 0.2893 | `runs/v2_forward_amplitude_off_axis/metrics_test.json`; `mean.D1.phase_mae_rad_in_cell` |
| +Fwd (free z) in-cell phase MAE | 0.2903 | 0.2886 | `runs/v2_learned_z_off_axis/metrics_test.json`; `mean.D2.phase_mae_rad_in_cell` |
| 0.9x probe, 32 validation fields, off-axis | -5e-5 (22 of 32 lower) | 0.00035 | `runs/z_calibration.json`; `disc.off_axis.margin.scaled_0.9` |
| Best phase scale, off-axis (median) | 0.68 (IQR 0.80) | 0.975 | `runs/z_calibration.json`; `scale.off_axis.best_scale_median` |
| 0.9x probe, test, global surface | -8e-5 | 0.00027 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_global.log`; `probe.global.phase_scaled_0.9.change` |
| 0.9x probe, test, per-field surfaces | -6e-4 | 0.0011 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_scaled_0.9.change` |
| Per-field-surface residual | 0.43 | 0.391 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_reference` |
| 0.5x probe, test, global | +0.003 | 0.0070 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_global.log`; `probe.global.phase_scaled_0.5.change` |
| 0.5x probe, test, per-field | +0.025 | 0.0359 | `logs/v2_20260930_065521_gpu2/amplitude_sensitivity_per_field.log`; `probe.per_field.phase_scaled_0.5.change` |
| Learned z after epoch 1 [um] | 34.12 | 34.10 | `runs/RESULTS.md`; `learnedz.initial_after_epoch1` |
| Learned z final [um] | 33.39 | 33.41 | `runs/RESULTS.md`; `learnedz.final` |
| Learned z excursion [um] | 0.958 | 0.884 | `runs/RESULTS.md`; `learnedz.excursion` |
| Learned z last-10 spread [um] | 0.026 | 0.018 | `runs/RESULTS.md`; `learnedz.last10_spread` |
| Learned z at selected checkpoint [um] | 33.35 | 33.41 | `runs/v2_learned_z_off_axis/metrics_test.json`; `learnedz.scored` |
| z scan IQR off-axis [um] | 156.4 | 48.1 | `runs/z_calibration.json`; `z.off_axis.per_image_iqr_um` |
| z scan in-line median [um] | 50.5 | 33.68 | `runs/z_calibration.json`; `z.gabor.per_image_median_z_um` |
| z scan IQR in-line [um] | 6.0 | 0.0 | `runs/z_calibration.json`; `z.gabor.per_image_iqr_um` |
| Gradient ratio median | 0.302 | 0.373 | `runs/gradient_path_b_cell_ipp_512.csv`; `grad.median` |
| Gradient ratio range | 0.18-0.67 | 0.243–0.655 | `runs/gradient_path_b_cell_ipp_512.csv`; `grad.min` |
| Baseline latency PyTorch FP32 [ms] | 12.32 ± 0.11 | 12.31 ± 0.12 | `runs/benchmark_results/results_hardware_arm_A.json`; `hw.A.mean.pytorch_fp32.latency_mean_ms` |
| Baseline latency ONNX FP16 [ms] | 8.42 ± 0.06 | 8.42 ± 0.07 | `runs/benchmark_results/results_hardware_arm_A.json`; `hw.A.mean.onnx_fp16.latency_mean_ms` |
| Compact latency ONNX FP16 [ms] | 7.98 | 7.97 | `runs/benchmark_results/results_hardware_arm_KA.json`; `hw.KA.mean.onnx_fp16.latency_mean_ms` |
| Membrane Dice (baseline) | 0.496 | 0.502 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_dice` |
| Membrane AJI | 0.235 | 0.236 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_aji` |
| Membrane boundary F1 | 0.062 | 0.064 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.seg_boundary_f1` |
| Membrane recall | 0.107 | 0.114 | `runs/benchmark_results/results_arm_A_membrane.json`; `membrane.A.mean.detection_recall` |
