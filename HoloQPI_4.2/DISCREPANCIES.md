# DISCREPANCIES — HoloQPI_4.2

> **Status note.** Written during the first synchronisation pass. It is superseded by `analysis/v42/AUDIT.md`, `analysis/v42/SYNC_AUDIT.md` and `analysis/v42/MANUSCRIPT_SYNC.md`. Items 1-15 remain a record of where file values differ from earlier expectations (the gradient ratio in particular: the CSV gives median 0.373, range 0.243-0.655). The registration archive is still not in the repository.


This file lists three kinds of item:
- places where the corrected result files disagree with the task prompt's "expected" values;
- sources the task asked for that are not in the project folder;
- figures that could not be regenerated here.

In every case the manuscript uses the file value, or carries a `% TODO-SOURCE` comment.

## 1. File values that differ from the prompt

| # | Item | Prompt / expected | File value used | Source |
|---|---|---|---|---|
| 1 | Gradient ratio of the per-cell IPP term at w = 1 | median 0.302 (range 0.18–0.67) | median **0.373** (range **0.243–0.655**), 30 batches | `runs/gradient_path_b_cell_ipp_512.csv`, column `ratio_at_weight_1`; same in `logs/v2_20260930_065521_gpu2/gradient_path.log` |
| 2 | Off-axis residual at the reference phase | "about 0.915" | **0.9147** on 32 *validation* fields (border 79 px) and **0.8914** on the 113 *test* fields | `runs/z_calibration.json` `off_axis.discrimination.floor`; `metrics_test.json` `forward_residual_reference`; `amplitude_sensitivity_global.log` |
| 3 | In-line residual at the reference phase | "about 0.70" | **0.6993** on 32 validation fields; **0.7654** on the 107 in-line test fields | as above, `gabor.*`; `runs/v2_baseline_gabor/metrics_test.json` |
| 4 | +Fwd residual | "do not assume they still lower the residual" | Both +Fwd configurations have ratio **1.0068**, above the baseline's **1.0039 ± 0.0002**. The v4.1 sentence saying they "reach lower residual ratios" is reversed. | `runs/v2_forward_amplitude_off_axis`, `runs/v2_learned_z_off_axis`, `metrics_test.json` |
| 5 | Off-axis z scan | "median 33.7 µm, IQR about 48 µm" | Median **33.68**, IQR **48.1**. The four per-field minima are 33.68, 33.68, −96.24 and +96.24 µm; the last two are the scan limits. The pooled minimum is also at 33.68. The text states all of this. | `runs/z_calibration.json` `off_axis.curve.per_image_best_z_um` |
| 6 | IPP decomposition: "no change in the domain or phase factors was resolved for all matched cells" | — | True for the geometric-mean factors. The domain \|log\| **spread** over all matched cells is resolved for +IPP (image) (+0.0025, 2×SD 0.0015). The text says so explicitly. | `runs/diagnostics/decomposition_summary.csv`, `analysis/v42/values.json` `dcmp.B1_vs_A.cell.all.domain_med_abs_log` |
| 7 | +IPP (per-cell) field-level foreground/detection | mention only if resolved | **Resolved**, so it is mentioned: field-level domain factor +0.0118 (2×SD 0.0081); recall +0.0075 (0.0061); false positives +119/run (72); near-boundary false positives +67 (39). | `values.json` `dcmp.B_vs_A.field.all.domain_gm`, `cmp.B_vs_A.*`, `lcmp.B_vs_A.*` |
| 8 | Benchmark values "expected within 0.02 ms" of v4.1 | — | The corrected benchmark sessions differ by more than 0.02 ms in several cells (see the next table). The new values are used. | `runs/benchmark_results/results_hardware_arm_{A,D0,KA}.json` |
| 9 | Expected numbers in the corrected-results summary (project note) | e.g. B vs A phase-spread verdicts | Recomputed with the repository rule. Only these phase/domain spread changes are resolved: B1 interior phase spread, B1 all and interior domain spread, and B1 edge phase spread and edge phase GM. | `values.json` `dcmp.*` |
| 10 | Synthetic field-total bias | three-set mean ≈ −2.59 % | **−2.592 ± 0.889 %** (mean of three sets). `RESULTS.md` quotes the last set only (−1.67 %). | `runs/benchmark_results/results_synthetic_validation.json` |
| 11 | Learned z | about 33.41 µm | 34.103 after epoch 1, 33.407 final; selected checkpoint epoch **44**, scored at **33.408 µm**. v4.1 gave 33.35 µm at the checkpoint. | `runs/RESULTS.md`, `results_arm_D2.json`, `v2_learned_z_off_axis/metrics_test.json` |
| 12 | Crop-origin source numbers | "+5.5 px (dy), +1.6 px (dx) for the exact centre crop" (test archive) | The test-field archive is missing (Sec. 2). The text uses the **validation** measurement that chose the origin: **+5.4 / +1.1 px** on 127 validation fields. Test confirmation at (−6, −2): dy −0.10 px (5–95 % −1.88 to +1.64), dx 0.00 px (−3.06 to +2.12). | `logs/run_everything_20260930_042722.txt`; `logs/corrected_20260930_054214/crop_check_after.txt` |
| 13 | Win rates in the probes | 0.9×: off-axis 0.56, in-line 0.94 | Given as field counts (18/32, 30/32, 20/32, 32/32). This avoids a rounding clash between 0.625 → "0.62" and "63 %". | `runs/z_calibration.json` `discrimination.win_rates` × `images` |
| 14 | In-line neural residual | not in the prompt | The In-Line Neural Configuration's residual (0.7573) is **below** that of the reference phase (0.7654), ratio 0.989. It is reported factually in Sec. 5.5. | `runs/v2_baseline_gabor/metrics_test.json` |
| 15 | w = 1.0 member of the weight sweep | v4.1: "trained once, it is +IPP (per-cell)" | A separate run `v2_ipp_w10_off_axis` exists. Its resolved config is identical to +IPP (per-cell) seed 42 apart from the name. It gave 0.1872 against 0.1875 for the B seed-42 run. Reported as a repeat-run note. The sweep and tables use +IPP (per-cell) (n = 3) for w = 1.0, as in the figure. | `runs/v2_ipp_w10_off_axis/resolved_config.yaml`, `metrics_test.json` |

**Benchmark changes (latency_mean_ms; v4.1 → 4.2):**

| Configuration | Runtime | v4.1 | 4.2 |
|---|---|---|---|
| Baseline | PyTorch FP32 | 12.32 ± 0.11 | 12.31 ± 0.12 |
| Baseline | ONNX FP32 | 14.20 ± 0.10 | 14.21 ± 0.08 |
| Baseline | PyTorch FP16 | 9.26 ± 0.16 | 9.19 ± 0.10 |
| Baseline | ONNX FP16 | 8.42 ± 0.06 | 8.42 ± 0.07 |
| +Amplitude | PyTorch FP16 | 9.60 | 11.26 |
| +Amplitude | ONNX FP16 | 9.20 | 9.18 |
| +Amplitude | ONNX FP16 p99/p50 | 1.15 | 1.01 |
| Compact | ONNX FP16 | 7.98 | 7.97 |

The top-level `runs/v2_*_hardware_benchmark_cuda.json` files come from an earlier benchmark stage, so `RESULTS.md` Table 4 differs from the collector. The manuscript uses the collector's `results_hardware_arm_*.json`.

## 2. Sources (updated 2026-10-01, second pass)

No `TODO-SOURCE` comment remains in main.tex.

1. **Registration archive** (`diagnostics/registration_summary.json`, `hologram_registration.csv`; band-pass variant unless stated). The values in Sec. 4.1 were supplied by the author from this archive:
   - matched pairs ≤1 px on 734/800 fields, control pairs on 3/800;
   - AUC 0.975 by correlation, 0.973 by registration error;
   - the 750 paired fields all within 4.7 px, 734 within 1 px, r 0.47–1.00, against a highest control value of 0.25;
   - SNU_01–50 r −0.13 to 0.18;
   - low-pass-only AUC 0.506.

   The archive is still not in the project folder, so `values.json` records these values with the provenance "author-supplied" (`reg.*`), and `check_numbers.py` matches them against those entries.

   **Flag:** the range for SNU_01–50 (−0.13 to 0.18) differs from the earlier diagnostics report (`Claude outputs/diagnostics_response.md`) and the `config/base.yaml` comment, which both give 0.09–0.18. The archive value was used as instructed. The two may describe different statistics, for example the minimum over all 50 fields against a summary range. Check this against `hologram_registration.csv`.
2. **Park et al. 2026.** The author verified these against the Supplementary Information and the main text:
   - S2: 500-iteration Gerchberg–Saxton;
   - S6: two segmentation strategies;
   - Table S4: parameters and latency;
   - main text: 256×256 inputs.

   The three TODO-SOURCE comments were deleted. The description of agreement in the Introduction and in Sec. 2.4 was changed to the author's wording.
3. **Cuche et al. 1999** was added to `references.bib` from bibliographic knowledge (Appl. Opt. 38(34):6994–7001, doi 10.1364/AO.38.006994). Verify the entry.
4. **Edge/interior median areas** (v4.1: 118.5 vs 411.4 µm²) could not be recomputed, because there are no per-instance bounding boxes for unmatched reference cells. The sentence was removed.
5. **Per-field aberration probe (Discussion 6.5, Table 11c).**
   - Computed from the per-field rows of `runs/amplitude_sensitivity_off_axis_global.csv` restricted to the batches present in `..._per_field.csv`.
   - Global surface on the same 112 fields: residual 0.8926; 0.9× change +0.00025; 0.5× change +0.0067.
   - The missing field is test index 99, T24_Staurosporine_100nM_10. Its per-field fit was rejected in `runs/aberration_fit_order5.csv` ("surface 405 rad exceeds 45 rad").

## 3. Figures

- **Figure 1:** rebuilt from its editable source (`figure_code/fig1/build_fig1.js`).
  - "Optical volume" → "Integrated phase Sₖ".
  - Vₖ → Sₖ.
  - Cₖ formula made to match the implementation.
  - "output / physics" → "output / forward model".
  - "bottleneck" → "projection".
  - Em dash removed.
  - "drug-condition classifier" note removed.
  - In-line text corrected.
  The panel images are the server-generated corrected-crop panels (`fig1_panels.json` records the crop offset [−6, −2]). These panels were **not** regenerated here.
- **Figure 3:** server-generated with the corrected crop; verified, not regenerated here (needs checkpoints). The field MAE labels (0.248, 0.296 rad) come from `figure_3.py`. `fig1_panels.json` independently gives 0.2959 rad for NCI_08.
- **Figures 2, 4, 5, 6 and 9:** regenerated earlier from the corrected runs by `paper/manuscript_images/figure_*.py`; verified against `values.json` here and not regenerated again.
- **The repository scripts render no decomposition figure.** Table 7 carries the decomposition, so no figure was added.
