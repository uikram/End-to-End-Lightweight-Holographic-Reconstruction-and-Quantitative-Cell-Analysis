# MANUSCRIPT_SYNC — what the manuscript must change, table by table

The manuscript was **not** edited. All numbers below are generated from `results_for_manuscript/` by `analysis/v42/make_sync.py` (units: fractions/rad as in the tables; field sets in the headings; training runs = n; every value is directly sourced unless marked derived in the JSON). Current printed numbers in `table_4…table_11.tex` already agree with these values at their printed precision; the changes needed are mostly wording/structure:

## Required changes (wording / structure)
| Where | Change | Reason / source |
|---|---|---|
| Table 3 | none (training-run counts 3/1 verified); optionally add a "Comparator" check against `metadata/configurations.json` | AUDIT §1 |
| Table 4, "Cells matched" Δ | replace `--` by **+20 ± 9**, labelled descriptive, not a resolution test | `baseline.json::table_4.cells_matched` |
| Table 5, "Recovered in-cell phase contrast" | fill the neural columns: in-line neural (1 training run) **+0.887 rad**, off-axis neural (3 training runs) **+0.899 ± 0.036 rad** (classical +0.922 / −0.084 already present). Report as recovered contrast only; reference median is +1.097 rad. | `inline.json::table_5.recovered_in_cell_phase_contrast`; `runs/common_fields/neural_phase_contrast.json` (classical re-check reproduced 0.921582 exactly) |
| Table 5 header | keep "n = training runs"; do not use n for fields/cells | — |
| Table 6 | circularity field-total bias/MAPE/r: `--` → **N/A** (circularity is not additive) | `baseline.json::table_6` |
| Table 7 | header `n` (matched cells/fields) → "N"; group labels `($n=3$)` → "(3 training runs)", classical "(1 deterministic run)" | AUDIT §3 |
| Table 8, 9 | same `n` → "training runs" / "N" separation | — |
| Table 10 | column "Verdict" → "Assessment"; "resolved" → "Exceeds 2× pooled SD", "not resolved" → "Within 2× pooled SD", "not resolvable" → "Not estimable from the available runs"; caption/tabnote likewise | `ipp.json::table_10[*].assessment` |
| Table 11c | replace `--` in "Fields worse" by counts; rename column **"Fields favouring reference phase"** (scaled phase gives the higher residual); counts: see Table 11c below | `amplitude.json::table_11c` |
| Table 11 tabnote | keep "Configured tolerance … 0.01" | `config/base.yaml` |
| §4.8 (main.tex l. 884) | "not a significance test" → e.g. "a separation criterion; no hypothesis test is performed" (word on the avoid list) | style |
| §4.1 registration paragraph | statistics are author-supplied (archive absent); SNU_01–50 r range −0.13…0.18 vs 0.09…0.18 in `config/base.yaml`: check `hologram_registration.csv` | AUDIT §6 |
| §3 / gradient ratio | already 0.373 (0.243–0.655); do **not** change to 0.302 / 0.18–0.67 | `runs/gradient_path_b_cell_ipp_512.csv` |
| Results text paragraphs | numeric values unchanged; add the neural contrast values above where Table 5 is discussed | — |

## Paragraph-level numeric sources (old concept → value → source)
* Forward-model z: recording 33.77 µm; in-line grid minimum 33.684 µm; free-z 34.10 → 33.41 µm (checkpoint epoch 44 scored 33.408 µm); off-axis per-field best z −96.24, +96.24, 33.68, 33.68 µm → `forward_model.json`. No 50.5 µm statement.
* Residual at the reference phase: validation 0.9147 (off-axis), 0.6993 (in-line); test 0.8914 (113 fields); +Fwd ratios 1.0068 vs baseline 1.0039 ± 0.0002 → `amplitude.json`, `forward_model.json`.
* Common 107-field comparison: in-line neural n = 1, off-axis neural n = 3 (same models re-scored), classical off-axis and in-line deterministic; 3055 reference cells → `inline.json`.
* Seeds/runs statement: three training runs for baseline, +IPP (per-cell), +IPP (image); one for all others including the In-Line Neural Configuration.



### Table 3 — configurations (training runs, seeds, comparator)

| Configuration | Training runs | Seeds | Comparator | Loss weights (non-zero) | Source |
|---|---|---|---|---|---|
| End-to-End Neural Baseline | 3 | 42, 1337, 2024 | -- | phase=1, segmentation=1 | `metadata/configurations.json` |
| In-Line Neural Configuration | 1 | 42 | End-to-End Neural Baseline | phase=1, segmentation=1 | `metadata/configurations.json` |
| +IPP (per-cell) | 3 | 42, 1337, 2024 | End-to-End Neural Baseline | phase=1, segmentation=1, cell_integrated_phase=1 | `metadata/configurations.json` |
| +IPP (image) | 3 | 42, 1337, 2024 | End-to-End Neural Baseline | phase=1, segmentation=1, phase_volume=1 | `metadata/configurations.json` |
| +Area | 1 | 42 | +IPP (per-cell) | phase=1, segmentation=1, cell_integrated_phase=1, cell_projected_area=1 | `metadata/configurations.json` |
| +BGA | 1 | 42 | +IPP (per-cell) | phase=1, segmentation=1, boundary_gradient_alignment=0.05, cell_integrated_phase=1 | `metadata/configurations.json` |
| +IPP (per-cell), w=0.1 | 1 | 42 | End-to-End Neural Baseline | phase=1, segmentation=1, cell_integrated_phase=0.1 | `metadata/configurations.json` |
| +IPP (per-cell), w=0.3 | 1 | 42 | End-to-End Neural Baseline | phase=1, segmentation=1, cell_integrated_phase=0.3 | `metadata/configurations.json` |
| +IPP (per-cell), w=3.0 | 1 | 42 | End-to-End Neural Baseline | phase=1, segmentation=1, cell_integrated_phase=3 | `metadata/configurations.json` |
| +Amplitude | 1 | 42 | +IPP (per-cell) | phase=1, segmentation=1, cell_integrated_phase=1, amplitude=0.1 | `metadata/configurations.json` |
| +Fwd (fixed z) | 1 | 42 | +Amplitude | phase=1, segmentation=1, cell_integrated_phase=1, amplitude=0.1, forward_model=0.02 | `metadata/configurations.json` |
| +Fwd (free z) | 1 | 42 | +Fwd (fixed z) | phase=1, segmentation=1, cell_integrated_phase=1, amplitude=0.1, forward_model=0.02 | `metadata/configurations.json` |
| Compact Baseline | 1 | 42 | End-to-End Neural Baseline | phase=1, segmentation=1 | `metadata/configurations.json` |
| Compact +IPP | 1 | 42 | +IPP (per-cell) | phase=1, segmentation=1, cell_integrated_phase=1 | `metadata/configurations.json` |

### Table 4 — Baseline (mean ± SD, 3 seeds) vs classical off-axis; 113 test fields, 3186 reference cells

| Metric key | Classical (n=1, deterministic) | Neural (3 training runs) | Δ neural − classical | Note |
|---|---|---|---|---|
| phase_mae_rad | 0.1968 | 0.1568 ± 0.0023 | -0.0400 | no resolution test (deterministic comparator) |
| phase_mae_rad_in_cell | 0.3112 | 0.2689 ± 0.0056 | -0.0423 | no resolution test (deterministic comparator) |
| phase_pearson_r | 0.7851 | 0.8741 ± 0.0023 | 0.0890 | no resolution test (deterministic comparator) |
| phase_ssim | 0.5299 | 0.6386 ± 0.0053 | 0.1087 | no resolution test (deterministic comparator) |
| seg_dice | 0.8253 | 0.8337 ± 0.0013 | 0.0085 | no resolution test (deterministic comparator) |
| seg_aji | 0.6387 | 0.6710 ± 0.0025 | 0.0322 | no resolution test (deterministic comparator) |
| seg_boundary_f1 | 0.3954 | 0.4687 ± 0.0053 | 0.0733 | no resolution test (deterministic comparator) |
| detection_recall | 0.5869 | 0.5931 ± 0.0027 | 0.0062 | no resolution test (deterministic comparator) |
| detection_precision | 0.7910 | 0.8488 ± 0.0124 | 0.0577 | no resolution test (deterministic comparator) |
| area_mape | 0.1578 | 0.1568 ± 0.0027 | -0.0010 | no resolution test (deterministic comparator) |
| dry_mass_mape | 0.2203 | 0.1742 ± 0.0045 | -0.0461 | no resolution test (deterministic comparator) |
| dry_mass_cell_pearson_r | 0.8937 | 0.9363 ± 0.0007 | 0.0427 | no resolution test (deterministic comparator) |
| dry_mass_field_total_mape | 0.1094 | 0.1631 ± 0.0141 | 0.0537 | no resolution test (deterministic comparator) |
| cells_matched | 1870 | 1890 ± 8.5049 | +20 ± 9 | descriptive only (not a resolution test) |

Source for every row: `baseline/baseline.json::table_4.<key>` (neural: `runs/v2_{baseline,A_seed1337,A_seed2024}*/metrics_test.json`; classical: `runs/conventional_off_axis/metrics_test.json`).

### Table 5 — 107 common fields, 3055 reference cells (n = training runs: in-line neural 1, off-axis neural 3, classical deterministic)

| Metric key | In-line neural (n=1, seed 42) | Off-axis neural (n=3) | Classical off-axis | Classical in-line |
|---|---|---|---|---|
| phase_mae_rad | 0.1625 | 0.1586 ± 0.0024 | 0.1990 | 0.3996 |
| phase_mae_rad_in_cell | 0.3005 | 0.2726 ± 0.0057 | 0.3157 | 1.0448 |
| phase_pearson_r | 0.8667 | 0.8745 ± 0.0023 | 0.7849 | -0.2075 |
| phase_ssim | 0.6296 | 0.6377 ± 0.0053 | 0.5297 | 0.0639 |
| seg_dice | 0.8148 | 0.8330 ± 0.0013 | 0.8240 | 0.1583 |
| seg_aji | 0.6399 | 0.6697 ± 0.0031 | 0.6355 | 0.0549 |
| seg_boundary_f1 | 0.4165 | 0.4706 ± 0.0052 | 0.3943 | 0.1087 |
| detection_recall | 0.5885 | 0.5936 ± 0.0031 | 0.5849 | 0.0347 |
| detection_precision | 0.8406 | 0.8487 ± 0.0140 | 0.7883 | 0.0077 |
| area_mape | 0.1669 | 0.1552 ± 0.0025 | 0.1576 | 0.5315 |
| dry_mass_mape | 0.2129 | 0.1740 ± 0.0045 | 0.2204 | 0.6942 |
| dry_mass_field_total_mape | 0.1356 | 0.1657 ± 0.0145 | 0.1101 | 0.6602 |
| dry_mass_cell_pearson_r | 0.9029 | 0.9374 ± 0.0010 | 0.8940 | 0.8758 |
| cells_matched | 1798 | 1813 ± 9.5044 | 1787 | 106 |
| recovered in-cell phase contrast [rad] | 0.8869 | 0.8992 ± 0.0362 | 0.922 | -0.084 |

### Table 6 — measurement agreement (baseline, 3 seeds; matched cells vs field totals); circularity field totals are N/A

| Quantity | Matched MAPE | Cov.-adj. MAPE | Pearson r (cells) | BA bias (per field) | LoA (mean over seeds) | Field-total bias | Field-total MAPE | Field-total r |
|---|---|---|---|---|---|---|---|---|
| Projected area | 0.1568 ± 0.0027 | 0.4999 ± 0.0036 | 0.8986 ± 0.0011 | 0.0649 ± 0.0028 | [-0.162, +0.292] | -0.0901 ± 0.0037 | 0.1553 ± 0.0047 | 0.9341 ± 0.0043 |
| Circularity | 0.0571 ± 0.0027 | 0.4408 ± 0.0039 | 0.8861 ± 0.0072 | 0.0217 ± 0.0057 | [-0.054, +0.097] | N/A | N/A | N/A |
| Dry mass | 0.1742 ± 0.0045 | 0.5102 ± 0.0041 | 0.9363 ± 0.0007 | -0.0372 ± 0.0172 | [-0.247, +0.172] | -0.1560 ± 0.0174 | 0.1631 ± 0.0141 | 0.9883 ± 0.0007 |

### Table 7 — mass-error decomposition (columns: training runs / N matched cells (or fields) kept apart)

| Config | Training runs | Level.group | N (matched cells or fields) | Recall | Domain GM | Phase GM | Total GM | |log| domain | |log| phase | |log| total |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | cell.all | 1889 | 0.593 ± 0.002 | 1.002 ± 0.002 | 0.929 ± 0.016 | 0.931 ± 0.014 | 0.042 | 0.102 | 0.132 |
| A | 3 | cell.interior | 1581 | 0.912 ± 0.002 | 1.006 ± 0.002 | 0.957 ± 0.017 | 0.963 ± 0.015 | 0.038 | 0.090 | 0.116 |
| A | 3 | cell.edge | 308 | 0.212 ± 0.003 | 0.982 ± 0.004 | 0.797 ± 0.014 | 0.782 ± 0.014 | 0.076 | 0.267 | 0.303 |
| A | 3 | field.all | 113 | -- | 0.897 ± 0.002 | 0.933 ± 0.021 | 0.838 ± 0.017 | 0.105 | 0.073 | 0.179 |
| B | 3 | cell.all | 1913 | 0.601 ± 0.003 | 1.010 ± 0.006 | 0.926 ± 0.015 | 0.935 ± 0.010 | 0.043 | 0.118 | 0.145 |
| B | 3 | cell.interior | 1594 | 0.919 ± 0.002 | 1.012 ± 0.005 | 0.949 ± 0.018 | 0.961 ± 0.014 | 0.039 | 0.104 | 0.130 |
| B | 3 | cell.edge | 320 | 0.220 ± 0.005 | 0.997 ± 0.008 | 0.819 ± 0.012 | 0.817 ± 0.010 | 0.081 | 0.253 | 0.306 |
| B | 3 | field.all | 113 | -- | 0.909 ± 0.005 | 0.929 ± 0.017 | 0.845 ± 0.015 | 0.096 | 0.081 | 0.176 |
| B1 | 3 | cell.all | 1875 | 0.589 ± 0.001 | 1.000 ± 0.002 | 0.938 ± 0.013 | 0.938 ± 0.012 | 0.044 | 0.118 | 0.145 |
| B1 | 3 | cell.interior | 1573 | 0.907 ± 0.003 | 1.004 ± 0.002 | 0.959 ± 0.013 | 0.964 ± 0.010 | 0.041 | 0.107 | 0.131 |
| B1 | 3 | cell.edge | 302 | 0.208 ± 0.000 | 0.979 ± 0.002 | 0.835 ± 0.021 | 0.817 ± 0.022 | 0.074 | 0.232 | 0.297 |
| B1 | 3 | field.all | 113 | -- | 0.894 ± 0.002 | 0.925 ± 0.012 | 0.827 ± 0.008 | 0.110 | 0.086 | 0.181 |
| classical | 1 | cell.all | 1870 | 0.587 | 0.966 | 1.014 | 0.980 | 0.046 | 0.119 | 0.169 |
| classical | 1 | cell.interior | 1543 | 0.890 | 0.977 | 1.028 | 1.004 | 0.041 | 0.106 | 0.152 |
| classical | 1 | cell.edge | 327 | 0.225 | 0.919 | 0.952 | 0.875 | 0.095 | 0.212 | 0.308 |
| classical | 1 | field.all | 113 | -- | 0.857 | 1.055 | 0.905 | 0.141 | 0.061 | 0.095 |

Source: `decomposition/decomposition.json::by_config` (recomputed from `runs/diagnostics/decomposition_{cells,fields}.csv`; agrees with `decomposition_summary.csv`). Derived.

### Table 8 — edge / interior stratification, false positives (all, within 15 px of the border)

| Configuration | Training runs | Edge recall | Interior recall | Edge dry-mass MAPE | Interior dry-mass MAPE | FP all | FP near boundary |
|---|---|---|---|---|---|---|---|
| End-to-End Neural Baseline | 3 | 0.212 ± 0.003 | 0.912 ± 0.002 | 0.305 ± 0.010 | 0.149 ± 0.004 | 337 ± 32 | 136 ± 16 |
| +IPP (per-cell) | 3 | 0.220 ± 0.005 | 0.919 ± 0.002 | 0.315 ± 0.012 | 0.163 ± 0.003 | 456 ± 40 | 203 ± 22 |
| +IPP (image) | 3 | 0.208 ± 0.000 | 0.907 ± 0.003 | 0.299 ± 0.009 | 0.167 ± 0.003 | 342 ± 27 | 136 ± 14 |
| Classical pipeline | 1 | 0.225 | 0.890 | 0.322 | 0.199 | 494 | 137 |

### Table 9 — ablation (mean ± SD for 3 training runs; point estimate, no SD, for 1 run)

| Configuration | Training runs | Matched MAPE | Cov.-adj. MAPE | Field-total MAPE | Area MAPE | Recall | Precision | Dice | Phase MAE [rad] |
|---|---|---|---|---|---|---|---|---|---|
| End-to-End Neural Baseline | 3 | 0.1742 ± 0.0045 | 0.5102 ± 0.0041 | 0.1631 ± 0.0141 | 0.1568 ± 0.0027 | 0.5931 ± 0.0027 | 0.8488 ± 0.0124 | 0.8337 ± 0.0013 | 0.1568 ± 0.0023 |
| +IPP (per-cell) | 3 | 0.1883 ± 0.0052 | 0.5125 ± 0.0008 | 0.1676 ± 0.0084 | 0.1679 ± 0.0059 | 0.6006 ± 0.0034 | 0.8077 ± 0.0138 | 0.8294 ± 0.0032 | 0.1656 ± 0.0032 |
| +IPP (image) | 3 | 0.1887 ± 0.0013 | 0.5222 ± 0.0008 | 0.1722 ± 0.0107 | 0.1597 ± 0.0014 | 0.5889 ± 0.0013 | 0.8460 ± 0.0100 | 0.8294 ± 0.0004 | 0.1644 ± 0.0015 |
| +Area | 1 | 0.1850 | 0.5075 | 0.1696 | 0.1779 | 0.6042 | 0.7958 | 0.8294 | 0.1646 |
| +BGA | 1 | 0.1921 | 0.5182 | 0.1651 | 0.1652 | 0.5964 | 0.8183 | 0.8309 | 0.1654 |
| +IPP (per-cell), w=0.1 | 1 | 0.1754 | 0.5158 | 0.1830 | 0.1504 | 0.5873 | 0.8428 | 0.8331 | 0.1567 |
| +IPP (per-cell), w=0.3 | 1 | 0.1881 | 0.5196 | 0.1953 | 0.1564 | 0.5917 | 0.8389 | 0.8328 | 0.1620 |
| +IPP (per-cell), w=3.0 | 1 | 0.1963 | 0.5197 | 0.1316 | 0.1729 | 0.5976 | 0.7855 | 0.8164 | 0.1813 |
| +Amplitude | 1 | 0.1857 | 0.5157 | 0.1528 | 0.1591 | 0.5948 | 0.8282 | 0.8297 | 0.1633 |
| +Fwd (fixed z) | 1 | 0.1855 | 0.5081 | 0.1573 | 0.1685 | 0.6039 | 0.8297 | 0.8318 | 0.1647 |
| +Fwd (free z) | 1 | 0.1920 | 0.5212 | 0.1599 | 0.1583 | 0.5926 | 0.8248 | 0.8290 | 0.1636 |
| Compact Baseline | 1 | 0.1740 | 0.5123 | 0.1738 | 0.1508 | 0.5904 | 0.8496 | 0.8345 | 0.1562 |
| Compact +IPP | 1 | 0.1887 | 0.5088 | 0.1539 | 0.1656 | 0.6055 | 0.8174 | 0.8306 | 0.1652 |

### Table 10 — resolution quantities (n/e = not estimable)

| Configuration | Comparator | Metric | Δ (config − comparator) | Pooled SD | 2× pooled SD | Runs (config/comparator) | Evaluable | Assessment |
|---|---|---|---|---|---|---|---|---|
| +IPP (per-cell) | End-to-End Neural Baseline | dry_mass_mape | +0.0141 | 0.0048 | 0.0097 | 3/3 | True | Exceeds 2x pooled SD |
| +IPP (per-cell) | End-to-End Neural Baseline | dry_mass_field_total_mape | +0.0045 | 0.0116 | 0.0232 | 3/3 | True | Within 2x pooled SD |
| +IPP (image) | End-to-End Neural Baseline | dry_mass_mape | +0.0145 | 0.0033 | 0.0066 | 3/3 | True | Exceeds 2x pooled SD |
| +IPP (image) | End-to-End Neural Baseline | dry_mass_field_total_mape | +0.0091 | 0.0125 | 0.0250 | 3/3 | True | Within 2x pooled SD |
| +Area | +IPP (per-cell) | dry_mass_mape | -0.0033 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +Area | +IPP (per-cell) | dry_mass_field_total_mape | +0.0020 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +BGA | +IPP (per-cell) | dry_mass_mape | +0.0038 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +BGA | +IPP (per-cell) | dry_mass_field_total_mape | -0.0025 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=0.1 | End-to-End Neural Baseline | dry_mass_mape | +0.0012 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=0.1 | End-to-End Neural Baseline | dry_mass_field_total_mape | +0.0199 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=0.3 | End-to-End Neural Baseline | dry_mass_mape | +0.0138 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=0.3 | End-to-End Neural Baseline | dry_mass_field_total_mape | +0.0322 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=3.0 | End-to-End Neural Baseline | dry_mass_mape | +0.0221 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (per-cell), w=3.0 | End-to-End Neural Baseline | dry_mass_field_total_mape | -0.0315 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +Amplitude | +IPP (per-cell) | dry_mass_mape | -0.0026 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +Amplitude | +IPP (per-cell) | dry_mass_field_total_mape | -0.0148 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +Fwd (fixed z) | +Amplitude | dry_mass_mape | -0.0002 | n/e | n/e | 1/1 | False | Not estimable from the available runs |
| +Fwd (fixed z) | +Amplitude | dry_mass_field_total_mape | +0.0046 | n/e | n/e | 1/1 | False | Not estimable from the available runs |
| +Fwd (free z) | +Fwd (fixed z) | dry_mass_mape | +0.0065 | n/e | n/e | 1/1 | False | Not estimable from the available runs |
| +Fwd (free z) | +Fwd (fixed z) | dry_mass_field_total_mape | +0.0025 | n/e | n/e | 1/1 | False | Not estimable from the available runs |
| Compact Baseline | End-to-End Neural Baseline | dry_mass_mape | -0.0002 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| Compact Baseline | End-to-End Neural Baseline | dry_mass_field_total_mape | +0.0107 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| Compact +IPP | +IPP (per-cell) | dry_mass_mape | +0.0003 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| Compact +IPP | +IPP (per-cell) | dry_mass_field_total_mape | -0.0137 | n/e | n/e | 1/3 | False | Not estimable from the available runs |
| +IPP (image) | +IPP (per-cell) | dry_mass_mape | +0.0004 | 0.0038 | 0.0076 | 3/3 | True | Within 2x pooled SD |
| +IPP (image) | +IPP (per-cell) | dry_mass_field_total_mape | +0.0046 | 0.0096 | 0.0193 | 3/3 | True | Within 2x pooled SD |

### Table 11a — amplitude output (1 training run each, 113 test fields)

| Quantity | +Amplitude | +Fwd (fixed z) | +Fwd (free z) |
|---|---|---|---|
| amplitude_mae | 0.0651 | 0.0665 | 0.0651 |
| amplitude_unity_mae | 0.1469 | 0.1469 | 0.1469 |
| ratio_mae_over_unity_mae | 0.4429 | 0.4526 | 0.4436 |
| amplitude_mae_in_cell | 0.0906 | 0.0922 | 0.0898 |
| amplitude_bias | 0.0339 | 0.0378 | 0.0323 |
| amplitude_pearson_r | 0.9047 | 0.9064 | 0.9045 |
| amplitude_pred_mean | 1.0324 | 1.0363 | 1.0308 |
| amplitude_pred_sd | 0.1207 | 0.1223 | 0.1200 |
| amplitude_reference_mean | 0.9985 | 0.9985 | 0.9985 |

### Table 11b — forward-model residual (A, B: mean ± SD of 3 runs)

| Quantity | Baseline | +IPP (per-cell) | +Amplitude | +Fwd (fixed z) | +Fwd (free z) |
|---|---|---|---|---|---|
| forward_residual | 0.8948 ± 0.0002 | 0.8968 ± 0.0005 | 0.8699 | 0.8702 | 0.8699 |
| forward_residual_reference | 0.8914 ± 0.0000 | 0.8914 ± 0.0000 | 0.8647 | 0.8643 | 0.8640 |
| forward_residual_ratio | 1.0039 ± 0.0002 | 1.0061 ± 0.0005 | 1.0061 | 1.0068 | 1.0068 |
| phase_mae_rad_in_cell | 0.2689 ± 0.0056 | 0.2876 ± 0.0064 | 0.2862 | 0.2893 | 0.2886 |

### Table 11c — phase-scale probes; 'Fields favouring reference phase' = fields where the scaled phase gives the HIGHER residual

| Geometry and fields | Residual (reference phase) | Margin 0.9× | Fields favouring reference 0.9× | Margin 0.5× | Fields favouring reference 0.5× |
|---|---|---|---|---|---|
| Off-axis, 32 validation fields | 0.9147 | 0.00035 | 18/32 | 0.0080 | 20/32 |
| In-line, 32 validation fields | 0.6993 | 0.00432 | 30/32 | 0.0622 | 32/32 |
| Off-axis, 113 test fields, global surface | 0.8914 | 0.00027 | 60/113 | 0.0070 | 71/113 |
| Off-axis, 112 test fields, per-field surfaces | 0.3906 | 0.00110 | 93/112 | 0.0359 | 110/112 |
| Off-axis, 112 test fields, global surface (same fields as per-field row) | 0.8926 | 0.00025 | 59/112 | 0.0067 | 70/112 |
