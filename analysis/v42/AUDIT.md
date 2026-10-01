# AUDIT — HoloQPI 4.2 code / configs / results / manuscript

Branch `bug_fixes/001` (worked on as `claude/admiring-maxwell-xwtq43`). Scope: audit and synchronise; the manuscript (`HoloQPI_4.2/`) was **not edited**.
Everything below was checked against files in this repository by `analysis/v42/compile_results.py` (510 consistency checks, 0 failed) and `analysis/v42/check_numbers.py` (`check_numbers.csv`: 1938 rows, 585 DIRECT_MATCH, 640 DERIVED_MATCH, 709 ROUNDED_MATCH, 4 non-numeric, **0 NOT_RECOMPUTED, 0 NOT_FOUND, 0 MISMATCH**).

## 0. What this environment could and could not do

| Item | Status |
|---|---|
| GPU / `CUDA_VISIBLE_DEVICES=2,3` | **Not available** in this container (no `nvidia-smi`). No training, evaluation or benchmark was run. |
| Checkpoints (`best_model.pt`), phase/mask/hologram data | **Not in the repository** (`.gitignore` excludes `*.pt`; `data/` holds only `manifest.csv`, `splits.json`). |
| Registration archive (`diagnostics/registration_summary.json`, `hologram_registration.csv`, `classical_phase_shift.csv`, `hologram_inventory.csv`) | Regenerated from `data/` with `scripts/register_holograms.py`; stored in `runs/diagnostics/` and `results_for_manuscript/registration/`. |
| Reruns performed | **No training or re-evaluation of any experiment.** One post-hoc diagnostic (neural recovered contrast, section 7) was run by the author on the server with the existing checkpoints: `CUDA_VISIBLE_DEVICES=2 python scripts/neural_phase_contrast.py --config config/base.yaml --device cuda --with-classical` (~3 min; output `runs/common_fields/neural_phase_contrast.json`). |
| Unit tests | `python scripts/selftest.py` passes before and after the code changes (torch CPU build installed here). |

## 1. Source of truth and seed/replication audit (verified)

All values come from `runs/*/metrics_test*.json`, `runs/common_fields/**`, `runs/diagnostics/*.csv`, `runs/benchmark_results/*.json`, `runs/z_calibration.json`, logs and `config/`.
Per run, `compile_results.py` checked: `resolved_config.yaml` seed = directory seed = provenance `train_seed`; `splits_sha256` equals `sha256(data/splits.json)` for every run; collector checkpoint SHA equals evaluation-provenance SHA.

| Configuration | Training runs | Seeds | Verified from |
|---|---|---|---|
| End-to-End Neural Baseline | 3 | 42, 1337, 2024 | run dirs, provenance |
| +IPP (per-cell) | 3 | 42, 1337, 2024 | " |
| +IPP (image) | 3 | 42, 1337, 2024 | " |
| +Area, +BGA, w=0.1/0.3/3.0, +Amplitude, +Fwd (fixed z), +Fwd (free z), Compact Baseline, Compact +IPP | 1 each | 42 | " |
| In-Line Neural Configuration | 1 | 42 | " |
| Classical off-axis / in-line | deterministic, run once | — | `runs/conventional_*` |

* The extra run `runs/v2_ipp_w10_off_axis` is a separate seed-42 training of the +IPP (per-cell) configuration (identical loss weights, verified). It is **not** a fourth seed of +IPP (per-cell) and is not counted (`metadata/configurations.json::W10_repeat_note`). Matched-cell MAPE 0.1872 vs 0.1875 for the seed-42 +IPP run.
* No `n` was inflated: every `training_runs` value is the count of run directories with a verified seed.
* Checkpoint SHA-256, selected epoch, epochs completed and wall hours per run: `metadata/configurations.json::<arm>.checkpoints`.

## 2. Field and cell counts (verified)

| Set | Fields | Reference cells | Verified |
|---|---|---|---|
| Off-axis test (Tables 4, 6–11) | 113 | 3186 (1452 edge + 1734 interior) | baseline ×3, classical, decomposition |
| Common set (Table 5) | 107 | 3055 | baseline ×3 (`metrics_test_common.json`), in-line, both classical |
| In-line split | 521 / 122 / 107 (train/val/test) | — | `logs/v2_20260930_074055_gpu3/train_G.log` |

* SNU_01–SNU_50 are excluded **before** training: 39 (train) + 5 (val) + 6 (test) = 50 (`train_G.log`, `logs/corrected_20260930_054214/common_G.log`). The in-line model was therefore **not** trained on the mismatched fields; the text must say excluded, not trained on.
* Matched-cell counts: the decomposition re-count (e.g. 1897 for baseline seed 42) differs from the stored evaluation (1898) by at most 3 cells per run (`decomposition.json::matched_count_reconciliation`). Baseline: 1890 ± 9 stored vs 1889.7 re-counted. Both are reported; they must not be mixed within one table.

## 3. Tables (all verified against stored outputs; values in `MANUSCRIPT_SYNC_numbers.md`)

* **Table 3.** Metadata now exported from configs (`metadata/configurations.json`): training runs, seeds, comparator, non-zero loss weights, architecture (encoder, projection channels, amplitude head, parameter count), geometry (modality, wavelength 0.666 µm, pitch 0.284871 µm, z 33.77 µm, trainable z, crop offset [−6, −2]). Seeds column in the manuscript (3/1/…) is correct.
* **Table 4.** All 14 quantities recomputed. `Cells matched`: classical 1870, neural 1889.67 ± 8.50 over three seeds → descriptive difference **+20 ± 9** (exported as `descriptive_difference_text`; no resolution rule applied). Other rows carry "no resolution test (deterministic comparator)".
* **Table 5.** In-line neural n=1 (seed 42), off-axis neural n=3 rescored on the same 107 fields, classical off-axis and in-line deterministic. All stored values agree with `common_fields_summary.csv`. Recovered contrast (107 fields): in-line neural +0.8869 rad (1 run), off-axis neural +0.8992 ± 0.0362 rad (3 runs), classical off-axis +0.9216, classical in-line −0.0836, reference +1.0969 rad.
* **Table 6.** Matched-cell statistics use matched cells; field totals use all predicted and all reference cells. Circularity has no field-total keys in any metrics file; exported as `"N/A"` with the reason. The manuscript currently prints `--` in these cells.
* **Table 7.** Decomposition recomputed independently from `decomposition_cells.csv` / `decomposition_fields.csv`: geometric means and median |log| agree with `decomposition_summary.csv` and `decomposition_by_config.csv` to 1e-9; domain × phase = total per cell (< 1e-6); field-level `domain = S_domain/S_ref`, `phase = S_pred/S_domain` verified from the stored S values. Edge rule: pixel index < 4 or ≥ size − 4 (strict inequality, `edge_margin_px: 4`), i.e. within 3 px of the border, consistent with the manuscript. The classical pipeline is a single deterministic result. **Wording issue:** Table 7 uses `n` for both training runs and matched cells (header `n` = cells, group rows `n=3`).
* **Table 8.** Edge/interior recall and MAPE and false positives (all / within 15 px) recomputed; false-positive counts equal `cells_false_positive` in each metrics file for all 10 runs.
* **Table 9 / 10.** Every row recomputed; means/SD for three-run configurations, point estimates for the rest (no SD assigned). Resolution quantities exported for every comparison: delta, pooled SD, 2× pooled SD, runs on each side, evaluability, assessment. Result is unchanged from the manuscript: only +IPP (per-cell) and +IPP (image) vs baseline exceed 2× pooled SD on matched-cell dry-mass MAPE; nothing exceeds it on field-total MAPE; all single-run comparisons are *Not estimable from the available runs* (no zero or borrowed variance used). The collector's `results_comparisons.json` agrees to 1e-9. `+IPP (image) vs +IPP (per-cell)` is included.
* **Table 11.** Amplitude and forward-residual values agree with `metrics_test.json`. `Fields favouring reference phase` is now computable for **every** row (previously `--` for test sets) from the per-field CSVs: 113 test fields, global surface: 60/113 (0.9×), 71/113 (0.5×); 112 test fields, per-field surfaces: 93/112, 110/112; global surface on the same 112 fields: 59/112, 70/112; validation: 18/32, 20/32 (off-axis), 30/32, 32/32 (in-line). Configured tolerance = 0.01 (`config/base.yaml`, `z_calibration.json`).
* **Table 13 / benchmarking.** Collector files `results_hardware_arm_{A,B,D0,KA,KB}.json` are consistent (500 timed runs, 50 warm-ups, batch 1, 900 px, RTX A5000, FP32/FP16 PyTorch and ONNX, p50, p99, p99/p50, parameters, GMACs, weights and activation memory). Collector statistics equal the mean of the per-seed sessions. Not rerun; nothing inconsistent. Parameters 9,598,099 (baseline), 3,360,403 (compact).

## 4. Forward model and z (verified)

Recording distance 33.77 µm (config). In-line pooled residual minimum on the grid at **33.684 µm** (grid step 4.81 µm; per-field best 33.68 µm for all fields). Off-axis: per-field best z = −96.24, +96.24, 33.68, 33.68 µm (two at the scan limits) — the off-axis residual does not constrain z. The old "in-line minimum near 50.5 µm" statement is not supported (check `forward_model.json`). Free-z: 34.10 µm after epoch 1, 33.41 µm final; selected checkpoint (epoch 44) scored at 33.408 µm. Residual margins at 0.9× / 0.5× in `forward_model.json` (validation, 32 fields). Both +Fwd configurations have ratio 1.0068, above the baseline's 1.0039 ± 0.0002. All wording must stay scoped to the implemented operator.

## 5. Gradient path (inconsistency in the brief, not in the manuscript)

`runs/gradient_path_b_cell_ipp_512.csv`, column `ratio_at_weight_1`, 30 batches: **median 0.373, range 0.243–0.655** (also in `logs/v2_20260930_065521_gpu2/gradient_path.log`). The values median 0.302 / range 0.18–0.67 in the brief are **not supported by the CSV**; the manuscript (main.tex §3, 0.373, 0.243–0.655) is already correct. **Stale code fixed:** the docstring of `scripts/check_gradient_path.py` quoted 0.299 / "0.25 to 0.29 on 512 px crops"; it now quotes the CSV values. Results unaffected.

## 6. Gabor registration (NOT verifiable here)

The registration files were regenerated with `scripts/register_holograms.py`, and `compile_results.py` recomputes the Sec. 4.1 values from `hologram_registration.csv` (734/800 within 1 px, AUC 0.975 / 0.973, low-pass-only AUC 0.506, 750 paired / 50 mismatched); they are marked `verified: true` in `metadata/gradient_registration_crop.json`. The SNU_01–50 correlation range in the file is −0.1335 to 0.1769, which matches the manuscript (−0.13 to 0.18); the earlier 0.09–0.18 in the `config/base.yaml` comment was wrong and has been corrected. The split (521/122/107) and the 39 + 5 + 6 = 50 exclusion are also verifiable from the logs.

## 7. Results that need recomputation / missing outputs

1. **Neural "Recovered in-cell phase contrast" (Table 5) — RESOLVED.** It was not derivable from stored files; `scripts/neural_phase_contrast.py` computes it with the same per-field function as the classical pipeline (per field: mean phase inside the reference mask minus outside; median over the 107 fields). The `--with-classical` re-check reproduced the stored classical value 0.921582 exactly, which validates the definition. Values: see section 3, Table 5. Stored in `runs/common_fields/neural_phase_contrast.json`; `compile_results.py` reads it.
2. **In-line neural has one training run.** Table 5/Table 4 comparisons involving it cannot be assessed by the 2× pooled-SD rule (not estimable). If a replicated in-line comparison is wanted, this is the highest-priority extra training (not run here, no GPU):
   ```
   CUDA_VISIBLE_DEVICES=2 python main.py train    --config config/v2/g_baseline_gabor.yaml --seed 1337 --set experiment_name=v2_G_seed1337
   CUDA_VISIBLE_DEVICES=2 python main.py evaluate --config config/v2/g_baseline_gabor.yaml --seed 1337 --set experiment_name=v2_G_seed1337
   ```
   (same for 2024; then add the runs to the registry in `compile_results.py` and to `ARMS`). Not required for correctness of anything currently reported.
3. **Common-field evaluation commands are not in the repository.** `run_corrected_study.sh` was deleted by `cleanup_server.sh`. The provenance (`tag: common`, 107 fields, `data.exclude.modalities` including `off_axis`) and the `config/base.yaml` comment document the method; reproduce with `python main.py evaluate --config <cfg> --seed S --tag common --set experiment_name=<name> data.exclude.modalities=[gabor,off_axis]` (reconstructed, not copied from the original).
4. **Figure 3** and figure-1 panels need checkpoints (cannot be regenerated here).

## 8. Stale code, stale results, items already correct

**Stale code (fixed or isolated)** — see `CLEANUP.md`: legacy names (`JointPhysicsAwareLoss`, `PhaseMaskContrast`, `PhaseVolumePreservation`, config key `phase_volume`) replaced by current names, old names resolved only through a deprecation shim `holoqpi/losses/_legacy.py`; "physics-aware" wording removed from generated text; gradient docstring; `scripts/mass_uncertainty.py` and `scripts/null_input_probe.py` moved to `legacy/`; superseded v42 helper scripts moved to `legacy/analysis_v42_old/`.

**Stale results (kept, raw, not used by the manuscript):** top-level `runs/v2_*_hardware_benchmark_cuda.{csv,json}` (earlier benchmark stage; `runs/RESULTS.md` Table 4 therefore differs from the collector — the manuscript correctly uses `results_hardware_arm_*.json`); `runs/RESULTS.md` Table 3b heading wording; repo-root v4.1 figure scripts/images and `assets/` (now archived in `legacy/figures_v4.1/`); `figures/` (exploratory `make_figures.py` output, not manuscript).

**Already correct (no change):** every number printed in `HoloQPI_4.2/table_*.tex` is found, at its printed precision, among the exported values (709 ROUNDED_MATCH lookups; a presence test, not a cell-by-cell comparison, so the structural wording issues below are listed separately), the gradient values in `main.tex`, split counts, the resolution assessments, seeds/n in Table 3.

## 9. Manuscript issues for the author (details per table in `MANUSCRIPT_SYNC.md`)

Wording/structure only (no numbers wrong): `--` should be `N/A` (Tables 4 matched-count Δ → descriptive "+20 ± 9"; 5 neural contrast; 6 circularity field totals; 11c counts now available); Table 10 verdict vocabulary ("resolved / not resolved / not resolvable" → "Exceeds 2× pooled SD / Within 2× pooled SD / Not estimable from the available runs", column "Assessment"); Table 11 "Fields worse" → "Fields favouring reference phase"; Tables 7, 8, 9 use `n` for both training runs and cells/fields (use "training runs" and "N"); `main.tex` §4.8 contains the phrase "not a significance test" (a negation; consider rewording since the word is on the avoid list); `table_6` CI note etc. unchanged. No occurrence of "preregistered", "50.5 µm", "physics-aware", "bottleneck", "novel", "state-of-the-art" etc. in the manuscript text or tables was found.
