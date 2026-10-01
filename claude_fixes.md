# claude_fixes — reference for finishing the manuscript

Branch `claude/admiring-maxwell-xwtq43` (last commit `0fe162c`). Nothing was retrained and no new
training experiment was run; no number was invented. HoloQPI_4.1 is not in this repo and was not edited.
Detailed per-table values: `analysis/v42/MANUSCRIPT_SYNC.md`. Number trace: `analysis/v42/check_numbers.csv`.

## 1. What is already changed in `HoloQPI_4.2/` (manuscript)
**Tables regenerated from `results_for_manuscript/` JSON** (by `analysis/v42/update_tables.py`): `table_3, 4, 5, 6, 7, 10, 11.tex`.
- Table 3: training runs 3 (Baseline, +IPP per-cell, +IPP image; seeds 42/1337/2024), 1 (seed 42) for everything else incl. In-Line Neural.
- Table 4: "Cells matched" difference is descriptive (+20 ± 9), not a resolution test.
- Table 5: neural recovered in-cell contrast added (107 fields): in-line +0.887, off-axis 0.899 ± 0.036, classical off-axis +0.922, classical in-line −0.084, reference +1.097 rad.
- Table 6: circularity field-total entries are N/A.
- Table 7: `n` split into "training runs" vs N (cells/fields).
- Table 10: column "Assessment" with "Exceeds 2× pooled SD" / "Within 2× pooled SD" / "Not estimable from the available runs".
- Table 11c: "Fields favouring reference phase" with counts; "configured tolerance 0.01".

**Text edits in `main.tex`**
- §4.8 wording (no significance language).
- §5.1: recovered contrast reported as a median, with the neural values.
- §5.5: counts.
- Field totals: "field-total dry-mass MAPE was lower for the tested classical pipeline" (abstract/intro paragraph ~l.916 and Discussion/Conclusion ~l.1548). Replaces "goes the other way" / "rank the two pipelines the other way".
- Classical in-line: "recovered near-background specimen contrast" (abstract ~l.108, §5.5 ~l.943, Discussion ~l.1297). Replaces "did not recover the specimen phase" / "recovered no specimen contrast". The values (r = −0.2075; −0.0836 rad vs +1.0969 rad reference) are unchanged.
- Decomposition: "compensating error structure" (describes structure, not mechanism) at ~l.1029, 1376, 1549.
- Compiles to 30 pages, 0 undefined references/citations. Forbidden-term search (significan*, preregist*, physics-aware, PMC, RBC, Applied Physics B, novel, state-of-the-art, phase_volume, 0.302) returns nothing in `main.tex`/`table_*.tex`.

**Files removed from `HoloQPI_4.2/`:** `analysis/` (moved to repo `analysis/v42/`), duplicate `figure_code` files. Added `VERSION_MANIFEST.md`; status notes in `DISCREPANCIES.md`, `CHANGELOG.md`.

## 2. Values you should NOT change (checked against source)
- Gradient ratio: median **0.373** (range 0.243–0.655, 30 batches) from `runs/gradient_path_b_cell_ipp_512.csv`. The 0.302 / 0.18–0.67 in the brief was wrong.
- Counts: off-axis test 113 fields / 3186 cells; common set 107 fields / 3055 cells; in-line split 521/122/107; exclusions 39 + 5 + 6 = 50 (SNU_01–50, excluded before training).
- z calibration: recording 33.77 µm; in-line grid minimum 33.684 µm; free-z 34.10 → 33.41 µm; off-axis does not constrain z. No "50.5 µm" statement.
- Resolution rule: |Δ| > 2·sqrt((s1²+s2²)/2) (ddof = 1) and ≥ 3 runs on both sides; otherwise "Not estimable from the available runs".

## 3. Sec. 4.1 registration — status
The original diagnostics archive could not be found. The four files were **regenerated** from `data/` with
`scripts/register_holograms.py` (`runs/diagnostics/`, copied to `results_for_manuscript/registration/`).
`compile_results.py` recomputes 15 claims from `hologram_registration.csv`; all agree with the manuscript text:
734/800 matched and 3 control within 1 px; AUC 0.975 (r) / 0.973 (error); low-pass AUC 0.506; 750 paired / 50 mismatched;
paired max shift 4.7 px; paired r 0.47–1.0; control r max 0.25; **SNU_01–50 r = −0.13 to 0.18** (file: −0.1335 to 0.1769).
Describe these as regenerated diagnostics if the provenance is asked. The old config comment "0.09–0.18" was wrong; fixed.

## 4. Figures (`analysis/v42/FIGURE_STATUS.md`)
- Figs 2, 4–9: regenerated from `results_for_manuscript/figures/` into `figures/regenerated/` (never over the manuscript PNGs); data verified. Regenerated PNGs differ from `HoloQPI_4.2/manuscript_images/` by size/font/layout only, not data. The manuscript PNGs were not edited.
- Figs 1 and 3: inspected visually, no stale terminology, seeds or v4.1 values; cannot be regenerated (need checkpoints and data). Fig. 3 NCI_08 MAE 0.296 rad matches `fig1_panels.json`; NCI_06 0.248 rad by inspection only.
- Do not use the repo-root `manuscript_images/figure_4.png` (old v4.1 ablation figure) or anything in `legacy/figures_v4.1/` (stale "physics-aware" labels).

## 5. Code and repository
- **Result tooling (new):** `analysis/v42/compile_results.py` (526 consistency checks, 0 failed), `check_numbers.py` (2336 rows: 588 direct, 642 derived, 1101 rounded, 4 non-numeric, 1 external [Park 96.97 M], 0 unresolved), `update_tables.py`, `make_sync.py`, plus AUDIT/SYNC_AUDIT/MANUSCRIPT_SYNC/FIGURE_STATUS.
- **New scripts:** `scripts/neural_phase_contrast.py` (run on the server; output `runs/common_fields/neural_phase_contrast.json`), `scripts/collect_registration_archive.sh`.
- **Losses:** current names `JointMeasurementLoss`, `ImageIntegratedPhase`, `LegacyPhaseMaskContrast`; old names (`JointPhysicsAwareLoss`, `PhaseMaskContrast`, `phase_volume`, `phase_mask_contrast`) still resolve through `holoqpi/losses/_legacy.py` with a DeprecationWarning. Config keys: `cell_integrated_phase`, `image_integrated_phase` (alias `phase_volume`), `forward_model`.
- **Docs rewritten:** `README.md`, `docs/documentation.md`, `config/v2/_shared.md`, `BENCHMARKING.md`, `MANIFEST.txt`, `CLEANUP.md`, `results_for_manuscript/README.md` and `RESULTS_INDEX.md`.
- **Cleanup:** historical scripts/figures moved to `legacy/` (`figures_v4.1/`, `mass_uncertainty`, `null_input_probe`, `analysis_v42_old/`); removed `server_figs.zip`, `cleanup_server.sh`, `figures/`. `runs/` and provenance sources are untouched. `.gitattributes` forces LF for `*.sh`/`*.py` (the server copy of the collector had CRLF).
- `results_for_manuscript/`: baseline, classical, inline, ipp, amplitude, forward_model, decomposition, boundary_sensitivity, benchmarking, metadata, summary, registration, figures. Every value has source file/key, field set, runs, seeds.

## 6. What was NOT changed / open
- No retraining; In-Line Neural stays **one run (seed 42)**. Any in-line comparison must say "not resolvable / not estimable". Three in-line seeds would be a new GPU job and your decision.
- Figs 1 and 3 not regenerated.
- Not merged into `bug_fixes/001`; no pull request opened.
- Wording rules to keep: no significance language; "n" only for training runs (N for cells/fields); decomposition = error structure, not mechanism; "near-background recovered specimen contrast".

## 7. Rebuild commands (CPU only)
```
python analysis/v42/compile_results.py
python analysis/v42/check_numbers.py
python analysis/v42/update_tables.py
python results_for_manuscript/figures/scripts/make_all.py
python scripts/selftest.py
```
