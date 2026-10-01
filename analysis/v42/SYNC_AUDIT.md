# SYNC_AUDIT — current state of CODE ↔ CONFIGS ↔ RESULTS ↔ results_for_manuscript ↔ MANUSCRIPT ↔ README/docs

Pre-final pass. Branch `claude/admiring-maxwell-xwtq43`. No training or re-evaluation was run in this phase.

## 0. Facts that shaped the plan

* **`HoloQPI_4.1` / `HoloQPI_Manuscript_v4_1` is not in this repository or in its git history.** The brief's step "copy v4.1 to HoloQPI_4.2" could not be executed. `HoloQPI_4.2/` already existed in the repository with the locked section structure (verified below), so it was synchronised **in place**. Nothing named 4.1 was touched.
* No GPU in this session. The only post-hoc diagnostic needed (neural recovered contrast) was run by the author on the server earlier; its output is in the repository.
* The registration archive (`diagnostics/…`) was regenerated afterwards (see §3 and the final-pass update).

## 1. Already synchronised (verified, no edit needed)

| Item | Evidence |
|---|---|
| Section structure of `main.tex` equals the locked list (1 Introduction … 8 Conclusions; 2.1–2.6; 3.1–3.5 with 3.5.1–3.5.4; 4.1–4.8; 5.1–5.7; 6.1–6.5; 7; 8) | `grep \\section/\\subsection` |
| Title; terminology S_k, L_IPP^cell, L_IPP^img, measurement-aware objectives, forward-model consistency | text scan: no "phase volume", "L_PV", "PMC", "physics-aware"; "optical volume" only in Related Work (other papers) |
| Relative integrated-phase error = relative dry-mass error, stated once (Sec. 4.5), reported as dry-mass MAPE | Sec. 4.5 |
| Park et al. description (phase and membrane-fluorescence segmentation applied to the phase; 256×256 inputs; distributions; narrow gap sentence; 500-iteration GS) | Introduction, Sec. 2.3–2.4, 5.1 |
| All decimal numbers in the running text (main.tex) | 0 NOT_FOUND in `check_numbers.csv`; one literature value (96.97 M, Park et al.) marked EXTERNAL_VALUE |
| Tables 4, 6–9, 11a–b numeric content | present in the package at printed precision; Table 10(a) likewise |
| Figures 1, 2: Integrated phase S_k, no old terminology; 107-field in-line panel; corrected classical values | viewed; Fig. 2 values equal Table 4/5 |
| Figure 7: in-line minimum at the grid point 33.68 µm, no 50.5 µm statement | regenerated and compared |
| Gradient ratio 0.373 (0.243–0.655) in the text | CSV |

## 2. Stale / inconsistent (and what was done)

| Item | State found | Action |
|---|---|---|
| Table 3 seeds column | "Seeds" with counts only | now **Training runs (seeds)**, generated from `metadata/configurations.json` (`update_tables.py`) |
| Table 4 "Cells matched" Δ | `--` | **+20 ± 9**, marked descriptive; tabnote added |
| Table 5 neural contrast | `--` | in-line +0.887, off-axis 0.899 ± 0.036, classical +0.922 / −0.084 (reference +1.097) with definition; text says **median**, not mean |
| Table 6 circularity field totals | `--` | N/A + "not defined for circularity, which is non-additive" |
| Table 7 `n` | used for cells and runs | column → N; `n` only for training runs |
| Table 10 | "Verdict", resolved/not resolved/not resolvable, side-by-side blocks | **Comparison / Δ / 2× pooled SD / Assessment** with the three required phrases; N/A where not estimable; field-total block separate; +IPP (image) vs +IPP (per-cell) kept |
| Table 11c | "Fields worse", `--` for test rows | **Fields favouring reference phase**, counts for all four rows; same-fields global-surface numbers in the note |
| Sec. 4.8 | "not a significance test" | "no hypothesis test is performed" |
| Sec. 5.1 | "mean in-cell phase contrast" (the metric is a median) | corrected; neural values added |
| Sec. 5.5 | counts only for validation probes | test-field counts added (60/71 of 113; 93/110 of 112) |
| `main.tex` header comment | cited a deleted generator | updated |
| `HoloQPI_4.2/analysis/v42/*` | first-generation extractor, `values.json` (superseded, contradicts new package) | removed (scripts archived in `legacy/analysis_v42_old/`) |
| `HoloQPI_4.2/figure_code/` duplicates of fig 7/8 scripts and `mstyle.py` | duplicated `results_for_manuscript/figures/scripts/` | removed; README updated |
| `DISCREPANCIES.md`, `CHANGELOG.md` | refer to removed scripts | status notes added at the top (content kept as a record) |
| README.md, docs/documentation.md, BENCHMARKING.md, config/v2/_shared.md | old title, optical volume, physics-aware, v1 weights, gradient 0.25–0.29, probe values contradicting the verified ones, "143 checks" fine | rewritten / corrected (README, docs, `_shared.md`); BENCHMARKING.md terms fixed |
| Loss/config names | `JointPhysicsAwareLoss`, `PhaseMaskContrast`, `PhaseVolumePreservation`, `phase_volume` | renamed; shim for old names (`CLEANUP.md`) |
| Stale figure infrastructure | v4.1 scripts/PNGs, `assets/`, `figures/` outputs | archived (`legacy/figures_v4.1/`) or removed; figure scripts now in the package |

## 3. Manuscript-only / code-only / obsolete / still unverifiable

* **Manuscript-only values** (no source file in the repository): the registration statistics of Sec. 4.1 (734/800, AUC 0.975/0.973, low-pass AUC 0.506, 4.7 px, r ranges) — originally author-supplied and flagged `verified: false`; since recomputed from the regenerated `hologram_registration.csv` (SNU_01–50 range −0.1335 to 0.1769, matching the manuscript). The 96.97 M parameter count (Park et al.) is a literature value.
* **Code-only** (not in the manuscript, kept because the workflow uses them): `scripts/make_figures.py` (exploratory plots, run by `run_v2.sh` stage 9), `scripts/collect_results.py`/`runs/RESULTS.md`, `scripts/audit_labels.py`, `scripts/diagnose_bias.py`, `scripts/estimate_aberration.py`, `scripts/prepare_membrane.py`, `scripts/aggregate_seeds.py`, `config/v2/l_lora.yaml`, `test/` notebook.
* **Obsolete (removed or archived):** see `CLEANUP.md`.
* **Not regenerable here:** Figures 1 and 3 (checkpoints/data); pixel-identical regeneration of Figures 2, 4, 5, 6, 9 (authoring scripts outside the repository); regenerated PNGs have the same data but not the same bytes.

## 4. What the manuscript tables/text still require from the author

Nothing numeric. Wording decisions left to the author are listed in `MANUSCRIPT_SYNC.md`.


**Update (final pass).** Registration files regenerated with `scripts/register_holograms.py` and stored in `runs/diagnostics/`; `compile_results.py` recomputes the Sec. 4.1 values (15 claims, all agree, `verified: true`; 526 consistency checks, 0 failed). The SNU_01-50 range is -0.1335 to 0.1769.
