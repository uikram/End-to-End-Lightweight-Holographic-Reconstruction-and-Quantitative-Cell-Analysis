# HoloQPI_4.2 version manifest

**Status.** v4.2 is the synchronized candidate derived from v4.1. HoloQPI_4.1 is not in this repository and
was not edited. No model was retrained and no new experiment was run; every number comes from stored files in `runs/`, `logs/`
and `config/`. In-line neural results remain one training run (seed 42).

**Source of truth.** `runs/`, `logs/`, `config/` -> `analysis/v42/compile_results.py` ->
`results_for_manuscript/` (JSON/CSV, 511 consistency checks) -> `analysis/v42/update_tables.py` (Tables 3-7, 10, 11)
-> `HoloQPI_4.2/`. `results_for_manuscript/RESULTS_INDEX.md` maps each table and figure to its source.
`analysis/v42/check_numbers.py` traces every manuscript number.

**What changed from v4.1**
- Tables 3-7, 10, 11 regenerated from the result package; training-run counts (3 for Baseline, +IPP per-cell, +IPP image; 1 otherwise) and the resolution rule stated consistently.
- Terminology unified: integrated phase, L_IPP^cell, L_IPP^img, measurement-aware objectives, forward-model consistency.
- Neural recovered in-cell contrast added (107 common fields) next to the classical values.
- Wording: field-total dry-mass MAPE "was lower for the tested classical pipeline"; classical in-line "near-background specimen contrast"; decomposition described as error structure, not mechanism; no significance language.
- Counts (113/3186 off-axis, 107/3055 common, in-line 521/122/107) and the gradient ratio (median 0.373) corrected to stored values.

**Not changed / open.** Sec. 4.1 registration statistics are author-supplied; the registration archive (`registration_summary.json`,
`hologram_registration.csv`, `classical_phase_shift.csv`, `hologram_inventory.csv`) is not in the repository, so they are
flagged unverified (`results_for_manuscript/metadata/gradient_registration_crop.json`). Figs 1 and 3 are not regenerated.
