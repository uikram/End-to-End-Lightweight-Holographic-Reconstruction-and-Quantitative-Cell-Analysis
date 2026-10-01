# CLEANUP

Rule followed: nothing deleted unless unreferenced; raw results in `runs/` untouched; aliases retained where old configs/checkpoints need them. `selftest.py` passes after every change.

## Renamed (current names only in the active code)
| Old | New | Compatibility |
|---|---|---|
| `JointPhysicsAwareLoss` | `JointMeasurementLoss` | old name not exported; resolves via `holoqpi/losses/_legacy.py` (PEP 562 `__getattr__`) with a `DeprecationWarning` |
| `PhaseVolumePreservation` | `ImageIntegratedPhase` (= L_IPP^img) | same shim |
| `PhaseMaskContrast` | `LegacyPhaseMaskContrast` (weight 0 in all manuscript configs) | same shim |
| loss key `phase_volume` | `image_integrated_phase` (`config/base.yaml`, `config/v2/b1_image_volume.yaml`, loss-component name, docs) | `loss.weights.phase_volume` is still **accepted** as a deprecated alias (warning) because every stored `resolved_config.yaml` of the +IPP (image) runs uses it; verified that the old resolved config and the new config give identical weights/active terms; setting both is an error |
Retained on purpose, because stored configs/logs use them: config section `loss.physics`, key `phase_mask_contrast`, `pmc_margin`, `active_physics_terms()`. `scripts/selftest.py` uses the new class names and passes. No active code path, documentation heading or generated output uses "physics-aware".

## Wording changed in code-generated text
`scripts/collect_results.py` (Table 3b heading, "significance" disclaimer), `collect_benchmark_results.py`, `aggregate_seeds.py` (the printed verdict hard-coded "physics-aware … do not change what this model measures", which is also false now that +IPP is resolved on matched-cell MAPE), `make_figures.py` and `conventional_baseline.py` ("FAILED" → "INVALID"), `metrics/measurement.py` docstring. `runs/RESULTS.md` (generated, raw) still has the old heading; regenerate with `scripts/collect_results.py` on the server if desired.

## Stale code corrected
`scripts/check_gradient_path.py` docstring/comment: 0.299 and "0.25–0.29 on 512 px crops" → 0.373 (0.243–0.655) from `runs/gradient_path_b_cell_ipp_512.csv`.

## Isolated (moved to `legacy/`)
* **v4.1 figure infrastructure** → `legacy/figures_v4.1/` (README inside): repo-root `manuscript_images/figure_{2,4,5,6,7,8,9}.{py,png}`, `FIGURE_MANUAL.md`, `build_fig1.js`, and `assets/`. `manuscript_images/` now only keeps the server-side Fig. 1/3 inputs (README inside). Current figure scripts: `results_for_manuscript/figures/scripts/` (`make_all.py`).
* Earlier (`legacy/`, confirmed unreferenced by `run_v2.sh`, docs, imports, configs and the manuscript)
* `scripts/mass_uncertainty.py`, `scripts/null_input_probe.py` → `legacy/scripts/` (only comment references remain; updated to the new path).
* Superseded helper copies `extract_values.py`, `make_tables_v42.py`, `make_changelog_numbers.py` → `legacy/analysis_v42_old/`; replaced by `analysis/v42/compile_results.py` + `check_numbers.py`. (The originals under `HoloQPI_4.2/analysis/v42/` are untouched; they belong to the manuscript folder.)

## Added
`analysis/v42/` (compiler, checker, audit/sync/figure docs), `results_for_manuscript/`, `scripts/neural_phase_contrast.py`.

## Left in place deliberately
`runs/` (all raw), top-level `runs/v2_*_hardware_benchmark_cuda.*` (early benchmark stage, not used), `figures/` (exploratory `make_figures.py` output), `server_figs.zip`, `cleanup_server.sh`, `HoloQPI_4.2/`. Suggested later (after your sign-off): delete `server_figs.zip` and `figures/`; remove the `phase_volume` alias only together with re-labelling stored configs.
