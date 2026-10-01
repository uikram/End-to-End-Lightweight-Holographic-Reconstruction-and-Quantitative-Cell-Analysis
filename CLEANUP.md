# CLEANUP

Rule followed: nothing deleted unless unreferenced; raw results in `runs/` untouched; aliases retained where old configs/checkpoints need them. `selftest.py` passes after every change.

## Renamed (alias kept)
| Old | New | Alias / compatibility |
|---|---|---|
| `JointPhysicsAwareLoss` (`holoqpi/losses/composite.py`) | `JointMeasurementLoss` | `JointPhysicsAwareLoss = JointMeasurementLoss`; `build_loss` unchanged |
| `PhaseVolumePreservation` (terms.py) | `ImageIntegratedPhase` (= L_IPP^img) | alias; config key **`phase_volume` retained** (it is stored in every `resolved_config.yaml`, incl. the +IPP (image) runs) |
| `PhaseMaskContrast` | `LegacyPhaseMaskContrast` (weight 0 in all manuscript configs) | alias; key `phase_mask_contrast` retained |
Config section `loss.physics` and `active_physics_terms()` keep their names (stored in resolved configs). Docstrings, comments, `config/base.yaml`, `l_lora.yaml` headings reworded ("measurement-aware", "forward-model consistency").

## Wording changed in code-generated text
`scripts/collect_results.py` (Table 3b heading, "significance" disclaimer), `collect_benchmark_results.py`, `aggregate_seeds.py` (the printed verdict hard-coded "physics-aware … do not change what this model measures", which is also false now that +IPP is resolved on matched-cell MAPE), `make_figures.py` and `conventional_baseline.py` ("FAILED" → "INVALID"), `metrics/measurement.py` docstring. `runs/RESULTS.md` (generated, raw) still has the old heading; regenerate with `scripts/collect_results.py` on the server if desired.

## Stale code corrected
`scripts/check_gradient_path.py` docstring/comment: 0.299 and "0.25–0.29 on 512 px crops" → 0.373 (0.243–0.655) from `runs/gradient_path_b_cell_ipp_512.csv`.

## Isolated (moved to `legacy/`, confirmed unreferenced by `run_v2.sh`, docs, imports, configs and the manuscript)
* `scripts/mass_uncertainty.py`, `scripts/null_input_probe.py` → `legacy/scripts/` (only comment references remain; updated to the new path).
* Superseded helper copies `extract_values.py`, `make_tables_v42.py`, `make_changelog_numbers.py` → `legacy/analysis_v42_old/`; replaced by `analysis/v42/compile_results.py` + `check_numbers.py`. (The originals under `HoloQPI_4.2/analysis/v42/` are untouched; they belong to the manuscript folder.)

## Added
`analysis/v42/` (compiler, checker, audit/sync/figure docs), `results_for_manuscript/`, `scripts/neural_phase_contrast.py`.

## Left in place deliberately
`runs/` (all raw), top-level `runs/v2_*_hardware_benchmark_cuda.*` (early benchmark stage, not used), `figures/`, `assets/`, repo-root `manuscript_images/` (stale v4.1 layout; see FIGURE_STATUS), `server_figs.zip`, `cleanup_server.sh`, `HoloQPI_4.2/`. Suggested later (after your sign-off): delete `server_figs.zip`, `figures/` and the stale repo-root `manuscript_images/*.png`/`assets/`; remove the `phase_volume` alias only together with re-labelling stored configs.
