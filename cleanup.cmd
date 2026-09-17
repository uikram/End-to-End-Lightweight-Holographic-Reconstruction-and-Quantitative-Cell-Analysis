@echo off
rem ===========================================================================
rem  HoloQPI - clean the folder. One run, no questions.
rem
rem      cleanup.cmd
rem
rem  Run it from inside the project folder (the one holding main.py).
rem
rem  KEEPS   code      main.py run_v2.sh study.sh holoqpi\ scripts\ config\
rem                    README.md requirements.txt environment.yml
rem          data      data\ weights\
rem          results   runs\v2_*  runs\RESULTS.md  runs\results_table.csv
rem                    the six current figures, the study's own logs
rem          docs      docs\  (every document, gathered here)
rem
rem  DELETES everything else: transfer zips, the v1 drivers, the 2026-09-09
rem          smoke-test runs and figures, old log archives, the extra
rem          checkpoints nothing reads.
rem
rem  Two of those are hazards rather than clutter:
rem    figures\fig01..09, fig11..15 are plots of the smoke run and look
rem      exactly like results;
rem    runs\*_modality_comparison.* are the files make_figures.py reads when
rem      building figures 5, 7, 13 and 14 - leave them and the next figure run
rem      draws smoke-test numbers into a paper figure.
rem ===========================================================================

if not exist main.py (
  echo   Run this from the project folder -- main.py is not here.
  exit /b 1
)

echo.
echo   Cleaning %CD%
echo.

rem --- gather every document into docs\ ------------------------------------
if not exist docs mkdir docs
if exist "Claude outputs" (
  move /y "Claude outputs\HoloQPI_Project_Documentation.pdf" docs\ >nul 2>nul
  move /y "Claude outputs\HoloQPI_Project_Documentation.md"  docs\ >nul 2>nul
  rmdir /s /q "Claude outputs"
)
if exist QPI_Extended_Lit (
  if not exist docs\literature mkdir docs\literature
  move /y QPI_Extended_Lit\* docs\literature\ >nul 2>nul
  rmdir /s /q QPI_Extended_Lit
)

rem --- this study's own narration, kept with its logs ----------------------
if not exist logs\study mkdir logs\study
for %%F in (p_main.out p_rest.out p_seeds.out p_rest_eval.out p_finish.out) do (
  move /y %%F logs\study\ >nul 2>nul
)

rem --- transfer archives ---------------------------------------------------
del /q docs.zip 2>nul
del /q holoqpi_membrane_weights_20260914.zip 2>nul
del /q holoqpi_v2_changes_20260910.zip 2>nul
del /q holoqpi_v2_pipeline_20260914.zip 2>nul
del /q holoqpi_v2_server.tar.gz 2>nul

rem --- v1 drivers and runtime leftovers ------------------------------------
rem run_all.sh re-evaluated "all four experiments"; run_study.sh was the v1
rem five-stage study. Neither knows about the thirteen-arm design.
del /q run_all.sh 2>nul
del /q run_study.sh 2>nul
del /q .study_driver.sh 2>nul
del /q full.out 2>nul
del /q pretrain.out 2>nul
del /q reeval.out 2>nul
del /q study.out 2>nul

rem scripts\cleanup_legacy.py removed the paper-1 RBC framework, a job long
rem finished. Deleted because its LEGACY_DIRECTORIES list still contains
rem "weights" - running it today would delete the ImageNet checkpoint that
rem model.pretrained_dir points at, and every arm would then train from
rem scratch while the config still said pretrained_encoder: true.
del /q scripts\cleanup_legacy.py 2>nul

if exist scripts\__pycache__ rmdir /s /q scripts\__pycache__
if exist holoqpi\__pycache__ rmdir /s /q holoqpi\__pycache__
for /d %%D in (holoqpi\*) do if exist "%%D\__pycache__" rmdir /s /q "%%D\__pycache__"

rem --- smoke-test figures --------------------------------------------------
for %%F in (01_qualitative_panel 02_bland_altman 03_error_decomposition 04_detection_gap 05_modality_comparison 06_loss_composition 07_ablation_deltas 08_efficiency_tradeoff 09_phase_error_structure 11_confusion_matrices 12_convergence 13_forward_model_consistency 14_learned_vs_classical 15_recall_by_cell_size) do (
  del /q figures\fig%%F.png 2>nul
  del /q figures\fig%%F.pdf 2>nul
)

rem --- v1 smoke-test run directories and their companions ------------------
rem Every metrics file in these came from a QUICK run: phase_ssim negative,
rem seg_dice identical to sixteen digits across four different conditions,
rem detection_f1 = nan. /d matches directories only, so the top-level files
rem that share these prefixes - conventional_baseline_test.csv above all,
rem which IS a result from this study - are not touched.
for /d %%D in (runs\base_* runs\classification_only_* runs\conventional_* runs\fixed_threshold_* runs\no_measurement_* runs\no_physics_* runs\physics_coupling_only_*) do (
  rmdir /s /q "%%D"
)
if exist runs\null_probe rmdir /s /q runs\null_probe

del /q runs\*_modality_comparison.csv 2>nul
del /q runs\*_modality_comparison.json 2>nul
del /q runs\base_hardware_benchmark_cuda.csv 2>nul
del /q runs\base_hardware_benchmark_cuda.json 2>nul
del /q runs\seed_aggregate.csv 2>nul
del /q runs\seed_aggregate.json 2>nul
del /q runs\label_audit_redundancy.json 2>nul
del /q runs\mass_uncertainty.json 2>nul

rem Empty Gabor stubs the export step leaves behind; no Gabor arm was trained.
for /d %%D in (runs\v2_*_gabor) do rmdir "%%D" 2>nul

rem --- the final-epoch checkpoints -----------------------------------------
rem Only best_model.pt is ever read: evaluation, export, the benchmark and the
rem figures all load it. last_model.pt is kept by training.save_last and is
rem useful only to RESUME a stopped arm. About 2 GB, no result lost.
rem del cannot take a wildcard in a directory component, so walk them.
for /d %%D in (runs\*) do if exist "%%D\last_model.pt" del /q "%%D\last_model.pt"

rem --- old logs ------------------------------------------------------------
del /q logs\*.tar.gz 2>nul
for %%D in (20260903_034814 20260903_052408 study_20260903_170803 study_20260903_221043 study_20260905_024054 study_20260905_165413 study_20260906_061844) do (
  if exist "logs\%%D" rmdir /s /q "logs\%%D"
)
rem The v2 attempts before the real run began at 20:56 on 2026-09-14. KEPT:
rem v2_20260914_205603, v2_20260914_222427, v2_20260915_055247 and the four
rem stage logs from 07:03 onwards -- that is the study.
for %%D in (v2_20260914_153044 v2_20260914_153050 v2_20260914_161854 v2_20260914_163720 v2_20260914_164403 v2_20260914_170107 v2_20260914_173553) do (
  if exist "logs\%%D" rmdir /s /q "logs\%%D"
)

rem --- the pre-v2 archive --------------------------------------------------
rem Snapshots of the smoke-test runs\ and figures\ just removed, so keeping
rem them keeps the hazard.
if exist archive rmdir /s /q archive

echo   Done. The folder now holds:
echo.
dir /b
echo.
echo   docs\ has every document. runs\ and figures\ have the results.
echo.
