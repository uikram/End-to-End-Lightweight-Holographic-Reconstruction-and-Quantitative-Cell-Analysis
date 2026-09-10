#!/usr/bin/env bash
# =============================================================================
# HoloQPI - the complete study, in dependency order.
#
#   CLEAN=1 bash run_study.sh          the full study, from scratch  <-- use this
#   bash run_study.sh --stage 5        re-run one stage only
#   QUICK=1 bash run_study.sh          tiny settings, to prove the plumbing works
#   NO_TRAIN=1 bash run_study.sh       reuse checkpoints, re-evaluate only
#   SKIP_BENCH=1 bash run_study.sh     skip the ONNX / hardware benchmark
#   SKIP_THRESHOLD=1 bash run_study.sh skip the fixed-threshold retrain
#   SEEDS="" bash run_study.sh         skip seed replication (the slow part)
#   SEEDS="1337 2024 7" bash run_study.sh   replicate over these seeds
#
# CLEAN=1 archives runs/ and figures/ under archive/ before starting. Use it
# whenever the objective has changed: a checkpoint trained under a different
# loss is not comparable with one trained under this loss.
#
# EXPECT ROUGHLY 12-16 HOURS for the main arms, plus about 4 h for the
# fixed-threshold arm and about 10 h per replication seed. With the default
# two extra seeds, budget roughly a day and a half; SEEDS="" cuts it back.
# Run it under nohup so a dropped connection does not kill it:
#
#   CLEAN=1 nohup bash run_study.sh > study.out 2>&1 &
#   tail -f study.out
#
# Stages, and what each one blocks:
#
#   0  environment + self-test        nothing runs if the physics is wrong
#   1  prepare data                   masks, splits, manifest, aberration surfaces
#   2  calibrate z                    BLOCKS stage 5; the forward model needs it
#   3  label audit                    threshold drift, head redundancy
#   4  main comparison                off-axis vs Gabor, the central contribution
#   5  four-way physics ablation      needs z from stage 2
#   6  supporting ablations           no_measurement, classification_only
#   7  conventional baseline          the floor the network must beat
#   8  diagnostics                    bias decomposition
#   9  hardware benchmark             efficiency and edge suitability
#  10  figures                        reads everything above
#  11  absolute-mass uncertainty      the alpha band on absolute picograms
#  12  fixed-threshold sensitivity    answers the label/condition confound
#  13  seed replication               error bars on the ablation table (SLOW)
#
# Every stage writes its own log under logs/<timestamp>/ and the script keeps
# going when one fails, so a single failure does not hide the rest. Send back
# the archive it prints at the end.
# =============================================================================

cd "$(dirname "$0")" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/study_${STAMP}"
mkdir -p "${LOGDIR}"
SUMMARY="${LOGDIR}/SUMMARY.txt"

: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

ONLY_STAGE=""
if [ "$1" = "--stage" ]; then ONLY_STAGE="$2"; fi

EXTRA=""
TRAIN_FLAG=""
if [ -n "${QUICK}" ]; then
  EXTRA="--set training.epochs=2 data.eval_size=192 data.train_crop=192 data.num_workers=0 deploy.benchmark.warmup_runs=2 deploy.benchmark.timed_runs=3 deploy.benchmark.input_size=192"
fi
# NO_TRAIN=1 reuses existing checkpoints and only re-evaluates.
if [ -n "${NO_TRAIN}" ]; then TRAIN_FLAG="--no-train"; fi

declare -a STEP_NAME
declare -a STEP_CODE

say() { echo "$@" | tee -a "${SUMMARY}"; }

want() { [ -z "${ONLY_STAGE}" ] || [ "${ONLY_STAGE}" = "$1" ]; }

run_step() {
  local label="$1"; shift
  local logfile="$1"; shift
  local started; started=$(date +%s)

  echo "" | tee -a "${SUMMARY}"
  say "--------------------------------------------------------------------"
  say "STEP: ${label}"
  say "  cmd: $*"
  say "  log: ${logfile}"

  "$@" > "${LOGDIR}/${logfile}" 2>&1
  local code=$?
  local elapsed=$(( $(date +%s) - started ))

  STEP_NAME+=("${label}"); STEP_CODE+=("${code}")
  if [ ${code} -eq 0 ]; then
    say "  result: OK (${elapsed}s)"
  else
    say "  result: FAILED with exit code ${code} (${elapsed}s)"
    say "  last 15 lines:"
    tail -n 15 "${LOGDIR}/${logfile}" | sed 's/^/    /' | tee -a "${SUMMARY}"
  fi
  return 0
}

# -- optional clean slate ------------------------------------------------------
# CLEAN=1 archives previous results and starts from nothing. Necessary whenever
# the loss, the dataloader or the checkpoint-selection rule has changed, because
# a checkpoint trained under a different objective is not comparable with one
# trained under this objective, and `compare --no-train` would silently
# re-evaluate the stale weights as if they belonged to the new experiment.
if [ -n "${CLEAN}" ]; then
  ARCHIVE_DIR="archive/before_${STAMP}"
  mkdir -p "${ARCHIVE_DIR}"
  moved=0
  for old in runs figures; do
    if [ -e "${old}" ]; then mv "${old}" "${ARCHIVE_DIR}/"; moved=$((moved + 1)); fi
  done
  # The aberration surfaces are regenerated in stage 1b and the stored file is
  # the one thing outside runs/ that a stale copy could silently poison: the
  # forward-model term and the calibration both read it, and a file written
  # before the conjugate fix carries the wrong sign on most fields.
  if [ -f data/aberration.json ]; then
    mv data/aberration.json "${ARCHIVE_DIR}/aberration.json"; moved=$((moved + 1))
  fi
  # Old study logs, kept but out of the way, so logs/ holds only this run.
  if compgen -G "logs/study_*" > /dev/null; then
    mkdir -p "${ARCHIVE_DIR}/logs"
    for d in logs/study_*; do
      [ "${d}" = "${LOGDIR}" ] || mv "${d}" "${ARCHIVE_DIR}/logs/" 2>/dev/null
    done
    moved=$((moved + 1))
  fi
  echo "CLEAN: moved ${moved} old output folder(s) into ${ARCHIVE_DIR}/"
  echo "       Nothing is deleted. Remove that folder yourself once the new run"
  echo "       has finished and you are satisfied with it."
fi

say "===================================================================="
say " HoloQPI full study - ${STAMP}"
say "===================================================================="
say " working directory : $(pwd)"
say " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
say " QUICK=${QUICK:-off}   NO_TRAIN=${NO_TRAIN:-off}   stage=${ONLY_STAGE:-all}"

# -- 0. environment and self-test ---------------------------------------------
if want 0; then
{
  echo "=== date ==="; date
  echo; echo "=== python ==="; python -V; which python
  echo; echo "=== torch / GPU ==="
  python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0 name  :", torch.cuda.get_device_name(0))
try:
    import onnxruntime as ort
    print("onnxruntime    :", ort.__version__, ort.get_available_providers())
except Exception as exc:
    print("onnxruntime    : NOT IMPORTABLE:", exc)
for name in ("matplotlib", "skimage", "scipy"):
    try:
        module = __import__(name)
        print(f"{name:<15}:", getattr(module, "__version__", "?"))
    except Exception as exc:
        print(f"{name:<15}: MISSING ({exc})")
PY
  echo; echo "=== nvidia-smi ==="; nvidia-smi 2>&1 | head -20
  echo; echo "=== installed packages ==="; pip list 2>/dev/null
} > "${LOGDIR}/00_environment.log" 2>&1
say ""; say "environment captured -> ${LOGDIR}/00_environment.log"

run_step "self-test (measurement chain, losses, forward model)" "00_selftest.log" \
  python scripts/selftest.py --config config/base.yaml
fi

# -- 1. prepare ----------------------------------------------------------------
if want 1; then
run_step "prepare data (masks, splits, manifest)" "01_prepare.log" \
  python main.py prepare --config config/base.yaml
fi

# -- 1b. aberration surfaces ---------------------------------------------------
# The delivered phase maps were aberration-corrected, background-subtracted and
# unwrapped before delivery, so they are not raw reconstructions. The forward
# model has to restore what was removed before it can reproduce the recorded
# hologram. This fits that surface per field and reports how well it explains
# the difference.
if want 1; then
run_step "estimate aberration surfaces" "01b_aberration.log" \
  python scripts/estimate_aberration.py --config config/base.yaml
fi

# -- 2. calibrate z ------------------------------------------------------------
# The forward-model term needs the sample-to-sensor distance and it is recorded
# nowhere. This recovers it from the data by matching the hologram against the
# reference phase in both directions. If it reports "not identifiable", stage 5
# must not be trusted and the number has to come from the acquiring group.
if want 2; then
# No --z-range and no --feature-um: the default is the largest |z| a 900 px
# field can model without the diffraction wrapping around, and every distance is
# scored on one fixed window sized for that. A wider range was previously passed
# here, which asked for a border wider than the field and quietly disabled the
# fixed window altogether. --refine is dropped for the same reason: z comes from
# the acquiring group, so the scan is a diagnostic and a check, not a search.
run_step "calibrate propagation distance z" "02_calibrate_z.log" \
  python scripts/calibrate_z.py --config config/base.yaml --steps "${Z_STEPS:-101}"

say ""
say "  >> ${LOGDIR}/02_calibrate_z.log is the first log to read."
fi

# The distance decides how stage 5 can run, so resolve it once, here, and record
# the decision in the summary. Three cases, in descending order of trust:
#
#   fixed     an identifiable z, or one set by hand in config/base.yaml. Best.
#   learned   z not identifiable, so it is initialised at the forward minimum
#             and refined by gradient descent during training. Defensible, and
#             the converged value is logged every epoch.
#   skip      no usable starting point; the forward-model arms are left out
#             rather than trained against a meaningless distance.
Z_MODE="skip"; Z_VALUE=""
read -r Z_MODE Z_VALUE <<< "$(python scripts/resolve_z.py)"

Z_ARGS=""
case "${Z_MODE}" in
  fixed)
    Z_ARGS="loss.forward_model.distance_um=${Z_VALUE}"
    say ""
    say "  z = ${Z_VALUE} um, held FIXED for the forward-model arms."
    ;;
  learned)
    Z_ARGS="loss.forward_model.distance_um=${Z_VALUE} loss.forward_model.learn_distance=true"
    say ""
    say "  z was NOT identifiable by grid search. Initialising at ${Z_VALUE} um and"
    say "  refining it by gradient descent during training (learn_distance: true)."
    say "  The converged value is logged each epoch as train_forward_distance_um;"
    say "  report it in the paper."
    ;;
  *)
    say ""
    say "  !! No usable propagation distance. The two forward-model arms will be"
    say "     SKIPPED rather than trained against a meaningless z. Ask the"
    say "     acquiring group for the reconstruction distance, then run:"
    say "        bash run_study.sh --stage 5"
    ;;
esac

# -- 4. the central comparison -------------------------------------------------
if want 4; then
# The base arm does not OPTIMISE the forward-model term, but it is SCORED on it
# whenever a distance is available, so the ablation table can show what
# optimising it actually buys rather than only that it was optimised.
BASE_Z=""
if [ -n "${Z_VALUE}" ]; then
  BASE_Z="--set loss.forward_model.distance_um=${Z_VALUE}"
fi
run_step "modality comparison: off-axis vs Gabor" "04_compare_base.log" \
  python main.py compare --config config/base.yaml ${TRAIN_FLAG} ${EXTRA} ${BASE_Z}
fi

# -- 5. four-way physics ablation ---------------------------------------------
# The experiment the study is actually about. no_physics is the floor;
# physics_coupling_only is the previous study's loss carried over unchanged;
# physics_forward_only is the constraint the reference literature means;
# physics_all is both. Arms 2 and 4 need loss.forward_model.distance_um.
if want 5; then
ZSET=""
[ -n "${Z_ARGS}" ] && ZSET="--set ${Z_ARGS}"

# These two need no distance and always run.
run_step "ablation: no physics at all" "05a_no_physics.log" \
  python main.py compare --config config/ablation/no_physics.yaml ${TRAIN_FLAG} ${EXTRA} ${BASE_Z}
run_step "ablation: mask-phase coupling only" "05c_physics_coupling_only.log" \
  python main.py compare --config config/ablation/physics_coupling_only.yaml ${TRAIN_FLAG} ${EXTRA} ${BASE_Z}

# The forward-model term is trained only on the arms where it discriminates in
# the right direction. On this data the operator is approximate -- the
# acquisition software's reconstruction cannot be inverted -- and that turns out
# to matter for one geometry and not the other, so the decision is per modality
# rather than global.
FWD_MODALITIES=$(python scripts/resolve_z.py --modalities)
if [ -n "${Z_ARGS}" ] && [ -n "${FWD_MODALITIES}" ]; then
  say ""
  say "  forward-model arms will train on: ${FWD_MODALITIES}"
  run_step "ablation: forward-model consistency only" "05b_physics_forward_only.log" \
    python main.py compare --config config/ablation/physics_forward_only.yaml \
      --modalities ${FWD_MODALITIES} ${TRAIN_FLAG} ${EXTRA} ${ZSET}
  run_step "ablation: every physics term" "05d_physics_all.log" \
    python main.py compare --config config/ablation/physics_all.yaml \
      --modalities ${FWD_MODALITIES} ${TRAIN_FLAG} ${EXTRA} ${ZSET}
else
  say ""
  say "  (forward-model arms skipped: the term does not discriminate on any arm,"
  say "   see the discrimination test in stage 2)"
fi
fi

# -- 6. supporting ablations ---------------------------------------------------
if want 6; then
run_step "ablation: no measurement terms" "06a_no_measurement.log" \
  python main.py compare --config config/ablation/no_measurement.yaml ${TRAIN_FLAG} ${EXTRA}
run_step "control: classification head alone" "06b_classification_only.log" \
  python main.py compare --config config/ablation/classification_only.yaml ${TRAIN_FLAG} ${EXTRA}
fi

# -- 3. label audit ------------------------------------------------------------
# AFTER the training stages, not before them. Its second section measures how
# much the segmentation head duplicates a threshold of its own predicted phase,
# and that needs a checkpoint: run earlier it prints "no checkpoints found;
# skipped" and the redundancy question -- which is the one that decides whether
# two heads are earning their keep -- goes unanswered while the stage still
# reports OK.
if want 3; then
run_step "label audit (threshold drift, head redundancy)" "03_audit_labels.log" \
  python scripts/audit_labels.py --config config/base.yaml ${EXTRA}
fi

# -- 7. conventional baseline --------------------------------------------------
# The floor. Read the "reconstruction_valid" line in the log before quoting any
# of these numbers: a failed classical reconstruction is not a baseline.
if want 7; then
BASELINE_Z=""
[ -n "${Z_VALUE}" ] && BASELINE_Z="--distance-um ${Z_VALUE}"
run_step "conventional reconstruction baseline" "07_conventional_baseline.log" \
  python scripts/conventional_baseline.py --config config/base.yaml ${EXTRA} ${BASELINE_Z}
fi

# -- 8. diagnostics ------------------------------------------------------------
if want 8; then
run_step "dry-mass bias decomposition" "08_diagnose_bias.log" \
  python scripts/diagnose_bias.py --config config/base.yaml ${EXTRA}
fi

# -- 9. hardware benchmark -----------------------------------------------------
if want 9 && [ -z "${SKIP_BENCH}" ]; then
# 1.20.1 is not published on this index. 1.20.2 is the CUDA 12 / cuDNN 9 build
# that matches torch cu124; 1.20.0 is the fallback. Anything from 1.21 upward is
# built against CUDA 13 and will report a CUDA provider while silently running
# on CPU, which is worse than not installing it at all.
: "${ORT_GPU_VERSION:=1.20.2}"
run_step "install onnxruntime-gpu (CUDA 12 build)" "09a_onnxruntime.log" \
  bash -c "pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1; \
           pip install 'onnxruntime-gpu==${ORT_GPU_VERSION}' || \
           pip install 'onnxruntime-gpu==1.20.0' || \
           pip install 'onnxruntime-gpu<1.21'"
run_step "hardware benchmark" "09b_benchmark.log" \
  python main.py benchmark --config config/base.yaml ${EXTRA}
fi

# -- 11. absolute-mass systematic uncertainty ---------------------------------
# Cheap, and it settles a question that would otherwise be answered wrongly.
# alpha cancels from every RELATIVE dry-mass quantity, so it explains none of the
# reported mass error; it does set the scale of every absolute picogram value.
# This prints the band to quote beside absolute masses, and demonstrates the
# invariance on the measured cells rather than asserting it.
if want 11; then
run_step "absolute dry-mass uncertainty from alpha" "11_mass_uncertainty.log" \
  python scripts/mass_uncertainty.py --config config/base.yaml ${EXTRA}
fi

# -- 12. fixed-threshold label sensitivity ------------------------------------
# The label audit shows the per-image Otsu level separates by drug condition
# (F = 15.27, p = 4.9e-12), so the label definition is partly confounded with the
# class the network predicts. This arm rebuilds the masks at one population-level
# threshold, where that confound cannot exist, and retrains. Compare its
# classification accuracy with the base arm: the drop, if any, is how much of the
# base number came from the label definition rather than from the specimen.
#
# Masks are written to data/mask_fixed/ so the main labels are untouched.
if want 12 && [ -z "${SKIP_THRESHOLD}" ]; then
run_step "prepare fixed-threshold masks" "12a_prepare_fixed_threshold.log" \
  python main.py prepare --config config/ablation/fixed_threshold.yaml
run_step "sensitivity: fixed label threshold" "12b_fixed_threshold.log" \
  python main.py compare --config config/ablation/fixed_threshold.yaml ${TRAIN_FLAG} ${EXTRA}
fi

# -- 13. seed replication ------------------------------------------------------
# The one stage that turns "the physics terms are null" from an assertion into a
# measurement. Every ablation conclusion here rests on differences below 1%, plus
# classification differences of about 5 points that point in OPPOSITE directions
# between the two modalities. Neither can be read off a single run.
#
# EXPENSIVE: each seed retrains every arm, roughly ten hours per seed. Set
# SEEDS="" to skip it; NO_TRAIN=1 re-evaluates existing seed runs instead.
: "${SEEDS=$(python -c "import sys; sys.path.insert(0, '.');
from holoqpi.config import load_config;
print(' '.join(str(s) for s in load_config('config/base.yaml').evaluation.seed_replication.seeds))" 2>/dev/null)}"

if want 13 && [ -n "${SEEDS}" ]; then
  say ""
  say "  seed replication over: ${SEEDS}"
  say "  (each seed retrains every arm; expect roughly 10 h per seed)"
  for seed in ${SEEDS}; do
    for arm in base no_physics physics_coupling_only no_measurement; do
      if [ "${arm}" = "base" ]; then
        arm_cfg="config/base.yaml"
      else
        arm_cfg="config/ablation/${arm}.yaml"
      fi
      run_step "seed ${seed}: ${arm}" "13_${arm}_s${seed}.log" \
        python main.py compare --config "${arm_cfg}" ${TRAIN_FLAG} ${EXTRA} \
          --set "project.seed=${seed}" "experiment_name=${arm}_s${seed}" \
                "loss.forward_model.distance_um=${Z_VALUE:-33.77}"
    done
  done
fi

# Runs whenever more than one repeat exists, and says plainly when nothing can be
# resolved -- which for this study is the expected, publishable answer.
if want 13; then
run_step "pool seeds and test which differences survive" "13_seed_aggregate.log" \
  python scripts/aggregate_seeds.py --config config/base.yaml
fi

# -- 10. figures ---------------------------------------------------------------
# Deliberately last in the file even though it is numbered 10: it reads every
# result produced above, including the seed replication and the fixed-threshold
# arm, so it must not run before them.
if want 10; then
run_step "figures" "10_figures.log" \
  python scripts/make_figures.py --config config/base.yaml ${EXTRA}
fi

# -- collect -------------------------------------------------------------------
{
  echo "########## MODALITY COMPARISON TABLES ##########"
  for f in runs/*_modality_comparison.csv; do
    [ -f "${f}" ] && { echo; echo "===== ${f} ====="; cat "${f}"; }
  done
  echo; echo "########## CONVENTIONAL BASELINE ##########"
  for f in runs/conventional_baseline_*.csv; do
    [ -f "${f}" ] && { echo; echo "===== ${f} ====="; cat "${f}"; }
  done
  echo; echo "########## LABEL AUDIT ##########"
  for f in runs/label_audit_*.json runs/z_calibration.json; do
    [ -f "${f}" ] && { echo; echo "===== ${f} ====="; head -c 4000 "${f}"; }
  done
  echo; echo "########## HARDWARE BENCHMARK ##########"
  for f in runs/*_hardware_benchmark_*.csv; do
    [ -f "${f}" ] && { echo; echo "===== ${f} ====="; cat "${f}"; }
  done
  echo; echo "########## FULL TEST METRICS PER RUN ##########"
  for d in runs/*/; do
    [ -f "${d}metrics_test.json" ] && { echo; echo "===== ${d}metrics_test.json ====="; cat "${d}metrics_test.json"; }
  done
  echo; echo "########## CONFUSION MATRICES ##########"
  for d in runs/*/; do
    [ -f "${d}confusion_test.json" ] && { echo; echo "===== ${d}confusion_test.json ====="; cat "${d}confusion_test.json"; }
  done
  echo; echo "########## RESOLVED CONFIGS ##########"
  for d in runs/*/; do
    [ -f "${d}resolved_config.yaml" ] && { echo; echo "===== ${d}resolved_config.yaml ====="; cat "${d}resolved_config.yaml"; }
  done
} > "${LOGDIR}/90_all_results.txt" 2>&1
say ""; say "all results concatenated -> ${LOGDIR}/90_all_results.txt"

python - > "${LOGDIR}/91_convergence.txt" 2>&1 <<'PY'
import json, glob, os
for path in sorted(glob.glob("runs/*/history.json")):
    run = os.path.basename(os.path.dirname(path))
    try:
        h = json.load(open(path))
    except Exception as exc:
        print(f"{run}: could not read history ({exc})"); continue
    if not h: continue
    print(f"\n=== {run}  ({len(h)} epochs) ===")
    keys = [k for k in ("val_seg_dice", "val_phase_mae_rad", "val_cls_accuracy",
                        "val_dry_mass_mape", "val_detection_f1", "train_total") if k in h[0]]
    print("  ep  " + "  ".join(f"{k[:18]:>18s}" for k in keys))
    for e in h[::10] + [h[-1]]:
        print(f"  {e['epoch']:>3} " + "  ".join(f"{e.get(k, float('nan')):18.4f}" for k in keys))
    last = h[-1]
    comp = {k: v for k, v in last.items() if k.startswith("train_") and k != "train_total"}
    if comp:
        print("  final loss components:", {k[6:]: round(v, 5) for k, v in comp.items()})
PY
say "convergence traces -> ${LOGDIR}/91_convergence.txt"

# -- verdict -------------------------------------------------------------------
say ""; say "===================================================================="
say " SUMMARY"; say "===================================================================="
FAILED=0
for i in "${!STEP_NAME[@]}"; do
  if [ "${STEP_CODE[$i]}" -eq 0 ]; then
    say "  OK      ${STEP_NAME[$i]}"
  else
    say "  FAILED  ${STEP_NAME[$i]}  (exit ${STEP_CODE[$i]})"
    FAILED=$((FAILED + 1))
  fi
done
say ""
[ ${FAILED} -eq 0 ] && say "All stages completed." || say "${FAILED} stage(s) failed - see the logs above."

say ""
say "Read these five first:"
say "  ${LOGDIR}/01b_aberration.log        how many fields needed the conjugate flip,"
say "                                      and how many surfaces were rejected"
say "  ${LOGDIR}/02_calibrate_z.log        does the in-line minimum land on the supplied z,"
say "                                      and what verdict does the forward-model term get"
say "  ${LOGDIR}/07_conventional_baseline.log   did the classical reconstruction recover"
say "                                      positive cell contrast in each geometry"
say "  ${LOGDIR}/03_audit_labels.log       does the label threshold drift with condition,"
say "                                      and is the segmentation head redundant"
say "  ${LOGDIR}/13_seed_aggregate.log     which ablation differences survive the spread"
say ""
say "Figures are in figures/ (PNG to read, PDF for the paper)."

cp -r figures "${LOGDIR}/figures" 2>/dev/null
ARCHIVE="logs/holoqpi_study_${STAMP}.tar.gz"
tar czf "${ARCHIVE}" -C logs "study_${STAMP}" 2>/dev/null \
  && say "" && say "Single archive to send: ${ARCHIVE}  ($(du -h "${ARCHIVE}" | cut -f1))"

exit 0
