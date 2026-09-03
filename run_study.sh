#!/usr/bin/env bash
# =============================================================================
# HoloQPI - the complete study, in dependency order.
#
#   bash run_study.sh                  everything, from prepare to figures
#   bash run_study.sh --stage 3        one stage only
#   QUICK=1 bash run_study.sh          tiny settings, to prove the plumbing works
#
# Stages, and what each one blocks:
#
#   0  environment + self-test        nothing runs if the physics is wrong
#   1  prepare data                   masks, splits, manifest
#   2  calibrate z                    BLOCKS stage 5; the forward model needs it
#   3  label audit                    threshold drift, head redundancy
#   4  main comparison                off-axis vs Gabor, the central contribution
#   5  four-way physics ablation      needs z from stage 2
#   6  supporting ablations           no_measurement, classification_only
#   7  conventional baseline          the floor the network must beat
#   8  diagnostics                    bias decomposition
#   9  hardware benchmark             efficiency and edge suitability
#  10  figures                        reads everything above
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

# -- 2. calibrate z ------------------------------------------------------------
# The forward-model term needs the sample-to-sensor distance and it is recorded
# nowhere. This recovers it from the data by matching the hologram against the
# reference phase in both directions. If it reports "not identifiable", stage 5
# must not be trusted and the number has to come from the acquiring group.
if want 2; then
run_step "calibrate propagation distance z" "02_calibrate_z.log" \
  python scripts/calibrate_z.py --config config/base.yaml --refine

say ""
say "  >> READ ${LOGDIR}/02_calibrate_z.log BEFORE trusting stage 5."
say "     If z is identifiable, set loss.forward_model.distance_um in"
say "     config/base.yaml to the reported value and re-run from stage 5."
fi

# -- 3. label audit ------------------------------------------------------------
if want 3; then
run_step "label audit (threshold drift, head redundancy)" "03_audit_labels.log" \
  python scripts/audit_labels.py --config config/base.yaml ${EXTRA}
fi

# -- 4. the central comparison -------------------------------------------------
if want 4; then
run_step "modality comparison: off-axis vs Gabor" "04_compare_base.log" \
  python main.py compare --config config/base.yaml ${TRAIN_FLAG} ${EXTRA}
fi

# -- 5. four-way physics ablation ---------------------------------------------
# The experiment the study is actually about. no_physics is the floor;
# physics_coupling_only is the previous study's loss carried over unchanged;
# physics_forward_only is the constraint the reference literature means;
# physics_all is both. Arms 2 and 4 need loss.forward_model.distance_um.
if want 5; then
Z_SET=$(python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, ".")
from holoqpi.config import load_config
try:
    z = load_config("config/base.yaml").loss.forward_model.distance_um
except Exception:
    z = None
print("" if z is None else "ok")
PY
)
if [ -z "${Z_SET}" ]; then
  say ""
  say "  !! loss.forward_model.distance_um is null, so the two forward-model arms"
  say "     would train a term with no distance and fail. Running only the arms"
<<<<<<< Updated upstream
  say "     that do not need z. Set the distance and re-run: bash run_study.sh --stage 5"
=======
  say "     that do not need z."
  say "     To unblock, in order of preference:"
  say "       1. ask the acquiring group for the reconstruction distance;"
  say "       2. widen the calibration scan:"
  say "          python scripts/calibrate_z.py --config config/base.yaml \\"
  say "              --z-range -600 600 --steps 121 --feature-um 2 --refine"
  say "       3. or set an approximate distance_um and learn_distance: true, which"
  say "          refines z by gradient descent alongside the weights."
  say "     Then: bash run_study.sh --stage 5"
>>>>>>> Stashed changes
  run_step "ablation: no physics at all" "05a_no_physics.log" \
    python main.py compare --config config/ablation/no_physics.yaml ${TRAIN_FLAG} ${EXTRA}
  run_step "ablation: mask-phase coupling only" "05c_physics_coupling_only.log" \
    python main.py compare --config config/ablation/physics_coupling_only.yaml ${TRAIN_FLAG} ${EXTRA}
else
  run_step "ablation: no physics at all" "05a_no_physics.log" \
    python main.py compare --config config/ablation/no_physics.yaml ${TRAIN_FLAG} ${EXTRA}
  run_step "ablation: forward-model consistency only" "05b_physics_forward_only.log" \
    python main.py compare --config config/ablation/physics_forward_only.yaml ${TRAIN_FLAG} ${EXTRA}
  run_step "ablation: mask-phase coupling only" "05c_physics_coupling_only.log" \
    python main.py compare --config config/ablation/physics_coupling_only.yaml ${TRAIN_FLAG} ${EXTRA}
  run_step "ablation: every physics term" "05d_physics_all.log" \
    python main.py compare --config config/ablation/physics_all.yaml ${TRAIN_FLAG} ${EXTRA}
fi
fi

# -- 6. supporting ablations ---------------------------------------------------
if want 6; then
run_step "ablation: no measurement terms" "06a_no_measurement.log" \
  python main.py compare --config config/ablation/no_measurement.yaml ${TRAIN_FLAG} ${EXTRA}
run_step "control: classification head alone" "06b_classification_only.log" \
  python main.py compare --config config/ablation/classification_only.yaml ${TRAIN_FLAG} ${EXTRA}
fi

# -- 7. conventional baseline --------------------------------------------------
# The floor. Read the "reconstruction_valid" line in the log before quoting any
# of these numbers: a failed classical reconstruction is not a baseline.
if want 7; then
run_step "conventional reconstruction baseline" "07_conventional_baseline.log" \
  python scripts/conventional_baseline.py --config config/base.yaml ${EXTRA}
fi

# -- 8. diagnostics ------------------------------------------------------------
if want 8; then
run_step "dry-mass bias decomposition" "08_diagnose_bias.log" \
  python scripts/diagnose_bias.py --config config/base.yaml ${EXTRA}
fi

# -- 9. hardware benchmark -----------------------------------------------------
if want 9; then
: "${ORT_GPU_VERSION:=1.20.1}"
run_step "install onnxruntime-gpu==${ORT_GPU_VERSION}" "09a_onnxruntime.log" \
  bash -c "pip uninstall -y onnxruntime onnxruntime-gpu; pip install 'onnxruntime-gpu==${ORT_GPU_VERSION}'"
run_step "hardware benchmark" "09b_benchmark.log" \
  python main.py benchmark --config config/base.yaml ${EXTRA}
fi

# -- 10. figures ---------------------------------------------------------------
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
say "Read these three first:"
say "  ${LOGDIR}/02_calibrate_z.log        is z identifiable? gates the physics ablation"
say "  ${LOGDIR}/07_conventional_baseline.log   is the classical reconstruction valid?"
say "  ${LOGDIR}/03_audit_labels.log       does the label threshold drift with condition?"
say ""
say "Figures are in figures/ (PNG to read, PDF for the paper)."

cp -r figures "${LOGDIR}/figures" 2>/dev/null
ARCHIVE="logs/holoqpi_study_${STAMP}.tar.gz"
tar czf "${ARCHIVE}" -C logs "study_${STAMP}" 2>/dev/null \
  && say "" && say "Single archive to send: ${ARCHIVE}  ($(du -h "${ARCHIVE}" | cut -f1))"

exit 0
