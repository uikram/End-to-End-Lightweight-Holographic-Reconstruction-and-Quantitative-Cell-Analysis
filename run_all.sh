#!/usr/bin/env bash
# =============================================================================
# HoloQPI - run everything, log everything.
#
#   bash run_all.sh
#
# Re-evaluates all four experiments with the corrected measurement metrics,
# re-runs the bias diagnosis, and re-runs the hardware benchmark after trying to
# get ONNX onto the GPU. Uses the checkpoints that already exist, so it does not
# retrain anything.
#
# Every step writes its own log under logs/<timestamp>/ and the script keeps
# going when a step fails, so one failure does not hide the rest. Zip that
# folder and send it back.
#
#   QUICK=1 bash run_all.sh     tiny settings, just to prove the plumbing works
# =============================================================================

cd "$(dirname "$0")" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "${LOGDIR}"
SUMMARY="${LOGDIR}/SUMMARY.txt"

: "${CUDA_VISIBLE_DEVICES:=2}"
export CUDA_VISIBLE_DEVICES

# QUICK mode shrinks the work so the script can be smoke-tested in a minute.
EXTRA=""
if [ -n "${QUICK}" ]; then
  EXTRA="--set data.eval_size=192 data.num_workers=0 deploy.benchmark.warmup_runs=2 deploy.benchmark.timed_runs=3 deploy.benchmark.input_size=192"
fi

declare -a STEP_NAME
declare -a STEP_CODE

say() { echo "$@" | tee -a "${SUMMARY}"; }

run_step() {
  # run_step <label> <logfile> <command...>
  local label="$1"; shift
  local logfile="$1"; shift
  local started
  started=$(date +%s)

  echo "" | tee -a "${SUMMARY}"
  say "--------------------------------------------------------------------"
  say "STEP: ${label}"
  say "  cmd: $*"
  say "  log: ${logfile}"

  "$@" > "${LOGDIR}/${logfile}" 2>&1
  local code=$?
  local elapsed=$(( $(date +%s) - started ))

  STEP_NAME+=("${label}")
  STEP_CODE+=("${code}")

  if [ ${code} -eq 0 ]; then
    say "  result: OK (${elapsed}s)"
  else
    say "  result: FAILED with exit code ${code} (${elapsed}s)"
    say "  last 15 lines:"
    tail -n 15 "${LOGDIR}/${logfile}" | sed 's/^/    /' | tee -a "${SUMMARY}"
  fi
  return 0
}

# -----------------------------------------------------------------------------
say "===================================================================="
say " HoloQPI full run - ${STAMP}"
say "===================================================================="
say " working directory : $(pwd)"
say " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
say " QUICK mode        : ${QUICK:-off}"

# -- 0. environment ------------------------------------------------------------
{
  echo "=== date ==="; date
  echo; echo "=== python ==="; python -V; which python
  echo; echo "=== torch / GPU ==="
  python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count   :", torch.cuda.device_count())
    print("device 0 name  :", torch.cuda.get_device_name(0))
    print("capability     :", torch.cuda.get_device_capability(0))
try:
    import onnxruntime as ort
    print("onnxruntime    :", ort.__version__)
    print("providers      :", ort.get_available_providers())
except Exception as exc:
    print("onnxruntime    : NOT IMPORTABLE:", exc)
PY
  echo; echo "=== nvidia-smi ==="; nvidia-smi 2>&1 | head -20
  echo; echo "=== installed packages ==="; pip list 2>/dev/null
} > "${LOGDIR}/00_environment.log" 2>&1
say ""
say "environment captured -> ${LOGDIR}/00_environment.log"

# -- 1. self-test --------------------------------------------------------------
run_step "self-test (verify the corrected code)" "01_selftest.log" \
  python scripts/selftest.py --config config/base.yaml

# -- 2. re-evaluate every experiment with the corrected metrics ----------------
# --no-train reuses the existing checkpoints and only redoes evaluation, so this
# rewrites metrics_test.json, per_cell_test.csv and the comparison tables with
# the per-image statistics computed over the matched population.
run_step "re-evaluate: base" "02_eval_base.log" \
  python main.py compare --config config/base.yaml --no-train ${EXTRA}

run_step "re-evaluate: no_physics" "03_eval_no_physics.log" \
  python main.py compare --config config/ablation/no_physics.yaml --no-train ${EXTRA}

run_step "re-evaluate: no_measurement" "04_eval_no_measurement.log" \
  python main.py compare --config config/ablation/no_measurement.yaml --no-train ${EXTRA}

run_step "re-evaluate: classification_only" "05_eval_classification_only.log" \
  python main.py compare --config config/ablation/classification_only.yaml --no-train ${EXTRA}

# -- 3. diagnostics ------------------------------------------------------------
run_step "dry-mass bias decomposition" "06_diagnose_bias.log" \
  python scripts/diagnose_bias.py --config config/base.yaml ${EXTRA}

# Two properties of the silver-standard labels that have to be reported rather
# than assumed: how much the per-image Otsu level moves (and whether it moves
# with drug condition, which would confound the classification result), and how
# much of the segmentation head is already implied by the phase head.
run_step "label audit: threshold stability and head redundancy" "06b_audit_labels.log" \
  python scripts/audit_labels.py --config config/base.yaml ${EXTRA}

# -- 4. get ONNX onto the GPU, then benchmark ---------------------------------
# The previous benchmark timed ONNX on CPU against PyTorch on the GPU, because
# the CPU-only onnxruntime wheel was installed. Swapping in onnxruntime-gpu is
# best-effort: if it fails the benchmark still runs and now flags the mismatch
# in a comparable_to_pytorch_row column instead of hiding it.
# The version matters: onnxruntime-gpu 1.28 is built against CUDA 13 and fails to
# load libcublasLt.so.13 on a CUDA 12 stack, reporting CUDAExecutionProvider as
# "available" while silently running on CPU. 1.20.1 is the CUDA 12 / cuDNN 9
# build, which matches torch cu124.
: "${ORT_GPU_VERSION:=1.20.1}"
run_step "install onnxruntime-gpu==${ORT_GPU_VERSION} (CUDA 12 build)" "07_onnxruntime_gpu.log" \
  bash -c "pip uninstall -y onnxruntime onnxruntime-gpu; pip install 'onnxruntime-gpu==${ORT_GPU_VERSION}'"

{
  echo "=== providers visible after the install attempt ==="
  python - <<'PY'
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
    print("providers  :", ort.get_available_providers())
    print("CUDA usable:", "CUDAExecutionProvider" in ort.get_available_providers())
except Exception as exc:
    print("onnxruntime NOT IMPORTABLE:", exc)
PY
} > "${LOGDIR}/08_providers_after.log" 2>&1
say ""
say "ONNX providers after install -> ${LOGDIR}/08_providers_after.log"

run_step "hardware benchmark" "09_benchmark.log" \
  python main.py benchmark --config config/base.yaml ${EXTRA}

# -- 5. collect the results ----------------------------------------------------
{
  echo "########## MODALITY COMPARISON TABLES ##########"
  for f in runs/*_modality_comparison.csv; do
    echo; echo "===== ${f} ====="; cat "${f}"
  done

  echo; echo "########## HARDWARE BENCHMARK ##########"
  for f in runs/*_hardware_benchmark_*.csv; do
    echo; echo "===== ${f} ====="; cat "${f}"
  done

  echo; echo "########## FULL TEST METRICS PER RUN ##########"
  for d in runs/*/; do
    if [ -f "${d}metrics_test.json" ]; then
      echo; echo "===== ${d}metrics_test.json ====="; cat "${d}metrics_test.json"
    fi
  done

  echo; echo "########## CONFUSION MATRICES ##########"
  for d in runs/*/; do
    if [ -f "${d}confusion_test.json" ]; then
      echo; echo "===== ${d}confusion_test.json ====="; cat "${d}confusion_test.json"
    fi
  done

  echo; echo "########## RESOLVED CONFIGS (loss weights actually used) ##########"
  for d in runs/*/; do
    if [ -f "${d}resolved_config.yaml" ]; then
      echo; echo "===== ${d}resolved_config.yaml ====="; cat "${d}resolved_config.yaml"
    fi
  done
} > "${LOGDIR}/10_all_results.txt" 2>&1
say ""
say "all result files concatenated -> ${LOGDIR}/10_all_results.txt"

# Convergence trace, so the training history can be read without the big files.
python - > "${LOGDIR}/11_convergence.txt" 2>&1 <<'PY'
import json, glob, os
for path in sorted(glob.glob("runs/*/history.json")):
    run = os.path.basename(os.path.dirname(path))
    try:
        h = json.load(open(path))
    except Exception as exc:
        print(f"{run}: could not read history ({exc})"); continue
    print(f"\n=== {run}  ({len(h)} epochs) ===")
    keys = [k for k in ("val_seg_dice", "val_phase_mae_rad", "val_cls_accuracy",
                        "val_dry_mass_mape", "train_total") if k in h[0]]
    print("  ep  " + "  ".join(f"{k[:18]:>18s}" for k in keys))
    for e in h[::10] + [h[-1]]:
        print(f"  {e['epoch']:>3} " + "  ".join(f"{e.get(k, float('nan')):18.4f}" for k in keys))
    last = h[-1]
    comp = {k: v for k, v in last.items() if k.startswith("train_") and k != "train_total"}
    if comp:
        print("  final loss components:", {k[6:]: round(v, 5) for k, v in comp.items()})
PY
say "convergence traces -> ${LOGDIR}/11_convergence.txt"

# -- 5b. figures ---------------------------------------------------------------
# Built last, because every figure reads a file an earlier step produced. One
# bad figure does not stop the rest; each prints its own reason and is skipped.
run_step "publication figures" "12_figures.log" \
  python scripts/make_figures.py --config config/base.yaml ${EXTRA}

{
  echo "=== figures/ ==="
  ls -la figures 2>&1
} > "${LOGDIR}/13_figure_listing.txt" 2>&1
say "figure listing -> ${LOGDIR}/13_figure_listing.txt"

# -- 6. verdict ----------------------------------------------------------------
say ""
say "===================================================================="
say " SUMMARY"
say "===================================================================="
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
if [ ${FAILED} -eq 0 ]; then
  say "All steps completed."
else
  say "${FAILED} step(s) failed - see the individual logs above."
fi

say ""
say "Send back this whole folder:  ${LOGDIR}"
say "  00_environment.log      python, torch, GPU, onnxruntime providers, packages"
say "  01..09_*.log            one log per step"
say "  10_all_results.txt      every table, metric and resolved config"
say "  11_convergence.txt      per-epoch traces and final loss components"
say "  12_figures.log          figure generation"
say "  13_figure_listing.txt   what landed in figures/"
say "  SUMMARY.txt             this summary"
say ""
say "The figures themselves are in  figures/  (PNG for reading, PDF for the paper)."

# Bundle it, so there is one file to send. The figures ride along: they are the
# part worth looking at first and they are small.
ARCHIVE="logs/holoqpi_run_${STAMP}.tar.gz"
cp -r figures "logs/${STAMP}/figures" 2>/dev/null
tar czf "${ARCHIVE}" -C logs "${STAMP}" 2>/dev/null \
  && say "" \
  && say "Single archive to send: ${ARCHIVE}  ($(du -h "${ARCHIVE}" | cut -f1))"

exit 0
