#!/usr/bin/env bash
#
# ONE command for the whole v2 study.
#
#     bash study.sh              stop anything running, then run the study and watch it
#     bash study.sh --status     re-attach the watcher to a study already running
#     bash study.sh --stop       stop everything and exit
#
# WHAT IT DOES, IN ORDER
#   1. stops every process left over from a previous run and waits for the GPUs
#   2. runs the no-GPU preparation stages, but only if their outputs are missing
#   3. trains both groups of arms in parallel, one group per GPU
#   4. re-evaluates every arm, which is what rebuilds unmatched_<split>.csv
#   5. runs stages 7-10: baseline, export, figures, runs/RESULTS.md
#   6. shows you where it is, the whole time
#
# WHY IT IS BUILT THIS WAY
#
# The work runs DETACHED and the terminal only watches it. Ctrl-C stops the
# watching, not the study, and closing the terminal or losing the ssh
# connection does not either. That is the one thing tmux was being used for,
# and it is not needed for this.
#
# The driver is a COPY of this file taken at launch (.study_driver.sh). Bash
# reads a script incrementally as it executes, so editing a running script
# corrupts it mid-run -- that happened once already in this project and
# produced `ig: command not found` out of nowhere. Editing study.sh while the
# study runs is now harmless.
#
# Arms that already trained to completion are SKIPPED, judged by
# history.json holding a full schedule of epochs rather than by the presence of
# best_model.pt -- which is written on every improvement and so exists for
# killed runs too. FORCE=1 retrains them anyway.
#
# W10 is deliberately not in the plan: its objective is identical to arm B
# (cell_integrated_phase: 1.0), and collect_results.py already reports B under
# both names.
#
# KNOBS
#   GPUS="2 3"   which two GPUs to use
#   FORCE=1      retrain arms that already finished
#   EVERY=45     seconds between watcher redraws
#   PYTHON=python3
#
# ---------------------------------------------------------------------------

set -u

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}" || exit 1

: "${GPUS:=2 3}"
: "${FORCE:=}"
: "${EVERY:=45}"
: "${PYTHON:=python}"

GPU_A="$(echo "${GPUS}" | awk '{print $1}')"
GPU_B="$(echo "${GPUS}" | awk '{print $2}')"
: "${GPU_B:=${GPU_A}}"

# Group A carries the paper's central question and its seed replication, so it
# gets the arms whose result the study cannot do without. Group B is everything
# else, and it is longer, which is why it has no seed replication.
ARMS_A="A B B1"
SEEDS_A="1337 2024"
ARMS_B="KA KB B2 C D0 D1 W01 W03 W30"

STUDY_DIR="logs/study"
STATE="${STUDY_DIR}/state"
DRIVER_LOG="${STUDY_DIR}/driver.log"
DRIVER_PID="${STUDY_DIR}/driver.pid"
DRIVER_COPY="${ROOT}/.study_driver.sh"

# arm code -> config file. MUST MATCH config_for() in run_v2.sh.
config_for() {
  case "$1" in
    A)   echo config/v2/a_baseline.yaml ;;
    B)   echo config/v2/b_cell_ipp.yaml ;;
    B1)  echo config/v2/b1_image_volume.yaml ;;
    B2)  echo config/v2/b2_cell_area.yaml ;;
    C)   echo config/v2/c_cell_ipp_bga.yaml ;;
    D0)  echo config/v2/d0_amplitude.yaml ;;
    D1)  echo config/v2/d_forward_amplitude.yaml ;;
    W01) echo config/v2/w_ipp_01.yaml ;;
    W03) echo config/v2/w_ipp_03.yaml ;;
    W10) echo config/v2/w_ipp_10.yaml ;;
    W30) echo config/v2/w_ipp_30.yaml ;;
    KA)  echo config/v2/k_compact_a.yaml ;;
    KB)  echo config/v2/k_compact_b.yaml ;;
    *)   echo "" ;;
  esac
}

say() { printf '%s\n' "$*"; }
rule() { printf '  %s\n' "------------------------------------------------------------"; }

state_put() { mkdir -p "${STUDY_DIR}"; printf '%s\n' "$1" > "${STATE}"; }
state_get() { [ -f "${STATE}" ] && cat "${STATE}" || echo "unknown"; }

driver_alive() {
  [ -f "${DRIVER_PID}" ] || return 1
  kill -0 "$(cat "${DRIVER_PID}" 2>/dev/null)" 2>/dev/null
}

# ---------------------------------------------------------------------------
# stopping
#
# Killing the bash that runs run_v2.sh does not kill the python it is waiting
# on, and killing the python lets run_v2.sh start the next arm. So the whole
# process GROUP goes, which is what nohup/setsid created it as. Our own group
# is protected, or this would kill the watcher too.
# ---------------------------------------------------------------------------
PATTERN='study_driver\.sh|run_v2\.sh|main\.py (train|evaluate|export|benchmark|prepare|compare)|scripts/(conventional_baseline|make_figures|collect_results|aggregate_seeds|estimate_aberration|prepare_membrane|prepare_amplitude|amplitude_sensitivity|error_propagation|synthetic_validation|check_gradient_path|calibrate_z|selftest)\.py'

stop_all() {
  local own_pgid pids pid pgid killed=0
  own_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"

  pids="$(pgrep -f "${PATTERN}" 2>/dev/null | grep -vx "$$" || true)"
  if [ -z "${pids}" ]; then
    say "  nothing of ours was running"
  else
    for pid in ${pids}; do
      kill -0 "${pid}" 2>/dev/null || continue
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
      if [ -n "${pgid}" ] && [ "${pgid}" != "${own_pgid}" ]; then
        kill -TERM "-${pgid}" 2>/dev/null && killed=1
      else
        kill -TERM "${pid}" 2>/dev/null && killed=1
      fi
    done
    say "  sent TERM to: $(echo ${pids} | tr '\n' ' ')"
  fi

  # Give them 20 s to close their files, then stop asking.
  local waited=0
  while [ ${waited} -lt 20 ]; do
    pgrep -f "${PATTERN}" 2>/dev/null | grep -vx "$$" >/dev/null || break
    sleep 2; waited=$((waited + 2))
  done
  pids="$(pgrep -f "${PATTERN}" 2>/dev/null | grep -vx "$$" || true)"
  if [ -n "${pids}" ]; then
    say "  still up after ${waited}s, sending KILL: $(echo ${pids} | tr '\n' ' ')"
    for pid in ${pids}; do
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
      if [ -n "${pgid}" ] && [ "${pgid}" != "${own_pgid}" ]; then
        kill -KILL "-${pgid}" 2>/dev/null
      else
        kill -KILL "${pid}" 2>/dev/null
      fi
    done
    sleep 3
  fi

  rm -f "${DRIVER_PID}"
  [ "${killed}" = 1 ] && state_put "phase=stopped"

  # CUDA memory is released by the kernel when the process dies, but not
  # instantly, and a new run that starts too early gets an OOM that looks like
  # a model-size problem.
  if command -v nvidia-smi >/dev/null 2>&1; then
    waited=0
    while [ ${waited} -lt 60 ]; do
      local busy
      busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
      [ -z "${busy}" ] && break
      say "  waiting for the GPUs to clear (pids: $(echo ${busy} | tr '\n' ' '))"
      sleep 5; waited=$((waited + 5))
    done
    say "  GPU state now:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null | sed 's/^/    GPU /'
  fi
}

# ---------------------------------------------------------------------------
# planning: which arms still need training
# ---------------------------------------------------------------------------
plan_runs() {
  # $1 = arm codes, $2 = replication seeds (optional).
  # Prints one "<arm> <seed|-> done|todo" line per run.
  local arms="$1" seeds="${2:-}" pairs="" arm cfg
  for arm in ${arms}; do
    cfg="$(config_for "${arm}")"
    [ -n "${cfg}" ] && pairs="${pairs} ${arm}=${cfg}"
  done
  ${PYTHON} - "${seeds}" ${pairs} <<'PY'
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from holoqpi.config import load_config
from holoqpi.utils import run_directory

seeds = sys.argv[1].split()


def complete(cfg, experiment_name):
    """Did this run train its whole schedule?

    Judged by history.json, not by best_model.pt: that checkpoint is rewritten
    on every improvement, so a run killed at epoch 3 leaves one behind and
    looks finished.
    """
    run_dir = Path(run_directory(cfg.paths.output_root, experiment_name, cfg.data.modality))
    history = run_dir / "history.json"
    if not history.is_file():
        return False
    try:
        records = json.loads(history.read_text())
    except Exception:
        return False
    return isinstance(records, list) and len(records) >= int(cfg.training.epochs)


for pair in sys.argv[2:]:
    arm, _, path = pair.partition("=")
    try:
        cfg = load_config(path)
    except Exception:
        print(arm, "-", "todo")
        continue
    print(arm, "-", "done" if complete(cfg, cfg.experiment_name) else "todo")
    for seed in seeds:
        # run_v2.sh names a replication run v2_<arm>_seed<seed>.
        print(arm, seed, "done" if complete(cfg, f"v2_{arm}_seed{seed}") else "todo")
PY
}

todo_arms() {
  if [ -n "${FORCE}" ]; then printf '%s' "$1"; return; fi
  local report
  report="$(plan_runs "$1")"
  # If the planner could not run at all -- a broken import, the wrong
  # interpreter -- it prints nothing, and an empty todo list would silently
  # mean "everything is already trained" and skip the whole study. Assume
  # nothing is done instead, which costs GPU time but never loses a result.
  if [ -z "${report}" ]; then
    say "  WARNING: could not read run history; assuming nothing has trained" >&2
    printf '%s' "$1"
    return
  fi
  printf '%s' "${report}" | awk '$2 == "-" && $3 == "todo" { printf "%s ", $1 }'
}

todo_seeds() {
  if [ -n "${FORCE}" ]; then printf '%s' "$2"; return; fi
  [ -z "${2:-}" ] && return
  local report
  report="$(plan_runs "$1" "$2")"
  if [ -z "${report}" ]; then printf '%s' "$2"; return; fi
  printf '%s' "${report}" | awk '$2 != "-" && $3 == "todo" { print $2 }' | sort -u | tr '\n' ' '
}

prep_needed() {
  [ -f data/splits.json ] && [ -f data/aberration.json ] \
    && [ -d data/amplitude_reference ] && return 1
  return 0
}

# ---------------------------------------------------------------------------
# the driver: everything that actually runs. Detached; never in the terminal.
# ---------------------------------------------------------------------------
run_driver() {
  local todo_a todo_b seeds_a pid_a pid_b stamp
  stamp="$(date '+%F %H:%M:%S')"
  say "=========================================================="
  say "study.sh driver up ${stamp}   GPUs ${GPU_A} and ${GPU_B}"

  if prep_needed; then
    state_put "phase=prep"
    say "--- preparation stages 0-5 (no GPU)"
    echo "study.sh: NO_PLAN -- stages 0-5" > p_prep.out
    for s in 0 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHON="${PYTHON}" bash run_v2.sh --stage "${s}" \
        >> "p_prep.out" 2>&1
    done
  else
    say "--- preparation outputs already present, skipping stages 0-5"
  fi

  todo_a="$(todo_arms "${ARMS_A}")"
  todo_b="$(todo_arms "${ARMS_B}")"
  seeds_a="$(todo_seeds "${ARMS_A}" "${SEEDS_A}")"
  say "--- training"
  say "    GPU ${GPU_A}: ${todo_a:-nothing left}   seeds: ${seeds_a:-none left}"
  say "    GPU ${GPU_B}: ${todo_b:-nothing left}"
  state_put "phase=training"

  # One background pipeline per GPU, each doing its own training and then its
  # own evaluations, so neither GPU waits on the other.
  #
  # The second invocation in each pipeline carries NO_TRAIN=1, which skips the
  # base trains and keeps the evaluations. That is what rebuilds
  # unmatched_<split>.csv -- figure 15 reads it, and every arm evaluated before
  # the main.py fix is missing it. It reuses the checkpoints, about a minute an
  # arm. In group A the same invocation also runs the replication seeds, which
  # NO_TRAIN deliberately does not skip.
  (
    if [ -n "${todo_a}" ]; then
      CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHON="${PYTHON}" \
        ARMS="${todo_a}" SEEDS="" \
        bash run_v2.sh --stage 6 > p_main.out 2>&1
    fi
    echo "study.sh: NO_TRAIN=1 -- evaluations and replication seeds only" > p_seeds.out
    CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHON="${PYTHON}" NO_TRAIN=1 \
      ARMS="${ARMS_A}" SEEDS="${seeds_a}" \
      bash run_v2.sh --stage 6 >> p_seeds.out 2>&1
  ) &
  pid_a=$!

  (
    if [ -n "${todo_b}" ]; then
      CUDA_VISIBLE_DEVICES="${GPU_B}" PYTHON="${PYTHON}" \
        ARMS="${todo_b}" SEEDS="" \
        bash run_v2.sh --stage 6 > p_rest.out 2>&1
    fi
    echo "study.sh: NO_TRAIN=1 -- evaluations only" > p_rest_eval.out
    CUDA_VISIBLE_DEVICES="${GPU_B}" PYTHON="${PYTHON}" NO_TRAIN=1 \
      ARMS="${ARMS_B}" SEEDS="" \
      bash run_v2.sh --stage 6 >> p_rest_eval.out 2>&1
  ) &
  pid_b=$!

  wait "${pid_a}"
  wait "${pid_b}"
  say "--- training and evaluation finished $(date '+%H:%M:%S')"

  # Stages 7-10 are minutes, not hours, and each one reads what the previous
  # wrote -- 9 and 10 both write into figures/ -- so they are strictly serial.
  echo "study.sh: NO_PLAN -- stages 7-10" > p_finish.out
  for s in 7 8 9 10; do
    state_put "phase=stage ${s}"
    say "--- stage ${s} $(date '+%H:%M:%S')"
    CUDA_VISIBLE_DEVICES="${GPU_A}" PYTHON="${PYTHON}" bash run_v2.sh --stage "${s}" \
      >> p_finish.out 2>&1
  done

  state_put "phase=finished"
  say "=========================================================="
  say "study.sh driver done $(date '+%F %H:%M:%S')"
  say "read runs/RESULTS.md"
}

# ---------------------------------------------------------------------------
# the watcher: the only thing that lives in your terminal
# ---------------------------------------------------------------------------
detach_note() {
  say ""
  rule
  say "  Watching stopped. THE STUDY IS STILL RUNNING."
  say "    bash study.sh --status     watch it again"
  say "    bash study.sh --stop       stop it for real"
  rule
  say ""
}

watch_loop() {
  trap 'detach_note; exit 0' INT
  local phase
  while :; do
    clear 2>/dev/null || true
    phase="$(state_get)"
    say ""
    say "  HoloQPI v2 study — $(date '+%F %H:%M:%S')   ${phase}"
    if driver_alive; then
      say "  driver pid $(cat "${DRIVER_PID}") is up"
    else
      say "  driver is NOT running"
    fi
    say ""

    if [ -f scripts/progress.py ]; then
      ${PYTHON} scripts/progress.py --no-header 2>/dev/null \
        || say "  (progress.py could not read the logs yet)"
    else
      for f in p_main.out p_rest.out p_main_eval.out p_rest_eval.out p_finish.out; do
        [ -f "${f}" ] || continue
        say "  ${f}"
        grep '^=== ' "${f}" 2>/dev/null | tail -n 2 | sed 's/^=== /      /'
      done
      say ""
    fi

    if [ -f "${DRIVER_LOG}" ]; then
      rule
      tail -n 5 "${DRIVER_LOG}" | sed 's/^/  /'
      rule
    fi

    case "${phase}" in
      phase=finished)
        say ""
        say "  FINISHED. Next:"
        say "    grep FAILED logs/v2_*/SUMMARY.txt"
        say "    cat runs/RESULTS.md"
        say ""
        return 0 ;;
    esac
    if ! driver_alive; then
      say ""
      say "  The driver has exited without reaching 'finished'. Read:"
      say "    ${DRIVER_LOG}"
      say ""
      return 1
    fi

    say "  redrawing every ${EVERY}s — Ctrl-C stops watching, not the study"
    sleep "${EVERY}"
  done
}

# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
case "${1:-}" in
  --driver)
    run_driver
    exit 0
    ;;
  --stop)
    say ""
    say "  stopping everything"
    stop_all
    say "  stopped."
    say ""
    exit 0
    ;;
  --status)
    if [ ! -f "${DRIVER_PID}" ]; then
      say ""
      say "  No study has been launched from this directory."
      say "  Start one with:  bash study.sh"
      say ""
      exit 1
    fi
    watch_loop
    exit $?
    ;;
  "")
    ;;
  *)
    say "usage: bash study.sh [--status | --stop]"
    exit 2
    ;;
esac

# --- default: stop, plan, launch, watch ------------------------------------
if [ ! -f main.py ] || [ ! -f run_v2.sh ]; then
  say "  run this from the project root (main.py and run_v2.sh must be here)"
  exit 1
fi

say ""
say "  HoloQPI v2 study"
rule
say "  This STOPS everything running now and starts the study again."
say "  Arms that already trained a full schedule are kept and skipped;"
say "  an arm that was only part-way through is lost and will retrain."
say ""
say "  5 seconds to Ctrl-C out of this."
rule
sleep 5

say ""
say "  1/3  stopping what is running"
stop_all

say ""
say "  2/3  planning"
mkdir -p "${STUDY_DIR}"
printf "  %-14s %s\n" "GPU ${GPU_A}:" "$(todo_arms "${ARMS_A}")  seeds: $(todo_seeds "${ARMS_A}" "${SEEDS_A}")"
printf "  %-14s %s\n" "GPU ${GPU_B}:" "$(todo_arms "${ARMS_B}")"
if prep_needed; then
  say "  preparation: stages 0-5 will run first"
else
  say "  preparation: outputs present, stages 0-5 skipped"
fi

say ""
say "  3/3  launching (detached, so this terminal is free)"

# The watcher reads every p_*.out in this directory. A file left by an earlier
# study is not overwritten until its own phase begins, so it would sit in the
# display for hours describing work that is not happening. Move them aside.
OLD="$(ls p_main.out p_rest.out p_seeds.out p_rest_eval.out p_prep.out p_finish.out 2>/dev/null || true)"
if [ -n "${OLD}" ]; then
  KEEP="${STUDY_DIR}/previous_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${KEEP}"
  # shellcheck disable=SC2086
  mv ${OLD} "${KEEP}/" 2>/dev/null || true
  say "  previous narration files moved to ${KEEP}/"
fi

cp "${SELF}" "${DRIVER_COPY}"
chmod +x "${DRIVER_COPY}" 2>/dev/null || true
: > "${DRIVER_LOG}"
state_put "phase=starting"
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash "${DRIVER_COPY}" --driver >> "${DRIVER_LOG}" 2>&1 &
else
  nohup bash "${DRIVER_COPY}" --driver >> "${DRIVER_LOG}" 2>&1 &
fi
echo $! > "${DRIVER_PID}"
sleep 3
say "  driver pid $(cat "${DRIVER_PID}")   log ${DRIVER_LOG}"
sleep 2

watch_loop
exit $?
