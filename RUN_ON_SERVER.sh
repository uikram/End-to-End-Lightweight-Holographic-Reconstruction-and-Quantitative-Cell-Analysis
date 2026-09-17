#!/usr/bin/env bash
# =============================================================================
# HoloQPI — the whole re-run, in one script. Run this ON THE SERVER.
#
#     nohup bash RUN_ON_SERVER.sh > run.out 2>&1 &
#
# Then check on it whenever you like with:
#
#     bash RUN_ON_SERVER.sh --status
#
# It uses GPU 2 and GPU 3 only, and never touches GPU 0 or 1.
#
# Takes about 38 hours. It trains on both GPUs at once, waits for them, and then
# runs the benchmark, the figures and the tables in the right order.
#
# IT IS SAFE TO RE-RUN. Each phase records that it finished, so if the script
# dies -- dropped connection, server reboot -- start it again with the same
# command and it picks up from the phase that did not complete. To force a
# phase to run again, delete its marker file from .run_state/ and re-run.
# =============================================================================

cd "$(dirname "$0")" || exit 1

STATE=".run_state"
mkdir -p "${STATE}" logs

say() { echo ""; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
done_phase()  { touch "${STATE}/$1"; }
is_done()     { [ -f "${STATE}/$1" ]; }

# -----------------------------------------------------------------------------
# --status: print where things are and stop.
# -----------------------------------------------------------------------------
if [ "$1" = "--status" ]; then
  echo ""
  echo "=== phases finished ==="
  for phase in archive selftest masks stage1 stage2 stage3 stage5 stage4 \
               train_gpu2 train_gpu3 stage7 stage8 compare stage9 stage10; do
    if is_done "${phase}"; then echo "  DONE     ${phase}"; else echo "  pending  ${phase}"; fi
  done
  echo ""
  echo "=== training steps started so far ==="
  grep -h "^=== " gpu2.out gpu3.out 2>/dev/null | tail -n 12
  echo ""
  echo "=== anything that failed ==="
  FAILURES="$(grep -h "FAILED" gpu2.out gpu3.out stage4.out finish.out \
              stage1.out stage2.out stage3.out 2>/dev/null)"
  if [ -n "${FAILURES}" ]; then echo "${FAILURES}"; else echo "  nothing"; fi
  echo ""
  echo "=== GPU right now ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
  exit 0
fi

say "HoloQPI re-run starting in $(pwd)"

# -----------------------------------------------------------------------------
# Sanity checks, before anything expensive.
# -----------------------------------------------------------------------------
for required in run_v2.sh main.py config/base.yaml scripts/selftest.py \
                holoqpi/losses/terms.py config/v2/l_lora.yaml; do
  if [ ! -f "${required}" ]; then
    say "STOP: ${required} is missing. The new code did not get copied across."
    exit 1
  fi
done
if [ ! -d data/phase ]; then
  say "STOP: data/phase is missing. Run this from the project folder on the server."
  exit 1
fi
say "code and data present"

# -----------------------------------------------------------------------------
# 1. Archive the old results. They came from the old objective and are not
#    comparable with what follows.
# -----------------------------------------------------------------------------
if is_done archive; then
  say "phase 1 archive: already done, skipping"
else
  say "phase 1 of 10: archiving the old runs/ and figures/"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  [ -d runs ]    && mv runs    "runs_before_audit_${STAMP}"    && echo "  runs -> runs_before_audit_${STAMP}"
  [ -d figures ] && mv figures "figures_before_audit_${STAMP}" && echo "  figures -> figures_before_audit_${STAMP}"
  mkdir -p runs figures
  done_phase archive
fi

# -----------------------------------------------------------------------------
# 2. Self-test. This is the gate: nothing below is worth running if the physics
#    or the metrics are wrong.
# -----------------------------------------------------------------------------
if is_done selftest; then
  say "phase 2 selftest: already done, skipping"
else
  say "phase 2 of 10: self-test (about 2 minutes)"
  if CUDA_VISIBLE_DEVICES=2 python scripts/selftest.py --config config/base.yaml \
       > selftest.out 2>&1; then
    grep -c "^  PASS" selftest.out | sed 's/^/  checks passed: /'
    done_phase selftest
  else
    say "STOP: the self-test FAILED. Nothing below is worth running."
    echo ""
    grep "^  FAIL" selftest.out
    echo ""
    say "Send me selftest.out"
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 3. Regenerate the masks once, so the settings that produced them are recorded.
#    The masks come out identical -- the parameters have not changed.
# -----------------------------------------------------------------------------
if is_done masks; then
  say "phase 3 masks: already done, skipping"
else
  say "phase 3 of 10: regenerating masks with provenance (about 10 minutes)"
  if CUDA_VISIBLE_DEVICES=2 python main.py prepare \
       --config config/base.yaml --force-masks > masks.out 2>&1; then
    grep -E "masks written|splits:" masks.out | sed 's/^/  /'
    done_phase masks
  else
    say "STOP: prepare failed. Send me masks.out"
    tail -n 20 masks.out
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 4. Stages 1, 2, 3 and 5: manifest and label audit, aberration surfaces,
#    amplitude reference, label-free analyses. Stage 3 is what arms D0, D1 and
#    D2 need; stage 5 needs stage 2 and nothing else.
#
#    STAGE 5 BELONGS HERE, and its omission from the first version of this
#    driver is why the 2026-09-17 run produced a `runs/RESULTS.md` whose
#    "Results that need no trained model" section was empty and left figures 17
#    and 18 standing from an earlier run. It runs on the reference phase and the
#    reference masks, so it needs no checkpoint and no GPU time worth the name:
#    it is the error-propagation bound and the synthetic-ground-truth floor, and
#    every measurement number in the study is read against them.
# -----------------------------------------------------------------------------
for stage in 1 2 3 5; do
  if is_done "stage${stage}"; then
    say "phase 4 stage ${stage}: already done, skipping"
  else
    say "phase 4 of 10: stage ${stage}"
    CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage "${stage}" > "stage${stage}.out" 2>&1
    if grep -q "steps that failed: 0" "stage${stage}.out"; then
      done_phase "stage${stage}"
      echo "  stage ${stage} OK"
    else
      say "STOP: stage ${stage} reported a failure. Send me stage${stage}.out"
      grep -A6 "FAILED" "stage${stage}.out" | head -n 20
      exit 1
    fi
  fi
done

# -----------------------------------------------------------------------------
# 5. Training on both GPUs, plus the pre-flight diagnostics alongside them.
#
#    Stage 4's diagnostics normally SET the loss weights, but the weights in the
#    configs are already fixed from earlier rounds -- so its corrected numbers go
#    into the write-up rather than changing anything this round. That is why it
#    can run beside the training instead of before it.
# -----------------------------------------------------------------------------
PIDS=""

if is_done stage4; then
  say "phase 5 stage 4: already done, skipping"
else
  say "phase 5 of 10: pre-flight diagnostics in the background (about 3 hours)"
  CUDA_VISIBLE_DEVICES=2 nohup bash run_v2.sh --stage 4 > stage4.out 2>&1 &
  PID4=$!
  PIDS="${PIDS} ${PID4}"
  echo "  stage 4 pid ${PID4} -> stage4.out"
fi

if is_done train_gpu2; then
  say "phase 5 GPU 2 training: already done, skipping"
else
  say "phase 5 of 10: GPU 2 training, 10 runs (about 35 hours)"
  # W10 is deliberately absent: its objective is arm B's exactly at the same
  # seed, so collect_results reports B's number for that row of the sweep and
  # says so. Training it would spend a GPU-day reproducing a number we have.
  CUDA_VISIBLE_DEVICES=2 nohup bash -c '
    ARMS="A B B1 B2 C D0" bash run_v2.sh --stage 6 &&
    NO_TRAIN=1 ARMS="A B" SEEDS="1337 2024" bash run_v2.sh --stage 6
  ' > gpu2.out 2>&1 &
  PID2=$!
  PIDS="${PIDS} ${PID2}"
  echo "  GPU 2 pid ${PID2} -> gpu2.out"
fi

if is_done train_gpu3; then
  say "phase 5 GPU 3 training: already done, skipping"
else
  say "phase 5 of 10: GPU 3 training, 10 runs (about 35 hours)"
  # D2 is the learned-z arm and G is arm A on in-line holograms. G is the only
  # thing that makes the off-axis / in-line comparison, and figures 5, 13 and 14,
  # possible at all; stage 6 pairs the two geometries once both have finished.
  CUDA_VISIBLE_DEVICES=3 nohup bash -c '
    ARMS="D1 D2 G W01 W03 W30 KA KB" bash run_v2.sh --stage 6 &&
    NO_TRAIN=1 ARMS="B1" SEEDS="1337 2024" bash run_v2.sh --stage 6
  ' > gpu3.out 2>&1 &
  PID3=$!
  PIDS="${PIDS} ${PID3}"
  echo "  GPU 3 pid ${PID3} -> gpu3.out"
fi

if [ -n "${PIDS}" ]; then
  say "waiting for${PIDS} -- this is the long wait, about 35 hours."
  echo "  check progress at any time from another terminal with:"
  echo "      bash RUN_ON_SERVER.sh --status"
  # shellcheck disable=SC2086
  wait ${PIDS}
  say "training and diagnostics finished"
fi

# Mark what actually completed, so a re-run does not repeat it.
grep -q "########## done" gpu2.out   2>/dev/null && done_phase train_gpu2
grep -q "########## done" gpu3.out   2>/dev/null && done_phase train_gpu3
grep -q "########## done" stage4.out 2>/dev/null && done_phase stage4

if ! is_done train_gpu2 || ! is_done train_gpu3; then
  say "STOP: a training job did not reach the end. Send me gpu2.out and gpu3.out"
  grep -h "FAILED" gpu2.out gpu3.out 2>/dev/null | head -n 20
  exit 1
fi

# -----------------------------------------------------------------------------
# 6-10. The finishing stages, in order, with the benchmark run alone.
# -----------------------------------------------------------------------------
{
  # 6. The classical reconstruction baseline: the floor the network must beat.
  if is_done stage7; then
    say "phase 6 stage 7: already done, skipping"
  else
    say "phase 6 of 10: classical baseline (about 1 hour)"
    CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 7
    done_phase stage7
  fi

  # 7. THE EFFICIENCY BENCHMARK, WHICH MUST HAVE THE GPU TO ITSELF.
  #
  # This is the step that was contaminated last time: it overlapped a training
  # job on the same device, the tail latencies roughly tripled, and the compact
  # decoder came out SLOWER than the full one -- inverting the conclusion. Both
  # training jobs have exited by now, and the benchmark itself checks the device
  # and records timing_stable per row, so a contaminated number is visible in
  # Table 4 instead of silently reported.
  if is_done stage8; then
    say "phase 7 stage 8: already done, skipping"
  else
    say "phase 7 of 10: ONNX export and benchmark, GPU 2 alone (about 20 minutes)"
    echo "  GPU state going in:"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 8
    done_phase stage8
  fi

  # 8. Pair the two acquisition geometries. `compare` no longer trains by
  # default, so this scores the existing checkpoints and writes the comparison
  # JSON that figures 5, 13 and 14 need -- it cannot overwrite an arm.
  if is_done compare; then
    say "phase 8 compare: already done, skipping"
  else
    say "phase 8 of 10: pairing off-axis with in-line"
    CUDA_VISIBLE_DEVICES=2 python main.py compare \
      --config config/v2/a_baseline.yaml --modalities off_axis gabor
    done_phase compare
  fi

  # 9. Every figure.
  if is_done stage9; then
    say "phase 9 stage 9: already done, skipping"
  else
    say "phase 9 of 10: figures (about 30 minutes)"
    CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 9
    done_phase stage9
  fi

  # 10. The seed aggregate and the tables.
  if is_done stage10; then
    say "phase 10 stage 10: already done, skipping"
  else
    say "phase 10 of 10: seed aggregate and the results tables"
    CUDA_VISIBLE_DEVICES=2 python scripts/aggregate_seeds.py --config config/base.yaml
    CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 10
    done_phase stage10
  fi
} > finish.out 2>&1

say "FINISHED. Read runs/RESULTS.md first."
echo ""
echo "Anything that failed anywhere:"
FAILURES="$(grep -h "FAILED" gpu2.out gpu3.out stage4.out finish.out \
            stage1.out stage2.out stage3.out 2>/dev/null)"
if [ -n "${FAILURES}" ]; then echo "${FAILURES}"; else echo "  nothing"; fi
echo ""
echo "Collect these into one archive and send them back:"
echo "    tar -czf holoqpi_results.tar.gz \\"
echo "        runs/RESULTS.md runs/results_table.csv runs/seed_aggregate.json \\"
echo "        runs/z_calibration.json runs/*/metrics_test.json runs/*/history.json \\"
echo "        runs/*/per_cell_test.csv runs/*_modality_comparison.json \\"
echo "        runs/*hardware_benchmark*.json figures/ \\"
echo "        selftest.out stage4.out gpu2.out gpu3.out finish.out"
echo ""
