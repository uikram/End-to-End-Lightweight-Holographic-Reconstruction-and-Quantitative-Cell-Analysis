#!/usr/bin/env bash
# =============================================================================
# HoloQPI — close the four gaps left by the 2026-09-17 run. Run this ON THE
# SERVER, in the same project folder.
#
#     nohup bash FINISH_ON_SERVER.sh > finish_gaps.out 2>&1 &
#
# Check on it at any time with:
#
#     bash FINISH_ON_SERVER.sh --status
#
# GPU 2 only. Nothing here trains anything: every trained checkpoint from the
# previous run is used exactly as it is, and no number already in
# runs/RESULTS.md changes except the four listed below.
#
# WHAT IT FIXES, AND WHY EACH ONE MATTERS
#
#  1. THE AMPLITUDE OUTPUT HAD NO METRIC. The framework's stated output is
#     phase AND amplitude AND segmentation. The amplitude head is trained in
#     arms D0, D1 and D2, but the predicted amplitude entered exactly one
#     reported number -- the forward-model residual -- where it is entangled
#     with the phase, the distance and the aberration surface. An amplitude head
#     that had collapsed to a constant would have left no trace in any table.
#     holoqpi/metrics/amplitude.py now scores it against the reconstruction
#     reference AND against the thin-phase assumption A = 1, and those three
#     arms are re-scored here. Re-scoring, not re-training: same checkpoint,
#     same data, same seed.
#
#  2. STAGE 5 NEVER RAN. RUN_ON_SERVER.sh omitted it, so RESULTS.md's "Results
#     that need no trained model" section came out empty and figures 17 and 18
#     were left standing from an earlier run. It is the error-propagation bound
#     and the synthetic-ground-truth floor: the numbers every measurement result
#     in the study is read against.
#
#  3. TABLE 4 SUPPRESSED ITS OWN GOOD MEASUREMENTS. All twenty timing rows were
#     marked UNSTABLE and "do not quote them", on rows whose own timing_stable
#     was true and whose p99/p50 ran 1.00-1.11. The benchmark was charging this
#     process's own CUDA context to "other processes". Fixed in the reporting,
#     so the existing benchmark JSONs are re-tabulated rather than re-measured.
#
#  4. THE FIGURE MANIFEST RECORDED THREE FIGURES OF SEVENTEEN, because stage 9
#     calls make_figures three times and each wrote the whole file. It now
#     merges.
#
# IT IS SAFE TO RE-RUN. Each step records that it finished; start it again with
# the same command and it picks up where it stopped. To force a step to run
# again, delete its marker from .finish_state/ and re-run.
#
# About an hour in total.
# =============================================================================

cd "$(dirname "$0")" || exit 1

STATE=".finish_state"
mkdir -p "${STATE}" logs

say() { echo ""; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
done_step() { touch "${STATE}/$1"; }
is_done()   { [ -f "${STATE}/$1" ]; }

if [ "$1" = "--status" ]; then
  echo ""
  echo "=== steps finished ==="
  for step in selftest stage5 eval_D0 eval_D1 eval_D2 figures tables; do
    if is_done "${step}"; then echo "  DONE     ${step}"; else echo "  pending  ${step}"; fi
  done
  echo ""
  exit 0
fi

say "HoloQPI gap-closing run starting in $(pwd)"

# -----------------------------------------------------------------------------
# 0. Refuse to run against the wrong folder, and refuse to run before there is
#    anything to finish. Both checks exist because this script edits files that
#    the previous run produced.
# -----------------------------------------------------------------------------
for required in run_v2.sh main.py config/base.yaml scripts/selftest.py \
                holoqpi/metrics/amplitude.py; do
  if [ ! -e "${required}" ]; then
    say "STOP: ${required} is missing. Run this from the project folder, and"
    say "      make sure the updated files were copied over."
    exit 1
  fi
done

if [ ! -f runs/RESULTS.md ]; then
  say "STOP: runs/RESULTS.md is not here, so there is no completed run to finish."
  say "      Run RUN_ON_SERVER.sh instead."
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. Self-test — the gate. 143 checks now, six of them new and all six on the
#    amplitude metric.
# -----------------------------------------------------------------------------
if is_done selftest; then
  say "step 1 self-test: already done, skipping"
else
  say "step 1 of 5: self-test (about 2 minutes)"
  if CUDA_VISIBLE_DEVICES=2 python scripts/selftest.py \
       --config config/base.yaml > selftest.out 2>&1 &&
     grep -q "All checks passed." selftest.out; then
    echo "  checks passed: $(grep -c '^  PASS' selftest.out)"
    done_step selftest
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
# 2. Stage 5 — the label-free analyses. Reference phase and reference masks
#    only, so no checkpoint is involved and nothing already reported moves.
# -----------------------------------------------------------------------------
if is_done stage5; then
  say "step 2 stage 5: already done, skipping"
else
  say "step 2 of 5: stage 5, label-free analyses (about 30 minutes)"
  CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 5 > stage5.out 2>&1
  if grep -q "steps that failed: 0" stage5.out; then
    done_step stage5
    echo "  stage 5 OK"
  else
    say "STOP: stage 5 reported a failure. Send me stage5.out"
    grep -A6 "FAILED" stage5.out | head -n 20
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 3. Re-score the three amplitude arms so the amplitude output has a number.
#
#    NO TRAINING. `main.py evaluate` loads the existing best_model.pt, runs the
#    test split through it and rewrites that arm's metrics_test.json. Because
#    the seed, the data and the weights are all unchanged, every metric other
#    than the new amplitude rows must come back identical -- which makes this
#    also a reproducibility check on the previous run.
# -----------------------------------------------------------------------------
eval_arm() {
  code="$1"; config="$2"
  if is_done "eval_${code}"; then
    say "step 3 arm ${code}: already done, skipping"
    return 0
  fi
  say "step 3 of 5: re-scoring arm ${code} for the amplitude metric (about 2 minutes)"
  if CUDA_VISIBLE_DEVICES=2 python main.py evaluate \
       --config "${config}" --split test > "eval_${code}.out" 2>&1; then
    done_step "eval_${code}"
    grep -E "amplitude_mae|amplitude_pred_sd" "eval_${code}.out" | sed 's/^/    /'
    echo "  arm ${code} OK"
  else
    say "STOP: re-scoring arm ${code} failed. Send me eval_${code}.out"
    tail -n 20 "eval_${code}.out"
    exit 1
  fi
}

eval_arm D0 config/v2/d0_amplitude.yaml
eval_arm D1 config/v2/d_forward_amplitude.yaml
eval_arm D2 config/v2/d2_learned_z.yaml

# -----------------------------------------------------------------------------
# 4. Figures. Stage 9 rebuilds all of them; 17 and 18 now have stage 5's inputs,
#    and the manifest merges instead of recording only the last invocation.
# -----------------------------------------------------------------------------
if is_done figures; then
  say "step 4 figures: already done, skipping"
else
  say "step 4 of 5: figures (about 5 minutes)"
  CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 9 > stage9.out 2>&1
  if grep -q "steps that failed: 0" stage9.out; then
    done_step figures
    echo "  figures OK: $(ls figures/fig*.png 2>/dev/null | wc -l) PNG(s) in figures/"
  else
    say "STOP: stage 9 reported a failure. Send me stage9.out"
    grep -A6 "FAILED" stage9.out | head -n 20
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
# 5. The tables, rebuilt from every file above.
# -----------------------------------------------------------------------------
if is_done tables; then
  say "step 5 tables: already done, skipping"
else
  say "step 5 of 5: seed aggregate and the results tables (about 2 minutes)"
  CUDA_VISIBLE_DEVICES=2 python scripts/aggregate_seeds.py \
    --config config/base.yaml > aggregate.out 2>&1
  CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 10 > stage10.out 2>&1
  if grep -q "steps that failed: 0" stage10.out; then
    done_step tables
    echo "  tables OK"
  else
    say "STOP: stage 10 reported a failure. Send me stage10.out"
    grep -A6 "FAILED" stage10.out | head -n 20
    exit 1
  fi
fi

# -----------------------------------------------------------------------------
say "FINISHED. Read runs/RESULTS.md."
echo ""
echo "What should be different from before:"
echo "  * Table 3c exists and carries the amplitude numbers for D0, D1 and D2."
echo "  * Table 4 no longer marks every row UNSTABLE."
echo "  * The 'Results that need no trained model' section is populated."
echo "  * figures/ contains figures 17 and 18, and _manifest.json lists them all."
echo ""
echo "Send these back:"
echo ""
echo "    tar -czf holoqpi_finish.tar.gz \\"
echo "        runs/RESULTS.md runs/results_table.csv runs/seed_aggregate.json \\"
echo "        runs/z_calibration.json runs/error_propagation_summary.csv \\"
echo "        runs/synthetic_validation_fields.csv \\"
echo "        runs/v2_amplitude_off_axis/metrics_test.json \\"
echo "        runs/v2_forward_amplitude_off_axis/metrics_test.json \\"
echo "        runs/v2_learned_z_off_axis/metrics_test.json \\"
echo "        figures/ selftest.out stage5.out stage9.out finish_gaps.out"
echo ""
