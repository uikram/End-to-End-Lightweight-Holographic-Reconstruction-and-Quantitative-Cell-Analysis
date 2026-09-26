#!/usr/bin/env bash
# =============================================================================
# HoloQPI v2 - the whole study, in dependency order.
#
#   bash run_v2.sh                     everything
#   bash run_v2.sh --stage 3           one stage only
#   QUICK=1 bash run_v2.sh             tiny settings, to prove the plumbing
#   NO_TRAIN=1 bash run_v2.sh          reuse checkpoints, re-evaluate only
#   ARMS="A B B1" bash run_v2.sh       train only these arms
#   CLEAN=1 bash run_v2.sh             archive runs/ and figures/ first
#
# Seed replication and repeated benchmarking (stages 11-14) are documented in
# BENCHMARKING.md; in short:
#   CUDA_VISIBLE_DEVICES=2 ARMS="A B B1 B2 C D0 D1 D2" nohup bash run_v2.sh --stage 11 > seeds_gpu2.out 2>&1 &
#   CUDA_VISIBLE_DEVICES=3 ARMS="G W01 W03 W30 KA KB" nohup bash run_v2.sh --stage 11 > seeds_gpu3.out 2>&1 &
#   CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 12     (only when both GPUs are idle)
#   bash run_v2.sh --stage 13
#   bash run_v2.sh --stage 14
#
# Run it under nohup so a dropped connection does not kill it:
#
#   CLEAN=1 nohup bash run_v2.sh > v2.out 2>&1 &
#   tail -f v2.out
#
# CLEAN=1 matters more than it looks. A checkpoint trained under a different
# objective is not comparable with one trained under this objective, and the
# constants changed on 2026-09-10 (pixel pitch, refraction increment), so every
# checkpoint written before then reports areas 34.4% low and masses 24.3% low.
# If runs/ contains anything from before that date, use CLEAN=1.
#
# ---------------------------------------------------------------------------
# HOW LONG THIS TAKES, AND WHAT TO RUN FIRST
#
# v1's run_study.sh budgeted 12-16 h for FOUR arms on one GPU, so roughly 3-4 h
# per arm at 60 epochs. This has FIFTEEN. Running all of them plus seed
# replication is 60-80 h, which is not a thing to start blind.
#
# So run it in three passes, and stop after any of them if the answer is
# already clear:
#
#   PASS 1, about 10-12 h.  The main hypothesis and its control.
#       ARMS="A B B1" bash run_v2.sh --stage 6
#       ARMS="A B B1" bash run_v2.sh --stage 11     (the other seeds)
#     Three arms, three seeds each = 9 runs. This is the ONLY pass that can
#     answer the paper's central question, because A vs B says a
#     measurement-aware loss helps and B vs B' says the PER-CELL part is what
#     helped. Without the seeds no difference can be called at all. If the
#     differences do not clear 2x the seed spread here, adding arms will not
#     change that -- write it up as unresolved and move to the label work.
#
#   PASS 2, about 8 h.  The efficiency claim and the remaining terms.
#       ARMS="KA KB B2 C" bash run_v2.sh
#     KA/KB measure what the 65% parameter saving costs; B2 and C are one extra
#     term each.
#
#   PASS 3, about 12 h.  Amplitude, the forward model, the sweep, the geometry.
#       ARMS="D0 D1 D2 G W01 W03 W10 W30" bash run_v2.sh
#     D1 and D2 are diagnostics, not loss ablations. G is arm A on in-line
#     holograms and is the only thing that makes the off-axis / in-line
#     comparison -- and figures 5, 13 and 14 -- possible; stage 6 pairs the two
#     geometries automatically once both have been trained.
#
# Stages 0-5 and 7 are cheap and run once; stage 6 is all of the cost. Stages
# 0-5 need NO GPU and produce publishable results on their own, so run them
# first whatever else you decide:
#
#       for s in 0 1 2 3 4 5; do bash run_v2.sh --stage $s; done
#
# ---------------------------------------------------------------------------
# STAGES, and what each one blocks
#
#   0  self-test                    nothing runs if the physics is wrong
#   1  prepare data                 masks, splits, manifest, MEMBRANE registration
#   2  aberration surfaces          BLOCKS stage 5; writes global + per_field
#   3  amplitude reference          BLOCKS arms D0 and D1
#   4  pre-flight diagnostics       sets the loss weights BEFORE training on them
#   5  label-free analyses          error propagation, synthetic ground truth
#   6  train the arms               the study
#   7  conventional baseline        the floor the network must beat
#   8  ONNX export + benchmark      the efficiency claim
#   9  figures                      reads everything above
#  10  collect results              the tables, assembled from the run files
#  11  seed replication             every arm at every seed: train if missing, evaluate
#  12  repeated hardware benchmark  every hardware arm at every seed, own file each
#  13  repeated synthetic floor     the label-free ground-truth check at every seed
#  14  collect benchmark results    runs/benchmark_results/results_*.json
#
# STAGE 4 IS NOT OPTIONAL AND IS NOT COSMETIC. It measures the gradient ratio
# that sets the per-cell loss weight and it measures whether the forward-model
# residual is correctly signed. Both were got wrong earlier in this project by
# reasoning instead of measuring. Read its output before stage 6.
#
# STAGE 5 NEEDS NO MODEL AT ALL. Both analyses run on the reference phase and
# the reference masks, so they produce the label-free headline results even if
# every training run fails. That is deliberate: they are the results that do
# not depend on anything working.
#
# Every stage writes its own log under logs/v2_<timestamp>/ and the script keeps
# going when one fails, so a single failure does not hide the rest.
# =============================================================================

cd "$(dirname "$0")" || exit 1

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# The GPU is part of the name because the seed stages are run as two
# concurrent invocations, one per GPU, and started in the same second they
# would otherwise write one shared SUMMARY.txt.
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/v2_${STAMP}_gpu${CUDA_VISIBLE_DEVICES//,/-}"
mkdir -p "${LOGDIR}"
SUMMARY="${LOGDIR}/SUMMARY.txt"
: "${PYTHON:=python}"
: "${SEEDS:=}"
# D2 and G are part of the study now -- D2 is the learned-distance arm and G is
# arm A on in-line holograms, which is what makes the modality comparison and
# figures 5, 13 and 14 possible at all. They were added after the arm list and
# left out of it, so a full run silently produced neither.
: "${ARMS:=A B B1 B2 C D0 D1 D2 G W01 W03 W10 W30 KA KB}"

ONLY_STAGE=""
if [ "$1" = "--stage" ]; then ONLY_STAGE="$2"; fi

# QUICK trades every expensive setting for speed. It proves the plumbing and
# proves nothing else: 2 epochs on 192 px crops cannot train a decoder, and a
# metric from a QUICK run must never appear in a table. The smoke runs already
# in runs/ from earlier in this project are exactly this and were mistaken for
# results once.
QUICK_OVERRIDES=""
GRAD_BATCHES=30
Z_STEPS=41
if [ -n "${QUICK}" ]; then
  QUICK_OVERRIDES="training.epochs=2 data.train_crop=192 data.batch_size=2 \
evaluation.phase_n_images=2 evaluation.measurement.field_bootstrap_resamples=50 \
deploy.benchmark.warmup_runs=2 deploy.benchmark.timed_runs=20"
  # The diagnostics need shrinking too, or QUICK is not quick: a 30-batch
  # gradient measurement and a 41-point z sweep dominate the wall clock on a
  # machine without a GPU, and neither is meaningful at QUICK's crop size
  # anyway (the gradient ratio is a function of how many cells a crop holds).
  GRAD_BATCHES=3
  Z_STEPS=5
  echo "QUICK MODE: results from this run are NOT reportable." | tee -a "${SUMMARY}"
  echo "  It proves every stage runs and every output file appears. The NUMBERS" \
    | tee -a "${SUMMARY}"
  echo "  are meaningless: 2 epochs on 192 px crops cannot train a decoder." \
    | tee -a "${SUMMARY}"
fi

note() { echo "$*" | tee -a "${SUMMARY}"; }

# Run one command, log it, record the outcome, never abort the script.
#
# A NON-ZERO EXIT IS NOT AUTOMATICALLY A FAILURE HERE. Several diagnostics use
# their exit code to carry a VERDICT: scripts/amplitude_sensitivity.py exits 2
# when the forward-model term is anti-discriminative, and
# scripts/check_gradient_path.py exits 1 when the loss term is swamped. Those
# runs succeeded -- they are reporting a finding, which is the whole point of
# running them -- and calling them FAILED sends someone hunting for a bug that
# is not there. It happened on the first 800-field run.
#
# A real crash leaves a Python traceback in the log, so that is one
# discriminator. EXIT CODE 3 IS THE OTHER, and it is reserved throughout this
# project for a hard error -- a missing checkpoint, a null distance, a split
# that read no images. Those are misconfigurations, not findings, and they used
# to exit 1 with no traceback and be logged here as "the script ran and is
# reporting a finding", so they did not count towards the failure total and
# nobody went looking. Anything else with a non-zero code is a verdict.
step() {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.log"
  note ""
  note "=== ${name}  $(date +%H:%M:%S)"
  note "    $*"

  # Capture the status immediately. The previous version read $? after the `if`
  # block had already reset it, so every failure was reported as "exit 0".
  local code=0
  "$@" > "${log}" 2>&1 || code=$?

  if [ "${code}" -eq 0 ]; then
    note "    OK   -> ${log}"
    return 0
  fi
  if grep -q "Traceback (most recent call last)" "${log}" 2>/dev/null; then
    note "    FAILED (exit ${code}, traceback) -> ${log}"
    note "    last lines:"
    tail -n 12 "${log}" | sed 's/^/      /' | tee -a "${SUMMARY}"
    return "${code}"
  fi
  if [ "${code}" -eq 3 ]; then
    note "    FAILED (exit 3, hard error -- a misconfiguration, not a finding) -> ${log}"
    note "    last lines:"
    tail -n 12 "${log}" | sed 's/^/      /' | tee -a "${SUMMARY}"
    return "${code}"
  fi
  note "    VERDICT exit ${code} -- the script ran and is reporting a finding "
  note "    (not a crash) -> ${log}"
  return 0
}

want() { [ -z "${ONLY_STAGE}" ] || [ "${ONLY_STAGE}" = "$1" ]; }
armed() { case " ${ARMS} " in *" $1 "*) return 0;; *) return 1;; esac; }

# arm code -> config file
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
    # Both are in the default ARMS list. D2 is the learned-distance arm and G is
    # arm A on in-line holograms; G is what makes the modality comparison, and
    # figures 5, 13 and 14, possible at all.
    D2)  echo config/v2/d2_learned_z.yaml ;;
    G)   echo config/v2/g_baseline_gabor.yaml ;;
    # L is arm A with LoRA adaptation of the encoder. NOT in the default ARMS
    # list, because whether the study claims LoRA at all is a decision for the
    # write-up rather than for this script -- see the long note at the top of
    # config/v2/l_lora.yaml. Run it by name:
    #     ARMS="L" bash run_v2.sh --stage 6
    L)   echo config/v2/l_lora.yaml ;;
    *)   echo "" ;;
  esac
}

note "HoloQPI v2 study, started $(date)"
note "log directory ${LOGDIR}"
note "GPU ${CUDA_VISIBLE_DEVICES}   arms: ${ARMS}   seeds (stages 11-14): ${SEEDS:-from config}"

if [ -n "${CLEAN}" ]; then
  ARCHIVE="archive/pre_v2_${STAMP}"
  mkdir -p "${ARCHIVE}"
  for d in runs figures; do
    [ -d "$d" ] && mv "$d" "${ARCHIVE}/" && mkdir -p "$d"
  done
  note "archived previous runs/ and figures/ to ${ARCHIVE}"
fi

# --------------------------------------------------------------------------
# 0  self-test
# --------------------------------------------------------------------------
if want 0; then
  note ""
  note "########## stage 0: self-test"
  if ! step selftest "${PYTHON}" scripts/selftest.py; then
    note ""
    note "SELF-TEST FAILED. Everything below rests on the physics and the metrics"
    note "being right, so stop here and read ${LOGDIR}/selftest.log."
    exit 1
  fi
fi

# --------------------------------------------------------------------------
# 1  prepare data
# --------------------------------------------------------------------------
if want 1; then
  note ""
  note "########## stage 1: prepare data"
  step prepare "${PYTHON}" main.py prepare --config config/base.yaml
  step audit_labels "${PYTHON}" scripts/audit_labels.py --config config/base.yaml

  # The membrane channel, which is the only route out of the label circularity:
  # every mask in data/mask is Otsu(gaussian * phase), so segmentation is
  # currently scored against a threshold of its own input. --estimate checks the
  # registration still holds on the full 800 fields before anything uses it.
  if [ -d data/membrane ]; then
    step membrane_registration "${PYTHON}" scripts/prepare_membrane.py \
      --config config/base.yaml --estimate
    grep -E "offset_|correlation|-> |    membrane" \
      "${LOGDIR}/membrane_registration.log" 2>/dev/null \
      | sed 's/^/    /' | tee -a "${SUMMARY}"
    step membrane_prepare "${PYTHON}" scripts/prepare_membrane.py \
      --config config/base.yaml --write-masks
    grep -E "illuminated|foreground|marginal|-> " \
      "${LOGDIR}/membrane_prepare.log" 2>/dev/null \
      | sed 's/^/    /' | tee -a "${SUMMARY}"
  else
    note "    data/membrane not present, skipping membrane registration"
  fi
fi

# --------------------------------------------------------------------------
# 2  aberration surfaces
#
# Writes three: per_field (target-derived, NOT deployable, kept only for the
# comparison), global (train-split median, deployable, the default), and hybrid
# (global curvature plus a carrier-derived ramp, measured not to help here).
# The log carries the numbers that justify the choice.
# --------------------------------------------------------------------------
if want 2; then
  note ""
  note "########## stage 2: aberration surfaces"
  step aberration "${PYTHON}" scripts/estimate_aberration.py --config config/base.yaml
  grep -E "peak-to-valley|departure|share of the|-> " "${LOGDIR}/aberration.log" \
    2>/dev/null | sed 's/^/    /' | tee -a "${SUMMARY}"
fi

# --------------------------------------------------------------------------
# 3  amplitude reference
# --------------------------------------------------------------------------
if want 3; then
  note ""
  note "########## stage 3: amplitude reference"
  step amplitude_reference "${PYTHON}" scripts/prepare_amplitude.py \
    --config config/base.yaml
fi

# --------------------------------------------------------------------------
# 4  pre-flight diagnostics
#
# READ THESE BEFORE STAGE 6. They set a loss weight and they decide whether a
# loss term is usable at all, and both questions were previously answered by
# argument rather than measurement, wrongly in both cases.
# --------------------------------------------------------------------------
if want 4; then
  note ""
  note "########## stage 4: pre-flight diagnostics"

  step gradient_path "${PYTHON}" scripts/check_gradient_path.py \
    --config config/v2/b_cell_ipp.yaml --batches "${GRAD_BATCHES}"
  grep -E "magnitude ratio|cosine with|parity|weight w|^    [0-9]|-> " \
    "${LOGDIR}/gradient_path.log" 2>/dev/null | sed 's/^/    /' | tee -a "${SUMMARY}"

  # Both aberration modes, because the difference between them IS the result:
  # the deployable surface is the one that has to work.
  for mode in per_field global; do
    step "amplitude_sensitivity_${mode}" "${PYTHON}" scripts/amplitude_sensitivity.py \
      --config config/base.yaml --split test \
      --set "optics.aberration.mode=${mode}" data.train_crop=null \
      data.augmentation.enabled=false
    grep -E "response|ratio, amplitude|-> " \
      "${LOGDIR}/amplitude_sensitivity_${mode}.log" 2>/dev/null \
      | sed 's/^/    /' | tee -a "${SUMMARY}"
  done

  step calibrate_z "${PYTHON}" scripts/calibrate_z.py --config config/base.yaml \
    --steps "${Z_STEPS}"
fi

# --------------------------------------------------------------------------
# 5  label-free analyses
#
# Neither needs a trained model, so these results survive any training failure.
# --------------------------------------------------------------------------
if want 5; then
  note ""
  note "########## stage 5: label-free analyses"

  step error_propagation "${PYTHON}" scripts/error_propagation.py \
    --config config/base.yaml --max-shift 5
  grep -E "ONE PIXEL|dilated|eroded|asymmetry|mass error / area|-> " \
    "${LOGDIR}/error_propagation.log" 2>/dev/null | sed 's/^/    /' | tee -a "${SUMMARY}"

  step synthetic_validation "${PYTHON}" scripts/synthetic_validation.py \
    --config config/base.yaml --fields 12
  grep -E "area  signed|mass  signed|field-summed|never measured|-> " \
    "${LOGDIR}/synthetic_validation.log" 2>/dev/null | sed 's/^/    /' | tee -a "${SUMMARY}"
fi

# --------------------------------------------------------------------------
# 6  train the arms
# --------------------------------------------------------------------------
if want 6; then
  note ""
  note "########## stage 6: training arms"
  for arm in ${ARMS}; do
    cfg="$(config_for "${arm}")"
    if [ -z "${cfg}" ]; then note "    unknown arm ${arm}, skipped"; continue; fi
    if [ ! -f "${cfg}" ]; then note "    ${cfg} missing, skipped"; continue; fi

    if [ -z "${NO_TRAIN}" ]; then
      # shellcheck disable=SC2086
      step "train_${arm}" "${PYTHON}" main.py train --config "${cfg}" \
        ${QUICK_OVERRIDES:+--set ${QUICK_OVERRIDES}}
    fi
    # shellcheck disable=SC2086
    step "evaluate_${arm}" "${PYTHON}" main.py evaluate --config "${cfg}" \
      ${QUICK_OVERRIDES:+--set ${QUICK_OVERRIDES}}

    # THE SAME CHECKPOINT, SCORED AGAINST INDEPENDENT LABELS. Without this the
    # segmentation numbers only say how well the model reproduces a phase
    # threshold. --tag keeps the two results in separate files; without it the
    # second evaluation silently overwrites the first.
    if [ -d data/membrane_mask ]; then
      # shellcheck disable=SC2086
      step "evaluate_${arm}_membrane" "${PYTHON}" main.py evaluate --config "${cfg}" \
        --tag membrane --set paths.manual_mask_dir=membrane_mask ${QUICK_OVERRIDES}
    fi
  done

  # Seed replication used to live here, for arms A, B and B' only. It is now
  # stage 11, which covers every arm, reads its seed list from config, never
  # retrains a finished run, and writes provenance for every evaluation.

  # PAIR THE TWO ACQUISITION GEOMETRIES, once both arms exist.
  #
  # `main.py compare` is the only thing that writes
  # <experiment>_modality_comparison.json, and nothing in this driver ever ran
  # it -- so figures 5, 13 and 14 could never be produced by a full run no
  # matter how many arms were trained, and the instruction to run it by hand
  # sat in a comment further down. It is also the step that scores BOTH
  # modalities through the same evaluator and the same instance labeller, which
  # is what makes the off-axis / in-line difference attributable to the hologram
  # type rather than to the evaluation.
  #
  # Training is opt-in for `compare` (see main.py), so this evaluates the
  # existing checkpoints and does not retrain either arm.
  if [ -d runs/v2_baseline_off_axis ] && [ -d runs/v2_baseline_gabor ]; then
    step compare_modalities "${PYTHON}" main.py compare \
      --config config/v2/a_baseline.yaml --modalities off_axis gabor
  else
    note "    modality comparison skipped: needs BOTH runs/v2_baseline_off_axis"
    note "    and runs/v2_baseline_gabor (arm A and arm G). Figures 5, 13 and 14"
    note "    depend on it."
  fi
fi

# --------------------------------------------------------------------------
# 7  conventional baseline
# --------------------------------------------------------------------------
if want 7; then
  note ""
  note "########## stage 7: conventional baseline"
  step conventional "${PYTHON}" scripts/conventional_baseline.py --config config/base.yaml
fi

# --------------------------------------------------------------------------
# 8  ONNX export and benchmark
#
# The export path was broken for every v2 config until 2026-09-10: the wrapper
# returned out["condition"] unconditionally and every classifier-free arm
# raised KeyError. It now derives its outputs from the heads that exist, so the
# compact-decoder arms are the ones worth exporting -- 3.36 M against 9.60 M.
# --------------------------------------------------------------------------
if want 8; then
  note ""
  note "########## stage 8: ONNX export and benchmark"
  for arm in A B KA KB D0; do
    armed "${arm}" || continue
    cfg="$(config_for "${arm}")"
    step "export_${arm}" "${PYTHON}" main.py export --config "${cfg}"
    # --modalities off_axis explicitly: the default is off_axis AND gabor, and
    # with no Gabor checkpoint the benchmark profiles an UNTRAINED graph and
    # writes an ONNX file for it. Harmless as a hardware measurement, but it
    # put untrained rows into the efficiency table for a while.
    step "benchmark_${arm}" "${PYTHON}" main.py benchmark --config "${cfg}" \
      --modalities off_axis
  done
fi

# --------------------------------------------------------------------------
# 9  figures
#
# TWO invocations, because make_figures.py resolves a run directory through
# cfg.experiment_name and the figures fall into two groups.
#
# The label-free and cross-arm figures (10, 16-20) read the CSVs that sit
# directly under runs/ -- error_propagation.csv, synthetic_validation_*.csv,
# label_audit_*.csv, the membrane tables -- and do not care which arm they are
# pointed at, so base.yaml serves them.
#
# Everything per-arm (1, 2, 3, 4, 6, 8, 9, 12, 15) needs a real checkpoint,
# per_cell_test.csv, unmatched_test.csv, history.json and
# <name>_hardware_benchmark_*.csv. With base.yaml that resolves to
# runs/base_off_axis/, which is a v1 directory that does not exist on the
# study machine -- so on 2026-09-14 all nine skipped with "no checkpoint" and
# "no per_cell CSV" while the v2 arms had every one of those files. Pointing
# this invocation at an arm config is the whole fix.
#
# FIGURE_ARM chooses that arm; A is the baseline and the natural illustration.
# Figure 3 additionally needs scripts/diagnose_bias.py, run just before it.
#
# A THIRD invocation covers figures 5, 13 and 14, which compare the two
# acquisition geometries and so need both modalities plus the
# <experiment>_modality_comparison.json that stage 6 now writes. They are
# skipped cleanly when arm G has not been trained.
#
# Not produced by any invocation, and not a failure:
#   7      the v1 ablation panel -- superseded by figure 19
#   11     condition confusion -- classifier_enabled is false in every v2 arm
# --------------------------------------------------------------------------
: "${FIGURE_ARM:=A}"
if want 9; then
  note ""
  note "########## stage 9: figures"
  FIGCFG="$(config_for "${FIGURE_ARM}")"
  note "    per-arm figures illustrated with arm ${FIGURE_ARM} (${FIGCFG})"

  # --modality off_axis explicitly: the default asks for gabor too, and no v2
  # arm has a Gabor checkpoint, so the default would fail on the second pass.
  step diagnose_bias "${PYTHON}" scripts/diagnose_bias.py \
    --config "${FIGCFG}" --split test --modality off_axis

  step figures_global "${PYTHON}" scripts/make_figures.py \
    --config config/base.yaml --only 10 16 17 18 19 20

  step figures_per_arm "${PYTHON}" scripts/make_figures.py \
    --config "${FIGCFG}" --modalities off_axis --only 1 2 3 4 6 8 9 12 15

  # Figures 5, 13 and 14 compare the two acquisition geometries, so they need
  # BOTH modalities and they need <experiment>_modality_comparison.json, which
  # only exists once a Gabor arm has been trained and `main.py compare` has run:
  #
  #     ARMS="G" bash run_v2.sh --stage 6
  #     python main.py compare --config config/v2/a_baseline.yaml
  #
  # Until then the three skip cleanly, so this is safe to leave in place.
  if [ -f runs/v2_baseline_modality_comparison.json ]; then
    step figures_modality "${PYTHON}" scripts/make_figures.py \
      --config config/v2/a_baseline.yaml --modalities off_axis gabor --only 5 13 14
  else
    note "    figures 5, 13, 14 skipped: no modality comparison yet (train arm G first)"
  fi
fi

# --------------------------------------------------------------------------
# 10  collect the results into the paper's tables
#
# Reads every runs/<arm>/metrics_test.json and assembles the ablation table,
# the difference table WITH the 2x-seed-spread rule applied, the measurement
# table and the efficiency table, plus the two label-free analyses. Nothing is
# transcribed by hand, so nothing can be mistyped, and a difference that does
# not clear the seed spread is printed as unresolved rather than as a finding.
# --------------------------------------------------------------------------
if want 10; then
  note ""
  note "########## stage 10: collect results"
  step collect_results "${PYTHON}" scripts/collect_results.py \
    --config config/base.yaml --split test
  if [ -f runs/RESULTS.md ]; then
    note "    -> runs/RESULTS.md is the document to read and to paste from"
  fi
fi

# ==========================================================================
# 11-14  SEED REPLICATION AND REPEATED BENCHMARKING.  See BENCHMARKING.md.
#
# The seed list is project.seed + evaluation.seed_replication.seeds in
# config/base.yaml, unless SEEDS="..." is given (see BENCHMARKING.md).
# Which (arm, seed) runs exist, and whether each is finished, is decided by
# scripts/collect_benchmark_results.py -- the same code that later collects
# them -- so the driver and the collector cannot disagree about "done".
#
# Nothing here ever retrains a finished run. A run that is missing or was
# killed part-way is trained; a finished run is only (re-)evaluated when its
# metrics file cannot be tied to its checkpoint. A run trained under
# different settings, or with a different seed, is reported and left alone:
# set FORCE_RETRAIN=1 to overwrite it.
# ==========================================================================
: "${SYN_FIELDS:=12}"
REP_ROOT="runs"
REP_OVERRIDES=""
if [ -n "${QUICK}" ]; then
  # QUICK replicates go to their own root so they can never be mistaken for, or
  # block, the real ones.
  REP_ROOT="runs_quick"
  REP_OVERRIDES="${QUICK_OVERRIDES} paths.output_root=${REP_ROOT}"
fi
SEEDS_ARG=""
[ -n "${SEEDS}" ] && SEEDS_ARG="--seeds ${SEEDS}"

# HW_DEVICE=cpu only for a CPU-only plumbing test; the paper's timings are cuda.
: "${HW_DEVICE:=cuda}"
collector() {
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/collect_benchmark_results.py "$@" ${SEEDS_ARG} \
    --hardware-device "${HW_DEVICE}" ${REP_OVERRIDES:+--set ${REP_OVERRIDES}}
}
seed_list() {
  if [ -n "${SEEDS}" ]; then echo "${SEEDS}"; else collector seeds; fi
}

# --------------------------------------------------------------------------
# 11  train (if needed) and evaluate every arm at every seed
# --------------------------------------------------------------------------
if want 11; then
  note ""
  note "########## stage 11: seed replication, seeds $(seed_list | tr '\n' ' ')"
  for seed in $(seed_list); do
    for arm in ${ARMS}; do
      [ "${arm}" = "W10" ] && continue          # W10 is arm B; reported once
      cfg="$(config_for "${arm}")"
      if [ -z "${cfg}" ]; then note "    unknown arm ${arm}, skipped"; continue; fi

      code=0
      info="$(collector status --arm "${arm}" --seed "${seed}" 2>"${LOGDIR}/.status")" || code=$?
      name="$(echo "${info}" | awk '{print $2}')"
      why="$(cat "${LOGDIR}/.status" 2>/dev/null)"
      train=""
      case "${code}" in
        20) continue ;;                           # not planned for this arm
        0)  if [ -z "${REEVALUATE}" ]; then
              note "    ${arm} seed ${seed}: done (trained and evaluated)"; continue
            fi ;;
        1)  ;;                                    # trained; evaluate only
        10|11) train=1 ;;                         # missing, or killed part-way
        12) if [ -n "${FORCE_RETRAIN}" ]; then train=1; else
              note "    ${arm} seed ${seed}: FAILED -- left untouched:${why}"
              note "      (FORCE_RETRAIN=1 retrains it and overwrites the directory)"
              continue
            fi ;;
        *)  note "    ${arm} seed ${seed}: FAILED -- status exit ${code}:${why}"; continue ;;
      esac

      if [ -n "${train}" ]; then
        # shellcheck disable=SC2086
        step "train_${arm}_seed${seed}" "${PYTHON}" main.py train --config "${cfg}" \
          --seed "${seed}" --set "experiment_name=${name}" ${REP_OVERRIDES} || continue
      fi
      # shellcheck disable=SC2086
      step "evaluate_${arm}_seed${seed}" "${PYTHON}" main.py evaluate --config "${cfg}" \
        --seed "${seed}" --set "experiment_name=${name}" ${REP_OVERRIDES}
      if [ -d data/membrane_mask ]; then
        # shellcheck disable=SC2086
        step "evaluate_${arm}_seed${seed}_membrane" "${PYTHON}" main.py evaluate \
          --config "${cfg}" --seed "${seed}" --tag membrane \
          --set "experiment_name=${name}" paths.manual_mask_dir=membrane_mask ${REP_OVERRIDES}
      fi
    done
  done
  # shellcheck disable=SC2086
  step plan_after_stage11 collector plan --arms ${ARMS}
  sed 's/^/    /' "${LOGDIR}/plan_after_stage11.log" | tee -a "${SUMMARY}"
fi

# --------------------------------------------------------------------------
# 12  repeated hardware benchmark
#
# RUN THIS ONLY WHEN NOTHING ELSE IS USING THE GPU -- including the other GPU's
# stage-11 run, which shares the CPU and PCIe bus. Each seed is a separate
# process timing that seed's checkpoint; its rows go to
# <run dir>/hardware_benchmark_cuda.json, never to the shared files at the top
# of runs/ that collect_results.py reads for Table 4.
# --------------------------------------------------------------------------
if want 12; then
  note ""
  note "########## stage 12: repeated hardware benchmark"
  for seed in $(seed_list); do
    for arm in $(collector seeds --hardware); do
      armed "${arm}" || continue
      cfg="$(config_for "${arm}")"
      code=0
      info="$(collector status --arm "${arm}" --seed "${seed}" --hardware 2>"${LOGDIR}/.status")" \
        || code=$?
      name="$(echo "${info}" | awk '{print $2}')"
      modality="$(echo "${info}" | awk '{print $4}')"
      why="$(cat "${LOGDIR}/.status" 2>/dev/null)"
      case "${code}" in
        20) continue ;;
        0)  if [ -z "${REBENCH}" ]; then
              note "    ${arm} seed ${seed}: already benchmarked"; continue
            fi ;;
        1)  ;;
        10) note "    ${arm} seed ${seed}: FAILED -- not trained yet (run stage 11 first)"
            continue ;;
        *)  note "    ${arm} seed ${seed}: FAILED -- status exit ${code}:${why}"; continue ;;
      esac
      # shellcheck disable=SC2086
      step "benchmark_${arm}_seed${seed}" "${PYTHON}" main.py benchmark --config "${cfg}" \
        --seed "${seed}" --modalities "${modality}" \
        --set "experiment_name=${name}" ${REP_OVERRIDES}
    done
  done
fi

# --------------------------------------------------------------------------
# 13  repeated synthetic ground-truth check (CPU; no model involved)
# --------------------------------------------------------------------------
if want 13; then
  note ""
  note "########## stage 13: synthetic ground truth at every seed"
  for seed in $(seed_list); do
    out="${REP_ROOT}/synthetic_validation_seeds/seed${seed}"
    if [ -f "${out}/synthetic_validation_summary.json" ] && [ -z "${REEVALUATE}" ]; then
      note "    seed ${seed}: already done"; continue
    fi
    # REP_OVERRIDES first: the per-seed output root must win over its
    # paths.output_root (later --set items override earlier ones).
    # shellcheck disable=SC2086
    step "synthetic_seed${seed}" "${PYTHON}" scripts/synthetic_validation.py \
      --config config/base.yaml --fields "${SYN_FIELDS}" \
      --set ${REP_OVERRIDES} "project.seed=${seed}" "paths.output_root=${out}"
  done
fi

# --------------------------------------------------------------------------
# 14  collect: one JSON per benchmark, plus the existing tables
# --------------------------------------------------------------------------
if want 14; then
  note ""
  note "########## stage 14: collect benchmark results"
  # shellcheck disable=SC2086
  step collect_benchmark_results collector collect --arms ${ARMS}
  grep -E "wrote results_|NOT COMPLETE|^  [A-Za-z0-9_]+$|^    - |all benchmarks complete|ignored" \
    "${LOGDIR}/collect_benchmark_results.log" 2>/dev/null | sed 's/^/    /' | tee -a "${SUMMARY}"
  if [ -z "${QUICK}" ]; then
    step aggregate_seeds "${PYTHON}" scripts/aggregate_seeds.py --config config/base.yaml
    step collect_results "${PYTHON}" scripts/collect_results.py --config config/base.yaml --split test
  fi
  note "    -> ${REP_ROOT}/benchmark_results/"
fi

note ""
note "########## done $(date)"
note "summary  ${SUMMARY}"
note "logs     ${LOGDIR}/"
note ""
note "WHAT TO READ FIRST, in this order:"
note "  1. ${LOGDIR}/error_propagation.log     the label-free headline result"
note "  2. ${LOGDIR}/synthetic_validation.log  the floor under every other number"
note "  3. ${LOGDIR}/gradient_path.log         whether the loss weight was heard"
note "  4. ${LOGDIR}/amplitude_sensitivity_global.log   whether the forward model is usable"
note "  5. runs/RESULTS.md                     every table, assembled"

FAILURES="$(grep -c 'FAILED' "${SUMMARY}" || true)"
note ""
note "steps that failed: ${FAILURES}"
[ "${FAILURES}" = "0" ] || note "grep FAILED ${SUMMARY}"
