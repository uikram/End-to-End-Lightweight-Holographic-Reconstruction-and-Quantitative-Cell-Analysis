# Runbook — every command, in order

Verified on 2026-09-14, and the whole sequence ran on the full 800 fields on
2026-09-16/17. The parts marked **GPU** need the compute machine.

> **If you want to run the study rather than one stage of it, use the drivers
> instead of this file.**
>
> ```bash
> nohup bash RUN_ON_SERVER.sh   > run.out 2>&1 &     # the whole study, ~38 h
> bash RUN_ON_SERVER.sh --status
>
> nohup bash FINISH_ON_SERVER.sh > finish_gaps.out 2>&1 &   # re-score only, ~1 h
> bash FINISH_ON_SERVER.sh --status
> ```
>
> They sequence every stage in dependency order, use GPU 2 and GPU 3 only, gate
> on the self-test, archive any existing `runs/` and `figures/` rather than
> overwriting them, and record each finished phase so an interrupted run resumes.
> This runbook remains the reference for running one piece at a time and for
> knowing what to read from each stage.
>
> **Delete `.run_state/` and `.finish_state/` before reusing a folder** — the
> drivers read them as "already done" and skip every phase while appearing to
> succeed.

Run everything from the project root. Use `python` or `python3` as your
environment requires — `run_v2.sh` takes `PYTHON=python3` if the default is
wrong.

---

## Cleaning up afterwards

```bash
bash cleanup.sh              # dry run: prints every path and its size
bash cleanup.sh --apply
bash cleanup.sh --apply --prune-last   # also drops runs/*/last_model.pt (~2 GB)
```

`cleanup.cmd` is the same thing for the Windows working copy.

Two of the things it removes are hazards rather than clutter:
`figures/fig01..09, fig11..15` are plots of the 2026-09-09 QUICK smoke run and
are indistinguishable from results at a glance, and
`runs/*_modality_comparison.*` are exactly the files `make_figures.py` looks for
when it builds figures 5, 7, 13 and 14 — while they exist, a figure run will
draw smoke-test numbers into a paper figure without warning.

---

## The one command

```bash
bash study.sh
```

That is the whole study. It stops anything already running, waits for the GPUs
to clear, works out which arms still need training, trains both groups in
parallel on GPUs 2 and 3, re-evaluates every arm, runs stages 7–10, and shows
you where it is the whole time.

The work runs **detached**. Ctrl-C stops the watching, not the study, and
closing the terminal or dropping the ssh connection does not stop it either.

```bash
bash study.sh --status     # watch it again
bash study.sh --stop       # stop it for real
```

Knobs, all optional: `GPUS="2 3"`, `FORCE=1` (retrain arms that already
finished), `EVERY=45` (watcher redraw seconds), `PYTHON=python3`.

Two things it decides for you, deliberately:

- **An arm that already trained a full schedule is skipped.** Completion is
  judged by `history.json` holding a full set of epochs, not by the presence of
  `best_model.pt` — that file is rewritten on every improvement, so a run killed
  at epoch 3 leaves one behind and would look finished. `FORCE=1` overrides.
- **W10 is not trained.** Its objective is identical to arm B
  (`cell_integrated_phase: 1.0`) and `collect_results.py` already reports B
  under both names.

The rest of this runbook is the same work done by hand, which is what to reach
for when one stage needs re-running on its own.

---

## 0. Once, before anything else

```bash
cd C:\Users\iivs\Desktop\lightweight_qpi_segmentation
pip install tifffile                    # only if missing; needed for the membrane TIFFs
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

That must print `True`. If it prints `False` every arm will train on CPU and the
50-hour budget below becomes weeks.

### Confirm the ImageNet weights are found locally

```bash
python -c "import sys; sys.path.insert(0,'.'); from holoqpi.config import load_config; from holoqpi.models import build_model; build_model(load_config('config/v2/a_baseline.yaml'))"
```

Look for this line:

```
encoder=mobilenet_v2 in_channels=1 pretrained=True weights=local (weights) stages=[...]
```

`weights=local (weights)` means your `weights/mobilenet_v2-b0353104.pth` was
used. If it says `weights=download cache or random`, the local file was not
found and — behind your proxy — **every arm will train from scratch while the
config still says `pretrained_encoder: true`**. Fix that before burning GPU
hours. `model.pretrained_dir` in `config/base.yaml` controls the folder.

---

## 1. The stages that need no GPU (~1 h)

These produce publishable results on their own and are prerequisites for the
rest, so run them first whatever else you decide.

```bash
bash run_v2.sh --stage 0     # 82 self-tests; nothing else runs if these fail
bash run_v2.sh --stage 1     # masks, splits, manifest, label audit, MEMBRANE registration
bash run_v2.sh --stage 2     # aberration surfaces (global + per_field + hybrid)
bash run_v2.sh --stage 3     # amplitude reference
bash run_v2.sh --stage 4     # pre-flight diagnostics  <- READ THIS ONE
bash run_v2.sh --stage 5     # error propagation + synthetic ground truth
```

Or in one line:

```bash
for s in 0 1 2 3 4 5; do PYTHON=python bash run_v2.sh --stage $s; done
```

### What to read from stage 1

```
logs/v2_<stamp>/membrane_registration.log
logs/v2_<stamp>/membrane_prepare.log
```

The registration must still report a **constant** offset on all 800 fields:

```
offset_y  median 314  sd 3.0  range 308..318
offset_x  median 280  sd 5.7  range 274..292
-> the offset is CONSTANT across fields (sd < 6 px)
```

If the standard deviation comes out above ~6 px on the full set, the two
cameras were not in a fixed relationship for every acquisition and the single
global transform is wrong. **Stop and tell me** — do not patch it with
per-field offsets, because fitting those needs the phase, which destroys the
independence that makes the membrane labels worth having.

### What to read from stage 4

This stage sets a loss weight and decides whether a loss term is usable. Both
questions were previously answered by argument instead of measurement, wrongly
in both cases.

```
logs/v2_<stamp>/gradient_path.log                  <- the weight arithmetic table
logs/v2_<stamp>/amplitude_sensitivity_global.log   <- is the forward model usable
logs/v2_<stamp>/amplitude_sensitivity_per_field.log
logs/v2_<stamp>/calibrate_z.log
```

In `gradient_path.log`, read the table, not the ratio:

```
    weight w    effective ratio      seg : term
       1.000             0.27           3.7 : 1
  parity (effective ratio 1.0) is at w = 3.7
```

If the parity weight on the full dataset differs a lot from what we measured on
13 fields, tell me and I will move `cell_integrated_phase` in
`config/v2/b_cell_ipp.yaml` and the sweep before you train.

---

## 2. Training — three passes, in priority order (**GPU**)

Thirteen arms at ~3–4 h each is 50–70 h. Do not start that blind.

### Pass 1 — the main hypothesis and its control (~10–12 h)

```bash
CLEAN=1 ARMS="A B B1" SEEDS="1337 2024" nohup bash run_v2.sh > v2_pass1.out 2>&1 &
tail -f v2_pass1.out
```

**This is the only pass that can answer the paper's central question.** A vs B
says a measurement-aware loss helps; B vs B′ says the *per-cell* part is what
helped. Three seeds each is what makes the significance column mean anything —
without it no difference can be called at all.

`CLEAN=1` archives `runs/` and `figures/` first. Use it: the pixel pitch and
refraction increment changed on 2026-09-10, so every earlier checkpoint reports
areas 34.4% low and masses 24.3% low and is not comparable.

**If the differences do not clear 2× the seed spread here, more arms will not
change that.** Write it up as unresolved — the v1 round resolved 0 of 54
comparisons by the same rule, and that is a publishable result, not a gap.

### Pass 2 — the efficiency claim and the remaining terms (~8 h)

```bash
ARMS="KA KB B2 C" nohup bash run_v2.sh > v2_pass2.out 2>&1 &
```

KA/KB measure what the 65% parameter saving (9.60 M → 3.36 M) costs in
accuracy. B2 and C each add one term.

### Pass 3 — amplitude, the forward-model diagnostic, the weight sweep (~10 h)

```bash
ARMS="D0 D1 W01 W03 W10 W30" nohup bash run_v2.sh > v2_pass3.out 2>&1 &
```

D1 is a **diagnostic**, not a loss ablation — see `config/v2/_shared.md`.

---

## 3. After the passes

Wait until `python scripts/progress.py` reports **FINISHED** for both passes.
Everything below reuses the checkpoints and takes minutes, not hours.

### First, regenerate the unmatched-cell tables

`main.py evaluate` used to write `unmatched_<split>.csv` only when `--tag` was
given, and then under the untagged name. So the arms evaluated before that was
fixed have either no file or the membrane-label file in the Otsu-label slot, and
**figure 15 (recall against cell size) is the figure that reads it.** The fix is
in `main.py`; the tables have to be rebuilt, which costs about a minute per arm
because it reuses the checkpoints and only re-scores them.

Leave `SEEDS` unset here. The replication loop is not covered by `NO_TRAIN`, so
setting it would retrain the seed runs.

```bash
CUDA_VISIBLE_DEVICES=2 NO_TRAIN=1 ARMS="A B B1" bash run_v2.sh --stage 6
CUDA_VISIBLE_DEVICES=3 NO_TRAIN=1 ARMS="<the arms the second pass ran>" bash run_v2.sh --stage 6
```

`grep "arms:" p_rest.out` prints the arm list that pass was launched with; paste
it in rather than retyping it from memory.

Check one file appeared before moving on:

```bash
ls runs/*/unmatched_test.csv
```

### Then the four assembly stages

These four stages read `runs/` and write `figures/` and `runs/RESULTS.md`; they
need a GPU only for stage 8, they take minutes rather than hours, and they must
run in this order because each one consumes what the previous wrote.

Run them in **one terminal, one at a time** — they are quick, and running two of
them at once would have both writing into `figures/`.

```bash
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 7      # conventional baseline (once is enough)
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 8      # ONNX export + hardware benchmark
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 9      # all figures
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 10     # assemble every table -> runs/RESULTS.md
```

Then read, in this order:

```bash
grep FAILED logs/v2_*/SUMMARY.txt
cat runs/RESULTS.md
ls figures/
```

`runs/RESULTS.md` is the document to read and to paste from. Nothing in it is
typed by hand: it is read out of `runs/*/metrics_test.json`.

`runs/RESULTS.md` is the document to read and to paste from. Nothing in it is
typed by hand: it is read out of `runs/*/metrics_test.json`.

---

## 4. Running one thing at a time

If you would rather not use the runner:

```bash
# train and evaluate one arm
python main.py train    --config config/v2/b_cell_ipp.yaml
python main.py evaluate --config config/v2/b_cell_ipp.yaml

# the SAME checkpoint against the independent membrane labels
python main.py evaluate --config config/v2/b_cell_ipp.yaml \
    --tag membrane --set paths.manual_mask_dir=membrane_mask

# a replication seed
python main.py train    --config config/v2/b_cell_ipp.yaml \
    --set project.seed=1337 experiment_name=v2_B_seed1337
python main.py evaluate --config config/v2/b_cell_ipp.yaml \
    --set project.seed=1337 experiment_name=v2_B_seed1337

# export and profile
python main.py export    --config config/v2/k_compact_b.yaml
python main.py benchmark --config config/v2/k_compact_b.yaml
```

`--tag membrane` matters. Without it the second evaluation overwrites
`metrics_test.json` with the membrane-label result under an identical filename,
and you lose the Otsu-label numbers. Two label sources are two results.

---

## 5. Diagnostics, on demand

```bash
python scripts/selftest.py                                              # 82 checks
python scripts/prepare_membrane.py   --config config/base.yaml --estimate
python scripts/prepare_membrane.py   --config config/base.yaml --write-masks
python scripts/estimate_aberration.py --config config/base.yaml
python scripts/prepare_amplitude.py  --config config/base.yaml
python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml --batches 30
python scripts/amplitude_sensitivity.py --config config/base.yaml --split test \
    --set optics.aberration.mode=global data.train_crop=null data.augmentation.enabled=false
python scripts/error_propagation.py   --config config/base.yaml --max-shift 5
python scripts/synthetic_validation.py --config config/base.yaml --fields 12
python scripts/collect_results.py     --config config/base.yaml --split test
python scripts/make_figures.py        --config config/base.yaml
python scripts/progress.py            --watch
python scripts/make_figures.py        --config config/base.yaml --only 17 18 19 20
```

---

## 6. Where the logging is

Every stage writes its own file, and the runner keeps going when one fails so a
single failure does not hide the rest.

```
logs/v2_<timestamp>/SUMMARY.txt        every step, its command, pass/fail, key numbers
logs/v2_<timestamp>/<step>.log         full stdout+stderr for that step
runs/<arm>_off_axis/train.log          per-arm training log
runs/<arm>_off_axis/history.json       per-epoch trajectory
runs/RESULTS.md                        every table, assembled
runs/results_table.csv                 the same, machine-readable
```

### Watching a run

`tail -f` is the wrong tool when two passes run at once: it blocks the terminal
and interleaves two unrelated streams. Use the status reader instead. It only
reads files, so it cannot disturb a run, and it can be run from any terminal at
any time.

```bash
python scripts/progress.py                # a snapshot, then gives the prompt back
python scripts/progress.py --watch        # redraws every 60 s; Ctrl-C to stop
python scripts/progress.py p_main.out     # one pass only
```

It prints, per pass: the arms and seeds it was launched with, how many of the
planned steps are done, which step is running and for how long, the epoch the
current arm has reached with its seconds-per-epoch, and an estimated finish time
for the whole pass from the mean duration of the arms that already finished.

A pass that is not finished but has nothing running is reported as **stopped
early**, with the steps that never ran — that state is otherwise easy to mistake
for completion.

Two greps worth knowing anyway:

```bash
grep "===" p_main.out                     # every step this pass has started
grep FAILED logs/v2_*/SUMMARY.txt         # anything that broke
```

### Two log lines worth checking on every run

```bash
grep "weights=" logs/v2_*/train_*.log      # must say weights=local (weights)
grep "objective:" logs/v2_*/train_*.log    # the arm's actual loss weights
```

The second prints the whole objective, e.g.

```
objective: amplitude=0.1  cell_integrated_phase=1  phase=1  segmentation=1
```

That line is how you confirm six weeks later which arm a checkpoint belongs to.

---

## 7. What is new since the last handover

| file | what it does |
|---|---|
| `scripts/prepare_membrane.py` | registers the membrane channel onto the phase grid |
| `scripts/collect_results.py` | assembles every table into `runs/RESULTS.md` |
| `scripts/prepare_amplitude.py` | writes the classical amplitude reference |
| `scripts/amplitude_sensitivity.py` | is the forward-model term usable as a loss |
| `scripts/error_propagation.py` | boundary error → measurement error |
| `scripts/synthetic_validation.py` | the pipeline's own error floor |
| `config/v2/_shared.md` | the thirteen arms and the question each answers |
| `figures/fig17` | the label-free headline result |
| `figures/fig18` | two error floors, two different claims |
| `figures/fig19` | the v2 ablation, with the significance band drawn |
| `figures/fig20` | the membrane labels against the phase labels |
| `scripts/progress.py` | read-only status of both training passes |
| `cleanup.sh` / `cleanup.cmd` | remove the smoke-test leftovers, on the server and on Windows |
| `docs/v2_code_changes.md` | before/after for every code change |
| `docs/HoloQPI_Technical_Documentation.pdf` | 108-page implementation reference, no results |

`model.pretrained_dir: weights` and the whole `membrane:` block are new in
`config/base.yaml`. `main.py evaluate` has a new `--tag`.

## 8. New on 2026-09-16 and 2026-09-17

| file | what it does |
|---|---|
| `RUN_ON_SERVER.sh` | the whole study in one resumable command, GPU 2 + GPU 3 |
| `FINISH_ON_SERVER.sh` | re-score and rebuild the reporting, no retraining |
| `holoqpi/metrics/amplitude.py` | scores the predicted amplitude against the reference **and** against the thin-phase assumption A = 1 |
| `config/v2/l_lora.yaml` | arm L, LoRA adaptation of the encoder (defined, not trained) |
| `docs/code_audit_2026-09-16.md` | the full-codebase audit and every fix |
| `docs/gap_closure_2026-09-17.md` | the four reporting defects found afterwards and their fixes |

Changes worth knowing when reading older logs:

* **The self-test is now 143 checks**, up from 137 and before that 89. It is the
  gate in both drivers; it must end with `All checks passed.`
* **`runs/RESULTS.md` gained Table 3c**, the amplitude output. It is blank for
  every arm whose amplitude head is off, which is correct — those arms predict no
  amplitude.
* **Table 4 no longer marks a row `UNSTABLE` on a start-of-run memory reading.**
  The decision now rests on `timing_stable` (p99/p50), which is measured from the
  timings themselves; device occupancy is reported as a note. A run before this
  change may have refused to quote sound measurements.
* **`figures/_manifest.json` merges** across the three `make_figures`
  invocations in stage 9 instead of being overwritten by the last one.
* **Stage 5 is in the driver.** It was omitted from the first version of
  `RUN_ON_SERVER.sh`, so a run from that version has an empty "Results that need
  no trained model" section and figures 17 and 18 left over from an earlier run.
* **The learned-distance pad is frozen at the configured z**, so the residual's
  comparison domain no longer moves as z is optimised, and
  `training.physics_parameter_lr_scale` is 100 rather than 10000 — Adam's step
  size is roughly the learning rate regardless of gradient magnitude, and the old
  value moved z by up to 3 µm per step.
