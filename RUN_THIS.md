# Run this

Two drivers. Both use **GPU 2 and GPU 3 only**, never touch GPU 0 or 1, and are
safe to re-run: each records the phase it finished, so an interrupted run picks
up where it stopped rather than starting over.

---

## The whole study — about 38 hours

```
nohup bash RUN_ON_SERVER.sh > run.out 2>&1 &
```

Check on it whenever you like, from any terminal:

```
bash RUN_ON_SERVER.sh --status
```

Ten phases, in dependency order:

| | |
|---|---|
| 1 | archive any existing `runs/` and `figures/` to `*_before_audit_<stamp>` |
| 2 | **self-test — the gate.** 143 checks; nothing below runs if it fails |
| 3 | regenerate the masks and record the parameters that produced them |
| 4 | stages 1, 2, 3, 5 — label audit, aberration surfaces, amplitude reference, label-free analyses |
| 5 | stage 4 diagnostics in the background, and training on both GPUs in parallel (~35 h) |
| 6 | stage 7 — the classical reconstruction baseline |
| 7 | stage 8 — ONNX export and the benchmark, **alone on the GPU** |
| 8 | pair off-axis with in-line |
| 9 | stage 9 — the figures |
| 10 | stage 10 — the seed aggregate and the tables |

Read `runs/RESULTS.md` first when it finishes.

## Re-score without retraining — about an hour

```
nohup bash FINISH_ON_SERVER.sh > finish_gaps.out 2>&1 &
bash FINISH_ON_SERVER.sh --status
```

Self-test, stage 5, re-score the amplitude arms, figures, tables. Every trained
checkpoint is used exactly as it is. Use this after any change to a metric, a
figure or a table.

---

## Before you run either one

**Delete `.run_state/` and `.finish_state/` if they exist.** The drivers read
them as "already done" and will skip every phase while appearing to succeed.

```
rm -rf .run_state .finish_state
```

**Check the code arrived complete.** Two partial folder copies have broken this
project's package before:

```
find holoqpi -name "*.py" | wc -l
ls data/phase | wc -l
python -c "import holoqpi.analysis.cells, holoqpi.deploy.benchmark, holoqpi.metrics.amplitude; print('imports OK')"
```

Expect **39**, **800**, `imports OK`. If the first number is lower, the copy is
incomplete — stop, because the self-test will fail with a `ModuleNotFoundError`
rather than anything informative.

---

## Moving files between machines

Never whole-folder paste in either direction. That is what broke the tree twice.

* **Off the server:** `tar -czf` the named files and move one archive.
* **Onto the server:** extract into an empty folder first, then copy.
* `data/` and `weights/` are multi-gigabyte **inputs**, not clutter — leave them
  alone. Replace only code, `runs/` and `figures/`.
* `runs/*/last_model.pt` is never read. Skip it and a transfer of `runs/` roughly
  halves.

To send a result set back:

```
tar -czf holoqpi_results.tar.gz \
    runs/RESULTS.md runs/results_table.csv runs/seed_aggregate.json \
    runs/z_calibration.json runs/error_propagation_summary.csv \
    runs/synthetic_validation_fields.csv \
    runs/*/metrics_test.json runs/*/history.json runs/*/per_cell_test.csv \
    runs/*_modality_comparison.json runs/*hardware_benchmark*.json \
    figures/ selftest.out stage4.out gpu2.out gpu3.out finish.out
```

---

## Running one thing at a time

`run_v2.sh` takes a single stage or a subset of arms, and
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) says what to read from each:

```
bash run_v2.sh --stage 5
ARMS="A B B1" bash run_v2.sh --stage 6
SEEDS="1337 2024" bash run_v2.sh --stage 6
NO_TRAIN=1 bash run_v2.sh --stage 6
QUICK=1 bash run_v2.sh
```

Arm L (LoRA) is defined in `config/v2/l_lora.yaml` and has not been trained. It
is one arm, about 3.5 hours, on either GPU once that GPU is free:

```
CUDA_VISIBLE_DEVICES=2 nohup bash -c 'ARMS="L" bash run_v2.sh --stage 6' > gpuL.out 2>&1 &
```

## Reading the output

`run_v2.sh` prints one line per step. Empty output from

```
grep FAILED gpu2.out gpu3.out
```

means nothing failed. A non-zero exit that is a **verdict** from a diagnostic
prints `VERDICT` and is not a failure; a hard error — a missing checkpoint, a
null distance — exits 3 and is reported as `FAILED`.
