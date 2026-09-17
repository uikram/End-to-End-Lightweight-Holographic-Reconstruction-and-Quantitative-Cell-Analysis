# Gap closure, 2026-09-17

Four defects found by reading the completed run's own output against the
research requirement. None of them changes a measurement: three are in the
reporting layer and the fourth adds a metric for an output that was already
being produced and trained. No arm is retrained.

The run itself was clean — 137 self-test checks, 800 masks, splits 560/127/113,
`steps that failed: 0` at every stage, no `FAILED` line anywhere, 14 arms plus 6
seed replicates — and every number already in `runs/RESULTS.md` stands except
where listed below.

---

## 1. The amplitude output had no metric

**What was wrong.** The framework's stated output is *raw hologram → phase **and
amplitude** and segmentation*. The amplitude head is built and supervised in
arms D0, D1 and D2 (`model.amplitude.enabled`, `loss.weights.amplitude: 0.1`
against the classical-reconstruction reference), and the predicted amplitude is
used in the forward-model residual — but nothing anywhere compared it to its
reference. `grep amplitude holoqpi/engine/evaluator.py` returned only the loss
plumbing and one argument to `ForwardModelMetrics`.

That is not a cosmetic omission. Inside the forward-model residual the amplitude
is entangled with the phase, the propagation distance and the aberration
surface, so an amplitude head that had collapsed to a constant — which is the
thin-phase-object assumption A = 1, and is exactly what an under-weighted head
converges to — would have left no trace in any table, and the study would have
claimed an amplitude output it had never measured.

**What changed.**

- **New `holoqpi/metrics/amplitude.py`.** `AmplitudeMetrics` accumulates MAE,
  RMSE, bias, Pearson r, the in-cell split, and the mean and standard deviation
  of the prediction itself.
- **The comparator is A = 1, not zero.** `amplitude_unity_mae` is the same MAE
  for the thin-phase assumption the rest of the study runs on, and
  `amplitude_mae_over_unity` is the ratio. Below 1.0 the head is closer to the
  reference than that assumption; at or above 1.0 no amplitude claim survives.
  A collapsed head scores exactly 1.0 by construction, which is the point.
- **`amplitude_pred_sd` is reported** because a near-zero spread says "collapsed
  head" more directly than any error figure.
- **`holoqpi/engine/evaluator.py`** builds the accumulator only when
  `cfg.model.amplitude.enabled`, so an arm that predicts no amplitude gets no
  row rather than a spurious perfect score against the unity field the dataset
  substitutes. The reference is now fetched before the loss block, because the
  metric is needed whether or not a loss is being evaluated — the final test
  pass of an arm runs with a loss, a bare `main.py evaluate` does not.
- **`scripts/collect_results.py`** gains **Table 3c**, listing only the arms
  whose head is on.
- **`scripts/selftest.py`** gains `test_amplitude_metric`, six checks: a
  collapsed head must score exactly 1.0 and report zero spread, a correct head
  must score 0, a head biased the wrong way must score above 1.0, the in-cell
  error must be separated, and an empty accumulator must return `{}` rather than
  zeros. 137 checks → **143**.

**What it does not claim.** The reference is the modulus of a classical off-axis
reconstruction written by `scripts/prepare_amplitude.py`. It is not a
measurement: it carries the sideband filter's lost high frequencies, residual
twin-image structure and any illumination vignetting. Every number in Table 3c
is *agreement with one reconstruction algorithm's output* and must be reported
in those words. The table says so itself.

## 2. Stage 5 never ran

**What was wrong.** `RUN_ON_SERVER.sh` ran stages 1, 2, 3, then 4, 6, 7, 8, 9,
10. Stage 5 was absent. Its two scripts produce the error-propagation bound
(`runs/error_propagation_summary.csv`) and the synthetic-ground-truth floor
(`runs/synthetic_validation_*.csv`), and the consequences were visible in the
output: `runs/RESULTS.md`'s *"Results that need no trained model"* section was
empty under its own heading, and figures 17 and 18 were left standing from an
earlier run.

Those two numbers are not optional extras. They are the floor every measurement
result in the study is read against: a dry-mass MAPE of 0.177 means one thing if
the pipeline reproduces exact ground truth to 1% and another entirely if it
reproduces it to 10%.

**What changed.** The stage loop in `RUN_ON_SERVER.sh` is now `for stage in 1 2
3 5`, with a note saying why stage 5 belongs there (it needs stage 2's
aberration surfaces and nothing else — no checkpoint, no GPU time worth the
name). `--status` lists `stage5`.

## 3. Table 4 suppressed twenty sound measurements

**What was wrong.** Table 4 opened with

> **20 of 20 timing rows have an unstable latency distribution and are marked
> `UNSTABLE` below. Do not quote them.**

while the same rows carried `timing_stable = True` and `p99/p50` between 1.00
and 1.11 — the numbers a quiet device produces. The efficiency claim was
withheld on the strength of a flag contradicted by the evidence in the next
column.

**The cause** was in `holoqpi/deploy/benchmark.py`. `foreign_memory_mb()`
computed `total − free − memory_reserved()`, which charges this process's own
CUDA context, cuDNN handles and cuBLAS workspaces — 400–600 MiB for this model —
to "other processes". On a genuinely idle GPU that exceeds the 256 MiB threshold
and `device_idle_at_start` comes out false. `collect_results.py` then treated
that flag as disqualifying.

**What changed.**

- **Per-process attribution.** `foreign_memory_mb()` now asks `nvidia-smi
  --query-compute-apps=gpu_uuid,pid,used_memory` and excludes this process by
  its own PID, matching the device by UUID. That answers the question actually
  being asked instead of inferring it from an allowance.
- **An honest fallback.** When nvidia-smi is unavailable, or its output cannot
  be attributed to this device, the memory arithmetic is used with a fixed
  1024 MiB allowance for this process's own context. `device_occupancy_source`
  records which of the two produced the number, so a reader can tell a
  PID-attributed zero from an estimate that merely came out low.
- **The measurement decides, the occupancy only cautions.**
  `collect_results.unstable()` now depends on `timing_stable` alone.
  `device_idle_at_start = False` on a row whose timings are stable becomes a
  separate, quieter note that reports the occupancy and says the timings are
  reported. This is the substantive change: contention is *visible in the
  timings* — it leaves p50 roughly intact and inflates the tail, which is what
  `timing_stable` measures — so a start-of-run memory reading is a prior and not
  evidence. A false alarm that suppresses good measurements is not the safe
  direction; it is the same failure as a missed alarm, pointing the other way.

The genuine contention this machinery exists to catch still trips it: the
episode it was built for produced p99/p50 of 2.7–3.0, which fails
`timing_stable` directly.

## 4. The figure manifest recorded three figures of seventeen

**What was wrong.** `figures/_manifest.json` listed figures 5, 13 and 14 only,
while seventeen figures sat in the directory beside it. Stage 9 calls
`make_figures.py` three times — global figures from `config/base.yaml`, per-arm
figures from one arm's config, the modality pair from two modalities — and each
invocation wrote the whole file, so the last one was the only one left. The
manifest exists precisely so nobody has to guess which config produced which
figure, and it was answering that question for three of them.

**What changed.** `make_figures.py` now reads any existing manifest and merges
into it. The schema is `{"figures": {"<n>": {name, config, experiment_name,
modalities, split}}, "last_run": {...}}`. A figure that this invocation skipped
or failed has its old entry **removed**, because the file on disk is then from
an earlier run and the manifest must not vouch for it. A corrupt manifest is
replaced and the loss is logged rather than allowed to block the record.

---

## Re-check of these four changes

- `scripts/selftest.py`: 143/143 pass.
- `Evaluator` constructs and resets for arms A, D0, D2, G and `config/base.yaml`;
  the amplitude accumulator is on for exactly the two configs whose head is
  enabled (and D2, which extends one of them) and off for the others.
- The `(B, 1, H, W)` prediction and reference against a `(B, H, W)` mask were run
  through the evaluator's own conversion; `amplitude_n_images` and the in-cell
  split come out right.
- `collect_results.py` was run against the completed run's `runs/`: it writes
  Table 3c, and Table 4 now carries no `UNSTABLE` mark.
- The manifest merge was exercised with stage 9's three invocations in order:
  nine figures recorded, each against the config that built it, and a figure
  reported as skipped is dropped from the record.

## Running it

`FINISH_ON_SERVER.sh` — self-test gate, stage 5, re-score D0/D1/D2, stage 9,
stage 10. GPU 2 only, nothing is retrained, resumable from `.finish_state/`,
about an hour.

Re-scoring D0, D1 and D2 with an unchanged checkpoint, seed and dataset must
reproduce every one of their existing metrics exactly. If any of them moves, the
re-scoring has found a reproducibility problem and that is worth more attention
than the amplitude row it was run for.

Stage 8 is deliberately **not** re-run. The existing benchmark JSONs record
`timing_stable = True` with `p99/p50` of 1.00–1.11, which is sound data; the
defect was in how it was tabulated, and re-measuring would risk replacing good
numbers with worse ones for no gain.
