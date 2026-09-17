# HoloQPI — full code audit and fix pass, 2026-09-16

Audit of the complete codebase (31 files changed), performed before the final
experimental re-run. Every finding below was verified against the code rather
than inferred, and every fix was re-checked by an independent adversarial pass
plus real train/evaluate runs on the 13 local fields.

Self-test: **137 checks, all passing** (was 89).

---

## 1. Verdict

The framework is structurally sound and does implement the intended pipeline:
raw hologram → lightweight network → quantitative phase + amplitude + cell
segmentation, with a genuine physics/data-fidelity term that propagates the
predicted complex field to the sensor and compares it with the recorded
hologram.

Three defects would have changed reported numbers. Nine more would have let a
failure pass unnoticed. The rest are consistency, provenance and honesty fixes.

**Two things are worth knowing before the re-run:**

1. **Arm D2's previous result was produced by a bug, not by physics.** The
   learned `z` was given an unclipped gradient at 10 000× the base learning
   rate. With Adam the step size *is* the learning rate, so z moved up to 3 µm
   per step — larger than the plausible range of the parameter. Its excursion to
   ±42 µm and settling at −4.95 µm was an artefact of that. Fixed; a real
   2-epoch run now holds z at 33.77 ± 0.03 µm.
2. **The forward-model term's effective weight changed.** It was being divided
   by the batch size while the logged value was divided by the number of usable
   fields. `loss.weights.forward_model = 0.02` therefore does not mean the same
   thing as in the previous D1/D2 runs, and those numbers are not comparable
   with the new ones.

---

## 2. Defects that would have changed the reported numbers

### 2.1 `z` had the only unclipped gradient in the run — `holoqpi/engine/trainer.py`, `config/base.yaml`

`torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)` clips the
network. The propagation distance is a parameter of the **objective**, not of the
model, so it was excluded — while simultaneously being the parameter with the
largest learning rate in the run.

`physics_parameter_lr_scale: 10000` was justified by "z lives in micrometres, so
it needs a far larger step than a network weight". That reasoning holds for SGD
and is wrong for Adam, which normalises each step by the gradient's own running
magnitude: the update is ≈ ±lr per step regardless of the gradient. At 10 000 the
rate was 3.0, i.e. up to 3 µm per step and ~420 µm per epoch.

- The criterion's parameters are now clipped, **separately** from the model's (a
  joint clip would let one large z gradient scale down every weight gradient).
- `physics_parameter_lr_scale: 100` → 0.03 µm/step. Over a 60-epoch schedule z
  can still travel ~250 µm if the gradient is consistently signed, so a real
  minimum near 33.77 µm remains reachable; a single noisy batch cannot move it
  by more than a thirtieth of a micrometre.
- z is **not** clamped. If it still leaves the sensible range that is the
  reportable finding about off-axis z-identifiability, and it must be reported
  rather than hidden behind a bound.

### 2.2 The residual's comparison domain moved with `z` — `holoqpi/losses/terms.py`

`required_pad()` derived the diffraction margin from the **live** value of z, and
`border_px` defaults to that margin — so the residual was computed over a
different set of pixels every time z moved: a 742 px window at z = 33.77 µm and a
776 px one at z = −5 µm. Gradient descent could lower the residual purely by
shrinking |z|, because that changes which pixels are compared, and two residuals
from different epochs were not on the same scale.

The pad is now frozen at the **configured** distance (`_reference_distance_um`),
so the objective is a smooth function of z over one fixed domain — the only form
in which "learn z" means anything. An explicit `pad_px` still overrides, which is
what a z sweep needs. A separate warning fires if a learned z drifts far from
where the pad was sized, since the field is then genuinely under-padded.

### 2.3 The forward-model weight fluctuated with the batch — `holoqpi/losses/composite.py`

`total` is a per-sample vector the caller reduces with `.mean()`. Adding the
per-sample `forward_term` contributed `Σ/B`, while the value written to the log
was `Σ/n_usable`. The applied weight therefore drifted from the logged one by
`n_usable/B`, batch by batch, as the number of fields with a recoverable
aberration surface changed.

Now reduced to a scalar once: the contribution is exactly
`w · mean_over_usable_fields`, which is how every other physics term is already
handled, and it equals the logged value. Verified numerically.

### 2.4 The verdict gating the forward-model weight was measured on the wrong window — `scripts/calibrate_z.py`

`fixed_border_px` sizes a border for the largest |z| in the sweep — 225 px for a
±96 µm scan — and that border was then reused for the two **fixed-distance**
tests: the discrimination verdict and the phase-scale probe. Those ran on a
450 px window while training and `metrics/forward.py` use 742 px. Their
thresholds are in *residual units* (`discrimination_tolerance: 0.01`), and a
residual's level depends on how many and which pixels it covers — so the decision
that gates `loss.weights.forward_model` was taken at a window size unrelated to
the term as trained.

Added `training_border_px()`; the fixed-z tests now use the objective's own
border. Also in this file:

- the sweep now freezes `pad_px` as well as `border_px` (the padded FFT grid
  still grew with |z| even though the output domain did not);
- `scan()` now drops fields whose aberration surface was rejected, as the two
  verdict tests already did — `z_forward_um` was computed over a superset of the
  images the verdict was;
- the inverse estimator is scored on the same window as the forward residual
  (the two z estimates came from different pixel regions and were printed side
  by side as corroboration);
- `phase_sensitivity` is now per radian (it divided by ε but not by the
  perturbation's size in radians, so the 1e-3 threshold scaled with each batch's
  own phase contrast);
- `--out` no longer replaces the whole file, so `--modality gabor` stops
  deleting the `off_axis` entry — the off-axis/in-line contrast in that file is
  the result it exists to record;
- `feature_um` comes from config, and the scan grid is built once rather than
  twice from the same arguments;
- the "no modality produced an identifiable z" advice was **unreachable** (the
  branch above it was a tautology) and the message that did print was the wrong
  diagnosis for the verdicts that can reach it. Both are now reachable and say
  what they mean.

### 2.5 The loss weight for the central term was set from a biased measurement — `scripts/check_gradient_path.py`

The objective gates crops below `min_foreground_pixels` and divides by the number
of **surviving** crops. This script used a plain `.mean()` over the whole batch —
including the gated crops, for which `CellIntegratedPhase` returns exactly zero.
Every below-threshold crop therefore pulled the measured gradient ratio down by
`n_surviving/B`, and that ratio is what set
`loss.weights.cell_integrated_phase` for arm B. Now reduced exactly as the
objective reduces it, and the number of gated crops is reported.

Also: the "comparable in scale" verdict fired for any ratio above **1e-3**, so a
term 10× to 1000× weaker than the segmentation loss was reported as comparable —
and that band is where this project's own measurements landed (0.25–0.54, with a
worked example of 0.090 described as "an 11:1 advantage to the segmentation
loss"). Threshold is now 0.1, the order-of-magnitude reading. The CSV is keyed by
config and crop size instead of one fixed name that each arm truncated.

### 2.6 Half the aberration-mode result was being destroyed — `scripts/amplitude_sensitivity.py`

`run_v2.sh` runs this twice, once per aberration mode, and its own comment says
"the difference between them IS the result". Both runs use the same modality, and
the CSV was keyed by modality only — so the `global` run overwrote the
`per_field` run and the rows carried no mode column to tell them apart. Now keyed
by mode, with the mode, border and distance recorded in every row.

Also: the border defaulted to a hardcoded 64 px, which is **smaller** than the
79 px pad the term applies, leaving a 15-pixel ring of wrap-around inside the
residual. It now defaults to the term's own pad. A fourth verdict sign
combination had no branch and fell through to a message that could declare the
unit-amplitude assumption adequate while a negative coarse response had already
been printed under a heading saying it must be positive.

---

## 3. Defects that would have let a failure pass unnoticed

### 3.1 The seed-replication result was never computed for v2 — `scripts/aggregate_seeds.py`

This script implements the project's significance rule, and it could not see its
own data. It globbed `*_modality_comparison.json`, written only by
`main.py compare`, which `run_v2.sh` never runs for a seed replicate — the seed
arms are trained and evaluated with `train`/`evaluate`. Its seed-suffix regex
(`_s(\d+)$`) also failed to match `run_v2.sh`'s `_seed1337` naming. With its
default baseline (`base`) absent from every v2 run it compared nothing, exited 0,
and printed **"NOTHING IS RESOLVED"** — which reads as a measured finding and was
a statement about an empty table.

Rewritten to read `runs/<experiment>_<modality>/metrics_<split>.json`, the same
place `collect_results` looks; both naming conventions recognised; arm codes
mapped to experiment names; baseline resolved from arm A's own config; restricted
to this study's arms (it was differencing the classical reconstruction baseline
and the v1 ablations against arm A as though they were ablations of it). A missing
baseline now exits 3, and "no replication" is reported as unresolvable rather
than as a null result.

Two arithmetic bugs went with it: the single-run guard accepted a finite SD from
**either** arm, so a one-run arm was judged against a three-run arm's spread and
printed as RESOLVED — a verdict the code's own comment said was impossible; and a
pooled SD of exactly zero forced `resolved = False`, labelling an arbitrarily
large real difference "within seed noise" when the noise was measurably nil.

### 3.2 The significance rule existed three times with two different formulas

| where | formula |
|---|---|
| `aggregate_seeds.py` | `sqrt(s_a² + s_b²)` |
| `collect_results.py` | `sqrt(0.5(v_a + v_b))` |
| `make_figures.py` fig 19 | `sqrt(0.5(v_a + v_b))` |
| `make_figures.py` fig 7 | a flat **±2%**, unrelated to any measurement |

The first is √2 larger than the second for equal n, so the same comparison could
be resolved by one document and noise by another. There is now one
implementation — `holoqpi.utils.pooled_between_seed_sd` — and `resolve_factor`
comes from config in all three places instead of a literal `2.0`.

### 3.3 Contaminated efficiency numbers were undetectable — `holoqpi/deploy/benchmark.py`

This is the defect that inverted the efficiency conclusion once already: stage 8
overlapped a training job on the same GPU, the tail latencies roughly tripled, and
the compact decoder came out *slower* than the full one. Nothing in the numbers
showed it.

- `foreign_memory_mb(device)` reports memory held by other processes and warns
  before any timing; `device_idle_at_start` goes into every row.
- `latency_p99_over_p50` and `timing_stable` go into every row (limit 1.25; an
  idle GPU measures 1.01, a contended one 2.7–3.0).
- `collect_results` marks unstable rows **UNSTABLE** and refuses to present them
  as measurements, distinguishing a GPU row (contention) from a CPU/ONNX row
  (scheduler jitter).
- Every CUDA call now names the requested device — `mem_get_info`,
  `memory_reserved`, `synchronize`, `reset_peak_memory_stats` all defaulted to
  the *current* device, so `--device cuda:1` measured GPU 0.
- fp16 timing runs on a deep copy. It used to `.half()` the caller's model and
  `.float()` it back, round-tripping every weight and BatchNorm statistic through
  10 mantissa bits — and the fp16 ONNX export, a deliverable, was written from
  the degraded model. `export.py` had the same in-place `.half()` on the wrapper.
- Peak memory is now reported as weights + activations rather than as whatever
  the process happens to hold, so the fp16 and fp32 rows are comparable.

### 3.4 Hard errors were logged as scientific findings — `run_v2.sh` and five scripts

`run_v2.sh` treats a traceback-free non-zero exit as "the script ran and is
reporting a finding", which is right for the diagnostics that carry verdicts in
their exit codes. But `raise SystemExit("message")` also exits non-zero without a
traceback — so "no images were read", "distance_um is null", "every aberration
surface was rejected" and "no checkpoint for any modality" were all recorded as
successful diagnostics and did not count towards the failure total.

Exit code **3** is now reserved for a hard error, and the runner treats it as
FAILED. Applied in `amplitude_sensitivity.py`, `diagnose_bias.py`,
`aggregate_seeds.py`; `calibrate_z.py` and `check_gradient_path.py` now raise
properly so the traceback reaches the log.

### 3.5 Stale artefacts were reused silently — `holoqpi/data/masks.py`, `scripts/prepare_amplitude.py`

Both generators skip files that already exist, neither recorded what produced
them, and the only consumer-side check is the array's **shape** — which none of
the relevant parameters changes. Changing `smoothing_sigma_px`, `otsu_scale`,
`border_buffer_px` or `sideband_radius_px` and re-running silently reused the old
files, and the run then trained and measured against artefacts from a different
configuration. For the masks that is acute: they are simultaneously the
segmentation target and the domain for every per-cell measurement.

A `_provenance.json` is now written beside each generated directory, and reuse
under changed parameters **raises** with a diff and the regeneration command. It
is written only when the run actually wrote files, so it cannot bless artefacts
it did not produce — and when everything was skipped it says so.

### 3.6 `trust_header_pitch` logged that it had done something, and did nothing — `scripts/prepare_data.py`

The flag appeared in exactly two places in the codebase, both of them a log line
saying "header values will be preferred". `calibration_from_config`
unconditionally reads `optics.pixel_pitch_*`. Since the pitch multiplies every
projected area, optical volume and dry mass, a flag that claims to change it and
does not is the most dangerous kind of no-op — and on this dataset honouring the
header would be actively wrong, because the acquiring group confirmed the
header's y value is the fluorescence camera's pitch. It now raises.

### 3.7 `compare` retrained by default, then scored random weights — `main.py`, `holoqpi/engine/compare.py`, `run_study.sh`

`python main.py compare --config config/v2/a_baseline.yaml` — the invocation
`run_v2.sh` documents for pairing two trained arms — retrained both for 60 epochs
and overwrote the checkpoints the study was about to report. Training is now
opt-in (`--train`).

That inversion had a second-order consequence I caught in the re-check pass:
`run_study.sh` trains every arm *through* `compare` and has no `train` step at
all, so it would have silently stopped training anything and written a full set
of metrics, tables and figures from randomly initialised weights behind a
per-arm WARNING. `run_study.sh` now passes `--train` explicitly, and
`compare` **refuses** to score a missing checkpoint rather than warning —
`main.py evaluate` already refused the same case.

### 3.8 Figures could not be produced, and reported themselves as built — `scripts/make_figures.py`

- Nothing in `run_v2.sh` ever ran `main.py compare`, so figures 5, 13 and 14
  could never be produced by a full run however many arms were trained. Stage 6
  now pairs the geometries once both arms exist.
- Figures 7 and 13 globbed every `*_modality_comparison.json`, which includes the
  **seed replicates** — so pure seed noise was plotted as separate ablation rows.
  Filtered.
- `built.append(number)` ran whenever a figure function *returned*, and every
  figure returns normally when it skips for a missing input. The summary then
  printed a directory listing of `fig*.png`, including files from previous runs.
  Writes are now tracked, skips are reported as skips, and a `_manifest.json`
  records which config, modality and split produced the figures — the filenames
  carry none of that, so one config's figures silently overwrite another's.
- Figure 19 dropped any metric that was genuinely `0.0` on every arm
  (`0.0 or np.nan` is `np.nan`), and drew one shaded band of the widest spread
  while colouring each bar against its own — so a grey "inside the band" bar
  could sit outside the shading. Each bar now shows the band it was judged
  against.
- Figure 5 lacked the `or {}` guard the other figures have and raised on a
  `"gabor": null` entry; figure 17 called `np.nanmax` on a possibly empty slice;
  figure 3's offset tuple was hardcoded to exactly two modalities; figure 18
  labelled a mean-absolute-error and the magnitude of a mean both as `|bias|`;
  figure 8 defaulted unknown ONNX provenance to "comparable" and marked PyTorch
  reference rows as not comparable to themselves.

---

## 4. Correctness and consistency fixes

- **`correlation` criterion was a verbatim copy of `l2`** — selecting it silently
  got `l2` while the docstring described something else. Now `1 − Pearson r`,
  and the self-test asserts the two differ.
- **Every pixel-sum and `scatter_add` integral is pinned to float32.** See §6 for
  the important caveat about what this did and did not fix.
- **The learned `z` was never saved and never reached evaluation.** It is a
  parameter of the loss, not the model, so `model_state` did not carry it: arm
  D2's reported `forward_residual` was computed at the configured 33.77 µm while
  the trained value was elsewhere. The checkpoint now carries `loss_state` and
  `forward_distance_um`, `apply_learned_physics` restores it at evaluate and
  compare time, and `forward_distance_um` is written into the metrics. The
  per-epoch trajectory (endpoint *and* step-mean) goes into `history.json`.
  Guarded so a fixed-z checkpoint cannot hijack an explicit distance override —
  the distance is a persistent buffer when `learn_distance` is false, so it
  appears in `loss_state` for every run ever saved.
- **`ForwardModelMetrics`** carried five duplicated geometry attributes and a
  private `_pad` reimplementation of the padding rule, none of it ever called.
  Two copies of one rule is how the metric and the objective drift apart, which
  is the one thing that class exists to prevent. Removed. Its
  `forward_residual_ratio` comment had the direction inverted.
- **`estimate_aberration.py`**: the pixel↔normalised-grid mapping used `height`
  where `np.linspace(-1,1,height)` implies `height − 1` (0.11%, confined to the
  hybrid mode, but the grid convention must match `polynomial_basis` and
  `render_aberration` exactly). An unguarded reduction over a possibly-empty
  array raised *after* the JSON was written, leaving a half-successful run. The
  reconstruction settings the surface was fitted under are now recorded in the
  payload — the surface is added back to the predicted phase by every
  forward-model residual in the study, so one fitted under a different sideband
  radius is quietly wrong in all of them. A peak-to-valley ratio labelled "the
  share that is piston+tilt" is not an additive decomposition and is now named
  for what it is.
- **`conventional_baseline.py`** duplicated `polynomial_basis` with the dtype
  dropped (float32) and solved it with a plain `lstsq` — at the configured
  `aberration_order: 3`, which is precisely the case the shared module's
  docstring warns loses accuracy. Now uses the ridge-stabilised float64 path.
- **`audit_labels.py`** hardcoded Otsu while `masks.py` branches on
  `threshold_method` — and this script's own verdict *recommends* switching to
  `fixed`, under which it was reporting a CV and an ANOVA p-value for a threshold
  that is a constant. It also reported the raw threshold's foreground fraction as
  the mask's, and its between/within condition SDs were unweighted. Outputs are
  now scoped by split and label source, and it warns when
  `paths.manual_mask_dir` makes its whole premise inapplicable.
- **`resolve_z.py`** swallowed every exception from config loading, so a typo'd
  path and a genuinely null `distance_um` were indistinguishable and the caller
  reported "no usable propagation distance" as a measurement. Paths now come from
  `paths.output_root` instead of the literal `"runs/"`, and both are overridable.
- **`diagnose_bias.py`** called `run_directory`, which *creates* the path, before
  testing for a checkpoint — leaving an empty `runs/<arm>_<modality>/` that
  `collect_results` and `make_figures` then saw as an untrained arm. Images with
  an empty GT or predicted mask were tallied nowhere, so a detector that found
  nothing was diagnosed as inverted contrast.
- **`prepare_data.py`** checked the geometry of the first **8** of ~800 files and
  phrased its warnings as statements about the dataset. A new header-only reader
  makes the full pass free; a disagreement now names the files and raises.
- **`SegmentationMetrics.update`** used `np.atleast_3d`, which *appends* the axis
  — a single `(H, W)` map became `(H, W, 1)` and every row was scored as its own
  image. `phase.py` already carried the fix.
- **`boundary_counts`** and **`aggregated_jaccard`** credited a perfect score for
  a field with no cells on either side, inflating `seg_boundary_f1` and `seg_aji`
  by however many empty fields a split holds.
- **Hardcoded physical constants moved to config**: the amplitude reference's
  `clip_max` and `min_background_fraction`, and the classical baseline's
  `min_recovered_contrast_rad` (which decides whether the whole baseline is
  reported as failed). `fit_radiometry` is read from config in both scripts that
  were hardcoding it `True`.
- **Unscoped output filenames** that one run truncated from another:
  `gradient_path.csv`, `aberration_fit.csv`, `amplitude_reference.csv` (a
  `--limit 4` diagnostic replaced the 800-field table), `amplitude_sensitivity_*`,
  `label_audit_*`.
- **`PhaseMaskContrast`** used `print()` instead of the logger, so its
  mask-collapse warning never reached the run's `train.log`.

---

## 5. Research-requirement gaps

### 5.1 No arm uses LoRA, and the project is named for it

`model.lora.enabled` is `false` in `base.yaml` and overridden nowhere. The
machinery is implemented and works — I verified it end to end: 35 layers
injected, gradients reaching only the low-rank factors, no gradient on frozen
weights, an exact merge (0.0 change), ONNX export fine, 7.56 M of 9.75 M
trainable. But **as the study currently stands the manuscript cannot claim
LoRA.**

This is a scope decision, not a defect, so I have added `config/v2/l_lora.yaml`
and registered it as arm **L** without putting it in the default `ARMS` list.
Three defensible options are written up in that file: train it and report LoRA as
an adaptation strategy (`ARMS="L" bash run_v2.sh --stage 6`, one extra arm);
drop LoRA from the title and cite paper 1 for it; or train it as a
parameter-efficiency comparison alongside KA/KB, which attack the same cost from
the decoder side. **This one needs your decision.**

### 5.2 D2 and G were not in the default arm list

Both were added after the list and left out of it, so a full run produced
neither — and G (arm A on in-line holograms) is the only thing that makes the
off-axis/in-line comparison, and figures 5, 13 and 14, possible at all. Both are
now in the default `ARMS`, and stage 6 pairs the geometries automatically.

### 5.3 Two of the five requested measurements were absent from the results document

The brief asks for projected area, circularity, integrated phase, optical volume
and dry mass. Optical volume and circularity were computed by the evaluator,
written into every metrics JSON, and appeared in no table. Added to Table 3,
along with the per-cell Pearson r and Bland-Altman bias for area.

### 5.4 The physics-aware half had no table at all

`forward_residual`, its ground-truth floor, the ratio and the learned distance
were all computed and never surfaced — so the component the extension is named
after was missing from the document the manuscript is assembled from. New
**Table 3b**, plus a per-arm learned-z trajectory section that says plainly that
the trajectory is the result and a drifting value must not be presented as a
measured distance.

---

## 6. One claim of mine I had to retract

My first pass concluded that float16 accumulation was **saturating the per-cell
integrals** and had flattened the central term of the study. That is wrong for
the GPU, and I want it on record rather than buried.

CUDA autocast places `softmax` and `sum` in its float32 list
(`AT_FORALL_FP32_SET_OPT_DTYPE` in `ATen/autocast_mode.h`, which I read in the
installed torch). So the foreground probability map came back float32, the
products with it were float32, and these integrals were **already** accumulating
in float32 on every run. There is also a proof that does not depend on reading
the policy table: before the change, `CellIntegratedPhase` built its accumulator
from `foreground * phase` while the reference map was float32-promoted, so if
`foreground` had been float16 the two would have disagreed and `scatter_add`
would have raised on the first step. Arm B trained for sixty epochs with that
term at weight 1.0.

**No recorded result is affected by the dtype change.** It stays in because the
CPU autocast policy is a different, smaller list that does *not* include
`softmax` — the same code raised under it, which the new self-test reproduces —
and because the correctness of the study's central measurement should not rest on
an op being on a version-specific promotion list.

---

## 7. Two things I deliberately did not change

### 7.1 The decoder produces its outputs at half resolution

`decoder_channels` has four entries, so four blocks from stride 32 reach stride 2
and both the phase map and the class map are bilinearly upsampled to full size.
The segmentation boundary that sets projected area and circularity is resolved on
a 2-pixel grid. A fifth entry reaches stride 1 and the code supports it directly.

I did not enable it: it is an architecture change, not a bug fix. It alters the
parameter count, the latency and therefore every number in the efficiency table,
and the arm comparison is only meaningful if they all share one architecture —
so enabling it means retraining everything. What I did fix is the config comment,
which described the fifth block as though it were present. The limitation is now
stated explicitly and should appear in the write-up;
`scripts/error_propagation.py` already quantifies what a boundary error costs.

### 7.2 `z` is not clamped

The config's own note says a runaway must be reported rather than quietly
clamped, and I agree. With the learning rate fixed, a drifting z is now
informative: off-axis, the carrier records phase at any distance, so a flat
residual in z is the *physically expected* answer and a reportable negative
result about identifiability — not a training failure.

---

## 8. Verification performed

- **137 self-test checks pass**, up from 89. New sections: the measurement
  integrals under autocast at this dataset's real magnitudes (with a check that
  establishes the hazard is real so the rest cannot pass vacuously); every one of
  the 15 arm objectives constructed and run under autocast with a backward pass;
  the learned-distance residual domain; and that the three criteria are genuinely
  distinct functions.
- **Real train + evaluate on CPU** for arms D2 (3 epochs), A, B2, KA, L and G
  (1 epoch each) on the 13 local fields. D2's z held at 33.770 → 33.686 µm at
  0.03 µm/step, the cosine schedule decayed the physics group with everything
  else, the checkpoint carried `forward_distance_um`, and evaluate restored it
  (448/448 tensors matched).
- **Guards exercised end to end**: the mask provenance guard refuses a changed
  `smoothing_sigma_px` with a diff; `trust_header_pitch: true` raises; `compare`
  refuses a missing checkpoint and lists what it did find; a fixed-z checkpoint
  no longer claims a learned distance and honours an explicit override.
- **`export` + `benchmark`** run, with the new stability and memory columns
  appearing in the CSV and JSON.
- **Every Python file compiles**; all five shell drivers pass `bash -n`; all 27
  configs resolve including the three new keys.
- An **independent adversarial re-check** of all 14 core changed files, which
  found 13 issues in my own edits — including the `run_study.sh` regression in
  §3.7 and the `apply_learned_physics` over-firing in §4. All are fixed and
  re-verified.
