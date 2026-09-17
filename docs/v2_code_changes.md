# What changed on 2026-09-10, what was happening before, and what happens now

Every entry is: the problem, what the code did before, what it does now, and how
to check. Nothing here is a guess — each item was either measured on the 13
local fields or asserted in `scripts/selftest.py` (82 checks, all passing).

Read the two blocks marked **CHANGES A REPORTED NUMBER** first. They are the
ones that move published values.

---

## Part 1 — Confirmed errors in the constants

### 1.1 Pixel pitch was anisotropic and wrong · CHANGES A REPORTED NUMBER

**Before.** `config/base.yaml` carried `pixel_pitch_x_um: 0.284871` and
`pixel_pitch_y_um: 0.211994`, taken from the phase `.bin` header. Every area,
optical volume and dry mass used `dx·dy = 0.060391 µm²`, and circularity was
computed on the pixel grid, which is only valid for square pixels.

**Now.** Both pitches are `0.284871`. `warn_on_header_pitch_mismatch` is
`false`, with a comment saying the header's y value is known wrong.

**Why.** Two independent confirmations. S. Park stated the sampling is
0.2849 µm square and the header's y field should be disregarded. Independently,
the EAAI 2026 paper on this same instrument publishes a field of view of
256.38 × 256.38 µm, and 900 × 0.284871 = **256.3839 µm** — which reproduces the
published number to four figures, while 900 × 0.211994 = 190.79 µm does not
reproduce anything.

**Effect.** Pixel area goes 0.060391 → 0.081168 µm², a factor of **1.3440**. So
every absolute area, optical volume and dry mass rises by **34.4%**. Every
*relative* metric — MAPE, Pearson r, Bland–Altman bias, every arm-to-arm
comparison — is unchanged, because the factor cancels.

**Check.** `python scripts/synthetic_validation.py --config config/base.yaml`
prints the calibration it used, and reproduces exact analytic ground truth to
0.04%, which it could not do with a wrong pixel area.

### 1.2 Specific refraction increment · CHANGES A REPORTED NUMBER

**Before.** `refraction_increment_ml_per_g: 0.185`, the common literature value
for protein.

**Now.** `0.2`, confirmed by S. Park as the value used for this dataset.

**Effect.** Dry mass scales by `0.185/0.2 = 0.925`. Combined with §1.1 the net
change to absolute dry mass is `1.3440 × 0.925` = **+24.3%**. Relative metrics
are again unchanged: α appears once, as a multiplicative constant, and cancels
from every ratio.

### 1.3 Circularity had no guard for anisotropic sampling

**Before.** `circularity = 4πA/P²` computed in pixel units, with a comment
assuming square pixels and nothing checking it.

**Now.** `calibration_from_config` warns when `|dx − dy| > 1e-6`, states the
approximate orientation-dependent bias, and says area, optical volume and dry
mass are unaffected because they use the pixel *area*. At the 1.344 aspect ratio
that was in effect, a physically circular cell would have scored about 0.978
instead of 1.000, and the bias depended on the cell's orientation, so it was not
even a constant offset.

**Now moot in practice** — the pitch is square — but the guard stays, because
the next dataset may not be.

### 1.4 The minimum-foreground floor was in the wrong units

**Before.** `loss.physics.min_foreground_pixels: 512`.

**Now.** `370`, which is `30 µm² / (0.284871 µm)²` — the same physical floor as
`min_cell_area_um2: 30.0`, expressed in the pixels the loss actually counts.
Previously the loss silently used a floor 38% larger than the metric's, so a
band of cells was measured but not constrained.

---

## Part 2 — The ONNX export was broken for every v2 config

**Before.** `ExportWrapper.forward` ended with

```python
return out["phase"], out["segmentation"], out["condition"]
```

Every v2 arm sets `model.classifier_enabled: false`, so `out` has no
`"condition"` key and **`main.py export` raised `KeyError` on all of them**. The
efficiency and edge-deployment claim rests on that command.

**Now.** The wrapper derives its outputs from the heads that exist:

```python
OUTPUT_ORDER = ("phase", "segmentation", "amplitude", "condition")

@property
def active_outputs(self):
    present = ["phase", "segmentation"]
    if self.model.amplitude_head is not None: present.append("amplitude")
    if self.model.condition_classifier is not None: present.append("condition")
    return [n for n in self.OUTPUT_ORDER if n in present]
```

and `holoqpi/deploy/export.py` derives `output_names` and `dynamic_axes` from
that same list rather than a fixed tuple, so the ONNX graph's names always match
its outputs.

**Check.** Verified on three configs:

| config | active outputs |
|---|---|
| `v2/a_baseline.yaml` | phase, segmentation |
| `v2/d0_amplitude.yaml` | phase, segmentation, amplitude |
| `base.yaml` | phase, segmentation, condition |

---

## Part 3 — The aberration surface was derived from the training target

This is the change that most affects what can be claimed.

**Before.** `scripts/estimate_aberration.py` fitted one polynomial surface per
field, from the difference between that field's classical reconstruction and
**that field's delivered reference phase** — the network's own target. Two
consequences, and the second is worse than the first:

1. It cannot be reproduced at inference. Given a new hologram and no reference
   phase, that field's surface is unobtainable, so a forward model depending on
   it is not deployable whatever it scores in training.
2. It makes the forward-model discrimination test **partly circular**. The
   surface was fitted assuming the reference phase is correct, so the reference
   phase is guaranteed a favourable residual and the comparison against
   degraded phases is not clean.

**Now.** `optics.aberration.mode` selects between three:

* `per_field` — the old behaviour, kept only so the comparison can be made. Not
  reportable.
* `global` — **the default.** One surface for the instrument: the
  per-coefficient median over the **train split's** valid fields, applied to
  every field including val and test. Obtainable from a one-off calibration, so
  it is deployable, and val/test never inform it.
* `hybrid` — global curvature plus a per-field ramp predicted from the
  hologram's own carrier. Also deployable. **Measured not to work here** (§3.2).

### 3.1 What the global surface costs, measured

```
global surface peak-to-valley                    15.21 rad
per-field departure from it, all terms   median   10.62 rad   (69.8%)
the same departure, degree >= 2 only     median    3.91 rad   (25.7%)
per-field tilt, degree == 1 only         median   13.98 rad
```

The **curvature is close to static**, at a quarter of the surface's magnitude,
which is what a fixed objective should look like. The ramp is not, and it is the
larger share.

### 3.2 A hybrid was tried, and the data rejected it

`reconstruct_off_axis` centres the sideband with `torch.roll`, which shifts by
**whole FFT bins only**, so a carrier that does not land on a bin provably
leaves a residual ramp of up to π rad across the field. `estimate_carrier`
refines the same carrier to a fraction of a bin **from the hologram alone**. If
the fitted tilt were that remainder, it could be predicted at inference and the
surface would be fully deployable.

It is not. The carrier prediction explains **10.9%** of the fitted degree-1 tilt
(12.46 rad left of 13.98 rad), and field by field the two are uncorrelated in
sign as well as magnitude. The tilt is something else — reference-beam drift
between acquisitions, or a ramp introduced by the 2-D unwrapping — and this
script cannot separate those without information it does not have.

The `hybrid` mode is kept selectable so the negative result can be reproduced. A
rejected hypothesis with a number attached is worth more than a deleted one.

---

## Part 4 — The forward-model term cannot be used as a loss · NEW FINDING

`scripts/amplitude_sensitivity.py` (new) scores the **training term itself** on
the reference phase and on degraded phases, with the same radiometric fit, the
same aberration surface, and crops and augmentation disabled so both columns are
the same pixels. A degraded phase must score **worse**.

| response | `per_field` surface | `global` surface |
|---|---|---|
| residual on the reference phase | 0.44 – 0.51 | 0.87 |
| phase × 0.9 (mild, a 10% bias) | **−0.0005** | **−0.0027** |
| phase × 0.5 (coarse, wrecked) | +0.053 | +0.0016 |

**The mild degradation is not penalised in either mode.** A 10% phase bias — the
regime training is in once it has nearly converged, and the only regime in which
a refinement term does anything — *lowers* the residual. Near the truth the
gradient points the wrong way.

**What was believed before.** `config/v2/d_forward_amplitude.yaml` said the
residual was "correctly signed against gross corruption but blind near the truth
(+0.0002 margin for a 10% phase rescaling)". That was too kind. It is not blind,
it is **inverted**, and a coarse-only discrimination test (×0.5, mirrored, zero)
passes it anyway — which is how this was missed.

**And the deployable surface makes it worse.** Under `global`, the coarse
discrimination collapses from +0.053 to +0.0016, a factor of about 35, and the
residual floor doubles. So even the coarse ranking survives only under the
target-derived per-field surface, which is circular by construction.

**What happens now.** Experiment D1's weight stays at 0.02 — low enough that a
wrong-signed gradient cannot drag the phase — and it is reported as a
**diagnostic**, with these numbers, rather than as a loss ablation. The script
refuses to interpret its own amplitude result when the mild phase response is
negative, and says why.

---

## Part 5 — New loss terms

### 5.1 `CellProjectedArea` — the symmetric partner of the mass term

**Before.** `cell_integrated_phase` constrained per-cell mass, and after v2
switched off the image-level `projected_area_consistency`, **nothing constrained
area at all**.

**Why that matters.** Mass is `m = k · Σφ · dA`: a product of a phase and an
area. A boundary pulled inwards while the phase inside is scaled up leaves the
product unchanged, so the mass term alone cannot pin the morphology — and
projected area and circularity are reported outputs in their own right, not
intermediates.

**Now.** A per-cell relative area error over the **reference** instance domains,
averaged over cells, formulated exactly like the mass term so the two are
directly comparable. Weight `0.0` by default; experiment **B″**
(`config/v2/b2_cell_area.yaml`) raises it.

**What it deliberately cannot do.** It sums predicted foreground inside each
reference cell, so it pushes confidence up within known cells and is blind to
false-positive area *outside* them. Suppressing that is the segmentation loss's
job, and detection precision is where it shows. Documented in the class
docstring.

**Check.** Six new assertions in `selftest.py`, and one in
`synthetic_validation.py` against a curved profile rather than flat squares:
losing half of every cell's foreground scores exactly 0.5000; one cell wrong by
25% with the other exact scores 0.1250 (= 0.25 / 2 cells), so a single bad cell
is not diluted by field size.

### 5.2 `AmplitudeReconstructionLoss` and the amplitude reference

**Before.** The forward model propagated a complex field whose amplitude was a
**fixture**: unity everywhere, the thin-phase-object assumption. Defensible for
a transparent cell, but an assumption the code could not check.

**Now.** `scripts/prepare_amplitude.py` (new) writes `|U|` from the classical
off-axis reconstruction, normalised to a robust background estimate, and
`loss.weights.amplitude` supervises the amplitude head against it. Measured over
13 fields: background-normalised median **1.0053** (mask-derived on 13/13), 1st
percentile 0.54, clipping above 1.5 on 0.85% of pixels.

**Stated plainly in the code and in the config:** this is **not a measurement**.
It is one algorithm's reconstruction, carrying the sideband filter's lost high
frequencies, residual twin-image structure and any illumination vignetting. So
the term is a **regulariser** towards a physically plausible modulus, never a
fidelity term against a truth.

**Is the amplitude worth modelling at all?** Measured, yes:

| variant (phase held at the reference) | `per_field` | `global` |
|---|---|---|
| unity → blend 0.25 | −0.0094 | −0.0051 |
| unity → blend 0.50 | −0.0164 | −0.0097 |
| unity → reference | **−0.0215** | **−0.0151** |
| unity → 1.5× past the reference | −0.0178 | −0.0160 |
| unity → **inverted** (2 − reference) | **+0.0438** | **+0.0027** |

Monotone along the blend and worse when inverted, so the direction is physically
meaningful rather than a fitting artefact. The unit-amplitude assumption was
costing the forward model real accuracy.

**But note the scale**, which is also the mechanism behind Part 4: under the
global surface the amplitude moves the residual (0.015) about **nine times more
than halving the phase does** (0.0016). The residual is dominated by amplitude
and aberration mismatch, not by phase.

### 5.3 The amplitude target was not reaching the loss

Found by smoke-testing arm D0 rather than by reading. `data.provide_amplitude:
true` loaded the reference into the batch, but neither
`holoqpi/engine/trainer.py` nor `holoqpi/engine/evaluator.py` copied it into the
target dict, so `KeyError` on the first step. Both now pass it **only when the
dataset actually loaded a reference** — the dataset returns ones otherwise, and
passing those would have the loss quietly score a field of ones and report a
small meaningless value instead of raising.

`trainer.py` also now logs the **whole objective**, not just the physics group:

```
objective: amplitude=0.1  cell_integrated_phase=1  phase=1  segmentation=1
```

An arm whose only extra term was `amplitude` previously logged `active physics
terms: none`, which is exactly the case where that line is load-bearing.

---

## Part 6 — The compact decoder

**Where the parameters were**, measured on `v2/b_cell_ipp.yaml` (9.5981 M):

| component | params | share |
|---|---|---|
| `phase_decoder` | 3.6781 M | 38.3% |
| `segmentation_decoder` | 3.6781 M | 38.3% |
| `encoder` | 2.2233 M | 23.2% |
| each head | 0.0093 M | — |

and inside each decoder a single layer dominates:
`decoder.blocks.0.up = ConvTranspose2d(1280 → 640, k2, s2)` = **3,277,440
params = 34.1% of the whole model**. There are two of them, one per decoder, so
**68.3% of the network is two copies of the same transposed convolution**, whose
only job is to lift the encoder's 1280-channel bottleneck.

**Now.** `model.decoder_bottleneck` (default `null`, so nothing changes unless
asked) inserts one shared 1×1 conv + BN + ReLU6 that compresses 1280 channels
**once**, before either decoder sees them.

**Measured effect** at `decoder_bottleneck: 256`:

| | full | compact |
|---|---|---|
| total | 9.5981 M | **3.3604 M** |
| phase_decoder | 3.6781 M | 0.3951 M |
| segmentation_decoder | 3.6781 M | 0.3951 M |
| shared bottleneck | — | 0.3282 M |

**a 65% reduction.** Whether accuracy survives is an experiment, not an
assertion: arms **KA** and **KB** run against their full-size twins on the same
data, the same seed and the same objective. A null cost is the interesting
outcome; a real cost is publishable too.

---

## Part 7 — Diagnostics that now measure instead of assert

### 7.1 `check_gradient_path.py` — multi-batch, plus cosine, plus the arithmetic

**Before.** One batch, one number: the ratio of the per-cell term's gradient to
the segmentation loss's gradient on the same parameters.

**That number was misread**, in this project, as if 0.299 were the effective
ratio at any weight. It is not — it is the ratio at **weight 1.0**, and the
objective adds `w · term`, so the effective ratio is `w × 0.299`. A weight of
0.1 therefore gave 0.030, a 33:1 advantage to the segmentation loss, not the
near-parity the bare ratio suggested.

**Now.**

* **A median over batches**, with the min and max. Per-batch variation is large
  and real: the same config gave cosine −0.197 and +0.634 on different batches,
  so a single batch cannot set a weight.
* **Cosine similarity** with the segmentation gradient, which answers a
  different question. Near +1 means the term asks for what Dice already asks
  for, so a better result from it is *not* evidence the measurement shaped the
  boundary. Near 0 means orthogonal — new information. The script now returns a
  warning verdict at cosine > 0.9 saying exactly that.
* **The weight arithmetic, printed as a table**, so the mistake cannot recur:

```
    weight w    effective ratio      seg : term
       0.100             0.0540      19.00 : 1
       0.300             0.1619       6.00 : 1
       1.000             0.5397       2.00 : 1
       3.000             1.6192       1 : 1.62
  parity (effective ratio 1.0) is at w = 1.853
```

* **A measured caveat about crop size.** The ratio is median 0.54 at 256 px
  (~7 cells) and 0.25–0.29 at 512 px (~20 cells) — about a factor of two, in the
  direction the formulation predicts, because the per-cell mean divides by the
  number of cells while the segmentation loss is a per-pixel mean that does not.
  The docstring says to re-run it if `data.train_crop` changes.

**Consequence for the configs.** Experiment B's weight moves from **0.1 to 1.0**
(effective 0.27, 3.7:1) because at 0.1 the term is swamped and a null result
would be a statement about the weight rather than about the hypothesis.
`config/v2/w_ipp_{01,03,10,30}.yaml` brackets it.

### 7.2 `MeasurementMetrics` — coverage adjustment and a field-level bootstrap

**Before.** Every per-cell error was computed over IoU-matched pairs only, and
the confidence in it came from the cell count.

**Two holes, both now closed.**

**COVERAGE.** A model that detects the easiest 40% of cells and measures those
perfectly reports an excellent MAPE; one that detects everything and measures it
well reports a worse one. Ranking on the matched-cell error **rewards missing
cells**. Now reported alongside:

```
mape_coverage_adjusted = recall · mape_matched + (1 − recall) · 1.0
```

A missed cell contributes a relative error of exactly 1.0 — its whole mass went
unreported. It is an upper bound on a missed cell's contribution, not a
measurement of one, and the convention is stated wherever the number appears.
Asserted in `selftest.py`: a detector that keeps half the cells and measures
them exactly scores matched MAPE 0.00e+00 and adjusted **0.5000**.

**FIELD TOTALS**, which need no pairing at all. Total dry mass on a field is
what a storage-lesion or drug-response study integrates; a missed cell lowers it
and a false positive raises it, so it includes the detection failures by
construction. Now reported as `*_field_total_bias`, `*_field_total_mape` and
`*_field_total_pearson_r`.

**INDEPENDENCE.** Cells in one field share an illumination, a focus, an
aberration surface and a segmentation threshold, so their errors are correlated
and an interval built from *n* cells treated as independent is too narrow. Pairs
are now grouped **by field**, and `field_bootstrap_resamples: 2000` resamples
fields with replacement. `_paired` became a derived property, so the pooled
point estimate is provably unchanged — asserted in `selftest.py`.

---

## Part 8 — Two analyses that need no trained model

Both run on the reference phase and the reference masks, so **these results
survive any training failure**. That is deliberate.

### 8.1 `error_propagation.py` — the exchange rate · the label-free headline

Move the reference boundary by a known number of pixels; measure what happens.
No network, nothing trained. So the result is a property of the specimen and the
optics rather than of any method — a calibration curve other people can use.

| shift | µm | Dice | \|ΔA/A\| | \|Δm/m\| | ΔA/A | Δm/m | mass/area |
|---|---|---|---|---|---|---|---|
| −5 | −1.424 | 0.8531 | 0.2672 | 0.1935 | −0.2672 | −0.1935 | 0.724 |
| −2 | −0.570 | 0.9482 | 0.1168 | 0.0828 | −0.1152 | −0.0824 | 0.708 |
| −1 | −0.285 | 0.9745 | 0.0601 | 0.0446 | −0.0583 | −0.0374 | 0.742 |
| +1 | +0.285 | 0.9753 | 0.0625 | 0.0399 | +0.0625 | +0.0399 | 0.639 |
| +2 | +0.570 | 0.9514 | 0.1430 | 0.0809 | +0.1340 | +0.0809 | 0.566 |
| +5 | +1.424 | 0.8855 | 0.3102 | 0.1230 | +0.3102 | +0.1161 | 0.396 |

**One pixel of boundary error — 0.285 µm — costs about 6% of area and about 4%
of dry mass at Dice 0.975.**

**Dry mass is more robust to segmentation error than area is, by a factor of
1.48** (median ratio 0.674). The reason is physical, not statistical: the
boundary sits where the cell is thinnest, so the pixels a boundary error adds or
removes carry little phase. **A measurement-grade mass therefore does not
require a measurement-grade boundary**, and this curve says how much boundary
error each mass tolerance buys.

The dilation and erosion curves are reported separately and **never averaged**.
Outward, mass saturates (+0.040 → +0.116 while area runs +0.063 → +0.310) because
dilation adds near-empty background; inward, mass tracks area (ratio flat at
0.71–0.74) because erosion removes real cell interior. The ratio therefore falls
with outward bias and not with inward, and a single "sensitivity" number would
hide the sign of the bias — which is exactly what a Bland–Altman plot of the
final result is about.

**Figure 17**, `figures/fig17_error_propagation.png`.

### 8.2 `synthetic_validation.py` — the floor under every other number

Cells are spherical caps, `φ(d) = φ₀√(1 − (d/r)²)`, whose integrals are exact:
`A = πr²` and `V = φ₀(2/3)πr²`. So for every synthetic cell there is a number
the pipeline must reproduce, and the difference is pipeline error and nothing
else.

```
cells placed 96   measured 89   paired 89
area  signed  mean +0.0350%   median +0.0121%   p95 |.| 0.6885%
mass  signed  mean +0.0080%   median +0.0049%   p95 |.| 0.1404%
field-summed mass  mean -1.8225%
```

**Two floors, and they are different numbers for different claims.**

* **Per cell: 0.04%** in dry mass. That is discretisation, and it falls with
  cell size: 0.087% at 3–4 µm radius down to 0.012% at 7–8 µm. So a measured
  per-cell error of 4% is about 100× the floor — model error, not pipeline
  error.
* **Field total: −1.82%**, an order of magnitude larger, and **entirely the area
  filter**. `min_cell_area_um2: 30` rejects anything below 3.09 µm of radius, and
  7 of 96 synthetic cells fell below it. Quoting the per-cell floor where the
  field total applies would understate a population mean's bias by that whole
  factor.

The script also checks the loss terms against the analytic answer on a **curved**
profile, where an off-by-one in the domain would actually show:

```
exact prediction            0.000e+00   (must be 0)
phase scaled by 1.07        0.070000   (must be 0.070000)
area term, exact foreground 0.000e+00   (must be 0)
```

**Figure 18**, `figures/fig18_synthetic_floor.png`.

---

## Part 9 — Documentation corrected

`docs/v2_state_and_plan.md` and `docs/HoloQPI_Project_Documentation.md` both
described the shared pretrain as running on "the full 800 fields". It does not.
`main.py:127` builds only the train and val loaders for a training command
(`build_dataloaders(cfg, splits_to_build=("train", "val"))`), so the test split
was never reachable and **there is no leak** — but the wording implied one, and a
reader checking the claim against the code would have found them contradicting
each other. Both now say "the train split", with the reason.

---

## Part 10 — The experiment matrix and one script to run it

`config/v2/_shared.md` describes all thirteen arms. `run_v2.sh` runs them in
dependency order.

| arm | file | the question it answers alone |
|---|---|---|
| A | `a_baseline.yaml` | what conventional losses alone achieve |
| B | `b_cell_ipp.yaml` | does a **per-cell** mass constraint help |
| B′ | `b1_image_volume.yaml` | …or would an **image-level** one have done |
| B″ | `b2_cell_area.yaml` | does constraining area as well as mass help |
| C | `c_cell_ipp_bga.yaml` | does boundary-gradient alignment help on *this* morphology |
| D0 | `d0_amplitude.yaml` | can the amplitude head be supervised at all |
| D1 | `d_forward_amplitude.yaml` | what the forward-model term does (**diagnostic**) |
| W×4 | `w_ipp_*.yaml` | the weight sweep for B |
| KA, KB | `k_compact_*.yaml` | what the 65% parameter saving costs |

**B′ is the arm that was missing, and it is not optional.** If B beats A, the
tempting claim is "a measurement-aware loss helps" — but an image-level
phase-volume term is also measurement-aware, so A vs B cannot establish the
contribution of the words **per-cell**. B vs B′ can. Its weight must be set from
`check_gradient_path.py` on its own config so both arms present the same
gradient magnitude; until that is run the comparison is not yet matched, and the
config says so.

Each pair of arms differs by exactly one term, which is why `cell_integrated_phase`
was raised to 1.0 in C, D0 and D1 as well: `C − B`, `D0 − B` and `D1 − D0` are
each one term.

`run_v2.sh` stages:

```
0  self-test                nothing runs if the physics is wrong
1  prepare data             masks, splits, manifest, label audit
2  aberration surfaces      writes global + per_field + hybrid, with the numbers
3  amplitude reference      blocks D0 and D1
4  pre-flight diagnostics   SETS THE LOSS WEIGHT; decides if D1 is usable
5  label-free analyses      error propagation, synthetic ground truth
6  train the arms           plus optional seed replication
7  conventional baseline    the floor the network must beat
8  ONNX export + benchmark  the efficiency claim (this is the path that was broken)
9  figures
```

Stage 4 is not cosmetic: it measures the gradient ratio that sets B's weight and
whether the forward-model residual is correctly signed. Both were previously
answered by reasoning instead of measurement, wrongly in both cases.

---

## How to verify all of it

```bash
python scripts/selftest.py                                    # 82 checks
bash run_v2.sh --stage 0                                      # the same, via the runner
python scripts/estimate_aberration.py --config config/base.yaml
python scripts/prepare_amplitude.py  --config config/base.yaml
bash run_v2.sh --stage 5                                      # the label-free results
python scripts/make_figures.py --config config/base.yaml --only 17 18
```

All of the above ran clean on the 13 local fields. The parts that need the
server are stage 4's 30-batch gradient measurement, stage 6's training, and
stage 8's benchmark.

## What is still open

* **B′'s weight** is a placeholder until `check_gradient_path.py` is run on
  `b1_image_volume.yaml`. The comparison is not matched until then.
* **The per-field ramp** in the aberration surface is unexplained: it is not the
  demodulation remainder (§3.2), and separating reference-beam drift from an
  unwrapping artefact needs either raw unprocessed reconstructions or the
  acquisition software's parameters.
* **Independent labels.** Every mask is `Otsu(Gσ * φ_GT)`, so segmentation is
  measured against a threshold of its own input, and head redundancy Dice is
  0.93. S. Park is organising the membrane channel at the same field of view. He
  has said it is not reliable enough for automated segmentation, so the plan is
  manual labels on **25–40 fields for evaluation only** — enough to break the
  circularity in the reported metric without needing enough to train on.
* **No trained v2 arm exists yet.** Everything in `runs/` from before today is a
  smoke test (2 epochs, 192 px crops, `seed_aggregate.json` with `runs: 1` and
  `sd: NaN`) *and* used the wrong pixel pitch. Use `CLEAN=1`.
