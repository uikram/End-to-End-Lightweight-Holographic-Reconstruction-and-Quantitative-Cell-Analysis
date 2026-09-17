# HoloQPI — Technical Documentation

Mathematical formulation, parameter reference, design rationale and experimental
protocol for the end-to-end holographic analysis framework.

- [1. Objective and scope](#1-objective-and-scope)
- [2. Data](#2-data)
- [3. Physical model and calibration](#3-physical-model-and-calibration)
- [4. Segmentation labels](#4-segmentation-labels)
- [5. Network architecture](#5-network-architecture)
- [6. Joint physics-aware objective](#6-joint-physics-aware-objective)
- [7. Evaluation metrics](#7-evaluation-metrics)
- [8. Experimental protocol](#8-experimental-protocol)
- [9. Configuration reference](#9-configuration-reference)
- [10. Extending the framework](#10-extending-the-framework)
- [11. Known limitations](#11-known-limitations)

---

## 1. Objective and scope

A conventional quantitative phase imaging (QPI) pipeline is two stages: numerical
reconstruction of the phase from a hologram, then segmentation and measurement of
the reconstructed phase. Each stage is optimised for its own criterion, and
neither is optimised for the biophysical quantity the experiment actually wants.

This framework collapses both stages into one network and optimises them against
that quantity directly. The requirement, stated in the project brief, is that the
network must not merely produce a *visually plausible* phase image but a
**measurement-ready** one: a reconstruction whose integrated optical path length,
inside a boundary the same network predicts, reproduces the dry mass a
conventional pipeline would report.

The network's outputs are the **quantitative phase**, the **transmitted
amplitude** and the **cell segmentation**, and the per-cell projected area,
circularity, optical volume and dry mass derived from the first and the third.
The amplitude completes the complex field, which is what makes the
physics-aware forward-model term (§5a, §6.2a) expressible at all; it is optional
and, when off, the specimen is treated as purely refractive (A = 1).

The second contribution is the comparison of two acquisition geometries —
off-axis and in-line Gabor — through this single pipeline, under identical data,
splits, architecture, objective and schedule.

---

## 2. Data

### 2.1 Contents

800 matched triplets: one off-axis hologram, one in-line Gabor hologram and one
reconstructed phase map per field of view.

| Cell line | Control | Blebbistatin | FCCP | Staurosporine | Rotenone | Total |
|---|---|---|---|---|---|---|
| NCI | 50 | 50 | 50 | 50 | 50 | 250 |
| SNU | 100 | 50 | 50 | 50 | 50 | 300 |
| T24 | 50 | 50 | 50 | 50 | 50 | 250 |
| **Total** | 200 | 150 | 150 | 150 | 150 | **800** |

The four drugs are chosen for their morphological effect — blebbistatin inhibits
myosin II, FCCP uncouples mitochondria, staurosporine induces apoptosis, rotenone
inhibits complex I — so the condition label functions as a phenotype label. This
is the 5-way classification target, replacing the four red-blood-cell
morphologies of the previous study.

### 2.2 File formats

**Holograms** — 1024 × 1024, 8-bit, LZW-compressed TIFF. `imagecodecs` is
required to decode them.

**Phase** — `<stem>_phase.bin`, a 23-byte header followed by 900 × 900 float32
values in radians:

| Offset | Type | Meaning |
|---|---|---|
| 0 | uint16 | format version |
| 2 | uint32 | header length (23) |
| 6 | uint32 | width (900) |
| 10 | uint32 | height (900) |
| 14 | float32 | x pixel pitch, metres |
| 18 | float32 | y pixel pitch, metres |
| 22 | uint8 | channel/type flag |

The layout is declared in `formats.phase_binary`, so a change of acquisition
software is a configuration edit rather than a code change.

Observed values: pitch 2.84871 × 10⁻⁷ m and 2.11994 × 10⁻⁷ m, phase spanning
roughly −2 to +5 rad with a background referenced to zero.

### 2.3 Hologram-to-phase alignment

The hologram is 1024 px and the phase 900 px over the same field of view. The two
candidate mappings were tested by correlating a cell-texture map derived from the
hologram against the phase:

| mapping | correlation |
|---|---|
| centre crop 1024 → 900 | **0.48** |
| resize 1024 → 900 | 0.23 |

Sweeping the size of a centred window and rescaling it to 900 peaks at
S ≈ 896–904, which is what a pure crop predicts and a rescaling would not. The
framework therefore uses `data.align: center_crop`. `resize` is implemented for
datasets whose reconstruction does resample the field.

**This matters quantitatively, not just cosmetically.** Resampling a hologram
interpolates its fringes, and interpolated fringes no longer encode the original
optical path length; cropping preserves the native sampling exactly.

---

## 3. Physical model and calibration

For a segmented region Ω of the phase map φ:

```
projected area     A     = N_Ω · dx · dy                        [µm²]
circularity        C     = 4π A / P²                            [-]
optical volume     V_φ   = Σ_{i∈Ω} φ_i · dx · dy                [rad·µm²]
dry mass           m     = λ / (2π α) · V_φ                     [pg]
```

with N_Ω the pixel count, P the Crofton perimeter, λ the illumination wavelength,
α the specific refraction increment of the intracellular solids, and dx, dy the
object-plane pixel pitches.

Because **m is a fixed scalar multiple of V_φ**, a relative error in the enclosed
phase integral is exactly the relative error in dry mass. That identity is why
the objective constrains V_φ as a *relative* penalty (§6.3) rather than an
absolute one, and it is verified numerically by `scripts/selftest.py`.

Default constants (`config/base.yaml`, section `optics`):

| Parameter | Value | Note |
|---|---|---|
| `wavelength_um` | 0.666 | as in the previous study's system |
| `refraction_increment_ml_per_g` | 0.2 | confirmed with the acquisition group; range `[0.173, 0.215]` is recorded in the config |
| `pixel_pitch_x_um` | 0.284871 | decoded from the phase headers |
| `pixel_pitch_y_um` | 0.284871 | **isotropic**; see the correction below |

These give **0.0811515 µm² per pixel**, **0.529986 pg per rad·µm²** and
**0.0430091 pg per rad·pixel**.

> **Two constants were corrected on 2026-09-10, and this table is the corrected
> one.** Earlier versions of this document and of `config/base.yaml` carried
> `refraction_increment_ml_per_g: 0.185` and an **anisotropic** pitch of
> 0.284871 × 0.211994 µm. Both were wrong: the y pitch had been read from a
> header field that does not mean what it was taken to mean, and α was a
> literature value for a different cell type than the one the acquisition group
> confirmed.
>
> The magnitudes are large. The pitch correction raises every absolute area by
> 34.4% and every absolute mass with it; the α correction lowers absolute mass by
> a further factor of 0.185/0.2. **Any absolute area or picogram figure written
> before 2026-09-10 is not comparable with one written after.**
>
> **No relative result changes.** α and dx·dy are fixed scalar multipliers that
> appear identically in the predicted and the reference quantity, so they cancel
> from every MAPE, correlation, relative bias and limit of agreement in
> `runs/RESULTS.md`. `scripts/selftest.py` proves this rather than asserting it:
> it recomputes a MAPE at both ends of the α range and requires the two to agree
> to 1e-12. The reported study ran on the corrected values above.

`prepare` re-reads the pitch from the headers and warns if it disagrees with the
configuration beyond `header_pitch_tolerance_um`. Every area, volume and mass
scales with these numbers, so a silent mismatch would corrupt the whole
measurement chain.

> **Confirm λ and α with the acquisition group before publishing absolute
> picograms.** The wavelength is inherited from the previous instrument and α is
> a literature value for non-erythrocyte cells. All *relative* comparisons, and
> every agreement statistic, are invariant to both.

**Sanity check on the delivered data.** Over the 1 889 reference cells of the
test split (`runs/v2_baseline_off_axis/per_cell_test.csv`, computed with the
corrected constants above), the measurement chain gives a median projected area
of 411 µm², a median equivalent diameter of **22.9 µm** and a median dry mass of
**212 pg** (IQR 137–304) — all within the expected range for adherent cancer
lines. This is an independent confirmation that the header decoding, the pixel
pitch and the calibration constants are mutually consistent. (The figures quoted
here before 2026-09-10 — 18.2 µm and 149 pg — were computed with the superseded
constants.)

---

## 4. Segmentation labels

No manual annotations were delivered. Masks are derived from the ground-truth
phase, on the physical argument that a cell is a connected region whose optical
path length rises measurably above the surrounding medium:

1. Gaussian smoothing, σ = `smoothing_sigma_px`
2. threshold — Otsu by default, or a fixed level in radians
3. binary closing, then hole filling
4. area filtering to `[min_object_area_um2, max_object_area_um2]`, which drops
   speckle at the lower bound and debris or confluent sheets at the upper
5. optional watershed instance split for touching cells

Mean foreground coverage on the delivered data is 19%.

These are **silver-standard** labels. Segmentation scores against them measure
agreement with a physically motivated automatic procedure, not with a human
annotator, and any write-up must say so.

Setting `paths.manual_mask_dir` redirects the loader to a folder of manual
annotations; no other change is needed anywhere.

Instance labels are produced by the *same* function for prediction and reference,
so instance metrics compare like with like.

### 4.1 Two consequences that must be reported, not assumed

Deriving the mask from the phase has two effects that no training curve reveals.
`scripts/audit_labels.py` measures both; run it once and quote the numbers.

**The threshold is chosen per image.** Otsu reads each image's own histogram, so
the operational definition of "cell" moves between fields of view. On the
delivered data the level has a coefficient of variation of roughly 35–40%. That
is tolerable as long as it does not move *with drug condition* — if it did, cells
would appear morphologically different between conditions partly because the
label definition differed, and the classification result would be confounded. The
audit reports a one-way ANOVA and a Kruskal–Wallis test across conditions; a
p-value above 0.05 rules this route out. Should it ever fail, switch to
`mask_generation.threshold_method: fixed` with `fixed_threshold_rad` near the
global mean and re-run as a sensitivity check.

**The two supervised heads are not independent.** The mask is a deterministic
function of the phase, so a perfect reconstruction determines the mask exactly.
The audit tests this directly by applying the mask-generation function to the
*predicted* phase and comparing the result with what the segmentation head
produced. Dice above ~0.85 there means the head is largely redundant. This is the
mechanism behind the null result for the physics-consistency terms (§6.3): those
terms constrain Σ(M ⊙ φ), a quantity already determined by the two primary
losses, so they can add no gradient information. State this in the paper — it
converts an unexplained negative result into a predicted one.

---

## 5. Network architecture

`holoqpi/models/holonet.py`

```
hologram (B,1,H,W)
      │
      ├─ optional angular-spectrum front end ──► (B,3,H,W)
      │
      ▼
 shared encoder  (MobileNetV2 / MobileNetV3-L / ResNet18)
      │  five feature maps at strides 2,4,8,16,32
      ├──────────────► phase decoder (U-Net)        ──► phase head       ──► (B,1,H,W) radians
      │                                             └──► amplitude head   ──► (B,1,H,W) transmittance
      ├──────────────► segmentation decoder (U-Net) ──► segmentation head ──► (B,2,H,W) logits
      └──────────────► global average pool          ──► classifier        ──► (B,5) logits
```

**Why one encoder feeds both decoders.** This is the design claim, not an
economy: the same latent description of the fringe field must support both
reconstruction and delineation. That shared representation is what ties the
segmentation boundary to the optical signal the boundary will later be used to
integrate. `model.share_decoder: true` collapses the two decoders into one trunk
with two output heads, as an ablation.

**Single-channel adaptation.** The first convolution is rebuilt for the required
input channels and the pretrained RGB filters are averaged across the colour
axis, preserving the spatial-frequency response rather than discarding it.

**Phase head.** Deliberately unbounded and unnormalised — it emits radians
directly. Any squashing activation would destroy the quantity being measured.

**Amplitude head.** Built only when `model.amplitude.enabled` is set, and it
shares the phase trunk rather than owning a decoder, because amplitude and phase
are two components of one field. Its output is bounded to 1 ± `model.amplitude.
deviation` (0.3), which keeps the predicted transmittance physical without
squashing it towards a constant. With the head off the model emits no amplitude at
all and the forward model uses A = 1; the evaluator then reports no amplitude
metric, rather than scoring the unity field the dataloader substitutes and
recording a perfect agreement for an arm that never predicted one.

**Classifier is disabled throughout the v2 study**
(`model.classifier_enabled: false`). Its accuracy moved ±9.5 points between
seeds, which is wider than any effect the study is testing, and the drug label is
a property of the dish rather than of a cell. The head and its metrics remain in
the code.

**Classifier.** The drug label is a property of the dish, not of an individual
cell, so it is predicted once per field of view and attributed to every cell
segmented within it. It uses `LayerNorm`, not `BatchNorm1d`, because a trailing
batch of one image is normal here and `BatchNorm1d` cannot normalise a single
sample in training mode.

**Non-divisible field sizes.** 900 is not a multiple of 32. The input is
reflection-padded to the next multiple internally and the padding is removed from
the outputs, so the network never resamples the quantitative field.

### 5.1 Angular-spectrum front end (optional)

`model.frontend.kind: angular_spectrum` isolates one first-order sideband in the
Fourier plane, shifts it to the origin to remove the reference-beam tilt, and
appends the demodulated amplitude and phase as two extra input channels. It is
differentiable and can be frozen (`detach: true`) or trained through.

It is meaningful **only for off-axis data**: an in-line Gabor hologram has no
separated sideband. That asymmetry is itself one of the findings the modality
comparison is meant to expose. Default is `none`, so both arms stay symmetric
unless the asymmetry is being studied deliberately.

Set `model.in_channels: 3` when enabling it.

### 5.2 LoRA (optional)

Carried forward from the previous study, where restricting adaptation to a
low-rank subspace preserved rare classes that full fine-tuning destroyed. The
encoder is frozen apart from its low-rank residuals; normalisation layers, the
input stem and every decoder and head remain trainable.

Grouped convolutions are skipped: a depthwise kernel has one filter per channel,
so a shared low-rank factorisation across channels is undefined for it and the
merge step could not be expressed as a single dense update.

`merge_lora()` folds the residuals into the base weights exactly (verified to
0.0 output difference) before export and timing, so the deployed graph carries no
adaptation branches.

Note that the trainable fraction under LoRA is high here (≈78% for MobileNetV2)
because the two U-Net decoders dominate the parameter count, exactly as
MobileNet-UNet did in the previous study.

---

## 5a. Forward model and hologram formation

`holoqpi/physics/propagation.py` holds the scalar diffraction the rest of the
framework was missing. It is used in three places -- the forward-model loss, the
conventional baseline, and z calibration -- so all three share one implementation
and one set of conventions.

**Angular spectrum propagation.** A field is propagated over z by

```
U(z) = F^-1 { F{u_0} . H },
H    = exp( i 2 pi z / lambda sqrt(1 - (lambda fx)^2 - (lambda fy)^2) )
```

Evanescent components (the square-root argument negative) are *attenuated* by
`exp(-k |z| sqrt(-arg))`, not zeroed. The distinction matters: zeroing them makes
the kernel a low-pass filter even at z = 0, so propagating by zero would not
return the field it was given, and an in-line hologram of a pure phase object
would acquire a contrast it does not physically have. The self-test checks that
propagating by zero is the identity to 1e-5.

**Hologram formation differs by geometry, and the difference is the study.**

```
in-line Gabor   I = |P_z(o)|^2
off-axis        I = |R + P_z(o)|^2      R = tilted plane reference
```

For in-line the unscattered beam travels with the object beam, so the two are
superposed and the twin image is inseparable. For off-axis a separate tilted
reference places the object in a sideband that can be isolated. The carrier is
read from the measured hologram's own spectrum rather than assumed, since the
reference tilt is a property of the setup and not of the specimen.

**A consequence worth stating in the paper.** At z = 0 an in-line hologram of a
pure phase object is *flat*: `|A exp(i phi)|^2 = A^2` does not contain phi. The
off-axis carrier records phase at any distance. Measured sensitivity of the
synthesised hologram to a 1-radian change in phase contrast:

| z (um) | in-line Gabor | off-axis |
|---:|---:|---:|
| 0 | 0.000 | 0.267 |
| 10 | 0.013 | 0.267 |
| 50 | 0.073 | 0.270 |
| 150 | 0.271 | 0.300 |
| 400 | 0.444 | 0.347 |

In-line holography needs defocus to encode phase; off-axis does not. The
forward-model term is therefore informative for the off-axis arm at any distance
and for the Gabor arm only away from focus, and `scripts/calibrate_z.py` reports
this sensitivity so the asymmetry is known before training rather than after.

### 5a.1 The conjugate sideband, and why it must be resolved on a detrended phase

`holoqpi/physics/surface.py`. An off-axis hologram carries the object in two
first-order sidebands that are complex conjugates of equal magnitude. Which one
a spectral `argmax` returns is arbitrary, and it flips between fields of a single
acquisition. Taking the wrong one returns the conjugate field, whose phase is
**negated** — a reconstruction that looks entirely plausible, unwraps cleanly,
and is anti-correlated with the truth.

The sideband is chosen on a physical rather than a numerical ground: cells are
optically denser than their medium, so they add optical path, and a field of
sparse cells on a flat background is right-skewed in phase. The rule needs no
mask, no threshold and no reference, so it applies at inference time as well as
during calibration.

It does, however, require the smooth surface to be removed first, and this is
not a detail. The objective's curvature is a quadratic bowl spanning tens of
radians whose own skewness far exceeds the few radians the cells contribute.
Measured on this dataset, over all 13 fields available locally:

| test applied to | picks the correct sideband |
|---|---:|
| wrapped phase (the previous implementation) | 6 / 13 — chance |
| unwrapped phase, no detrending | 0 / 13 — systematically wrong |
| detrended, order 1 (plane only) | 0 / 13 |
| **detrended, order 2** | **13 / 13**, margin ≥ 1.2 in skewness units |
| detrended, order 5 | 13 / 13 |

`optics.conjugate.detrend_order` is therefore required to be at least 2, and
`phase_skewness` raises if it is not. Ground truth for the table is the
correlation between the detrended reconstruction and the detrended delivered
phase, which is ±0.87–0.95 — bimodal, with no field near zero.

**What the bug cost, before it was found.** `scripts/estimate_aberration.py`
folded the unresolved sign into the stored aberration surface on 11 of 13
fields. The difference it fitted was then not a surface at all but roughly
`-(2 phi_ref + Psi)`, which a fifth-order polynomial still fits to R² ≈ 0.99
because the cells are a small part of the variance — so the fit's own quality
metric could not see it. The forward-model term then added a surface of the
opposite handedness to the predicted phase, and its residual *fell* as the phase
was removed. That is precisely the anti-discriminative behaviour the off-axis arm
was showing, and it was an artefact of this sign, not a property of the geometry.

| off-axis forward residual at z = 33.77 µm | before | after |
|---|---:|---:|
| reference (true) phase — the floor | 0.871 | **0.498** |
| phase × 0.9 | 0.868 | 0.499 |
| phase × 0.5 | 0.848 | 0.526 |
| phase + 0.3 noise | 0.877 | 0.518 |
| mirrored phase | 0.847 | 0.627 |
| zero phase | 0.817 | 0.600 |
| verdict | anti-discriminative | correctly ordered |

Aberration-fit quality moved with it: R² minimum 0.81 → 0.991, median 0.982 →
0.996, and the median surface span tightened from 23.4 to 20.0 rad. The ordering
is unchanged across every padding setting tested (`feature_um` 0.5/1.0/2.0,
`pad_px` 0/64/derived), so it is not an artefact of the crop geometry.

**At full scale (800 fields).** 577 of 800 fields — 72% — needed the flip, median
|skewness| 2.41. Detrended agreement with the delivered phase: median +0.920.
Aberration R²: median 0.9975, p10 0.9951, min 0.9809. 780 of 800 fields pass both
gates; the 20 rejected are unwrapping failures with surfaces of 388–493 rad, and
every one of them scores R² = 1.000, which is the second independent
demonstration that R² cannot serve as the quality gate here.

**A consequence: z becomes identifiable from the in-line arm.** With the surfaces
corrected, the Gabor scan places its minimum at **+34.377 µm** with a per-image
IQR of **0.72 µm** across independent fields — an interior minimum, well depth
2.8. The acquiring group independently supplied **33.77 µm**. Two unrelated
routes agreeing to within one grid step (1.43 µm) is the strongest evidence in
the study that the forward operator is now physically correct, and it is worth
stating as such. Before the sign fix no distance was identifiable at all.

### 5a.2 Two traps in the z scan itself

**The scoring window must not depend on z.** `border_px` defaults to `pad`, and
`pad` is derived from z, so left alone every distance in a scan is scored on a
different set of pixels: at |z| = 66 µm the residual covers 29% of a 900 px field
and at 34 µm it covers 58%. Fewer, more central pixels are easier for four free
radiometric coefficients to fit, so the residual falls with |z| for a reason that
has nothing to do with focus, and the off-axis minimum duly ran to the edge of
the scanned range. `calibrate_z.py` now computes one border from the largest |z|
scanned and applies it at every distance. The Gabor minimum survives this — it is
an *interior* minimum found against the bias, which makes it stronger, not weaker.

**A small margin is not a wrong sign.** The verdict function previously fell
through to `anti_discriminative` whenever a margin failed to clear the tolerance,
and so printed "a degraded phase scores BETTER than the truth" for an off-axis
run in which every degraded phase in fact scored worse (+0.0001 to +0.0333). The
taxonomy is now four-way — `usable`, `marginal`, `uninformative`,
`anti_discriminative` — and the last requires a margin below *minus* the
tolerance. A correctly signed term with tiny margins is `uninformative`, which is
a different finding with a different remedy.

**And the verdict is a sign test, not a mean.** The residual varies more between
fields than between a true phase and a mildly degraded one, so an average over
four fields is decided by which four were read — which is exactly what made two
earlier measurements at the same distance disagree. The test now records a
per-field margin and reports in what fraction of fields each degraded variant
scores worse than the truth; chance is 50%, and
`loss.forward_model.discrimination_win_rate` sets the bar.

### 5a.3 The unmeasurable piston phase, and why off-axis needs four cross-terms

Propagation multiplies the field by `exp(i 2 pi z / lambda)`. That is a global
phase across the whole field. It is not measurable — a camera records intensity —
and pinning it down would require knowing z to a small fraction of a wavelength,
which no experiment provides.

An in-line residual never sees it: `|U|^2` cancels any global phase exactly. An
off-axis residual sees it directly, because the object interferes with a
reference and the fringe positions depend on it. With only the real parts of the
two conjugate cross-terms in the radiometric fit, the residual is therefore a
function of where z happens to fall modulo half a wavelength. Measured on this
dataset, sampling z in steps of lambda/8:

| z (µm) | offset (z/λ) | residual, real parts only | with quadratures |
|---:|---:|---:|---:|
| 33.7700 | 0.000 | 0.5327 | 0.1310 |
| 33.8533 | 0.125 | **0.9910** | 0.1309 |
| 34.0198 | 0.375 | **0.1335** | 0.1309 |
| 34.1030 | 0.500 | 0.5322 | 0.1309 |
| 34.4360 | 1.000 | 0.5318 | 0.1308 |

The residual swung across nearly its whole range over a 0.33 µm change in z, so
any single reported value was a lottery on the piston. Carrying the imaginary
parts of both cross-terms fixes it, because

```
Re(R* U e^{i phi}) = cos(phi) Re(R* U) - sin(phi) Im(R* U)
```

so the four components span every global phase and the least-squares fit removes
the piston instead of being defeated by it. The residual then varies by 2e-4 over
a full cycle, and `scripts/selftest.py` asserts that invariance. The in-line arm
is unchanged, which is the control: its curve was already smooth.

**What it changed.** The off-axis floor at z = 33.77 µm fell from 0.7477 to
0.1815 — the operator explains 82% of the recorded hologram rather than 25%. The
off-axis scan's own minimum moved to **+33.42 µm**, within one grid step of the
supplied 33.77 µm, so the off-axis arm now corroborates the distance too, having
previously placed it at +37.2 µm or at the edge of the range. The calibration
probe still minimises at ×0.97, but its per-field IQR tightened from 0.50 to
0.03. Every degraded phase except a 10% rescaling now scores worse than the truth
in **100%** of fields.

Figure 16's left panel before the fix shows the aliasing directly: a sawtooth of
period ~λ/2 undersampled at a 1.4 µm grid step, against a smooth in-line curve.

### 5a.4 The verdict, and the calibration probe that sharpens it

Measured on 32 val fields at z = 33.77 µm, with a fixed 448 px scoring window.
The margin is the mean rise in residual over the true phase; the last column is
the fraction of *individual* fields in which the degraded phase scores worse
(chance is 50%).

| phase variant | off-axis margin | fields worse | Gabor margin | fields worse |
|---|---:|---:|---:|---:|
| × 0.9 | +0.0003 | 53% | −0.0020 | 19% |
| × 0.5 | +0.0157 | 72% | +0.0067 | 75% |
| + 0.3 noise | +0.0109 | **100%** | +0.0711 | **100%** |
| zero | +0.0468 | 62% | +0.1334 | **100%** |
| mirrored | +0.0942 | 84% | +0.1328 | **100%** |
| **verdict** | **uninformative** | | **marginal** | |

Both arms are correctly signed against gross corruption — nothing scores better
than the truth by more than the tolerance — and both are blind near it. Neither
is trained. `loss.weights.forward_model` stays at 0.0, and this table is the
result, not a gap in it.

**The calibration probe is the sharper statement.** Multiplying the reference
phase by *s* and minimising the residual over *s* asks whether the operator
agrees with the delivered phase about its **magnitude**, which is the operational
content of "measurement-ready":

| phase × s | 0.5 | 0.7 | 0.9 | 1.0 | 1.1 | 1.4 |
|---|---:|---:|---:|---:|---:|---:|
| off-axis residual | 0.4264 | 0.4018 | 0.3880 | 0.3851 | **0.3846** | 0.3937 |
| in-line residual | 0.9459 | **0.9450** | 0.9462 | 0.9475 | 0.9491 | 0.9552 |

**The off-axis forward model recovers the correct phase magnitude to within
5–10%** — a genuine parabolic well centred near s = 1. **The in-line model
minimises near s ≈ 0.7**, roughly 30% low, which is what a single-term forward
model does when the twin image is superposed on the object and it accounts for
only part of the measured modulation. That asymmetry is physical, not incidental,
and it is a direct quantitative answer on the phase-reconstruction-accuracy axis
of the modality comparison. Figure 16 draws all three diagnostics.

The same resolution is applied in `scripts/conventional_baseline.py`, which
previously tested the *wrapped* phase, and in the optional angular-spectrum front
end (`model.frontend.resolve_conjugate`), which used a bare `argmax` and would
otherwise hand the encoder a negated phase channel on an unpredictable subset of
the training set — noise a network cannot learn around, because nothing in its
input says which sign it received.

**Crop size is a physical constraint.** Light scattered inside a training crop
travels `lambda z / feature` micrometres sideways before reaching the sensor, and
light from outside travels the same distance inward. Neither is available on a
crop smaller than that spread. The field is reflection-padded by the required
radius and the same margin is dropped from the residual; when the crop cannot
support the full radius the shortfall is logged once, so the approximation is
known rather than silent.

---

## 6. Joint physics-aware objective

`holoqpi/losses/composite.py`

```
L = w_phase · L_phase
  + w_seg   · L_seg
  + w_cls   · L_cls
  + w_pmc   · L_PMC + w_bga · L_BGA + w_pv · L_PV
  + w_mass  · L_mass + w_area · L_area
```

The first three terms supervise each head against its own target. The remaining
five are the extension of the previous study's physics-aware loss to the
end-to-end setting: **they act on the predicted phase and the predicted mask
together**, so the network cannot satisfy them by getting either output right in
isolation. That coupling is what makes the reconstruction measurement-ready.

Setting any weight to zero removes its term, which is how the component-wise
ablation is run.

### 6.1 Phase reconstruction

```
L_phase = w_l1 · ‖φ̂ − φ‖₁ + w_grad · ‖∇φ̂ − ∇φ‖₁ + w_ssim · (1 − SSIM(φ̂, φ))
```

The gradient term protects the membrane transitions where the phase falls
steeply, which are precisely the locations that determine where a boundary can be
placed. SSIM is implemented locally with a Gaussian window; no extra dependency.

### 6.2 Segmentation

Class-weighted Dice plus cross-entropy, with Laplace smoothing on the Dice
denominator for numerical stability under mixed precision.

### 6.2a Forward-model consistency -- the term the reference literature means

```
L_forward = || standardise(I_synthesised) - standardise(I_measured) ||
I_synthesised = form_hologram( A_hat exp(i phi_hat), z, geometry )
```

**This is the only term that reads the raw hologram**, and therefore the only one
that introduces information the supervised losses have not already consumed. The
raw hologram enters the network as input and, without this term, never appears in
the objective again -- so nothing in training checks that the predicted field is
consistent with the measurement it came from.

It is what all four reference papers mean by physics consistency:

* Huang, Chen, Liu & Ozcan, *Nature Machine Intelligence* 2023 (GedankenNet) --
  trained with *no* labelled data at all, using only the residual between the
  input hologram and the hologram predicted by forward-propagating the network's
  output complex field.
* Galande et al., *J. Biomed. Opt.* (HDPhysNet) -- "the data fidelity term
  promotes data consistency using the hologram formation model."
* Lee, Mammadova, Barg & Jang, *APL Mach. Learn.* 4, 026106 (2026) and
  arXiv:2507.00482 -- object-to-sensor distance treated as an implicit style,
  inverse mapping learned from intensity measurements only.

Three implementation details carry weight.

**The radiometric model is fitted, not assumed.** A sensor does not record
`|U|^2`; it records `gain * (physical intensity) + offset + noise`, with gain set
by illumination power, exposure and camera response, and offset by the black
level. None of these are known and none carry phase information. They are removed
by *fitting* them, per image, in closed form, as part of the observation model.
Writing the off-axis intensity in its physical components,

```
I = |R|^2 + |U|^2 + 2 Re(R* U)
  = c0 . 1 + c1 . |U|^2 + c2 . Re(R* U) + c3 . Re(R U)
```

the unknown reference power, object gain and reference-to-object amplitude ratio
are exactly the coefficients c0..c3, recovered by a 4 x 1 least-squares solve.
In-line has no separate reference, so the model is two-parameter,
`I = c0 + c1 |U|^2`.

Three things this buys, none of which a z-score of both sides provides:

* It absorbs the off-axis **reference ratio**, which a normalisation cannot.
* It absorbs the **conjugate-sideband ambiguity**. The two first-order sidebands
  have equal magnitude, so any rule for picking one is a coin flip that flips the
  sign of the fringe term. Both cross-terms enter as separate fitted components
  and the fit weights them, so the ambiguity disappears rather than being
  resolved by guess.
* The coefficients are **logged** (`forward_offset`, `forward_object_gain`,
  `forward_fringe_gain`). A gain that drifts or changes sign is a modelling error
  announcing itself, which a hidden normalisation would conceal.

**The residual is computed against the RAW measurement.** The dataset carries two
representations of every hologram: `hologram`, z-scored for the network, and
`hologram_raw`, the intensity the sensor recorded. The network needs a
well-conditioned input; the physics term needs the observation. Using one array
for both would silently redefine the observation model this term exists to test,
so the loss reads `hologram_raw` and warns loudly if only the normalised version
reaches it.

**The carrier is estimated to sub-bin precision.** A half-bin error is a phase
ramp of pi across the field, which no per-image gain can absorb. The spectral
peak is refined by parabolic interpolation and then polished by maximising the
demodulated DC magnitude. The DC exclusion radius is a *fraction* of the field,
not a pixel count: an absolute radius means something different on a 900 px
evaluation field and a 512 px training crop, and on a small enough crop it masks
out the carrier itself.

**Border exclusion.** See "crop size is a physical constraint" in section 5a.

**Amplitude.** A hologram is formed by a complex field, so synthesising one from
phase alone assumes a purely refractive specimen. `model.amplitude.enabled` adds
a head for transmitted amplitude, bounded to `1 +/- deviation` and
zero-initialised so it starts from the pure-phase assumption. It has no ground
truth and is trained *only* by this residual, exactly as in the self-supervised
hologram-reconstruction literature.

**z is required and is not in the data.** The phase `.bin` header carries width,
height and the two pixel pitches and nothing else. Three routes, in order of
preference:

1. **Ask the acquiring group.** One email settles it outright.
2. **Recover it** with `scripts/calibrate_z.py`, which scans the *same* residual
   the loss minimises and decides identifiability by whether individual images
   independently agree on the minimum — not by whether two estimators of unequal
   quality agree with each other.
3. **Learn it.** `loss.forward_model.learn_distance: true` makes z an
   `nn.Parameter` initialised at `distance_um`. The angular-spectrum kernel is
   differentiable in z, so this is a real refinement: the self-test recovers
   z = 200 um from an initialisation of 140 um in eighty steps. Report the
   converged value in the paper.

Training with a wrong fixed z is worse than not training the term: the residual
then measures the error in z rather than the error in the reconstruction.

**Only the in-line arm constrains z, and that is physics rather than a defect.**
An off-axis hologram encodes phase in its carrier fringes at any distance, so its
residual is nearly flat in z; an in-line hologram encodes phase only through
defocus, so its residual has a real minimum. The same specimen was recorded at
the same distance in both, so **the Gabor arm determines z and the value applies
to both**. `calibrate_z.py` says this explicitly when it happens.

**Verification.** Four checks run in `scripts/selftest.py` section [6] and must
pass before any result from this term is quoted:

| Check | Result |
|---|---|
| residual vanishes on the true field | 7.7e-11 (in-line), 8.2e-3 (off-axis, limited by carrier estimation) |
| residual is invariant to camera gain and black level | identical at gain 1 and gain 37 with offset 120 |
| residual grows monotonically as the phase degrades | 0.000 → 0.028 → 1.000 (in-line) |
| gradient is finite and non-zero | yes, both geometries |
| z recoverable by gradient descent | 140 um → 200.9 um (true 200) |

### 6.3 Physics coupling terms

Let `f = 1 − softmax(seg)[background]` be the predicted foreground probability.

**Phase-Mask Contrast** — the segmented interior must carry more optical path
than its surround:

```
L_PMC = max(0, µ_bg(φ̂) − µ_cell(φ̂) + margin)
```

**Boundary-Gradient Alignment** — drives the mask edge onto the ridge of
steepest optical path change:

```
L_BGA = ‖ ∇f / max(∇f) − ∇φ_ref / max(∇φ_ref) ‖₁
```

Both fields are max-normalised per sample first: a probability map and a phase
map in radians are otherwise on incomparable scales. `bga_reference` selects
whether the anchor is the predicted phase (cross-head self-consistency, the
default) or the ground truth.

**Phase-Volume Preservation** — conserves the phase integral enclosed by the
predicted boundary:

```
L_PV = | Σ(f ⊙ φ̂) − Σ(M ⊙ φ) | / ( |Σ(M ⊙ φ)| + ε )
```

Relative rather than absolute, both because dry mass inherits exactly this
relative error and because an absolute integral over a full field reaches
magnitudes that destabilise FP16 training.

**Dry-Mass Consistency** — a smooth-L1 penalty on the ratio of predicted to
reference enclosed integral, closing the loop from raw hologram to picograms. The
calibration constant cancels in this relative form. It is kept separate from
L_PV because it is evaluated against the reference mask *and* phase jointly,
constraining the quantity the study reports rather than either head alone.

**Projected-Area Consistency** — keeps the segmented footprint calibrated:
`|Σf − ΣM| / (ΣM + ε)`.

All five return per-sample values and are averaged only over fields of view that
contain cells, so an empty patch contributes nothing rather than a spurious zero.

### 6.4 Default weights

| Term | Weight |
|---|---|
| phase | 1.0 |
| segmentation | 1.0 |
| classification | 0.2 |
| phase_mask_contrast | 0.1 |
| boundary_gradient_alignment | 0.05 |
| phase_volume | 0.1 |
| dry_mass_consistency | 0.1 |
| projected_area_consistency | 0.05 |

The three carried-over physics weights match the previous study. The two new
measurement terms start at the same order of magnitude; the ablation grid in §8.2
is how they get tuned.

---

## 7. Evaluation metrics

| Family | Metrics |
|---|---|
| **Phase** | MAE and RMSE in radians, bias, PSNR, SSIM, Pearson r, **plus MAE and bias restricted to the inside of reference cells and to background separately** |
| **Amplitude** | MAE, RMSE, bias and Pearson r against the reconstruction reference; the same MAE inside reference cells; **the same MAE for the thin-phase assumption A = 1 and the ratio between them**; the mean and standard deviation of the prediction itself. Produced only for an arm whose amplitude head is on (§7.2) |
| **Segmentation** | Dice, IoU, Aggregated Jaccard Index, Boundary F1 (2 px tolerance) |
| **Detection** | recall, precision and F1 of cell instances at `match_iou_threshold`; counts of reference, detected, matched, missed and false-positive cells |
| **Classification** | accuracy, macro F1, balanced accuracy, per-class F1, confusion matrix. Not produced while the condition head is disabled |
| **Measurement** | per-cell MAPE for area, circularity, optical volume and dry mass; per-cell and per-image Pearson r; Bland-Altman relative bias and limits of agreement; field-total bias; bootstrap confidence intervals over fields; **coverage-adjusted MAPE** |
| **Forward model** | the hologram data-fidelity residual, the same residual from the reference phase as its floor, their ratio, and the propagation distance it was scored at. Reported for **every** arm, including those that never optimised it (§6.2a) |
| **Efficiency** | parameters, GMACs, latency mean/p50/p99, **p99/p50 and a stability flag**, FPS, weights and peak activation memory, under PyTorch and ONNX Runtime |

Phase errors are reported in **radians**, not as normalised image-quality scores,
because the downstream measurement inherits them in physical units.

**Whole-field phase error understates what matters.** Roughly four fifths of every
image is background, which is easy to reconstruct, so a field-wide MAE is
dominated by the part of the image dry mass never integrates. On the delivered
data the in-cell MAE runs several times the field-wide figure. Both are reported;
`phase_mae_rad_in_cell` and `phase_bias_rad_in_cell` are the ones a
measurement-readiness claim rests on.

**Detection is reported separately from measurement.** Per-cell errors are
computed over IoU-matched pairs only, so they are silent about cells that were
never detected — and missed cells, not mismeasured ones, dominate whole-field
mass error on this data. `detection_recall`, `detection_precision` and
`detection_f1` state that directly. A cell-count ratio cannot: a model that misses
half the cells and invents an equal number of false positives scores a ratio of
1.0. Unmatched cells are written to `unmatched_<split>.csv` with their areas and
masses, so the size distribution of what was missed can be inspected.

**Bland–Altman bias and limits share one centre.** The bias reported alongside the
limits is the mean relative difference and the limits are mean ± 1.96 SD, which is
the standard construction; a median bias is reported separately as a robust
cross-check rather than substituted for the mean. Mixing a median centre with
mean-based limits would produce an interval not centred on its own bias.

**Instance separation is not a secondary metric here.** Every reported quantity
is per cell, so a prediction that covers the right pixels but merges two touching
cells yields two wrong measurements despite an excellent Dice score. AJI is
reported alongside Dice for that reason.

Predicted and reference cells are paired by greedy best IoU above
`match_iou_threshold`; only matched pairs enter the per-cell error statistics.

### 7.2 The amplitude metric, and what it can and cannot claim

`holoqpi/metrics/amplitude.py`

The framework's stated output is phase **and amplitude** and segmentation, but
the predicted amplitude is easy to leave unmeasured: it enters the forward-model
residual, where it is entangled with the phase, the propagation distance and the
aberration surface, so an amplitude head that had collapsed to a constant would
leave no trace in any table. This metric family exists to close that gap.

**The comparator is A = 1, not zero.** The honest baseline for a transmittance
prediction is the thin-phase-object assumption the rest of the study runs on, so
`amplitude_unity_mae` is the same error computed for A = 1 everywhere and
`amplitude_mae_over_unity` is the ratio. Below 1.0 the head is closer to the
reference than that assumption; at or above 1.0 no amplitude claim survives. A
collapsed head scores **exactly 1.000** by construction, with
`amplitude_pred_sd` at zero — which is the point, and is pinned by six checks in
`scripts/selftest.py`.

**What it is agreement with.** The reference is the modulus of a classical
off-axis reconstruction written by `scripts/prepare_amplitude.py`, normalised so
the mask background reads 1. It is not a measured transmittance: it carries the
sideband filter's lost high frequencies, residual twin-image structure and any
illumination vignetting. Every number in this family is therefore agreement with
one reconstruction algorithm's output, and the caveat must travel with it into
any write-up. The in-cell split is reported for the same reason it is for the
phase — the specimen is about a fifth of the field, so a head that predicts the
background modulus perfectly and the cells not at all still scores well
field-wide.

### 7.1 Checkpoint selection

`training.checkpoint_metric: composite` combines the three axes so a model cannot
be selected for excelling at one head while failing the quantity the study
reports:

```
composite = 0.35 · Dice
          + 0.25 · max(Pearson_phase, 0)
          + 0.25 · clip(1 − MAPE_drymass, 0, 1)
          + 0.15 · detection_F1
```

The detection term is not cosmetic. Because the measurement terms are computed
over matched cells only, a model that detects a handful of large, easy cells and
misses the rest scores an excellent dry-mass MAPE; without a detection term the
selection rule actively prefers it. The weights are configurable under
`training.composite_metric`, and any single metric can be used instead.

---

## 8. Experimental protocol

### 8.1 The modality comparison

`python main.py compare --config config/base.yaml`

Both arms share the split file, architecture, objective, schedule and seed. The
input modality is the only free variable, so any difference in the table is
attributable to the hologram type.

Splits are stratified over (cell line × drug condition) at 70/15/15 and written
once to `data/splits.json`, so every experiment sees the same images. Strata too
small to divide three ways are assigned whole to whichever split is furthest
below its global target, which prevents a split ending up empty.

The comparison writes `runs/<experiment>_modality_comparison.csv` with one row
per metric and a difference column, covering every axis the brief lists: phase
accuracy, segmentation, classification, area, optical volume, dry mass, and
computational efficiency.

### 8.2 Suggested ablations

Each is a configuration edit; no code change.

| Question | Setting |
|---|---|
| Do the physics terms help? | zero `loss.weights.phase_mask_contrast`, `..._alignment`, `phase_volume` |
| Do the measurement terms help? | zero `dry_mass_consistency`, `projected_area_consistency` |
| Hierarchical or additive? | add the terms one at a time, as in the previous study |
| Is the shared encoder doing work? | `model.share_decoder: true` versus `false` |
| Does the physics front end help off-axis? | `model.frontend.kind: angular_spectrum`, `model.in_channels: 3` |
| Does low-rank adaptation regularise? | `model.lora.enabled: true`, sweep `rank` over {2,4,8,16,32} |
| Which backbone? | `model.encoder` over the three registered encoders |

### 8.3 Statistical treatment

Cells within one field of view share an acquisition and are not independent. Take
the **image as the unit of analysis** for any longitudinal or between-condition
claim, summarising each image by the median of its cells, and report per-cell
distributions as descriptive only. The per-image Pearson r and Bland-Altman
statistics in the measurement family are already computed on that basis.

---

## 9. Configuration reference

`config/base.yaml` holds every value. `config/off_axis.yaml` and
`config/gabor.yaml` inherit from it through `extends:` and override only the
modality.

| Section | Governs |
|---|---|
| `project` | seed, determinism |
| `paths` | data root, per-modality directories, mask override, output root |
| `formats` | phase header layout, file suffixes |
| `optics` | λ, α, pixel pitches, header cross-check |
| `data` | modality, sizes, alignment, normalisation, batching, splits, augmentation |
| `labels` | cell lines, drug conditions, filename aliases |
| `mask_generation` | smoothing, threshold, morphology, area filter, watershed |
| `model` | encoder, decoder widths, heads, front end, LoRA |
| `loss` | all eight weights and each term's internal parameters |
| `training` | epochs, optimiser, schedule, AMP, checkpoint selection |
| `evaluation` | tolerances, instance method, measurement bounds, matching |
| `deploy` | ONNX opset and export options, benchmark protocol |

Inline overrides use dotted paths:

```bash
python main.py train --config config/gabor.yaml \
  --set training.epochs=100 loss.weights.dry_mass_consistency=0.2
```

### 9.1 Augmentation policy

Only isometries of the sampling grid are permitted: horizontal and vertical
flips, and quarter-turns, applied identically to hologram, phase and mask.

Brightness, contrast, noise and blur augmentation are **deliberately absent**. A
hologram encodes optical path length in the position and contrast of its fringes,
so intensity-domain augmentation would change the very quantity the network is
being asked to measure.

---

## 10. Extending the framework

| Task | Where |
|---|---|
| New encoder | add to `_ENCODERS` in `holoqpi/models/encoders.py`; must return five feature maps and expose `out_channels` |
| New loss term | add to `holoqpi/losses/terms.py`, wire into `composite.py`, add its weight to the YAML |
| New metric | add to the relevant module in `holoqpi/metrics/`; the evaluator merges every family's `compute()` |
| Third modality | add a directory to `paths.hologram_dirs` and a suffix to `formats.hologram.suffixes`; pass it to `compare --modalities` |
| Manual annotations | set `paths.manual_mask_dir` |
| Different acquisition software | edit `formats.phase_binary` |

`scripts/selftest.py` should be re-run after any change to the measurement chain
or the loss. It verifies dry mass against the analytic value on a synthetic cell,
confirms that an identical prediction scores perfectly and an eroded one degrades
in the expected direction, and checks that every loss term is finite,
differentiable and correctly signed.

---

## 10a. Figures

`scripts/make_figures.py` builds the figure set. Every figure answers a question
the paper has to answer; none restates a table. Most are built from files earlier
steps already wrote, so they regenerate in seconds; figures 1 and 9 need a forward
pass and are skipped automatically when no checkpoint exists.

**Twenty figures are registered and eighteen build on the current study.**
Figure 7 needs the v1 ablation result set and figure 11 needs the condition head,
which is disabled — both skip with their reason printed rather than failing.

```
python scripts/make_figures.py --config config/base.yaml          # all
python scripts/make_figures.py --config config/base.yaml --only 2 4 5
python scripts/make_figures.py --config config/base.yaml --list   # what each one is
```

Output is `figures/fig<NN>_<name>.png` (for reading) and `.pdf` (for the paper).
A figure whose inputs are missing prints its reason and is skipped; the rest still
build.

`figures/_manifest.json` records, for each figure, the config, experiment name,
modalities and split that produced it. The filenames carry none of that, and one
figure built from two different configs overwrites itself, so the manifest is the
only record of which is which. It is **merged** across invocations — stage 9
calls this script three times, from `config/base.yaml`, from one arm's config and
from the modality pair — and a figure this run skipped or failed has its entry
**removed**, because the file on disk is then from an earlier run and the
manifest must not vouch for it.

| # | Figure | Tier | What it shows and why it earns its place | Reads |
|---|---|---|---|---|
| 1 | Qualitative panel | **Essential** | The two hologram types side by side on the same field, with recovered phase, signed error and both segmentation boundaries. No table conveys what distinguishes an off-axis carrier from a Gabor in-line record with its superposed twin image; this is the physical premise of the whole study. | checkpoint |
| 2 | Bland–Altman | **Essential** | Agreement, not correlation, for dry mass and projected area per cell. Two methods can correlate at r = 0.99 and still disagree by 30% on every cell. The limits of agreement are the number a biologist needs to decide whether the method resolves the mass differences their experiment is about — which is exactly the "measurement-ready" claim. | `per_cell_<split>.csv` |
| 3 | Error decomposition | **Essential** | Mass ratio factorised into a domain (boundary) term and a phase (reconstruction) term, plotted against each other with the iso-mass diagonal. Turns a mass error into a cause and settles the only actionable question it raises: which head to work on. | `bias_diagnosis_<split>.csv` |
| 4 | Detection gap | **Essential** | The cascade from reference cells to detections to matches, the size distribution of missed versus matched cells, and the IoU distribution of the matches made. Every measurement statistic is conditioned on matching, so this is the figure that says what the measurements are silent about. | `per_cell_`, `unmatched_`, `metrics_` |
| 5 | Modality comparison | **Essential** | All the axes the brief names, on one page, every bar oriented so longer is better, with a signed-advantage panel. This is the central contribution; it should be legible without reconciling a table in which some metrics improve upward and others downward. | `*_modality_comparison.json` |
| 6 | Loss composition | **Essential** | The share each term holds in the weighted objective over training, and the physics terms alone on a symlog axis. This is the *evidence* for the negative result: the physics group collapses to a few percent and the phase-mask hinge reaches exactly zero early. It makes the null ablation a prediction rather than a surprise. | `history.json` |
| 7 | Ablation deltas | **Essential** | Signed percentage change of each ablation against the full objective, with a zero line and a ±2% noise band. An ablation table invites the reader to hunt for the largest number; this shows at a glance that most deltas straddle zero. | all `*_modality_comparison.json` |
| 8 | Efficiency trade-off | **Essential** | Accuracy against measured latency, median and tail latency per configuration, and compute cost against latency. Edge suitability is a trade-off, not a score. Rows flagged as not comparable across devices are drawn hollow so a CPU ONNX timing cannot be misread against a GPU PyTorch timing. | `*_hardware_benchmark_*.csv` |
| 9 | Phase error structure | Optional | Reconstruction error binned by reference phase amplitude, and the radially averaged error spectrum. Dry mass is an integral: insensitive to zero-mean high-frequency error, very sensitive to a slow offset inside cells. A single MAE cannot tell those apart; this can. | checkpoint |
| 10 | Label audit | Important | Per-image Otsu threshold by condition, and the head-redundancy bars from §4.1. The two properties of the silver-standard labels a reviewer will probe, answered pre-emptively. | `label_audit_*` |
| 11 | Confusion matrices | Important | Row-normalised condition confusion for both arms. An accuracy number cannot explain why the arm that reconstructs phase better classifies worse; whether the gap is one collapsed class or diffuse confusion decides whether the effect is biological or an acquisition artefact. | `confusion_<split>.json` |
| 12 | Convergence | Important | Validation trajectories with each metric's own best epoch marked. The heads converge at different times — segmentation and phase plateau while condition accuracy is still climbing — so this is the evidence that the reported numbers were read at a defensible point. | `history.json` |
| 13 | Forward-model consistency | **Essential** | The hologram data-fidelity residual per arm against its own reference-phase floor, for both geometries. Reading the level alone is meaningless — what the residual can reach is set by acquisition-chain mismatch, not by the network — so this plots the ratio and makes the anti-discriminative result (§6.2a) visible rather than asserted. | `*_modality_comparison.json` |
| 14 | Learned vs classical | **Essential** | The network and the textbook reconstruction on the same holograms, scored by the same evaluator, for both geometries. This is the study's primary claim and its clearest single panel: the in-line arm is where the classical route fails outright. | `conventional_baseline_<split>.json` |
| 15 | Recall by cell size | Important | Detection recall binned by reference cell area, with the size distribution of missed cells. Detection recall is the binding constraint on every measurement number, and this says *which* cells are lost — the answer is the small ones, near the area filter. | `unmatched_<split>.csv` |
| 16 | Forward-model diagnostics | **Essential** | The residual's response to a scaled, noised, mirrored and zeroed phase, and its z scan for both geometries. This is the measurement behind the claim that the term is a diagnostic and not a loss: a 10% phase error *lowers* it. | `amplitude_sensitivity_*`, `z_calibration.json` |
| 17 | Error propagation | **Essential** | Area and dry-mass error against a boundary moved a known number of pixels, and the ratio between them. No model is involved. It converts a segmentation error into a measurement error and shows why integrated quantities are intrinsically more robust than areal ones. | `error_propagation_summary.csv` |
| 18 | Synthetic floor | **Essential** | The measurement chain against exact analytic ground truth. Every other number in the study is read against this floor: it separates pipeline error from model error, and it is what licenses the statement that the reported error is reconstruction and segmentation rather than calibration arithmetic. | `synthetic_validation_*.csv` |
| 19 | v2 ablation | **Essential** | The v2 arms on the primary metric with per-bar between-seed bands and the significance rule applied. The bands are the figure's whole purpose: without them a reader ranks bars that are inside seed noise. | `runs/*/metrics_<split>.json` |
| 20 | Membrane labels | **Essential** | The phase-derived silver labels against the independently acquired membrane labels on the same fields. This is the only panel in the set that addresses the circularity caveat (§4, §11) with data rather than with a disclaimer. | `membrane_registration.csv`, checkpoint |

Colour is consistent across the set: off-axis blue, Gabor red. Modalities appear
in the order given by `--modalities`, so panels line up between figures.

---

## 10b. Conventional reconstruction baseline

`scripts/conventional_baseline.py` runs the textbook pipeline for each geometry
and scores it through the *same* evaluator, instance labelling and measurement
chain as the network, so the difference between them is attributable to
reconstruction and nothing else.

```
off-axis   isolate a first-order sideband -> shift the carrier to the origin
           -> back-propagate -> argument
in-line    back-propagate the recorded intensity -> argument
           (optionally N Gerchberg-Saxton iterations first)
```

Three corrections are applied, and each is necessary for the baseline to be a
fair opponent rather than a strawman:

1. **Conjugate selection.** The two first-order sidebands are conjugates and
   taking the wrong one returns the *negated* phase -- which looks plausible and
   scores as an anti-correlation. Which peak is numerically larger is arbitrary,
   so the choice is made physically: cells add optical path, so the correct
   reconstruction is right-skewed in phase.
2. **Unwrapping.** Reconstructed phase is known modulo 2 pi. Uses scikit-image's
   quality-guided unwrapper, with a least-squares Poisson solve as fallback.
3. **Numerical aberration compensation.** The objective imposes curvature and the
   reference beam a tilt; together they dominate the recovered phase. A
   polynomial surface is fitted to the background and subtracted. **A plane fit
   is not enough on this data** -- order 3 is needed before the recovered in-cell
   phase contrast matches the reference.

**Read `reconstruction_valid` before quoting any baseline number.** The script
reports the recovered in-cell phase contrast and marks the run invalid when it is
not positive. A failed classical reconstruction is not a baseline, and reporting
one as if it were would overstate what the network achieves. The usual causes, in
order: the propagation distance is wrong; the aberration order is too low; or,
for the in-line arm at z = 0, the reconstruction is degenerate by construction
(section 5a).

---

## 11. Known limitations

**Segmentation labels are phase-derived**, not manual (§4). Segmentation and
measurement scores are agreement with an automatic procedure. This is the single
largest caveat on any number the framework produces. It has a second edge (§4.1):
because the mask is a deterministic function of the phase, the two supervised
tasks are not independent, so Dice partly measures phase accuracy and the physics
coupling terms constrain a quantity the primary losses already determine.

**Detection recall is well below one.** Roughly two fifths of reference cells are
never matched, so per-cell measurement accuracy is conditioned on the subset the
model finds. Whole-field dry mass is biased low mainly through this route rather
than through mismeasurement of the cells it does find. Any per-cell number should
be quoted with its detection recall beside it.

**Augmentation diversity was reduced in runs produced before the worker-seeding
fix.** With `num_workers > 0`, each dataloader worker held an identical random
stream, so crop offsets and flips repeated across workers within an epoch and the
effective augmentation diversity was 1/`num_workers` of the intended. This is
fixed by `_seed_worker` in `holoqpi/data/dataset.py`; results produced before it
remain valid but were trained under weaker augmentation than configured.

**Absolute dry mass depends on λ and α**, which are inherited and literature
values respectively. Confirm both with the acquisition group. Relative
comparisons are invariant to them.

**The pixel pitch was decoded from an undocumented header field.** It is
**isotropic at 0.284871 µm** in both axes. An earlier reading of the headers took
a second field to be an anisotropic y pitch of 0.211994 µm; that was corrected on
2026-09-10 (§3), and the correction changes every absolute area by 34.4%.
`prepare` re-reads the pitch from the headers on every run and warns on
disagreement beyond `header_pitch_tolerance_um`, but the header interpretation
itself rests on the acquisition group's confirmation rather than on documentation.

**The amplitude reference is a reconstruction, not a measurement.** The
amplitude target is the modulus of a classical off-axis reconstruction (§7.2),
so it carries that algorithm's sideband-filter losses, residual twin-image
structure and any illumination vignetting. Every amplitude number is agreement
with that output. The comparison against the thin-phase assumption A = 1 is
therefore the defensible claim, and the comparison against a physical
transmittance is not available from this data.

**The forward-model residual is anti-discriminative near the truth** on this
data (§6.2a): perturbing the reference phase by 10% lowers it, and every trained
arm scores below its own reference-phase floor. It is reported as a consistency
diagnostic and as a measure of acquisition-chain mismatch, and it is not used as
a training signal in any configuration the study recommends.

**The propagation distance is not identifiable off-axis.** An off-axis carrier
records the phase at any reconstruction distance, so the residual is nearly flat
in z and `calibrate_z.py` correctly refuses to return a value; in-line it is
identifiable, with per-field agreement. An arm that makes z a free parameter and
settles near its initialisation is therefore exhibiting a weak gradient, not
recovering a distance, and must not be reported as a measurement of one.

**The reference phase is itself a reconstruction**, not a ground truth. The
network is trained to reproduce the lab's numerical reconstruction, so it
inherits that pipeline's aberration compensation and any error in it. Phase
accuracy should be read as agreement with the existing pipeline.

**The Gabor arm carries twin-image artefacts** that off-axis reconstruction
removes analytically. This is a genuine physical asymmetry between the two
configurations and is expected to be one of the study's findings, not a defect to
be corrected away.

**Efficiency is profiled on the training machine.** Latency, throughput and
memory establish the relative ordering of configurations; they are not a
demonstration of embedded deployment, which requires evaluation on the target
hardware.
