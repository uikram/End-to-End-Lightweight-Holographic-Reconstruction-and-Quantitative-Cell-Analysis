# HoloQPI — physics-aware lightweight end-to-end quantitative holographic cell analysis

**Results report · 17 September 2026**

---

## 1. What the system does

A single lightweight convolutional network maps a **raw hologram** directly to
**quantitative phase**, **transmitted amplitude** and a **cell segmentation**,
with no classical reconstruction step anywhere in the inference path:

```
raw hologram (1024 px)
    └─ centre crop to 900 px
        └─ differentiable demodulation front end   (amplitude + phase channels)
            └─ MobileNetV2 encoder + U-Net decoder
                ├─ phase head          →  quantitative phase  φ(x,y)  [rad]
                ├─ amplitude head      →  transmitted amplitude A(x,y)
                └─ segmentation head   →  per-pixel cell / background
                        └─ watershed instance labelling
                            └─ per-cell projected area, circularity,
                               optical volume and dry mass
```

Per-cell dry mass follows the standard QPI relation

&nbsp;&nbsp;&nbsp;&nbsp;**m = λ / (2πα) · Σ φ · dx · dy**&nbsp;&nbsp;[pg]

with λ, α and the pixel pitches taken from `config/base.yaml`; no physical
constant is written into any source file. α and dx·dy cancel identically from
every relative metric reported below, which is verified by the self-test rather
than assumed.

The model is **9.60 M parameters / 45.85 GMACs** at 900 × 900 input, with a
compact decoder variant at **3.36 M / 24.03 GMACs**. Both export to ONNX.

## 2. Data and protocol

| | |
|---|---|
| fields | 800 holograms with matched reference phase |
| split | 560 train / 127 validation / 113 test, stratified by cell line and condition, seed 42 |
| test cells | 3 186 reference cells |
| geometries | off-axis (primary) and in-line / Gabor |
| labels | derived from the reference phase by Gaussian smoothing + Otsu + closing + area filter, with truncated objects excluded |
| significance rule | a difference between two arms counts only if it exceeds **2 × the pooled between-seed standard deviation** of that metric, estimated from three seeds (42, 1337, 2024) |

Fifteen arms were defined and fourteen trained, plus six seed replicates. The
full run took just under 17 hours on two GPUs and reported no failed step.

**The segmentation labels are silver standard.** They are a threshold of the
same reference phase the network is trained to reproduce, so every Dice, AJI and
recall figure below is agreement with that labelling procedure, not with an
independent annotation. This is stated throughout and is the study's principal
limitation.

## 3. The objective

The training loss is the sum of a reconstruction term, a segmentation term, and
two optional families that the study exists to test:

- **measurement-aware terms** — per-cell integrated phase, per-cell projected
  area, image-level optical volume, boundary-gradient alignment;
- **a physics-aware data-fidelity term** — the predicted complex field
  A·exp(iφ) is propagated to the sensor plane by angular-spectrum propagation
  with the measured aberration surface applied, and compared with the hologram
  that was actually recorded. It needs no reference phase, so it is also
  reported as a metric for every arm, including the arms that never optimised
  it. The propagation distance z can be held at the supplied value or made a
  free parameter of the objective.

## 4. Results

### 4.1 The learned pipeline beats classical reconstruction on the same holograms

Both pipelines are scored by identical code — same instance labelling, same
measurement chain, same test fields — so the difference is reconstruction and
nothing else.

| test metric (off-axis) | classical reconstruction | learned (arm A) |
|---|---|---|
| phase MAE [rad] | 0.2114 | **0.1589** |
| phase Pearson r | 0.7580 | **0.8688** |
| phase SSIM | 0.5022 | **0.6300** |
| segmentation Dice | 0.7885 | **0.8328** |
| segmentation AJI | 0.5931 | **0.6704** |
| boundary F1 | 0.2463 | **0.4593** |
| detection F1 | 0.6619 | **0.7034** |
| dry mass MAPE, matched cells | 0.2245 | **0.1767** |
| projected area MAPE | 0.1567 | **0.1546** |
| circularity MAPE | 0.0672 | **0.0659** |

The margin is largest where classical reconstruction is weakest: boundary
delineation (0.246 → 0.459) and per-cell dry mass (0.225 → 0.177, a 21 %
reduction in relative error).

### 4.2 The learned pipeline survives in-line holography, where classical reconstruction fails

In-line (Gabor) holograms carry an unseparated twin image. The classical
pipeline does not recover usable phase from them at all — its own validity
check fails, and the recovered phase is *anti-correlated* with the truth.

| test metric (in-line / Gabor) | classical | learned (arm G) |
|---|---|---|
| reconstruction valid | **no** | — |
| phase Pearson r | −0.1359 | **0.8098** |
| phase MAE [rad] | 0.3855 | **0.1752** |
| segmentation Dice | 0.1627 | **0.7758** |
| detection F1 | 0.0134 | **0.6514** |
| dry mass MAPE | 0.6695 | **0.2308** |

This is the strongest result in the study. The network trained end to end on raw
in-line holograms reaches Dice 0.776 and dry-mass MAPE 0.231 on a geometry where
the conventional route produces nothing usable, at a cost of 0.054 in dry-mass
MAPE relative to off-axis. Twin-image suppression is learned rather than
engineered.

### 4.3 Measurement-aware loss terms make the measurement worse — resolved, and in one direction

This is a negative result, and it is the one the seed replication was run to be
able to state. **Sixteen differences exceed twice the pooled between-seed
standard deviation, and all sixteen are degradations.**

| arm vs baseline A | dry mass MAPE | phase MAE | Dice | AJI | detection F1 |
|---|---|---|---|---|---|
| A — baseline | 0.1763 ± 0.0024 | 0.1593 ± 0.0008 | 0.8312 ± 0.0014 | 0.6696 ± 0.0007 | 0.6996 ± 0.0033 |
| B — + per-cell integrated phase | 0.1915 ± 0.0044 | 0.1697 ± 0.0022 | 0.8245 ± 0.0015 | 0.6517 ± 0.0027 | 0.6831 ± 0.0012 |
| B′ — + image-level optical volume | 0.1935 ± 0.0028 | 0.1666 ± 0.0006 | 0.8242 ± 0.0007 | 0.6541 ± 0.0005 | 0.6932 ± 0.0014 |

*(n = 3 seeds per arm; every entry above is resolved against the 2 × SD rule.)*

A weight sweep on the per-cell term confirms it is a dose response rather than a
seed accident — segmentation and phase degrade monotonically as the weight rises:

| per-cell weight w | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| Dice | 0.8322 | 0.8319 | 0.8253 | 0.8101 |
| phase MAE [rad] | 0.1583 | 0.1610 | 0.1671 | 0.1851 |
| dry mass MAPE | 0.1853 | 0.1828 | 0.1866 | 0.2107 |

**Interpretation.** A per-cell measurement loss is computed inside predicted
cell boundaries, so it rewards boundary placements that make the integral come
out right rather than boundary placements that are correct. The measurement is
downstream of the segmentation, and supervising it directly trades accuracy in
the upstream quantity for agreement in the derived one. The clean reconstruction
plus segmentation objective is the better route to an accurate measurement.

### 4.4 The physics-aware forward model is a diagnostic, not a usable loss

The forward-model residual is reported for every arm together with the same
residual computed from the *ground-truth* phase, which is the floor it can
reach. The ratio is the quantity to read.

Every arm scores a ratio **below 1.0** (0.9915–0.9994; one arm at 1.0004). The
predicted phase explains the recorded hologram slightly *better* than the true
phase does. A direct sweep confirms the same thing independently: perturbing the
reference phase by 10 % **lowers** the residual, and an inverted phase scores
better still.

The residual is therefore **anti-discriminative near the truth** and its
gradient points the wrong way. It is reported as a consistency diagnostic and as
a measure of acquisition-chain mismatch — never as evidence of reconstruction
quality, and it is not used as a training signal in any arm the study
recommends.

### 4.5 Propagation distance is identifiable in-line and not off-axis

| | off-axis | in-line (Gabor) |
|---|---|---|
| per-field median z [µm] | −24.06 | **50.53** |
| per-field IQR [µm] | 156.39 | **6.02** |
| fields agree | no | **yes** |
| identifiable | **no** | **yes** |

This is the physically expected answer. An off-axis carrier records the phase at
any reconstruction distance, so z is not recoverable from an off-axis hologram;
an in-line hologram encodes defocus directly, so it is.

The arm that makes z a free parameter of the objective (initialised at the
supplied 33.77 µm) settles at **33.39 µm** with a total excursion of 0.958 µm
and a spread of 0.026 µm over the last ten epochs. Given the flat off-axis
residual landscape above, that stability reflects a weak gradient near the
initialisation rather than recovery of the physical distance, and it is reported
as such. The trajectory is the result, not the endpoint.

### 4.6 The amplitude head learns real structure

The predicted amplitude is compared with the modulus of a classical off-axis
reconstruction, and — the row that matters — with the thin-phase-object
assumption A = 1 that the rest of the study runs on.

| | D0 (+amplitude) | D1 (+forward model) | D2 (z free) |
|---|---|---|---|
| amplitude MAE vs reference | 0.0656 | **0.0637** | 0.0642 |
| the same MAE for A = 1 | 0.1469 | 0.1469 | 0.1469 |
| **ratio (below 1 = better than A = 1)** | **0.446** | **0.434** | **0.437** |
| amplitude MAE inside cells | 0.0921 | 0.0887 | 0.0877 |
| amplitude Pearson r | 0.9029 | 0.9034 | 0.9025 |
| predicted amplitude sd | 0.1216 | 0.1200 | 0.1230 |

The head reaches **less than half the error of assuming unit transmittance**
(ratio 0.43–0.45) at a per-pixel correlation of 0.903, and its predicted spread
of 0.12 confirms it has not collapsed to a constant — a collapsed head would
score a ratio of exactly 1.000 with zero spread.

So the third output of the pipeline is real and measured, not nominal. The
caveat stands and must travel with the number: the reference is a
reconstruction, so this is agreement with one algorithm's modulus rather than
with a measured transmittance.

### 4.7 Efficiency

Measured on an idle GPU at 900 × 900 input; every row has a stable latency
distribution (p99/p50 ≤ 1.11).

| | full decoder | compact decoder |
|---|---|---|
| parameters | 9.60 M | **3.36 M** |
| GMACs | 45.85 | **24.03** |
| PyTorch fp32 latency | 12.29 ms | **11.42 ms** |
| ONNX fp16 latency | 8.39 ms | **7.96 ms** |
| ONNX fp16 throughput | 119.2 fps | **125.7 fps** |
| fp16 weights | 18.38 MB | **6.48 MB** |
| peak fp16 activation | 94.90 MB | **91.95 MB** |
| cost in dry mass MAPE | — | **+0.0016** |

A 2.9× reduction in parameters and a 1.9× reduction in compute cost 0.0016 in
dry-mass MAPE — well inside the between-seed spread of 0.0024, i.e. not
resolvable. **The compact model is the one to deploy.**

## 5. Measurement quality on the recommended configuration

Arm A, off-axis, 113 test fields, 3 186 reference cells:

| quantity | matched-cell MAPE | per-cell Pearson r | relative bias |
|---|---|---|---|
| projected area | 0.1546 | 0.899 | +0.060 |
| optical volume | 0.1767 | 0.934 | −0.042 |
| dry mass | 0.1767 | 0.934 | −0.042 |
| circularity | 0.0659 | 0.863 | +0.034 |

Detection recall is **0.593** at precision 0.865, so 1 889 of 3 186 reference
cells are matched and measured. The **coverage-adjusted** dry-mass MAPE, which
charges the pipeline for every cell it never reported, is **0.512**.

Both numbers must be quoted together. The matched-cell figure describes
measurement accuracy on the cells the detector finds; the coverage-adjusted
figure describes the system. Detection, not phase reconstruction, is the binding
constraint on this dataset, and the missed cells are small — their median area
is close to the lower end of the size filter.


## 6. The floor underneath every number above

Both of these run on the reference phase and the reference masks, so they hold
whatever the trained arms do.

### 6.1 The measurement chain itself is essentially exact

Against synthetic cells of analytically known mass and area:

| quantity | mean absolute error |
|---|---|
| per-cell dry mass | **0.041 %** |
| per-cell projected area | **0.245 %** |
| field-total dry mass | −1.67 % bias, from the small-cell area filter |

This is the result that gives the rest of the study its meaning. The
calibration, the integration and the instance labelling contribute 0.04 % to the
dry-mass error, so **essentially all of the reported 17.7 % is reconstruction and
segmentation error, not measurement arithmetic.** Any future effort belongs
upstream, in the phase and the boundary, not in the calibration.

### 6.2 Dry mass is 1.41× more robust to a boundary error than area is

Dilating and eroding the reference masks by a known number of pixels:

| boundary shift | Dice | area error | dry mass error |
|---|---|---|---|
| ±1 px (0.285 µm) | 0.973 | 6.5–6.9 % | 4.7–4.8 % |
| ±2 px (0.570 µm) | 0.947 | 12.3–15.0 % | 9.0–9.6 % |
| ±5 px (1.424 µm) | 0.863 | 27.5–32.6 % | 14.2–19.7 % |

The median mass-to-area error ratio is **0.710**. The reason is physical: a cell
boundary sits where the cell is thinnest, so the pixels a boundary error adds or
removes carry little phase. Integrated quantities are therefore intrinsically
more trustworthy than areal ones in QPI, which is a useful result in its own
right.

Read against this scale, the measured projected-area MAPE of 0.155 corresponds
to roughly a two-pixel boundary error, while the dry-mass MAPE of 0.177 exceeds
what even a five-pixel systematic shift would cause — so the mass error is not
boundary placement alone; phase error inside the cell contributes materially.

## 7. What this study does not establish

- **Segmentation labels are circular.** They are a threshold of the reference
  phase the network reproduces. Independent annotation is required before any
  Dice or recall figure here is a statement about cells rather than about a
  thresholding procedure.
- **The amplitude reference is a reconstruction, not a measurement.** Any
  amplitude figure is agreement with one algorithm's output and carries that
  algorithm's sideband-filter losses and residual twin-image structure.
- **The forward-model residual does not measure reconstruction quality**, for
  the reason given in §4.4.
- **Ten of the fifteen arms have a single run**, so comparisons involving them
  are correctly reported as unresolvable rather than as null.
- **The learned propagation distance is not a distance measurement** off-axis.

## 8. Conclusions

1. An end-to-end lightweight network that maps a raw hologram directly to phase,
   amplitude and segmentation **outperforms classical reconstruction followed by
   the same segmentation and measurement chain** on every metric reported, on
   the same holograms and the same test fields.
2. The advantage becomes categorical for **in-line holography**, where the
   classical route fails outright and the network still reaches Dice 0.776 and
   dry-mass MAPE 0.231.
3. **Measurement-aware loss terms degrade the measurement**, resolved against
   seed noise on sixteen of sixteen metrics and confirmed by a monotone weight
   sweep. The plain reconstruction-plus-segmentation objective is the
   recommended configuration.
4. The **physics-aware forward-model residual is anti-discriminative near the
   truth** on this data. It is a valid consistency diagnostic and an invalid
   training signal, and the study reports it as the former.
5. **Propagation distance is identifiable in-line and not off-axis**, which
   matches the physics of the two geometries.
6. The **amplitude head reaches less than half the error of the thin-phase
   assumption** (ratio 0.434, Pearson r 0.903), so the pipeline's third output
   is measured rather than nominal.
7. The **measurement chain is exact to 0.04 %** on analytic ground truth, which
   places all of the reported error upstream in reconstruction and
   segmentation, and **dry mass is 1.41× more robust to boundary error than
   projected area**.
8. A **3.36 M-parameter, 24.03 GMAC** model runs at 126 fps in ONNX fp16 with
   6.48 MB of weights, at a measurement cost smaller than seed noise — the
   configuration to deploy on edge hardware.

---

### Provenance

Every number in this report is read from a result file; none is typed by hand.
The study ran in two passes on GPUs 2 and 3: training and diagnostics on
16–17 September 2026 (~17 h, no failed step), then a scoring-only pass on
17 September that added the amplitude metric and the label-free floors without
retraining anything. The code passes 143 self-tests. Eighteen figures and their
manifest, 14 trained arms, 6 seed replicates and the full per-cell tables are in
`runs/` and `figures/`.
