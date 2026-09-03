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
| `refraction_increment_ml_per_g` | 0.185 | adherent cancer cells; 0.20 is haemoglobin-specific |
| `pixel_pitch_x_um` | 0.284871 | decoded from the phase headers |
| `pixel_pitch_y_um` | 0.211994 | decoded from the phase headers |

These give 0.060391 µm² per pixel and 0.034601 pg per rad·pixel.

`prepare` re-reads the pitch from the headers and warns if it disagrees with the
configuration beyond `header_pitch_tolerance_um`. Every area, volume and mass
scales with these numbers, so a silent mismatch would corrupt the whole
measurement chain.

> **Confirm λ and α with the acquisition group before publishing absolute
> picograms.** The wavelength is inherited from the previous instrument and α is
> a literature value for non-erythrocyte cells. All *relative* comparisons, and
> every agreement statistic, are invariant to both.

**Sanity check on the delivered data.** Running the measurement chain over the
phase-derived masks gives a median equivalent diameter of 18.2 µm and a median
dry mass of 149 pg (IQR 61–240), both within the expected range for adherent
cancer lines. This is an independent confirmation that the header decoding, the
pixel pitch and the calibration constants are mutually consistent.

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
| **Segmentation** | Dice, IoU, Aggregated Jaccard Index, Boundary F1 (2 px tolerance) |
| **Detection** | recall, precision and F1 of cell instances at `match_iou_threshold`; counts of reference, detected, matched, missed and false-positive cells |
| **Classification** | accuracy, macro F1, balanced accuracy, per-class F1, confusion matrix |
| **Measurement** | per-cell MAPE for area, optical volume and dry mass; per-cell and per-image Pearson r; Bland-Altman relative bias and limits of agreement |
| **Efficiency** | parameters, GMACs, latency p50/p99, FPS, peak VRAM, under PyTorch and ONNX Runtime |

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
the paper has to answer; none restates a table. Nine of the twelve are built from
files earlier steps already wrote, so they regenerate in seconds; figures 1 and 9
need a forward pass and are skipped automatically when no checkpoint exists.

```
python scripts/make_figures.py --config config/base.yaml          # all
python scripts/make_figures.py --config config/base.yaml --only 2 4 5
python scripts/make_figures.py --config config/base.yaml --list   # what each one is
```

Output is `figures/fig<NN>_<name>.png` (for reading) and `.pdf` (for the paper).
A figure whose inputs are missing prints its reason and is skipped; the rest still
build.

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

Colour is consistent across the set: off-axis blue, Gabor red. Modalities appear
in the order given by `--modalities`, so panels line up between figures.

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

**The pixel pitch is anisotropic** (0.285 × 0.212 µm) and was decoded from an
undocumented header field. `prepare` cross-checks it and warns on disagreement,
but the interpretation should be confirmed.

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
