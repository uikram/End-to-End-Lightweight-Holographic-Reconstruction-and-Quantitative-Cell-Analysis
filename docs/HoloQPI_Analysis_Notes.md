# HoloQPI — Analysis Notes

Every analysis note written during this project, in the order it was produced,
verbatim. This is the reasoning record: what was measured, what was found, what
was withdrawn, and why each decision was taken. The companion documents are:

- `docs/HoloQPI_Technical_Documentation.pdf` — 108-page implementation
  reference, what the code does, no results.
- `docs/HoloQPI_Project_Documentation.pdf` — the 2026-09-09 package, results
  and recommendations.
- `docs/RUNBOOK.md` — every command, in order.
- `docs/v2_code_changes.md` — before/after for every code change.
- `runs/RESULTS.md` — the tables, assembled from the run files.

## Contents

1. Baseline framework analysis (paper 1, RBC) — 2026-09-01
2. HoloQPI framework notes — 2026-09-01
3. The physics-loss finding — 2026-09-03
4. The conjugate sideband sign error — 2026-09-04
5. Forward-model consistency: the settled verdict — 2026-09-04
6. Extended framework status — 2026-09-05
7. Literature review: physics-aware lightweight end-to-end holography — 2026-09-05
8. Seed replication and the label confound — 2026-09-07
9. v2 operating brief: independent verification — 2026-09-08/09
10. Null-input probe — 2026-09-09
11. Findings from building the full documentation package — 2026-09-09
12. Literature analysis: QPI_Extended_Lit — 2026-09-10
13. Forward-model term and amplitude reference: measured verdict — 2026-09-10
14. v2 code-fix round — 2026-09-10
15. Membrane channel: registration solved — 2026-09-14
16. Complete technical documentation — 2026-09-14
17. The v2 study: what it produced — 2026-09-15

---

# 1. Baseline framework analysis (paper 1, RBC) — 2026-09-01

Technical reference for the existing `lightweight_qpi_segmentation` codebase, produced before extending
the framework to a new project. Covers architecture, LoRA integration, the physics-aware loss, the data
and dry-mass pipelines, evaluation metrics, and the edge-deployment path.

## System architecture and control flow

`main.py` is the single entry point with three modes — `train`, `evaluate`, `sweep` — all driven by a YAML path.
`utils/config.py` flattens nested sections into a flat `SimpleNamespace`, so `config.lora_r` and
`config.lambda1_pmc` are reachable regardless of nesting depth. Everything downstream reads config via
`getattr(config, key, default)` — defensive by default, no schema class.

`models/get_model()` is the registry: it builds the architecture, then injects LoRA **only if `lora_r` is
present in the config**. That single conditional separates a LoRA run from a full fine-tune run — the
full-FT YAMLs simply omit the `lora:` block, which makes the Table 3 comparison genuinely apples-to-apples.

`run_sweep()` loops ranks in-process, overriding `lora_r` **and** `lora_alpha = float(r)` before construction
(so scaling `α/r ≡ 1.0` across the whole sweep), redirecting `results_dir → {base}_r{rank}`, training,
auto-evaluating the best checkpoint, then `del model; gc.collect(); torch.cuda.empty_cache()` between ranks.

Root-level `train.py` is a cosmetic terminal simulator, not a trainer. The real loop is
`training/trainer_seg.py`.

## Models and LoRA integration

Three architectures, each single-channel-adapted by **averaging pretrained RGB stem weights across the
channel dim** (`weight.data.mean(dim=1, keepdim=True)`), applied to `features[0][0]` (MobileNetV2),
`patch_embed.seq[0].c` (TinyViT), and the first `in_channels==3` Conv2d found by traversal (EdgeSAM).

`models/lora_utils.py` is the core:

- **`LoRALinear`** — frozen `weight`/`bias` as non-grad `nn.Parameter`, trainable `lora_A` (Kaiming) and
  `lora_B` (zeros); output `= Wx + b + (dropout(x) A^T B^T)·α/r`.
- **`LoRAConv2d`** — low-rank as a 1×1 `lora_A` into a k×k `lora_B`; merge via `einsum('orhw,ri->oihw')`.
- **`inject_lora_into_model()`** — freezes everything, walks `named_modules()`, swaps matching layers, then
  selectively re-enables three groups: head patterns
  (`final_conv / mask_decoder / simple_decoder / prompt / outc / final_up`); **all**
  BatchNorm/LayerNorm/GroupNorm affine params; any `Conv2d` with `in_channels == 1` (the QPI stem).

  This tri-part unfreeze is the methodological signature — LoRA supplies the low-rank subspace, but domain
  shift is absorbed by norms and stem.
- Guard: `groups > 1` convs are skipped entirely, so depthwise layers never get LoRA (avoids the merge-shape
  crash). Only pointwise/expand convs are adapted in the CNN paths.
- `merge_lora_weights()` merges **and deletes** `lora_A`/`lora_B`/`lora_dropout` submodules — explicitly for
  a clean ONNX graph.

Injection target and strategy differ per architecture: EdgeSAM and MobileSAM inject into `self.encoder`;
MobileNet-UNet injects into `self` then force-unfreezes every `dec*`/`up*`/`final*` param. Configs use
`encoder_only` for EdgeSAM and MobileNet-UNet, `attention_blocks` (q/v/qkv keyword match) for MobileSAM.

| Architecture | Trainable @ r=8 | Total | % |
|---|---|---|---|
| EdgeSAM | 534,205 | 6,004,333 | 8.90% |
| MobileSAM | 673,184 | 6,719,932 | 10.0% |
| MobileNet-UNet | 5,584,173 | 7,773,357 | 71.8% |

## Physics-aware loss

`training/losses.py`, `PhysicsAwarePhaseLoss`:

**L = L_Dice + 0.1·L_PMC + 0.05·L_BGA + 0.1·L_PV**

Structural decision: the **biology is multi-class, the physics is binary**. `MultiClassDiceLoss` runs over
all 5 classes with weights `[0.5, 1.0, 1.5, 2.0, 2.0]` and Laplace smoothing `+1` (chosen for FP16
stability, not just epsilon-hygiene). The three physics terms operate on a foreground probability derived
as `1 - softmax[:, 0:1]` against a binarized target, with `apply_sigmoid=False` since softmax already
normalized.

- **PMC** — `relu(μ_bg − μ_cell + margin)`, area-normalized, with a throttled degenerate-collapse warning
  at `fg_ratio > 0.85`.
- **BGA** — forward-difference gradient magnitude of mask and phase, each **per-sample max-normalized** to
  [0,1] before an L1 penalty. That normalization fixes the probability-vs-radians scale mismatch.
- **PV** — `|Σ(M_pred⊙Φ) − Σ(M_gt⊙Φ)| / (|Σ(M_gt⊙Φ)| + ε)`, relative by default because unnormalized
  volume sums blew up FP16 gradients.

All three return **unreduced (B,) tensors**, masked by `has_cells` and averaged only over patches
containing cells — background-only patches contribute nothing rather than a spurious zero.

**Ablation grid:** `yaml_gen.py` programmatically emits 12 YAMLs — 3 architectures × {`dice_only`,
`pmc` (λ2=λ3=0), `pmc_bga` (λ3=0), `full`} — holding rank, schedule and split fixed. The rank sweep is
orthogonal, at r ∈ {2,4,8,16,32} with the `full` loss throughout.

## Data pipeline

`QPIDataset` expects `data_root/X_{split}` + `Y_{split}`: float32 TIFF phase (radians), uint8 masks with
pixel values 0–4 = background / discocyte / echinocyte / spherocyte / stomatocyte. **Storage day is parsed
from the filename prefix** (`YYYYMMDD`) and converted to days-since-earliest in a single post-loop pass —
that is how the 11 timepoints (0, 5, 8, 12, 15, 19, 23, 27, 30, 37, 47) reach the morphology CSVs without a
separate manifest. `__getitem__` failures self-heal by recursing to `idx+1`.

`datasets/qpi_augmentation.py` documents the physics-preserving policy — geometry only (flips, 90°
rotations, pad-crop translation), explicitly **no** intensity jitter, noise, or contrast change.

`SegmentationTrainer`: AdamW over `requires_grad` params only (hard error if the set is empty),
CosineAnnealingLR, `torch.amp` autocast + GradScaler gated to CUDA, best-checkpoint-on-`mean_dice`.
Validation runs **outside autocast in forced FP32** — a deliberate workaround for EdgeSAM FP16 eval
overflow.

Dataset scale: 1,090 densely annotated images → 1,024 train / 66 val. No independent held-out test set;
acknowledged in manuscript §6.5.

## Evaluation metrics

- **mean Dice / mean IoU** macro-averaged over classes **1–4 only** — background excluded, so a class the
  model never predicts scores 0 and drags the mean down. This makes MobileNet-UNet's mode collapse visible
  as Dice 0.417 rather than hidden.
- **AJI** on connected components of the binarized foreground, vectorized via `bincount` on
  `target*max_p + pred` hashes.
- **Boundary F1** at 2-px tolerance.
- **Phase-volume error**, global and per class: `|V_pred − V_gt| / |V_gt|` on raw phase.

## Morphology and dry mass

Three extraction paths, in order of physical rigour:

1. **`analysis/morphology_analysis.py`** — in-eval OpenCV analyzer. Arbitrary units.
2. **`dry_mass.py`** — physically calibrated: `V_φ = Σφ·Δx²`, `m = λ/(2πα)·V_φ`, with λ=666 nm,
   α=0.2 mL/g, Δx=0.1441 µm/px → **0.011005 pg per rad·px²**. Carries `_load_checkpoint_verified()`,
   which aborts if <90% of tensors match — a guard against `strict=False` silently producing a random
   network that still emits plausible masks.
3. **`calibrated_dry_mass.py`** — the two-pass path actually used for the manuscript, because the archived
   reconstructions are 8-bit with an unrecorded radian range. Pass 1 subtracts the per-image **modal
   non-cell intensity** as phase zero; pass 2 solves one global scale `s` so the day-0 median equals 30 pg
   MCH (s = 0.0139 rad/level). Gates on foreground Dice ≥ 0.40 against GT on 12 samples, else abort.

Outputs: **`dry_mass_pred.csv` = 1,974 cells** from EdgeSAM r=8 predictions (the headline number);
`dry_mass_calibrated.csv` = 2,467 cells from ground-truth masks (annotator reference, r = 0.966).

Statistics use the **image as the unit of analysis (n=66)** with Spearman ρ, Mann–Whitney U, and partial
correlation controlling for area — central finding: area −40.9%, dry mass −14.4%, residual mass trend not
significant after controlling for area, i.e. geometric contraction with approximate optical-mass
conservation.

## Edge deployment

`benchmark/benchmark.py` runs 3 configs × ranks {0,2,4,8,16,32} × precisions {fp32, fp16, int8} ×
{PyTorch, ONNX}. `merge_lora()` before export so low-rank branches vanish; `opset_version=17` with
`dynamic_axes` removed; native `.half()` for CNNs, `quantize_dynamic(QUInt8)` for INT8; ORT CUDA EP with
`EXHAUSTIVE` cudnn search and a 4 GB arena cap; two module-scope monkeypatches needed to trace TinyViT;
50 warmup + 500 synchronized runs.

Headline: EdgeSAM r=8, PyTorch FP32 13.36 ms / 177.9 MB → ONNX FP16 4.33 ms / 228.3 FPS / 16.4 MB
(NVIDIA RTX A5000).

## Observations flagged (no changes proposed)

1. `calibrated_dry_mass.py`'s hard-coded `_Cfg` defaults disagree with `configs/edge_sam_lora.yaml`. The
   YAML overrides them when present and the checkpoint-match gate catches the failure, but the defaults are
   a silent trap if the YAML path is wrong.
2. `results/dry_mass_by_day.csv` reports ~4,000 pg/cell — 8-bit levels integrated as if they were radians.
   Superseded by `calibrated_dry_mass.py` rather than wrong.
3. `results/benchmarks_without_trt/`, `_without_MobilenetFix/` and `_inclduing_mobilesamwrongresults/` are
   superseded benchmark runs; `results/benchmarks/` (20260619_025602) matches Table 7.

---

# 2. HoloQPI framework notes — 2026-09-01

Working notes for the follow-up study: an end-to-end network taking a raw hologram to a quantitative phase
image and a cell segmentation map, comparing off-axis against in-line Gabor acquisition.

## Data findings (verified, not assumed)

800 matched triplets: off-axis hologram + Gabor hologram + reconstructed phase.

- **3 cell lines × 5 drug conditions.** NCI 250, SNU 300 (control has 100), T24 250. Conditions: control,
  blebbistatin 5 µM, FCCP 10 µM, staurosporine 100 nM, rotenone 500 nM. All four drugs are morphologically
  active, so condition serves as the phenotype label.
- **Filename inconsistency.** SNU uses `blebbistatin` / `staurosporine` with no concentration; NCI and T24
  use `Blebbistatin_5uM` / `Staurosporine_100nM`. Rotenone capitalisation varies. The parser is
  case-insensitive with an optional concentration token.
- **Phase `.bin` format decoded**: 23-byte header (uint16 version, uint32 header length, uint32 width,
  uint32 height, float32 dx, float32 dy in metres, uint8 flag) then 900×900 float32 radians. Values roughly
  −2 to +5 rad, background referenced to zero.
- **Pixel pitch** decoded from an undocumented header field as dx = 0.284871 µm, dy = 0.211994 µm.
  *(Later corrected: dy is 0.284871 too — 0.211994 is the fluorescence camera's pitch. See note 12.)*
- **Holograms are 1024×1024 8-bit LZW TIFF** (needs `imagecodecs`).

## Alignment: centre crop, not resize

Tested by correlating a hologram-derived cell-texture map against the phase: centre crop scored 0.48,
resize 0.23. Sweeping the centred window size and rescaling to 900 peaks at S = 896–904, exactly what a
pure crop predicts. Usama independently confirmed centre crop.

This matters physically: resampling interpolates the fringes, and interpolated fringes no longer encode the
original optical path length.

## Architecture

Shared lightweight encoder (MobileNetV2 default; V3-Large and ResNet18 registered) feeding two U-Net
decoders plus an image-level classifier:

- phase head → radians, unbounded and unnormalised
- segmentation head → background/cell logits
- classifier → 5-way condition, `LayerNorm` not `BatchNorm1d` (batch-of-one is normal here)

Optional angular-spectrum front end (off-axis only, differentiable sideband demodulation) and optional
LoRA. Input is reflection-padded to a multiple of 32 internally, so 900 px never gets resampled.

## Verification performed

- `scripts/selftest.py`: dry mass matches the analytic value on a synthetic cell; identical prediction
  scores perfectly on all metric families; eroded prediction degrades as expected, with area and mass MAPE
  coinciding exactly as the physics requires; every loss term finite, differentiable and correctly signed.
- Full pipeline run on 13 real staged samples: prepare, train, evaluate, compare, benchmark, export.
- ONNX parity with PyTorch to 1e-7. LoRA merge bit-exact (0.0 output difference).
- **Plausibility on real data**: phase-derived masks give median equivalent diameter 18.2 µm and median dry
  mass 149 pg (IQR 61–240), both in range for adherent cancer lines.

## Bugs found and fixed during the build

1. **LoRA module replacement** — sliced `nn.Sequential` keeps the parent's child keys (`stage1.2`, not
   `stage1.0`), so positional indexing raised `IndexError`. The previous study's `_replace_module` had the
   same flaw; it only avoided the crash because it injected into unsliced modules. Now uses attribute access.
2. **`write_csv` truncation** — the header came from the first row only, so a row with extra keys (ONNX
   `providers`) raised mid-write and left a partial file. Now takes the union of keys.
3. `BatchNorm1d` in the classifier failed on trailing batches of one.
4. Stratified splits could leave validation empty when strata were too small to divide; small strata are now
   assigned by global deficit.

---

# 3. The physics-loss finding — 2026-09-03

The single most important result of the review. Read this before touching the loss functions.

## The claim

The "physics-aware" losses in `holoqpi/losses/terms.py` are **not** what the professor's reference papers
mean by physics-aware, and that mismatch is the direct cause of the null ablation.

All four references define physics consistency as **data fidelity against the measured hologram through the
forward propagation model**, `L(H(o), i)`:

- Huang, Chen, Liu & Ozcan, *Nature Machine Intelligence* 2023 (GedankenNet) — physics-consistency loss
  between the input hologram and the hologram predicted by forward-propagating the network's output complex
  field. No ground-truth object fields used at all.
- Galande et al., *J. Biomed. Opt.* — "the data fidelity term promotes data consistency using the hologram
  formation model."
- Lee, Mammadova, Barg & Jang, *APL Mach. Learn.* 4, 026106 (2026) and arXiv:2507.00482 — object-to-sensor
  distance as an implicit style; inverse mapping learned from intensity measurements only.

The current implementation has five physics terms and **none touches the raw hologram**. All couple
predicted mask to predicted phase.

## Why the old loss worked and the ported version does not

| | Old paper | Current code | Correct extension |
|---|---|---|---|
| Phase | **measured input** | learned, supervised | learned, supervised |
| Mask | learned | learned, supervised (= Otsu of phase target) | same |
| Term couples | learned mask ↔ **measured** phase | learned mask ↔ learned phase | learned phase ↔ **measured hologram** |
| New information | the measured phase field | none | the raw hologram |
| Effect | 15.52% → 3.01% phase-volume error | all deltas < 3.5% | untested |

Measured confirmation of the redundancy: Dice between the segmentation head and a threshold of its own
predicted phase is **0.920**, higher than either agrees with ground truth (0.822 head, 0.828 threshold).

## Three levels of constraint

```
L1  forward model    || H_z(A_hat * exp(i*phi_hat)) - I ||   MISSING, informative
L2  mask <-> phase   PMC, BGA, phase-volume                  carried over, redundant
L3  measurement      dry mass, projected area                added, redundant
```

## Blocker

L1 needs sample-to-sensor distance `z`. Not in the repo — the phase `.bin` header carries width, height,
pitch_x, pitch_y only; λ = 0.666 µm is an inherited config constant. Fallbacks: per-field autofocus over a
z sweep, or make z a learnable parameter.

The propagation kernel already exists in `holoqpi/models/frontend.py` (`AngularSpectrumFrontEnd`). It needs
moving from the input path into the loss.

## Two other config-level facts only visible in the code

- `lora.enabled: false` in **every** resolved config. LoRA has never run in this study. Do not describe it
  as a component of the reported results.
- `frontend.kind: none` in every resolved config. The angular-spectrum front end has never run either.
  Opportunity: it exists only for off-axis, so switching it on measures how much of off-axis's advantage a
  network extracts unaided.

---

# 4. The conjugate sideband sign error, and what it explains — 2026-09-04

*Updated with the 800-field server run. Supersedes the open question in note 3 about why the off-axis
forward-model term was anti-discriminative.*

## Summary

The off-axis forward-model consistency term was reported as **anti-discriminative** — a degraded phase
scored a *lower* residual than the true phase, so minimising it would have driven the reconstruction away
from the reference. That was an artefact of an unresolved conjugate-sideband sign, not a property of the
geometry.

Fixing it also made **z identifiable for the first time**: the in-line arm now places the propagation
distance at **+34.377 µm** (per-field IQR 0.72 µm) against the **33.77 µm** the acquiring group supplied
independently. Two unrelated routes agreeing to within one grid step is the strongest evidence in the study
that the forward operator is physically correct.

## The mechanism

An off-axis hologram carries the object in two first-order sidebands that are complex conjugates of equal
magnitude. Which one a spectral `argmax` returns is arbitrary and flips between fields of one acquisition.
Taking the wrong one returns the conjugate field, whose phase is **negated** — a reconstruction that looks
plausible and unwraps cleanly.

`scripts/estimate_aberration.py` reconstructed each off-axis field classically and fitted a polynomial to
`raw − reference`, calling the result the aberration surface. On **577 of 800** fields (72%) the
reconstruction was conjugated, so the difference it fitted was not a surface at all but roughly
`−(2·φ_ref + Ψ)`.

A fifth-order polynomial still fits that to R² ≈ 0.99, because the cells are a small part of the total
variance — **the fit's own quality metric could not see the error**. (This is the second time R² has failed
as a gate here; the first was polynomials perfectly fitting unwrapping failures. It should not be trusted as
a quality gate anywhere in this pipeline.)

## The fix

`holoqpi/physics/surface.py` — `resolve_conjugate()`. The sideband is chosen on a physical ground: cells are
optically denser than their medium, so they add optical path, and a field of sparse cells on a flat
background is right-skewed in phase. No mask, no threshold, no reference — so it applies at inference time
too.

**The surface must be removed before measuring the skewness.** The objective's curvature is a quadratic bowl
spanning tens of radians whose own skewness far exceeds the cells' few radians. Measured over all 13 locally
available fields:

| test applied to | picks the correct sideband |
|---|---:|
| wrapped phase (what `conventional_baseline.py` used) | 6 / 13 — chance |
| unwrapped phase, no detrending | 0 / 13 — systematically wrong |
| detrended, order 1 (plane only) | 0 / 13 |
| **detrended, order 2** | **13 / 13**, margin ≥ 1.2 |
| detrended, order 5 | 13 / 13 |

`optics.conjugate.detrend_order` is required to be ≥ 2 and `phase_skewness()` raises otherwise.

## Measured effect, 800 fields

| aberration fit | before | after |
|---|---:|---:|
| R² median | 0.982 | **0.9975** |
| R² minimum | 0.81 | **0.9809** |
| median surface span | 23.4 rad | 18.1 rad |
| usable fields | — | 780 / 800 |
| fields needing the flip | — | 577 / 800 (72%) |
| detrended agreement, median | — | +0.920 |

The 20 rejected fields are unwrapping failures with 388–493 rad surfaces, every one scoring R² = 1.000.

## Two further traps found in the z scan itself

**1. The scoring window depended on z.** `border_px` defaults to `pad`, and `pad` is derived from z, so every
distance in a scan was scored on a different set of pixels: 29% of the field at |z| = 66 µm, 58% at 34 µm.
Fewer, more central pixels are easier for four free radiometric coefficients to fit, so the residual fell
with |z| for reasons unrelated to focus. Now one border, sized for the largest |z|, applied at every distance.

**2. "Anti-discriminative" was mislabelled.** The verdict fell through to that label whenever a margin failed
to clear the tolerance. Taxonomy is now four-way — `usable` / `marginal` / `uninformative` /
`anti_discriminative` — with the last requiring a margin below *minus* the tolerance.

**3. The verdict is now a per-field sign test**, not a mean over four images.

## Consumers fixed

1. `scripts/estimate_aberration.py` — the stored surface (primary).
2. `scripts/conventional_baseline.py` — tested the *wrapped* phase, i.e. chance. The classical baseline's
   reported phase was negated on a random subset, so any previously reported classical-baseline numbers are
   invalid and must be re-run.
3. `holoqpi/models/frontend.py` — used a bare `argmax` with no half-plane rule at all.

---

# 5. Forward-model consistency: the settled verdict — 2026-09-04

## The decision

**The forward-model consistency term is not trained.** `loss.weights.forward_model` stays at 0.0 for every
arm. The diagnostics below are the result, not a gap in it.

## What was measured

32 val fields at z = 33.77 µm, fixed 448 px scoring window, per-field sign test. Margin = mean rise in
residual over the true phase. "Fields worse" = fraction of individual fields where the degraded phase scores
worse; chance is 50%.

| phase variant | off-axis margin | fields worse | Gabor margin | fields worse |
|---|---:|---:|---:|---:|
| × 0.9 | +0.0003 | 53% | −0.0020 | 19% |
| × 0.5 | +0.0157 | 72% | +0.0067 | 75% |
| + 0.3 noise | +0.0109 | **100%** | +0.0711 | **100%** |
| zero | +0.0468 | 62% | +0.1334 | **100%** |
| mirrored | +0.0942 | 84% | +0.1328 | **100%** |
| floor (true phase) | 0.7477 | | 0.8662 | |
| **verdict** | **uninformative** | | **marginal** | |

Both arms are correctly signed against gross corruption and blind near the truth. Neither can refine a
prediction that is already close.

## Three results worth reporting

**1. z is identifiable, and it corroborates the acquiring group.** The in-line arm places the minimum at
**+34.377 µm**, per-field IQR **0.00 µm** across 4 fields, well depth 2.8, interior to the scan. The
acquiring group independently supplied **33.77 µm** — agreement to within one grid step. The off-axis
residual is flat in z, which is physical: an off-axis carrier records phase at any distance, so z is not
recoverable from it. Before the conjugate fix nothing was identifiable at all.

**2. The calibration probe separates the geometries.** Multiplying the reference phase by *s* and minimising
the residual over *s* asks whether the operator agrees about phase **magnitude**.

| phase × s | 0.5 | 0.7 | 0.9 | 1.0 | 1.1 | 1.4 |
|---|---:|---:|---:|---:|---:|---:|
| off-axis | 0.4264 | 0.4018 | 0.3880 | 0.3851 | **0.3846** | 0.3937 |
| in-line | 0.9459 | **0.9450** | 0.9462 | 0.9475 | 0.9491 | 0.9552 |

Off-axis minimises at **s ≈ 1.05** — a genuine parabolic well centred on the delivered phase, i.e. the
operator recovers the correct magnitude to within 5–10%. In-line minimises at **s ≈ 0.68**, ~30% low, which
is what a single-term forward model does when the twin image is superposed on the object.

**3. The negative result is citable.** A physics-consistency loss of the kind Huang et al., Galande et al.
and Lee et al. define requires an invertible acquisition chain. Here the reconstruction runs inside closed
acquisition software, no uncorrected phase is saved, and the accumulated approximation puts a floor of 0.75
(off-axis) / 0.87 (in-line) under the residual that no prediction can beat. Seonghwan hit the same wall
independently. That makes this a reproducible limitation of the method under realistic data conditions, not
a local failure.

## What this does NOT invalidate

The conjugate fix touches `data/aberration.json` (used only by the forward-model term, weight 0.0) and the
angular-spectrum front end (`kind: none`, inactive). **The trained models and the base modality comparison
are unaffected and do not need retraining.**

---

# 6. Extended framework status — after the clean run of 2026-09-05

All 15 stages completed. Trained models are **valid and do not need retraining**. Two evaluation-only
defects were found afterwards and fixed.

## Headline results, test split, 113 fields

| metric | off-axis | in-line (Gabor) |
|---|---:|---:|
| phase MAE (rad) | **0.164** | 0.184 |
| phase MAE in-cell | **0.298** | 0.375 |
| phase Pearson r | **0.864** | 0.794 |
| Dice | **0.828** | 0.761 |
| AJI | **0.663** | 0.565 |
| boundary F1 | **0.438** | 0.292 |
| dry-mass MAPE | **0.188** | 0.233 |
| area MAPE | **0.154** | 0.172 |
| detection recall | **0.612** | 0.558 |
| detection precision | **0.863** | 0.784 |
| classification accuracy | 0.637 | **0.788** |

Off-axis wins every reconstruction and measurement axis. In-line wins classification, which is the one
reversal and needs an explanation in the paper.

## Learned vs classical — the strongest contrast in the study

| | classical off-axis | learned off-axis | classical in-line | learned in-line |
|---|---:|---:|---:|---:|
| in-cell contrast (rad) | +0.900 | — | **−0.073 (fails)** | — |
| phase MAE | 0.213 | **0.164** | 0.393 | **0.184** |
| phase Pearson r | 0.752 | **0.864** | **−0.155** | **0.794** |
| Dice | 0.786 | **0.828** | 0.160 | **0.761** |
| boundary F1 | 0.251 | **0.438** | 0.086 | **0.292** |
| dry-mass MAPE | 0.228 | **0.188** | 0.692 | **0.233** |

**The classical in-line pipeline fails outright** — at z ≈ 34 µm the object is essentially in focus and
`|A e^{iφ}|² = A²` contains no φ, and Gerchberg–Saxton (20 iterations tested, also 0/10/30) does not rescue
it. That is the study's value proposition for the in-line arm.

## Ablations: null, and honestly so

Across `no_physics`, `physics_coupling_only`, `no_measurement` and `base`, every
reconstruction/segmentation/measurement metric agrees within ±1%. Classification moves by up to 5 points but
in **contradictory directions** between modalities — the signature of single-seed noise, not effect.

`classification_only` control: phase Pearson −0.01, Dice 0.20, i.e. the shared encoder is genuinely doing
reconstruction work rather than the metrics coming free from the architecture.

## Dry-mass bias decomposition

| | off-axis | in-line |
|---|---:|---:|
| mass ratio pred/ref | 0.808 (−19.2%) | 0.867 (−13.3%) |
| from the boundary | 0.890 | 0.769 |
| from the phase | 0.908 | 1.127 |
| area ratio | 0.884 | 0.886 |
| phase bias in-cell | −0.212 rad | −0.179 rad |

Both heads contribute; neither alone explains it.

## Label audit — a confound that must be reported

Otsu threshold over 800 fields: 0.406 ± 0.151 rad, CV 37%. Between-condition SD 0.046 rad vs within-condition
0.146 rad. One-way ANOVA **F = 15.27, p = 4.9e-12**; Kruskal–Wallis H = 56.1, p = 1.9e-11.

The label definition is **partly confounded with the class label**: the threshold that defines "cell" shifts
systematically with drug condition.

## Defects found after the run (evaluation-only; no retraining)

1. **Unmeasurable piston phase.** Propagation multiplies by `exp(i2πz/λ)`, a global phase that no camera
   measures. The off-axis radiometric fit carried only the real parts of the two conjugate cross-terms, so
   the residual depended on where z fell modulo λ/2: measured, it swung between **0.133 and 0.991** over
   0.33 µm of z. Fixed by carrying both quadratures. In-line was always immune (`|U|²` cancels it).

   Effect: off-axis floor 0.7477 → **0.1815**; the off-axis scan's own minimum moves to **+33.42 µm**,
   within one grid step of the supplied 33.77 µm; calibration-probe per-field IQR 0.50 → 0.03.

2. **`audit_labels` ran before training**, so its head-redundancy section found no checkpoints and skipped
   while still reporting OK. Moved after the training stages.

---

# 7. Literature review — physics-aware lightweight end-to-end holography (2026-09-05)

*Every reference fetched and verified; unverifiable items flagged rather than dropped silently.*

## Citation problems to resolve before citing

- **Galande et al., J. Biomed. Opt. 29(10):106502** — the paper is real, but the network name
  **"HDPhysNet" appears in no indexed source**. Full text CAPTCHA-blocked at SPIE, PMC and PubMed. Confirm
  the name and author list from the PDF before citing it that way.
- **Lee, Mammadova, Barg, Jang, arXiv:2507.00482** — the preprint is real and confirmed. The
  *APL Machine Learning* 4, 026106 (2026) placement could **not** be confirmed (AIP returned 403). Cite the
  arXiv preprint until verified.

## A. Lightweight hologram → phase, and edge deployment

| # | Reference | Contribution | Gap |
|---|---|---|---|
| A1 | **OAH-Net**, Liu et al. 2025, arXiv:2410.13592 | Fourier Imager Head + complex net, raw off-axis → amplitude + phase, no unwrapping. **441K–9.24M params, 2.65–5.6 ms/frame** | No edge deployment, no quantisation/ONNX, no segmentation |
| A2 | **NAS-PRNet**, arXiv:2210.14231 | NAS encoder-decoder: 5.0M params, 31 ms, 36.1 dB vs U-Net 37.7M / 373 ms / 34.7 dB | No physics loss, no edge benchmarking |
| A3 | Li et al. 2025, Photonics 12(7):708 | Distils depthwise-separable U-Net to ~5.4% of params | No absolute latency; not biological QPI |
| A4 | Wang et al. 2025, Biophotonics Discovery | Full QPM pipeline on a $249 **Jetson Orin Nano**: 1,200 cells/s, >100,000 cells in <3 min, <5% error | Not a neural hologram→phase reconstruction |
| A5 | **HoloPhaseNet**, Jaferzadeh & Fevens 2022 | cGAN off-axis → phase without propagation or twin-image removal | No parameter count; systematic centre-of-frame phase error |
| A6 | **GedankenNet**, Nat. Mach. Intell. 5:895 | Physics-consistency on synthetic data only; ~128× faster than iterative | Not parameter-efficient; needs 2–7 multi-height holograms |
| A7 | **MorpHoloNet**, Nat. Commun. 16:4840 | Coordinate net + differentiable angular spectrum; single-hologram 3D RI | **10–20 min per sample** |
| A9 | Lee et al. 2026, Nat. Commun. | Physics-informed U-Net + metasurface, phase at **74 Hz**, sub-840 nm | Requires custom metasurface hardware |
| A10 | Endo et al. 2025, Appl. Opt. 64:A12 | **INT8 post-training quantisation**: ~70% smaller, ~4× faster | Computer-generated holography — technique transfers, application does not |

## B. Joint reconstruction + segmentation, and task redundancy

| # | Reference | Why it matters |
|---|---|---|
| B3 | **LACSS**, Commun. Biol. 6:232 | Instance segmentation from machine-generated weak labels; validates on independently annotated public benchmarks. **That validation pattern is the fix we lack** |
| B4 | **SegNetMRI**, IPMI 2019 | Shared encoders for reconstruction + segmentation. GT is **independent**, so it never tests the degenerate case |
| B7 | **DenoiSeg**, ECCV-W 2020 | **Explicitly reports that on clean/easy data the auxiliary task is solved trivially and its regularising benefit disappears** — our exact failure mode, in another modality. Closest citable precedent |
| B9 | **PCGrad**, NeurIPS 2020 | Gradient interference between tasks. Does not treat the opposite pathology — tasks so correlated the head adds no signal |
| B10 | **ForkMerge**, NeurIPS 2023 | Negative transfer driven by distribution shift; branch/merge machinery reusable to test whether our head earns its parameters |

## C. Physics-consistency losses, and amplitude without ground truth

| # | Reference | Why it matters |
|---|---|---|
| C4 | Wang & Lam 2024, arXiv:2404.01360 | **States PD losses are insensitive to low-frequency/background phase because it barely perturbs the hologram** — the mechanism explaining our rescaling blindness |
| C6 | Xiang et al. arXiv:2212.06725 | **Directly compares supervised vs unsupervised**: unsupervised amplitude MSE **0.00029** vs supervised 0.00018; unsupervised phase far worse, **0.06 vs 0.00048**. Read: supervise phase, let amplitude ride on physics |
| C2 | MorpHoloNet | Incident-field amplitude a **trainable parameter initialised from √(mean hologram intensity)**, never given GT |
| C9 | **Y-Net**, Opt. Lett. 44(19):4765 | One encoder, two decoders → intensity and phase from a single hologram. **This is our architecture** |
| C10 | **PhaseGAN**, Opt. Express 29(13):19593 | GAN + forward model learns from **unpaired** data |

## D. Measurement-preserving losses, dry mass, LoRA

| # | Reference | Why it matters |
|---|---|---|
| D1 | Barer 1952, Nature 169:366 | The Barer relation — origin of m = (λ/2πα)∫φ dA |
| D2 | Zhao, Brown, Schuck 2011, Biophys. J. | dn/dc mean **0.190 mL/g (SD 0.003)**, individual range **0.173–0.215** |
| D3 | Chaumet et al. 2024, Light Sci. Appl. | α for biological media **0.18–0.21 µm³/pg** |
| D6 | Hansen et al. 2023, ICML | **Soft conservation penalties are unreliable**; enforce the **integral (finite-volume) form** as a hard constraint. A stronger form of our idea |
| D7 | Idrees et al. 2018, ECCV | **Composition loss**: forces the spatial integral of a predicted density map to match the true scalar count. Direct precedent for an integrated-quantity loss |
| D9 | Bland & Altman 1986, Lancet 327:307 | Explicitly rejects correlation for method comparison. Our proportional bias needs the regression-based variant |
| D13 | Sui et al. 2025, BMC Med. Imaging 25:248 | LoRA-SAM: **81–89% Dice using 6.39%** of SAM's parameters. Cross-dataset, not cross-modality |

## What the review establishes

1. **The four-way gap is real.** No paper combines single-shot raw hologram → phase *and* amplitude, a
   lightweight backbone with reported parameters, a physics-aware loss, and real edge benchmarking.
2. **OAH-Net is the direct competitor** — already published, with parameter and latency numbers we must beat
   or match. Differentiate explicitly: + segmentation, + measurement-preserving loss, + edge deployment.
3. **Amplitude without ground truth has three established strategies**, none GT-equivalent. C6 is the
   decisive experiment: supervise phase, let amplitude ride on physics plus constraints.
4. **Our forward-model characterisation is novel.** C4 states the mechanism; **no paper found runs a
   degradation/discrimination test**. Our numbers are a reportable methods contribution.
5. **The segmentation-head redundancy has a precedent in B7, not a solution.** Every joint paper uses
   independent labels, so none of their arguments transfers. B3 shows the accepted fix.
6. **α = 0.185 is defensible but low.** Report with an uncertainty band.
7. **D6 is a stronger form of Integrated-Phase Preservation than ours**, and our dominant bias is exactly an
   integration-domain error.
8. **Cross-modality LoRA is an open gap**, which supports the proposed follow-up study.

---

# 8. Seed replication and the label confound — the decisive run (2026-09-07)

*Run 20260906_061844. Three runs of every ablation arm (seeds 42, 1337, 2024) plus a fixed-threshold label
arm. 27 stages, all green.*

## The headline: nothing resolves

**Zero of 54 comparisons** — two modalities × nine metrics × three ablation arms — exceeds twice the
between-run spread. The physics-aware and measurement terms do not change what this model measures. The null
result now has error bars.

Off-axis, mean ± SD over three runs:

| Metric | base | no_physics | coupling_only | no_measurement |
|---|---|---|---|---|
| Phase MAE (rad) | 0.1630 ±.0006 | 0.1621 ±.0002 | 0.1624 ±.0010 | 0.1627 ±.0006 |
| Dice | 0.8280 ±.0003 | 0.8281 ±.0007 | 0.8281 ±.0008 | 0.8281 ±.0011 |
| AJI | 0.6629 ±.0035 | 0.6634 ±.0015 | 0.6628 ±.0026 | 0.6610 ±.0038 |
| Boundary F1 | 0.4374 ±.0052 | 0.4375 ±.0016 | 0.4396 ±.0048 | 0.4360 ±.0051 |
| Dry-mass MAPE | 0.1835 ±.0045 | 0.1786 ±.0009 | 0.1822 ±.0020 | 0.1803 ±.0029 |
| Detection F1 | 0.7137 ±.0038 | 0.7124 ±.0018 | 0.7141 ±.0046 | 0.7100 ±.0053 |
| **Classification** | **0.6431 ±.0946** | 0.6106 ±.0580 | 0.6932 ±.0640 | 0.7080 ±.0265 |

## What the spreads reveal

Reconstruction, segmentation and measurement are **extraordinarily stable**: Dice varies by ±0.0003 between
runs, phase MAE by ±0.0006. Off-axis classification varies by **±0.0946** — nearly ten points.

**This retires the headline classification reversal.** In-line 0.8024 ±.0051 against off-axis 0.6431 ±.0946
is a difference of 0.159 against a pooled spread of 0.095 — under the same 2σ rule applied everywhere else,
it does **not** resolve. Report both with their spread; do not claim the reversal. Every classification claim
made from a single run in earlier drafts is withdrawn.

## The label confound: real, but not the driver

Masks rebuilt at a fixed 0.406 rad threshold, where the Otsu-drift confound cannot exist.

| | off-axis base | off-axis fixed | in-line base | in-line fixed |
|---|---:|---:|---:|---:|
| Classification accuracy | 0.593 | 0.522 | 0.797 | **0.850** |
| Phase MAE (rad) | 0.163 | 0.165 | 0.184 | 0.184 |

**Removing the confound does not cost the classifier anything.** The threshold drift is real in the labels
and must still be reported, but the network is not living off it.

**Caveat: segmentation metrics are NOT comparable across these two arms.** The fixed-threshold arm is scored
against its own masks, so its higher Dice (0.842 vs 0.828) means only that a fixed threshold produces more
self-consistent labels. Classification is the only clean cross-arm comparison.

## Absolute dry mass, and the α correction

Measured over 1,852 matched off-axis cells: reference mean **188.65 pg**, predicted mean 173.46 pg. The
literature range for α moves the reference figure across **[162.3, 201.7] pg**, a systematic band of
**+6.9% / −14.0%**.

**An earlier claim in this project was wrong and is corrected here.** α cannot explain any part of the
dry-mass error. The same λ/(2πα) multiplies the predicted and the reference mass, so it cancels exactly from
MAPE, from the mass ratio and from the relative bias. The run demonstrates it: MAPE = 0.188688 at α = 0.185
and 0.188688 at α = 0.215. `selftest.py` now pins this with three checks so the temptation cannot resurface.

α belongs in the paper as a systematic band on **absolute** picograms, and MAPE should be quoted with no α
caveat at all.

## What the study can now claim

1. A single 9.76 M-parameter network reconstructs quantitative phase and segments cells directly from a raw
   hologram — dry mass within 19%, Dice 0.83, 8 ms per 900×900 field as ONNX fp16.
2. The acquisition geometry dominates the loss function. Off-axis beats in-line on every reconstruction and
   measurement axis, while three runs of four objective variants produce not one difference that clears the
   between-run spread.
3. Learning buys most where classical reconstruction fails. Against a competent classical off-axis pipeline
   the gain is real but bounded (boundary F1 0.438 vs 0.251); against classical in-line, which recovers
   nothing at this defocus, it is the difference between a measurement and no measurement.
4. The negative result, with a mechanism and now with statistics: physics terms coupling reconstruction to
   segmentation cannot help when the segmentation labels are derived from the reconstruction target —
   0.93 head redundancy, a coupling term that was exactly zero throughout training, and 0/54 resolved.

## Remaining

- Detection recall 0.61 / 0.56 — the weakest headline number.
- **"Edge" is still unsupported**: all timings are from an RTX A5000 workstation GPU.
- Independent (manual) segmentation labels — the blocker for the Integrated-Phase Preservation extension.

---

# 9. v2 operating brief — independent verification against the code (2026-09-08/09)

*The joint Claude/GPT/Gemini brief was checked against the actual repository. Its priorities were right;
two of its factual claims were not, and one of its instructions would have been expensive and harmful.*

## Endorsed

Check the gradient path before anything else, independent labels are the critical path, multi-seed
discipline is non-negotiable, Option C (padded bounding box) is correctly rejected, and the in-line arm
needs a null-input control before any manuscript claim.

## Correction 1 — the term already works, no fix needed

```
cell_integrated_phase alone   sum |grad| 1.158e+02   disconnected 0/37
segmentation (Dice+CE) alone  sum |grad| 4.976e+02   disconnected 0/37
ratio 2.328e-01
```

Connected, 23% of the segmentation loss's magnitude. `foreground` is the softmax probability, and the domain
is the reference instance labelling, so `∂L_i/∂M̂(q) = 0` for `q ∉ Ω_i^GT` by construction. **The existing
implementation already is the brief's Option A** and cannot be gamed by mask expansion. The brief's Section 2
fix must not be applied on top.

## Correction 2 — both border mechanism stories were wrong

The brief said `binary_closing` is extensive and cannot erode, and that the real mechanism is the area filter
running before closing. Measured:

- scipy extensivity: a 12-px blob touching the top edge → **4 px** after closing. `A ⊆ A•B` is **False**.
  A 1-px rim → **0 px**. `scipy.ndimage.binary_closing` applies `border_value=0` to its erosion step, so at
  an array edge it is not extensive.
- Execution order in `masks.py` is threshold → closing → fill_holes → border → **area filter last**.
- Stage by stage over 5 fields: 4–11 border components at threshold, **zero immediately after closing**,
  before the area filter runs.

So the closing explanation is right, for a reason neither review had — scipy's border convention, not
textbook morphology.

## Disagreement — the vignetting-derived border buffer

Measured fall-off (cells masked out, 13 fields): −0.50 rad at 2 px against a −0.167 rad plateau, settling
within 3 sd at **34 px**. Cost of applying that:

| buffer | cells kept | % lost | foreground |
|---:|---:|---:|---:|
| none | 315 | 0.0% | 19.13% |
| **0** | **315** | **0.0%** | **19.13%** |
| 4 | 236 | 25.1% | 14.16% |
| 34 | 204 | **35.2%** | 12.20% |

A 34 px buffer discards a third of every cell in the dataset to suppress an artefact that provably never
reaches the masks. **Implemented `clear_border` with `border_buffer_px: 0`** — documented as
truncated-object exclusion, which is independently valid, costs nothing today, and becomes load-bearing the
moment `binary_closing_px` changes or an external annotator produces labels.

## Server results — verification round closed

**A. Gradient path — PASS.** At full scale, ratio **2.986e-01**, 0/37 disconnected. Reproduces the local
0.233 within batch-to-batch spread. **Set `cell_integrated_phase` weight from 0.299, not the placeholder 0.1.**

**B. Null-input probe — the two arms differ, and not in the direction the memorisation hypothesis predicts.**
See note 10.

**C. Label integrity — no cells lost.** Checked with the same labeller the evaluator uses (watershed, not
connected components — an earlier check of mine used CC and undercounted by 1.59×): 24,446 watershed
instances, 30.6 per field, against ~26.9/field implied by v1. The feared ~32% cell loss is excluded.

**Net: nothing to retrain.** The critical path is now entirely the independent segmentation labels.

---

# 10. Null-input probe — the in-line arm is not memorising (2026-09-09)

*Two failed versions of this probe before a usable one. Both failures are recorded because the failure
modes are instructive.*

## The question

The in-line (Gabor) arm reaches Dice 0.761 and phase MAE 0.184 rad where the classical pipeline recovers
nothing. Either the network extracts a real weak signal, or it has learned a shape prior and draws plausible
cells largely regardless of input. The second would make the in-line numbers uncitable as reconstruction.

## Two bugs in the probe, both mine

**1. A collapsed tier.** The script z-scored each synthetic input, which subtracts the constant and rescales
added noise to unit variance — so the "flat background" tier became a second copy of the noise tier. The
outputs were identical to three decimals. Not fixable by tuning: the network is fed z-scored holograms, so a
genuinely flat field standardises to exactly zeros and *is* the zero tier. Replaced with a
**pixel-shuffled real hologram** — identical intensity histogram, no spatial structure.

**2. An adaptive counting metric.** Cells were counted with Otsu, which splits whatever histogram it is
handed and can never report "no structure". The first run reported **57 "cells" on pure noise against 26.6
on real holograms** — a broken metric rather than a finding. Demonstrated directly on an untrained model:
**68 and 87 components from an output with sd 0.0002**. Replaced with a fixed +0.406 rad threshold.

**3. A binary verdict that was too crude**, replaced with a graded one keyed on the *fraction of pixels
above threshold*, which is scale-free.

## Results

Real input: gabor 27.0 cells/field, off-axis 26.6, ~13.5% of pixels above threshold.

| tier | gabor cells | gabor area | off-axis cells | off-axis area |
|---|---:|---:|---:|---:|
| zero | **0** | 0.00% | **0** | 0.00% |
| noise | 3 | 0.36% | 11 | 1.58% |
| shuffled real | 2 | 0.23% | 5 | 0.79% |
| **fraction of real area** | | **2.7%** | | **11.7%** |
| **verdict** | | **clean** | | **OOD hallucination** |

## What this establishes

**A constant input produces zero cells in both arms.** There is no unconditional shape prior — the model
responds to its input rather than drawing from memory. That clears the in-line result of the memorisation
charge.

**The in-line arm is the CLEANER of the two.** 2.7% against off-axis's 11.7%. This is the opposite of what
the memorisation hypothesis predicts: if Gabor's 0.761 Dice came from a learned prior, Gabor should invent
*more* than off-axis, not four times less. The hypothesis is not merely unsupported, it is contradicted in
the direction of the effect.

**Off-axis shows mild out-of-distribution hallucination** — ~12% of the real above-threshold area on inputs
with realistic contrast and no content. Report as a robustness caveat in the limitations.

---

# 11. Findings from building the full documentation package (2026-09-09)

*A 124-page documentation package was produced by re-reading the entire codebase, the run artefacts and
three parallel literature sweeps. Nine things surfaced that were not previously recorded.*

## 1. SEVERE — `runs/` contains smoke tests, not results

Every neural metrics file in the working copy came from a `QUICK=1` run: 2 epochs, 192-px crops,
`phase_n_images: 2`, `cells_reference: 4`. `v2_cell_ipp_off_axis/history.json` has **one epoch**;
`seed_aggregate.json` reports `"runs": 1`, `"sd": NaN`.

Three independent proofs they are degenerate: `phase_ssim` is **negative** (−0.249); `seg_dice` is
**byte-identical to sixteen digits (0.2627095679631393)** across four different conditions and two
modalities; `detection_f1` is `nan`.

**They prove the 27-stage pipeline runs. They are not evidence.**

## 2. SEVERE — 68.3% of the network is two copies of one layer

`ConvTranspose2d(1280→640, k2, s2)` is **3,277,440 parameters = 34.1% of the model**, duplicated in both
decoders.

| Module | Params | Share |
|---|---:|---:|
| phase_decoder | 3.6781 M | 38.3% |
| segmentation_decoder | 3.6781 M | 38.3% |
| encoder (MobileNetV2) | 2.2233 M | 23.2% |
| heads (2 ×) | 0.0186 M | 0.2% |

**Fix:** a `Conv2d(1280→256, k1)` before the first decoder block (cost 0.33 M) saves ~2.9 M per decoder.
**9.60 M → ~3.8 M.** This matters because OAH-Net does the same task at **3.7 M / 2.65 ms/frame** and
NAS-PRNet at **4.4 M / 11.3 GFLOPs / 31 ms** — on the efficiency axis we currently lose.

## 3. Both outputs are computed at stride 2 and bilinearly upsampled

Five encoder stages but only four decoder blocks, so the trunk is `(B,32,464,464)` for a 928-px padded
input. **The segmentation boundary — which sets projected area and near-edge phase integral — is resolved on
a 2-pixel grid.** A candidate contributor to the recall behaviour, but not demonstrated.

## 4. SEVERE — circularity assumes square pixels; the pixels are anisotropic

`cells.py` computes `C = 4π·A_px / P_px²` in **pixel** units, but `dx/dy = 1.344`. A physically circular
cell reports **≈ 0.978**, and the bias is **orientation-dependent**. Circularity is one of the four
measurements the professor named.

## 5. SEVERE — the configured `dy` contradicts the published instrument

Park *et al.*, *Microsyst. Nanoeng.* 12:311 (2026), DOI 10.1038/s41378-026-01424-9 describes a **666 nm**
laser and a **0.2849 µm** pixel. Our λ and `dx` match **exactly**; our `dy = 0.211994 µm` does not.

If the true pitch is 0.2849 µm square, **every absolute area and dry mass in the project is low by 34%.**
Relative metrics are unaffected.

## 6. The professor's datasets — found, and not public

Three human cancer lines **SNU-475 / T-24 / NCI-H1299** and five conditions — an exact match to our labels.
One rig produces **both** off-axis and in-line Gabor via a motorised reference-beam shutter.

> Data availability, verbatim: *"available from the corresponding author upon reasonable request."*

**The highest-value question to ask:** the M&N paper co-acquired **DAPI / MitoTracker / CellMask
fluorescence** with co-registration validated by fluorescent microspheres. **If fluorescence exists for any
of our 800 fields it is a far better route to independent segmentation labels than a Cellpose pilot** — an
orthogonal physical measurement rather than another algorithm's opinion.

## 7. `calibrate_z.py --refine` crashes

`KeyError: 'estimators_agree'` at line 361. Not load-bearing — z is known from the acquisition — but it is a
crash in a script that will ship with the paper.

## 8. Experiment B′ does not exist, and it is the study's own claim

`phase_volume` is 0.0 in **all four** v2 configs. So the claim "per-cell beats image-level" rests entirely
on the synthetic self-test and **cannot currently be measured on real data.** Of every gap found, this is the
one most likely to be raised by a reviewer.

Related: nothing constrains **projected area** in v2. The right fix is a **per-cell area term** — the exact
analogue of `CellIntegratedPhase`, same scatter-add, ~15 lines.

## 9. Literature verdict — the novelty is the loss, not the architecture

Searched ~70 papers across three sweeps. **No paper was found doing all three of (a) raw hologram in,
(b) phase + amplitude + segmentation out, (c) an explicit constraint on the integrated phase inside each
cell.** No region-integral conservation loss inside a segmentation mask was found in *any* biomedical
imaging modality.

**Already taken — do not claim:**

| Claim | Pre-empted by |
|---|---|
| segmenting from a raw measurement | Schlemper 2018 (k-space); KMAE 2024 |
| multi-decoder hologram network | **Y-Net**, Opt. Lett. 2019 — *this is our architecture* |
| shared-encoder joint recon+seg | SegNetMRI 2019, Deep-SLR 2021, MTLRS 2023 |
| physics-informed hologram loss | PhysenNet 2020, GedankenNet 2023 |
| morphology + phase from one hologram with a physics loss | **MorpHoloNet**, Nat. Commun. 2025 |
| deep-learning dry mass | **PICS**, Nat. Commun. 2020 (<2% nuclear mass error) |
| lightweight off-axis phase retrieval | **NAS-PRNet** 4.4 M; **OAH-Net** 3.7 M / 2.65 ms |

**Also worth knowing:** α = 0.185 mL/g is **not** a value any primary source recommends. Barer & Joseph 1954
recommend **0.18**; Zhao, Brown & Schuck 2011 give **0.1899 ± 0.0030**; HoloPhaseNet uses **0.2**.

## One framing recommendation

**Call the contribution measurement-preserving, not physics-aware.** Four of the six physics-labelled terms
are switched off and the one that reads the sensor is blind near the truth. Present the physics findings —
the conjugate sideband and the piston-phase identifiability result — as separate, self-contained results.

---

# 12. Literature analysis: QPI_Extended_Lit — and what it does to our novelty claim (2026-09-10)

*Full analysis in `QPI_Extended_Lit/Research_Analysis.pdf` (65 pp).*

The folder holds **four distinct papers, not five** — `PEDS_20111635.pdf` is a byte-identical duplicate of
the Microsystems & Nanoengineering paper (same MD5). **Two of the four are Seonghwan's**, not one.

| Paper | Venue | Input | Params |
|---|---|---|---|
| **Park, Lee, Park & Moon (2026)** — dual-mode phase + label-free fluorescence | *Microsyst. Nanoeng.* **12**:311 | **Gabor** | **96.97 M** |
| **Park, Park, Kim, Moon & Javidi (2026)** — single-shot DH via unsupervised diffusion | *Eng. Appl. Artif. Intell.* **163**:112970 | **Gabor** | **361.26 M** |
| Liu *et al.* (2025) — **OAH-Net** | *Biomed. Opt. Express* **16**(3):894 | **off-axis** | **3.7 M** |
| Rivenson *et al.* (2018) | *Light Sci. Appl.* **7**:17141 | back-propagated in-line | not reported |

## 1. Most of our "quantitative cell analysis" contribution is already published on this dataset

Both Seonghwan papers use the same rig, the same three cell lines and the same five conditions, 750 fields
at 900×900, 666 nm, 0.2849 µm pixel.

Already published on it: **dry mass, projected area, optical/phase volume, circularity, OPD**; segmentation
by **smoothing + Otsu** (our exact method); cell-type LDA **75.0%**; drug-response LDA **47/47/42%**;
Gerchberg–Saxton comparison. EAAI Table 8 reports phase reconstruction on **our three lines** at PSNR
34.5–36.6 / SSIM 0.857–0.879 — in one row those lines were **never in training**.

**We cannot claim end-to-end hologram→phase, dry mass, area, volume, circularity, or quantitative cell
analysis as contributions on this dataset.**

## 2. Confirmed configuration error — `pixel_pitch_y_um` is wrong

EAAI §2.3 states the 20× cancer-cell field of view is **256.38 × 256.38 µm** at 900×900 — **square**. And
900 × 0.284871 = **256.384 µm** ✓. Our config has `pixel_pitch_y_um: 0.211994` ✗ (which gives 190.8 µm).
**Every absolute projected area and dry mass is 34.4% too low.** Fixing it also removes the circularity
anisotropy bug.

## 3. We are solving the easy direction

Our off-axis target is the **classical reconstruction of our own input**. Deterministic mapping, ceiling =
the algorithm. Both Seonghwan papers take **Gabor** input, where the twin image makes the problem genuinely
ill-posed.

OAH-Net is in our position and handles it correctly: it claims **speed (2.65 ms vs a 9.5 ms camera), 16×
storage reduction and robustness**, never accuracy. We must do the same, and say it in the methods before a
reviewer does. Framing that works: *off-axis well-posedness removes reconstruction ambiguity as a confound,
which is what lets the measurement question be isolated.*

## 4. What is actually still open — build the paper here

Every loss in all four papers is an **image-similarity** loss. Every reported number is an **image-quality**
metric. Two papers compute dry mass and **neither reports a per-cell error** — their validation is
distributional.

Genuinely open:

- **A loss that constrains the measurement.** Our `CellIntegratedPhase` has no analogue in these papers.
- **Segmentation as a learned output.** No paper outputs a mask; none reports Dice, IoU, AJI or boundary F1.
- **Per-cell measurement error** — MAPE, relative bias, Bland–Altman, with recall alongside.
- **Error propagation** — how a k-pixel boundary shift becomes a mass error. Needs no new data. Predicts
  error scales with **perimeter**, not area, and therefore that **Dice is the wrong metric to optimise for
  mass accuracy**.
- **★ The measurement cost of cheap optics.** Both papers argue Gabor ≈ off-axis on PSNR/SSIM. **Neither
  asks what it costs in picograms.** Only a framework with a segmentation output can answer it.

## 5. Efficiency — mixed, and currently unsupported

We are **10× smaller than M&N (97 M) and 37× smaller than EAAI (361 M)**. But **OAH-Net is 3.7 M at
2.65 ms**, and its **9.24 M variant-2 performs *worse* than the 3.7 M vanilla** (phase SSIM 0.943 vs 0.997)
at twice the latency. Our 9.60 M sits almost exactly there. And we report **no latency, no FLOPs, no
memory** — every other paper reports at least latency.

## 6. Ideas worth adopting

1. **★★★ Ask Seonghwan for the fluorescence channels.** M&N has NucBlue/MitoTracker/CellMask for these
   fields, microsphere-calibrated, and **explicitly reports that fluorescence-guided segmentation beats
   phase-only** (Suppl. S6). That dissolves our circular-label problem in a way no segmentation algorithm
   can. **One email.**
2. **★★★ OAH-Net's spatially weighted L1**:
   `W = 40·clamp(max(|φ̂|,|φ|),0,0.05) + 20·clamp(grad(φ),0,0.1) + 1`, with `w_A = 0.1`. Our evaluator
   already diagnosed the problem it solves — in-cell MAE 0.446 rad vs 0.273 whole-field. Tune the 40× down
   for our ~19% foreground (theirs is 1.04%).
3. **★★ Benchmark against instrument repeatability.** OAH-Net measured frame-to-frame MAE on a *static*
   sample: **0.030 ± 0.001 rad**, and their error (0.012) is below it. Ask for repeated frames; then express
   our error — and a **dry-mass repeatability floor in pg** — against that. Most rigorous idea in these papers.
4. **★★ The LDA-overlay figure** alongside Bland–Altman. It is possible to pass the LDA test and fail
   Bland–Altman; if we observe that, it is a headline result.
5. **★★ Their robustness battery** — six OOD scenarios, and the text says reviewers asked for them.
   **A physics-aware model should beat a learned prior on defocus.**
6. **★★ Physics-initialised trainable Fourier filters** — `frontend.kind: angular_spectrum` is already
   implemented and disabled. Enabling it puts physics in the *architecture*, a much stronger reading of
   "physics-aware" than a switched-off forward model.
7. **★ EAAI's limited-data result is a challenge:** unsupervised beat supervised at 50 labels/type. Answer
   with a data-efficiency curve at 25/50/100%.
8. **Saved experiment:** wrapped-vs-unwrapped targets is already settled (EAAI Table 9, unwrapped better).

## 7. Recommended reframing

**Stop describing a pipeline; describe a measurement problem.** Four pillars: error propagation;
measurement-preserving learning (A vs B vs **B′** vs B″, 3 seeds, weight swept from the measured 0.299);
the measurement cost of cheap optics, reported in pg and µm²; honest efficiency with FLOPs, latency and
memory measured.

This framing is **closer to the professor's brief** than our current one, it cannot be scooped by a bigger
model, and it **survives a null result**.

## 8. Also ask Seonghwan

The pixel pitch; **750 vs our 800 fields**; are the Gabor and off-axis exposures close enough in time to be
treated as **paired** for fixed cells; repeated frames of a static sample; **what is he working on next?**

---

# 13. Forward-model term and amplitude reference: measured verdict (2026-09-10)

*Measured on the 13 local fields, off-axis arm, z = 33.77 µm, border 64 px, criterion l2, crops and
augmentation disabled so the comparison is on identical pixels.*

## 1. The aberration surface is not static, and the deployable version costs a lot

| quantity | value |
|---|---|
| global surface peak-to-valley | 15.21 rad |
| per-field departure from it, all terms | median 10.62 rad (69.8%) |
| per-field departure, degree ≥ 2 (curvature) only | median 3.91 rad (25.7%) |
| per-field tilt, degree == 1 only | median 13.98 rad |

The **curvature is close to static**, which is what a fixed objective should look like. The ramp is not.

### A hybrid was tried and rejected by its own test

`reconstruct_off_axis` centres the sideband with `torch.roll`, which shifts by whole FFT bins only, so a
non-integer carrier provably leaves a residual ramp of up to π rad across the field. If the fitted tilt were
that remainder it could be predicted from the hologram alone and the surface would be deployable.
`optics.aberration.mode: hybrid` builds exactly that.

**It does not work here.** The carrier prediction explains **10.9%** of the fitted degree-1 tilt, and field
by field the two are uncorrelated in sign as well as magnitude. The tilt is therefore something else —
reference-beam drift between acquisitions, or a ramp introduced by the 2-D unwrapping — and this script
cannot separate those.

**Consequence:** the deployable aberration model on this instrument is the global surface, and it leaves an
unmodelled per-field ramp of order 10 rad.

## 2. The forward-model term cannot serve as a loss, in either mode

| response | per_field | global |
|---|---|---|
| residual on the reference phase | 0.44 – 0.51 | 0.87 |
| phase × 0.9 (mild, 10% bias) | **−0.0005 / −0.0006** | **−0.0027** |
| phase × 0.5 (coarse, wrecked) | +0.053 / +0.060 | +0.0016 |

**The mild degradation is not penalised in either mode.** A 10% phase bias — the regime training is in once
it has nearly converged, and the only regime in which a refinement term does anything — *lowers* the
residual. Near the truth the gradient points the wrong way. The term must not be used as a loss at any
weight. A coarse-only discrimination test passes it, which is how this was missed before.

**The global surface also collapses the coarse discrimination**, from +0.053 to +0.0016 — a factor of about
35 — and doubles the residual floor. So even the coarse ranking survives only under the target-derived
per-field surface, which is circular by construction.

**Reportable position:** the forward-model consistency term is a **diagnostic** on this instrument, not a
deployable physical constraint.

## 3. The amplitude is material, and the classical reference beats unity

Background median 1.0053 across 13 fields, 1st percentile 0.54, clipping above 1.5 on 0.85% of pixels.

| variant (phase held at reference) | per_field | global |
|---|---|---|
| unity → blend 0.25 | −0.0094 | −0.0051 |
| unity → blend 0.50 | −0.0164 | −0.0097 |
| unity → reference | **−0.0215** | **−0.0151** |
| unity → 1.5× past reference | −0.0178 | −0.0160 |
| unity → inverted (2 − reference) | **+0.0438** | **+0.0027** |

The trend is monotone along the blend, the inverted amplitude is worse, so the direction is meaningful and
not a fitting artefact. The unit-amplitude (thin-phase-object) assumption was costing the forward model real
accuracy.

**But note the scale.** Under the global surface, the amplitude moves the residual (0.015) roughly **nine
times more than halving the phase does** (0.0016). The residual is dominated by amplitude and aberration
mismatch, not by phase — which is the mechanism behind §2's verdict.

Caveat that must appear in any writeup: the reference is one algorithm's reconstruction, not a measurement.

## 4. Reproducibility note

The train loader shuffles and drops the short final batch, so which 8 of 9 fields are seen changes between
runs and the absolute residual moves with them. Every *difference* is computed within a run on identical
pixels and reproduces to the third decimal. **Report differences.**

---

# 14. v2 code-fix round — 2026-09-10

Full before/after writeup lives at `docs/v2_code_changes.md`.

## Constants corrected (changes absolute reported numbers)

| constant | was | now | source |
|---|---|---|---|
| `pixel_pitch_y_um` | 0.211994 | 0.284871 | S. Park + EAAI's published 256.38 µm FOV |
| `refraction_increment_ml_per_g` | 0.185 | 0.2 | S. Park |
| `min_foreground_pixels` | 512 | 370 | = 30 µm² / (0.284871 µm)² |

Net effect: absolute **areas +34.4%**, absolute **dry masses +24.3%**. **Every relative metric unchanged** —
the factors cancel. Any checkpoint or metric from before this date is not comparable; use `CLEAN=1`.

## Bugs fixed

* **ONNX export was broken for all 13 v2 arms.** `ExportWrapper` returned `out["condition"]` unconditionally
  → `KeyError` whenever `classifier_enabled: false`. The efficiency claim depends on this command.
* **The amplitude target never reached the loss.** `trainer.py` and `evaluator.py` did not copy it into the
  target dict.
* **Circularity had no anisotropy guard.** Now warns and quantifies the bias.

## New results that need no trained model

**Boundary error → measurement error** (figure 17). One pixel of boundary error (0.285 µm, Dice 0.975) costs
**~6% of area and ~4% of dry mass**. Median mass/area error ratio **0.674**, so **dry mass is 1.48× more
robust to segmentation error than area is** — the boundary sits where the cell is thinnest. Asymmetric:
dilation saturates mass (ratio falls to 0.40 at +5 px) while erosion tracks area (flat at 0.71–0.74).
Report both curves; never average them.

**Pipeline floor** (figure 18). Spherical caps with closed-form area and integrated phase. **Two floors:**
per cell **0.04%** in dry mass; field total **−1.82%**, entirely `min_cell_area_um2: 30` discarding cells
below 3.09 µm radius. Quote the first for "the mass of this cell", the second for "the mass on this field".

## Compact decoder

68.3% of the network was two copies of one `ConvTranspose2d(1280→640)`. A shared 1×1 bottleneck
(`model.decoder_bottleneck: 256`) takes the model from **9.5981 M → 3.3604 M, a 65% reduction**.

## Loss weight corrected

The gradient ratio 0.25–0.29 is measured **at weight 1.0**; the effective ratio is `w × ratio`. Experiment B
moves from **w = 0.1 (37:1, swamped) to w = 1.0 (3.7:1)**, with `w_ipp_{01,03,10,30}.yaml` bracketing it.
`check_gradient_path.py` now prints the arithmetic table, a multi-batch median, and the **cosine** with the
segmentation gradient (measured −0.20 to +0.63, highly batch-dependent — which is why one batch cannot set a
weight).

## New arms

**B′ (`b1_image_volume.yaml`) was the missing control** and is not optional: if B beats A, an image-level
phase-volume term would also be "measurement-aware", so only B vs B′ isolates the words *per-cell*.
Also new: B″ (+per-cell area), D0 (amplitude only), W×4 (weight sweep), KA/KB (compact decoder).

## Still open

* The per-field aberration ramp is unexplained — **not** the demodulation remainder (carrier explains 10.9%).
* Label circularity — plan is manual labels on 25–40 fields for **evaluation only**.
* No trained v2 arm exists yet.

---

# 15. Membrane channel: registration solved, labels still provisional (2026-09-14)

Data: `data/membrane/` — **800** `<stem>_membrane.tif`, one per field, 2048×2048 uint16. The full dataset.

## The geometry, measured (this is the reusable result)

The membrane is **not** the same field of view or sampling as the phase. Phase is 900×900 (centre crop of a
1024 hologram); membrane is 2048×2048 over a **larger** physical area. No crop or resize alone aligns them.

Registration by scale + translation search, **with both channels detrended first** (essential — the phase
carries tens of radians of smooth aberration and the membrane a bright illumination envelope; leaving either
in lets the search match the two *envelopes* and report a confident alignment unrelated to the specimen):

| | value |
|---|---|
| scale | 1.32–1.34 membrane px per phase px, winning **13/13 fields** |
| mean xcorr at that scale | **0.277** |
| mean xcorr, "same FOV" hypotheses (f = 2.0, 2.2756) | 0.11 — excluded, not merely worse |
| offset (at f = 1.343769) | **dy = 314 ± 3.0 px, dx = 280 ± 5.7 px — CONSTANT** |
| xcorr after refinement | median 0.443, min 0.344 |

### The header's "wrong" pitch was the fluorescence camera's pitch

The preferred scale is indistinguishable from `0.284871 / 0.211994 = 1.343769`, and **0.211994 µm is exactly
the second pitch field in the phase `.bin` header** — the one S. Park said to disregard. It was never a wrong
y-pitch for the phase. It is the membrane camera's pixel pitch, and it was in the header all along.

At that scale membrane 2048 px span 434 µm against the phase's 256 µm. Illuminated fraction of the warped
900×900 window: median 0.909.

## The labels are NOT solved, and the reason is biological

Membrane stain marks the **perimeter**, not the interior. Thresholding returns rings; filling them fails
wherever a lamellipodium leaves one open, so the mask leaks into background. Confirmed by a falsification
test — a cell cannot carry negative optical path, so ask what the pixels a threshold *adds* are worth:

| threshold pct | foreground | Dice vs Otsu | mean phase of added pixels |
|---|---|---|---|
| 60 | 41.3% | 0.521 | **−0.187** background |
| 70 | 30.9% | 0.519 | **−0.039** background |
| 75 | 25.9% | 0.542 | +0.083 signal |
| 80 | 20.8% | 0.545 | +0.246 |

Crossing is between 70 and 75 → `threshold_percentile: 75` is the loosest defensible value. Otsu foreground
for reference: 19.0%.

### Numbers I measured and then had to withdraw

An earlier naive Otsu-on-log membrane mask gave field-total **area +160%** and **dry mass −49%** versus the
Otsu labels. The mass *dropping* while area grows is the tell: post-background-subtraction phase is negative
in background, so the extra area was background, not cell periphery. **Those bias figures are not
established** and must not be quoted.

## Independence must be guarded

A membrane mask that uses the phase to decide its **shape** is not independent and reintroduces exactly the
circularity it exists to break. The mask is derived from the membrane alone; the phase enters only through
the marginal-phase diagnostic, which calibrates **one scalar** and is reported rather than applied silently.

## Still the gold standard

Manual annotation on **25–40 fields, for evaluation only**. Now much cheaper: the annotator traces on a
membrane image already aligned to the phase grid.

## Pretrained weights

`weights/mobilenet_v2-b0353104.pth` is now loaded via `model.pretrained_dir: weights`, searched **before**
the download cache, so the proxy block no longer silently forces random initialisation. The encoder logs
`weights=local (weights)` vs `weights=download cache or random` — check that line before trusting a run.

---

# 16. Complete technical documentation — 2026-09-14

`docs/HoloQPI_Technical_Documentation.pdf` (108 pages) and its source `.md` + `docs/figures/*.svg`.

Describes **only what the code does**, derived by reading the repository file by file, and deliberately
contains **no experimental results**. Numbers in it are configuration defaults, tensor shapes, parameter
counts and calibration constants only. Where a docstring quotes a measurement to justify a design decision,
that is reported as a *claim made in the source*, attributed, and not as a verified result.

26 sections plus **Appendix A**, the full resolved-configuration reference: all **230** leaf keys of
`config/base.yaml`, each with its default and a one-line definition.

## §23 — claimed vs actual (nine discrepancies, documented not fixed)

1. The composite-loss docstring omits three terms that are implemented.
2. A dataset comment says connected components; the config selects watershed.
3. `ForwardModelMetrics`' docstring describes a bounded correlation residual that the `l2` code path does
   not produce.
4. Two different SSIM implementations — Gaussian window in the loss, uniform box in the metric — under one name.
5. `deploy.onnx.output_names` is inert; the exporter derives names from the heads that exist.
6. `model.in_channels` does not size the encoder stem, which is always rebuilt for one channel.
7. `model.lora.train_input_stem` is a no-op in the current ordering (it runs before injection).
8. The "800 fields" pretrain wording does not match what the code loads.
9. `loss.forward_model.criterion: correlation` aliases `l2`.

## §24.1 — a regression introduced by the `--tag` edit, since fixed

In `main.py evaluate`, the `save_per_cell(evaluation["unmatched"], ...)` call sat inside the `if args.tag:`
block while its filename carried no tag. A plain `evaluate` wrote no `unmatched_<split>.csv` at all, and a
tagged one wrote the membrane result under the Otsu result's name. Figure 15 reads that file.

---

# 17. The v2 study: what it produced — 2026-09-15

800 fields, 113 test fields, 3158 reference cells, 60 epochs per arm, ImageNet weights confirmed loaded from
`weights/` on every run. **12 of 13 arms trained and evaluated.** Source: `runs/RESULTS.md`, assembled by
`scripts/collect_results.py` from `runs/*/metrics_test.json` — nothing typed by hand.

These are the first real results in this project. Everything in `runs/` before 2026-09-14 was a `QUICK=1`
smoke run.

## 1. The central hypothesis is REFUTED, and cleanly

| comparison | Δ dry-mass MAPE | 2× seed SD | verdict |
|---|---|---|---|
| B vs A (per-cell mass term) | **+0.0291** | 0.0070 | resolved — **worse** |
| B′ vs A (image-level volume) | **+0.0113** | 0.0078 | resolved — **worse** |

The weight sweep says the same thing monotonically:

| weight on `cell_integrated_phase` | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| dry-mass MAPE | 0.1732 | 0.1792 | 0.2076 | 0.2081 |
| Dice | 0.8330 | 0.8310 | 0.8258 | 0.8134 |

w = 0.1 is nominally better than A (−0.0053) but that is below the 0.0070 resolution threshold.

**This is the result, not a failure to get one.** A falsified hypothesis with a dose–response curve and a
significance rule behind it, on 800 fields with seed replication — which the v1 round could not produce for
any of its 54 comparisons.

## 2. The efficiency claim holds, and is the strongest positive result

| | params | GMACs | ONNX fp16 latency | fps | dry-mass MAPE |
|---|---|---|---|---|---|
| full decoder (A) | 9,598,099 | 45.85 | 8.46 ms | 118 | 0.1784 |
| compact decoder (KA) | **3,360,403** | **24.03** | 8.00 ms | 125 | 0.1808 |

−65% parameters, −48% MACs, and ΔMAPE = +0.0024 against a 0.0070 resolution threshold: no accuracy
difference distinguishable from seed noise. Note the latency gain is only 5% — at 900×900 the network is
bandwidth-bound, not compute-bound, which is worth saying plainly rather than implying 65% fewer parameters
buys 65% less time.

## 3. The learned model beats classical reconstruction

| | classical off-axis | learned (arm A) |
|---|---|---|
| dry-mass MAPE | 0.2240 | **0.1784** |
| Dice | 0.7886 | **0.8333** |
| phase Pearson r | 0.7580 | — |
| detection recall | 0.5807 | 0.5953 |

The classical **in-line** pipeline recovers −0.065 rad of in-cell contrast, i.e. none. Expected physics, and
it should be reported as the classical in-line baseline failing on this data.

## 4. Detection is the bottleneck, not measurement

Over IoU-matched cells: dry-mass MAPE 0.178, Pearson r 0.932, Bland–Altman bias −3.9%. But recall is 0.595 —
1880 of 3158 cells — so the coverage-adjusted MAPE is **0.511**. Both numbers must be quoted together.

## 5. The membrane labels do NOT resolve the circularity question

| | Otsu labels | membrane labels |
|---|---|---|
| Dice | 0.8333 | 0.5033 |
| detection recall | 0.5953 | 0.1119 |
| matched cells | 1880 / 3158 | 401 / 3584 |
| **reference area, median** | **412 µm²** | **631 µm²** |
| predicted area, median | 427 µm² | 466 µm² |
| matched IoU, median | 0.86 | 0.58 |
| dry-mass MAPE, matched | 0.1784 | **0.1676** |

The membrane reference cells are **53% larger in area**. The predicted areas barely move — same model — and
the median matched IoU lands at 0.58, just above the 0.5 matching threshold, so most pairs fall below it and
are dropped. That, not a segmentation failure, is what produces recall 0.11.

`scripts/prepare_membrane.py` predicted this in its own docstring. The 800-field run confirms the threshold
sits on the edge of its own falsification test — `marginal_phase_rad` median **+0.0235**, barely positive —
with `foreground_fraction` 0.254.

**Do not report Dice 0.5033 as the model's segmentation accuracy.** The correct statement is that the two
label sets disagree with each other by about as much as the model disagrees with either, so the circularity
question is still open and needs manual annotation on 25–40 fields — which is now much cheaper, because the
registration itself *is* verified.

One real positive: over the 401 cells that do match, dry-mass MAPE is 0.1676 — no worse than against the
Otsu labels. The measurement chain is robust to the label source even where the segmentation agreement is not.

## 6. The two label-free results stand whatever the arms do

- **Boundary error → measurement error.** Median mass/area error ratio 0.708: dry mass is 1.41× more robust
  to a boundary error than projected area. Figure 17.
- **Pipeline floor against exact analytic ground truth.** Per-cell dry mass 0.041%, per-cell area 0.245%,
  field-total dry mass −1.67%. Two floors for two different claims. Figure 18.

## What did not run

**Arm D1 crashed at epoch 1** — `torch.linalg.solve` in `ForwardModelConsistency._fit_components` received A
as Half (autocast) and B as Float. Fixed 2026-09-15: the solve is forced to float32 regardless of autocast,
which also restores the 1e-6 ridge that fp16 (eps ≈ 1e-3) had rendered inert. D1 is a diagnostic, not a loss
ablation, so no conclusion above depends on it.

**14 of 20 figures were not regenerated.** `make_figures.py` resolves a run directory through
`cfg.experiment_name`, and stage 9 ran it with `config/base.yaml` → `runs/base_off_axis/`, a v1 directory
that does not exist on the study machine. Fixed: stage 9 now runs twice, `base.yaml` for the label-free
figures (10, 16–20) and an arm config for the per-arm ones (1, 2, 3, 4, 6, 8, 9, 12, 15), plus
`diagnose_bias.py` for figure 3.

Out of scope by design: figures 5 and 13 (off-axis vs in-line — no Gabor arm, deferred to study 2), 7
(superseded by figure 19), 11 (`classifier_enabled: false` in every v2 arm), 14 (wants a modality_comparison
JSON; the numbers are in `runs/conventional_baseline_test.csv`).

**No LoRA anywhere in this study.** `model.lora.enabled` is false in `config/base.yaml` and no v2 arm
overrides it. The parameter-efficiency result above is the decoder bottleneck, not LoRA. The project title's
"LoRA-adapted" belongs to paper 1.

## To close the gaps

```bash
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 9              # ~5 min, +9 figures
CUDA_VISIBLE_DEVICES=2 ARMS="D1" bash run_v2.sh --stage 6     # ~1.5 h
CUDA_VISIBLE_DEVICES=2 bash run_v2.sh --stage 10             # rebuild RESULTS.md
```
