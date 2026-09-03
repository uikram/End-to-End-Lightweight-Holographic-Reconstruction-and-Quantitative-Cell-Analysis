# Research plan — end-to-end off-axis vs in-line Gabor study

Written after reading the old manuscript (`lightweightsegmentationui.docx`), the
professor's four reference papers, and the repository including every resolved
config and the completed `runs/` outputs.

---

## The finding that reorganises the plan

**The physics-aware losses in the code are not what the professor's reference
papers mean by "physics-aware", and that mismatch is the direct cause of the null
result.**

All four references define physics consistency as *data fidelity against the
measured hologram through the forward propagation model*, `L(H(o), i)`:

- **Huang, Chen, Liu & Ozcan, Nature Machine Intelligence 2023 (GedankenNet)** —
  "trained using a physics-consistency loss"; the loss is between the input
  holograms and the holograms numerically predicted by forward-propagating the
  network's output complex field. No ground-truth object fields are used at all.
- **Galande et al., J. Biomed. Opt. (HDPhysNet)** — "the data fidelity term
  promotes data consistency using the hologram formation model."
- **Lee, Mammadova, Barg & Jang, APL Mach. Learn. 4, 026106 (2026) /
  arXiv:2507.00482** — object-to-sensor distance treated as an implicit style in
  the diffraction pattern; inverse mapping learned from intensity measurements
  only.

The current implementation has **five physics terms and none of them touches the
raw hologram.** Every one couples predicted mask to predicted phase — both
directly supervised, and the mask is a deterministic Otsu threshold of the phase.
They are structurally redundant, which is exactly what the ablation measured
(all deltas within 3.5%, most within 1%).

### Why the old loss worked and this one does not

| | Old paper | Current implementation | What it should be |
|---|---|---|---|
| Phase | **measured input** | network output, supervised | network output, supervised |
| Mask | network output | network output, supervised | same |
| Physics term couples | learned mask ↔ **measured** phase | learned mask ↔ learned phase | learned phase ↔ **measured hologram** |
| Information source | the measured phase field | none new | the raw hologram |
| Result | 15.52% → 3.01% phase-volume error | no measurable effect | untested |

The old physics loss was informative because the phase was external evidence and
the only free variable was the boundary. Port it into a setting where both
operands are learned and mutually determined, and it has no residual gradient to
supply. Measured confirmation: Dice between the segmentation head and a threshold
of its own predicted phase is **0.920**, higher than either agrees with ground
truth (0.822 / 0.828).

**The three levels of physics constraint:**

```
L1  FORWARD MODEL     || H_z(A_hat * exp(i*phi_hat)) - I ||     <- MISSING
      uses the raw hologram as evidence                            informative

L2  MASK <-> PHASE    PMC, BGA, phase-volume                    <- carried over
      both operands already supervised                             redundant here

L3  MEASUREMENT       dry mass, projected area                  <- added
      a function of L2's operands                                  redundant
```

Only L1 introduces evidence the rest of the objective does not already use. The
raw hologram enters the network as input and never appears in the loss again.

### Practical blocker

L1 needs the sample-to-sensor distance `z`. It is nowhere in the repository — the
phase `.bin` header carries width, height and pixel pitch only. **Sunghwan
produced the reference phase, so he knows the reconstruction distance.** If it is
genuinely unavailable: estimate per field by autofocus over a `z` sweep, or make
it a learnable parameter (which is what Lee et al. do).

---

## Answers to the eleven questions

**1. What the professor is asking for.** One model, raw hologram in, phase and
segmentation out, biophysics derived from those two, on both geometries. His
sentence names three things to jointly optimise: *hologram reconstruction*,
segmentation, measurement consistency. The repo has the last two.

**2. What carries forward.** The conceptual chain carries forward completely and
is validated — boundary defines the integration domain, domain defines optical
volume, optical volume × λ/(2πα) gives dry mass. The three loss terms carry
forward as an *ablation arm*, not a working component. LoRA carries forward as an
open question: `lora.enabled: false` in every resolved config, so it has never
run in this study.

**3. What must change.** Add the forward-model term. Add a conventional
reconstruction baseline. Attack detection. Everything else is sound.

**4-5. What the repo has, and whether it meets the brief.** Substantially yes,
with three caveats found in the configs: **LoRA never enabled**, **angular-spectrum
front end never enabled** (`frontend.kind: none`), **no hologram-consistency term**.
The first two are omissions to state or fix; the third is the brief.

Note the disabled front end is an opportunity: it exists only for off-axis, so
switching it on measures how much of off-axis's advantage a network extracts
unaided versus needs handed to it analytically.

**6-7. What makes phase measurement-ready.** Not PSNR/SSIM. Four tests, three
already running:

- **In-cell vs background error.** Field-wide MAE 0.164 rad; in-cell 0.288 rad.
- **Amplitude dependence.** −0.15 rad at 1 rad true phase → −0.55 rad at 5 rad.
- **Spatial scale.** The radial error spectrum is overwhelmingly low-frequency —
  the component an integral accumulates rather than averages away.
- **Forward-model residual (new).** Once L1 exists, `||H_z(phi_hat) - I||` is a
  ground-truth-free per-image measure of physical plausibility.

Also report the decomposition, not just the product: per-cell mass bias −4.5% is
area +1.3% times mean phase −5.8%. Two errors that oppose.

**8. Fair comparison.** The hard part is already done (shared split, identical
architecture/objective/schedule/seed). Add: paired statistics with bootstrap CIs
(the arms see the same fields); keep the front end symmetric or declare it as an
asymmetric arm; split by acquisition session for anything involving the condition
label.

**9. Necessary ablations.** Forward-model term on/off (new, and the one that tests
the actual request); the three existing ones; and one that changes the
*supervision* rather than the loss — the manual-label arm, which is the
falsification test for the redundancy explanation.

**10. Figures.** Twelve already build. Add three: forward-model residual by
modality, conventional-baseline comparison, recall-vs-cell-size curve.

**11. Minimum protocol before writing.** Phases 0–3 below. Do not start the
manuscript before Phase 2 reports.

---

## The plan

### Phase 0 — Unblock (two emails, no compute)

| | Ask Sunghwan for | Unblocks |
|---|---|---|
| 0-A | **Reconstruction distance z** and the exact numerical reconstruction procedure | The forward-model loss and the conventional baseline |
| 0-B | **Acquisition metadata** — session, dish, date per field, both modalities | Whether the classification result is biology or a batch signature |
| 0-C | **Manual annotations** that exist, or agreement to annotate 50–100 fields | Turns the physics-loss explanation from hypothesis into result |

### Phase 1 — Establish the floor (~1 day, mostly classical)

Without this a reviewer can ask why a network is needed and you have no answer.

- **1-A Conventional off-axis**: FFT sideband → unwrap → Otsu → same measurement
  chain, same test split, same metrics. Expect it to be competitive; report honestly.
- **1-B Conventional Gabor**: back-propagation to the sample plane, twin-image
  limited; optionally a few Gerchberg–Saxton iterations. Expect it to fail — this
  is the quantitative motivation for learning, currently asserted rather than shown.
- **1-C Reference-phase oracle**: feed ground-truth phase to the segmentation head.
  Separates "we miss cells because the phase is imperfect" from "the segmentation
  task is hard." Cheap, and it decides where to spend Phase 3.
- **1-D Two-stage learned**: hologram→phase net, then a separate phase→mask net.
  Tests the joint-model claim directly.

### Phase 2 — Build the extension the professor asked for (~8 h, needs 0-A)

- **2-A Add the forward-model term.** Angular-spectrum propagate the predicted
  complex field to the sensor plane; penalise the residual against the measured
  hologram intensity. The propagation kernel already exists in
  `holoqpi/models/frontend.py` — it needs moving from the input path into the loss.
  New config block under `loss.physics`, new term in `losses/terms.py`.
- **2-B Four-way ablation**: none / L2+L3 only (current) / L1 only / all three,
  both modalities. This is the table the paper is built on.
- **2-C Report the forward-model residual as a metric**, not only as a loss.

### Phase 3 — Attack detection (~4 h, guided by 1-C)

At 61% / 56% recall every per-cell number is conditioned on the cells you find.

- **3-A Lower the area-filter floor** from 30 µm² and re-score. Missed cells have
  median area 81 µm² vs 308 µm² matched — check the filter isn't doing some of the
  missing itself. Free; do it first.
- **3-B Boundary-aware supervision**: three-class (background/interior/boundary)
  or a distance-transform regression head. Boundary F1 of 0.29 on Gabor is the
  weakest number in the study.
- **3-C Report recall stratified by cell size** regardless of whether 3-A/3-B
  work. Pre-empts the strongest objection: that mass agreement holds because the
  model preferentially finds large cells.

### Phase 4 — Close the indefensible results (~6 h, needs 0-B and 0-C)

- **4-A Leave-one-session-out classification.** If the Gabor advantage survives it
  is real; if it collapses you have found a batch artefact, which is also a result.
- **4-B Manual-label arm.** Retrain on human annotations, re-run the L2/L3
  ablation. Prediction: the mask↔phase terms recover value because the mask stops
  being a function of the phase.
- **4-C Fixed-threshold sensitivity.** Otsu level varies with drug condition at
  p = 5e-12. Rerun mask generation at a fixed 0.406 rad.

### Phase 5 — Decide what the paper may claim (optional)

- **5-A** Jetson benchmark, or restate as "lightweight, profiled on a workstation
  GPU". 45.9 GMACs at full field is not an edge claim.
- **5-B** Enable LoRA and run the rank sweep, or drop it from the paper. It has
  never run in this study.
- **5-C** Angular-spectrum front end on/off, off-axis only.
- **5-D** Bootstrap CIs on every headline modality difference.

---

## Where I agree and disagree with the GPT review

| Point | Position |
|---|---|
| "For mass-based assays the simpler in-line setup suffices" is too strong | **Agreed — that was my overreach.** The correct statement is narrower: the *difference between modalities* in whole-field dry mass is not significant. Both are around −15%; neither is measurement-grade in absolute terms. |
| Don't stop modelling — attack detection | **Agreed.** Phase 3 exists because of this. Add 1-C first: it tells you whether to spend the effort on the phase head or the segmentation head. |
| Don't conclude the physics loss is useless | **Agreed, and now far better supported.** The reference papers show the loss was never ported correctly. It is not that physics-aware learning fails here; the term with the information in it was never written. |
| Classification confound must be resolved | **Agreed.** One addition from the confusion matrices: off-axis fails in a biologically sensible pattern (FCCP confused with rotenone — both mitochondrial inhibitors), while Gabor is uniformly 0.71–1.00 across five drug classes from single images. The *structure* of the errors is what should make you suspicious, not just the accuracy. |
| Conventional baseline is the biggest gap | **Agreed** — jointly with the missing forward-model term, which the GPT review could not see without the code. |
| "Cannot verify the numbers without the repository" | **Fair, and now resolved.** Every figure is traced to `runs/` and the staged outputs. Three things only visible in the code: LoRA never ran, the front end never ran, no loss term touches the hologram. |

---

## What to tell the professor

The end-to-end framework is built and works: raw hologram → phase → segmentation
→ dry mass, on both geometries, with a controlled comparison and full error
attribution. Off-axis reconstructs and segments significantly better; the
whole-field dry-mass difference between geometries is *not* significant, because
both are limited by the same detection bottleneck rather than by the optics.

On the physics loss: the old formulation does not transfer, and we can say
exactly why — it constrained a learned boundary against a *measured* phase, and
in the new setting both are learned and mutually determined. The extension he
asked for needs the reconstruction-consistency term his own reference papers
define, and we have not built it yet. We need `z` from Sunghwan to build it.
