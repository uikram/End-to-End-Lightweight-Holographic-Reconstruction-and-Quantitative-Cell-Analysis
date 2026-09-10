# HoloQPI v2 — current state, decisions, and what to run next

*Written after independently verifying the joint Claude/GPT/Gemini operating brief
against the actual repository. Every claim below that was checked against code or
data is marked **[verified]** with the measurement. Every claim still resting on
reasoning alone is marked **[unverified]**. Nothing here is asserted from prose.*

---

## 0. Verdict on the operating brief

**Endorsed, with three corrections and one substantive disagreement.** The brief's
priorities are right: the gradient path had to be checked first, independent
labels are the critical path, and multi-seed discipline is non-negotiable. Four of
its instructions were carried out and two of its factual claims did not survive
contact with the code.

| Brief's claim | Outcome |
|---|---|
| §1 gradient path must be isolated and measured | **Done. Term is connected.** No fix needed |
| §2 Option A is the safe formulation | **Already implemented.** What exists *is* Option A |
| §2 Option C (padded bbox) should be rejected | **Agreed**, not implemented |
| §3 "`binary_closing` is extensive and cannot erode" | **Wrong** for scipy at array borders — measured |
| §3 "area-filter-before-closing is the mechanism" | **Wrong** — the area filter runs *last* |
| §3b set border buffer `k` from vignetting width (~34 px) | **Disagree.** Costs 35.2% of all cells for zero benefit. Use 0 |
| §5 check if classical amplitude extraction is easy | **Done. It is trivial** — proceed with precompute |
| §7 shared pretrain checkpoint for all conditions | **Implemented** (`--init-from`) |

---

## 1. The gradient path — resolved, no fix needed

### What was measured

`scripts/check_gradient_path.py` (new). Isolates the term with
`torch.autograd.grad(..., allow_unused=True)` in fp32, on one real batch, and
compares against the segmentation loss on the same parameters.

```
probing 37 segmentation parameters
  cell_integrated_phase alone   sum |grad| 1.158e+02   max 2.37e-03   disconnected 0/37
  segmentation (Dice+CE) alone  sum |grad| 4.976e+02   max 2.07e-01   disconnected 0/37
  ratio measurement/segmentation: 2.328e-01
```

**[verified]** The term reaches the segmentation decoder, 0 of 37 parameters
disconnected, at 23% of the segmentation loss's gradient magnitude. This is the
brief's *second* decision branch, so §2's fix must **not** be applied on top.

### Why it already works, traced

`holoqpi/losses/composite.py:157` — `foreground = 1.0 - softmax(seg_logits)[:, 0:1]`.
That is the soft class probability, not a hard threshold, so gradient flows.

### Is it gameable by mask expansion? No, by construction

The integration domain is `instances`, the **reference** instance labelling
carried by the dataloader. For any pixel `q ∉ Ω_i^GT`, `∂L_i/∂M̂(q) = 0` exactly —
that pixel is never summed into any bin. A mask cannot expand outside the
reference region to capture more phase, because doing so cannot change the loss.

**The implementation already is the brief's Option A.** One cosmetic difference:
the `dx·dy` factor is omitted, which is a constant multiplying numerator and
denominator of a relative error and therefore cancels identically.

### Accepted limitation, unchanged

The term cannot correct a genuinely wrong predicted boundary, since it never sees
outside the reference region. It answers *"given roughly correct localisation,
does the physics term improve per-cell measurement"* — not *"can it fix bad
segmentation"*. That is the intended scope of Experiment B.

---

## 2. The border artifact — mechanism settled, and the brief's fix is too expensive

### 2a. Both previous mechanism explanations were wrong

**[verified]** scipy extensivity test:

```
blob touching top edge:  12 px -> 4 px after closing.  A ⊆ A•B is FALSE
1-px rim:                21 px -> 0 px.                rim does not survive
```

Closing *is* extensive on an infinite domain. `scipy.ndimage.binary_closing`
applies `border_value=0` to its erosion step, so at an array edge it is **not**
extensive and does erode. The brief's correction ("cannot erode anything") is
wrong for this implementation.

**[verified]** Stage-by-stage, in the order `masks.py` actually executes
(threshold → closing → fill_holes → border → area filter — the area filter is
**last**, not first):

| stage | NCI_01 border comps | largest | foreground |
|---|---:|---:|---:|
| 1 threshold | 6 | 1479 µm² | 19.40% |
| 2 closing | **0** | 0 | 19.05% |
| 3 fill_holes | 0 | 0 | 19.05% |
| 4 area filter | 0 | 0 | 19.00% |

Same pattern in 5 of 5 fields tested (NCI_01, NCI_02, NCI_FCCP_10uM_01, SNU_100,
T24_rotenone_500nM_50): 4–11 border components at threshold, **zero immediately
after closing**, before the area filter runs at all. So the brief's hypothesised
mechanism (area filter rejecting fragments before they merge) is ruled out by
execution order, and the closing explanation is confirmed — for a reason neither
review had: scipy's border convention, not textbook morphology.

### 2b. Disagreement: the vignetting-derived buffer

**[verified]** Background phase vs distance from edge, cells masked out, median
over 13 fields: −0.50 rad at 2 px against a −0.167 rad plateau, settling within
3 sd at **34 px**. (Note the sign — the rim is *depressed*, not elevated; an
earlier +0.09 rad figure included cell pixels and was misleading.)

**[verified]** Cost of each buffer on the real masks:

| buffer | cells kept | cells lost | % lost | foreground |
|---:|---:|---:|---:|---:|
| none (current) | 315 | 0 | 0.0% | 19.13% |
| **0 (touching)** | **315** | **0** | **0.0%** | **19.13%** |
| 4 | 236 | 79 | 25.1% | 14.16% |
| 8 | 232 | 83 | 26.3% | 13.87% |
| 34 (brief's rule) | 204 | 111 | **35.2%** | 12.20% |

Buffer 0 removes nothing because nothing touches the edge after closing. The
brief's rule would discard **more than a third of every cell in the dataset** to
suppress an artefact that provably never reaches the masks.

### 2c. What was implemented

`clear_border(buffer_size=cfg.border_buffer_px)` with **`border_buffer_px: 0`**,
placed after fill_holes and before the area filter. Documented in code as
**truncated-object exclusion** — a cell crossing the sensor boundary has a
physically incomplete area and integrated phase — which is independently valid
and does not depend on the artefact story at all.

It costs nothing today and becomes load-bearing the moment `binary_closing_px`
changes, or an external annotator (Cellpose, StarDist, a human) produces labels
without scipy's incidental border erosion.

**[verified]** Masks are byte-identical to those on disk after the change.

---

## 3. What the code now does

### The per-cell loss

`holoqpi/losses/terms.py :: CellIntegratedPhase`

```
V̂_i = Σ_{p ∈ Ω_i^GT}  M̂_seg(p) · φ̂(p)          M̂_seg = soft probability
V_i  = Σ_{p ∈ Ω_i^GT}  M_GT(p)  · φ_GT(p)
L_i  = |V̂_i − V_i| / (|V_i| + ε)                averaged over CELLS, not pixels
```

* Domains `Ω_i^GT` come from `batch["instances"]`, computed once per field by the
  configured instance method and carried through cropping and augmentation.
* Cells with `|V_i| < cell_min_reference_rad` (50.0) are skipped — a near-zero
  denominator would dominate the batch.
* Labels are re-densified after cropping (`_relabel_dense`), because a crop can
  delete whole cells and leave gaps that would otherwise allocate empty bins and
  divide the per-cell mean by the wrong count.

**[verified]** `scripts/selftest.py` asserts the cancellation this exists to
catch: two cells at +20% and −20%, image-level phase-volume scores **0.0000**,
per-cell scores **0.2000**. Plus: vanishes on an exact prediction, scales
linearly with error (10% → 0.1000), skips sub-floor cells without dividing by
zero, and is differentiable.

### Losses now off by default, each for a measured reason

| Term | Weight | Why |
|---|---:|---|
| `phase_mask_contrast` | 0.0 | **[verified]** cell−background phase gap median **1.370 rad** (range 0.748–2.000, n=13) vs a 0.1 rad hinge — 13.7×. Could not activate in 13/13 fields. Its logged value was exactly 0.00000 in every v1 run |
| `dry_mass_consistency` | 0.0 | Same per-image ratio as `phase_volume`, different penalty shape. Double-weights one constraint |
| `phase_volume` | 0.0 | Superseded by `cell_integrated_phase` — same constraint without the image-level cancellation |
| `boundary_gradient_alignment` | 0.0 | Not assumed. Ablated in Experiment C |
| `forward_model` | 0.0 | Diagnostic only. **[verified]** blind to a 10% phase rescale (+0.0002 margin, worse in 56% of fields ≈ chance) |

### Architecture

`model.classifier_enabled: false` in all v2 configs **removes the head's
parameters**, not just its loss weight. **[verified]** 9.76 M → 9.60 M. Losses,
evaluator and metrics all handle its absence without emitting a placeholder
accuracy.

### Pretrain → fine-tune

`--init-from <checkpoint>` on both `train` and `compare`. Every experiment
condition starts from the **same** pretrained checkpoint, so a difference between
conditions is attributable to the objective and not to independent pretraining
runs. The source path is recorded in each run directory as
`initialised_from.txt`.

### Experiment configs

`config/v2/{a_baseline, b_cell_ipp, c_cell_ipp_bga, d_forward_amplitude}.yaml` —
off-axis only, no classifier head, with the measurement behind every disabled
term written into each header.

---

## 4. Assumptions and open risks

1. **[unverified] Instance domains use connected components, not watershed.**
   Touching cells are integrated as one region. This makes the constraint coarser
   but never wrong — a merged region is still a legitimate domain. Evaluation
   still uses watershed, so reported per-cell metrics are unaffected. If Cellpose
   labels separate touching cells well, revisit whether the loss should use
   watershed domains too.
2. **[unverified] The 0.1 weight on `cell_integrated_phase` is a starting value.**
   The measured gradient ratio is 0.233 at weight 1.0, so at 0.1 the term
   contributes roughly 2% of the segmentation gradient. If Experiment B is null,
   check this before concluding the term does not work — rerun
   `check_gradient_path.py` and set the weight from the ratio.
3. **[unverified] `cell_min_reference_rad: 50.0` has not been tuned.** Report how
   many cells it excludes per batch during the first run.
4. **[verified but consequential] Fine-tuning on ~150–200 independent fields**
   with a 9.6 M-parameter network is thin. Pretraining on the 800 Otsu fields
   first is what makes it viable, and is why §7 is not optional.
5. **[unverified] Whether Cellpose works on this morphology.** Adherent cells with
   lamellipodia are not what generalist models are usually validated on. Pilot
   before committing to 150–200 fields.

---

## 5. What to run now

### Step 1 — confirm the gradient path on the server (2 minutes)

```
python scripts/check_gradient_path.py --config config/v2/b_cell_ipp.yaml \
    --set data.train_crop=512 data.num_workers=0 data.batch_size=2
```

Expect `disconnected 0/37` and a ratio near 0.2. **If it reports DISCONNECTED,
stop and send me the output** — that would contradict the local measurement and
nothing downstream is trustworthy until it is explained.

### Step 2 — regenerate labels with explicit border handling (2 minutes)

```
python main.py prepare --config config/base.yaml \
    --set mask_generation.overwrite_existing=true
```

Expect the mask count and foreground fraction to be **unchanged** (315 cells,
19.13% on the local subset; the server has 800 fields so absolute numbers differ,
but the before/after must match). A change means something other than the border
step moved.

### Step 3 — null-input probe on the in-line arm (10 minutes, decoupled)

```
python scripts/null_input_probe.py --config config/base.yaml --modality gabor
python scripts/null_input_probe.py --config config/base.yaml --modality off_axis
```

Runs on the **existing** trained checkpoints. Nothing else depends on it, and it
answers the one question that could change how the in-line result is written up.

### Step 4 — pretrain the shared checkpoint (about 2 hours)

```
python main.py train --config config/v2/a_baseline.yaml \
    --set experiment_name=v2_pretrain training.epochs=60
```

This is Experiment A's objective (`L_φ + L_seg`) on the **full 800 Otsu fields**,
producing the checkpoint every later condition fine-tunes from. Report its
held-out phase MAE and Dice so we can confirm pretraining converged before
anything is built on it.

### Not yet runnable — needs the independent labels

Experiments A/B/C and the Phase-2 gate all require the annotated set. Step 4
produces the checkpoint they will start from, so it is worth doing now regardless.

---

## 6. What I need back from you

1. `check_gradient_path.py` output — the full table.
2. `prepare` output — cell count and foreground before/after.
3. Both null-probe results — the `cells found` column for all three tiers, and
   the verdict line.
4. The pretraining run's final validation phase MAE, Dice, and detection recall,
   plus its `runs/v2_pretrain_off_axis/history.json`.

With those I can confirm the gradient path at full scale, confirm the label
change is inert, settle the in-line question, and set the
`cell_integrated_phase` weight from a measured gradient ratio rather than a
guess.

---

## 7. Next steps, in order

1. **Independent labels — the critical path.** Cellpose/StarDist pilot on 20–30
   fields spanning cell lines, conditions and densities; genuine expert
   correction, not passive acceptance. Then the circularity check:
   `Dice(M̂, threshold(φ̂))` must fall well below **0.93** before scaling to
   150–200 fields. If it does not, diagnose before annotating further.
2. **Amplitude precompute.** **[verified]** `reconstruct_off_axis` returns a
   complex field, so `.abs()` is a one-line extraction — the brief's "easy"
   branch applies. Store as `amplitude_reference`, never `amplitude_ground_truth`:
   it is itself a reconstruction with its own errors.
3. **Experiment A, 3 seeds**, from the shared checkpoint, on independent labels.
   Then the four gate conditions: circularity broken, recall ≥ 0.60–0.65 *reported
   against both label sets*, seed variance characterised, and headroom confirmed
   to exist.
4. **Recall diagnostic** alongside A. Missed cells skew small (**[verified]**
   median missed 80.8 µm² vs 122.7 µm² for false positives). Check whether they
   were absent from the semantic mask or lost in watershed — those need
   completely different fixes, and only a decoder-resolution finding would justify
   an architecture change.
5. **Experiment B**, then **B′** separately, then **D** (amplitude-scoped only).
6. Every comparison: mean ± SD over 3 seeds, and a difference counts only if it
   clears 2× the pooled spread.

---

## 8. Standing corrections to carry into any writeup

* The border artefact **never contaminated the ground-truth masks**. It exists in
  raw Otsu output and is removed before the labels are written. Do not write that
  the labels were contaminated.
* The mechanism is **scipy's `border_value=0` in the erosion half of
  `binary_closing`**, not the area filter, and not textbook extensivity.
* The dataset is **adherent cancer lines (NCI/SNU/T24) under drug perturbation**,
  not the RBC storage-lesion biology of paper 1. The link between the studies is
  methodological.
* α cancels from every relative dry-mass quantity. It applies only to absolute
  picograms, as +6.9%/−14.0%.
* No LoRA result exists. No amplitude result exists. The in-line arm's mechanism
  is unexplained pending the null probe.
