# What this study actually shows

Assessment of the completed run: 4 experiments x 2 modalities, 800 fields prepared,
113 test fields, 3045 reference cells, full-dataset label audit, 12 figures.

**Verdict: publishable, but not as the paper the current framing points at.** The
architecture and the physics losses were meant to be the contribution and neither
is. What you actually built is an unusually well-controlled comparison with the
diagnostic apparatus to explain its own results.

---

## The story, in the order it should be told

These are not seven separate results. They are one argument.

### 1. The unified model works

One 9.8M-parameter network takes a raw hologram and returns phase, a segmentation
and a condition label. Per matched cell, dry mass agrees with the reference at
r = 0.92 with a bias under 1%. The measured population is physiologically sane
(median 308 um^2, 171 pg, equivalent diameter 19.8 um) and the mass chain was
verified analytically against `lambda/(2*pi*alpha) * sum(phi) * dx * dy`.

```
off_axis   phase r 0.863   Dice 0.826   dry-mass MAPE 0.186   cell r 0.917
gabor      phase r 0.792   Dice 0.759   dry-mass MAPE 0.232   cell r 0.866
```

### 2. "Measurement-ready" needs three qualifications

- **The residual error is low-frequency, not noise.** The radial error spectrum
  falls six orders of magnitude from the largest scales to the pixel scale.
  Integration does not average it away; it accumulates it.
- **The error is compressive and proportional.** At 1 rad of true phase the model
  is low by 0.15 rad; at 5 rad it is low by 0.55 rad. Dense material is
  systematically under-recovered, so the heaviest cells are the least accurately
  weighed.
- **Per-cell mass accuracy is achieved partly by cancellation.** Area runs +1.3%,
  mean phase runs -5.8%, and their product is the -4.5% mass figure. Two errors
  that oppose is not the same as two errors that are small, and the cancellation
  will not transfer to a new dataset.

### 3. The system is detection-limited, not measurement-limited

61% recall at 85% precision (off-axis). Whole-field dry mass is -14.2%, of which
-10.8% is the boundary and only -3.8% the reconstruction. The phase head is not
the problem.

The misses are **size-selective**: median missed cell 81 um^2 against 308 um^2
matched; 92% of misses sit below the matched median; they carry only 20% of total
reference area. Mass-weighted quantities largely survive. *Number*-weighted
quantities — cell counts, size distributions, morphology histograms — are badly
skewed toward large cells. In a drug-response study where staurosporine drives
rounding and shrinkage, that bias points the wrong way.

Checked and ruled out: this is not a confluence artefact. Recall correlates with
field density only weakly (rho = -0.28; 0.69 sparse vs 0.59 dense).

### 4. Off-axis wins the optics, and the win is statistically established

Paired across 110 test images:

```
boundary (domain) factor    off-axis better by +0.050 [+0.026, +0.082]  p = 1e-15
in-cell phase bias          off-axis better by  0.042 rad [0.018, 0.071] p = 3e-4
```

Boundary F1 is the widest gap in the study: 0.431 vs 0.290.

### 5. ...and yet the end quantity converges. This is the headline.

Whole-field dry-mass ratio 0.863 (off-axis) vs 0.855 (Gabor); **not significant**
(p = 0.135). Different routes, same destination:

```
off_axis   boundary -10.8%  x  phase -3.8%  =  mass -14.2%
gabor      boundary -21.1%  x  phase +7.6%  =  mass -15.1%
```

Both are dominated by the same detection bottleneck. Practical reading: **for
mass-based assays the simpler in-line setup suffices. For morphometry it does
not.**

### 6. The classification reversal is the result to be most careful with

Gabor 0.832 vs off-axis 0.611. The classification-only control moves off-axis to
0.655 and *drops* Gabor to 0.788, so multi-task interference explains about a
third of the gap and no more.

The confusion matrices should give you pause. Off-axis fails in a **biologically
interpretable** way — blebbistatin collapses into control (recall 0.14), FCCP
confuses with rotenone (recall 0.24), and those two are both mitochondrial
inhibitors, which is exactly the mistake a genuine morphology classifier should
make. Gabor makes almost no structured errors: per-class recall 0.71-1.00 across
all five conditions from single images. That is better than the biology
plausibly supports.

Two live hypotheses your data cannot separate: the unfiltered in-line record
retains axial information that sideband filtering discards, or the two stacks
were acquired in different sessions and the network reads an acquisition
signature. **Do not claim Gabor is better for morphology classification.** Report
the gap, report the control, name the confound.

### 7. The physics-consistency losses do nothing — and you have measured why

Removing them moves every metric by <3.5%, most by <1%. The mechanism:

```
seg head vs ground truth            off_axis 0.822   gabor 0.751
threshold(pred phase) vs GT                  0.828           0.747
seg head vs threshold(pred phase)            0.920           0.912
```

The segmentation head agrees with a threshold of its own phase output more than
either agrees with ground truth — and thresholding the predicted phase beats the
segmentation head against ground truth outright. Because the masks are a
deterministic Otsu threshold of the phase, `sum(M * phi)` is already fully
determined by the two primary losses; a consistency term has no residual gradient
to supply.

**This is the most transferable thing in the study.** The literature is full of
papers adding physics-consistency terms and reporting gains without testing
whether the term is informative given the supervision. It also generates a
falsifiable prediction: the terms should recover their value the moment manual
annotations replace derived masks.

---

## Strongest parts

1. **The experimental control.** Shared split, architecture, objective, schedule
   and seed, with modality the only free variable. This is why your differences
   mean anything.
2. **The error attribution machinery.** Multiplicative domain x phase budget,
   detection cascade, missed-cell size distribution, amplitude-dependent bias
   curve, error spectrum. Better diagnostics than the field's norm, and what lets
   you write findings 2, 3 and 5 at all.
3. **The redundancy result with a measured mechanism.** A null with an
   explanation and a falsifiable prediction attached.
4. **Pre-empted limitations.** The label audit found a real confound (Otsu level
   varies with drug condition, p = 5e-12, eta^2 ~ 7%) before a reviewer did.
5. **A physically validated measurement chain.**

## Weakest parts

1. **No external baseline.** Nothing compares the network against conventional
   angular-spectrum reconstruction plus a threshold. Without it, Dice 0.83 has no
   reference point, and for off-axis that baseline is easy to compute and could
   beat you. Single most likely reason for a reject.
2. **Detection at 61% / 56%.** Boundary F1 0.29 for Gabor is poor by any standard.
   Dominant error source, and not yet attacked.
3. **The classification confound is unresolved.** One of the professor's seven
   axes currently rests on a result you cannot defend against a batch-artefact
   objection.
4. **"Edge-efficient" is asserted, not shown.** 45.9 GMACs at full field is not a
   small model, 12.3 ms on a datacenter GPU is not an edge measurement, and the
   ONNX rows are missing from the final benchmark entirely.
5. **Single site, single setup.** 800 fields, one microscope, three cell lines
   very unevenly represented.
6. **LoRA earns no keep.** Nothing in these results shows it contributing. If it
   is not a contribution, take it out of the title.

---

## What paper to write

Not a novel-architecture paper — the backbone is standard, the new loss terms
demonstrably do nothing, and there is no baseline showing the joint model beats a
two-stage alternative. Framed that way it gets rejected on novelty.

It *is* a good controlled-comparison paper with negative results and unusually
careful error attribution. Realistic targets: **Biomedical Optics Express**,
**Journal of Biomedical Optics**, **Optics Express**, **Scientific Reports**. A
well-written version should land in the first two.

Suggested structure:

| Section | Content |
|---|---|
| Framing | Can a lightweight network make the cheaper in-line geometry measurement-grade, and does hologram geometry matter for the biophysics you want? |
| Methods | Unified architecture, joint objective, silver labels **and their audit** (in Methods, not buried in Limitations) |
| Result 1 | The joint model works; measurement chain validated analytically |
| Result 2 | Off-axis wins the optics, with paired statistics |
| Result 3 | **But whole-field dry mass converges.** The detection bottleneck dominates the geometry. This is the contribution. |
| Result 4 | Characterising the residual: low-frequency, compressive, proportional |
| Result 5 | Physics losses inert under phase-derived supervision, with the redundancy mechanism |
| Result 6 | The classification reversal as an open observation with the confound named. Secondary. |
| Conclusion | For mass-based assays the simpler geometry suffices; for morphometry use off-axis. The bottleneck is **instance detection, not phase retrieval.** |

Working title: *"Off-axis versus in-line holography for end-to-end quantitative
phase imaging and single-cell biophysics: a controlled comparison, and why
physics-consistency losses are redundant under phase-derived supervision."*

---

## Before submission, in priority order

| # | Action | Priority |
|---|---|---|
| 1 | **Add a conventional-reconstruction baseline** — angular spectrum + Otsu on the same test split. Cheap for off-axis; the twin image should make it fail for Gabor, which is the motivation for learning. | Blocking |
| 2 | **Get acquisition metadata, or demote the classification claim.** Session/dish IDs would let a leave-one-dish-out split settle the batch question. | Blocking |
| 3 | **Fixed-threshold sensitivity check.** Rerun mask generation at a fixed level near the global mean of 0.406 rad and report whether the morphology comparisons move. Note the honest ambiguity: the drift may be downstream of real biology. | Strongly advised |
| 4 | **Bootstrap CIs on every headline modality difference.** The machinery already works — extend it to Dice, AJI and MAPE. | Strongly advised |
| 5 | **One attempt at the detection gap.** A boundary-aware loss or a lower area-filter floor. Even an unsuccessful attempt, reported, shows engagement. | Improves odds |
| 6 | **Resolve the edge claim.** Measure on real target hardware, or restate as "lightweight backbone, latency profiled on a workstation GPU". | Improves odds |

---

## Bottom line

You have more than you think, but it is not what the current framing points at.
Two findings a careful reader will value: hologram geometry is second-order to
instance detection for field-level dry mass, and physics-consistency losses are
structurally redundant when the labels derive from the regression target.

Both are deflationary. That is fine — deflationary results with a measured
mechanism are more useful to a field than another incremental architecture, and
much harder to argue with. The work left before submission is not more modelling.
It is the baseline, the confound, and the statistics.
