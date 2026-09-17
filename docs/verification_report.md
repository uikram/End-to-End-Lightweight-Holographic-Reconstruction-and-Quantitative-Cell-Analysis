# HoloQPI — Second Full Verification

> **HISTORICAL DOCUMENT — superseded by the 17 September 2026 final study.**
>
> This file records the project as it stood on an earlier verification round and is kept for provenance:
> it shows what was known, planned or open at that point. **Do not read it as the
> current state of the system.** Recommendations in it may already be
> implemented, and any constant, result or gap it names may have changed.
>
> Current truth, in order: `runs/RESULTS.md` for results, `README.md` for how to
> run it, `docs/documentation.md` for the method and the physical constants,
> `docs/RUNBOOK.md` for operations. Change records:
> `docs/code_audit_2026-09-16.md`, then `docs/gap_closure_2026-09-17.md`.
> 
> This is a verification record for an earlier round. The later rounds are in the
> two change records named above.

---

Date: 2026-09-02. Covers the framework at
`C:\Users\iivs\Desktop\lightweight_qpi_segmentation` after the fixes described
below were applied.

**Verdict: Mostly Ready.** The methodology is sound, the professor's brief is
fully covered, and the numbers you have are trustworthy *after* the measurement
fix from the last round. This pass found four things worth changing before the
final run — one of which was silently costing you augmentation diversity, and
one of which was making a headline weakness invisible in the metrics. None of
them invalidate what you have; all of them are now fixed in the code.

---

## 1. What I re-verified

### 1.1 Methodology — sound

The pipeline does what the brief asks. A raw hologram of either modality goes in;
a quantitative phase map, a cell segmentation and an image-level condition label
come out; per-cell morphology, projected area, optical volume and dry mass are
derived from the first two by the documented physics. Both arms share the split
file, architecture, objective, schedule and seed, so modality is genuinely the
only free variable. That is the property the whole comparison rests on and it
holds.

The dry-mass chain is correct and was verified analytically, not by inspection:
on a synthetic disc of known phase the computed mass matches
`λ/(2πα)·Σφ·dx·dy` to 1e-4 pg. The physical sanity check also passes — median
equivalent diameter 18.2 µm and median reference dry mass 171 pg are both squarely
in the literature range for adherent cancer cells.

### 1.2 The professor's requirements — all covered

| Requirement | Status |
|---|---|
| Raw hologram in, phase **and** segmentation out, one model | Yes — shared encoder, two decoders |
| Morphology, projected area, optical volume, dry mass per cell | Yes — `holoqpi/analysis/cells.py` |
| Off-axis vs Gabor on phase accuracy | Yes |
| …on segmentation performance | Yes |
| …on morphology classification accuracy | Yes (5-way drug condition) |
| …on projected-area estimation | Yes |
| …on optical-volume and dry-mass accuracy | Yes |
| …on computational efficiency | Yes |
| …on lightweight/edge suitability | Yes, with one honest caveat (§5, Optional O1) |
| Measurement-ready phase, not merely plausible | Yes — and now measured *inside cells*, which is where it matters |
| Physics loss extended to jointly optimise reconstruction, segmentation and measurement consistency | Yes — five coupling terms, two of them new |

### 1.3 Data processing — verified, with one item to state in the paper

**Alignment.** Center crop 1024 → 900, confirmed quantitatively: texture
correlation between hologram and phase is 0.48 under center crop versus 0.23
under resize, with the optimal window at S ≈ 896–904. This matches what you said
independently.

**Split leakage — clean.** The obvious risk was that consecutive indices within a
group (`NCI_01`, `NCI_02`) might be the same or overlapping field, which would
make a random stratified split leak. Pearson correlation between consecutive
phase maps:

```
NCI_01                     vs NCI_02                      r = -0.0442
NCI_Blebbistatin_5uM_01    vs NCI_Blebbistatin_5uM_02     r = +0.0264
NCI_FCCP_10uM_01           vs NCI_FCCP_10uM_02            r = +0.0344
NCI_Staurosporine_100nM_01 vs NCI_Staurosporine_100nM_02  r = -0.0257
NCI_rotenone_500nM_01      vs NCI_rotenone_500nM_02       r = -0.0221
cross-group control                                       r = -0.0247
```

Within-group correlation is indistinguishable from the cross-group control, so
consecutive indices are genuinely distinct fields. **No pixel-level leakage.**
This does not rule out same-dish or acquisition-batch effects on the image-level
condition label — see §4.4.

**Label provenance — needs a sentence in the paper, not a fix.** Masks are Otsu
thresholds of the ground-truth phase, which is also the phase head's regression
target. Two consequences, both now measured by `scripts/audit_labels.py`:

- The Otsu level is chosen per image, so the definition of "cell" moves between
  fields. On the sample I hold locally the coefficient of variation is 38%
  (0.24 → 0.81 rad). What matters is whether it moves *with condition*; on 13
  images the test is not powered (ANOVA p = 0.39), so run the audit on the full
  set. If p > 0.05 there, the classification result is not confounded by this
  route and you can say so.
- Because the mask is a deterministic function of the phase, the two supervised
  heads are not independent. I tested this directly: apply the mask-generation
  function to the *predicted* phase and compare with the segmentation head's
  output. Even on a 1-epoch smoke checkpoint the Dice between them is **0.96**.
  This is the mechanism behind your null ablation — see §4.3.

### 1.4 Evaluation validity — one real inconsistency found and fixed

The per-image / per-cell population mix-up from the last round is fixed and the
corrected numbers stand. New this round:

- **Bland–Altman was internally inconsistent.** The reported bias was a *median*
  while the limits of agreement were *mean* ± 1.96 SD. A reviewer will notice
  immediately that the interval is not centred on its own bias. Now both use the
  mean (the standard construction), with the median reported separately as a
  robust cross-check.
- **Whole-field phase error understates the error that matters.** Roughly four
  fifths of every image is background. Dry mass integrates only the inside. On
  real data from your checkpoint, field-wide MAE was 0.400 rad while in-cell MAE
  was **1.087 rad** — a 2.7× understatement, and on the synthetic control the gap
  is 11×. In-cell and background errors are now reported separately.
- **Detection recall had no metric.** `cell_count_ratio` cannot express it: a
  model that misses half the cells and invents an equal number of false positives
  scores 1.0. Your run has 61% recall, which is the single biggest weakness in
  the results, and nothing in `metrics_test.json` said so. Now
  `detection_recall`, `detection_precision`, `detection_f1` plus counts of missed
  and false-positive cells, and an `unmatched_test.csv` listing them.
- **Checkpoint selection could be gamed by under-detection.** The composite score
  combined Dice, phase Pearson and dry-mass MAPE — but MAPE is computed over
  matched cells only, so a model that finds a few large easy cells and misses the
  rest scores brilliantly. Detection F1 now carries 15% of the composite.

### 1.5 Bugs found

| # | Bug | Effect | Status |
|---|---|---|---|
| B1 | Every dataloader worker held an identical RNG | With `num_workers=4`, only **25% of crop offsets were distinct** — augmentation diversity was 1/4 of what the config asked for. Measured, not inferred. | Fixed (`_seed_worker`) |
| B2 | Bland–Altman mixed a median centre with mean-based limits | Reported bias not centred on its own limits | Fixed |
| B3 | `np.atleast_3d` on a 2-D phase array | Latent: would have iterated image *rows* as if they were images. Not reachable from the current call sites, but one refactor away. | Fixed (`_as_batch`) |
| B4 | No detection recall/precision anywhere | The dominant error mode was invisible | Fixed |

### 1.6 Missing experiments — two added

- **Head-redundancy test** (`audit_labels.py`). Turns the null ablation from an
  unexplained negative into a predicted one. Cheap, and it is the strongest thing
  you can say about the physics terms.
- **Threshold-stability test** (`audit_labels.py`). Pre-empts the obvious reviewer
  question about silver-standard labels confounding the classification result.

### 1.7 Reproducibility

Splits are written once and reused. Seeds are set. The resolved config is saved
per run. Environment capture, per-step logs and a single archive are produced by
`run_all.sh`. With B1 fixed, a rerun at the same seed is now genuinely
reproducible; before it, worker count silently changed the augmentation stream.

---

## 2. Everything I changed (already written into your folder)

| File | Change |
|---|---|
| `holoqpi/data/dataset.py` | `_seed_worker` gives each worker an independent stream |
| `holoqpi/analysis/cells.py` | `match_cells` stamps `match_iou` on every record, NaN when unmatched |
| `holoqpi/metrics/measurement.py` | Detection recall/precision/F1, missed and false-positive tracking, `unmatched_rows()`, Bland–Altman fix |
| `holoqpi/metrics/phase.py` | In-cell and background MAE and bias; `_as_batch` |
| `holoqpi/engine/evaluator.py` | Passes masks to `PhaseMetrics`; stamps stems; returns unmatched rows; detection F1 in the composite |
| `holoqpi/engine/compare.py` | Comparison table grouped by the brief's seven axes, with the new metrics |
| `main.py` | Writes `unmatched_<split>.csv` |
| `config/base.yaml` | `composite_metric.detection_f1: 0.15`, other weights rebalanced |
| `scripts/audit_labels.py` | **New** — threshold stability and head redundancy |
| `scripts/make_figures.py` | **New** — twelve research figures |
| `scripts/selftest.py` | Nine new checks covering everything above |
| `run_all.sh` | Two new steps; figures ride along in the archive |
| `requirements.txt` | `matplotlib>=3.6` |
| `README.md`, `docs/documentation.md` | Documented |

---

## 3. Prioritised fix list

### CRITICAL — none outstanding

Everything critical found in this pass is already fixed in your folder. The list
below records what they were, because they belong in the paper's methods section.

- **C1. Detection recall was unreported** — `holoqpi/metrics/measurement.py`.
  Why it mattered: per-cell measurement accuracy is conditioned on matching, so
  quoting "dry-mass MAPE 14%" without "recall 61%" beside it overstates the
  method. **Fixed.** Verify: `python scripts/selftest.py --config config/base.yaml`
  → "an empty prediction scores zero detection recall".

- **C2. Bland–Altman bias and limits had different centres** —
  `holoqpi/metrics/measurement.py`, `_bland_altman`. **Fixed.** Verify: selftest
  → "Bland-Altman bias lies inside its own limits of agreement".

### IMPORTANT

- **I1. Worker RNG duplication** — `holoqpi/data/dataset.py`, `_seed_worker`.
  What was wrong: each of your 4 workers replayed the same crop-offset and
  flip sequence, so the effective augmentation diversity was a quarter of what
  the config specified. **Fixed, but it only takes effect if you retrain.**
  Decision: your existing checkpoints remain valid and publishable — they were
  simply trained under weaker augmentation than the config claims. If you retrain
  the base arms anyway, take the fix; if not, the limitation is already written
  into `docs/documentation.md` §11. Verify:
  ```bash
  python - <<'PY'
  import random, torch, sys; sys.path.insert(0, ".")
  from torch.utils.data import Dataset, DataLoader
  from holoqpi.data.dataset import _seed_worker
  class D(Dataset):
      def __init__(s): s._rng = random.Random(1); s.augment = None
      def __len__(s): return 64
      def __getitem__(s, i): return torch.tensor([s._rng.randint(0, 388)])
  dl = DataLoader(D(), batch_size=4, num_workers=4, worker_init_fn=_seed_worker)
  v = [int(x) for b in dl for x in b]
  print(f"{len(set(v))}/{len(v)} distinct  (was 16/64 before the fix)")
  PY
  ```

- **I2. Phase error was only reported field-wide** — `holoqpi/metrics/phase.py`.
  Why it mattered: the "measurement-ready" claim rests on error *inside cells*,
  and the field-wide figure understated it by 2.7× on your own data. **Fixed.**
  Verify: selftest → section [5], and `phase_mae_rad_in_cell` appears in
  `metrics_test.json` after a rerun.

- **I3. Checkpoint selection rewarded under-detection** —
  `holoqpi/engine/evaluator.py` `composite_score`, `config/base.yaml`
  `training.composite_metric`. **Fixed.** This changes which epoch is selected, so
  it only takes effect on a retrain. If you do not retrain, say in the methods
  that the checkpoint was selected on Dice + phase Pearson + dry-mass accuracy.

- **I4. Label provenance is unstated** — not a code fix, a paper fix. Run
  `scripts/audit_labels.py` on the full dataset and quote both numbers. Verify:
  `python scripts/audit_labels.py --config config/base.yaml`.

### OPTIONAL

- **O1. ONNX on GPU.** `run_all.sh` now pins `onnxruntime-gpu==1.20.1` (the CUDA
  12 / cuDNN 9 build that matches your torch cu124) and preloads the pip-installed
  CUDA libraries so the provider actually instantiates. If it still falls back,
  the benchmark now flags it in a `comparable_to_pytorch_row` column and figure 8
  draws that point hollow. **This is cosmetic for the study** — you can report
  PyTorch latency alone and say ONNX export was verified but not timed on GPU.
  Verify: `grep -i provider logs/*/09_benchmark.log`.

- **O2. Per-image fixed threshold as a sensitivity check.** Only if I4 comes back
  with p < 0.05. Set `mask_generation.threshold_method: fixed` and
  `fixed_threshold_rad` near the global mean, rerun `prepare --force-masks`, and
  report both.

- **O3. SSIM uses the biased variance estimator.** Differs from scikit-image by a
  factor N/(N−1) per window. Sub-1% on a 7×7 window; irrelevant to any conclusion.

---

## 4. What your current results mean

### 4.1 The measurements are good; the detection is not

After the population fix, per-image dry-mass bias is **−1.2%** with image-level
Pearson **r = 0.913**, and projected area **+6.5%**, **r = 0.784**. Those are
respectable numbers for a method-agreement study — a small negative mass bias and
tight correlation means that *for the cells the model finds*, it measures them
well.

The problem is which cells it finds. Of 3,045 reference cells the model detects
2,175 and matches 1,858 — **61% recall**. Boundary F1 is 0.431. The bias
decomposition says the same thing from the other side: whole-field dry mass is
**−14.2%**, and that splits into **−10.8% from the boundary** and only **−3.8%
from the reconstruction**. The error is missing cells, not mismeasured ones.

The honest framing for the paper: *this is a measurement-capable but
detection-limited system.* Do not quote a per-cell MAPE without the recall beside
it. That is a legitimate and interesting result — it says the bottleneck for
lightweight QPI cell analysis is instance detection, not phase retrieval — and it
points cleanly at future work.

### 4.2 Off-axis reconstructs better; Gabor classifies better

Off-axis wins on phase, as physics predicts: the sideband is separable, the twin
image is not. Gabor nonetheless classifies drug condition at **0.832** against
off-axis **0.611** — a 22-point gap in the wrong direction.

The `classification_only` control moved off-axis 0.611 → 0.655 and *dropped*
Gabor 0.832 → 0.788. So multi-task interference explains part of it: off-axis is
paying more for its shared representation. But the gap only narrows from 22 to 13
points; it does not close. Two hypotheses remain, and you cannot separate them
with the data you have:

1. The Gabor encoding preserves something class-discriminative that off-axis
   reconstruction discards (defocus signature, axial information in the twin
   image).
2. The Gabor and off-axis holograms were acquired in different sessions, and the
   classifier is partly reading an acquisition batch signature rather than
   biology.

Write both down. Hypothesis 2 is testable only with acquisition metadata; if you
can get session or dish identifiers from the acquisition group, a
leave-one-dish-out split settles it. Absent that, report the gap, report the
control, and name the confound. That is a stronger paper than an unexplained
result presented as a finding.

### 4.3 The physics terms are inert — and now you know why

At epoch 60 the objective is phase 40.0%, classification 31.6%, segmentation
25.2%, and **all five physics terms together 3.1%**. `phase_mask_contrast` is
exactly 0.00000 from epoch 5 onward — its hinge is always satisfied. Removing the
terms changes nothing measurable.

Until now the best you could say was "they didn't help." The redundancy test gives
you the mechanism. In the RBC baseline the phase was a *measured input* and only
the boundary was learned, so a term coupling phase to mask supplied information
Dice lacked. Here both phase and mask are *directly supervised*, and the mask is a
deterministic Otsu threshold of the phase — so Σ(M ⊙ φ) is a deterministic
function of quantities the two primary losses are already fitting. There is no
residual gradient for a consistency term to supply. Thresholding the predicted
phase reproduces the segmentation head at Dice 0.96, which is that statement
measured.

**This is a publishable negative result with a mechanism**, which is worth
considerably more than a null. It also generates a concrete prediction: physics
coupling terms should recover their value the moment the mask stops being derived
from the phase — i.e. with manual annotations. Say so.

### 4.4 What the leakage check does and does not establish

It establishes that consecutive indices are distinct fields, so the random
stratified split does not put the same image on both sides. It does not establish
that images from the same dish are absent from both sides. Since the
classification label is *per image* and dish-level effects are exactly what
hypothesis 2 in §4.2 is about, this is the one open methodological question. Say
so plainly rather than letting a reviewer find it.

---

## 5. Exactly what to run

Everything is in one script. On the server:

```bash
cd /sda/usama/QPI_Extended
conda activate qpi_extended
pip install matplotlib
bash run_all.sh
```

That is the whole thing. It takes roughly 30–60 minutes with existing
checkpoints, writes every step's log under `logs/<timestamp>/`, keeps going when
a step fails, and ends by printing the single archive to send back.

If you prefer to run the steps yourself, this is what the script does:

| # | Command | What it does | Where the output lands | Success looks like |
|---|---|---|---|---|
| 0 | *(automatic)* | Captures python, torch, GPU, onnxruntime providers, package list | `logs/<ts>/00_environment.log` | `cuda available : True` |
| 1 | `python scripts/selftest.py --config config/base.yaml` | Verifies the measurement chain, the new detection and Bland–Altman metrics, in-cell phase errors, and every loss term | `logs/<ts>/01_selftest.log` | `All checks passed.` |
| 2–5 | `python main.py compare --config config/<...>.yaml --no-train` | Re-evaluates all four experiments with the new metrics, using existing checkpoints | `runs/*/metrics_test.json`, `per_cell_test.csv`, **`unmatched_test.csv`**, `*_modality_comparison.csv` | `comparison table written to ...` and `detection_recall` present in the JSON |
| 6 | `python scripts/diagnose_bias.py --config config/base.yaml` | Splits the dry-mass error into boundary and reconstruction | `runs/*/bias_diagnosis_test.csv` | Prints `domain x phase` equal to the mass ratio |
| 6b | `python scripts/audit_labels.py --config config/base.yaml` | Threshold stability by condition; segmentation/phase head redundancy | `runs/label_audit_*.csv/json` | Prints an ANOVA p-value and three Dice columns |
| 7–8 | `pip install onnxruntime-gpu==1.20.1` then a provider check | Tries to get ONNX onto the GPU | `logs/<ts>/07`, `08` | `CUDA usable: True` (optional) |
| 9 | `python main.py benchmark --config config/base.yaml` | Latency, memory, GMACs, PyTorch and ONNX | `runs/base_hardware_benchmark_cuda.csv` | `comparable_to_pytorch_row` is `True` |
| 10–11 | *(automatic)* | Concatenates every table and traces convergence | `logs/<ts>/10`, `11` | Non-empty |
| 12 | `python scripts/make_figures.py --config config/base.yaml` | Builds all twelve figures | `figures/fig*.png` and `.pdf` | Twelve PNGs listed |

Then send back `logs/holoqpi_run_<timestamp>.tar.gz` — the figures are inside it.

**One thing to decide before you start.** If you want the augmentation fix (I1)
and the detection-aware checkpoint selection (I3) reflected in the results, the
base arms need retraining: roughly 1 hour per arm, so about 2 hours for off-axis
and Gabor, or 8 hours if you redo the ablations too. If you do, replace step 2
with `python main.py compare --config config/base.yaml` (no `--no-train`). If you
would rather keep the results you have — which is entirely defensible for a study
paper — run exactly as written above and the limitation is already documented.

---

## 6. Figures

Twelve, built by `scripts/make_figures.py`. Full rationale for each is in
`docs/documentation.md` §10a; the short version:

**Essential (8):** qualitative panel showing what an off-axis record looks like
next to a Gabor record on the same field; Bland–Altman agreement for dry mass and
area; the boundary-versus-reconstruction error decomposition; the detection
cascade with the size distribution of missed cells; the full head-to-head
comparison on one page; the loss composition over training that proves the
physics terms were never in a position to matter; the ablation deltas with a
visible zero line; and the accuracy-versus-latency trade-off.

**Important (3):** the label audit (threshold drift and head redundancy);
confusion matrices for both arms, which is the only way to interpret the
classification paradox; and convergence trajectories showing the heads plateau at
different epochs.

**Optional (1):** phase error structure — error binned by phase amplitude and the
radial error spectrum. Valuable because dry mass is an integral and therefore
cares about low-frequency error specifically, but it is a supplementary figure.

`python scripts/make_figures.py --list` prints this at the terminal. Any figure
whose inputs are missing prints its reason and is skipped; the rest still build.

---

## 7. Execution plan

1. `pip install matplotlib` on the server.
2. Decide: retrain the base arms (2 h, picks up I1 and I3) or keep the current
   checkpoints (0 h, limitations already documented). Either is defensible.
3. `bash run_all.sh`.
4. Open `figures/fig10_label_audit.png` first — the ANOVA p-value there decides
   whether O2 is needed.
5. Open `figures/fig04_detection_gap.png` second — it is the story the paper
   should lead the limitations section with.
6. Send back `logs/holoqpi_run_<timestamp>.tar.gz`.
7. I do the final end-to-end review against the real numbers.
