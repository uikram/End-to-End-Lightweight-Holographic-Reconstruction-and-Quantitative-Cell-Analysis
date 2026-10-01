# FIGURE_MANUAL.md — every generated figure, what it is and how to rebuild it

This directory holds the **generated** figures of the manuscript: one PNG per
figure and the Python script that produced it. Nothing here is drawn by hand and
nothing here is edited after generation — if a figure needs to change, the script
changes and the figure is regenerated.

The one figure that is **not** here is the framework-overview schematic
(manuscript Figure 1). It is designed by hand and lives in `../figures/`; see
`../figures/README.md`.

---

## 0. The mapping you will need first

Manuscript figure numbers are assigned by LaTeX from the order of appearance;
`main.tex` never writes a number by hand — every reference is a `\ref`. The
script names are kept in step with those numbers, and this is the current
correspondence.

| Script / PNG | Manuscript figure | LaTeX label | Referenced from |
|---|---|---|---|
| *(hand-drawn, `../figures/`)* | Figure 1 | `fig:overview` | Sec. 5 |
| `figure_2.py` / `.png` | Figure 2 | `fig:learned-vs-classical` | Sec. 7.1 |
| `figure_3.py` / `.png` | Figure 3 | `fig:qualitative` | Sec. 7.1 |
| `figure_4.py` / `.png` | Figure 4 | `fig:ablation` | Sec. 7.2 |
| `figure_5.py` / `.png` | Figure 5 | `fig:sweep` | Sec. 7.3 |
| `figure_6.py` / `.png` | Figure 6 | `fig:agreement` | Sec. 7.4 |
| `figure_7.py` / `.png` | Figure 7 | `fig:forward` | Sec. 7.5 |
| `figure_8.py` / `.png` | Figure 8 | `fig:propagation` | Sec. 7.6 |
| `figure_9.py` / `.png` | Figure 9 | `fig:efficiency` | Sec. 7.7 |

The numbering is deliberately aligned: **`figure_N.py` produces manuscript
Figure N**. `figure_1` does not exist here because manuscript Figure 1 is the
hand-drawn framework schematic in `../figures/`.

If you reorder the results sections the numbers change by themselves and the
manuscript stays correct, but the script names will no longer match — either
rename them to match or update this table.

---

## 1. How to rebuild all of them

```bash
cd manuscript_writing/manuscript_images
python mstyle.py                 # self-check: prints the data dir and the tables it found
for f in figure_2 figure_4 figure_5 figure_6 figure_7 figure_8 figure_9; do
    python $f.py
done
```

`figure_3.py` is the exception: it runs the trained network, so it needs the
research repository and a checkpoint. See its entry below.

Requirements: Python 3.10+, `numpy`, `matplotlib`. `figure_3.py` additionally
needs the project's own environment (`torch`, `holoqpi`, `imagecodecs`).

Each script prints the file it wrote and its size. They overwrite in place, so a
rebuild is safe and repeatable.

---

## Naming

Scripts and result files use short internal codes (A, B, B1, …, KA, KB, G);
every label drawn on a figure uses the paper's descriptive names (End-to-End
Neural Baseline, +IPP (per-cell), …). The full mapping is in
`../README_figures.md`, section "Configuration names". Where this manual says
"arm A" it means the End-to-End Neural Baseline, and so on.

---

## 2. Where the data comes from

Every figure reads from `../data/`, which is a **copy** of the result artefacts
produced by the study. It is copied rather than referenced so that the
manuscript folder is self-contained and the figures can be rebuilt without the
research repository.

| File in `../data/` | Origin in the repository | Used by |
|---|---|---|
| `benchmark_results/results_*.json` | `runs/benchmark_results/`, written by `scripts/collect_benchmark_results.py collect` | Figures 2, 4, 5, 7a, 8c, 9 and every generated table |
| `per_cell_test_A.csv` | `runs/v2_baseline_off_axis/per_cell_test.csv` (seed-42 run of arm A) | Figure 6 |
| `error_propagation_summary.csv` | `runs/error_propagation_summary.csv` | Figure 8a, 8b |
| `z_calibration.json` | `runs/z_calibration.json` | Figure 7b (supplied z), 7c |
| `RESULTS.md` | `runs/RESULTS.md`, written by `scripts/collect_results.py` | Figure 7b (learned-z trajectory), Figure 8b (reported median ratio) |

**`benchmark_results/` is authoritative.** Each file lists the runs it was
built from (seed, checkpoint sha256, provenance) and a `statistics` block with
mean, sample SD and n per metric; `mstyle.arm()` and `mstyle.bench()` read it.
Arms A, B and B′ have three seeds and are plotted as mean ± SD; every other
arm is its single seed-42 run. The same files generate the manuscript tables
(`../make_tables.py`), so a figure and its table cannot disagree.

The earlier arm-A discrepancy (0.17844 over 3158 cells in one per-arm file) is
resolved by the re-evaluation: seed 42 now reads 0.1767 with 1889 of 3186 cells
matched, and the three-seed mean is 0.1763 ± 0.0024.

### To point the figures at a new set of results

Refresh `../data/` from the repository (`scripts/collect_benchmark_results.py
collect` first, then copy `runs/benchmark_results/`), then rerun the scripts. If you would rather read the
repository directly, edit one line in `mstyle.py`:

```python
DATA = ROOT / "data"          # -> DATA = Path("/path/to/project/runs")
```

Nothing else changes. No plotting code contains a path.

---

## 3. Shared style — `mstyle.py`

Every script imports `mstyle`, so typography, figure widths, colours and data
loading are defined once.

- `style()` — the manuscript rcParams: 400 dpi, DejaVu Sans, 7.5 pt base,
  recessive grid, no legend frames.
- `COL1 = 3.5`, `COL2 = 7.16` — single- and double-column widths in inches.
  All eight generated figures are double-column.
- `C` / `ORDER` — a five-slot categorical palette derived from Okabe–Ito and
  validated for colour-vision deficiency (worst adjacent-pair ΔE 11.0 under
  deuteranopia, 30.9 under tritanopia; every slot above the chroma floor and
  above 3:1 contrast against white). Hues are assigned in fixed order and never
  cycled.
- Image colour maps: `gray` for recorded intensity, `cividis` for phase
  magnitude (perceptually uniform, CVD-safe, not a rainbow), `RdBu_r` for signed
  error (two hues about a neutral midpoint).
- Data helpers: `results_tables`, `table_by_prefix`, `keyed_table`,
  `load_json`, `load_csv`, `column`, `results_table_csv`, `seed_stats`,
  `results_scalar`, `learned_z_trajectory`.

**If you change the palette, change it here.** A hue changed in one script and
not the others is the most common way a figure set stops being a set.

---

## 4. The figures, one by one

---

### `figure_2.py` → manuscript Figure 2 (`fig:learned-vs-classical`)

**What it shows.** Ten matched test metrics for the end-to-end network and for
the classical reconstruct-then-segment pipeline, off-axis (left panel) and
in-line Gabor (right panel). Both pipelines are scored by the same evaluator, on
the same 113 test fields, with the same instance labelling and the same
measurement chain.

**Why it is included.** It carries the manuscript's primary claim. The
right-hand panel carries the strongest single result in the study: the classical
pipeline fails outright on in-line holograms while the network does not.

**Data used.**
- `../data/benchmark_results/results_conventional_{off_axis,gabor}.json` — classical pipeline
- `../data/benchmark_results/results_arm_A.json` (mean of three seeds) and `results_arm_G.json` — learned

**Run.** `python figure_2.py` → writes `figure_2.png`.

**Parameters you may want to change.** The metric list and its order are the
`METRICS` table near the top of the script: each entry is a JSON key, a display
name, and whether the metric improves upward. Removing a row removes a bar pair.

**Read before using it.** Every bar is oriented so that **longer is better**;
error metrics are plotted as `1 − error` and labelled accordingly, with the raw
value printed at the bar end. The classical in-line phase Pearson *r* is
negative (−0.1359) and is drawn clipped at zero with a left-pointing marker — if
you change the metric set, check that any other negative value is handled the
same way rather than silently vanishing.

---

### `figure_3.py` → manuscript Figure 3 (`fig:qualitative`)

**What it shows.** One row per test field, six panels each: the recorded
hologram the network is given, the predicted quantitative phase, the reference
phase, the signed phase error, the predicted segmentation and the reference
segmentation.

**Why it is included.** No table conveys that the input is a fringe pattern
bearing no resemblance to a cell. The signed-error panel is the informative
one: dry mass is an integral, so a slow offset inside a cell matters far more
than zero-mean high-frequency error, and an absolute-error map hides the sign
that distinguishes them.

**Data used.** This is the only script that **requires the research
repository**: it loads a checkpoint and runs a forward pass.
- the project root (`--project`), for `holoqpi/` and `config/`
- `runs/<experiment_name>_<modality>/best_model.pt`
- `data/off_axis_hologram/`, `data/phase/`, `data/mask/`, `data/splits.json`

**Run.**
```bash
python figure_3.py --project /path/to/lightweight_qpi_segmentation \
                   --config config/v2/a_baseline.yaml \
                   --split test --rows 2
```

**Parameters.** `--project` (project root), `--config` (any `config/v2/*.yaml`
whose checkpoint exists), `--split` (`train`/`val`/`test`), `--rows` (how many
fields to show), `--out` (output filename).

**Read before using it.** Three things.
1. Colour scales are **shared across rows** so that rows are comparable; if you
   change `--rows`, confirm the shared scale still resolves the cells in every
   row rather than saturating one of them.
2. `--split test` is the honest default. Training fields look better and are not
   a result.
3. If the checkpoint or the fields are absent the script says so and exits
   without writing a partial figure. It will not quietly fall back to a
   different split.

---

### `figure_4.py` → manuscript Figure 4 (`fig:ablation`)

**What it shows.** Dry-mass MAPE (a) and segmentation Dice (b) for every trained
arm, with the between-seed standard deviation as an error bar on the three arms
replicated at seeds 42, 1337 and 2024. The shaded band is the baseline ± twice
its pooled between-seed SD — the study's resolution criterion.

**Why it is included.** The band is the whole point. Without it a reader ranks
differences that are inside seed noise. With it, the figure shows that the
measurement-aware terms move the primary metric outside the band in the *wrong*
direction, and that the compact variant stays inside it.

**Data used.**
- `../data/benchmark_results/results_arm_<ARM>.json` — mean, SD and n per arm

**Run.** `python figure_4.py` → writes `figure_4.png`.

**Parameters.** `ARMS` near the top fixes which arms appear and in what order;
the order follows the ablation logic, not the alphabet. The colour grouping
(baseline / measurement-aware / amplitude-physics / compact) is a dict just
below it.

**Read before using it.** It is a **dot-and-interval** plot, not a bar chart,
and deliberately so: the vertical axis cannot begin at zero for a metric in the
0.17–0.21 range, and truncated bars misrepresent ratios. If you convert it to
bars you reintroduce that error. Value labels sit on the far side of each point
from the baseline band so that they clear the error bars — check that after any
change to the arm list.

---

### `figure_5.py` → manuscript Figure 5 (`fig:sweep`)

**What it shows.** Segmentation Dice, phase MAE and dry-mass MAPE against the
weight on the per-cell integrated-phase term, *w* ∈ {0.1, 0.3, 1.0, 3.0}, each
against the baseline (*w* = 0) and its ±2×SD band.

**Why it is included.** It separates two very different conclusions — "the term
hurts" and "this particular weight hurts". The degradation is monotonic beyond
*w* = 0.3, which makes it a dose response rather than a tuning accident, and
that is the strongest form the negative result can take.

**Data used.**
- `../data/benchmark_results/results_arm_{W01,W03,B,W30,A}.json` (w = 1.0 is arm B, mean of three seeds with its SD)

**Run.** `python figure_5.py` → writes `figure_5.png`.

**Parameters.** `WEIGHTS` (the sweep points) and `PANELS` (which metric goes in
which panel, with its axis direction).

**Read before using it.** The point at *w* = 1.0 **is arm B** — the two configs
specify the same objective at the same seed, so the study does not train it
twice. The figure annotates this, and the annotation must survive any edit,
otherwise a reader will count arm B twice.

---

### `figure_6.py` → manuscript Figure 6 (`fig:agreement`)

**What it shows.** For every IoU-matched cell of the test split: predicted
against reference dry mass and projected area (top row), with the corresponding
relative Bland–Altman plots (bottom row).

**Why it is included.** Correlation and agreement are different claims, and only
the second matters for a measurement — two methods can correlate at *r* = 0.99
and still disagree by 30% on every cell. The limits of agreement are the number
a biologist needs in order to decide whether the method resolves the mass
differences their experiment is about, which is exactly the
measurement-readiness claim the manuscript makes.

**Data used.**
- `../data/per_cell_test_A.csv` (from `runs/v2_baseline_off_axis/per_cell_test.csv`)

**Run.** `python figure_6.py` → writes `figure_6.png`. It plots the seed-42 run
of arm A (per-cell data exist per run; the tables report the three-seed mean),
and prints the statistics it computed for that run:

```
  matched cells: 1889
  dry mass: per-cell r=0.9337 MAPE=0.1767 (n=1889)
      per-field bias=-0.0423 LoA=[-0.2532, +0.1686]  (n=113 fields)
  projected area: per-cell r=0.8986 MAPE=0.1546 (n=1889)
      per-field bias=+0.0600 LoA=[-0.1552, +0.2752]  (n=113 fields)
```

**Parameters.** `QUANTITIES` selects which measurands are plotted; any
`<key>_pred` / `<key>_ref` column pair in the CSV works.

**Read before using it.** The **unit of analysis differs between the rows**, on
purpose. Cells segmented from the same field share one acquisition scale and are
not independent observations, so the agreement panels use the **field** as the
unit, each field summarised by the median of its matched cells; the correlation
panels keep the per-cell scatter, which is what the reported per-cell Pearson
*r* describes. The relative difference is taken against the **mean of the pair**,
not against the reference. Both conventions match the project's own evaluator,
which is why the printed numbers above reproduce the manuscript's
Table 7 exactly. Change either convention and the figure will no longer agree
with the text — the printed output is the check.

---

### `figure_7.py` → manuscript Figure 7 (`fig:forward`)

**What it shows.** (a) The hologram forward-model residual for every arm,
divided by the residual obtained from the reference phase; a ratio below 1 means
the predicted field explains the recorded hologram better than the true field
does. (b) The per-epoch trajectory of the propagation distance in the arm that
makes *z* a free parameter, 34.12 → 33.39 µm. (c) Per-field agreement of the
distance recovered by an independent scan, in the two geometries.

**Why it is included.** The three panels support the study's second negative
result and keep it correctly scoped. (a) shows the residual cannot be a training
signal on this data. (b) and (c) show why a stable learned *z* must not be
reported as a measured distance: the off-axis residual carries almost no
information about *z* (per-field IQR 156.4 µm, against 6.0 µm in-line), so
stability near the initialisation reflects a weak gradient rather than recovery
of the physical distance.

**Data used.**
- `../data/benchmark_results/results_arm_<ARM>.json` — residual ratio per arm (13 trained off-axis arms)
- `../data/RESULTS.md` — the *z* trajectory
- `../data/z_calibration.json` — the independent scan, both geometries

**Run.** `python figure_7.py` → writes `figure_7.png`.

**Parameters.** The legend split in panel (a) — arms that optimise the term
versus arms that do not — is a list of arm codes near the top.

**Read before using it.** The *z* trajectory is parsed out of `RESULTS.md`, not
out of `history_v2_learned_z.json`. The copy of that history file in `../data/`
is from the **superseded pre-audit run**: it logs only
`train_forward_distance_um` and its trajectory runs away to −4.94 µm, which is
the behaviour the code audit fixed. The current run logs a top-level
`forward_distance_um` and `RESULTS.md` prints the resulting 60-value trajectory
in full. If `RESULTS.md` carries no `per-epoch trajectory:` line the script
raises with an explanation rather than plotting the wrong run.

---

### `figure_8.py` → manuscript Figure 8 (`fig:propagation`)

**What it shows.** (a) Relative error in projected area and in dry mass when the
reference masks are dilated or eroded by a known number of pixels, with the
equivalent micrometre displacement on a secondary axis. (b) The ratio of the
two — the exchange rate between a segmentation error and a measurement error.
(c) The measurement chain against exact analytic ground truth, on a log axis.

**Why it is included.** It converts "the segmentation boundary matters" from an
assertion into a number, and it licenses the claim that the reported error is
reconstruction and segmentation error rather than calibration arithmetic. The
sub-unity ratio is a physical result in its own right: a cell boundary sits
where the cell is thinnest, so the pixels a boundary error adds or removes carry
little phase.

**Data used.**
- `../data/error_propagation_summary.csv`
- `../data/benchmark_results/results_synthetic_validation.json` — the floor, mean ± SD over three sets of synthetic fields
- `../data/RESULTS.md` — the median mass/area ratio, read from the prose

**Run.** `python figure_8.py` → writes `figure_8.png`.

**Parameters.** Nothing needs changing; the shift range comes from the CSV.

**Read before using it.** Panel (b) annotates the **reported** median ratio,
read out of `RESULTS.md` by `results_scalar()` rather than recomputed. This is
deliberate: recomputing it from the CSV gives 0.708 against the reported 0.710,
a rounding and inclusion difference, and a figure that disagrees with the text
about the same number is worse than either value. If `RESULTS.md` loses that
line the script will annotate nothing rather than guess — check the annotation
is present after a data refresh.

---

### `figure_9.py` → manuscript Figure 9 (`fig:efficiency`)

**What it shows.** (a) Measured latency against parameter count for every
runtime × precision combination, full decoder versus compact, with p99 whiskers.
(b) Throughput for the same combinations. (c) The accuracy cost of the compact
decoder against the between-seed interval.

**Why it is included.** Edge suitability is a trade-off, not a score, so the
accuracy axis has to appear beside the cost axes. Panel (c) is what licenses the
recommendation: the compact model's accuracy penalty (+0.0016 dry-mass MAPE) is
smaller than the spread the same configuration produces when only the seed
changes (±0.0047).

**Data used.**
- `../data/benchmark_results/results_hardware_arm_{A,B,D0,KA,KB}.json` — timing (A and B: three sessions)
- `../data/benchmark_results/results_arm_{A,KA}.json` — accuracy for panel (c)

**Run.** `python figure_9.py` → writes `figure_9.png`.

**Parameters.** The arms and runtime × precision combinations are listed at the
top of the script (`A, B, D0, KA, KB` × four combinations = 20 points in (a)).
Panel (b) plots arms A and KA only, the same rows as Table 10.

**Read before using it.** Arms A and B are means over one benchmark session per
trained seed; D0, KA and KB are single sessions. Every session's
`p99/p50` is at most 1.19, inside the benchmark's stability limit of 1.25; the
script prints the worst value so a contaminated refresh is visible.

---

## 5. Checklist before a figure goes into the manuscript

1. `python mstyle.py` runs and reports the expected data directory and tables.
2. The script printed a file size in the tens-to-hundreds of KB — a few KB means
   it wrote an empty axis.
3. Open the PNG and look at it. The validator in `mstyle.py` checks colour, not
   layout; label collisions, clipped labels and axis overflow are only visible
   by eye.
4. Any number annotated on the figure also appears in `main.tex` or in a
   `table_*.tex`, with the same value.
5. If the results were refreshed, `scripts/collect_benchmark_results.py collect`
   reported "all benchmarks complete" and `../data/benchmark_results/` was
   replaced as a whole, then `../make_tables.py` was rerun.
