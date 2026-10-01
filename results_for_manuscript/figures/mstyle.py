"""Shared style, palette and data loading for every manuscript figure.

WHY THIS FILE EXISTS
--------------------
Every figure script imports from here rather than defining its own colours,
fonts or file paths. Two consequences:

1. Typography and sizing are identical across the figure set, which is what a
   two-column paper needs.
2. When the experiments are re-run, only `DATA` has to point somewhere else and
   every figure regenerates unchanged. No plotting code needs editing.

DATA PROVENANCE
---------------
The single source of every plotted number is `runs/benchmark_results/`, written
by `scripts/collect_benchmark_results.py` (a `data/benchmark_results/` copy
beside this folder takes precedence if present).
It holds one JSON file per arm and per benchmark; each file lists every run it
was built from (seed, checkpoint sha256, provenance status) and a `statistics`
block with mean, sample SD and n per metric. `arm()` and `bench()` below read
it. The same files generate the manuscript tables (`../make_tables.py`), so a
figure and its table cannot disagree.

Reporting convention, identical to the tables: arms A, B and B' were trained at
three seeds and are plotted as the mean with the between-seed SD; every other
arm was trained once and is plotted as that run.

Three inputs are not in the benchmark files because they are per-cell or
per-epoch series rather than summary statistics: `per_cell_test_A.csv` (seed-42
run of arm A, Figure 6), `z_calibration.json` (the per-field z scan, Figure 7c)
and the learned-z trajectory in `RESULTS.md` (Figure 7b).

COLOUR
------
The categorical palette is an Okabe-Ito-derived ordered set, validated for
colour-vision deficiency: worst adjacent-pair separation is Delta-E 11.0
(deuteranopia) and 30.9 (tritanopia), with every slot above the chroma floor and
above 3:1 contrast against white. Hues are assigned in fixed order and never
cycled; where more than four categories appear, a single hue plus direct value
labels is used instead of inventing more hues.

Image colour maps: `gray` for recorded intensity (field convention, monotonic
luminance), `cividis` for phase magnitude (perceptually uniform and explicitly
CVD-safe, not a rainbow), and `RdBu_r` for signed error (two hues about a
neutral midpoint, which is the correct encoding for a quantity whose sign
matters).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # HoloQPI_4.2/
# HoloQPI_4.2 lives in <repository>/Claude outputs/HoloQPI_4.2/, so the result
# files are in <repository>/runs. Override with the environment variable
# HOLOQPI_RUNS if the folder is somewhere else.
import os as _os
DATA = Path(_os.environ.get("HOLOQPI_RUNS", HERE.parents[2] / "runs"))
OUT = ROOT / "manuscript_images"         # figures are written into the manuscript

# --------------------------------------------------------------------------
# Palette — validated, fixed order, never cycled
# --------------------------------------------------------------------------
C = {
    "blue":    "#0072B2",   # slot 1 — the learned model / arm A
    "orange":  "#D55E00",   # slot 2 — the classical baseline
    "green":   "#009E73",   # slot 3 — compact variant
    "purple":  "#882255",   # slot 4 — amplitude / physics arms
    "gold":    "#DDAA33",   # slot 5 — only with direct value labels
}
ORDER = [C["blue"], C["orange"], C["green"], C["purple"], C["gold"]]

INK = "#1a1a1a"          # primary text
INK2 = "#555555"         # secondary text
GRID = "#d8d8d8"         # recessive grid
NEUTRAL = "#9a9a9a"

CMAP_INTENSITY = "gray"
CMAP_PHASE = "cividis"
CMAP_SIGNED = "RdBu_r"

# Column widths for a standard two-column article, in inches.
COL1 = 3.5               # single column
COL2 = 7.16              # full width (double column)


def style():
    """Apply the manuscript-wide rcParams. Call once at the top of a script."""
    plt.rcParams.update({
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 8.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name):
    """Write a figure and report where it went."""
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {path.name}  ({size_kb:.0f} KB)")
    return path


# --------------------------------------------------------------------------
# RESULTS.md parsing
# --------------------------------------------------------------------------
def _clean(cell: str) -> str:
    return cell.replace("**", "").strip()


def results_tables(path: Path | None = None) -> dict[str, list[list[str]]]:
    """Parse every pipe-delimited table out of RESULTS.md.

    Returns {section heading: [[cells], ...]} with the separator row dropped.
    Markdown is a fragile interchange format, so this is deliberately tolerant:
    it keys on the '## ' heading above each table and ignores anything that is
    not a table row.
    """
    path = path or (DATA / "RESULTS.md")
    tables: dict[str, list[list[str]]] = {}
    heading = None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            continue
        if not line.startswith("|") or heading is None:
            continue
        cells = [_clean(c) for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):      # separator row
            continue
        tables.setdefault(heading, []).append(cells)
    return tables


def table_by_prefix(prefix: str) -> list[list[str]]:
    """The first RESULTS.md table whose heading starts with `prefix`."""
    for heading, rows in results_tables().items():
        if heading.startswith(prefix):
            return rows
    raise KeyError(f"no RESULTS.md table with heading starting {prefix!r}")


def keyed_table(prefix: str) -> tuple[list[str], dict[str, list[str]]]:
    """A RESULTS.md table as (column headers, {row label: row cells})."""
    rows = table_by_prefix(prefix)
    header = rows[0]
    body = {r[0]: r[1:] for r in rows[1:]}
    return header[1:], body


def num(value: str) -> float:
    """Parse a RESULTS.md cell to float; '--' and junk become NaN."""
    value = _clean(value).replace("+", "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------
# Direct artefact loaders
# --------------------------------------------------------------------------
def load_json(name: str):
    return json.loads((DATA / name).read_text())


def load_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def results_table_csv() -> dict[str, dict[str, float]]:
    """runs/results_table.csv keyed by arm code."""
    out = {}
    for row in load_csv("results_table.csv"):
        arm = row["arm"]
        out[arm] = {k: (v if k in ("arm", "label", "comparator") else num(v))
                    for k, v in row.items()}
    return out


def seed_stats(metric: str, modality: str = "off_axis") -> dict[str, dict]:
    """{experiment_name: {mean, sd, runs, values}} from seed_aggregate.json."""
    doc = load_json("seed_aggregate.json")
    block = doc["modalities"][modality]
    return {exp: stats[metric] for exp, stats in block.items() if metric in stats}


# A copy in data/ if there is one; otherwise <project>/runs/benchmark_results,
# with this folder either at <project>/paper/ (public repository) or at
# <project>/Claude outputs/manuscript_writing/ (working copy).
_CANDIDATES = (DATA / "benchmark_results",
               ROOT.parent / "runs" / "benchmark_results",
               ROOT.parents[1] / "runs" / "benchmark_results")
BENCH = next((c for c in _CANDIDATES if c.is_dir()), _CANDIDATES[0])


def bench(name: str) -> dict:
    """One collected benchmark file, e.g. bench("arm_A"), bench("hardware_arm_KA")."""
    return json.loads((BENCH / f"results_{name}.json").read_text())


def arm(code: str, metric: str, kind: str = "arm") -> dict:
    """{mean, sd, n} of one metric for one arm (sd = 0.0 when n == 1).

    `kind` selects the file family: "arm" (evaluation), "hardware_arm".
    """
    s = bench(f"{kind}_{code}")["statistics"][metric]
    sd = s.get("std")
    return {"mean": s["mean"], "sd": float(sd) if sd else 0.0, "n": int(s.get("n", 1))}


def benchmark_rows(source: str) -> list[dict]:
    """Rows from one hardware-benchmark JSON, off-axis only."""
    doc = load_json(source)
    rows = doc if isinstance(doc, list) else doc.get("rows", doc.get("results", []))
    return [r for r in rows if r.get("modality") in (None, "off_axis")]


def results_scalar(pattern: str, group: int = 1) -> float | None:
    """Pull one number out of RESULTS.md's prose by regex.

    Some headline quantities are stated in the text of the label-free section
    rather than in a table. Reading them from RESULTS.md rather than recomputing
    them guarantees that a figure and the manuscript cannot disagree about the
    same number through a rounding or inclusion difference.
    """
    text = (DATA / "RESULTS.md").read_text()
    match = re.search(pattern, text)
    return float(match.group(group)) if match else None


def learned_z_trajectory() -> np.ndarray:
    """Per-epoch propagation distance for the learned-z arm, from RESULTS.md.

    WHY RESULTS.md AND NOT history.json. An earlier copy of
    `runs/v2_learned_z_off_axis/history.json` was from a superseded pre-audit
    run. The current copy (data/history_v2_learned_z.json, identical to
    runs/v2_learned_z_off_axis/history.json) logs a top-level
    `forward_distance_um`, and the audit of 2026-09-26 confirmed that its 60
    values agree with the trajectory RESULTS.md prints to within 0.005 um
    (RESULTS.md rounds to 0.01 um).

    Reading the trajectory out of RESULTS.md therefore guarantees the figure
    shows the reported run. If a future RESULTS.md is regenerated from a history
    that carries `forward_distance_um`, this still works unchanged.
    """
    text = (DATA / "RESULTS.md").read_text()
    match = re.search(r"per-epoch trajectory:\s*([+\-0-9.,\s]+)", text)
    if not match:
        raise RuntimeError(
            "RESULTS.md carries no 'per-epoch trajectory:' line. Regenerate it "
            "with scripts/collect_results.py from a run whose history.json has "
            "a top-level forward_distance_um."
        )
    values = [v.strip() for v in match.group(1).split(",") if v.strip()]
    return np.asarray([float(v) for v in values], dtype=float)


if __name__ == "__main__":
    style()
    print("mstyle self-check")
    print(f"  data dir            {DATA}")
    print(f"  RESULTS.md tables   {list(results_tables())}")
    print(f"  arms in CSV         {sorted(results_table_csv())}")
    print(f"  z trajectory epochs {len(learned_z_trajectory())}")
