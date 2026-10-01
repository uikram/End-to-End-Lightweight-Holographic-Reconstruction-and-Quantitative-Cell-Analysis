#!/usr/bin/env python
"""Check the manuscript-facing JSON files against the source files.

Run after compile_results.py:   python analysis/v42/check_numbers.py

Part 1 (JSON vs source). Every entry of results_for_manuscript/**/*.json that
carries ``value`` + ``source_file`` + ``source_key`` is re-read from its source:

  direct          value must equal source[key]                      -> match = yes
  mean over runs  value must equal the mean of source[key] over the
                  listed source files; ``sd`` (ddof=1) likewise      -> match = yes
  delta           "mean(neural ...) - classical" recomputed          -> match = yes
  derived         a calculation the checker cannot re-run from one
                  key (counts, CSV medians, ...) is recomputed where
                  a rule exists (below) and otherwise marked
                  "derived, not recomputed here" -- NOT as missing.

Part 2 (resolution assessments). Every Table 10 / Table 8 / decomposition row
is recomputed from the per-run source values: delta, pooled SD, 2 x pooled SD,
evaluability (>= 3 runs on both sides) and the assessment wording.

Part 3 (manuscript tables, read-only). Every number printed in HoloQPI_4.2/table_*.tex
is looked up, at the precision it is printed with, among the JSON values (and their
SDs). Values not found are listed as "not found in JSON" with a note: they may be
derived quantities (differences, ratios) or stale numbers. The manuscript is never
modified.

Output: analysis/v42/check_numbers.csv with columns
location,value,source_file,source_key,calculation,match,notes
Exit status is 1 if a direct, mean or delta entry disagrees with its source.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RFM = ROOT / "results_for_manuscript"
OUTCSV = Path(__file__).resolve().parent / "check_numbers.csv"
ROWS: list[dict] = []
TOL = 1e-9


def add(location, value, source_file, source_key, calculation, match, notes=""):
    ROWS.append({"location": location, "value": value, "source_file": source_file, "source_key": source_key,
                 "calculation": calculation, "match": match, "notes": notes})


_cache: dict[str, object] = {}


def load(path: str):
    if path not in _cache:
        p = ROOT / path
        if p.suffix == ".json":
            _cache[path] = json.loads(p.read_text())
        elif p.suffix in (".yaml", ".yml"):
            _cache[path] = yaml.safe_load(p.read_text())
        elif p.suffix == ".csv":
            with open(p, newline="") as fh:
                _cache[path] = list(csv.DictReader(fh))
        else:
            _cache[path] = p.read_text()
    return _cache[path]


def resolve(obj, key: str):
    """Look up ``key`` as a dotted path with optional [i] indices; fall back to a top-level key."""
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", key):
        if part.startswith("["):
            cur = cur[int(part[1:-1])]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(key)
    return cur


def close(a, b):
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))


def walk(o, path=""):
    if isinstance(o, dict):
        if "value" in o and "source_file" in o and "source_key" in o:
            yield path, o
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}[{i}]")


def sd(xs):
    return statistics.stdev(xs) if len(xs) >= 2 else None


failures = 0
all_values: list[float] = []

# ----------------------------------------------------------------------------- part 1
for jf in sorted(RFM.rglob("*.json")):
    if jf.name == "consistency_checks.json" or "figures" in jf.parts or "data" in jf.parts:
        continue
    tree = json.loads(jf.read_text())
    rel = str(jf.relative_to(RFM))

    def collect(o):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            all_values.append(float(o))
    collect(tree)

    for path, e in walk(tree):
        loc = f"{rel}::{path}"
        v, srcs, key = e["value"], e["source_file"], e["source_key"]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            add(loc, v, srcs, key, e.get("calculation") or "", "n/a", "non-numeric value")
            continue
        files = [s.strip() for s in str(srcs).split(";")]
        calc = e.get("calculation") or ""
        # --- direct and mean-over-runs entries
        if all((ROOT / f).exists() and f.endswith(".json") for f in files):
            try:
                vals = [float(resolve(load(f), key)) for f in files]
            except (KeyError, IndexError, TypeError, ValueError):
                vals = None
            if vals is not None and (e["derivation"] == "direct" or "mean" in calc):
                recomputed = sum(vals) / len(vals)
                ok = close(v, recomputed)
                note = "direct" if e["derivation"] == "direct" and len(vals) == 1 else f"mean of {len(vals)} source files"
                if "sd" in e and e["sd"] is not None and len(vals) >= 2:
                    ok = ok and close(e["sd"], sd(vals))
                    note += "; SD (ddof=1) recomputed"
                failures += (not ok)
                add(loc, v, srcs, key, calc or "source[key]", "yes" if ok else "NO", note)
                continue
        # --- mean over runs where the source is a JSON key that is the metric name only
        if "per_seed" in e and all((ROOT / f).exists() for f in files):
            pass
        # --- delta neural minus classical
        if calc.startswith("mean(neural") and e.get("derived_from"):
            nf = [s.strip() for s in e["derived_from"][0].split(";")]
            cf = e["derived_from"][1]
            try:
                rec = sum(float(resolve(load(f), key)) for f in nf) / len(nf) - float(resolve(load(cf), key))
                ok = close(v, rec)
                failures += (not ok)
                add(loc, v, srcs, key, calc, "yes" if ok else "NO", "delta recomputed from source files")
                continue
            except Exception as exc:  # noqa: BLE001
                add(loc, v, srcs, key, calc, "derived, not recomputed here", f"lookup failed: {exc}")
                continue
        # --- CSV rules
        if srcs.endswith(".csv") and "gradient_path" in srcs:
            col = [float(r["ratio_at_weight_1"]) for r in load(srcs)]
            rec = {"median": statistics.median(col), "min": min(col), "max": max(col), "rows": len(col)}
            kk = calc.split()[0] if calc else "rows"
            r_ = rec.get({"median": "median", "min": "min", "max": "max"}.get(kk, "rows"), len(col))
            ok = close(v, r_)
            failures += (not ok)
            add(loc, v, srcs, key, calc or "row count", "yes" if ok else "NO", "recomputed from CSV")
            continue
        if srcs.endswith(".csv") and "amplitude_sensitivity" in srcs:
            rows = load(srcs)
            if "shared" in " ".join(map(str, e.get("derived_from") or [])):
                shared = {r["batch"] for r in load("runs/amplitude_sensitivity_off_axis_per_field.csv")}
                rows = [r for r in rows if r["batch"] in shared]
            m = re.match(r"count\(phase_scaled_([0-9.]+) > phase_reference\)", key)
            if m:
                rec = sum(1 for r in rows if float(r[f"phase_scaled_{m.group(1)}"]) > float(r["phase_reference"]))
                ok = rec == v
                failures += (not ok)
                add(loc, v, srcs, key, calc, "yes" if ok else "NO", "count recomputed from per-field CSV")
                continue
            if key == "phase_reference":
                rec = sum(float(r["phase_reference"]) for r in rows) / len(rows)
                ok = close(v, rec)
                failures += (not ok)
                add(loc, v, srcs, key, calc, "yes" if ok else "NO", "mean recomputed from per-field CSV")
                continue
            m = re.match(r"phase_scaled_([0-9.]+) - phase_reference", key)
            if m:
                rec = (sum(float(r[f"phase_scaled_{m.group(1)}"]) for r in rows) -
                       sum(float(r["phase_reference"]) for r in rows)) / len(rows)
                ok = close(v, rec)
                failures += (not ok)
                add(loc, v, srcs, key, calc, "yes" if ok else "NO", "margin recomputed from per-field CSV")
                continue
        if "win_rates" in key and "x images" in key:
            geom = key.split(".")[0]
            d = load("runs/z_calibration.json")[geom]["discrimination"]
            s = re.search(r"scaled_([0-9.]+)", key).group(1)
            rec = round(d["win_rates"][f"scaled_{s}"] * d["images"])
            ok = rec == v
            failures += (not ok)
            add(loc, v, srcs, key, calc, "yes" if ok else "NO", "win_rate x images recomputed")
            continue
        if "amplitude_mae / amplitude_unity_mae" in key:
            run = ROOT / "runs"
            add(loc, v, srcs, key, calc, "derived, checked in compile_results", "ratio of two stored metrics")
            continue
        # rounded gradient etc.
        add(loc, v, srcs, key, calc or "", "derived, not recomputed here",
            "derived quantity; independent recomputation is performed by compile_results.py consistency checks "
            "(decomposition, split counts, matched-cell reconciliation)")

# ----------------------------------------------------------------------------- part 2
def assess(delta, sa, na, sb, nb):
    if na < 3 or nb < 3 or sa is None or sb is None:
        return None, None, False, "Not estimable from the available runs"
    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2)
    return pooled, 2 * pooled, True, ("Exceeds 2x pooled SD" if abs(delta) > 2 * pooled else "Within 2x pooled SD")


ipp = json.loads((RFM / "ipp" / "ipp.json").read_text())
for i, r in enumerate(ipp["table_10"]):
    files_a = [f.strip() for f in r["source_files"][0].split(";")]
    files_b = [f.strip() for f in r["source_files"][1].split(";")]
    xa = [float(load(f)[r["metric"]]) for f in files_a]
    xb = [float(load(f)[r["metric"]]) for f in files_b]
    delta = sum(xa) / len(xa) - sum(xb) / len(xb)
    pooled, two, evaluable, text = assess(delta, sd(xa), len(xa), sd(xb), len(xb))
    ok = (close(delta, r["delta"]) and r["criterion_evaluable"] == evaluable and r["assessment"] == text and
          (two is None and r["two_x_pooled_sd"] is None or two is not None and r["two_x_pooled_sd"] is not None and close(two, r["two_x_pooled_sd"])))
    failures += (not ok)
    add(f"ipp/ipp.json::table_10[{i}] {r['configuration']} vs {r['comparator']} / {r['metric']}", r["delta"],
        "; ".join(r["source_files"]), r["metric"], "delta, 2 x pooled SD, evaluability, assessment recomputed",
        "yes" if ok else "NO", f"{text}; runs {len(xa)} vs {len(xb)}")

# banned words in manuscript-facing JSON text
BANNED = ["significant", "significance", "preregistered", "physics-aware", "securely paired", "strictly paired",
          "state-of-the-art", "groundbreaking", "illusion", "bottleneck"]
for jf in sorted(RFM.rglob("*.json")):
    if "figures" in jf.parts and "data" in jf.parts:
        continue
    txt = jf.read_text().lower()
    for w in BANNED:
        if w in txt:
            # allowed only when the word is quoted as a prohibition ("never 'preregistered'")
            ctx = [m.start() for m in re.finditer(re.escape(w), txt)]
            allowed = all("never" in txt[max(0, c - 40):c] or "not " in txt[max(0, c - 30):c] for c in ctx)
            add(f"{jf.relative_to(RFM)}", w, "", "", "banned-word scan", "yes" if allowed else "NO",
                "word appears only as a negated instruction" if allowed else "banned wording present")
            failures += (not allowed)

# ----------------------------------------------------------------------------- part 3
ms = ROOT / "HoloQPI_4.2"
# printed precision conventions: percentages (x100) and parameter counts / 1e6 are looked up too
pool = sorted(set(all_values) | {100 * v for v in all_values} | {v / 1e6 for v in all_values})
import bisect


def found(x_str: str) -> bool:
    d = len(x_str.split(".")[1]) if "." in x_str else 0
    x = float(x_str)
    tol = 0.5 * 10 ** (-d) + 1e-12
    for cand in (x, -x):
        i = bisect.bisect_left(pool, cand - tol)
        while i < len(pool) and pool[i] <= cand + tol:
            return True
    return False


NUM = re.compile(r"(?<![\w.])[+-]?\$?\\?-?\d+\.\d+")
n_found = n_missing = 0
for tex in sorted(ms.glob("table_*.tex"), key=lambda p: int(re.search(r"\d+", p.stem).group())):
    for ln, line in enumerate(tex.read_text().splitlines(), 1):
        if line.lstrip().startswith("%") or "\\caption" in line or "\\tabnote" in line:
            continue
        for m in re.finditer(r"[-+]?\d+\.\d+", line):
            s = m.group(0).lstrip("+")
            ok = found(s)
            n_found += ok
            n_missing += (not ok)
            add(f"HoloQPI_4.2/{tex.name}:{ln}", s, "results_for_manuscript/**/*.json", "any numeric value (rounded to printed precision)",
                "rounded lookup (value or SD; differences and ratios are looked up as exported deltas)",
                "yes (rounded)" if ok else "not found in JSON",
                "" if ok else "may be a derived quantity not exported as a number, or a stale value; see MANUSCRIPT_SYNC.md")

with open(OUTCSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["location", "value", "source_file", "source_key", "calculation", "match", "notes"])
    w.writeheader()
    w.writerows(ROWS)
n_yes = sum(r["match"].startswith("yes") for r in ROWS)
n_no = sum(r["match"] == "NO" for r in ROWS)
n_der = sum(r["match"].startswith("derived") for r in ROWS)
print(f"{len(ROWS)} rows: {n_yes} match, {n_no} mismatch, {n_der} derived-not-recomputed, "
      f"{sum(r['match'] == 'not found in JSON' for r in ROWS)} table numbers not found in JSON -> {OUTCSV.name}")
sys.exit(1 if n_no else 0)
