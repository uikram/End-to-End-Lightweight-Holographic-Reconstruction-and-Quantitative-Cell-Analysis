"""Tabulate every configuration scored on the same test fields.

The in-line (Gabor) arms exclude SNU_01-SNU_50, whose Gabor frames show a
different field, so their test set is 107 fields while the off-axis arms keep
all 113. For the geometry comparison and the learned/classical comparison the
off-axis baseline and both classical pipelines are therefore also scored on the
same 107 fields (run_corrected_study.sh, step 7), and this script collects them.

It refuses to write a table whose entries were not scored on one test set: every
entry must report the same phase_n_images and cells_reference.

    python scripts/summarise_common_fields.py

Parameters: config/diagnostics.yaml (common_fields block).
Output: <paths.output_root>/<common_fields.output> (CSV), printed as a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from holoqpi.config import load_config
from holoqpi.utils import write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--diagnostics", default="config/diagnostics.yaml")
    args = parser.parse_args()

    base = load_config(args.config)
    settings = load_config(args.diagnostics).common_fields
    root = Path(base.paths.output_root)
    metrics = list(settings.metrics)

    loaded, missing = {}, []
    for label, relative in settings.entries.items():
        path = root / relative
        if path.is_file():
            loaded[label] = json.loads(path.read_text())
        else:
            missing.append(f"{label}: {path}")
    if missing:
        print("MISSING -- these evaluations have not been run:")
        for item in missing:
            print("  ", item)
        return 3

    for key in ("phase_n_images", "cells_reference"):
        values = {label: m.get(key) for label, m in loaded.items()}
        if len(set(values.values())) != 1:
            print(f"NOT ONE TEST SET: {key} differs between entries: {values}")
            return 4

    rows = []
    for label, m in loaded.items():
        rows.append({"entry": label, **{k: m.get(k) for k in metrics}})

    # Mean and SD over seeds for any configuration with several seed entries.
    groups: dict[str, list[dict]] = {}
    for label, m in loaded.items():
        if "_s" in label:
            groups.setdefault(label.split("_s")[0], []).append(m)
    for name, runs in groups.items():
        if len(runs) < 2:
            continue
        mean = {"entry": f"{name}_mean"}
        sd = {"entry": f"{name}_sd"}
        for k in metrics:
            values = np.array([r.get(k) for r in runs], dtype=float)
            mean[k] = float(values.mean())
            sd[k] = float(values.std(ddof=1))
        rows += [mean, sd]

    destination = root / settings.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, destination)

    width = max(len(r["entry"]) for r in rows) + 2
    print("\n=== Common test fields: "
          f"{loaded[next(iter(loaded))].get('phase_n_images')} fields, "
          f"{loaded[next(iter(loaded))].get('cells_reference')} reference cells ===\n")
    print(f"{'metric':<28}" + "".join(f"{r['entry']:>{width}}" for r in rows))
    for k in metrics:
        cells = []
        for r in rows:
            v = r.get(k)
            cells.append(f"{v:>{width}.4f}" if isinstance(v, float) else f"{str(v):>{width}}")
        print(f"{k:<28}" + "".join(cells))
    print(f"\nWritten to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
