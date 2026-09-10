"""Pool repeated runs and decide which ablation differences are real.

WHY THIS EXISTS
---------------
The physics ablations differ by less than 1% on every reconstruction and
measurement metric, and by up to 5 points on classification -- but in opposite
directions between the two modalities. A single run cannot tell a 5-point effect
from a 5-point coincidence, and with 113 test fields over five classes the
standard error on accuracy alone is around 4.5 points. Until the same
configuration has been trained more than once, "the physics terms are null" is an
assertion, and so is any claim that one of them helped.

This script reads every `<experiment>[_s<seed>]_modality_comparison.json` under
the output root, groups them by experiment and modality, and reports mean and
spread across seeds. It then compares each ablation against the base arm using
the seed spread as the yardstick, which is the only honest way to read a table
whose differences are this small.

THE RULE IT APPLIES
-------------------
A difference counts as resolved only when it exceeds the pooled seed spread by
the factor in `evaluation.seed_replication.resolve_factor`. Everything else is
reported as "within seed noise" -- which for this study is the expected and
publishable outcome for the physics terms, not a disappointment.

    python scripts/aggregate_seeds.py --config config/base.yaml
    python scripts/aggregate_seeds.py --config config/base.yaml --metrics seg_dice cls_accuracy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.utils import get_logger, write_csv, write_json

LOGGER = get_logger(__name__)

_SEED_SUFFIX = re.compile(r"^(?P<stem>.+?)_s(?P<seed>\d+)$")


def group_runs(output_root: Path) -> dict[str, dict[int, dict]]:
    """{experiment: {seed: payload}} from the comparison JSONs on disk."""
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(output_root.glob("*_modality_comparison.json")):
        name = path.name.replace("_modality_comparison.json", "")
        match = _SEED_SUFFIX.match(name)
        if match:
            experiment, seed = match.group("stem"), int(match.group("seed"))
        else:
            experiment, seed = name, -1        # -1 marks the original run
        try:
            grouped[experiment][seed] = json.loads(path.read_text())
        except Exception as exc:
            LOGGER.warning("could not read %s (%s)", path, exc)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--baseline", default="base",
                        help="experiment the others are compared against")
    parser.add_argument("--modality", nargs="+", default=["off_axis", "gabor"])
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="default: evaluation.seed_replication.metrics")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    settings = cfg.evaluation.seed_replication
    metrics = args.metrics or list(settings.metrics)
    factor = float(settings.resolve_factor)
    output_root = Path(cfg.paths.output_root)

    grouped = group_runs(output_root)
    if not grouped:
        raise SystemExit(f"no *_modality_comparison.json under {output_root}")

    counts = {name: len(seeds) for name, seeds in grouped.items()}
    print("\n=== runs found ===")
    for name in sorted(counts):
        seeds = sorted(s for s in grouped[name] if s >= 0)
        label = ", ".join(str(s) for s in seeds) or "-"
        print(f"  {name:<26} {counts[name]} run(s)   seeds: {label or 'original only'}")

    single = [n for n, c in counts.items() if c < 2]
    if single:
        print(f"\n  NOTE: {len(single)} experiment(s) have a single run, so no spread can be")
        print("  estimated for them. Their rows below carry no error bar and any")
        print("  difference involving them is unresolved by construction.")

    payload: dict = {"resolve_factor": factor, "modalities": {}}
    rows: list[dict] = []

    for modality in args.modality:
        print(f"\n{'=' * 74}\n {modality}\n{'=' * 74}")
        pooled: dict[str, dict[str, tuple[float, float, int]]] = {}
        for experiment, by_seed in grouped.items():
            for metric in metrics:
                values = []
                for payload_one in by_seed.values():
                    entry = (payload_one.get(modality) or {})
                    value = entry.get(metric)
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        values.append(float(value))
                if values:
                    array = np.asarray(values)
                    pooled.setdefault(experiment, {})[metric] = (
                        float(array.mean()),
                        float(array.std(ddof=1)) if array.size > 1 else float("nan"),
                        int(array.size),
                    )

        base = pooled.get(args.baseline, {})
        others = [e for e in sorted(pooled) if e != args.baseline]

        for metric in metrics:
            if metric not in base:
                continue
            base_mean, base_sd, base_n = base[metric]
            print(f"\n  {metric}")
            print(f"    {args.baseline:<26} {base_mean:8.4f}"
                  + (f" +/- {base_sd:.4f}  (n={base_n})" if base_n > 1 else f"  (n={base_n})"))
            for experiment in others:
                if metric not in pooled[experiment]:
                    continue
                mean, sd, n = pooled[experiment][metric]
                difference = mean - base_mean
                # The yardstick is the spread of the two arms combined; where an
                # arm has only one run its spread is unknown and the comparison
                # cannot be resolved at all.
                spreads = [s for s in (base_sd, sd) if np.isfinite(s)]
                if spreads:
                    noise = float(np.sqrt(np.sum(np.square(spreads))))
                    resolved = abs(difference) > factor * noise and noise > 0
                    verdict = "RESOLVED" if resolved else "within seed noise"
                    band = f"+/- {sd:.4f}" if np.isfinite(sd) else "        "
                    print(f"    {experiment:<26} {mean:8.4f} {band}  (n={n})"
                          f"   d={difference:+.4f}   {verdict}")
                else:
                    verdict = "unresolved (single run)"
                    print(f"    {experiment:<26} {mean:8.4f}           (n={n})"
                          f"   d={difference:+.4f}   {verdict}")
                    noise = float("nan")
                rows.append({
                    "modality": modality, "metric": metric, "experiment": experiment,
                    "mean": mean, "sd": sd, "runs": n,
                    "baseline_mean": base_mean, "baseline_sd": base_sd,
                    "difference": difference, "noise": noise, "verdict": verdict,
                })
        payload["modalities"][modality] = {
            experiment: {m: {"mean": v[0], "sd": v[1], "runs": v[2]}
                         for m, v in values.items()}
            for experiment, values in pooled.items()
        }

    destination = Path(args.out) if args.out else output_root / "seed_aggregate.json"
    write_json(payload, destination)
    table = write_csv(rows, output_root / "seed_aggregate.csv") if rows else None

    resolved = [r for r in rows if r["verdict"] == "RESOLVED"]
    print(f"\n{'=' * 74}")
    if not rows:
        print(" Nothing to compare.")
    elif not resolved:
        print(" NOTHING IS RESOLVED. Every ablation difference sits inside the spread")
        print(" between repeated runs of the same configuration. That is the result:")
        print(" the physics-aware and measurement terms do not change what this model")
        print(" measures, and the classification differences seen in any single run")
        print(" are seed noise. Report the table with error bars and say so.")
    else:
        print(f" {len(resolved)} difference(s) exceed {factor:g}x the seed spread:")
        for row in resolved:
            print(f"   {row['modality']:<9} {row['metric']:<22} {row['experiment']:<24}"
                  f" d={row['difference']:+.4f}")
        print(" Everything else is within seed noise.")
    print(f"\n  -> {destination}" + (f"\n  -> {table}" if table else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
