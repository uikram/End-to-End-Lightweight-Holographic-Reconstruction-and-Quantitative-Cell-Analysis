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

WHERE IT READS FROM, AND WHY THAT CHANGED
-----------------------------------------
It reads `runs/<experiment>_<modality>/metrics_<split>.json`, which is what
`main.py evaluate` writes -- the same place `collect_results.py` looks.

It used to glob `*_modality_comparison.json` at the top of the output root
instead. That file is written by exactly one code path, `main.py compare`, which
this study never runs for a seed replicate: `run_v2.sh` trains and evaluates the
seed arms with `main.py train` / `main.py evaluate`, so no comparison JSON is
ever produced for them. The script was therefore structurally unable to see the
runs it exists to pool, and with its default baseline absent it compared nothing
and exited 0 -- printing "NOTHING IS RESOLVED", which reads as a measured
finding and was in fact a statement about an empty table.

Two naming conventions are recognised, because the project has used both:
`v2_<arm>_seed<N>` (run_v2.sh) and `<arm>_s<N>` (run_study.sh). The old regex
matched only the second, so even a comparison JSON from a v2 seed run would have
been parsed as a separate experiment with no replicates.

THE RULE IT APPLIES
-------------------
A difference counts as resolved only when it exceeds the pooled between-seed
standard deviation by the factor in
`evaluation.seed_replication.resolve_factor`, and only when BOTH arms have at
least two runs. Everything else is reported as unresolved -- which for this
study is the expected and publishable outcome for the physics terms, not a
disappointment.

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
from holoqpi.utils import (
    get_logger,
    pooled_between_seed_sd,
    write_csv,
    write_json,
)

LOGGER = get_logger(__name__)

# Both conventions this project has used for a seed replicate's run directory.
# `_seed<N>` is run_v2.sh's; `_s<N>` is run_study.sh's.
_SEED_SUFFIXES = (
    re.compile(r"^(?P<stem>.+?)_seed(?P<seed>\d+)$"),
    re.compile(r"^(?P<stem>.+?)_s(?P<seed>\d+)$"),
)

# run_v2.sh names a seed replicate after the ARM CODE (v2_A_seed1337) while the
# unreplicated run is named after the arm's experiment_name (v2_baseline), so the
# two do not group together by string alone. This maps one onto the other.
_ARM_CODE_TO_EXPERIMENT = {
    "v2_A": "v2_baseline",
    "v2_B": "v2_cell_ipp",
    "v2_B1": "v2_image_volume",
    "v2_B2": "v2_cell_ipp_area",
    "v2_C": "v2_cell_ipp_bga",
    "v2_D0": "v2_amplitude",
    "v2_D1": "v2_forward_amplitude",
    "v2_D2": "v2_learned_z",
    "v2_KA": "v2_compact_baseline",
    "v2_KB": "v2_compact_cell_ipp",
    "v2_L": "v2_lora",
}


def _split_seed(name: str) -> tuple[str, int | None]:
    for pattern in _SEED_SUFFIXES:
        match = pattern.match(name)
        if match:
            return match.group("stem"), int(match.group("seed"))
    return name, None


def study_experiments() -> set[str]:
    """The experiment names that belong to this study's arm set.

    Taken from ``collect_results.ARMS``, so the two documents describe the same
    experiment. Without this filter the run root also yields the classical
    reconstruction baseline (``conventional_*``) and the v1 ablations
    (``classification_only``, ``no_measurement``), and they were being pooled and
    differenced against arm A as though they were ablations of it -- a classical
    reconstruction is not an arm of the network and its phase MAE is not
    comparable in that column.
    """
    try:
        from collect_results import ARMS
    except Exception:                                  # pragma: no cover
        return set()
    names = set()
    for _, config_path, _, _ in ARMS:
        path = Path(config_path)
        if path.is_file():
            try:
                names.add(load_config(str(path)).experiment_name)
            except Exception as exc:
                LOGGER.warning("could not read %s (%s)", path, exc)
    return names


def group_runs(
    output_root: Path, modality: str, split: str, default_seed: int,
    keep: set[str] | None = None,
) -> dict[str, dict[int, dict]]:
    """``{experiment: {seed: metrics}}`` from the run directories on disk.

    The unreplicated run is filed under ``default_seed`` (``project.seed``)
    rather than a sentinel, so it counts as one of the seeds instead of being a
    special case the arithmetic has to remember.
    """
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    suffix = f"_{modality}"
    for path in sorted(output_root.glob(f"*{suffix}/metrics_{split}.json")):
        directory = path.parent.name
        if not directory.endswith(suffix):
            continue
        stem, seed = _split_seed(directory[: -len(suffix)])
        experiment = _ARM_CODE_TO_EXPERIMENT.get(stem, stem)
        if keep is not None and experiment not in keep:
            continue
        try:
            grouped[experiment][default_seed if seed is None else seed] = json.loads(
                path.read_text()
            )
        except Exception as exc:
            LOGGER.warning("could not read %s (%s)", path, exc)
    return grouped


#: The significance rule, from holoqpi.utils so that this script, the figures
#: and the results tables cannot disagree about it. See
#: ``pooled_between_seed_sd`` for what the three previous implementations were.
pooled_sd = pooled_between_seed_sd


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--baseline", default=None,
                        help="experiment the others are compared against; "
                             "default: arm A's experiment_name")
    parser.add_argument("--modality", nargs="+", default=None,
                        help="default: data.modality")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="default: evaluation.seed_replication.metrics")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--all-runs", action="store_true",
        help="pool every run directory, not only this study's arms. Off by "
             "default: the run root also holds the classical baseline and the v1 "
             "ablations, which are not arms of this network.",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    settings = cfg.evaluation.seed_replication
    metrics = args.metrics or list(settings.metrics)
    factor = float(settings.resolve_factor)
    output_root = Path(cfg.paths.output_root)
    # The v2 arms are single-modality, so asking for gabor by default produced a
    # whole empty section under a heading that looked like a result.
    modalities = args.modality or [cfg.data.modality]

    # The baseline is arm A, resolved from its own config rather than from a
    # string literal. The old default was "base", which no v2 run is ever named,
    # so every metric was skipped and the script reported an empty table as a
    # null result.
    baseline = args.baseline
    if baseline is None:
        arm_a = Path("config/v2/a_baseline.yaml")
        baseline = (
            load_config(str(arm_a)).experiment_name if arm_a.is_file() else "v2_baseline"
        )
    keep = None if args.all_runs else (study_experiments() or None)
    print(f"\nbaseline arm: {baseline}   split: {args.split}   "
          f"resolve factor: {factor:g}x the pooled between-seed SD")
    if keep is not None:
        print(f"restricted to the study's {len(keep)} arms "
              f"(pass --all-runs to pool every run directory)")

    payload: dict = {
        "resolve_factor": factor,
        "baseline": baseline,
        "split": args.split,
        "modalities": {},
    }
    rows: list[dict] = []
    absent_baseline: list[str] = []

    for modality in modalities:
        grouped = group_runs(
            output_root, modality, args.split, int(cfg.project.seed), keep=keep
        )
        print(f"\n{'=' * 74}\n {modality}\n{'=' * 74}")
        if not grouped:
            print(f"  no runs/*_{modality}/metrics_{args.split}.json found")
            continue

        counts = {name: len(seeds) for name, seeds in grouped.items()}
        print("  runs found:")
        for name in sorted(counts):
            seeds = ", ".join(str(s) for s in sorted(grouped[name]))
            print(f"    {name:<26} {counts[name]} run(s)   seeds: {seeds}")

        single = sorted(n for n, c in counts.items() if c < 2)
        if single:
            print(f"\n  {len(single)} arm(s) have a single run, so no spread can be")
            print("  estimated for them and every comparison involving one is")
            print(f"  unresolvable by construction: {', '.join(single)}")
            print("  Fill this in with:  SEEDS=\"1337 2024\" bash run_v2.sh --stage 6")

        # Raw per-seed values, kept as arrays so the pooled SD can be computed
        # from the two populations rather than from two summary numbers.
        samples: dict[str, dict[str, np.ndarray]] = {}
        for experiment, by_seed in grouped.items():
            for metric in metrics:
                values = [
                    float(by_seed[seed][metric])
                    for seed in sorted(by_seed)
                    if isinstance(by_seed[seed].get(metric), (int, float))
                    and np.isfinite(by_seed[seed][metric])
                ]
                if values:
                    samples.setdefault(experiment, {})[metric] = np.asarray(values)

        if baseline not in samples:
            absent_baseline.append(modality)
            print(f"\n  THE BASELINE ARM {baseline!r} HAS NO {args.split} METRICS for "
                  f"{modality}, so nothing can be compared against it.")
            print(f"  Arms present: {', '.join(sorted(samples)) or 'none'}")
            print("  This is a missing input, NOT a finding that no difference exists.")
            continue

        base = samples[baseline]
        others = [e for e in sorted(samples) if e != baseline]

        for metric in metrics:
            if metric not in base:
                continue
            base_values = base[metric]
            base_mean = float(base_values.mean())
            base_sd = (
                float(base_values.std(ddof=1)) if base_values.size > 1 else float("nan")
            )
            print(f"\n  {metric}")
            print(f"    {baseline:<26} {base_mean:8.4f}"
                  + (f" +/- {base_sd:.4f}  (n={base_values.size})"
                     if base_values.size > 1 else f"  (n={base_values.size})"))
            for experiment in others:
                if metric not in samples[experiment]:
                    continue
                values = samples[experiment][metric]
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
                difference = mean - base_mean

                noise, reason = pooled_sd(base_values, values)
                if reason is not None:
                    verdict = f"unresolvable ({reason})"
                else:
                    verdict = (
                        "RESOLVED" if abs(difference) > factor * noise
                        else "within seed noise"
                    )
                band = f"+/- {sd:.4f}" if np.isfinite(sd) else "          "
                print(f"    {experiment:<26} {mean:8.4f} {band}  (n={values.size})"
                      f"   d={difference:+.4f}   {verdict}")
                rows.append({
                    "modality": modality, "metric": metric, "experiment": experiment,
                    "mean": mean, "sd": sd, "runs": int(values.size),
                    "baseline": baseline,
                    "baseline_mean": base_mean, "baseline_sd": base_sd,
                    "baseline_runs": int(base_values.size),
                    "difference": difference, "pooled_sd": noise,
                    "threshold": factor * noise if np.isfinite(noise) else float("nan"),
                    "verdict": verdict,
                })
        payload["modalities"][modality] = {
            experiment: {
                metric: {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                    "runs": int(values.size),
                    "values": [float(v) for v in values],
                }
                for metric, values in by_metric.items()
            }
            for experiment, by_metric in samples.items()
        }

    destination = Path(args.out) if args.out else output_root / "seed_aggregate.json"
    write_json(payload, destination)
    table = write_csv(rows, output_root / "seed_aggregate.csv") if rows else None

    resolved = [r for r in rows if r["verdict"] == "RESOLVED"]
    unresolvable = [r for r in rows if r["verdict"].startswith("unresolvable")]
    print(f"\n{'=' * 74}")
    if absent_baseline:
        print(f" NOT RUN for {', '.join(absent_baseline)}: the baseline arm "
              f"{baseline!r} has no metrics there.")
        print(" Train and evaluate it before reading anything into this file.")
    if not rows:
        print(" Nothing was compared. That is a missing input, not a null result:")
        print(f" no arm other than the baseline had {args.split} metrics for the")
        print(" requested modality. Check that stage 6 finished.")
    elif unresolvable and not resolved and len(unresolvable) == len(rows):
        print(f" NO COMPARISON IS RESOLVABLE: all {len(rows)} of them lack seed")
        print(" replication on one or both arms. This is NOT the same statement as")
        print(" 'no difference exists' -- the yardstick does not exist yet.")
        print(' Run:  SEEDS="1337 2024" bash run_v2.sh --stage 6')
    elif not resolved:
        comparable = len(rows) - len(unresolvable)
        print(f" NOTHING IS RESOLVED among the {comparable} comparison(s) that had a")
        print(" yardstick. Every one of those differences sits inside the spread between")
        print(" repeated runs of the same configuration. That is the result: the")
        print(" physics-aware and measurement terms do not change what this model")
        print(" measures, and a difference seen in any single run is seed noise.")
        print(" Report the table with error bars and say so.")
        if unresolvable:
            print(f" The other {len(unresolvable)} comparison(s) had no replication and are")
            print(" reported as unresolvable rather than folded into the statement above.")
    else:
        print(f" {len(resolved)} difference(s) exceed {factor:g}x the pooled seed SD:")
        for row in resolved:
            print(f"   {row['modality']:<9} {row['metric']:<22} {row['experiment']:<24}"
                  f" d={row['difference']:+.4f}  (threshold {row['threshold']:.4f})")
        print(f" Of the rest, {len(rows) - len(resolved) - len(unresolvable)} are within")
        print(f" seed noise and {len(unresolvable)} have no replication to judge them by.")
    print(f"\n  -> {destination}" + (f"\n  -> {table}" if table else ""))
    # A missing baseline is a hard input error, not a verdict: run_v2.sh treats a
    # traceback-free non-zero code as a finding, so this uses the reserved
    # hard-error code 3 to make sure it is counted as a failure instead.
    return 3 if absent_baseline else 0


if __name__ == "__main__":
    sys.exit(main())
