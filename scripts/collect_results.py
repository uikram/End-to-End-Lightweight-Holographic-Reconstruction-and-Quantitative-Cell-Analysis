"""Turn a directory of finished runs into the tables the paper needs.

WHY THIS EXISTS
---------------
Training writes `runs/<experiment>_<modality>/metrics_test.json`, one file per
arm, thirty-five keys each. Reading thirteen of those by hand and typing the
differences into a table is where transcription errors come from, and it is
also where a difference that is not real gets promoted to a finding because
nobody computed the spread it has to clear.

So this does the whole last mile: it reads every run, assembles the three
tables, applies the significance rule, folds in the two label-free analyses,
and writes both a machine-readable CSV and a paper-ready Markdown document.

THE SIGNIFICANCE RULE, APPLIED RATHER THAN DESCRIBED
----------------------------------------------------
A difference between two arms counts ONLY if it exceeds
`evaluation.seed_replication.resolve_factor` times the pooled between-seed
standard deviation of the same metric. That rule exists because the v1
replication round resolved 0 of 54 comparisons by it: the arm-to-arm
differences were smaller than the run-to-run noise, and reporting them as
findings would have been reporting seed noise.

The factor and the pooled SD both come from one place --
`holoqpi.utils.pooled_between_seed_sd` and the config key above -- so this
document, `scripts/aggregate_seeds.py` and figure 19 cannot disagree about the
verdict. They previously used two different formulas, sqrt(v_a + v_b) and
sqrt(0.5 (v_a + v_b)), which differ by sqrt(2) for equal n.

This script refuses to mark anything significant when fewer than two seeds
exist for EITHER arm, and says why in the verdict column rather than implying a
null result. Run `SEEDS="1337 2024" bash run_v2.sh` to make the column mean
something.

WHAT IT WILL NOT DO
-------------------
It will not rank arms by a metric that rewards missing cells. Every per-cell
error here is over IoU-matched pairs, so a selective detector scores well on
it; the coverage-adjusted column and the detection recall are printed
alongside every time, and the Markdown says why.

    python scripts/collect_results.py --config config/base.yaml
    python scripts/collect_results.py --config config/base.yaml --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoqpi.config import load_config, parse_overrides
from holoqpi.utils import get_logger, pooled_between_seed_sd, write_csv

LOGGER = get_logger(__name__)

# arm code -> (config file, human label, what it adds over its comparator)
ARMS = [
    ("A",   "config/v2/a_baseline.yaml",          "A  baseline",              None),
    ("B",   "config/v2/b_cell_ipp.yaml",          "B  + per-cell mass",       "A"),
    ("B1",  "config/v2/b1_image_volume.yaml",     "B' + image-level volume",  "A"),
    ("B2",  "config/v2/b2_cell_area.yaml",        "B\" + per-cell area",      "B"),
    ("C",   "config/v2/c_cell_ipp_bga.yaml",      "C  + boundary gradient",   "B"),
    ("D0",  "config/v2/d0_amplitude.yaml",        "D0 + amplitude",           "B"),
    ("D1",  "config/v2/d_forward_amplitude.yaml", "D1 + forward model",       "D0"),
    ("W01", "config/v2/w_ipp_01.yaml",            "w = 0.1",                  "A"),
    ("W03", "config/v2/w_ipp_03.yaml",            "w = 0.3",                  "A"),
    # w = 1.0 is arm B's objective exactly, same seed and all. Training it twice
    # would burn a GPU-day to reproduce a number we already have, so the sweep
    # borrows B's result for that row and says so.
    ("W10", "config/v2/w_ipp_10.yaml",            "w = 1.0 (= arm B)",        "A"),
    ("W30", "config/v2/w_ipp_30.yaml",            "w = 3.0",                  "A"),
    ("KA",  "config/v2/k_compact_a.yaml",         "KA compact baseline",      "A"),
    ("KB",  "config/v2/k_compact_b.yaml",         "KB compact + per-cell",    "B"),
    # D2 is D1 with the propagation distance free. It is off-axis like every
    # other row here, so it belongs in this table; arm G is Gabor and does not,
    # because a single-modality table cannot mix acquisition geometries -- its
    # numbers live in runs/v2_baseline_modality_comparison.json and figure 5.
    ("D2",  "config/v2/d2_learned_z.yaml",        "D2 + forward model, z free", "D1"),
    # L is arm A with LoRA adaptation of the encoder. It is listed so that IF it
    # is trained its numbers reach the tables; when it is not, it simply appears
    # under "not yet run" like any other untrained arm. Whether the study claims
    # LoRA at all is a write-up decision -- see config/v2/l_lora.yaml.
    ("L",   "config/v2/l_lora.yaml",              "L  + LoRA encoder",        "A"),
]

# The metrics that go in the headline table, and which direction is better.
HEADLINE = [
    ("dry_mass_mape",                   "mass MAPE",        "lower"),
    ("dry_mass_mape_coverage_adjusted", "mass MAPE (cov)",  "lower"),
    ("area_mape",                       "area MAPE",        "lower"),
    ("detection_recall",                "recall",           "higher"),
    ("detection_precision",             "precision",        "higher"),
    ("seg_dice",                        "Dice",             "higher"),
    ("phase_mae_rad",                   "phase MAE",        "lower"),
]

MEASUREMENT = [
    ("dry_mass_mape",                   "dry mass MAPE, matched cells"),
    ("dry_mass_mape_coverage_adjusted", "dry mass MAPE, coverage-adjusted"),
    ("dry_mass_mape_ci_lower",          "  bootstrap CI lower (fields)"),
    ("dry_mass_mape_ci_upper",          "  bootstrap CI upper (fields)"),
    ("dry_mass_cell_pearson_r",         "dry mass Pearson r, per cell"),
    ("dry_mass_relative_bias",          "dry mass Bland-Altman bias"),
    ("dry_mass_loa_lower",              "  limit of agreement, lower"),
    ("dry_mass_loa_upper",              "  limit of agreement, upper"),
    ("dry_mass_field_total_bias",       "dry mass, field total bias"),
    ("area_mape",                       "projected area MAPE"),
    ("area_cell_pearson_r",             "projected area Pearson r, per cell"),
    ("area_relative_bias",              "projected area Bland-Altman bias"),
    ("area_field_total_bias",           "projected area, field total bias"),
    # THE TWO QUANTITIES THE BRIEF ASKS FOR THAT THIS TABLE DID NOT CARRY.
    #
    # The study reports projected area, circularity, integrated phase, optical
    # volume and dry mass. Optical volume and circularity were computed by the
    # evaluator and written into every metrics JSON, and then appeared in no
    # table -- so two of the five requested measurements were absent from the
    # document the manuscript is assembled from.
    ("optical_volume_mape",             "optical volume MAPE"),
    ("optical_volume_cell_pearson_r",   "optical volume Pearson r, per cell"),
    ("optical_volume_field_total_bias", "optical volume, field total bias"),
    ("circularity_mape",                "circularity MAPE"),
    ("circularity_cell_pearson_r",      "circularity Pearson r, per cell"),
    ("circularity_relative_bias",       "circularity Bland-Altman bias"),
    ("coverage",                        "coverage (detection recall)"),
    ("cells_matched",                   "cells matched"),
    ("cells_reference",                 "cells in reference"),
]

# The physics-aware half of the study, which had no table at all. The forward
# residual, its ground-truth floor and the learned propagation distance were
# computed, written to every metrics JSON and then never surfaced -- so the one
# component the extension is named after was absent from the results document.
PHYSICS = [
    ("forward_residual",           "forward-model residual (lower is better)"),
    ("forward_residual_reference", "  the same residual from the REFERENCE phase"),
    ("forward_residual_ratio",     "  ratio: 1.0 = as good as the reference phase"),
    ("forward_distance_um",        "propagation distance z used to score [um]"),
    ("phase_mae_rad_in_cell",      "phase MAE inside cells [rad]"),
    ("phase_bias_rad_in_cell",     "phase bias inside cells [rad]"),
    ("phase_mae_rad_background",   "phase MAE in background [rad]"),
    ("phase_pearson_r",            "phase Pearson r"),
    ("seg_aji",                    "segmentation AJI"),
    ("seg_boundary_f1",            "segmentation boundary F1"),
]

# THE THIRD OUTPUT OF THE FRAMEWORK, WHICH HAD NO ROW ANYWHERE.
#
# The stated pipeline is raw hologram -> phase AND amplitude AND segmentation,
# and the amplitude head is trained in arms D0, D1 and D2 -- but until
# holoqpi/metrics/amplitude.py existed the predicted amplitude entered exactly
# one reported number, the forward-model residual, where it is entangled with
# the phase, the propagation distance and the aberration surface. An amplitude
# head that had collapsed to a constant would have left no trace in any table.
#
# These rows are blank for every arm whose amplitude head is off, which is the
# correct appearance: those arms predict no amplitude at all.
#
# Read `amplitude_mae_over_unity` first. The reference is the modulus of a
# classical reconstruction, not a measurement, so the only defensible claim is
# comparative: below 1 the head is closer to that reference than the
# thin-phase-object assumption A = 1 is, at or above 1 it is not.
AMPLITUDE = [
    ("amplitude_mae",             "amplitude MAE vs the reconstruction reference"),
    ("amplitude_unity_mae",       "  the same MAE for the thin-phase assumption A = 1"),
    ("amplitude_mae_over_unity",  "  ratio: below 1.0 = better than assuming A = 1"),
    ("amplitude_mae_in_cell",     "amplitude MAE inside cells"),
    ("amplitude_bias",            "amplitude bias"),
    ("amplitude_pearson_r",       "amplitude Pearson r"),
    ("amplitude_pred_mean",       "predicted amplitude, mean"),
    ("amplitude_pred_sd",         "predicted amplitude, sd (0 = collapsed head)"),
    ("amplitude_reference_mean",  "reference amplitude, mean"),
]


def load_metrics(root: Path, experiment: str, modality: str, split: str) -> dict | None:
    path = root / f"{experiment}_{modality}" / f"metrics_{split}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def experiment_name(config_path: Path) -> str | None:
    """Read the arm's experiment_name without instantiating anything heavy."""
    if not config_path.is_file():
        return None
    try:
        return load_config(str(config_path)).experiment_name
    except Exception as exc:                          # a broken config is not fatal here
        LOGGER.warning("could not read %s: %s", config_path, exc)
        return None


def seed_spread(root: Path, code: str, modality: str, split: str, key: str) -> list[float]:
    """Values of one metric across the seed-replication runs of one arm.

    run_v2.sh names those `v2_<code>_seed<N>`, so they are found by pattern
    rather than by being listed anywhere.
    """
    values = []
    for directory in sorted(root.glob(f"v2_{code}_seed*_{modality}")):
        path = directory / f"metrics_{split}.json"
        if path.is_file():
            value = json.loads(path.read_text()).get(key)
            if value is not None and np.isfinite(value):
                values.append(float(value))
    return values


def format_value(value, digits: int = 4) -> str:
    """Render one cell. Non-numeric values pass through as themselves.

    The benchmark payload mixes numbers with strings (experiment name, device),
    so coercing everything to float turns a perfectly good table into a
    traceback. A string is a legitimate cell here.
    """
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "--" if not np.isfinite(value) else f"{float(value):.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--modality", default=None, help="default: data.modality")
    parser.add_argument("--out", default=None,
                        help="Markdown output; default runs/RESULTS.md")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    modality = args.modality or cfg.data.modality
    root = Path(cfg.paths.output_root)
    destination = Path(args.out) if args.out else root / "RESULTS.md"

    # ---- gather -------------------------------------------------------
    found: dict[str, dict] = {}
    missing: list[str] = []
    for code, config_path, label, _ in ARMS:
        name = experiment_name(Path(config_path))
        if name is None:
            missing.append(f"{code} (config unreadable)")
            continue
        metrics = load_metrics(root, name, modality, args.split)
        if metrics is None:
            missing.append(f"{code} ({name}_{modality})")
            continue
        found[code] = metrics

    # The sweep's w = 1.0 row is arm B. Alias it rather than reporting a gap.
    aliased = False
    if "W10" not in found and "B" in found:
        found["W10"] = found["B"]
        missing = [m for m in missing if not m.startswith("W10 ")]
        aliased = True

    print(f"\n=== runs found: {len(found)} of {len(ARMS)} arms, "
          f"{modality}, {args.split} split ===")
    if aliased:
        print("  w = 1.0 taken from arm B (identical objective, so it is not trained twice)")
    if missing:
        print("  not yet run: " + ", ".join(missing))

    if not found:
        print("\n  Nothing to collect. Train first:")
        print("      CLEAN=1 nohup bash run_v2.sh > v2.out 2>&1 &")
        print("  Then re-run this. Stage 10 of run_v2.sh does it automatically.")
        return 1

    lines: list[str] = []
    rows: list[dict] = []

    lines.append(f"# HoloQPI v2 results — {modality}, {args.split} split")
    lines.append("")
    lines.append(f"Generated by `scripts/collect_results.py` from "
                 f"`{root}/*/metrics_{args.split}.json`. Every number is read from a "
                 f"file; none is typed.")
    lines.append("")
    if missing:
        lines.append(f"**Arms not yet run:** {', '.join(missing)}. Any comparison "
                     f"involving them is absent below rather than estimated.")
        lines.append("")
    if aliased:
        lines.append("The weight sweep's `w = 1.0` row is **arm B itself**: the two "
                     "configs specify the same objective at the same seed, so it is "
                     "reported once rather than trained twice.")
        lines.append("")

    # ---- Table 1: the ablation ----------------------------------------
    lines.append("## Table 1 — Ablation")
    lines.append("")
    lines.append("Each arm differs from its comparator by **one term**. The "
                 "significance column applies the project's rule: a difference "
                 "counts only if it exceeds "
                 f"**{float(cfg.evaluation.seed_replication.resolve_factor):g}x the "
                 "pooled between-seed SD** of the same metric.")
    lines.append("")
    header = "| arm | " + " | ".join(label for _, label, _ in HEADLINE) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(HEADLINE) + 1))

    for code, _, label, comparator in ARMS:
        metrics = found.get(code)
        if metrics is None:
            continue
        cells = [format_value(metrics.get(key)) for key, _, _ in HEADLINE]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        rows.append({"arm": code, "label": label, "comparator": comparator or "",
                     **{key: metrics.get(key) for key, _, _ in HEADLINE}})
    lines.append("")

    # ---- Table 2: the differences, with the rule applied ---------------
    lines.append("## Table 2 — Differences against the comparator, and whether they survive")
    lines.append("")
    primary = "dry_mass_mape"
    # From configuration, so the rule stated in the document is the rule that
    # was applied. It was a literal 2.0 here while base.yaml also declares it.
    resolve_factor = float(cfg.evaluation.seed_replication.resolve_factor)
    lines.append(f"Primary metric: `{primary}`. Negative delta = better. "
                 f"A difference counts only if it exceeds {resolve_factor:g}x the "
                 f"pooled between-seed SD.")
    lines.append("")
    lines.append("| comparison | delta | between-seed SD | 2x SD | verdict |")
    lines.append("|---|---|---|---|---|")

    any_resolved = False
    comparisons = 0
    for code, _, label, comparator in ARMS:
        if comparator is None or code not in found or comparator not in found:
            continue
        value = found[code].get(primary)
        reference = found[comparator].get(primary)
        if value is None or reference is None:
            continue
        delta = float(value) - float(reference)
        comparisons += 1

        spread_a = seed_spread(root, code, modality, args.split, primary)
        spread_b = seed_spread(root, comparator, modality, args.split, primary)
        # The yardstick is how much the SAME configuration moves when only the
        # seed changes. Computed by holoqpi.utils.pooled_between_seed_sd, which
        # is the single implementation this project's figures, tables and
        # aggregate all now share -- there used to be three, with two different
        # formulas differing by a factor of sqrt(2).
        pooled, reason = pooled_between_seed_sd(spread_a, spread_b)
        if reason is None:
            threshold = resolve_factor * pooled
            resolved = abs(delta) > threshold
            any_resolved = any_resolved or resolved
            verdict = "**resolved**" if resolved else "not resolved (noise)"
            sd_text, threshold_text = f"{pooled:.4f}", f"{threshold:.4f}"
        else:
            verdict = f"not resolvable — {reason}"
            sd_text = threshold_text = "--"

        lines.append(f"| {code} vs {comparator} | {delta:+.4f} | {sd_text} | "
                     f"{threshold_text} | {verdict} |")
    lines.append("")
    if comparisons == 0:
        # An empty table is not evidence of anything, and saying "no difference
        # is resolved" here would read as a finding when in fact the metric was
        # never produced. The usual cause is that no cell was detected at all,
        # so no pair exists to compute a per-cell error over.
        lines.append(f"> **This table is empty: `{primary}` is absent from every run.** "
                     f"That is not a null result. The per-cell error is computed over "
                     f"IoU-matched pairs, so it does not exist when no cell was "
                     f"detected — check `detection_recall` and `cells_matched` in "
                     f"Table 3 before reading anything into this. A QUICK run always "
                     f"lands here, because two epochs cannot train a decoder.")
        lines.append("")
    elif not any_resolved:
        lines.append("> **No difference in this table is resolved.** That is a result, "
                     "not a gap: it means the arm-to-arm differences are smaller than "
                     "the run-to-run noise of the same configuration, and reporting "
                     "them as findings would be reporting seed noise. The v1 round "
                     "resolved 0 of 54 comparisons by the same rule.")
        lines.append("")

    # ---- Table 3: measurement quality, in full --------------------------
    lines.append("## Table 3 — Measurement quality")
    lines.append("")
    lines.append("Read the matched-cell MAPE and the coverage-adjusted MAPE "
                 "**together**. The first is over IoU-matched pairs only, so a "
                 "detector that finds the easiest cells and measures them perfectly "
                 "scores well on it; the second charges it for every cell it never "
                 "reported.")
    lines.append("")
    codes = [c for c, _, _, _ in ARMS if c in found]
    lines.append("| quantity | " + " | ".join(codes) + " |")
    lines.append("|" + "---|" * (len(codes) + 1))
    for key, label in MEASUREMENT:
        cells = [format_value(found[c].get(key)) for c in codes]
        if all(cell == "--" for cell in cells):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Table 3b: the physics-aware half -------------------------------
    lines.append("## Table 3b — Physics-aware forward model and reconstruction")
    lines.append("")
    lines.append("The forward-model residual asks whether the predicted field could "
                 "have produced the hologram that was actually recorded, so it needs "
                 "no reference phase. It is reported for **every** arm, including the "
                 "arms that never optimised it, which is what makes the column a "
                 "comparison rather than a restatement of the objective.")
    lines.append("")
    lines.append("`forward_residual_reference` is the same residual computed from the "
                 "GROUND-TRUTH phase: it is the floor this metric can reach, and "
                 "whatever the reference phase itself fails to explain is model "
                 "mismatch in the acquisition chain rather than an error the network "
                 "made. Read the ratio, not the level.")
    lines.append("")
    lines.append("| quantity | " + " | ".join(codes) + " |")
    lines.append("|" + "---|" * (len(codes) + 1))
    physics_present = False
    for key, label in PHYSICS:
        cells = [format_value(found[c].get(key)) for c in codes]
        if all(cell == "--" for cell in cells):
            continue
        physics_present = True
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if not physics_present:
        lines.append("| *(no forward-model metric was recorded)* | "
                     + " | ".join("--" for _ in codes) + " |")
    lines.append("")

    # ---- Table 3c: the amplitude output ---------------------------------
    amplitude_codes = [
        c for c in codes
        if any(found[c].get(key) is not None for key, _ in AMPLITUDE)
    ]
    lines.append("## Table 3c — Amplitude output")
    lines.append("")
    if not amplitude_codes:
        lines.append("No arm in this table recorded an amplitude metric. The "
                     "amplitude head is off by default (`model.amplitude.enabled`), "
                     "and an arm that predicts no amplitude is left out of this "
                     "table rather than given a blank column.")
        lines.append("")
    else:
        lines.append("Only the arms whose amplitude head is switched on appear here. "
                     "The reference is the modulus of a classical off-axis "
                     "reconstruction written by `scripts/prepare_amplitude.py`: it is "
                     "**not a measurement**, so every number below is agreement with "
                     "one reconstruction algorithm's output and must be reported in "
                     "those words.")
        lines.append("")
        lines.append("The row to read is the ratio against the thin-phase-object "
                     "assumption A = 1, which is what the rest of the study runs on. "
                     "Below 1.0 the head is closer to the reference than that "
                     "assumption; at or above 1.0 no amplitude claim survives.")
        lines.append("")
        lines.append("| quantity | " + " | ".join(amplitude_codes) + " |")
        lines.append("|" + "---|" * (len(amplitude_codes) + 1))
        for key, label in AMPLITUDE:
            cells = [format_value(found[c].get(key)) for c in amplitude_codes]
            if all(cell == "--" for cell in cells):
                continue
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    # The learned propagation distance, read from the arm's history rather than
    # its metrics: the trajectory is the reportable quantity for that arm, not a
    # single endpoint, and a mean over an epoch's steps can hide an oscillation.
    for code, config_path, label, _ in ARMS:
        if code not in found:
            continue
        name = experiment_name(Path(config_path))
        history_path = root / f"{name}_{modality}" / "history.json"
        if not history_path.is_file():
            continue
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            continue
        trajectory = [
            entry["forward_distance_um"] for entry in history
            if isinstance(entry, dict) and entry.get("forward_distance_um") is not None
        ]
        if len(trajectory) < 2:
            continue
        span = max(trajectory) - min(trajectory)
        settled = trajectory[-min(10, len(trajectory)):]
        lines.append(f"### Learned propagation distance, arm {label}")
        lines.append("")
        lines.append(f"- initial **{trajectory[0]:+.3f} um**, final "
                     f"**{trajectory[-1]:+.3f} um**")
        lines.append(f"- full excursion over training **{span:.3f} um**; spread over "
                     f"the last {len(settled)} epochs **"
                     f"{max(settled) - min(settled):.3f} um**")
        lines.append(f"- per-epoch trajectory: "
                     f"{', '.join(f'{v:+.2f}' for v in trajectory)}")
        lines.append("")
        lines.append("The **trajectory** is the result here, not the endpoint. A z that "
                     "settles near the supplied distance corroborates the instrument; "
                     "one that drifts without settling is the physically expected answer "
                     "for an off-axis geometry, where the carrier records phase at any "
                     "distance and z is therefore not identifiable from the hologram. "
                     "Report whichever of those the numbers above show, and do not "
                     "present a drifting value as a measured distance.")
        lines.append("")

    # ---- Table 4: efficiency -------------------------------------------
    benchmark_rows: list[dict] = []
    for path in sorted(root.glob("*hardware_benchmark*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if isinstance(entry, dict):
                benchmark_rows.append({"source": path.name, **entry})
    # KEEP ONLY THE MODALITY BEING REPORTED, and it has to happen here rather
    # than in the de-duplication below.
    #
    # `main.py benchmark` loops over every modality in --modalities (default
    # off_axis AND gabor) and, when a modality has no checkpoint, logs
    # "no checkpoint for <m>; profiling the untrained graph" and profiles it
    # anyway. That is legitimate for a hardware measurement -- latency, params
    # and MACs do not depend on the weights -- but it means each benchmark JSON
    # holds eight rows, four of them from an UNTRAINED graph.
    #
    # The de-duplication key below is (source, experiment, runtime, precision)
    # and does not include modality, so the two modalities collided and the
    # dict kept whichever came last: gabor. Table 4 was therefore reporting the
    # untrained Gabor profile under an off-axis heading. The architecture is the
    # same either way, so the numbers agreed to about 1% and nothing in the
    # conclusion changed -- but the row a reader would check in the CSV said
    # `modality: gabor`, which is indefensible in a manuscript.
    reported = [r for r in benchmark_rows if r.get("modality") in (None, modality)]
    dropped = len(benchmark_rows) - len(reported)
    benchmark_rows = reported

    # CONTAMINATED TIMING ROWS ARE MARKED, NOT QUIETLY TABULATED.
    #
    # A latency measured while another job held the same GPU is not a property of
    # the model, and it is invisible in the number. It happened: stage 8 once
    # overlapped a training run, the tail latencies roughly tripled, and the
    # COMPACT decoder came out slower than the full one -- inverting the whole
    # efficiency conclusion. holoqpi/deploy/benchmark.py now records
    # `timing_stable` (p99/p50 within tolerance) and `device_idle_at_start`, and
    # this is where they have to be acted on.
    # THE MEASUREMENT DECIDES, THE OCCUPANCY ONLY CAUTIONS.
    #
    # `timing_stable` is measured from the timings themselves and is the evidence
    # that matters: contention leaves p50 roughly intact and inflates the tail,
    # so a stable p99/p50 is direct evidence that nothing disturbed the run.
    # `device_idle_at_start` is a prior, and when it comes from
    # `device_occupancy_source = allocator_arithmetic` it is an estimate that
    # cannot tell whose memory it is seeing.
    #
    # Treating that estimate as disqualifying is what produced "20 of 20 timing
    # rows UNSTABLE ... do not quote them" in the 2026-09-17 run, on rows whose
    # own `timing_stable` was true and whose p99/p50 ran 1.00-1.11 -- the
    # benchmark was charging this process's own CUDA context to "other
    # processes". The table suppressed twenty sound measurements and printed the
    # remedy for a problem that was not there.
    def unstable(row: dict) -> bool:
        stable = row.get("timing_stable")
        if stable is None:
            return False                # written before the flags existed
        return str(stable).lower() == "false"

    def occupancy_caution(row: dict) -> bool:
        """A start-of-run occupancy warning on a row whose timings look clean."""
        idle = row.get("device_idle_at_start")
        return (
            idle is not None
            and str(idle).lower() == "false"
            and not unstable(row)
        )

    contaminated = [r for r in benchmark_rows if unstable(r)]
    cautioned = [r for r in benchmark_rows if occupancy_caution(r)]

    if benchmark_rows:
        lines.append("## Table 4 — Efficiency")
        lines.append("")
        lines.append(f"Modality `{modality}`, from the trained checkpoints."
                     + (f" {dropped} row(s) for other modalities were excluded;"
                        " `main.py benchmark` profiles an untrained graph when a"
                        " modality has no checkpoint." if dropped else ""))
        lines.append("")
        if contaminated:
            on_gpu = [r for r in contaminated if r.get("timing_device") == "cuda"]
            lines.append(
                f"> **{len(contaminated)} of {len(benchmark_rows)} timing rows have an "
                f"unstable latency distribution and are marked `UNSTABLE` below. Do "
                f"not quote them.** The `p99/p50` column is the tell: contention "
                f"leaves the median roughly intact and inflates the tail."
            )
            if on_gpu:
                # Only a GPU row supports the contention claim. An ONNX row that
                # fell back to CPU is routinely jittery on a shared machine, and
                # asserting GPU contention for it sends someone hunting for a job
                # that was never running.
                lines.append(
                    f">\n> {len(on_gpu)} of them were measured on a GPU, where this "
                    f"means another process was almost certainly using the device. "
                    f"Re-run stage 8 with nothing else on it:\n>\n"
                    f">     CUDA_VISIBLE_DEVICES=<idle gpu> bash run_v2.sh --stage 8"
                )
            if len(contaminated) > len(on_gpu):
                lines.append(
                    f">\n> {len(contaminated) - len(on_gpu)} were measured on the CPU "
                    f"(an ONNX row falls back there without `onnxruntime-gpu`), where "
                    f"scheduler jitter alone can trip the check. Raise "
                    f"`deploy.benchmark.timed_runs` before reading anything into it."
                )
            lines.append("")
        elif any("timing_stable" in r for r in benchmark_rows):
            lines.append("All timing rows have a stable latency distribution "
                         "(`p99/p50` within tolerance), which is the direct "
                         "evidence that nothing else disturbed the run.")
            lines.append("")
        if cautioned:
            # Reported, because the occupancy check exists for a reason and a
            # silent override would defeat it -- but separated from UNSTABLE,
            # because it is a prior and not a measurement.
            estimated = [
                r for r in cautioned
                if str(r.get("device_occupancy_source", "")) != "pid"
            ]
            lines.append(
                f"> {len(cautioned)} row(s) recorded memory in use on the device at "
                f"the start of the run while their own latency distribution came out "
                f"stable. The timings are reported."
                + (f" {len(estimated)} of those figures are allocator arithmetic "
                   f"rather than per-process attribution, which cannot separate this "
                   f"process's own CUDA context from another job's memory."
                   if estimated else "")
            )
            lines.append("")
        # The columns worth showing, in the order a reader wants them, but
        # kept only when the benchmark actually wrote them. A fixed allow-list
        # silently drops every column whose name the benchmark spells
        # differently, which is how this table first came out as two columns of
        # strings and nothing else.
        preferred = ["experiment", "modality", "runtime", "precision", "device", "input_size",
                     "params_total", "gmacs", "latency_mean_ms", "latency_p50_ms",
                     "latency_p99_ms", "latency_p99_over_p50", "timing_stable",
                     "timing_device", "fps", "weights_mb", "peak_activation_mb",
                     "peak_allocated_mb", "onnx_bytes"]
        keys = [k for k in preferred if any(k in row for row in benchmark_rows)]
        extra = sorted({k for row in benchmark_rows for k in row}
                       - set(keys) - {"source"})
        if extra:
            lines.append(f"Columns not shown: `{'`, `'.join(extra)}` — "
                         f"see the JSON under `{root}/`.")
            lines.append("")
        lines.append("| " + " | ".join(["", "source"] + keys) + " |")
        lines.append("|" + "---|" * (len(keys) + 2))
        # De-duplicate: a benchmark file holds one entry per runtime, and
        # re-running appends rather than replaces, so the same configuration can
        # appear several times. Keep the last of each (source, experiment,
        # runtime, precision), which is the most recent measurement.
        # `modality` is part of the key as well, so that even if a second
        # modality ever reaches this point it cannot silently displace the one
        # being reported.
        unique: dict[tuple, dict] = {}
        for row in benchmark_rows:
            unique[(row.get("source"), row.get("experiment"), row.get("modality"),
                    row.get("runtime"), row.get("precision"))] = row
        for row in unique.values():
            marker = "**UNSTABLE**" if unstable(row) else ""
            lines.append("| " + " | ".join(
                [marker, row["source"]]
                + [format_value(row.get(k), 2) for k in keys]) + " |")
        lines.append("")

    # ---- the label-free results, which need no trained model -----------
    lines.append("## Results that need no trained model")
    lines.append("")
    lines.append("These two run on the reference phase and the reference masks, so "
                 "they stand whatever the arms above do.")
    lines.append("")

    propagation = root / "error_propagation_summary.csv"
    if propagation.is_file():
        with open(propagation, newline="") as handle:
            summary = list(csv.DictReader(handle))
        lines.append("### Boundary error propagated into measurement")
        lines.append("")
        lines.append("| shift (px) | shift (um) | Dice | area error | mass error | mass/area |")
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(summary, key=lambda r: float(r["shift_px"])):
            lines.append(
                f"| {int(float(row['shift_px']))} | {float(row['shift_um']):+.3f} | "
                f"{float(row['dice']):.4f} | {float(row['area_signed']):+.4f} | "
                f"{float(row['mass_signed']):+.4f} | "
                f"{format_value(row.get('mass_over_area'), 3)} |"
            )
        lines.append("")
        ratios = [float(r["mass_over_area"]) for r in summary
                  if r.get("mass_over_area") not in (None, "", "nan")
                  and np.isfinite(float(r["mass_over_area"]))]
        if ratios:
            lines.append(f"Median mass/area error ratio **{np.median(ratios):.3f}**: dry "
                         f"mass is **{1 / np.median(ratios):.2f}x more robust** to a "
                         f"boundary error than projected area is, because the boundary "
                         f"sits where the cell is thinnest. Figure 17.")
            lines.append("")

    cells_file = root / "synthetic_validation_cells.csv"
    fields_file = root / "synthetic_validation_fields.csv"
    if cells_file.is_file():
        with open(cells_file, newline="") as handle:
            synthetic = list(csv.DictReader(handle))
        mass = np.array([float(r["mass_relative_error"]) for r in synthetic])
        area = np.array([float(r["area_relative_error"]) for r in synthetic])
        lines.append("### Pipeline error against exact analytic ground truth")
        lines.append("")
        lines.append(f"- per-cell dry mass: mean |error| **{np.abs(mass).mean():.4%}**")
        lines.append(f"- per-cell projected area: mean |error| **{np.abs(area).mean():.4%}**")
        if fields_file.is_file():
            with open(fields_file, newline="") as handle:
                fields = list(csv.DictReader(handle))
            totals = np.array([float(r["mass_total_relative_error"]) for r in fields])
            totals = totals[np.isfinite(totals)]
            if totals.size:
                lines.append(f"- **field-total** dry mass: mean bias "
                             f"**{totals.mean():+.4%}**, from the "
                             f"`min_cell_area_um2` filter discarding small cells")
        lines.append("")
        lines.append("These are two different floors for two different claims. Quote the "
                     "per-cell figure for 'the mass of this cell' and the field-total "
                     "figure for 'the mass on this field'. Figure 18.")
        lines.append("")

    # ---- what is not established ---------------------------------------
    lines.append("## What this document does not establish")
    lines.append("")
    if comparisons == 0:
        lines.append(f"- **No per-cell measurement result exists**: `{primary}` is "
                     f"absent from every run, because no cell was detected. Nothing "
                     f"in Table 1 or Table 2 is a finding.")
    elif not any_resolved:
        lines.append("- **No arm-to-arm difference is statistically resolved** at the "
                     "project's 2x-seed-spread rule. Without seed replication the "
                     "column cannot be filled at all; with it, a difference smaller "
                     "than the noise is noise.")
    lines.append("- **Segmentation labels are circular.** Every mask is "
                 "`Otsu(gaussian * phase_reference)`, so segmentation is scored against "
                 "a threshold of its own input. Independent labels are needed before "
                 "any Dice or recall here is a statement about the world.")
    lines.append("- **The forward-model term is a diagnostic, not a loss.** A 10% phase "
                 "error lowers its residual, so its gradient points the wrong way near "
                 "the truth. See `docs/v2_code_changes.md` Part 4.")
    lines.append("- **The amplitude reference is a reconstruction, not a measurement.** "
                 "Any amplitude result is agreement with one algorithm's output.")
    lines.append("")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    if rows:
        write_csv(rows, root / "results_table.csv")

    print(f"\n  wrote {destination}")
    if rows:
        print(f"  wrote {root / 'results_table.csv'}")
    print(f"\n  {len(found)} arms collected, {len(missing)} missing")
    if comparisons == 0 and found:
        print(f"\n  -> '{primary}' is absent from every run: no cell was detected, so")
        print("     no per-cell error exists. Check detection_recall. If this was a")
        print("     QUICK run that is expected and means nothing.")
    elif not any_resolved and found:
        print("\n  -> no arm-to-arm difference is resolved at the 2x-seed-spread rule.")
        print("     Run SEEDS=\"1337 2024\" bash run_v2.sh to fill that column, and")
        print("     report an unresolved difference as unresolved rather than as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
