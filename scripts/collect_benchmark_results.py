"""Collect repeated benchmark runs into one JSON file per benchmark, with statistics.

    python scripts/collect_benchmark_results.py                  # collect everything
    python scripts/collect_benchmark_results.py plan             # what exists, what is missing
    python scripts/collect_benchmark_results.py status --arm A --seed 123
    python scripts/collect_benchmark_results.py seeds --arm A

WHAT IS A REPLICATE
-------------------
For accuracy and measurement metrics the independent unit is a separately
TRAINED model: evaluation is deterministic, so scoring one checkpoint several
times would only repeat a number. Each seed is therefore one training run of the
arm (seed 42 in `runs/<experiment_name>_<modality>/`, every other seed in
`runs/v2_<ARM>_seed<N>_<modality>/`), evaluated once on the fixed test split.
The 113 test fields and ~3,000 cells inside one run are NOT replicates of the
experiment; they are summarised inside the run by the evaluator and enter here
as one value per run.

For latency and memory the unit is one benchmark session: a separate process
timing that seed's checkpoint (`<run dir>/hardware_benchmark_<device>.json`).
The 500 timed passes inside a session give that session's mean; the spread
reported here is between sessions.

Everything the project computes without a random component (the classical
reconstruction baseline, the boundary-displacement propagation, the z scan) is
collected once and marked `deterministic`: repeating it reproduces the same
numbers, so it carries n = 1 and no interval.

WHAT IT REFUSES
---------------
A run is used only if all of these hold, and the reason is printed otherwise:
  * its directory is the one the plan assigns to that arm and seed;
  * resolved_config.yaml records that seed and experiment name, and the same
    model / loss / training / data / optics / label settings as the arm's
    config today (a QUICK smoke test or an older objective fails here);
  * history.json holds every epoch (a killed run fails here);
  * the metrics file has a provenance file whose checkpoint hash matches the
    best_model.pt on disk, and whose training seed matches (a metrics file left
    over from another checkpoint or another seed fails here);
  * all runs of a benchmark scored the same number of test images and the same
    number of reference cells (a different dataset or label set fails here);
  * hardware runs were timed on the same GPU model and PyTorch version, with a
    stable latency distribution and ONNX actually on the GPU.
Seed directories that are not in the plan (e.g. an old seed-7 smoke test) are
listed and ignored, never pooled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from holoqpi.config import load_config, parse_overrides      # noqa: E402
from holoqpi.utils import (  # noqa: E402
    analysis_fingerprint,
    file_sha256,
    pooled_between_seed_sd,
    run_name,
)

from collect_results import ARMS                              # noqa: E402

# ---------------------------------------------------------------------------
# The arm registry: collect_results.ARMS plus the in-line arm, which that
# script leaves out of its single-modality tables.
# ---------------------------------------------------------------------------
ARM_TABLE: dict[str, dict] = {
    code: {"config": config, "label": label, "comparator": comparator}
    for code, config, label, comparator in ARMS
}
ARM_TABLE["G"] = {"config": "config/v2/g_baseline_gabor.yaml",
                  "label": "G  in-line Gabor baseline", "comparator": "A"}
# W10 specifies arm B's objective at arm B's seed: it is arm B, reported once.
ALIASES = {"W10": "B"}
# L is defined but was never part of the study; include it with --arms L.
NOT_IN_STUDY = {"L"}

# Settings that decide what a run trained on and how. Compared between a run's
# resolved_config.yaml and the arm's config today; anything else (paths,
# workers, logging, evaluation settings applied at scoring time) may differ.
PROTOCOL_SECTIONS = ("model", "loss", "training", "data", "optics",
                     "mask_generation", "labels", "project.deterministic",
                     "paths.mask_dir", "paths.manual_mask_dir")
PROTOCOL_IGNORED = {"data.num_workers", "data.pin_memory", "data.cache_in_memory",
                    "data.eval_batch_size", "training.log_every_n_steps"}

# Non-numeric fields of a hardware row that must agree between runs of one
# benchmark; a difference means results from different machines or stacks.
HARDWARE_MUST_MATCH = ("gpu_name", "torch_version", "encoder", "frontend", "lora",
                       "input_size")

EXIT_OK, EXIT_EVALUATE, EXIT_TRAIN, EXIT_INCOMPLETE = 0, 1, 10, 11
EXIT_MISMATCH, EXIT_NOT_PLANNED, EXIT_HARD = 12, 20, 3


# ===========================================================================
# Plan
# ===========================================================================
class Plan:
    """Which seeds each arm is run at, and where each run lives."""

    def __init__(self, base_config: str, overrides: dict, seeds: list[int] | None):
        self.base_path = base_config
        self.overrides = overrides
        self.base = load_config(base_config, overrides)
        replication = self.base.evaluation.seed_replication
        self.primary_seed = int(self.base.project.seed)
        extra = [int(s) for s in replication.seeds]
        full = [self.primary_seed] + [s for s in extra if s != self.primary_seed]
        self.full_seeds = list(dict.fromkeys(seeds if seeds else full))
        self.reduced_arms = set(replication.get("reduced_arms", []) or [])
        self.reduced_runs = int(replication.get("reduced_runs", 3))
        self.min_runs = int(replication.get("min_runs", 3))
        self.hardware_arms = list(replication.get("hardware_arms", []) or [])
        self.ci_level = float(replication.get("ci_level", 0.95))
        self.resolve_factor = float(replication.resolve_factor)
        self.output_root = ROOT / self.base.paths.output_root
        self._configs: dict[tuple, object] = {}

    # -- seeds ------------------------------------------------------------
    def seeds_for(self, code: str) -> list[int]:
        if code in self.reduced_arms:
            return self.full_seeds[: self.reduced_runs]
        return list(self.full_seeds)

    # -- identity of one run -----------------------------------------------
    def config_for(self, code: str, seed: int):
        """The arm's config exactly as a run at this seed must have used it."""
        key = (code, seed)
        if key not in self._configs:
            path = ROOT / ARM_TABLE[code]["config"]
            primary = load_config(str(path), self.overrides)
            name = (primary.experiment_name if seed == self.primary_seed
                    else f"v2_{code}_seed{seed}")
            extra = {"project": {"seed": seed}, "experiment_name": name}
            merged = dict(self.overrides)
            for section, value in extra.items():
                if isinstance(value, dict):
                    merged[section] = {**merged.get(section, {}), **value}
                else:
                    merged[section] = value
            self._configs[key] = load_config(str(path), merged)
        return self._configs[key]

    def run_dir(self, code: str, seed: int) -> Path:
        cfg = self.config_for(code, seed)
        return self.output_root / run_name(cfg.experiment_name, cfg.data.modality)

    def evaluation_tags(self, code: str) -> list[str]:
        """'' always; 'membrane' when the independent membrane labels exist."""
        cfg = self.config_for(code, self.primary_seed)
        membrane = ROOT / cfg.paths.data_root / cfg.membrane.mask_dir
        return ["", "membrane"] if membrane.is_dir() else [""]


# ===========================================================================
# Checking one run
# ===========================================================================
def _flatten(data, prefix: str = "") -> dict:
    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = data
    return out


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def protocol_differences(resolved: dict, expected: dict) -> tuple[list[str], list[str]]:
    """(differences, keys the run predates) between two configs, protocol keys only."""
    run_flat, want_flat = _flatten(resolved), _flatten(expected)

    def relevant(key: str) -> bool:
        if key in PROTOCOL_IGNORED:
            return False
        return any(key == s or key.startswith(s + ".") for s in PROTOCOL_SECTIONS)

    differences, predates = [], []
    for key in sorted(k for k in want_flat if relevant(k)):
        if key not in run_flat:
            predates.append(key)
        elif not _same(run_flat[key], want_flat[key]):
            differences.append(f"{key}: run {run_flat[key]!r}, config {want_flat[key]!r}")
    return differences, predates


class RunCheck:
    """Everything known about one (arm, seed) run directory."""

    def __init__(self, plan: Plan, code: str, seed: int, allow_legacy: bool = False):
        self.code, self.seed = code, seed
        self.cfg = plan.config_for(code, seed)
        self.dir = plan.run_dir(code, seed)
        self.tags = plan.evaluation_tags(code)
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.trained = False
        self.state = "missing"
        self.checkpoint = self.dir / "best_model.pt"
        self.sha = None
        self.history: list[dict] = []
        self.evaluations: dict[str, dict] = {}
        self._check_training()
        if self.trained:
            self._check_evaluations(allow_legacy)

    # -- training ---------------------------------------------------------
    def _check_training(self) -> None:
        if not self.checkpoint.is_file():
            self.state = "missing"
            return
        resolved_path = self.dir / "resolved_config.yaml"
        if not resolved_path.is_file():
            self._mismatch("no resolved_config.yaml, so the seed and settings it "
                           "was trained with cannot be verified")
            return
        resolved = yaml.safe_load(resolved_path.read_text()) or {}
        run_seed = (resolved.get("project") or {}).get("seed")
        if run_seed is None or int(run_seed) != self.seed:
            self._mismatch(f"trained with seed {run_seed}, expected {self.seed}")
            return
        if resolved.get("experiment_name") != self.cfg.experiment_name:
            self._mismatch(f"experiment_name {resolved.get('experiment_name')!r}, "
                           f"expected {self.cfg.experiment_name!r}")
            return
        differences, predates = protocol_differences(resolved, self.cfg.to_dict())
        if differences:
            shown = "; ".join(differences[:4])
            more = f" (+{len(differences) - 4} more)" if len(differences) > 4 else ""
            self._mismatch(f"trained under different settings: {shown}{more}")
            return
        if predates:
            self.warnings.append(f"{len(predates)} config key(s) added after this run "
                                 f"was trained, e.g. {predates[0]}")

        history_path = self.dir / "history.json"
        if not history_path.is_file():
            self._mismatch("no history.json, so it cannot be shown the run finished")
            return
        self.history = json.loads(history_path.read_text())
        epochs = int(self.cfg.training.epochs)
        patience = self.cfg.training.get("early_stopping_patience")
        if len(self.history) < epochs and not patience:
            self.state = "incomplete"
            self.errors.append(f"history.json holds {len(self.history)} of {epochs} "
                               f"epochs: the run did not finish")
            return
        self.sha = file_sha256(self.checkpoint)
        self.trained = True
        self.state = "trained"

    def _mismatch(self, reason: str) -> None:
        self.state = "mismatch"
        self.errors.append(reason)

    # -- evaluation -------------------------------------------------------
    def _check_evaluations(self, allow_legacy: bool) -> None:
        split = "test"
        all_good = True
        for tag in self.tags:
            suffix = f"_{tag}" if tag else ""
            metrics_path = self.dir / f"metrics_{split}{suffix}.json"
            provenance_path = self.dir / f"metrics_{split}{suffix}.provenance.json"
            label = f"metrics_{split}{suffix}.json"
            record = {"file": str(metrics_path), "provenance": None, "ok": False}
            self.evaluations[tag] = record
            if not metrics_path.is_file():
                record["problem"] = f"{label} missing"
                all_good = False
                continue
            if provenance_path.is_file():
                provenance = json.loads(provenance_path.read_text())
                if provenance.get("checkpoint_sha256") != self.sha:
                    record["problem"] = (f"{label} was produced from a different "
                                         f"checkpoint than the best_model.pt on disk")
                    all_good = False
                    continue
                expected = self._fingerprint(tag)
                if provenance.get("config_digest") not in (None, expected["config_digest"]):
                    record["problem"] = (f"{label} was scored under different measurement "
                                         f"settings than the config now specifies; "
                                         f"re-evaluate (bash run_v2.sh --stage 11)")
                    all_good = False
                    continue
                if provenance.get("splits_sha256") not in (None, expected["splits_sha256"]):
                    record["problem"] = (f"{label} was scored on a different split file "
                                         f"than data/{self.cfg.paths.splits_file}")
                    all_good = False
                    continue
                train_seed = provenance.get("train_seed")
                if train_seed is not None and int(train_seed) != self.seed:
                    record["problem"] = (f"{label} scored a checkpoint trained with "
                                         f"seed {train_seed}, expected {self.seed}")
                    all_good = False
                    continue
                record.update(provenance="verified", ok=True, details=provenance)
            elif allow_legacy and metrics_path.stat().st_mtime >= self.checkpoint.stat().st_mtime:
                record.update(provenance="legacy", ok=True)
                self.warnings.append(f"{label} has no provenance file; accepted as legacy "
                                     f"because it is newer than the checkpoint")
            else:
                record["problem"] = (f"{label} has no provenance file, so it cannot be "
                                     f"tied to this checkpoint; re-evaluate "
                                     f"(bash run_v2.sh --stage 11)")
                all_good = False
        if all_good:
            self.state = "evaluated"

    def _fingerprint(self, tag: str) -> dict:
        cfg = self.cfg
        if tag == "membrane":
            cfg = cfg.merged({"paths": {"manual_mask_dir": cfg.membrane.mask_dir}})
        return analysis_fingerprint(cfg)

    # -- hardware ---------------------------------------------------------
    def hardware_file(self, device: str) -> Path:
        return self.dir / f"hardware_benchmark_{device}.json"

    def hardware_problem(self, device: str) -> str | None:
        path = self.hardware_file(device)
        if not path.is_file():
            return f"{path.name} missing"
        rows = json.loads(path.read_text())
        if not rows:
            return f"{path.name} is empty"
        for row in rows:
            if row.get("checkpoint_sha256") != self.sha:
                return f"{path.name} timed a different checkpoint than best_model.pt"
            if row.get("seed") is None or int(row["seed"]) != self.seed:
                return f"{path.name} records seed {row.get('seed')}, expected {self.seed}"
        return None

    # -- summaries ----------------------------------------------------------
    def training_summary(self) -> dict:
        seconds = [e.get("seconds") for e in self.history if isinstance(e.get("seconds"), (int, float))]
        return {
            "epochs_completed": len(self.history),
            "wall_hours": round(sum(seconds) / 3600.0, 3) if seconds else None,
        }


# ===========================================================================
# Statistics
# ===========================================================================
def _t_critical(level: float, df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.5 + level / 2.0, df))
    except Exception:
        # Two-sided 95% Student t for df 1..30; beyond that the normal value.
        table = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
                 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
                 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
                 2.048, 2.045, 2.042]
        if not math.isclose(level, 0.95):
            raise RuntimeError("scipy is required for a confidence level other than 0.95")
        return table[df - 1] if df <= len(table) else 1.960


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def describe(values: list, level: float) -> dict:
    """mean, sample SD, n, SEM, Student-t interval, min, max over the finite values."""
    finite = [float(v) for v in values if _finite(v)]
    n = len(finite)
    ci_key = f"ci{int(round(level * 100))}"
    out = {"mean": None, "std": None, "n": n, "sem": None, ci_key: None,
           "min": None, "max": None, "n_missing": len(values) - n}
    if n == 0:
        return out
    array = np.asarray(finite)
    out.update(mean=float(array.mean()), min=float(array.min()), max=float(array.max()))
    if n > 1:
        sd = float(array.std(ddof=1))
        sem = sd / math.sqrt(n)
        half = _t_critical(level, n - 1) * sem
        out.update(std=sd, sem=sem, **{ci_key: [out["mean"] - half, out["mean"] + half]})
    if n > 1 and out["std"] == 0.0:
        out["constant_across_runs"] = True
    return out


def split_fields(record: dict) -> tuple[dict, dict]:
    """(numeric metrics, everything else). NaN and inf stay as metrics, as null."""
    numeric, other = {}, {}
    for key, value in record.items():
        if isinstance(value, bool) or value is None or isinstance(value, (str, list, dict)):
            other[key] = value
        elif isinstance(value, (int, float)):
            numeric[key] = float(value) if math.isfinite(float(value)) else None
        else:
            other[key] = str(value)
    return numeric, other


def statistics_for(runs: list[dict], level: float) -> dict:
    keys = sorted({key for run in runs for key in run["metrics"]})
    stats = {}
    for key in keys:
        values = [run["metrics"].get(key) for run in runs]
        stats[key] = describe(values, level)
        missing = [run["seed"] for run in runs if key not in run["metrics"]]
        non_finite = [run["seed"] for run in runs
                      if key in run["metrics"] and run["metrics"][key] is None]
        if missing:
            stats[key]["seeds_without_metric"] = missing
        if non_finite:
            stats[key]["seeds_non_finite"] = non_finite
    return stats


# ===========================================================================
# Categories
# ===========================================================================
class Collector:
    def __init__(self, plan: Plan, args):
        self.plan = plan
        self.args = args
        self.categories: dict[str, dict] = {}
        self.errors: dict[str, list[str]] = {}
        self.ignored: list[dict] = []
        self.legacy: dict[str, list[str]] = {}
        self._checks: dict[tuple, RunCheck] = {}

    def check(self, code: str, seed: int) -> RunCheck:
        key = (code, seed)
        if key not in self._checks:
            self._checks[key] = RunCheck(self.plan, code, seed, self.args.allow_legacy)
        return self._checks[key]

    def _fail(self, name: str, message: str) -> None:
        self.errors.setdefault(name, []).append(message)

    def _finish(self, name: str, payload: dict, runs: list[dict], expected: list[int]) -> None:
        got = [run["seed"] for run in runs]
        missing = [s for s in expected if s not in got]
        payload.update(
            n_runs=len(runs),
            seeds=got,
            seeds_planned=expected,
            seeds_missing=missing,
            complete=not missing and name not in self.errors,
            runs=runs,
            statistics=statistics_for(runs, self.plan.ci_level) if runs else {},
        )
        if len(runs) < self.plan.min_runs and payload.get("replication") == "independent_runs":
            payload.setdefault("warnings", []).append(
                f"only {len(runs)} run(s); the plan's minimum is {self.plan.min_runs}")
        if payload["complete"] or self.args.allow_partial:
            self.categories[name] = payload

    # -- trained arms: accuracy and measurement -----------------------------
    def evaluation(self, code: str) -> None:
        seeds = self.plan.seeds_for(code)
        for tag in self.plan.evaluation_tags(code):
            name = f"arm_{code}" + (f"_{tag}" if tag else "")
            cfg = self.plan.config_for(code, self.plan.primary_seed)
            payload = {
                "benchmark": name,
                "kind": "evaluation",
                "arm": code,
                "label": ARM_TABLE[code]["label"],
                "config": ARM_TABLE[code]["config"],
                "experiment_name": cfg.experiment_name,
                "modality": cfg.data.modality,
                "split": "test",
                "labels": "membrane (independent)" if tag else "phase-derived (Otsu)",
                "replication": "independent_runs",
                "unit_of_replication": "one training run per seed, evaluated once",
                "warnings": [],
            }
            runs = []
            for seed in seeds:
                check = self.check(code, seed)
                payload["warnings"].extend(f"seed {seed}: {w}" for w in check.warnings)
                evaluation = check.evaluations.get(tag)
                if not check.trained:
                    self._fail(name, f"seed {seed}: {check.state} -- "
                                     + ("; ".join(check.errors) or f"no run at {check.dir}"))
                    continue
                if not evaluation or not evaluation.get("ok"):
                    self._fail(name, f"seed {seed}: {evaluation.get('problem') if evaluation else 'not evaluated'}")
                    continue
                raw = json.loads(Path(evaluation["file"]).read_text())
                metrics, other = split_fields(raw)
                details = evaluation.get("details") or {}
                runs.append({
                    "seed": seed,
                    "run_dir": _rel(check.dir),
                    "experiment_name": check.cfg.experiment_name,
                    "checkpoint_sha256": check.sha,
                    "checkpoint_epoch": details.get("checkpoint_epoch"),
                    "provenance": evaluation["provenance"],
                    "evaluated_at": details.get("evaluated_at"),
                    "training": check.training_summary(),
                    "metrics": metrics,
                    "non_numeric": other,
                })
            # The same test set and the same labels in every run, or they are
            # not replicates of one experiment.
            for key in ("phase_n_images", "seg_n_images", "cells_reference"):
                values = {run["metrics"].get(key) for run in runs}
                if len(values) > 1:
                    self._fail(name, f"runs disagree on {key} ({sorted(map(str, values))}): "
                                     f"they were not scored on the same test set/labels")
            self._finish(name, payload, runs, seeds)

    # -- trained arms: latency and memory ------------------------------------
    def hardware(self, code: str) -> None:
        device = self.args.hardware_device
        name = f"hardware_arm_{code}"
        seeds = self.plan.seeds_for(code)
        payload = {
            "benchmark": name,
            "kind": "hardware",
            "arm": code,
            "label": ARM_TABLE[code]["label"],
            "device_type": device,
            "replication": "independent_runs",
            "unit_of_replication": "one benchmark process per seed, timing that "
                                   "seed's checkpoint; metrics are session means",
            "metric_naming": "<runtime>_<precision>.<field>; architecture fields un-prefixed",
            "warnings": [],
        }
        runs, reference_meta, reference_combos = [], None, None
        for seed in seeds:
            check = self.check(code, seed)
            if not check.trained:
                self._fail(name, f"seed {seed}: not trained ({check.state})")
                continue
            problem = check.hardware_problem(device)
            if problem:
                self._fail(name, f"seed {seed}: {problem}")
                continue
            rows = json.loads(check.hardware_file(device).read_text())
            metrics, meta, combos = {}, {}, []
            architecture = ("params_total", "params_trainable", "params_frozen",
                            "params_trainable_fraction", "gmacs", "input_size")
            for row in rows:
                combo = f"{row['runtime']}_{row['precision']}"
                combos.append(combo)
                if row.get("timing_stable") is False and not self.args.allow_unstable:
                    self._fail(name, f"seed {seed} {combo}: unstable latency "
                                     f"(p99/p50 = {row.get('latency_p99_over_p50'):.2f}); "
                                     f"re-run this seed on an idle GPU")
                if row.get("comparable_to_pytorch_row") is False:
                    self._fail(name, f"seed {seed} {combo}: ONNX ran on "
                                     f"{row.get('providers')}, not the GPU")
                if row.get("device_idle_at_start") is False:
                    payload["warnings"].append(
                        f"seed {seed} {combo}: another process held "
                        f"{row.get('foreign_gpu_memory_mb')} MiB at the start "
                        f"(source: {row.get('device_occupancy_source')})")
                numeric, other = split_fields(row)
                for key, value in numeric.items():
                    if key in architecture:
                        if key in metrics and not _same(metrics[key], value):
                            self._fail(name, f"seed {seed}: {key} differs between rows")
                        metrics[key] = value
                    elif key not in ("seed", "measured_at", "checkpoint_bytes",
                                     "checkpoint_mtime", "checkpoint_epoch", "train_seed"):
                        metrics[f"{combo}.{key}"] = value
                for key in HARDWARE_MUST_MATCH:
                    meta[key] = row.get(key)
                meta[f"{combo}.providers"] = other.get("providers")
            if reference_meta is None:
                reference_meta, reference_combos = meta, sorted(combos)
            else:
                for key in HARDWARE_MUST_MATCH:
                    if meta.get(key) != reference_meta.get(key):
                        self._fail(name, f"seed {seed}: {key} is {meta.get(key)!r}, other "
                                         f"runs {reference_meta.get(key)!r} -- results from "
                                         f"different machines or software are not replicates")
                if sorted(combos) != reference_combos:
                    self._fail(name, f"seed {seed}: runtime/precision set {sorted(combos)} "
                                     f"differs from {reference_combos}")
            runs.append({
                "seed": seed,
                "run_dir": _rel(check.dir),
                "checkpoint_sha256": check.sha,
                "file": check.hardware_file(device).name,
                "metrics": metrics,
                "non_numeric": meta,
            })
        self._finish(name, payload, runs, seeds)

    # -- label-free, seeded: the synthetic ground-truth floor ---------------
    def synthetic(self) -> None:
        name = "synthetic_validation"
        seeds = list(self.plan.full_seeds)
        payload = {
            "benchmark": name,
            "kind": "measurement_chain_floor",
            "replication": "independent_runs",
            "unit_of_replication": "one set of synthetic fields per seed "
                                   "(scripts/synthetic_validation.py)",
            "warnings": [],
        }
        runs, field_counts = [], set()
        for seed in seeds:
            path = self.plan.output_root / "synthetic_validation_seeds" / f"seed{seed}" \
                / "synthetic_validation_summary.json"
            if not path.is_file():
                self._fail(name, f"seed {seed}: {_rel(path)} missing")
                continue
            summary = json.loads(path.read_text())
            if int(summary.get("seed", -1)) != seed:
                self._fail(name, f"seed {seed}: file records seed {summary.get('seed')}")
                continue
            current = analysis_fingerprint(self.plan.base)
            if summary.get("config_digest") != current["config_digest"]:
                self._fail(name, f"seed {seed}: produced under different measurement "
                                 f"settings; rerun bash run_v2.sh --stage 13 with REEVALUATE=1")
                continue
            field_counts.add((summary.get("fields"), summary.get("cells_per_field")))
            metrics, other = split_fields({k: v for k, v in summary.items()
                                           if k not in ("seed", "config_digest", "splits_sha256")})
            runs.append({"seed": seed, "file": _rel(path),
                         "metrics": metrics, "non_numeric": other})
        if len(field_counts) > 1:
            self._fail(name, f"seeds used different field/cell counts: {sorted(field_counts)}")
        self._finish(name, payload, runs, seeds)

    # -- deterministic analyses -------------------------------------------
    def _deterministic(self, name: str, source: Path, metrics: dict, other: dict,
                       description: str) -> None:
        payload = {
            "benchmark": name,
            "kind": "deterministic",
            "replication": "deterministic",
            "unit_of_replication": description,
            "warnings": list(self.legacy.get(name, [])),
        }
        runs = [{"seed": None, "file": _rel(source),
                 "modified_at": source.stat().st_mtime,
                 "metrics": metrics, "non_numeric": other}]
        payload.update(n_runs=1, seeds=[], seeds_planned=[], seeds_missing=[],
                       complete=True, runs=runs,
                       statistics=statistics_for(runs, self.plan.ci_level))
        self.categories[name] = payload

    def _verified(self, name: str, provenance_path: Path, rerun: str,
                  block: str | None = None) -> bool:
        """True if an analysis output was produced under today's settings."""
        current = analysis_fingerprint(self.plan.base)
        record = None
        if provenance_path.is_file():
            record = json.loads(provenance_path.read_text())
            if block is not None:
                record = record.get(block)
        if record is None:
            if self.args.allow_legacy:
                self.legacy_note(name, f"{_rel(provenance_path)} missing; accepted as legacy")
                return True
            self._fail(name, f"no provenance ({_rel(provenance_path)}), so it cannot be "
                             f"shown to match the current settings and split; rerun: {rerun}")
            return False
        for key in ("config_digest", "splits_sha256"):
            if record.get(key) != current[key]:
                self._fail(name, f"produced under different {'settings' if key == 'config_digest' else 'split file'} "
                                 f"than the config now specifies; rerun: {rerun}")
                return False
        return True

    def legacy_note(self, name: str, message: str) -> None:
        self.legacy.setdefault(name, []).append(message)

    def deterministic(self) -> None:
        root = self.plan.output_root
        note_conventional = ("classical reconstruction has no random component; its point "
                             "metrics do not depend on the seed (only the within-run "
                             "bootstrap interval keys use project.seed)")
        for modality in ("off_axis", "gabor"):
            path = root / f"conventional_{modality}" / "metrics_test.json"
            name = f"conventional_{modality}"
            if path.is_file() and self._verified(
                name, path.with_name("metrics_test.provenance.json"),
                "python scripts/conventional_baseline.py --config config/base.yaml",
            ):
                metrics, other = split_fields(json.loads(path.read_text()))
                self._deterministic(name, path, metrics, other, note_conventional)
            elif not path.is_file():
                self._fail(name, f"{_rel(path)} missing "
                                 f"(python scripts/conventional_baseline.py --config config/base.yaml)")

        path = root / "error_propagation_summary.csv"
        if not path.is_file():
            self._fail("error_propagation", f"{_rel(path)} missing")
        elif self._verified("error_propagation", root / "error_propagation_summary.provenance.json",
                            "python scripts/error_propagation.py --config config/base.yaml --max-shift 5"):
            metrics = {}
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    shift = int(float(row["shift_px"]))
                    for key, value in row.items():
                        if key == "shift_px":
                            continue
                        try:
                            number = float(value)
                        except (TypeError, ValueError):
                            continue
                        metrics[f"shift_{shift:+d}px.{key}"] = number if math.isfinite(number) else None
            self._deterministic("error_propagation", path, metrics, {},
                                "reference masks displaced by known pixel shifts; no random component")

        path = root / "z_calibration.json"
        if path.is_file():
            data = json.loads(path.read_text())
            metrics, other = {}, {}
            verified = True
            for modality, block in data.items():
                if not isinstance(block, dict):
                    continue
                verified &= self._verified(
                    "z_calibration", root / "z_calibration.provenance.json",
                    "python scripts/calibrate_z.py --config config/base.yaml --steps 41",
                    block=modality)
                numeric, rest = split_fields(block)
                metrics.update({f"{modality}.{k}": v for k, v in numeric.items()})
                other.update({f"{modality}.{k}": v for k, v in rest.items()
                              if not isinstance(v, (dict, list))})
            if verified:
                self._deterministic("z_calibration", path, metrics, other,
                                    "scan of the forward-model residual over z; no random component")
        else:
            self._fail("z_calibration", f"{_rel(path)} missing")

    # -- comparisons ----------------------------------------------------------
    def comparisons(self) -> dict:
        pairs = [(code, info["comparator"]) for code, info in ARM_TABLE.items()
                 if info["comparator"]]
        out = {"rule": f"resolved when |difference of means| > {self.plan.resolve_factor:g} x "
                       f"the pooled between-seed SD (holoqpi.utils.pooled_between_seed_sd); "
                       f"a resolution criterion, not a significance test",
               "resolve_factor": self.plan.resolve_factor,
               "comparisons": {}}
        for code, comparator in pairs:
            a, b = self.categories.get(f"arm_{code}"), self.categories.get(f"arm_{comparator}")
            if not a or not b:
                continue
            rows = {}
            for metric in sorted(set(a["statistics"]) & set(b["statistics"])):
                values_a = [r["metrics"].get(metric) for r in a["runs"]]
                values_b = [r["metrics"].get(metric) for r in b["runs"]]
                values_a = [v for v in values_a if _finite(v)]
                values_b = [v for v in values_b if _finite(v)]
                if not values_a or not values_b:
                    continue
                difference = float(np.mean(values_a) - np.mean(values_b))
                pooled, reason = pooled_between_seed_sd(values_a, values_b)
                if reason is not None:
                    verdict, threshold = f"unresolvable ({reason})", None
                else:
                    threshold = self.plan.resolve_factor * pooled
                    verdict = "resolved" if abs(difference) > threshold else "within seed noise"
                rows[metric] = {
                    "mean": float(np.mean(values_a)), "comparator_mean": float(np.mean(values_b)),
                    "difference": difference, "n": len(values_a),
                    "comparator_n": len(values_b),
                    "pooled_sd": pooled if reason is None else None,
                    "threshold": threshold, "verdict": verdict,
                }
            out["comparisons"][f"{code}_vs_{comparator}"] = {
                "arm": code, "comparator": comparator, "metrics": rows}
        return out

    # -- ignored directories ------------------------------------------------
    def find_ignored(self) -> None:
        import re
        pattern = re.compile(r"^v2_(?P<code>[A-Za-z0-9]+)_seed(?P<seed>\d+)_(?P<mod>.+)$")
        for directory in sorted(self.plan.output_root.glob("v2_*_seed*_*")):
            match = pattern.match(directory.name)
            if not match or not directory.is_dir():
                continue
            code, seed = match.group("code"), int(match.group("seed"))
            if code in ARM_TABLE and seed in self.plan.seeds_for(code):
                continue
            self.ignored.append({"dir": directory.name,
                                 "reason": "seed not in the plan" if code in ARM_TABLE
                                 else "arm not in the registry"})


# ===========================================================================
# Output
# ===========================================================================
def _clean(value):
    """NaN/inf -> null so every file is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _rel(path: Path) -> str:
    """A path relative to the repository when it is inside it, else absolute."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_strict_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(payload), indent=2, allow_nan=False)
    path.write_text(text + "\n")


def study_arms(args) -> list[str]:
    if args.arms:
        unknown = [a for a in args.arms if a not in ARM_TABLE and a not in ALIASES]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARM_TABLE)}")
        return [ALIASES.get(a, a) for a in dict.fromkeys(args.arms)]
    return [c for c in ARM_TABLE if c not in ALIASES and c not in NOT_IN_STUDY]


def command_collect(plan: Plan, args) -> int:
    collector = Collector(plan, args)
    only = set(args.only or ["evaluation", "hardware", "synthetic", "deterministic"])
    arms = study_arms(args)

    if "evaluation" in only:
        for code in arms:
            collector.evaluation(code)
    if "hardware" in only:
        for code in [c for c in plan.hardware_arms if c in arms]:
            collector.hardware(code)
    if "synthetic" in only:
        collector.synthetic()
    if "deterministic" in only:
        collector.deterministic()
    collector.find_ignored()

    out_dir = Path(args.out) if args.out else plan.output_root / "benchmark_results"
    if out_dir.exists():
        for stale in out_dir.glob("results_*.json"):
            stale.unlink()                     # never leave a previous collection's file behind
    written = []
    for name, payload in collector.categories.items():
        payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        payload["ci_level"] = plan.ci_level
        path = out_dir / f"results_{name}.json"
        write_strict_json(payload, path)
        written.append(path)

    comparisons = collector.comparisons()
    if comparisons["comparisons"]:
        write_strict_json(comparisons, out_dir / "results_comparisons.json")
        written.append(out_dir / "results_comparisons.json")

    # One long table for building manuscript tables.
    ci_key = f"ci{int(round(plan.ci_level * 100))}"
    summary_rows = []
    for name, payload in collector.categories.items():
        for metric, stats in payload["statistics"].items():
            interval = stats.get(ci_key) or [None, None]
            summary_rows.append({
                "benchmark": name, "metric": metric, "n": stats["n"],
                "mean": stats["mean"], "std": stats["std"], "sem": stats["sem"],
                f"{ci_key}_low": interval[0], f"{ci_key}_high": interval[1],
                "min": stats["min"], "max": stats["max"],
                "complete": payload["complete"],
            })
    if summary_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "benchmark_summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(_clean(summary_rows))

    index = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": plan.base_path,
        "overrides": plan.overrides,
        "seeds": plan.full_seeds,
        "seeds_by_arm": {c: plan.seeds_for(c) for c in arms},
        "hardware_arms": [c for c in plan.hardware_arms if c in arms],
        "hardware_device": args.hardware_device,
        "ci_level": plan.ci_level,
        "aliases": ALIASES,
        "benchmarks": {
            name: {"file": f"results_{name}.json", "n_runs": p["n_runs"],
                   "seeds": p["seeds"], "complete": p["complete"],
                   "replication": p["replication"]}
            for name, p in collector.categories.items()
        },
        "not_written": collector.errors,
        "ignored_directories": collector.ignored,
    }
    write_strict_json(index, out_dir / "benchmark_index.json")

    # ---- report ----------------------------------------------------------
    print(f"\nseeds {plan.full_seeds}   output {out_dir}")
    for name, payload in collector.categories.items():
        flag = "" if payload["complete"] else "   PARTIAL"
        print(f"  wrote results_{name}.json   n={payload['n_runs']}   "
              f"seeds {payload['seeds']}{flag}")
    if collector.ignored:
        print("\nignored (not in the plan, never pooled):")
        for item in collector.ignored:
            print(f"  {item['dir']}: {item['reason']}")
    if collector.errors:
        print("\nNOT COMPLETE -- fix these, or pass --allow-partial to write what exists:")
        for name, problems in collector.errors.items():
            print(f"  {name}")
            for problem in problems:
                print(f"    - {problem}")
        return EXIT_HARD
    print("\nall benchmarks complete")
    return EXIT_OK


def command_status(plan: Plan, args) -> int:
    code = ALIASES.get(args.arm, args.arm)
    if code not in ARM_TABLE:
        print(f"unknown arm {args.arm}")
        return EXIT_HARD
    if args.seed not in plan.seeds_for(code):
        print("not-planned - - -")
        return EXIT_NOT_PLANNED
    check = RunCheck(plan, code, args.seed, allow_legacy=False)
    relative = _rel(check.dir)
    reason = "; ".join(check.errors) or "-"
    if args.hardware:
        if not check.trained:
            state, code_out = ("mismatch", EXIT_MISMATCH) if check.state == "mismatch" \
                else ("not-trained", EXIT_TRAIN)
        elif check.hardware_problem(args.hardware_device):
            state, code_out = "needs-benchmark", EXIT_EVALUATE
            reason = check.hardware_problem(args.hardware_device)
        else:
            state, code_out = "benchmarked", EXIT_OK
    else:
        state = check.state
        code_out = {"evaluated": EXIT_OK, "trained": EXIT_EVALUATE, "missing": EXIT_TRAIN,
                    "incomplete": EXIT_INCOMPLETE, "mismatch": EXIT_MISMATCH}[state]
        if state == "trained":
            reason = "; ".join(e.get("problem", "") for e in check.evaluations.values()
                               if not e.get("ok")) or "-"
    print(f"{state} {check.cfg.experiment_name} {relative} {check.cfg.data.modality}")
    print(f"  {reason}", file=sys.stderr)
    return code_out


def command_plan(plan: Plan, args) -> int:
    arms = study_arms(args)
    print(f"seeds {plan.full_seeds}   (reduced arms {sorted(plan.reduced_arms) or 'none'}: "
          f"first {plan.reduced_runs})")
    print(f"{'arm':<5}" + "".join(f"{s:>12}" for s in plan.full_seeds) + "   run directory (seed 42)")
    todo = 0
    for code in arms:
        cells = []
        for seed in plan.full_seeds:
            if seed not in plan.seeds_for(code):
                cells.append("-")
                continue
            check = RunCheck(plan, code, seed)
            cells.append(check.state)
            todo += check.state != "evaluated"
        print(f"{code:<5}" + "".join(f"{c:>12}" for c in cells)
              + f"   {plan.run_dir(code, plan.primary_seed).name}")
    hardware = [c for c in plan.hardware_arms if c in arms]
    if hardware:
        print(f"\nhardware ({args.hardware_device})")
        for code in hardware:
            cells = []
            for seed in plan.full_seeds:
                if seed not in plan.seeds_for(code):
                    cells.append("-")
                    continue
                check = RunCheck(plan, code, seed)
                cells.append("done" if check.trained and not
                             check.hardware_problem(args.hardware_device) else "todo")
            print(f"{code:<5}" + "".join(f"{c:>12}" for c in cells))
    print(f"\n{todo} training/evaluation run(s) not yet complete")
    return EXIT_OK


def command_seeds(plan: Plan, args) -> int:
    if args.arm:
        print(" ".join(str(s) for s in plan.seeds_for(ALIASES.get(args.arm, args.arm))))
    elif args.hardware:
        print(" ".join(plan.hardware_arms))
    else:
        print(" ".join(str(s) for s in plan.full_seeds))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "collect")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default="config/base.yaml")
        p.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="override the seed list (default: project.seed + "
                            "evaluation.seed_replication.seeds)")
        p.add_argument("--arms", nargs="+", default=None,
                       help="restrict to these arm codes (default: every study arm)")
        p.add_argument("--hardware-device", default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="the SAME overrides the runs were trained with, if any")

    collect = sub.add_parser("collect", help="write results_*.json (default)")
    common(collect)
    collect.add_argument("--out", default=None, help="default: runs/benchmark_results")
    collect.add_argument("--only", nargs="+",
                         choices=["evaluation", "hardware", "synthetic", "deterministic"])
    collect.add_argument("--allow-partial", action="store_true",
                         help="write benchmarks that are missing seeds, marked complete=false")
    collect.add_argument("--allow-legacy", action="store_true",
                         help="accept metrics files without provenance if newer than "
                              "their checkpoint")
    collect.add_argument("--allow-unstable", action="store_true",
                         help="accept hardware rows with an unstable latency distribution")

    status = sub.add_parser("status", help="state of one run, for run_v2.sh")
    common(status)
    status.add_argument("--arm", required=True)
    status.add_argument("--seed", type=int, required=True)
    status.add_argument("--hardware", action="store_true")

    plan_parser = sub.add_parser("plan", help="table of every planned run and its state")
    common(plan_parser)

    seeds = sub.add_parser("seeds", help="print the planned seeds")
    common(seeds)
    seeds.add_argument("--arm", default=None)
    seeds.add_argument("--hardware", action="store_true",
                       help="print the hardware arms instead")

    args = parser.parse_args(argv)
    args.allow_legacy = getattr(args, "allow_legacy", False)
    args.allow_partial = getattr(args, "allow_partial", False)
    args.allow_unstable = getattr(args, "allow_unstable", False)

    import os
    os.chdir(ROOT)
    plan = Plan(args.config, parse_overrides(args.set), args.seeds)
    return {"collect": command_collect, "status": command_status,
            "plan": command_plan, "seeds": command_seeds}[args.command](plan, args)


if __name__ == "__main__":
    sys.exit(main())
