#!/usr/bin/env python
"""Build results_for_manuscript/ from the corrected experiment outputs.

Run from anywhere:    python analysis/v42/compile_results.py

Every number is read from, or computed from, files under runs/, logs/, config/
and data/. Nothing is typed in by hand except (a) the planned design (which
configuration is compared with which, and how many training runs were
intended), which is asserted against the files, and (b) the registration
statistics, whose source archive is not in the repository (marked
``verified: false``).

Entry format used throughout the JSON outputs::

    {"value", "units", "source_file", "source_key", "field_set", "n_runs",
     "seeds", "derivation": "direct" | "derived", "derived_from", "calculation"}

Resolution criterion (Methods, statistical analysis): a difference between two
configurations is "resolved" only if |delta| > 2*sqrt((s1^2 + s2^2)/2) AND both
configurations have at least three training runs (s = between-seed sample SD,
ddof = 1). With fewer than three runs on either side, 2xSD is not estimable and
no substitute variance is used.

The script stops with an error if a file it needs is missing or if a
consistency check fails (run counts, seeds, field counts, split hash).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
OUT = ROOT / "results_for_manuscript"
CHECKS: list[dict] = []          # consistency checks, written to metadata/consistency_checks.json


# ----------------------------------------------------------------------------- helpers
def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def jload(p: Path):
    with open(p) as fh:
        return json.load(fh)


def rcsv(p: Path):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def sd(xs):
    xs = [float(x) for x in xs]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def E(value, units, src, key, field_set, n_runs, seeds, derived_from=None, calc=None):
    return {"value": value, "units": units, "source_file": src, "source_key": key,
            "field_set": field_set, "n_runs": n_runs, "seeds": seeds,
            "derivation": "derived" if calc else "direct",
            "derived_from": derived_from, "calculation": calc}


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    if not ok:
        print(f"  CONSISTENCY FAILURE: {name}: {detail}", file=sys.stderr)


def write(path: Path, obj):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str, allow_nan=False)
                    if not _has_nan(obj) else json.dumps(_scrub(obj), indent=1, default=str))
    return path


def _has_nan(o):
    if isinstance(o, float):
        return math.isnan(o) or math.isinf(o)
    if isinstance(o, dict):
        return any(_has_nan(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return any(_has_nan(v) for v in o)
    return False


def _scrub(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_scrub(v) for v in o]
    return o


def assess(delta, sa, na, sb, nb):
    """Resolution criterion. Returns the raw quantities and the assessment."""
    out = {"delta": delta, "n_runs_config": na, "n_runs_comparator": nb,
           "sd_config": sa, "sd_comparator": sb}
    if na < 3 or nb < 3 or sa is None or sb is None:
        out.update(pooled_sd=None, two_x_pooled_sd=None, criterion_evaluable=False,
                   assessment="Not estimable from the available runs")
        return out
    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2)
    out.update(pooled_sd=pooled, two_x_pooled_sd=2 * pooled, criterion_evaluable=True,
               assessment="Exceeds 2x pooled SD" if abs(delta) > 2 * pooled else "Within 2x pooled SD")
    return out


# ----------------------------------------------------------------------------- design registry
# (label, arm, {seed: run dir}, comparator arm, change relative to comparator, planned training runs)
SEEDS3 = (42, 1337, 2024)
REG = {
    "A": ("End-to-End Neural Baseline", {42: "v2_baseline_off_axis", 1337: "v2_A_seed1337_off_axis",
                                         2024: "v2_A_seed2024_off_axis"}, None,
          "off-axis; reconstruction + segmentation losses only", 3),
    "G": ("In-Line Neural Configuration", {42: "v2_baseline_gabor"}, "A",
          "baseline objective on in-line Gabor holograms; SNU_01-SNU_50 excluded", 1),
    "B": ("+IPP (per-cell)", {42: "v2_cell_ipp_off_axis", 1337: "v2_B_seed1337_off_axis",
                              2024: "v2_B_seed2024_off_axis"}, "A", "+ L_IPP^cell, w=1.0", 3),
    "B1": ("+IPP (image)", {42: "v2_image_volume_off_axis", 1337: "v2_B1_seed1337_off_axis",
                            2024: "v2_B1_seed2024_off_axis"}, "A", "+ L_IPP^img, w=1.0", 3),
    "B2": ("+Area", {42: "v2_cell_ipp_area_off_axis"}, "B", "+ L_area (per-cell), w=1.0", 1),
    "C": ("+BGA", {42: "v2_cell_ipp_bga_off_axis"}, "B", "+ L_BGA, w=0.05", 1),
    "W01": ("+IPP (per-cell), w=0.1", {42: "v2_ipp_w01_off_axis"}, "A", "L_IPP^cell weight 0.1", 1),
    "W03": ("+IPP (per-cell), w=0.3", {42: "v2_ipp_w03_off_axis"}, "A", "L_IPP^cell weight 0.3", 1),
    "W30": ("+IPP (per-cell), w=3.0", {42: "v2_ipp_w30_off_axis"}, "A", "L_IPP^cell weight 3.0", 1),
    "D0": ("+Amplitude", {42: "v2_amplitude_off_axis"}, "B", "+ amplitude head, + L_amp, w=0.1", 1),
    "D1": ("+Fwd (fixed z)", {42: "v2_forward_amplitude_off_axis"}, "D0", "+ L_fwd, w=0.02, z fixed", 1),
    "D2": ("+Fwd (free z)", {42: "v2_learned_z_off_axis"}, "D1", "z becomes a trainable parameter", 1),
    "KA": ("Compact Baseline", {42: "v2_compact_baseline_off_axis"}, "A",
           "shared 1x1 projection to 256 channels before the decoders", 1),
    "KB": ("Compact +IPP", {42: "v2_compact_cell_ipp_off_axis"}, "B", "compact decoder with L_IPP^cell", 1),
}
W10_REPEAT = ("W10", "v2_ipp_w10_off_axis")      # separate seed-42 run of B's configuration; not an extra seed of B


def label(arm):
    return REG[arm][0]


# ----------------------------------------------------------------------------- 0. runs, provenance
print("0. runs and provenance")
SPLITS_SHA = sha256(ROOT / "data" / "splits.json")
RUN = {}                      # arm -> seed -> dict(metrics, common, cfg, prov, dir)
for arm, (lab, dirs, comp, chg, planned) in REG.items():
    RUN[arm] = {}
    for seed, d in dirs.items():
        p = RUNS / d
        m = jload(p / "metrics_test.json")
        cfg = yaml.safe_load((p / "resolved_config.yaml").read_text())
        prov = jload(p / "metrics_test.provenance.json")
        common = jload(p / "metrics_test_common.json") if (p / "metrics_test_common.json").exists() else None
        RUN[arm][seed] = {"dir": d, "m": m, "common": common, "cfg": cfg, "prov": prov}
        check(f"{arm} s{seed}: resolved_config project.seed == directory seed",
              cfg["project"]["seed"] == seed, f"{cfg['project']['seed']} vs {seed}")
        check(f"{arm} s{seed}: provenance train_seed == seed", prov.get("train_seed") == seed, str(prov.get("train_seed")))
        check(f"{arm} s{seed}: provenance splits_sha256 == sha256(data/splits.json)",
              prov.get("splits_sha256") == SPLITS_SHA, "")
        check(f"{arm} s{seed}: split is test", prov.get("split") == "test", str(prov.get("split")))
    check(f"{arm}: number of training runs == planned ({planned})", len(dirs) == planned, f"{len(dirs)}")
    check(f"{arm}: seeds distinct", len(set(dirs)) == len(dirs))

w10 = jload(RUNS / W10_REPEAT[1] / "metrics_test.json")
w10cfg = yaml.safe_load((RUNS / W10_REPEAT[1] / "resolved_config.yaml").read_text())
check("W10 repeat is the configuration of B", w10cfg["loss"]["weights"] == RUN["B"][42]["cfg"]["loss"]["weights"])

# checkpoint hashes / epochs from the collector (these were verified at collection time)
CKPT = {}
for arm in REG:
    bj = RUNS / "benchmark_results" / f"results_arm_{arm}.json"
    if bj.exists():
        for r in jload(bj)["runs"]:
            CKPT[(arm, r["seed"])] = {"checkpoint_sha256": r["checkpoint_sha256"], "checkpoint_epoch": r["checkpoint_epoch"],
                                      "epochs_completed": r.get("training", {}).get("epochs_completed"),
                                      "wall_hours": r.get("training", {}).get("wall_hours"), "source": rel(bj)}
for arm in REG:
    for seed, r in RUN[arm].items():
        c = CKPT.get((arm, seed))
        if c:
            check(f"{arm} s{seed}: collector checkpoint sha == evaluation provenance sha",
                  c["checkpoint_sha256"] == r["prov"]["checkpoint_sha256"])

# ---- configuration metadata (Table 3)
configs_meta = {}
for arm, (lab, dirs, comp, chg, planned) in REG.items():
    s0 = min(dirs)
    cfg = RUN[arm][s0]["cfg"]
    weights = {k: v for k, v in cfg["loss"]["weights"].items() if v}
    fwd = cfg["loss"]["forward_model"]
    hw = RUNS / "benchmark_results" / f"results_hardware_arm_{arm}.json"
    params = None
    if hw.exists():
        params = jload(hw)["statistics"]["params_total"]["mean"]
    configs_meta[arm] = {
        "label": lab, "arm": arm,
        "training_runs": len(dirs), "seeds": sorted(dirs), "run_dirs": {str(s): "runs/" + d for s, d in dirs.items()},
        "comparator": label(comp) if comp else None, "comparator_arm": comp,
        "change_relative_to_comparator": chg,
        "loss_weights_nonzero": weights,
        "architecture": {"encoder": cfg["model"]["encoder"], "decoder_projection_channels": cfg["model"]["decoder_bottleneck"],
                         "decoder_channels": cfg["model"]["decoder_channels"],
                         "amplitude_head": cfg["model"]["amplitude"]["enabled"],
                         "params_total": params,
                         "params_source": rel(hw) if hw.exists() else None},
        "geometry": {"modality": cfg["data"]["modality"], "wavelength_um": cfg["optics"]["wavelength_um"],
                     "pixel_pitch_um": cfg["optics"]["pixel_pitch_x_um"],
                     "refraction_increment_ml_per_g": cfg["optics"]["refraction_increment_ml_per_g"],
                     "propagation_distance_um": fwd["distance_um"], "z_trainable": fwd["learn_distance"],
                     "crop_offset_px": cfg["data"]["crop_offset_px"]},
        "epochs_max": cfg["training"]["epochs"],
        "checkpoints": {str(s): CKPT.get((arm, s)) for s in dirs},
        "source_file": [rel(RUNS / d / "resolved_config.yaml") for d in dirs.values()],
        "field_set": "107 fields (in-line common set)" if arm == "G" else "113 test fields (off-axis)",
    }
configs_meta["W10_repeat_note"] = {
    "description": "A separate seed-42 training of the configuration of +IPP (per-cell); not counted as a seed of "
                   "+IPP (per-cell) and not used for the w=1.0 sweep point.",
    "run_dir": "runs/" + W10_REPEAT[1], "dry_mass_mape": w10["dry_mass_mape"],
    "same_loss_weights_as_B": True, "B_seed42_dry_mass_mape": RUN["B"][42]["m"]["dry_mass_mape"]}
# geometry constants shared by all runs
check("all runs use the same crop offset, z and optics",
      len({json.dumps([c["geometry"]["crop_offset_px"], c["geometry"]["propagation_distance_um"],
                       c["geometry"]["wavelength_um"], c["geometry"]["pixel_pitch_um"]])
           for k, c in configs_meta.items() if k in REG}) == 1)
write("metadata/configurations.json", configs_meta)


# ----------------------------------------------------------------------------- seed statistics
def stat(arm, key, which="m", fs=None):
    """mean, sd, n and an E() entry for a metric of an arm (mean/SD over its training runs)."""
    seeds = sorted(RUN[arm])
    xs = [RUN[arm][s][which][key] for s in seeds]
    src = "; ".join(rel(RUNS / RUN[arm][s]["dir"] / ("metrics_test.json" if which == "m" else "metrics_test_common.json"))
                    for s in seeds)
    fs = fs or ("113 off-axis test fields" if arm != "G" else "107 in-line common fields")
    mu, s_ = mean(xs), sd(xs)
    return {"mean": mu, "sd": s_, "n": len(xs), "per_seed": dict(zip(map(str, seeds), xs)),
            "source_file": src, "source_key": key, "field_set": fs, "seeds": seeds}


def entry(arm, key, units, which="m", fs=None):
    s = stat(arm, key, which, fs)
    calc = None if s["n"] == 1 else "mean and sample SD (ddof=1) over training runs"
    e = E(s["mean"], units, s["source_file"], key, s["field_set"], s["n"], s["seeds"],
          derived_from=None if s["n"] == 1 else [f"{k}: seed {sd_}" for sd_, k in
                                                 zip(s["seeds"], [key] * len(s["seeds"]))], calc=calc)
    e["sd"] = s["sd"]
    e["per_seed"] = s["per_seed"]
    return e


UNITS = {"phase_mae_rad": "rad", "phase_mae_rad_in_cell": "rad", "phase_pearson_r": "1", "phase_ssim": "1",
         "seg_dice": "1", "seg_aji": "1", "seg_boundary_f1": "1", "detection_recall": "1", "detection_precision": "1",
         "detection_f1": "1", "area_mape": "fraction", "dry_mass_mape": "fraction",
         "dry_mass_mape_coverage_adjusted": "fraction", "dry_mass_field_total_mape": "fraction",
         "dry_mass_field_total_bias": "fraction", "dry_mass_cell_pearson_r": "1", "cells_matched": "cells",
         "cells_false_positive": "cells", "cells_reference": "cells", "cells_detected": "cells",
         "forward_residual": "1", "forward_residual_reference": "1", "forward_residual_ratio": "1"}


def units(k):
    return UNITS.get(k, "1")


# ----------------------------------------------------------------------------- 1. baseline + classical (Table 4, 6)
print("1. baseline, classical, Table 4 and 6")
CL113 = jload(RUNS / "conventional_off_axis" / "metrics_test.json")
CL107 = jload(RUNS / "common_fields" / "conventional_off_axis" / "metrics_test.json")
CL107_INLINE = jload(RUNS / "common_fields" / "conventional_gabor" / "metrics_test.json")
CL113_INLINE = jload(RUNS / "conventional_gabor" / "metrics_test.json")
check("classical off-axis: 113 fields, 3186 reference cells",
      CL113["phase_n_images"] == 113 and CL113["cells_reference"] == 3186)
check("classical common off-axis: 107 fields, 3055 reference cells",
      CL107["phase_n_images"] == 107 and CL107["cells_reference"] == 3055)
check("classical common in-line: 107 fields, 3055 reference cells",
      CL107_INLINE["phase_n_images"] == 107 and CL107_INLINE["cells_reference"] == 3055)
for s in SEEDS3:
    check(f"baseline s{s}: 113 fields, 3186 reference cells",
          RUN["A"][s]["m"]["phase_n_images"] == 113 and RUN["A"][s]["m"]["cells_reference"] == 3186)
    check(f"baseline s{s} common: 107 fields, 3055 reference cells",
          RUN["A"][s]["common"]["phase_n_images"] == 107 and RUN["A"][s]["common"]["cells_reference"] == 3055)
check("in-line neural: 107 fields, 3055 reference cells",
      RUN["G"][42]["m"]["phase_n_images"] == 107 and RUN["G"][42]["m"]["cells_reference"] == 3055)

T4_ROWS = [("Phase MAE", "phase_mae_rad"), ("Phase MAE inside cells", "phase_mae_rad_in_cell"),
           ("Phase Pearson r", "phase_pearson_r"), ("Phase SSIM", "phase_ssim"), ("Dice", "seg_dice"),
           ("AJI", "seg_aji"), ("Boundary F1", "seg_boundary_f1"), ("Detection recall", "detection_recall"),
           ("Detection precision", "detection_precision"), ("Projected-area MAPE", "area_mape"),
           ("Dry-mass MAPE (matched cells)", "dry_mass_mape"),
           ("Dry-mass Pearson r, per cell", "dry_mass_cell_pearson_r"),
           ("Field-total dry-mass MAPE", "dry_mass_field_total_mape"),
           ("Cells matched", "cells_matched")]
table4 = {}
for name, k in T4_ROWS:
    neural = entry("A", k, units(k))
    cl = E(CL113[k], units(k), rel(RUNS / "conventional_off_axis" / "metrics_test.json"), k,
           "113 off-axis test fields", 1, None)
    row = {"neural": neural, "classical": cl}
    delta = neural["value"] - CL113[k]
    row["delta_neural_minus_classical"] = E(
        delta, units(k), "derived", k, "113 off-axis test fields", 3, neural["seeds"],
        derived_from=[neural["source_file"], cl["source_file"]], calc="mean(neural over 3 seeds) - classical")
    if k == "cells_matched":
        row["delta_neural_minus_classical"]["note"] = (
            "DESCRIPTIVE ONLY. A matched-cell count is not a quality-error metric and the 2x pooled-SD "
            "resolution rule is not applied to it. Classical is deterministic (single run).")
        row["descriptive_difference_text"] = f"{delta:+.0f} ± {neural['sd']:.0f}"
        row["descriptive_difference_sd_note"] = "SD is the between-seed SD of the neural count; classical has none."
    else:
        row["assessment"] = ("Not applicable: comparator is deterministic (no between-run SD); "
                             "difference reported without a resolution test")
    table4[k] = row
baseline = {"description": "End-to-End Neural Baseline (off-axis), three training runs, 113 test fields.",
            "field_set": "113 off-axis test fields; 3186 reference cells", "n_runs": 3, "seeds": list(SEEDS3),
            "table_4": table4, "per_seed_checkpoints": {str(s): CKPT.get(("A", s)) for s in SEEDS3}}

# Table 6
MEAS = {"Projected area": "area", "Circularity": "circularity", "Dry mass": "dry_mass"}
table6 = {}
for name, q in MEAS.items():
    r = {"matched_cells": {
        "mape": entry("A", f"{q}_mape", "fraction") if f"{q}_mape" in RUN["A"][42]["m"] else None,
        "mape_ci95_seed42": [E(RUN["A"][42]["m"][f"{q}_mape_ci_lower"], "fraction", rel(RUNS / RUN["A"][42]["dir"] / "metrics_test.json"),
                               f"{q}_mape_ci_lower", "113 fields", 1, [42]),
                             E(RUN["A"][42]["m"][f"{q}_mape_ci_upper"], "fraction", rel(RUNS / RUN["A"][42]["dir"] / "metrics_test.json"),
                               f"{q}_mape_ci_upper", "113 fields", 1, [42])]
        if f"{q}_mape_ci_lower" in RUN["A"][42]["m"] else None,
        "coverage_adjusted_mape": entry("A", f"{q}_mape_coverage_adjusted", "fraction")
        if f"{q}_mape_coverage_adjusted" in RUN["A"][42]["m"] else None,
        "pearson_r_per_cell": entry("A", f"{q}_cell_pearson_r", "1"),
        "n_matched_cells": entry("A", f"{q}_n_cells", "cells")},
        "per_field_bland_altman": {
            "relative_bias": entry("A", f"{q}_relative_bias", "fraction") if f"{q}_relative_bias" in RUN["A"][42]["m"] else None,
            "loa_lower": entry("A", f"{q}_loa_lower", "fraction") if f"{q}_loa_lower" in RUN["A"][42]["m"] else None,
            "loa_upper": entry("A", f"{q}_loa_upper", "fraction") if f"{q}_loa_upper" in RUN["A"][42]["m"] else None},
    }
    ft = {k: f"{q}_field_total_{k}" for k in ("bias", "mape", "pearson_r")}
    if all(v in RUN["A"][42]["m"] for v in ft.values()):
        r["field_total"] = {k: entry("A", v, "fraction" if k != "pearson_r" else "1") for k, v in ft.items()}
        r["field_total"]["basis"] = "all predicted cells vs all reference cells per field (not only matched cells)"
    else:
        r["field_total"] = {"bias": "N/A", "mape": "N/A", "pearson_r": "N/A",
                            "reason": "Circularity is not additive, so no field total is defined; no field-total "
                                      "circularity statistic exists in the result files."}
        check("circularity has no field_total keys in metrics_test.json",
              not any(k.startswith("circularity_field_total") for k in RUN["A"][42]["m"]))
    table6[name] = r
check("matched-cell n equals cells_matched for area and dry mass (seed 42)",
      RUN["A"][42]["m"]["area_n_cells"] == RUN["A"][42]["m"]["cells_matched"] ==
      RUN["A"][42]["m"]["dry_mass_n_cells"])
check("Bland-Altman fields are matched-cell based; field totals are on all cells",
      "dry_mass_field_total_mape" in RUN["A"][42]["m"])
baseline["table_6"] = table6
write("baseline/baseline.json", baseline)

classical = {
    "description": "Classical off-axis pipeline (deterministic, single run) and the classical in-line pipeline.",
    "n_runs": 1, "seeds": None,
    "off_axis_113_fields": {k: E(v, units(k), rel(RUNS / "conventional_off_axis" / "metrics_test.json"), k,
                                 "113 off-axis test fields; 3186 reference cells", 1, None)
                            for k, v in CL113.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
    "off_axis_107_common_fields": {k: E(v, units(k), rel(RUNS / "common_fields" / "conventional_off_axis" / "metrics_test.json"),
                                        k, "107 common test fields; 3055 reference cells", 1, None)
                                   for k, v in CL107.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
    "in_line_107_common_fields": {k: E(v, units(k), rel(RUNS / "common_fields" / "conventional_gabor" / "metrics_test.json"),
                                       k, "107 common test fields; 3055 reference cells", 1, None)
                                  for k, v in CL107_INLINE.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
    "reconstruction_distance_um": E(CL113["distance_um"], "um", rel(RUNS / "conventional_off_axis" / "metrics_test.json"),
                                    "distance_um", "113 fields", 1, None),
    "in_line_reconstruction_valid": E(CL107_INLINE["reconstruction_valid"], "bool",
                                      rel(RUNS / "common_fields" / "conventional_gabor" / "metrics_test.json"),
                                      "reconstruction_valid", "107 common fields", 1, None),
    "table_4": table4,
}
write("classical/classical.json", classical)

# ----------------------------------------------------------------------------- 2. in-line / Table 5
print("2. Table 5 (107 common fields)")
T5_ROWS = [("Phase MAE", "phase_mae_rad"), ("Phase MAE inside cells", "phase_mae_rad_in_cell"),
           ("Phase Pearson r", "phase_pearson_r"), ("Phase SSIM", "phase_ssim"), ("Dice", "seg_dice"),
           ("AJI", "seg_aji"), ("Boundary F1", "seg_boundary_f1"), ("Detection recall", "detection_recall"),
           ("Detection precision", "detection_precision"), ("Projected-area MAPE", "area_mape"),
           ("Dry-mass MAPE (matched cells)", "dry_mass_mape"), ("Field-total dry-mass MAPE", "dry_mass_field_total_mape"),
           ("Dry-mass Pearson r, per cell", "dry_mass_cell_pearson_r"), ("Cells matched", "cells_matched")]
summary_csv = {r["entry"]: r for r in rcsv(RUNS / "common_fields" / "common_fields_summary.csv")}
FS107 = "107 common test fields; 3055 reference cells"
table5 = {}
for name, k in T5_ROWS:
    inl = RUN["G"][42]["common"] if RUN["G"][42]["common"] else RUN["G"][42]["m"]
    inl_src = rel(RUNS / RUN["G"][42]["dir"] / "metrics_test_common.json")
    table5[k] = {
        "in_line_neural": E(inl[k], units(k), inl_src, k, FS107, 1, [42]),
        "off_axis_neural": entry("A", k, units(k), which="common", fs=FS107),
        "classical_off_axis": E(CL107[k], units(k), rel(RUNS / "common_fields" / "conventional_off_axis" / "metrics_test.json"),
                                k, FS107, 1, None),
        "classical_in_line": E(CL107_INLINE[k], units(k), rel(RUNS / "common_fields" / "conventional_gabor" / "metrics_test.json"),
                               k, FS107, 1, None),
    }
table5["dry_mass_cell_pearson_r"]["classical_in_line"]["note"] = (
    "Computed over the matched cells of the classical in-line pipeline only (106 cells); not comparable with the other columns.")
table5["cells_matched"]["classical_in_line"]["note"] = "106 matched cells"
# cross-check against the collector's common-field summary CSV
for k, kk in (("phase_mae_rad", "phase_mae_rad"), ("dry_mass_mape", "dry_mass_mape"), ("seg_dice", "seg_dice")):
    gname = [e for e in summary_csv if e.startswith("G")]
    if gname:
        check(f"common_fields_summary.csv G {k} == metrics_test_common.json",
              abs(fnum(summary_csv[gname[0]][kk]) - RUN["G"][42]["common"][k]) < 1e-9)

# Recovered in-cell phase contrast. Definition from scripts/conventional_baseline.py (predict()):
#   per field:  c_f = mean(phase[reference cell pixels]) - mean(phase[background pixels])
#   reported:   median over the test fields of c_f
cl_contrast_off = CL107["recovered_phase_contrast_rad"]
cl_contrast_in = CL107_INLINE["recovered_phase_contrast_rad"]
NEURAL_CONTRAST_FILE = RUNS / "common_fields" / "neural_phase_contrast.json"
contrast = {
    "definition": "median over test fields of [mean phase inside reference cell pixels - mean phase outside them]; "
                  "scripts/conventional_baseline.py predict(), diagnostics['contrast']",
    "classical_off_axis": E(cl_contrast_off, "rad", rel(RUNS / "common_fields" / "conventional_off_axis" / "metrics_test.json"),
                            "recovered_phase_contrast_rad", FS107, 1, None),
    "classical_in_line": E(cl_contrast_in, "rad", rel(RUNS / "common_fields" / "conventional_gabor" / "metrics_test.json"),
                           "recovered_phase_contrast_rad", FS107, 1, None),
}
if NEURAL_CONTRAST_FILE.exists():
    nc = jload(NEURAL_CONTRAST_FILE)
    contrast["in_line_neural"] = E(nc["in_line_neural"]["median_contrast_rad"], "rad", rel(NEURAL_CONTRAST_FILE),
                                   "in_line_neural.median_contrast_rad", FS107, 1, [42])
    xs = [nc["off_axis_neural"][str(s)]["median_contrast_rad"] for s in SEEDS3]
    contrast["off_axis_neural"] = E(mean(xs), "rad", rel(NEURAL_CONTRAST_FILE), "off_axis_neural.<seed>.median_contrast_rad",
                                    FS107, 3, list(SEEDS3), derived_from=[rel(NEURAL_CONTRAST_FILE)],
                                    calc="mean over 3 seeds of the per-seed median contrast")
    contrast["off_axis_neural"]["sd"] = sd(xs)
    contrast["reference_median_contrast_rad"] = nc.get("reference_median_contrast_rad")
else:
    for who in ("in_line_neural", "off_axis_neural"):
        contrast[who] = {"value": "N/A (not yet computed)", "reason":
                         "Needs the trained checkpoints and the phase/mask data, which are not in the repository. "
                         "Run scripts/neural_phase_contrast.py on the server (see AUDIT.md section 7) to create "
                         "runs/common_fields/neural_phase_contrast.json; this compiler picks it up automatically."}
table5["recovered_in_cell_phase_contrast"] = contrast
inline = {"description": "In-Line Neural Configuration: one training run (seed 42), scored on the 107 common test fields; "
                         "Table 5 puts it beside the off-axis three-seed baseline re-scored on the same fields and the "
                         "two deterministic classical pipelines.",
          "field_set": FS107, "n_runs": {"in_line_neural": 1, "off_axis_neural": 3, "classical": 1},
          "seeds": {"in_line_neural": [42], "off_axis_neural": list(SEEDS3)},
          "table_5": table5, "checkpoint": CKPT.get(("G", 42))}

# training-split disclosure (in-line)
tg = (LOGS / "v2_20260930_074055_gpu3" / "train_G.log").read_text()
split = {
    "train_excluded": int(re.search(r"train split: (\d+) of 560 fields excluded", tg).group(1)),
    "train": int(re.search(r"train split: (\d+) samples \| modality=gabor", tg).group(1)),
    "val_excluded": int(re.search(r"val split: (\d+) of 127 fields excluded", tg).group(1)),
    "val": int(re.search(r"val split: (\d+) samples \| modality=gabor", tg).group(1)),
    "test": RUN["G"][42]["m"]["phase_n_images"], "test_reference_cells": RUN["G"][42]["m"]["cells_reference"]}
check("in-line split is 521/122/107", (split["train"], split["val"], split["test"]) == (521, 122, 107), str(split))
_cg = (LOGS / "corrected_20260930_054214" / "common_G.log").read_text()
split["test_excluded"] = int(re.search(r"test split: (\d+) of 113 fields excluded", _cg).group(1))
check("excluded fields: 39 train + 5 val + 6 test == 50 (SNU_01-SNU_50)",
      split["train_excluded"] + split["val_excluded"] + split["test_excluded"] == 50, str(split))
excl_note = re.findall(r"exclu[^\n]*", tg)[:6]
inline["split"] = {k: E(v, "fields", "logs/v2_20260930_074055_gpu3/train_G.log", k, "in-line Gabor", 1, [42])
                   for k, v in split.items()}
inline["split_log_excerpt"] = excl_note
inline["statement"] = ("SNU_01-SNU_50 in-line frames are excluded from train/val/test before training "
                       "(train_G.log), so the in-line model was not trained on the mismatched fields.")
write("inline/inline.json", inline)

# ----------------------------------------------------------------------------- 3. ablation + replicated comparisons (Tables 3, 9, 10)
print("3. Tables 9 and 10")
T9_COLS = [("dry_mass_mape", "matched-cell dry-mass MAPE"), ("dry_mass_mape_coverage_adjusted", "coverage-adjusted dry-mass MAPE"),
           ("dry_mass_field_total_mape", "field-total dry-mass MAPE"), ("area_mape", "projected-area MAPE"),
           ("detection_recall", "recall"), ("detection_precision", "precision"), ("seg_dice", "Dice"),
           ("phase_mae_rad", "phase MAE")]
table9 = {}
ABL_ORDER = ["A", "B", "B1", "B2", "C", "W01", "W03", "B", "W30", "D0", "D1", "D2", "KA", "KB"]
for arm in dict.fromkeys(ABL_ORDER):
    n = len(RUN[arm])
    row = {"label": label(arm), "training_runs": n, "seeds": sorted(RUN[arm]),
           "reporting": "mean +/- SD over training runs" if n >= 3 else "point estimate (single run; no SD assigned)"}
    for k, nm in T9_COLS:
        e = entry(arm, k, units(k))
        if n == 1:
            e.pop("sd", None)
        row[k] = e
    table9[arm] = row
table9["W10_sweep_point"] = {"label": "+IPP (per-cell), w=1.0", "same_as_arm": "B",
                             "note": "the w=1.0 point of the weight sweep is the three-seed +IPP (per-cell) configuration"}

CMP_KEYS = ["dry_mass_mape", "dry_mass_field_total_mape", "dry_mass_field_total_bias", "area_mape", "phase_mae_rad",
            "phase_mae_rad_in_cell", "phase_pearson_r", "seg_dice", "seg_aji", "seg_boundary_f1", "detection_recall",
            "detection_precision", "dry_mass_mape_coverage_adjusted", "cells_false_positive", "dry_mass_cell_pearson_r"]
PAIRS = [(a, REG[a][2]) for a in REG if REG[a][2] and a != "G"] + [("B1", "B")]   # B1 vs B: both three-seed, used in Table 10
table10 = []
for a, b in PAIRS:
    for k in CMP_KEYS:
        if k not in RUN[a][min(RUN[a])]["m"] or k not in RUN[b][min(RUN[b])]["m"]:
            continue
        sa, sb = stat(a, k), stat(b, k)
        r = assess(sa["mean"] - sb["mean"], sa["sd"], sa["n"], sb["sd"], sb["n"])
        r.update(comparator=label(b), configuration=label(a), metric=k, units=units(k),
                 mean_config=sa["mean"], mean_comparator=sb["mean"],
                 source_files=[sa["source_file"], sb["source_file"]], field_set="113 off-axis test fields",
                 seeds_config=sa["seeds"], seeds_comparator=sb["seeds"], derivation="derived",
                 calculation="delta = mean(config) - mean(comparator); pooled SD = sqrt((s1^2+s2^2)/2), "
                             "criterion evaluable only if both have >= 3 runs; ddof=1")
        table10.append(r)
# cross-check with collector's results_comparisons.json for the dry-mass MAPE rows
rc = jload(RUNS / "benchmark_results" / "results_comparisons.json")["comparisons"]
for name, c in rc.items():
    a, b = c["arm"], c["comparator"]
    m = c["metrics"].get("dry_mass_mape")
    mine = [x for x in table10 if x["metric"] == "dry_mass_mape" and x["configuration"] == label(a)
            and x["comparator"] == label(b)] if a in REG and b in REG and a != "G" else []
    if m and mine:
        check(f"collector comparison {name}: delta and threshold agree",
              abs(mine[0]["delta"] - m["difference"]) < 1e-9 and
              (m["threshold"] is None or abs(mine[0]["two_x_pooled_sd"] - m["threshold"]) < 1e-9))
# every single-run row must be non-evaluable
check("no n=1 comparison is assessed", all(not r["criterion_evaluable"] for r in table10
                                           if r["n_runs_config"] < 3 or r["n_runs_comparator"] < 3))
ipp = {"description": "Ablation (Table 9) and replicated/unreplicated comparisons (Table 10). Assessment wording: "
                      "'Exceeds 2x pooled SD' / 'Within 2x pooled SD' / 'Not estimable from the available runs'.",
       "field_set": "113 off-axis test fields; 3186 reference cells",
       "table_9": table9, "table_10": table10,
       "w10_repeat_note": configs_meta["W10_repeat_note"],
       "gradient_path": None}
write("ipp/ipp.json", ipp)

# ----------------------------------------------------------------------------- 4. forward model / z / amplitude (Table 11)
print("4. Table 11, forward model, z")
zc = jload(RUNS / "z_calibration.json")
amp_keys = ["amplitude_mae", "amplitude_unity_mae", "amplitude_in_cell_mae", "amplitude_bias", "amplitude_pearson_r",
            "amplitude_pred_mean", "amplitude_pred_std", "amplitude_ref_mean", "amplitude_rmse"]
present = [k for k in RUN["D0"][42]["m"] if k.startswith("amplitude")]
amp = {"description": "Amplitude output (+Amplitude, +Fwd fixed z, +Fwd free z); one training run each; 113 test fields.",
       "n_runs": 1, "seeds": [42], "field_set": "113 off-axis test fields", "table_11a": {}, "table_11b": {}, "table_11c": {}}
for arm in ("D0", "D1", "D2"):
    amp["table_11a"][arm] = {"label": label(arm)}
    for k in present:
        amp["table_11a"][arm][k] = entry(arm, k, "1")
    mm = RUN[arm][42]["m"]
    if "amplitude_unity_mae" in mm and "amplitude_mae" in mm:
        amp["table_11a"][arm]["ratio_mae_over_unity_mae"] = E(
            mm["amplitude_mae"] / mm["amplitude_unity_mae"], "1", "derived", "amplitude_mae / amplitude_unity_mae",
            "113 fields", 1, [42], derived_from=["amplitude_mae", "amplitude_unity_mae"],
            calc="amplitude_mae / amplitude_unity_mae (<1: closer to the reference than A=1)")
amp["amplitude_keys_available"] = present
for arm in ("A", "B", "D0", "D1", "D2"):
    row = {"label": label(arm)}
    for k in ("forward_residual", "forward_residual_reference", "forward_residual_ratio", "phase_mae_rad_in_cell"):
        row[k] = entry(arm, k, units(k))
    row["forward_distance_um"] = entry(arm, "forward_distance_um", "um")
    amp["table_11b"][arm] = row
# in-line neural residual (off-axis tables do not include it; reported for completeness, 107 fields)
amp["table_11b"]["G_in_line_107_fields"] = {
    "label": label("G"), **{k: entry("G", k, units(k), fs="107 in-line test fields") for k in
                            ("forward_residual", "forward_residual_reference", "forward_residual_ratio")}}

# phase-scale probes. "Fields favouring reference phase" = number of fields on which the SCALED phase gives the
# HIGHER residual (scaled - reference > 0). Validation counts = win_rate * images (calibrate_z.py: (difference>0).mean()).
probes = {}
for geom, nm in (("off_axis", "Off-axis, 32 validation fields"), ("gabor", "In-line, 32 validation fields")):
    d = zc[geom]["discrimination"]
    probes[nm] = {"source_file": "runs/z_calibration.json", "field_set": f"{d['images']} validation fields",
                  "reference_residual": E(d["floor"], "1", "runs/z_calibration.json", f"{geom}.discrimination.floor",
                                          f"{d['images']} validation fields", 1, None)}
    for s in ("0.9", "0.5"):
        n_fav = round(d["win_rates"][f"scaled_{s}"] * d["images"])
        check(f"{geom} scaled_{s}: win_rate*images is an integer",
              abs(d["win_rates"][f"scaled_{s}"] * d["images"] - n_fav) < 1e-9)
        probes[nm][f"scale_{s}"] = {
            "margin": E(d["margins"][f"scaled_{s}"], "1", "runs/z_calibration.json", f"{geom}.discrimination.margins.scaled_{s}",
                        f"{d['images']} validation fields", 1, None),
            "fields_favouring_reference_phase": E(
                n_fav, "fields", "runs/z_calibration.json", f"{geom}.discrimination.win_rates.scaled_{s} x images",
                f"{d['images']} validation fields", 1, None, derived_from=[f"{geom}.discrimination.win_rates.scaled_{s}",
                                                                           f"{geom}.discrimination.images"],
                calc="win_rate x images; win_rate = mean(scaled residual - reference residual > 0)"),
            "of": d["images"]}
for mode, nm in (("global", "Off-axis, 113 test fields, global surface"),
                 ("per_field", "Off-axis, 112 test fields, per-field surfaces")):
    p = RUNS / f"amplitude_sensitivity_off_axis_{mode}.csv"
    rows = rcsv(p)
    n = len(rows)
    ref = [fnum(r["phase_reference"]) for r in rows]
    probes[nm] = {"source_file": rel(p), "field_set": f"{n} test fields",
                  "reference_residual": E(mean(ref), "1", rel(p), "phase_reference", f"{n} test fields", 1, None,
                                          derived_from=[rel(p)], calc="mean over fields of phase_reference")}
    for s in ("0.9", "0.5"):
        sc = [fnum(r[f"phase_scaled_{s}"]) for r in rows]
        n_fav = sum(1 for a, b in zip(sc, ref) if a > b)
        n_tie = sum(1 for a, b in zip(sc, ref) if a == b)
        probes[nm][f"scale_{s}"] = {
            "margin": E(mean(sc) - mean(ref), "1", rel(p), f"phase_scaled_{s} - phase_reference", f"{n} test fields", 1, None,
                        derived_from=[rel(p)], calc="mean(scaled) - mean(reference)"),
            "fields_favouring_reference_phase": E(
                n_fav, "fields", rel(p), f"count(phase_scaled_{s} > phase_reference)", f"{n} test fields", 1, None,
                derived_from=[rel(p)], calc="number of fields whose scaled-phase residual exceeds the reference-phase residual"),
            "ties": n_tie, "of": n}
_gl = rcsv(RUNS / "amplitude_sensitivity_off_axis_global.csv")
_pf = {r["batch"] for r in rcsv(RUNS / "amplitude_sensitivity_off_axis_per_field.csv")}
_gc = [r for r in _gl if r["batch"] in _pf]
_ref = [fnum(r["phase_reference"]) for r in _gc]
_nm = "Off-axis, 112 test fields, global surface (same fields as per-field row)"
probes[_nm] = {"source_file": "runs/amplitude_sensitivity_off_axis_global.csv restricted to batches in the per-field CSV",
               "field_set": f"{len(_gc)} test fields",
               "reference_residual": E(mean(_ref), "1", "runs/amplitude_sensitivity_off_axis_global.csv", "phase_reference",
                                       f"{len(_gc)} test fields", 1, None, derived_from=["batches shared with the per-field CSV"],
                                       calc="mean over the 112 shared fields")}
for s in ("0.9", "0.5"):
    _sc = [fnum(r[f"phase_scaled_{s}"]) for r in _gc]
    probes[_nm][f"scale_{s}"] = {
        "margin": E(mean(_sc) - mean(_ref), "1", "runs/amplitude_sensitivity_off_axis_global.csv", f"phase_scaled_{s} - phase_reference",
                    f"{len(_gc)} test fields", 1, None, derived_from=["shared batches"], calc="mean(scaled) - mean(reference), 112 shared fields"),
        "fields_favouring_reference_phase": E(sum(1 for a, b in zip(_sc, _ref) if a > b), "fields",
                                              "runs/amplitude_sensitivity_off_axis_global.csv", f"count(phase_scaled_{s} > phase_reference)",
                                              f"{len(_gc)} test fields", 1, None, derived_from=["shared batches"],
                                              calc="count over the 112 shared fields"), "of": len(_gc)}
amp["table_11c"] = probes
amp["configured_tolerance"] = E(0.01, "1", "config/base.yaml", "loss.forward_model.discrimination_tolerance",
                                "n/a", None, None)
cfgtxt = (ROOT / "config" / "base.yaml").read_text()
check("configured tolerance in config/base.yaml is 0.01",
      re.search(r"discrimination_tolerance:\s*0\.01\b", cfgtxt) is not None)
check("validation discrimination tolerance stored with z calibration is 0.01",
      zc["off_axis"]["discrimination"]["tolerance"] == 0.01 and zc["gabor"]["discrimination"]["tolerance"] == 0.01)
amp["terminology"] = "'configured tolerance' (never 'preregistered'); 'Fields favouring reference phase' (not 'Fields worse')"
write("amplitude/amplitude.json", amp)

# forward model / z
fm = {"scope": "All statements apply to the implemented operator (angular-spectrum propagation with the stored "
               "aberration surface and amplitude convention); no general identifiability claim is made.",
      "recording_distance_um": E(RUN["A"][42]["cfg"]["loss"]["forward_model"]["distance_um"], "um",
                                 rel(RUNS / RUN["A"][42]["dir"] / "resolved_config.yaml"),
                                 "loss.forward_model.distance_um", "n/a", None, None),
      "z_calibration": {}}
for geom in ("off_axis", "gabor"):
    g = zc[geom]
    c = g["curve"]
    import numpy as _np
    dist, res = _np.array(c["distances_um"]), _np.array(c["forward_residual"])
    g_ = {"grid_minimum_z_um": E(float(dist[res.argmin()]), "um", "runs/z_calibration.json", f"{geom}.curve.distances_um[argmin forward_residual]",
                                 f"{c['images']} scan fields", 1, None, derived_from=["distances_um", "forward_residual"],
                                 calc="distance at the minimum of the pooled forward residual"),
          "grid_step_um": E(g["grid_step_um"], "um", "runs/z_calibration.json", f"{geom}.grid_step_um", "", 1, None),
          "per_image_best_z_um": E(c["per_image_best_z_um"], "um", "runs/z_calibration.json", f"{geom}.curve.per_image_best_z_um",
                                   f"{c['images']} scan fields", 1, None),
          "per_image_median_z_um": E(g["per_image_median_z_um"], "um", "runs/z_calibration.json", f"{geom}.per_image_median_z_um", "", 1, None),
          "per_image_iqr_um": E(g["per_image_iqr_um"], "um", "runs/z_calibration.json", f"{geom}.per_image_iqr_um", "", 1, None),
          "images_agree": g["images_agree"], "identifiable_flag_in_file": g["identifiable"],
          "forward_model_verdict": g["forward_model_verdict"],
          "reference_residual_validation": E(g["discrimination"]["floor"], "1", "runs/z_calibration.json",
                                             f"{geom}.discrimination.floor", f"{g['discrimination']['images']} validation fields", 1, None),
          "residual_margin_0.9x": E(g["discrimination"]["margins"]["scaled_0.9"], "1", "runs/z_calibration.json",
                                    f"{geom}.discrimination.margins.scaled_0.9", "32 validation fields", 1, None),
          "residual_margin_0.5x": E(g["discrimination"]["margins"]["scaled_0.5"], "1", "runs/z_calibration.json",
                                    f"{geom}.discrimination.margins.scaled_0.5", "32 validation fields", 1, None)}
    fm["z_calibration"][geom] = g_
rec = fm["recording_distance_um"]["value"]
check("recording distance is 33.77 um", abs(rec - 33.77) < 1e-9, str(rec))
gm = fm["z_calibration"]["gabor"]["grid_minimum_z_um"]["value"]
check("in-line grid minimum within one grid step of the recording distance",
      abs(gm - rec) < zc["gabor"]["grid_step_um"], f"{gm}")
check("in-line grid minimum is not near 50.5 um (stale statement)", abs(gm - 50.5) > 10, f"{gm}")
off_best = zc["off_axis"]["curve"]["per_image_best_z_um"]
check("off-axis per-field best z has large spread (does not constrain z)",
      max(off_best) - min(off_best) > 100, str(off_best))
d2 = RUN["D2"][42]["m"]
fm["free_z"] = {
    "learned_distance_at_selected_checkpoint_um": E(d2["forward_distance_um"], "um", rel(RUNS / RUN["D2"][42]["dir"] / "metrics_test.json"),
                                                    "forward_distance_um", "113 fields", 1, [42]),
    "selected_checkpoint_epoch": E(CKPT[("D2", 42)]["checkpoint_epoch"], "epoch", CKPT[("D2", 42)]["source"], "runs[0].checkpoint_epoch", "", 1, [42]),
    "fixed_z_distance_um": E(RUN["D1"][42]["m"]["forward_distance_um"], "um", rel(RUNS / RUN["D1"][42]["dir"] / "metrics_test.json"),
                             "forward_distance_um", "113 fields", 1, [42]),
}
rt = (RUNS / "RESULTS.md").read_text()
mm = re.search(r"initial \*\*\+([0-9.]+) um\*\*, final \*\*\+([0-9.]+) um\*\*", rt)
if mm:
    fm["free_z"]["trajectory_initial_after_epoch1_um"] = E(float(mm.group(1)), "um", "runs/RESULTS.md",
                                                           "learned propagation distance, initial", "", 1, [42])
    fm["free_z"]["trajectory_final_um"] = E(float(mm.group(2)), "um", "runs/RESULTS.md",
                                            "learned propagation distance, final", "", 1, [42])
write("forward_model/forward_model.json", fm)

# ----------------------------------------------------------------------------- 5. decomposition (Table 7) -- independent recompute
print("5. decomposition (Table 7), recomputed from the per-cell and per-field CSVs")
dcells = rcsv(RUNS / "diagnostics" / "decomposition_cells.csv")
dfields = rcsv(RUNS / "diagnostics" / "decomposition_fields.csv")
dsum = rcsv(RUNS / "diagnostics" / "decomposition_summary.csv")
DS = {(r["run"], r["level"], r["group"]): r for r in dsum}


def gm(xs):
    xs = [x for x in xs if x > 0 and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def med_abs_log(xs):
    return statistics.median(abs(math.log(x)) for x in xs if x > 0)


RUNS_DCMP = sorted({r["run"] for r in dcells})
dec_runs = {}
for run in RUNS_DCMP:
    cs = [c for c in dcells if c["run"] == run and c["status"] == "ok"]
    fs = [f for f in dfields if f["run"] == run and f["status"] == "ok"]
    res = {}
    for grp in ("all", "interior", "edge"):
        sel = [c for c in cs if grp == "all" or (c["edge"] == "1") == (grp == "edge")]
        dom, pha, tot = ([fnum(c[k]) for c in sel] for k in ("domain", "phase", "total"))
        n_ref = sum(int(fnum(f["n_ref_cells"])) for f in fs)
        n_edge = sum(int(fnum(f["n_ref_edge"])) for f in fs)
        denom = n_ref if grp == "all" else (n_edge if grp == "edge" else n_ref - n_edge)
        res[("cell", grp)] = {"n_matched": len(sel), "n_reference": denom, "recall": len(sel) / denom,
                              "domain_gm": gm(dom), "phase_gm": gm(pha), "total_gm": gm(tot),
                              "domain_med_abs_log": med_abs_log(dom), "phase_med_abs_log": med_abs_log(pha),
                              "total_med_abs_log": med_abs_log(tot),
                              "mape": mean([abs(fnum(c["dry_mass_pg_pred"]) / fnum(c["dry_mass_pg_ref"]) - 1) for c in sel])}
        # identity: domain x phase == total per cell
        check(f"{run} cell/{grp}: domain x phase == total (max abs err < 1e-6)",
              max(abs(a * b - t) for a, b, t in zip(dom, pha, tot)) < 1e-6)
    dom, pha, tot = ([fnum(f[k]) for f in fs] for k in ("domain", "phase", "total"))
    res[("field", "all")] = {"n_fields": len(fs), "domain_gm": gm(dom), "phase_gm": gm(pha), "total_gm": gm(tot),
                             "domain_med_abs_log": med_abs_log(dom), "phase_med_abs_log": med_abs_log(pha),
                             "total_med_abs_log": med_abs_log(tot)}
    # definition checks at field level: domain = S_domain/S_ref, phase = S_pred/S_domain
    check(f"{run} field: domain == S(Omega_pred,phi_ref)/S(Omega_ref,phi_ref) and phase == S(Omega_pred,phi_pred)/S(Omega_pred,phi_ref)",
          max(abs(fnum(f["domain"]) - fnum(f["S_domain"]) / fnum(f["S_ref"])) for f in fs) < 1e-6 and
          max(abs(fnum(f["phase"]) - fnum(f["S_pred"]) / fnum(f["S_domain"])) for f in fs) < 1e-6)
    # agreement with the stored summary
    for (lvl, grp), v in res.items():
        s_ = DS[(run, lvl, grp)]
        for k in ("domain_gm", "phase_gm", "total_gm", "domain_med_abs_log", "phase_med_abs_log", "total_med_abs_log"):
            check(f"{run} {lvl}/{grp} {k}: recompute == decomposition_summary.csv",
                  abs(v[k] - fnum(s_[k])) < 1e-9, f"{v[k]} vs {s_[k]}")
    dec_runs[run] = res

dec = {"definitions": {
    "domain_factor": "S(Omega_pred, phi_ref) / S(Omega_ref, phi_ref)",
    "phase_factor": "S(Omega_pred, phi_pred) / S(Omega_pred, phi_ref)",
    "total": "domain factor x phase factor",
    "matched_cell_level": "predicted and reference instances paired (IoU matching); edge/interior by the reference instance",
    "field_level": "whole predicted foreground vs whole reference foreground; missed and spurious cells enter the domain factor",
    "edge_rule": "reference instance with any pixel at index < edge_margin_px or >= size - edge_margin_px "
                 "(config/diagnostics.yaml edge_margin_px = 4, i.e. within 3 px of the border; strict inequality in "
                 "decompose_mass_error._edge_labels)",
    "summary_statistic": "geometric mean (exact: domain x phase = total) and median absolute log ratio (spread)"},
       "n_definitions": {"training_runs": "number of trained models (3 for the baseline and IPP configurations)",
                         "N_matched_cells": "matched instance pairs per run", "N_fields": "fields"},
       "field_set": "113 off-axis test fields; 3186 reference cells; 1452 edge / 1734 interior reference cells",
       "per_run": {run: {f"{lvl}.{grp}": v for (lvl, grp), v in res.items()} for run, res in dec_runs.items()},
       "by_config": {}, "comparisons": []}
cfg_runs = {"A": [f"A_s{s}" for s in SEEDS3], "B": [f"B_s{s}" for s in SEEDS3], "B1": [f"B1_s{s}" for s in SEEDS3],
            "classical": ["classical"]}
DSTAT = {}
for cfg, rs in cfg_runs.items():
    dec["by_config"][cfg] = {"training_runs": len(rs) if cfg != "classical" else 1,
                             "note": "classical: deterministic, single run (not repeated)" if cfg == "classical" else None}
    for lvl, grp in (("cell", "all"), ("cell", "interior"), ("cell", "edge"), ("field", "all")):
        keys = [k for k in dec_runs[rs[0]][(lvl, grp)]]
        d = {}
        for k in keys:
            xs = [dec_runs[r][(lvl, grp)][k] for r in rs]
            d[k] = {"mean": mean(xs), "sd": sd(xs) if len(xs) >= 3 else None, "n_runs": len(xs)}
            DSTAT[(cfg, f"{lvl}.{grp}", k)] = (mean(xs), sd(xs), len(xs))
        dec["by_config"][cfg][f"{lvl}.{grp}"] = d
# cross-check with decomposition_by_config.csv
for r in rcsv(RUNS / "diagnostics" / "decomposition_by_config.csv"):
    if r["config"] in ("A", "B", "B1"):
        for k in ("domain_gm", "phase_gm", "total_gm"):
            m_, s_, n_ = DSTAT[(r["config"], f"{r['level']}.{r['group']}", k)]
            check(f"by_config {r['config']} {r['level']}/{r['group']} {k}",
                  abs(m_ - fnum(r[f"{k}_mean"])) < 1e-9 and abs(s_ - fnum(r[f"{k}_sd"])) < 1e-9)
for a in ("B", "B1"):
    for lvl, grp in (("cell", "all"), ("cell", "interior"), ("cell", "edge"), ("field", "all")):
        for k in ("domain_gm", "phase_gm", "total_gm", "domain_med_abs_log", "phase_med_abs_log", "total_med_abs_log", "recall"):
            if (a, f"{lvl}.{grp}", k) not in DSTAT:
                continue
            ma, sa, na = DSTAT[(a, f"{lvl}.{grp}", k)]
            mb, sb, nb = DSTAT[("A", f"{lvl}.{grp}", k)]
            r = assess(ma - mb, sa, na, sb, nb)
            r.update(configuration=label(a), comparator=label("A"), level=lvl, group=grp, metric=k,
                     derivation="derived")
            dec["comparisons"].append(r)
# matched counts: decomposition recount vs stored evaluation (documented difference)
dec["matched_count_reconciliation"] = {
    run: {"decomposition_recount": dec_runs[run][("cell", "all")]["n_matched"],
          "stored_evaluation_cells_matched": RUN[run.split("_")[0]][int(run.split("_s")[1])]["m"]["cells_matched"]}
    for run in RUNS_DCMP if run != "classical"}
dec["matched_count_reconciliation"]["classical"] = {
    "decomposition_recount": dec_runs["classical"][("cell", "all")]["n_matched"],
    "stored_evaluation_cells_matched": CL113["cells_matched"]}
mx = max(abs(v["decomposition_recount"] - v["stored_evaluation_cells_matched"]) for v in dec["matched_count_reconciliation"].values())
check("decomposition matched-cell recount within 3 cells of the stored evaluation for every run", mx <= 3, f"max {mx}")
write("decomposition/decomposition.json", dec)

# ----------------------------------------------------------------------------- 6. edge / interior and false positives (Table 8)
print("6. Table 8")
H = W = 900
FP_MARGIN = 15
LOC = {}
for run, res in dec_runs.items():
    LOC[run] = {"edge_recall": res[("cell", "edge")]["recall"], "interior_recall": res[("cell", "interior")]["recall"],
                "edge_mape": res[("cell", "edge")]["mape"], "interior_mape": res[("cell", "interior")]["mape"]}
UNM = {}
for arm in ("A", "B", "B1"):
    for s, r in RUN[arm].items():
        UNM[f"{arm}_s{s}"] = RUNS / r["dir"] / "unmatched_test.csv"
UNM["classical"] = RUNS / "conventional_off_axis" / "unmatched_test.csv"
for run, p in UNM.items():
    rows = rcsv(p)
    fps = [r for r in rows if r["kind"] != "missed_reference"]
    near = [r for r in fps if min(fnum(r["centroid_y"]), fnum(r["centroid_x"]), H - 1 - fnum(r["centroid_y"]),
                                  W - 1 - fnum(r["centroid_x"])) <= FP_MARGIN]
    LOC[run]["fp_all"], LOC[run]["fp_near_boundary"] = len(fps), len(near)
    stored = RUN[run.split("_")[0]][int(run.split("_s")[1])]["m"]["cells_false_positive"] if run != "classical" else CL113["cells_false_positive"]
    check(f"{run}: unmatched.csv false positives == metrics cells_false_positive", len(fps) == stored, f"{len(fps)} vs {stored}")
boundary = {"description": "Edge/interior stratification and false positives near the field boundary (Table 8) and the "
                           "boundary-displacement / synthetic analyses (Table 12).",
            "field_set": "113 off-axis test fields", "edge_definition": dec["definitions"]["edge_rule"],
            "false_positive_boundary_margin_px": FP_MARGIN, "table_8": {}, "table_8_comparisons": []}
LS = {}
for cfg, rs in cfg_runs.items():
    boundary["table_8"][cfg] = {"label": label(cfg) if cfg in REG else "Classical pipeline", "training_runs": len(rs)}
    for k in ("edge_recall", "interior_recall", "edge_mape", "interior_mape", "fp_all", "fp_near_boundary"):
        xs = [LOC[r][k] for r in rs]
        LS[(cfg, k)] = (mean(xs), sd(xs), len(xs))
        boundary["table_8"][cfg][k] = {"mean": mean(xs), "sd": sd(xs) if len(xs) >= 3 else None, "n_runs": len(xs),
                                       "per_run": dict(zip(rs, xs)), "derivation": "derived",
                                       "source_file": "runs/diagnostics/decomposition_cells.csv; runs/diagnostics/decomposition_fields.csv; "
                                                      "runs/*/unmatched_test.csv"}
for a in ("B", "B1"):
    for k in ("edge_recall", "interior_recall", "edge_mape", "interior_mape", "fp_all", "fp_near_boundary"):
        ma, sa, na = LS[(a, k)]
        mb, sb, nb = LS[("A", k)]
        r = assess(ma - mb, sa, na, sb, nb)
        r.update(configuration=label(a), comparator=label("A"), metric=k, derivation="derived")
        boundary["table_8_comparisons"].append(r)
boundary["subgroup_note"] = ("Subgroup changes are descriptive; no causal mechanism is inferred from them.")
refcells = {"edge": int(DS[("A_s42", "cell", "edge")]["n_reference_cells"].split(".")[0]),
            "interior": int(DS[("A_s42", "cell", "interior")]["n_reference_cells"].split(".")[0])}
check("1452 edge + 1734 interior == 3186 reference cells", refcells["edge"] == 1452 and refcells["interior"] == 1734 and
      refcells["edge"] + refcells["interior"] == 3186, str(refcells))
boundary["reference_cells"] = refcells

# Table 12: boundary displacement + synthetic validation
ep = rcsv(RUNS / "error_propagation_summary.csv")
_ratios = [fnum(r["mass_over_area"]) for r in ep if int(float(r["shift_px"])) != 0]
boundary["median_mass_over_area_ratio"] = E(statistics.median(_ratios), "1", "runs/error_propagation_summary.csv", "mass_over_area",
                                            "113 off-axis test fields", 1, None, derived_from=["mass_over_area at shifts +-1..5 px"],
                                            calc="median over the ten non-zero displacements")
boundary["table_12_boundary_displacement"] = {
    "source_file": "runs/error_propagation_summary.csv",
    "rows": [{k: (fnum(v) if k != "kind" else v) for k, v in row.items()} for row in ep],
    "derivation": "direct; see scripts/error_propagation.py"}
sv = jload(RUNS / "benchmark_results" / "results_synthetic_validation.json")
syn = {}
for k in ("mass_abs_relative_error_mean", "area_abs_relative_error_mean", "field_total_mass_error_mean",
          "field_total_mass_abs_error_mean", "cells_measured", "cells_placed"):
    xs = [r["metrics"][k] for r in sv["runs"]]
    syn[k] = E(mean(xs), "fraction" if "error" in k else "cells", "runs/benchmark_results/results_synthetic_validation.json",
               f"runs[*].metrics.{k}", "3 synthetic sets", len(xs), [r.get("seed") for r in sv["runs"]],
               derived_from=[f"runs[{i}].metrics.{k}" for i in range(len(xs))], calc="mean over 3 synthetic sets")
    syn[k]["sd"] = sd(xs)
boundary["table_12_synthetic_floor"] = syn
write("boundary_sensitivity/boundary.json", boundary)

# ----------------------------------------------------------------------------- 7. benchmarking (Table 13)
print("7. benchmarking")
_blk = cfgtxt.split("  benchmark:\n", 1)[1].split("\n\n", 1)[0]
_bm = [int(re.search(rf"{k}:\s*(\d+)", _blk).group(1)) for k in ("warmup_runs", "timed_runs", "batch_size")]
BENCH_WARMUP, BENCH_TIMED, BENCH_BATCH = _bm
check("config deploy.benchmark: 50 warmup, 500 timed runs", (BENCH_WARMUP, BENCH_TIMED) == (50, 500), str(_bm))
bench = {"description": "Computational benchmarking (Table 13). Sources are the collector files "
                        "runs/benchmark_results/results_hardware_arm_*.json (one benchmark process per training seed; "
                        "values are means/SD over those sessions). Not rerun.",
         "device": None, "arms": {}}
for arm in ("A", "B", "D0", "KA", "KB"):
    p = RUNS / "benchmark_results" / f"results_hardware_arm_{arm}.json"
    h = jload(p)
    st = h["statistics"]
    first = h["runs"][0]
    bench["device"] = bench["device"] or first["non_numeric"].get("gpu_name")
    a = {"label": label(arm), "benchmark_sessions": h["n_runs"], "seeds": h["seeds"],
         "gpu": first["non_numeric"].get("gpu_name"), "torch_version": first["non_numeric"].get("torch_version"),
         "input_size": E(st["input_size"]["mean"], "px", rel(p), "statistics.input_size.mean", "", h["n_runs"], h["seeds"]),
         "batch_size": BENCH_BATCH,
         "params_total": E(st["params_total"]["mean"], "parameters", rel(p), "statistics.params_total.mean", "", h["n_runs"], h["seeds"]),
         "gmacs": E(st["gmacs"]["mean"], "GMACs", rel(p), "statistics.gmacs.mean", "", h["n_runs"], h["seeds"]),
         "runtimes": {}}
    for rt in ("pytorch_fp32", "onnx_fp32", "pytorch_fp16", "onnx_fp16"):
        d = {}
        for q, u in (("latency_mean_ms", "ms"), ("latency_p50_ms", "ms"), ("latency_p99_ms", "ms"),
                     ("latency_p99_over_p50", "1"), ("fps", "1/s"), ("timed_runs", "runs"), ("weights_mb", "MB"),
                     ("peak_activation_mb", "MB"), ("peak_allocated_mb", "MB")):
            key = f"{rt}.{q}"
            if key in st:
                d[q] = E(st[key]["mean"], u, rel(p), f"statistics.{key}.mean", "", st[key]["n"], h["seeds"])
                d[q]["sd"] = st[key]["std"]
                d[q]["session_min"], d[q]["session_max"] = st[key]["min"], st[key]["max"]
        a["runtimes"][rt] = d
    bench["arms"][arm] = a
    # recompute mean latency from the per-session values
    xs = [r["metrics"]["pytorch_fp32.latency_mean_ms"] for r in h["runs"]]
    check(f"{arm}: collector statistics pytorch_fp32 latency mean == mean of sessions",
          abs(mean(xs) - st["pytorch_fp32.latency_mean_ms"]["mean"]) < 1e-9)
check("benchmark: 500 timed runs per runtime (all arms)", all(
    bench["arms"][a]["runtimes"]["pytorch_fp32"]["timed_runs"]["value"] == 500 for a in bench["arms"]))
bench["protocol"] = {"warmup_runs": BENCH_WARMUP, "timed_runs": BENCH_TIMED, "batch_size": BENCH_BATCH, "source_file": "config/base.yaml deploy.benchmark"}
bench["protocol_source"] = "holoqpi/deploy/benchmark.py; config/base.yaml deploy.* (warmup and timed-run counts)"
write("benchmarking/benchmarking.json", bench)

# ----------------------------------------------------------------------------- 8. gradient path, registration, crop, label audit
print("8. metadata")
g = rcsv(RUNS / "gradient_path_b_cell_ipp_512.csv")
r = [fnum(x["ratio_at_weight_1"]) for x in g]
meta_grad = {"median": E(statistics.median(r), "1", "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "30 training batches (512x512 crops)", 1, [42],
                         derived_from=["ratio_at_weight_1"], calc="median over rows"),
             "min": E(min(r), "1", "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "30 batches", 1, [42],
                      derived_from=["ratio_at_weight_1"], calc="min over rows"),
             "max": E(max(r), "1", "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "30 batches", 1, [42],
                      derived_from=["ratio_at_weight_1"], calc="max over rows"),
             "batches": E(len(r), "batches", "runs/gradient_path_b_cell_ipp_512.csv", "rows", "", 1, [42]),
             "config": g[0]["config"], "train_crop": g[0]["train_crop"],
             "note": "Source of truth is the CSV. A previously quoted median 0.302 (range 0.18-0.67) is NOT supported by it."}
check("gradient path: 30 batches, median 0.373, range 0.243-0.655 (rounded)",
      len(r) == 30 and round(meta_grad["median"]["value"], 3) == 0.373 and round(min(r), 3) == 0.243 and round(max(r), 3) == 0.655)
check("gradient path: the value 0.302 is not supported by the CSV", abs(meta_grad["median"]["value"] - 0.302) > 0.05)
ipp["gradient_path"] = meta_grad
write("ipp/ipp.json", ipp)

reg_src = "diagnostics/registration_summary.json, hologram_registration.csv (NOT in the repository)"
registration = {"verified": False,
                "reason": "The diagnostics archive (registration_summary.json, hologram_registration.csv, "
                          "classical_phase_shift.csv, hologram_inventory.csv) is not present in this repository, so these "
                          "values cannot be re-derived or checked here. They are carried over as author-supplied.",
                "values": {k: {"value": v, "source_file": reg_src, "verified": False} for k, v in {
                    "fields_total": 800, "correctly_paired_fields": 750, "mismatched_fields_SNU_01_to_SNU_50": 50,
                    "matched_pairs_within_1px": 734, "control_pairs_within_1px": 3, "auc_correlation": 0.975,
                    "auc_registration_error": 0.973, "lowpass_only_auc": 0.506, "paired_max_shift_px": 4.7,
                    "snu_r_min": -0.13, "snu_r_max": 0.18, "paired_r_min": 0.47, "control_r_max": 0.25}.items()},
                "flag": "SNU_01-50 correlation range: -0.13 to 0.18 here vs 0.09-0.18 in config/base.yaml comments.",
                "split_from_train_G_log": split}
cb = (ROOT / "config" / "base.yaml").read_text()
registration["config_base_yaml_mentions_0.09_to_0.18"] = bool(re.search(r"0\.09", cb))
check("750 + 50 == 800 registration fields", 750 + 50 == 800)

# ---- registration provenance (Sec. 4.1): verified against the diagnostics archive when it is present
REGDIR = RUNS / "diagnostics"
REG_FILES = ("registration_summary.json", "hologram_registration.csv", "classical_phase_shift.csv", "hologram_inventory.csv")
if all((REGDIR / f).exists() for f in REG_FILES):
    import numpy as _np
    (OUT / "registration").mkdir(parents=True, exist_ok=True)
    for f in REG_FILES:
        shutil_dst = OUT / "registration" / f
        shutil_dst.write_bytes((REGDIR / f).read_bytes())
    rows_ = rcsv(REGDIR / "hologram_registration.csv")

    def _auc(pos, neg):
        pos, neg = _np.asarray(pos, float), _np.asarray(neg, float)
        gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        return float(gt / (len(pos) * len(neg)))

    snu = {f"SNU_{i:02d}" for i in range(1, 51)}
    got = {}
    for variant in ("bandpass", "lowpass"):
        m = [r for r in rows_ if r["variant"] == variant and r["pairing"] == "matched"]
        c = [r for r in rows_ if r["variant"] == variant and r["pairing"] == "control"]
        mag = lambda r: math.hypot(fnum(r["dy"]), fnum(r["dx"]))
        got[variant] = {
            "fields": len(m), "matched_within_1px": sum(mag(r) <= 1.0 for r in m), "control_within_1px": sum(mag(r) <= 1.0 for r in c),
            "auc_r": _auc([fnum(r["r_after_shift"]) for r in m], [fnum(r["r_after_shift"]) for r in c]),
            "auc_error": _auc([-fnum(r["error"]) for r in m], [-fnum(r["error"]) for r in c]),
            "control_r_max": max(fnum(r["r_after_shift"]) for r in c)}
        pr = [r for r in m if r["stem"] not in snu]
        sn = [r for r in m if r["stem"] in snu]
        got[variant].update(paired_fields=len(pr), paired_max_shift_px=max(mag(r) for r in pr), paired_within_1px=sum(mag(r) <= 1.0 for r in pr),
                            paired_r_min=min(fnum(r["r_after_shift"]) for r in pr), paired_r_max=max(fnum(r["r_after_shift"]) for r in pr),
                            snu_fields=len(sn), snu_r_min=min(fnum(r["r_after_shift"]) for r in sn), snu_r_max=max(fnum(r["r_after_shift"]) for r in sn))
    b, lo = got["bandpass"], got["lowpass"]
    claims = [("matched_within_1px", 734, 0), ("control_within_1px", 3, 0), ("fields", 800, 0), ("paired_fields", 750, 0), ("paired_within_1px", 734, 0),
              ("snu_fields", 50, 0), ("auc_r", 0.975, 3), ("auc_error", 0.973, 3), ("paired_max_shift_px", 4.7, 1), ("paired_r_min", 0.47, 2),
              ("paired_r_max", 1.00, 2), ("control_r_max", 0.25, 2), ("snu_r_min", -0.13, 2), ("snu_r_max", 0.18, 2)]
    reg_checks = {}
    for k, v, d in claims:
        val = b[k]
        ok = (round(val, d) == v) if d else (val == v)
        reg_checks[k] = {"manuscript": v, "recomputed": val, "match": ok}
        check(f"registration (bandpass) {k}: manuscript {v} vs archive {val}", ok)
    ok = round(lo["auc_r"], 3) == 0.506
    reg_checks["lowpass_auc_r"] = {"manuscript": 0.506, "recomputed": lo["auc_r"], "match": ok}
    check(f"registration (lowpass) auc_r: manuscript 0.506 vs archive {lo['auc_r']}", ok)
    summ_ = jload(REGDIR / "registration_summary.json")
    registration = {"verified": all(c_["match"] for c_ in reg_checks.values()),
                    "source_files": [f"results_for_manuscript/registration/{f}" for f in REG_FILES], "origin": "runs/diagnostics/ (scripts/register_holograms.py)",
                    "recomputed_from_hologram_registration_csv": got, "manuscript_claims_checked": reg_checks,
                    "summary_json_bandpass_separation": summ_["registration"]["bandpass"]["separation"],
                    "split_from_train_G_log": split}


_mb = yaml.safe_load(cfgtxt)["membrane"] if "membrane" in yaml.safe_load(cfgtxt) else {}
membrane = {"scale": E(_mb.get("scale"), "membrane px per phase px", "config/base.yaml", "membrane.scale", "", 1, None),
            "offset_y_px": E(_mb.get("offset_y"), "px", "config/base.yaml", "membrane.offset_y", "", 1, None),
            "offset_x_px": E(_mb.get("offset_x"), "px", "config/base.yaml", "membrane.offset_x", "", 1, None),
            "second_header_pitch_um": E(0.211994, "um", "config/base.yaml (comment at membrane.scale) / scripts/prepare_data.py", "phase-file header pitch_y", "", 1, None),
            "check": "0.284871 / 0.211994 = %.6f" % (0.284871 / 0.211994)}
check("membrane scale equals the pitch ratio 0.284871/0.211994", abs(_mb.get("scale", 0) - 0.284871 / 0.211994) < 1e-6)
crop = {"configured_offset_px": E(RUN["A"][42]["cfg"]["data"]["crop_offset_px"], "px", rel(RUNS / RUN["A"][42]["dir"] / "resolved_config.yaml"),
                                  "data.crop_offset_px", "", 1, None)}
cj = jload(RUNS / "diagnostics" / "crop_offset_test_dy-6_dx-2.json")
crop["test_confirmation"] = {"source_file": "runs/diagnostics/crop_offset_test_dy-6_dx-2.json", "content": cj}
lab = jload(RUNS / "label_audit_thresholds.json")
write("metadata/gradient_registration_crop.json", {"gradient_path": meta_grad, "membrane_registration": membrane, "registration": registration, "crop": crop,
                                                   "label_audit_thresholds": {k: lab[k] for k in lab if not isinstance(lab[k], (list, dict))}})

# splits
splits = jload(ROOT / "data" / "splits.json")
check("test split has 113 fields", len(splits["splits"]["test"]) == 113, str(len(splits["splits"]["test"])))
write("metadata/splits.json", {"splits_sha256": SPLITS_SHA, "off_axis": {k: len(v) for k, v in splits["splits"].items()},
                               "in_line": split, "common_fields": 107,
                               "source_file": "data/splits.json; logs/v2_20260930_074055_gpu3/train_G.log"})


# ----------------------------------------------------------------------------- 10. figure source data
print("10. figure source data")
import shutil
FIG = OUT / "figures"
(FIG / "data").mkdir(parents=True, exist_ok=True)
FIG_SOURCES = {
    "benchmark_results": RUNS / "benchmark_results",          # figures 2, 4-6, 9 (every per-arm statistic)
    "z_calibration.json": RUNS / "z_calibration.json",        # figure 7c
    "RESULTS.md": RUNS / "RESULTS.md",                        # figure 7b (learned-z trajectory)
    "error_propagation_summary.csv": RUNS / "error_propagation_summary.csv",   # figure 8a,b
    "synthetic_validation_seeds": RUNS / "synthetic_validation_seeds",         # figure 8c
    "per_cell_test_A.csv": RUNS / "v2_baseline_off_axis" / "per_cell_test.csv",   # figure 4 (agreement)
    "amplitude_sensitivity_off_axis_global.csv": RUNS / "amplitude_sensitivity_off_axis_global.csv",
    "amplitude_sensitivity_off_axis_per_field.csv": RUNS / "amplitude_sensitivity_off_axis_per_field.csv",
    "diagnostics": RUNS / "diagnostics",                      # decomposition tables
}
for name, src in FIG_SOURCES.items():
    dst = FIG / "data" / name
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
write("figures/figure_index.json", {
    "note": "Figure source data copied from runs/ (data/). Scripts: figures/scripts/ (make_all.py regenerates figures 2, 4-9 into figures/regenerated/). The per-figure data lists below are indicative (taken from the script docstrings). See analysis/v42/FIGURE_STATUS.md.",
    "figures": {
        "figure_2": {"data": ["data/benchmark_results/results_arm_A.json", "data/benchmark_results/results_conventional_off_axis.json",
                              "data/benchmark_results/results_conventional_gabor.json"]},
        "figure_3": {"data": "needs checkpoints (not in repository); see FIGURE_STATUS.md"},
        "figure_4": {"data": ["data/per_cell_test_A.csv", "data/benchmark_results/results_arm_A.json"]},
        "figure_5": {"data": ["data/benchmark_results/results_arm_*.json", "data/benchmark_results/results_comparisons.json"]},
        "figure_6": {"data": ["data/benchmark_results/results_arm_{A,W01,W03,W30,B}.json"]},
        "figure_7": {"data": ["data/benchmark_results/results_arm_*.json", "data/RESULTS.md", "data/z_calibration.json"]},
        "figure_8": {"data": ["data/error_propagation_summary.csv", "data/synthetic_validation_seeds/"]},
        "figure_9": {"data": ["data/benchmark_results/results_hardware_arm_*.json", "data/benchmark_results/results_arm_{A,KA,B,KB}.json"]},
    }})

# ----------------------------------------------------------------------------- 9. summary
write("metadata/consistency_checks.json", {"n_checks": len(CHECKS), "n_failed": sum(not c["passed"] for c in CHECKS), "checks": CHECKS})
summary = {"generated_by": "analysis/v42/compile_results.py",
           "design": {"three_seed_configurations": [label(a) for a in REG if len(RUN[a]) == 3],
                      "single_run_configurations": [label(a) for a in REG if len(RUN[a]) == 1],
                      "deterministic": ["Classical Off-Axis", "Classical In-Line"]},
           "files": {"baseline": "baseline/baseline.json", "classical": "classical/classical.json",
                     "inline": "inline/inline.json", "ipp": "ipp/ipp.json", "amplitude": "amplitude/amplitude.json",
                     "forward_model": "forward_model/forward_model.json", "decomposition": "decomposition/decomposition.json",
                     "boundary": "boundary_sensitivity/boundary.json", "benchmarking": "benchmarking/benchmarking.json",
                     "configurations": "metadata/configurations.json"},
           "consistency_checks": {"n": len(CHECKS), "failed": [c for c in CHECKS if not c["passed"]]}}
write("summary/manuscript_summary.json", summary)
failed = [c for c in CHECKS if not c["passed"]]
print(f"compiled {len(CHECKS)} consistency checks, {len(failed)} failed")
for c in failed:
    print("  FAILED:", c["check"], c["detail"])
sys.exit(1 if failed else 0)
