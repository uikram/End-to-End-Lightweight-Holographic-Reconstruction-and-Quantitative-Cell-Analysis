"""Collect every number used in HoloQPI_4.2 from the corrected result files.

Run from the repository root:
    python analysis/v42/extract_values.py

Writes analysis/v42/values.json: one entry per value, with the file and the
field it was read from, or, for derived values, the formula and its inputs.
Nothing is typed by hand: every entry is read from a result file or computed
from entries that were.

Rule for comparisons (Methods, statistical analysis): a difference is
"resolved" when |mean_a - mean_b| > 2 * sqrt((s_a^2 + s_b^2) / 2), with s the
between-seed sample SD (ddof = 1). Fewer than three runs on either side:
"not resolvable".
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
OUT = Path(__file__).resolve().parent / "values.json"

VALUES: dict[str, dict] = {}


def put(key, value, source, field=None, formula=None, inputs=None):
    VALUES[key] = {"value": value, "source": source, "field": field,
                   "formula": formula, "inputs": inputs}
    return value


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def load_json(p: Path):
    with open(p) as fh:
        return json.load(fh)


def read_csv(p: Path):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ----------------------------------------------------------------------------- arms
ARMS = {
    "A": {42: "v2_baseline_off_axis", 1337: "v2_A_seed1337_off_axis", 2024: "v2_A_seed2024_off_axis"},
    "B": {42: "v2_cell_ipp_off_axis", 1337: "v2_B_seed1337_off_axis", 2024: "v2_B_seed2024_off_axis"},
    "B1": {42: "v2_image_volume_off_axis", 1337: "v2_B1_seed1337_off_axis", 2024: "v2_B1_seed2024_off_axis"},
    "B2": {42: "v2_cell_ipp_area_off_axis"},
    "C": {42: "v2_cell_ipp_bga_off_axis"},
    "D0": {42: "v2_amplitude_off_axis"},
    "D1": {42: "v2_forward_amplitude_off_axis"},
    "D2": {42: "v2_learned_z_off_axis"},
    "W01": {42: "v2_ipp_w01_off_axis"},
    "W03": {42: "v2_ipp_w03_off_axis"},
    "W10": {42: "v2_ipp_w10_off_axis"},
    "W30": {42: "v2_ipp_w30_off_axis"},
    "KA": {42: "v2_compact_baseline_off_axis"},
    "KB": {42: "v2_compact_cell_ipp_off_axis"},
    "G": {42: "v2_baseline_gabor"},
}

print("step 1/12: per-run metrics")
RUNMETRICS: dict[str, dict[int, dict]] = {}
for arm, seeds in ARMS.items():
    RUNMETRICS[arm] = {}
    for seed, d in seeds.items():
        p = RUNS / d / "metrics_test.json"
        m = load_json(p)
        RUNMETRICS[arm][seed] = m
        for k, v in m.items():
            if isinstance(v, (int, float)):
                put(f"run.{arm}.s{seed}.{k}", v, rel(p), k)
    print(f"  arm {arm}: {len(seeds)} run(s)")

# Cross-check against the collector's benchmark JSON where it exists.
for arm in RUNMETRICS:
    p = RUNS / "benchmark_results" / f"results_arm_{arm}.json"
    if not p.exists():
        continue
    b = load_json(p)
    for run in b["runs"]:
        seed = run["seed"]
        stored = RUNMETRICS[arm].get(seed)
        if stored is None:
            continue
        for k in ("dry_mass_mape", "seg_dice", "phase_pearson_r"):
            if abs(run["metrics"][k] - stored[k]) > 1e-9:
                raise SystemExit(f"mismatch {arm} s{seed} {k}")
print("  benchmark_results/results_arm_*.json agree with runs/*/metrics_test.json")


def sd(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2 or any(math.isnan(x) for x in xs):
        return float("nan")
    return statistics.stdev(xs)


print("step 2/12: seed means and SDs")
METRICS = sorted({k for arm in RUNMETRICS for m in RUNMETRICS[arm].values()
                  for k, v in m.items() if isinstance(v, (int, float))})
MEAN, SD, NRUN = {}, {}, {}
for arm, runs in RUNMETRICS.items():
    for k in METRICS:
        xs = [r[k] for r in runs.values() if k in r and isinstance(r[k], (int, float))]
        if not xs:
            continue
        mean = sum(xs) / len(xs)
        MEAN[(arm, k)] = mean
        SD[(arm, k)] = sd(xs)
        NRUN[(arm, k)] = len(xs)
        srcs = [rel(RUNS / ARMS[arm][s] / "metrics_test.json") for s in runs]
        put(f"mean.{arm}.{k}", mean, "; ".join(srcs), k, "mean over runs", [f"run.{arm}.s{s}.{k}" for s in runs])
        if len(xs) >= 2:
            put(f"sd.{arm}.{k}", SD[(arm, k)], "; ".join(srcs), k, "sample SD (ddof=1) over runs",
                [f"run.{arm}.s{s}.{k}" for s in runs])
        put(f"n.{arm}.{k}", len(xs), "; ".join(srcs), k, "number of runs")


def compare(a, b, k, key=None, mean=MEAN, sdd=SD, nrun=NRUN, source="runs/*/metrics_test.json"):
    key = key or f"cmp.{a}_vs_{b}.{k}"
    ma, mb = mean[(a, k)], mean[(b, k)]
    diff = ma - mb
    put(f"{key}.diff", diff, source, k, f"mean.{a} - mean.{b}", [f"mean.{a}.{k}", f"mean.{b}.{k}"])
    if nrun[(a, k)] >= 3 and nrun[(b, k)] >= 3:
        pooled = math.sqrt((sdd[(a, k)] ** 2 + sdd[(b, k)] ** 2) / 2)
        thr = 2 * pooled
        verdict = "resolved" if abs(diff) > thr else "not resolved"
        put(f"{key}.thr", thr, source, k, "2*sqrt((sa^2+sb^2)/2)", [f"sd.{a}.{k}", f"sd.{b}.{k}"])
    else:
        verdict = "not resolvable"
    put(f"{key}.verdict", verdict, source, k, "resolution rule")
    return diff, verdict


print("step 3/12: comparisons")
CMP_METRICS = ["dry_mass_mape", "dry_mass_field_total_mape", "dry_mass_field_total_bias",
               "area_mape", "circularity_mape", "phase_mae_rad", "phase_mae_rad_in_cell",
               "phase_pearson_r", "phase_ssim", "seg_dice", "seg_aji", "seg_boundary_f1",
               "detection_recall", "detection_precision", "detection_f1",
               "dry_mass_mape_coverage_adjusted", "cells_false_positive", "cells_matched",
               "forward_residual", "forward_residual_ratio", "dry_mass_cell_pearson_r"]
PAIRS = [("B", "A"), ("B1", "A"), ("B1", "B"), ("B2", "B"), ("C", "B"), ("D0", "B"),
         ("D1", "D0"), ("D2", "D1"), ("W01", "A"), ("W03", "A"), ("W10", "A"), ("W30", "A"),
         ("KA", "A"), ("KB", "B")]
for a, b in PAIRS:
    for k in CMP_METRICS:
        if (a, k) in MEAN and (b, k) in MEAN:
            compare(a, b, k)
# cross-check with the collector's comparison file
rc = load_json(RUNS / "benchmark_results" / "results_comparisons.json")["comparisons"]
for name, c in rc.items():
    a, b = c["arm"], c["comparator"]
    m = c["metrics"].get("dry_mass_mape")
    if name == "G_vs_A":
        continue  # mixes the 107-field in-line run with the 113-field off-axis runs; not used
    if m and (a, "dry_mass_mape") in MEAN and b in ("A", "B", "D0", "D1"):
        mine = VALUES[f"cmp.{a}_vs_{b}.dry_mass_mape.diff"]["value"]
        if abs(mine - m["difference"]) > 1e-9:
            raise SystemExit(f"comparison mismatch {name}")
        if m["threshold"] is not None:
            if abs(VALUES[f"cmp.{a}_vs_{b}.dry_mass_mape.thr"]["value"] - m["threshold"]) > 1e-9:
                raise SystemExit(f"threshold mismatch {name}")
print("  differences and thresholds agree with benchmark_results/results_comparisons.json")

print("step 4/12: classical pipeline, 113 fields and 107 common fields")
cl = load_json(RUNS / "conventional_baseline_test.json")
for geom in ("off_axis", "gabor"):
    for k, v in cl[geom].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            put(f"classical113.{geom}.{k}", v, "runs/conventional_baseline_test.json", f"{geom}.{k}")
clc = load_json(RUNS / "common_fields" / "conventional_baseline_test.json")
for geom in ("off_axis", "gabor"):
    for k, v in clc[geom].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            put(f"classical107.{geom}.{k}", v, "runs/common_fields/conventional_baseline_test.json", f"{geom}.{k}")
for row in read_csv(RUNS / "common_fields" / "common_fields_summary.csv"):
    for k, v in row.items():
        if k != "entry":
            put(f"common.{row['entry']}.{k}", fnum(v), "runs/common_fields/common_fields_summary.csv", f"{row['entry']}.{k}")
# per-seed common-field metrics for A (all metrics, including those not in the summary)
for seed, d in ARMS["A"].items():
    p = RUNS / d / "metrics_test_common.json"
    m = load_json(p)
    for k, v in m.items():
        if isinstance(v, (int, float)):
            put(f"common_run.A.s{seed}.{k}", v, rel(p), k)
mg = load_json(RUNS / ARMS["G"][42] / "metrics_test_common.json")
for k, v in mg.items():
    if isinstance(v, (int, float)):
        put(f"common_run.G.s42.{k}", v, rel(RUNS / ARMS["G"][42] / "metrics_test_common.json"), k)
for k in ("dry_mass_cell_pearson_r", "cells_matched", "dry_mass_field_total_bias"):
    xs = [VALUES[f"common_run.A.s{s}.{k}"]["value"] for s in (42, 1337, 2024)]
    put(f"common_mean.A.{k}", sum(xs) / 3, "runs/v2_*/metrics_test_common.json", k, "mean over 3 seeds",
        [f"common_run.A.s{s}.{k}" for s in (42, 1337, 2024)])
    put(f"common_sd.A.{k}", sd(xs), "runs/v2_*/metrics_test_common.json", k, "SD over 3 seeds")

print("step 5/12: mass-error decomposition")
for row in read_csv(RUNS / "diagnostics" / "decomposition_by_config.csv"):
    base = f"decomp.{row['config']}.{row['level']}.{row['group']}"
    for k, v in row.items():
        if k not in ("config", "level", "group"):
            put(f"{base}.{k}", fnum(v), "runs/diagnostics/decomposition_by_config.csv", f"{base}.{k}")
summ = read_csv(RUNS / "diagnostics" / "decomposition_summary.csv")
DSTAT = {}
for row in summ:
    run = row["run"]
    cfg = run.split("_")[0]
    for k in ("domain_gm", "phase_gm", "total_gm", "domain_med_abs_log", "phase_med_abs_log",
              "total_med_abs_log", "recall", "n", "n_reference_cells"):
        DSTAT.setdefault((cfg, row["level"], row["group"], k), []).append(fnum(row[k]))
        put(f"decomp_run.{run}.{row['level']}.{row['group']}.{k}", fnum(row[k]),
            "runs/diagnostics/decomposition_summary.csv", f"{run}.{row['level']}.{row['group']}.{k}")
DM, DS, DN = {}, {}, {}
for (cfg, lvl, grp, k), xs in DSTAT.items():
    DM[(cfg, f"{lvl}.{grp}.{k}")] = sum(xs) / len(xs)
    DS[(cfg, f"{lvl}.{grp}.{k}")] = sd(xs)
    DN[(cfg, f"{lvl}.{grp}.{k}")] = len(xs)
for a in ("B", "B1"):
    for lvl, grp in (("cell", "all"), ("cell", "interior"), ("cell", "edge"), ("field", "all")):
        for k in ("domain_gm", "phase_gm", "total_gm", "domain_med_abs_log", "phase_med_abs_log",
                  "total_med_abs_log", "recall"):
            kk = f"{lvl}.{grp}.{k}"
            if (a, kk) in DM and not math.isnan(DM[(a, kk)]):
                compare(a, "A", kk, key=f"dcmp.{a}_vs_A.{kk}", mean=DM, sdd=DS, nrun=DN,
                        source="runs/diagnostics/decomposition_summary.csv")

print("step 6/12: location stratification (edge / interior)")
cells = read_csv(RUNS / "diagnostics" / "decomposition_cells.csv")
LOC = {}
for c in cells:
    ape = abs(fnum(c["dry_mass_pg_pred"]) / fnum(c["dry_mass_pg_ref"]) - 1)
    LOC.setdefault((c["run"], c["edge"]), []).append(ape)
for (run, edge), apes in LOC.items():
    grp = "edge" if edge == "1" else "interior"
    put(f"loc.{run}.{grp}.mape", sum(apes) / len(apes), "runs/diagnostics/decomposition_cells.csv",
        "mean |dry_mass_pg_pred/dry_mass_pg_ref - 1|", "mean over matched cells of the group")
    put(f"loc.{run}.{grp}.n_matched", len(apes), "runs/diagnostics/decomposition_cells.csv", "count")
# false positives by location, from the stored unmatched files
H = W = 900
FP_MARGIN = 15
for arm in ("A", "B", "B1"):
    for seed, d in ARMS[arm].items():
        rows = read_csv(RUNS / d / "unmatched_test.csv")
        fps = [r for r in rows if r["kind"] != "missed_reference"]
        near = [r for r in fps if min(fnum(r["centroid_y"]), fnum(r["centroid_x"]),
                                      H - 1 - fnum(r["centroid_y"]), W - 1 - fnum(r["centroid_x"])) <= FP_MARGIN]
        put(f"loc.{arm}_s{seed}.fp_all", len(fps), rel(RUNS / d / "unmatched_test.csv"), "rows with kind != missed_reference")
        put(f"loc.{arm}_s{seed}.fp_near", len(near), rel(RUNS / d / "unmatched_test.csv"),
            f"false positives with centroid within {FP_MARGIN} px of the field boundary")
# classical pipeline (single run, 113 fields)
rows = read_csv(RUNS / "conventional_off_axis" / "unmatched_test.csv")
fps = [r for r in rows if r["kind"] != "missed_reference"]
near = [r for r in fps if min(fnum(r["centroid_y"]), fnum(r["centroid_x"]),
                              H - 1 - fnum(r["centroid_y"]), W - 1 - fnum(r["centroid_x"])) <= FP_MARGIN]
put("loc.classical.fp_all", len(fps), "runs/conventional_off_axis/unmatched_test.csv", "rows with kind != missed_reference")
put("loc.classical.fp_near", len(near), "runs/conventional_off_axis/unmatched_test.csv",
    f"false positives with centroid within {FP_MARGIN} px of the field boundary")
# classical run in the decomposition file is 'classical'
LM, LS = {}, {}
for cfg in ("A", "B", "B1", "classical"):
    runs = [f"{cfg}_s{s}" for s in (42, 1337, 2024)] if cfg != "classical" else ["classical"]
    for stat in ("edge.mape", "interior.mape", "edge.n_matched", "interior.n_matched", "fp_all", "fp_near"):
        xs = [VALUES[f"loc.{r}.{stat}"]["value"] for r in runs if f"loc.{r}.{stat}" in VALUES]
        if not xs:
            continue
        LM[(cfg, stat)] = sum(xs) / len(xs)
        LS[(cfg, stat)] = sd(xs)
        put(f"locmean.{cfg}.{stat}", LM[(cfg, stat)], "derived", stat, "mean over runs",
            [f"loc.{r}.{stat}" for r in runs])
        if len(xs) >= 2:
            put(f"locsd.{cfg}.{stat}", LS[(cfg, stat)], "derived", stat, "SD over runs")
LN = {k: 3 for k in LM}
for a in ("B", "B1"):
    for stat in ("edge.mape", "interior.mape", "fp_all", "fp_near"):
        compare(a, "A", stat, key=f"lcmp.{a}_vs_A.{stat}", mean=LM, sdd=LS, nrun=LN,
                source="runs/diagnostics/decomposition_cells.csv; runs/*/unmatched_test.csv")
# reference instances at the edge (from the decomposition summary)
for row in summ:
    if row["run"] == "A_s42" and row["level"] == "cell" and row["group"] in ("edge", "all"):
        put(f"refcells.{row['group']}", fnum(row["n_reference_cells"]),
            "runs/diagnostics/decomposition_summary.csv", f"A_s42.cell.{row['group']}.n_reference_cells")

print("step 7/12: forward model and z calibration")
zc = load_json(RUNS / "z_calibration.json")
for geom in ("off_axis", "gabor"):
    g = zc[geom]
    for k in ("z_forward_um", "per_image_median_z_um", "per_image_iqr_um", "grid_step_um", "identifiable",
              "reference_floor", "forward_model_verdict"):
        put(f"z.{geom}.{k}", g[k], "runs/z_calibration.json", f"{geom}.{k}")
    put(f"z.{geom}.per_image_best_z_um", g["curve"]["per_image_best_z_um"], "runs/z_calibration.json",
        f"{geom}.curve.per_image_best_z_um")
    put(f"z.{geom}.scan_images", g["curve"]["images"], "runs/z_calibration.json", f"{geom}.curve.images")
    dsc = g["discrimination"]
    for k in ("floor", "images", "tolerance", "win_threshold", "verdict", "distance_um"):
        put(f"disc.{geom}.{k}", dsc[k], "runs/z_calibration.json", f"{geom}.discrimination.{k}")
    for v in dsc["margins"]:
        put(f"disc.{geom}.margin.{v}", dsc["margins"][v], "runs/z_calibration.json", f"{geom}.discrimination.margins.{v}")
        put(f"disc.{geom}.win.{v}", dsc["win_rates"][v], "runs/z_calibration.json", f"{geom}.discrimination.win_rates.{v}")
    sr = g["scale_response"]
    for k in ("best_scale_pooled", "best_scale_median", "best_scale_iqr", "images"):
        put(f"scale.{geom}.{k}", sr[k], "runs/z_calibration.json", f"{geom}.scale_response.{k}")
put("config.discrimination_tolerance", 0.01, "config/base.yaml", "loss.forward_model.discrimination_tolerance")
cfgtxt = (ROOT / "config" / "base.yaml").read_text()
assert re.search(r"discrimination_tolerance:\s*0\.01", cfgtxt)
m = re.search(r"crop_offset_px:\s*\[(-?\d+),\s*(-?\d+)\]", cfgtxt)
put("config.crop_offset_dy", int(m.group(1)), "config/base.yaml", "data.crop_offset_px[0]")
put("config.crop_offset_dx", int(m.group(2)), "config/base.yaml", "data.crop_offset_px[1]")
# test-field probes (amplitude_sensitivity logs: global and per-field aberration surfaces)
for mode in ("global", "per_field"):
    p = LOGS / "v2_20260930_065521_gpu2" / f"amplitude_sensitivity_{mode}.log"
    t = p.read_text()
    n = int(re.search(r"=== residual over (\d+) images", t).group(1))
    put(f"probe.{mode}.images", n, rel(p), "residual over N images")
    for name in ("amp_unity", "amp_reference", "phase_reference"):
        mm = re.search(rf"^\s*{name}\s+([0-9.]+)", t, re.M)
        put(f"probe.{mode}.{name}", float(mm.group(1)), rel(p), name)
    for name in ("phase_scaled_0.9", "phase_scaled_0.5"):
        mm = re.search(rf"^\s*{re.escape(name)}\s+([0-9.]+)\s+([+-][0-9.]+)", t, re.M)
        put(f"probe.{mode}.{name}.residual", float(mm.group(1)), rel(p), name)
        put(f"probe.{mode}.{name}.change", float(mm.group(2)), rel(p), f"{name} change vs reference")
# learned z (D2): trajectory from RESULTS.md (collector) and history
rtxt = (RUNS / "RESULTS.md").read_text()
mm = re.search(r"initial \*\*\+([0-9.]+) um\*\*, final \*\*\+([0-9.]+) um\*\*", rtxt)
put("learnedz.initial_after_epoch1", float(mm.group(1)), "runs/RESULTS.md", "learned propagation distance, initial")
put("learnedz.final", float(mm.group(2)), "runs/RESULTS.md", "learned propagation distance, final")
mm = re.search(r"full excursion over training \*\*([0-9.]+) um\*\*; spread over the last 10 epochs \*\*([0-9.]+) um\*\*", rtxt)
put("learnedz.excursion", float(mm.group(1)), "runs/RESULTS.md", "full excursion")
put("learnedz.last10_spread", float(mm.group(2)), "runs/RESULTS.md", "spread over last 10 epochs")
put("learnedz.scored", RUNMETRICS["D2"][42]["forward_distance_um"], rel(RUNS / ARMS["D2"][42] / "metrics_test.json"),
    "forward_distance_um")
ra = load_json(RUNS / "benchmark_results" / "results_arm_D2.json")
put("learnedz.checkpoint_epoch", ra["runs"][0]["checkpoint_epoch"], "runs/benchmark_results/results_arm_D2.json",
    "runs[0].checkpoint_epoch")

# Global surface on the same fields as the per-field surfaces (per-field CSV lacks
# the test fields whose per-field aberration fit was rejected).
gl = read_csv(RUNS / "amplitude_sensitivity_off_axis_global.csv")
pf = read_csv(RUNS / "amplitude_sensitivity_off_axis_per_field.csv")
pf_batches = {r["batch"] for r in pf}
common = [r for r in gl if r["batch"] in pf_batches]
for k in ("phase_reference", "phase_scaled_0.9", "phase_scaled_0.5"):
    put(f"probe.global_common.{k}", sum(fnum(r[k]) for r in common) / len(common),
        "runs/amplitude_sensitivity_off_axis_global.csv", f"mean {k} over batches present in the per-field CSV")
put("probe.global_common.images", len(common), "runs/amplitude_sensitivity_off_axis_global.csv", "rows shared with per-field CSV")
for k in ("phase_scaled_0.9", "phase_scaled_0.5"):
    put(f"probe.global_common.{k}.change", VALUES[f"probe.global_common.{k}"]["value"] - VALUES["probe.global_common.phase_reference"]["value"],
        "derived", k, "scaled minus reference, common fields")
splits = load_json(ROOT / "data" / "splits.json")["splits"]["test"]
missing = sorted(int(r["batch"]) for r in gl if r["batch"] not in pf_batches)
ab = {r["stem"]: r for r in read_csv(RUNS / "aberration_fit_order5.csv")}
for i in missing:
    stem = splits[i]
    put(f"probe.per_field.missing.{i}.stem", stem, "data/splits.json", f"test[{i}]")
    put(f"probe.per_field.missing.{i}.reason", ab[stem]["reason"], "runs/aberration_fit_order5.csv", f"{stem}.reason")
    put(f"probe.per_field.missing.{i}.surface_pv_rad", fnum(ab[stem]["surface_peak_to_valley_rad"]),
        "runs/aberration_fit_order5.csv", f"{stem}.surface_peak_to_valley_rad")

# Registration of the in-line frames (diagnostics archive produced before the
# correction: diagnostics/registration_summary.json, hologram_registration.csv).
# The archive is not in the project folder; these values were supplied by the
# author from it on 2026-10-01 and are recorded here with that provenance.
REG_SRC = "diagnostics archive registration_summary.json / hologram_registration.csv (author-supplied; archive not in project folder)"
for k, x in (("matched_within_1px", 734), ("control_within_1px", 3), ("fields", 800), ("auc_r", 0.975),
             ("auc_error", 0.973), ("paired_fields", 750), ("paired_max_shift_px", 4.7),
             ("paired_within_1px", 734), ("paired_r_min", 0.47), ("paired_r_max", 1.00),
             ("control_r_max", 0.25), ("snu_r_min", -0.13), ("snu_r_max", 0.18), ("lowpass_auc", 0.506)):
    put(f"reg.{k}", x, REG_SRC, k)

print("step 8/12: amplitude")
for arm in ("D0", "D1", "D2"):
    m = RUNMETRICS[arm][42]
    for k, v in m.items():
        if k.startswith("amplitude"):
            put(f"amp.{arm}.{k}", v, rel(RUNS / ARMS[arm][42] / "metrics_test.json"), k)

print("step 9/12: efficiency")
for arm in ("A", "B", "D0", "KA", "KB"):
    p = RUNS / "benchmark_results" / f"results_hardware_arm_{arm}.json"
    h = load_json(p)
    runs = h["runs"]
    keys = sorted({k for r in runs for k in r["metrics"]})
    for k in keys:
        xs = [r["metrics"][k] for r in runs if isinstance(r["metrics"].get(k), (int, float))]
        if not xs:
            continue
        put(f"hw.{arm}.mean.{k}", sum(xs) / len(xs), rel(p), k, "mean over benchmark sessions",
            [f"session seed {r['seed']}" for r in runs])
        if len(xs) >= 2:
            put(f"hw.{arm}.sd.{k}", sd(xs), rel(p), k, "SD over benchmark sessions")
        put(f"hw.{arm}.n.{k}", len(xs), rel(p), k)
        put(f"hw.{arm}.min.{k}", min(xs), rel(p), k)
        put(f"hw.{arm}.max.{k}", max(xs), rel(p), k)

print("step 10/12: model-free analyses")
for row in read_csv(RUNS / "error_propagation_summary.csv"):
    s = int(float(row["shift_px"]))
    for k, v in row.items():
        put(f"ep.{s}.{k}", fnum(v), "runs/error_propagation_summary.csv", f"shift {s}: {k}")
ratios = [fnum(r["mass_over_area"]) for r in read_csv(RUNS / "error_propagation_summary.csv")
          if int(float(r["shift_px"])) != 0]
put("ep.median_ratio", statistics.median(ratios), "runs/error_propagation_summary.csv", "mass_over_area",
    "median over shifts +-1..5")
sv = load_json(RUNS / "benchmark_results" / "results_synthetic_validation.json")
for k in ("mass_abs_relative_error_mean", "area_abs_relative_error_mean", "field_total_mass_error_mean",
          "field_total_mass_abs_error_mean", "cells_measured", "cells_placed"):
    xs = [r["metrics"][k] for r in sv["runs"]]
    put(f"syn.mean.{k}", sum(xs) / len(xs), "runs/benchmark_results/results_synthetic_validation.json",
        f"runs[*].metrics.{k}", "mean over 3 sets")
    put(f"syn.sd.{k}", sd(xs), "runs/benchmark_results/results_synthetic_validation.json",
        f"runs[*].metrics.{k}", "SD over 3 sets")
    put(f"syn.min.{k}", min(xs), "runs/benchmark_results/results_synthetic_validation.json", k)
    put(f"syn.max.{k}", max(xs), "runs/benchmark_results/results_synthetic_validation.json", k)

print("step 11/12: gradient ratio, splits, crop offset, label audit")
g = read_csv(RUNS / "gradient_path_b_cell_ipp_512.csv")
r = [fnum(x["ratio_at_weight_1"]) for x in g]
put("grad.median", statistics.median(r), "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "median")
put("grad.min", min(r), "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "min")
put("grad.max", max(r), "runs/gradient_path_b_cell_ipp_512.csv", "ratio_at_weight_1", "max")
put("grad.batches", len(r), "runs/gradient_path_b_cell_ipp_512.csv", "rows")
tg = (LOGS / "v2_20260930_074055_gpu3" / "train_G.log").read_text()
put("split.G.train_excluded", int(re.search(r"train split: (\d+) of 560 fields excluded", tg).group(1)),
    "logs/v2_20260930_074055_gpu3/train_G.log", "train split excluded")
put("split.G.train", int(re.search(r"train split: (\d+) samples \| modality=gabor", tg).group(1)),
    "logs/v2_20260930_074055_gpu3/train_G.log", "train samples")
put("split.G.val", int(re.search(r"val split: (\d+) samples \| modality=gabor", tg).group(1)),
    "logs/v2_20260930_074055_gpu3/train_G.log", "val samples")
put("split.G.val_excluded", int(re.search(r"val split: (\d+) of 127 fields excluded", tg).group(1)),
    "logs/v2_20260930_074055_gpu3/train_G.log", "val excluded")
put("split.G.test", RUNMETRICS["G"][42]["phase_n_images"], rel(RUNS / ARMS["G"][42] / "metrics_test.json"), "phase_n_images")
put("split.G.test_cells", RUNMETRICS["G"][42]["cells_reference"], rel(RUNS / ARMS["G"][42] / "metrics_test.json"), "cells_reference")
ta = (LOGS / "v2_20260930_074050_gpu2" / "train_A.log").read_text()
put("split.A.train", int(re.search(r"train split: (\d+) samples \| modality=off_axis", ta).group(1)),
    "logs/v2_20260930_074050_gpu2/train_A.log", "train samples")
put("split.A.val", int(re.search(r"val split: (\d+) samples \| modality=off_axis", ta).group(1)),
    "logs/v2_20260930_074050_gpu2/train_A.log", "val samples")
put("split.A.test", RUNMETRICS["A"][42]["phase_n_images"], rel(RUNS / ARMS["A"][42] / "metrics_test.json"), "phase_n_images")
put("split.A.test_cells", RUNMETRICS["A"][42]["cells_reference"], rel(RUNS / ARMS["A"][42] / "metrics_test.json"), "cells_reference")
re_log = (LOGS / "run_everything_20260930_042722.txt").read_text()
val_block = re_log.split("=== Crop offset, val split")[1].split("=== Crop offset, test split")[0]
mm = re.search(r"residual dy\s+median ([+-][0-9.]+) px\s+IQR \[([+-][0-9.]+), ([+-][0-9.]+)\]", val_block)
put("crop.val_centre.dy_median", float(mm.group(1)), "logs/run_everything_20260930_042722.txt", "val split, offset [0,0], residual dy median")
mm = re.search(r"residual dx\s+median ([+-][0-9.]+) px", val_block)
put("crop.val_centre.dx_median", float(mm.group(1)), "logs/run_everything_20260930_042722.txt", "val split, offset [0,0], residual dx median")
put("crop.val_fields", int(re.search(r"val split, (\d+) fields", re_log).group(1)), "logs/run_everything_20260930_042722.txt", "val fields")
cands = re.findall(r"\[\s*(-?\d+),\s*(-?\d+)\]\s+([+-][0-9.]+)\s+([+-][0-9.]+)", val_block)
put("crop.val_candidates", len([c for c in cands if not (c[0] == "0" and c[1] == "0")]),
    "logs/run_everything_20260930_042722.txt", "candidate offsets searched (excluding [0,0])")
ct = (RUNS.parent / "logs" / "corrected_20260930_054214" / "crop_check_after.txt").read_text()
mm = re.search(r"residual dy\s+median ([+-][0-9.]+) px\s+IQR \[([+-][0-9.]+), ([+-][0-9.]+)\]\s+5-95% \[([+-][0-9.]+), ([+-][0-9.]+)\]", ct)
for i, n in enumerate(("median", "iqr_lo", "iqr_hi", "p5", "p95")):
    put(f"crop.test.dy_{n}", float(mm.group(i + 1)), "logs/corrected_20260930_054214/crop_check_after.txt", f"residual dy {n}")
mm = re.search(r"residual dx\s+median ([+-][0-9.]+) px\s+IQR \[([+-][0-9.]+), ([+-][0-9.]+)\]\s+5-95% \[([+-][0-9.]+), ([+-][0-9.]+)\]", ct)
for i, n in enumerate(("median", "iqr_lo", "iqr_hi", "p5", "p95")):
    put(f"crop.test.dx_{n}", float(mm.group(i + 1)), "logs/corrected_20260930_054214/crop_check_after.txt", f"residual dx {n}")
put("crop.test_fields", int(re.search(r"test split, (\d+) fields", ct).group(1)), "logs/corrected_20260930_054214/crop_check_after.txt", "fields")
la = load_json(RUNS / "label_audit_thresholds.json")
for k in ("level_mean_rad", "level_sd_rad", "anova_F", "anova_p", "n_images"):
    put(f"otsu.{k}", la[k], "runs/label_audit_thresholds.json", k)
# membrane-derived scoring (A, 3 seeds)
mb = load_json(RUNS / "benchmark_results" / "results_arm_A_membrane.json")
for k in ("seg_dice", "seg_aji", "seg_boundary_f1", "detection_recall"):
    xs = [r["metrics"][k] for r in mb["runs"]]
    put(f"membrane.A.mean.{k}", sum(xs) / len(xs), "runs/benchmark_results/results_arm_A_membrane.json", k, "mean over seeds")
    put(f"membrane.A.sd.{k}", sd(xs), "runs/benchmark_results/results_arm_A_membrane.json", k, "SD over seeds")
put("membrane.A.n_runs", len(mb["runs"]), "runs/benchmark_results/results_arm_A_membrane.json", "runs")

print("step 12/12: writing values.json")
OUT.write_text(json.dumps(VALUES, indent=1, default=str))
print(f"  {len(VALUES)} values -> {rel(OUT)}")
