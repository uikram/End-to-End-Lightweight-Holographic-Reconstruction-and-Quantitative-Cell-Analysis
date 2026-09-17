"""Is this machine ready to run the v2 study? Answer before burning GPU hours.

WHY THIS EXISTS
---------------
The study runs on a different machine from the one the code is edited on, and
the two drift. A missing script is obvious the moment you run it; a STALE one is
not, and that is the dangerous case: the old encoder logged `pretrained=True`
whether or not it had actually obtained ImageNet weights, so a whole study could
train from scratch with nothing in the log to catch it.

So this checks three separate things and never guesses:

  PRESENT   does the file exist at all
  CURRENT   is it the version that carries the fixes, tested by looking for a
            specific marker inside it rather than by a date or a version string
  WORKING   does it actually do the thing, run right here

Everything it reports is a fact it just established on this machine. It exits
non-zero when something would invalidate a run, so it can gate a job script.

    python scripts/preflight.py
    python scripts/preflight.py --config config/base.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# file -> (marker that only the fixed version contains, what the fix is)
FRESHNESS = {
    "holoqpi/models/encoders.py": (
        "_LOCAL_WEIGHT_DIR",
        "loads ImageNet weights from model.pretrained_dir and logs which route it took",
    ),
    "holoqpi/models/holonet.py": (
        "decoder_bottleneck",
        "shared 1x1 bottleneck (9.60M -> 3.36M) and the fixed ONNX ExportWrapper",
    ),
    "holoqpi/deploy/export.py": (
        "active_outputs",
        "derives ONNX output names from the heads that exist",
    ),
    "holoqpi/losses/terms.py": (
        "class CellProjectedArea",
        "per-cell area loss and the amplitude loss",
    ),
    "holoqpi/metrics/measurement.py": (
        "coverage_adjusted",
        "coverage-adjusted error, field totals and the field-level bootstrap",
    ),
    "holoqpi/engine/trainer.py": (
        "provide_amplitude",
        "passes the amplitude target through, and logs the whole objective",
    ),
    "holoqpi/engine/evaluator.py": (
        "provide_amplitude",
        "passes the amplitude target through",
    ),
    "main.py": (
        "unmatched_{args.split}{tag}",
        "evaluate --tag, and the tagged unmatched_*.csv that figure 15 reads",
    ),
    "scripts/progress.py": ("epoch_state", "read-only status of both training passes"),
    "scripts/prepare_membrane.py": ("illumination_window", "membrane registration"),
    "scripts/prepare_amplitude.py": ("clip_max", "amplitude reference"),
    "scripts/amplitude_sensitivity.py": ("ANTI-DISCRIMINATIVE", "forward-model usability test"),
    "scripts/error_propagation.py": ("shift_boundary", "boundary -> measurement error"),
    "scripts/synthetic_validation.py": ("phase_sum_exact", "analytic ground-truth floor"),
    "scripts/collect_results.py": ("seed_spread", "assembles every table"),
    "scripts/check_gradient_path.py": ("cosine", "multi-batch ratio + cosine + weight table"),
    "scripts/estimate_aberration.py": ("__global__", "deployable global aberration surface"),
    "scripts/make_figures.py": ("figure_membrane", "figures 17-20"),
    "run_v2.sh": ("membrane_registration", "the ten-stage runner"),
    "study.sh": ("NO_PLAN", "the one-command driver: stop, plan, train, finish, watch"),
    "config/base.yaml": ("pretrained_dir", "pretrained_dir + the membrane block"),
    "config/v2/b1_image_volume.yaml": ("phase_volume", "experiment B' (the missing control)"),
    "config/v2/k_compact_b.yaml": ("decoder_bottleneck", "compact-decoder arms"),
}

CONFIG_KEYS = [
    ("membrane.scale", "membrane registration constants"),
    ("membrane.offset_y", "membrane registration constants"),
    ("model.pretrained_dir", "local ImageNet weights"),
    ("model.decoder_bottleneck", "compact decoder"),
    ("loss.weights.cell_projected_area", "per-cell area loss"),
    ("loss.weights.amplitude", "amplitude loss"),
    ("evaluation.measurement.report_coverage_adjusted", "coverage-adjusted error"),
    ("evaluation.measurement.field_bootstrap_resamples", "field bootstrap"),
    ("optics.aberration.mode", "deployable aberration surface"),
]

ARMS = ["a_baseline", "b_cell_ipp", "b1_image_volume", "b2_cell_area",
        "c_cell_ipp_bga", "d0_amplitude", "d_forward_amplitude",
        "w_ipp_01", "w_ipp_03", "w_ipp_10", "w_ipp_30",
        "k_compact_a", "k_compact_b"]

PASS, WARN, FAIL = "ok  ", "WARN", "FAIL"
_counts = {PASS: 0, WARN: 0, FAIL: 0}


def report(status: str, label: str, detail: str = "") -> None:
    _counts[status] += 1
    print(f"  [{status}] {label:<52} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()

    print(f"\nHoloQPI preflight — {ROOT}")
    print(f"python {sys.version.split()[0]}")

    # ---- 1. the machine ------------------------------------------------
    print("\n=== 1. environment ===")
    try:
        import torch

        report(PASS, "torch", torch.__version__)
        if torch.cuda.is_available():
            names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
            report(PASS, "CUDA", f"{torch.cuda.device_count()} device(s): {', '.join(names)}")
        else:
            report(FAIL, "CUDA", "not available — training would run on CPU, which is weeks")
    except ImportError:
        report(FAIL, "torch", "not importable")

    for module in ("numpy", "scipy", "skimage", "matplotlib", "tifffile", "yaml", "onnx"):
        if importlib.util.find_spec(module) is not None:
            report(PASS, module, "")
        else:
            severity = WARN if module in ("onnx",) else FAIL
            report(severity, module, "missing" + (" (needed for stage 8 export)"
                                                  if module == "onnx" else ""))

    # ---- 2. are the files here, and are they the fixed versions? -------
    print("\n=== 2. code freshness (marker found inside the file, not a version string) ===")
    for relative, (marker, what) in sorted(FRESHNESS.items()):
        path = ROOT / relative
        if not path.is_file():
            report(FAIL, relative, f"MISSING — {what}")
            continue
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            report(WARN, relative, f"unreadable: {exc}")
            continue
        if marker in text:
            report(PASS, relative, what)
        else:
            report(FAIL, relative, f"STALE (no '{marker}') — {what}")

    # ---- 3. the config actually parses and carries the new keys --------
    print("\n=== 3. configuration ===")
    cfg = None
    try:
        from holoqpi.config import load_config

        cfg = load_config(args.config)
        report(PASS, args.config, "parses")
    except Exception as exc:
        report(FAIL, args.config, f"{type(exc).__name__}: {exc}")

    if cfg is not None:
        for dotted, what in CONFIG_KEYS:
            node = cfg
            try:
                for part in dotted.split("."):
                    node = getattr(node, part)
                report(PASS, dotted, f"{node}")
            except Exception:
                report(FAIL, dotted, f"absent — {what}")

        missing_arms = [a for a in ARMS if not (ROOT / "config" / "v2" / f"{a}.yaml").is_file()]
        if missing_arms:
            report(FAIL, "config/v2 arms", f"{len(ARMS) - len(missing_arms)}/{len(ARMS)}; "
                                           f"missing {', '.join(missing_arms)}")
        else:
            report(PASS, "config/v2 arms", f"all {len(ARMS)} present")

    # ---- 4. the data ---------------------------------------------------
    print("\n=== 4. data ===")
    if cfg is not None:
        root = ROOT / cfg.paths.data_root
        if not root.is_dir():
            report(FAIL, str(root), "data_root does not exist")
        else:
            report(PASS, "data_root", str(root))
            manifest = root / cfg.paths.manifest_file
            fields = 0
            if manifest.is_file():
                import csv

                with open(manifest, newline="") as handle:
                    fields = sum(1 for _ in csv.DictReader(handle))
                report(PASS, "manifest", f"{fields} fields")
            else:
                report(WARN, "manifest", "absent — run `python main.py prepare`")

            for label, directory in (
                ("phase", cfg.paths.phase_dir),
                ("off-axis holograms", cfg.paths.hologram_dirs.off_axis),
                ("masks", cfg.paths.mask_dir),
            ):
                path = root / directory
                count = len(list(path.glob("*"))) if path.is_dir() else 0
                if count:
                    report(PASS, label, f"{count} files in {directory}/")
                else:
                    report(FAIL, label, f"nothing in {directory}/")

            membrane = root / cfg.membrane.source_dir
            count = len(list(membrane.glob("*_membrane.tif"))) if membrane.is_dir() else 0
            if count == 0:
                report(WARN, "membrane channel", f"no *_membrane.tif in "
                                                 f"{cfg.membrane.source_dir}/ — the "
                                                 f"independent labels are unavailable")
            elif fields and count < fields:
                report(WARN, "membrane channel", f"{count} files for {fields} fields; "
                                                 f"the rest will be skipped")
            else:
                report(PASS, "membrane channel", f"{count} files")

            for label, directory in (
                ("aberration surfaces", cfg.paths.aberration_file),
                ("amplitude reference", cfg.paths.amplitude_dir),
                ("membrane, aligned", cfg.membrane.output_dir),
            ):
                path = root / directory
                if path.exists():
                    report(PASS, label, "present")
                else:
                    report(WARN, label, f"absent — produced by stage "
                                        f"{'2' if 'aberration' in label else '3' if 'amplitude' in label else '1'}")

    # ---- 5. the ImageNet weights, and which route they take ------------
    print("\n=== 5. pretrained encoder ===")
    if cfg is not None:
        directory = getattr(cfg.model, "pretrained_dir", None)
        if directory:
            path = ROOT / directory
            found = sorted(path.glob("*.pth")) if path.is_dir() else []
            if found:
                report(PASS, f"pretrained_dir ({directory})",
                       ", ".join(p.name for p in found))
            else:
                report(WARN, f"pretrained_dir ({directory})",
                       "no .pth inside; the download cache will be used instead")
        else:
            report(WARN, "model.pretrained_dir", "not set; the download cache will be used")

        # The only test that matters: build it and see what the log says.
        try:
            from holoqpi.models import build_model

            build_model(cfg)
            report(PASS, "build_model", "succeeded — read the encoder's `weights=` line above")
        except Exception as exc:
            report(FAIL, "build_model", f"{type(exc).__name__}: {exc}")

    # ---- 6. disk -------------------------------------------------------
    print("\n=== 6. disk ===")
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 1e9
    # Thirteen arms keep two checkpoints each at roughly 115 MB per copy, plus
    # seeds, ONNX exports and figures.
    report(PASS if free_gb > 40 else WARN if free_gb > 15 else FAIL,
           "free space", f"{free_gb:.0f} GB (13 arms x 2 checkpoints ~ 3 GB, "
                         f"plus seeds and exports)")

    # ---- verdict -------------------------------------------------------
    print(f"\n=== {_counts[PASS]} ok, {_counts[WARN]} warnings, {_counts[FAIL]} failures ===")
    if _counts[FAIL]:
        print("\n  -> NOT READY. Every FAIL above is something that would either stop a run")
        print("     or silently change what it measures. A STALE file is the dangerous")
        print("     one: the code runs, it just is not the code the configs assume.")
        print("     Copy the current tree over and run this again.")
        return 1
    if _counts[WARN]:
        print("\n  -> READY, with warnings. Each warning names the stage that produces the")
        print("     missing thing, so work through them in order rather than skipping to")
        print("     training.")
        return 0
    print("\n  -> READY. Start with the stages that need no GPU:")
    print("       for s in 0 1 2 3 4 5; do bash run_v2.sh --stage $s; done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
