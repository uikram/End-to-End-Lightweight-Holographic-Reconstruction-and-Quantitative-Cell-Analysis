"""Remove the previous red-blood-cell framework from this repository.

The new study replaces the old codebase entirely: different data, different
task, different architecture. Everything the old pipeline needed is listed here
and deleted, leaving only the new framework and the `data/` directory.

The script prints a plan first and does nothing without an explicit flag:

    python scripts/cleanup_legacy.py                 # show what would be removed
    python scripts/cleanup_legacy.py --apply         # remove it
    python scripts/cleanup_legacy.py --apply --keep-manuscript
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Directories from the previous RBC study, plus the vendored MMDetection tree.
LEGACY_DIRECTORIES = [
    "projects",                     # vendored third-party detection models, never imported
    "results",                      # RBC rank sweep, ablation and benchmark outputs
    "weights",                      # EdgeSAM / MobileSAM checkpoints
    "morphology_analysis",
    "training_validation_plots",
    "global_ablations_and_comparisons",
    "fig",
    "visualization",
    "benchmark",
    "evaluation",
    "training",
    "datasets",
    "analysis",
    "utils",
    "models",
    "configs",
    "cache",
]

LEGACY_FILES = [
    "main.py.bak",
    "train.py",
    "demo.py",
    "dry_mass.py",
    "calibrated_dry_mass.py",
    "collect_metrics.py",
    "extract_visual.py",
    "yaml_gen.py",
    "ablation_curve.png",
    "drymass_validation.png",
    "lora_vs_full_finetune.png",
    "loss_ablation.png",
    "pareto_frontier.png",
    "per_class_dice.png",
    "population_shift.png",
    "radar_chart.png",
]

MANUSCRIPT_FILES = ["lightweight-segmentation-ui.pdf"]

# Never removed, whatever else happens.
PROTECTED = {"data", "holoqpi", "config", "scripts", "docs", "runs", "main.py",
             "README.md", "requirements.txt", ".git", ".gitignore"}


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def collect_targets(root: Path, keep_manuscript: bool) -> list[Path]:
    names = list(LEGACY_DIRECTORIES) + list(LEGACY_FILES)
    if not keep_manuscript:
        names += MANUSCRIPT_FILES

    targets = []
    for name in names:
        if name in PROTECTED:
            continue
        candidate = root / name
        if candidate.exists():
            targets.append(candidate)
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    parser.add_argument("--apply", action="store_true", help="actually delete; otherwise dry-run")
    parser.add_argument("--keep-manuscript", action="store_true",
                        help="keep the previous study's PDF for citation")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    targets = collect_targets(root, args.keep_manuscript)

    if not targets:
        print("Nothing to remove: the repository is already clean.")
        return 0

    total = 0
    print(f"{'ACTION':8}  {'SIZE':>10}  PATH")
    print("-" * 70)
    for target in targets:
        size = _tree_size(target)
        total += size
        action = "DELETE" if args.apply else "would"
        marker = "/" if target.is_dir() else ""
        print(f"{action:8}  {_human_size(size):>10}  {target.relative_to(root)}{marker}")
    print("-" * 70)
    print(f"{'':8}  {_human_size(total):>10}  across {len(targets)} entries")

    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    removed = 0
    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            removed += 1
        except OSError as exc:
            print(f"  could not remove {target}: {exc}", file=sys.stderr)

    print(f"\nRemoved {removed}/{len(targets)} entries, freeing {_human_size(total)}.")
    print("Remaining framework: holoqpi/, config/, scripts/, docs/, data/, main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
