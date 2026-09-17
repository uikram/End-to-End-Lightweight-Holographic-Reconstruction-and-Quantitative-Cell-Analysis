"""Where is the study right now? Read-only, safe to run at any moment.

WHY THIS EXISTS
---------------
Two passes train at once on two GPUs, each writing its own narration file and
its own log directory, and each arm takes over an hour. Watching that with
`tail -f` puts two unrelated streams into one terminal interleaved, which is
how the last check turned into a guess about which pass had got where.

This reads the files instead and prints one block per pass: the plan, what is
finished, what is running, and -- from the epoch lines of the arm currently
training -- how long the rest is likely to take. It opens nothing for writing
and holds no locks, so running it while the study runs cannot disturb it.

    python scripts/progress.py                     once
    python scripts/progress.py --watch             refresh every 60 s, Ctrl-C to stop
    python scripts/progress.py p_main.out          just one pass

Nothing here is a substitute for `logs/v2_*/SUMMARY.txt`, which is the record.
This is the dashboard.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The arms that stage 6 replicates over extra seeds. Kept in step with the
# `for arm in A B B1` loop in run_v2.sh; if that loop changes, change this.
SEED_ARMS = ("A", "B", "B1")

RE_LOGDIR = re.compile(r"^log directory (\S+)")
RE_HEADER = re.compile(r"^GPU (\S*)\s+arms: (.*?)\s+extra seeds: (.*)$")
RE_STEP = re.compile(r"^=== (\S+)\s+(\d\d:\d\d:\d\d)")
RE_OK = re.compile(r"^    OK   -> (\S+)")
RE_VERDICT = re.compile(r"^    VERDICT exit (\d+)")
RE_FAILED = re.compile(r"^    FAILED \(exit (\d+)")
RE_EPOCH = re.compile(r"epoch (\d+)/(\d+) \((\d+)s")

# A narration file written this recently is between steps, not abandoned. Long
# enough to cover a step's start-up, short enough that a real stall still shows.
IDLE_GRACE = 600.0


def hms(seconds: float) -> str:
    """Seconds as the coarsest useful unit. 4500 -> '1h 15m'."""
    seconds = int(max(seconds, 0))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def read_pass(out_file: Path) -> dict:
    """Everything knowable about one pass, from its narration file alone."""
    info = {
        "file": out_file.name,
        "idle": time.time() - out_file.stat().st_mtime,
        "logdir": None,
        "gpu": "?",
        "arms": [],
        "seeds": [],
        "no_train": False,
        "no_plan": False,
        "done": [],       # (name, start, outcome)
        "running": None,  # (name, start)
        "started": None,
    }
    lines = out_file.read_text(errors="replace").splitlines()

    pending: tuple[str, str] | None = None
    for line in lines:
        if "NO_TRAIN" in line:
            info["no_train"] = True
            continue
        if "NO_PLAN" in line:
            info["no_plan"] = True
            continue
        if info["logdir"] is None:
            m = RE_LOGDIR.match(line)
            if m:
                info["logdir"] = m.group(1)
                continue
        m = RE_HEADER.match(line)
        if m:
            info["gpu"] = m.group(1) or "unset"
            info["arms"] = m.group(2).split()
            seeds = m.group(3).strip()
            info["seeds"] = [] if seeds in ("none", "") else seeds.split()
            continue
        m = RE_STEP.match(line)
        if m:
            if pending is not None:          # a step with no outcome line: it
                info["done"].append((*pending, "?"))   # was interrupted
            pending = (m.group(1), m.group(2))
            if info["started"] is None:
                info["started"] = m.group(2)
            continue
        if pending is None:
            continue
        if RE_OK.match(line):
            info["done"].append((*pending, "ok"))
            pending = None
        elif RE_VERDICT.match(line):
            info["done"].append((*pending, "verdict"))
            pending = None
        elif RE_FAILED.match(line):
            info["done"].append((*pending, "FAILED"))
            pending = None

    info["running"] = pending
    return info


def plan(info: dict) -> list[str]:
    """The step names stage 6 will run, in order, for this pass.

    NO_TRAIN=1 skips the base train of each arm and keeps everything else --
    the evaluations and, in run_v2.sh, the replication seeds. study.sh writes a
    marker line into those narration files, because otherwise every skipped
    train looks like a step that never ran.
    """
    if info["no_plan"]:
        # Stages other than 6 are not an arm sweep, so there is no plan to
        # measure against. run_v2.sh still prints its default ARMS list in the
        # header, and reading that as a plan invented 39 steps that were never
        # going to run.
        return []
    steps: list[str] = []
    membrane = (ROOT / "data" / "membrane_mask").is_dir()
    for arm in info["arms"]:
        if not info["no_train"]:
            steps.append(f"train_{arm}")
        steps.append(f"evaluate_{arm}")
        if membrane:
            steps.append(f"evaluate_{arm}_membrane")
    for seed in info["seeds"]:
        for arm in SEED_ARMS:
            if arm in info["arms"]:
                steps.append(f"train_{arm}_seed{seed}")
                steps.append(f"evaluate_{arm}_seed{seed}")
    if info["seeds"]:
        steps.append("aggregate_seeds")
    return steps


def epoch_state(logdir: str | None, step_name: str) -> tuple[int, int, float] | None:
    """(epoch, total, mean seconds per epoch) from a training step's own log."""
    if not logdir:
        return None
    path = Path(logdir) / f"{step_name}.log"
    if not path.is_file():
        path = ROOT / logdir / f"{step_name}.log"
        if not path.is_file():
            return None
    epochs = RE_EPOCH.findall(path.read_text(errors="replace"))
    if not epochs:
        return None
    last = epochs[-1]
    seconds = [float(e[2]) for e in epochs[-5:]]
    return int(last[0]), int(last[1]), sum(seconds) / len(seconds)


def elapsed_since(clock: str) -> float:
    """Seconds from an HH:MM:SS stamp to now, assuming it is today or yesterday."""
    now = datetime.now()
    try:
        h, m, s = (int(x) for x in clock.split(":"))
    except ValueError:
        return 0.0
    then = now.replace(hour=h, minute=m, second=s, microsecond=0)
    if then > now:
        then -= timedelta(days=1)
    return (now - then).total_seconds()


def mean_train_seconds(info: dict) -> float | None:
    """How long a finished train step took in this pass, on average."""
    order = [(name, clock) for name, clock, _ in info["done"]]
    spans = []
    for i, (name, clock) in enumerate(order):
        if not name.startswith("train_") or i + 1 >= len(order):
            continue
        spans.append(elapsed_since(clock) - elapsed_since(order[i + 1][1]))
    spans = [s for s in spans if s > 0]
    return sum(spans) / len(spans) if spans else None


def render(info: dict) -> list[str]:
    out = []
    steps = plan(info)
    done_names = {name for name, _, _ in info["done"]}
    bad = [(n, o) for n, _, o in info["done"] if o in ("FAILED", "?")]
    finished = [s for s in steps if s in done_names]

    if info["no_plan"]:
        ran = [name for name, _, _ in info["done"]]
        out.append(f"  {info['file']:<16} GPU {info['gpu']}   the finishing stages")
        out.append(f"  {'':<16} {len(ran)} step(s) done"
                   + (f": {' '.join(ran[-4:])}" if ran else "")
                   + (f"   [{len(bad)} not ok]" if bad else ""))
        if info["running"] is not None:
            name, clock = info["running"]
            out.append(f"  {'':<16} now: {name}   started {clock}"
                       f"  ({hms(elapsed_since(clock))} ago)")
        for name, outcome in bad:
            out.append(f"  {'':<16} !! {name}: {outcome}")
        return out

    out.append(f"  {info['file']:<16} GPU {info['gpu']}   arms: {' '.join(info['arms'])}"
               f"   seeds: {' '.join(info['seeds']) or 'none'}"
               + ("   (evaluations only)" if info["no_train"] else ""))
    out.append(f"  {'':<16} {len(finished)} of {len(steps)} steps done"
               + (f"   [{len(bad)} not ok]" if bad else ""))

    running = info["running"]
    if running is None:
        if len(finished) >= len(steps) and not bad:
            out.append(f"  {'':<16} nothing running -- this pass has FINISHED")
        elif info["idle"] < IDLE_GRACE:
            # No step is open, but the file was written moments ago: the runner
            # is between two steps. Calling that "stopped" -- which an earlier
            # version did -- turns a one-second gap into a false alarm.
            out.append(f"  {'':<16} between steps (last write"
                       f" {hms(info['idle'])} ago)")
        else:
            # Nothing open and nothing written for a while. Either the process
            # was killed or the plan does not match what was launched; say so
            # rather than calling it finished.
            out.append(f"  {'':<16} nothing running, and {len(steps) - len(finished)}"
                       f" planned step(s) have no record -- the pass STOPPED early,"
                       f" or it was launched with different ARMS/SEEDS")
            missing = [s for s in steps if s not in done_names][:4]
            if missing:
                out.append(f"  {'':<16} never ran: {' '.join(missing)}"
                           + (" ..." if len(steps) - len(finished) > 4 else ""))
    else:
        name, clock = running
        run_for = elapsed_since(clock)
        out.append(f"  {'':<16} now: {name}   started {clock}  ({hms(run_for)} ago)")
        state = epoch_state(info["logdir"], name)
        if state:
            epoch, total, per = state
            left_here = (total - epoch) * per
            out.append(f"  {'':<16}      epoch {epoch}/{total}"
                       f"   {per:.0f}s/epoch   ~{hms(left_here)} left in this arm")
        elif name.startswith("train_"):
            out.append(f"  {'':<16}      no epoch line yet (still building the model"
                       f" or loading data)")

        # everything after the running step
        idx = steps.index(name) if name in steps else -1
        remaining = steps[idx + 1:] if idx >= 0 else []
        trains_left = sum(1 for s in remaining if s.startswith("train_"))
        mean = mean_train_seconds(info)
        if trains_left and mean:
            eta = trains_left * mean + (left_here if state else 0.0)
            done_at = datetime.now() + timedelta(seconds=eta)
            out.append(f"  {'':<16} {trains_left} more arm(s) to train"
                       f"   ~{hms(eta)}   finishes about {done_at:%H:%M}")
        elif remaining:
            out.append(f"  {'':<16} then: {' '.join(remaining[:4])}"
                       + (" ..." if len(remaining) > 4 else ""))

    for name, outcome in bad:
        out.append(f"  {'':<16} !! {name}: {outcome}")
    return out


def gpu_line() -> list[str]:
    try:
        raw = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if raw.returncode != 0:
        return []
    rows = []
    for line in raw.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append(f"GPU{parts[0]} {parts[1]:>3}% "
                        f"{int(parts[2]) / 1024:.1f}/{int(parts[3]) / 1024:.0f}G")
    return ["  " + "    ".join(rows)] if rows else []


def once(out_files: list[Path], header: bool = True) -> None:
    print()
    if header:
        print(f"  HoloQPI v2 study — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print()
    if not out_files:
        print("  No p_*.out file found. Run this from the project root, or pass the")
        print("  narration file explicitly: python scripts/progress.py p_main.out")
        return
    for path in out_files:
        for line in render(read_pass(path)):
            print(line)
        print()
    for line in gpu_line():
        print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="*", help="narration files; default every p_*.out")
    parser.add_argument("--watch", action="store_true",
                        help="redraw every --every seconds until Ctrl-C")
    parser.add_argument("--every", type=int, default=60, help="seconds between redraws")
    parser.add_argument("--no-header", action="store_true",
                        help="omit the title line (study.sh prints its own)")
    args = parser.parse_args()

    def resolve() -> list[Path]:
        if args.out:
            return [Path(p) for p in args.out if Path(p).is_file()]
        return sorted(Path(p) for p in glob.glob("p_*.out") + glob.glob("v2_*.out"))

    if not args.watch:
        once(resolve(), header=not args.no_header)
        return 0

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            once(resolve(), header=not args.no_header)
            print(f"  refreshing every {args.every}s — Ctrl-C to stop")
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
