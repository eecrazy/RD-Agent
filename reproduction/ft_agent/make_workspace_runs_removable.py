"""Keep generated immutable run directories removable for legacy workers.

FT-Agent workers started before ``FBWorkspace`` learned to remove read-only
trees can fail while restoring a checkpoint if generated data processing code
marks ``workspace/*/processed_runs/*`` directories ``0555``.  This operational
guard adds only owner-write permission to those directories.  It never follows
symlinks, changes file contents, or changes payload-file permissions.
"""

from __future__ import annotations

import argparse
import os
import stat
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Matrix run roots to guard")
    parser.add_argument("--interval", type=float, default=0.25, help="Polling interval in seconds")
    parser.add_argument("--heartbeat", type=float, default=30.0, help="Heartbeat interval in seconds")
    parser.add_argument("--once", action="store_true", help="Scan once and exit")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.heartbeat <= 0:
        parser.error("--heartbeat must be positive")
    return args


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def candidate_directories(root: Path) -> Iterator[Path]:
    """Yield workspace-local immutable data-run directories."""
    try:
        task_roots = tuple(root.iterdir())
    except OSError:
        return
    for task_root in task_roots:
        workspace = task_root / "workspace"
        if not workspace.is_dir():
            continue
        try:
            iterations = tuple(workspace.iterdir())
        except OSError:
            continue
        for iteration in iterations:
            processed_runs = iteration / "processed_runs"
            if not processed_runs.is_dir() or processed_runs.is_symlink():
                continue
            try:
                runs = tuple(processed_runs.iterdir())
            except OSError:
                continue
            for run in runs:
                if run.is_dir() and not run.is_symlink():
                    yield run


def make_removable(directory: Path) -> str:  # noqa: C901
    """Add owner-write permission to one safe local directory if needed."""
    try:
        before = directory.lstat()
    except FileNotFoundError:
        outcome = "raced"
    except OSError:
        outcome = "error"
    else:
        if not stat.S_ISDIR(before.st_mode):
            outcome = "raced"
        elif before.st_uid != os.geteuid():
            outcome = "foreign_owner"
        elif before.st_mode & stat.S_IWUSR:
            outcome = "already_writable"
        else:
            try:
                directory.chmod(stat.S_IMODE(before.st_mode) | stat.S_IWUSR, follow_symlinks=False)
                after = directory.lstat()
            except FileNotFoundError:
                outcome = "raced"
            except OSError:
                outcome = "error"
            else:
                same_directory = (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
                if not same_directory or not stat.S_ISDIR(after.st_mode):
                    outcome = "raced"
                else:
                    outcome = "made_removable" if after.st_mode & stat.S_IWUSR else "error"
    return outcome


def scan(roots: list[Path], totals: dict[str, int]) -> None:
    for root in roots:
        for directory in candidate_directories(root):
            outcome = make_removable(directory)
            totals[outcome] = totals.get(outcome, 0) + 1
            if outcome == "made_removable":
                print(f"{timestamp()} MADE_REMOVABLE {directory}", flush=True)


def main() -> None:
    args = parse_args()
    roots = [root.resolve() for root in args.roots]
    totals: dict[str, int] = {}
    next_heartbeat = time.monotonic()
    while True:
        scan(roots, totals)
        now = time.monotonic()
        if now >= next_heartbeat:
            counts = ",".join(f"{key}={value}" for key, value in sorted(totals.items())) or "no-runs"
            print(f"{timestamp()} WORKSPACE_MODE_GUARD {counts}", flush=True)
            next_heartbeat = now + args.heartbeat
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
