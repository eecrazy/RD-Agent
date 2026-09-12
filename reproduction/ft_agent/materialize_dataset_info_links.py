"""Atomically materialize generated workspace-local dataset-info symlinks.

This is an operational compatibility guard for FT-Agent workers that were
started before ``FBWorkspace.inject_files`` learned to replace symlinks.  It
only touches ``workspace/*/dataset_info.json`` links whose targets are regular,
valid JSON files inside the same task workspace.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import tempfile
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


def candidate_links(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield task-workspace roots and top-level dataset-info symlinks."""
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
            candidate = iteration / "dataset_info.json"
            if candidate.is_symlink():
                yield workspace, candidate


def materialize(workspace: Path, link: Path) -> str:
    """Replace one safe symlink atomically and return an outcome label."""
    try:
        before = link.lstat()
        target = link.resolve(strict=True)
        workspace_resolved = workspace.resolve(strict=True)
    except FileNotFoundError:
        return "dangling"
    except OSError:
        return "raced"

    try:
        target.relative_to(workspace_resolved)
    except ValueError:
        return "external"
    if not target.is_file():
        return "non_file"

    try:
        payload = target.read_bytes()
        json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid_json"

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=link.parent,
            prefix=".dataset_info.materialize-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(stat.S_IMODE(target.stat().st_mode))

        after = link.lstat()
        if not stat.S_ISLNK(after.st_mode) or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            outcome = "raced"
        else:
            temporary_path.replace(link)
            temporary_path = None
            outcome = "materialized"
    except FileNotFoundError:
        outcome = "raced"
    except OSError:
        outcome = "error"
    finally:
        if temporary_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()
    return outcome


def scan(roots: list[Path], totals: dict[str, int]) -> None:
    for root in roots:
        for workspace, link in candidate_links(root):
            outcome = materialize(workspace, link)
            totals[outcome] = totals.get(outcome, 0) + 1
            if outcome == "materialized":
                print(f"{timestamp()} MATERIALIZED {link}", flush=True)


def main() -> None:
    args = parse_args()
    roots = [root.resolve() for root in args.roots]
    totals: dict[str, int] = {}
    next_heartbeat = time.monotonic()
    while True:
        scan(roots, totals)
        now = time.monotonic()
        if now >= next_heartbeat:
            counts = ",".join(f"{key}={value}" for key, value in sorted(totals.items())) or "no-links"
            print(f"{timestamp()} DATASET_LINK_GUARD {counts}", flush=True)
            next_heartbeat = now + args.heartbeat
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
