#!/usr/bin/env python3
"""Run a frozen matrix manager's API threads without scheduling more tasks.

The matrix manager owns the ephemeral Responses adapter used by its children.
It was intentionally group-stopped after duplicate work was discovered, which
also stopped that adapter.  Keep the asyncio scheduler thread in SCHED_IDLE on
one deliberately busy CPU while allowing every other thread to run normally.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/github/RD-Agent")
MANAGER_PID = 38651
MANAGER_STARTTIME = 592896965
EXPECTED_LISTENER = "socket:[3857750359]"
POLL_SECONDS = 0.01
HEARTBEAT_SECONDS = 30.0
HOG_COUNT = 2


def log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat()} {message}", flush=True)


def proc_stat(pid: int) -> tuple[str, int, int, int, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    fields = raw.rsplit(")", 1)[1].split()
    return fields[0], int(fields[1]), int(fields[19]), int(fields[11]), int(fields[12])


def expected_manager(*, require_stopped: bool = False) -> tuple[str, int, int, int, int]:
    stat = proc_stat(MANAGER_PID)
    if stat is None or stat[2] != MANAGER_STARTTIME:
        raise RuntimeError(f"manager identity mismatch: {stat}")
    if require_stopped and stat[0] not in {"T", "t"}:
        raise RuntimeError(f"manager is not stopped: {stat}")
    return stat


def direct_children() -> set[tuple[int, int]]:
    children: set[tuple[int, int]] = set()
    children_path = Path(f"/proc/{MANAGER_PID}/task/{MANAGER_PID}/children")
    try:
        child_pids = [int(value) for value in children_path.read_text(encoding="utf-8").split()]
    except FileNotFoundError:
        return children
    for pid in child_pids:
        stat = proc_stat(pid)
        if stat is not None and stat[1] == MANAGER_PID:
            children.add((pid, stat[2]))
    return children


def stop_manager(reason: str) -> None:
    stat = proc_stat(MANAGER_PID)
    if stat is not None and stat[2] == MANAGER_STARTTIME:
        with contextlib.suppress(ProcessLookupError):
            os.kill(MANAGER_PID, signal.SIGSTOP)
        log(f"SAFETY_STOP manager={MANAGER_PID} reason={reason}")


def hog(cage_cpu: int) -> None:
    os.sched_setaffinity(0, {cage_cpu})
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    signal.signal(signal.SIGTERM, lambda *_args: raise_exit())
    while True:
        pass


def raise_exit() -> None:
    raise SystemExit(0)


def spawn_hog(cage_cpu: int) -> int:
    pid = os.fork()
    if pid == 0:
        try:
            hog(cage_cpu)
        finally:
            os._exit(0)
    return pid


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def reap_hogs(hogs: set[int]) -> None:
    for pid in tuple(hogs):
        waited, _status = os.waitpid(pid, os.WNOHANG)
        if waited:
            hogs.remove(pid)


def terminate_hogs(hogs: set[int]) -> None:
    for pid in hogs:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 2
    while hogs and time.monotonic() < deadline:
        reap_hogs(hogs)
        if hogs:
            time.sleep(0.01)
    for pid in hogs:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    for pid in tuple(hogs):
        with contextlib.suppress(ChildProcessError, ProcessLookupError):
            os.waitpid(pid, 0)
        hogs.discard(pid)


def main() -> int:
    lock_path = ROOT / "finetune_files/logs/orchestration/frozen_manager_api_rescue.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another API rescue process is already active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    initial = expected_manager(require_stopped=True)
    listener = os.readlink(f"/proc/{MANAGER_PID}/fd/3")
    if listener != EXPECTED_LISTENER:
        raise RuntimeError(f"unexpected manager fd 3: {listener}")
    baseline_children = direct_children()
    allowed_cpus = os.sched_getaffinity(MANAGER_PID)
    if len(allowed_cpus) < 2:
        raise RuntimeError(f"at least two CPUs are required: {allowed_cpus}")
    cage_cpu = max(allowed_cpus)
    worker_cpus = allowed_cpus - {cage_cpu}

    # Every existing non-main thread can serve adapter traffic. Threads later
    # created by ThreadingHTTPServer inherit the serving thread's worker mask.
    for task in Path(f"/proc/{MANAGER_PID}/task").iterdir():
        tid = int(task.name)
        if tid != MANAGER_PID:
            try:
                os.sched_setaffinity(tid, worker_cpus)
            except ProcessLookupError:
                pass
    os.sched_setaffinity(MANAGER_PID, {cage_cpu})
    os.sched_setscheduler(MANAGER_PID, os.SCHED_IDLE, os.sched_param(0))

    hogs = {spawn_hog(cage_cpu) for _ in range(HOG_COUNT)}
    time.sleep(0.05)
    reap_hogs(hogs)
    if len(hogs) != HOG_COUNT or not all(alive(pid) for pid in hogs):
        terminate_hogs(hogs)
        raise RuntimeError("failed to establish both scheduler starvation guards")

    os.kill(MANAGER_PID, signal.SIGCONT)
    log(
        f"API_RESCUE manager={MANAGER_PID} cage_cpu={cage_cpu} "
        f"worker_cpus={min(worker_cpus)}-{max(worker_cpus)} hogs={sorted(hogs)} "
        f"children={len(baseline_children)} initial_ticks={initial[3] + initial[4]}",
    )

    last_heartbeat = 0.0
    try:
        while True:
            stat = proc_stat(MANAGER_PID)
            if stat is None:
                log(f"manager={MANAGER_PID} exited; API rescue complete")
                return 0
            if stat[2] != MANAGER_STARTTIME:
                raise RuntimeError(f"manager PID was reused: {stat}")

            reap_hogs(hogs)
            while len(hogs) < HOG_COUNT:
                if not hogs:
                    stop_manager("all scheduler starvation guards exited")
                    raise RuntimeError("all scheduler starvation guards exited")
                replacement = spawn_hog(cage_cpu)
                hogs.add(replacement)
                log(f"replaced scheduler starvation guard pid={replacement}")

            observed_children = direct_children()
            unexpected = observed_children - baseline_children
            if unexpected:
                stop_manager(f"unexpected direct children: {sorted(unexpected)}")
                raise RuntimeError(f"manager scheduled unexpected children: {sorted(unexpected)}")
            if os.sched_getscheduler(MANAGER_PID) != os.SCHED_IDLE:
                stop_manager("scheduler thread left SCHED_IDLE")
                raise RuntimeError("manager scheduler thread left SCHED_IDLE")
            if os.sched_getaffinity(MANAGER_PID) != {cage_cpu}:
                stop_manager("scheduler thread affinity changed")
                raise RuntimeError("manager scheduler thread affinity changed")

            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                log(
                    f"HEARTBEAT state={stat[0]} scheduler_ticks={stat[3] + stat[4]} "
                    f"children={len(observed_children)} hogs={sorted(hogs)}",
                )
                last_heartbeat = now
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        stop_manager(f"rescue exception: {type(error).__name__}: {error}")
        raise
    finally:
        terminate_hogs(hogs)


if __name__ == "__main__":
    raise SystemExit(main())
