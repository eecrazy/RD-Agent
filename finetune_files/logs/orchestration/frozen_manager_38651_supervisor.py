#!/usr/bin/env python3
"""Finalize valid children of the intentionally frozen v10 matrix manager."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/github/RD-Agent")
RUN_ROOT = ROOT / "finetune_files/logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
MANAGER_PID = 38651
MANAGER_STARTTIME = 592896965
POLL_SECONDS = 5
TASKS = {
    39434: (592897955, "main/aime25/run-1"),
    39438: (592897955, "main/panorama_noc4pc/run-1"),
    39440: (592897955, "main/panorama_pi4pc/run-1"),
    39442: (592897955, "main/chemcotbench_mol_und/run-1"),
    39446: (592897955, "main/chemcotbench_mol_opt/run-1"),
    39450: (592897955, "main/FinanceIQ_gen/run-1"),
    39452: (592897955, "main/tablebench_data_analysis/run-1"),
    39454: (592897955, "main/tablebench_fact_checking/run-1"),
    39456: (592897955, "main/tablebench_numerical_reasoning/run-1"),
    39458: (592897956, "main/tablebench_visualization/run-1"),
    39460: (592897956, "main/aime25/run-2"),
    39464: (592897956, "main/panorama_noc4pc/run-2"),
    56119: (592915811, "main/panorama_pi4pc/run-2"),
}


def log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat()} {message}", flush=True)


def proc_stat(pid: int) -> tuple[str, int, int, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    fields = raw.rsplit(")", 1)[1].split()
    return fields[0], int(fields[1]), int(fields[19]), int(fields[49])


def status_path(experiment_id: str) -> Path:
    return RUN_ROOT / experiment_id.replace("/", "__") / "status.json"


def read_status(experiment_id: str) -> dict[str, object]:
    path = status_path(experiment_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError(f"status identity mismatch: {path}")
    return payload


def write_completion(experiment_id: str, return_code: int) -> None:
    path = status_path(experiment_id)
    payload = read_status(experiment_id)
    payload.update(
        {
            "state": "succeeded" if return_code == 0 else "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "return_code": return_code,
            "outer_timeout": False,
        },
    )
    temporary = path.with_name(f".{path.name}.supervisor-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_initial_processes() -> None:
    manager = proc_stat(MANAGER_PID)
    if manager is None or manager[2] != MANAGER_STARTTIME or manager[0] not in {"T", "t"}:
        raise RuntimeError(f"manager {MANAGER_PID} is not the expected frozen process: {manager}")
    for pid, (starttime, experiment_id) in TASKS.items():
        stat = proc_stat(pid)
        if stat is None:
            payload = read_status(experiment_id)
            if payload.get("state") in {"succeeded", "failed"}:
                continue
            raise RuntimeError(f"valid child {pid} disappeared before supervision: {experiment_id}")
        if stat[1] != MANAGER_PID or stat[2] != starttime:
            raise RuntimeError(f"valid child identity mismatch for {pid}: {stat}")


def collect_report() -> None:
    sys.path.insert(0, str(ROOT / "reproduction/ft_agent"))
    from collect_results import collect_and_write_run

    report = collect_and_write_run(RUN_ROOT, "ft-agent")
    coverage = report["coverage"]
    log(f"report refreshed: supplied_tasks={coverage['supplied_tasks']}")


def main() -> int:
    lock_path = ROOT / "finetune_files/logs/orchestration/frozen_manager_38651_supervisor.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another frozen-manager supervisor is already active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    verify_initial_processes()
    pending = set(TASKS)
    for pid in tuple(pending):
        experiment_id = TASKS[pid][1]
        if proc_stat(pid) is None and read_status(experiment_id).get("state") in {"succeeded", "failed"}:
            pending.remove(pid)
    log(f"supervising manager={MANAGER_PID} valid_children={len(pending)}")

    while pending:
        for pid in tuple(pending):
            starttime, experiment_id = TASKS[pid]
            stat = proc_stat(pid)
            if stat is None:
                raise RuntimeError(f"child {pid} vanished without a wait status: {experiment_id}")
            state, ppid, observed_starttime, wait_status = stat
            if ppid != MANAGER_PID or observed_starttime != starttime:
                raise RuntimeError(f"child identity changed for {pid}: {stat}")
            if state != "Z":
                continue
            return_code = os.waitstatus_to_exitcode(wait_status)
            write_completion(experiment_id, return_code)
            pending.remove(pid)
            log(
                f"finalized pid={pid} experiment={experiment_id} "
                f"return_code={return_code} remaining={len(pending)}",
            )
        if pending:
            time.sleep(POLL_SECONDS)

    manager = proc_stat(MANAGER_PID)
    if manager is not None and manager[2] == MANAGER_STARTTIME:
        os.kill(MANAGER_PID, signal.SIGKILL)
        log(f"terminated frozen manager={MANAGER_PID}")
    else:
        log(f"manager={MANAGER_PID} already absent after all valid children completed")
    collect_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
