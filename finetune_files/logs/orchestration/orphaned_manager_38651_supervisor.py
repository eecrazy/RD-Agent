#!/usr/bin/env python3
"""Replace manager 38651's adapter and supervise its orphaned task sessions."""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import os
import select
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/data/github/RD-Agent")
RUN_ROOT = ROOT / "finetune_files/logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
MANAGER_PID = 38651
MANAGER_STARTTIME = 592896965
OLD_SUPERVISOR_PID = 3460714
OLD_SUPERVISOR_STARTTIME = 601217265
ADAPTER_PORT = 40843
POLL_MILLISECONDS = 200
HEARTBEAT_SECONDS = 30.0
PIDFD_OPEN_SYSCALL = 434
SUCCESS_MARKERS = (
    b"Reach stop criterion and stop loop: Timer timeout",
    b"Pipeline claim",  # Used only together with a succeeded claim below.
)
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


def process_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return {
        key.decode(): value.decode()
        for item in raw
        if item
        for key, value in [item.split(b"=", 1)]
    }


def status_path(experiment_id: str) -> Path:
    return RUN_ROOT / experiment_id.replace("/", "__") / "status.json"


def read_status(experiment_id: str) -> dict[str, object]:
    path = status_path(experiment_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise RuntimeError(f"status identity mismatch: {path}")
    return payload


def terminal_evidence(experiment_id: str) -> tuple[int, str]:
    task_root = status_path(experiment_id).parent
    claim_path = task_root / ".pipeline-claim.json"
    if claim_path.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            if claim.get("state") == "succeeded":
                return 0, "pipeline_claim_succeeded"
            if claim.get("state") == "failed":
                return 1, f"pipeline_claim_failed: {claim.get('error', 'unknown error')}"

    console_path = task_root / "console.log"
    try:
        with console_path.open("rb") as console:
            console.seek(0, os.SEEK_END)
            size = console.tell()
            console.seek(max(0, size - 2 * 1024 * 1024))
            tail = console.read()
    except OSError as error:
        return 1, f"console_unreadable: {error}"
    if SUCCESS_MARKERS[0] in tail:
        return 0, "console_timer_completion_marker"
    return 1, "no_terminal_success_marker"


def write_completion(experiment_id: str, return_code: int, evidence: str, *, exact: bool) -> None:
    path = status_path(experiment_id)
    payload = read_status(experiment_id)
    payload.update(
        {
            "state": "succeeded" if return_code == 0 else "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "return_code": return_code,
            "outer_timeout": False,
            "completion_observation": {
                "supervisor": "orphaned_manager_38651_supervisor",
                "return_code_exact": exact,
                "evidence": evidence,
            },
        },
    )
    temporary = path.with_name(f".{path.name}.orphan-supervisor-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pidfd_open(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(PIDFD_OPEN_SYSCALL, pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), pid)
    return fd


def terminate_old_supervisor() -> None:
    stat = proc_stat(OLD_SUPERVISOR_PID)
    if stat is None:
        return
    if stat[2] != OLD_SUPERVISOR_STARTTIME:
        raise RuntimeError(f"old supervisor PID identity changed: {stat}")
    os.kill(OLD_SUPERVISOR_PID, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while proc_stat(OLD_SUPERVISOR_PID) is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    if proc_stat(OLD_SUPERVISOR_PID) is not None:
        os.kill(OLD_SUPERVISOR_PID, signal.SIGKILL)
    log(f"terminated old supervisor={OLD_SUPERVISOR_PID}")


def verify_and_open_tasks() -> tuple[dict[int, tuple[int, str]], dict[int, int]]:
    pending: dict[int, tuple[int, str]] = {}
    pidfds: dict[int, int] = {}
    for pid, (starttime, experiment_id) in TASKS.items():
        stat = proc_stat(pid)
        if stat is None or stat[1] != MANAGER_PID or stat[2] != starttime:
            raise RuntimeError(f"valid child identity mismatch for {pid}: {stat}")
        if stat[0] == "Z":
            return_code = os.waitstatus_to_exitcode(stat[3])
            write_completion(experiment_id, return_code, "pre_orphan_wait_status", exact=True)
            log(f"finalized pre-existing zombie pid={pid} experiment={experiment_id} return_code={return_code}")
            continue
        pending[pid] = (starttime, experiment_id)
        pidfds[pid] = pidfd_open(pid)
    return pending, pidfds


def collect_report() -> None:
    sys.path.insert(0, str(ROOT / "reproduction/ft_agent"))
    from collect_results import collect_and_write_run

    report = collect_and_write_run(RUN_ROOT, "ft-agent")
    coverage = report["coverage"]
    log(f"report refreshed: supplied_tasks={coverage['supplied_tasks']}")


def main() -> int:
    lock_path = ROOT / "finetune_files/logs/orchestration/orphaned_manager_38651_supervisor.lock"
    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another orphan supervisor is already active") from None
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    manager = proc_stat(MANAGER_PID)
    if manager is None or manager[2] != MANAGER_STARTTIME or manager[0] not in {"T", "t"}:
        raise RuntimeError(f"manager is not the expected stopped process: {manager}")
    environment = process_environment(MANAGER_PID)

    sys.path.insert(0, str(ROOT / "reproduction/ft_agent"))
    from responses_adapter import ResponsesAdapterConfig, ResponsesChatAdapter

    config = ResponsesAdapterConfig.from_environment(environment)
    if config is None:
        raise RuntimeError("manager did not use a Responses adapter")
    if config.served_model != "gpt-5.6-sol":
        raise RuntimeError(f"unexpected served model: {config.served_model}")

    pending, pidfds = verify_and_open_tasks()
    terminate_old_supervisor()
    os.kill(MANAGER_PID, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while proc_stat(MANAGER_PID) is not None and time.monotonic() < deadline:
        time.sleep(0.005)
    if proc_stat(MANAGER_PID) is not None:
        raise RuntimeError("manager did not exit after SIGKILL")

    adapter = None
    bind_error: OSError | None = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            adapter = ResponsesChatAdapter(config, host="127.0.0.1", port=ADAPTER_PORT)
            break
        except OSError as error:
            bind_error = error
            time.sleep(0.01)
    if adapter is None:
        raise RuntimeError(f"could not take over adapter port {ADAPTER_PORT}: {bind_error}")
    adapter.start()
    log(
        f"ADAPTER_TAKEOVER port={ADAPTER_PORT} model={config.served_model} "
        f"orphaned_tasks={len(pending)}",
    )

    poller = select.poll()
    fd_to_pid: dict[int, int] = {}
    for pid, fd in pidfds.items():
        poller.register(fd, select.POLLIN)
        fd_to_pid[fd] = pid

    last_heartbeat = 0.0
    try:
        while pending:
            for fd, _event in poller.poll(POLL_MILLISECONDS):
                pid = fd_to_pid.pop(fd, None)
                if pid is None or pid not in pending:
                    continue
                poller.unregister(fd)
                os.close(fd)
                pidfds.pop(pid, None)
                _starttime, experiment_id = pending.pop(pid)
                return_code, evidence = terminal_evidence(experiment_id)
                write_completion(experiment_id, return_code, evidence, exact=False)
                log(
                    f"finalized orphan pid={pid} experiment={experiment_id} "
                    f"inferred_return_code={return_code} evidence={evidence} remaining={len(pending)}",
                )
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                live = sum(proc_stat(pid) is not None for pid in pending)
                log(f"HEARTBEAT adapter_port={ADAPTER_PORT} pending={len(pending)} live={live}")
                last_heartbeat = now
        collect_report()
        return 0
    finally:
        adapter.close()
        for fd in pidfds.values():
            with contextlib.suppress(OSError):
                os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
