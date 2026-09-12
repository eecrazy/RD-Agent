#!/usr/bin/env python3
# ruff: noqa: C901, E402, EM101, EM102, PLR0911, PLR0912, PLR0915, TRY003, TRY300
"""Safely transition the H20 FT-Dojo run from search to formal validation."""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("/data/github/RD-Agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.ft_agent.run_validation_sweep import (
    load_targets as load_validation_targets,
)
from reproduction.ft_agent.run_validation_sweep import (
    validate_current_selection_provenance,
)

RUN_ROOT = (
    ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-consolidated"
)
ORCHESTRATION_ROOT = ROOT / "finetune_files/logs/orchestration"
LOCK_PATH = ORCHESTRATION_ROOT / "post_search_validation_supervisor.lock"
STATE_PATH = ORCHESTRATION_ROOT / "post_search_validation_supervisor_state.json"
GATE_PID_PATH = ORCHESTRATION_ROOT / "gpu_priority_gate_v4.pid"
GPU_LOCK_ROOT = ROOT / "finetune_files/gpu_leases/locks"
PYTHON = ROOT / ".venv/bin/python"
VALIDATION_RUNNER = ROOT / "reproduction/ft_agent/run_validation_sweep.py"
POLL_SECONDS = 30
QUIESCENT_POLLS_REQUIRED = 2
FORMAL_VALIDATION_ATTEMPTS = 3
EXPECTED_TASKS = 39


def timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def log(message: str) -> None:
    print(f"{timestamp()} POST_SEARCH_VALIDATION {message}", flush=True)


def write_state(stage: str, **fields: Any) -> None:
    payload = {
        "schema_version": 1,
        "stage": stage,
        "updated_at": timestamp(),
        "pid": os.getpid(),
        "run_root": str(RUN_ROOT),
        **fields,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def safe_id(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", experiment_id).strip("_")


def manifest_tasks() -> list[dict[str, Any]]:
    payload = json.loads((RUN_ROOT / "matrix.json").read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TASKS} manifest tasks, "
            f"found {len(tasks) if isinstance(tasks, list) else 'invalid'}",
        )
    ids = [str(task["experiment_id"]) for task in tasks]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Consolidated manifest contains duplicate experiment ids")
    return tasks


def search_states(tasks: list[dict[str, Any]]) -> tuple[Counter[str], list[str]]:
    states: Counter[str] = Counter()
    errors: list[str] = []
    for task in tasks:
        experiment_id = str(task["experiment_id"])
        status_path = RUN_ROOT / safe_id(experiment_id) / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{experiment_id}: {type(error).__name__}: {error}")
            continue
        if status.get("experiment_id") != experiment_id:
            errors.append(f"{experiment_id}: status experiment_id mismatch")
            continue
        states[str(status.get("state", "unknown"))] += 1
    return states, errors


def process_cmdlines() -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        result.append((int(entry.name), command))
    return result


def matching_processes(*needles: str) -> list[tuple[int, str]]:
    return [
        (pid, command)
        for pid, command in process_cmdlines()
        if pid != os.getpid() and all(needle in command for needle in needles)
    ]


def busy_gpu_locks() -> list[int]:
    busy: list[int] = []
    for gpu in range(8):
        path = GPU_LOCK_ROOT / f"gpu-{gpu}.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy.append(gpu)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return busy


def stop_priority_gate() -> None:
    if not GATE_PID_PATH.is_file():
        log("priority gate pid file already absent")
        return
    try:
        pid = int(GATE_PID_PATH.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot validate priority gate pid: {error}") from error
    if "gpu_priority_gate_v4.zsh" not in command:
        raise RuntimeError(f"Refusing to signal pid {pid}: it is not gpu_priority_gate_v4")
    os.kill(pid, signal.SIGCONT)
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not Path(f"/proc/{pid}").exists():
            log(f"priority gate stopped pid={pid}")
            return
        time.sleep(0.5)
    raise RuntimeError(f"Priority gate pid {pid} did not stop after SIGTERM")


def validation_command(*extra: str) -> list[str]:
    return [
        str(PYTHON),
        str(VALIDATION_RUNNER),
        "--matrix-run",
        str(RUN_ROOT),
        "--gpus",
        "0,1,2,3,4,5,6,7",
        "--max-parallel",
        "8",
        "--resume",
        # Base-7B is evaluated once by run_base_eval.py.  Checkpoint ranking is
        # restricted to compliant RS-LoRA artifacts so final-test selection
        # cannot accidentally choose the unchanged full model.
        "--no-baseline",
        *extra,
    ]


def run_command(command: list[str]) -> int:
    log("START " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    log(f"END exit={result.returncode} command={Path(command[1]).name}")
    return result.returncode


def audit_selections(tasks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    signatures: set[str] = set()
    try:
        targets = {target.experiment_id: target for target in load_validation_targets(RUN_ROOT.resolve())}
    except Exception as error:  # noqa: BLE001
        return [f"Unable to load formal validation targets: {type(error).__name__}: {error}"]
    for task in tasks:
        experiment_id = str(task["experiment_id"])
        path = RUN_ROOT / safe_id(experiment_id) / "validation_selection.json"
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{experiment_id}: {type(error).__name__}: {error}")
            continue
        expected = {
            "state": "succeeded",
            "experiment_id": experiment_id,
            "held_out_test_used": False,
        }
        mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
        if artifact.get("search", {}).get("state") != "succeeded":
            mismatches.append("search.state")
        signature = artifact.get("artifact_signature")
        if not isinstance(signature, str) or not signature:
            mismatches.append("artifact_signature")
        else:
            signatures.add(signature)
        target = targets.get(experiment_id)
        if target is None:
            mismatches.append("formal_target")
        else:
            try:
                validate_current_selection_provenance(target, artifact)
            except Exception as error:  # noqa: BLE001
                mismatches.append(f"formal_provenance({type(error).__name__}: {error})")
        if mismatches:
            errors.append(f"{experiment_id}: incompatible fields {', '.join(mismatches)}")
    if len(signatures) != EXPECTED_TASKS:
        errors.append(f"Expected {EXPECTED_TASKS} unique selection signatures, found {len(signatures)}")
    return errors


def main() -> int:
    ORCHESTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another supervisor owns the singleton lock")
        os.close(descriptor)
        return 0

    try:
        tasks = manifest_tasks()
        last_summary: str | None = None
        write_state("waiting_for_search")
        while True:
            states, errors = search_states(tasks)
            loops = matching_processes("/rdagent/app/finetune/llm/loop.py")
            summary = f"states={dict(sorted(states.items()))} errors={len(errors)} loops={len(loops)}"
            if summary != last_summary:
                log(summary)
                last_summary = summary
            write_state(
                "waiting_for_search",
                search_states=dict(sorted(states.items())),
                status_errors=errors,
                loop_processes=len(loops),
            )
            unexpected = sum(count for state, count in states.items() if state not in {"running", "succeeded"})
            if errors or unexpected:
                write_state(
                    "search_needs_intervention",
                    search_states=dict(sorted(states.items())),
                    status_errors=errors,
                    loop_processes=len(loops),
                )
                log("effective search source failed or became unreadable; refusing automatic transition")
                return 2
            if states == Counter({"succeeded": EXPECTED_TASKS}) and not loops:
                break
            time.sleep(POLL_SECONDS)

        write_state("draining_snapshot_validation")
        quiet_polls = 0
        last_summary = None
        while quiet_polls < QUIESCENT_POLLS_REQUIRED:
            snapshot_processes = matching_processes("run_validation_sweep.py", "--snapshot-only")
            busy = busy_gpu_locks()
            summary = f"snapshot_processes={len(snapshot_processes)} busy_gpus={busy} quiet={quiet_polls}"
            if summary != last_summary:
                log(summary)
                last_summary = summary
            if not snapshot_processes and not busy:
                quiet_polls += 1
            else:
                quiet_polls = 0
            write_state(
                "draining_snapshot_validation",
                snapshot_processes=len(snapshot_processes),
                busy_gpus=busy,
                quiet_polls=quiet_polls,
            )
            if quiet_polls < QUIESCENT_POLLS_REQUIRED:
                time.sleep(POLL_SECONDS)

        stop_priority_gate()
        write_state("formal_validation_preflight")
        if run_command(validation_command("--preflight-only")) != 0:
            write_state("formal_validation_preflight_failed")
            return 3

        for attempt in range(1, FORMAL_VALIDATION_ATTEMPTS + 1):
            write_state("formal_validation_running", attempt=attempt)
            if run_command(validation_command()) == 0:
                break
            if attempt == FORMAL_VALIDATION_ATTEMPTS:
                write_state("formal_validation_needs_intervention", attempts=attempt)
                return 4
            log(f"formal validation attempt {attempt} was incomplete; resuming after 30 seconds")
            time.sleep(30)

        errors = audit_selections(tasks)
        if errors:
            write_state("selection_audit_failed", errors=errors)
            for error in errors:
                log("AUDIT_ERROR " + error)
            return 5
        write_state("formal_validation_complete", selection_count=EXPECTED_TASKS)
        log(f"formal validation complete; {EXPECTED_TASKS} signed selections await held-out final-test audit")
        return 0
    except Exception as error:  # noqa: BLE001
        write_state("supervisor_failed", error=f"{type(error).__name__}: {error}")
        log(f"FATAL {type(error).__name__}: {error}")
        return 1
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
