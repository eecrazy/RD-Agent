#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, TRY003
"""Resume one failed matrix task from an RD-Agent session snapshot."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .responses_adapter import responses_configuration_errors, routed_api_environment
    from .run_matrix import (
        FORMAL_EXPECTED_SAMPLES_ENV,
        FORMAL_METHOD_LOCK_ENV,
        FORMAL_METHOD_LOCK_FILE,
        ROOT,
        configure_project_environment,
        duration_seconds,
        outer_process_timeout_seconds,
        status_write,
        stop_process,
        validate_task_formal_training,
    )
    from .run_validation_sweep import trusted_gpu_owner
else:
    from responses_adapter import responses_configuration_errors, routed_api_environment
    from run_matrix import (
        FORMAL_EXPECTED_SAMPLES_ENV,
        FORMAL_METHOD_LOCK_ENV,
        FORMAL_METHOD_LOCK_FILE,
        ROOT,
        configure_project_environment,
        duration_seconds,
        outer_process_timeout_seconds,
        status_write,
        stop_process,
        validate_task_formal_training,
    )
    from run_validation_sweep import trusted_gpu_owner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, help="Existing matrix task directory")
    parser.add_argument("--snapshot", required=True, help="Session snapshot below task-root/trace")
    parser.add_argument(
        "--remaining-time",
        required=True,
        help="Remaining loop budget, using one unit such as 334m or 5h",
    )
    parser.add_argument(
        "--requested-gpu",
        default=None,
        help="Preferred physical GPU; the project dynamic lease pool may choose another free GPU",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def option_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Recorded command has no value for {option}") from error
    return value


def resume_command(status: dict[str, Any], snapshot: Path, remaining_time: str) -> list[str]:
    command = [str(item) for item in status.get("command", [])]
    if not command:
        raise RuntimeError("Task status has no recorded command")
    if "--path" in command:
        raise RuntimeError("Recorded task command is already a resume command")
    try:
        timeout_index = command.index("--timeout") + 1
        command[timeout_index] = remaining_time
    except (ValueError, IndexError) as error:
        raise RuntimeError("Recorded task command has no --timeout value") from error
    command.extend(("--path", str(snapshot)))
    return command


def stable_routing(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "adapter_base"}


def task_environment(
    status: dict[str, Any],
    command: list[str],
    requested_gpu: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": requested_gpu,
            "CHAT_MODEL": str(status["planner"]),
            "FT_BASE_MODEL": str(status["model"]),
            "FT_TARGET_BENCHMARK": str(status["benchmark"]),
            "FT_BENCHMARK_DESCRIPTION": option_value(command, "--benchmark-description"),
            "FT_UPPER_DATA_SIZE_LIMIT": str(status["data_limit"]),
            FORMAL_EXPECTED_SAMPLES_ENV: str(status["data_limit"]),
            FORMAL_METHOD_LOCK_ENV: str(status["formal_method_lock_path"]),
            "LOG_TRACE_PATH": str(status["trace_path"]),
            "WORKSPACE_PATH": str(status["workspace_path"]),
            "FT_EXPERIMENT_ID": str(status["experiment_id"]),
            "FT_TRAINING_POLICY": str(status.get("training_policy", "paper")),
            "FT_BENCHMARK_DATASET_PATH": str(status["benchmark_dataset_path"]),
            "FT_EVALUATE_HELD_OUT_DURING_SEARCH": "false",
            # Loop snapshots contain the timer that was active when they were
            # serialized.  Tell the resumed process to replace that stale
            # value with the explicitly budgeted remaining time.
            "FT_RESUME_TIMER_BUDGET": option_value(command, "--timeout"),
        },
    )
    return environment


def validate_inputs(
    task_root: Path,
    snapshot: Path,
    status: dict[str, Any],
    remaining_time: str,
) -> None:
    if status.get("state") != "failed":
        raise RuntimeError(f"Task state must be 'failed', found {status.get('state')!r}")
    data_limit = status.get("data_limit")
    if isinstance(data_limit, bool) or not isinstance(data_limit, int) or data_limit < 1:
        raise RuntimeError("Task status has no positive integer data_limit")
    if status.get("formal_expected_samples") != data_limit:
        raise RuntimeError(
            "Task formal sample contract does not match data_limit; refusing an unbound resume",
        )
    expected_lock_path = (task_root / FORMAL_METHOD_LOCK_FILE).resolve()
    recorded_lock_path = Path(str(status.get("formal_method_lock_path", ""))).resolve()
    if recorded_lock_path != expected_lock_path:
        raise RuntimeError("Task formal method lock path is missing or inconsistent")
    if not snapshot.is_file():
        raise RuntimeError(f"Session snapshot is missing: {snapshot}")
    trace_root = (task_root / "trace").resolve()
    if not snapshot.is_relative_to(trace_root):
        raise RuntimeError(f"Session snapshot must be below {trace_root}")
    seconds = duration_seconds(remaining_time)
    if not 0 < seconds <= 48 * 3600:
        raise RuntimeError("Remaining task budget must be greater than 0 and at most 48h")


async def run_child(
    command: list[str],
    environment: dict[str, str],
    task_root: Path,
    status_path: Path,
    status: dict[str, Any],
    timeout_seconds: int,
) -> int:
    attempts = list(status.get("resume_attempts", []))
    attempt: dict[str, Any] = {
        "attempt": len(attempts) + 1,
        "started_at": datetime.now().astimezone().isoformat(),
        "snapshot": command[-1],
        "remaining_time": option_value(command, "--timeout"),
        "requested_gpu": environment["CUDA_VISIBLE_DEVICES"],
        "command": command,
    }
    attempts.append(attempt)
    status.update(
        {
            "state": "running",
            "resume_attempts": attempts,
            "resume_snapshot": command[-1],
            "resume_requested_gpu": environment["CUDA_VISIBLE_DEVICES"],
        },
    )
    for key in ("finished_at", "return_code", "outer_timeout", "abort_reason"):
        status.pop(key, None)
    status_write(status_path, status)

    marker = (
        f"\n===== RESUME attempt={attempt['attempt']} at={attempt['started_at']} "
        f"snapshot={attempt['snapshot']} remaining={attempt['remaining_time']} =====\n"
    ).encode()
    with (task_root / "console.log").open("ab", buffering=0) as log:
        log.write(marker)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        attempt["pid"] = process.pid
        status["resume_pid"] = process.pid
        status_write(status_path, status)
        timed_out = False
        try:
            return_code = await asyncio.wait_for(
                process.wait(),
                timeout=outer_process_timeout_seconds(timeout_seconds, environment),
            )
        except TimeoutError:
            timed_out = True
            await stop_process(process)
            return_code = process.returncode
        except (asyncio.CancelledError, KeyboardInterrupt):
            await stop_process(process)
            attempt.update(
                {
                    "state": "aborted",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "return_code": process.returncode,
                    "outer_timeout": False,
                },
            )
            status.update(
                {
                    "state": "aborted",
                    "finished_at": attempt["finished_at"],
                    "return_code": process.returncode,
                    "outer_timeout": False,
                    "abort_reason": "resume_supervisor_cancelled",
                },
            )
            status_write(status_path, status)
            raise

    evidence_records, evidence_errors = validate_task_formal_training(
        task_root,
        expected_samples=int(status["data_limit"]),
        experiment_id=str(status["experiment_id"]),
        training_policy=str(status.get("training_policy", "paper")),
    )
    success = return_code == 0 and not timed_out and bool(evidence_records)
    locked_method = evidence_records[0]["training_method"] if evidence_records else None
    method_lock = (
        evidence_records[0].get("formal_training_method_lock") if evidence_records else None
    )
    finished_at = datetime.now().astimezone().isoformat()
    attempt.update(
        {
            "state": "succeeded" if success else "failed",
            "finished_at": finished_at,
            "return_code": return_code,
            "outer_timeout": timed_out,
            "formal_training_evidence": evidence_records,
            "formal_training_validation_errors": evidence_errors,
            "formal_training_method": locked_method,
            "formal_training_method_lock": method_lock,
        },
    )
    status.update(
        {
            "state": attempt["state"],
            "finished_at": finished_at,
            "return_code": return_code,
            "outer_timeout": timed_out,
            "formal_training_evidence": evidence_records,
            "formal_training_validation_errors": evidence_errors,
            "formal_training_method": locked_method,
            "formal_training_method_lock": method_lock,
        },
    )
    status.pop("resume_pid", None)
    status_write(status_path, status)
    return 0 if success else 1


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    configure_project_environment()

    task_root = Path(args.task_root).expanduser().resolve()
    status_path = task_root / "status.json"
    if not status_path.is_file():
        raise SystemExit(f"Task status is missing: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    configure_project_environment(str(status.get("training_policy", "paper")))
    snapshot = Path(args.snapshot).expanduser().resolve()
    validate_inputs(task_root, snapshot, status, args.remaining_time)
    command = resume_command(status, snapshot, args.remaining_time)
    requested_gpu = str(args.requested_gpu if args.requested_gpu is not None else status["gpu"])
    environment = task_environment(status, command, requested_gpu)

    errors = responses_configuration_errors(environment)
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))

    lock_path = task_root / "resume.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"Another resume supervisor already owns {lock_path}") from error

        with trusted_gpu_owner(), routed_api_environment(environment) as (runtime_environment, api_routing):
            recorded_routing = status.get("api_routing", {})
            if stable_routing(recorded_routing) != stable_routing(api_routing):
                raise SystemExit("Current API routing is incompatible with the recorded task routing")
            if args.dry_run:
                print(f"Task: {status['experiment_id']}")
                print(f"Snapshot: {snapshot}")
                print(f"Requested GPU: {requested_gpu} (dynamic lease pool enabled)")
                print(f"Remaining time: {args.remaining_time}")
                print("Command: " + " ".join(command))
                return 0

            status["resume_api_routing"] = api_routing
            return asyncio.run(
                run_child(
                    command,
                    runtime_environment,
                    task_root,
                    status_path,
                    status,
                    duration_seconds(args.remaining_time),
                ),
            )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
