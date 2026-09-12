#!/usr/bin/env python3
"""Finish the strict main, ordinary-LoRA, and paired-rsLoRA experiment protocol.

The supervisor is intentionally downstream-only.  It never repairs or starts
the main matrix; it waits until the read-only retry audit proves 39/39 strict
tasks and no associated Python process remains.  Every later phase is
idempotent except held-out evaluation, whose runner enforces one attempt per
checkpoint and is never called with ``--allow-retest``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .prepare_matrix_retry import build_plan, resolve_run_root
    from .run_matrix import FT_ROOT, ROOT
    from .run_rslora_paired import audit_paired_matrix
else:
    from prepare_matrix_retry import build_plan, resolve_run_root
    from run_matrix import FT_ROOT, ROOT
    from run_rslora_paired import audit_paired_matrix

SCRIPT_ROOT = Path(__file__).resolve().parent
ORCHESTRATION_ROOT = FT_ROOT / "logs" / "orchestration"
REPORT_ROOT = FT_ROOT / "logs" / "paper-report"
EXPECTED_MAIN_TASKS = 39
EXPECTED_GPUS = tuple(str(index) for index in range(8))
TEMPORARY_FAILURE_EXIT_CODE = 75


class SupervisorError(RuntimeError):
    """Raised when a terminal protocol phase cannot be completed safely."""


class SupervisorLockError(SupervisorError):
    """Raised when another terminal supervisor already owns the singleton lock."""


def utc_now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-matrix", required=True, help="Strict 39-task paper-policy matrix")
    parser.add_argument("--base-run", required=True, help="Completed 13-task Base-7B run")
    parser.add_argument("--paired-matrix", default=None, help="Defaults to <main>-rslora-paired")
    parser.add_argument("--gpus", default=",".join(EXPECTED_GPUS))
    parser.add_argument("--max-parallel", type=int, default=len(EXPECTED_GPUS))
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--paired-attempts", type=int, default=3)
    parser.add_argument("--validation-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_gpus(value: str, max_parallel: int) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if gpus != EXPECTED_GPUS:
        message = (
            "The terminal reproduction is configured for all eight H20s in order: "
            f"{','.join(EXPECTED_GPUS)}"
        )
        raise SupervisorError(message)
    if max_parallel != len(EXPECTED_GPUS):
        message = f"--max-parallel must be {len(EXPECTED_GPUS)} for the terminal audit"
        raise SupervisorError(message)
    return gpus


@contextmanager
def exclusive_supervisor(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"Another post-main supervisor owns {lock_path}"
            raise SupervisorLockError(message) from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.main_root = resolve_run_root(args.main_matrix)
        self.main_name = self.main_root.name
        self.paired_name = args.paired_matrix or f"{self.main_name}-rslora-paired"
        self.paired_root = resolve_run_root(self.paired_name)
        self.report_root = (args.report_root or REPORT_ROOT / f"{self.main_name}-complete").resolve()
        self.state_path = (
            args.state_file or ORCHESTRATION_ROOT / f"{self.main_name}-post-main-supervisor-state.json"
        ).resolve()
        self.lock_path = (
            args.lock_file or ORCHESTRATION_ROOT / f"{self.main_name}-post-main-supervisor.lock"
        ).resolve()
        self.commit_log = ORCHESTRATION_ROOT / f"{self.main_name}-held-out-commit.log"
        self.gpus = parse_gpus(args.gpus, args.max_parallel)
        # Keep the virtual-environment entrypoint intact.  Resolving the
        # ``.venv/bin/python`` symlink selects the base interpreter directly,
        # which drops the venv site-packages for every child command.
        self.python = Path(sys.executable)
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "kind": "ft-dojo-main-and-rslora-terminal-supervisor",
            "stage": "starting",
            "return_code": None,
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "main_matrix_root": str(self.main_root),
            "paired_matrix_root": str(self.paired_root),
            "base_run": args.base_run,
            "gpus": list(self.gpus),
            "max_parallel": args.max_parallel,
            "report_root": str(self.report_root),
            "completed_commands": [],
        }

    def save_state(self, stage: str, **updates: Any) -> None:
        self.state.update(updates)
        self.state["stage"] = stage
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def command(self, script: str, *arguments: str) -> list[str]:
        return [str(self.python), str(SCRIPT_ROOT / script), *arguments]

    def run_command(
        self,
        stage: str,
        command: list[str],
        log_path: Path,
        *,
        update_state: bool = True,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = shlex.join(command)
        if update_state:
            self.save_state(stage, current_command=rendered, current_log=str(log_path))
        print(f"{utc_now()} START stage={stage} command={rendered}", flush=True)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{utc_now()} START stage={stage} command={rendered}\n")
            log.flush()
            process = subprocess.Popen(  # noqa: S603 - fixed local interpreter and repository scripts.
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE.
                message = f"No output stream for stage {stage}"
                raise SupervisorError(message)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
            log.write(f"{utc_now()} END stage={stage} return_code={return_code}\n")
            log.flush()
        print(f"{utc_now()} END stage={stage} return_code={return_code}", flush=True)
        if return_code == 0 and update_state:
            completed = [*self.state.get("completed_commands", []), {"stage": stage, "log": str(log_path)}]
            self.save_state(stage, completed_commands=completed, last_return_code=0)
        return return_code

    def checked_command(self, stage: str, command: list[str], log_path: Path) -> None:
        return_code = self.run_command(stage, command, log_path)
        if return_code != 0:
            message = f"Stage {stage} failed with return code {return_code}; see {log_path}"
            raise SupervisorError(message)

    def retry_command(
        self,
        stage: str,
        command: list[str],
        log_path: Path,
        attempts: int,
        *,
        retry_gpu_conflicts_forever: bool = False,
    ) -> None:
        attempt = 0
        while True:
            attempt += 1
            return_code = self.run_command(f"{stage}_attempt_{attempt}", command, log_path)
            if return_code == 0:
                return
            if retry_gpu_conflicts_forever and return_code == TEMPORARY_FAILURE_EXIT_CODE:
                self.save_state(
                    "waiting_for_exclusive_gpus",
                    failed_stage=stage,
                    attempt=attempt,
                    last_return_code=return_code,
                )
                time.sleep(self.args.retry_delay)
                continue
            if attempt >= attempts:
                message = (
                    f"Stage {stage} failed after {attempt} attempts with return code {return_code}; see {log_path}"
                )
                raise SupervisorError(message)
            self.save_state(
                "retry_wait",
                failed_stage=stage,
                attempt=attempt,
                last_return_code=return_code,
            )
            time.sleep(self.args.retry_delay)

    def wait_for_main(self) -> dict[str, Any]:
        while True:
            plan = build_plan(self.main_root, [])
            counts = plan["action_counts"]
            active = plan["active_python_processes"]
            summary = {
                "strict_task_count": plan["strict_task_count"],
                "task_count": plan["task_count"],
                "action_counts": counts,
                "active_python_processes": [item["pid"] for item in active],
            }
            self.save_state("waiting_for_main", main_audit=summary)
            print(
                f"{utc_now()} MAIN strict={plan['strict_task_count']}/{plan['task_count']} "
                f"keep={counts['keep']} repair={counts['repair_status']} archive={counts['archive']} "
                f"create={counts['create_on_resume']} active={len(active)}",
                flush=True,
            )
            if (
                plan["task_count"] == EXPECTED_MAIN_TASKS
                and plan["strict_task_count"] == EXPECTED_MAIN_TASKS
                and counts == {"keep": EXPECTED_MAIN_TASKS, "repair_status": 0, "archive": 0, "create_on_resume": 0}
                and not active
            ):
                return summary
            time.sleep(self.args.poll_interval)

    def common_gpu_arguments(self) -> list[str]:
        return ["--gpus", ",".join(self.gpus), "--max-parallel", str(self.args.max_parallel)]

    def validation_command(self, matrix: str, profile: str, *, preflight: bool, no_baseline: bool) -> list[str]:
        arguments = [
            "--matrix-run",
            matrix,
            "--selection-profile",
            profile,
            *self.common_gpu_arguments(),
            "--resume",
            "--retry-failed-snapshots",
        ]
        if no_baseline:
            arguments.append("--no-baseline")
        if preflight:
            arguments.append("--preflight-only")
        return self.command("run_validation_sweep.py", *arguments)

    def final_command(self, matrix: str, profile: str, *, preflight: bool) -> list[str]:
        arguments = [
            "--matrix-run",
            matrix,
            "--selection-profile",
            profile,
            *self.common_gpu_arguments(),
            "--resume",
        ]
        if preflight:
            arguments.append("--preflight-only")
        return self.command("run_final_test.py", *arguments)

    def finalize_audit(self) -> None:
        main_results = self.report_root / "main-results" / "results.json"
        audit_output = self.report_root / "audit"
        command = self.command(
            "render_experiment_audit.py",
            "--matrix-run",
            str(self.main_root),
            "--results",
            str(main_results),
            "--supervisor-state",
            str(self.state_path),
            "--supervisor-log",
            str(self.commit_log),
            "--final-test-log",
            str(ORCHESTRATION_ROOT / f"{self.main_name}-final-test-main.log"),
            "--output-dir",
            str(audit_output),
        )
        return_code = self.run_command(
            "render_terminal_audit",
            command,
            ORCHESTRATION_ROOT / f"{self.main_name}-render-terminal-audit.log",
            update_state=False,
        )
        if return_code != 0:
            message = "Terminal main-experiment audit failed"
            raise SupervisorError(message)

    def run(self) -> None:
        if self.state_path.is_file():
            existing = json.loads(self.state_path.read_text(encoding="utf-8"))
            if existing.get("stage") == "complete" and existing.get("return_code") == 0:
                self.state = existing
                audit_manifest = self.report_root / "audit" / "evidence_manifest.json"
                if not audit_manifest.is_file():
                    self.finalize_audit()
                print(f"{utc_now()} supervisor already complete: {self.state_path}", flush=True)
                return

        self.save_state("starting")
        main_audit = self.wait_for_main()

        paired_log = ORCHESTRATION_ROOT / f"{self.main_name}-rslora-paired-training.log"
        pair_base = [
            "--source-matrix",
            str(self.main_root),
            "--run-name",
            self.paired_name,
            *self.common_gpu_arguments(),
            "--require-exclusive-gpus",
        ]
        self.checked_command(
            "paired_preflight",
            self.command("run_rslora_paired.py", *pair_base, "--preflight-only"),
            paired_log,
        )
        self.retry_command(
            "paired_training",
            self.command("run_rslora_paired.py", *pair_base, "--resume"),
            paired_log,
            self.args.paired_attempts,
            retry_gpu_conflicts_forever=True,
        )

        paired_audit = audit_paired_matrix(self.paired_root, source_root=self.main_root)
        paired_audit_path = self.report_root / "paired-results" / "paired_training_audit.json"
        atomic_write_json(paired_audit_path, paired_audit)
        self.save_state("paired_training_audited", paired_training_audit=paired_audit)

        validation_specs = (
            ("validation_main", str(self.main_root), "main", False),
            ("validation_lora_comparison", str(self.main_root), "lora_comparison", False),
            ("validation_rslora", str(self.paired_root), "main", True),
        )
        for stage, matrix, profile, no_baseline in validation_specs:
            log = ORCHESTRATION_ROOT / f"{self.main_name}-{stage.replace('_', '-')}.log"
            self.checked_command(
                f"{stage}_preflight",
                self.validation_command(matrix, profile, preflight=True, no_baseline=no_baseline),
                log,
            )
            self.retry_command(
                stage,
                self.validation_command(matrix, profile, preflight=False, no_baseline=no_baseline),
                log,
                self.args.validation_attempts,
            )

        final_specs = (
            ("final_test_main", str(self.main_root), "main"),
            ("final_test_lora_comparison", str(self.main_root), "lora_comparison"),
            ("final_test_rslora", str(self.paired_root), "main"),
        )
        for stage, matrix, profile in final_specs:
            log = ORCHESTRATION_ROOT / f"{self.main_name}-{stage.replace('_', '-')}-preflight.log"
            self.checked_command(
                f"{stage}_preflight",
                self.final_command(matrix, profile, preflight=True),
                log,
            )

        atomic_write_text(
            self.commit_log,
            f"{utc_now()} COMMIT one-shot held-out audit --matrix-run {self.main_root} "
            f"--gpus {','.join(self.gpus)} --max-parallel {self.args.max_parallel}\n",
        )
        for stage, matrix, profile in final_specs:
            log = ORCHESTRATION_ROOT / f"{self.main_name}-{stage.replace('_', '-')}.log"
            self.checked_command(stage, self.final_command(matrix, profile, preflight=False), log)

        paired_audit = audit_paired_matrix(self.paired_root, source_root=self.main_root)
        atomic_write_json(paired_audit_path, paired_audit)

        main_results_dir = self.report_root / "main-results"
        paired_results_dir = self.report_root / "paired-results"
        self.checked_command(
            "collect_main_results",
            self.command(
                "collect_results.py",
                "--matrix-run",
                str(self.main_root),
                "--base-run",
                self.args.base_run,
                "--output",
                str(main_results_dir),
            ),
            ORCHESTRATION_ROOT / f"{self.main_name}-collect-main.log",
        )
        self.checked_command(
            "render_main_table",
            self.command(
                "render_main_report.py",
                "--results",
                str(main_results_dir / "results.json"),
                "--output",
                str(self.report_root / "MAIN_TABLE.md"),
            ),
            ORCHESTRATION_ROOT / f"{self.main_name}-render-main.log",
        )
        self.checked_command(
            "collect_paired_results",
            self.command(
                "collect_results.py",
                "--matrix-run",
                str(self.main_root),
                "--matrix-run",
                str(self.paired_root),
                "--output",
                str(paired_results_dir),
            ),
            ORCHESTRATION_ROOT / f"{self.main_name}-collect-paired.log",
        )
        self.checked_command(
            "render_paired_table",
            self.command(
                "render_lora_rslora_report.py",
                "--results",
                str(paired_results_dir / "results.json"),
                "--paired-audit",
                str(paired_audit_path),
                "--output",
                str(self.report_root / "LORA_RSLORA_COMPARISON.md"),
            ),
            ORCHESTRATION_ROOT / f"{self.main_name}-render-paired.log",
        )

        terminal = {
            "return_code": 0,
            "completed_at": utc_now(),
            "current_command": None,
            "current_log": None,
            "main_audit": main_audit,
            "paired_training_audit": paired_audit,
            "outputs": {
                "main_table": str(self.report_root / "MAIN_TABLE.md"),
                "paired_table": str(self.report_root / "LORA_RSLORA_COMPARISON.md"),
                "main_results": str(main_results_dir / "results.json"),
                "paired_results": str(paired_results_dir / "results.json"),
                "paired_training_audit": str(paired_audit_path),
                "experiment_audit": str(self.report_root / "audit" / "EXPERIMENT_AUDIT.md"),
            },
        }
        self.save_state("complete", **terminal)
        self.finalize_audit()


def run_supervisor(supervisor: Supervisor) -> int:
    """Run one singleton supervisor without corrupting an existing owner's state."""
    try:
        with exclusive_supervisor(supervisor.lock_path):
            try:
                supervisor.run()
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                supervisor.save_state(
                    "failed",
                    return_code=1,
                    failed_at=utc_now(),
                    error=f"{type(error).__name__}: {error}",
                )
                print(f"SUPERVISOR FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                return 1
    except SupervisorLockError as error:
        # The state file belongs to the process that currently holds the lock.
        # A duplicate launch must leave that shared progress record untouched.
        print(f"SUPERVISOR NOT STARTED: {error}", file=sys.stderr, flush=True)
        return TEMPORARY_FAILURE_EXIT_CODE
    return 0


def main() -> int:
    args = parse_args()
    if args.poll_interval <= 0 or args.retry_delay <= 0:
        message = "Poll and retry delays must be positive"
        raise SystemExit(message)
    if args.paired_attempts < 1 or args.validation_attempts < 1:
        message = "Attempt counts must be positive"
        raise SystemExit(message)
    load_dotenv(ROOT / ".env", override=False)
    return run_supervisor(Supervisor(args))


if __name__ == "__main__":
    raise SystemExit(main())
