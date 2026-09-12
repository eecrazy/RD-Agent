#!/usr/bin/env python3
# ruff: noqa: C901, E402, EM101, EM102, PLR0911, PLR0912, PLR0915, TRY003, TRY004, TRY300
"""Run the one-shot held-out audit after formal validation, then collect results."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
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

ORCHESTRATION_ROOT = ROOT / "finetune_files/logs/orchestration"
VALIDATION_STATE_PATH = ORCHESTRATION_ROOT / "post_search_validation_supervisor_state.json"
LOCK_PATH = ORCHESTRATION_ROOT / "post_validation_finalization_supervisor.lock"
STATE_PATH = ORCHESTRATION_ROOT / "post_validation_finalization_supervisor_state.json"
RUN_ROOT = (
    ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-consolidated"
)
BASE_ROOT = ROOT / "finetune_files/logs/paper-base/h20-gpt56-base-v1"
REPORT_ROOT = (
    ROOT
    / "finetune_files/logs/paper-report"
    / "h20-gpt56-main-rslora-48h-v10-consolidated-final"
)
PYTHON = ROOT / ".venv/bin/python"
FINAL_TEST_RUNNER = ROOT / "reproduction/ft_agent/run_final_test.py"
COLLECTOR = ROOT / "reproduction/ft_agent/collect_results.py"
MAIN_REPORT_RENDERER = ROOT / "reproduction/ft_agent/render_main_report.py"
EXPECTED_FT_TASKS = 39
EXPECTED_BASE_TASKS = 13
POLL_SECONDS = 30

WAITING_VALIDATION_STAGES = {
    "waiting_for_search",
    "draining_snapshot_validation",
    "formal_validation_preflight",
    "formal_validation_running",
}
FAILED_VALIDATION_STAGES = {
    "search_needs_intervention",
    "formal_validation_preflight_failed",
    "formal_validation_needs_intervention",
    "selection_audit_failed",
    "supervisor_failed",
}
ONE_SHOT_COMMITTED_STAGES = {
    "final_test_launch_committed",
    "final_test_running",
    "final_test_needs_intervention",
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def log(message: str) -> None:
    print(f"{timestamp()} POST_VALIDATION_FINALIZATION {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_state(stage: str, **fields: Any) -> None:
    payload = {
        "schema_version": 1,
        "stage": stage,
        "updated_at": timestamp(),
        "pid": os.getpid(),
        "run_root": str(RUN_ROOT),
        "base_root": str(BASE_ROOT),
        "report_root": str(REPORT_ROOT),
        **fields,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def manifest_experiment_ids() -> list[str]:
    manifest = read_json(RUN_ROOT / "matrix.json")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_FT_TASKS:
        found = len(tasks) if isinstance(tasks, list) else "invalid"
        raise RuntimeError(f"Expected {EXPECTED_FT_TASKS} FT tasks, found {found}")
    result = [str(task["experiment_id"]) for task in tasks]
    if len(set(result)) != len(result):
        raise RuntimeError("FT manifest contains duplicate experiment ids")
    return result


def safe_id(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", experiment_id).strip("_")


def final_test_command(*extra: str) -> list[str]:
    command = [
        str(PYTHON),
        str(FINAL_TEST_RUNNER),
        "--matrix-run",
        str(RUN_ROOT),
        "--gpus",
        "0,1,2,3,4,5,6,7",
        "--max-parallel",
        "8",
        "--resume",
        *extra,
    ]
    if "--allow-retest" in command:
        raise AssertionError("The finalization supervisor must never allow a held-out retest")
    return command


def collector_command() -> list[str]:
    return [
        str(PYTHON),
        str(COLLECTOR),
        "--matrix-run",
        str(RUN_ROOT),
        "--base-run",
        str(BASE_ROOT),
        "--output",
        str(REPORT_ROOT),
    ]


def renderer_command() -> list[str]:
    return [
        str(PYTHON),
        str(MAIN_REPORT_RENDERER),
        "--results",
        str(REPORT_ROOT / "results.json"),
        "--output",
        str(REPORT_ROOT / "MAIN_TABLE.md"),
    ]


def process_matches(pid: int, needle: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return needle in command


def audit_validation_provenance(experiment_ids: list[str]) -> list[str]:
    """Block the one-shot held-out launch if a selection lacks formal training lineage."""
    errors: list[str] = []
    signatures: set[str] = set()
    try:
        targets = {target.experiment_id: target for target in load_validation_targets(RUN_ROOT.resolve())}
    except Exception as error:  # noqa: BLE001
        return [f"Unable to load formal validation targets: {type(error).__name__}: {error}"]
    for experiment_id in experiment_ids:
        target = targets.get(experiment_id)
        if target is None:
            errors.append(f"{experiment_id}: formal validation target is missing")
            continue
        path = RUN_ROOT / safe_id(experiment_id) / "validation_selection.json"
        try:
            artifact = read_json(path)
            validate_current_selection_provenance(target, artifact)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{experiment_id}: {type(error).__name__}: {error}")
            continue
        signature = artifact.get("artifact_signature")
        if isinstance(signature, str) and signature:
            signatures.add(signature)
        else:
            errors.append(f"{experiment_id}: selection artifact signature is missing")
    if len(signatures) != EXPECTED_FT_TASKS:
        errors.append(f"Expected {EXPECTED_FT_TASKS} unique formal selection signatures, found {len(signatures)}")
    return errors


def audit_final_tests(experiment_ids: list[str]) -> list[str]:
    errors: list[str] = []
    signatures: set[str] = set()
    for experiment_id in experiment_ids:
        task_root = RUN_ROOT / safe_id(experiment_id)
        artifact_path = task_root / "final_test.json"
        selection_path = task_root / "validation_selection.json"
        try:
            artifact = read_json(artifact_path)
            selection = read_json(selection_path)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            errors.append(f"{experiment_id}: {type(error).__name__}: {error}")
            continue
        expected = {
            "experiment_id": experiment_id,
            "state": "succeeded",
            "evaluation_mode": "post_selection",
            "test_range": "[-min(100, len(index_list)//2):]",
        }
        mismatches = [key for key, value in expected.items() if artifact.get(key) != value]
        artifact_signature = artifact.get("selection", {}).get("signature")
        selected_signature = selection.get("selection", {}).get("selection_signature")
        if not isinstance(artifact_signature, str) or not artifact_signature:
            mismatches.append("selection.signature")
        elif artifact_signature != selected_signature:
            mismatches.append("selection.signature_changed")
        else:
            signatures.add(artifact_signature)
        if artifact.get("benchmark_dataset_path") != selection.get("benchmark_dataset_path"):
            mismatches.append("benchmark_dataset_path")
        if mismatches:
            errors.append(f"{experiment_id}: incompatible fields {', '.join(mismatches)}")
    if len(signatures) != EXPECTED_FT_TASKS:
        errors.append(f"Expected {EXPECTED_FT_TASKS} unique final-test signatures, found {len(signatures)}")
    return errors


def audit_report() -> list[str]:
    errors: list[str] = []
    coverage = read_json(REPORT_ROOT / "coverage.json")
    if coverage.get("supplied_tasks") != EXPECTED_FT_TASKS + EXPECTED_BASE_TASKS:
        errors.append(
            f"Expected {EXPECTED_FT_TASKS + EXPECTED_BASE_TASKS} supplied tasks, "
            f"found {coverage.get('supplied_tasks')!r}",
        )
    states = coverage.get("task_states", {})
    if states != {"succeeded": EXPECTED_FT_TASKS + EXPECTED_BASE_TASKS}:
        errors.append(f"Unexpected collected task states: {states!r}")
    diagnostics = coverage.get("diagnostics", {})
    forbidden = {
        name: count
        for name, count in diagnostics.items()
        if count and name.startswith(("missing_", "invalid_", "unusable_"))
    }
    if forbidden:
        errors.append(f"Incomplete collected artifacts: {forbidden!r}")
    return errors


def run_preflight() -> int:
    command = final_test_command("--preflight-only")
    log("START preflight " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    log(f"END preflight exit={result.returncode}")
    return result.returncode


def launch_final_test() -> tuple[int, int]:
    command = final_test_command()
    write_state("final_test_launch_committed", command=command)
    log("COMMIT one-shot held-out audit " + " ".join(command))
    log_path = ORCHESTRATION_ROOT / "post_validation_final_test.log"
    with log_path.open("ab", buffering=0) as output:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_state("final_test_running", command=command, child_pid=process.pid, log_path=str(log_path))
        return process.wait(), process.pid


def wait_for_adopted_final_test(state: dict[str, Any]) -> None:
    child_pid = state.get("child_pid")
    if not isinstance(child_pid, int):
        raise RuntimeError("A committed one-shot final test has no auditable child pid")
    while process_matches(child_pid, "reproduction/ft_agent/run_final_test.py"):
        write_state(
            "final_test_running",
            command=state.get("command"),
            child_pid=child_pid,
            log_path=state.get("log_path"),
            adopted=True,
        )
        time.sleep(POLL_SECONDS)


def collect_results() -> int:
    command = collector_command()
    write_state("collecting_results", command=command)
    log("START collector " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    log(f"END collector exit={result.returncode}")
    return result.returncode


def render_main_report() -> int:
    command = renderer_command()
    write_state("rendering_main_report", command=command)
    log("START renderer " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
    log(f"END renderer exit={result.returncode}")
    return result.returncode


def main() -> int:
    ORCHESTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another finalization supervisor owns the singleton lock")
        os.close(descriptor)
        return 0

    try:
        experiment_ids = manifest_experiment_ids()
        previous = read_json(STATE_PATH) if STATE_PATH.is_file() else None
        if previous and previous.get("stage") == "report_complete":
            log("final report is already complete")
            return 0
        if previous and previous.get("stage") in ONE_SHOT_COMMITTED_STAGES:
            if previous.get("stage") == "final_test_running":
                log("adopting the already-committed one-shot final-test process")
                wait_for_adopted_final_test(previous)
                errors = audit_final_tests(experiment_ids)
                if errors:
                    write_state("final_test_needs_intervention", errors=errors, adopted=True)
                    return 5
            else:
                write_state(
                    "final_test_needs_intervention",
                    error="A one-shot launch was already committed; refusing to launch it again",
                    previous_stage=previous.get("stage"),
                )
                return 6
        elif previous and previous.get("stage") in {
            "final_test_complete",
            "collecting_results",
            "rendering_main_report",
            "report_needs_intervention",
        }:
            errors = audit_final_tests(experiment_ids)
            if errors:
                write_state("final_test_needs_intervention", errors=errors)
                return 7
        else:
            write_state("waiting_for_formal_validation")
            last_stage: str | None = None
            while True:
                try:
                    validation_state = read_json(VALIDATION_STATE_PATH)
                except (OSError, json.JSONDecodeError, TypeError) as error:
                    write_state("waiting_for_formal_validation", transient_error=f"{type(error).__name__}: {error}")
                    time.sleep(POLL_SECONDS)
                    continue
                stage = str(validation_state.get("stage", "unknown"))
                if stage != last_stage:
                    log(f"validation_stage={stage}")
                    last_stage = stage
                write_state(
                    "waiting_for_formal_validation",
                    validation_stage=stage,
                    validation_updated_at=validation_state.get("updated_at"),
                )
                if stage == "formal_validation_complete":
                    if validation_state.get("selection_count") != EXPECTED_FT_TASKS:
                        write_state("validation_needs_intervention", error="Formal selection count is not 39")
                        return 2
                    break
                if stage in FAILED_VALIDATION_STAGES:
                    write_state("validation_needs_intervention", validation_state=validation_state)
                    return 3
                if stage not in WAITING_VALIDATION_STAGES:
                    write_state("validation_needs_intervention", error=f"Unexpected validation stage: {stage}")
                    return 4
                time.sleep(POLL_SECONDS)

            provenance_errors = audit_validation_provenance(experiment_ids)
            if provenance_errors:
                write_state("final_test_preflight_failed", errors=provenance_errors)
                for error in provenance_errors:
                    log("PROVENANCE_AUDIT_ERROR " + error)
                return 8

            write_state("final_test_preflight")
            if run_preflight() != 0:
                write_state("final_test_preflight_failed")
                return 8

            return_code, child_pid = launch_final_test()
            if return_code != 0:
                write_state(
                    "final_test_needs_intervention",
                    child_pid=child_pid,
                    return_code=return_code,
                    error="The sole held-out audit command did not complete successfully; no retry was launched",
                )
                return 9
            errors = audit_final_tests(experiment_ids)
            if errors:
                write_state("final_test_needs_intervention", child_pid=child_pid, errors=errors)
                return 10

        write_state("final_test_complete", final_test_count=EXPECTED_FT_TASKS)
        if collect_results() != 0:
            write_state("report_needs_intervention", error="Collector exited nonzero")
            return 11
        errors = audit_report()
        if errors:
            write_state("report_needs_intervention", errors=errors)
            return 12
        if render_main_report() != 0:
            write_state("report_needs_intervention", error="Strict main-table renderer exited nonzero")
            return 13
        write_state(
            "report_complete",
            final_test_count=EXPECTED_FT_TASKS,
            base_count=EXPECTED_BASE_TASKS,
            supplied_task_count=EXPECTED_FT_TASKS + EXPECTED_BASE_TASKS,
            main_table_path=str(REPORT_ROOT / "MAIN_TABLE.md"),
        )
        log(f"final report complete: {REPORT_ROOT}")
        return 0
    except Exception as error:  # noqa: BLE001
        write_state("supervisor_failed", error=f"{type(error).__name__}: {error}")
        log(f"FATAL {type(error).__name__}: {error}")
        return 1
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
