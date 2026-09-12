#!/usr/bin/env python3
"""Strictly audit a terminal matrix and archive task roots that must be retried.

The matrix runner deliberately refuses to append to an existing failed task
root. This helper performs the matching recovery operation: it moves every
non-strict task root into a timestamped archive and restores only a validated
paper-policy method lock. It can also quarantine explicitly named, failed
generated datasets and remove their stale global registrations.

The default mode is read-only. Pass ``--apply`` only after the matrix has no
running tasks or live Python processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

if __package__:
    from .run_matrix import FORMAL_METHOD_LOCK_FILE, FT_ROOT, safe_id, validate_task_formal_training
else:
    from run_matrix import FORMAL_METHOD_LOCK_FILE, FT_ROOT, safe_id, validate_task_formal_training

from rdagent.scenarios.finetune.train.formal_training import (
    FormalTrainingEvidenceError,
    formal_training_provenance_files,
    validate_formal_training_evidence,
    validate_formal_training_method_lock,
)

TERMINAL_STATES = {"succeeded", "failed", "aborted"}
FAILED_DATASET_STATES = {"aborted", "failed"}
DATASET_NAME = re.compile(r"[A-Za-z0-9_.-]+")


class RetryPreparationError(RuntimeError):
    """Raised when retry preparation cannot be performed safely."""


def _require(*, condition: bool, message: str) -> None:
    """Raise a retry-preparation error when a transaction invariant fails."""
    if not condition:
        raise RetryPreparationError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", required=True, help="Matrix run name or explicit run-root path")
    parser.add_argument(
        "--quarantine-dataset",
        action="append",
        default=[],
        help="Explicit failed generated-dataset name; repeat as needed",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the printed archive/quarantine plan; default is read-only",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Regex applied to experiment ids. In selective mode, active unselected tasks are left untouched; "
            "an active selected task still blocks the transaction."
        ),
    )
    return parser.parse_args()


def resolve_run_root(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = FT_ROOT / "logs" / "paper-matrix" / candidate
    return candidate.resolve()


def read_json(path: Path, expected_type: type = dict) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        message = f"Cannot read JSON {path}: {error}"
        raise RetryPreparationError(message) from error
    if not isinstance(value, expected_type):
        message = f"JSON has the wrong top-level type: {path}"
        raise RetryPreparationError(message)
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix = read_json(run_root / "matrix.json")
    tasks = matrix.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        message = "Matrix manifest has no task inventory"
        raise RetryPreparationError(message)
    if any(not isinstance(task, dict) for task in tasks):
        message = "Matrix task inventory contains a non-object"
        raise RetryPreparationError(message)
    identifiers = [task.get("experiment_id") for task in tasks]
    if any(not isinstance(item, str) or not item for item in identifiers):
        message = "Matrix task inventory contains an invalid experiment id"
        raise RetryPreparationError(message)
    if len(set(identifiers)) != len(identifiers):
        message = "Matrix task inventory contains duplicate experiment ids"
        raise RetryPreparationError(message)
    policy = matrix.get("training_policy")
    if not isinstance(policy, str) or not policy:
        message = "Matrix manifest has no training policy"
        raise RetryPreparationError(message)
    return matrix, tasks


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def preserved_method_lock(
    task_root: Path,
    *,
    experiment_id: str,
    training_policy: str,
) -> dict[str, Any] | None:
    """Return a standalone valid method lock, irrespective of task status."""
    path = task_root / FORMAL_METHOD_LOCK_FILE
    payload = optional_json(path)
    method = payload.get("training_method") if payload is not None else None
    if not isinstance(method, str):
        return None
    try:
        record = validate_formal_training_method_lock(
            path,
            experiment_id=experiment_id,
            training_policy=training_policy,
            training_method=method,
        )
    except (FormalTrainingEvidenceError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    return {"path": str(path), "sha256": sha256(path), "record": record}


def _status_contract_reasons(
    status: dict[str, Any] | None,
    *,
    expected_samples: int,
    training_policy: str,
) -> list[str]:
    reasons: list[str] = []
    if status is None:
        reasons.append("missing_or_invalid_status")
        return reasons
    if status.get("state") != "succeeded":
        reasons.append(f"state={status.get('state')!r}")
    if status.get("formal_expected_samples") != expected_samples:
        reasons.append("status_sample_contract_mismatch")
    if status.get("training_policy", "paper") != training_policy:
        reasons.append("status_training_policy_mismatch")
    if status.get("formal_training_validation_errors"):
        reasons.append("status_records_formal_errors")
    return reasons


def _record_consistency_reasons(
    status: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> list[str]:
    if not records:
        return []
    reasons: list[str] = []
    methods = {record.get("training_method") for record in records}
    method_locks = {json.dumps(record.get("formal_training_method_lock"), sort_keys=True) for record in records}
    if len(methods) != 1:
        reasons.append("formal_methods_disagree")
    if len(method_locks) != 1:
        reasons.append("formal_method_locks_disagree")
    if status is None:
        return reasons
    signatures = {
        record.get("evidence_signature") for record in records if isinstance(record.get("evidence_signature"), str)
    }
    status_records = status.get("formal_training_evidence")
    status_signatures = (
        {
            record.get("evidence_signature")
            for record in status_records
            if isinstance(record, dict) and isinstance(record.get("evidence_signature"), str)
        }
        if isinstance(status_records, list)
        else set()
    )
    if status_signatures != signatures:
        reasons.append("status_evidence_signatures_mismatch")
    method = next(iter(methods)) if len(methods) == 1 else None
    if status.get("formal_training_method") != method:
        reasons.append("status_formal_method_mismatch")
    lock = records[0].get("formal_training_method_lock")
    if status.get("formal_training_method_lock") != lock:
        reasons.append("status_method_lock_mismatch")
    return reasons


def status_matches_strict_evidence(
    status: dict[str, Any] | None,
    records: list[dict[str, Any]],
    errors: list[str],
    *,
    expected_samples: int,
    training_policy: str,
) -> tuple[bool, list[str]]:
    reasons = _status_contract_reasons(
        status,
        expected_samples=expected_samples,
        training_policy=training_policy,
    )
    if errors:
        reasons.append("strict_scan_errors")
    if not records:
        reasons.append("no_strict_formal_evidence")
    reasons.extend(_record_consistency_reasons(status, records))
    return not reasons, reasons


def durable_evidence_is_repairable(
    records: list[dict[str, Any]],
    errors: list[str],
) -> tuple[bool, list[str]]:
    """Return whether durable evidence can safely replace stale visible state."""
    reasons: list[str] = []
    if errors:
        reasons.append("durable_scan_errors")
    if not records:
        reasons.append("no_durable_formal_evidence")
    reasons.extend(_record_consistency_reasons(None, records))
    return not reasons, reasons


def _selected_tasks(tasks: list[dict[str, Any]], only_pattern: str | None) -> list[dict[str, Any]]:
    """Return the manifest tasks selected by the same regex contract as run_matrix."""
    if only_pattern is None:
        return list(tasks)
    try:
        pattern = re.compile(only_pattern)
    except re.error as error:
        message = f"Invalid --only regular expression {only_pattern!r}: {error}"
        raise RetryPreparationError(message) from error
    selected = [task for task in tasks if pattern.search(str(task["experiment_id"]))]
    if not selected:
        message = f"--only selected no matrix tasks: {only_pattern!r}"
        raise RetryPreparationError(message)
    return selected


def _process_environment(environment_bytes: bytes) -> dict[str, str]:
    """Decode only the non-secret process fields needed for task ownership."""
    wanted = {b"FT_EXPERIMENT_ID", b"WORKSPACE_PATH"}
    result: dict[str, str] = {}
    for entry in environment_bytes.split(b"\0"):
        key, separator, value = entry.partition(b"=")
        if separator and key in wanted:
            result[key.decode("ascii")] = value.decode("utf-8", errors="replace")
    return result


def _scheduler_only_filter(command_parts: list[bytes]) -> tuple[bool, str | None]:
    """Return whether a scheduler has ``--only`` and its exact regex value."""
    for index, part in enumerate(command_parts):
        if part == b"--only":
            if index + 1 >= len(command_parts):
                return True, None
            return True, command_parts[index + 1].decode("utf-8", errors="replace")
        if part.startswith(b"--only="):
            return True, part.partition(b"=")[2].decode("utf-8", errors="replace")
    return False, None


def active_python_processes(run_root: Path) -> list[dict[str, Any]]:
    """Find live Python schedulers/workers associated with this run and task."""
    run_name = run_root.name
    root_bytes = os.fsencode(str(run_root))
    active: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text(encoding="utf-8").strip().lower()
            if "python" not in comm:
                continue
            command_bytes = (proc / "cmdline").read_bytes()
            environment_bytes = (proc / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, UnicodeError):
            continue
        scheduler_match = b"run_matrix.py" in command_bytes and os.fsencode(run_name) in command_bytes
        environment = _process_environment(environment_bytes)
        workspace_path = environment.get("WORKSPACE_PATH")
        worker_match = root_bytes in environment_bytes
        if workspace_path is not None:
            try:
                Path(workspace_path).resolve().relative_to(run_root)
            except (OSError, ValueError):
                pass
            else:
                worker_match = True
        if scheduler_match or worker_match:
            process = {
                "pid": int(proc.name),
                "command": command_bytes.replace(b"\0", b" ").decode("utf-8", errors="replace").strip(),
                "kind": "scheduler" if scheduler_match else "worker",
                "experiment_id": environment.get("FT_EXPERIMENT_ID") if worker_match else None,
                "workspace_path": workspace_path if worker_match else None,
            }
            if scheduler_match:
                command_parts = [part for part in command_bytes.split(b"\0") if part]
                has_only_filter, only_pattern = _scheduler_only_filter(command_parts)
                process["has_only_filter"] = has_only_filter
                process["only_pattern"] = only_pattern
            active.append(process)
    return sorted(active, key=lambda item: item["pid"])


def _scheduler_can_select(
    process: dict[str, Any],
    selected_experiment_ids: set[str],
) -> bool:
    """Conservatively decide whether a live scheduler can own a selected task."""
    if not process.get("has_only_filter"):
        return True
    only_pattern = process.get("only_pattern")
    if not isinstance(only_pattern, str) or not only_pattern:
        return True
    try:
        pattern = re.compile(only_pattern)
    except re.error:
        return True
    return any(pattern.search(experiment_id) for experiment_id in selected_experiment_ids)


def blocking_active_processes(
    active: list[dict[str, Any]],
    selected_experiment_ids: set[str],
    *,
    selective: bool,
) -> list[dict[str, Any]]:
    """Return processes that can race with the selected retry transaction."""
    if not selective:
        return list(active)
    blocking: list[dict[str, Any]] = []
    for process in active:
        experiment_id = process.get("experiment_id")
        if experiment_id is not None:
            if experiment_id in selected_experiment_ids:
                blocking.append(process)
            continue
        if process.get("kind") != "scheduler" or _scheduler_can_select(
            process,
            selected_experiment_ids,
        ):
            blocking.append(process)
    return blocking


def audit_tasks(
    run_root: Path,
    matrix: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policy = str(matrix["training_policy"])
    audited: list[dict[str, Any]] = []
    for task in tasks:
        experiment_id = str(task["experiment_id"])
        expected_samples = task.get("formal_expected_samples", task.get("data_limit"))
        if not isinstance(expected_samples, int) or expected_samples <= 0:
            message = f"Invalid sample contract: {experiment_id}"
            raise RetryPreparationError(message)
        task_root = run_root / safe_id(experiment_id)
        if task_root.exists() and (not task_root.is_dir() or task_root.is_symlink()):
            message = f"Task root is not a real directory: {task_root}"
            raise RetryPreparationError(message)
        status = optional_json(task_root / "status.json") if task_root.is_dir() else None
        durable_records: list[dict[str, Any]] = []
        durable_errors: list[str] = []
        visible_records: list[dict[str, Any]] = []
        visible_errors: list[str] = []
        if task_root.is_dir():
            durable_records, durable_errors = validate_task_formal_training(
                task_root,
                expected_samples=expected_samples,
                experiment_id=experiment_id,
                training_policy=policy,
                require_visible_evidence=False,
            )
            visible_records, visible_errors = validate_task_formal_training(
                task_root,
                expected_samples=expected_samples,
                experiment_id=experiment_id,
                training_policy=policy,
                require_visible_evidence=True,
            )
        strict, reasons = status_matches_strict_evidence(
            status,
            visible_records,
            visible_errors,
            expected_samples=expected_samples,
            training_policy=policy,
        )
        repairable, durable_reasons = durable_evidence_is_repairable(
            durable_records,
            durable_errors,
        )
        if not task_root.exists():
            reasons = ["task_root_absent"]
            action = "create_on_resume"
        elif strict:
            action = "keep"
        elif repairable:
            action = "repair_status"
        else:
            action = "archive"
        audited.append(
            {
                "experiment_id": experiment_id,
                "task_root": str(task_root),
                "status_state": status.get("state") if status is not None else None,
                "status_sha256": sha256(task_root / "status.json") if (task_root / "status.json").is_file() else None,
                "strict": strict,
                "strict_record_count": len(visible_records),
                "strict_scan_errors": visible_errors,
                "durable_record_count": len(durable_records),
                "durable_scan_errors": durable_errors,
                "durable_repairable": repairable,
                "durable_repair_reasons": durable_reasons,
                "formal_training_evidence": durable_records,
                "retry_reasons": reasons,
                "task_manifest": task,
                "formal_expected_samples": expected_samples,
                "training_policy": policy,
                "method_lock": (
                    preserved_method_lock(
                        task_root,
                        experiment_id=experiment_id,
                        training_policy=policy,
                    )
                    if task_root.is_dir()
                    else None
                ),
                "action": action,
            },
        )
    return audited


def validate_failed_dataset(name: str) -> dict[str, Any]:
    if not DATASET_NAME.fullmatch(name):
        message = f"Unsafe dataset name: {name!r}"
        raise RetryPreparationError(message)
    path = FT_ROOT / "datasets" / name
    if not path.is_dir() or path.is_symlink():
        message = f"Dataset directory is absent or unsafe: {path}"
        raise RetryPreparationError(message)
    manifest = read_json(path / "processing_manifest.json")
    if manifest.get("status") not in FAILED_DATASET_STATES or manifest.get("production_eligible") is not False:
        message = f"Refusing to quarantine dataset without an explicit failed/ineligible manifest: {path}"
        raise RetryPreparationError(message)
    registry = read_json(FT_ROOT / "datasets" / "dataset_info.json")
    return {
        "name": name,
        "path": str(path),
        "manifest": manifest,
        "registered": name in registry,
    }


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy one regular file and atomically expose the completed destination."""
    if not source.is_file() or source.is_symlink():
        message = f"Recovery source is not a regular file: {source}"
        raise RetryPreparationError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copy2(source, temporary, follow_symlinks=True)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _link_or_copy(source: str, destination: str) -> str:
    """Materialize immutable durable output without rewriting its source."""
    try:
        os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)
    return destination


def _validated_record_paths(
    item: dict[str, Any],
    record: dict[str, Any],
) -> tuple[str, Path, Path, Path]:
    """Bind scanner-provided paths back to the expected task-root layout."""
    task_root = Path(item["task_root"]).resolve()
    workspace_root = task_root / "workspace"
    workspace_id = record.get("workspace_id")
    if (
        not isinstance(workspace_id, str)
        or not workspace_id
        or Path(workspace_id).name != workspace_id
        or workspace_id in {".", ".."}
    ):
        message = f"Unsafe durable workspace id: {workspace_id!r}"
        raise RetryPreparationError(message)

    visible = workspace_root / workspace_id
    provenance = workspace_root / ".ft_model_checkpoints" / workspace_id / "formal_training"
    output = provenance.parent / "output"
    supplied = {
        "workspace_path": visible,
        "provenance_path": provenance,
        "output_path": output,
    }
    for key, expected in supplied.items():
        raw = record.get(key)
        if not isinstance(raw, str) or Path(raw).resolve() != expected.resolve():
            message = f"Durable evidence {key} escapes its task root: {raw!r}"
            raise RetryPreparationError(message)
    if not provenance.is_dir() or provenance.is_symlink():
        message = f"Durable provenance is absent or unsafe: {provenance}"
        raise RetryPreparationError(message)
    if not output.is_dir() or output.is_symlink():
        message = f"Durable model output is absent or unsafe: {output}"
        raise RetryPreparationError(message)
    if visible.is_symlink() or (visible.exists() and not visible.is_dir()):
        message = f"Visible workspace is unsafe: {visible}"
        raise RetryPreparationError(message)
    return workspace_id, visible, provenance, output


def _restore_visible_workspace(  # noqa: C901, PLR0912, PLR0915 - transactional restore/rollback
    item: dict[str, Any],
    record: dict[str, Any],
    repair_archive: Path,
) -> dict[str, Any]:
    """Restore one visible candidate from immutable durable provenance/output."""
    workspace_id, visible, provenance, durable_output = _validated_record_paths(item, record)
    expected_samples = int(item["formal_expected_samples"])
    experiment_id = str(item["experiment_id"])
    training_policy = str(item["training_policy"])
    relative_files = formal_training_provenance_files(provenance)
    workspace_root = visible.parent
    workspace_root.mkdir(parents=True, exist_ok=True)

    backup_root = repair_archive / "workspaces" / workspace_id
    if backup_root.exists():
        message = f"Workspace recovery backup already exists: {backup_root}"
        raise RetryPreparationError(message)
    backup_root.mkdir(parents=True)

    with TemporaryDirectory(prefix=f".{workspace_id}.retry-recovery.", dir=workspace_root) as raw_staging:
        staging = Path(raw_staging) / "workspace"
        shutil.copytree(provenance, staging, copy_function=shutil.copy2, symlinks=False)
        staged_output = staging / "output"
        shutil.copytree(
            durable_output,
            staged_output,
            copy_function=_link_or_copy,
            symlinks=True,
        )
        staged_artifact = validate_formal_training_evidence(
            staging,
            expected_samples=expected_samples,
            experiment_id=experiment_id,
            training_policy=training_policy,
            output_path=staged_output,
        )
        if staged_artifact.get("evidence_signature") != record.get("evidence_signature"):
            message = f"Staged evidence signature changed for workspace {workspace_id}"
            raise RetryPreparationError(message)

        visible.mkdir(parents=True, exist_ok=True)
        provenance_backup = backup_root / "provenance.before"
        file_backups: list[dict[str, Any]] = []
        for relative in relative_files:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                message = f"Unsafe provenance-relative path: {relative}"
                raise RetryPreparationError(message)
            target = visible / relative_path
            existed = target.exists() or target.is_symlink()
            entry: dict[str, Any] = {"path": relative, "existed": existed}
            if existed:
                if not target.is_file() or target.is_symlink():
                    message = f"Visible provenance target is unsafe: {target}"
                    raise RetryPreparationError(message)
                backup = provenance_backup / relative_path
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
                entry["original_sha256"] = sha256(target)
            entry["restored_sha256"] = sha256(staging / relative_path)
            file_backups.append(entry)

        visible_output = visible / "output"
        output_backup = backup_root / "output.before"
        output_existed = visible_output.exists() or visible_output.is_symlink()
        if output_existed:
            if not visible_output.is_dir() or visible_output.is_symlink():
                message = f"Visible output is unsafe: {visible_output}"
                raise RetryPreparationError(message)
            visible_output.replace(output_backup)

        installed_output = False
        try:
            staged_output.replace(visible_output)
            installed_output = True
            for relative in relative_files:
                _atomic_copy_file(staging / relative, visible / relative)
            restored_artifact = validate_formal_training_evidence(
                visible,
                expected_samples=expected_samples,
                experiment_id=experiment_id,
                training_policy=training_policy,
                output_path=visible_output,
            )
            _require(
                condition=restored_artifact == staged_artifact,
                message=f"Visible evidence changed during recovery for workspace {workspace_id}",
            )
        except Exception:
            if installed_output:
                shutil.rmtree(visible_output, ignore_errors=True)
            if output_existed and output_backup.exists():
                output_backup.replace(visible_output)
            for entry in file_backups:
                target = visible / entry["path"]
                if entry["existed"]:
                    _atomic_copy_file(provenance_backup / entry["path"], target)
                else:
                    target.unlink(missing_ok=True)
            raise

    restore_record = {
        "workspace_id": workspace_id,
        "workspace_path": str(visible),
        "provenance_path": str(provenance),
        "output_path": str(durable_output),
        "evidence_signature": record["evidence_signature"],
        "training_method": record["training_method"],
        "visible_output_existed": output_existed,
        "provenance_files": file_backups,
    }
    atomic_write_json(backup_root / "restore.json", restore_record)
    return restore_record


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("workspace_id"),
        record.get("workspace_path"),
        record.get("provenance_path"),
        record.get("output_path"),
        record.get("evidence_signature"),
        record.get("training_method"),
        record.get("expected_samples"),
        record.get("global_step"),
        record.get("max_steps"),
        json.dumps(record.get("formal_training_method_lock"), sort_keys=True),
    )


def apply_status_repairs(  # noqa: C901, PLR0912, PLR0915 - verify-before-write transaction
    audited: list[dict[str, Any]],
    archive_root: Path,
) -> None:
    """Restore durable candidates and atomically repair only their stale status."""
    for item in audited:
        if item["action"] != "repair_status":
            continue
        task_root = Path(item["task_root"])
        status_path = task_root / "status.json"
        planned_status_sha = item.get("status_sha256")
        if status_path.is_file():
            current_status_sha = sha256(status_path)
            if current_status_sha != planned_status_sha:
                message = f"Task status changed after audit: {status_path}"
                raise RetryPreparationError(message)
        elif status_path.exists() or planned_status_sha is not None:
            message = f"Task status changed after audit: {status_path}"
            raise RetryPreparationError(message)

        repair_archive = archive_root / "repairs" / safe_id(str(item["experiment_id"]))
        if repair_archive.exists():
            message = f"Repair archive already exists: {repair_archive}"
            raise RetryPreparationError(message)
        repair_archive.mkdir(parents=True)
        status_backup = repair_archive / "status.before.json"
        if status_path.is_file():
            shutil.copy2(status_path, status_backup)
        original_status = optional_json(status_path)

        current_records, current_errors = validate_task_formal_training(
            task_root,
            expected_samples=int(item["formal_expected_samples"]),
            experiment_id=str(item["experiment_id"]),
            training_policy=str(item["training_policy"]),
            require_visible_evidence=False,
        )
        repairable, repair_reasons = durable_evidence_is_repairable(current_records, current_errors)
        if not repairable:
            message = (
                f"Durable evidence changed after audit for {item['experiment_id']}: "
                + ", ".join(repair_reasons)
            )
            raise RetryPreparationError(message)
        planned_records = item.get("formal_training_evidence", [])
        if [_record_identity(record) for record in current_records] != [
            _record_identity(record) for record in planned_records
        ]:
            message = f"Durable evidence inventory changed after audit: {item['experiment_id']}"
            raise RetryPreparationError(message)

        restored = [
            _restore_visible_workspace(item, record, repair_archive)
            for record in current_records
        ]
        visible_records, visible_errors = validate_task_formal_training(
            task_root,
            expected_samples=int(item["formal_expected_samples"]),
            experiment_id=str(item["experiment_id"]),
            training_policy=str(item["training_policy"]),
            require_visible_evidence=True,
        )
        if visible_errors or [_record_identity(record) for record in visible_records] != [
            _record_identity(record) for record in current_records
        ]:
            message = f"Visible formal evidence did not revalidate after recovery: {item['experiment_id']}"
            raise RetryPreparationError(message)

        recovered_at = datetime.now(UTC).astimezone().isoformat()
        status = dict(original_status) if original_status is not None else dict(item["task_manifest"])
        methods = {str(record["training_method"]) for record in visible_records}
        if len(methods) != 1:
            message = f"Recovered formal methods disagree: {item['experiment_id']}"
            raise RetryPreparationError(message)
        method = methods.pop()
        method_lock = visible_records[0]["formal_training_method_lock"]
        recovery = {
            "schema_version": 1,
            "recovered_at": recovered_at,
            "original_status_present": status_backup.is_file(),
            "original_status_sha256": planned_status_sha,
            "original_state": original_status.get("state") if original_status is not None else None,
            "original_return_code": (
                original_status.get("return_code") if original_status is not None else None
            ),
            "status_backup_path": str(status_backup) if status_backup.is_file() else None,
            "evidence_sources": [
                {
                    "workspace_id": record["workspace_id"],
                    "provenance_path": record["provenance_path"],
                    "output_path": record["output_path"],
                    "evidence_signature": record["evidence_signature"],
                }
                for record in visible_records
            ],
            "restored_workspaces": restored,
        }
        status.update(
            {
                "state": "succeeded",
                "finished_at": status.get("finished_at") or recovered_at,
                "formal_expected_samples": int(item["formal_expected_samples"]),
                "training_policy": str(item["training_policy"]),
                "formal_training_evidence": visible_records,
                "formal_training_validation_errors": [],
                "formal_training_method": method,
                "formal_training_method_lock": method_lock,
                "formal_training_recovery": recovery,
            },
        )
        strict, reasons = status_matches_strict_evidence(
            status,
            visible_records,
            visible_errors,
            expected_samples=int(item["formal_expected_samples"]),
            training_policy=str(item["training_policy"]),
        )
        if not strict:
            message = f"Recovered status is not strict for {item['experiment_id']}: {', '.join(reasons)}"
            raise RetryPreparationError(message)
        try:
            atomic_write_json(status_path, status)
            persisted = read_json(status_path)
            persisted_strict, persisted_reasons = status_matches_strict_evidence(
                persisted,
                visible_records,
                visible_errors,
                expected_samples=int(item["formal_expected_samples"]),
                training_policy=str(item["training_policy"]),
            )
            _require(
                condition=persisted_strict,
                message=(
                    f"Persisted recovered status is not strict for {item['experiment_id']}: "
                    + ", ".join(persisted_reasons)
                ),
            )
        except Exception:
            if status_backup.is_file():
                _atomic_copy_file(status_backup, status_path)
            else:
                status_path.unlink(missing_ok=True)
            raise
        atomic_write_json(
            repair_archive / "repair.json",
            {
                "experiment_id": item["experiment_id"],
                "status_path": str(status_path),
                "recovery": recovery,
                "repaired_status_sha256": sha256(status_path),
            },
        )


def apply_task_archives(audited: list[dict[str, Any]], archive_root: Path) -> None:
    task_archive = archive_root / "tasks"
    for item in audited:
        if item["action"] != "archive":
            continue
        source = Path(item["task_root"])
        destination = task_archive / safe_id(str(item["experiment_id"]))
        if destination.exists():
            message = f"Archive destination already exists: {destination}"
            raise RetryPreparationError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        method_lock = item.get("method_lock")
        if method_lock is not None:
            source.mkdir(parents=True, exist_ok=False)
            archived_lock = destination / FORMAL_METHOD_LOCK_FILE
            if sha256(archived_lock) != method_lock["sha256"]:
                message = f"Archived method lock changed unexpectedly: {archived_lock}"
                raise RetryPreparationError(message)
            shutil.copy2(archived_lock, source / FORMAL_METHOD_LOCK_FILE)


def apply_dataset_quarantine(datasets: list[dict[str, Any]], archive_root: Path) -> None:
    if not datasets:
        return
    registry_path = FT_ROOT / "datasets" / "dataset_info.json"
    registry = read_json(registry_path)
    snapshot = archive_root / "datasets" / "dataset_info.before.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_path, snapshot)
    for item in datasets:
        name = str(item["name"])
        source = Path(item["path"])
        destination = archive_root / "datasets" / name
        if destination.exists():
            message = f"Dataset quarantine destination already exists: {destination}"
            raise RetryPreparationError(message)
        source.replace(destination)
        registry.pop(name, None)
    atomic_write_json(registry_path, registry)


def build_plan(
    run_root: Path,
    dataset_names: list[str],
    only_pattern: str | None = None,
) -> dict[str, Any]:
    matrix, tasks = load_matrix(run_root)
    if only_pattern is not None and dataset_names:
        message = "Dataset quarantine cannot be combined with selective --only retry preparation"
        raise RetryPreparationError(message)
    selected_tasks = _selected_tasks(tasks, only_pattern)
    audited = audit_tasks(run_root, matrix, selected_tasks)
    datasets = [validate_failed_dataset(name) for name in dict.fromkeys(dataset_names)]
    active = active_python_processes(run_root)
    selected_experiment_ids = {str(task["experiment_id"]) for task in selected_tasks}
    blocking_active = blocking_active_processes(
        active,
        selected_experiment_ids,
        selective=only_pattern is not None,
    )
    state_counts: dict[str, int] = {}
    for item in audited:
        state = str(item["status_state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    action_counts = {
        action: sum(item["action"] == action for item in audited)
        for action in ("keep", "repair_status", "archive", "create_on_resume")
    }
    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).astimezone().isoformat(),
        "run_root": str(run_root),
        "matrix_sha256": sha256(run_root / "matrix.json"),
        "training_policy": matrix["training_policy"],
        "matrix_task_count": len(tasks),
        "task_count": len(selected_tasks),
        "selected_experiment_ids": sorted(selected_experiment_ids),
        "only_pattern": only_pattern,
        "strict_task_count": action_counts["keep"],
        "repair_task_count": action_counts["repair_status"],
        "retry_task_count": action_counts["archive"] + action_counts["create_on_resume"],
        "action_counts": action_counts,
        "status_counts": state_counts,
        "active_python_processes": active,
        "blocking_active_python_processes": blocking_active,
        "tasks": audited,
        "datasets": datasets,
    }


def validate_apply_preconditions(
    plan: dict[str, Any],
    current_active: list[dict[str, Any]],
) -> None:
    """Reject an apply transaction that can overlap a selected live task."""
    selective = plan.get("only_pattern") is not None
    selected_experiment_ids = set(plan.get("selected_experiment_ids", []))
    blocking_active = blocking_active_processes(
        current_active,
        selected_experiment_ids,
        selective=selective,
    )
    nonterminal = [
        item
        for item in plan["tasks"]
        if item["status_state"] not in TERMINAL_STATES and item["status_state"] is not None
    ]
    if blocking_active:
        identifiers = ", ".join(
            f"pid={item['pid']} task={item.get('experiment_id') or '<unassigned>'}"
            for item in blocking_active
        )
        message = f"Selected matrix tasks are still active; refusing retry preparation: {identifiers}"
        raise RetryPreparationError(message)
    # A selective retry intentionally repairs stale `running` states once no
    # process owns those selected task roots. Preserve the stricter historical
    # gate for an unfiltered whole-matrix transaction.
    if nonterminal and not selective:
        message = "Matrix has nonterminal task states; refusing to archive or quarantine any artifact"
        raise RetryPreparationError(message)


def main() -> int:
    args = parse_args()
    run_root = resolve_run_root(args.matrix_run)
    plan = build_plan(run_root, args.quarantine_dataset, args.only)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.apply:
        return 0
    # Rescan immediately before mutation so a task that became active after
    # the read-only audit cannot be archived or repaired from a stale plan.
    current_active = active_python_processes(run_root)
    validate_apply_preconditions(plan, current_active)
    plan["active_python_processes_at_apply"] = current_active
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%dT%H%M%S%z")
    archive_root = FT_ROOT / "quarantine" / "matrix-retries" / run_root.name / stamp
    if archive_root.exists():
        message = f"Archive root already exists: {archive_root}"
        raise RetryPreparationError(message)
    archive_root.mkdir(parents=True)
    journal = {
        **plan,
        "apply_started_at": datetime.now(UTC).astimezone().isoformat(),
        "archive_root": str(archive_root),
        "apply_state": "running",
    }
    atomic_write_json(archive_root / "retry_preparation.json", journal)
    try:
        apply_status_repairs(plan["tasks"], archive_root)
        apply_task_archives(plan["tasks"], archive_root)
        apply_dataset_quarantine(plan["datasets"], archive_root)
    except Exception as error:
        journal["apply_state"] = "failed"
        journal["apply_error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(archive_root / "retry_preparation.json", journal)
        raise
    journal["apply_state"] = "complete"
    journal["apply_completed_at"] = datetime.now(UTC).astimezone().isoformat()
    atomic_write_json(archive_root / "retry_preparation.json", journal)
    print(f"Prepared retry archive: {archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
