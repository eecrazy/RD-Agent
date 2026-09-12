import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rdagent.scenarios.finetune.train.formal_training import enforce_formal_training_method_lock

SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "reproduction" / "ft_agent"
sys.path.insert(0, str(SCRIPT_ROOT))
prepare_matrix_retry = importlib.import_module("prepare_matrix_retry")

EXPECTED_SAMPLES = 2000
ORIGINAL_RETURN_CODE = 17
PLAN_SCHEMA_VERSION = 2
EXPECTED_RETRY_TASKS = 2


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_single_task(task_root: Path) -> list[dict[str, Any]]:
    experiment_id = "main/aime25/run-1"
    return prepare_matrix_retry.audit_tasks(
        task_root.parent,
        {"training_policy": "paper"},
        [{"experiment_id": experiment_id, "formal_expected_samples": 2000}],
    )


def test_status_requires_success_and_matching_strict_signatures() -> None:
    lock = {"training_method": "lora"}
    records = [
        {
            "training_method": "lora",
            "evidence_signature": "a" * 64,
            "formal_training_method_lock": lock,
        },
    ]
    status = {
        "state": "succeeded",
        "formal_expected_samples": 2000,
        "training_policy": "paper",
        "formal_training_validation_errors": [],
        "formal_training_method": "lora",
        "formal_training_method_lock": lock,
        "formal_training_evidence": [{"evidence_signature": "a" * 64}],
    }

    strict, reasons = prepare_matrix_retry.status_matches_strict_evidence(
        status,
        records,
        [],
        expected_samples=2000,
        training_policy="paper",
    )

    assert strict
    assert reasons == []
    status["formal_training_evidence"] = [{"evidence_signature": "b" * 64}]
    strict, reasons = prepare_matrix_retry.status_matches_strict_evidence(
        status,
        records,
        [],
        expected_samples=2000,
        training_policy="paper",
    )
    assert not strict
    assert "status_evidence_signatures_mismatch" in reasons


def test_archive_restores_only_valid_method_lock(tmp_path: Path) -> None:
    task_root = tmp_path / "run" / "task"
    task_root.mkdir(parents=True)
    lock_path = task_root / prepare_matrix_retry.FORMAL_METHOD_LOCK_FILE
    enforce_formal_training_method_lock(
        lock_path,
        experiment_id="main/example/run-1",
        training_policy="paper",
        training_method="lora",
    )
    (task_root / "status.json").write_text('{"state":"failed"}', encoding="utf-8")
    (task_root / "console.log").write_text("old attempt", encoding="utf-8")
    method_lock = prepare_matrix_retry.preserved_method_lock(
        task_root,
        experiment_id="main/example/run-1",
        training_policy="paper",
    )
    audited = [
        {
            "action": "archive",
            "task_root": str(task_root),
            "experiment_id": "main/example/run-1",
            "method_lock": method_lock,
        },
    ]
    archive_root = tmp_path / "archive"

    prepare_matrix_retry.apply_task_archives(audited, archive_root)

    assert sorted(path.name for path in task_root.iterdir()) == [prepare_matrix_retry.FORMAL_METHOD_LOCK_FILE]
    archived = archive_root / "tasks" / "main__example__run-1"
    assert (archived / "console.log").read_text(encoding="utf-8") == "old attempt"
    assert (archived / "status.json").is_file()


def test_failed_dataset_quarantine_removes_only_named_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ft_root = tmp_path / "finetune_files"
    datasets_root = ft_root / "datasets"
    failed = datasets_root / "failed_generated"
    _write_json(
        failed / "processing_manifest.json",
        {"status": "aborted", "production_eligible": False},
    )
    (failed / "partial.json").write_text("[]", encoding="utf-8")
    _write_json(
        datasets_root / "dataset_info.json",
        {"failed_generated": {"stale": True}, "keep": {"valid": True}},
    )
    monkeypatch.setattr(prepare_matrix_retry, "FT_ROOT", ft_root)
    item = prepare_matrix_retry.validate_failed_dataset("failed_generated")
    archive_root = tmp_path / "archive"

    prepare_matrix_retry.apply_dataset_quarantine([item], archive_root)

    registry = json.loads((datasets_root / "dataset_info.json").read_text(encoding="utf-8"))
    assert registry == {"keep": {"valid": True}}
    assert (archive_root / "datasets" / "failed_generated" / "partial.json").is_file()
    assert (archive_root / "datasets" / "dataset_info.before.json").is_file()


def test_valid_durable_evidence_repairs_status_and_visible_workspace(
    tmp_path: Path,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "run" / "main__aime25__run-1"
    workspace_id = "workspace-1"
    formal_session_writer(task_root, {workspace_id: 5})
    visible = task_root / "workspace" / workspace_id
    durable = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id
    durable_provenance = durable / "formal_training"
    durable_output = durable / "output"
    durable_evidence = json.loads(
        (durable_provenance / "formal_training_evidence.json").read_text(encoding="utf-8"),
    )
    evidence_signature = durable_evidence["evidence_signature"]
    durable_model = durable_output / "adapter_model.safetensors"
    durable_model_sha256 = _sha256(durable_model)
    method_lock = task_root / prepare_matrix_retry.FORMAL_METHOD_LOCK_FILE
    method_lock_sha256 = _sha256(method_lock)

    (visible / "train.yaml").write_text("corrupted visible configuration\n", encoding="utf-8")
    visible_output = visible / "output"
    visible_output.mkdir()
    (visible_output / "stale.txt").write_text("stale visible output", encoding="utf-8")
    original_status = (
        b'{\n  "state": "failed",\n  "return_code": 17,\n'
        b'  "finished_at": "2026-09-08T00:00:00+00:00"\n}\n'
    )
    (task_root / "status.json").write_bytes(original_status)

    audited = _audit_single_task(task_root)

    assert len(audited) == 1
    assert audited[0]["action"] == "repair_status"
    assert audited[0]["durable_record_count"] == 1
    # The visible scan binds the durable run to its visible signed evidence.
    # The stale status triggers recovery, which also restores all provenance.
    assert audited[0]["strict_record_count"] == 1
    archive_root = tmp_path / "archive"
    prepare_matrix_retry.apply_status_repairs(audited, archive_root)

    repair_root = archive_root / "repairs" / "main__aime25__run-1"
    assert (repair_root / "status.before.json").read_bytes() == original_status
    assert (repair_root / "workspaces" / workspace_id / "output.before" / "stale.txt").read_text(
        encoding="utf-8",
    ) == "stale visible output"
    repaired_status = json.loads((task_root / "status.json").read_text(encoding="utf-8"))
    assert repaired_status["state"] == "succeeded"
    assert repaired_status["return_code"] == ORIGINAL_RETURN_CODE
    assert repaired_status["formal_expected_samples"] == EXPECTED_SAMPLES
    assert repaired_status["training_policy"] == "paper"
    assert repaired_status["formal_training_method"] == "lora"
    assert repaired_status["formal_training_method_lock"]["training_method"] == "lora"
    assert repaired_status["formal_training_evidence"][0]["workspace_id"] == workspace_id
    assert repaired_status["formal_training_evidence"][0]["evidence_signature"] == evidence_signature
    assert repaired_status["formal_training_recovery"]["original_return_code"] == ORIGINAL_RETURN_CODE
    assert (visible / "train.yaml").read_bytes() == (durable_provenance / "train.yaml").read_bytes()
    assert (visible_output / "adapter_model.safetensors").read_bytes() == durable_model.read_bytes()

    records, errors = prepare_matrix_retry.validate_task_formal_training(
        task_root,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id="main/aime25/run-1",
        training_policy="paper",
        require_visible_evidence=True,
    )
    assert errors == []
    assert [record["workspace_id"] for record in records] == [workspace_id]
    assert [record["evidence_signature"] for record in records] == [evidence_signature]
    assert _sha256(method_lock) == method_lock_sha256
    assert _sha256(durable_model) == durable_model_sha256
    assert _audit_single_task(task_root)[0]["action"] == "keep"


def test_corrupted_durable_evidence_is_archived_not_repaired(
    tmp_path: Path,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "run" / "main__aime25__run-1"
    workspace_id = "workspace-1"
    formal_session_writer(task_root, {workspace_id: 5})
    durable_train = (
        task_root
        / "workspace"
        / ".ft_model_checkpoints"
        / workspace_id
        / "formal_training"
        / "train.yaml"
    )
    durable_train.write_text(durable_train.read_text(encoding="utf-8") + "seed: 999\n", encoding="utf-8")

    audited = _audit_single_task(task_root)

    assert audited[0]["action"] == "archive"
    assert audited[0]["durable_record_count"] == 0
    assert audited[0]["durable_scan_errors"]
    archive_root = tmp_path / "archive"
    prepare_matrix_retry.apply_task_archives(audited, archive_root)
    assert not (task_root / "status.json").exists()
    assert (task_root / prepare_matrix_retry.FORMAL_METHOD_LOCK_FILE).is_file()
    assert (archive_root / "tasks" / "main__aime25__run-1" / "status.json").is_file()


def test_build_plan_counts_repairs_separately_from_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    tasks = [
        {"experiment_id": f"main/task-{index}/run-1", "formal_expected_samples": 2000}
        for index in range(4)
    ]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})
    actions = ["keep", "repair_status", "archive", "create_on_resume"]
    monkeypatch.setattr(
        prepare_matrix_retry,
        "audit_tasks",
        lambda *_args: [
            {"action": action, "strict": action == "keep", "status_state": "failed"}
            for action in actions
        ],
    )
    monkeypatch.setattr(prepare_matrix_retry, "active_python_processes", lambda _run_root: [])

    plan = prepare_matrix_retry.build_plan(run_root, [])

    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["strict_task_count"] == 1
    assert plan["repair_task_count"] == 1
    assert plan["retry_task_count"] == EXPECTED_RETRY_TASKS
    assert plan["action_counts"] == dict.fromkeys(actions, 1)


def test_selective_plan_allows_unselected_active_task_and_stale_running_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    tasks = [
        {"experiment_id": "main/selected/run-1", "formal_expected_samples": 2000},
        {"experiment_id": "main/live/run-1", "formal_expected_samples": 2000},
    ]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})
    monkeypatch.setattr(
        prepare_matrix_retry,
        "audit_tasks",
        lambda _root, _matrix, selected: [
            {
                "experiment_id": selected[0]["experiment_id"],
                "action": "repair_status",
                "strict": False,
                "status_state": "running",
            },
        ],
    )
    active = [
        {
            "pid": 123,
            "command": "python worker.py",
            "kind": "worker",
            "experiment_id": "main/live/run-1",
            "workspace_path": str(run_root / "main__live__run-1" / "workspace"),
        },
    ]
    monkeypatch.setattr(prepare_matrix_retry, "active_python_processes", lambda _run_root: active)

    plan = prepare_matrix_retry.build_plan(run_root, [], r"^main/selected/run-1$")

    assert plan["matrix_task_count"] == len(tasks)
    assert plan["task_count"] == 1
    assert plan["blocking_active_python_processes"] == []
    prepare_matrix_retry.validate_apply_preconditions(plan, active)


def test_selective_plan_rejects_active_selected_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    tasks = [{"experiment_id": "main/selected/run-1", "formal_expected_samples": 2000}]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})
    monkeypatch.setattr(
        prepare_matrix_retry,
        "audit_tasks",
        lambda _root, _matrix, selected: [
            {
                "experiment_id": selected[0]["experiment_id"],
                "action": "archive",
                "strict": False,
                "status_state": "running",
            },
        ],
    )
    active = [
        {
            "pid": 456,
            "command": "python worker.py",
            "kind": "worker",
            "experiment_id": "main/selected/run-1",
            "workspace_path": str(run_root / "main__selected__run-1" / "workspace"),
        },
    ]
    monkeypatch.setattr(prepare_matrix_retry, "active_python_processes", lambda _run_root: active)

    plan = prepare_matrix_retry.build_plan(run_root, [], r"^main/selected/run-1$")

    assert plan["blocking_active_python_processes"] == active
    with pytest.raises(prepare_matrix_retry.RetryPreparationError, match="pid=456"):
        prepare_matrix_retry.validate_apply_preconditions(plan, active)


def test_selective_plan_allows_scheduler_with_disjoint_only_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    tasks = [
        {"experiment_id": "main/selected/run-1", "formal_expected_samples": 2000},
        {"experiment_id": "main/live/run-1", "formal_expected_samples": 2000},
    ]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})
    monkeypatch.setattr(
        prepare_matrix_retry,
        "audit_tasks",
        lambda _root, _matrix, selected: [
            {
                "experiment_id": selected[0]["experiment_id"],
                "action": "archive",
                "strict": False,
                "status_state": "running",
            },
        ],
    )
    active = [
        {
            "pid": 789,
            "command": "python run_matrix.py --only ^main/live/run-1$",
            "kind": "scheduler",
            "experiment_id": None,
            "workspace_path": None,
            "has_only_filter": True,
            "only_pattern": r"^main/live/run-1$",
        },
    ]
    monkeypatch.setattr(prepare_matrix_retry, "active_python_processes", lambda _run_root: active)

    plan = prepare_matrix_retry.build_plan(run_root, [], r"^main/selected/run-1$")

    assert plan["blocking_active_python_processes"] == []
    prepare_matrix_retry.validate_apply_preconditions(plan, active)


def test_selective_plan_rejects_scheduler_with_overlapping_only_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    tasks = [{"experiment_id": "main/selected/run-1", "formal_expected_samples": 2000}]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})
    monkeypatch.setattr(
        prepare_matrix_retry,
        "audit_tasks",
        lambda _root, _matrix, selected: [
            {
                "experiment_id": selected[0]["experiment_id"],
                "action": "archive",
                "strict": False,
                "status_state": "running",
            },
        ],
    )
    active = [
        {
            "pid": 987,
            "command": "python run_matrix.py --only ^main/selected/run-1$",
            "kind": "scheduler",
            "experiment_id": None,
            "workspace_path": None,
            "has_only_filter": True,
            "only_pattern": r"^main/selected/run-1$",
        },
    ]
    monkeypatch.setattr(prepare_matrix_retry, "active_python_processes", lambda _run_root: active)

    plan = prepare_matrix_retry.build_plan(run_root, [], r"^main/selected/run-1$")

    assert plan["blocking_active_python_processes"] == active
    with pytest.raises(prepare_matrix_retry.RetryPreparationError, match="pid=987"):
        prepare_matrix_retry.validate_apply_preconditions(plan, active)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([b"python", b"run_matrix.py", b"--only", b"^main/task/run-1$"], (True, "^main/task/run-1$")),
        ([b"python", b"run_matrix.py", b"--only=^main/task/run-1$"], (True, "^main/task/run-1$")),
        ([b"python", b"run_matrix.py"], (False, None)),
        ([b"python", b"run_matrix.py", b"--only"], (True, None)),
    ],
)
def test_scheduler_only_filter(arguments: list[bytes], expected: tuple[bool, str | None]) -> None:
    assert prepare_matrix_retry._scheduler_only_filter(arguments) == expected


def test_selective_plan_forbids_dataset_quarantine(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    tasks = [{"experiment_id": "main/selected/run-1", "formal_expected_samples": 2000}]
    _write_json(run_root / "matrix.json", {"training_policy": "paper", "tasks": tasks})

    with pytest.raises(prepare_matrix_retry.RetryPreparationError, match="Dataset quarantine"):
        prepare_matrix_retry.build_plan(run_root, ["failed-generated"], r"selected")
