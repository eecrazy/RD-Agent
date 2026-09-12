# ruff: noqa: PLR2004
"""End-to-end provenance tests for strict ordinary-LoRA/rsLoRA pairs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reproduction.ft_agent.collect_results import collect_ft_task, lora_rslora_comparison_rows
from reproduction.ft_agent.final_test_protocol import FINAL_TEST_SCHEMA_VERSION, TEST_RANGE
from reproduction.ft_agent.rslora_pairing import (
    PAIRING_ARTIFACT_FILE,
    PAIRING_RUN_KIND,
    PairingContractError,
    build_pair_input_contract,
    file_sha256,
    make_pairing_artifact,
    paired_experiment_id,
    paired_train_yaml,
    paired_workspace_id,
    validate_pairing_artifact,
)
from reproduction.ft_agent.run_final_test import target_from_validation_selection
from reproduction.ft_agent.run_matrix import safe_id, validate_task_formal_training
from reproduction.ft_agent.run_rslora_paired import _finalization_finished_at, _schedule_longest_first
from reproduction.ft_agent.run_validation_sweep import (
    SweepTarget,
    candidate_set_signature,
    discover_candidates,
    validate_target_pairing,
)
from reproduction.ft_agent.validation_selection import (
    ValidationSelectionError,
    make_selection_artifact,
)

SOURCE_ID = "main/aime25/run-1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_SAMPLES = 2000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _scheduling_spec(tmp_path: Path, name: str, runtime: float) -> SimpleNamespace:
    task_root = tmp_path / name
    workspace_id = f"workspace-{name}"
    _write_json(
        task_root / "workspace" / workspace_id / "output" / "train_results.json",
        {"train_runtime": runtime},
    )
    return SimpleNamespace(
        source_task_root=task_root,
        source_workspace_id=workspace_id,
        paired_experiment_id=f"rslora-paired/main/{name}/run-1",
        paired_workspace_id=f"paired-{name}",
    )


def test_paired_queue_schedules_longest_measured_source_first(tmp_path: Path) -> None:
    short = _scheduling_spec(tmp_path, "short", 12.0)
    longest = _scheduling_spec(tmp_path, "longest", 300.0)
    medium = _scheduling_spec(tmp_path, "medium", 80.0)

    assert _schedule_longest_first([short, longest, medium]) == [longest, medium, short]


def test_paired_queue_rejects_missing_source_runtime(tmp_path: Path) -> None:
    invalid = _scheduling_spec(tmp_path, "invalid", 1.0)
    _write_json(
        invalid.source_task_root
        / "workspace"
        / invalid.source_workspace_id
        / "output"
        / "train_results.json",
        {},
    )

    with pytest.raises(PairingContractError, match="runtime is missing or invalid"):
        _schedule_longest_first([invalid])


def test_successful_paired_resume_preserves_task_completion_time(monkeypatch: pytest.MonkeyPatch) -> None:
    original = "2026-09-10T15:20:22.479733+08:00"
    monkeypatch.setattr(
        "reproduction.ft_agent.run_rslora_paired.utc_now",
        lambda: "2026-09-10T17:43:51.717982+08:00",
    )

    assert _finalization_finished_at({"state": "succeeded", "finished_at": original}) == original
    assert _finalization_finished_at({"state": "running", "finished_at": original}) != original


def _formal_records(task_root: Path, experiment_id: str, policy: str) -> list[dict[str, Any]]:
    records, errors = validate_task_formal_training(
        task_root,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id=experiment_id,
        training_policy=policy,
        require_visible_evidence=True,
    )
    assert errors == []
    assert records
    return records


def _complete_status(
    task_root: Path,
    *,
    experiment_id: str,
    policy: str,
    records: list[dict[str, Any]],
    source_experiment_id: str | None = None,
) -> dict[str, Any]:
    status = json.loads((task_root / "status.json").read_text(encoding="utf-8"))
    status.update(
        {
            "state": "succeeded",
            "experiment_id": experiment_id,
            "benchmark": "aime25",
            "model": MODEL,
            "planner": "gpt-5.6-sol",
            "data_limit": EXPECTED_SAMPLES,
            "formal_expected_samples": EXPECTED_SAMPLES,
            "benchmark_dataset_path": "pinned/aime25",
            "training_policy": policy,
            "formal_training_method": records[0]["training_method"],
            "formal_training_method_lock": records[0]["formal_training_method_lock"],
            "formal_training_evidence": records,
        },
    )
    if source_experiment_id is not None:
        status.update(
            {
                "run_kind": PAIRING_RUN_KIND,
                "source_experiment_id": source_experiment_id,
                "pairing_artifact_signature": None,
            },
        )
    _write_json(task_root / "status.json", status)
    return status


def _build_strict_pair(
    tmp_path: Path,
    formal_session_writer: Callable[..., None],
    *,
    workspace_count: int = 1,
) -> dict[str, Any]:
    source_root = tmp_path / "paper-matrix"
    source_task = source_root / safe_id(SOURCE_ID)
    source_plans = {f"ordinary-{index}": 4 + index for index in range(workspace_count)}
    formal_session_writer(
        source_task,
        source_plans,
        experiment_id=SOURCE_ID,
        training_policy="paper",
        training_method="lora",
    )
    source_records = _formal_records(source_task, SOURCE_ID, "paper")
    _complete_status(source_task, experiment_id=SOURCE_ID, policy="paper", records=source_records)
    source_manifest = {
        "schema_version": 1,
        "suite": "main",
        "training_policy": "paper",
        "formal_training_contract": {"rslora_is_paired_comparison_only": True},
        "tasks": [
            {
                "experiment_id": SOURCE_ID,
                "benchmark": "aime25",
                "model": MODEL,
                "planner": "gpt-5.6-sol",
                "data_limit": EXPECTED_SAMPLES,
                "formal_expected_samples": EXPECTED_SAMPLES,
            },
        ],
    }
    _write_json(source_root / "matrix.json", source_manifest)

    pair_id = paired_experiment_id(SOURCE_ID)
    pair_root = tmp_path / "paired-matrix"
    pair_task = pair_root / safe_id(pair_id)
    pair_plans: dict[str, int] = {}
    contracts = []
    for source_record in source_records:
        target_workspace_id = paired_workspace_id(
            str(source_record["workspace_id"]),
            str(source_record["evidence_signature"]),
        )
        pair_plans[target_workspace_id] = int(source_record["max_steps"])
        contracts.append(
            build_pair_input_contract(
                source_experiment_id=SOURCE_ID,
                paired_experiment_id_value=pair_id,
                source_workspace_id=str(source_record["workspace_id"]),
                paired_workspace_id_value=target_workspace_id,
                source_evidence_signature=str(source_record["evidence_signature"]),
                source_provenance=source_record["provenance_path"],
                expected_samples=EXPECTED_SAMPLES,
            ),
        )
    formal_session_writer(
        pair_task,
        pair_plans,
        experiment_id=pair_id,
        training_policy="rslora",
        training_method="rslora",
    )
    pair_records = _formal_records(pair_task, pair_id, "rslora")
    status = _complete_status(
        pair_task,
        experiment_id=pair_id,
        policy="rslora",
        records=pair_records,
        source_experiment_id=SOURCE_ID,
    )
    pair_manifest = {
        "schema_version": 1,
        "suite": "main-rslora-paired",
        "training_policy": "rslora",
        "paired_comparison": {
            "schema_version": 1,
            "run_kind": PAIRING_RUN_KIND,
            "source_matrix_root": str(source_root.resolve()),
            "source_matrix_manifest_sha256": file_sha256(source_root / "matrix.json"),
            "source_task_count": 1,
            "paired_task_count": 1,
            "paired_workspace_count": workspace_count,
        },
        "tasks": [
            {
                "experiment_id": pair_id,
                "benchmark": "aime25",
                "model": MODEL,
                "planner": "gpt-5.6-sol",
                "data_limit": EXPECTED_SAMPLES,
                "formal_expected_samples": EXPECTED_SAMPLES,
                "pairing": {
                    "source_experiment_id": SOURCE_ID,
                    "expected_pairs": sorted(contracts, key=lambda item: item["paired_workspace_id"]),
                },
            },
        ],
    }
    _write_json(pair_root / "matrix.json", pair_manifest)
    artifact = make_pairing_artifact(pair_task)
    _write_json(pair_task / PAIRING_ARTIFACT_FILE, artifact)
    status["pairing_artifact_signature"] = artifact["artifact_signature"]
    _write_json(pair_task / "status.json", status)
    validate_pairing_artifact(
        pair_task,
        experiment_id=pair_id,
        expected_samples=EXPECTED_SAMPLES,
        expected_signature=artifact["artifact_signature"],
    )
    return {
        "source_root": source_root,
        "source_task": source_task,
        "source_records": source_records,
        "pair_root": pair_root,
        "pair_task": pair_task,
        "pair_records": pair_records,
        "pair_id": pair_id,
        "status": status,
        "artifact": artifact,
    }


def _write_validation_selection(pair: dict[str, Any], value: float = 75.0) -> dict[str, Any]:
    signature = pair["artifact"]["artifact_signature"]
    target = SweepTarget(
        experiment_id=pair["pair_id"],
        benchmark="aime25",
        model=MODEL,
        benchmark_dataset_path="pinned/aime25",
        task_root=pair["pair_task"],
        expected_samples=EXPECTED_SAMPLES,
        training_policy="rslora",
        pairing_artifact_signature=signature,
    )
    discovered = discover_candidates(target, include_baseline=False, include_final_outputs=False)
    assert discovered
    selected_candidate = discovered[0]
    candidate = {
        **selected_candidate.identity(),
        "accuracy_summary": {"synthetic": {"accuracy": value}},
        "paper_metrics": [
            {
                "metric": "accuracy",
                "label": "Accuracy",
                "value": value,
                "higher_is_better": True,
                "unit": "percent",
                "formula": "reported accuracy",
            },
        ],
    }
    selection = make_selection_artifact(
        experiment_id=pair["pair_id"],
        benchmark="aime25",
        model=MODEL,
        benchmark_dataset_path="pinned/aime25",
        search_status={"state": "succeeded"},
        candidate_set_signature=candidate_set_signature(discovered),
        candidates=[candidate],
        expected_samples=EXPECTED_SAMPLES,
        include_baseline=False,
        include_final_outputs=False,
        pairing_artifact_signature=signature,
    )
    _write_json(pair["pair_task"] / "validation_selection.json", selection)
    return selection


def test_paired_yaml_changes_only_the_rslora_boolean() -> None:
    source = (
        b"# retained comment\n"
        b"finetuning_type: lora\n"
        b"use_rslora: false  # method bit\n"
        b"learning_rate: 1.0e-4\n"
    )
    target = paired_train_yaml(source)

    assert target == source.replace(b"use_rslora: false", b"use_rslora: true")
    with pytest.raises(PairingContractError, match="explicitly set"):
        paired_train_yaml(source.replace(b"use_rslora: false", b"use_rslora: true"))


def test_pairing_covers_every_ordinary_lora_workspace_once(
    tmp_path: Path,
    formal_session_writer: Callable[..., None],
) -> None:
    pair = _build_strict_pair(tmp_path, formal_session_writer, workspace_count=2)

    assert len(pair["artifact"]["pairs"]) == 2
    assert {item["source_workspace_id"] for item in pair["artifact"]["pairs"]} == {
        "ordinary-0",
        "ordinary-1",
    }

    manifest_path = pair["pair_root"] / "matrix.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"][0]["pairing"]["expected_pairs"].pop()
    _write_json(manifest_path, manifest)
    with pytest.raises(PairingContractError, match="every ordinary-LoRA"):
        validate_pairing_artifact(
            pair["pair_task"],
            experiment_id=pair["pair_id"],
            expected_samples=EXPECTED_SAMPLES,
        )


@pytest.mark.parametrize("tamper", ["yaml", "data", "evidence", "signature"])
def test_pairing_rejects_any_provenance_tampering(
    tmp_path: Path,
    formal_session_writer: Callable[..., None],
    tamper: str,
) -> None:
    pair = _build_strict_pair(tmp_path, formal_session_writer)
    provenance = Path(pair["pair_records"][0]["provenance_path"])
    if tamper == "yaml":
        (provenance / "train.yaml").write_text(
            (provenance / "train.yaml").read_text(encoding="utf-8") + "learning_rate: 9.9e-1\n",
            encoding="utf-8",
        )
    elif tamper == "data":
        (provenance / "data.json").write_text("[]\n", encoding="utf-8")
    elif tamper == "evidence":
        evidence_path = provenance / "formal_training_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["expected_samples"] = 1999
        _write_json(evidence_path, evidence)
    else:
        artifact_path = pair["pair_task"] / PAIRING_ARTIFACT_FILE
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["artifact_signature"] = "0" * 64
        _write_json(artifact_path, artifact)

    with pytest.raises((PairingContractError, RuntimeError)):
        validate_pairing_artifact(
            pair["pair_task"],
            experiment_id=pair["pair_id"],
            expected_samples=EXPECTED_SAMPLES,
            expected_signature=pair["artifact"]["artifact_signature"],
        )


def test_unsigned_rslora_is_rejected(tmp_path: Path) -> None:
    target = SweepTarget(
        experiment_id=paired_experiment_id(SOURCE_ID),
        benchmark="aime25",
        model=MODEL,
        benchmark_dataset_path="pinned/aime25",
        task_root=tmp_path,
        expected_samples=EXPECTED_SAMPLES,
        training_policy="rslora",
    )
    with pytest.raises(ValidationSelectionError, match="requires a signed"):
        validate_target_pairing(target)


def test_trace_free_pair_enters_validation_final_test_and_collection(
    tmp_path: Path,
    formal_session_writer: Callable[..., None],
) -> None:
    pair = _build_strict_pair(tmp_path, formal_session_writer)
    selection = _write_validation_selection(pair)

    target = target_from_validation_selection(pair["pair_task"], pair["status"])
    assert target.pairing_artifact_signature == pair["artifact"]["artifact_signature"]
    assert not (pair["pair_task"] / "trace").exists()

    _write_json(
        pair["pair_task"] / "final_test.json",
        {
            "schema_version": FINAL_TEST_SCHEMA_VERSION,
            "experiment_id": pair["pair_id"],
            "benchmark": "aime25",
            "model": MODEL,
            "training_policy": "rslora",
            "state": "succeeded",
            "evaluation_mode": "post_selection",
            "selection": target.selection(),
            "test_range": TEST_RANGE,
            "benchmark_dataset_path": "pinned/aime25",
            "pairing_artifact_signature": pair["artifact"]["artifact_signature"],
            "result": {"accuracy_summary": {"synthetic": {"accuracy": 80.0}}},
        },
    )
    record = collect_ft_task(pair["pair_task"], pair["status"])

    assert record["trace"] is None
    assert record["pairing"]["artifact_signature"] == pair["artifact"]["artifact_signature"]
    assert record["validation_selection"]["artifact_signature"] == selection["artifact_signature"]
    assert record["final"]["test"]["paper_metrics"][0]["value"] == 80.0

    source_record = {
        "run_root": record["pairing"]["source_matrix_root"],
        "experiment_id": SOURCE_ID,
        "benchmark": "aime25",
        "run_index": 1,
        "training_method": "lora",
        "pairing": None,
        "lora_comparison": {
            "final": {
                "validation": {
                    "paper_metrics": [
                        {"metric": "accuracy", "value": 70.0, "unit": "percent", "higher_is_better": True},
                    ],
                },
                "test": {
                    "paper_metrics": [
                        {"metric": "accuracy", "value": 72.0, "unit": "percent", "higher_is_better": True},
                    ],
                },
            },
        },
    }
    rows = lora_rslora_comparison_rows([source_record, record])
    test_row = next(item for item in rows if item["split"] == "test")
    assert test_row["source_value"] == 72.0
    assert test_row["paired_value"] == 80.0
    assert test_row["improvement_delta"] == 8.0
