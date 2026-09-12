"""Tests for terminal experiment-audit helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from reproduction.ft_agent.render_experiment_audit import (
    ExperimentAuditError,
    financeiq_pipeline_evidence,
    matrix_provenance_record,
    method_choice_checks,
    parse_final_test_log,
    select_best_successful_run,
    selected_model_evidence,
    stable_api_route,
)

FIXTURE_TRAINING_SAMPLES = 3
FIXTURE_HOLDOUT_SAMPLES = 2
FIXTURE_HASHED_FILES = 5


def test_direct_matrix_uses_hashed_matrix_manifest_as_root_provenance(tmp_path: Path) -> None:
    matrix = tmp_path / "direct"
    matrix.mkdir()
    matrix_manifest = matrix / "matrix.json"
    matrix_manifest.write_text('{"tasks": [{"experiment_id": "main/a/run-1"}]}\n', encoding="utf-8")
    (matrix / "main__a__run-1").mkdir()

    record = matrix_provenance_record(matrix, [{"experiment_id": "main/a/run-1"}])

    assert record["kind"] == "direct_matrix_manifest"
    assert record["materialized_task_roots"] == 1
    assert record["resolved_path"] == str(matrix_manifest)
    assert record["sha256"] == hashlib.sha256(matrix_manifest.read_bytes()).hexdigest()


def test_composed_matrix_without_provenance_is_rejected(tmp_path: Path) -> None:
    matrix = tmp_path / "composed"
    source = tmp_path / "source-task"
    matrix.mkdir()
    source.mkdir()
    (matrix / "matrix.json").write_text("{}\n", encoding="utf-8")
    (matrix / "main__a__run-1").symlink_to(source, target_is_directory=True)

    with pytest.raises(ExperimentAuditError, match="missing its required provenance"):
        matrix_provenance_record(matrix, [{"experiment_id": "main/a/run-1"}])


def _financeiq_rescue_fixture(tmp_path: Path) -> tuple[Path, Path]:
    task_root = tmp_path / "task"
    source = task_root / "workspace" / "source"
    target = task_root / "workspace" / "target"
    source.mkdir(parents=True)
    target.mkdir()

    files = {
        "process_data.py": b"# frozen generator\n",
        "data.json": json.dumps([{"row": index} for index in range(FIXTURE_TRAINING_SAMPLES)]).encode(),
        "validation.json": b'[{"row": "validation"}]',
        "data_stats.json": b'{"total_samples": 3}',
        "train.yaml": b"stage: sft\n",
    }
    for name, content in files.items():
        (source / name).write_bytes(content)
        (target / name).write_bytes(content)

    (source / "audit.json").write_text(
        json.dumps(
            {
                "mode": "full",
                "raw_source_count": 10,
                "training_counts": {"subject": FIXTURE_TRAINING_SAMPLES},
                "validation_counts": {"subject": 1},
                "formal_contract": {
                    "passed": True,
                    "expected_training_records": FIXTURE_TRAINING_SAMPLES,
                    "actual_training_records": FIXTURE_TRAINING_SAMPLES,
                    "actual_validation_records": 1,
                },
                "partition_overlap_assertions": {
                    "passed": True,
                    "train_validation_source_overlap": 0,
                    "train_benchmark_source_overlap": 0,
                    "validation_benchmark_source_overlap": 0,
                },
                "benchmark_partition": {
                    "total_tail_rows": FIXTURE_HOLDOUT_SAMPLES,
                    "excluded_from_training_and_validation": True,
                },
                "artifact_paths": {
                    "training": "data.json",
                    "validation": "validation.json",
                },
            },
        ),
        encoding="utf-8",
    )
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    (target / "formal_rescue_manifest.json").write_text(
        json.dumps(
            {
                "workspace_id": "target",
                "source_workspace_id": "source",
                "training_file": "data.json",
                "validation_file": "validation.json",
                "input_hashes": hashes,
                "source_hashes": hashes,
            },
        ),
        encoding="utf-8",
    )
    return task_root, target


def test_financeiq_pipeline_follows_hash_bound_rescue_lineage(tmp_path: Path) -> None:
    task_root, target = _financeiq_rescue_fixture(tmp_path)

    evidence = financeiq_pipeline_evidence(
        task_root,
        target,
        expected_training_samples=FIXTURE_TRAINING_SAMPLES,
    )

    assert evidence["audit_source_workspace_id"] == "source"
    assert evidence["accepted_training_count"] == FIXTURE_TRAINING_SAMPLES
    assert evidence["validation_count"] == 1
    assert evidence["holdout_count"] == FIXTURE_HOLDOUT_SAMPLES
    assert evidence["pairwise_intersection_counts"] == {
        "train_validation": 0,
        "train_holdout": 0,
        "validation_holdout": 0,
    }
    assert evidence["lineage_hashes"]["verified_input_count"] == FIXTURE_HASHED_FILES
    assert evidence["lineage_hashes"]["verified_source_count"] == FIXTURE_HASHED_FILES


def test_financeiq_pipeline_rejects_changed_rescue_input(tmp_path: Path) -> None:
    task_root, target = _financeiq_rescue_fixture(tmp_path)
    (target / "data.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ExperimentAuditError, match="rescue input artifact changed"):
        financeiq_pipeline_evidence(
            task_root,
            target,
            expected_training_samples=FIXTURE_TRAINING_SAMPLES,
        )


def test_method_choice_checks_prove_logical_b200_precedes_h20_execution() -> None:
    checks, rationale = method_choice_checks(
        "[Training] Use ordinary unquantized LoRA rather than full SFT; explicitly disable rsLoRA.",
        "With only 2,000 examples, full tuning has overfitting and catastrophic-forgetting risk.",
        locked_method="lora",
        training_resource={
            "source": "logical_resource_override",
            "gpu_count": 1,
            "gpu_name": "NVIDIA B200",
            "memory_per_gpu_gb": 178.0,
            "total_memory_gb": 178.0,
        },
    )

    assert all(checks.values())
    assert rationale["two_thousand_sample_overfit_or_forgetting_risk"] is True


def test_method_choice_checks_reject_physical_h20_resource() -> None:
    checks, _ = method_choice_checks(
        "Use full-parameter SFT and no rsLoRA.",
        "The full SFT configuration is preferred.",
        locked_method="full",
        training_resource={
            "source": "logical_resource_override",
            "gpu_count": 1,
            "gpu_name": "NVIDIA H20",
            "memory_per_gpu_gb": 96.0,
            "total_memory_gb": 96.0,
        },
    )

    assert checks["locked_method_named_in_initial_hypothesis"] is True
    assert checks["physical_h20_not_exposed_to_hypothesis"] is False
    assert checks["logical_gpu_is_b200"] is False


def _task(run_index: int, validation: float, test: float, state: str = "succeeded") -> dict[str, Any]:
    def metric(value: float) -> dict[str, list[dict[str, str | float | bool]]]:
        return {
            "paper_metrics": [
                {
                    "metric": "accuracy",
                    "label": "Acc",
                    "value": value,
                    "higher_is_better": True,
                    "unit": "percent",
                },
            ],
        }

    return {
        "paper_experiment_id": f"ft-agent/main/chemcotbench_mol_edit/run-{run_index}",
        "benchmark": "chemcotbench_mol_edit",
        "run_index": run_index,
        "final_test": {"state": state},
        "final": {"validation": metric(validation), "test": metric(test) if state == "succeeded" else None},
    }


def test_best_successful_supplement_uses_validation_not_test() -> None:
    expected_run_index = 2
    tasks = [
        _task(1, 100.0, 0.0, state="failed"),
        _task(2, 80.0, 1.0),
        _task(3, 70.0, 100.0),
    ]

    selected = select_best_successful_run(tasks, "chemcotbench_mol_edit")

    assert selected["selected_run_index"] == expected_run_index
    assert selected["replaces_primary_table"] is False
    assert selected["excluded_operationally_unavailable"][0]["run_index"] == 1


def test_final_test_log_reconstructs_peak_and_attempts(tmp_path: Path) -> None:
    expected_attempts = 2
    log = tmp_path / "final.log"
    lines = (
        "START gpu=0 main/a/run-1 final-test",
        "START gpu=1 main/b/run-1 final-test",
        "DONE  gpu=0 main/a/run-1 final-test",
        "FAIL  gpu=1 main/b/run-1 final-test",
        "Final tests: 0 reused, 2 evaluated",
    )
    log.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    parsed = parse_final_test_log(log)

    assert parsed["peak_parallelism"] == expected_attempts
    assert parsed["reused"] == 0
    assert parsed["evaluated"] == expected_attempts
    assert parsed["attempts"]["main/a/run-1"][0]["outcome"] == "done"
    assert parsed["attempts"]["main/b/run-1"][0]["outcome"] == "fail"


def test_final_test_log_accumulates_resume_without_duplicate_evaluation(tmp_path: Path) -> None:
    log = tmp_path / "final-resume.log"
    log.write_text(
        "START gpu=0 main/a/run-1 final-test\n"
        "DONE  gpu=0 main/a/run-1 final-test\n"
        "Final tests: 0 reused, 1 evaluated\n"
        "Final tests ready: 1 reused, 0 evaluated",
        encoding="utf-8",
    )

    parsed = parse_final_test_log(log)

    assert parsed["evaluated"] == 1
    assert parsed["reused"] == 1
    assert len(parsed["attempts"]["main/a/run-1"]) == 1
    assert len(parsed["invocations"]) == len(("evaluation", "reuse"))


def test_stable_api_route_excludes_only_ephemeral_adapter() -> None:
    route = {
        "protocol": "responses",
        "upstream_base": "http://127.0.0.1:8313/v1",
        "served_model": "gpt-5.6-sol",
        "adapter_base": "http://127.0.0.1:40843/v1",
    }

    assert stable_api_route(route) == {
        "protocol": "responses",
        "upstream_base": "http://127.0.0.1:8313/v1",
        "served_model": "gpt-5.6-sol",
    }


def test_selected_model_evidence_supports_full_and_adapter_artifacts(tmp_path: Path) -> None:
    full = tmp_path / "full"
    full.mkdir()
    (full / "config.json").write_text("{}\n", encoding="utf-8")
    (full / "model.safetensors").write_bytes(b"full")
    (full / "trainer_state.json").write_text('{"global_step": 1, "epoch": 1}\n', encoding="utf-8")

    full_evidence = selected_model_evidence(
        full,
        candidate_source="durable_checkpoint",
        training_policy="full",
    )

    assert full_evidence["training_method"] == "full"
    assert full_evidence["artifact_type"] == "full"
    assert full_evidence["adapter_config"] is None
    assert full_evidence["is_policy_compliant"] is True

    for method, use_rslora in (("lora", False), ("rslora", True)):
        adapter = tmp_path / method
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(method.encode())
        (adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "use_rslora": use_rslora,
                    "use_dora": False,
                },
            ),
            encoding="utf-8",
        )

        evidence = selected_model_evidence(
            adapter,
            candidate_source="durable_checkpoint",
            training_policy=method,
        )

        assert evidence["training_method"] == method
        assert evidence["adapter_config"] is not None
        assert evidence["is_policy_compliant"] is True


def test_baseline_selection_is_policy_compliant_without_trainer_state(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}\n", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"base")

    evidence = selected_model_evidence(
        baseline,
        candidate_source="baseline",
        training_policy="rslora",
    )

    assert evidence["training_method"] == "baseline"
    assert evidence["trainer_state"] is None
    assert evidence["is_policy_compliant"] is True
