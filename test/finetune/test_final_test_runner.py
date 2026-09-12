"""Regression tests for one-shot held-out test scheduling."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reproduction.ft_agent.final_test_protocol import selection_signature, trace_protocol_pollution
from reproduction.ft_agent.run_final_test import (
    FinalTestTarget,
    _artifact_payload,
    persist_legacy_result,
    prepare_targets,
    target_from_trace,
    target_from_validation_selection,
)
from reproduction.ft_agent.run_validation_sweep import (
    SweepTarget,
    candidate_set_signature,
    discover_candidates,
)
from reproduction.ft_agent.validation_selection import (
    LORA_COMPARISON_SELECTION_PROFILE,
    final_test_file,
    make_selection_artifact,
)

SELECTED_LOOP_ID = 8
SELECTED_CHECKPOINT_STEP = 4


def _score(value: float) -> dict[str, Any]:
    return {"accuracy_summary": {"synthetic": {"accuracy": value}}}


def _target(tmp_path: Path, *, legacy: bool = False) -> FinalTestTarget:
    task_root = tmp_path / "task"
    model_path = task_root / "workspace" / "output"
    model_path.mkdir(parents=True)
    (model_path / "adapter_model.safetensors").write_bytes(b"weights")
    signature = selection_signature("accepted_loop", 2, SELECTED_LOOP_ID, model_path)
    return FinalTestTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        task_root=task_root,
        model_path=model_path,
        selection_source="accepted_loop",
        history_index=2,
        loop_id=SELECTED_LOOP_ID,
        selection_signature=signature,
        benchmark_dataset_path="pinned/aime25",
        expected_samples=2000,
        formal_training_evidence_signature=None,
        legacy_result=_score(50.0) if legacy else None,
        protocol_pollution=("hist[2].benchmark_test",) if legacy else (),
    )


def _write_artifact(target: FinalTestTarget, *, state: str, signature: str | None = None) -> None:
    target.task_root.mkdir(parents=True, exist_ok=True)
    (target.task_root / "final_test.json").write_text(
        json.dumps(
            {
                "state": state,
                "evaluation_mode": "post_selection",
                "selection": {
                    **target.selection(),
                    "signature": signature or target.selection_signature,
                },
            },
        ),
        encoding="utf-8",
    )


def test_matching_success_is_reused_only_with_resume(tmp_path: Path) -> None:
    target = _target(tmp_path)
    _write_artifact(target, state="succeeded")

    with pytest.raises(RuntimeError, match="already succeeded"):
        prepare_targets([target], resume=False, allow_retest=False, audit_reuse_legacy=False)

    pending, reused = prepare_targets(
        [target],
        resume=True,
        allow_retest=False,
        audit_reuse_legacy=False,
    )
    assert pending == []
    assert reused == [target]


def test_failed_attempt_requires_explicit_retest_authorization(tmp_path: Path) -> None:
    target = _target(tmp_path)
    _write_artifact(target, state="failed")

    with pytest.raises(RuntimeError, match="already attempted"):
        prepare_targets([target], resume=True, allow_retest=False, audit_reuse_legacy=False)

    pending, reused = prepare_targets(
        [target],
        resume=True,
        allow_retest=True,
        audit_reuse_legacy=False,
    )
    assert pending == [target]
    assert reused == []


def test_same_checkpoint_is_reused_across_selection_profiles(tmp_path: Path) -> None:
    main_target = _target(tmp_path)
    main_payload = _artifact_payload(
        main_target,
        state="succeeded",
        evaluation_mode="post_selection",
    )
    main_payload.update(
        {
            "started_at": "2026-09-08T00:00:00+00:00",
            "finished_at": "2026-09-08T00:01:00+00:00",
            "result": _score(77.0),
            "workspace": str(tmp_path / "main-final-test-workspace"),
        },
    )
    (main_target.task_root / final_test_file("main")).write_text(
        json.dumps(main_payload),
        encoding="utf-8",
    )
    comparison_target = replace(
        main_target,
        selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
        selection_artifact_signature="comparison-selection-signature",
    )

    pending, reused = prepare_targets(
        [comparison_target],
        resume=False,
        allow_retest=False,
        audit_reuse_legacy=False,
    )

    assert pending == []
    assert reused == [comparison_target]
    comparison_path = comparison_target.task_root / final_test_file(LORA_COMPARISON_SELECTION_PROFILE)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["result"] == _score(77.0)
    assert comparison["selection"] == comparison_target.selection()
    assert comparison["reused_from"]["selection_profile"] == "main"


def test_legacy_result_requires_explicit_audit_and_is_never_scheduled(tmp_path: Path) -> None:
    target = _target(tmp_path, legacy=True)

    with pytest.raises(RuntimeError, match="protocol-polluted"):
        prepare_targets([target], resume=False, allow_retest=False, audit_reuse_legacy=False)

    pending, reused = prepare_targets(
        [target],
        resume=False,
        allow_retest=False,
        audit_reuse_legacy=True,
    )
    assert pending == []
    assert reused == [target]

    persist_legacy_result(target)
    artifact = json.loads((target.task_root / "final_test.json").read_text(encoding="utf-8"))
    assert artifact["state"] == "succeeded"
    assert artifact["evaluation_mode"] == "legacy_same_node_reuse"
    assert artifact["selection"]["signature"] == target.selection_signature
    assert artifact["result"] == _score(50.0)


def test_pollution_detection_scans_unselected_history_nodes() -> None:
    clean_workspace = SimpleNamespace(running_info=SimpleNamespace(result={"benchmark": _score(10.0)}))
    polluted_workspace = SimpleNamespace(
        running_info=SimpleNamespace(
            result={"benchmark": _score(20.0), "benchmark_test": _score(99.0)},
        ),
    )
    trace = SimpleNamespace(
        scen=SimpleNamespace(baseline_benchmark_score_test={}),
        hist=[
            (SimpleNamespace(experiment_workspace=clean_workspace), SimpleNamespace(decision=True)),
            (SimpleNamespace(experiment_workspace=polluted_workspace), SimpleNamespace(decision=False)),
        ],
    )

    assert trace_protocol_pollution(trace) == ("hist[1].benchmark_test",)


def test_target_from_trace_rejects_search_time_held_out_payload(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    workspace = task_root / "workspace" / "selected"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "adapter_model.safetensors").write_bytes(b"weights")
    experiment = SimpleNamespace(
        experiment_workspace=SimpleNamespace(
            workspace_path=workspace,
            running_info=SimpleNamespace(
                result={"benchmark": _score(40.0), "benchmark_test": _score(41.0)},
            ),
        ),
    )
    trace = SimpleNamespace(
        hist=[(experiment, SimpleNamespace(decision=True))],
        dag_parent=[(0,)],
        current_selection=(-1,),
        idx2loop_id={0: SELECTED_LOOP_ID},
        scen=SimpleNamespace(),
    )
    session = task_root / "trace" / "__session__" / "8" / "1_session.pkl"
    session.parent.mkdir(parents=True)
    with session.open("wb") as stream:
        pickle.dump(SimpleNamespace(trace=trace), stream)
    status = {
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "benchmark_dataset_path": "pinned/aime25",
        "data_limit": 2000,
        "formal_expected_samples": 2000,
        "training_policy": "paper",
    }

    with pytest.raises(RuntimeError, match=r"Protocol-polluted.*hist\[0\]\.benchmark_test"):
        target_from_trace(task_root, status)

    target = target_from_trace(task_root, status, allow_polluted_legacy_audit=True)

    assert target.history_index == 0
    assert target.loop_id == SELECTED_LOOP_ID
    assert target.model_path == output.resolve()
    assert target.legacy_result == _score(41.0)
    assert target.protocol_pollution == ("hist[0].benchmark_test",)


def test_clean_final_test_requires_signed_validation_selection(
    tmp_path: Path,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "task"
    trace = SimpleNamespace(hist=[], scen=SimpleNamespace(baseline_benchmark_score_test={}))
    session = task_root / "trace" / "__session__" / "0" / "1_session.pkl"
    session.parent.mkdir(parents=True)
    with session.open("wb") as stream:
        pickle.dump(SimpleNamespace(trace=trace), stream)
    status = {
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "benchmark_dataset_path": "pinned/aime25",
        "state": "succeeded",
        "data_limit": 2000,
        "formal_expected_samples": 2000,
        "training_policy": "paper",
    }

    with pytest.raises(RuntimeError, match="artifact is missing"):
        target_from_validation_selection(task_root, status)

    formal_session_writer(task_root, {"workspace": SELECTED_CHECKPOINT_STEP})
    sweep_target = SweepTarget(
        experiment_id=status["experiment_id"],
        benchmark=status["benchmark"],
        model=status["model"],
        benchmark_dataset_path=status["benchmark_dataset_path"],
        task_root=task_root,
        expected_samples=2000,
        training_policy="paper",
    )
    discovered = discover_candidates(
        sweep_target,
        include_baseline=False,
        include_final_outputs=False,
    )
    assert len(discovered) == 1
    discovered_candidate = discovered[0]
    candidate = {
        **discovered_candidate.identity(),
        "accuracy_summary": {"synthetic": {"accuracy": 75.0}},
        "paper_metrics": [
            {
                "metric": "accuracy",
                "label": "accuracy",
                "value": 75.0,
                "higher_is_better": True,
                "unit": "percent",
                "formula": "synthetic",
            },
        ],
    }
    artifact = make_selection_artifact(
        experiment_id=status["experiment_id"],
        benchmark=status["benchmark"],
        model=status["model"],
        benchmark_dataset_path=status["benchmark_dataset_path"],
        search_status={"state": "succeeded"},
        candidate_set_signature=candidate_set_signature(discovered),
        candidates=[candidate],
        expected_samples=2000,
        include_baseline=False,
        include_final_outputs=False,
    )
    selection_path = task_root / "validation_selection.json"
    selection_path.write_text(json.dumps(artifact), encoding="utf-8")

    target = target_from_validation_selection(task_root, status)

    assert target.selection_source == "validation_sweep"
    assert target.candidate_id == discovered_candidate.candidate_id
    assert target.checkpoint_step == SELECTED_CHECKPOINT_STEP
    assert target.selection_artifact_signature == artifact["artifact_signature"]
    assert target.selection()["signature"] == discovered_candidate.selection_signature

    artifact["selection"]["selection_metrics"][0]["value"] = 99.0
    selection_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact signature"):
        target_from_validation_selection(task_root, status)
