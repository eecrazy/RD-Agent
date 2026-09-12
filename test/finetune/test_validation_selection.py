"""Regression tests for validation-only checkpoint selection."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import reproduction.ft_agent.run_validation_sweep as validation_sweep
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    enforce_formal_training_method_lock,
    make_formal_training_evidence,
)
from reproduction.ft_agent.final_test_protocol import selection_signature
from reproduction.ft_agent.run_validation_sweep import (
    CheckpointCandidate,
    SweepTarget,
    _archive_failed_attempt,
    _candidate_claim,
    _stage_model_snapshot,
    _stage_pending_candidates,
    candidate_set_signature,
    discover_baseline_candidates,
    discover_candidates,
    filter_candidates,
)
from reproduction.ft_agent.validation_selection import (
    LORA_COMPARISON_SELECTION_PROFILE,
    MAIN_SELECTION_PROFILE,
    VALIDATION_RANGE,
    ValidationSelectionError,
    final_test_file,
    make_selection_artifact,
    rank_candidates,
    validate_selection_artifact,
    validation_selection_file,
    validation_sweep_directory,
)

EXPERIMENT_ID = "main/aime25/run-1"
EXPECTED_SAMPLES = 2000


def test_baseline_discovery_and_validation_do_not_require_formal_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ft_root = tmp_path / "finetune_files"
    baseline = ft_root / "models" / "Qwen" / "example"
    baseline.mkdir(parents=True)
    (baseline / "config.json").write_text("{}\n", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"base-weights")
    monkeypatch.setattr(validation_sweep, "FT_ROOT", ft_root)
    target = SweepTarget(
        experiment_id="main/example/run-1",
        benchmark="example",
        model="Qwen/example",
        benchmark_dataset_path="pinned/example",
        task_root=tmp_path / "unfinished-task",
        expected_samples=EXPECTED_SAMPLES,
    )

    candidates = discover_baseline_candidates(target)

    assert len(candidates) == 1
    validation_sweep.validate_candidate_formal_provenance(target, candidates[0])
    assert not target.task_root.exists()


def test_baseline_staging_hardlinks_the_signed_pinned_model(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "config.json").write_text("{}\n", encoding="utf-8")
    (baseline / "model.safetensors").write_bytes(b"base-weights")
    signature = selection_signature("validation_sweep", None, None, baseline)

    staged = _stage_model_snapshot(
        tmp_path / "candidate-workspace",
        baseline,
        candidate_source="baseline",
        expected_signature=signature,
    )

    assert staged == tmp_path / "candidate-workspace" / "checkpoint_model"
    assert not staged.is_symlink()
    assert (staged / "model.safetensors").stat().st_ino == (baseline / "model.safetensors").stat().st_ino



def _metric(metric: str, value: float, *, higher: bool) -> dict[str, object]:
    return {
        "metric": metric,
        "label": metric,
        "value": value,
        "higher_is_better": higher,
        "unit": "percent",
        "formula": "synthetic",
    }


def _candidate(candidate_id: str, metrics: list[dict[str, object]]) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source": "checkpoint_archive",
        "workspace_id": "workspace",
        "checkpoint_step": int(candidate_id.removeprefix("step-")),
        "model_path": f"/synthetic/{candidate_id}",
        "selection_signature": f"signature-{candidate_id}",
        "paper_metrics": metrics,
    }


def _trainer_state(
    max_steps: int,
    *,
    global_step: int | None = None,
    train_batch_size: int = 2,
) -> dict[str, object]:
    completed_step = max_steps if global_step is None else global_step
    return {
        "max_steps": max_steps,
        "global_step": completed_step,
        "train_batch_size": train_batch_size,
        "num_train_epochs": 1,
        "log_history": [
            {
                "step": completed_step,
                "epoch": 1.0,
                "loss": float(max_steps) / 1000,
            },
        ],
    }


def _write_trainer_state(
    path: Path,
    max_steps: int,
    *,
    global_step: int | None = None,
    train_batch_size: int = 2,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps(
            _trainer_state(
                max_steps,
                global_step=global_step,
                train_batch_size=train_batch_size,
            ),
        ),
        encoding="utf-8",
    )


def _write_formal_session(
    task_root: Path,
    plans: dict[str, int],
    *,
    expected_samples: int = EXPECTED_SAMPLES,
    experiment_id: str = EXPERIMENT_ID,
    training_policy: str = "paper",
    training_method: str = "lora",
) -> None:
    """Write the real durable evidence contract used by production discovery."""
    if training_method not in {"full", "lora", "rslora"}:
        raise ValueError(training_method)
    method_lock = enforce_formal_training_method_lock(
        task_root / "formal_training_method.json",
        experiment_id=experiment_id,
        training_policy=training_policy,
        training_method=training_method,
    )
    for workspace_id, max_steps in plans.items():
        durable_root = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id
        output = durable_root / "output"
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"checkpoint-{max_steps}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        train_batch_size = 1 if training_method == "full" else 2
        if training_method == "full":
            (checkpoint / "model.safetensors").write_bytes(b"full-weights")
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            weight_name = "model.safetensors"
            config_name = "config.json"
        else:
            adapter = {
                "peft_type": "LORA",
                "use_rslora": training_method == "rslora",
                "use_dora": False,
            }
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-weights")
            (checkpoint / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
            weight_name = "adapter_model.safetensors"
            config_name = "adapter_config.json"
        for name in (weight_name, config_name):
            destination = output / name
            if not destination.exists():
                os.link(checkpoint / name, destination)
        _write_trainer_state(
            checkpoint,
            max_steps,
            train_batch_size=train_batch_size,
        )
        _write_trainer_state(
            output,
            max_steps,
            train_batch_size=train_batch_size,
        )

        workspace_path = task_root / "workspace" / workspace_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        data = [{"instruction": f"sample-{index}", "output": "ok"} for index in range(expected_samples)]
        (workspace_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
        (workspace_path / "dataset_info.json").write_text(
            json.dumps({"processed_data": {"file_name": "data.json"}}),
            encoding="utf-8",
        )
        (workspace_path / "data_stats.json").write_text(
            json.dumps({"total_samples": expected_samples}),
            encoding="utf-8",
        )
        method_yaml = (
            "finetuning_type: full\n"
            "deepspeed: /tmp/ds_z3_config.json\n"
            if training_method == "full"
            else (
                "finetuning_type: lora\n"
                f"use_rslora: {'true' if training_method == 'rslora' else 'false'}\n"
            )
        )
        runtime_markers = (
            "# rdagent_global_batch_size: 2\n# rdagent_world_size: 2\n"
            if training_method == "full"
            else ""
        )
        (workspace_path / "train.yaml").write_text(
            runtime_markers
            + "model_name_or_path: Qwen/Qwen2.5-7B-Instruct\n"
            "stage: sft\n"
            "do_train: true\n"
            + method_yaml
            + "dataset: processed_data\n"
            "dataset_dir: ./\n"
            "num_train_epochs: 1\n"
            f"per_device_train_batch_size: {train_batch_size}\n"
            "gradient_accumulation_steps: 1\n"
            "seed: 42\n"
            "data_seed: 42\n"
            "output_dir: ./output\n",
            encoding="utf-8",
        )
        evidence = make_formal_training_evidence(
            workspace_path,
            expected_samples=expected_samples,
            experiment_id=experiment_id,
            training_policy=training_policy,
            output_path=output,
        )
        (workspace_path / FORMAL_TRAINING_EVIDENCE_FILE).write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = durable_root / "formal_training"
        shutil.rmtree(provenance, ignore_errors=True)
        provenance.mkdir(parents=True)
        for name in (
            "train.yaml",
            "data.json",
            "dataset_info.json",
            "data_stats.json",
            FORMAL_TRAINING_EVIDENCE_FILE,
        ):
            shutil.copy2(workspace_path / name, provenance / name)

    (task_root / "status.json").write_text(
        json.dumps(
            {
                "state": "succeeded",
                "formal_training_method": training_method,
                "formal_training_method_lock": method_lock,
            },
        ),
        encoding="utf-8",
    )


def test_single_primary_metric_selects_highest_value() -> None:
    ranked = rank_candidates(
        "aime25",
        [
            _candidate("step-1", [_metric("accuracy", 25.0, higher=True)]),
            _candidate("step-2", [_metric("accuracy", 75.0, higher=True)]),
        ],
    )

    assert ranked[0]["candidate_id"] == "step-2"
    assert ranked[0]["mean_ordinal_rank"] == 1.0


def test_selection_profiles_use_independent_artifact_paths() -> None:
    assert validation_selection_file(MAIN_SELECTION_PROFILE) == "validation_selection.json"
    assert validation_selection_file(LORA_COMPARISON_SELECTION_PROFILE) == (
        "lora_comparison_validation_selection.json"
    )
    assert validation_sweep_directory(MAIN_SELECTION_PROFILE) != validation_sweep_directory(
        LORA_COMPARISON_SELECTION_PROFILE,
    )
    assert final_test_file(MAIN_SELECTION_PROFILE) != final_test_file(LORA_COMPARISON_SELECTION_PROFILE)


def test_lora_comparison_selection_requires_an_ordinary_lora_candidate(
    tmp_path: Path,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "task"
    formal_session_writer(task_root, {"workspace": 4}, training_method="lora")
    target = SweepTarget(
        experiment_id=EXPERIMENT_ID,
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
        selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
    )
    discovered = discover_candidates(target, include_baseline=False, include_final_outputs=False)
    assert len(discovered) == 1
    candidate = {
        **discovered[0].identity(),
        "accuracy_summary": {"synthetic": {"accuracy": 75.0}},
        "paper_metrics": [_metric("accuracy", 75.0, higher=True)],
    }
    artifact = make_selection_artifact(
        experiment_id=EXPERIMENT_ID,
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        search_status={"state": "succeeded"},
        candidate_set_signature=candidate_set_signature(discovered),
        candidates=[candidate],
        expected_samples=EXPECTED_SAMPLES,
        include_baseline=False,
        include_final_outputs=False,
        selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
    )
    assert validate_selection_artifact(
        artifact,
        experiment_id=EXPERIMENT_ID,
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
    )["candidate_id"] == discovered[0].candidate_id

    with pytest.raises(ValidationSelectionError, match="exclude the unchanged baseline"):
        make_selection_artifact(
            experiment_id=EXPERIMENT_ID,
            benchmark="aime25",
            model="Qwen/Qwen2.5-7B-Instruct",
            benchmark_dataset_path="pinned/aime25",
            search_status={"state": "succeeded"},
            candidate_set_signature=candidate_set_signature(discovered),
            candidates=[candidate],
            expected_samples=EXPECTED_SAMPLES,
            include_baseline=True,
            include_final_outputs=False,
            selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
        )


def test_lora_comparison_selection_rejects_full_sft_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "full-model"
    model_path.mkdir()
    (model_path / "model.safetensors").write_bytes(b"full-weights")
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    candidate = {
        "candidate_id": "full-step-4",
        "source": "durable_checkpoint",
        "workspace_id": "workspace",
        "checkpoint_step": 4,
        "model_path": str(model_path),
        "selection_signature": selection_signature("validation_sweep", None, 4, model_path),
        "formal_training_evidence_signature": "evidence",
        "accuracy_summary": {"synthetic": {"accuracy": 75.0}},
        "paper_metrics": [_metric("accuracy", 75.0, higher=True)],
    }

    with pytest.raises(ValidationSelectionError, match="non-LoRA candidate"):
        make_selection_artifact(
            experiment_id=EXPERIMENT_ID,
            benchmark="aime25",
            model="Qwen/Qwen2.5-7B-Instruct",
            benchmark_dataset_path="pinned/aime25",
            search_status={"state": "succeeded"},
            candidate_set_signature="candidate-set",
            candidates=[candidate],
            expected_samples=EXPECTED_SAMPLES,
            include_baseline=False,
            include_final_outputs=False,
            selection_profile=LORA_COMPARISON_SELECTION_PROFILE,
        )


def test_joint_metrics_use_direction_aware_mean_average_rank() -> None:
    ranked = rank_candidates(
        "chemcotbench_mol_und",
        [
            _candidate(
                "step-1",
                [
                    _metric("mae", 1.0, higher=False),
                    _metric("tanimoto_similarity", 0.5, higher=True),
                    _metric("accuracy", 80.0, higher=True),
                ],
            ),
            _candidate(
                "step-2",
                [
                    _metric("mae", 2.0, higher=False),
                    _metric("tanimoto_similarity", 0.8, higher=True),
                    _metric("accuracy", 90.0, higher=True),
                ],
            ),
        ],
    )

    assert ranked[0]["candidate_id"] == "step-2"
    assert ranked[0]["ordinal_ranks"] == {
        "mae": 2.0,
        "tanimoto_similarity": 1.0,
        "accuracy": 1.0,
    }


def test_signed_selection_requires_completed_search_and_detects_tampering(tmp_path: Path) -> None:
    model_path = tmp_path / "checkpoint-4"
    model_path.mkdir()
    (model_path / "adapter_model.safetensors").write_bytes(b"weights")
    candidate = {
        "candidate_id": "step-4",
        "source": "checkpoint_archive",
        "workspace_id": "workspace",
        "checkpoint_step": 4,
        "model_path": str(model_path),
        "selection_signature": selection_signature("validation_sweep", None, 4, model_path),
        "formal_training_evidence_signature": "formal-evidence",
        "paper_metrics": [_metric("accuracy", 75.0, higher=True)],
    }
    kwargs = {
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "benchmark_dataset_path": "pinned/aime25",
        "candidate_set_signature": "candidate-set",
        "candidates": [candidate],
        "expected_samples": EXPECTED_SAMPLES,
        "include_baseline": False,
        "include_final_outputs": False,
    }

    with pytest.raises(ValidationSelectionError, match="premature"):
        make_selection_artifact(search_status={"state": "running"}, **kwargs)

    artifact = make_selection_artifact(
        search_status={"state": "succeeded", "finished_at": "now"},
        **kwargs,
    )
    assert artifact["validation_range"] == VALIDATION_RANGE
    assert artifact["held_out_test_used"] is False
    selection = validate_selection_artifact(
        artifact,
        experiment_id=kwargs["experiment_id"],
        benchmark=kwargs["benchmark"],
        model=kwargs["model"],
    )
    assert selection["model_path"] == str(model_path)

    tampered = copy.deepcopy(artifact)
    tampered["selection"]["selection_metrics"][0]["value"] = 100.0
    with pytest.raises(ValidationSelectionError, match="artifact signature"):
        validate_selection_artifact(
            tampered,
            experiment_id=kwargs["experiment_id"],
            benchmark=kwargs["benchmark"],
            model=kwargs["model"],
        )


def test_stable_selection_content_ignores_only_reaudit_timestamps() -> None:
    original = {
        "created_at": "2026-09-10T15:20:00+08:00",
        "artifact_signature": "a" * 64,
        "search": {
            "state": "succeeded",
            "started_at": "2026-09-10T14:00:00+08:00",
            "finished_at": "2026-09-10T15:00:00+08:00",
        },
        "selection": {"candidate_id": "step-4", "selection_metrics": [{"value": 75.0}]},
    }
    resumed = copy.deepcopy(original)
    resumed["created_at"] = "2026-09-10T17:45:00+08:00"
    resumed["artifact_signature"] = "b" * 64
    resumed["search"]["started_at"] = "2026-09-10T16:00:00+08:00"
    resumed["search"]["finished_at"] = "2026-09-10T17:44:00+08:00"

    assert validation_sweep._stable_selection_artifact_content(  # noqa: SLF001
        original,
    ) == validation_sweep._stable_selection_artifact_content(resumed)  # noqa: SLF001

    resumed["selection"]["selection_metrics"][0]["value"] = 76.0
    assert validation_sweep._stable_selection_artifact_content(  # noqa: SLF001
        original,
    ) != validation_sweep._stable_selection_artifact_content(resumed)  # noqa: SLF001


def test_sweep_discovers_durable_intermediate_checkpoints_and_deduplicates_outputs(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "task"
    workspace_id = "candidate-a"
    checkpoint_step = 5
    durable_output = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id / "output"
    checkpoint = durable_output / f"checkpoint-{checkpoint_step}"
    checkpoint.mkdir(parents=True)
    checkpoint_weights = checkpoint / "adapter_model.safetensors"
    checkpoint_weights.write_bytes(b"step-five")
    adapter_config = json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False})
    (checkpoint / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
    _write_trainer_state(checkpoint, checkpoint_step)

    # LLaMA-Factory may hard-link the final output to the last checkpoint, and
    # FTWorkspace restores that same inode into the visible workspace.
    os.link(checkpoint_weights, durable_output / checkpoint_weights.name)
    (durable_output / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
    live_output = task_root / "workspace" / workspace_id / "output"
    live_output.mkdir(parents=True)
    os.link(checkpoint_weights, live_output / checkpoint_weights.name)
    (live_output / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
    _write_formal_session(task_root, {workspace_id: checkpoint_step})

    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    candidates = discover_candidates(target, include_baseline=False, include_final_outputs=True)

    assert len(candidates) == 1
    assert candidates[0].source == "durable_checkpoint"
    assert candidates[0].workspace_id == workspace_id
    assert candidates[0].checkpoint_step == checkpoint_step
    assert candidates[0].model_path == checkpoint.resolve()
    assert filter_candidates(candidates, re.compile(workspace_id)) == candidates
    assert filter_candidates(candidates, re.compile("candidate-b")) == []


@pytest.mark.parametrize(
    ("policy", "training_method"),
    [
        ("paper", "full"),
        ("paper", "lora"),
        ("full", "full"),
        ("lora", "lora"),
    ],
)
def test_sweep_accepts_only_the_task_locked_training_method(
    tmp_path: Path,
    policy: str,
    training_method: str,
) -> None:
    task_root = tmp_path / f"{policy}-{training_method}"
    workspace_id = f"{training_method}-workspace"
    _write_formal_session(
        task_root,
        {workspace_id: 1},
        training_policy=policy,
        training_method=training_method,
    )

    target = SweepTarget(
        experiment_id=EXPERIMENT_ID,
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
        training_policy=policy,
    )

    candidates = discover_candidates(target, include_baseline=False, include_final_outputs=False)

    assert [candidate.workspace_id for candidate in candidates] == [workspace_id]


def test_sweep_rejects_multiple_methods_inside_one_task(tmp_path: Path) -> None:
    task_root = tmp_path / "mixed-method-task"
    _write_formal_session(task_root, {"ordinary-lora": 1}, training_method="lora")
    second_root = tmp_path / "second-full-task"
    _write_formal_session(second_root, {"full": 1}, training_method="full")
    source = second_root / "workspace" / ".ft_model_checkpoints" / "full"
    destination = task_root / "workspace" / ".ft_model_checkpoints" / "full"
    shutil.copytree(source, destination)

    target = SweepTarget(
        experiment_id=EXPERIMENT_ID,
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
        training_policy="paper",
    )

    with pytest.raises(RuntimeError, match="disagree on the locked training method"):
        discover_candidates(target, include_baseline=False, include_final_outputs=False)


def test_sweep_filters_artifacts_by_selected_training_policy(tmp_path: Path) -> None:
    """A policy mismatch is rejected even when model files look usable."""
    task_root = tmp_path / "task"
    _write_formal_session(task_root, {"ordinary-lora": 1}, training_method="lora")
    for policy, error_match in (
        ("full", "no valid formal training evidence"),
        ("rslora", "requires a signed ordinary-LoRA pairing artifact"),
    ):
        target = SweepTarget(
            experiment_id="main/aime25/run-1",
            benchmark="aime25",
            model="Qwen/Qwen2.5-7B-Instruct",
            benchmark_dataset_path="pinned/aime25",
            task_root=task_root,
            expected_samples=EXPECTED_SAMPLES,
            training_policy=policy,
        )

        with pytest.raises(RuntimeError, match=error_match):
            discover_candidates(target, include_baseline=False, include_final_outputs=False)


def test_sweep_excludes_micro_batch_checkpoint_from_formal_workspace(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    workspace_id = "candidate-a"
    output = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id / "output"
    adapter_config = json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False})
    for step, max_steps in ((2, 2), (320, 320)):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(f"weights-{step}".encode())
        (checkpoint / "adapter_config.json").write_text(adapter_config, encoding="utf-8")
        _write_trainer_state(checkpoint, max_steps)
    _write_formal_session(task_root, {workspace_id: 320})
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )

    candidates = discover_candidates(target, include_baseline=False, include_final_outputs=False)

    assert [(candidate.checkpoint_step, candidate.model_path.name) for candidate in candidates] == [
        (320, "checkpoint-320"),
    ]


def test_sweep_excludes_checkpoint_without_formal_session_evidence(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    workspace_id = "candidate-a"
    output = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id / "output"
    checkpoint = output / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 10)
    _write_trainer_state(output, 10)
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )

    with pytest.raises(RuntimeError, match="no valid formal training evidence"):
        discover_candidates(target, include_baseline=False, include_final_outputs=False)


@pytest.mark.parametrize(
    "invalid_provenance",
    ["missing-data-stats", "zero-samples", "batch-mismatch", "step-mismatch"],
)
def test_sweep_rejects_formal_session_with_invalid_workspace_provenance(
    tmp_path: Path,
    invalid_provenance: str,
) -> None:
    task_root = tmp_path / "task"
    workspace_id = "candidate-a"
    checkpoint = (
        task_root / "workspace" / ".ft_model_checkpoints" / workspace_id / "output" / "checkpoint-10"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 10)
    _write_formal_session(task_root, {workspace_id: 10})
    provenance = (
        task_root
        / "workspace"
        / ".ft_model_checkpoints"
        / workspace_id
        / "formal_training"
    )
    if invalid_provenance == "missing-data-stats":
        (provenance / "data_stats.json").unlink()
    elif invalid_provenance == "zero-samples":
        (provenance / "data_stats.json").write_text('{"total_samples": 0}', encoding="utf-8")
    elif invalid_provenance == "batch-mismatch":
        (provenance / "train.yaml").write_text(
            "per_device_train_batch_size: 8\nmax_steps: 10\n",
            encoding="utf-8",
        )
    else:
        (provenance / "train.yaml").write_text(
            "per_device_train_batch_size: 2\nmax_steps: 11\n",
            encoding="utf-8",
        )
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )

    with pytest.raises(RuntimeError, match="no valid formal training evidence"):
        discover_candidates(target, include_baseline=False, include_final_outputs=False)


def test_failed_sweep_attempt_archives_partial_benchmark_cache(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    benchmark_results = candidate_root / "workspace" / "benchmark_results"
    partial_summary = benchmark_results / "validation" / "candidate" / "summary.csv"
    partial_summary.parent.mkdir(parents=True)
    partial_summary.write_text("dataset,metric\nadd,-\n", encoding="utf-8")
    result = {
        "state": "failed",
        "started_at": "2026-09-04T00:45:14+08:00",
        "error": "missing metric",
    }
    (candidate_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (candidate_root / "spec.json").write_text(json.dumps({"candidate_id": "candidate"}), encoding="utf-8")

    _archive_failed_attempt(candidate_root)

    assert not benchmark_results.exists()
    attempts = list((candidate_root / "failed_attempts").iterdir())
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "result.json").read_text(encoding="utf-8")) == result
    assert (attempts[0] / "benchmark_results" / "validation" / "candidate" / "summary.csv").is_file()


def test_sweep_stages_candidate_locally_before_mutable_source_disappears(tmp_path: Path) -> None:
    source = tmp_path / "durable" / "output" / "checkpoint-10"
    source.mkdir(parents=True)
    source_weights = source / "adapter_model.safetensors"
    source_weights.write_bytes(b"ordinary-lora-weights")
    (source / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    workspace = tmp_path / "candidate" / "workspace"

    snapshot = _stage_model_snapshot(workspace, source)

    assert snapshot == workspace / "checkpoint_model"
    assert snapshot.is_dir()
    assert not snapshot.is_symlink()
    assert (snapshot / source_weights.name).stat().st_ino != source_weights.stat().st_ino

    # A live output can be rewritten in place before FTWorkspace rotates it.
    # The candidate snapshot must therefore own independent file inodes.
    source_weights.write_bytes(b"newer-ordinary-lora-weights")
    assert (snapshot / source_weights.name).read_bytes() == b"ordinary-lora-weights"

    # FTWorkspace also replaces its durable output directory on every checkpoint.
    # The candidate-local files must remain usable after that replacement.
    shutil.rmtree(source.parents[1])
    assert (snapshot / source_weights.name).read_bytes() == b"ordinary-lora-weights"
    assert _stage_model_snapshot(workspace, source) == snapshot


def test_sweep_recovers_signed_snapshot_after_original_checkpoint_rotates(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    checkpoint = task_root / "workspace" / ".ft_model_checkpoints" / "workspace-a" / "output" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"validated-weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 5)
    _write_formal_session(task_root, {"workspace-a": 5})
    # The trainer can write the final trainer state separately from the last
    # checkpoint even when their JSON payloads are identical.  Make that
    # production ordering deterministic here: provenance recovery must bind
    # by content, not by an incidental small-file timestamp.
    checkpoint_state = checkpoint / "trainer_state.json"
    output_state = checkpoint.parent / "trainer_state.json"
    assert checkpoint_state.read_bytes() == output_state.read_bytes()
    output_mtime_ns = output_state.stat().st_mtime_ns
    checkpoint_mtime_ns = max(1, output_mtime_ns - 2_000_000_000)
    os.utime(checkpoint_state, ns=(checkpoint_mtime_ns, checkpoint_mtime_ns))
    assert checkpoint_state.stat().st_mtime_ns != output_state.stat().st_mtime_ns
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    original = discover_candidates(target, include_baseline=False, include_final_outputs=False)[0]
    candidate_root = task_root / "validation_sweep" / "candidates" / original.candidate_id
    snapshot = _stage_model_snapshot(candidate_root / "workspace", original.model_path)
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "result.json").write_text(
        json.dumps(
            {
                **original.identity(),
                "state": "succeeded",
                "experiment_id": target.experiment_id,
                    "benchmark": target.benchmark,
                    "model": target.model,
                    "training_policy": target.training_policy,
                    "expected_samples": target.expected_samples,
                    "validation_range": VALIDATION_RANGE,
                "paper_metrics": [_metric("accuracy", 75.0, higher=True)],
            },
        ),
        encoding="utf-8",
    )

    shutil.rmtree(checkpoint)
    recovered = discover_candidates(target, include_baseline=False, include_final_outputs=False)

    assert len(recovered) == 1
    assert recovered[0].candidate_id == original.candidate_id
    assert recovered[0].model_path == snapshot.resolve()
    assert recovered[0].selection_signature != original.selection_signature
    result = validation_sweep._matching_success(target, recovered[0])  # noqa: SLF001
    assert result is not None
    assert result["model_path"] == str(snapshot.resolve())
    assert result["selection_signature"] == recovered[0].selection_signature
    assert result["model_relocation"]["validated_model_path"] == str(original.model_path)
    assert result["model_relocation"]["validated_selection_signature"] == original.selection_signature


@pytest.mark.parametrize("exit_code", [127, 143])
def test_sweep_recovers_retryable_failed_snapshot_after_original_rotates(
    tmp_path: Path,
    exit_code: int,
) -> None:
    task_root = tmp_path / "task"
    checkpoint = task_root / "workspace" / ".ft_model_checkpoints" / "workspace-a" / "output" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"interrupted-weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 5)
    _write_formal_session(task_root, {"workspace-a": 5})
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    original = discover_candidates(target, include_baseline=False, include_final_outputs=False)[0]
    candidate_root = task_root / "validation_sweep" / "candidates" / original.candidate_id
    snapshot = _stage_model_snapshot(candidate_root / "workspace", original.model_path)
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "result.json").write_text(
        json.dumps(
            {
                **original.identity(),
                "state": "failed",
                "experiment_id": target.experiment_id,
                    "benchmark": target.benchmark,
                    "model": target.model,
                    "training_policy": target.training_policy,
                    "expected_samples": target.expected_samples,
                    "validation_range": VALIDATION_RANGE,
                "error": f"RuntimeError: Benchmark execution failed (exit_code={exit_code})",
            },
        ),
        encoding="utf-8",
    )

    shutil.rmtree(checkpoint)
    assert discover_candidates(target, include_baseline=False, include_final_outputs=False) == []

    recovered = discover_candidates(
        target,
        include_baseline=False,
        include_final_outputs=False,
        include_retryable_failed_snapshots=True,
    )

    assert len(recovered) == 1
    assert recovered[0].candidate_id == original.candidate_id
    assert recovered[0].model_path == snapshot.resolve()
    assert recovered[0].selection_signature != original.selection_signature
    assert validation_sweep._matching_success(target, recovered[0]) is None  # noqa: SLF001


def test_sweep_does_not_recover_deterministic_failed_snapshot(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    checkpoint = task_root / "workspace" / ".ft_model_checkpoints" / "workspace-a" / "output" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"invalid-metric-weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 5)
    _write_formal_session(task_root, {"workspace-a": 5})
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    original = discover_candidates(target, include_baseline=False, include_final_outputs=False)[0]
    candidate_root = task_root / "validation_sweep" / "candidates" / original.candidate_id
    _stage_model_snapshot(candidate_root / "workspace", original.model_path)
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "result.json").write_text(
        json.dumps(
            {
                **original.identity(),
                "state": "failed",
                "experiment_id": target.experiment_id,
                    "benchmark": target.benchmark,
                    "model": target.model,
                    "training_policy": target.training_policy,
                    "expected_samples": target.expected_samples,
                    "validation_range": VALIDATION_RANGE,
                "error": "RuntimeError: Validation result has no paper-primary metrics",
            },
        ),
        encoding="utf-8",
    )

    shutil.rmtree(checkpoint)

    assert (
        discover_candidates(
            target,
            include_baseline=False,
            include_final_outputs=False,
            include_retryable_failed_snapshots=True,
        )
        == []
    )


def test_sweep_rejects_snapshot_whose_files_changed_after_validation(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    checkpoint = task_root / "workspace" / ".ft_model_checkpoints" / "workspace-a" / "output" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"validated-weights")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    _write_trainer_state(checkpoint, 5)
    _write_formal_session(task_root, {"workspace-a": 5})
    target = SweepTarget(
        experiment_id="main/aime25/run-1",
        benchmark="aime25",
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_dataset_path="pinned/aime25",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    original = discover_candidates(target, include_baseline=False, include_final_outputs=False)[0]
    candidate_root = task_root / "validation_sweep" / "candidates" / original.candidate_id
    snapshot = _stage_model_snapshot(candidate_root / "workspace", original.model_path)
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "result.json").write_text(
        json.dumps(
            {
                **original.identity(),
                "state": "succeeded",
                "experiment_id": target.experiment_id,
                    "benchmark": target.benchmark,
                    "model": target.model,
                    "training_policy": target.training_policy,
                    "expected_samples": target.expected_samples,
                    "validation_range": VALIDATION_RANGE,
                "paper_metrics": [_metric("accuracy", 75.0, higher=True)],
            },
        ),
        encoding="utf-8",
    )

    shutil.rmtree(checkpoint)
    snapshot_weights = snapshot / "adapter_model.safetensors"
    original_stat = snapshot_weights.stat()
    assert len(b"tampered-weight") == original_stat.st_size
    snapshot_weights.write_bytes(b"tampered-weight")
    os.utime(
        snapshot_weights,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    assert discover_candidates(target, include_baseline=False, include_final_outputs=False) == []


def test_candidate_claim_prevents_overlapping_validation_workers(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"

    with _candidate_claim(candidate_root) as first_claim, _candidate_claim(candidate_root) as overlapping_claim:
        assert first_claim is True
        assert overlapping_claim is False

    with _candidate_claim(candidate_root) as later_claim:
        assert later_claim is True


def test_sweep_stages_all_pending_candidates_before_queue_wait(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    _write_formal_session(
        task_root,
        {"workspace-1": 1, "workspace-2": 2},
        experiment_id="main/example/run-1",
    )
    target = SweepTarget(
        experiment_id="main/example/run-1",
        benchmark="example",
        model="Qwen/example",
        benchmark_dataset_path="pinned/example",
        task_root=task_root,
        expected_samples=EXPECTED_SAMPLES,
    )
    candidates = discover_candidates(target, include_baseline=False, include_final_outputs=False)
    work = [(target, candidate) for candidate in candidates]

    assert _stage_pending_candidates(work) == work
    shutil.rmtree(task_root / "workspace" / ".ft_model_checkpoints")
    for _, candidate in work:
        snapshot = task_root / "validation_sweep" / "candidates" / candidate.candidate_id
        assert (snapshot / "workspace" / "checkpoint_model" / "adapter_model.safetensors").read_bytes() == (
            b"adapter-weights"
        )


def test_sweep_skips_candidate_that_aged_out_before_batch_staging(tmp_path: Path) -> None:
    target = SweepTarget(
        experiment_id="main/example/run-1",
        benchmark="example",
        model="Qwen/example",
        benchmark_dataset_path="pinned/example",
        task_root=tmp_path / "task",
        expected_samples=EXPECTED_SAMPLES,
    )
    candidate = CheckpointCandidate(
        candidate_id="durable_checkpoint__missing",
        source="durable_checkpoint",
        workspace_id="workspace",
        checkpoint_step=1,
        model_path=tmp_path / "missing-checkpoint",
        selection_signature="missing-signature",
        formal_training_evidence_signature="formal-evidence",
    )

    assert _stage_pending_candidates([(target, candidate)]) == []
    assert not (target.task_root / "validation_sweep" / "candidates" / candidate.candidate_id / "result.json").exists()


def test_run_one_rechecks_success_after_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_formal_session(
        tmp_path,
        {"workspace": 1},
        experiment_id="main/example/run-1",
    )
    target = SweepTarget(
        experiment_id="main/example/run-1",
        benchmark="example",
        model="Qwen/example",
        benchmark_dataset_path="pinned/example",
        task_root=tmp_path,
        expected_samples=EXPECTED_SAMPLES,
    )
    candidate = discover_candidates(target, include_baseline=False, include_final_outputs=False)[0]
    candidate_root = tmp_path / "validation_sweep" / "candidates" / candidate.candidate_id
    candidate_root.mkdir(parents=True)
    (candidate_root / "result.json").write_text(
        json.dumps(
            {
                **candidate.identity(),
                "state": "succeeded",
                "experiment_id": target.experiment_id,
                "benchmark": target.benchmark,
                "model": target.model,
                "training_policy": target.training_policy,
                "expected_samples": target.expected_samples,
                "validation_range": VALIDATION_RANGE,
            },
        ),
        encoding="utf-8",
    )

    async def unexpected_run(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError

    monkeypatch.setattr(validation_sweep, "_run_claimed_candidate", unexpected_run)

    assert asyncio.run(validation_sweep.run_one(target, candidate, "0", 1, {}, {})) is True


def test_claimed_candidate_worker_spec_includes_experiment_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SweepTarget(
        experiment_id="main/example/run-1",
        benchmark="example",
        model="Qwen/example",
        benchmark_dataset_path="pinned/example",
        task_root=tmp_path / "task",
        expected_samples=EXPECTED_SAMPLES,
    )
    candidate = CheckpointCandidate(
        candidate_id="baseline",
        source="baseline",
        workspace_id=None,
        checkpoint_step=None,
        model_path=tmp_path / "model",
        selection_signature="baseline-signature",
        formal_training_evidence_signature=None,
    )

    monkeypatch.setattr(validation_sweep, "validate_candidate_formal_provenance", lambda *_args: None)
    monkeypatch.setattr(
        validation_sweep,
        "_stage_model_snapshot",
        lambda *_args, **_kwargs: tmp_path / "staged-model",
    )

    class FailedProcess:
        returncode = 1

        async def wait(self) -> int:
            return self.returncode

    async def create_failed_process(*_args: object, **_kwargs: object) -> FailedProcess:
        return FailedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_failed_process)

    assert (
        asyncio.run(
            validation_sweep._run_claimed_candidate(  # noqa: SLF001
                target,
                candidate,
                "0",
                1,
                {},
                {},
            ),
        )
        is False
    )
    spec_path = target.task_root / "validation_sweep" / "candidates" / candidate.candidate_id / "spec.json"
    assert json.loads(spec_path.read_text(encoding="utf-8"))["experiment_id"] == target.experiment_id
