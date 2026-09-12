"""Shared fixtures for formal fine-tuning provenance tests."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    enforce_formal_training_method_lock,
    make_formal_training_evidence,
)


def _trainer_state(max_steps: int, train_batch_size: int) -> dict[str, Any]:
    return {
        "max_steps": max_steps,
        "global_step": max_steps,
        "train_batch_size": train_batch_size,
        "num_train_epochs": 1,
        "log_history": [{"step": max_steps, "epoch": 1.0, "loss": 0.1}],
    }


def _write_formal_session(
    task_root: Path,
    plans: dict[str, int],
    *,
    expected_samples: int = 2000,
    experiment_id: str = "main/aime25/run-1",
    training_policy: str = "paper",
    training_method: str = "lora",
) -> None:
    """Create the same durable 2k/full-epoch evidence consumed in production."""
    method_lock = enforce_formal_training_method_lock(
        task_root / "formal_training_method.json",
        experiment_id=experiment_id,
        training_policy=training_policy,
        training_method=training_method,
    )
    for workspace_id, max_steps in plans.items():
        durable_root = task_root / "workspace" / ".ft_model_checkpoints" / workspace_id
        output = durable_root / "output"
        checkpoint = output / f"checkpoint-{max_steps}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        train_batch_size = 1 if training_method == "full" else 2
        if training_method == "full":
            (checkpoint / "model.safetensors").write_bytes(b"full-weights")
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            artifact_names = ("model.safetensors", "config.json")
            method_yaml = "finetuning_type: full\ndeepspeed: /tmp/ds_z3_config.json\n"
            runtime_markers = "# rdagent_global_batch_size: 2\n# rdagent_world_size: 2\n"
        else:
            adapter = {
                "peft_type": "LORA",
                "use_rslora": training_method == "rslora",
                "use_dora": False,
            }
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-weights")
            (checkpoint / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
            artifact_names = ("adapter_model.safetensors", "adapter_config.json")
            method_yaml = (
                "finetuning_type: lora\n"
                f"use_rslora: {'true' if training_method == 'rslora' else 'false'}\n"
            )
            runtime_markers = ""
        for name in artifact_names:
            os.link(checkpoint / name, output / name)

        state = _trainer_state(max_steps, train_batch_size)
        (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
        (output / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

        workspace = task_root / "workspace" / workspace_id
        workspace.mkdir(parents=True, exist_ok=True)
        records = [{"instruction": f"sample-{index}", "output": "ok"} for index in range(expected_samples)]
        (workspace / "data.json").write_text(json.dumps(records), encoding="utf-8")
        (workspace / "dataset_info.json").write_text(
            json.dumps({"processed_data": {"file_name": "data.json"}}),
            encoding="utf-8",
        )
        (workspace / "data_stats.json").write_text(
            json.dumps({"total_samples": expected_samples}),
            encoding="utf-8",
        )
        (workspace / "train.yaml").write_text(
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
            workspace,
            expected_samples=expected_samples,
            experiment_id=experiment_id,
            training_policy=training_policy,
            output_path=output,
        )
        (workspace / FORMAL_TRAINING_EVIDENCE_FILE).write_text(
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
            shutil.copy2(workspace / name, provenance / name)

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


@pytest.fixture
def formal_session_writer() -> Callable[..., None]:
    return _write_formal_session
