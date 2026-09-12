"""Regression tests for durable FT model checkpointing."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from rdagent.scenarios.finetune.experiment.workspace import FTWorkspace
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    make_formal_training_evidence,
    validate_recorded_formal_training_evidence,
)


def test_ft_workspace_restores_large_selected_model(tmp_path: Path) -> None:
    workspace = FTWorkspace()
    workspace.workspace_path = tmp_path / "workspace" / "candidate"
    workspace.inject_files(**{"train.yaml": "output_dir: ./output\n"})

    output = workspace.workspace_path / "output"
    output.mkdir()
    weights = output / "adapter_model.safetensors"
    weights.write_bytes(b"model" * 30_000)  # Larger than the 100 KiB code-checkpoint limit.
    (output / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir()
    checkpoint_weights = checkpoint / "adapter_model.safetensors"
    checkpoint_weights.write_bytes(b"epoch" * 30_000)
    (checkpoint / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1}\n', encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer" * 30_000)

    workspace.create_ws_ckp()
    durable_weights = (
        workspace.workspace_path.parent
        / ".ft_model_checkpoints"
        / workspace.workspace_path.name
        / "output"
        / weights.name
    )
    assert durable_weights.is_file()
    assert durable_weights.stat().st_ino == weights.stat().st_ino

    shutil.rmtree(output)
    (workspace.workspace_path / "untracked.txt").write_text("remove me", encoding="utf-8")
    workspace.recover_ws_ckp()

    restored_weights = workspace.workspace_path / "output" / weights.name
    assert restored_weights.read_bytes() == b"model" * 30_000
    assert restored_weights.stat().st_ino == durable_weights.stat().st_ino
    assert (workspace.workspace_path / "output" / "adapter_config.json").is_file()
    restored_checkpoint = workspace.workspace_path / "output" / "checkpoint-1"
    restored_checkpoint_weights = restored_checkpoint / "adapter_model.safetensors"
    assert restored_checkpoint_weights.read_bytes() == b"epoch" * 30_000
    assert restored_checkpoint_weights.stat().st_ino == checkpoint_weights.stat().st_ino
    assert (restored_checkpoint / "adapter_config.json").is_file()
    assert (restored_checkpoint / "trainer_state.json").is_file()
    assert not (restored_checkpoint / "optimizer.pt").exists()
    assert not (workspace.workspace_path / "untracked.txt").exists()
    assert (workspace.workspace_path / "train.yaml").is_file()


def test_ft_workspace_drops_stale_formal_evidence_without_model(tmp_path: Path) -> None:
    """A derived code-only candidate must not claim its parent's formal run."""
    workspace = FTWorkspace()
    workspace.workspace_path = tmp_path / "workspace" / "candidate"
    workspace.inject_files(
        **{
            "train.yaml": "output_dir: ./output\n",
            "formal_training_evidence.json": '{"stale": true}\n',
        },
    )

    # Simulate durable state left by an earlier incarnation of the same
    # workspace id.  Neither it nor the visible evidence belongs to this
    # code-only candidate because the candidate has no model output.
    durable_root = (
        workspace.workspace_path.parent
        / ".ft_model_checkpoints"
        / workspace.workspace_path.name
    )
    (durable_root / "output").mkdir(parents=True)
    (durable_root / "output" / "adapter_model.safetensors").write_bytes(b"old-model")
    (durable_root / "formal_training").mkdir()
    (durable_root / "formal_training" / "formal_training_evidence.json").write_text(
        '{"stale": true}\n',
        encoding="utf-8",
    )

    workspace.create_ws_ckp()

    assert workspace._model_checkpoint_path is None
    assert workspace._formal_training_checkpoint_path is None
    assert not (workspace.workspace_path / "formal_training_evidence.json").exists()
    assert not durable_root.exists()

    workspace.recover_ws_ckp()
    assert not (workspace.workspace_path / "formal_training_evidence.json").exists()


def test_ft_workspace_restores_valid_formal_training_provenance(tmp_path: Path) -> None:
    workspace = FTWorkspace()
    workspace.workspace_path = tmp_path / "workspace" / "trained"
    workspace.prepare()
    records = [
        {"instruction": f"sample-{index}", "input": "", "output": "ok"}
        for index in range(2)
    ]
    (workspace.workspace_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    (workspace.workspace_path / "dataset_info.json").write_text(
        json.dumps({"processed_data": {"file_name": "data.json"}}),
        encoding="utf-8",
    )
    (workspace.workspace_path / "data_stats.json").write_text(
        json.dumps({"total_samples": 2}),
        encoding="utf-8",
    )
    (workspace.workspace_path / "train.yaml").write_text(
        "model_name_or_path: Qwen/Qwen2.5-7B-Instruct\n"
        "stage: sft\n"
        "do_train: true\n"
        "finetuning_type: lora\n"
        "use_rslora: false\n"
        "use_dora: false\n"
        "dataset: processed_data\n"
        "dataset_dir: ./\n"
        "num_train_epochs: 1\n"
        "per_device_train_batch_size: 1\n"
        "gradient_accumulation_steps: 1\n"
        "seed: 42\n"
        "data_seed: 42\n"
        "output_dir: ./output\n",
        encoding="utf-8",
    )

    output = workspace.workspace_path / "output"
    checkpoint = output / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    adapter_config = {"peft_type": "LORA", "use_rslora": False, "use_dora": False}
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (checkpoint / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    os.link(checkpoint / "adapter_model.safetensors", output / "adapter_model.safetensors")
    os.link(checkpoint / "adapter_config.json", output / "adapter_config.json")
    trainer_state = {
        "max_steps": 2,
        "global_step": 2,
        "train_batch_size": 1,
        "num_train_epochs": 1,
        "log_history": [{"step": 2, "epoch": 1.0, "loss": 0.1}],
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    (output / "trainer_state.json").write_text(json.dumps(trainer_state), encoding="utf-8")
    evidence = make_formal_training_evidence(
        workspace.workspace_path,
        expected_samples=2,
        experiment_id="main/test/run-1",
        training_policy="paper",
        output_path=output,
    )
    (workspace.workspace_path / FORMAL_TRAINING_EVIDENCE_FILE).write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    workspace.create_ws_ckp()
    durable_root = workspace.workspace_path.parent / ".ft_model_checkpoints" / "trained"
    assert (durable_root / "output" / "adapter_model.safetensors").is_file()
    assert (durable_root / "formal_training" / FORMAL_TRAINING_EVIDENCE_FILE).is_file()

    shutil.rmtree(workspace.workspace_path)
    workspace.recover_ws_ckp()

    restored = validate_recorded_formal_training_evidence(
        workspace.workspace_path,
        output_path=workspace.workspace_path / "output",
    )
    assert restored["training_sample_count"] == 2
    assert restored["training_method"] == "lora"
