# ruff: noqa: PLR2004
"""Regression tests for the exact formal-training sample contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdagent.scenarios.finetune.train.formal_training import (
    FormalTrainingEvidenceError,
    validate_formal_training_inputs,
)
from rdagent.utils.agent import tpl as tpl_module
from rdagent.utils.agent.tpl import T

EXPECTED_SAMPLES = 2000
EXPERIMENT_ID = "main/chemcotbench_reaction/run-1"


def _write_formal_workspace(path: Path, *, extra_yaml: str = "") -> list[dict[str, str]]:
    records = [
        {"instruction": f"training-{index}", "input": "", "output": "ok"}
        for index in range(EXPECTED_SAMPLES)
    ]
    (path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    (path / "dataset_info.json").write_text(
        json.dumps({"processed_data": {"file_name": "data.json"}}),
        encoding="utf-8",
    )
    (path / "data_stats.json").write_text(
        json.dumps({"total_samples": EXPECTED_SAMPLES}),
        encoding="utf-8",
    )
    (path / "train.yaml").write_text(
        "model_name_or_path: Qwen/Qwen2.5-7B-Instruct\n"
        "stage: sft\n"
        "do_train: true\n"
        "finetuning_type: lora\n"
        "use_rslora: false\n"
        "use_dora: false\n"
        "dataset: processed_data\n"
        "dataset_dir: ./\n"
        "num_train_epochs: 1\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 1\n"
        "seed: 42\n"
        "data_seed: 42\n"
        "output_dir: ./output\n"
        + extra_yaml,
        encoding="utf-8",
    )
    return records


def _validate(path: Path) -> dict:
    return validate_formal_training_inputs(
        path,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id=EXPERIMENT_ID,
        training_policy="paper",
        require_runtime_contract=False,
    )


def test_formal_training_rejects_val_split_from_exact_2000(tmp_path: Path) -> None:
    _write_formal_workspace(tmp_path, extra_yaml="val_size: 0.1\n")

    with pytest.raises(FormalTrainingEvidenceError, match="positive val_size"):
        _validate(tmp_path)


def test_formal_training_accepts_additional_disjoint_validation_data(tmp_path: Path) -> None:
    _write_formal_workspace(
        tmp_path,
        extra_yaml=(
            "val_size: 0\n"
            "eval_dataset: processed_data_validation\n"
            "do_eval: true\n"
            "eval_strategy: epoch\n"
            "save_strategy: epoch\n"
            "load_best_model_at_end: true\n"
        ),
    )
    validation = [
        {"instruction": f"validation-{index}", "input": "", "output": "ok"}
        for index in range(20)
    ]
    (tmp_path / "validation.json").write_text(json.dumps(validation), encoding="utf-8")
    (tmp_path / "dataset_info.json").write_text(
        json.dumps(
            {
                "processed_data": {"file_name": "data.json"},
                "processed_data_validation": {"file_name": "validation.json"},
            },
        ),
        encoding="utf-8",
    )

    facts = _validate(tmp_path)

    assert facts["training_sample_count"] == EXPECTED_SAMPLES
    assert facts["validation"]["source"] == "independent_eval_dataset"
    assert facts["validation"]["sample_count"] == 20


def test_formal_training_rejects_overlapping_validation_records(tmp_path: Path) -> None:
    training = _write_formal_workspace(
        tmp_path,
        extra_yaml=(
            "val_size: 0\n"
            "eval_dataset: processed_data_validation\n"
            "do_eval: true\n"
            "eval_strategy: epoch\n"
        ),
    )
    (tmp_path / "validation.json").write_text(json.dumps([training[0]]), encoding="utf-8")
    (tmp_path / "dataset_info.json").write_text(
        json.dumps(
            {
                "processed_data": {"file_name": "data.json"},
                "processed_data_validation": {"file_name": "validation.json"},
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(FormalTrainingEvidenceError, match="overlaps the training data"):
        _validate(tmp_path)


def test_formal_training_rejects_evaluation_without_independent_dataset(tmp_path: Path) -> None:
    _write_formal_workspace(
        tmp_path,
        extra_yaml="val_size: 0\ndo_eval: true\neval_strategy: epoch\n",
    )

    with pytest.raises(FormalTrainingEvidenceError, match="no independent eval_dataset"):
        _validate(tmp_path)


def test_formal_prompts_describe_2000_as_exact_not_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tpl_module.logger, "log_object", lambda *_args, **_kwargs: None)
    data_prompt = T("rdagent.components.coder.finetune.prompts:data_coder.system").r(
        scenario="formal scenario",
        task_desc="prepare data",
        dataset_info="dataset",
        queried_former_failed_knowledge=[],
        api_max_workers=1,
        datasets_path="/datasets/",
        workspace_path="./",
        force_think_token=False,
        formal_expected_samples=EXPECTED_SAMPLES,
    )
    training_prompt = T("rdagent.components.coder.finetune.prompts:finetune_coder.system").r(
        scenario="formal scenario",
        task_desc="train",
        queried_former_failed_knowledge=[],
        available_methods="full, lora",
        shared_params="",
        methods_specific_params={},
        training_policy="paper",
        training_policy_guidance="choose Full SFT or ordinary LoRA",
        formal_expected_samples=EXPECTED_SAMPLES,
    )

    assert "exact count, not an upper bound" in data_prompt
    assert "estimated rejection allowance" in data_prompt
    assert "from actual outcomes" in data_prompt
    assert "reserved validation identity for a training variant" in data_prompt
    assert "Never carve it out" in data_prompt
    assert "Set `val_size: 0`" in training_prompt
    assert "positive `val_size` is rejected" in training_prompt
