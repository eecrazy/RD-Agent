"""Regression tests for the controlled Full/LoRA/rsLoRA boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rdagent.components.coder.finetune import eval as coder_eval_module
from rdagent.components.coder.finetune.conf import FT_YAML_FILE_NAME
from rdagent.components.coder.finetune.eval import FTCoderEvaluator
from rdagent.components.coder.finetune.unified_validator import (
    LLMConfigValidator,
    normalize_rs_lora_config,
    normalize_training_config,
    training_policy_guidance,
    validate_rs_lora_policy,
    validate_training_policy,
)
from rdagent.scenarios.finetune.proposal.proposal import proposal_training_methods
from rdagent.scenarios.finetune.train import eval as runner_eval_module
from rdagent.scenarios.finetune.train.eval import FTRunnerEvaluator

VALID_CONFIG = """\
model_name_or_path: local-model
finetuning_type: lora
use_rslora: true
use_dora: false
"""


def test_training_guidance_uses_single_b200_logical_batch_contract() -> None:
    guidance = training_policy_guidance("full")

    assert "logical single-B200 contract" in guidance
    assert (
        "logical_global_batch = per_device_train_batch_size * gradient_accumulation_steps"
        in guidance
    )
    assert "Do not multiply or divide either field by the H20 world size" in guidance
    assert (
        "runtime_per_device_batch * runtime_accumulation * world_size = logical_global_batch"
        in guidance
    )
    assert "QLoRA/on-the-fly quantization" in guidance


def test_prompt_does_not_pre_scale_paper_batch_for_h20_world_size() -> None:
    prompt_source = (Path(coder_eval_module.__file__).with_name("prompts.yaml")).read_text(
        encoding="utf-8",
    )

    assert "logical_global_batch = batch_size * gradient_accumulation_steps" in prompt_source
    assert "do not include H20 num_gpus" in prompt_source
    multiplication_sign = "\u00d7"
    assert (
        f"batch_size {multiplication_sign} gradient_accumulation_steps "
        f"{multiplication_sign} num_gpus"
        not in prompt_source
    )
    assert (
        f"effective_batch = batch {multiplication_sign} accum {multiplication_sign} gpus"
        not in prompt_source
    )


def test_legacy_rs_lora_policy_wrapper_accepts_required_adapter() -> None:
    assert validate_rs_lora_policy(VALID_CONFIG) == []


def test_legacy_rs_lora_normalizer_remains_compatible() -> None:
    normalized = normalize_rs_lora_config(
        "finetuning_type: full\nlearning_rate: 1.0e-5\nuse_dora: true\n",
    )

    assert validate_rs_lora_policy(normalized) == []
    assert "finetuning_type: lora" in normalized
    assert "use_rslora: true" in normalized
    assert "use_dora: false" in normalized
    assert "lora_target: all" in normalized
    assert "learning_rate: 1.0e-05" in normalized


@pytest.mark.parametrize(
    ("policy", "config"),
    [
        ("paper", "finetuning_type: full\n"),
        ("paper", "finetuning_type: lora\nuse_rslora: false\n"),
        ("full", "finetuning_type: full\n"),
        ("lora", "finetuning_type: lora\nuse_rslora: false\n"),
        ("rslora", "finetuning_type: lora\nuse_rslora: true\n"),
    ],
)
def test_training_policy_accepts_its_supported_methods(policy: str, config: str) -> None:
    assert validate_training_policy(config, policy) == []


def test_paper_policy_rejects_rs_lora_as_a_separate_comparison_method() -> None:
    errors = validate_training_policy(
        "finetuning_type: lora\nuse_rslora: true\nuse_dora: false\n",
        "paper",
    )

    assert errors == [
        "training method must be 'full' or ordinary 'lora' under the 'paper' policy "
        "(got 'rslora')",
    ]
    assert "rsLoRA is excluded from the paper policy" in training_policy_guidance("paper")


@pytest.mark.parametrize(
    ("policy", "config", "expected_method"),
    [
        ("full", "finetuning_type: lora\nuse_rslora: false\n", "full"),
        ("full", "finetuning_type: lora\nuse_rslora: true\n", "full"),
        ("lora", "finetuning_type: full\n", "lora"),
        ("lora", "finetuning_type: lora\nuse_rslora: true\n", "lora"),
        ("rslora", "finetuning_type: full\n", "rslora"),
        ("rslora", "finetuning_type: lora\nuse_rslora: false\n", "rslora"),
    ],
)
def test_controlled_training_policy_rejects_other_methods(
    policy: str,
    config: str,
    expected_method: str,
) -> None:
    assert any(
        f"training method must be '{expected_method}'" in error
        for error in validate_training_policy(config, policy)
    )


@pytest.mark.parametrize("policy", ["paper", "full", "lora", "rslora"])
def test_all_training_policies_reject_dora(policy: str) -> None:
    errors = validate_training_policy(
        "finetuning_type: lora\nuse_rslora: false\nuse_dora: true\n",
        policy,
    )

    assert "use_dora must be false or omitted" in errors


@pytest.mark.parametrize("policy", ["paper", "full", "lora", "rslora"])
def test_all_training_policies_reject_quantized_training(policy: str) -> None:
    errors = validate_training_policy(
        "finetuning_type: lora\nuse_rslora: false\nquantization_bit: 4\n",
        policy,
    )

    assert any("QLoRA/quantized training is not allowed" in error for error in errors)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("paper", ["full", "lora"]),
        ("full", ["full"]),
        ("lora", ["lora"]),
        ("rslora", ["lora"]),
    ],
)
def test_proposal_exposes_only_policy_methods(policy: str, expected: list[str]) -> None:
    assert proposal_training_methods(policy, ["full", "freeze", "lora"]) == expected


@pytest.mark.parametrize(
    ("policy", "expected_type", "expected_rslora"),
    [
        ("full", "full", None),
        ("lora", "lora", False),
        ("rslora", "lora", True),
    ],
)
def test_controlled_policy_normalizes_only_the_training_method(
    policy: str,
    expected_type: str,
    expected_rslora: bool | None,  # noqa: FBT001 - parametrized test case value
) -> None:
    normalized = normalize_training_config(
        "finetuning_type: full\nlearning_rate: 2.0e-5\nuse_dora: true\n",
        policy,
    )

    assert f"finetuning_type: {expected_type}" in normalized
    assert "learning_rate: 2.0e-05" in normalized
    if expected_rslora is None:
        assert "use_rslora:" not in normalized
        assert "use_dora:" not in normalized
    else:
        assert f"use_rslora: {str(expected_rslora).lower()}" in normalized
        assert "use_dora: false" in normalized


def test_config_validator_rejects_policy_violation_without_micro_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT_TRAINING_POLICY", "lora")
    validator = LLMConfigValidator()
    monkeypatch.setattr(
        validator,
        "_run_micro_batch_test",
        lambda *_args, **_kwargs: pytest.fail("micro-batch must not run"),
    )

    result = validator.validate_and_test(
        "finetuning_type: lora\nuse_rslora: true\n",
        workspace=SimpleNamespace(),
        env=object(),
    )

    assert result.success is False
    assert "training method must be 'lora'" in result.execution_output


def test_full_training_boundary_checks_selected_policy_before_environment_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT_TRAINING_POLICY", "full")
    workspace = SimpleNamespace(
        file_dict={
            FT_YAML_FILE_NAME: "finetuning_type: lora\nuse_rslora: false\nuse_dora: false\n",
        },
        feedback=None,
    )
    monkeypatch.setattr(
        runner_eval_module,
        "get_ft_env",
        lambda **_kwargs: pytest.fail("training environment must not be created"),
    )
    monkeypatch.setattr(runner_eval_module.logger, "log_object", lambda *_args, **_kwargs: None)

    feedback = FTRunnerEvaluator(scen=SimpleNamespace(gpu_count=1)).evaluate(
        SimpleNamespace(),
        workspace,
        gt_implementation=workspace,
    )

    assert feedback.final_decision is False
    assert "training method must be 'full'" in feedback.execution
    assert workspace.feedback is feedback


def test_coder_policy_feedback_reports_selected_method_without_environment_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT_TRAINING_POLICY", "full")
    workspace = SimpleNamespace(
        file_dict={FT_YAML_FILE_NAME: "finetuning_type: lora\nuse_rslora: false\n"},
        feedback=None,
    )
    task = SimpleNamespace(get_task_information=lambda: "Full tuning is required; do not use LoRA")
    monkeypatch.setattr(
        coder_eval_module,
        "get_ft_env",
        lambda **_kwargs: pytest.fail("micro-batch environment must not be created"),
    )
    monkeypatch.setattr(
        coder_eval_module,
        "build_cls_from_json_with_retry",
        lambda *_args, **_kwargs: pytest.fail("policy failure must not be reinterpreted by an LLM"),
    )
    monkeypatch.setattr(coder_eval_module.logger, "log_object", lambda *_args, **_kwargs: None)

    feedback = FTCoderEvaluator(scen=SimpleNamespace()).evaluate(
        task,
        workspace,
        gt_implementation=workspace,
    )

    assert feedback.final_decision is False
    assert "active 'full' training policy" in feedback.return_checking
    assert "use full-parameter SFT" in feedback.code
    assert "H20 multi-GPU ZeRO-3" in feedback.code
    assert workspace.feedback is feedback
