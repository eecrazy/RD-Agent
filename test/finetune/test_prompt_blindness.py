"""Regression tests for the FT-Agent blind-test prompt boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from rdagent.components.coder.finetune.conf import FT_YAML_FILE_NAME
from rdagent.core.experiment import RunningInfo
from rdagent.scenarios.finetune.dev import feedback as experiment_feedback_module
from rdagent.scenarios.finetune.dev.feedback import FTExperiment2Feedback
from rdagent.scenarios.finetune.proposal.trace import FTTrace
from rdagent.scenarios.finetune.scen import scenario as scenario_module
from rdagent.scenarios.finetune.train import eval as runner_eval_module
from rdagent.scenarios.finetune.train.eval import FTRunnerEvaluator
from rdagent.utils.agent import tpl as tpl_module
from rdagent.utils.agent.tpl import T

if TYPE_CHECKING:
    import pytest
    from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback

TEST_SENTINEL = "HELD_OUT_TEST_SENTINEL_7f0c23"
VALIDATION_MARKER = "VALIDATION_MARKER_5bd911"


class _Workspace:
    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = workspace_path
        self.file_dict = {
            FT_YAML_FILE_NAME: (
                "model_name_or_path: local-model\n"
                "finetuning_type: lora\n"
                "use_rslora: false\n"
                "use_dora: false\n"
            ),
        }
        self.running_info = RunningInfo()
        self.feedback = None

    def run(self, **_kwargs: Any) -> SimpleNamespace:
        output = self.workspace_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").touch()
        return SimpleNamespace(exit_code=0, stdout="training complete", running_time=1.0)


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        benchmark="aime25",
        get_task_information=lambda: "A held-out benchmark task",
    )


def _results() -> tuple[dict[str, Any], dict[str, Any]]:
    validation = {
        "accuracy_summary": {"accuracy": 50.0, "marker": VALIDATION_MARKER},
        "error_samples": [],
    }
    test = {
        "accuracy_summary": {"accuracy": 100.0, "marker": TEST_SENTINEL},
        "error_samples": [],
    }
    return validation, test


def test_baseline_cache_separates_validation_only_from_legacy_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = object.__new__(scenario_module.LLMFinetuneScen)
    monkeypatch.setenv("FT_BENCHMARK_DATASET_PATH", "pinned/aime25")
    monkeypatch.setattr(scenario_module.FT_RD_SETTING, "evaluate_held_out_during_search", False)
    validation_key = scenario.benchmark_hash("Qwen/Qwen2.5-7B-Instruct", "aime25")

    monkeypatch.setattr(scenario_module.FT_RD_SETTING, "evaluate_held_out_during_search", True)
    legacy_key = scenario.benchmark_hash("Qwen/Qwen2.5-7B-Instruct", "aime25")

    monkeypatch.setattr(scenario_module.FT_RD_SETTING, "evaluate_held_out_during_search", False)
    monkeypatch.setenv("FT_BENCHMARK_DATASET_PATH", "pinned/aime25-v2")
    other_dataset_key = scenario.benchmark_hash("Qwen/Qwen2.5-7B-Instruct", "aime25")

    assert validation_key != legacy_key
    assert validation_key != other_dataset_key


def test_runner_search_is_validation_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Iterative search must neither run nor persist held-out test metrics."""
    validation, test = _results()
    workspace = _Workspace(tmp_path)
    captured: dict[str, str] = {}

    monkeypatch.setattr(runner_eval_module, "get_ft_env", lambda **_kwargs: object())
    monkeypatch.setattr(runner_eval_module, "clear_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_eval_module, "extract_loss_history", lambda _path: {"train": [], "eval": []})
    monkeypatch.setattr(runner_eval_module.logger, "log_object", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner_eval_module.LLMConfigValidator,
        "_parse_execution_log",
        lambda *_args, **_kwargs: {"status": "ok"},
    )

    calls: list[str] = []

    def fake_benchmark(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["result_subdir"])
        return validation if kwargs["result_subdir"] == "validation" else test

    monkeypatch.setattr(runner_eval_module, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(runner_eval_module.FT_RD_SETTING, "evaluate_held_out_during_search", False)

    def capture_feedback(feedback_type: type[CoSTEERSingleFeedback], **kwargs: Any) -> CoSTEERSingleFeedback:
        captured.update(system_prompt=kwargs["system_prompt"], user_prompt=kwargs["user_prompt"])
        return feedback_type(execution="ok", return_checking="ok", code="ok", final_decision=True)

    monkeypatch.setattr(runner_eval_module, "build_cls_from_json_with_retry", capture_feedback)

    evaluator = FTRunnerEvaluator(scen=SimpleNamespace(gpu_count=1))
    monkeypatch.setattr(
        evaluator,
        "_run_full_data_processing",
        lambda _implementation: SimpleNamespace(exit_code=0, stdout="data complete"),
    )

    evaluator.evaluate(_task(), workspace, gt_implementation=workspace)

    assert workspace.running_info.result["benchmark"] == validation
    assert "benchmark_test" not in workspace.running_info.result
    assert calls == ["validation"]
    rendered_prompt = captured["system_prompt"] + captured["user_prompt"]
    assert VALIDATION_MARKER in rendered_prompt
    assert TEST_SENTINEL not in rendered_prompt


def test_feedback_trace_and_proposal_prompts_exclude_held_out_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Later feedback and proposal stages must retain the same prompt boundary."""
    validation, test = _results()
    monkeypatch.setattr(tpl_module.logger, "log_object", lambda *_args, **_kwargs: None)
    workspace = SimpleNamespace(
        file_dict={
            FT_YAML_FILE_NAME: "model_name_or_path: local-model\n",
            "process_data.py": "print('safe training data')\n",
        },
        running_info=RunningInfo(
            result={
                "benchmark": validation,
                "benchmark_test": test,
                "training_metrics": {"loss_history": {"train": [], "eval": []}},
            },
            running_time=1.0,
        ),
    )
    experiment = SimpleNamespace(
        sub_tasks=[_task()],
        sub_workspace_list=[],
        experiment_workspace=workspace,
        hypothesis="validation-only hypothesis",
    )

    trace = object.__new__(FTTrace)
    trace.get_sota_experiment = lambda: experiment

    assert trace.sota_benchmark() == validation
    parent_info = trace.get_experiment_info(experiment)
    assert VALIDATION_MARKER in json.dumps(parent_info)
    assert TEST_SENTINEL not in json.dumps(parent_info)

    proposal_prompt = T(
        "rdagent.scenarios.finetune.proposal.prompts:unified_hypothesis_gen.user_prompt",
    ).r(
        parent_exp=experiment,
        siblings=[],
        trace=trace,
        based_on_a_successful_parent=True,
    )
    assert VALIDATION_MARKER in proposal_prompt
    assert TEST_SENTINEL not in proposal_prompt

    scenario = SimpleNamespace(
        get_scenario_all_desc=lambda: "scenario without result payloads",
        baseline_benchmark_score=validation,
        baseline_benchmark_score_test=test,
    )
    captured: dict[str, str] = {}

    class CapturingBackend:
        def build_messages_and_create_chat_completion(
            self,
            *,
            user_prompt: str,
            system_prompt: str,
            **_kwargs: Any,
        ) -> str:
            captured.update(system_prompt=system_prompt, user_prompt=user_prompt)
            return json.dumps(
                {
                    "Code Summary": "safe",
                    "Reason": "validation improved",
                    "Decision": "yes",
                },
            )

    monkeypatch.setattr(experiment_feedback_module, "APIBackend", CapturingBackend)
    summarizer = FTExperiment2Feedback(scenario)
    summarizer.generate_feedback(experiment, trace)

    rendered_prompt = captured["system_prompt"] + captured["user_prompt"]
    assert VALIDATION_MARKER in rendered_prompt
    assert TEST_SENTINEL not in rendered_prompt
