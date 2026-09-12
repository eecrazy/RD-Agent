"""Regression tests for paper-resource selection versus H20 execution."""

from __future__ import annotations

import json

import pytest
from rdagent.scenarios.finetune.scen import scenario as scenario_module
from rdagent.utils.agent.tpl import T

H20_MEMORY_GB = 95.0
H20_TOTAL_MEMORY_GB = 190.0


def test_scenario_prompt_accepts_nonformal_or_legacy_context() -> None:
    """A worker started before the formal-count field existed must not crash."""
    context = {
        "user_target_scenario": None,
        "target_benchmark": "aime25",
        "benchmark_description": "AIME validation",
        "training_resource_info": "one logical B200",
        "memory_report": "",
        "chosen_model": True,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "model_info": {},
        "enable_dataset_description": False,
        "upper_data_size_limit": 2000,
    }
    nonformal = T("rdagent.scenarios.finetune.scen.prompts:scenario_description").r(
        **context,
        physical_execution_mapping="",
    )
    legacy_formal = T("rdagent.scenarios.finetune.scen.prompts:scenario_description").r(
        **context,
        physical_execution_mapping="LoRA on one H20; Full SFT on an H20 group",
    )

    assert "The upper limit is 2000 samples" in nonformal
    assert "contain exactly 2000 training records" in legacy_formal
    assert "comfort margin" in legacy_formal
    assert "reuse a reserved validation identity in training" in legacy_formal


def test_scenario_description_injects_formal_sample_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production caller must pass the formal count into StrictUndefined."""
    scenario = object.__new__(scenario_module.LLMFinetuneScen)
    scenario.user_target_scenario = None
    scenario.target_benchmark = "aime25"
    scenario.benchmark_description = "AIME validation"
    scenario.training_resource_info = "one logical B200"
    scenario.physical_execution_mapping = ""
    scenario.memory_report = ""
    scenario.dataset_config = {}
    scenario.model_info = {}
    monkeypatch.setenv("FT_FORMAL_EXPECTED_SAMPLES", "2000")

    description = scenario.get_scenario_all_desc()

    assert "contain exactly 2000 training records" in description
    assert "The upper limit is 2000 samples" not in description


def _h20_worker_report() -> str:
    return json.dumps(
        {
            "gpu_count": 1,
            "gpu": {
                "source": "pytorch",
                "gpu_count": 1,
                "gpus": [
                    {
                        "name": "NVIDIA H20",
                        "memory_total_gb": 95.07,
                    },
                ],
                "summary": {"total_memory_gb": 95.07},
            },
        },
    )


def test_paper_logical_resource_overrides_worker_local_h20(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = object.__new__(scenario_module.LLMFinetuneScen)
    scenario.device_info = _h20_worker_report()
    monkeypatch.setenv(scenario_module.LOGICAL_GPU_COUNT_ENV, "1")
    monkeypatch.setenv(scenario_module.LOGICAL_GPU_MEMORY_ENV, "178")
    monkeypatch.setenv(scenario_module.LOGICAL_GPU_NAME_ENV, "NVIDIA B200")
    monkeypatch.setenv(
        scenario_module.LOGICAL_RESOURCE_SCOPE_ENV,
        "paper logical per-experiment method-selection envelope",
    )

    resource = scenario._resolve_training_resource()  # noqa: SLF001

    assert resource == {
        "scope": "paper logical per-experiment method-selection envelope",
        "source": "logical_resource_override",
        "gpu_count": 1,
        "gpu_name": "NVIDIA B200",
        "memory_per_gpu_gb": 178.0,
        "total_memory_gb": 178.0,
    }


def test_paper_memory_report_keeps_full_sft_in_method_space(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = object.__new__(scenario_module.LLMFinetuneScen)
    scenario.base_model = "Qwen/Qwen2.5-7B-Instruct"
    scenario.model_info = {"specs": "max_position_embeddings: 32768"}
    scenario.training_resource = {
        "scope": "paper logical per-experiment method-selection envelope",
        "source": "logical_resource_override",
        "gpu_count": 1,
        "gpu_name": "NVIDIA B200",
        "memory_per_gpu_gb": 178.0,
        "total_memory_gb": 178.0,
    }
    monkeypatch.setenv("FT_TRAINING_POLICY", "paper")

    report = scenario._generate_memory_report()  # noqa: SLF001

    assert "1x NVIDIA B200, 178GB each" in report
    assert "| full_gc |" in report
    assert "| full_gc | Not viable |" not in report
    assert "| lora |" in report
    assert "qlora" not in report.lower()


def test_runtime_resource_uses_per_gpu_memory_not_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = object.__new__(scenario_module.LLMFinetuneScen)
    scenario.device_info = json.dumps(
        {
            "gpu_count": 2,
            "gpu": {
                "source": "pytorch",
                "gpu_count": 2,
                "gpus": [
                    {"name": "NVIDIA H20", "memory_total_gb": H20_MEMORY_GB},
                    {"name": "NVIDIA H20", "memory_total_gb": H20_MEMORY_GB},
                ],
                "summary": {"total_memory_gb": H20_TOTAL_MEMORY_GB},
            },
        },
    )
    monkeypatch.delenv(scenario_module.LOGICAL_GPU_COUNT_ENV, raising=False)
    monkeypatch.delenv(scenario_module.LOGICAL_GPU_MEMORY_ENV, raising=False)

    resource = scenario._resolve_training_resource()  # noqa: SLF001

    assert resource["memory_per_gpu_gb"] == H20_MEMORY_GB
    assert resource["total_memory_gb"] == H20_TOTAL_MEMORY_GB
