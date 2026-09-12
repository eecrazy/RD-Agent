from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdagent.scenarios.finetune.benchmark.benchmark import get_lora_max_rank
from rdagent.utils.agent.tpl import T

LARGE_RANK = 256


def test_get_lora_max_rank_includes_rank_pattern(tmp_path: Path) -> None:
    config = {"r": 128, "rank_pattern": {"model.layers.0.self_attn.q_proj": LARGE_RANK}}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    assert get_lora_max_rank(tmp_path) == LARGE_RANK


@pytest.mark.parametrize(("adapter_rank", "vllm_rank"), [(1, 1), (2, 8), (96, 128), (300, 320)])
def test_get_lora_max_rank_rounds_up_to_vllm_capacity(
    tmp_path: Path, adapter_rank: int, vllm_rank: int,
) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": adapter_rank}), encoding="utf-8")

    assert get_lora_max_rank(tmp_path) == vllm_rank


def test_get_lora_max_rank_rejects_rank_above_vllm_limit(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 513}), encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds vLLM's supported maximum"):
        get_lora_max_rank(tmp_path)


@pytest.mark.parametrize("rank", [None, True, 0, -1, "128"])
def test_get_lora_max_rank_rejects_invalid_rank(tmp_path: Path, rank: object) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": rank}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid LoRA rank"):
        get_lora_max_rank(tmp_path)


def test_opencompass_template_uses_adapter_rank() -> None:
    source = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(
        dataset_imports=["fake_dataset"],
        dataset_path_literal="None",
        test_range_literal="None",
        num_runs=1,
        pass_k=None,
        model_abbr="test-model",
        model_path="/models/test-model",
        is_lora=True,
        lora_path="/adapters/test-model",
        max_lora_rank=128,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        enable_thinking=False,
        use_cot_postprocessor=False,
        work_dir="results",
    )

    assert "max_lora_rank=128" in source


def test_opencompass_template_has_inflight_worker_fallback() -> None:
    source = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(
        dataset_imports=["fake_dataset"],
        dataset_path_literal="None",
        test_range_literal="None",
        num_runs=1,
        pass_k=None,
        model_abbr="test-model",
        model_path="/models/test-model",
        is_lora=True,
        lora_path="/adapters/test-model",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_seq_len=4096,
        max_out_len=1024,
        batch_size=1,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        enable_thinking=False,
        use_cot_postprocessor=False,
        work_dir="results",
    )

    assert "max_lora_rank=512" in source
