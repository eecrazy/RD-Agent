from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType

from rdagent.utils.agent.tpl import T


def test_opencompass_template_removes_runtime_helpers(monkeypatch) -> None:
    @contextmanager
    def read_base():
        yield

    mmengine = ModuleType("mmengine")
    mmengine_config = ModuleType("mmengine.config")
    mmengine_config.read_base = read_base
    opencompass = ModuleType("opencompass")
    opencompass_models = ModuleType("opencompass.models")
    opencompass_models.VLLMwithChatTemplate = object
    fake_dataset = ModuleType("fake_dataset")
    fake_dataset.datasets = [
        {
            "path": "original/path",
            "eval_cfg": {
                "evaluator": {
                    "dataset_cfg": {
                        "path": "original/path",
                        "reader_cfg": {},
                    },
                },
            },
        },
    ]

    monkeypatch.setitem(__import__("sys").modules, "mmengine", mmengine)
    monkeypatch.setitem(__import__("sys").modules, "mmengine.config", mmengine_config)
    monkeypatch.setitem(__import__("sys").modules, "opencompass", opencompass)
    monkeypatch.setitem(__import__("sys").modules, "opencompass.models", opencompass_models)
    monkeypatch.setitem(__import__("sys").modules, "fake_dataset", fake_dataset)

    source = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(
        dataset_imports=["fake_dataset"],
        dataset_path_literal=repr("pinned/dataset"),
        test_range_literal=repr("[:50]"),
        num_runs=1,
        pass_k=None,
        model_abbr="test-model",
        model_path="/models/test-model",
        is_lora=False,
        lora_path="",
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
        work_dir="/tmp/results",
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<opencompass-config>", "exec"), namespace)

    assert "sync_nested_dataset_cfgs" not in namespace
    dataset = namespace["datasets"][0]
    assert dataset["path"] == "pinned/dataset"
    assert dataset["reader_cfg"]["test_range"] == "[:50]"
    nested = dataset["eval_cfg"]["evaluator"]["dataset_cfg"]
    assert nested["path"] == "pinned/dataset"
    assert nested["reader_cfg"]["test_range"] == "[:50]"
