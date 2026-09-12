"""One-sample OpenCompass smoke configuration for the local 3B target."""

# ruff: noqa: A001

from pathlib import Path

from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

with read_base():
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

for dataset in aime2024_datasets:
    dataset["reader_cfg"]["test_range"] = "[:1]"

datasets = aime2024_datasets

models = [
    {
        "type": VLLMwithChatTemplate,
        "abbr": "qwen2.5-3b-instruct-local-smoke",
        "path": str(
            Path(__file__).resolve().parents[2] / "finetune_files" / "models" / "Qwen" / "Qwen2.5-3B-Instruct",
        ),
        "model_kwargs": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.3,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 4096,
        },
        "max_seq_len": 4096,
        "max_out_len": 1024,
        "batch_size": 1,
        "generation_kwargs": {"temperature": 0},
        "run_cfg": {"num_gpus": 1, "num_procs": 1},
    },
]

infer = {
    "partitioner": {"type": "NaivePartitioner"},
    "runner": {
        "type": "LocalRunner",
        "max_num_workers": 1,
        "task": {"type": "OpenICLInferTask"},
    },
}

eval = {
    "partitioner": {"type": "NaivePartitioner"},
    "runner": {
        "type": "LocalRunner",
        "max_num_workers": 1,
        "task": {"type": "OpenICLEvalTask", "dump_details": True},
    },
}
