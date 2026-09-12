from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

# ==================== Dataset Import ====================
with read_base():

    from opencompass.configs.datasets.tablebench.tablebench_fact_checking_gen import *

# Aggregate all dataset variables
datasets = sum([v for k, v in locals().items() if (k == "datasets" or k.endswith("_datasets")) and isinstance(v, list)], [])

pinned_dataset_path = "/data/github/RD-Agent/finetune_files/benchmarks/pinned/tablebench"
test_range_override = "[-min(100, len(index_list)//2):]"

def sync_nested_dataset_cfgs(value):
    if isinstance(value, dict):
        dataset_cfg = value.get("dataset_cfg")
        if isinstance(dataset_cfg, dict):
            if pinned_dataset_path is not None and "path" in dataset_cfg:
                dataset_cfg["path"] = pinned_dataset_path
            if test_range_override is not None:
                if "reader_cfg" not in dataset_cfg:
                    dataset_cfg["reader_cfg"] = {}
                dataset_cfg["reader_cfg"]["test_range"] = test_range_override
        for nested in value.values():
            sync_nested_dataset_cfgs(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            sync_nested_dataset_cfgs(nested)

# Apply dataset modifications
for ds in datasets:
    if pinned_dataset_path is not None:
        ds["path"] = pinned_dataset_path

    # Apply dataset range (e.g., "[:100]" for validation, "[-100:]" for test)
    if test_range_override is not None:
        if "reader_cfg" not in ds:
            ds["reader_cfg"] = {}
        ds["reader_cfg"]["test_range"] = test_range_override

    # Cascade evaluators can nest GenericLLMEvaluator dataset_cfg objects.
    sync_nested_dataset_cfgs(ds.get("eval_cfg", {}))


# mmengine treats every remaining top-level name as config data.  Keeping a
# local helper here makes OpenCompass dump ``<function ...>`` into its
# resolved config, which is not valid Python when it reloads that file.
del sync_nested_dataset_cfgs

# ==================== Model Configuration ====================
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="base-qwen2.5-7b-instruct-tablebench_fact_checking",
        path="/data/github/RD-Agent/finetune_files/models/Qwen/Qwen2.5-7B-Instruct",
        model_kwargs=dict(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=32768,

        ),

        max_seq_len=32768,
        max_out_len=8192,
        batch_size=16,
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
            top_k=1,

        ),

        run_cfg=dict(
            num_gpus=1,
            num_procs=1,
        ),
    ),
]

# ==================== Inference Configuration ====================
infer = dict(
    partitioner=dict(
        type="NaivePartitioner",
    ),
    runner=dict(
        type="LocalRunner",
        max_num_workers=16,
        task=dict(
            type="OpenICLInferTask",
        ),
    ),
)

# ==================== Evaluation Configuration ====================
eval = dict(
    partitioner=dict(
        type="NaivePartitioner",
    ),
    runner=dict(
        type="LocalRunner",
        max_num_workers=16,
        task=dict(
            type="OpenICLEvalTask",
            dump_details=True,
        ),
    ),
)

# ==================== Work Directory ====================
work_dir = "/data/github/RD-Agent/finetune_files/logs/paper-base/h20-gpt56-base-v1/base__7b__tablebench_fact_checking/test"
