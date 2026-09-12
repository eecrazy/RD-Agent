import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock
from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.finetune.benchmark import get_benchmark_ranges, run_benchmark
from rdagent.scenarios.finetune.datasets import prepare_all
from rdagent.scenarios.finetune.experiment.workspace import FTWorkspace
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager
from rdagent.scenarios.finetune.scen.memory_estimator import MemoryEstimator
from rdagent.scenarios.finetune.scen.utils import (
    FinetuneDatasetDescriptor,
    generate_dataset_info_config,
    valid_dataset_info_cache,
)
from rdagent.scenarios.finetune.train.formal_training import formal_expected_samples
from rdagent.scenarios.finetune.utils import ensure_ft_assets_exist
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T

LOGICAL_GPU_COUNT_ENV = "FT_LOGICAL_TRAINING_GPU_COUNT"
LOGICAL_GPU_MEMORY_ENV = "FT_LOGICAL_TRAINING_GPU_MEMORY_GB"
LOGICAL_GPU_NAME_ENV = "FT_LOGICAL_TRAINING_GPU_NAME"
LOGICAL_RESOURCE_SCOPE_ENV = "FT_LOGICAL_TRAINING_RESOURCE_SCOPE"
PHYSICAL_EXECUTION_MAPPING_ENV = "FT_PHYSICAL_TRAINING_MAPPING"


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self) -> None:
        """Initialize LLM finetune scenario using configuration from FT_RD_SETTING."""
        logger.info("Initializing LLM Fine-tune scenario")

        # Basic attributes
        self.user_target_scenario = FT_RD_SETTING.user_target_scenario
        self.target_benchmark = FT_RD_SETTING.target_benchmark
        self.benchmark_description = FT_RD_SETTING.benchmark_description
        self.dataset = FT_RD_SETTING.dataset
        self.base_model = FT_RD_SETTING.base_model

        # Validate and prepare environment
        self._validate_and_prepare_environment()

        # Initialize LLaMA Factory manager
        self._initialize_llama_factory()

        # Generate dataset configuration for all datasets first
        self.dataset_config = self._prepare_dataset_config()

        # Select relevant datasets based on user target scenario (using full config info)
        self.selected_datasets = self._select_relevant_datasets()

        # Filter dataset_config to only include selected datasets
        self.dataset_config = {k: v for k, v in self.dataset_config.items() if k in self.selected_datasets}

        # timeout tracking
        self.timeout_increase_count = 0

        # NOTE: we disable the cache for environment. in case of changing cuda config
        self.device_info = get_runtime_environment_by_env(get_ft_env(enable_cache=False))
        self.gpu_count = json.loads(self.device_info).get("gpu_count", 0)
        self.model_info = FinetuneDatasetDescriptor().describe_model(self.base_model)

        # Method selection uses an explicit logical resource envelope.  The
        # reproduction runner supplies the paper's B200 envelope while the
        # worker-local CUDA view remains available for execution and baseline
        # evaluation only.
        self.training_resource = self._resolve_training_resource()
        self.training_resource_info = json.dumps(self.training_resource, indent=2)
        self.physical_execution_mapping = os.environ.get(PHYSICAL_EXECUTION_MAPPING_ENV, "").strip()
        self.memory_report = self._generate_memory_report()

        baseline_result = self.run_baseline_model_evaluation(
            model_name=self.base_model, benchmark_name=self.target_benchmark,
        )
        # Agent only sees validation score
        self.baseline_benchmark_score = baseline_result.get("benchmark", {})
        # Normally populated only by the separate final-test runner.  The
        # legacy search-time opt-in may still populate it for UI callers.
        self.baseline_benchmark_score_test = (
            baseline_result.get("benchmark_test", {})
            if FT_RD_SETTING.evaluate_held_out_during_search
            else {}
        )

    def benchmark_hash(self, model_name: str, benchmark_name: str) -> str:
        payload = {
            "schema_version": 2,
            "evaluation_mode": (
                "legacy_with_held_out"
                if FT_RD_SETTING.evaluate_held_out_during_search
                else "validation_only"
            ),
            "model": model_name,
            "benchmark": benchmark_name,
            "benchmark_dataset_path": os.environ.get("FT_BENCHMARK_DATASET_PATH"),
            "benchmark_limit": FT_RD_SETTING.benchmark_limit,
            "benchmark_num_runs": FT_RD_SETTING.benchmark_num_runs,
            "benchmark_pass_k": FT_RD_SETTING.benchmark_pass_k,
            "judge_model": FT_RD_SETTING.judge_model,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"llm_finetune_baseline_eval_v2_{digest}"

    @cache_with_pickle(benchmark_hash)
    def run_baseline_model_evaluation(self, model_name: str, benchmark_name: str) -> dict[str, Any]:
        ws = FTWorkspace()
        shutil.copytree(
            Path(FT_RD_SETTING.file_path) / "models" / model_name,
            ws.workspace_path / "models" / model_name,
            dirs_exist_ok=True,
        )
        val_range, test_range = get_benchmark_ranges()

        # Validation set - visible to agent
        validation_result = run_benchmark(
            workspace_path=str(ws.workspace_path),
            model_path=ws.workspace_path / "models" / model_name,
            model_name=model_name,
            benchmark_name=benchmark_name,
            gpu_count=self.gpu_count,
            test_range=val_range,
            result_subdir="validation",
        )
        result = {"benchmark": validation_result}
        if FT_RD_SETTING.evaluate_held_out_during_search:
            result["benchmark_test"] = run_benchmark(
                workspace_path=str(ws.workspace_path),
                model_path=ws.workspace_path / "models" / model_name,
                model_name=model_name,
                benchmark_name=benchmark_name,
                gpu_count=self.gpu_count,
                test_range=test_range,
                result_subdir="test",
            )
        return result

    def real_full_timeout(self):
        return FT_RD_SETTING.full_timeout

    def _physical_training_resource(self) -> dict[str, Any]:
        """Return one-GPU memory and topology from the runtime-visible device report."""
        device_info = json.loads(self.device_info) if isinstance(self.device_info, str) else self.device_info
        gpu_info = device_info.get("gpu", {})
        gpus = gpu_info.get("gpus", [])
        num_gpus = gpu_info.get("gpu_count") or len(gpus) or device_info.get("gpu_count")
        first_gpu = gpus[0] if gpus else {}
        gpu_name = first_gpu.get("name") or "GPU"
        gpu_mem = first_gpu.get("memory_total_gb")
        if not gpu_mem:
            total_mem = gpu_info.get("summary", {}).get("total_memory_gb")
            gpu_mem = total_mem / num_gpus if total_mem and num_gpus else None
        if not num_gpus or not gpu_mem:
            message = "GPU topology or per-device memory is unavailable"
            raise ValueError(message)
        return {
            "scope": "runtime-visible physical resources",
            "source": "runtime_device_report",
            "gpu_count": int(num_gpus),
            "gpu_name": str(gpu_name),
            "memory_per_gpu_gb": float(gpu_mem),
            "total_memory_gb": float(gpu_mem) * int(num_gpus),
        }

    def _resolve_training_resource(self) -> dict[str, Any]:
        """Resolve the resource envelope used for autonomous method selection."""
        raw_count = os.environ.get(LOGICAL_GPU_COUNT_ENV)
        raw_memory = os.environ.get(LOGICAL_GPU_MEMORY_ENV)
        if raw_count is None and raw_memory is None:
            return self._physical_training_resource()
        if raw_count is None or raw_memory is None:
            message = f"{LOGICAL_GPU_COUNT_ENV} and {LOGICAL_GPU_MEMORY_ENV} must be set together"
            raise ValueError(message)
        try:
            count = int(raw_count)
            memory = float(raw_memory)
        except ValueError as error:
            message = "Logical training GPU count and memory must be numeric"
            raise ValueError(message) from error
        if count < 1 or memory <= 0:
            message = "Logical training GPU count and memory must be positive"
            raise ValueError(message)
        name = os.environ.get(LOGICAL_GPU_NAME_ENV, "GPU").strip() or "GPU"
        scope = os.environ.get(
            LOGICAL_RESOURCE_SCOPE_ENV,
            "logical method-selection resources",
        ).strip()
        return {
            "scope": scope,
            "source": "logical_resource_override",
            "gpu_count": count,
            "gpu_name": name,
            "memory_per_gpu_gb": memory,
            "total_memory_gb": memory * count,
        }

    def _generate_memory_report(self) -> str:
        """Generate the method-selection report from the logical resource envelope."""
        try:
            resource = self.training_resource
            estimator = MemoryEstimator.from_model_name(
                name=self.base_model,
                gpu_mem=resource["memory_per_gpu_gb"],
                num_gpus=resource["gpu_count"],
                model_specs=self.model_info.get("specs", ""),
                gpu_name=resource["gpu_name"],
                resource_scope=resource["scope"],
            )
            policy_methods = {
                "paper": ("full", "full_gc", "lora"),
                "full": ("full", "full_gc"),
                "lora": ("lora",),
                "rslora": ("lora",),
            }.get(os.environ.get("FT_TRAINING_POLICY", "").strip().lower().replace("-", ""))
            return estimator.format(methods=policy_methods)
        except Exception as e:
            logger.warning(f"Failed to generate memory report: {e}")
            return ""

    def _validate_and_prepare_environment(self):
        """Validate FT_FILE_PATH and prepare all registered datasets"""
        ft_root = Path(FT_RD_SETTING.file_path)
        if not ft_root.exists():
            os.makedirs(ft_root, mode=0o777, exist_ok=True)
            logger.info(f"FT_FILE_PATH not exists, created FT_FILE_PATH directory: {ft_root}")

        # Prepare all registered datasets
        prepare_all()

        # Ensure model assets exist
        if self.base_model:
            ensure_ft_assets_exist(model=self.base_model, check_model=True)

    def _initialize_llama_factory(self):
        """Initialize LLaMA Factory information manager"""

        # Extract LLaMA Factory information (pulls latest code automatically)
        info = LLaMAFactory_manager.get_info()

        # Log extracted information
        methods_count = len(info.get("methods", []))
        params_count = sum(len(p) if isinstance(p, dict) else 0 for p in info.get("parameters", {}).values())
        logger.info(f"LLaMA Factory initialized: {methods_count} methods, {params_count} parameters")

    def _select_relevant_datasets(self) -> list[str]:
        """Select relevant datasets based on user target scenario using LLM.

        Uses self.dataset_config which contains full information (stats, description, samples).
        """
        total = len(self.dataset_config)

        # If user specified a dataset, use it directly
        if self.dataset:
            selected, reasoning = [self.dataset], "User specified dataset directly"
        elif not self.dataset_config:
            logger.warning("No datasets found for selection")
            return []
        else:
            # Use LLM to select relevant datasets
            logger.info(f"Found {total} datasets, selecting relevant ones...")
            selected, reasoning = self._llm_select_datasets()

        # Log results
        logger.info(f"Dataset selection: {len(selected)}/{total} - {selected}")
        logger.log_object(
            {"selected_datasets": selected, "total_datasets": total, "reasoning": reasoning},
            tag="dataset_selection",
        )
        return selected

    def _llm_select_datasets(self) -> tuple[list[str], str]:
        """Use LLM to select relevant datasets."""
        # Pass dataset_config directly - it already has the unified tasks structure
        dataset_summaries = [
            {
                "name": ds_name,
                "total_samples": ds_config.get("total_samples"),
                "total_size_mb": ds_config.get("total_size_mb"),
                "tasks": ds_config.get("tasks", {}),
                "readme": ds_config.get("readme"),
            }
            for ds_name, ds_config in self.dataset_config.items()
        ]

        system_prompt = T(".prompts:dataset_selection.system").r(
            user_target_scenario=self.user_target_scenario,
            target_benchmark=self.target_benchmark,
            benchmark_description=self.benchmark_description,
        )
        user_prompt = T(".prompts:dataset_selection.user").r(datasets=dataset_summaries)

        response = APIBackend().build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )

        result = json.loads(response)
        return result.get("selected_datasets", []), result.get("reasoning", "")

    def _prepare_dataset_config(self) -> dict:
        """Generate dataset_info.json configuration.

        This is the single source of truth for dataset information, containing:
        - LlamaFactory compatible fields (file_name, formatting, columns)
        - Auto-computed statistics (stats.column_stats)
        - Data samples (truncated)
        - AI-generated description

        Returns:
            dict: Complete dataset configuration
        """
        datasets_dir = Path(FT_RD_SETTING.file_path) / "datasets"
        dataset_info_path = datasets_dir / "dataset_info.json"

        # Check if already configured
        existing_config = {}
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, encoding="utf-8") as f:
                    existing_config = json.load(f)

                existing_config = valid_dataset_info_cache(FT_RD_SETTING.file_path, existing_config)

            except Exception as e:
                logger.warning(f"Failed to load existing dataset_info.json: {e}")

        # Generate config for all datasets (will be filtered later by _select_relevant_datasets)
        target_dataset_list = [] if self.dataset is None else [self.dataset]
        logger.info(
            f"Generating dataset_info.json configuration for: {target_dataset_list or 'all datasets'}",
        )
        generated_config = generate_dataset_info_config(target_dataset_list, FT_RD_SETTING.file_path, existing_config)
        for dataset_name, config in generated_config.items():
            existing_config[dataset_name] = config

        try:
            os.makedirs(datasets_dir, mode=0o777, exist_ok=True)

            # Many matrix workers initialize this scenario concurrently.  A
            # direct ``open(..., 'w')`` exposes a truncated/partially-written
            # JSON document to readers and can also discard entries generated
            # by another worker.  Merge once more while holding a short write
            # lock, then atomically replace the cache file.
            lock_path = dataset_info_path.with_suffix(dataset_info_path.suffix + ".lock")
            with FileLock(lock_path):
                latest_config: dict[str, Any] = {}
                if dataset_info_path.exists():
                    try:
                        with open(dataset_info_path, encoding="utf-8") as f:
                            latest_config = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to reload existing dataset_info.json before update: {e}")

                latest_config = valid_dataset_info_cache(FT_RD_SETTING.file_path, latest_config)
                latest_config.update(existing_config)

                temp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=datasets_dir,
                        prefix=f".{dataset_info_path.name}.",
                        suffix=".tmp",
                        delete=False,
                    ) as f:
                        temp_path = Path(f.name)
                        json.dump(latest_config, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_path, dataset_info_path)
                    temp_path = None
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

                existing_config = latest_config
            logger.info(f"Successfully updated dataset_info.json with configuration for: {target_dataset_list}")
        except Exception as e:
            message = f"Failed to write dataset_info.json: {e}"
            raise RuntimeError(message) from e
        return existing_config

    @property
    def metric_direction(self) -> bool:
        """Metric direction for LLM fine-tuning (higher is better)"""
        return True

    def get_scenario_all_desc(self, enable_dataset_description: bool = False) -> str:
        """Get complete scenario description for LLM fine-tuning.

        Uses dataset_config as the single source of truth for dataset information.
        The prompt template renders tasks with their statistics and samples.
        """
        return T(".prompts:scenario_description").r(
            user_target_scenario=self.user_target_scenario,
            target_benchmark=self.target_benchmark,
            benchmark_description=self.benchmark_description,
            training_resource_info=self.training_resource_info,
            physical_execution_mapping=self.physical_execution_mapping,
            memory_report=self.memory_report,
            chosen_model=FT_RD_SETTING.base_model is not None,
            base_model=FT_RD_SETTING.base_model,
            dataset_config=self.dataset_config,
            model_info=self.model_info,
            full_timeout=f"{self.real_full_timeout() / 60 / 60:.2f} hours",
            data_processing_timeout=f"{FT_RD_SETTING.data_processing_timeout / 60:.0f} minutes",
            enable_dataset_description=enable_dataset_description,
            upper_data_size_limit=FT_RD_SETTING.upper_data_size_limit,
            formal_expected_samples=formal_expected_samples(),
        )
