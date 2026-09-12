"""
Simplified LLM Fine-tuning Configuration Validator

Two-step validation:
1. Parameter filtering - Remove unsupported parameters
2. Micro-batch testing - Runtime validation with small dataset
"""

import ast
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rdagent.components.coder.finetune.conf import (
    FT_DEBUG_YAML_FILE_NAME,
    FT_TEST_PARAMS_FILE_NAME,
)
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.scen.llama_factory_manager import LLaMAFactory_manager

DIRNAME = Path(__file__).absolute().resolve().parent
MICRO_BATCH_OUTPUT_PREFIX = ".rdagent_micro_batch_output-"
TIMEOUT_EXIT_CODE = 124
MAX_ERROR_TEXT_LENGTH = 4000
MAX_RAW_LOG_TAIL_LENGTH = 2000
TRAINING_POLICY_ENV = "FT_TRAINING_POLICY"
DEFAULT_TRAINING_POLICY = "paper"
TRAINING_POLICIES = ("paper", "full", "lora", "rslora")
TRAINING_METHODS = ("full", "lora", "rslora")


def get_training_policy(policy: str | None = None) -> str:
    """Return the validated fine-tuning policy selected for this run."""
    value = (policy if policy is not None else os.getenv(TRAINING_POLICY_ENV, DEFAULT_TRAINING_POLICY))
    normalized = value.strip().lower().replace("-", "")
    if normalized not in TRAINING_POLICIES:
        supported = ", ".join(TRAINING_POLICIES)
        message = f"{TRAINING_POLICY_ENV} must be one of: {supported}; got {value!r}"
        raise ValueError(message)
    return normalized


def training_method_from_config(config: dict[str, Any]) -> str | None:
    """Classify a LlamaFactory configuration as full, LoRA, or rsLoRA."""
    finetuning_type = str(config.get("finetuning_type", "")).strip().lower()
    if finetuning_type == "full":
        return "full"
    if finetuning_type == "lora":
        return "rslora" if config.get("use_rslora") is True else "lora"
    return None


def training_policy_guidance(policy: str | None = None) -> str:
    """Render the method constraint supplied to the configuration coder."""
    active = get_training_policy(policy)
    common = (
        "All allowed methods use the unquantized base model in BF16; QLoRA/on-the-fly "
        "quantization and DoRA are disabled so they cannot become uncontrolled experimental "
        "variables. For full SFT, the runtime uses H20 multi-GPU ZeRO-3 (2 GPUs through 8K context, "
        "4 GPUs above 8K). Express train.yaml as the paper's logical single-B200 contract: "
        "logical_global_batch = per_device_train_batch_size * gradient_accumulation_steps. "
        "Do not multiply or divide either field by the H20 world size. The runtime alone maps "
        "that contract onto the leased GPUs while preserving actual_global_batch = "
        "runtime_per_device_batch * runtime_accumulation * world_size = logical_global_batch."
    )
    guidance = {
        "paper": (
            "Paper-equivalent policy: follow the task description and hypothesis when choosing "
            "full-parameter SFT or ordinary unquantized LoRA. rsLoRA is excluded from the paper "
            "policy and must run only under the separate rslora comparison policy; do not rewrite "
            "full SFT into an adapter method."
        ),
        "full": "Controlled policy: use full-parameter SFT (finetuning_type: full).",
        "lora": (
            "Controlled policy: use ordinary LoRA (finetuning_type: lora, use_rslora: false)."
        ),
        "rslora": (
            "Controlled policy: use rsLoRA (finetuning_type: lora, use_rslora: true)."
        ),
    }[active]
    return f"{guidance} {common}"


def _configuration_method_errors(config: dict[str, Any], method: str | None) -> list[str]:
    """Return method-shape violations shared by every training policy."""
    errors: list[str] = []
    if method is None:
        errors.append("finetuning_type must be 'full' or 'lora'")
    if config.get("use_dora", False) is not False:
        errors.append("use_dora must be false or omitted")
    if config.get("quantization_bit") is not None:
        errors.append("quantization_bit must be null or omitted; QLoRA/quantized training is not allowed")
    if method == "full" and "use_rslora" in config:
        errors.append("use_rslora must be omitted for full-parameter SFT")
    if "use_rslora" in config and not isinstance(config.get("use_rslora"), bool):
        errors.append("use_rslora must be a boolean")
    return errors


def validate_training_policy(config_yaml: str, policy: str | None = None) -> list[str]:
    """Return deterministic method-policy violations before a GPU is leased."""
    try:
        config = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        return [f"train.yaml is not valid YAML: {exc}"]

    if not isinstance(config, dict):
        return ["train.yaml must contain a YAML mapping"]

    try:
        active = get_training_policy(policy)
    except ValueError as exc:
        return [str(exc)]

    method = training_method_from_config(config)
    errors = _configuration_method_errors(config, method)

    if active == "paper" and method == "rslora":
        errors.append(
            "training method must be 'full' or ordinary 'lora' under the 'paper' policy "
            "(got 'rslora')",
        )
    expected = None if active == "paper" else active
    if expected is not None and method is not None and method != expected:
        errors.append(f"training method must be '{expected}' under the '{active}' policy (got '{method}')")
    return errors


_LORA_ONLY_KEYS = {
    "additional_target",
    "create_new_adapter",
    "lora_alpha",
    "lora_dropout",
    "lora_rank",
    "lora_target",
    "loraplus_lr_embedding",
    "loraplus_lr_ratio",
    "pissa_convert",
    "pissa_init",
    "pissa_iter",
    "use_dora",
    "use_rslora",
}


def normalize_training_config(config_yaml: str, policy: str | None = None) -> str:
    """Normalize only the method fields controlled by the selected policy."""
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict):
        message = "train.yaml must contain a YAML mapping"
        raise TypeError(message)

    active = get_training_policy(policy)
    if active == "full":
        config["finetuning_type"] = "full"
    elif active in {"lora", "rslora"}:
        config["finetuning_type"] = "lora"
        config["use_rslora"] = active == "rslora"

    method = training_method_from_config(config)
    if method == "full":
        for key in _LORA_ONLY_KEYS:
            config.pop(key, None)
    elif method in {"lora", "rslora"}:
        config["use_rslora"] = method == "rslora"
        config["use_dora"] = False
        config.setdefault("lora_target", "all")

    normalized = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
    errors = validate_training_policy(normalized, active)
    if errors:
        message = "Training policy rejected generated train.yaml:\n- " + "\n- ".join(errors)
        raise ValueError(message)
    return normalized

# System-managed parameters that are automatically injected during validation.
# These should NOT be checked for alignment in eval prompts.
# Single source of truth: modify here to change injected parameters.
SYSTEM_MANAGED_PARAMS = {
    "overwrite_cache": True,  # Avoid HF datasets cache lock contention
    # Preserve optimizer/scheduler/RNG state so interrupted H20 runs can be
    # resumed from a checkpoint instead of silently restarting training.
    "save_only_model": False,
    # "save_total_limit": 1,  # Limit checkpoint count to save disk space
    "output_dir": "./output",  # Standardize model output location
    "per_device_eval_batch_size": 1,  # Prevent OOM during evaluation
    # The generic SFT Trainer only emits eval_loss. Paper metrics are computed
    # later by the benchmark validator, so using them here fails after the
    # first evaluation and wastes a GPU training pass.
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def validate_rs_lora_policy(config_yaml: str) -> list[str]:
    """Compatibility wrapper for historical callers and reports."""
    return validate_training_policy(config_yaml, "rslora")


def normalize_rs_lora_config(config_yaml: str) -> str:
    """Compatibility wrapper that explicitly requests the rsLoRA policy."""
    return normalize_training_config(config_yaml, "rslora")


@dataclass
class ValidationResult:
    """Configuration validation result"""

    success: bool
    filtered_config: str
    execution_output: str = ""  # Parsed/summarized output for LLM
    raw_stdout: str = ""  # Full raw stdout for UI display
    errors: list[str] = field(default_factory=list)
    execution_time: float = 0.0


class LLMConfigValidator:
    """LLM configuration validator with two-step validation:

    1. Parameter filtering - Remove unsupported parameters
    2. Micro-batch test - Runtime validation with small dataset

    The micro-batch test inherently validates completeness, so no separate completeness check is needed.
    """

    def __init__(self) -> None:
        self._supported_params_cache: set[str] | None = None

    def validate_and_test(self, config_yaml: str, workspace: FBWorkspace, env: Any) -> ValidationResult:
        """Three-step validation: parameter filtering + injection + micro-batch testing"""
        start_time = time.time()

        policy = get_training_policy()
        policy_errors = validate_training_policy(config_yaml, policy)
        if policy_errors:
            return ValidationResult(
                success=False,
                filtered_config=config_yaml,
                execution_output=(
                    f"Training policy '{policy}' rejected train.yaml before micro-batch execution:\n- "
                    + "\n- ".join(policy_errors)
                ),
                errors=policy_errors,
                execution_time=time.time() - start_time,
            )

        # Step 1: Parameter filtering
        filtered_config, removed_params = self._filter_parameters(config_yaml)

        # Step 2: Inject required parameters for multi-task environments
        injected_config = self._inject_required_parameters(filtered_config)

        # Step 3: Micro-batch testing (validates everything at runtime)
        result = self._run_micro_batch_test(injected_config, workspace, env)
        result.execution_time = time.time() - start_time

        # Add filtered params info to execution_output for agent learning
        if removed_params:
            filter_info = (
                f"\n\n[Filtered Parameters] {len(removed_params)} unsupported params removed: {removed_params}"
            )
            result.execution_output += filter_info

        return result

    def _filter_parameters(self, config_yaml: str) -> tuple[str, list[str]]:
        """Filter configuration parameters to only include supported ones.

        Returns:
            tuple: (filtered_yaml, removed_params_list)
        """
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            return config_yaml, []

        supported_params = self._get_supported_parameters()

        filtered_config = {}
        removed_params = []
        for k, v in config_dict.items():
            if k in supported_params:
                filtered_config[k] = v
            else:
                removed_params.append(k)

        if removed_params:
            logger.info(f"Filtered out {len(removed_params)} unsupported parameters: {removed_params}")

        return yaml.dump(filtered_config, default_flow_style=False, sort_keys=False), removed_params

    def _inject_required_parameters(self, config_yaml: str) -> str:
        """Inject required parameters for multi-task environments.

        Uses SYSTEM_MANAGED_PARAMS as the single source of truth.
        """
        config = yaml.safe_load(config_yaml)
        if not isinstance(config, dict):
            return config_yaml

        config.update(SYSTEM_MANAGED_PARAMS)

        logger.info(f"Injected required parameters: {SYSTEM_MANAGED_PARAMS}")
        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _get_supported_parameters(self) -> set[str]:
        """Get supported parameters from LlamaFactory Manager"""
        if self._supported_params_cache is not None:
            return self._supported_params_cache

        all_params = LLaMAFactory_manager.get_parameters()

        # Extract all parameter names from all parameter types (including nested structures)
        supported_params = set()
        for params_dict in all_params.values():
            if isinstance(params_dict, dict):
                # Recursively extract parameter names from nested dictionaries
                for key, value in params_dict.items():
                    if isinstance(value, dict) and "name" in value:
                        # This is a parameter definition with metadata
                        supported_params.add(key)
                    elif isinstance(value, dict):
                        # This is a nested category (e.g., BaseModelArguments, LoraArguments)
                        # Extract parameter names from the nested structure
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, dict) and "name" in nested_value:
                                supported_params.add(nested_key)

        if not supported_params:
            message = "No parameters found in LlamaFactory Manager"
            raise RuntimeError(message)

        logger.info(f"Loaded {len(supported_params)} parameters from LlamaFactory Manager")
        self._supported_params_cache = supported_params
        return supported_params

    def _parse_execution_log(  # noqa: C901, PLR0912
        self,
        stdout: str,
        exit_code: int,
        failed_stage: str | None = None,
    ) -> str:
        """Parse execution log and extract key information for LLM evaluation.

        Reduces log from ~36k tokens to ~500 tokens by extracting only:
        - Status and exit code
        - Error messages (if any)
        - Training metrics (if successful)
        - Warnings (limited)
        - Timeout and stage information (if applicable)

        Args:
            stdout: The execution output
            exit_code: The process exit code
            failed_stage: Which stage failed - "data_processing" or "training"
        """
        result = {
            "status": "success" if exit_code == 0 else "failed",
            "exit_code": exit_code,
        }

        # Handle timeout (exit_code 124)
        if exit_code == TIMEOUT_EXIT_CODE:
            result["timeout"] = True
            if failed_stage:
                result["failed_stage"] = failed_stage

        # 1. Extract error information (highest priority)
        # Strategy: extract rank0's error block (each line prefixed with [rank0]:)
        error_text = None

        # Method A: Extract [rank0]: prefixed lines and reconstruct traceback
        rank0_lines = re.findall(r"\[rank0\]:[^\n]+", stdout)
        if rank0_lines:
            rank0_block = "\n".join(line.replace("[rank0]: ", "").replace("[rank0]:", "") for line in rank0_lines)
            # Find traceback in rank0 block
            tb_match = re.search(
                r"Traceback \(most recent call last\):.*?(?:Error|Exception):[^\n]+",
                rank0_block,
                re.DOTALL,
            )
            if tb_match:
                error_text = tb_match.group(0)

        # Method B: Fallback to generic traceback (no rank prefix)
        # Use findall to get ALL tracebacks, then keep the first one (root cause)
        if not error_text:
            all_tracebacks = re.findall(
                r"Traceback \(most recent call last\):.*?(?:Error|Exception):[^\n]+",
                stdout,
                re.DOTALL,
            )
            if all_tracebacks:
                # First traceback is usually the root cause
                error_text = all_tracebacks[0]
                if len(all_tracebacks) > 1:
                    error_text += f"\n\n[Note: {len(all_tracebacks)} total errors, showing root cause]"

        if error_text:
            # Limit length but keep from the END (actual error type/message is at the end of traceback)
            result["error"] = (
                error_text[-MAX_ERROR_TEXT_LENGTH:] if len(error_text) > MAX_ERROR_TEXT_LENGTH else error_text
            )

        # 2. Extract training information
        if "Running training" in stdout:
            result["training_started"] = True

            # Extract training config
            # NOTE: we may have log like "Num examples = 1,000,000" and "Num Epochs = 1,000"; So we need to handle ","
            num_examples = re.search(r"Num examples\s*=\s*([\d,]+)", stdout)
            num_epochs = re.search(r"Num Epochs\s*=\s*([\d,]+)", stdout)
            if num_examples:
                result["num_examples"] = int(num_examples.group(1).replace(",", ""))
            if num_epochs:
                result["num_epochs"] = int(num_epochs.group(1).replace(",", ""))

            # Extract final metrics (JSON format from trainer output)
            final_metrics = re.search(r"\{[\"']train_runtime[\"']:[^}]+\}", stdout)
            if final_metrics:
                try:
                    metrics = ast.literal_eval(final_metrics.group(0))
                    result["final_metrics"] = {
                        "train_loss": metrics.get("train_loss"),
                        "train_runtime": metrics.get("train_runtime"),
                        "train_samples_per_second": metrics.get("train_samples_per_second"),
                    }
                except (SyntaxError, ValueError):
                    logger.debug("Could not parse final trainer metrics from the micro-batch log")

            # Check completion
            if "Training completed" in stdout:
                result["completed"] = True

        # 3. Extract warnings (limit to 20)
        warnings = re.findall(r"\[WARNING[^\]]*\][^\n]+", stdout)
        if warnings:
            result["warnings"] = list(set(warnings))[:20]

        # 4. Fallback: if parsing failed, include truncated raw log
        if not result.get("error") and not result.get("training_started"):
            result["raw_log_tail"] = (
                stdout[-MAX_RAW_LOG_TAIL_LENGTH:] if len(stdout) > MAX_RAW_LOG_TAIL_LENGTH else stdout
            )

        return json.dumps(result, indent=2, ensure_ascii=False)

    def _run_micro_batch_test(self, config_yaml: str, workspace: FBWorkspace, env: Any) -> ValidationResult:
        """Run micro-batch training test for runtime validation"""
        result = ValidationResult(success=True, filtered_config=config_yaml)

        # Create micro-batch test configuration
        config = yaml.safe_load(config_yaml)
        if not isinstance(config, dict):
            result.success = False
            result.execution_output = "Invalid YAML configuration"
            result.errors.append("Invalid configuration for micro-batch test")
            return result

        test_config = config.copy()

        # Load extra test parameters from workspace (generated by coder in 2nd turn)
        extra_test_params = yaml.safe_load(workspace.file_dict[FT_TEST_PARAMS_FILE_NAME])

        # Merge extra test parameters (overrides previous settings)
        if extra_test_params:
            test_config.update(extra_test_params)

        # The generated formal configuration may tune prefetching for several
        # data-loader processes, while the debug override deliberately reduces
        # the worker count.  The pinned Transformers version rejects an
        # explicit prefetch factor unless more than one worker remains.
        if test_config.get("dataloader_num_workers", 0) <= 1:
            test_config.pop("dataloader_prefetch_factor", None)

        # Packing can collapse a tiny debug subset into a single sequence.  A
        # subsequent validation split then fails even though the source data
        # contains many valid rows.  Keep packing as a formal-training
        # optimization only, and do not let a debug run create or reuse the
        # formal tokenized cache.
        test_config["packing"] = False
        test_config["neat_packing"] = False
        test_config["tokenized_path"] = None

        # The micro-batch run proves that forward/backward/optimizer steps are
        # executable; it is not a candidate model and its artifacts are always
        # deleted below.  In particular, a ZeRO-3 full-SFT checkpoint includes
        # optimizer shards and can transiently consume hundreds of gigabytes
        # for a four-step smoke test.  Disable intermediate checkpoints while
        # leaving the formal train.yaml (including its checkpoint policy)
        # untouched.
        test_config["save_strategy"] = "no"
        test_config["load_best_model_at_end"] = False
        test_config["save_only_model"] = True
        test_config.pop("save_steps", None)
        test_config.pop("save_total_limit", None)

        # A successful coder evaluation is checkpointed before the formal
        # runner starts.  Reusing ``./output`` here therefore makes the debug
        # adapter look like a formal trainer checkpoint to the asynchronous
        # validation sweep.  Give every invocation its own non-formal output
        # directory, prohibit formal-checkpoint resume, and remove the debug
        # artifact even when LlamaFactory raises or times out.
        debug_output_name = f"{MICRO_BATCH_OUTPUT_PREFIX}{uuid.uuid4().hex}"
        debug_output_path = workspace.workspace_path / debug_output_name
        test_config["output_dir"] = f"./{debug_output_name}"
        test_config["overwrite_output_dir"] = True
        test_config.pop("resume_from_checkpoint", None)

        # Run micro-batch training
        workspace.inject_files(**{FT_DEBUG_YAML_FILE_NAME: yaml.dump(test_config, default_flow_style=False)})
        try:
            training_result = workspace.run(
                env=env,
                entry=f"llamafactory-cli train {FT_DEBUG_YAML_FILE_NAME}",
            )
        finally:
            workspace.remove_files([FT_DEBUG_YAML_FILE_NAME, FT_TEST_PARAMS_FILE_NAME])
            FBWorkspace._remove_workspace_tree(debug_output_path)  # noqa: SLF001

        # Parse and store structured execution output (reduces ~36k tokens to ~500)
        raw_stdout = training_result.stdout or ""
        result.raw_stdout = raw_stdout  # Keep full log for UI
        result.execution_output = self._parse_execution_log(raw_stdout, training_result.exit_code)

        # Check results
        progress_indicators = ["train_loss", "Training:", "Epoch", "loss:", "step"]
        has_progress = any(ind.lower() in raw_stdout.lower() for ind in progress_indicators)

        if training_result.exit_code == 0 and has_progress:
            logger.info("Micro-batch test passed")
            result.success = True
        else:
            result.success = False
            result.errors.append(f"Micro-batch test failed (exit_code={training_result.exit_code})")

        return result
