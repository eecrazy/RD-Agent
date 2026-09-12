"""Auditable evidence for full-dataset formal fine-tuning runs.

The iterative FT workflow also performs tiny debug runs.  Model files alone are
therefore not proof that a checkpoint consumed the paper-sized dataset or that
the trainer reached the end of its declared schedule.  This module defines the
strict, filesystem-backed contract used by the matrix runner, checkpoint
selection, and the one-shot final-test protocol.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

FORMAL_EXPECTED_SAMPLES_ENV = "FT_FORMAL_EXPECTED_SAMPLES"
FORMAL_METHOD_LOCK_ENV = "FT_FORMAL_METHOD_LOCK_PATH"
FORMAL_TRAINING_EVIDENCE_FILE = "formal_training_evidence.json"
FORMAL_TRAINING_EVIDENCE_SCHEMA_VERSION = 1
_FORBIDDEN_SAMPLE_LIMIT_KEYS = {
    "max_eval_samples",
    "max_predict_samples",
    "max_samples",
    "max_train_samples",
}
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_FULL_WEIGHTS = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
)
_TRAINING_POLICIES = {"paper", "full", "lora", "rslora"}


class FormalTrainingEvidenceError(RuntimeError):
    """Raised when a workspace cannot prove a complete formal training run."""


def enforce_formal_training_method_lock(
    lock_path: str | Path,
    *,
    experiment_id: str,
    training_policy: str,
    training_method: str,
) -> dict[str, Any]:
    """Atomically bind one matrix task to its first valid formal method.

    The paper policy lets the agent choose Full SFT or ordinary LoRA.  Once a
    task's first complete-dataset configuration passes the pre-GPU contract,
    later iterations must retain that choice.  A separate advisory lock makes
    concurrent pipeline attempts deterministic.
    """
    path = Path(lock_path).expanduser().resolve()
    if not experiment_id.strip():
        raise FormalTrainingEvidenceError("Method lock requires a non-empty experiment id")
    policy = _normalized_policy(training_policy)
    if training_method not in _TRAINING_POLICIES - {"paper"}:
        raise FormalTrainingEvidenceError(f"Unsupported method lock value: {training_method!r}")
    if policy == "paper" and training_method not in {"full", "lora"}:
        raise FormalTrainingEvidenceError("Paper method lock must be Full SFT or ordinary LoRA")
    if policy != "paper" and training_method != policy:
        raise FormalTrainingEvidenceError(
            f"Method lock {training_method!r} does not match controlled policy {policy!r}",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    advisory_path = path.with_name(path.name + ".lock")
    with advisory_path.open("a+", encoding="utf-8") as advisory:
        fcntl.flock(advisory.fileno(), fcntl.LOCK_EX)
        try:
            if path.is_file():
                return validate_formal_training_method_lock(
                    path,
                    experiment_id=experiment_id,
                    training_policy=policy,
                    training_method=training_method,
                )
            if path.exists():
                raise FormalTrainingEvidenceError(f"Method lock path is not a regular file: {path}")

            record = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "training_policy": policy,
                "training_method": training_method,
            }
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=path.name + ".",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                temporary.replace(path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return record
        finally:
            fcntl.flock(advisory.fileno(), fcntl.LOCK_UN)


def validate_formal_training_method_lock(
    lock_path: str | Path,
    *,
    experiment_id: str,
    training_policy: str,
    training_method: str,
) -> dict[str, Any]:
    """Validate an existing task method lock without creating or changing it."""
    path = Path(lock_path).expanduser().resolve()
    record = _read_json(path, dict)
    policy = _normalized_policy(training_policy)
    expected = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "training_policy": policy,
        "training_method": training_method,
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise FormalTrainingEvidenceError(
            "Formal training method lock mismatch: " + ", ".join(mismatches),
        )
    created_at = record.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise FormalTrainingEvidenceError("Formal training method lock has no creation timestamp")
    return record


def formal_expected_samples(environment: Mapping[str, str] | None = None) -> int | None:
    """Return the enabled formal sample contract, or ``None`` outside a matrix run."""
    values = os.environ if environment is None else environment
    raw = values.get(FORMAL_EXPECTED_SAMPLES_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        expected = int(raw)
    except ValueError as error:
        raise FormalTrainingEvidenceError(
            f"{FORMAL_EXPECTED_SAMPLES_ENV} must be a positive integer; got {raw!r}",
        ) from error
    if expected < 1:
        raise FormalTrainingEvidenceError(
            f"{FORMAL_EXPECTED_SAMPLES_ENV} must be a positive integer; got {raw!r}",
        )
    return expected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FormalTrainingEvidenceError(f"Unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalTrainingEvidenceError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, expected_type):
        raise FormalTrainingEvidenceError(
            f"{path} must contain a JSON {expected_type.__name__}; got {type(value).__name__}",
        )
    return value


def _read_training_plan(path: Path) -> tuple[dict[str, Any], str]:
    try:
        source = path.read_text(encoding="utf-8")
        value = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as error:
        raise FormalTrainingEvidenceError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FormalTrainingEvidenceError(f"{path} must contain a YAML mapping")
    return value, source


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalTrainingEvidenceError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FormalTrainingEvidenceError(f"{label} must be a positive number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FormalTrainingEvidenceError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FormalTrainingEvidenceError(f"{label} must be a non-negative integer")
    return value


def _forbidden_limit_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if str(key).strip().lower() in _FORBIDDEN_SAMPLE_LIMIT_KEYS:
                paths.append(label)
            paths.extend(_forbidden_limit_paths(child, label))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_limit_paths(child, f"{prefix}[{index}]"))
    return paths


def _training_method(plan: dict[str, Any]) -> str:
    finetuning_type = str(plan.get("finetuning_type", "")).strip().lower()
    if finetuning_type == "full":
        if "use_rslora" in plan:
            raise FormalTrainingEvidenceError("Full SFT must omit use_rslora")
        return "full"
    if finetuning_type == "lora":
        use_rslora = plan.get("use_rslora", False)
        if not isinstance(use_rslora, bool):
            raise FormalTrainingEvidenceError("use_rslora must be a boolean")
        return "rslora" if use_rslora else "lora"
    raise FormalTrainingEvidenceError("finetuning_type must be full or lora")


def _normalized_policy(policy: str | None) -> str:
    raw = policy if policy is not None else os.environ.get("FT_TRAINING_POLICY", "paper")
    active = str(raw).strip().lower().replace("-", "")
    if active not in _TRAINING_POLICIES:
        raise FormalTrainingEvidenceError(f"Unsupported formal training policy: {raw!r}")
    return active


def _validate_method_policy(plan: dict[str, Any], policy: str) -> str:
    method = _training_method(plan)
    if plan.get("use_dora", False) is not False:
        raise FormalTrainingEvidenceError("use_dora must be false or omitted")
    if plan.get("quantization_bit") is not None:
        raise FormalTrainingEvidenceError(
            "quantization_bit must be null or omitted; QLoRA/quantized training is not allowed",
        )
    if policy == "paper" and method not in {"full", "lora"}:
        raise FormalTrainingEvidenceError("Paper runs must use Full SFT or ordinary LoRA")
    if policy != "paper" and method != policy:
        raise FormalTrainingEvidenceError(
            f"Training method {method!r} does not match policy {policy!r}",
        )
    return method


def _dataset_names(
    value: Any,
    *,
    role: str = "training",
    required: bool = True,
) -> list[str]:
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        names = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if len(names) != len(value):
            raise FormalTrainingEvidenceError(f"{role} dataset entries must be non-empty strings")
    else:
        names = []
    if not names and required:
        raise FormalTrainingEvidenceError(f"train.yaml must select at least one {role} dataset")
    if len(names) != len(set(names)):
        raise FormalTrainingEvidenceError(f"train.yaml contains duplicate {role} dataset names")
    return names


def _workspace_relative(path: Path, workspace: Path, label: str) -> str:
    try:
        return str(path.absolute().relative_to(workspace.absolute()))
    except ValueError as error:
        raise FormalTrainingEvidenceError(f"{label} must stay inside the formal workspace: {path}") from error


def _dataset_inventory(
    workspace_path: Path,
    plan: dict[str, Any],
    *,
    dataset_key: str = "dataset",
    role: str = "training",
) -> tuple[list[Any], list[dict[str, Any]], Path]:
    workspace = workspace_path.absolute()
    raw_dataset_dir = plan.get("dataset_dir", ".")
    if not isinstance(raw_dataset_dir, str) or not raw_dataset_dir.strip():
        raise FormalTrainingEvidenceError("dataset_dir must be a non-empty path string")
    dataset_dir = Path(raw_dataset_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = workspace / dataset_dir
    _workspace_relative(dataset_dir, workspace, "dataset_dir")
    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info = _read_json(dataset_info_path, dict)

    all_records: list[Any] = []
    inventory: list[dict[str, Any]] = []
    for name in _dataset_names(plan.get(dataset_key), role=role):
        registration = dataset_info.get(name)
        if not isinstance(registration, dict):
            raise FormalTrainingEvidenceError(f"Dataset {name!r} is absent from {dataset_info_path}")
        file_name = registration.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise FormalTrainingEvidenceError(f"Dataset {name!r} has no JSON file_name")
        data_path = Path(file_name).expanduser()
        if not data_path.is_absolute():
            data_path = dataset_dir / data_path
        relative_path = _workspace_relative(data_path, workspace, f"Dataset {name!r}")
        records = _read_json(data_path, list)
        all_records.extend(records)
        inventory.append(
            {
                "name": name,
                "path": relative_path,
                "samples": len(records),
                "sha256": _sha256(data_path),
            },
        )
    return all_records, inventory, dataset_info_path


def _validation_contract(
    workspace_path: Path,
    plan: dict[str, Any],
    *,
    training_records: list[Any],
    training_datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that validation cannot reduce or leak the formal training set."""
    val_size = plan.get("val_size")
    if val_size is not None:
        if isinstance(val_size, bool) or not isinstance(val_size, (int, float)):
            raise FormalTrainingEvidenceError("Formal train.yaml val_size must be numeric zero or omitted")
        if not math.isfinite(float(val_size)) or val_size < 0:
            raise FormalTrainingEvidenceError("Formal train.yaml val_size must be numeric zero or omitted")
        if val_size > 0:
            raise FormalTrainingEvidenceError(
                "Formal train.yaml cannot set a positive val_size: it would split examples out of "
                "the required complete training set; use val_size: 0 and an independent eval_dataset",
            )

    strategy_keys = [key for key in ("eval_strategy", "evaluation_strategy") if key in plan]
    strategy_values: list[str] = []
    for key in strategy_keys:
        value = plan[key]
        if not isinstance(value, str) or not value.strip():
            raise FormalTrainingEvidenceError(f"{key} must be a non-empty string when present")
        strategy_values.append(value.strip().lower())
    if len(set(strategy_values)) > 1:
        raise FormalTrainingEvidenceError("eval_strategy and evaluation_strategy disagree")
    strategy = strategy_values[0] if strategy_values else "no"

    do_eval = plan.get("do_eval", False)
    load_best = plan.get("load_best_model_at_end", False)
    if not isinstance(do_eval, bool):
        raise FormalTrainingEvidenceError("do_eval must be a boolean when present")
    if not isinstance(load_best, bool):
        raise FormalTrainingEvidenceError("load_best_model_at_end must be a boolean when present")
    evaluation_enabled = do_eval or load_best or strategy not in {"no", "none"}

    eval_names = _dataset_names(
        plan.get("eval_dataset"),
        role="validation",
        required=False,
    )
    if evaluation_enabled and not eval_names:
        raise FormalTrainingEvidenceError(
            "Formal training enables evaluation but has no independent eval_dataset; "
            "set val_size: 0 and register a separate validation dataset",
        )

    validation_records: list[Any] = []
    validation_datasets: list[dict[str, Any]] = []
    if eval_names:
        training_names = {str(item["name"]) for item in training_datasets}
        duplicate_names = training_names.intersection(eval_names)
        if duplicate_names:
            raise FormalTrainingEvidenceError(
                "Training and validation dataset names overlap: " + ", ".join(sorted(duplicate_names)),
            )
        validation_records, validation_datasets, _ = _dataset_inventory(
            workspace_path,
            plan,
            dataset_key="eval_dataset",
            role="validation",
        )
        if not validation_records:
            raise FormalTrainingEvidenceError("Independent eval_dataset must contain at least one sample")

        training_paths = {str(item["path"]) for item in training_datasets}
        validation_paths = {str(item["path"]) for item in validation_datasets}
        duplicate_paths = training_paths.intersection(validation_paths)
        if duplicate_paths:
            raise FormalTrainingEvidenceError(
                "Training and validation datasets resolve to the same file(s): "
                + ", ".join(sorted(duplicate_paths)),
            )

        training_fingerprints = {_canonical_sha256(record) for record in training_records}
        overlapping_records = sum(
            1 for record in validation_records if _canonical_sha256(record) in training_fingerprints
        )
        if overlapping_records:
            raise FormalTrainingEvidenceError(
                f"Independent eval_dataset overlaps the training data by {overlapping_records} record(s)",
            )

    return {
        "val_size": 0,
        "eval_strategy": strategy,
        "do_eval": do_eval,
        "load_best_model_at_end": load_best,
        "source": "independent_eval_dataset" if eval_names else "none",
        "sample_count": len(validation_records),
        "datasets": validation_datasets,
    }


def load_formal_training_records(workspace_path: str | Path) -> list[Any]:
    """Load exactly the records selected by ``train.yaml`` (never eval_dataset)."""
    workspace = Path(workspace_path)
    plan, _ = _read_training_plan(workspace / "train.yaml")
    records, _, _ = _dataset_inventory(workspace, plan)
    return records


def _input_facts(
    workspace_path: Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    require_runtime_contract: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    workspace = workspace_path.absolute()
    if expected_samples < 1:
        raise FormalTrainingEvidenceError("expected_samples must be positive")
    config_path = workspace / "train.yaml"
    plan, config_source = _read_training_plan(config_path)

    forbidden = _forbidden_limit_paths(plan)
    if forbidden:
        raise FormalTrainingEvidenceError(
            "Formal train.yaml contains sample truncation key(s): " + ", ".join(forbidden),
        )
    configured_max_steps = plan.get("max_steps")
    if configured_max_steps is not None:
        if isinstance(configured_max_steps, bool) or not isinstance(configured_max_steps, (int, float)):
            raise FormalTrainingEvidenceError("max_steps must be numeric when present")
        if configured_max_steps > 0:
            raise FormalTrainingEvidenceError(
                "Formal train.yaml must complete its epoch schedule and cannot set a positive max_steps cap",
            )
    if str(plan.get("stage", "")).strip().lower() != "sft":
        raise FormalTrainingEvidenceError("Formal training requires stage: sft")
    if plan.get("do_train") is not True:
        raise FormalTrainingEvidenceError("Formal training requires do_train: true")

    policy = _normalized_policy(training_policy)
    method = _validate_method_policy(plan, policy)
    epochs = _positive_number(plan.get("num_train_epochs"), "num_train_epochs")
    per_device_batch = _positive_int(
        plan.get("per_device_train_batch_size"),
        "per_device_train_batch_size",
    )
    accumulation = _positive_int(
        plan.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
    )
    seed = _nonnegative_int(plan.get("seed"), "seed")
    data_seed = _nonnegative_int(plan.get("data_seed"), "data_seed")

    output_dir_value = plan.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise FormalTrainingEvidenceError("Formal training requires an explicit output_dir")
    output_dir = Path(output_dir_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    if output_dir.absolute() != (workspace / "output").absolute():
        raise FormalTrainingEvidenceError("Formal training output_dir must resolve to workspace/output")

    records, datasets, dataset_info_path = _dataset_inventory(workspace, plan)
    validation = _validation_contract(
        workspace,
        plan,
        training_records=records,
        training_datasets=datasets,
    )
    if len(records) != expected_samples:
        raise FormalTrainingEvidenceError(
            f"Formal training dataset has {len(records)} samples; expected exactly {expected_samples}",
        )
    data_stats_path = workspace / "data_stats.json"
    data_stats = _read_json(data_stats_path, dict)
    total_samples = data_stats.get("total_samples")
    if isinstance(total_samples, bool) or not isinstance(total_samples, int):
        raise FormalTrainingEvidenceError("data_stats.total_samples must be an integer")
    if total_samples != expected_samples:
        raise FormalTrainingEvidenceError(
            f"data_stats.total_samples is {total_samples}; expected exactly {expected_samples}",
        )

    marker_global_batch = None
    marker_world_size = None
    global_match = re.search(r"^# rdagent_global_batch_size: ([0-9]+)$", config_source, re.MULTILINE)
    world_match = re.search(r"^# rdagent_world_size: ([0-9]+)$", config_source, re.MULTILINE)
    if method == "full":
        if (global_match is None) != (world_match is None):
            raise FormalTrainingEvidenceError(
                "Full SFT must provide both runtime logical-batch/world-size markers or neither",
            )
        if require_runtime_contract and (global_match is None or world_match is None):
            raise FormalTrainingEvidenceError("Full SFT is missing the runtime logical-batch/world-size markers")
        if global_match is not None and world_match is not None:
            marker_global_batch = _positive_int(int(global_match.group(1)), "rdagent_global_batch_size")
            marker_world_size = _positive_int(int(world_match.group(1)), "rdagent_world_size")
            if per_device_batch * accumulation * marker_world_size != marker_global_batch:
                raise FormalTrainingEvidenceError("Full SFT runtime batch no longer preserves the logical batch")
        if require_runtime_contract and (
            not isinstance(plan.get("deepspeed"), str) or not plan["deepspeed"].strip()
        ):
            raise FormalTrainingEvidenceError("Full SFT formal evidence requires the injected ZeRO-3 config")

    facts = {
        "experiment_id": experiment_id,
        "training_policy": policy,
        "training_method": method,
        "expected_samples": expected_samples,
        "training_sample_count": len(records),
        "datasets": datasets,
        "validation": validation,
        "configuration": {
            "num_train_epochs": epochs,
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": accumulation,
            "logical_global_batch_size": marker_global_batch,
            "world_size": marker_world_size,
            "seed": seed,
            "data_seed": data_seed,
        },
        "hashes": {
            "train_yaml_sha256": _sha256(config_path),
            "dataset_info_sha256": _sha256(dataset_info_path),
            "data_stats_sha256": _sha256(data_stats_path),
        },
    }
    return facts, plan, config_source


def validate_formal_training_inputs(
    workspace_path: str | Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    require_runtime_contract: bool = True,
) -> dict[str, Any]:
    """Validate the formal data/config contract before a GPU is leased."""
    facts, _, _ = _input_facts(
        Path(workspace_path),
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
        require_runtime_contract=require_runtime_contract,
    )
    return facts


def formal_training_provenance_files(workspace_path: str | Path) -> tuple[str, ...]:
    """Return every relative input/evidence file needed to revalidate a run.

    The list follows the datasets actually selected by ``train.yaml`` rather
    than assuming that the generated training set is always ``data.json``.
    """
    workspace = Path(workspace_path).absolute()
    plan, _ = _read_training_plan(workspace / "train.yaml")
    _, datasets, dataset_info_path = _dataset_inventory(workspace, plan)
    eval_names = _dataset_names(
        plan.get("eval_dataset"),
        role="validation",
        required=False,
    )
    validation_datasets: list[dict[str, Any]] = []
    if eval_names:
        _, validation_datasets, _ = _dataset_inventory(
            workspace,
            plan,
            dataset_key="eval_dataset",
            role="validation",
        )
    paths = [
        "train.yaml",
        "data_stats.json",
        _workspace_relative(dataset_info_path, workspace, "dataset_info.json"),
        *(str(dataset["path"]) for dataset in datasets),
        *(str(dataset["path"]) for dataset in validation_datasets),
        FORMAL_TRAINING_EVIDENCE_FILE,
    ]
    unique: list[str] = []
    for relative in paths:
        if relative not in unique:
            unique.append(relative)
    for relative in unique:
        path = workspace / relative
        if not path.is_file():
            raise FormalTrainingEvidenceError(f"Formal training provenance file is missing: {path}")
    return tuple(unique)


def _completed_epoch(state: dict[str, Any], global_step: int) -> float:
    candidates: list[float] = []
    value = state.get("epoch")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        candidates.append(float(value))
    history = state.get("log_history")
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict) or entry.get("step") != global_step:
                continue
            value = entry.get("epoch")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                candidates.append(float(value))
    if not candidates:
        raise FormalTrainingEvidenceError("trainer_state has no final completed epoch")
    return max(candidates)


def _validate_output_method(output_path: Path, method: str) -> None:
    if method == "full":
        if not (output_path / "config.json").is_file() or not any(
            any(output_path.glob(pattern)) for pattern in _FULL_WEIGHTS
        ):
            raise FormalTrainingEvidenceError("Full SFT output has no complete full-model artifact")
        return

    adapter_config = _read_json(output_path / "adapter_config.json", dict)
    if not any((output_path / pattern).is_file() for pattern in _ADAPTER_WEIGHTS):
        raise FormalTrainingEvidenceError("LoRA output has no adapter weights")
    actual_rslora = adapter_config.get("use_rslora", False)
    if not isinstance(actual_rslora, bool):
        raise FormalTrainingEvidenceError("adapter_config.use_rslora must be a boolean")
    expected_rslora = method == "rslora"
    if actual_rslora != expected_rslora or adapter_config.get("use_dora", False) is not False:
        raise FormalTrainingEvidenceError("Output adapter method does not match the formal train.yaml")


def _evidence_content(
    workspace_path: Path,
    *,
    output_path: Path,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
) -> dict[str, Any]:
    facts, plan, _ = _input_facts(
        workspace_path,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
    )
    state_path = output_path / "trainer_state.json"
    state = _read_json(state_path, dict)
    global_step = _positive_int(state.get("global_step"), "trainer_state.global_step")
    max_steps = _positive_int(state.get("max_steps"), "trainer_state.max_steps")
    if global_step != max_steps:
        raise FormalTrainingEvidenceError(
            f"Trainer stopped at global_step={global_step}; expected max_steps={max_steps}",
        )

    planned_epochs = float(facts["configuration"]["num_train_epochs"])
    state_epochs = _positive_number(state.get("num_train_epochs"), "trainer_state.num_train_epochs")
    if state_epochs + 1e-9 < planned_epochs or state_epochs > math.ceil(planned_epochs) + 1e-9:
        raise FormalTrainingEvidenceError(
            f"trainer_state.num_train_epochs={state_epochs} does not cover configured epochs={planned_epochs}",
        )
    completed_epoch = _completed_epoch(state, global_step)
    epoch_per_step = completed_epoch / global_step
    if completed_epoch + epoch_per_step + 1e-9 < planned_epochs:
        raise FormalTrainingEvidenceError(
            f"Trainer completed epoch={completed_epoch}; configured epochs={planned_epochs}",
        )
    if completed_epoch > planned_epochs + epoch_per_step + 1e-9:
        raise FormalTrainingEvidenceError(
            f"Trainer overran the configured epoch schedule: {completed_epoch} > {planned_epochs}",
        )

    actual_batch = _positive_int(state.get("train_batch_size"), "trainer_state.train_batch_size")
    planned_batch = int(facts["configuration"]["per_device_train_batch_size"])
    if actual_batch != planned_batch:
        raise FormalTrainingEvidenceError(
            f"trainer_state.train_batch_size={actual_batch} does not match train.yaml={planned_batch}",
        )
    _validate_output_method(output_path, str(facts["training_method"]))

    facts["completion"] = {
        "global_step": global_step,
        "max_steps": max_steps,
        "configured_epochs": planned_epochs,
        "trainer_num_train_epochs": state_epochs,
        "completed_epoch": completed_epoch,
        "train_batch_size": actual_batch,
    }
    facts["hashes"]["trainer_state_sha256"] = _sha256(state_path)
    # Record the exact model/base selection without expanding the artifact with
    # a multi-gigabyte weight hash.  Checkpoint identity is signed separately by
    # the validation/final-test protocol.
    facts["model_name_or_path"] = str(plan.get("model_name_or_path", ""))
    return facts


def make_formal_training_evidence(
    workspace_path: str | Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a signed evidence document after a successful trainer exit."""
    workspace = Path(workspace_path)
    output = Path(output_path) if output_path is not None else workspace / "output"
    content = _evidence_content(
        workspace,
        output_path=output,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
    )
    artifact: dict[str, Any] = {
        "schema_version": FORMAL_TRAINING_EVIDENCE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **content,
    }
    artifact["evidence_signature"] = _canonical_sha256(artifact)
    return artifact


def validate_formal_training_evidence(
    workspace_path: str | Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute every fact and reject missing, stale, or edited evidence."""
    workspace = Path(workspace_path)
    artifact = _read_json(workspace / FORMAL_TRAINING_EVIDENCE_FILE, dict)
    if artifact.get("schema_version") != FORMAL_TRAINING_EVIDENCE_SCHEMA_VERSION:
        raise FormalTrainingEvidenceError("Unsupported formal training evidence schema")
    signature = artifact.get("evidence_signature")
    signed = {key: value for key, value in artifact.items() if key != "evidence_signature"}
    if not isinstance(signature, str) or signature != _canonical_sha256(signed):
        raise FormalTrainingEvidenceError("Formal training evidence signature is invalid")
    created_at = artifact.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise FormalTrainingEvidenceError("Formal training evidence has no creation timestamp")

    output = Path(output_path) if output_path is not None else workspace / "output"
    current = _evidence_content(
        workspace,
        output_path=output,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
    )
    recorded = {
        key: value
        for key, value in artifact.items()
        if key not in {"schema_version", "created_at", "evidence_signature"}
    }
    if recorded != current:
        mismatches = sorted(
            key for key in set(recorded) | set(current) if recorded.get(key) != current.get(key)
        )
        raise FormalTrainingEvidenceError(
            "Formal training evidence no longer matches workspace files: " + ", ".join(mismatches),
        )
    return artifact


def validate_recorded_formal_training_evidence(
    workspace_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an evidence file using its recorded contract.

    Callers enforcing a paper matrix must use
    :func:`validate_formal_training_evidence` with their independently known
    experiment/sample/policy values.  This helper is for copying or restoring
    an already-bound provenance snapshot without trusting a session pickle.
    """
    workspace = Path(workspace_path)
    artifact = _read_json(workspace / FORMAL_TRAINING_EVIDENCE_FILE, dict)
    expected_samples = _positive_int(artifact.get("expected_samples"), "evidence.expected_samples")
    experiment_id = artifact.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise FormalTrainingEvidenceError("evidence.experiment_id must be a non-empty string")
    policy_value = artifact.get("training_policy")
    if not isinstance(policy_value, str):
        raise FormalTrainingEvidenceError("evidence.training_policy must be a string")
    return validate_formal_training_evidence(
        workspace,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=policy_value,
        output_path=output_path,
    )


def scan_durable_formal_training_evidence(
    task_root: str | Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    require_visible_evidence: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find independently revalidated durable formal-training snapshots.

    Each accepted record couples one immutable provenance directory with the
    corresponding durable model output.  Invalid workspaces are returned as
    diagnostics instead of making a different valid candidate unusable.
    """
    task = Path(task_root)
    workspace_root = task / "workspace"
    durable_root = workspace_root / ".ft_model_checkpoints"
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for provenance in sorted(durable_root.glob("*/formal_training")):
        workspace_id = provenance.parent.name
        output = provenance.parent / "output"
        try:
            if not provenance.is_dir():
                raise FormalTrainingEvidenceError(f"Durable provenance is not a directory: {provenance}")
            if not output.is_dir():
                raise FormalTrainingEvidenceError(f"Durable model output is missing: {output}")
            visible_evidence = workspace_root / workspace_id / FORMAL_TRAINING_EVIDENCE_FILE
            durable_evidence = provenance / FORMAL_TRAINING_EVIDENCE_FILE
            if require_visible_evidence:
                if not visible_evidence.is_file():
                    raise FormalTrainingEvidenceError(
                        f"Visible formal training evidence is missing: {visible_evidence}",
                    )
                if visible_evidence.read_bytes() != durable_evidence.read_bytes():
                    raise FormalTrainingEvidenceError(
                        f"Visible and durable evidence differ for workspace {workspace_id}",
                    )
            artifact = validate_formal_training_evidence(
                provenance,
                expected_samples=expected_samples,
                experiment_id=experiment_id,
                training_policy=training_policy,
                output_path=output,
            )
            formal_training_provenance_files(provenance)
            records.append(
                {
                    "workspace_id": workspace_id,
                    "workspace_path": str(workspace_root / workspace_id),
                    "provenance_path": str(provenance),
                    "output_path": str(output),
                    "evidence_signature": artifact["evidence_signature"],
                    "training_method": artifact["training_method"],
                    "expected_samples": artifact["expected_samples"],
                    "global_step": artifact["completion"]["global_step"],
                    "max_steps": artifact["completion"]["max_steps"],
                },
            )
        except (FormalTrainingEvidenceError, OSError) as error:
            errors.append(f"{workspace_id}: {error}")
    return records, errors
