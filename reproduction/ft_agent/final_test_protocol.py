"""Shared identity and artifact constants for one-shot held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

TEST_RANGE = "[-min(100, len(index_list)//2):]"
FINAL_TEST_FILE = "final_test.json"
FINAL_TEST_SCHEMA_VERSION = 1
SMALL_FILE_HASH_LIMIT = 16 * 1024 * 1024
ADAPTER_WEIGHT_PATTERNS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)
FULL_MODEL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
)
MODEL_WEIGHT_PATTERNS = ADAPTER_WEIGHT_PATTERNS + FULL_MODEL_WEIGHT_PATTERNS
TRAINING_POLICIES = ("paper", "full", "lora", "rslora")
IDENTITY_METADATA_FILES = (
    "adapter_config.json",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tokenizer_config.json",
    "trainer_state.json",
)


def has_model_weights(model_path: Path) -> bool:
    return model_path.is_dir() and any(any(model_path.glob(pattern)) for pattern in MODEL_WEIGHT_PATTERNS)


def _has_weights(model_path: Path, patterns: tuple[str, ...]) -> bool:
    return model_path.is_dir() and any(any(model_path.glob(pattern)) for pattern in patterns)


def model_artifact_type(model_path: Path) -> str | None:
    """Classify a complete non-DoRA fine-tuned artifact."""
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.is_file():
        config: Any = None
        if _has_weights(model_path, ADAPTER_WEIGHT_PATTERNS):
            with suppress(OSError, json.JSONDecodeError):
                config = json.loads(adapter_config.read_text(encoding="utf-8"))
        if isinstance(config, dict):
            use_rslora = config.get("use_rslora", False)
            if (
                str(config.get("peft_type", "")).upper() == "LORA"
                and config.get("use_dora", False) is False
                and isinstance(use_rslora, bool)
            ):
                return "rslora" if use_rslora else "lora"
        return None

    if (model_path / "config.json").is_file() and _has_weights(model_path, FULL_MODEL_WEIGHT_PATTERNS):
        return "full"
    return None


def is_finetuned_model(model_path: Path) -> bool:
    """Return whether *model_path* is a complete supported training artifact."""
    return model_artifact_type(model_path) is not None


def normalize_training_policy(policy: str | None = None) -> str:
    """Return a validated controlled training policy name."""
    raw = policy if policy is not None else os.environ.get("FT_TRAINING_POLICY", "paper")
    active = str(raw).strip().lower().replace("-", "")
    if active not in TRAINING_POLICIES:
        supported = ", ".join(TRAINING_POLICIES)
        message = f"FT_TRAINING_POLICY must be one of: {supported}; got {raw!r}"
        raise ValueError(message)
    return active


def is_policy_compliant_model(model_path: Path, policy: str | None = None) -> bool:
    """Match an artifact to the active controlled training policy."""
    active = normalize_training_policy(policy)
    artifact_type = model_artifact_type(model_path)
    if active == "paper":
        return artifact_type in {"full", "lora"}
    return artifact_type == active


def is_policy_compliant_selection(
    model_path: Path,
    candidate_source: str | None,
    policy: str | None = None,
) -> bool:
    """Allow the untouched baseline or require the selected trained method."""
    return has_model_weights(model_path) and (
        candidate_source == "baseline" or is_policy_compliant_model(model_path, policy)
    )


def is_rs_lora_model(model_path: Path) -> bool:
    """Historical convenience predicate retained for audit compatibility."""
    return model_artifact_type(model_path) == "rslora"


def _has_result_payload(value: Any) -> bool:
    """Distinguish the empty compatibility placeholders from an evaluated result."""
    if value is None:
        return False
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return bool(value)
    return True


def trace_protocol_pollution(trace: Any) -> tuple[str, ...]:
    """Return search-time held-out artifacts embedded anywhere in an FT trace.

    Validation-only scenarios retain an empty ``baseline_benchmark_score_test``
    compatibility attribute, so only non-empty payloads count as evidence that
    held-out evaluation actually happened.
    """
    locations: list[str] = []
    scenario = getattr(trace, "scen", None)
    if _has_result_payload(getattr(scenario, "baseline_benchmark_score_test", None)):
        locations.append("scen.baseline_benchmark_score_test")

    for index, node in enumerate(getattr(trace, "hist", ())):
        if not isinstance(node, (tuple, list)) or not node:
            continue
        workspace = getattr(node[0], "experiment_workspace", None)
        running_info = getattr(workspace, "running_info", None)
        result = getattr(running_info, "result", None)
        if (
            isinstance(result, dict)
            and "benchmark_test" in result
            and _has_result_payload(result.get("benchmark_test"))
        ):
            locations.append(f"hist[{index}].benchmark_test")
    return tuple(locations)


def model_identity(model_path: Path) -> dict[str, Any]:
    """Return a stable-enough local identity without hashing multi-GB weights."""
    resolved = model_path.resolve()
    identity: dict[str, Any] = {"path": str(resolved)}
    candidates = {resolved / name for name in IDENTITY_METADATA_FILES}
    for pattern in MODEL_WEIGHT_PATTERNS:
        candidates.update(resolved.glob(pattern))
    for path in sorted(candidates, key=lambda item: item.name):
        if not path.is_file():
            continue
        stat = path.stat()
        item: dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= SMALL_FILE_HASH_LIMIT:
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity[path.name] = item
    return identity


def selection_signature(
    source: str,
    history_index: int | None,
    loop_id: int | None,
    model_path: Path,
    *,
    identity_path: Path | None = None,
) -> str:
    """Sign model files and their logical path.

    ``identity_path`` is used only to verify an immutable snapshot after its
    original trainer-owned path has rotated away.  The bytes and metadata are
    always read from ``model_path``.
    """
    model = model_identity(model_path)
    if identity_path is not None:
        model["path"] = str(identity_path.resolve())
    payload = {
        "source": source,
        "history_index": history_index,
        "loop_id": loop_id,
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
