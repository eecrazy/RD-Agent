#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PERF401, PLR0911, PLR0912, TRY003, TRY300, TRY301
"""Evaluate every durable FT checkpoint on validation only, in parallel."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from dotenv import load_dotenv

# Session unpickling imports the configured LiteLLM backend.  Keep that import
# offline and deterministic on hosts without access to LiteLLM's remote cost map.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

SCRIPT_PATH = Path(__file__).resolve()
CHECKPOINT_RE = re.compile(r"^checkpoint-(?P<step>\d+)$")
RETRYABLE_SNAPSHOT_FAILURE_MARKERS = ("exit_code=143", "exit_code=127")

if __package__:
    from .collect_results import json_safe, result_view
    from .final_test_protocol import (
        has_model_weights,
        is_policy_compliant_model,
        is_policy_compliant_selection,
        model_identity,
        normalize_training_policy,
        selection_signature,
    )
    from .responses_adapter import responses_configuration_errors, routed_api_environment
    from .rslora_pairing import (
        PAIRING_TARGET_POLICY,
        pairing_signature_from_status,
        validate_pairing_artifact,
    )
    from .run_matrix import (
        FT_ROOT,
        PYTHON,
        ROOT,
        configure_project_environment,
        duration_seconds,
        safe_id,
        status_write,
        stop_process,
        validate_max_parallel,
        validate_task_formal_training,
        visible_gpus,
    )
    from .validation_selection import (
        LORA_COMPARISON_SELECTION_PROFILE,
        MAIN_SELECTION_PROFILE,
        SELECTION_PROFILES,
        VALIDATION_RANGE,
        ValidationSelectionError,
        make_selection_artifact,
        normalize_selection_profile,
        rank_candidates,
        validate_selection_artifact,
        validation_selection_file,
        validation_sweep_directory,
        validation_sweep_snapshot_file,
    )
else:
    from collect_results import json_safe, result_view
    from final_test_protocol import (
        has_model_weights,
        is_policy_compliant_model,
        is_policy_compliant_selection,
        model_identity,
        normalize_training_policy,
        selection_signature,
    )
    from responses_adapter import responses_configuration_errors, routed_api_environment
    from rslora_pairing import (
        PAIRING_TARGET_POLICY,
        pairing_signature_from_status,
        validate_pairing_artifact,
    )
    from run_matrix import (
        FT_ROOT,
        PYTHON,
        ROOT,
        configure_project_environment,
        duration_seconds,
        safe_id,
        status_write,
        stop_process,
        validate_max_parallel,
        validate_task_formal_training,
        visible_gpus,
    )
    from validation_selection import (
        LORA_COMPARISON_SELECTION_PROFILE,
        MAIN_SELECTION_PROFILE,
        SELECTION_PROFILES,
        VALIDATION_RANGE,
        ValidationSelectionError,
        make_selection_artifact,
        normalize_selection_profile,
        rank_candidates,
        validate_selection_artifact,
        validation_selection_file,
        validation_sweep_directory,
        validation_sweep_snapshot_file,
    )

GPU_TRUST_DIRECTORY = FT_ROOT / "gpu_leases"
GPU_TRUST_REGISTRY = GPU_TRUST_DIRECTORY / "trusted_owners"
GPU_TRUST_REGISTRY_LOCK = GPU_TRUST_DIRECTORY / "trusted_owners.lock"


def _valid_candidate_artifact(
    path: Path,
    source: str | None = None,
    training_policy: str | None = None,
) -> bool:
    """Allow the unchanged base model while enforcing policy on trained outputs."""
    return is_policy_compliant_selection(path, source, training_policy)


@dataclass(frozen=True)
class SweepTarget:
    experiment_id: str
    benchmark: str
    model: str
    benchmark_dataset_path: str
    task_root: Path
    expected_samples: int
    training_policy: str = "paper"
    pairing_artifact_signature: str | None = None
    selection_profile: str = MAIN_SELECTION_PROFILE


def validate_target_pairing(target: SweepTarget) -> None:
    """Revalidate the strict ordinary-LoRA/rsLoRA contract when present."""
    selection_profile = normalize_selection_profile(target.selection_profile)
    if selection_profile == LORA_COMPARISON_SELECTION_PROFILE and target.pairing_artifact_signature is not None:
        raise ValidationSelectionError(
            f"{target.experiment_id}: ordinary-LoRA comparison cannot target an rsLoRA paired run",
        )
    if target.pairing_artifact_signature is None:
        if target.training_policy == PAIRING_TARGET_POLICY:
            raise ValidationSelectionError(
                f"{target.experiment_id}: rsLoRA evaluation requires a signed ordinary-LoRA pairing artifact",
            )
        return
    validate_pairing_artifact(
        target.task_root,
        experiment_id=target.experiment_id,
        expected_samples=target.expected_samples,
        expected_signature=target.pairing_artifact_signature,
    )


@dataclass(frozen=True)
class CheckpointCandidate:
    candidate_id: str
    source: str
    workspace_id: str | None
    checkpoint_step: int | None
    model_path: Path
    selection_signature: str
    formal_training_evidence_signature: str | None

    def identity(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "workspace_id": self.workspace_id,
            "checkpoint_step": self.checkpoint_step,
            "model_path": str(self.model_path),
            "selection_signature": self.selection_signature,
            "formal_training_evidence_signature": self.formal_training_evidence_signature,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", help="Matrix run name or path")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--only", default=None, help="Regex applied to experiment ids")
    parser.add_argument(
        "--selection-profile",
        choices=SELECTION_PROFILES,
        default=MAIN_SELECTION_PROFILE,
        help="Independent checkpoint-selection contract to evaluate",
    )
    parser.add_argument(
        "--candidate-only",
        default=None,
        help="Regex applied to validation candidate ids after checkpoint discovery",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse matching successful validation results")
    parser.add_argument(
        "--retry-failed-snapshots",
        action="store_true",
        help="Retry signed candidate-local snapshots interrupted by retryable infrastructure failures",
    )
    parser.add_argument("--snapshot-only", action="store_true", help="Never finalize a checkpoint selection")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Evaluate only the unchanged base model. This is safe before formal training finishes, "
            "but requires --snapshot-only and the main selection profile."
        ),
    )
    parser.add_argument("--task-timeout", default="3h")
    parser.add_argument("--no-baseline", action="store_true", help="Do not evaluate the unchanged base model")
    parser.add_argument("--no-final-outputs", action="store_true", help="Do not evaluate workspace output directories")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker_spec and not args.matrix_run:
        parser.error("--matrix-run is required")
    if args.retry_failed_snapshots and not args.resume:
        parser.error("--retry-failed-snapshots requires --resume")
    if args.baseline_only and not args.snapshot_only:
        parser.error("--baseline-only requires --snapshot-only")
    if args.baseline_only and args.selection_profile != MAIN_SELECTION_PROFILE:
        parser.error("--baseline-only is available only for the main selection profile")
    if args.baseline_only and args.no_baseline:
        parser.error("--baseline-only cannot be combined with --no-baseline")
    return args


def resolve_run_root(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = FT_ROOT / "logs" / "paper-matrix" / candidate
    return candidate.resolve()


def load_targets(
    run_root: Path,
    only: str | None = None,
    *,
    selection_profile: str = MAIN_SELECTION_PROFILE,
) -> list[SweepTarget]:
    selection_profile = normalize_selection_profile(selection_profile)
    matrix_path = run_root / "matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError(f"Matrix manifest is missing: {matrix_path}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    training_policy = normalize_training_policy(matrix.get("training_policy", "paper"))
    pattern = re.compile(only) if only else None
    targets: list[SweepTarget] = []
    errors: list[str] = []
    for task in matrix.get("tasks", []):
        experiment_id = str(task["experiment_id"])
        if pattern and not pattern.search(experiment_id):
            continue
        task_root = run_root / safe_id(experiment_id)
        status_path = task_root / "status.json"
        if not status_path.is_file():
            errors.append(f"{experiment_id}: task has not started")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            selection_profile == LORA_COMPARISON_SELECTION_PROFILE
            and status.get("formal_training_method") != "lora"
        ):
            # A matrix may legitimately mix Full SFT and ordinary LoRA tasks.
            # This profile is defined only on the ordinary-LoRA subset.
            continue
        status_policy = normalize_training_policy(status.get("training_policy", training_policy))
        if status_policy != training_policy:
            errors.append(
                f"{experiment_id}: task training policy {status_policy!r} does not match "
                f"matrix policy {training_policy!r}",
            )
            continue
        benchmark_dataset_path = status.get("benchmark_dataset_path")
        if not isinstance(benchmark_dataset_path, str) or not benchmark_dataset_path:
            errors.append(f"{experiment_id}: pinned benchmark dataset path is missing")
            continue
        expected_samples = status.get("formal_expected_samples")
        if (
            isinstance(expected_samples, bool)
            or not isinstance(expected_samples, int)
            or expected_samples < 1
            or status.get("data_limit") != expected_samples
            or task.get("formal_expected_samples") != expected_samples
        ):
            errors.append(f"{experiment_id}: formal sample contract is missing or inconsistent")
            continue
        try:
            pairing_signature = pairing_signature_from_status(status)
            provisional_target = SweepTarget(
                experiment_id=experiment_id,
                benchmark=str(status["benchmark"]),
                model=str(status["model"]),
                benchmark_dataset_path=benchmark_dataset_path,
                task_root=task_root.resolve(),
                expected_samples=expected_samples,
                training_policy=training_policy,
                pairing_artifact_signature=pairing_signature,
                selection_profile=selection_profile,
            )
            validate_target_pairing(provisional_target)
        except (KeyError, RuntimeError, ValueError) as error:
            errors.append(f"{experiment_id}: invalid rsLoRA pairing contract: {error}")
            continue
        targets.append(
            provisional_target,
        )
    if errors:
        raise RuntimeError("Validation sweep targets are not ready:\n- " + "\n- ".join(errors))
    if not targets:
        raise RuntimeError("No validation sweep targets were selected")
    return targets


def _candidate_id(
    source: str,
    workspace_id: str | None,
    step: int | None,
    signature: str,
    formal_training_evidence_signature: str | None,
) -> str:
    readable = "__".join(
        part
        for part in (
            source,
            workspace_id,
            f"step-{step}" if step is not None else None,
        )
        if part
    )
    digest = hashlib.sha256(
        f"{readable}|{signature}|{formal_training_evidence_signature or 'baseline'}".encode(),
    ).hexdigest()[:12]
    return f"{safe_id(readable)}__{digest}"


def _weight_inode_key(path: Path) -> tuple[tuple[int, int, int], ...]:
    files: set[Path] = set()
    for pattern in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ):
        files.update(path.glob(pattern))
    return tuple(sorted((item.stat().st_dev, item.stat().st_ino, item.stat().st_size) for item in files))


def _make_candidate(
    path: Path,
    *,
    source: str,
    workspace_id: str | None,
    checkpoint_step: int | None,
    formal_training_evidence_signature: str | None,
) -> CheckpointCandidate:
    resolved = path.resolve()
    signature = selection_signature("validation_sweep", None, checkpoint_step, resolved)
    return CheckpointCandidate(
        candidate_id=_candidate_id(
            source,
            workspace_id,
            checkpoint_step,
            signature,
            formal_training_evidence_signature,
        ),
        source=source,
        workspace_id=workspace_id,
        checkpoint_step=checkpoint_step,
        model_path=resolved,
        selection_signature=signature,
        formal_training_evidence_signature=formal_training_evidence_signature,
    )


def discover_baseline_candidates(target: SweepTarget) -> list[CheckpointCandidate]:
    """Discover the immutable base model without requiring completed FT evidence.

    A baseline has not consumed generated training data, so its validation is
    independent of whether the task's formal 2k/full-epoch training has
    finished.  The resulting candidate identity is the same one discovered by
    the normal post-training sweep and can therefore be reused later.
    """
    validate_target_pairing(target)
    baseline = FT_ROOT / "models" / target.model
    if not has_model_weights(baseline):
        return []
    return [
        _make_candidate(
            baseline,
            source="baseline",
            workspace_id=None,
            checkpoint_step=None,
            formal_training_evidence_signature=None,
        ),
    ]


def _checkpoint_directories(
    root: Path,
    pattern: str,
    *,
    workspace_parent: int,
) -> list[tuple[str, int, Path]]:
    paths: list[tuple[str, int, Path]] = []
    for path in root.glob(pattern):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match and has_model_weights(path):
            paths.append(
                (
                    path.parents[workspace_parent].name,
                    int(match.group("step")),
                    path,
                ),
            )
    return sorted(paths, key=lambda item: (item[0], item[1]))


def _formal_training_records(target: SweepTarget) -> dict[str, dict[str, Any]]:
    """Revalidate and index the task's durable, full-schedule training proof."""
    validate_target_pairing(target)
    records, errors = validate_task_formal_training(
        target.task_root,
        expected_samples=target.expected_samples,
        experiment_id=target.experiment_id,
        training_policy=target.training_policy,
        require_visible_evidence=False,
    )
    if not records:
        detail = "; ".join(errors) if errors else "no durable evidence directories were found"
        raise RuntimeError(f"{target.experiment_id}: no valid formal training evidence: {detail}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        workspace_id = str(record["workspace_id"])
        if workspace_id in indexed:
            raise RuntimeError(f"{target.experiment_id}: duplicate formal evidence for {workspace_id}")
        indexed[workspace_id] = record

    try:
        status = json.loads((target.task_root / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{target.experiment_id}: task status is unavailable: {error}") from error
    locked_methods = {record["training_method"] for record in records}
    method_locks = {json.dumps(record["formal_training_method_lock"], sort_keys=True) for record in records}
    if len(locked_methods) != 1 or status.get("formal_training_method") not in locked_methods:
        raise RuntimeError(f"{target.experiment_id}: task status does not match the formal method lock")
    if (
        normalize_selection_profile(target.selection_profile) == LORA_COMPARISON_SELECTION_PROFILE
        and locked_methods != {"lora"}
    ):
        raise RuntimeError(
            f"{target.experiment_id}: LoRA comparison requires an ordinary-LoRA formal method lock",
        )
    if (
        len(method_locks) != 1
        or json.dumps(status.get("formal_training_method_lock"), sort_keys=True) not in method_locks
    ):
        raise RuntimeError(f"{target.experiment_id}: task status contains a stale formal method lock")
    return indexed


def _same_model_artifact(candidate: Path, reference: Path, checkpoint_step: int | None) -> bool:
    """Compare model content while normalizing its logical path.

    The trainer may serialize ``trainer_state.json`` independently into the
    final output and the last checkpoint.  Those small files can have identical
    bytes but different mtimes.  Their SHA-256 digest is the identity; mtimes
    remain part of the identity for large weights that are deliberately not
    hashed.
    """
    if not has_model_weights(candidate) or not has_model_weights(reference):
        return False
    expected = selection_signature("validation_sweep", None, checkpoint_step, reference)
    actual = selection_signature(
        "validation_sweep",
        None,
        checkpoint_step,
        candidate,
        identity_path=reference,
    )
    if actual == expected:
        return True

    def content_identity(path: Path) -> dict[str, Any]:
        identity = model_identity(path)
        identity["path"] = str(reference.resolve())
        for name, item in tuple(identity.items()):
            if not isinstance(item, dict) or "sha256" not in item:
                continue
            normalized = dict(item)
            normalized.pop("mtime_ns", None)
            identity[name] = normalized
        return identity

    return content_identity(candidate) == content_identity(reference)


def _has_formal_training_provenance(
    candidate: CheckpointCandidate,
    records: dict[str, dict[str, Any]],
) -> bool:
    """Bind a candidate to one revalidated 2k/full-epoch durable evidence record."""
    if candidate.source == "baseline":
        return candidate.formal_training_evidence_signature is None
    if candidate.workspace_id is None:
        return False
    record = records.get(candidate.workspace_id)
    if record is None or candidate.formal_training_evidence_signature != record["evidence_signature"]:
        return False
    max_steps = int(record["max_steps"])
    if candidate.checkpoint_step is not None and candidate.checkpoint_step != max_steps:
        return False
    output = Path(record["output_path"]).resolve()
    reference = output
    if candidate.checkpoint_step is not None:
        checkpoint = output / f"checkpoint-{candidate.checkpoint_step}"
        if checkpoint.is_dir() and has_model_weights(checkpoint):
            reference = checkpoint
    return _same_model_artifact(candidate.model_path, reference, candidate.checkpoint_step)


def validate_candidate_formal_provenance(
    target: SweepTarget,
    candidate: CheckpointCandidate,
) -> None:
    """Fail closed if formal evidence or its bound model changed after discovery."""
    if candidate.source == "baseline":
        expected = discover_baseline_candidates(target)
        if not expected:
            raise ValidationSelectionError(f"{target.experiment_id}: baseline model is unavailable")
        baseline = expected[0]
        if (
            candidate.candidate_id != baseline.candidate_id
            or candidate.workspace_id is not None
            or candidate.checkpoint_step is not None
            or candidate.formal_training_evidence_signature is not None
            or candidate.selection_signature != baseline.selection_signature
            or not _same_model_artifact(candidate.model_path, baseline.model_path, None)
        ):
            raise ValidationSelectionError(
                f"{target.experiment_id}: baseline candidate identity changed: {candidate.candidate_id}",
            )
        return
    records = _formal_training_records(target)
    if not _has_formal_training_provenance(candidate, records):
        raise ValidationSelectionError(
            f"{target.experiment_id}: candidate is no longer bound to valid formal training evidence: "
            f"{candidate.candidate_id}",
        )


def discover_candidates(
    target: SweepTarget,
    *,
    include_baseline: bool = True,
    include_final_outputs: bool = True,
    include_retryable_failed_snapshots: bool = False,
) -> list[CheckpointCandidate]:
    formal_records = _formal_training_records(target)
    candidates: list[CheckpointCandidate] = []

    # Keep supporting explicit post-run archives, but also consume the durable
    # store written by ``FTWorkspace.create_ws_ckp``.  The latter is the source
    # of truth during a normal matrix run; requiring a separate copy step here
    # would silently drop every intermediate trainer checkpoint.
    archive_root = target.task_root / "checkpoint_archive"
    for workspace_id, step, path in _checkpoint_directories(
        archive_root,
        "*/checkpoint-*",
        workspace_parent=0,
    ):
        if not is_policy_compliant_model(path, target.training_policy):
            continue
        candidates.append(
            _make_candidate(
                path,
                source="checkpoint_archive",
                workspace_id=workspace_id,
                checkpoint_step=step,
                formal_training_evidence_signature=(
                    formal_records.get(workspace_id, {}).get("evidence_signature")
                ),
            ),
        )

    durable_root = target.task_root / "workspace" / ".ft_model_checkpoints"
    for workspace_id, step, path in _checkpoint_directories(
        durable_root,
        "*/output/checkpoint-*",
        workspace_parent=1,
    ):
        if not is_policy_compliant_model(path, target.training_policy):
            continue
        candidates.append(
            _make_candidate(
                path,
                source="durable_checkpoint",
                workspace_id=workspace_id,
                checkpoint_step=step,
                formal_training_evidence_signature=(
                    formal_records.get(workspace_id, {}).get("evidence_signature")
                ),
            ),
        )

    if include_final_outputs:
        # Prefer the durable copy over its live-workspace hard link.  The inode
        # de-duplicator below then suppresses the live duplicate while keeping
        # a distinct final adapter when it is not identical to a saved step.
        for path in sorted(durable_root.glob("*/output")):
            if path.is_dir() and has_model_weights(path) and is_policy_compliant_model(path, target.training_policy):
                candidates.append(
                    _make_candidate(
                        path,
                        source="durable_output",
                        workspace_id=path.parent.name,
                        checkpoint_step=None,
                        formal_training_evidence_signature=(
                            formal_records.get(path.parent.name, {}).get("evidence_signature")
                        ),
                    ),
                )
        for path in sorted((target.task_root / "workspace").glob("*/output")):
            if path.is_dir() and has_model_weights(path) and is_policy_compliant_model(path, target.training_policy):
                candidates.append(
                    _make_candidate(
                        path,
                        source="workspace_output",
                        workspace_id=path.parent.name,
                        checkpoint_step=None,
                        formal_training_evidence_signature=(
                            formal_records.get(path.parent.name, {}).get("evidence_signature")
                        ),
                    ),
                )

    if include_baseline:
        baseline = FT_ROOT / "models" / target.model
        if has_model_weights(baseline):
            candidates.append(
                _make_candidate(
                    baseline,
                    source="baseline",
                    workspace_id=None,
                    checkpoint_step=None,
                    formal_training_evidence_signature=None,
                ),
            )

    # A trainer-owned checkpoint can legitimately rotate away after its
    # validation finishes.  Keep such a candidate selectable when (and only
    # when) its candidate-local snapshot still matches the exact identity that
    # was evaluated.  The live candidate is preferred whenever it still has
    # the original identity.
    candidates.extend(
        _validated_snapshot_candidates(
            target,
            existing_candidate_ids={candidate.candidate_id for candidate in candidates},
            include_retryable_failures=include_retryable_failed_snapshots,
        ),
    )

    # Debug/micro-batch outputs have no durable evidence.  Intermediate
    # checkpoints are also excluded: reported models must have consumed all
    # samples and reached the end of the configured epoch schedule.
    candidates = [
        candidate for candidate in candidates if _has_formal_training_provenance(candidate, formal_records)
    ]

    # LLaMA-Factory commonly hard-links the final output to its last checkpoint.
    # Validate that inode only once, preferring the immutable archive candidate.
    deduplicated: list[CheckpointCandidate] = []
    seen_weights: set[tuple[tuple[int, int, int], ...]] = set()
    for candidate in candidates:
        weight_key = _weight_inode_key(candidate.model_path)
        if weight_key and weight_key in seen_weights:
            continue
        seen_weights.add(weight_key)
        deduplicated.append(candidate)
    return deduplicated


def candidate_set_signature(candidates: list[CheckpointCandidate]) -> str:
    identities = [candidate.identity() for candidate in sorted(candidates, key=lambda item: item.candidate_id)]
    encoded = json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_current_selection_provenance(
    target: SweepTarget,
    artifact: dict[str, Any],
    *,
    include_baseline: bool | None = None,
    include_final_outputs: bool | None = None,
) -> dict[str, Any]:
    """Verify that a signed selection still belongs to the current formal candidate set."""
    selection = validate_selection_artifact(
        artifact,
        experiment_id=target.experiment_id,
        benchmark=target.benchmark,
        model=target.model,
        selection_profile=target.selection_profile,
    )
    validate_target_pairing(target)
    if artifact.get("pairing_artifact_signature") != target.pairing_artifact_signature:
        raise ValidationSelectionError("Validation selection rsLoRA pairing contract changed")
    if artifact.get("benchmark_dataset_path") != target.benchmark_dataset_path:
        raise ValidationSelectionError("Validation selection benchmark dataset changed")
    if artifact.get("expected_samples") != target.expected_samples:
        raise ValidationSelectionError("Validation selection formal sample contract changed")
    if (
        normalize_selection_profile(target.selection_profile) == LORA_COMPARISON_SELECTION_PROFILE
        and artifact.get("candidate_discovery", {}).get("include_baseline") is not False
    ):
        raise ValidationSelectionError("LoRA comparison validation selection must exclude the baseline")

    discovery = artifact.get("candidate_discovery")
    if not isinstance(discovery, dict):
        raise ValidationSelectionError("Validation selection has no signed candidate-discovery contract")
    recorded_baseline = discovery.get("include_baseline")
    recorded_outputs = discovery.get("include_final_outputs")
    candidate_filter = discovery.get("candidate_filter")
    if not isinstance(recorded_baseline, bool) or not isinstance(recorded_outputs, bool):
        raise ValidationSelectionError("Validation selection candidate-discovery flags are invalid")
    if candidate_filter is not None and not isinstance(candidate_filter, str):
        raise ValidationSelectionError("Validation selection candidate filter is invalid")
    if include_baseline is not None and include_baseline != recorded_baseline:
        raise ValidationSelectionError("Validation selection baseline-discovery contract changed")
    if include_final_outputs is not None and include_final_outputs != recorded_outputs:
        raise ValidationSelectionError("Validation selection output-discovery contract changed")

    candidates = discover_candidates(
        target,
        include_baseline=recorded_baseline,
        include_final_outputs=recorded_outputs,
    )
    if candidate_filter is not None:
        candidates = filter_candidates(candidates, re.compile(candidate_filter))
    actual_signature = candidate_set_signature(candidates)
    if artifact.get("candidate_set_signature") != actual_signature:
        raise ValidationSelectionError("Validation selection candidate set is no longer formally eligible")
    if artifact.get("candidate_count") != len(candidates):
        raise ValidationSelectionError("Validation selection candidate count does not match formal provenance")

    candidate_id = selection.get("candidate_id")
    matches = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValidationSelectionError("Selected checkpoint is absent from the formal candidate set")
    expected_identity = matches[0].identity()
    mismatches = [key for key, value in expected_identity.items() if selection.get(key) != value]
    if mismatches:
        raise ValidationSelectionError(
            "Selected checkpoint no longer matches formal provenance: " + ", ".join(mismatches),
        )
    return selection


def filter_candidates(
    candidates: list[CheckpointCandidate],
    pattern: re.Pattern[str] | None,
) -> list[CheckpointCandidate]:
    if pattern is None:
        return candidates
    return [candidate for candidate in candidates if pattern.search(candidate.candidate_id)]


def _candidate_root(target: SweepTarget, candidate: CheckpointCandidate) -> Path:
    return (
        target.task_root
        / validation_sweep_directory(target.selection_profile)
        / "candidates"
        / candidate.candidate_id
    )


def _result_path(target: SweepTarget, candidate: CheckpointCandidate) -> Path:
    return _candidate_root(target, candidate) / "result.json"


def _matching_success(target: SweepTarget, candidate: CheckpointCandidate) -> dict[str, Any] | None:
    path = _result_path(target, candidate)
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "state": "succeeded",
        "experiment_id": target.experiment_id,
        "benchmark": target.benchmark,
        "model": target.model,
        "expected_samples": target.expected_samples,
        "validation_range": VALIDATION_RANGE,
        "pairing_artifact_signature": target.pairing_artifact_signature,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return None
    try:
        result_profile = normalize_selection_profile(result.get("selection_profile"))
    except ValueError:
        return None
    if result_profile != normalize_selection_profile(target.selection_profile):
        return None
    if normalize_training_policy(result.get("training_policy", "paper")) != target.training_policy:
        return None
    if all(result.get(key) == value for key, value in candidate.identity().items()):
        return result

    # Successful validation may outlive a rotating trainer path.  A relocated
    # result is reusable only when the local snapshot reproduces the signature
    # at the path recorded when validation ran.  This prevents a later model
    # from inheriting metrics merely because it occupies the same directory.
    candidate_root = _candidate_root(target, candidate)
    snapshot = (candidate_root / "workspace" / "checkpoint_model").resolve()
    if candidate.model_path.resolve() != snapshot:
        return None
    identity_fields = {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "workspace_id": candidate.workspace_id,
        "checkpoint_step": candidate.checkpoint_step,
        "formal_training_evidence_signature": candidate.formal_training_evidence_signature,
    }
    if any(result.get(key) != value for key, value in identity_fields.items()):
        return None
    recorded_path_value = result.get("model_path")
    recorded_signature = result.get("selection_signature")
    if not isinstance(recorded_path_value, str) or not recorded_path_value:
        return None
    if not isinstance(recorded_signature, str) or not recorded_signature:
        return None
    recorded_path = Path(recorded_path_value).resolve()
    if (
        selection_signature(
            "validation_sweep",
            None,
            candidate.checkpoint_step,
            snapshot,
            identity_path=recorded_path,
        )
        != recorded_signature
    ):
        return None
    if (
        selection_signature("validation_sweep", None, candidate.checkpoint_step, snapshot)
        != candidate.selection_signature
    ):
        return None

    normalized = dict(result)
    normalized.update(
        {
            "model_path": str(snapshot),
            "selection_signature": candidate.selection_signature,
            "model_relocation": {
                "reason": "validated_source_rotated",
                "validated_model_path": str(recorded_path),
                "validated_selection_signature": recorded_signature,
                "selected_model_path": str(snapshot),
                "selected_selection_signature": candidate.selection_signature,
            },
        },
    )
    return normalized


def _validated_snapshot_candidates(
    target: SweepTarget,
    *,
    existing_candidate_ids: set[str],
    include_retryable_failures: bool = False,
) -> list[CheckpointCandidate]:
    """Recover byte-identical snapshots whose trainer-owned source rotated."""
    recovered: list[CheckpointCandidate] = []
    candidates_root = target.task_root / validation_sweep_directory(target.selection_profile) / "candidates"
    for result_path in sorted(candidates_root.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected_target = {
            "experiment_id": target.experiment_id,
            "benchmark": target.benchmark,
            "model": target.model,
            "expected_samples": target.expected_samples,
            "validation_range": VALIDATION_RANGE,
            "pairing_artifact_signature": target.pairing_artifact_signature,
        }
        if any(result.get(key) != value for key, value in expected_target.items()):
            continue
        try:
            result_profile = normalize_selection_profile(result.get("selection_profile"))
        except ValueError:
            continue
        if result_profile != normalize_selection_profile(target.selection_profile):
            continue
        try:
            result_policy = normalize_training_policy(result.get("training_policy"))
        except ValueError:
            continue
        if result_policy != target.training_policy:
            continue
        candidate_id = result.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id != result_path.parent.name
            or candidate_id in existing_candidate_ids
        ):
            continue
        source = result.get("source")
        workspace_id = result.get("workspace_id")
        checkpoint_step = result.get("checkpoint_step")
        evidence_signature = result.get("formal_training_evidence_signature")
        if not isinstance(source, str) or not source:
            continue
        if workspace_id is not None and not isinstance(workspace_id, str):
            continue
        if checkpoint_step is not None and (isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int)):
            continue
        if evidence_signature is not None and not isinstance(evidence_signature, str):
            continue
        snapshot = (result_path.parent / "workspace" / "checkpoint_model").resolve()
        if snapshot.is_symlink() or not _valid_candidate_artifact(snapshot, source, target.training_policy):
            continue
        candidate = CheckpointCandidate(
            candidate_id=candidate_id,
            source=source,
            workspace_id=workspace_id,
            checkpoint_step=checkpoint_step,
            model_path=snapshot,
            selection_signature=selection_signature(
                "validation_sweep",
                None,
                checkpoint_step,
                snapshot,
            ),
            formal_training_evidence_signature=evidence_signature,
        )
        if _matching_success(target, candidate) is None:
            error = result.get("error")
            retryable_failure = (
                include_retryable_failures
                and result.get("state") == "failed"
                and isinstance(error, str)
                and any(marker in error for marker in RETRYABLE_SNAPSHOT_FAILURE_MARKERS)
            )
            recorded_path_value = result.get("model_path")
            recorded_signature = result.get("selection_signature")
            if (
                not retryable_failure
                or not isinstance(recorded_path_value, str)
                or not recorded_path_value
                or not isinstance(recorded_signature, str)
                or not recorded_signature
                or selection_signature(
                    "validation_sweep",
                    None,
                    checkpoint_step,
                    snapshot,
                    identity_path=Path(recorded_path_value).resolve(),
                )
                != recorded_signature
            ):
                continue
        recovered.append(candidate)
        existing_candidate_ids.add(candidate_id)
    return recovered


def _shares_top_level_inodes(left: Path, right: Path) -> bool:
    """Return whether two model directories share any mutable top-level file."""
    for left_file in left.iterdir():
        right_file = right / left_file.name
        if not left_file.is_file() or not right_file.is_file():
            continue
        left_stat = left_file.stat()
        right_stat = right_file.stat()
        if (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino):
            return True
    return False


def _stage_baseline_snapshot(
    workspace: Path,
    source: Path,
    *,
    training_policy: str,
    expected_signature: str | None,
    checkpoint_step: int | None,
) -> Path:
    """Hard-link a pinned baseline into the workspace without duplicating its bytes."""
    snapshot = workspace / "checkpoint_model"
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Baseline source is unavailable before validation: {source}") from error
    if not _valid_candidate_artifact(resolved_source, "baseline", training_policy):
        raise RuntimeError(f"Baseline source is not a complete artifact: {resolved_source}")
    if (
        expected_signature is not None
        and selection_signature("validation_sweep", None, checkpoint_step, resolved_source) != expected_signature
    ):
        raise RuntimeError(f"Baseline source identity changed before validation: {resolved_source}")
    snapshot_matches = (
        snapshot.is_dir()
        and not snapshot.is_symlink()
        and _valid_candidate_artifact(snapshot, "baseline", training_policy)
        and (
            expected_signature is None
            or selection_signature(
                "validation_sweep",
                None,
                checkpoint_step,
                snapshot,
                identity_path=resolved_source,
            )
            == expected_signature
        )
    )
    if snapshot_matches:
        return snapshot

    temporary = workspace / "checkpoint_model.tmp"
    shutil.rmtree(temporary, ignore_errors=True)

    def link_or_copy(source_file: str, destination_file: str) -> str:
        try:
            os.link(source_file, destination_file)
        except OSError:
            return shutil.copy2(source_file, destination_file)
        return destination_file

    try:
        shutil.copytree(
            resolved_source,
            temporary,
            copy_function=link_or_copy,
            ignore=shutil.ignore_patterns("checkpoint-*"),
            symlinks=False,
        )
        if not _valid_candidate_artifact(temporary, "baseline", training_policy):
            raise RuntimeError(f"Staged baseline is incomplete: {temporary}")
        if (
            expected_signature is not None
            and selection_signature(
                "validation_sweep",
                None,
                checkpoint_step,
                temporary,
                identity_path=resolved_source,
            )
            != expected_signature
        ):
            raise RuntimeError(f"Baseline changed while its snapshot was staged: {resolved_source}")
        if snapshot.is_symlink() or snapshot.is_file():
            snapshot.unlink()
        elif snapshot.exists():
            shutil.rmtree(snapshot)
        temporary.replace(snapshot)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return snapshot


def _stage_model_snapshot(
    workspace: Path,
    source: Path,
    *,
    candidate_source: str = "trained",
    training_policy: str = "paper",
    expected_signature: str | None = None,
    checkpoint_step: int | None = None,
) -> Path:
    """Materialize a candidate-local model before its mutable source can vanish."""
    workspace.mkdir(parents=True, exist_ok=True)
    snapshot = workspace / "checkpoint_model"
    # The prepared base model is a pinned, read-only experiment asset.  Keep
    # its bytes as hard links inside the candidate workspace instead of copying
    # ~15 GiB for every task.  This preserves the workspace-relative benchmark
    # contract (including container mounts) while its signed identity is
    # checked here, again in the worker, and again before result reuse.
    if candidate_source == "baseline":
        return _stage_baseline_snapshot(
            workspace,
            source,
            training_policy=training_policy,
            expected_signature=expected_signature,
            checkpoint_step=checkpoint_step,
        )

    snapshot_is_valid = (
        snapshot.is_dir()
        and not snapshot.is_symlink()
        and _valid_candidate_artifact(snapshot, candidate_source, training_policy)
    )
    snapshot_matches = snapshot_is_valid and (
        expected_signature is None
        or selection_signature(
            "validation_sweep",
            None,
            checkpoint_step,
            snapshot,
            identity_path=source,
        )
        == expected_signature
    )
    if snapshot_matches and (not source.is_dir() or not _shares_top_level_inodes(snapshot, source.resolve())):
        return snapshot

    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Checkpoint source is unavailable before validation: {source}") from error
    if not _valid_candidate_artifact(resolved_source, candidate_source, training_policy):
        raise RuntimeError(f"Checkpoint source is not a complete policy-compliant artifact: {resolved_source}")
    if (
        expected_signature is not None
        and selection_signature("validation_sweep", None, checkpoint_step, resolved_source) != expected_signature
    ):
        raise RuntimeError(f"Checkpoint source identity changed before validation: {resolved_source}")

    temporary = workspace / "checkpoint_model.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        shutil.copytree(
            resolved_source,
            temporary,
            copy_function=shutil.copy2,
            ignore=shutil.ignore_patterns("checkpoint-*"),
            symlinks=False,
        )
        if not _valid_candidate_artifact(temporary, candidate_source, training_policy):
            raise RuntimeError(f"Staged checkpoint is incomplete: {temporary}")
        if (
            expected_signature is not None
            and selection_signature(
                "validation_sweep",
                None,
                checkpoint_step,
                temporary,
                identity_path=resolved_source,
            )
            != expected_signature
        ):
            raise RuntimeError(f"Checkpoint changed while its snapshot was staged: {resolved_source}")

        if snapshot.is_symlink() or snapshot.is_file():
            snapshot.unlink()
        elif snapshot.exists():
            shutil.rmtree(snapshot)
        temporary.replace(snapshot)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return snapshot


def _archive_failed_attempt(candidate_root: Path) -> None:
    """Move a failed attempt's benchmark cache aside before retrying it."""
    result_path = candidate_root / "result.json"
    if not result_path.is_file():
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = {"state": "unreadable"}
    if result.get("state") == "succeeded":
        return

    benchmark_results = candidate_root / "workspace" / "benchmark_results"
    if not benchmark_results.exists():
        return
    timestamp = str(result.get("finished_at") or result.get("started_at") or datetime.now().astimezone().isoformat())
    attempt_root = candidate_root / "failed_attempts" / f"{safe_id(timestamp)}-{result_path.stat().st_mtime_ns}"
    attempt_root.mkdir(parents=True)
    shutil.copy2(result_path, attempt_root / "result.json")
    spec_path = candidate_root / "spec.json"
    if spec_path.is_file():
        shutil.copy2(spec_path, attempt_root / "spec.json")
    benchmark_results.replace(attempt_root / "benchmark_results")


@contextlib.contextmanager
def _candidate_claim(candidate_root: Path) -> Iterator[bool]:
    """Prevent overlapping sweep processes from evaluating one candidate twice."""
    candidate_root.mkdir(parents=True, exist_ok=True)
    with (candidate_root / ".validation.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _live_pids(values: set[int]) -> set[int]:
    return {pid for pid in values if (Path("/proc") / str(pid)).exists()}


def _update_trusted_owners(*, add: int | None = None, remove: int | None = None) -> None:
    GPU_TRUST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with GPU_TRUST_REGISTRY_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            values = (
                {
                    int(line)
                    for line in GPU_TRUST_REGISTRY.read_text(encoding="utf-8").splitlines()
                    if line.strip().isdigit()
                }
                if GPU_TRUST_REGISTRY.is_file()
                else set()
            )
            values = _live_pids(values)
            if add is not None:
                values.add(add)
            if remove is not None:
                values.discard(remove)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=GPU_TRUST_DIRECTORY,
                prefix="trusted_owners.",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write("".join(f"{pid}\n" for pid in sorted(values)))
            temporary.replace(GPU_TRUST_REGISTRY)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def trusted_gpu_owner() -> Iterator[None]:
    """Allow the current sweep tree through the live matrix exclusivity guard."""
    owner = os.getpid()
    _update_trusted_owners(add=owner)
    try:
        yield
    finally:
        _update_trusted_owners(remove=owner)


def worker_main(spec_path: Path) -> int:
    load_dotenv(ROOT / ".env", override=False)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    configure_project_environment(normalize_training_policy(spec.get("training_policy", "paper")))
    result_path = Path(spec["result_path"])
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    try:
        os.environ.update(
            {
                "FT_BASE_MODEL": spec["model"],
                "FT_TARGET_BENCHMARK": spec["benchmark"],
                "FT_BENCHMARK_DATASET_PATH": spec["benchmark_dataset_path"],
                "FT_EVALUATE_HELD_OUT_DURING_SEARCH": "false",
                "FT_BENCHMARK_TIMEOUT": str(spec["timeout_seconds"]),
            },
        )
        from rdagent.scenarios.finetune.benchmark import run_benchmark  # noqa: PLC0415

        workspace = Path(spec["workspace"])
        staged_model = Path(spec["staged_model_path"])
        if not _valid_candidate_artifact(staged_model, spec.get("source"), spec.get("training_policy")):
            raise RuntimeError(f"Staged checkpoint is unavailable or invalid: {staged_model}")
        original_model = Path(spec["model_path"])
        if selection_signature(
            "validation_sweep",
            None,
            spec.get("checkpoint_step"),
            staged_model,
            identity_path=original_model,
        ) != spec["selection_signature"]:
            raise RuntimeError("Staged checkpoint identity changed after validation scheduling")
        target = SweepTarget(
            experiment_id=spec["experiment_id"],
            benchmark=spec["benchmark"],
            model=spec["model"],
            benchmark_dataset_path=spec["benchmark_dataset_path"],
            task_root=Path(spec["task_root"]),
            expected_samples=spec["expected_samples"],
            training_policy=spec["training_policy"],
            pairing_artifact_signature=spec.get("pairing_artifact_signature"),
            selection_profile=spec.get("selection_profile", MAIN_SELECTION_PROFILE),
        )
        candidate = CheckpointCandidate(
            candidate_id=spec["candidate_id"],
            source=spec["source"],
            workspace_id=spec.get("workspace_id"),
            checkpoint_step=spec.get("checkpoint_step"),
            model_path=staged_model,
            selection_signature=spec["selection_signature"],
            formal_training_evidence_signature=spec.get("formal_training_evidence_signature"),
        )
        validate_candidate_formal_provenance(target, candidate)
        result = run_benchmark(
            workspace_path=str(workspace),
            model_path=str(staged_model),
            model_name=spec["model"],
            benchmark_name=spec["benchmark"],
            gpu_count=1,
            test_range=VALIDATION_RANGE,
            result_subdir=f"validation/{spec['candidate_id']}",
        )
        view = result_view(result, spec["benchmark"])
        if view is None or not view.get("paper_metrics"):
            raise RuntimeError("Validation result has no paper-primary metrics")
        payload.update(
            {
                "state": "succeeded",
                "finished_at": datetime.now().astimezone().isoformat(),
                "result": json_safe(result),
                "accuracy_summary": view["accuracy_summary"],
                "paper_metrics": view["paper_metrics"],
                "workspace": str(workspace),
            },
        )
        status_write(result_path, payload)
        return 0
    except Exception as error:
        payload.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(result_path, payload)
        raise


async def _run_claimed_candidate(
    target: SweepTarget,
    candidate: CheckpointCandidate,
    gpu: str,
    timeout_seconds: int,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    candidate_root = _candidate_root(target, candidate)
    workspace = candidate_root / "workspace"
    result_path = candidate_root / "result.json"
    spec_path = candidate_root / "spec.json"
    candidate_root.mkdir(parents=True, exist_ok=True)
    _archive_failed_attempt(candidate_root)
    payload = {
        **candidate.identity(),
        "schema_version": 1,
        "experiment_id": target.experiment_id,
        "benchmark": target.benchmark,
        "model": target.model,
        "training_policy": target.training_policy,
        "expected_samples": target.expected_samples,
        "pairing_artifact_signature": target.pairing_artifact_signature,
        "selection_profile": normalize_selection_profile(target.selection_profile),
        "benchmark_dataset_path": target.benchmark_dataset_path,
        "validation_range": VALIDATION_RANGE,
        "held_out_test_used": False,
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "api_routing": api_routing,
    }
    status_write(result_path, payload)
    try:
        validate_candidate_formal_provenance(target, candidate)
        staged_model_path = _stage_model_snapshot(
            workspace,
            candidate.model_path,
            candidate_source=candidate.source,
            training_policy=target.training_policy,
            expected_signature=candidate.selection_signature,
            checkpoint_step=candidate.checkpoint_step,
        )
    except Exception as error:  # noqa: BLE001 - persist fast staging failures for resume.
        payload.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(result_path, payload)
        print(f"FAIL  gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
        return False
    spec = {
        **candidate.identity(),
        "experiment_id": target.experiment_id,
        "result_path": str(result_path),
        "workspace": str(workspace),
        "task_root": str(target.task_root),
        "staged_model_path": str(staged_model_path),
        "model": target.model,
        "benchmark": target.benchmark,
        "benchmark_dataset_path": target.benchmark_dataset_path,
        "timeout_seconds": timeout_seconds,
        "training_policy": target.training_policy,
        "expected_samples": target.expected_samples,
        "pairing_artifact_signature": target.pairing_artifact_signature,
        "selection_profile": normalize_selection_profile(target.selection_profile),
    }
    status_write(spec_path, spec)
    environment = runtime_environment.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(PYTHON), str(SCRIPT_PATH), "--worker-spec", str(spec_path)]
    print(f"START gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
    try:
        with (candidate_root / "console.log").open("ab", buffering=0) as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds + 60)
            except TimeoutError:
                timed_out = True
                await stop_process(process)
                return_code = process.returncode
            except asyncio.CancelledError:
                await stop_process(process)
                raise
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - persist orchestration failure for resume.
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        artifact.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(result_path, artifact)
        print(f"FAIL  gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
        return False
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    success = return_code == 0 and not timed_out and artifact.get("state") == "succeeded"
    if not success and artifact.get("state") == "running":
        artifact.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "return_code": return_code,
                "outer_timeout": timed_out,
            },
        )
        status_write(result_path, artifact)
    print(f"{'DONE ' if success else 'FAIL '} gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
    return success


async def run_one(
    target: SweepTarget,
    candidate: CheckpointCandidate,
    gpu: str,
    timeout_seconds: int,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    candidate_root = _candidate_root(target, candidate)
    with _candidate_claim(candidate_root) as claimed:
        if not claimed:
            print(f"BUSY  gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
            return True
        # The pending queue is a point-in-time snapshot.  Another sweep may
        # finish this candidate while this worker is waiting to claim it, so
        # re-check under the candidate lock before overwriting the successful
        # artifact or launching a redundant benchmark.
        if _matching_success(target, candidate) is not None:
            print(f"REUSE gpu={gpu} {target.experiment_id} {candidate.candidate_id}", flush=True)
            return True
        return await _run_claimed_candidate(
            target,
            candidate,
            gpu,
            timeout_seconds,
            runtime_environment,
            api_routing,
        )


async def run_workers(
    work: list[tuple[SweepTarget, CheckpointCandidate]],
    gpus: list[str],
    timeout_seconds: int,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    queue: asyncio.Queue[tuple[SweepTarget, CheckpointCandidate]] = asyncio.Queue()
    for item in work:
        queue.put_nowait(item)
    results: list[bool] = []

    async def worker(gpu: str) -> None:
        while True:
            try:
                target, candidate = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(
                    await run_one(
                        target,
                        candidate,
                        gpu,
                        timeout_seconds,
                        runtime_environment,
                        api_routing,
                    ),
                )
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(gpu) for gpu in gpus))
    return all(results)


def _stage_pending_candidates(
    work: list[tuple[SweepTarget, CheckpointCandidate]],
) -> list[tuple[SweepTarget, CheckpointCandidate]]:
    """Pin every pending model before it can age out while waiting in the queue."""
    available: list[tuple[SweepTarget, CheckpointCandidate]] = []
    for target, candidate in work:
        candidate_root = _candidate_root(target, candidate)
        with _candidate_claim(candidate_root) as claimed:
            if not claimed or _matching_success(target, candidate) is not None:
                available.append((target, candidate))
                continue
            try:
                validate_candidate_formal_provenance(target, candidate)
                _stage_model_snapshot(
                    candidate_root / "workspace",
                    candidate.model_path,
                    candidate_source=candidate.source,
                    training_policy=target.training_policy,
                    expected_signature=candidate.selection_signature,
                    checkpoint_step=candidate.checkpoint_step,
                )
            except Exception:
                local_snapshot = candidate_root / "workspace" / "checkpoint_model"
                if has_model_weights(local_snapshot) or has_model_weights(candidate.model_path):
                    raise
                print(
                    f"STALE {target.experiment_id} {candidate.candidate_id}: "
                    "checkpoint disappeared before queue staging",
                    flush=True,
                )
                continue
        available.append((target, candidate))
    return available


def preflight(targets: list[SweepTarget], candidates: dict[SweepTarget, list[CheckpointCandidate]]) -> None:
    errors: list[str] = []
    opencompass = FT_ROOT / "conda_envs" / "opencompass" / "bin" / "opencompass"
    if not PYTHON.is_file():
        errors.append(f"Python environment is missing: {PYTHON}")
    if not opencompass.is_file():
        errors.append(f"OpenCompass backend is missing: {opencompass}")
    for target in targets:
        dataset = FT_ROOT / "benchmarks" / target.benchmark_dataset_path
        if not dataset.exists():
            errors.append(f"{target.experiment_id}: benchmark dataset is missing: {dataset}")
        for candidate in candidates[target]:
            if not has_model_weights(candidate.model_path):
                errors.append(f"{target.experiment_id}: checkpoint weights are missing: {candidate.model_path}")
                continue
            try:
                validate_candidate_formal_provenance(target, candidate)
            except (RuntimeError, ValidationSelectionError) as error:
                errors.append(str(error))
    errors.extend(responses_configuration_errors(os.environ))
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))


def _completed_candidates(
    target: SweepTarget,
    candidates: list[CheckpointCandidate],
) -> tuple[list[dict[str, Any]], list[str]]:
    completed: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for candidate in candidates:
        result = _matching_success(target, candidate)
        if result is None:
            unavailable.append(candidate.candidate_id)
        else:
            completed.append(result)
    return completed, unavailable


def _stable_selection_artifact_content(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the signed selection content without non-semantic run timestamps.

    A completed rsLoRA matrix may be re-audited with ``--resume``.  That audit
    historically refreshed the task-level ``finished_at`` timestamp even when
    no training artifact changed.  Rebuilding a validation selection from that
    status produced a different outer artifact signature and broke the binding
    to an already-completed one-shot held-out evaluation.  Selection identity,
    candidate provenance, validation metrics, and ranking remain part of this
    comparison; only creation/search timestamps are ignored.
    """
    content = {
        key: value
        for key, value in artifact.items()
        if key not in {"created_at", "artifact_signature"}
    }
    search = content.get("search")
    if isinstance(search, dict):
        content["search"] = {
            key: value for key, value in search.items() if key not in {"started_at", "finished_at"}
        }
    return json_safe(content)


def finalize_target(
    target: SweepTarget,
    initial_candidates: list[CheckpointCandidate],
    *,
    include_baseline: bool,
    include_final_outputs: bool,
    snapshot_only: bool,
    baseline_only: bool = False,
    candidate_pattern: re.Pattern[str] | None = None,
) -> bool:
    discovered = (
        discover_baseline_candidates(target)
        if baseline_only
        else discover_candidates(
            target,
            include_baseline=include_baseline,
            include_final_outputs=include_final_outputs,
        )
    )
    current_candidates = filter_candidates(discovered, candidate_pattern)
    initial_signature = candidate_set_signature(initial_candidates)
    current_signature = candidate_set_signature(current_candidates)
    completed, unavailable = _completed_candidates(target, current_candidates)
    status = json.loads((target.task_root / "status.json").read_text(encoding="utf-8"))
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(),
        "experiment_id": target.experiment_id,
        "benchmark": target.benchmark,
        "model": target.model,
        "training_policy": target.training_policy,
        "expected_samples": target.expected_samples,
        "pairing_artifact_signature": target.pairing_artifact_signature,
        "selection_profile": normalize_selection_profile(target.selection_profile),
        "validation_range": VALIDATION_RANGE,
        "held_out_test_used": False,
        "baseline_only": baseline_only,
        "search_state": status.get("state", "unknown"),
        "initial_candidate_set_signature": initial_signature,
        "candidate_set_signature": current_signature,
        "candidate_set_changed_during_sweep": initial_signature != current_signature,
        "candidate_count": len(current_candidates),
        "completed_count": len(completed),
        "unavailable_candidates": unavailable,
        "candidate_results": [
            {
                "candidate_id": candidate.candidate_id,
                "result_path": str(_result_path(target, candidate)),
                "state": "succeeded" if _matching_success(target, candidate) else "unavailable",
            }
            for candidate in current_candidates
        ],
    }
    if completed:
        try:
            ranked = rank_candidates(target.benchmark, completed)
        except ValidationSelectionError as error:
            snapshot["ranking_error"] = str(error)
        else:
            snapshot["provisional_selection"] = {
                key: ranked[0].get(key)
                for key in (
                    "candidate_id",
                    "model_path",
                    "selection_signature",
                    "formal_training_evidence_signature",
                    "selection_metrics",
                    "mean_ordinal_rank",
                )
            }

    snapshot_state = "complete"
    if unavailable:
        snapshot_state = "incomplete"
    elif initial_signature != current_signature:
        snapshot_state = "candidate_set_changed"
    elif status.get("state") != "succeeded":
        snapshot_state = "search_running"
    elif snapshot_only:
        snapshot_state = "snapshot_only"
    snapshot["state"] = snapshot_state
    status_write(target.task_root / validation_sweep_snapshot_file(target.selection_profile), json_safe(snapshot))

    if snapshot_state != "complete":
        print(
            f"SNAPSHOT {target.experiment_id}: state={snapshot_state} "
            f"completed={len(completed)}/{len(current_candidates)}",
            flush=True,
        )
        return not unavailable

    artifact = make_selection_artifact(
        experiment_id=target.experiment_id,
        benchmark=target.benchmark,
        model=target.model,
        benchmark_dataset_path=target.benchmark_dataset_path,
        search_status=status,
        candidate_set_signature=current_signature,
        candidates=completed,
        expected_samples=target.expected_samples,
        include_baseline=include_baseline,
        include_final_outputs=include_final_outputs,
        candidate_filter=candidate_pattern.pattern if candidate_pattern is not None else None,
        pairing_artifact_signature=target.pairing_artifact_signature,
        selection_profile=target.selection_profile,
    )
    artifact_path = target.task_root / validation_selection_file(target.selection_profile)
    preserved = False
    if artifact_path.is_file():
        try:
            existing_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if (
                isinstance(existing_artifact, dict)
                and _stable_selection_artifact_content(existing_artifact)
                == _stable_selection_artifact_content(artifact)
            ):
                validate_current_selection_provenance(
                    target,
                    existing_artifact,
                    include_baseline=include_baseline,
                    include_final_outputs=include_final_outputs,
                )
                artifact = existing_artifact
                preserved = True
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
            # An invalid or substantively changed artifact is replaced by the
            # newly rebuilt, fully validated selection below.
            preserved = False
    if not preserved:
        status_write(artifact_path, artifact)
    print(
        f"{'PRESERVE' if preserved else 'SELECT'} {target.experiment_id}: "
        f"{artifact['selection']['candidate_id']} "
        f"signature={artifact['selection']['selection_signature'][:12]}",
        flush=True,
    )
    return True


def main() -> int:
    args = parse_args()
    if args.worker_spec:
        return worker_main(Path(args.worker_spec))
    load_dotenv(ROOT / ".env", override=False)
    validate_max_parallel(args.max_parallel)
    run_root = resolve_run_root(args.matrix_run)
    targets = load_targets(run_root, args.only, selection_profile=args.selection_profile)
    configure_project_environment(targets[0].training_policy)
    include_baseline = (
        False
        if args.selection_profile == LORA_COMPARISON_SELECTION_PROFILE
        else not args.no_baseline
    )
    include_final_outputs = not args.no_final_outputs
    candidate_pattern = re.compile(args.candidate_only) if args.candidate_only else None
    candidates = {}
    for target in targets:
        discovered = (
            discover_baseline_candidates(target)
            if args.baseline_only
            else discover_candidates(
                target,
                include_baseline=include_baseline,
                include_final_outputs=include_final_outputs,
                include_retryable_failed_snapshots=args.retry_failed_snapshots,
            )
        )
        candidates[target] = filter_candidates(discovered, candidate_pattern)
    for target in targets:
        if not candidates[target]:
            raise RuntimeError(f"{target.experiment_id}: no model checkpoints are available")
        print(f"{target.experiment_id}: {len(candidates[target])} validation candidates")

    pending: list[tuple[SweepTarget, CheckpointCandidate]] = []
    reused = 0
    for target in targets:
        for candidate in candidates[target]:
            existing_path = _result_path(target, candidate)
            existing = _matching_success(target, candidate)
            if existing is not None:
                if not args.resume:
                    raise RuntimeError(
                        f"Validation result already exists for {candidate.candidate_id}; pass --resume to reuse it",
                    )
                reused += 1
                continue
            if existing_path.exists() and not args.resume:
                raise RuntimeError(
                    f"Validation attempt already exists for {candidate.candidate_id}; pass --resume to retry it",
                )
            pending.append((target, candidate))
    print(f"Validation sweep: {len(pending)} pending, {reused} reusable")
    if args.dry_run:
        return 0
    if pending:
        preflight(targets, candidates)
    if args.preflight_only:
        print("Preflight OK")
        return 0

    success = True
    if pending:
        pending = _stage_pending_candidates(pending)
    if pending:
        timeout_seconds = duration_seconds(args.task_timeout)
        gpus = visible_gpus(args.gpus)[: args.max_parallel]
        with trusted_gpu_owner(), routed_api_environment(os.environ) as (runtime_environment, api_routing):
            success = asyncio.run(
                run_workers(
                    pending,
                    gpus,
                    timeout_seconds,
                    runtime_environment,
                    api_routing,
                ),
            )
    finalization = [
        finalize_target(
            target,
            candidates[target],
            include_baseline=include_baseline,
            include_final_outputs=include_final_outputs,
            snapshot_only=args.snapshot_only,
            baseline_only=args.baseline_only,
            candidate_pattern=candidate_pattern,
        )
        for target in targets
    ]
    return 0 if success and all(finalization) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
