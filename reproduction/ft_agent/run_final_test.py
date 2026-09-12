#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, SIM102, TRY003, TRY004, TRY300
"""Evaluate each validation-selected FT-Agent checkpoint on held-out test once."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SCRIPT_PATH = Path(__file__).resolve()

if __package__:
    from .collect_results import MATRIX_LOG_ROOT, json_safe, load_latest_session, select_sota_index
    from .final_test_protocol import (
        FINAL_TEST_SCHEMA_VERSION,
        TEST_RANGE,
        has_model_weights,
        is_policy_compliant_selection,
        normalize_training_policy,
        selection_signature,
        trace_protocol_pollution,
    )
    from .responses_adapter import responses_configuration_errors, routed_api_environment
    from .rslora_pairing import pairing_signature_from_status, validate_pairing_artifact
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
        visible_gpus,
    )
    from .run_validation_sweep import SweepTarget, validate_current_selection_provenance
    from .validation_selection import (
        LORA_COMPARISON_SELECTION_PROFILE,
        MAIN_SELECTION_PROFILE,
        SELECTION_PROFILES,
        final_test_file,
        final_test_log_file,
        final_test_spec_file,
        final_test_workspace_directory,
        normalize_selection_profile,
        validation_selection_file,
    )
else:
    from collect_results import MATRIX_LOG_ROOT, json_safe, load_latest_session, select_sota_index
    from final_test_protocol import (
        FINAL_TEST_SCHEMA_VERSION,
        TEST_RANGE,
        has_model_weights,
        is_policy_compliant_selection,
        normalize_training_policy,
        selection_signature,
        trace_protocol_pollution,
    )
    from responses_adapter import responses_configuration_errors, routed_api_environment
    from rslora_pairing import pairing_signature_from_status, validate_pairing_artifact
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
        visible_gpus,
    )
    from run_validation_sweep import SweepTarget, validate_current_selection_provenance
    from validation_selection import (
        LORA_COMPARISON_SELECTION_PROFILE,
        MAIN_SELECTION_PROFILE,
        SELECTION_PROFILES,
        final_test_file,
        final_test_log_file,
        final_test_spec_file,
        final_test_workspace_directory,
        normalize_selection_profile,
        validation_selection_file,
    )

@dataclass(frozen=True)
class FinalTestTarget:
    experiment_id: str
    benchmark: str
    model: str
    task_root: Path
    model_path: Path
    selection_source: str
    history_index: int | None
    loop_id: int | None
    selection_signature: str
    benchmark_dataset_path: str
    expected_samples: int
    formal_training_evidence_signature: str | None
    training_policy: str = "paper"
    selection_profile: str = MAIN_SELECTION_PROFILE
    candidate_source: str | None = None
    candidate_id: str | None = None
    checkpoint_step: int | None = None
    selection_artifact_signature: str | None = None
    pairing_artifact_signature: str | None = None
    legacy_result: dict[str, Any] | None = None
    protocol_pollution: tuple[str, ...] = ()

    def selection(self) -> dict[str, Any]:
        return {
            "source": self.selection_source,
            "candidate_source": self.candidate_source,
            "training_policy": self.training_policy,
            "history_index": self.history_index,
            "loop_id": self.loop_id,
            "candidate_id": self.candidate_id,
            "checkpoint_step": self.checkpoint_step,
            "model_path": str(self.model_path),
            "signature": self.selection_signature,
            "expected_samples": self.expected_samples,
            "formal_training_evidence_signature": self.formal_training_evidence_signature,
            "validation_selection_artifact_signature": self.selection_artifact_signature,
            "pairing_artifact_signature": self.pairing_artifact_signature,
            "selection_profile": normalize_selection_profile(self.selection_profile),
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
        help="Independent validation selection whose checkpoint should be tested",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse a matching successful final-test artifact")
    parser.add_argument(
        "--allow-retest",
        action="store_true",
        help="Allow test evaluation after an already successful artifact's selection changed",
    )
    parser.add_argument(
        "--audit-reuse-legacy",
        action="store_true",
        help=(
            "Audit only: accept a polluted pre-protocol trace and persist its same-selection "
            "legacy test result without executing another held-out evaluation"
        ),
    )
    parser.add_argument("--task-timeout", default="3h")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker_spec and not args.matrix_run:
        parser.error("--matrix-run is required")
    return args


def resolve_run_root(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = MATRIX_LOG_ROOT / candidate
    return candidate.resolve()


def _usable_legacy_result(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("accuracy_summary")
    return value if isinstance(summary, dict) and bool(summary) else None


def _status_expected_samples(status: dict[str, Any]) -> int:
    expected = status.get("formal_expected_samples")
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected < 1
        or status.get("data_limit") != expected
    ):
        raise RuntimeError("Task formal sample contract is missing or inconsistent")
    return expected


def _load_task_trace(task_root: Path) -> Any:
    session_path, session = load_latest_session(task_root / "trace")
    trace = getattr(session, "trace", None)
    if trace is None:
        raise RuntimeError(f"Latest session has no trace: {session_path}")
    return trace


def _validated_pairing_signature(task_root: Path, status: dict[str, Any]) -> str | None:
    """Return a revalidated strict-pair signature, or ``None`` for normal tasks."""
    try:
        signature = pairing_signature_from_status(status)
        if signature is not None:
            validate_pairing_artifact(
                task_root,
                experiment_id=str(status["experiment_id"]),
                expected_samples=_status_expected_samples(status),
                expected_signature=signature,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid rsLoRA pairing contract: {error}") from error
    return signature


def target_from_trace(
    task_root: Path,
    status: dict[str, Any],
    *,
    allow_polluted_legacy_audit: bool = False,
) -> FinalTestTarget:
    training_policy = normalize_training_policy(status.get("training_policy", "paper"))
    expected_samples = _status_expected_samples(status)
    if _validated_pairing_signature(task_root, status) is not None:
        raise RuntimeError("Strict rsLoRA pairs must use their signed validation selection, not a search trace")
    trace = _load_task_trace(task_root)
    pollution = trace_protocol_pollution(trace)
    if pollution and not allow_polluted_legacy_audit:
        raise RuntimeError(
            "Protocol-polluted trace contains search-time held-out result(s): "
            + ", ".join(pollution)
            + "; quarantine this task and rerun validation-only search",
        )
    history_index = select_sota_index(trace)
    if history_index is None:
        selection_source = "baseline"
        candidate_source = "baseline"
        loop_id = None
        model_path = FT_ROOT / "models" / str(status["model"])
        legacy_result = _usable_legacy_result(getattr(trace.scen, "baseline_benchmark_score_test", None))
    else:
        selection_source = "accepted_loop"
        candidate_source = "accepted_loop"
        loop_id = getattr(trace, "idx2loop_id", {}).get(history_index)
        node = trace.hist[history_index]
        workspace = getattr(node[0], "experiment_workspace", None)
        if workspace is None:
            raise RuntimeError(f"Selected history node {history_index} has no workspace")
        model_path = Path(workspace.workspace_path) / "output"
        running_info = getattr(workspace, "running_info", None)
        result = getattr(running_info, "result", None)
        legacy_result = _usable_legacy_result(result.get("benchmark_test") if isinstance(result, dict) else None)
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Selected model directory is missing: {model_path}")
    signature = selection_signature(selection_source, history_index, loop_id, model_path)
    return FinalTestTarget(
        experiment_id=str(status["experiment_id"]),
        benchmark=str(status["benchmark"]),
        model=str(status["model"]),
        task_root=task_root.resolve(),
        model_path=model_path,
        selection_source=selection_source,
        history_index=history_index,
        loop_id=loop_id,
        selection_signature=signature,
        benchmark_dataset_path=str(status["benchmark_dataset_path"]),
        expected_samples=expected_samples,
        formal_training_evidence_signature=None,
        training_policy=training_policy,
        candidate_source=candidate_source,
        legacy_result=legacy_result,
        protocol_pollution=pollution,
    )


def target_from_validation_selection(  # noqa: PLR0912, PLR0915
    task_root: Path,
    status: dict[str, Any],
    *,
    selection_profile: str = MAIN_SELECTION_PROFILE,
) -> FinalTestTarget:
    """Load the immutable validation-selected checkpoint for one-shot testing."""
    selection_profile = normalize_selection_profile(selection_profile)
    if status.get("state") != "succeeded":
        raise RuntimeError(f"Search is not complete: state={status.get('state', 'unknown')!r}")
    expected_samples = _status_expected_samples(status)
    pairing_artifact_signature = _validated_pairing_signature(task_root, status)
    # Ordinary FT-Agent tasks must retain their search trace so held-out
    # leakage can be audited.  A strict paired rsLoRA task is produced by the
    # standalone trainer and intentionally has no agent search trace; its
    # signed source/target workspace contract replaces that provenance link.
    if pairing_artifact_signature is None:
        pollution = trace_protocol_pollution(_load_task_trace(task_root))
        if pollution:
            raise RuntimeError(
                "Protocol-polluted trace contains search-time held-out result(s): "
                + ", ".join(pollution)
                + "; quarantine this task and rerun validation-only search",
            )

    artifact_path = task_root / validation_selection_file(selection_profile)
    if not artifact_path.is_file():
        raise RuntimeError(f"Validation selection artifact is missing: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unable to read validation selection artifact: {type(error).__name__}: {error}",
        ) from error
    if not isinstance(artifact, dict):
        raise TypeError("Validation selection artifact is not a JSON object")

    experiment_id = str(status["experiment_id"])
    benchmark = str(status["benchmark"])
    model = str(status["model"])
    training_policy = normalize_training_policy(status.get("training_policy", "paper"))
    benchmark_dataset_path = status.get("benchmark_dataset_path")
    if not isinstance(benchmark_dataset_path, str) or not benchmark_dataset_path:
        raise RuntimeError("Pinned benchmark dataset path is missing from task status")
    target = SweepTarget(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        benchmark_dataset_path=benchmark_dataset_path,
        task_root=task_root.resolve(),
        expected_samples=expected_samples,
        training_policy=training_policy,
        pairing_artifact_signature=pairing_artifact_signature,
        selection_profile=selection_profile,
    )
    selection = validate_current_selection_provenance(target, artifact)

    candidate_id = selection.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise RuntimeError("Validation selection candidate id is missing")
    checkpoint_step = selection.get("checkpoint_step")
    if checkpoint_step is not None and (isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int)):
        raise RuntimeError("Validation selection checkpoint step is invalid")
    model_path = Path(str(selection.get("model_path", ""))).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Validation-selected model directory is missing: {model_path}")
    candidate_source = selection.get("source")
    if not isinstance(candidate_source, str) or not candidate_source:
        raise RuntimeError("Validation selection candidate source is missing")
    if not is_policy_compliant_selection(model_path, candidate_source, training_policy):
        raise RuntimeError(
            f"Validation-selected model is not compliant with training policy {training_policy!r}: {model_path}",
        )
    artifact_signature = artifact.get("artifact_signature")
    if not isinstance(artifact_signature, str) or not artifact_signature:
        raise RuntimeError("Validation selection artifact signature is missing")
    evidence_signature = selection.get("formal_training_evidence_signature")
    if candidate_source == "baseline":
        if evidence_signature is not None:
            raise RuntimeError("Baseline selection must not claim formal training evidence")
    elif not isinstance(evidence_signature, str) or not evidence_signature:
        raise RuntimeError("Validation-selected model has no formal training evidence signature")

    return FinalTestTarget(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        task_root=task_root.resolve(),
        model_path=model_path,
        selection_source="validation_sweep",
        history_index=None,
        loop_id=None,
        selection_signature=str(selection["selection_signature"]),
        benchmark_dataset_path=benchmark_dataset_path,
        expected_samples=expected_samples,
        formal_training_evidence_signature=evidence_signature,
        training_policy=training_policy,
        selection_profile=selection_profile,
        candidate_source=candidate_source,
        candidate_id=candidate_id,
        checkpoint_step=checkpoint_step,
        selection_artifact_signature=artifact_signature,
        pairing_artifact_signature=pairing_artifact_signature,
    )


def validate_target_current_provenance(target: FinalTestTarget) -> None:
    """Revalidate the signed candidate set and formal evidence immediately before test."""
    if target.selection_source != "validation_sweep":
        return
    artifact_path = target.task_root / validation_selection_file(target.selection_profile)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to re-read validation selection: {error}") from error
    if not isinstance(artifact, dict):
        raise RuntimeError("Validation selection artifact is not a JSON object")
    sweep_target = SweepTarget(
        experiment_id=target.experiment_id,
        benchmark=target.benchmark,
        model=target.model,
        benchmark_dataset_path=target.benchmark_dataset_path,
        task_root=target.task_root,
        expected_samples=target.expected_samples,
        training_policy=target.training_policy,
        pairing_artifact_signature=target.pairing_artifact_signature,
        selection_profile=target.selection_profile,
    )
    selection = validate_current_selection_provenance(sweep_target, artifact)
    expected = {
        "candidate_id": target.candidate_id,
        "source": target.candidate_source,
        "checkpoint_step": target.checkpoint_step,
        "model_path": str(target.model_path),
        "selection_signature": target.selection_signature,
        "formal_training_evidence_signature": target.formal_training_evidence_signature,
    }
    mismatches = [key for key, value in expected.items() if selection.get(key) != value]
    if mismatches:
        raise RuntimeError("Validation selection changed before final test: " + ", ".join(mismatches))
    if artifact.get("artifact_signature") != target.selection_artifact_signature:
        raise RuntimeError("Validation selection artifact changed before final test")


def load_targets(  # noqa: PLR0912
    run_root: Path,
    only: str | None = None,
    *,
    allow_polluted_legacy_audit: bool = False,
    selection_profile: str = MAIN_SELECTION_PROFILE,
) -> list[FinalTestTarget]:
    selection_profile = normalize_selection_profile(selection_profile)
    matrix_path = run_root / "matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError(f"Matrix manifest is missing: {matrix_path}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_policy = normalize_training_policy(matrix.get("training_policy", "paper"))
    pattern = re.compile(only) if only else None
    targets: list[FinalTestTarget] = []
    errors: list[str] = []
    for task in matrix.get("tasks", []):
        experiment_id = str(task["experiment_id"])
        if pattern and not pattern.search(experiment_id):
            continue
        task_root = run_root / safe_id(experiment_id)
        status_path = task_root / "status.json"
        if not status_path.is_file():
            errors.append(f"{experiment_id}: status is missing")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            selection_profile == LORA_COMPARISON_SELECTION_PROFILE
            and status.get("formal_training_method") != "lora"
        ):
            continue
        if task.get("formal_expected_samples") != status.get("formal_expected_samples"):
            errors.append(f"{experiment_id}: matrix and task formal sample contracts differ")
            continue
        status_policy = normalize_training_policy(status.get("training_policy", matrix_policy))
        if status_policy != matrix_policy:
            errors.append(
                f"{experiment_id}: task training policy {status_policy!r} does not match "
                f"matrix policy {matrix_policy!r}",
            )
            continue
        status["training_policy"] = status_policy
        if status.get("state") != "succeeded":
            errors.append(f"{experiment_id}: search is not complete: state={status.get('state', 'unknown')!r}")
            continue
        try:
            pairing_artifact_signature = _validated_pairing_signature(task_root, status)
            pollution = (
                ()
                if pairing_artifact_signature is not None
                else trace_protocol_pollution(_load_task_trace(task_root))
            )
            if (
                selection_profile == MAIN_SELECTION_PROFILE
                and pairing_artifact_signature is None
                and pollution
                and allow_polluted_legacy_audit
            ):
                target = target_from_trace(
                    task_root,
                    status,
                    allow_polluted_legacy_audit=True,
                )
            else:
                target = target_from_validation_selection(
                    task_root,
                    status,
                    selection_profile=selection_profile,
                )
            targets.append(target)
        except Exception as error:  # noqa: BLE001 - report every unavailable task together.
            errors.append(f"{experiment_id}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("Final-test selection is not ready:\n- " + "\n- ".join(errors))
    if not targets:
        raise RuntimeError("No final-test targets were selected")
    return targets


def _artifact_payload(
    target: FinalTestTarget,
    *,
    state: str,
    evaluation_mode: str,
    api_routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FINAL_TEST_SCHEMA_VERSION,
        "experiment_id": target.experiment_id,
        "benchmark": target.benchmark,
        "model": target.model,
        "training_policy": target.training_policy,
        "selection_profile": normalize_selection_profile(target.selection_profile),
        "state": state,
        "evaluation_mode": evaluation_mode,
        "selection": target.selection(),
        "test_range": TEST_RANGE,
        "benchmark_dataset_path": target.benchmark_dataset_path,
        "pairing_artifact_signature": target.pairing_artifact_signature,
        "api_routing": api_routing,
    }


def _existing_artifact(target: FinalTestTarget) -> dict[str, Any] | None:
    path = target.task_root / final_test_file(target.selection_profile)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_matches_target(artifact: dict[str, Any], target: FinalTestTarget) -> bool:
    return artifact.get("selection") == target.selection()


def _cross_profile_attempt(
    target: FinalTestTarget,
) -> tuple[str, Path, dict[str, Any]] | None:
    """Find an evaluation attempt for the exact checkpoint in another profile."""
    active_profile = normalize_selection_profile(target.selection_profile)
    for profile in SELECTION_PROFILES:
        if profile == active_profile:
            continue
        path = target.task_root / final_test_file(profile)
        if not path.is_file():
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(artifact, dict):
            continue
        selection = artifact.get("selection")
        if not isinstance(selection, dict):
            continue
        expected = {
            "schema_version": FINAL_TEST_SCHEMA_VERSION,
            "experiment_id": target.experiment_id,
            "benchmark": target.benchmark,
            "model": target.model,
            "test_range": TEST_RANGE,
            "benchmark_dataset_path": target.benchmark_dataset_path,
            "pairing_artifact_signature": target.pairing_artifact_signature,
        }
        if any(artifact.get(key) != value for key, value in expected.items()):
            continue
        selection_expected = {
            "signature": target.selection_signature,
            "expected_samples": target.expected_samples,
            "formal_training_evidence_signature": target.formal_training_evidence_signature,
            "pairing_artifact_signature": target.pairing_artifact_signature,
        }
        if any(selection.get(key) != value for key, value in selection_expected.items()):
            continue
        return profile, path, artifact
    return None


def _materialize_cross_profile_reuse(
    target: FinalTestTarget,
    source_profile: str,
    source_path: Path,
    source_artifact: dict[str, Any],
) -> None:
    """Bind one successful held-out evaluation to a second signed selection."""
    if source_artifact.get("state") != "succeeded" or source_artifact.get("evaluation_mode") != "post_selection":
        raise RuntimeError("Cross-profile reuse requires a successful clean post-selection artifact")
    if not isinstance(source_artifact.get("result"), dict):
        raise RuntimeError("Cross-profile reuse source has no result payload")
    payload = _artifact_payload(
        target,
        state="succeeded",
        evaluation_mode="post_selection",
        api_routing=source_artifact.get("api_routing"),
    )
    payload.update(
        {
            "started_at": source_artifact.get("started_at"),
            "finished_at": source_artifact.get("finished_at"),
            "result": source_artifact["result"],
            "workspace": source_artifact.get("workspace"),
            "reused_at": datetime.now().astimezone().isoformat(),
            "reused_from": {
                "selection_profile": source_profile,
                "artifact_path": str(source_path.resolve()),
                "validation_selection_artifact_signature": (
                    source_artifact.get("selection", {}).get("validation_selection_artifact_signature")
                    if isinstance(source_artifact.get("selection"), dict)
                    else None
                ),
            },
        },
    )
    status_write(target.task_root / final_test_file(target.selection_profile), payload)


def _prepare_legacy_audit(
    target: FinalTestTarget,
    existing: dict[str, Any] | None,
    *,
    resume: bool,
    audit_reuse_legacy: bool,
) -> None:
    if not audit_reuse_legacy:
        raise RuntimeError(
            f"{target.experiment_id}: protocol-polluted search trace cannot enter final evaluation",
        )
    if target.legacy_result is None:
        raise RuntimeError(
            f"{target.experiment_id}: polluted trace has no same-selection legacy test to audit; "
            "refusing a new held-out evaluation",
        )
    if existing is None:
        return
    if existing.get("state") == "succeeded" and _artifact_matches_target(existing, target):
        if resume:
            return
        raise RuntimeError(
            f"{target.experiment_id}: legacy audit artifact already succeeded; pass --resume to reuse it",
        )
    raise RuntimeError(
        f"{target.experiment_id}: a legacy audit was already attempted; refusing to overwrite it",
    )


def prepare_targets(
    targets: list[FinalTestTarget],
    *,
    resume: bool,
    allow_retest: bool,
    audit_reuse_legacy: bool,
) -> tuple[list[FinalTestTarget], list[FinalTestTarget]]:
    pending: list[FinalTestTarget] = []
    reused: list[FinalTestTarget] = []
    for target in targets:
        existing = _existing_artifact(target)
        if target.protocol_pollution:
            _prepare_legacy_audit(
                target,
                existing,
                resume=resume,
                audit_reuse_legacy=audit_reuse_legacy,
            )
            reused.append(target)
            continue
        if existing is None:
            cross_profile = _cross_profile_attempt(target)
            if cross_profile is not None:
                source_profile, source_path, source_artifact = cross_profile
                if source_artifact.get("state") == "succeeded":
                    _materialize_cross_profile_reuse(
                        target,
                        source_profile,
                        source_path,
                        source_artifact,
                    )
                    reused.append(target)
                    continue
                if not allow_retest:
                    raise RuntimeError(
                        f"{target.experiment_id}: the same checkpoint was already attempted by "
                        f"selection profile {source_profile!r} ({source_artifact.get('state')}); "
                        "refusing another held-out evaluation without --allow-retest",
                    )
        if existing and existing.get("state") == "succeeded":
            if _artifact_matches_target(existing, target):
                if existing.get("evaluation_mode") != "post_selection":
                    raise RuntimeError(
                        f"{target.experiment_id}: matching artifact is not a clean post-selection test",
                    )
                if resume:
                    reused.append(target)
                    continue
                raise RuntimeError(
                    f"{target.experiment_id}: final test already succeeded; pass --resume to reuse it",
                )
        if existing and not allow_retest:
            detail = "selection changed" if not _artifact_matches_target(existing, target) else existing.get("state")
            raise RuntimeError(
                f"{target.experiment_id}: held-out evaluation was already attempted ({detail}); "
                "refusing another attempt without --allow-retest",
            )
        pending.append(target)
    return pending, reused


def persist_legacy_result(target: FinalTestTarget) -> None:
    if not target.protocol_pollution or target.legacy_result is None:
        raise RuntimeError("Legacy audit artifacts require a polluted trace and a same-selection test result")
    payload = _artifact_payload(target, state="succeeded", evaluation_mode="legacy_same_node_reuse")
    payload.update(
        {
            "started_at": None,
            "finished_at": datetime.now().astimezone().isoformat(),
            "result": json_safe(target.legacy_result),
            "protocol_note": (
                "Audit-only artifact: this held-out result was produced by the legacy iterative "
                "runner and is excluded from clean protocol reports."
            ),
            "protocol_pollution": list(target.protocol_pollution),
        },
    )
    status_write(target.task_root / final_test_file(target.selection_profile), payload)


def _ensure_model_link(workspace: Path, source: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    link = workspace / "selected_model"
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"Existing model link points to a different selection: {link}")
    elif link.exists():
        raise RuntimeError(f"Refusing to replace non-symlink final-test model path: {link}")
    else:
        link.symlink_to(source, target_is_directory=True)
    return link


def _validate_scheduled_selection(spec: dict[str, Any]) -> None:
    validation_artifact_path = Path(spec["validation_selection_path"])
    validation_artifact = json.loads(validation_artifact_path.read_text(encoding="utf-8"))
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
    selection = validate_current_selection_provenance(target, validation_artifact)
    if validation_artifact.get("artifact_signature") != spec["selection_artifact_signature"]:
        raise RuntimeError("Validation selection artifact changed after final-test scheduling")
    if validation_artifact.get("pairing_artifact_signature") != spec.get("pairing_artifact_signature"):
        raise RuntimeError("rsLoRA pairing artifact changed after final-test scheduling")
    if validation_artifact.get("benchmark_dataset_path") != spec["benchmark_dataset_path"]:
        raise RuntimeError("Validation selection benchmark dataset changed after scheduling")
    if selection.get("selection_signature") != spec["selection_signature"]:
        raise RuntimeError("Validation-selected checkpoint identity changed after scheduling")
    if selection.get("formal_training_evidence_signature") != spec.get(
        "formal_training_evidence_signature",
    ):
        raise RuntimeError("Formal training evidence changed after final-test scheduling")
    if selection.get("source") != spec.get("candidate_source"):
        raise RuntimeError("Validation-selected candidate source changed after scheduling")
    if selection.get("candidate_id") != spec.get("candidate_id"):
        raise RuntimeError("Validation-selected candidate id changed after scheduling")
    if selection.get("checkpoint_step") != spec.get("checkpoint_step"):
        raise RuntimeError("Validation-selected checkpoint step changed after scheduling")
    if Path(str(selection.get("model_path", ""))).resolve() != Path(spec["model_path"]).resolve():
        raise RuntimeError("Validation-selected model path changed after scheduling")
    if not is_policy_compliant_selection(
        Path(spec["model_path"]),
        spec.get("candidate_source"),
        spec.get("training_policy"),
    ):
        raise RuntimeError("Validation-selected model no longer complies with its training policy")


def worker_main(spec_path: Path) -> int:
    load_dotenv(ROOT / ".env", override=False)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    configure_project_environment(normalize_training_policy(spec.get("training_policy", "paper")))
    artifact_path = Path(spec["artifact_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    try:
        _validate_scheduled_selection(spec)
        os.environ.update(
            {
                "FT_BASE_MODEL": spec["model"],
                "FT_TARGET_BENCHMARK": spec["benchmark"],
                "FT_BENCHMARK_DATASET_PATH": spec["benchmark_dataset_path"],
                "FT_EVALUATE_HELD_OUT_DURING_SEARCH": "false",
                "FT_BENCHMARK_TIMEOUT": str(spec["timeout_seconds"]),
            },
        )
        # Import only after worker-specific environment variables are installed.
        from rdagent.scenarios.finetune.benchmark import run_benchmark  # noqa: PLC0415

        workspace = Path(spec["workspace"])
        model_link = _ensure_model_link(workspace, Path(spec["model_path"]))
        result = run_benchmark(
            workspace_path=str(workspace),
            model_path=str(model_link),
            model_name=spec["model"],
            benchmark_name=spec["benchmark"],
            gpu_count=1,
            test_range=TEST_RANGE,
            result_subdir="test",
        )
        payload.update(
            {
                "state": "succeeded",
                "finished_at": datetime.now().astimezone().isoformat(),
                "result": json_safe(result),
                "workspace": str(workspace),
            },
        )
        status_write(artifact_path, payload)
        return 0
    except Exception as error:
        payload.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(artifact_path, payload)
        raise


async def run_one(
    target: FinalTestTarget,
    gpu: str,
    timeout_seconds: int,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    if target.selection_source != "validation_sweep" or not target.selection_artifact_signature:
        raise RuntimeError(f"{target.experiment_id}: final test requires a signed validation selection")
    validate_target_current_provenance(target)
    artifact_path = target.task_root / final_test_file(target.selection_profile)
    workspace = (
        target.task_root
        / final_test_workspace_directory(target.selection_profile)
        / target.selection_signature[:16]
    )
    spec_path = target.task_root / final_test_spec_file(target.selection_profile)
    payload = _artifact_payload(
        target,
        state="running",
        evaluation_mode="post_selection",
        api_routing=api_routing,
    )
    payload["started_at"] = datetime.now().astimezone().isoformat()
    status_write(artifact_path, payload)
    spec = {
        "artifact_path": str(artifact_path),
        "validation_selection_path": str(
            target.task_root / validation_selection_file(target.selection_profile),
        ),
        "task_root": str(target.task_root),
        "selection_artifact_signature": target.selection_artifact_signature,
        "selection_signature": target.selection_signature,
        "formal_training_evidence_signature": target.formal_training_evidence_signature,
        "pairing_artifact_signature": target.pairing_artifact_signature,
        "selection_profile": normalize_selection_profile(target.selection_profile),
        "experiment_id": target.experiment_id,
        "workspace": str(workspace),
        "model_path": str(target.model_path),
        "model": target.model,
        "training_policy": target.training_policy,
        "candidate_source": target.candidate_source,
        "candidate_id": target.candidate_id,
        "checkpoint_step": target.checkpoint_step,
        "benchmark": target.benchmark,
        "benchmark_dataset_path": target.benchmark_dataset_path,
        "expected_samples": target.expected_samples,
        "timeout_seconds": timeout_seconds,
    }
    status_write(spec_path, spec)
    environment = runtime_environment.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(PYTHON), str(SCRIPT_PATH), "--worker-spec", str(spec_path)]
    print(f"START gpu={gpu} {target.experiment_id} final-test", flush=True)
    try:
        with (target.task_root / final_test_log_file(target.selection_profile)).open("ab", buffering=0) as log:
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
    except Exception as error:  # noqa: BLE001 - persist orchestration failures for audit/resume.
        artifact = _existing_artifact(target) or payload
        artifact.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(artifact_path, artifact)
        print(f"FAIL  gpu={gpu} {target.experiment_id} final-test", flush=True)
        return False
    artifact = _existing_artifact(target) or payload
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
        status_write(artifact_path, artifact)
    print(f"{'DONE ' if success else 'FAIL '} gpu={gpu} {target.experiment_id} final-test", flush=True)
    return success


async def run_workers(
    targets: list[FinalTestTarget],
    gpus: list[str],
    timeout_seconds: int,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    queue: asyncio.Queue[FinalTestTarget] = asyncio.Queue()
    for target in targets:
        queue.put_nowait(target)
    results: list[bool] = []

    async def worker(gpu: str) -> None:
        while True:
            try:
                target = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await run_one(target, gpu, timeout_seconds, runtime_environment, api_routing))
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(gpu) for gpu in gpus))
    return all(results)


def preflight(targets: list[FinalTestTarget]) -> None:
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
        if not has_model_weights(target.model_path):
            errors.append(f"{target.experiment_id}: selected model weights are missing: {target.model_path}")
        elif not is_policy_compliant_selection(
            target.model_path,
            target.candidate_source,
            target.training_policy,
        ):
            errors.append(
                f"{target.experiment_id}: selected model does not comply with "
                f"training policy {target.training_policy!r}",
            )
        try:
            validate_target_current_provenance(target)
        except Exception as error:  # noqa: BLE001 - aggregate every provenance failure.
            errors.append(f"{target.experiment_id}: {type(error).__name__}: {error}")
    errors.extend(responses_configuration_errors(os.environ))
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))


def main() -> int:
    args = parse_args()
    if args.worker_spec:
        return worker_main(Path(args.worker_spec))
    load_dotenv(ROOT / ".env", override=False)
    validate_max_parallel(args.max_parallel)
    run_root = resolve_run_root(args.matrix_run)
    targets = load_targets(
        run_root,
        args.only,
        allow_polluted_legacy_audit=args.audit_reuse_legacy,
        selection_profile=args.selection_profile,
    )
    configure_project_environment(targets[0].training_policy)
    pending, reused = prepare_targets(
        targets,
        resume=args.resume,
        allow_retest=args.allow_retest,
        audit_reuse_legacy=args.audit_reuse_legacy,
    )
    for target in targets:
        action = "evaluate" if target in pending else ("audit" if target.protocol_pollution else "reuse")
        print(
            f"{action:8s} {target.experiment_id}: {target.selection_source} "
            f"profile={target.selection_profile} history={target.history_index} "
            f"loop={target.loop_id} signature={target.selection_signature[:12]}",
        )
    if args.dry_run:
        return 0
    if pending:
        preflight(pending)
    if args.preflight_only:
        print(f"Preflight OK: {len(pending)} pending checkpoints, {len(reused)} reusable results")
        return 0
    for target in reused:
        existing = _existing_artifact(target)
        if existing and existing.get("state") == "succeeded":
            continue
        persist_legacy_result(target)
    if not pending:
        print(f"Final tests ready: {len(reused)} reused, 0 evaluated")
        return 0
    timeout_seconds = duration_seconds(args.task_timeout)
    gpus = visible_gpus(args.gpus)[: args.max_parallel]
    with routed_api_environment(os.environ) as (runtime_environment, api_routing):
        success = asyncio.run(run_workers(pending, gpus, timeout_seconds, runtime_environment, api_routing))
    print(f"Final tests: {len(reused)} reused, {len(pending)} evaluated")
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
