#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, PLR2004, TRY003, TRY300, TRY301
"""Train a strict rsLoRA counterpart for every formal ordinary-LoRA workspace."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .matrix import Experiment, main_experiments
    from .rslora_pairing import (
        PAIRING_ARTIFACT_FILE,
        PAIRING_EXPECTED_SAMPLES,
        PAIRING_MATRIX_SCHEMA_VERSION,
        PAIRING_RUN_KIND,
        PAIRING_SOURCE_POLICY,
        PAIRING_TARGET_POLICY,
        PairingContractError,
        build_pair_input_contract,
        file_sha256,
        make_pairing_artifact,
        paired_experiment_id,
        paired_train_yaml,
        paired_workspace_id,
        validate_pairing_artifact,
    )
    from .run_matrix import (
        FORMAL_METHOD_LOCK_FILE,
        FT_ROOT,
        GPU_CONFLICT_EXIT_CODE,
        ROOT,
        GPUExclusivityError,
        check_gpu_exclusivity,
        configure_project_environment,
        duration_seconds,
        monitor_gpu_exclusivity,
        safe_id,
        selected_gpu_uuids,
        status_write,
        stop_process,
        validate_max_parallel,
        validate_task_formal_training,
        visible_gpus,
    )
else:
    from matrix import Experiment, main_experiments
    from rslora_pairing import (
        PAIRING_ARTIFACT_FILE,
        PAIRING_EXPECTED_SAMPLES,
        PAIRING_MATRIX_SCHEMA_VERSION,
        PAIRING_RUN_KIND,
        PAIRING_SOURCE_POLICY,
        PAIRING_TARGET_POLICY,
        PairingContractError,
        build_pair_input_contract,
        file_sha256,
        make_pairing_artifact,
        paired_experiment_id,
        paired_train_yaml,
        paired_workspace_id,
        validate_pairing_artifact,
    )
    from run_matrix import (
        FORMAL_METHOD_LOCK_FILE,
        FT_ROOT,
        GPU_CONFLICT_EXIT_CODE,
        ROOT,
        GPUExclusivityError,
        check_gpu_exclusivity,
        configure_project_environment,
        duration_seconds,
        monitor_gpu_exclusivity,
        safe_id,
        selected_gpu_uuids,
        status_write,
        stop_process,
        validate_max_parallel,
        validate_task_formal_training,
        visible_gpus,
    )

from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    FormalTrainingEvidenceError,
    formal_training_provenance_files,
    make_formal_training_evidence,
    validate_formal_training_evidence,
)

TRAINING_BIN = FT_ROOT / "conda_envs" / "llm_finetune" / "bin" / "llamafactory-cli"
# The strict source par4pc runs take about 18 hours on one H20.  This guard
# covers trainer wall time (not the matrix agent's 12-hour search budget), so
# it must allow the unchanged full epoch schedule to finish.
DEFAULT_TASK_TIMEOUT = "24h"
PAIR_STATUS_FILE = "rslora_pair_status.json"
DOWNSTREAM_ARTIFACTS = (
    "validation_sweep",
    "validation_sweep.json",
    "validation_selection.json",
    "final_test.json",
    "final_test_spec.json",
    "final_test_workspaces",
)


@dataclass(frozen=True)
class PairSpec:
    source_experiment: Experiment
    source_task_root: Path
    source_workspace_id: str
    source_provenance: Path
    source_evidence_signature: str
    paired_experiment_id: str
    paired_workspace_id: str
    input_contract: dict[str, Any]
    benchmark_dataset_path: str

    @property
    def paired_task_id(self) -> str:
        return safe_id(self.paired_experiment_id)


def utc_now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", required=True, help="Completed 39-task paper-policy matrix name or path")
    parser.add_argument("--run-name", default=None, help="Output matrix name; defaults to <source>-rslora-paired")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids; default: all visible GPUs")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--task-timeout", default=DEFAULT_TASK_TIMEOUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-exclusive-gpus", action="store_true")
    parser.add_argument("--gpu-guard-interval", type=float, default=5.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Revalidate a completed paired matrix without training or evaluation",
    )
    return parser.parse_args()


def resolve_matrix_root(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = FT_ROOT / "logs" / "paper-matrix" / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairingContractError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise PairingContractError(f"{path} must contain a JSON object")
    return value


def _expected_main_tasks() -> dict[str, Experiment]:
    expected = main_experiments()
    if len(expected) != 39 or any(experiment.data_limit != PAIRING_EXPECTED_SAMPLES for experiment in expected):
        raise PairingContractError("Released main experiment inventory is no longer the 39 x 2,000 contract")
    return {experiment.experiment_id: experiment for experiment in expected}


def _validate_source_task_definition(task: dict[str, Any], expected: Experiment) -> None:
    expected_fields = {**expected.to_dict(), "formal_expected_samples": PAIRING_EXPECTED_SAMPLES}
    mismatches = [key for key, value in expected_fields.items() if task.get(key) != value]
    if mismatches:
        raise PairingContractError(
            f"Source matrix task {expected.experiment_id} differs from the released main contract: "
            + ", ".join(mismatches),
        )


def audit_source_matrix(source_root: Path) -> tuple[dict[str, Any], list[PairSpec]]:
    """Require a clean, complete 39-run paper matrix before deriving pairs."""
    matrix_path = source_root / "matrix.json"
    matrix = _read_json(matrix_path)
    if matrix.get("suite") != "main" or matrix.get("training_policy", "paper") != PAIRING_SOURCE_POLICY:
        raise PairingContractError("Source must be a suite=main, training_policy=paper matrix")
    contract = matrix.get("formal_training_contract")
    if not isinstance(contract, dict) or contract.get("rslora_is_paired_comparison_only") is not True:
        raise PairingContractError("Source matrix does not declare the paper/rsLoRA separation contract")

    expected_tasks = _expected_main_tasks()
    tasks = matrix.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(expected_tasks):
        raise PairingContractError(f"Source matrix must contain all {len(expected_tasks)} main tasks")
    by_id = {task.get("experiment_id"): task for task in tasks if isinstance(task, dict)}
    if len(by_id) != len(tasks) or set(by_id) != set(expected_tasks):
        raise PairingContractError("Source matrix experiment inventory is incomplete or duplicated")

    pair_specs: list[PairSpec] = []
    for experiment_id, expected in expected_tasks.items():
        task = by_id[experiment_id]
        _validate_source_task_definition(task, expected)
        task_root = source_root / safe_id(experiment_id)
        status = _read_json(task_root / "status.json")
        required_status = {
            "state": "succeeded",
            "experiment_id": experiment_id,
            "benchmark": expected.benchmark,
            "model": expected.model,
            "planner": expected.planner,
            "data_limit": PAIRING_EXPECTED_SAMPLES,
            "formal_expected_samples": PAIRING_EXPECTED_SAMPLES,
            "training_policy": PAIRING_SOURCE_POLICY,
        }
        mismatches = [key for key, value in required_status.items() if status.get(key) != value]
        if mismatches:
            raise PairingContractError(
                f"Source task {experiment_id} is not a matching formal success: {', '.join(mismatches)}",
            )
        benchmark_dataset_path = status.get("benchmark_dataset_path")
        if not isinstance(benchmark_dataset_path, str) or not benchmark_dataset_path:
            raise PairingContractError(f"Source task {experiment_id} has no pinned benchmark dataset")

        records, errors = validate_task_formal_training(
            task_root,
            expected_samples=PAIRING_EXPECTED_SAMPLES,
            experiment_id=experiment_id,
            training_policy=PAIRING_SOURCE_POLICY,
            require_visible_evidence=True,
        )
        if errors or not records:
            detail = "; ".join(errors) if errors else "no formal evidence"
            raise PairingContractError(f"Source task {experiment_id} has invalid durable evidence: {detail}")
        methods = {record["training_method"] for record in records}
        if methods not in ({"full"}, {"lora"}):
            raise PairingContractError(f"Source task {experiment_id} violates its Full-SFT/ordinary-LoRA method lock")
        method = next(iter(methods))
        if status.get("formal_training_method") != method:
            raise PairingContractError(f"Source task {experiment_id} status has a stale method")
        if status.get("formal_training_method_lock") != records[0].get("formal_training_method_lock"):
            raise PairingContractError(f"Source task {experiment_id} status has a stale method lock")
        status_evidence = status.get("formal_training_evidence")
        if not isinstance(status_evidence, list) or {
            item.get("evidence_signature") for item in status_evidence if isinstance(item, dict)
        } != {record["evidence_signature"] for record in records}:
            raise PairingContractError(f"Source task {experiment_id} status does not enumerate current formal evidence")

        if method == "full":
            continue
        target_experiment_id = paired_experiment_id(experiment_id)
        for record in records:
            source_workspace_id = str(record["workspace_id"])
            target_workspace_id = paired_workspace_id(source_workspace_id, record["evidence_signature"])
            input_contract = build_pair_input_contract(
                source_experiment_id=experiment_id,
                paired_experiment_id_value=target_experiment_id,
                source_workspace_id=source_workspace_id,
                paired_workspace_id_value=target_workspace_id,
                source_evidence_signature=record["evidence_signature"],
                source_provenance=record["provenance_path"],
                expected_samples=PAIRING_EXPECTED_SAMPLES,
            )
            pair_specs.append(
                PairSpec(
                    source_experiment=expected,
                    source_task_root=task_root,
                    source_workspace_id=source_workspace_id,
                    source_provenance=Path(record["provenance_path"]).resolve(),
                    source_evidence_signature=str(record["evidence_signature"]),
                    paired_experiment_id=target_experiment_id,
                    paired_workspace_id=target_workspace_id,
                    input_contract=input_contract,
                    benchmark_dataset_path=benchmark_dataset_path,
                ),
            )
    if not pair_specs:
        raise PairingContractError("The complete paper matrix chose no ordinary-LoRA task to pair")
    return matrix, pair_specs


def _group_specs(specs: list[PairSpec]) -> dict[str, list[PairSpec]]:
    grouped: dict[str, list[PairSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.paired_experiment_id, []).append(spec)
    return grouped


def matrix_manifest(source_root: Path, source_matrix: dict[str, Any], specs: list[PairSpec]) -> dict[str, Any]:
    grouped = _group_specs(specs)
    source_assets = source_matrix.get("benchmark_assets", {})
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "suite": "main-rslora-paired",
        "training_policy": PAIRING_TARGET_POLICY,
        "formal_training_contract": {
            "schema_version": 1,
            "sample_count": PAIRING_EXPECTED_SAMPLES,
            "require_complete_epoch_schedule": True,
            "method": "rslora",
            "paired_source_method": "lora",
            "only_yaml_change": "use_rslora: false -> true",
        },
        "paired_comparison": {
            "schema_version": PAIRING_MATRIX_SCHEMA_VERSION,
            "run_kind": PAIRING_RUN_KIND,
            "source_matrix_root": str(source_root.resolve()),
            "source_matrix_manifest_sha256": file_sha256(source_root / "matrix.json"),
            "source_task_count": len(source_matrix["tasks"]),
            "paired_task_count": len(grouped),
            "paired_workspace_count": len(specs),
        },
        "benchmark_assets": {
            spec.source_experiment.benchmark: source_assets[spec.source_experiment.benchmark]
            for spec in specs
        },
        "tasks": [
            {
                "experiment_id": experiment_id,
                "suite": "main-rslora-paired",
                "benchmark": task_specs[0].source_experiment.benchmark,
                "model": task_specs[0].source_experiment.model,
                "planner": task_specs[0].source_experiment.planner,
                "data_limit": PAIRING_EXPECTED_SAMPLES,
                "timeout": task_specs[0].source_experiment.timeout,
                "formal_expected_samples": PAIRING_EXPECTED_SAMPLES,
                "pairing": {
                    "source_experiment_id": task_specs[0].source_experiment.experiment_id,
                    "expected_pairs": [
                        spec.input_contract
                        for spec in sorted(task_specs, key=lambda item: item.paired_workspace_id)
                    ],
                },
            }
            for experiment_id, task_specs in sorted(grouped.items())
        ],
    }


def write_or_validate_manifest(run_root: Path, manifest: dict[str, Any]) -> None:
    path = run_root / "matrix.json"
    if not path.exists():
        status_write(path, manifest)
        return
    existing = _read_json(path)
    stable = lambda value: {key: item for key, item in value.items() if key != "created_at"}  # noqa: E731
    if stable(existing) != stable(manifest):
        raise PairingContractError(f"Existing paired matrix manifest differs: {path}")


def audit_paired_matrix(
    run_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Revalidate complete one-to-one ordinary-LoRA/rsLoRA training evidence."""
    run_root = run_root.resolve()
    manifest = _read_json(run_root / "matrix.json")
    comparison = manifest.get("paired_comparison")
    if not isinstance(comparison, dict):
        raise PairingContractError("Paired matrix has no paired-comparison declaration")
    declared_source = Path(str(comparison.get("source_matrix_root", ""))).resolve()
    if source_root is not None and declared_source != source_root.resolve():
        raise PairingContractError("Paired matrix points to a different source matrix")
    source_root = declared_source
    source_matrix, specs = audit_source_matrix(source_root)
    expected_manifest = matrix_manifest(source_root, source_matrix, specs)
    stable = lambda value: {key: item for key, item in value.items() if key != "created_at"}  # noqa: E731
    if stable(manifest) != stable(expected_manifest):
        raise PairingContractError("Paired matrix manifest no longer matches the current strict source evidence")

    grouped = _group_specs(specs)
    task_signatures: dict[str, str] = {}
    validated_pairs = 0
    for experiment_id, task_specs in sorted(grouped.items()):
        task_root = run_root / safe_id(experiment_id)
        artifact = validate_pairing_artifact(
            task_root,
            experiment_id=experiment_id,
            expected_samples=PAIRING_EXPECTED_SAMPLES,
        )
        pairs = artifact.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != len(task_specs):
            raise PairingContractError(f"Paired artifact count changed for {experiment_id}")
        signature = artifact.get("artifact_signature")
        if not isinstance(signature, str):
            raise PairingContractError(f"Paired artifact signature is missing for {experiment_id}")
        task_signatures[experiment_id] = signature
        validated_pairs += len(pairs)

    if validated_pairs != len(specs):
        raise PairingContractError("Paired workspace audit did not cover every ordinary-LoRA workspace")
    return {
        "schema_version": 1,
        "source_matrix_root": str(source_root),
        "paired_matrix_root": str(run_root),
        "source_task_count": len(source_matrix["tasks"]),
        "ordinary_lora_task_count": len(grouped),
        "paired_task_count": len(task_signatures),
        "paired_workspace_count": validated_pairs,
        "expected_samples_per_workspace": PAIRING_EXPECTED_SAMPLES,
        "only_yaml_change": "use_rslora: false -> true",
        "state": "succeeded",
        "task_artifact_signatures": task_signatures,
    }


def _copy_pair_inputs(spec: PairSpec, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    for relative in formal_training_provenance_files(spec.source_provenance):
        if relative == FORMAL_TRAINING_EVIDENCE_FILE:
            continue
        source = spec.source_provenance / relative
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "train.yaml":
            destination.write_bytes(paired_train_yaml(source.read_bytes()))
        else:
            shutil.copy2(source, destination, follow_symlinks=True)
            if source.read_bytes() != destination.read_bytes():
                raise PairingContractError(f"Paired input copy changed bytes: {source}")
    current = build_pair_input_contract(
        source_experiment_id=spec.source_experiment.experiment_id,
        paired_experiment_id_value=spec.paired_experiment_id,
        source_workspace_id=spec.source_workspace_id,
        paired_workspace_id_value=spec.paired_workspace_id,
        source_evidence_signature=spec.source_evidence_signature,
        source_provenance=spec.source_provenance,
        paired_provenance=None,
        expected_samples=PAIRING_EXPECTED_SAMPLES,
    )
    # The target has no evidence yet, so validate the copied files directly.
    target_yaml = (workspace / "train.yaml").read_bytes()
    if file_sha256(workspace / "train.yaml") != current["paired_train_yaml_sha256"]:
        raise PairingContractError("Prepared rsLoRA train.yaml does not match its signed input contract")
    for item in current["input_files"]:
        if file_sha256(workspace / item["path"]) != item["sha256"]:
            raise PairingContractError(f"Prepared paired input changed: {item['path']}")
    source_yaml = (spec.source_provenance / "train.yaml").read_bytes()
    if current != spec.input_contract or target_yaml != paired_train_yaml(source_yaml):
        raise PairingContractError("Prepared workspace does not match the matrix pair contract")


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)
    return destination


def _has_weights(path: Path) -> bool:
    patterns = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    )
    return any(any(path.glob(pattern)) for pattern in patterns)


def _preserve_output(source: Path, destination: Path) -> None:
    temporary = destination.with_name("output.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        target = temporary / item.name
        if item.is_file():
            _link_or_copy(str(item), str(target))
        elif item.is_symlink():
            target.symlink_to(item.readlink())
        elif item.is_dir() and item.name.startswith("checkpoint-") and _has_weights(item):
            target.mkdir()
            for child in item.iterdir():
                child_target = target / child.name
                if child.is_symlink():
                    child_target.symlink_to(child.readlink())
                elif child.is_file() and child.name not in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}:
                    _link_or_copy(str(child), str(child_target))
    if not _has_weights(temporary):
        shutil.rmtree(temporary, ignore_errors=True)
        raise PairingContractError(f"Training output has no rsLoRA weights: {source}")
    shutil.rmtree(destination, ignore_errors=True)
    temporary.replace(destination)


def _persist_formal_run(workspace: Path, durable_root: Path, spec: PairSpec) -> dict[str, Any]:
    output = workspace / "output"
    evidence = make_formal_training_evidence(
        workspace,
        expected_samples=PAIRING_EXPECTED_SAMPLES,
        experiment_id=spec.paired_experiment_id,
        training_policy=PAIRING_TARGET_POLICY,
        output_path=output,
    )
    status_write(workspace / FORMAL_TRAINING_EVIDENCE_FILE, evidence)
    validate_formal_training_evidence(
        workspace,
        expected_samples=PAIRING_EXPECTED_SAMPLES,
        experiment_id=spec.paired_experiment_id,
        training_policy=PAIRING_TARGET_POLICY,
        output_path=output,
    )

    durable_output = durable_root / "output"
    _preserve_output(output, durable_output)
    provenance = durable_root / "formal_training"
    temporary = durable_root / "formal_training.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    for relative in formal_training_provenance_files(workspace):
        source = workspace / relative
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
        if source.read_bytes() != target.read_bytes():
            raise PairingContractError(f"Durable paired provenance changed bytes: {source}")
    validate_formal_training_evidence(
        temporary,
        expected_samples=PAIRING_EXPECTED_SAMPLES,
        experiment_id=spec.paired_experiment_id,
        training_policy=PAIRING_TARGET_POLICY,
        output_path=durable_output,
    )
    shutil.rmtree(provenance, ignore_errors=True)
    temporary.replace(provenance)
    return evidence


def _pair_complete(run_root: Path, spec: PairSpec) -> bool:
    task_root = run_root / spec.paired_task_id
    workspace = task_root / "workspace" / spec.paired_workspace_id
    durable = task_root / "workspace" / ".ft_model_checkpoints" / spec.paired_workspace_id
    try:
        visible = validate_formal_training_evidence(
            workspace,
            expected_samples=PAIRING_EXPECTED_SAMPLES,
            experiment_id=spec.paired_experiment_id,
            training_policy=PAIRING_TARGET_POLICY,
            output_path=workspace / "output",
        )
        durable_evidence = validate_formal_training_evidence(
            durable / "formal_training",
            expected_samples=PAIRING_EXPECTED_SAMPLES,
            experiment_id=spec.paired_experiment_id,
            training_policy=PAIRING_TARGET_POLICY,
            output_path=durable / "output",
        )
        if visible["evidence_signature"] != durable_evidence["evidence_signature"]:
            return False
        contract = build_pair_input_contract(
            source_experiment_id=spec.source_experiment.experiment_id,
            paired_experiment_id_value=spec.paired_experiment_id,
            source_workspace_id=spec.source_workspace_id,
            paired_workspace_id_value=spec.paired_workspace_id,
            source_evidence_signature=spec.source_evidence_signature,
            source_provenance=spec.source_provenance,
            paired_provenance=durable / "formal_training",
            expected_samples=PAIRING_EXPECTED_SAMPLES,
        )
        return contract == spec.input_contract
    except (FormalTrainingEvidenceError, PairingContractError, OSError):
        return False


def _source_train_runtime_seconds(spec: PairSpec) -> float:
    """Return the measured ordinary-LoRA runtime used only for queue ordering."""
    results_path = (
        spec.source_task_root
        / "workspace"
        / spec.source_workspace_id
        / "output"
        / "train_results.json"
    )
    results = _read_json(results_path)
    value = results.get("train_runtime")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise PairingContractError(f"Source training runtime is missing or invalid: {results_path}")
    return float(value)


def _schedule_longest_first(specs: list[PairSpec]) -> list[PairSpec]:
    """Minimize the eight-GPU tail using measured paired-source runtimes."""
    runtimes = {id(spec): _source_train_runtime_seconds(spec) for spec in specs}
    return sorted(
        specs,
        key=lambda spec: (
            -runtimes[id(spec)],
            spec.paired_experiment_id,
            spec.paired_workspace_id,
        ),
    )


def _prepare_pair(run_root: Path, spec: PairSpec, *, resume: bool) -> bool:
    task_root = run_root / spec.paired_task_id
    workspace_root = task_root / "workspace"
    workspace = workspace_root / spec.paired_workspace_id
    durable = workspace_root / ".ft_model_checkpoints" / spec.paired_workspace_id
    if workspace.exists() or durable.exists():
        if not resume:
            raise PairingContractError(f"Paired workspace already exists; use --resume or a new run: {workspace}")
        if _pair_complete(run_root, spec):
            return False
        for relative in DOWNSTREAM_ARTIFACTS:
            if (task_root / relative).exists():
                raise PairingContractError(
                    "Cannot replace incomplete paired training after downstream evaluation exists: "
                    f"{task_root / relative}",
                )
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(durable, ignore_errors=True)
    _copy_pair_inputs(spec, workspace)
    status_write(
        workspace / PAIR_STATUS_FILE,
        {
            "schema_version": 1,
            "state": "prepared",
            "source_experiment_id": spec.source_experiment.experiment_id,
            "paired_experiment_id": spec.paired_experiment_id,
            "source_workspace_id": spec.source_workspace_id,
            "paired_workspace_id": spec.paired_workspace_id,
            "input_contract_signature": spec.input_contract["input_contract_signature"],
            "expected_samples": PAIRING_EXPECTED_SAMPLES,
            "prepared_at": utc_now(),
        },
    )
    return True


def _initialize_task_status(run_root: Path, task_specs: list[PairSpec]) -> None:
    first = task_specs[0]
    task_root = run_root / first.paired_task_id
    task_root.mkdir(parents=True, exist_ok=True)
    existing_path = task_root / "status.json"
    if existing_path.is_file():
        existing = _read_json(existing_path)
        stable = {
            "experiment_id": first.paired_experiment_id,
            "source_experiment_id": first.source_experiment.experiment_id,
            "benchmark": first.source_experiment.benchmark,
            "model": first.source_experiment.model,
            "data_limit": PAIRING_EXPECTED_SAMPLES,
            "formal_expected_samples": PAIRING_EXPECTED_SAMPLES,
            "training_policy": PAIRING_TARGET_POLICY,
            "run_kind": PAIRING_RUN_KIND,
        }
        if any(existing.get(key) != value for key, value in stable.items()):
            raise PairingContractError(f"Existing paired task status differs: {existing_path}")
        return
    status_write(
        existing_path,
        {
            "schema_version": 1,
            "state": "running",
            "run_kind": PAIRING_RUN_KIND,
            "experiment_id": first.paired_experiment_id,
            "source_experiment_id": first.source_experiment.experiment_id,
            "suite": "main-rslora-paired",
            "benchmark": first.source_experiment.benchmark,
            "model": first.source_experiment.model,
            "planner": first.source_experiment.planner,
            "data_limit": PAIRING_EXPECTED_SAMPLES,
            "timeout": first.source_experiment.timeout,
            "benchmark_dataset_path": first.benchmark_dataset_path,
            "training_policy": PAIRING_TARGET_POLICY,
            "formal_expected_samples": PAIRING_EXPECTED_SAMPLES,
            "formal_method_lock_path": str(task_root / FORMAL_METHOD_LOCK_FILE),
            "formal_training_method": None,
            "formal_training_method_lock": None,
            "formal_training_evidence": [],
            "pairing_artifact_signature": None,
            "expected_pair_count": len(task_specs),
            "started_at": utc_now(),
        },
    )


def _finalization_finished_at(status: dict[str, Any]) -> str:
    """Keep a successful task's original completion time during ``--resume``.

    The task-level timestamp is provenance consumed by validation selection.
    Re-auditing an already-complete paired matrix must therefore not make a
    semantically unchanged training run appear to be newly completed.
    """
    existing = status.get("finished_at")
    if status.get("state") == "succeeded" and isinstance(existing, str) and existing:
        return existing
    return utc_now()


def _training_environment(run_root: Path, task_root: Path, spec: PairSpec, gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "FT_EXPERIMENT_ID": spec.paired_experiment_id,
            "FT_TRAINING_POLICY": PAIRING_TARGET_POLICY,
            "FT_FORMAL_EXPECTED_SAMPLES": str(PAIRING_EXPECTED_SAMPLES),
            "FT_FORMAL_METHOD_LOCK_PATH": str(task_root / FORMAL_METHOD_LOCK_FILE),
            "FT_GPU_LEASE_POOL_FILE": str(run_root / ".disabled_dynamic_gpu_pool"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    return environment


async def run_one(run_root: Path, spec: PairSpec, gpu: str, timeout_seconds: int) -> bool:
    task_root = run_root / spec.paired_task_id
    workspace = task_root / "workspace" / spec.paired_workspace_id
    durable = task_root / "workspace" / ".ft_model_checkpoints" / spec.paired_workspace_id
    pair_status_path = workspace / PAIR_STATUS_FILE
    status = _read_json(pair_status_path)
    status.update({"state": "running", "gpu": gpu, "started_at": utc_now()})
    status_write(pair_status_path, status)
    print(f"START gpu={gpu} {spec.paired_experiment_id} workspace={spec.paired_workspace_id}", flush=True)
    command = [str(TRAINING_BIN), "train", "train.yaml"]
    try:
        with (workspace / "train.console.log").open("ab", buffering=0) as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=_training_environment(run_root, task_root, spec, gpu),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except TimeoutError:
                timed_out = True
                await stop_process(process)
                return_code = process.returncode
            except asyncio.CancelledError:
                await stop_process(process)
                raise
        if return_code != 0 or timed_out:
            raise RuntimeError(f"trainer exit_code={return_code}, outer_timeout={timed_out}")
        evidence = await asyncio.to_thread(_persist_formal_run, workspace, durable, spec)
        if not _pair_complete(run_root, spec):
            raise PairingContractError("Completed pair failed independent durable revalidation")
        status.update(
            {
                "state": "succeeded",
                "finished_at": utc_now(),
                "return_code": return_code,
                "outer_timeout": timed_out,
                "formal_training_evidence_signature": evidence["evidence_signature"],
                "global_step": evidence["completion"]["global_step"],
                "max_steps": evidence["completion"]["max_steps"],
            },
        )
        status_write(pair_status_path, status)
        print(f"DONE  gpu={gpu} {spec.paired_experiment_id} workspace={spec.paired_workspace_id}", flush=True)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - persist every orchestration/training failure.
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(pair_status_path, status)
        print(f"FAIL  gpu={gpu} {spec.paired_experiment_id} workspace={spec.paired_workspace_id}", flush=True)
        return False


async def run_workers(
    run_root: Path,
    specs: list[PairSpec],
    gpus: list[str],
    timeout_seconds: int,
    *,
    guarded_gpu_uuids: set[str] | None,
    guard_interval: float,
) -> bool:
    queue: asyncio.Queue[PairSpec] = asyncio.Queue()
    for spec in specs:
        queue.put_nowait(spec)
    results: list[bool] = []

    async def worker(gpu: str) -> None:
        while True:
            try:
                spec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(await run_one(run_root, spec, gpu, timeout_seconds))
            finally:
                queue.task_done()

    workers = asyncio.gather(*(worker(gpu) for gpu in gpus))
    if guarded_gpu_uuids is None:
        await workers
        return all(results)
    guard = asyncio.create_task(monitor_gpu_exclusivity(os.getpid(), guarded_gpu_uuids, guard_interval))
    done, _ = await asyncio.wait({workers, guard}, return_when=asyncio.FIRST_COMPLETED)
    if workers in done:
        guard.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await guard
        await workers
        return all(results)
    try:
        await guard
    finally:
        workers.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await workers
    return False


def _finalize_task(run_root: Path, task_specs: list[PairSpec]) -> bool:
    first = task_specs[0]
    task_root = run_root / first.paired_task_id
    status_path = task_root / "status.json"
    status = _read_json(status_path)
    finished_at = _finalization_finished_at(status)
    pair_statuses = [
        _read_json(task_root / "workspace" / spec.paired_workspace_id / PAIR_STATUS_FILE) for spec in task_specs
    ]
    if any(item.get("state") != "succeeded" for item in pair_statuses):
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "pair_runs": pair_statuses,
                "formal_training_validation_errors": ["one or more paired workspace runs failed"],
            },
        )
        status_write(status_path, status)
        return False
    records, errors = validate_task_formal_training(
        task_root,
        expected_samples=PAIRING_EXPECTED_SAMPLES,
        experiment_id=first.paired_experiment_id,
        training_policy=PAIRING_TARGET_POLICY,
        require_visible_evidence=True,
    )
    if errors or len(records) != len(task_specs) or {item["training_method"] for item in records} != {"rslora"}:
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "pair_runs": pair_statuses,
                "formal_training_validation_errors": errors or ["paired formal record count/method mismatch"],
            },
        )
        status_write(status_path, status)
        return False

    status.update(
        {
            "state": "finalizing",
            "finished_at": finished_at,
            "pair_runs": pair_statuses,
            "formal_training_evidence": records,
            "formal_training_validation_errors": [],
            "formal_training_method": "rslora",
            "formal_training_method_lock": records[0]["formal_training_method_lock"],
        },
    )
    status_write(status_path, status)
    try:
        artifact = make_pairing_artifact(task_root)
        status_write(task_root / PAIRING_ARTIFACT_FILE, artifact)
        status.update(
            {
                "state": "succeeded",
                "pairing_artifact_signature": artifact["artifact_signature"],
            },
        )
        status_write(status_path, status)
        validate_pairing_artifact(
            task_root,
            experiment_id=first.paired_experiment_id,
            expected_samples=PAIRING_EXPECTED_SAMPLES,
            expected_signature=artifact["artifact_signature"],
        )
    except Exception as error:  # noqa: BLE001 - task must fail closed on any provenance defect.
        status.update(
            {
                "state": "failed",
                "pairing_artifact_signature": None,
                "formal_training_validation_errors": [f"{type(error).__name__}: {error}"],
            },
        )
        status_write(status_path, status)
        return False
    return True


def _select_gpus(args: argparse.Namespace) -> tuple[list[str], set[str] | None]:
    gpus = visible_gpus(args.gpus)
    if args.max_parallel is not None:
        gpus = gpus[: args.max_parallel]
    if not gpus:
        raise PairingContractError("No GPU remains after --max-parallel")
    guarded = None
    if args.require_exclusive_gpus:
        guarded = selected_gpu_uuids(gpus)
        check_gpu_exclusivity(os.getpid(), guarded)
    return gpus, guarded


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    configure_project_environment(PAIRING_TARGET_POLICY)
    validate_max_parallel(args.max_parallel)
    if args.gpu_guard_interval <= 0:
        raise SystemExit("--gpu-guard-interval must be positive")
    timeout_seconds = duration_seconds(args.task_timeout)
    source_root = resolve_matrix_root(args.source_matrix)
    source_matrix, specs = audit_source_matrix(source_root)
    grouped = _group_specs(specs)
    print(
        f"Source audit OK: 39/39 tasks; ordinary-LoRA tasks={len(grouped)}; paired workspaces={len(specs)}",
        flush=True,
    )
    for spec in specs:
        print(
            f"  {spec.source_experiment.experiment_id}/{spec.source_workspace_id} -> "
            f"{spec.paired_experiment_id}/{spec.paired_workspace_id} "
            f"contract={spec.input_contract['input_contract_signature'][:12]}",
        )
    run_name = args.run_name or f"{source_root.name}-rslora-paired"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise SystemExit(f"Unsafe run name: {run_name!r}")
    run_root = FT_ROOT / "logs" / "paper-matrix" / run_name
    if args.audit_only:
        print(json.dumps(audit_paired_matrix(run_root, source_root=source_root), indent=2, sort_keys=True))
        return 0
    if not TRAINING_BIN.is_file():
        raise SystemExit(f"Training backend is missing: {TRAINING_BIN}")
    if args.preflight_only or args.dry_run:
        return 0

    if run_root.exists() and not args.resume:
        raise SystemExit(f"Paired run already exists; use --resume or another --run-name: {run_root}")
    gpus, guarded = _select_gpus(args)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = matrix_manifest(source_root, source_matrix, specs)
    write_or_validate_manifest(run_root, manifest)
    for task_specs in grouped.values():
        _initialize_task_status(run_root, task_specs)

    pending: list[PairSpec] = []
    for spec in specs:
        if _prepare_pair(run_root, spec, resume=args.resume):
            pending.append(spec)
        else:
            print(f"REUSE {spec.paired_experiment_id} workspace={spec.paired_workspace_id}", flush=True)
    pending = _schedule_longest_first(pending)
    print(
        f"Paired training: {len(pending)} pending; GPUs={','.join(gpus)}; "
        f"schedule=longest-source-runtime-first; output={run_root}",
        flush=True,
    )
    for index, spec in enumerate(pending, start=1):
        print(
            f"  QUEUE {index:02d} estimated_seconds={_source_train_runtime_seconds(spec):.1f} "
            f"{spec.paired_experiment_id} workspace={spec.paired_workspace_id}",
            flush=True,
        )
    success = True
    if pending:
        try:
            success = asyncio.run(
                run_workers(
                    run_root,
                    pending,
                    gpus,
                    timeout_seconds,
                    guarded_gpu_uuids=guarded,
                    guard_interval=args.gpu_guard_interval,
                ),
            )
        except GPUExclusivityError as error:
            status_write(
                run_root / "invalid.json",
                {
                    "schema_version": 1,
                    "reason": "external_gpu_conflict",
                    "detected_at": utc_now(),
                    "selected_gpus": gpus,
                    "processes": error.processes,
                    "query_error": error.query_error,
                },
            )
            print(f"GPU EXCLUSIVITY FAILURE: {error}", flush=True)
            return GPU_CONFLICT_EXIT_CODE
    finalized = [_finalize_task(run_root, task_specs) for task_specs in grouped.values()]
    print(
        f"Paired matrix complete: tasks={sum(finalized)}/{len(finalized)}, "
        "workspaces="
        f"{len(specs) - len(pending) + sum(1 for spec in pending if _pair_complete(run_root, spec))}/"
        f"{len(specs)}",
        flush=True,
    )
    return 0 if success and all(finalized) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
