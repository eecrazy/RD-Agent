#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, TRY003, TRY004
"""Retrain missing matrix runs from clean same-task formal inputs.

Each target gets a new deterministic workspace and a fresh model.  Only the
data-generation script, 2,000-row training set, an independent validation set,
the dataset registry, data statistics, and the ordinary-LoRA three-epoch
configuration are copied. Prior model outputs and formal-training evidence are
never copied.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from run_formal_training_rescue import (
    EXPECTED_SAMPLES,
    MANIFEST_FILE,
    MATRIX_ROOT,
    SCHEMA_VERSION,
    TRAINING_BIN,
    TRAINING_POLICY,
    atomic_write_json,
    parse_gpus,
    read_json,
    run_one,
    safe_matrix_run,
    sha256,
    utc_now,
    validate_prepared,
)


@dataclass(frozen=True)
class CrossRunSpec:
    experiment_id: str
    task_id: str
    source_task_id: str
    source_workspace_id: str
    training_file: str
    validation_file: str
    validation_workspace_id: str | None = None
    enable_validation: bool = False


RESCUES = (
    CrossRunSpec(
        experiment_id="main/FinanceIQ_gen/run-2",
        task_id="main__FinanceIQ_gen__run-2",
        source_task_id="main__FinanceIQ_gen__run-1",
        source_workspace_id="9afb3a9bff57034002cf1d182e0dbf47",
        training_file="data.json",
        validation_file="validation.json",
    ),
    CrossRunSpec(
        experiment_id="main/panorama_pi4pc/run-1",
        task_id="main__panorama_pi4pc__run-1",
        source_task_id="main__panorama_pi4pc__run-2",
        source_workspace_id="92739247514f498fb5ebf0921106ebfb",
        validation_workspace_id="af05c17d3be440bcbae4f321cfbdf88b",
        training_file="data.json",
        validation_file="validation_data.json",
        enable_validation=True,
    ),
    CrossRunSpec(
        experiment_id="main/tablebench_data_analysis/run-1",
        task_id="main__tablebench_data_analysis__run-1",
        source_task_id="main__tablebench_data_analysis__run-3",
        source_workspace_id="2951a9f61d214a86a4e61b8dc77378e3",
        training_file="data.json",
        validation_file="data_validation.json",
    ),
    CrossRunSpec(
        experiment_id="main/tablebench_data_analysis/run-2",
        task_id="main__tablebench_data_analysis__run-2",
        source_task_id="main__tablebench_data_analysis__run-3",
        source_workspace_id="2951a9f61d214a86a4e61b8dc77378e3",
        training_file="data.json",
        validation_file="data_validation.json",
    ),
    CrossRunSpec(
        experiment_id="main/panorama_par4pc/run-2",
        task_id="main__panorama_par4pc__run-2",
        source_task_id="main__panorama_par4pc__run-1",
        source_workspace_id="073af76ba8b44a63847bbcac3bf05420",
        training_file="data.json",
        validation_file="data_validation.json",
    ),
    CrossRunSpec(
        experiment_id="main/panorama_par4pc/run-3",
        task_id="main__panorama_par4pc__run-3",
        source_task_id="main__panorama_par4pc__run-1",
        source_workspace_id="073af76ba8b44a63847bbcac3bf05420",
        training_file="data.json",
        validation_file="data_validation.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument("--gpus", default="0,2,3,5,6,7")
    parser.add_argument("--only", default=None, help="Regex matched against target experiment ids")
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def selected_specs(pattern: str | None) -> list[CrossRunSpec]:
    if pattern is None:
        return list(RESCUES)
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise SystemExit(f"Invalid --only expression: {error}") from error
    selected = [spec for spec in RESCUES if compiled.search(spec.experiment_id)]
    if not selected:
        raise SystemExit("--only did not select any cross-run rescue")
    return selected


def source_workspace(run_root: Path, spec: CrossRunSpec) -> Path:
    return run_root / spec.source_task_id / "workspace" / spec.source_workspace_id


def validation_workspace(run_root: Path, spec: CrossRunSpec) -> Path:
    workspace_id = spec.validation_workspace_id or spec.source_workspace_id
    return run_root / spec.source_task_id / "workspace" / workspace_id


def target_workspace_id(matrix_run: str, spec: CrossRunSpec) -> str:
    identity = "\0".join(
        (
            "rdagent-cross-run-formal-rescue-v1",
            matrix_run,
            spec.experiment_id,
            spec.source_task_id,
            spec.source_workspace_id,
            spec.validation_workspace_id or spec.source_workspace_id,
        ),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def source_paths(run_root: Path, spec: CrossRunSpec) -> dict[str, Path]:
    source = source_workspace(run_root, spec)
    validation_source = validation_workspace(run_root, spec)
    return {
        "process_data.py": source / "process_data.py",
        "train.yaml": source / "train.yaml",
        "dataset_info.json": source / "dataset_info.json",
        "data_stats.json": source / "data_stats.json",
        spec.training_file: source / spec.training_file,
        spec.validation_file: validation_source / spec.validation_file,
    }


def assert_sources(run_root: Path, spec: CrossRunSpec) -> dict[str, Path]:
    paths = source_paths(run_root, spec)
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Cross-run source must be a regular file ({label}): {path}")
    training = read_json(paths[spec.training_file], list)
    validation = read_json(paths[spec.validation_file], list)
    if len(training) != EXPECTED_SAMPLES:
        raise RuntimeError(f"{spec.experiment_id} source has {len(training)} rows, expected 2000")
    if not validation:
        raise RuntimeError(f"{spec.experiment_id} source has an empty validation set")
    return paths


def rendered_train_yaml(path: Path, spec: CrossRunSpec) -> str:
    original = path.read_text(encoding="utf-8")
    if not spec.enable_validation:
        return original
    plan = yaml.safe_load(original)
    if not isinstance(plan, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping")
    plan.update(
        {
            "eval_dataset": "processed_data_validation",
            "do_eval": True,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_total_limit": 1,
            "val_size": 0,
        },
    )
    return yaml.safe_dump(plan, default_flow_style=False, sort_keys=False)


def formal_dataset_info(path: Path, spec: CrossRunSpec) -> dict[str, Any]:
    registrations = read_json(path, dict)
    result: dict[str, Any] = {}
    for name, file_name in (
        ("processed_data", spec.training_file),
        ("processed_data_validation", spec.validation_file),
    ):
        registration = registrations.get(name)
        if not isinstance(registration, dict):
            raise RuntimeError(f"Dataset registration {name!r} is missing from {path}")
        normalized = dict(registration)
        normalized["file_name"] = file_name
        result[name] = normalized
    return result


def stable_manifest(matrix_run: str, spec: CrossRunSpec, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manager": "run_cross_run_training_rescue.py",
        "matrix_run": matrix_run,
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "source_task_id": spec.source_task_id,
        "source_workspace_id": spec.source_workspace_id,
        "validation_workspace_id": spec.validation_workspace_id or spec.source_workspace_id,
        "workspace_id": workspace_id,
        "expected_samples": EXPECTED_SAMPLES,
        "training_policy": TRAINING_POLICY,
        "training_method": "lora",
        "training_file": spec.training_file,
        "validation_file": spec.validation_file,
    }


def validate_cross_run_prepared(target: Path, task_root: Path, spec: CrossRunSpec) -> dict[str, Any]:
    facts = validate_prepared(target, task_root, spec)
    validation = facts["validation"]
    if validation["source"] != "independent_eval_dataset" or validation["sample_count"] < 1:
        raise RuntimeError(f"Cross-run rescue requires independent validation: {target}")
    return facts


def verify_managed_workspace(
    target: Path,
    *,
    matrix_run: str,
    spec: CrossRunSpec,
    workspace_id: str,
) -> None:
    manifest = read_json(target / MANIFEST_FILE, dict)
    expected = stable_manifest(matrix_run, spec, workspace_id)
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Managed workspace manifest mismatch ({', '.join(mismatches)}): {target}")
    hashes = manifest.get("input_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"Managed workspace has no input hashes: {target}")
    changed = [name for name, digest in hashes.items() if sha256(target / name) != digest]
    if changed:
        raise RuntimeError(f"Managed workspace inputs changed ({', '.join(changed)}): {target}")


def backfill_process_data(target: Path, source_path: Path) -> None:
    """Migrate a legacy cross-run workspace with its declared source script."""
    target_path = target / "process_data.py"
    if not source_path.is_file() or source_path.is_symlink():
        raise RuntimeError(f"Cross-run source process_data.py must be a regular file: {source_path}")
    source_digest = sha256(source_path)
    if target_path.exists():
        if not target_path.is_file() or target_path.is_symlink():
            raise RuntimeError(f"Cross-run process_data.py must be a regular file: {target_path}")
        if sha256(target_path) != source_digest:
            raise RuntimeError(f"Cross-run process_data.py differs from its declared source: {target_path}")
    else:
        shutil.copy2(source_path, target_path)

    manifest_path = target / MANIFEST_FILE
    manifest = read_json(manifest_path, dict)
    source_files = dict(manifest.get("source_files") or {})
    input_hashes = dict(manifest.get("input_hashes") or {})
    source_record = {"path": str(source_path.resolve()), "sha256": source_digest}
    existing_source = source_files.get("process_data.py")
    if existing_source is not None and existing_source != source_record:
        raise RuntimeError(f"Managed cross-run process_data.py source changed: {target}")
    existing_input = input_hashes.get("process_data.py")
    if existing_input is not None and existing_input != source_digest:
        raise RuntimeError(f"Managed cross-run process_data.py input changed: {target}")
    source_files["process_data.py"] = source_record
    input_hashes["process_data.py"] = source_digest
    if manifest.get("source_files") != source_files or manifest.get("input_hashes") != input_hashes:
        manifest["source_files"] = source_files
        manifest["input_hashes"] = input_hashes
        atomic_write_json(manifest_path, manifest)


def prepare_workspace(
    run_root: Path,
    matrix_run: str,
    spec: CrossRunSpec,
    *,
    resume: bool,
) -> Path:
    task_root = run_root / spec.task_id
    paths = assert_sources(run_root, spec)
    workspace_id = target_workspace_id(matrix_run, spec)
    target = task_root / "workspace" / workspace_id
    if target.exists():
        if not resume:
            raise RuntimeError(f"Cross-run workspace exists; pass --resume: {target}")
        verify_managed_workspace(
            target,
            matrix_run=matrix_run,
            spec=spec,
            workspace_id=workspace_id,
        )
        backfill_process_data(target, paths["process_data.py"])
        facts = validate_cross_run_prepared(target, task_root, spec)
        print(
            f"PREPARED resume {spec.experiment_id} workspace={workspace_id} "
            f"train={facts['training_sample_count']} val={facts['validation']['sample_count']}",
            flush=True,
        )
        return target

    temporary = target.with_name(f".{workspace_id}.prepare")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        shutil.copy2(paths["process_data.py"], temporary / "process_data.py")
        shutil.copy2(paths["data_stats.json"], temporary / "data_stats.json")
        shutil.copy2(paths[spec.training_file], temporary / spec.training_file)
        shutil.copy2(paths[spec.validation_file], temporary / spec.validation_file)
        (temporary / "train.yaml").write_text(
            rendered_train_yaml(paths["train.yaml"], spec),
            encoding="utf-8",
        )
        atomic_write_json(
            temporary / "dataset_info.json",
            formal_dataset_info(paths["dataset_info.json"], spec),
        )
        facts = validate_cross_run_prepared(temporary, task_root, spec)
        input_names = (
            "process_data.py",
            "train.yaml",
            "dataset_info.json",
            "data_stats.json",
            spec.training_file,
            spec.validation_file,
        )
        manifest = {
            **stable_manifest(matrix_run, spec, workspace_id),
            "created_at": utc_now(),
            "source_files": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in paths.items()
            },
            "input_hashes": {name: sha256(temporary / name) for name in input_names},
            "validation_samples": facts["validation"]["sample_count"],
        }
        atomic_write_json(temporary / MANIFEST_FILE, manifest)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"PREPARED new {spec.experiment_id} workspace={workspace_id} "
        f"train={facts['training_sample_count']} val={facts['validation']['sample_count']}",
        flush=True,
    )
    return target


async def async_main(args: argparse.Namespace) -> int:
    matrix_run = safe_matrix_run(args.matrix_run)
    run_root = MATRIX_ROOT / matrix_run
    if not run_root.is_dir():
        raise SystemExit(f"Matrix run does not exist: {run_root}")
    if not TRAINING_BIN.is_file():
        raise SystemExit(f"Training wrapper does not exist: {TRAINING_BIN}")
    specs = selected_specs(args.only)
    gpus = parse_gpus(args.gpus, len(specs))
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    prepared = [
        prepare_workspace(run_root, matrix_run, spec, resume=args.resume)
        for spec in specs
    ]
    if args.prepare_only:
        print("PREFLIGHT_OK " + ",".join(spec.experiment_id for spec in specs), flush=True)
        return 0

    results = await asyncio.gather(
        *(
            run_one(
                run_root,
                spec,
                workspace,
                gpu,
                args.timeout_seconds,
                resume=args.resume,
            )
            for spec, workspace, gpu in zip(specs, prepared, gpus, strict=True)
        ),
    )
    return 0 if all(results) else 1


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
