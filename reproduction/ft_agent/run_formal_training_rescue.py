#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, FLY002, PLR2004, TRY003, TRY004, TRY300, TRY301
"""Run isolated ordinary-LoRA formal-training rescues for known matrix workspaces.

The source workspaces are left untouched.  Each rescue copies only the formal
configuration, data-generation script, selected 2,000-row training set,
independent validation set, and data statistics into a deterministic sibling
workspace.  The normal H20 training wrapper still owns the shared GPU lease
and formal-method lock.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rdagent.scenarios.finetune.experiment.workspace import FTWorkspace
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    FormalTrainingEvidenceError,
    make_formal_training_evidence,
    validate_formal_training_evidence,
    validate_formal_training_inputs,
    validate_formal_training_method_lock,
)

ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = ROOT / "finetune_files"
MATRIX_ROOT = FT_ROOT / "logs" / "paper-matrix"
TRAINING_BIN = FT_ROOT / "conda_envs" / "llm_finetune" / "bin" / "llamafactory-cli"
GPU_LOCK_ROOT = FT_ROOT / "gpu_leases" / "locks"
EXPECTED_SAMPLES = 2_000
TRAINING_POLICY = "paper"
FORMAL_METHOD_LOCK_FILE = "formal_training_method.json"
MANIFEST_FILE = "formal_rescue_manifest.json"
STATUS_FILE = "formal_rescue_status.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RescueSpec:
    experiment_id: str
    task_id: str
    source_workspace_id: str
    training_file: str
    validation_file: str
    enable_validation: bool = False


RESCUES = (
    RescueSpec(
        experiment_id="main/FinanceIQ_gen/run-1",
        task_id="main__FinanceIQ_gen__run-1",
        source_workspace_id="f33b65a087ee44d7bc5b75213162f37e",
        training_file="data.json",
        validation_file="validation.json",
    ),
    RescueSpec(
        experiment_id="main/chemcotbench_mol_und/run-1",
        task_id="main__chemcotbench_mol_und__run-1",
        source_workspace_id="a00a237a2cfe4b6fa5279bd0a61af91f",
        training_file="processed_data_train_full.json",
        validation_file="processed_data_validation_full.json",
    ),
    RescueSpec(
        experiment_id="main/tablebench_visualization/run-1",
        task_id="main__tablebench_visualization__run-1",
        source_workspace_id="17035a0985af4744a510ee54c65302ff",
        training_file="data.json",
        validation_file="validation_data.json",
        enable_validation=True,
    ),
    RescueSpec(
        experiment_id="main/aime25/run-3",
        task_id="main__aime25__run-3",
        source_workspace_id="f05d474df7d84ec588a92321bfc5ee41",
        training_file="data.json",
        validation_file="validation_data.json",
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument("--gpus", default="0,2,4", help="One physical GPU id per selected rescue")
    parser.add_argument("--only", default=None, help="Regular expression matched against experiment ids")
    parser.add_argument("--timeout-seconds", type=int, default=12 * 60 * 60)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def safe_matrix_run(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise SystemExit(f"Unsafe matrix run name: {value!r}")
    return value


def selected_specs(pattern: str | None) -> list[RescueSpec]:
    if pattern is None:
        return list(RESCUES)
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise SystemExit(f"Invalid --only expression: {error}") from error
    selected = [spec for spec in RESCUES if compiled.search(spec.experiment_id)]
    if not selected:
        raise SystemExit("--only did not select any rescue")
    return selected


def parse_gpus(value: str, expected: int) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if len(gpus) != expected or len(set(gpus)) != expected:
        raise SystemExit(f"Expected exactly {expected} distinct GPU ids; got {value!r}")
    if any(re.fullmatch(r"[0-9]+", gpu) is None for gpu in gpus):
        raise SystemExit("GPU ids must be non-negative integers")
    return gpus


def target_workspace_id(matrix_run: str, spec: RescueSpec) -> str:
    identity = "\0".join(
        (
            "rdagent-formal-rescue-v1",
            matrix_run,
            spec.experiment_id,
            spec.source_workspace_id,
        ),
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, expected_type):
        raise RuntimeError(f"{path} must contain {expected_type.__name__}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping")
    return value


def source_files(source: Path, spec: RescueSpec) -> tuple[Path, ...]:
    return (
        source / "process_data.py",
        source / "train.yaml",
        source / "dataset_info.json",
        source / "data_stats.json",
        source / spec.training_file,
        source / spec.validation_file,
    )


def assert_source(source: Path, spec: RescueSpec) -> None:
    if not source.is_dir():
        raise RuntimeError(f"Source workspace is missing: {source}")
    for path in source_files(source, spec):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Rescue source must be a regular file: {path}")
    training = read_json(source / spec.training_file, list)
    validation = read_json(source / spec.validation_file, list)
    if len(training) != EXPECTED_SAMPLES:
        raise RuntimeError(f"{spec.experiment_id} has {len(training)} training rows, expected 2000")
    if not validation:
        raise RuntimeError(f"{spec.experiment_id} has an empty validation set")


def rendered_train_yaml(source: Path, spec: RescueSpec) -> str:
    original = source.read_text(encoding="utf-8")
    plan = load_yaml(source)
    if spec.enable_validation:
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
        original = yaml.safe_dump(plan, default_flow_style=False, sort_keys=False)
    return original


def formal_dataset_info(source: Path, spec: RescueSpec) -> dict[str, Any]:
    registrations = read_json(source / "dataset_info.json", dict)
    result: dict[str, Any] = {}
    for name, file_name in (
        ("processed_data", spec.training_file),
        ("processed_data_validation", spec.validation_file),
    ):
        registration = registrations.get(name)
        if not isinstance(registration, dict):
            raise RuntimeError(f"Dataset registration {name!r} is missing from {source}")
        normalized = dict(registration)
        normalized["file_name"] = file_name
        result[name] = normalized
    return result


def manifest_stable_fields(matrix_run: str, spec: RescueSpec, workspace_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manager": "run_formal_training_rescue.py",
        "matrix_run": matrix_run,
        "experiment_id": spec.experiment_id,
        "source_workspace_id": spec.source_workspace_id,
        "workspace_id": workspace_id,
        "expected_samples": EXPECTED_SAMPLES,
        "training_policy": TRAINING_POLICY,
        "training_method": "lora",
        "training_file": spec.training_file,
        "validation_file": spec.validation_file,
    }


def verify_managed_workspace(
    target: Path,
    *,
    matrix_run: str,
    spec: RescueSpec,
    workspace_id: str,
) -> None:
    manifest = read_json(target / MANIFEST_FILE, dict)
    expected = manifest_stable_fields(matrix_run, spec, workspace_id)
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Managed workspace manifest mismatch ({', '.join(mismatches)}): {target}")
    hashes = manifest.get("input_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"Managed workspace has no input hashes: {target}")
    changed = [name for name, digest in hashes.items() if sha256(target / name) != digest]
    if changed:
        raise RuntimeError(f"Managed workspace inputs changed ({', '.join(changed)}): {target}")


def backfill_process_data(target: Path, source: Path) -> None:
    """Migrate a legacy rescue workspace without changing its training inputs."""
    source_path = source / "process_data.py"
    target_path = target / "process_data.py"
    if not source_path.is_file() or source_path.is_symlink():
        raise RuntimeError(f"Rescue source process_data.py must be a regular file: {source_path}")
    source_digest = sha256(source_path)
    if target_path.exists():
        if not target_path.is_file() or target_path.is_symlink():
            raise RuntimeError(f"Rescue process_data.py must be a regular file: {target_path}")
        if sha256(target_path) != source_digest:
            raise RuntimeError(f"Rescue process_data.py differs from its declared source: {target_path}")
    else:
        shutil.copy2(source_path, target_path)

    manifest_path = target / MANIFEST_FILE
    manifest = read_json(manifest_path, dict)
    source_hashes = dict(manifest.get("source_hashes") or {})
    input_hashes = dict(manifest.get("input_hashes") or {})
    for hashes, label in ((source_hashes, "source"), (input_hashes, "input")):
        existing = hashes.get("process_data.py")
        if existing is not None and existing != source_digest:
            raise RuntimeError(
                f"Managed workspace {label} process_data.py hash changed: {target}",
            )
        hashes["process_data.py"] = source_digest
    provenance = {
        "path": str(source_path.resolve()),
        "sha256": source_digest,
    }
    if (
        manifest.get("source_hashes") != source_hashes
        or manifest.get("input_hashes") != input_hashes
        or manifest.get("process_data_provenance") != provenance
    ):
        manifest["source_hashes"] = source_hashes
        manifest["input_hashes"] = input_hashes
        manifest["process_data_provenance"] = provenance
        atomic_write_json(manifest_path, manifest)


def validate_prepared(target: Path, task_root: Path, spec: RescueSpec) -> dict[str, Any]:
    facts = validate_formal_training_inputs(
        target,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id=spec.experiment_id,
        training_policy=TRAINING_POLICY,
        require_runtime_contract=False,
    )
    if facts["training_method"] != "lora":
        raise RuntimeError(f"Rescue unexpectedly selected {facts['training_method']}: {target}")
    if float(facts["configuration"]["num_train_epochs"]) != 3.0:
        raise RuntimeError(f"Rescue must preserve the three-epoch paper schedule: {target}")
    method_lock = task_root / FORMAL_METHOD_LOCK_FILE
    if method_lock.is_file():
        validate_formal_training_method_lock(
            method_lock,
            experiment_id=spec.experiment_id,
            training_policy=TRAINING_POLICY,
            training_method="lora",
        )
    return facts


def prepare_workspace(run_root: Path, matrix_run: str, spec: RescueSpec, *, resume: bool) -> Path:
    task_root = run_root / spec.task_id
    source = task_root / "workspace" / spec.source_workspace_id
    workspace_id = target_workspace_id(matrix_run, spec)
    target = task_root / "workspace" / workspace_id
    assert_source(source, spec)

    if target.exists():
        if not resume:
            raise RuntimeError(f"Rescue workspace already exists; pass --resume: {target}")
        verify_managed_workspace(
            target,
            matrix_run=matrix_run,
            spec=spec,
            workspace_id=workspace_id,
        )
        backfill_process_data(target, source)
        facts = validate_prepared(target, task_root, spec)
        print(
            f"PREPARED resume {spec.experiment_id} workspace={workspace_id} "
            f"train={facts['training_sample_count']} val={facts['validation']['sample_count']}",
            flush=True,
        )
        return target

    temporary = target.with_name(f".{workspace_id}.prepare-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        shutil.copy2(source / "process_data.py", temporary / "process_data.py")
        shutil.copy2(source / "data_stats.json", temporary / "data_stats.json")
        shutil.copy2(source / spec.training_file, temporary / spec.training_file)
        shutil.copy2(source / spec.validation_file, temporary / spec.validation_file)
        (temporary / "train.yaml").write_text(
            rendered_train_yaml(source / "train.yaml", spec),
            encoding="utf-8",
        )
        atomic_write_json(temporary / "dataset_info.json", formal_dataset_info(source, spec))
        facts = validate_prepared(temporary, task_root, spec)
        inputs = (
            "process_data.py",
            "train.yaml",
            "dataset_info.json",
            "data_stats.json",
            spec.training_file,
            spec.validation_file,
        )
        manifest = {
            **manifest_stable_fields(matrix_run, spec, workspace_id),
            "created_at": utc_now(),
            "source_hashes": {path.name: sha256(path) for path in source_files(source, spec)},
            "input_hashes": {name: sha256(temporary / name) for name in inputs},
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


def durable_root(workspace: Path) -> Path:
    return workspace.parent / ".ft_model_checkpoints" / workspace.name


def validated_complete(workspace: Path, spec: RescueSpec) -> dict[str, Any] | None:
    durable = durable_root(workspace)
    try:
        visible = validate_formal_training_evidence(
            workspace,
            expected_samples=EXPECTED_SAMPLES,
            experiment_id=spec.experiment_id,
            training_policy=TRAINING_POLICY,
            output_path=workspace / "output",
        )
        preserved = validate_formal_training_evidence(
            durable / "formal_training",
            expected_samples=EXPECTED_SAMPLES,
            experiment_id=spec.experiment_id,
            training_policy=TRAINING_POLICY,
            output_path=durable / "output",
        )
    except (FormalTrainingEvidenceError, OSError):
        return None
    if visible["evidence_signature"] != preserved["evidence_signature"]:
        raise RuntimeError(f"Visible and durable formal evidence disagree: {workspace}")
    return visible


def persist_formal_run(workspace: Path, spec: RescueSpec) -> dict[str, Any]:
    evidence = make_formal_training_evidence(
        workspace,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id=spec.experiment_id,
        training_policy=TRAINING_POLICY,
        output_path=workspace / "output",
    )
    atomic_write_json(workspace / FORMAL_TRAINING_EVIDENCE_FILE, evidence)
    validate_formal_training_evidence(
        workspace,
        expected_samples=EXPECTED_SAMPLES,
        experiment_id=spec.experiment_id,
        training_policy=TRAINING_POLICY,
        output_path=workspace / "output",
    )

    checkpoint = FTWorkspace()
    checkpoint.workspace_path = workspace
    checkpoint.create_ws_ckp()
    completed = validated_complete(workspace, spec)
    if completed is None:
        raise RuntimeError(f"Durable formal checkpoint failed revalidation: {workspace}")
    return completed


def process_is_alive(pid: Any) -> bool:
    return isinstance(pid, int) and pid > 0 and Path(f"/proc/{pid}").exists()


async def stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()


def training_environment(run_root: Path, task_root: Path, spec: RescueSpec, gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "FT_EXPERIMENT_ID": spec.experiment_id,
            "FT_TRAINING_POLICY": TRAINING_POLICY,
            "FT_FORMAL_EXPECTED_SAMPLES": str(EXPECTED_SAMPLES),
            "FT_FORMAL_METHOD_LOCK_PATH": str(task_root / FORMAL_METHOD_LOCK_FILE),
            "FT_GPU_LEASE_POOL_FILE": str(run_root / ".disabled_formal_rescue_gpu_pool"),
            "FT_GPU_LEASE_LOCK_ROOT": str(GPU_LOCK_ROOT),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    return environment


async def run_one(
    run_root: Path,
    spec: RescueSpec,
    workspace: Path,
    gpu: str,
    timeout_seconds: int,
    *,
    resume: bool,
) -> bool:
    existing = validated_complete(workspace, spec)
    if existing is not None:
        print(
            f"COMPLETE gpu={gpu} {spec.experiment_id} workspace={workspace.name} "
            f"step={existing['completion']['global_step']}",
            flush=True,
        )
        return True

    status_path = workspace / STATUS_FILE
    if status_path.is_file():
        status = read_json(status_path, dict)
        if status.get("state") == "running" and process_is_alive(status.get("pid")):
            print(f"ACTIVE gpu={gpu} {spec.experiment_id} pid={status['pid']}", flush=True)
            return True
        if not resume:
            raise RuntimeError(f"Prior rescue status exists; pass --resume: {status_path}")
    if (workspace / "output").exists():
        if not resume:
            raise RuntimeError(f"Partial rescue output exists; pass --resume: {workspace / 'output'}")
        shutil.rmtree(workspace / "output")

    status = {
        "schema_version": SCHEMA_VERSION,
        "state": "starting",
        "experiment_id": spec.experiment_id,
        "workspace_id": workspace.name,
        "gpu_requested": gpu,
        "started_at": utc_now(),
    }
    atomic_write_json(status_path, status)
    log_path = workspace / "formal_rescue.console.log"
    print(f"START gpu={gpu} {spec.experiment_id} workspace={workspace.name}", flush=True)
    try:
        with log_path.open("ab", buffering=0) as log:
            process = await asyncio.create_subprocess_exec(
                str(TRAINING_BIN),
                "train",
                "train.yaml",
                cwd=workspace,
                env=training_environment(run_root, workspace.parent.parent, spec, gpu),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            status.update({"state": "running", "pid": process.pid})
            atomic_write_json(status_path, status)
            try:
                return_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                timed_out = False
            except TimeoutError:
                timed_out = True
                await stop_process(process)
                return_code = process.returncode
            except asyncio.CancelledError:
                await stop_process(process)
                raise
        if return_code != 0 or timed_out:
            raise RuntimeError(f"trainer exit_code={return_code}, outer_timeout={timed_out}")
        evidence = await asyncio.to_thread(persist_formal_run, workspace, spec)
        status.update(
            {
                "state": "succeeded",
                "finished_at": utc_now(),
                "return_code": return_code,
                "outer_timeout": timed_out,
                "evidence_signature": evidence["evidence_signature"],
                "global_step": evidence["completion"]["global_step"],
                "max_steps": evidence["completion"]["max_steps"],
            },
        )
        atomic_write_json(status_path, status)
        print(
            f"DONE gpu={gpu} {spec.experiment_id} workspace={workspace.name} "
            f"step={evidence['completion']['global_step']}",
            flush=True,
        )
        return True
    except asyncio.CancelledError:
        status.update({"state": "aborted", "finished_at": utc_now()})
        atomic_write_json(status_path, status)
        raise
    except Exception as error:  # noqa: BLE001 - every operational failure must be durable.
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        atomic_write_json(status_path, status)
        print(f"FAIL gpu={gpu} {spec.experiment_id}: {error}", flush=True)
        return False


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
