#!/usr/bin/env python3
"""Run the released FT-Agent portion of the paper matrix with pipelined workers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import io
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .collect_results import collect_and_write_run
    from .matrix import Experiment, ablation_experiments, all_experiments, main_experiments, planner_experiments
    from .responses_adapter import litellm_model_name, responses_configuration_errors, routed_api_environment
else:
    from collect_results import collect_and_write_run
    from matrix import Experiment, ablation_experiments, all_experiments, main_experiments, planner_experiments
    from responses_adapter import litellm_model_name, responses_configuration_errors, routed_api_environment

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
LOOP = ROOT / "rdagent" / "app" / "finetune" / "llm" / "loop.py"
SCENARIOS = ROOT / "rdagent" / "app" / "finetune" / "llm" / "job" / "scenarios.json"
ASSET_MANIFEST = Path(__file__).with_name("assets.json")
FT_ROOT = ROOT / "finetune_files"
CONDA_CONFIG = ROOT / "reproduction" / "condarc"
DEFAULT_DEBUG_DATA_PROCESSING_TIMEOUT = 3600
DEFAULT_DATA_PROCESSING_TIMEOUT = 21600
DEFAULT_FULL_TRAINING_TIMEOUT = 360000
DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT = 360000
OUTER_TIMEOUT_GRACE_SECONDS = 600
DEFAULT_GPU_GUARD_INTERVAL = 5.0
DEFAULT_WORKERS_PER_GPU = 2
GPU_CONFLICT_EXIT_CODE = 75
GPU_INVENTORY_COLUMNS = 2
COMPUTE_INVENTORY_COLUMNS = 3
TRAINING_POLICIES = ("paper", "full", "lora", "rslora")
FORMAL_EXPECTED_SAMPLES_ENV = "FT_FORMAL_EXPECTED_SAMPLES"
FORMAL_METHOD_LOCK_ENV = "FT_FORMAL_METHOD_LOCK_PATH"
FORMAL_METHOD_LOCK_FILE = "formal_training_method.json"
LOGICAL_GPU_COUNT_ENV = "FT_LOGICAL_TRAINING_GPU_COUNT"
LOGICAL_GPU_MEMORY_ENV = "FT_LOGICAL_TRAINING_GPU_MEMORY_GB"
LOGICAL_GPU_NAME_ENV = "FT_LOGICAL_TRAINING_GPU_NAME"
LOGICAL_RESOURCE_SCOPE_ENV = "FT_LOGICAL_TRAINING_RESOURCE_SCOPE"
PHYSICAL_EXECUTION_MAPPING_ENV = "FT_PHYSICAL_TRAINING_MAPPING"
PAPER_LOGICAL_GPU_COUNT = 1
PAPER_LOGICAL_GPU_MEMORY_GB = 178
PAPER_LOGICAL_GPU_NAME = "NVIDIA B200"
PAPER_LOGICAL_RESOURCE_SCOPE = "paper logical per-experiment method-selection envelope"
H20_EXECUTION_MAPPING = (
    "Ordinary LoRA executes on one H20. Full SFT uses BF16 ZeRO-3 on two H20s "
    "for cutoff_len <= 8192 and four H20s for longer contexts; the runtime "
    "preserves the logical single-B200 global batch."
)


class GPUExclusivityError(RuntimeError):
    """Raised when a selected GPU is used by a process outside this matrix."""

    def __init__(self, processes: list[dict[str, Any]], query_error: str | None = None) -> None:
        self.processes = processes
        self.query_error = query_error
        if query_error is not None:
            message = f"GPU exclusivity query failed: {query_error}"
        else:
            details = ", ".join(
                f"pid={item['pid']} uuid={item['gpu_uuid']} process={item['process_name']}" for item in processes
            )
            message = f"Selected GPU(s) acquired by external process(es): {details}"
        super().__init__(message)


def enforce_complete_training_timeout() -> None:
    """Prevent the paper search budget from truncating a formal train step.

    The 12-hour task budget is checked by the FT-Agent loop between steps.  A
    train step that has already started must still be allowed to finish every
    configured epoch, which is substantially slower on H20 for long-context
    examples.  Keep larger operator overrides, but reject malformed values and
    raise stale smaller values to the formal-training safety floor.
    """
    raw = os.environ.get("FT_FULL_TIMEOUT", str(DEFAULT_FULL_TRAINING_TIMEOUT))
    try:
        configured = int(raw)
    except ValueError as error:
        message = f"FT_FULL_TIMEOUT must be an integer number of seconds; got {raw!r}"
        raise ValueError(message) from error
    if configured < 1:
        message = "FT_FULL_TIMEOUT must be positive"
        raise ValueError(message)
    os.environ["FT_FULL_TIMEOUT"] = str(max(configured, DEFAULT_FULL_TRAINING_TIMEOUT))


def enforce_complete_responses_data_processing_timeout() -> None:
    """Give formal Responses-backed generation enough time to reach 2,000 rows.

    Slow-rejection datasets such as AIME can require many more than 2,000 API
    calls before producing 2,000 accepted records.  The generic Responses
    default is therefore too short for the formal matrix.  Preserve larger
    operator overrides while raising stale smaller values to the same safety
    floor used for a complete formal training step.
    """
    variable = "FT_RESPONSES_DATA_PROCESSING_TIMEOUT"
    raw = os.environ.get(variable, str(DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT))
    message = f"{variable} must be a positive integer"
    try:
        configured = int(raw)
    except ValueError as error:
        raise ValueError(message) from error
    if configured < 1:
        raise ValueError(message)
    os.environ[variable] = str(max(configured, DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "ablation", "planner", "all"), default="main")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids; default: all visible GPUs")
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum concurrent tasks")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
        help=(
            "Concurrent FT-Agent pipelines assigned to each GPU (default: 2). "
            "Training and benchmark entrypoints must use the project GPU lease so only one GPU stage runs per card."
        ),
    )
    parser.add_argument("--only", default=None, help="Regex applied to experiment ids")
    parser.add_argument("--run-name", default=None, help="Stable output folder name, useful with --resume")
    parser.add_argument(
        "--training-policy",
        choices=TRAINING_POLICIES,
        default=os.environ.get("FT_TRAINING_POLICY", "paper").lower().replace("-", ""),
        help="Method policy: paper lets the hypothesis choose; others force one method",
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks with a successful status file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print selected tasks and run no experiments")
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate selected tasks without querying GPUs or creating a run directory",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--task-timeout",
        default=None,
        help="Override each selected FT task budget for a fresh run; maximum: 48h",
    )
    parser.add_argument(
        "--require-exclusive-gpus",
        action="store_true",
        help="Abort if a selected GPU is used by a process outside this matrix",
    )
    parser.add_argument(
        "--gpu-guard-interval",
        type=float,
        default=DEFAULT_GPU_GUARD_INTERVAL,
        help="Seconds between exclusive-GPU checks (default: 5)",
    )
    return parser.parse_args()


def selected_experiments(suite: str) -> list[Experiment]:
    return {
        "main": main_experiments,
        "ablation": ablation_experiments,
        "planner": planner_experiments,
        "all": all_experiments,
    }[suite]()


def configure_project_environment(training_policy: str | None = None) -> None:
    if training_policy is None:
        training_policy = os.environ.get("FT_TRAINING_POLICY", "paper")
    os.environ.update(
        {
            "CONDARC": str(CONDA_CONFIG),
            "CONDA_ENVS_PATH": str(FT_ROOT / "conda_envs"),
            "CONDA_PKGS_DIRS": str(FT_ROOT / "cache" / "conda_pkgs"),
            "PIP_CACHE_DIR": str(FT_ROOT / "cache" / "pip"),
            "TMPDIR": str(FT_ROOT / "tmp"),
            "FT_GPU_LEASE_LOCK_ROOT": str(FT_ROOT / "gpu_leases" / "locks"),
            "FT_GPU_LEASE_POOL_FILE": str(FT_ROOT / "gpu_leases" / "dynamic_pool"),
            # The reproduction search is validation-only even if a caller's
            # shell or .env enabled the legacy UI behavior.
            "FT_EVALUATE_HELD_OUT_DURING_SEARCH": "false",
            "FT_TRAINING_POLICY": training_policy,
            # The released paper lets the agent choose Full SFT or LoRA under
            # one 178GB B200. Worker-local CUDA visibility is intentionally
            # decoupled from that autonomous method-selection envelope.
            LOGICAL_GPU_COUNT_ENV: str(PAPER_LOGICAL_GPU_COUNT),
            LOGICAL_GPU_MEMORY_ENV: str(PAPER_LOGICAL_GPU_MEMORY_GB),
            LOGICAL_GPU_NAME_ENV: PAPER_LOGICAL_GPU_NAME,
            LOGICAL_RESOURCE_SCOPE_ENV: PAPER_LOGICAL_RESOURCE_SCOPE,
            PHYSICAL_EXECUTION_MAPPING_ENV: H20_EXECUTION_MAPPING,
            "FT_DEEPSPEED_ZERO3_CONFIG": str(
                ROOT / "rdagent" / "scenarios" / "finetune" / "env" / "conda" / "deepspeed" / "ds_z3_config.json",
            ),
        },
    )
    enforce_complete_training_timeout()
    enforce_complete_responses_data_processing_timeout()
    # A responses-backed data coder can legitimately need one or two model
    # calls for each of roughly 2,000 selected records.  The upstream endpoint
    # is shared across concurrent matrix workers, so keeping per-worker API
    # concurrency conservative and allowing the work more wall time is safer
    # than multiplying requests.  Explicit user settings still take priority.
    os.environ.setdefault(
        "FT_DEBUG_DATA_PROCESSING_TIMEOUT",
        str(DEFAULT_DEBUG_DATA_PROCESSING_TIMEOUT),
    )
    os.environ.setdefault(
        "FT_DATA_PROCESSING_TIMEOUT",
        str(DEFAULT_DATA_PROCESSING_TIMEOUT),
    )


def visible_gpus(argument: str | None) -> list[str]:
    if argument:
        result = [item.strip() for item in argument.split(",") if item.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        result = [item.strip() for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
    else:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            message = "nvidia-smi is not available; pass --gpus explicitly"
            raise SystemExit(message)
        query = subprocess.run(  # noqa: S603
            [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if not result:
        message = "No GPUs selected"
        raise SystemExit(message)
    return result


def parse_gpu_inventory(output: str) -> dict[str, str]:
    """Parse ``nvidia-smi --query-gpu=index,uuid`` output."""
    result: dict[str, str] = {}
    for row in csv.reader(io.StringIO(output)):
        if len(row) < GPU_INVENTORY_COLUMNS:
            continue
        index, gpu_uuid = (value.strip() for value in row[:GPU_INVENTORY_COLUMNS])
        if index and gpu_uuid:
            result[index] = gpu_uuid
    return result


def parse_compute_inventory(output: str) -> list[dict[str, Any]]:
    """Parse the stable subset of the compute-app query used by the guard."""
    result: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) < COMPUTE_INVENTORY_COLUMNS:
            continue
        pid_text, gpu_uuid, process_name = (value.strip() for value in row[:COMPUTE_INVENTORY_COLUMNS])
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        result.append(
            {
                "pid": pid,
                "gpu_uuid": gpu_uuid,
                "process_name": process_name,
            },
        )
    return result


def _nvidia_smi_query(kind: str, fields: str) -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        message = "nvidia-smi is not available"
        raise RuntimeError(message)
    result = subprocess.run(  # noqa: S603
        [nvidia_smi, f"--query-{kind}={fields}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def selected_gpu_uuids(gpus: list[str]) -> set[str]:
    inventory = parse_gpu_inventory(_nvidia_smi_query("gpu", "index,uuid"))
    known_uuids = set(inventory.values())
    selected: set[str] = set()
    missing: list[str] = []
    for gpu in gpus:
        if gpu in inventory:
            selected.add(inventory[gpu])
        elif gpu in known_uuids:
            selected.add(gpu)
        else:
            missing.append(gpu)
    if missing:
        message = f"Selected GPU(s) are absent from nvidia-smi inventory: {', '.join(missing)}"
        raise RuntimeError(message)
    return selected


def process_parent_table() -> dict[int, tuple[int, int]]:
    """Return ``pid -> (ppid, session_id)`` for live Linux processes."""
    result: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            state, ppid, session_id = fields[0], int(fields[1]), int(fields[3])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if state != "Z":
            result[int(entry.name)] = (ppid, session_id)
    return result


def is_descendant(pid: int, ancestor: int, table: dict[int, tuple[int, int]]) -> bool:
    """Return whether *pid* currently descends from *ancestor*."""
    current = pid
    seen: set[int] = set()
    while current not in seen and current in table:
        if current == ancestor:
            return True
        seen.add(current)
        current = table[current][0]
    return current == ancestor


def foreign_gpu_processes(
    owner_pid: int,
    gpu_uuids: set[str],
    *,
    compute_processes: list[dict[str, Any]] | None = None,
    process_table: dict[int, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Return selected-GPU processes that do not belong to this scheduler."""
    processes = (
        parse_compute_inventory(_nvidia_smi_query("compute-apps", "pid,gpu_uuid,process_name"))
        if compute_processes is None
        else compute_processes
    )
    table = process_parent_table() if process_table is None else process_table
    # A PID can disappear between the NVML and /proc snapshots.  Such a process
    # cannot still interfere, so only live PIDs are classified.
    return [
        process
        for process in processes
        if process["gpu_uuid"] in gpu_uuids
        and process["pid"] in table
        and not is_descendant(process["pid"], owner_pid, table)
    ]


def check_gpu_exclusivity(owner_pid: int, gpu_uuids: set[str]) -> None:
    try:
        conflicts = foreign_gpu_processes(owner_pid, gpu_uuids)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise GPUExclusivityError([], query_error=str(error)) from error
    if conflicts:
        raise GPUExclusivityError(conflicts)


async def monitor_gpu_exclusivity(owner_pid: int, gpu_uuids: set[str], interval: float) -> None:
    while True:
        await asyncio.to_thread(check_gpu_exclusivity, owner_pid, gpu_uuids)
        await asyncio.sleep(interval)


def duration_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)([smhd])", value)
    if not match:
        message = f"Unsupported duration: {value}"
        raise ValueError(message)
    number, unit = match.groups()
    return int(number) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def outer_process_timeout_seconds(
    task_budget_seconds: int,
    environment: Mapping[str, str],
) -> int:
    """Keep the supervisor alive for a formal training step already in flight."""
    raw_full_timeout = environment.get("FT_FULL_TIMEOUT", str(DEFAULT_FULL_TRAINING_TIMEOUT))
    try:
        full_training_timeout = int(raw_full_timeout)
    except ValueError as error:
        message = f"FT_FULL_TIMEOUT must be an integer number of seconds; got {raw_full_timeout!r}"
        raise ValueError(message) from error
    if task_budget_seconds < 1 or full_training_timeout < 1:
        message = "Task and full-training timeouts must be positive"
        raise ValueError(message)
    return max(task_budget_seconds, full_training_timeout) + OUTER_TIMEOUT_GRACE_SECONDS


def override_task_timeout(experiments: list[Experiment], value: str | None) -> list[Experiment]:
    """Apply a bounded task budget override for isolated difficult-task retries."""
    if value is None:
        return experiments
    seconds = duration_seconds(value)
    if not 0 < seconds <= 48 * 3600:
        message = "--task-timeout must be greater than 0 and at most 48h"
        raise SystemExit(message)
    return [replace(experiment, timeout=value) for experiment in experiments]


def validate_max_parallel(value: int | None) -> None:
    if value is not None and value < 1:
        message = "--max-parallel must be positive"
        raise SystemExit(message)


def validate_workers_per_gpu(value: int) -> None:
    if value < 1:
        message = "--workers-per-gpu must be positive"
        raise SystemExit(message)


def worker_gpu_assignments(
    gpus: list[str],
    workers_per_gpu: int,
    max_parallel: int | None = None,
) -> list[str]:
    """Return round-robin pipeline lanes while honoring the global cap."""
    assignments = [gpu for _lane in range(workers_per_gpu) for gpu in gpus]
    return assignments[:max_parallel]


def validate_gpu_lease_entrypoints(workers_per_gpu: int) -> None:
    """Require the policy boundary and serialized GPU entrypoints."""
    training_env = os.environ.get("FT_CONDA_CONDA_ENV_NAME", "llm_finetune")
    benchmark_env = os.environ.get("BENCHMARK_CONDA_CONDA_ENV_NAME", "opencompass")
    training_entrypoint = FT_ROOT / "conda_envs" / training_env / "bin" / "llamafactory-cli"
    try:
        training_source = training_entrypoint.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        training_source = ""
    policy_snippets = (
        "def prepare_training_invocation",
        "FT_TRAINING_POLICY",
        "finetuning_type",
        "use_rslora",
        "use_dora",
        "DEFAULT_DEEPSPEED_ZERO3_CONFIG",
        "def validate_formal_training_contract",
        "FT_FORMAL_EXPECTED_SAMPLES",
        "FT_FORMAL_METHOD_LOCK_PATH",
        "enforce_formal_training_method_lock",
        "require_runtime_contract=True",
        "def acquire_exact_gpu_leases",
        "prepare_training_invocation()\n    _gpu_leases = acquire_gpu_leases(_training_config)",
    )
    if any(snippet not in training_source for snippet in policy_snippets):
        message = (
            "Matrix runs require the pre-lease multi-method training-policy wrapper; "
            f"reinstall the project training backend first: {training_entrypoint}"
        )
        raise SystemExit(message)
    if workers_per_gpu == 1:
        return
    entrypoints = (
        training_entrypoint,
        FT_ROOT / "conda_envs" / benchmark_env / "bin" / "opencompass",
    )
    unsafe = []
    for path in entrypoints:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            unsafe.append(path)
            continue
        required_snippets = (
            "import fcntl",
            "def acquire_gpu_leases",
            "def wait_for_gpu_memory_ready",
            "memory.free,memory.total",
            "wait_for_gpu_memory_ready(gpu)",
            "CUDA_VISIBLE_DEVICES",
            str(FT_ROOT / "gpu_leases" / "locks"),
            "fcntl.flock",
            "fcntl.LOCK_EX",
        )
        if any(snippet not in source for snippet in required_snippets):
            unsafe.append(path)
    if unsafe:
        joined = ", ".join(str(path) for path in unsafe)
        message = f"--workers-per-gpu > 1 requires GPU lease wrappers; reinstall the project backends first: {joined}"
        raise SystemExit(message)


def safe_id(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", experiment_id).strip("_")


def validate_task_formal_training(
    task_root: Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
    require_visible_evidence: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return durable runs that prove the matrix contract and its method lock."""
    from rdagent.scenarios.finetune.train.formal_training import (  # noqa: PLC0415
        FormalTrainingEvidenceError,
        scan_durable_formal_training_evidence,
        validate_formal_training_method_lock,
    )

    records, errors = scan_durable_formal_training_evidence(
        task_root,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
        require_visible_evidence=require_visible_evidence,
    )
    if not records:
        errors.append(
            "no durable workspace proves the formal "
            f"{expected_samples}-sample/full-epoch training contract",
        )
        return records, errors

    methods = {str(record.get("training_method", "")) for record in records}
    if len(methods) != 1 or "" in methods:
        errors.append(
            "durable formal workspaces disagree on the locked training method: "
            + ", ".join(sorted(methods)),
        )
        return [], errors

    training_method = methods.pop()
    lock_path = task_root / FORMAL_METHOD_LOCK_FILE
    try:
        method_lock = validate_formal_training_method_lock(
            lock_path,
            experiment_id=experiment_id,
            training_policy=training_policy,
            training_method=training_method,
        )
    except (FormalTrainingEvidenceError, OSError) as error:
        errors.append(f"formal training method lock is invalid: {error}")
        return [], errors

    bound_records = [
        {
            **record,
            "formal_training_method_lock": method_lock,
        }
        for record in records
    ]
    return bound_records, errors


def read_scenarios() -> dict[str, dict[str, Any]]:
    return json.loads(SCENARIOS.read_text(encoding="utf-8"))


def read_asset_manifest() -> dict[str, Any]:
    return json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))


def asset_target(asset: dict[str, Any]) -> Path:
    if target := asset.get("target"):
        relative = Path(target)
        if relative.is_absolute() or ".." in relative.parts:
            message = f"Asset target must stay below finetune_files: {target!r}"
            raise ValueError(message)
        return FT_ROOT / relative
    if asset["kind"] == "dataset":
        return FT_ROOT / "datasets" / asset["name"]
    return FT_ROOT / "models" / asset["name"]


def benchmark_assets(experiments: list[Experiment], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    requested = {experiment.benchmark for experiment in experiments}
    result: dict[str, dict[str, Any]] = {}
    for asset in manifest["assets"]:
        for benchmark in asset.get("benchmarks", []):
            if benchmark not in requested:
                continue
            if benchmark in result:
                message = f"Multiple pinned assets configured for benchmark: {benchmark}"
                raise ValueError(message)
            result[benchmark] = asset
    missing = requested - result.keys()
    if missing:
        message = f"No pinned evaluation asset configured for: {', '.join(sorted(missing))}"
        raise ValueError(message)
    return result


def benchmark_dataset_path(asset: dict[str, Any]) -> str:
    target = Path(asset["target"])
    try:
        relative = target.relative_to("benchmarks")
    except ValueError as error:
        message = f"Benchmark asset target must be below finetune_files/benchmarks: {target}"
        raise ValueError(message) from error
    dataset_path = Path(asset.get("dataset_path", "."))
    if dataset_path.is_absolute() or ".." in dataset_path.parts:
        message = f"Benchmark dataset path must stay below its asset: {dataset_path}"
        raise ValueError(message)
    return (relative / dataset_path).as_posix()


def conda_ready(conda: str, env_name: str, module: str) -> bool:
    result = subprocess.run(  # noqa: S603
        [conda, "run", "-n", env_name, "python", "-c", f"import {module}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env=os.environ.copy(),
    )
    return result.returncode == 0


def marker_matches(asset: dict[str, Any]) -> bool:
    marker_path = asset_target(asset) / ".rdagent-asset.json"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    identity = ("id", "kind", "repo_id", "revision")
    return all(marker.get(key) == asset[key] for key in identity)


def asset_errors(
    experiments: list[Experiment],
    manifest: dict[str, Any],
    pinned_benchmarks: dict[str, dict[str, Any]],
) -> list[str]:
    required: dict[str, dict[str, Any]] = {
        asset["id"]: asset
        for asset in manifest["assets"]
        if asset["kind"] == "dataset" and asset.get("role", "training") == "training"
    }
    required.update({asset["id"]: asset for asset in pinned_benchmarks.values()})
    selected_models = {experiment.model for experiment in experiments}
    required.update(
        {
            asset["id"]: asset
            for asset in manifest["assets"]
            if asset["kind"] == "model" and asset["name"] in selected_models
        },
    )
    return [
        f"asset not prepared: {asset_id} ({asset_target(asset)})"
        for asset_id, asset in sorted(required.items())
        if not marker_matches(asset)
    ]


def backend_errors() -> list[str]:
    conda = shutil.which("conda")
    if conda is None:
        return ["conda is not available"]
    training_env = os.environ.get("FT_CONDA_CONDA_ENV_NAME", "llm_finetune")
    benchmark_env = os.environ.get("BENCHMARK_CONDA_CONDA_ENV_NAME", "opencompass")
    errors = []
    if not conda_ready(conda, training_env, "llamafactory"):
        errors.append(f"training conda environment is not ready: {training_env}")
    if not conda_ready(conda, benchmark_env, "opencompass"):
        errors.append(f"benchmark conda environment is not ready: {benchmark_env}")
    return errors


def preflight(
    experiments: list[Experiment],
    manifest: dict[str, Any],
    pinned_benchmarks: dict[str, dict[str, Any]],
) -> None:
    errors: list[str] = []
    if not PYTHON.is_file():
        errors.append(f"uv environment missing: {PYTHON}")
    if not os.environ.get("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not exported")
    errors.extend(responses_configuration_errors(os.environ))
    errors.extend(asset_errors(experiments, manifest, pinned_benchmarks))
    errors.extend(backend_errors())
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))


def status_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stable_api_routing(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop the adapter's ephemeral listen address before compatibility checks."""
    return {key: value for key, value in metadata.items() if key != "adapter_base"}


def write_or_validate_manifest(run_root: Path, manifest: dict[str, Any]) -> None:  # noqa: C901
    """Create a run manifest once and preserve its full task inventory on retries.

    A filtered ``--only`` retry must not replace a complete matrix manifest with
    its subset; the collector uses that manifest as the run's coverage contract.
    """
    path = run_root / "matrix.json"
    if not path.is_file():
        status_write(path, manifest)
        return

    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        message = f"Existing matrix manifest is unreadable: {path}: {error}"
        raise SystemExit(message) from error

    errors: list[str] = []
    if existing.get("suite") != manifest.get("suite"):
        errors.append(f"suite differs ({existing.get('suite')!r} != {manifest.get('suite')!r})")
    if existing.get("training_policy", "paper") != manifest.get("training_policy", "paper"):
        errors.append(
            "training policy differs "
            f"({existing.get('training_policy', 'paper')!r} != {manifest.get('training_policy', 'paper')!r})",
        )
    if existing.get("formal_training_contract") != manifest.get("formal_training_contract"):
        errors.append("formal training contract differs")
    if _stable_api_routing(existing.get("api_routing", {})) != _stable_api_routing(
        manifest.get("api_routing", {}),
    ):
        errors.append("stable API routing differs")

    existing_assets = existing.get("benchmark_assets", {})
    for benchmark, expected in manifest.get("benchmark_assets", {}).items():
        if existing_assets.get(benchmark) != expected:
            errors.append(f"benchmark asset differs for {benchmark}")

    existing_tasks = {
        task.get("experiment_id"): task
        for task in existing.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("experiment_id"), str)
    }
    for expected in manifest.get("tasks", []):
        experiment_id = expected["experiment_id"]
        actual = existing_tasks.get(experiment_id)
        if actual is None:
            errors.append(f"task is absent from existing manifest: {experiment_id}")
        elif actual != expected:
            errors.append(f"task definition differs: {experiment_id}")

    if errors:
        message = "Existing matrix manifest is incompatible:\n- " + "\n- ".join(errors)
        raise SystemExit(message)


def _is_preserved_method_lock_root(
    experiment: Experiment,
    task_root: Path,
    training_policy: str,
) -> bool:
    """Return whether a retry root contains only its immutable method lock."""
    from rdagent.scenarios.finetune.train.formal_training import (  # noqa: PLC0415
        FormalTrainingEvidenceError,
        validate_formal_training_method_lock,
    )

    if not task_root.is_dir() or task_root.is_symlink():
        return False
    allowed_names = {FORMAL_METHOD_LOCK_FILE, FORMAL_METHOD_LOCK_FILE + ".lock"}
    try:
        children = list(task_root.iterdir())
    except OSError:
        return False
    if not children or any(
        child.name not in allowed_names or not child.is_file() or child.is_symlink()
        for child in children
    ):
        return False

    method_lock_path = task_root / FORMAL_METHOD_LOCK_FILE
    try:
        payload = json.loads(method_lock_path.read_text(encoding="utf-8"))
        training_method = payload.get("training_method") if isinstance(payload, dict) else None
        if not isinstance(training_method, str):
            return False
        validate_formal_training_method_lock(
            method_lock_path,
            experiment_id=experiment.experiment_id,
            training_policy=training_policy,
            training_method=training_method,
        )
    except (FormalTrainingEvidenceError, OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def refuse_existing_task_roots(
    experiments: list[Experiment],
    run_root: Path,
    training_policy: str | None = None,
) -> None:
    """Prevent retries from appending to stale logs or restoring old traces.

    An archived failed attempt may leave behind only its validated method lock.
    That exception preserves the paper-policy task's autonomous first choice
    without reusing any status, trace, workspace, or console artifact.
    """
    active_policy = training_policy or os.environ.get("FT_TRAINING_POLICY", "paper")
    existing = [
        path
        for experiment in experiments
        if (
            (path := run_root / safe_id(experiment.experiment_id)).exists()
            and not _is_preserved_method_lock_root(experiment, path, active_policy)
        )
    ]
    if not existing:
        return
    listed = "\n- ".join(str(path) for path in existing)
    message = (
        "Selected task roots already exist and cannot be reused safely. "
        "Archive them before retrying so trace, workspace, and console artifacts stay isolated:\n- " + listed
    )
    raise SystemExit(message)


def session_processes(session_id: int) -> list[int]:
    """Return live processes in a Linux session, including nested process groups."""
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # Fields following ``comm`` begin with state, ppid, pgrp, session.
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            state = fields[0]
            process_session = int(fields[3])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if process_session == session_id and state != "Z":
            result.append(int(entry.name))
    return result


def signal_session(session_id: int, signal_number: signal.Signals) -> None:
    # Stop the session leader first so it cannot create more descendants while
    # we enumerate child-created process groups (LocalEnv and timeout do this).
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(session_id, signal_number)
    for pid in session_processes(session_id):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal_number)


async def wait_for_session_exit(process: asyncio.subprocess.Process, session_id: int) -> None:
    if process.returncode is None:
        await process.wait()
    # Linux does not expose an asyncio event for unrelated descendants leaving
    # a session, so polling /proc is intentional here.
    while session_processes(session_id):  # noqa: ASYNC110
        await asyncio.sleep(0.2)


async def stop_process(process: asyncio.subprocess.Process) -> None:
    session_id = process.pid
    if process.returncode is not None and not session_processes(session_id):
        return
    signal_session(session_id, signal.SIGTERM)
    try:
        await asyncio.wait_for(wait_for_session_exit(process, session_id), timeout=60)
    except TimeoutError:
        signal_session(session_id, signal.SIGKILL)
        if process.returncode is None:
            await process.wait()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wait_for_session_exit(process, session_id), timeout=10)


async def run_one(
    experiment: Experiment,
    gpu: str,
    run_root: Path,
    scenarios: dict[str, dict[str, Any]],
    pinned_benchmarks: dict[str, dict[str, Any]],
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    task_root = run_root / safe_id(experiment.experiment_id)
    trace_path = task_root / "trace"
    workspace_path = task_root / "workspace"
    status_path = task_root / "status.json"
    task_root.mkdir(parents=True, exist_ok=True)
    description = scenarios.get(experiment.benchmark, {}).get("benchmark_description")
    if not description:
        message = f"No benchmark description registered for {experiment.benchmark}"
        raise RuntimeError(message)

    command = [
        str(PYTHON),
        str(LOOP),
        "--benchmark",
        experiment.benchmark,
        "--benchmark-description",
        description,
        "--base-model",
        experiment.model,
        "--upper-data-size-limit",
        str(experiment.data_limit),
        "--timeout",
        experiment.timeout,
    ]
    environment = runtime_environment.copy()
    benchmark_asset = pinned_benchmarks[experiment.benchmark]
    pinned_dataset_path = benchmark_dataset_path(benchmark_asset)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CHAT_MODEL": litellm_model_name(experiment.planner, api_routing),
            "FT_BASE_MODEL": experiment.model,
            "FT_TARGET_BENCHMARK": experiment.benchmark,
            "FT_BENCHMARK_DESCRIPTION": description,
            "FT_UPPER_DATA_SIZE_LIMIT": str(experiment.data_limit),
            FORMAL_EXPECTED_SAMPLES_ENV: str(experiment.data_limit),
            FORMAL_METHOD_LOCK_ENV: str(task_root / FORMAL_METHOD_LOCK_FILE),
            "LOG_TRACE_PATH": str(trace_path),
            "WORKSPACE_PATH": str(workspace_path),
            "FT_EXPERIMENT_ID": experiment.experiment_id,
            "FT_BENCHMARK_DATASET_PATH": pinned_dataset_path,
        },
    )
    started = datetime.now().astimezone().isoformat()
    status = {
        **experiment.to_dict(),
        "gpu": gpu,
        "state": "running",
        "started_at": started,
        "command": command,
        "trace_path": str(trace_path),
        "workspace_path": str(workspace_path),
        "benchmark_asset": benchmark_asset["id"],
        "benchmark_dataset_path": pinned_dataset_path,
        "api_routing": api_routing,
        "training_policy": environment.get("FT_TRAINING_POLICY", "paper"),
        "formal_expected_samples": experiment.data_limit,
        "formal_method_lock_path": str(task_root / FORMAL_METHOD_LOCK_FILE),
        "formal_training_method": None,
        "formal_training_method_lock": None,
    }
    status_write(status_path, status)
    print(f"START gpu={gpu} {experiment.experiment_id}", flush=True)
    with (task_root / "console.log").open("ab", buffering=0) as log:
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
            # The loop enforces the paper search budget at step boundaries. A
            # formal training step that has already started must still be able
            # to finish its configured epochs, which can take longer than the
            # search budget on H20s. This outer guard therefore covers both.
            return_code = await asyncio.wait_for(
                process.wait(),
                timeout=outer_process_timeout_seconds(
                    duration_seconds(experiment.timeout),
                    environment,
                ),
            )
        except TimeoutError:
            timed_out = True
            await stop_process(process)
            return_code = process.returncode
        except asyncio.CancelledError:
            await stop_process(process)
            status.update(
                {
                    "state": "aborted",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "return_code": process.returncode,
                    "outer_timeout": False,
                    "abort_reason": "scheduler_cancelled",
                },
            )
            status_write(status_path, status)
            print(f"ABORT gpu={gpu} {experiment.experiment_id}", flush=True)
            raise
    evidence_records, evidence_errors = validate_task_formal_training(
        task_root,
        expected_samples=experiment.data_limit,
        experiment_id=experiment.experiment_id,
        training_policy=str(environment.get("FT_TRAINING_POLICY", "paper")),
    )
    success = return_code == 0 and not timed_out and bool(evidence_records)
    locked_method = evidence_records[0]["training_method"] if evidence_records else None
    method_lock = (
        evidence_records[0].get("formal_training_method_lock") if evidence_records else None
    )
    status.update(
        {
            "state": "succeeded" if success else "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "return_code": return_code,
            "outer_timeout": timed_out,
            "formal_training_evidence": evidence_records,
            "formal_training_validation_errors": evidence_errors,
            "formal_training_method": locked_method,
            "formal_training_method_lock": method_lock,
        },
    )
    status_write(status_path, status)
    print(f"{'DONE ' if success else 'FAIL '} gpu={gpu} {experiment.experiment_id}", flush=True)
    return success


async def run_workers(
    experiments: list[Experiment],
    gpus: list[str],
    run_root: Path,
    scenarios: dict[str, dict[str, Any]],
    pinned_benchmarks: dict[str, dict[str, Any]],
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
    guarded_gpu_uuids: set[str] | None = None,
    gpu_guard_interval: float = DEFAULT_GPU_GUARD_INTERVAL,
    workers_per_gpu: int = 1,
    max_parallel: int | None = None,
) -> bool:
    queue: asyncio.Queue[Experiment] = asyncio.Queue()
    for experiment in experiments:
        queue.put_nowait(experiment)
    results: list[bool] = []

    async def worker(gpu: str) -> None:
        while True:
            try:
                experiment = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results.append(
                    await run_one(
                        experiment,
                        gpu,
                        run_root,
                        scenarios,
                        pinned_benchmarks,
                        runtime_environment,
                        api_routing,
                    ),
                )
            finally:
                queue.task_done()

    # Keep more than one end-to-end task in flight per GPU so slow API/CPU data
    # preparation overlaps training and evaluation from another task.  The
    # project-local ``llamafactory-cli`` and ``opencompass`` entrypoints take an
    # exclusive file lease for CUDA_VISIBLE_DEVICES, serializing only the
    # actual GPU stages while preserving this front-end pipeline parallelism.
    assignments = worker_gpu_assignments(gpus, workers_per_gpu, max_parallel)
    workers = asyncio.gather(*(worker(gpu) for gpu in assignments))
    if guarded_gpu_uuids is None:
        await workers
        return all(results)

    guard = asyncio.create_task(
        monitor_gpu_exclusivity(os.getpid(), guarded_gpu_uuids, gpu_guard_interval),
    )
    done, _pending = await asyncio.wait({workers, guard}, return_when=asyncio.FIRST_COMPLETED)
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
    return all(results)


def filter_succeeded_experiments(experiments: list[Experiment], run_root: Path) -> list[Experiment]:
    """Drop only tasks whose success still has valid formal evidence."""
    pending: list[Experiment] = []
    active_policy = os.environ.get("FT_TRAINING_POLICY", "paper")
    for experiment in experiments:
        task_root = run_root / safe_id(experiment.experiment_id)
        status_path = task_root / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pending.append(experiment)
            continue
        if (
            status.get("state") != "succeeded"
            or status.get("formal_expected_samples") != experiment.data_limit
            or status.get("training_policy", "paper") != active_policy
        ):
            pending.append(experiment)
            continue
        records, _errors = validate_task_formal_training(
            task_root,
            expected_samples=experiment.data_limit,
            experiment_id=experiment.experiment_id,
            training_policy=active_policy,
        )
        if (
            not records
            or status.get("formal_training_method") != records[0].get("training_method")
            or status.get("formal_training_method_lock")
            != records[0].get("formal_training_method_lock")
        ):
            pending.append(experiment)
    return pending


def print_experiment_selection(experiments: list[Experiment], gpus: list[str], run_root: Path) -> None:
    print(f"Selected {len(experiments)} tasks; GPUs={','.join(gpus)}; output={run_root}")
    for experiment in experiments:
        print(
            f"  {experiment.experiment_id}: {experiment.model}, {experiment.planner}, "
            f"limit={experiment.data_limit}, timeout={experiment.timeout}",
        )


def guarded_gpu_selection(gpus: list[str], *, required: bool) -> set[str] | None:
    if not required:
        return None
    try:
        gpu_uuids = selected_gpu_uuids(gpus)
        check_gpu_exclusivity(os.getpid(), gpu_uuids)
    except GPUExclusivityError:
        raise
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise GPUExclusivityError([], query_error=str(error)) from error
    return gpu_uuids


def matrix_manifest(
    suite: str,
    api_routing: dict[str, Any],
    pinned_benchmarks: dict[str, dict[str, Any]],
    experiments: list[Experiment],
    training_policy: str = "paper",
) -> dict[str, Any]:
    return {
        "suite": suite,
        "training_policy": training_policy,
        "formal_training_contract": {
            "schema_version": 1,
            "sample_count_source": "task.formal_expected_samples",
            "require_complete_epoch_schedule": True,
            "paper_methods": ["full", "lora"],
            "rslora_is_paired_comparison_only": True,
            "method_lock_scope": "experiment_id",
            "method_lock_file": FORMAL_METHOD_LOCK_FILE,
        },
        "method_selection_resource": {
            "scope": PAPER_LOGICAL_RESOURCE_SCOPE,
            "gpu_count": PAPER_LOGICAL_GPU_COUNT,
            "gpu_name": PAPER_LOGICAL_GPU_NAME,
            "memory_per_gpu_gb": PAPER_LOGICAL_GPU_MEMORY_GB,
            "physical_execution_mapping": H20_EXECUTION_MAPPING,
        },
        "api_routing": api_routing,
        "benchmark_assets": {
            benchmark: {
                "id": asset["id"],
                "revision": asset["revision"],
                "dataset_path": benchmark_dataset_path(asset),
            }
            for benchmark, asset in sorted(pinned_benchmarks.items())
        },
        "tasks": [
            {
                **experiment.to_dict(),
                "formal_expected_samples": experiment.data_limit,
            }
            for experiment in experiments
        ],
    }


def execute_workers(
    experiments: list[Experiment],
    gpus: list[str],
    run_root: Path,
    pinned_benchmarks: dict[str, dict[str, Any]],
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
    guarded_gpu_uuids: set[str] | None,
    gpu_guard_interval: float,
    workers_per_gpu: int,
    max_parallel: int | None,
) -> int:
    try:
        success = asyncio.run(
            run_workers(
                experiments,
                gpus,
                run_root,
                read_scenarios(),
                pinned_benchmarks,
                runtime_environment,
                api_routing,
                guarded_gpu_uuids,
                gpu_guard_interval,
                workers_per_gpu,
                max_parallel,
            ),
        )
    except GPUExclusivityError as error:
        status_write(
            run_root / "invalid.json",
            {
                "schema_version": 1,
                "reason": "external_gpu_conflict",
                "detected_at": datetime.now().astimezone().isoformat(),
                "selected_gpus": gpus,
                "processes": error.processes,
                "query_error": error.query_error,
            },
        )
        print(f"GPU EXCLUSIVITY FAILURE: {error}", flush=True)
        return GPU_CONFLICT_EXIT_CODE
    return 0 if success else 1


def execute_run(
    args: argparse.Namespace,
    experiments: list[Experiment],
    manifest_experiments: list[Experiment],
    gpus: list[str],
    run_root: Path,
    pinned_benchmarks: dict[str, dict[str, Any]],
    guarded_gpu_uuids: set[str] | None,
) -> int:
    with routed_api_environment(os.environ) as (runtime_environment, api_routing):
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = matrix_manifest(
            args.suite,
            api_routing,
            pinned_benchmarks,
            manifest_experiments,
            args.training_policy,
        )
        write_or_validate_manifest(run_root, manifest)
        try:
            return execute_workers(
                experiments,
                gpus,
                run_root,
                pinned_benchmarks,
                runtime_environment,
                api_routing,
                guarded_gpu_uuids,
                args.gpu_guard_interval,
                args.workers_per_gpu,
                args.max_parallel,
            )
        finally:
            report = collect_and_write_run(run_root, "ft-agent")
            print(f"Report: {run_root / 'report'} ({report['coverage']['supplied_tasks']} tasks)")


def finish_empty_run(run_root: Path, *, dry_run: bool) -> int:
    print(f"Selected 0 tasks; GPUs=none; output={run_root}")
    if not dry_run:
        report = collect_and_write_run(run_root, "ft-agent")
        print(f"Report: {run_root / 'report'} ({report['coverage']['supplied_tasks']} tasks)")
    return 0


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    configure_project_environment(args.training_policy)
    validate_max_parallel(args.max_parallel)
    validate_workers_per_gpu(args.workers_per_gpu)
    if args.gpu_guard_interval <= 0:
        message = "--gpu-guard-interval must be positive"
        raise SystemExit(message)

    experiments = selected_experiments(args.suite)
    if args.only:
        pattern = re.compile(args.only)
        experiments = [experiment for experiment in experiments if pattern.search(experiment.experiment_id)]
    if not experiments:
        message = "No experiments selected"
        raise SystemExit(message)
    experiments = override_task_timeout(experiments, args.task_timeout)
    manifest_experiments = list(experiments)
    asset_manifest = read_asset_manifest()
    pinned_benchmarks = benchmark_assets(manifest_experiments, asset_manifest)

    if args.preflight_only:
        preflight(experiments, asset_manifest, pinned_benchmarks)
        print(f"Preflight OK: {len(experiments)} tasks")
        return 0

    run_name = args.run_name or datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = FT_ROOT / "logs" / "paper-matrix" / run_name
    if args.resume:
        experiments = filter_succeeded_experiments(experiments, run_root)
    if not experiments:
        return finish_empty_run(run_root, dry_run=args.dry_run)

    candidate_gpus = visible_gpus(args.gpus)
    assignments = worker_gpu_assignments(
        candidate_gpus,
        args.workers_per_gpu,
        args.max_parallel,
    )
    gpus = list(dict.fromkeys(assignments))
    try:
        guarded_gpu_uuids = guarded_gpu_selection(gpus, required=args.require_exclusive_gpus)
    except GPUExclusivityError as error:
        print(f"GPU EXCLUSIVITY FAILURE: {error}", flush=True)
        return GPU_CONFLICT_EXIT_CODE
    print_experiment_selection(experiments, gpus, run_root)
    print(
        f"Pipeline workers={len(assignments)} ({args.workers_per_gpu} per GPU; GPU stages lease-serialized)",
    )
    if args.dry_run:
        return 0
    validate_gpu_lease_entrypoints(args.workers_per_gpu)
    refuse_existing_task_roots(experiments, run_root)
    if not args.skip_preflight:
        preflight(experiments, asset_manifest, pinned_benchmarks)
    return execute_run(
        args,
        experiments,
        manifest_experiments,
        gpus,
        run_root,
        pinned_benchmarks,
        guarded_gpu_uuids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
