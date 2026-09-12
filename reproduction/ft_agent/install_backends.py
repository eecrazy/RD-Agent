#!/usr/bin/env python3
"""Install and verify the project-local FT-Agent conda backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = ROOT / "finetune_files"
CONDA_ENVS = FT_ROOT / "conda_envs"
LOCK_ROOT = FT_ROOT / "environment-locks"
CONDA_CONFIG = ROOT / "reproduction" / "condarc"
REQUIREMENTS_ROOT = ROOT / "rdagent" / "scenarios" / "finetune" / "env" / "conda"
TRAINING_REQUIREMENTS = REQUIREMENTS_ROOT / "llm_finetune_requirements.txt"
BENCHMARK_REQUIREMENTS = REQUIREMENTS_ROOT / "opencompass_requirements.txt"
FLASH_ATTN_VERSION = "2.8.3.post1"
CUDA_NVCC_VERSION = "12.8.93"
CUDA_CUDART_VERSION = "12.8.90"
PYTHON_VERSION = "3.10"
CONDA_FORGE_CHANNEL = "https://conda.anaconda.org/conda-forge"
GPU_LEASE_LOCK_ROOT = FT_ROOT / "gpu_leases" / "locks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", "training", "benchmark"), default="all")
    parser.add_argument("--force", action="store_true", help="Reinstall requirements and FlashAttention")
    parser.add_argument("--max-jobs", type=int, default=8, help="Parallel compiler jobs for FlashAttention")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without changing environments")
    return parser.parse_args()


def project_environment() -> dict[str, str]:
    load_dotenv(ROOT / ".env", override=False)
    environment = os.environ.copy()
    environment.update(
        {
            "CONDARC": str(CONDA_CONFIG),
            "CONDA_ENVS_PATH": str(CONDA_ENVS),
            "CONDA_PKGS_DIRS": str(FT_ROOT / "cache" / "conda_pkgs"),
            "PIP_CACHE_DIR": str(FT_ROOT / "cache" / "pip"),
            "TMPDIR": str(FT_ROOT / "tmp"),
        },
    )
    return environment


def run(command: list[str], environment: dict[str, str], *, dry_run: bool, capture: bool = False) -> str:
    print(f"+ {shlex.join(command)}", flush=True)
    if dry_run:
        return ""
    result = subprocess.run(  # noqa: S603
        command,
        check=True,
        env=environment,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def valid_env_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        message = f"Unsafe conda environment name: {name!r}"
        raise ValueError(message)
    return name


def env_prefix(name: str) -> Path:
    return CONDA_ENVS / valid_env_name(name)


def conda_forge_channel(environment: dict[str, str]) -> str:
    return environment.get("FT_CONDA_FORGE_CHANNEL", CONDA_FORGE_CHANNEL)


def ensure_environment(
    conda: str,
    name: str,
    environment: dict[str, str],
    *,
    dry_run: bool,
) -> Path:
    prefix = env_prefix(name)
    if not (prefix / "bin" / "python").is_file():
        run(
            [
                conda,
                "create",
                "--yes",
                "--prefix",
                str(prefix),
                "--override-channels",
                "--channel",
                conda_forge_channel(environment),
                f"python={PYTHON_VERSION}",
                "pip",
            ],
            environment,
            dry_run=dry_run,
        )
    return prefix


def pip_command(prefix: Path, *arguments: str) -> list[str]:
    return [str(prefix / "bin" / "python"), "-m", "pip", *arguments]


def install_requirements(
    prefix: Path,
    requirements: Path,
    environment: dict[str, str],
    *,
    force: bool,
    dry_run: bool,
) -> None:
    run(
        pip_command(
            prefix,
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "ninja",
            "packaging",
            "psutil",
        ),
        environment,
        dry_run=dry_run,
    )
    command = pip_command(prefix, "install", "--requirement", str(requirements))
    if force:
        command.insert(4, "--force-reinstall")
    run(command, environment, dry_run=dry_run)


def install_gpu_lease_entrypoint(
    prefix: Path,
    script_name: str,
    main_module: str,
    *,
    dry_run: bool,
) -> None:
    """Replace a console entrypoint with a per-CUDA-device lease wrapper."""
    # Console scripts may be invoked from arbitrary experiment workspaces.  An
    # installer caller can legitimately pass a relative prefix, but a relative
    # interpreter in the generated shebang is then resolved against the
    # *runtime* working directory and fails with ENOENT.  Normalize it before
    # generating the wrapper so the entrypoint is location-independent.
    prefix = prefix.expanduser().resolve()
    target = prefix / "bin" / script_name
    print(f"+ install GPU lease wrapper {target}", flush=True)
    if dry_run:
        return
    python = prefix / "bin" / "python"
    if not python.is_file() or not target.is_file():
        message = f"Cannot install GPU lease wrapper for missing entrypoint: {target}"
        raise RuntimeError(message)
    opportunistic_multi_gpu = script_name == "opencompass"
    content = f"""#!{python}
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time


OPPORTUNISTIC_MULTI_GPU = {opportunistic_multi_gpu!r}
IS_TRAINING_WRAPPER = {not opportunistic_multi_gpu!r}
DEFAULT_DEEPSPEED_ZERO3_CONFIG = {str(REQUIREMENTS_ROOT / "deepspeed" / "ds_z3_config.json")!r}
PROJECT_ROOT = {str(ROOT)!r}
TRAINING_POLICIES = ("paper", "full", "lora", "rslora")
BENCHMARK_GPU_LIMITS = {{
    "chemcotbench_mol_und": 5,
    "chemcotbench_reaction": 5,
    "tablebench_data_analysis": 5,
}}


def active_training_policy():
    raw = os.environ.get("FT_TRAINING_POLICY", "paper")
    policy = raw.strip().lower().replace("-", "")
    if policy not in TRAINING_POLICIES:
        print(
            f"FT_TRAINING_POLICY must be one of {{', '.join(TRAINING_POLICIES)}}; got {{raw!r}}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    return policy


def training_method(config):
    finetuning_type = str(config.get("finetuning_type", "")).strip().lower()
    if finetuning_type == "full":
        return "full"
    if finetuning_type == "lora":
        return "rslora" if config.get("use_rslora") is True else "lora"
    return None


def full_sft_gpu_count(config):
    raw = os.environ.get("FT_FULL_SFT_GPUS")
    if raw is None:
        try:
            cutoff_len = int(config.get("cutoff_len", 0) or 0)
        except (TypeError, ValueError):
            cutoff_len = 0
        raw = "4" if cutoff_len > 8192 else "2"
    try:
        count = int(raw)
    except ValueError as error:
        print(f"Invalid FT_FULL_SFT_GPUS={{raw!r}}", file=sys.stderr, flush=True)
        raise SystemExit(78) from error
    if count < 2:
        print("FT_FULL_SFT_GPUS must be at least 2", file=sys.stderr, flush=True)
        raise SystemExit(78)
    return count


def normalize_full_sft_batch(config, config_path, source, world_size):
    try:
        batch = int(config.get("per_device_train_batch_size", 8))
        accumulation = int(config.get("gradient_accumulation_steps", 1))
    except (TypeError, ValueError) as error:
        print(f"Full SFT batch settings must be integers: {{error}}", file=sys.stderr, flush=True)
        raise SystemExit(78) from error
    if batch < 1 or accumulation < 1:
        print("Full SFT batch settings must be positive", file=sys.stderr, flush=True)
        raise SystemExit(78)

    match = re.search(
        r"^# rdagent_global_batch_size: ([0-9]+)$",
        source,
        re.MULTILINE,
    )
    target_global_batch = int(match.group(1)) if match else batch * accumulation
    if config_path.name == "debug_train.yaml" and target_global_batch < world_size:
        print(
            f"Debug full-SFT global batch {{target_global_batch}} cannot be preserved on "
            f"{{world_size}} GPUs; allowing the debug-only batch to expand",
            flush=True,
        )
        return None

    selected = None
    for candidate_batch in range(min(batch, target_global_batch // world_size), 0, -1):
        denominator = candidate_batch * world_size
        if target_global_batch % denominator == 0:
            selected = (candidate_batch, target_global_batch // denominator)
            break
    if selected is None:
        print(
            f"Cannot preserve full-SFT global batch {{target_global_batch}} on {{world_size}} GPUs; "
            "choose a batch divisible by the world size",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)

    config["per_device_train_batch_size"], config["gradient_accumulation_steps"] = selected
    print(
        f"Full SFT batch contract: global={{target_global_batch}}, per_device={{selected[0]}}, "
        f"accumulation={{selected[1]}}, world_size={{world_size}}",
        flush=True,
    )
    return target_global_batch


def validate_formal_training_contract(config_path, experiment_id, policy):
    raw_expected = os.environ.get("FT_FORMAL_EXPECTED_SAMPLES")
    # Debug/micro-batch invocations deliberately use a tiny dataset and must
    # never be mistaken for the formal train.yaml boundary.
    if raw_expected is None or not raw_expected.strip() or config_path.name != "train.yaml":
        return
    try:
        expected_samples = int(raw_expected)
        if expected_samples < 1:
            raise ValueError("expected sample count must be positive")
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from rdagent.scenarios.finetune.train.formal_training import (
            enforce_formal_training_method_lock,
            validate_formal_training_inputs,
        )

        facts = validate_formal_training_inputs(
            config_path.resolve().parent,
            expected_samples=expected_samples,
            experiment_id=experiment_id,
            training_policy=policy,
            require_runtime_contract=True,
        )
        method_lock_path = os.environ.get("FT_FORMAL_METHOD_LOCK_PATH", "").strip()
        if not method_lock_path:
            raise RuntimeError("FT_FORMAL_METHOD_LOCK_PATH is required for formal matrix training")
        enforce_formal_training_method_lock(
            method_lock_path,
            experiment_id=experiment_id,
            training_policy=policy,
            training_method=facts["training_method"],
        )
    except Exception as error:
        print(
            f"Formal training contract rejected {{experiment_id}} before GPU lease: {{error}}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78) from error
    print(
        f"Formal training contract accepted {{experiment_id}}: exactly {{expected_samples}} samples; "
        f"method={{facts['training_method']}} locked",
        flush=True,
    )


def prepare_training_invocation():
    if not IS_TRAINING_WRAPPER or len(sys.argv) < 2 or sys.argv[1] != "train":
        return None
    experiment_id = os.environ.get("FT_EXPERIMENT_ID", "standalone").strip() or "standalone"
    if len(sys.argv) < 3:
        print(
            f"Training policy rejected {{experiment_id}}: missing training config",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    config_path = Path(sys.argv[2]).expanduser()
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as error:
        print(
            f"Training policy rejected {{experiment_id}}: cannot read {{config_path}}: {{error}}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78) from error
    if not isinstance(config, dict):
        print(
            f"Training policy rejected {{experiment_id}}: {{config_path}} is not a mapping",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)

    # Every wrapper invocation is a fresh process, so this boundary also
    # protects matrix workers that imported an older in-process validator.
    # A debug run only proves that forward/backward/optimizer steps execute;
    # saving and then reloading ZeRO-3 optimizer shards can consume hundreds
    # of gigabytes and deadlock after the useful smoke test has finished.
    # Never change the formal train.yaml checkpoint policy here.
    if config_path.name == "debug_train.yaml":
        config["save_strategy"] = "no"
        config["load_best_model_at_end"] = False
        config["save_only_model"] = True
        config.pop("save_steps", None)
        config.pop("save_total_limit", None)
        config.pop("resume_from_checkpoint", None)
        config_path.write_text(
            yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        print("Debug training checkpoint creation/reload disabled", flush=True)

    policy = active_training_policy()
    method = training_method(config)
    violations = []
    if method is None:
        violations.append("finetuning_type must be full or lora")
    if config.get("use_dora", False) is not False:
        violations.append("use_dora must be false or omitted")
    if config.get("quantization_bit") is not None:
        violations.append("quantization_bit must be null or omitted; QLoRA is not allowed")
    if str(config.get("finetuning_type", "")).strip().lower() == "full" and "use_rslora" in config:
        violations.append("use_rslora must be omitted for full SFT")
    if policy == "paper" and method == "rslora":
        violations.append("method must be full or ordinary lora under policy paper (got rslora)")
    expected = None if policy == "paper" else policy
    if expected is not None and method is not None and method != expected:
        violations.append(f"method must be {{expected}} under policy {{policy}} (got {{method}})")
    if violations:
        print(
            f"Training policy {{policy}} rejected {{experiment_id}} before GPU lease: "
            + "; ".join(violations),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)

    if method == "full":
        zero3 = Path(os.environ.get("FT_DEEPSPEED_ZERO3_CONFIG", DEFAULT_DEEPSPEED_ZERO3_CONFIG))
        if not zero3.is_file():
            print(f"ZeRO-3 config is missing: {{zero3}}", file=sys.stderr, flush=True)
            raise SystemExit(78)
        world_size = full_sft_gpu_count(config)
        target_global_batch = normalize_full_sft_batch(
            config,
            config_path,
            config_path.read_text(encoding="utf-8"),
            world_size,
        )
        config["deepspeed"] = str(zero3)
        rendered = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
        if target_global_batch is not None:
            rendered = (
                f"# rdagent_global_batch_size: {{target_global_batch}}\\n"
                f"# rdagent_world_size: {{world_size}}\\n"
                + rendered
            )
        config_path.write_text(rendered, encoding="utf-8")

    validate_formal_training_contract(config_path, experiment_id, policy)

    os.environ["FT_TRAINING_METHOD"] = method
    print(f"Training policy={{policy}} method={{method}} config={{config_path}}", flush=True)
    return config


def dynamic_gpu_candidates(requested, lock_root):
    if len(requested) != 1:
        return requested
    pool_file = Path(
        os.environ.get("FT_GPU_LEASE_POOL_FILE", str(lock_root.parent / "dynamic_pool"))
    )
    if not pool_file.is_file():
        return requested
    configured = pool_file.read_text(encoding="utf-8").replace(",", " ").split()
    return requested + [gpu for gpu in configured if gpu not in requested]


def desired_gpu_lease_count(training_config=None):
    if IS_TRAINING_WRAPPER:
        if training_config is not None and training_method(training_config) == "full":
            return full_sft_gpu_count(training_config)
        return 1
    benchmark = os.environ.get("FT_TARGET_BENCHMARK", "").strip()
    default = BENCHMARK_GPU_LIMITS.get(benchmark, 1)
    raw = os.environ.get("FT_OPENCOMPASS_MAX_GPUS", str(default))
    try:
        value = int(raw)
    except ValueError as error:
        print(
            f"Invalid FT_OPENCOMPASS_MAX_GPUS={{raw!r}}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78) from error
    if value < 1:
        print(
            "FT_OPENCOMPASS_MAX_GPUS must be positive",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    return value


def invocation_requires_gpu_lease():
    if not IS_TRAINING_WRAPPER:
        return True
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    return command not in {"help", "version", "--help", "-h", "--version"}


def gpu_memory_snapshot(gpu):
    command = [
        os.environ.get("FT_NVIDIA_SMI", "nvidia-smi"),
        f"--id={{gpu}}",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(
            f"Skipping GPU memory readiness check for {{gpu}}: {{error}}",
            file=sys.stderr,
            flush=True,
        )
        return None
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {{result.returncode}}"
        print(
            f"Skipping GPU memory readiness check for {{gpu}}: {{detail}}",
            file=sys.stderr,
            flush=True,
        )
        return None
    try:
        row = result.stdout.strip().splitlines()[0]
        free_text, total_text = (item.strip() for item in row.split(",", maxsplit=1))
        return float(free_text), float(total_text)
    except (IndexError, TypeError, ValueError) as error:
        print(
            f"Skipping GPU memory readiness check for {{gpu}}: invalid nvidia-smi output "
            f"{{result.stdout!r}} ({{error}})",
            file=sys.stderr,
            flush=True,
        )
        return None


def wait_for_gpu_memory_ready(gpu):
    try:
        required_fraction = float(os.environ.get("FT_GPU_MEMORY_READY_FRACTION", "0.95"))
        timeout = float(os.environ.get("FT_GPU_MEMORY_READY_TIMEOUT", "300"))
        poll_interval = float(os.environ.get("FT_GPU_MEMORY_READY_POLL_INTERVAL", "1"))
    except ValueError as error:
        print(f"Invalid GPU memory readiness setting: {{error}}", file=sys.stderr, flush=True)
        raise SystemExit(78) from error
    if not 0 <= required_fraction <= 1 or timeout < 0 or poll_interval <= 0:
        print(
            "GPU memory readiness settings require a fraction in [0, 1], "
            "a non-negative timeout, and a positive poll interval",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    if required_fraction == 0:
        return

    started = time.monotonic()
    deadline = started + timeout
    next_log = started
    waited = False
    while True:
        snapshot = gpu_memory_snapshot(gpu)
        if snapshot is None:
            return
        free_memory, total_memory = snapshot
        free_fraction = free_memory / total_memory if total_memory > 0 else 0
        now = time.monotonic()
        if free_fraction >= required_fraction:
            if waited:
                print(
                    f"GPU {{gpu}} memory ready after {{now - started:.1f}}s: "
                    f"{{free_memory / 1024:.2f}}/{{total_memory / 1024:.2f}} GiB free",
                    flush=True,
                )
            return
        if now >= next_log:
            print(
                f"Waiting for GPU {{gpu}} memory reclamation: "
                f"{{free_memory / 1024:.2f}}/{{total_memory / 1024:.2f}} GiB free "
                f"({{free_fraction:.1%}}; need {{required_fraction:.1%}})",
                flush=True,
            )
            next_log = now + 15
        waited = True
        if now >= deadline:
            print(
                f"Timed out after {{timeout:.1f}}s waiting for GPU {{gpu}} memory reclamation",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(75)
        time.sleep(min(poll_interval, deadline - now))


def gpu_memory_ready_now(gpu):
    try:
        required_fraction = float(os.environ.get("FT_GPU_MEMORY_READY_FRACTION", "0.95"))
    except ValueError as error:
        print(f"Invalid GPU memory readiness setting: {{error}}", file=sys.stderr, flush=True)
        raise SystemExit(78) from error
    if not 0 <= required_fraction <= 1:
        print(
            "GPU memory readiness fraction must be in [0, 1]",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    if required_fraction == 0:
        return True
    snapshot = gpu_memory_snapshot(gpu)
    if snapshot is None:
        return True
    free_memory, total_memory = snapshot
    return total_memory > 0 and free_memory / total_memory >= required_fraction


def acquire_opportunistic_gpu_leases(visible, candidates, lock_root, limit):
    requested = visible[0]
    print(
        f"Waiting for any GPU lease (requested={{requested}}; pool={{','.join(candidates)}})",
        flush=True,
    )
    while True:
        for gpu in candidates:
            safe_gpu = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in gpu
            )
            lease = (lock_root / f"gpu-{{safe_gpu}}.lock").open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            leases = [lease]
            acquired = [gpu]
            wait_for_gpu_memory_ready(gpu)
            break
        else:
            time.sleep(0.2)
            continue
        break

    for gpu in candidates:
        if len(leases) >= min(limit, len(candidates)):
            break
        if gpu in acquired:
            continue
        safe_gpu = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in gpu
        )
        lease = (lock_root / f"gpu-{{safe_gpu}}.lock").open("a+")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        if not gpu_memory_ready_now(gpu):
            lease.close()
            continue
        leases.append(lease)
        acquired.append(gpu)

    os.environ["FT_REQUESTED_CUDA_VISIBLE_DEVICES"] = requested
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(acquired)
    print(
        f"Acquired GPU leases: {{','.join(acquired)}} (requested={{requested}}; limit={{limit}})",
        flush=True,
    )
    return leases


def acquire_exact_gpu_leases(visible, candidates, lock_root, desired):
    if desired > len(candidates):
        print(
            f"Training requires {{desired}} GPUs but the lease pool has only {{len(candidates)}}: "
            f"{{','.join(candidates)}}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(78)
    requested = ",".join(visible)
    print(
        f"Waiting for an atomic {{desired}}-GPU lease (requested={{requested}}; "
        f"pool={{','.join(candidates)}})",
        flush=True,
    )
    while True:
        leases = []
        acquired = []
        for gpu in candidates:
            safe_gpu = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in gpu
            )
            lease = (lock_root / f"gpu-{{safe_gpu}}.lock").open("a+")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            leases.append(lease)
            acquired.append(gpu)
            if len(acquired) == desired:
                break
        if len(acquired) == desired:
            break
        for lease in reversed(leases):
            lease.close()
        time.sleep(0.2)

    os.environ["FT_REQUESTED_CUDA_VISIBLE_DEVICES"] = requested
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(acquired)
    for gpu in acquired:
        wait_for_gpu_memory_ready(gpu)
    print(f"Acquired atomic GPU leases: {{','.join(acquired)}}", flush=True)
    return leases


def acquire_gpu_leases(training_config=None):
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    lock_root = Path(os.environ.get("FT_GPU_LEASE_LOCK_ROOT", {str(GPU_LEASE_LOCK_ROOT)!r}))
    lock_root.mkdir(parents=True, exist_ok=True)
    candidates = dynamic_gpu_candidates(visible, lock_root)
    desired = desired_gpu_lease_count(training_config)
    if IS_TRAINING_WRAPPER:
        return acquire_exact_gpu_leases(visible, candidates, lock_root, desired)
    if len(visible) == 1 and len(candidates) > 1 and desired > 1:
        return acquire_opportunistic_gpu_leases(visible, candidates, lock_root, desired)
    if len(visible) == 1 and len(candidates) > 1:
        requested = visible[0]
        print(
            f"Waiting for any GPU lease (requested={{requested}}; pool={{','.join(candidates)}})",
            flush=True,
        )
        while True:
            for gpu in candidates:
                safe_gpu = "".join(
                    character if character.isalnum() or character in "-_." else "_"
                    for character in gpu
                )
                lease = (lock_root / f"gpu-{{safe_gpu}}.lock").open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                os.environ["FT_REQUESTED_CUDA_VISIBLE_DEVICES"] = requested
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu
                print(f"Acquired GPU lease: {{gpu}} (requested={{requested}})", flush=True)
                wait_for_gpu_memory_ready(gpu)
                return [lease]
            time.sleep(0.2)
    leases = []
    for gpu in sorted(visible):
        safe_gpu = "".join(character if character.isalnum() or character in "-_." else "_" for character in gpu)
        lease = (lock_root / f"gpu-{{safe_gpu}}.lock").open("a+")
        print(f"Waiting for GPU lease: {{gpu}}", flush=True)
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        print(f"Acquired GPU lease: {{gpu}}", flush=True)
        wait_for_gpu_memory_ready(gpu)
        leases.append(lease)
    return leases


def configure_training_launch(training_config):
    if training_config is None or training_method(training_config) != "full":
        return
    world_size = len(
        [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    )
    if world_size < 2:
        print("Full SFT requires at least two leased GPUs", file=sys.stderr, flush=True)
        raise SystemExit(78)
    os.environ["FORCE_TORCHRUN"] = "1"
    os.environ["NPROC_PER_NODE"] = str(world_size)
    print(f"Configured ZeRO-3 torchrun with world_size={{world_size}}", flush=True)


def prioritize_active_environment_path():
    # Make subprocess launchers resolve from the wrapper's Python environment.
    active_bin = str(Path(sys.executable).resolve().parent)
    existing = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    os.environ["PATH"] = os.pathsep.join(
        [active_bin, *(entry for entry in existing if Path(entry).resolve() != Path(active_bin))],
    )


if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    prioritize_active_environment_path()
    _training_config = prepare_training_invocation()
    _gpu_leases = acquire_gpu_leases(_training_config) if invocation_requires_gpu_lease() else []
    configure_training_launch(_training_config)
    from {main_module} import main

    sys.exit(main())
"""
    temporary = target.with_name(f".{target.name}.gpu-lease.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o755)
    temporary.replace(target)


def installed_version(prefix: Path, module: str, environment: dict[str, str], *, dry_run: bool) -> str | None:
    if dry_run or not (prefix / "bin" / "python").is_file():
        return None
    code = f"import importlib.metadata as m; print(m.version({module!r}))"
    result = subprocess.run(  # noqa: S603
        [str(prefix / "bin" / "python"), "-c", code],
        check=False,
        env=environment,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def install_flash_attention(
    conda: str,
    prefix: Path,
    environment: dict[str, str],
    *,
    force: bool,
    max_jobs: int,
    dry_run: bool,
) -> None:
    current = installed_version(prefix, "flash-attn", environment, dry_run=dry_run)
    if current == FLASH_ATTN_VERSION and not force:
        print(f"FlashAttention {current} is already installed", flush=True)
        return

    run(
        [
            conda,
            "install",
            "--yes",
            "--prefix",
            str(prefix),
            "--override-channels",
            "--channel",
            conda_forge_channel(environment),
            "--freeze-installed",
            f"cuda-nvcc={CUDA_NVCC_VERSION}",
            f"cuda-cudart-dev={CUDA_CUDART_VERSION}",
        ],
        environment,
        dry_run=dry_run,
    )
    nvcc = prefix / "bin" / "nvcc"
    run([str(nvcc), "--version"], environment, dry_run=dry_run)

    build_environment = environment.copy()
    cuda_target = prefix / "targets" / "x86_64-linux"
    cuda_include = cuda_target / "include"
    cuda_library = cuda_target / "lib"
    build_environment.update(
        {
            "CUDA_HOME": str(prefix),
            "CPATH": os.pathsep.join(filter(None, (str(cuda_include), environment.get("CPATH")))),
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_CUDA_ARCHS": "90",
            "LD_LIBRARY_PATH": os.pathsep.join(
                filter(None, (str(cuda_library), environment.get("LD_LIBRARY_PATH"))),
            ),
            "LIBRARY_PATH": os.pathsep.join(
                filter(None, (str(cuda_library), environment.get("LIBRARY_PATH"))),
            ),
            "MAX_JOBS": str(max_jobs),
            "PATH": f"{prefix / 'bin'}:{environment.get('PATH', '')}",
        },
    )
    if Path("/usr/bin/gcc").is_file() and Path("/usr/bin/g++").is_file():
        build_environment.update({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"})
    run(
        pip_command(
            prefix,
            "install",
            f"flash-attn=={FLASH_ATTN_VERSION}",
            "--no-build-isolation",
            "--no-cache-dir",
        ),
        build_environment,
        dry_run=dry_run,
    )


def verify_named_environment(
    conda: str,
    name: str,
    prefix: Path,
    environment: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    code = "import pathlib,sys; print(pathlib.Path(sys.prefix).resolve())"
    actual = run(
        [conda, "run", "--name", name, "python", "-c", code],
        environment,
        dry_run=False,
        capture=True,
    ).strip()
    if Path(actual).resolve() != prefix.resolve():
        message = f"Conda name {name!r} resolves to {actual}, expected {prefix}"
        raise RuntimeError(message)


def verify_training(prefix: Path, environment: dict[str, str], *, dry_run: bool) -> None:
    code = """
import flash_attn
import torch
from importlib.metadata import version

assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert (major, minor) >= (9, 0), (major, minor)
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "flash_attn": flash_attn.__version__, "llamafactory": version("llamafactory")})
"""
    run([str(prefix / "bin" / "python"), "-c", code], environment, dry_run=dry_run)
    run([str(prefix / "bin" / "llamafactory-cli"), "version"], environment, dry_run=dry_run)
    run(pip_command(prefix, "check"), environment, dry_run=dry_run)


def verify_benchmark(prefix: Path, environment: dict[str, str], *, dry_run: bool) -> None:
    code = """
import torch
import opencompass
import vllm

assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "vllm": vllm.__version__, "opencompass": opencompass.__file__})
"""
    run([str(prefix / "bin" / "python"), "-c", code], environment, dry_run=dry_run)
    run([str(prefix / "bin" / "opencompass"), "--help"], environment, dry_run=dry_run)
    run(pip_command(prefix, "check"), environment, dry_run=dry_run)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_locks(
    conda: str,
    name: str,
    prefix: Path,
    requirements: Path,
    environment: dict[str, str],
) -> None:
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    pip_freeze = run(
        pip_command(prefix, "freeze", "--all"),
        environment,
        dry_run=False,
        capture=True,
    )
    conda_explicit = run(
        [conda, "list", "--explicit", "--prefix", str(prefix)],
        environment,
        dry_run=False,
        capture=True,
    )
    (LOCK_ROOT / f"{name}.pip.txt").write_text(pip_freeze, encoding="utf-8")
    (LOCK_ROOT / f"{name}.conda.txt").write_text(conda_explicit, encoding="utf-8")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "prefix": prefix.resolve().relative_to(ROOT.resolve()).as_posix(),
        "python": PYTHON_VERSION,
        "requirements": requirements.relative_to(ROOT).as_posix(),
        "requirements_sha256": sha256(requirements),
    }
    (LOCK_ROOT / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def prepare_directories(environment: dict[str, str]) -> None:
    for path in (
        CONDA_ENVS,
        Path(environment["CONDA_PKGS_DIRS"]),
        Path(environment["PIP_CACHE_DIR"]),
        Path(environment["TMPDIR"]),
    ):
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.max_jobs < 1:
        message = "--max-jobs must be positive"
        raise SystemExit(message)
    environment = project_environment()
    conda = shutil.which("conda", path=environment.get("PATH"))
    if conda is None:
        message = "conda is not available"
        raise SystemExit(message)
    if not args.dry_run:
        prepare_directories(environment)

    training_name = valid_env_name(environment.get("FT_CONDA_CONDA_ENV_NAME", "llm_finetune"))
    benchmark_name = valid_env_name(environment.get("BENCHMARK_CONDA_CONDA_ENV_NAME", "opencompass"))

    if args.backend in {"all", "training"}:
        training_prefix = ensure_environment(
            conda,
            training_name,
            environment,
            dry_run=args.dry_run,
        )
        install_requirements(
            training_prefix,
            TRAINING_REQUIREMENTS,
            environment,
            force=args.force,
            dry_run=args.dry_run,
        )
        install_flash_attention(
            conda,
            training_prefix,
            environment,
            force=args.force,
            max_jobs=args.max_jobs,
            dry_run=args.dry_run,
        )
        install_gpu_lease_entrypoint(
            training_prefix,
            "llamafactory-cli",
            "llamafactory.cli",
            dry_run=args.dry_run,
        )
        verify_named_environment(
            conda,
            training_name,
            training_prefix,
            environment,
            dry_run=args.dry_run,
        )
        verify_training(training_prefix, environment, dry_run=args.dry_run)
        if not args.dry_run:
            write_locks(conda, training_name, training_prefix, TRAINING_REQUIREMENTS, environment)

    if args.backend in {"all", "benchmark"}:
        benchmark_prefix = ensure_environment(
            conda,
            benchmark_name,
            environment,
            dry_run=args.dry_run,
        )
        install_requirements(
            benchmark_prefix,
            BENCHMARK_REQUIREMENTS,
            environment,
            force=args.force,
            dry_run=args.dry_run,
        )
        install_gpu_lease_entrypoint(
            benchmark_prefix,
            "opencompass",
            "opencompass.cli.main",
            dry_run=args.dry_run,
        )
        verify_named_environment(
            conda,
            benchmark_name,
            benchmark_prefix,
            environment,
            dry_run=args.dry_run,
        )
        verify_benchmark(benchmark_prefix, environment, dry_run=args.dry_run)
        if not args.dry_run:
            write_locks(conda, benchmark_name, benchmark_prefix, BENCHMARK_REQUIREMENTS, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
