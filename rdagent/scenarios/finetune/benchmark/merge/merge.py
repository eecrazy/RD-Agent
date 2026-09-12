import contextlib
import fcntl
import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

from rdagent.components.coder.finetune.conf import get_workspace_prefix
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T

BLACKWELL_GPU_KEYWORDS = ["b100", "b200", "b300"]


@contextlib.contextmanager
def gpu_lease_environment() -> Iterator[dict[str, str]]:  # noqa: C901
    """Lease the GPU used by an out-of-band model merge when configured."""
    lock_root_value = os.environ.get("FT_GPU_LEASE_LOCK_ROOT")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not lock_root_value or not visible:
        yield {}
        return

    lock_root = Path(lock_root_value)
    lock_root.mkdir(parents=True, exist_ok=True)
    candidates = visible
    if len(visible) == 1:
        pool_file = Path(
            os.environ.get("FT_GPU_LEASE_POOL_FILE", str(lock_root.parent / "dynamic_pool")),
        )
        if pool_file.is_file():
            configured = pool_file.read_text(encoding="utf-8").replace(",", " ").split()
            candidates = visible + [gpu for gpu in configured if gpu not in visible]

    leases = []
    acquired: list[str] = []
    requested = ",".join(visible)
    try:
        if len(visible) == 1 and len(candidates) > 1:
            logger.info(f"Waiting for any GPU lease for model merge (requested={requested})")
            while not leases:
                for gpu in candidates:
                    safe_gpu = "".join(
                        character if character.isalnum() or character in "-_." else "_" for character in gpu
                    )
                    lease = (lock_root / f"gpu-{safe_gpu}.lock").open("a+")
                    try:
                        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close()
                        continue
                    leases.append(lease)
                    acquired.append(gpu)
                    break
                if not leases:
                    time.sleep(0.2)
        else:
            for gpu in sorted(visible):
                safe_gpu = "".join(
                    character if character.isalnum() or character in "-_." else "_" for character in gpu
                )
                lease = (lock_root / f"gpu-{safe_gpu}.lock").open("a+")
                logger.info(f"Waiting for GPU lease for model merge: {gpu}")
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
                leases.append(lease)
                acquired.append(gpu)
        logger.info(f"Acquired GPU lease for model merge: {','.join(acquired)} (requested={requested})")
        yield {
            "CUDA_VISIBLE_DEVICES": ",".join(acquired),
            "FT_REQUESTED_CUDA_VISIBLE_DEVICES": requested,
        }
    finally:
        for lease in reversed(leases):
            lease.close()


def is_blackwell_gpu() -> bool:
    """Check if the current GPU is NVIDIA Blackwell architecture (B100, B200, B300)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu_names = result.stdout.strip().lower()
            return any(kw in gpu_names for kw in BLACKWELL_GPU_KEYWORDS)
    except Exception:
        pass
    return False


def check_if_merging_needed(model_path: str | Path) -> bool:
    """
    Check if the model needs to be merged before benchmarking.
    Required for adapter features that vLLM cannot load directly, such as
    modules_to_save and DoRA.
    """
    config_path = Path(model_path) / "adapter_config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        config = json.load(f)
    # Check for modules_to_save which requires merging for vLLM
    # The logic is based in https://github.com/vllm-project/vllm/issues/9280
    if config.get("modules_to_save") is not None:
        logger.info(f"Model merging required due to modules_to_save: {config.get('modules_to_save')}")
        return True
    if config.get("use_dora") is True:
        logger.info("Model merging required because vLLM does not support DoRA adapters")
        return True
    if is_blackwell_gpu():
        logger.info("Model merging required due to Blackwell GPU (B100/B200/B300)")
        return True
    return False


def merge_model(env, workspace_path: Path, base_model_path: str, adapter_path: str, output_path: str):
    """
    Merge LoRA adapter into base model using a template-generated script.
    """
    # Prepare template variables
    template_vars = {
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "output_path": output_path,
    }

    # Render Jinja2 template
    merge_script = T("rdagent.scenarios.finetune.benchmark.merge.merge_model_template:template").r(**template_vars)

    script_path = workspace_path / "merge_model.py"
    script_path.write_text(merge_script)

    logger.info(f"Starting model merging from {adapter_path}...")

    ws_prefix = get_workspace_prefix(env)
    cmd = f"python {ws_prefix}/merge_model.py"

    with gpu_lease_environment() as lease_environment:
        result = env.run(
            cmd,
            local_path=str(workspace_path),
            env=lease_environment or None,
        )
    if result.exit_code != 0:
        raise RuntimeError(f"Model merging failed (exit_code={result.exit_code}):\n{result.stdout}")
    logger.info("Model merging completed.")
