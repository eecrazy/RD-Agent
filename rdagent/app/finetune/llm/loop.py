"""
LLM Fine-tuning Entry Point

Standard RDLoop entry point for LLM fine-tuning, consistent with data science implementation.
"""

import asyncio
import fcntl
import functools
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import fire
from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.scenarios.finetune.loop import LLMFinetuneRDLoop

REPRODUCTION_API_MAX_RETRY = 30
RESUME_TIMER_BUDGET_ENV = "FT_RESUME_TIMER_BUDGET"
PIPELINE_CLAIM_ENV = "FT_PIPELINE_CLAIM"
PIPELINE_CLAIM_FILE = ".pipeline-claim.json"
PIPELINE_CLAIM_LOCK = ".pipeline-claim.lock"
P = ParamSpec("P")
R = TypeVar("R")


def _claim_paths() -> tuple[Path, Path] | None:
    trace_path = os.getenv("LOG_TRACE_PATH")
    if not trace_path:
        return None
    task_root = Path(trace_path).expanduser().resolve().parent
    return task_root / PIPELINE_CLAIM_FILE, task_root / PIPELINE_CLAIM_LOCK


def _write_claim(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_claim(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Pipeline claim is unreadable: {path}: {error}") from error  # noqa: EM102, TRY003
    if not isinstance(payload, dict) or payload.get("state") not in {"running", "succeeded", "failed"}:
        raise RuntimeError(f"Pipeline claim has an invalid state: {path}")  # noqa: EM102, TRY003
    return payload


@contextmanager
def pipeline_claim() -> Iterator[bool]:
    """Serialize an explicitly overflowed task and let the original queue adopt it.

    A second matrix scheduler may opt a not-yet-started task into this protocol by
    setting ``FT_PIPELINE_CLAIM=1``.  The original scheduler was already running
    before overflow was enabled, so its later child discovers the per-task marker,
    waits for the first attempt, and reuses only a successful terminal state.
    Ordinary matrix tasks and isolated retry directories remain untouched.
    """
    paths = _claim_paths()
    if paths is None:
        yield True
        return
    claim_path, lock_path = paths
    enabled = os.getenv(PIPELINE_CLAIM_ENV) == "1" or claim_path.exists() or lock_path.exists()
    if not enabled:
        yield True
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if claim_path.exists():
            claim = _read_claim(claim_path)
            if claim["state"] == "succeeded":
                logger.info(f"Reusing successful overflow pipeline: {claim_path.parent}")
                yield False
                return
            if claim["state"] == "failed":
                message = f"Overflow pipeline already failed; retry it in a fresh run directory: {claim_path.parent}"
                raise RuntimeError(message)
            claim.update(
                {
                    "state": "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": "The prior overflow owner exited without a terminal claim",
                },
            )
            _write_claim(claim_path, claim)
            raise RuntimeError(f"Overflow pipeline owner disappeared: {claim_path.parent}")  # noqa: EM102, TRY003

        claim = {
            "schema_version": 1,
            "experiment_id": os.getenv("FT_EXPERIMENT_ID"),
            "state": "running",
            "owner_pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        _write_claim(claim_path, claim)
        try:
            yield True
        except BaseException as error:
            claim.update(
                {
                    "state": "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            _write_claim(claim_path, claim)
            raise
        else:
            claim.update(
                {
                    "state": "succeeded",
                    "finished_at": datetime.now().astimezone().isoformat(),
                },
            )
            _write_claim(claim_path, claim)


def use_pipeline_claim(function: Callable[P, R]) -> Callable[P, R | None]:
    """Wrap the Fire entrypoint without changing its inspected signature."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R | None:
        with pipeline_claim() as should_run:
            if not should_run:
                return None
            return function(*args, **kwargs)

    return wrapped


def reset_resumed_timer(loop: LLMFinetuneRDLoop, path: str | None) -> None:
    """Replace a snapshot's serialized timer for an explicitly budgeted resume."""
    resume_budget = os.getenv(RESUME_TIMER_BUDGET_ENV)
    if not resume_budget:
        return
    if path is None:
        raise RuntimeError(f"{RESUME_TIMER_BUDGET_ENV} requires a resumed --path")  # noqa: EM102, TRY003
    loop.timer.reset(resume_budget)
    logger.info(f"Reset resumed workflow timer to explicit remaining budget: {resume_budget}")


@use_pipeline_claim
def main(
    path: str | None = None,
    checkout: bool = True,  # noqa: FBT001, FBT002
    user_target_scenario: str | None = None,
    benchmark: str | None = None,
    benchmark_description: str | None = None,
    dataset: str | None = None,
    base_model: str | None = None,
    upper_data_size_limit: int | None = None,
    step_n: int | None = None,
    loop_n: int | None = None,
    timeout: str | None = None,
) -> None:
    """
    LLM fine-tuning entry point

    Parameters
    ----------
    path :
        A path like `$LOG_PATH/__session__/1/0_propose`. This indicates that we restore the state
        after finishing step 0 in loop 1.
    checkout :
        Used to control the log session path. Boolean type, default is True.
        - If True, the new loop will use the existing folder and clear logs for sessions after the
          one corresponding to the given path.
        - If False, the new loop will use the existing folder but keep the logs for sessions after
          the one corresponding to the given path.
    dataset : str
        Dataset name for fine-tuning (e.g., 'shibing624/alpaca-zh')
    base_model : str, optional
        Model name for fine-tuning (e.g., 'Qwen/Qwen2.5-1.5B-Instruct').
        If not provided, auto-selects optimal model based on hardware and dataset.
    step_n : int, optional
        Number of steps to run; if None, runs indefinitely until completion or error
    loop_n : int, optional
        Number of loops to run; if None, runs indefinitely until completion or error
    timeout : str, optional
        Maximum duration for the entire process

    Examples:
    .. code-block:: bash
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh \
            --base-model Qwen/Qwen2.5-1.5B-Instruct
        dotenv run -- python rdagent/app/finetune/llm/loop.py --dataset shibing624/alpaca-zh    # TODO: not enabled yet
    """

    # Matrix experiments can spend hours producing and validating a candidate
    # before a planner call.  Do not discard that work because the routed API
    # has a short transient outage: keep the generic RD-Agent default for
    # interactive runs, but give explicitly identified reproduction tasks a
    # larger, still-bounded retry window.
    if os.getenv("FT_EXPERIMENT_ID"):
        LLM_SETTINGS.max_retry = max(LLM_SETTINGS.max_retry, REPRODUCTION_API_MAX_RETRY)

    if user_target_scenario:
        FT_RD_SETTING.user_target_scenario = user_target_scenario
    assert (
        FT_RD_SETTING.user_target_scenario is None
    ), "user_target_scenario is not yet supported, please specify via benchmark and benchmark_description"
    if upper_data_size_limit:
        FT_RD_SETTING.upper_data_size_limit = upper_data_size_limit
        logger.info(f"Set upper_data_size_limit to {FT_RD_SETTING.upper_data_size_limit}")
    if benchmark:
        FT_RD_SETTING.target_benchmark = benchmark
    if benchmark_description:
        FT_RD_SETTING.benchmark_description = benchmark_description
    assert FT_RD_SETTING.user_target_scenario or (
        FT_RD_SETTING.target_benchmark and FT_RD_SETTING.benchmark_description
    ), "Either user_target_scenario or target_benchmark must be specified for LLM fine-tuning."

    # Update configuration with provided parameters
    if dataset:
        FT_RD_SETTING.dataset = dataset
    if base_model:
        FT_RD_SETTING.base_model = base_model

    # Create and run LLM fine-tuning loop
    data_set_target = FT_RD_SETTING.dataset or "auto generated dataset"
    model_target = FT_RD_SETTING.base_model or "auto selected model"

    # Temporary assertion until auto-selection is implemented
    assert (
        FT_RD_SETTING.base_model is not None
    ), "Base model auto selection not yet supported, please specify via --base-model"

    logger.info(f"Starting LLM fine-tuning on dataset='{data_set_target}' with model='{model_target}'")

    if path is None:
        loop = LLMFinetuneRDLoop(FT_RD_SETTING)
    else:
        loop = cast("LLMFinetuneRDLoop", LLMFinetuneRDLoop.load(str(path), checkout=checkout))

    reset_resumed_timer(loop, path)

    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=timeout))


if __name__ == "__main__":
    fire.Fire(main)
