#!/usr/bin/env python3
"""Run the paper's 7B and 3B base-model evaluations with the released evaluator."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

if __package__:
    from .collect_results import collect_and_write_run
    from .paper_matrix import PaperExperiment, base_experiments
    from .responses_adapter import responses_configuration_errors, routed_api_environment
    from .run_matrix import (
        ASSET_MANIFEST,
        FT_ROOT,
        ROOT,
        asset_target,
        benchmark_assets,
        benchmark_dataset_path,
        configure_project_environment,
        duration_seconds,
        marker_matches,
        safe_id,
        status_write,
        stop_process,
        validate_max_parallel,
        visible_gpus,
    )
else:
    from collect_results import collect_and_write_run
    from paper_matrix import PaperExperiment, base_experiments
    from responses_adapter import responses_configuration_errors, routed_api_environment
    from run_matrix import (
        ASSET_MANIFEST,
        FT_ROOT,
        ROOT,
        asset_target,
        benchmark_assets,
        benchmark_dataset_path,
        configure_project_environment,
        duration_seconds,
        marker_matches,
        safe_id,
        status_write,
        stop_process,
        validate_max_parallel,
        visible_gpus,
    )

OPENCOMPASS = FT_ROOT / "conda_envs" / "opencompass" / "bin" / "opencompass"
LOG_ROOT = FT_ROOT / "logs" / "paper-base"
SPLITS = (
    ("validation", "[:min(100, len(index_list)//2)]"),
    ("test", "[-min(100, len(index_list)//2):]"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("base-7b", "base-3b", "all"), default="all")
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU ids; default: all visible GPUs")
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum concurrent tasks")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Override vLLM's per-GPU memory fraction for base evaluation only; "
            "useful when safely co-scheduling with the FT matrix"
        ),
    )
    parser.add_argument("--only", default=None, help="Regex applied to experiment ids")
    parser.add_argument("--run-name", default=None, help="Stable output folder name, useful with --resume")
    parser.add_argument("--resume", action="store_true", help="Skip tasks with a successful status file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print selected tasks and run no evaluations")
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate selected tasks without querying GPUs or creating a run directory",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--task-timeout",
        default="12h",
        help="Operational guard covering both splits for one task; default: 12h",
    )
    return parser.parse_args()


def selected_experiments(suite: str) -> list[PaperExperiment]:
    groups = {"base-7b"} if suite == "base-7b" else {"base-3b"} if suite == "base-3b" else {"base-7b", "base-3b"}
    return [experiment for experiment in base_experiments() if experiment.paper_group in groups]


def validate_and_report_gpu_memory_utilization(value: float | None) -> None:
    if value is not None and not 0 < value <= 1:
        message = "--gpu-memory-utilization must be greater than 0 and at most 1"
        raise SystemExit(message)
    if value is not None:
        print(f"Base vLLM GPU memory utilization={value}")


def read_asset_manifest() -> dict[str, Any]:
    return json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))


def model_assets(
    experiments: list[PaperExperiment],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    requested = {experiment.target_model for experiment in experiments}
    result = {
        asset["name"]: asset for asset in manifest["assets"] if asset["kind"] == "model" and asset["name"] in requested
    }
    missing = requested - result.keys()
    if missing:
        message = f"No pinned model asset configured for: {', '.join(sorted(missing))}"
        raise ValueError(message)
    return result


def preflight(
    experiments: list[PaperExperiment],
    pinned_benchmarks: dict[str, dict[str, Any]],
    pinned_models: dict[str, dict[str, Any]],
) -> None:
    errors: list[str] = []
    if not OPENCOMPASS.is_file():
        errors.append(f"OpenCompass backend missing: {OPENCOMPASS}")
    errors.extend(
        f"asset not prepared: {asset['id']} ({asset_target(asset)})"
        for asset in {asset["id"]: asset for asset in pinned_benchmarks.values()}.values()
        if not marker_matches(asset)
    )
    errors.extend(
        f"asset not prepared: {asset['id']} ({asset_target(asset)})"
        for asset in pinned_models.values()
        if not marker_matches(asset)
    )
    judge_key = os.environ.get("FT_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if any(experiment.benchmark == "aime25" for experiment in experiments) and not judge_key:
        errors.append("AIME cascade evaluation requires FT_JUDGE_API_KEY or OPENAI_API_KEY for unresolved answers")
    errors.extend(responses_configuration_errors(os.environ))
    if errors:
        raise SystemExit("Preflight failed:\n- " + "\n- ".join(errors))


def render_config(
    experiment: PaperExperiment,
    benchmark_asset: dict[str, Any],
    model_asset: dict[str, Any],
    split_range: str,
    config_path: Path,
    split_work_dir: Path,
    gpu_memory_utilization: float | None,
) -> None:
    # Import after load_dotenv so RD-Agent settings see the reproduction environment.
    from rdagent.app.finetune.llm.conf import FT_RD_SETTING  # noqa: PLC0415
    from rdagent.scenarios.finetune.benchmark.benchmark import get_model_inference_config  # noqa: PLC0415
    from rdagent.scenarios.finetune.benchmark.data.adaptor import BENCHMARK_CONFIG_DICT  # noqa: PLC0415
    from rdagent.utils.agent.tpl import T  # noqa: PLC0415

    inference_config = get_model_inference_config(experiment.target_model, gpu_count=1)
    if gpu_memory_utilization is not None:
        inference_config["gpu_memory_utilization"] = gpu_memory_utilization
    if not FT_RD_SETTING.force_think_token:
        inference_config["use_cot_postprocessor"] = False
    dataset_path = FT_ROOT / "benchmarks" / benchmark_dataset_path(benchmark_asset)
    model_path = asset_target(model_asset)
    template_vars = {
        "model_abbr": f"base-{model_path.name.lower()}-{experiment.benchmark}",
        "model_path": str(model_path),
        "is_lora": False,
        "lora_path": "",
        "dataset_imports": [BENCHMARK_CONFIG_DICT[experiment.benchmark].dataset],
        "test_range": split_range,
        "num_runs": 1,
        "pass_k": None,
        "work_dir": str(split_work_dir),
        "dataset_path_literal": repr(str(dataset_path)),
        "test_range_literal": repr(split_range),
        **inference_config,
    }
    content = T("rdagent.scenarios.finetune.benchmark.configs.opencompass_template:template").r(**template_vars)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")


def read_summary(split_work_dir: Path) -> dict[str, Any]:
    timestamped_dirs = sorted((path for path in split_work_dir.glob("20*_*") if path.is_dir()), reverse=True)
    if not timestamped_dirs:
        message = f"OpenCompass produced no timestamped result directory below {split_work_dir}"
        raise RuntimeError(message)
    summary_files = sorted(timestamped_dirs[0].joinpath("summary").glob("*.csv"), reverse=True)
    if not summary_files:
        message = f"OpenCompass produced no summary CSV below {timestamped_dirs[0]}"
        raise RuntimeError(message)
    summary_path = summary_files[0]
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metadata_columns = {"dataset", "version", "metric", "mode"}
    score_columns = sorted({column for row in rows for column in row} - metadata_columns)
    if len(score_columns) != 1:
        message = f"Expected one OpenCompass score column in {summary_path}, found {score_columns}"
        raise RuntimeError(message)
    score_column = score_columns[0]
    unusable_datasets: list[str] = []
    for row in rows:
        dataset = str(row.get("dataset", "")).strip() or "<unknown>"
        metric = str(row.get("metric", "")).strip()
        score = str(row.get(score_column, "")).strip()
        try:
            numeric_score = float(score)
        except ValueError:
            unusable_datasets.append(dataset)
            continue
        if not math.isfinite(numeric_score) or not metric or metric == "-":
            unusable_datasets.append(dataset)
    if unusable_datasets:
        joined = ", ".join(sorted(set(unusable_datasets)))
        message = f"OpenCompass summary contains failed dataset rows: {joined}"
        raise RuntimeError(message)
    return {
        "timestamp_dir": str(timestamped_dirs[0]),
        "summary_csv": str(summary_path),
        "rows": rows,
    }


async def run_command(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: float,
) -> tuple[int | None, bool]:
    with log_path.open("ab", buffering=0) as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return await asyncio.wait_for(process.wait(), timeout=timeout), False
        except TimeoutError:
            await stop_process(process)
            return process.returncode, True
        except asyncio.CancelledError:
            await stop_process(process)
            raise


async def run_one(
    experiment: PaperExperiment,
    gpu: str,
    run_root: Path,
    benchmark_asset: dict[str, Any],
    model_asset: dict[str, Any],
    task_timeout: int,
    gpu_memory_utilization: float | None,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    task_root = run_root / safe_id(experiment.experiment_id)
    status_path = task_root / "status.json"
    task_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        **experiment.to_dict(),
        "gpu": gpu,
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "benchmark_asset": benchmark_asset["id"],
        "benchmark_dataset_path": benchmark_dataset_path(benchmark_asset),
        "model_asset": model_asset["id"],
        "task_timeout": task_timeout,
        "gpu_memory_utilization": gpu_memory_utilization,
        "api_routing": api_routing,
        "splits": {},
    }
    status_write(status_path, status)
    environment = runtime_environment.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["OC_JUDGE_MODEL"] = environment.get("FT_JUDGE_MODEL", "gpt-5")
    judge_key = environment.get("FT_JUDGE_API_KEY") or environment.get("OPENAI_API_KEY")
    judge_base = environment.get("FT_JUDGE_API_BASE") or environment.get("OPENAI_API_BASE")
    if judge_key:
        environment["OC_JUDGE_API_KEY"] = judge_key
    if judge_base:
        environment["OC_JUDGE_API_BASE"] = judge_base
    environment["OC_JUDGE_RETRY"] = environment.get("FT_JUDGE_RETRY", "10")
    deadline = asyncio.get_running_loop().time() + task_timeout
    print(f"START gpu={gpu} {experiment.experiment_id}", flush=True)
    try:
        for split, split_range in SPLITS:
            split_work_dir = task_root / split
            config_path = task_root / "configs" / f"{split}.py"
            render_config(
                experiment,
                benchmark_asset,
                model_asset,
                split_range,
                config_path,
                split_work_dir,
                gpu_memory_utilization,
            )
            command = [str(OPENCOMPASS), str(config_path), "--work-dir", str(split_work_dir)]
            status["splits"][split] = {
                "state": "running",
                "range": split_range,
                "config": str(config_path),
                "command": command,
            }
            status_write(status_path, status)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                message = "task timeout expired before starting split"
                raise TimeoutError(message)  # noqa: TRY301
            return_code, timed_out = await run_command(
                command,
                environment,
                task_root / f"{split}.console.log",
                remaining,
            )
            status["splits"][split].update(
                {
                    "state": "succeeded" if return_code == 0 and not timed_out else "failed",
                    "return_code": return_code,
                    "outer_timeout": timed_out,
                },
            )
            status_write(status_path, status)
            if return_code != 0 or timed_out:
                message = f"OpenCompass {split} failed with return code {return_code}"
                raise RuntimeError(message)  # noqa: TRY301
            status["splits"][split]["summary"] = read_summary(split_work_dir)
            status_write(status_path, status)
    except Exception as error:  # noqa: BLE001 - record any task failure before the worker continues.
        status.update(
            {
                "state": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        status_write(status_path, status)
        print(f"FAIL  gpu={gpu} {experiment.experiment_id}: {error}", flush=True)
        return False
    status.update(
        {
            "state": "succeeded",
            "finished_at": datetime.now().astimezone().isoformat(),
        },
    )
    status_write(status_path, status)
    print(f"DONE  gpu={gpu} {experiment.experiment_id}", flush=True)
    return True


async def run_workers(
    experiments: list[PaperExperiment],
    gpus: list[str],
    run_root: Path,
    pinned_benchmarks: dict[str, dict[str, Any]],
    pinned_models: dict[str, dict[str, Any]],
    task_timeout: int,
    gpu_memory_utilization: float | None,
    runtime_environment: dict[str, str],
    api_routing: dict[str, Any],
) -> bool:
    queue: asyncio.Queue[PaperExperiment] = asyncio.Queue()
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
                        pinned_benchmarks[experiment.benchmark],
                        pinned_models[experiment.target_model],
                        task_timeout,
                        gpu_memory_utilization,
                        runtime_environment,
                        api_routing,
                    ),
                )
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(gpu) for gpu in gpus))
    return all(results)


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    configure_project_environment()
    experiments = selected_experiments(args.suite)
    if args.only:
        pattern = re.compile(args.only)
        experiments = [experiment for experiment in experiments if pattern.search(experiment.experiment_id)]
    if not experiments:
        message = "No experiments selected"
        raise SystemExit(message)
    manifest_experiments = list(experiments)
    asset_manifest = read_asset_manifest()
    pinned_benchmarks = benchmark_assets(experiments, asset_manifest)
    pinned_models = model_assets(experiments, asset_manifest)
    validate_max_parallel(args.max_parallel)
    validate_and_report_gpu_memory_utilization(args.gpu_memory_utilization)
    task_timeout = duration_seconds(args.task_timeout)

    if args.preflight_only:
        preflight(experiments, pinned_benchmarks, pinned_models)
        print(f"Preflight OK: {len(experiments)} tasks")
        return 0

    run_name = args.run_name or datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = LOG_ROOT / run_name
    if args.resume:
        experiments = [
            experiment
            for experiment in experiments
            if not (
                (status_path := run_root / safe_id(experiment.experiment_id) / "status.json").is_file()
                and json.loads(status_path.read_text(encoding="utf-8")).get("state") == "succeeded"
            )
        ]
    if not experiments:
        print(f"Selected 0 tasks; GPUs=none; output={run_root}")
        if args.dry_run:
            return 0
        report = collect_and_write_run(run_root, "base")
        print(f"Report: {run_root / 'report'} ({report['coverage']['supplied_tasks']} tasks)")
        return 0
    gpus = visible_gpus(args.gpus)[: args.max_parallel]
    print(f"Selected {len(experiments)} tasks; GPUs={','.join(gpus)}; output={run_root}")
    for experiment in experiments:
        print(f"  {experiment.experiment_id}: {experiment.target_model}, validation+test")
    if args.dry_run:
        return 0
    if not args.skip_preflight:
        preflight(experiments, pinned_benchmarks, pinned_models)
    with routed_api_environment(os.environ) as (runtime_environment, api_routing):
        run_root.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "suite": args.suite,
            "task_timeout": args.task_timeout,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "api_routing": api_routing,
            "tasks": [experiment.to_dict() for experiment in manifest_experiments],
            "benchmark_assets": {
                benchmark: {
                    "id": asset["id"],
                    "revision": asset["revision"],
                    "dataset_path": benchmark_dataset_path(asset),
                }
                for benchmark, asset in sorted(pinned_benchmarks.items())
            },
            "model_assets": {
                name: {"id": asset["id"], "revision": asset["revision"]}
                for name, asset in sorted(pinned_models.items())
            },
        }
        (run_root / "matrix.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        try:
            success = asyncio.run(
                run_workers(
                    experiments,
                    gpus,
                    run_root,
                    pinned_benchmarks,
                    pinned_models,
                    task_timeout,
                    args.gpu_memory_utilization,
                    runtime_environment,
                    api_routing,
                ),
            )
        finally:
            report = collect_and_write_run(run_root, "base")
            print(f"Report: {run_root / 'report'} ({report['coverage']['supplied_tasks']} tasks)")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
