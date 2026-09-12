#!/usr/bin/env python3
# ruff: noqa: C901, EM101, EM102, PERF401, PERF403, PLR0912, PLR2004, TRY003
"""Run a controlled Full-SFT/LoRA/rsLoRA feasibility ablation on eight H20s."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_ROOT = Path(__file__).with_name("training_constraint_ablation")
FT_ROOT = ROOT / "finetune_files"
TRAINING_BIN = FT_ROOT / "conda_envs" / "llm_finetune" / "bin" / "llamafactory-cli"
DEFAULT_OUTPUT_ROOT = FT_ROOT / "logs" / "training-constraint-ablation"
MODEL_ROOT = FT_ROOT / "models" / "Qwen" / "Qwen2.5-7B-Instruct"
GPU_LOCK_ROOT = FT_ROOT / "gpu_leases" / "locks"
PAIR_METHODS = ("lora", "rslora")
METHOD_FIELD = "use_rslora"


@dataclass(frozen=True)
class DatasetSpec:
    benchmark: str
    template: str
    train_source: Path
    validation_source: Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    benchmark: str
    method: str
    gpus: tuple[str, ...]
    template: Path
    train_source: Path
    validation_source: Path


@dataclass
class ActiveRun:
    spec: RunSpec
    workspace: Path
    process: subprocess.Popen[str]
    log_handle: TextIO
    started_monotonic: float
    started_at: str
    gpu_samples: dict[str, list[dict[str, float]]]


DATASETS = (
    DatasetSpec(
        benchmark="chemcotbench_mol_und",
        template="chemcotbench_mol_und.yaml",
        train_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__chemcotbench_mol_und__run-1/workspace/2657b0f5e42e407bb6168d8cbcc243d5"
            / "data_train.json"
        ),
        validation_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__chemcotbench_mol_und__run-1/workspace/2657b0f5e42e407bb6168d8cbcc243d5"
            / "data_validation.json"
        ),
    ),
    DatasetSpec(
        benchmark="FinanceIQ_gen",
        template="FinanceIQ_gen.yaml",
        train_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__FinanceIQ_gen__run-1/workspace/7c293cc65d0548609b4854eb07492be6"
            / "data.json"
        ),
        validation_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__FinanceIQ_gen__run-1/workspace/7c293cc65d0548609b4854eb07492be6"
            / "internal_validation_alpaca.json"
        ),
    ),
    DatasetSpec(
        benchmark="tablebench_fact_checking",
        template="tablebench_fact_checking.yaml",
        train_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__tablebench_fact_checking__run-1/workspace/eb316a345fde4af5a4fce8a542668ddb"
            / "data_train_full.json"
        ),
        validation_source=(
            FT_ROOT
            / "logs/paper-matrix/h20-gpt56-main-rslora-48h-v10"
            / "main__tablebench_fact_checking__run-1/workspace/eb316a345fde4af5a4fce8a542668ddb"
            / "data_validation_full.json"
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Optional full-match regex for pre-filling a safe subset. Re-run without "
            "this option and with --resume to finish the complete controlled layout."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override both seed and data_seed for an independent controlled repeat.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise SystemExit("This controlled layout requires exactly eight distinct GPU ids")
    if any(not re.fullmatch(r"[0-9]+", gpu) for gpu in gpus):
        raise SystemExit("GPU ids must be non-negative integers")
    return gpus


def safe_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise SystemExit(f"Unsafe run name: {value!r}")
    return value


def build_specs(gpus: list[str]) -> list[RunSpec]:
    table = next(item for item in DATASETS if item.benchmark == "tablebench_fact_checking")
    specs = [
        RunSpec(
            run_id="tablebench_fact_checking__full_smoke",
            benchmark=table.benchmark,
            method="full",
            gpus=(gpus[0], gpus[1]),
            template=TEMPLATE_ROOT / "full_sft_smoke.yaml",
            train_source=table.train_source,
            validation_source=table.validation_source,
        ),
    ]
    for index, dataset in enumerate(DATASETS):
        for offset, method in enumerate(PAIR_METHODS):
            specs.append(
                RunSpec(
                    run_id=f"{dataset.benchmark}__{method}",
                    benchmark=dataset.benchmark,
                    method=method,
                    gpus=(gpus[2 + 2 * index + offset],),
                    template=TEMPLATE_ROOT / dataset.template,
                    train_source=dataset.train_source,
                    validation_source=dataset.validation_source,
                ),
            )
    return specs


def select_specs(specs: list[RunSpec], pattern: str | None) -> list[RunSpec]:
    if pattern is None:
        return specs
    try:
        matcher = re.compile(pattern)
    except re.error as error:
        raise SystemExit(f"Invalid --only regex: {error}") from error
    selected = [spec for spec in specs if matcher.fullmatch(spec.run_id)]
    if not selected:
        raise SystemExit(f"--only matched no controlled runs: {pattern}")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Controlled dataset is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, list) or not payload:
        raise SystemExit(f"Controlled dataset must be a non-empty JSON list: {path}")
    required = {"instruction", "input", "output"}
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or not required.issubset(record):
            raise SystemExit(f"Invalid Alpaca record {index} in {path}")
        if not str(record["instruction"]).strip() or not str(record["output"]).strip():
            raise SystemExit(f"Empty instruction/output at record {index} in {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "records": len(payload)}


def load_template(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Training template is not a mapping: {path}")
    return value


def rendered_config(spec: RunSpec, *, seed: int | None = None) -> dict[str, Any]:
    config = load_template(spec.template)
    config["model_name_or_path"] = str(MODEL_ROOT.resolve())
    if spec.method in PAIR_METHODS:
        config["finetuning_type"] = "lora"
        config[METHOD_FIELD] = spec.method == "rslora"
        config["use_dora"] = False
    elif spec.method == "full":
        if config.get("finetuning_type") != "full":
            raise SystemExit("Full-SFT template does not request finetuning_type: full")
        for field in tuple(config):
            if field.startswith("lora_") or field in {"use_rslora", "use_dora", "pissa_init"}:
                config.pop(field)
    config["dataset_dir"] = "./"
    config["tokenized_path"] = "./tokenized_cache"
    config["output_dir"] = "./output"
    if seed is not None:
        config["seed"] = seed
        config["data_seed"] = seed
    return config


def masked_pair_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result[METHOD_FIELD] = "<controlled-method-switch>"
    return result


def pair_contract_hash(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_masked = masked_pair_config(left)
    right_masked = masked_pair_config(right)
    if left_masked != right_masked:
        raise SystemExit("LoRA/rsLoRA pair differs in fields other than use_rslora")
    return canonical_hash(left_masked)


def link_exact(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise SystemExit(f"Existing controlled data differs from its source: {target}")
        return
    os.link(source, target)


def prepare_workspace(
    spec: RunSpec,
    run_root: Path,
    *,
    resume: bool,
    seed: int | None = None,
) -> dict[str, Any]:
    workspace = run_root / spec.run_id
    status_path = workspace / "status.json"
    if workspace.exists() and not resume:
        raise SystemExit(f"Controlled workspace already exists; use --resume or a new run name: {workspace}")
    if status_path.is_file() and read_json(status_path).get("state") == "succeeded":
        return read_json(status_path)
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"Incomplete workspace cannot be overwritten; use a new run name: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)

    link_exact(spec.train_source, workspace / "data_train.json")
    link_exact(spec.validation_source, workspace / "data_validation.json")
    dataset_info = {
        "processed_data": {
            "file_name": "data_train.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "processed_data_validation": {
            "file_name": "data_validation.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    write_json(workspace / "dataset_info.json", dataset_info)
    config = rendered_config(spec, seed=seed)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    (workspace / "train.requested.yaml").write_text(rendered, encoding="utf-8")
    (workspace / "train.yaml").write_text(rendered, encoding="utf-8")
    status = {
        "state": "prepared",
        "run_id": spec.run_id,
        "benchmark": spec.benchmark,
        "method": spec.method,
        "gpus": list(spec.gpus),
        "config_sha256": canonical_hash(config),
        "seed": config.get("seed"),
        "data_seed": config.get("data_seed"),
        "train_data": validate_dataset(spec.train_source),
        "validation_data": validate_dataset(spec.validation_source),
        "prepared_at": utc_now(),
    }
    write_json(status_path, status)
    return status


def build_manifest(
    specs: list[RunSpec],
    run_root: Path,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    rendered = {spec.run_id: rendered_config(spec, seed=seed) for spec in specs}
    pair_contracts: dict[str, str] = {}
    for dataset in DATASETS:
        pair_contracts[dataset.benchmark] = pair_contract_hash(
            rendered[f"{dataset.benchmark}__lora"],
            rendered[f"{dataset.benchmark}__rslora"],
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "seed_override": seed,
        "run_root": str(run_root.resolve()),
        "model": str(MODEL_ROOT.resolve()),
        "model_config_sha256": sha256_file(MODEL_ROOT / "config.json"),
        "training_binary": str(TRAINING_BIN.resolve()),
        "pair_contracts": pair_contracts,
        "runs": [
            {
                "run_id": spec.run_id,
                "benchmark": spec.benchmark,
                "method": spec.method,
                "gpus": list(spec.gpus),
                "template": str(spec.template.resolve()),
                "train_source": str(spec.train_source.resolve()),
                "validation_source": str(spec.validation_source.resolve()),
            }
            for spec in specs
        ],
    }


def process_environment(spec: RunSpec, run_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(spec.gpus),
            "FT_EXPERIMENT_ID": f"constraint-ablation/{spec.run_id}",
            "FT_TRAINING_POLICY": spec.method,
            "FT_FULL_SFT_GPUS": "2",
            "FT_GPU_LEASE_LOCK_ROOT": str(GPU_LOCK_ROOT),
            "FT_GPU_LEASE_POOL_FILE": str(run_root / ".disabled-dynamic-pool"),
            "FT_GPU_MEMORY_READY_FRACTION": "0.95",
            "FT_GPU_MEMORY_READY_TIMEOUT": "300",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    return environment


def launch(spec: RunSpec, run_root: Path) -> ActiveRun:
    workspace = run_root / spec.run_id
    log_handle = (workspace / "train.log").open("w", encoding="utf-8")
    started_at = utc_now()
    status = read_json(workspace / "status.json")
    status.update({"state": "running", "started_at": started_at})
    write_json(workspace / "status.json", status)
    process = subprocess.Popen(  # noqa: S603
        [str(TRAINING_BIN), "train", "train.yaml"],
        cwd=workspace,
        env=process_environment(spec, run_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return ActiveRun(
        spec=spec,
        workspace=workspace,
        process=process,
        log_handle=log_handle,
        started_monotonic=time.monotonic(),
        started_at=started_at,
        gpu_samples={gpu: [] for gpu in spec.gpus},
    )


def gpu_snapshot() -> dict[str, dict[str, float]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
    if result.returncode != 0:
        return {}
    snapshots: dict[str, dict[str, float]] = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 4:
            continue
        try:
            snapshots[row[0].strip()] = {
                "memory_mib": float(row[1].strip()),
                "utilization_percent": float(row[2].strip()),
                "power_watts": float(row[3].strip()),
            }
        except ValueError:
            continue
    return snapshots


def summarize_gpu_samples(samples: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for gpu, rows in samples.items():
        result[gpu] = {
            "samples": len(rows),
            "peak_memory_mib": max((row["memory_mib"] for row in rows), default=None),
            "peak_utilization_percent": max((row["utilization_percent"] for row in rows), default=None),
            "mean_utilization_percent": (sum(row["utilization_percent"] for row in rows) / len(rows) if rows else None),
            "mean_power_watts": sum(row["power_watts"] for row in rows) / len(rows) if rows else None,
        }
    return result


def numeric_metrics(output: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for name in ("all_results.json", "train_results.json", "eval_results.json"):
        path = output / name
        if not path.is_file():
            continue
        payload = read_json(path)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = value
    state_path = output / "trainer_state.json"
    if state_path.is_file():
        state = read_json(state_path)
        for key in ("global_step", "max_steps", "epoch", "best_metric"):
            value = state.get(key) if isinstance(state, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
        history = state.get("log_history", []) if isinstance(state, dict) else []
        eval_losses = [row["eval_loss"] for row in history if isinstance(row, dict) and "eval_loss" in row]
        if eval_losses:
            metrics["last_eval_loss"] = eval_losses[-1]
            metrics["min_eval_loss"] = min(eval_losses)
    return metrics


def artifact_evidence(output: Path, method: str) -> dict[str, Any]:
    adapter_config = output / "adapter_config.json"
    adapter_weights = any((output / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    full_weights = any(output.glob("model*.safetensors")) or any(output.glob("pytorch_model*.bin"))
    evidence: dict[str, Any] = {
        "path": str(output.resolve()),
        "adapter_config": adapter_config.is_file(),
        "adapter_weights": adapter_weights,
        "full_weights": full_weights,
    }
    if adapter_config.is_file():
        config = read_json(adapter_config)
        evidence["use_rslora"] = config.get("use_rslora")
        evidence["use_dora"] = config.get("use_dora")
        evidence["rank"] = config.get("r")
    if method == "full":
        evidence["valid"] = bool((output / "config.json").is_file() and full_weights and not adapter_config.exists())
    else:
        evidence["valid"] = bool(
            adapter_weights
            and evidence.get("use_dora") is False
            and evidence.get("use_rslora") is (method == "rslora"),
        )
    return evidence


def finalize(active: ActiveRun, returncode: int) -> dict[str, Any]:
    active.log_handle.close()
    finished_at = utc_now()
    duration = time.monotonic() - active.started_monotonic
    output = active.workspace / "output"
    metrics = numeric_metrics(output) if output.is_dir() else {}
    artifact = artifact_evidence(output, active.spec.method)
    log_text = (active.workspace / "train.log").read_text(encoding="utf-8", errors="replace")
    launched_config = load_template(active.workspace / "train.yaml")
    zero3_configured = active.spec.method == "full" and bool(launched_config.get("deepspeed"))
    zero3_log_evidence = "Configured ZeRO-3 torchrun" in log_text
    succeeded = returncode == 0 and artifact["valid"]
    if active.spec.method == "full":
        succeeded = succeeded and zero3_configured and zero3_log_evidence
    status = read_json(active.workspace / "status.json")
    status.update(
        {
            "state": "succeeded" if succeeded else "failed",
            "returncode": returncode,
            "finished_at": finished_at,
            "duration_seconds": duration,
            "metrics": metrics,
            "gpu": summarize_gpu_samples(active.gpu_samples),
            "artifact": artifact,
            "zero3_configured": zero3_configured,
            "zero3_log_evidence": zero3_log_evidence,
        },
    )
    write_json(active.workspace / "status.json", status)
    return status


def monitor(active: list[ActiveRun], poll_interval: float) -> None:
    if poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")
    pending = list(active)
    try:
        while pending:
            snapshot = gpu_snapshot()
            for item in pending:
                for gpu in item.spec.gpus:
                    if gpu in snapshot:
                        item.gpu_samples[gpu].append(snapshot[gpu])
            finished: list[ActiveRun] = []
            for item in pending:
                returncode = item.process.poll()
                if returncode is not None:
                    finalize(item, returncode)
                    finished.append(item)
            for item in finished:
                pending.remove(item)
            if pending:
                time.sleep(poll_interval)
    except BaseException:
        for item in pending:
            if item.process.poll() is None:
                os.killpg(item.process.pid, signal.SIGTERM)
        for item in pending:
            try:
                item.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(item.process.pid, signal.SIGKILL)
            item.log_handle.close()
        raise


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def primary_gpu_value(status: dict[str, Any], key: str, *, aggregate: str = "max") -> float | None:
    values = [row.get(key) for row in status.get("gpu", {}).values() if row.get(key) is not None]
    if not values:
        return None
    return max(values) if aggregate == "max" else sum(values) / len(values)


def collect_results(specs: list[RunSpec], run_root: Path) -> dict[str, Any]:
    statuses = {spec.run_id: read_json(run_root / spec.run_id / "status.json") for spec in specs}
    pairs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ordinary = statuses[f"{dataset.benchmark}__lora"]
        scaled = statuses[f"{dataset.benchmark}__rslora"]
        ordinary_eval = ordinary.get("metrics", {}).get("min_eval_loss")
        scaled_eval = scaled.get("metrics", {}).get("min_eval_loss")
        pairs.append(
            {
                "benchmark": dataset.benchmark,
                "rank": ordinary.get("artifact", {}).get("rank"),
                "lora_state": ordinary.get("state"),
                "rslora_state": scaled.get("state"),
                "lora_train_loss": ordinary.get("metrics", {}).get("train_loss"),
                "rslora_train_loss": scaled.get("metrics", {}).get("train_loss"),
                "lora_eval_loss": ordinary_eval,
                "rslora_eval_loss": scaled_eval,
                "eval_loss_delta_rslora_minus_lora": (
                    scaled_eval - ordinary_eval
                    if isinstance(scaled_eval, (int, float)) and isinstance(ordinary_eval, (int, float))
                    else None
                ),
                "lora_duration_seconds": ordinary.get("duration_seconds"),
                "rslora_duration_seconds": scaled.get("duration_seconds"),
                "lora_peak_memory_mib": primary_gpu_value(ordinary, "peak_memory_mib"),
                "rslora_peak_memory_mib": primary_gpu_value(scaled, "peak_memory_mib"),
            },
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_root": str(run_root.resolve()),
        "all_succeeded": all(status.get("state") == "succeeded" for status in statuses.values()),
        "runs": statuses,
        "pairs": pairs,
    }


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# H20 training-constraint ablation",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "## Runs",
        "",
        "| Task | Method | GPUs | State | Steps | Train loss | Min eval loss "
        "| Peak/GPU GiB | Mean util. | Runtime min | Artifact |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for status in results["runs"].values():
        metrics = status.get("metrics", {})
        peak_mib = primary_gpu_value(status, "peak_memory_mib")
        mean_util = primary_gpu_value(status, "mean_utilization_percent", aggregate="mean")
        lines.append(
            "| "
            + " | ".join(
                (
                    status["benchmark"],
                    status["method"],
                    ",".join(status["gpus"]),
                    status["state"],
                    fmt(metrics.get("global_step"), 0),
                    fmt(metrics.get("train_loss")),
                    fmt(metrics.get("min_eval_loss")),
                    fmt(peak_mib / 1024 if peak_mib is not None else None, 2),
                    fmt(mean_util, 1),
                    fmt(status.get("duration_seconds", 0) / 60, 1),
                    "yes" if status.get("artifact", {}).get("valid") else "no",
                ),
            )
            + " |",
        )
    lines.extend(
        [
            "",
            "## Paired method isolation",
            "",
            "A positive eval-loss delta means rsLoRA was worse under the byte-identical paired contract.",
            "",
            "| Task | Rank | LoRA eval | rsLoRA eval | rsLoRA - LoRA | LoRA GiB | rsLoRA GiB | LoRA min | rsLoRA min |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for pair in results["pairs"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    pair["benchmark"],
                    fmt(pair["rank"], 0),
                    fmt(pair["lora_eval_loss"]),
                    fmt(pair["rslora_eval_loss"]),
                    fmt(pair["eval_loss_delta_rslora_minus_lora"]),
                    fmt(pair["lora_peak_memory_mib"] / 1024 if pair["lora_peak_memory_mib"] else None, 2),
                    fmt(pair["rslora_peak_memory_mib"] / 1024 if pair["rslora_peak_memory_mib"] else None, 2),
                    fmt(pair["lora_duration_seconds"] / 60 if pair["lora_duration_seconds"] else None, 1),
                    fmt(pair["rslora_duration_seconds"] / 60 if pair["rslora_duration_seconds"] else None, 1),
                ),
            )
            + " |",
        )
    full = results["runs"]["tablebench_fact_checking__full_smoke"]
    full_peak_mib = primary_gpu_value(full, "peak_memory_mib")
    full_peak_gib = full_peak_mib / 1024 if full_peak_mib else None
    lines.extend(
        [
            "",
            "## Full-SFT feasibility contract",
            "",
            "| Check | Result |",
            "| --- | --- |",
            f"| ZeRO-3 injected | {full.get('zero3_configured', False)} |",
            f"| Distributed torchrun evidence | {full.get('zero3_log_evidence', False)} |",
            f"| Complete full-model artifact | {full.get('artifact', {}).get('valid', False)} |",
            f"| Peak memory per H20 | {fmt(full_peak_gib, 2)} GiB |",
            "",
        ],
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    run_name = safe_run_name(args.run_name)
    gpus = parse_gpus(args.gpus)
    all_specs = build_specs(gpus)
    specs = select_specs(all_specs, args.only)
    if not TRAINING_BIN.is_file():
        raise SystemExit(f"Training backend is missing: {TRAINING_BIN}")
    if not MODEL_ROOT.is_dir():
        raise SystemExit(f"Base model is missing: {MODEL_ROOT}")
    for spec in all_specs:
        validate_dataset(spec.train_source)
        validate_dataset(spec.validation_source)
        rendered_config(spec, seed=args.seed)
    manifest = build_manifest(all_specs, args.output_root / run_name, seed=args.seed)
    if args.preflight_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    run_root = args.output_root / run_name
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        comparable_keys = ("model", "model_config_sha256", "training_binary", "pair_contracts", "runs")
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise SystemExit(f"Existing manifest differs: {manifest_path}")
    else:
        write_json(manifest_path, manifest)
    active: list[ActiveRun] = []
    for spec in specs:
        status = prepare_workspace(spec, run_root, resume=args.resume, seed=args.seed)
        if status.get("state") != "succeeded":
            active.append(launch(spec, run_root))
    if active:
        monitor(active, args.poll_interval)
    if len(specs) != len(all_specs):
        statuses = {spec.run_id: read_json(run_root / spec.run_id / "status.json") for spec in specs}
        partial_results = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "run_root": str(run_root.resolve()),
            "selected_run_ids": [spec.run_id for spec in specs],
            "all_selected_succeeded": all(status.get("state") == "succeeded" for status in statuses.values()),
            "runs": statuses,
        }
        write_json(run_root / "prefill_results.json", partial_results)
        print(json.dumps(partial_results, indent=2, ensure_ascii=False))
        return 0 if partial_results["all_selected_succeeded"] else 1
    results = collect_results(all_specs, run_root)
    write_json(run_root / "results.json", results)
    (run_root / "RESULTS.md").write_text(render_markdown(results), encoding="utf-8")
    print(render_markdown(results))
    return 0 if results["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
