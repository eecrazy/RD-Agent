#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Build a machine-readable and human-readable audit of a terminal FT-Dojo run.

The audit is deliberately read-only with respect to experiment artifacts.  It
resolves the canonical matrix symlinks, hashes the small data/SFT evidence,
validates the signed validation selection, and verifies that the one-shot
held-out result refers to that exact selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rdagent.scenarios.finetune.train.formal_training import (
    FormalTrainingEvidenceError,
    scan_durable_formal_training_evidence,
    validate_formal_training_method_lock,
)

if __package__:
    from .collect_results import PAPER_METRICS
    from .final_test_protocol import (
        MODEL_WEIGHT_PATTERNS,
        has_model_weights,
        is_policy_compliant_model,
        is_policy_compliant_selection,
        model_artifact_type,
        normalize_training_policy,
    )
    from .validation_selection import validate_selection_artifact
else:
    from collect_results import PAPER_METRICS
    from final_test_protocol import (
        MODEL_WEIGHT_PATTERNS,
        has_model_weights,
        is_policy_compliant_model,
        is_policy_compliant_selection,
        model_artifact_type,
        normalize_training_policy,
    )
    from validation_selection import validate_selection_artifact

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TASKS = 39
EXPECTED_BASE_TASKS = 13
EXPECTED_RUNS_PER_TASK = 3
EXPECTED_TRAINING_SAMPLES = 2000
EXPECTED_TIMEOUT = "12h"
EXPECTED_GPUS = tuple(str(index) for index in range(8))
EXPECTED_MAX_PARALLEL = len(EXPECTED_GPUS)
EXPECTED_SUPERVISOR_STAGE = "complete"
EXPECTED_LOGICAL_GPU_MEMORY_GB = 178.0
MAX_SUPPORTING_AUDIT_DEPTH = 3
AUDIT_NAME_MARKERS = ("audit", "manifest", "metadata", "report", "spec", "stats", "summary")
AUDIT_DISCOVERY_EXCLUDED_DIRS = {
    ".ft_model_checkpoints",
    "__pycache__",
    "benchmark_results",
    "output",
    "tokenized_cache",
}


class ExperimentAuditError(RuntimeError):
    """Raised when terminal evidence is missing or internally inconsistent."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ExperimentAuditError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--supervisor-state", type=Path, required=True)
    parser.add_argument("--supervisor-log", type=Path, required=True)
    parser.add_argument("--final-test-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path, *, object_required: bool = True) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Cannot read JSON {path}: {type(error).__name__}: {error}"
        raise ExperimentAuditError(message) from error
    if object_required:
        require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def artifact_record(path: Path, *, canonical_view_path: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"Required artifact is missing: {path}")
    resolved = path.resolve()
    return {
        "canonical_view_path": canonical_view_path or project_relative(path),
        "resolved_path": project_relative(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _safe_workspace_file(workspace: Path, value: object, *, label: str) -> Path:
    require(isinstance(value, str) and bool(value), f"Missing {label}")
    relative = Path(value)
    require(
        not relative.is_absolute() and len(relative.parts) == 1 and relative.name == value,
        f"Unsafe {label}: {value!r}",
    )
    return workspace / relative


def _verify_declared_hashes(
    workspace: Path,
    hashes: object,
    *,
    label: str,
) -> dict[str, str]:
    require(isinstance(hashes, dict) and bool(hashes), f"Missing {label} hashes")
    verified: dict[str, str] = {}
    for name, expected_digest in sorted(hashes.items()):
        path = _safe_workspace_file(workspace, name, label=f"{label} artifact name")
        require(
            isinstance(expected_digest, str) and re.fullmatch(r"[0-9a-f]{64}", expected_digest) is not None,
            f"Invalid {label} SHA-256 for {name!r}",
        )
        require(path.is_file(), f"Declared {label} artifact is missing: {path}")
        actual_digest = sha256_file(path)
        require(actual_digest == expected_digest, f"Declared {label} artifact changed: {path}")
        verified[str(name)] = actual_digest
    return verified


def financeiq_pipeline_evidence(  # noqa: PLR0915 - explicit audit checks are intentionally linear.
    task_root: Path,
    workspace: Path,
    *,
    expected_training_samples: int = EXPECTED_TRAINING_SAMPLES,
) -> dict[str, Any]:
    """Resolve and validate FinanceIQ's durable data-generation audit.

    A formal-training rescue copies the immutable training inputs into a new
    workspace, but deliberately does not copy the source ``audit.json``.  The
    rescue manifest is the durable link back to that source workspace and
    records hashes for both sides of the copy.  Follow that link instead of
    assuming a transient ``manifest.json`` exists in the selected workspace.
    """
    workspace = workspace.resolve()
    workspace_root = (task_root / "workspace").resolve()
    require(workspace.parent == workspace_root, f"Workspace is outside the task root: {workspace}")

    lineage_path = workspace / "formal_rescue_manifest.json"
    lineage: dict[str, Any] | None = None
    source_workspace = workspace
    input_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    if lineage_path.is_file():
        lineage = load_json(lineage_path)
        require(lineage.get("workspace_id") == workspace.name, "FinanceIQ rescue workspace id mismatch")
        source_workspace_id = lineage.get("source_workspace_id")
        require(
            isinstance(source_workspace_id, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", source_workspace_id) is not None,
            "FinanceIQ rescue source workspace id is unsafe",
        )
        source_workspace = (workspace_root / source_workspace_id).resolve()
        require(
            source_workspace.parent == workspace_root,
            f"FinanceIQ rescue source is outside the task root: {source_workspace}",
        )
        input_hashes = _verify_declared_hashes(
            workspace,
            lineage.get("input_hashes"),
            label="FinanceIQ rescue input",
        )
        source_hashes = _verify_declared_hashes(
            source_workspace,
            lineage.get("source_hashes"),
            label="FinanceIQ rescue source",
        )

        # The audit describes the source generation.  These immutable files
        # must be byte-identical in the workspace that was actually trained.
        for name in ("process_data.py", "data.json", "validation.json", "data_stats.json", "train.yaml"):
            require(
                input_hashes.get(name) == source_hashes.get(name),
                f"FinanceIQ rescue lineage differs for {name}",
            )

    audit_path = source_workspace / "audit.json"
    audit = load_json(audit_path)
    require(audit.get("mode") == "full", f"FinanceIQ processing audit is not a full run: {audit_path}")

    artifact_paths = audit.get("artifact_paths")
    require(isinstance(artifact_paths, dict), f"FinanceIQ audit has no artifact paths: {audit_path}")
    training_name = lineage.get("training_file") if lineage else artifact_paths.get("training")
    validation_name = lineage.get("validation_file") if lineage else artifact_paths.get("validation")
    training_path = _safe_workspace_file(workspace, training_name, label="FinanceIQ training artifact")
    validation_path = _safe_workspace_file(workspace, validation_name, label="FinanceIQ validation artifact")
    training = load_json(training_path, object_required=False)
    validation = load_json(validation_path, object_required=False)
    require(isinstance(training, list), f"FinanceIQ training artifact is not a list: {training_path}")
    require(isinstance(validation, list), f"FinanceIQ validation artifact is not a list: {validation_path}")

    contract = audit.get("formal_contract")
    require(isinstance(contract, dict) and contract.get("passed") is True, "FinanceIQ formal contract failed")
    require(
        len(training) == expected_training_samples
        and contract.get("actual_training_records") == len(training)
        and contract.get("expected_training_records") == expected_training_samples,
        "FinanceIQ processing audit does not prove the exact training count",
    )
    require(
        len(validation) > 0 and contract.get("actual_validation_records") == len(validation),
        "FinanceIQ processing audit validation count mismatch",
    )

    training_counts = audit.get("training_counts")
    validation_counts = audit.get("validation_counts")
    require(
        isinstance(training_counts, dict) and sum(training_counts.values()) == len(training),
        "FinanceIQ per-subject training counts do not match the artifact",
    )
    require(
        isinstance(validation_counts, dict) and sum(validation_counts.values()) == len(validation),
        "FinanceIQ per-subject validation counts do not match the artifact",
    )

    overlap = audit.get("partition_overlap_assertions")
    require(isinstance(overlap, dict) and overlap.get("passed") is True, "FinanceIQ partition audit failed")
    overlap_keys = {
        "train_validation": "train_validation_source_overlap",
        "train_holdout": "train_benchmark_source_overlap",
        "validation_holdout": "validation_benchmark_source_overlap",
    }
    pairwise_intersection_counts = {name: overlap.get(key) for name, key in overlap_keys.items()}
    require(
        all(value == 0 for value in pairwise_intersection_counts.values()),
        f"FinanceIQ source partitions overlap: {pairwise_intersection_counts}",
    )

    benchmark_partition = audit.get("benchmark_partition")
    require(
        isinstance(benchmark_partition, dict)
        and benchmark_partition.get("excluded_from_training_and_validation") is True,
        "FinanceIQ benchmark tail is not proven isolated",
    )
    holdout_count = benchmark_partition.get("total_tail_rows")
    require(isinstance(holdout_count, int) and holdout_count > 0, "FinanceIQ benchmark-tail count is invalid")

    raw_source_count = audit.get("raw_source_count")
    require(isinstance(raw_source_count, int) and raw_source_count > 0, "FinanceIQ raw-source count is invalid")
    return {
        "processing_audit": artifact_record(audit_path),
        "audit_source_workspace_id": source_workspace.name,
        "lineage_manifest": artifact_record(lineage_path) if lineage is not None else None,
        "lineage_hashes": {
            "verified_input_count": len(input_hashes),
            "verified_source_count": len(source_hashes),
            "input": input_hashes,
            "source": source_hashes,
        },
        "pre_llm_sample_count": raw_source_count,
        "accepted_training_count": len(training),
        "validation_count": len(validation),
        "holdout_count": holdout_count,
        "pairwise_intersection_counts": pairwise_intersection_counts,
    }


def weight_records(model_path: Path) -> list[dict[str, Any]]:
    files: set[Path] = set()
    for pattern in MODEL_WEIGHT_PATTERNS:
        files.update(path.resolve() for path in model_path.glob(pattern) if path.is_file())
    return [
        {
            "resolved_path": project_relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": None,
            "digest_policy": "existence-and-size-only; selection signature also binds size and mtime",
        }
        for path in sorted(files, key=str)
    ]


def safe_id(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", experiment_id).strip("_")


def matrix_provenance_record(
    matrix_run: Path,
    matrix_tasks: object,
) -> dict[str, Any]:
    """Record root provenance for composed and directly executed matrices.

    A composed canonical view has a dedicated ``provenance.json`` describing
    its source runs and overrides.  ``run_matrix.py`` materializes task roots
    directly and intentionally writes only ``matrix.json``.  In that direct
    case, the immutable matrix specification is the root provenance artifact;
    the per-task evidence below supplies the artifact-level chain.

    Refuse the fallback when any task root is a symbolic link.  Such a view is
    composed and must retain its explicit composition manifest rather than
    silently presenting itself as a direct run.
    """
    provenance_path = matrix_run / "provenance.json"
    if provenance_path.is_file():
        return {
            "kind": "composition_manifest",
            **artifact_record(provenance_path),
        }

    require(isinstance(matrix_tasks, list) and bool(matrix_tasks), "Matrix task inventory is missing")
    for task in matrix_tasks:
        require(isinstance(task, dict), "Malformed matrix task inventory")
        experiment_id = task.get("experiment_id")
        require(isinstance(experiment_id, str) and bool(experiment_id), "Matrix task has no experiment ID")
        task_root = matrix_run / safe_id(experiment_id)
        require(task_root.is_dir(), f"Direct matrix task root is missing: {task_root}")
        require(
            not task_root.is_symlink(),
            "Composed matrix is missing its required provenance.json: " + str(matrix_run),
        )

    return {
        "kind": "direct_matrix_manifest",
        "materialized_task_roots": len(matrix_tasks),
        **artifact_record(matrix_run / "matrix.json"),
    }


def canonical_prefix(matrix_run: Path, experiment_id: str) -> str:
    return f"{project_relative(matrix_run)}/{safe_id(experiment_id)}"


def canonical_workspace_prefix(matrix_run: Path, experiment_id: str, workspace_id: str) -> str:
    return f"{canonical_prefix(matrix_run, experiment_id)}/workspace/{workspace_id}"


def discover_supporting_audits(workspace: Path, canonical_workspace: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*"), key=str):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        if len(relative.parts) > MAX_SUPPORTING_AUDIT_DEPTH or set(relative.parts[:-1]) & AUDIT_DISCOVERY_EXCLUDED_DIRS:
            continue
        name = relative.name.lower()
        if not name.endswith((".json", ".sha256")):
            continue
        if not any(marker in name for marker in AUDIT_NAME_MARKERS):
            continue
        if relative.as_posix() in {"data_stats.json", "dataset_info.json"}:
            continue
        records.append(
            artifact_record(path, canonical_view_path=f"{canonical_workspace}/{relative.as_posix()}"),
        )
    return records


def parse_final_test_log(path: Path) -> dict[str, Any]:
    start_pattern = re.compile(r"^START gpu=(\S+) (main/.+?) final-test$")
    end_pattern = re.compile(r"^(DONE|FAIL)\s+gpu=(\S+) (main/.+?) final-test$")
    summary_pattern = re.compile(r"^Final tests(?: ready)?: (\d+) reused, (\d+) evaluated$")
    attempts: dict[str, list[dict[str, Any]]] = {}
    active: dict[str, str] = {}
    peak_parallelism = 0
    summaries: list[dict[str, int]] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        start = start_pattern.match(line)
        if start:
            gpu, experiment_id = start.groups()
            require(experiment_id not in active, f"Duplicate active final-test start: {experiment_id}")
            attempt = {"gpu": gpu, "start_line": line_number, "outcome": None, "end_line": None}
            attempts.setdefault(experiment_id, []).append(attempt)
            active[experiment_id] = gpu
            peak_parallelism = max(peak_parallelism, len(active))
            continue
        end = end_pattern.match(line)
        if end:
            outcome, gpu, experiment_id = end.groups()
            require(active.get(experiment_id) == gpu, f"Unmatched final-test completion: {experiment_id}")
            attempt = attempts[experiment_id][-1]
            attempt["outcome"] = outcome.lower()
            attempt["end_line"] = line_number
            del active[experiment_id]
            continue
        summary = summary_pattern.match(line)
        if summary:
            reused, evaluated = (int(value) for value in summary.groups())
            summaries.append({"reused": reused, "evaluated": evaluated})

    require(not active, f"Final-test log ends with active tasks: {sorted(active)}")
    require(summaries, "Final-test summary line is missing")
    return {
        "attempts": attempts,
        "peak_parallelism": peak_parallelism,
        "reused": sum(item["reused"] for item in summaries),
        "evaluated": sum(item["evaluated"] for item in summaries),
        "invocations": summaries,
    }


def parse_supervisor_commit(path: Path) -> dict[str, Any]:
    pattern = re.compile(r"COMMIT one-shot held-out audit .* --gpus (\S+) --max-parallel (\d+)(?:\s|$)")
    matches = [pattern.search(line) for line in path.read_text(encoding="utf-8").splitlines()]
    found = [match for match in matches if match]
    require(len(found) == 1, f"Expected one held-out COMMIT line, found {len(found)}")
    gpu_text, max_parallel = found[0].groups()
    return {"gpus": gpu_text.split(","), "max_parallel": int(max_parallel)}


def stable_api_route(route: object) -> dict[str, Any]:
    """Remove only the ephemeral per-process adapter listener from a recorded route."""
    require(isinstance(route, dict), "API routing evidence is not an object")
    return {key: value for key, value in route.items() if key != "adapter_base"}


def metric_map(task: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    view = task.get("final", {}).get(split)
    require(isinstance(view, dict), f"Missing {split} view: {task.get('paper_experiment_id')}")
    metrics = view.get("paper_metrics")
    require(isinstance(metrics, list), f"Missing {split} metrics: {task.get('paper_experiment_id')}")
    result: dict[str, dict[str, Any]] = {}
    for item in metrics:
        require(isinstance(item, dict) and isinstance(item.get("metric"), str), "Malformed paper metric")
        value = item.get("value")
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            "Paper metric is not finite",
        )
        result[str(item["metric"])] = item
    return result


def validation_utilities(tasks: list[dict[str, Any]], benchmark: str) -> dict[str, float]:
    """Rank candidates without accessing their held-out metric values."""
    require(tasks, f"No operationally available runs for {benchmark}")
    expected_metrics = PAPER_METRICS[benchmark]
    utilities = {str(task["paper_experiment_id"]): [] for task in tasks}
    for metric in expected_metrics:
        records = [(task, metric_map(task, "validation")[metric]) for task in tasks]
        directions = {bool(item["higher_is_better"]) for _, item in records}
        require(len(directions) == 1, f"Inconsistent validation metric direction: {benchmark}/{metric}")
        higher_is_better = directions.pop()
        values = [float(item["value"]) for _, item in records]
        low, high = min(values), max(values)
        for (task, _), value in zip(records, values, strict=True):
            if high == low:
                utility = 1.0
            elif higher_is_better:
                utility = (value - low) / (high - low)
            else:
                utility = (high - value) / (high - low)
            utilities[str(task["paper_experiment_id"])].append(utility)
    return {experiment_id: sum(parts) / len(parts) for experiment_id, parts in utilities.items()}


def select_best_successful_run(tasks: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    benchmark_tasks = [task for task in tasks if task.get("benchmark") == benchmark]
    available = [task for task in benchmark_tasks if task.get("final_test", {}).get("state") == "succeeded"]
    excluded = [task for task in benchmark_tasks if task not in available]
    utilities = validation_utilities(available, benchmark)
    selected = min(
        available,
        key=lambda task: (
            -utilities[str(task["paper_experiment_id"])],
            int(task.get("run_index") or 0),
            str(task["paper_experiment_id"]),
        ),
    )
    # Held-out values are intentionally read only after the validation-only
    # choice above has been frozen in ``selected``.
    return {
        "benchmark": benchmark,
        "view_kind": "post-hoc operational-availability supplement",
        "replaces_primary_table": False,
        "availability_filter": "final_test.state == succeeded; no held-out metric value is used",
        "ranking": "direction-aware, equal-weight min-max utility over validation metrics only",
        "selected_experiment_id": selected["paper_experiment_id"],
        "selected_run_index": selected["run_index"],
        "validation_utility": utilities[str(selected["paper_experiment_id"])],
        "validation_metrics": list(metric_map(selected, "validation").values()),
        "held_out_metrics_display_only": list(metric_map(selected, "test").values()),
        "excluded_operationally_unavailable": [
            {
                "experiment_id": task["paper_experiment_id"],
                "run_index": task["run_index"],
                "final_test_state": task.get("final_test", {}).get("state"),
                "validation_metrics": list(metric_map(task, "validation").values()),
            }
            for task in excluded
        ],
    }


def optional_artifact_record(path: Path, *, canonical_view_path: str | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return artifact_record(path, canonical_view_path=canonical_view_path)


def training_method_from_config(config: dict[str, Any]) -> str | None:
    finetuning_type = str(config.get("finetuning_type", "")).strip().lower()
    if finetuning_type == "full":
        return "full"
    if finetuning_type == "lora":
        return "rslora" if config.get("use_rslora") is True else "lora"
    return None


def method_choice_checks(
    hypothesis: str,
    reason: str,
    *,
    locked_method: str,
    training_resource: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Check that the initial hypothesis chose under the logical paper envelope.

    The formal method lock proves what was executed.  These deliberately
    conservative text checks add the missing causal evidence: the method was
    already named in the step-0 hypothesis, rsLoRA was explicitly excluded,
    and that hypothesis saw one logical 178 GB B200 rather than the physical
    H20 worker assigned later by the scheduler.
    """
    text = f"{hypothesis}\n{reason}"
    normalized = text.lower().replace("‑", "-").replace("–", "-")
    if locked_method == "lora":
        without_other_lora_variants = re.sub(r"(?:rs|q)lora|use_rslora", "", normalized)
        locked_method_named = re.search(r"\blora\b", without_other_lora_variants) is not None
    elif locked_method == "full":
        locked_method_named = re.search(r"\bfull(?:[- ]parameter|\s+sft)\b", normalized) is not None
    else:
        locked_method_named = False

    rslora_disabled = re.search(
        r"(?:\bno\b|\bwithout\b|\bdisable(?:d)?\b|\bdisabl(?:e|ed|ing)\b|"
        r"\bexclude(?:d)?\b)[^.]{0,100}\brslora\b|"
        r"\brslora\b[^.]{0,100}(?:\bdisabled\b|\bfalse\b|\bexcluded\b|"
        r"\bmust remain\b|\bremain disabled\b)|"
        r"\buse_rslora\s*[:=]\s*(?:false|0)\b",
        normalized,
    ) is not None
    resource_text = json.dumps(training_resource, sort_keys=True).lower()
    checks = {
        "locked_method_named_in_initial_hypothesis": locked_method_named,
        "rslora_explicitly_excluded": rslora_disabled,
        "logical_resource_override_recorded": training_resource.get("source") == "logical_resource_override",
        "logical_single_gpu": training_resource.get("gpu_count") == 1,
        "logical_gpu_is_b200": training_resource.get("gpu_name") == "NVIDIA B200",
        "logical_memory_is_178gb": training_resource.get("memory_per_gpu_gb")
        == EXPECTED_LOGICAL_GPU_MEMORY_GB
        and training_resource.get("total_memory_gb") == EXPECTED_LOGICAL_GPU_MEMORY_GB,
        "physical_h20_not_exposed_to_hypothesis": "h20" not in resource_text,
    }
    rationale = {
        "two_thousand_sample_overfit_or_forgetting_risk": (
            ("2000" in normalized or "2,000" in normalized)
            and re.search(
                r"overfitt|catastrophic|forgetting|capability (?:drift|regression)|"
                r"instruction[- ]forgetting|instruction[- ]format drift|destructive",
                normalized,
            )
            is not None
        ),
        "full_sft_context_or_memory_tradeoff_discussed": (
            re.search(r"\bfull(?:[- ]parameter|\s+sft)\b", normalized) is not None
            and re.search(r"context|token|cutoff|window|memory", normalized) is not None
        ),
    }
    return checks, rationale


def relevant_method_excerpt(hypothesis: str, reason: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", f"{hypothesis} {reason}")
    selected = [
        sentence.strip()
        for sentence in sentences
        if re.search(r"(?i)\b(?:lora|rslora)\b|full[- ]parameter|full\s+sft|overfitt|catastrophic", sentence)
    ]
    return " ".join(selected[:6])


def autonomous_method_selection_evidence(
    task_root: Path,
    *,
    locked_method: str,
) -> dict[str, Any]:
    trace_path = task_root / "trace" / "__session__" / "0" / "0_direct_exp_gen"
    require(trace_path.is_file(), f"Initial direct-hypothesis trace is missing: {task_root}")
    try:
        with trace_path.open("rb") as stream:
            loop = pickle.load(stream)
        direct_experiment = loop.loop_prev_out[0]["direct_exp_gen"]
        hypothesis_object = direct_experiment.hypothesis
        hypothesis = str(hypothesis_object.hypothesis)
        reason = str(hypothesis_object.reason)
        training_resource = loop.hypothesis_gen.scen.training_resource
    except (AttributeError, KeyError, OSError, pickle.PickleError, TypeError) as error:
        message = f"Cannot extract initial method-selection evidence from {trace_path}: {error}"
        raise ExperimentAuditError(message) from error
    require(isinstance(training_resource, dict), f"Training resource is not a mapping: {trace_path}")
    checks, rationale = method_choice_checks(
        hypothesis,
        reason,
        locked_method=locked_method,
        training_resource=training_resource,
    )
    require(all(checks.values()), f"Non-autonomous or mismatched method-selection evidence: {task_root}: {checks}")
    return {
        "source_stage": "trace/__session__/0/0_direct_exp_gen",
        "source_trace": artifact_record(trace_path),
        "locked_method": locked_method,
        "hypothesis_sha256": hashlib.sha256(hypothesis.encode()).hexdigest(),
        "reason_sha256": hashlib.sha256(reason.encode()).hexdigest(),
        "method_rationale_excerpt": relevant_method_excerpt(hypothesis, reason),
        "training_resource": training_resource,
        "rationale": rationale,
        "checks": checks,
    }


def strict_formal_training_evidence(
    task_root: Path,
    *,
    experiment_id: str,
    training_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently revalidate the exact-sample/full-epoch training contract."""
    records, errors = scan_durable_formal_training_evidence(
        task_root,
        expected_samples=EXPECTED_TRAINING_SAMPLES,
        experiment_id=experiment_id,
        training_policy=training_policy,
        require_visible_evidence=True,
    )
    require(not errors, f"Invalid durable formal-training evidence: {experiment_id}: {errors}")
    require(records, f"No durable formal-training evidence: {experiment_id}")
    methods = {str(record.get("training_method", "")) for record in records}
    require(
        len(methods) == 1 and "" not in methods,
        f"Formal training methods disagree within task: {experiment_id}: {sorted(methods)}",
    )
    method = methods.pop()
    try:
        method_lock = validate_formal_training_method_lock(
            task_root / "formal_training_method.json",
            experiment_id=experiment_id,
            training_policy=training_policy,
            training_method=method,
        )
    except (FormalTrainingEvidenceError, OSError) as error:
        message = f"Invalid formal training method lock: {experiment_id}: {error}"
        raise ExperimentAuditError(message) from error
    return records, method_lock


def selected_model_evidence(
    model_path: Path,
    *,
    candidate_source: str = "trained",
    training_policy: str = "paper",
) -> dict[str, Any]:
    adapter_path = model_path / "adapter_config.json"
    trainer_state_path = model_path / "trainer_state.json"
    adapter = load_json(adapter_path) if adapter_path.is_file() else None
    trainer_state = load_json(trainer_state_path) if trainer_state_path.is_file() else None
    weights = weight_records(model_path)
    artifact_type = model_artifact_type(model_path)
    selected_method = "baseline" if candidate_source == "baseline" else artifact_type
    return {
        "resolved_path": project_relative(model_path),
        "candidate_source": candidate_source,
        "training_method": selected_method,
        "artifact_type": artifact_type,
        "adapter_config": optional_artifact_record(adapter_path),
        "trainer_state": optional_artifact_record(trainer_state_path),
        "weights": weights,
        "peft_type": adapter.get("peft_type") if adapter else None,
        "use_rslora": adapter.get("use_rslora") if adapter else None,
        "use_dora": adapter.get("use_dora", False) if adapter else False,
        "global_step": trainer_state.get("global_step") if trainer_state else None,
        "epoch": trainer_state.get("epoch") if trainer_state else None,
        "has_model_weights": has_model_weights(model_path),
        "is_policy_compliant": is_policy_compliant_selection(
            model_path,
            candidate_source,
            training_policy,
        ),
    }


def task_evidence(  # noqa: PLR0915
    *,
    matrix_run: Path,
    matrix_task: dict[str, Any],
    attempt_map: dict[str, list[dict[str, Any]]],
    report_task: dict[str, Any],
    training_policy: str = "paper",
) -> dict[str, Any]:
    experiment_id = str(matrix_task["experiment_id"])
    task_root = matrix_run / safe_id(experiment_id)
    resolved_task_root = task_root.resolve()
    prefix = canonical_prefix(matrix_run, experiment_id)
    status = load_json(task_root / "status.json")
    selection_artifact = load_json(task_root / "validation_selection.json")
    selection = validate_selection_artifact(
        selection_artifact,
        experiment_id=experiment_id,
        benchmark=str(matrix_task["benchmark"]),
        model=str(matrix_task["model"]),
    )
    training_policy = normalize_training_policy(training_policy)
    status_policy = normalize_training_policy(status.get("training_policy", training_policy))
    require(
        status_policy == training_policy,
        f"Status/matrix training-policy mismatch: {experiment_id}: {status_policy} != {training_policy}",
    )
    require(
        status.get("formal_expected_samples") == EXPECTED_TRAINING_SAMPLES,
        f"Status does not require exactly {EXPECTED_TRAINING_SAMPLES} training samples: {experiment_id}",
    )
    formal_records, method_lock = strict_formal_training_evidence(
        task_root,
        experiment_id=experiment_id,
        training_policy=training_policy,
    )
    locked_method = str(method_lock["training_method"])
    require(
        status.get("formal_training_method") == locked_method,
        f"Status/formal method-lock mismatch: {experiment_id}",
    )
    require(
        status.get("formal_training_method_lock") == method_lock,
        f"Status/formal method-lock artifact mismatch: {experiment_id}",
    )
    method_selection = autonomous_method_selection_evidence(
        task_root,
        locked_method=locked_method,
    )
    workspace_value = selection.get("workspace_id")
    if not isinstance(workspace_value, str) or not workspace_value:
        workspace_value = next(
            (
                candidate.get("workspace_id")
                for candidate in selection_artifact.get("candidates", [])
                if isinstance(candidate, dict)
                and candidate.get("source") != "baseline"
                and isinstance(candidate.get("workspace_id"), str)
                and candidate.get("workspace_id")
            ),
            None,
        )
    require(isinstance(workspace_value, str) and bool(workspace_value), f"No trained workspace: {experiment_id}")
    workspace_id = workspace_value
    formal_by_workspace = {str(record["workspace_id"]): record for record in formal_records}
    require(
        workspace_id in formal_by_workspace,
        f"Selected trained workspace has no strict formal evidence: {experiment_id}/{workspace_id}",
    )
    formal_record = formal_by_workspace[workspace_id]
    workspace = (task_root / "workspace" / workspace_id).resolve()
    workspace_prefix = canonical_workspace_prefix(matrix_run, experiment_id, workspace_id)

    process_path = workspace / "process_data.py"
    data_stats_path = workspace / "data_stats.json"
    dataset_info_path = workspace / "dataset_info.json"
    train_path = workspace / "train.yaml"
    output = workspace / "output"
    trainer_state_path = output / "trainer_state.json"
    adapter_path = output / "adapter_config.json"
    data_stats = load_json(data_stats_path)
    trainer_state = load_json(trainer_state_path)
    adapter = load_json(adapter_path) if adapter_path.is_file() else None
    try:
        train_config = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        message = f"Cannot read SFT config {train_path}: {error}"
        raise ExperimentAuditError(message) from error
    require(isinstance(train_config, dict), f"SFT config is not a mapping: {train_path}")

    total_samples = data_stats.get("total_samples")
    require(
        isinstance(total_samples, int)
        and not isinstance(total_samples, bool)
        and total_samples == EXPECTED_TRAINING_SAMPLES,
        f"data_stats.total_samples is not exactly {EXPECTED_TRAINING_SAMPLES}: {experiment_id}",
    )
    weights = weight_records(output)
    training_method = training_method_from_config(train_config)
    output_artifact_type = model_artifact_type(output)
    sft_checks = {
        "stage_is_sft": train_config.get("stage") == "sft",
        "training_enabled": train_config.get("do_train") is True,
        "training_method_supported": training_method in {"full", "lora", "rslora"},
        "config_not_dora": train_config.get("use_dora", False) is False,
        "config_bf16": train_config.get("bf16") is True,
        "trainer_advanced": isinstance(trainer_state.get("global_step"), int) and trainer_state["global_step"] > 0,
        "formal_evidence_revalidated": formal_record.get("expected_samples") == EXPECTED_TRAINING_SAMPLES,
        "complete_epoch_schedule": formal_record.get("global_step") == formal_record.get("max_steps"),
        "config_matches_method_lock": training_method == locked_method,
        "artifact_matches_training_method": output_artifact_type == training_method,
        "artifact_matches_policy": is_policy_compliant_model(output, training_policy),
        "weights_present": bool(weights),
    }
    if training_method == "full":
        sft_checks.update(
            {
                "full_has_no_rslora_setting": "use_rslora" not in train_config,
                "full_has_no_adapter": adapter is None,
            },
        )
    else:
        expected_rslora = training_method == "rslora"
        sft_checks.update(
            {
                "adapter_config_present": isinstance(adapter, dict),
                "adapter_is_lora": isinstance(adapter, dict)
                and str(adapter.get("peft_type", "")).upper() == "LORA",
                "config_rslora_matches_method": (train_config.get("use_rslora", False) is expected_rslora),
                "adapter_rslora_matches_method": isinstance(adapter, dict)
                and adapter.get("use_rslora", False) is expected_rslora,
                "adapter_not_dora": isinstance(adapter, dict) and adapter.get("use_dora", False) is False,
            },
        )
    require(all(sft_checks.values()), f"Non-compliant SFT evidence: {experiment_id}: {sft_checks}")

    final_spec = load_json(task_root / "final_test_spec.json")
    final_result = load_json(task_root / "final_test.json")
    selected_model = Path(str(selection["model_path"])).resolve()
    candidate_source = str(selection.get("source") or "trained")
    selected_evidence = selected_model_evidence(
        selected_model,
        candidate_source=candidate_source,
        training_policy=training_policy,
    )
    require(selected_evidence["has_model_weights"], f"Selected model weights are missing: {experiment_id}")
    require(selected_evidence["is_policy_compliant"], f"Selected model violates policy: {experiment_id}")

    artifact_signature = selection_artifact.get("artifact_signature")
    selection_signature = selection.get("selection_signature")
    result_selection = final_result.get("selection", {})
    signature_checks = {
        "selection_artifact_valid": True,
        "spec_selection_artifact_matches": final_spec.get("selection_artifact_signature") == artifact_signature,
        "spec_selection_signature_matches": final_spec.get("selection_signature") == selection_signature,
        "spec_model_path_matches": Path(str(final_spec.get("model_path", ""))).resolve() == selected_model,
        "result_selection_artifact_matches": (
            result_selection.get("validation_selection_artifact_signature") == artifact_signature
        ),
        "result_selection_signature_matches": result_selection.get("signature") == selection_signature,
        "result_model_path_matches": Path(str(result_selection.get("model_path", ""))).resolve() == selected_model,
        "result_is_post_selection": final_result.get("evaluation_mode") == "post_selection",
    }
    require(
        all(signature_checks.values()),
        f"Selection/final-test signature mismatch: {experiment_id}: {signature_checks}",
    )

    attempts = attempt_map.get(experiment_id, [])
    require(len(attempts) == 1, f"Expected exactly one held-out attempt for {experiment_id}, found {len(attempts)}")
    attempt = attempts[0]
    expected_outcome = "done" if final_result.get("state") == "succeeded" else "fail"
    require(attempt.get("outcome") == expected_outcome, f"Log/artifact final-test state mismatch: {experiment_id}")
    require(
        report_task.get("final_test", {}).get("state") == final_result.get("state"),
        f"Report state mismatch: {experiment_id}",
    )

    status_route = status.get("api_routing")
    result_route = final_result.get("api_routing")
    stable_status_route = stable_api_route(status_route)
    stable_result_route = stable_api_route(result_route)
    require(stable_status_route == stable_result_route, f"Search/final-test stable API route mismatch: {experiment_id}")
    require(status.get("state") == "succeeded", f"Search did not succeed: {experiment_id}")
    require(status.get("timeout") == EXPECTED_TIMEOUT, f"Unexpected timeout: {experiment_id}")
    require(
        matrix_task.get("timeout") == EXPECTED_TIMEOUT,
        f"Matrix timeout is not {EXPECTED_TIMEOUT}: {experiment_id}",
    )

    return {
        "experiment_id": experiment_id,
        "benchmark": matrix_task["benchmark"],
        "run_index": int(experiment_id.rsplit("-", maxsplit=1)[-1]),
        "canonical_task_root": prefix,
        "resolved_task_root": project_relative(resolved_task_root),
        "search": {
            "state": status.get("state"),
            "timeout": status.get("timeout"),
            "gpu": status.get("gpu"),
            "training_policy": training_policy,
            "started_at": status.get("started_at"),
            "finished_at": status.get("finished_at"),
            "status": artifact_record(task_root / "status.json", canonical_view_path=f"{prefix}/status.json"),
        },
        "api_routing": stable_status_route,
        "method_selection": method_selection,
        "runtime_adapter_bases": {
            "search": status_route.get("adapter_base"),
            "final_test": result_route.get("adapter_base"),
        },
        "data_pipeline": {
            "workspace_id": workspace_id,
            "canonical_workspace": workspace_prefix,
            "resolved_workspace": project_relative(workspace),
            "process_data": artifact_record(
                process_path,
                canonical_view_path=f"{workspace_prefix}/process_data.py",
            ),
            "data_stats": {
                "artifact": artifact_record(
                    data_stats_path,
                    canonical_view_path=f"{workspace_prefix}/data_stats.json",
                ),
                "total_samples": total_samples,
            },
            "dataset_info": artifact_record(
                dataset_info_path,
                canonical_view_path=f"{workspace_prefix}/dataset_info.json",
            ),
            "supporting_audits": discover_supporting_audits(workspace, workspace_prefix),
            "checks": {
                "process_script_present": True,
                "exact_training_sample_count": total_samples == EXPECTED_TRAINING_SAMPLES,
                "dataset_registration_present": True,
            },
        },
        "sft": {
            "training_policy": training_policy,
            "formal_expected_samples": EXPECTED_TRAINING_SAMPLES,
            "formal_training_method_lock": method_lock,
            "formal_training_records": formal_records,
            "training_method": training_method,
            "artifact_type": output_artifact_type,
            "train_config": artifact_record(train_path, canonical_view_path=f"{workspace_prefix}/train.yaml"),
            "trainer_state": artifact_record(
                trainer_state_path,
                canonical_view_path=f"{workspace_prefix}/output/trainer_state.json",
            ),
            "adapter_config": optional_artifact_record(
                adapter_path,
                canonical_view_path=f"{workspace_prefix}/output/adapter_config.json",
            ),
            "weights": weights,
            "configuration": {
                key: train_config.get(key)
                for key in (
                    "stage",
                    "finetuning_type",
                    "do_train",
                    "use_rslora",
                    "use_dora",
                    "bf16",
                    "num_train_epochs",
                    "max_steps",
                    "lora_rank",
                    "lora_alpha",
                    "learning_rate",
                    "seed",
                )
            },
            "completion": {
                "global_step": trainer_state.get("global_step"),
                "epoch": trainer_state.get("epoch"),
            },
            "adapter": {
                key: adapter.get(key)
                for key in ("peft_type", "r", "lora_alpha", "use_rslora", "use_dora", "base_model_name_or_path")
            }
            if adapter
            else None,
            "checks": sft_checks,
        },
        "validation_selection": {
            "artifact": artifact_record(
                task_root / "validation_selection.json",
                canonical_view_path=f"{prefix}/validation_selection.json",
            ),
            "artifact_signature": artifact_signature,
            "held_out_test_used": selection_artifact.get("held_out_test_used"),
            "candidate_count": selection_artifact.get("candidate_count"),
            "candidate_id": selection.get("candidate_id"),
            "checkpoint_step": selection.get("checkpoint_step"),
            "selection_signature": selection_signature,
            "selected_model": selected_evidence,
        },
        "final_test": {
            "state": final_result.get("state"),
            "attempt": attempt,
            "spec": artifact_record(
                task_root / "final_test_spec.json",
                canonical_view_path=f"{prefix}/final_test_spec.json",
            ),
            "result": artifact_record(task_root / "final_test.json", canonical_view_path=f"{prefix}/final_test.json"),
            "selection_signature": result_selection.get("signature"),
            "selection_artifact_signature": result_selection.get("validation_selection_artifact_signature"),
            "started_at": final_result.get("started_at"),
            "finished_at": final_result.get("finished_at"),
            "error": final_result.get("error"),
            "checks": signature_checks,
        },
    }


def build_manifest(
    *,
    matrix_run: Path,
    results_path: Path,
    supervisor_state_path: Path,
    supervisor_log_path: Path,
    final_test_log_path: Path,
) -> dict[str, Any]:
    matrix = load_json(matrix_run / "matrix.json")
    results = load_json(results_path)
    supervisor_state = load_json(supervisor_state_path)
    execution = parse_final_test_log(final_test_log_path)
    commit = parse_supervisor_commit(supervisor_log_path)
    matrix_tasks = matrix.get("tasks")
    training_policy = normalize_training_policy(matrix.get("training_policy", "paper"))
    require(training_policy == "paper", "Main reproduction must use the paper method-selection policy")
    require(isinstance(matrix_tasks, list) and len(matrix_tasks) == EXPECTED_TASKS, "Expected a 39-task matrix")

    ft_report_tasks = [task for task in results.get("tasks", []) if task.get("paper_group") == "ft-agent-main"]
    base_report_tasks = [task for task in results.get("tasks", []) if task.get("paper_group") == "base-7b"]
    require(len(ft_report_tasks) == EXPECTED_TASKS, "Expected 39 FT report tasks")
    require(len(base_report_tasks) == EXPECTED_BASE_TASKS, "Expected 13 Base-7B report tasks")
    report_by_id = {
        str(task["paper_experiment_id"]).removeprefix("ft-agent/"): task for task in ft_report_tasks
    }

    tasks = [
        task_evidence(
            matrix_run=matrix_run,
            matrix_task=matrix_task,
            attempt_map=execution["attempts"],
            report_task=report_by_id[str(matrix_task["experiment_id"])],
            training_policy=training_policy,
        )
        for matrix_task in matrix_tasks
    ]
    states: dict[str, int] = {}
    for task in tasks:
        state = str(task["final_test"]["state"])
        states[state] = states.get(state, 0) + 1
    training_method_counts: dict[str, int] = {}
    for task in tasks:
        method = str(task["sft"]["training_method"])
        training_method_counts[method] = training_method_counts.get(method, 0) + 1
    method_rationale_counts = {
        key: sum(task["method_selection"]["rationale"][key] for task in tasks)
        for key in next(iter(tasks))["method_selection"]["rationale"]
    }

    require(states == {"succeeded": EXPECTED_TASKS}, f"Unexpected final-test states: {states}")
    require(execution["evaluated"] == EXPECTED_TASKS, "Held-out log does not prove exactly 39 evaluations")
    require(execution["peak_parallelism"] == len(EXPECTED_GPUS), "Observed held-out peak parallelism is not eight")
    observed_gpus = sorted(
        {attempt[0]["gpu"] for attempt in execution["attempts"].values()},
        key=int,
    )
    require(tuple(observed_gpus) == EXPECTED_GPUS, "Held-out execution did not use every requested GPU")
    require(
        tuple(commit["gpus"]) == EXPECTED_GPUS and commit["max_parallel"] == EXPECTED_MAX_PARALLEL,
        "Commit did not request all eight GPUs",
    )
    require(
        supervisor_state.get("stage") == EXPECTED_SUPERVISOR_STAGE,
        f"Supervisor did not reach {EXPECTED_SUPERVISOR_STAGE!r}",
    )
    require(supervisor_state.get("return_code") == 0, "Supervisor did not exit successfully")

    routes = {json.dumps(task["api_routing"], sort_keys=True) for task in tasks}
    require(len(routes) == 1, "Tasks do not share one sanitized API route")
    route = json.loads(routes.pop())
    require(route.get("upstream_base") == "http://127.0.0.1:8313/v1", "Unexpected API upstream")
    require(route.get("served_model") == "gpt-5.6-sol", "Unexpected served model")

    finance = next(task for task in tasks if task["experiment_id"] == "main/FinanceIQ_gen/run-1")
    finance_task_root = matrix_run / safe_id(finance["experiment_id"])
    finance_workspace = Path(finance["data_pipeline"]["resolved_workspace"])
    if not finance_workspace.is_absolute():
        finance_workspace = ROOT / finance_workspace
    pipeline = financeiq_pipeline_evidence(finance_task_root, finance_workspace)
    representative = {
        "experiment_id": finance["experiment_id"],
        "workspace_id": finance["data_pipeline"]["workspace_id"],
        **pipeline,
        "sft_global_step": finance["sft"]["completion"]["global_step"],
        "sft_epoch": finance["sft"]["completion"]["epoch"],
    }

    mol_aggregate = next(
        row
        for row in results["aggregates"]
        if row.get("paper_group") == "ft-agent-main"
        and row.get("benchmark") == "chemcotbench_mol_edit"
        and row.get("split") == "test"
        and row.get("metric") == "accuracy"
    )
    require(
        mol_aggregate.get("n") == EXPECTED_RUNS_PER_TASK
        and mol_aggregate.get("expected_n") == EXPECTED_RUNS_PER_TASK,
        "Molecule Editing aggregate is not the strict n=3/3 result",
    )

    return {
        "schema_version": 1,
        "kind": "ft-dojo-main-terminal-evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "matrix_suite": matrix.get("suite"),
            "training_policy": training_policy,
            "matrix_tasks": EXPECTED_TASKS,
            "base_tasks": EXPECTED_BASE_TASKS,
            "supplied_tasks": EXPECTED_TASKS + EXPECTED_BASE_TASKS,
            "strictly_complete_supplied_tasks": EXPECTED_TASKS + EXPECTED_BASE_TASKS,
            "terminal_status": "complete",
        },
        "source_artifacts": {
            "matrix": artifact_record(matrix_run / "matrix.json"),
            "provenance": matrix_provenance_record(matrix_run, matrix_tasks),
            "results": artifact_record(results_path),
            "supervisor_state": artifact_record(supervisor_state_path),
            "supervisor_log": artifact_record(supervisor_log_path),
            "final_test_log": artifact_record(final_test_log_path),
        },
        "execution": {
            "search_succeeded": sum(task["search"]["state"] == "succeeded" for task in tasks),
            "exact_2k_training_datasets": sum(
                task["data_pipeline"]["data_stats"]["total_samples"] == EXPECTED_TRAINING_SAMPLES
                for task in tasks
            ),
            "sft_completed": sum(all(task["sft"]["checks"].values()) for task in tasks),
            "training_policy": training_policy,
            "training_method_counts": training_method_counts,
            "autonomous_method_selections": sum(
                all(task["method_selection"]["checks"].values()) for task in tasks
            ),
            "logical_b200_method_envelopes": sum(
                task["method_selection"]["checks"]["logical_single_gpu"]
                and task["method_selection"]["checks"]["logical_gpu_is_b200"]
                and task["method_selection"]["checks"]["logical_memory_is_178gb"]
                for task in tasks
            ),
            "physical_h20_exposed_to_hypothesis": sum(
                not task["method_selection"]["checks"]["physical_h20_not_exposed_to_hypothesis"]
                for task in tasks
            ),
            "method_rationale_counts": method_rationale_counts,
            "signed_validation_selections": sum(
                task["validation_selection"]["held_out_test_used"] is False for task in tasks
            ),
            "held_out_attempted_once": sum(len(execution["attempts"][task["experiment_id"]]) == 1 for task in tasks),
            "held_out_states": states,
            "held_out_reused": execution["reused"],
            "held_out_evaluated": execution["evaluated"],
            "held_out_invocations": execution["invocations"],
            "requested_gpus": commit["gpus"],
            "requested_max_parallel": commit["max_parallel"],
            "observed_gpus": observed_gpus,
            "observed_peak_parallelism": execution["peak_parallelism"],
            "task_timeout": EXPECTED_TIMEOUT,
            "api_routing": route,
            "supervisor_terminal_stage": supervisor_state.get("stage"),
            "supervisor_return_code": supervisor_state.get("return_code"),
        },
        "representative_pipeline_evidence": representative,
        "molecule_editing_primary_aggregate": {
            key: mol_aggregate.get(key) for key in ("mean", "std", "n", "expected_n", "experiment_ids")
        },
        "tasks": tasks,
    }


def display_metric(items: list[dict[str, Any]]) -> str:
    return "; ".join(f"{item.get('label') or item['metric']}={float(item['value']):.2f}" for item in items)


def render_markdown(manifest: dict[str, Any]) -> str:
    execution = manifest["execution"]
    representative = manifest["representative_pipeline_evidence"]
    source = manifest["source_artifacts"]
    overlap = representative["pairwise_intersection_counts"]
    method_summary = ", ".join(
        f"{method}={count}"
        for method, count in sorted(execution["training_method_counts"].items())
    )
    representative_method = next(
        task["sft"]["training_method"]
        for task in manifest["tasks"]
        if task["experiment_id"] == representative["experiment_id"]
    )
    return "\n".join(
        (
            "# FT-Dojo 主实验终态审计",
            "",
            f"生成时间：{manifest['generated_at']}。本页与 `evidence_manifest.json` 由原始终态文件重新计算。",
            "",
            "> 结论：13 个 task × 3 次独立运行全部完成；搜索、严格 2k/full-epoch SFT 与 held-out 均为 39/39。",
            "> 加上 13 项 Base-7B，对应主表输入为严格 52/52 完整。",
            "",
            "## 覆盖与协议链",
            "",
            "| 阶段 | 可核验结果 |",
            "| --- | ---: |",
            f"| 恰好 2,000 条训练数据及注册证据 | {execution['exact_2k_training_datasets']}/39 |",
            f"| 完整 epoch、trainer state、方法锁及模型权重（{method_summary}） | {execution['sft_completed']}/39 |",
            f"| 初始 hypothesis 自主方法选择与最终方法锁一致 | {execution['autonomous_method_selections']}/39 |",
            f"| 方法选择时看到 1×B200 178GB 逻辑资源 | {execution['logical_b200_method_envelopes']}/39 |",
            f"| 方法选择时看到物理 H20 | {execution['physical_h20_exposed_to_hypothesis']}/39 |",
            f"| 搜索成功 | {execution['search_succeeded']}/39 |",
            "| validation-only 签名选择（held_out_test_used=false） | "
            f"{execution['signed_validation_selections']}/39 |",
            f"| 一次性 held-out 提交 | {execution['held_out_attempted_once']}/39 |",
            f"| held-out 成功 | {execution['held_out_states']['succeeded']}/39 |",
            "",
            "流程为：任务工作区中的 `process_data.py` 产出并登记恰好 2,000 条训练数据；"
            "随后由 task 在论文策略下自主选择 Full SFT 或普通 LoRA。正式证据会重新计算数据条数、"
            "配置 epoch、trainer 完成步数、方法锁及权重类型，主实验不接受 rsLoRA/DoRA/QLoRA；"
            "搜索结束后只按 validation 冻结并签名 checkpoint；最终 test 的 spec 与 result 必须同时匹配该签名。",
            "",
            "每项小型证据均记录 SHA-256；权重文件记录存在性与字节数，"
            "并由 checkpoint selection signature 绑定大小和 mtime。"
            f"机器可读明细：`evidence_manifest.json`（{len(manifest['tasks'])} 项）。",
            "",
            "## 自主方法选择审计",
            "",
            f"39 个 step-0 hypothesis 中，最终锁定的方法分布为 `{method_summary}`；"
            f"其中 {execution['method_rationale_counts']['two_thousand_sample_overfit_or_forgetting_risk']}/39 "
            "明确讨论 2,000 条小样本带来的过拟合、灾难性遗忘或能力漂移风险。",
            "",
            "方法选择阶段统一注入论文的单卡 B200 逻辑资源包络，而物理 H20 只在之后由执行层动态租约。"
            "因此若 39 项最终均选择 LoRA，这一结果不能归因于 hypothesis 被 H20 显存硬性限制；"
            "每项原始 hypothesis 摘要、资源字典、方法锁与 SHA-256 均保存在机器清单中。",
            "",
            "## 代表性清洗/SFT 证据",
            "",
            f"FinanceIQ run-1 所选 workspace `{representative['workspace_id']}`：pre-LLM "
            f"{representative['pre_llm_sample_count']} 条、接受训练 {representative['accepted_training_count']} 条、"
            f"validation {representative['validation_count']} 条、保护 holdout {representative['holdout_count']} 条。"
            f"train/validation={overlap['train_validation']}、train/holdout={overlap['train_holdout']}、"
            f"validation/holdout={overlap['validation_holdout']}；正式救援链已核验 "
            f"{representative['lineage_hashes']['verified_input_count']} 个训练输入 hash 与 "
            f"{representative['lineage_hashes']['verified_source_count']} 个源 workspace hash。",
            "",
            f"该 workspace 的训练方法为 {representative_method}（SFT/bf16）；trainer state 为 global step "
            f"{representative['sft_global_step']}、epoch {representative['sft_epoch']}。其源处理审计 SHA-256 为 "
            f"`{representative['processing_audit']['sha256']}`。其余 38 项逐项证据见机器清单。",
            "",
            "## 8 GPU 并行",
            "",
            f"统一 COMMIT 请求 GPU `{','.join(execution['requested_gpus'])}`、"
            f"`max-parallel={execution['requested_max_parallel']}`；"
            f"从 START/DONE/FAIL 事件重建的实际峰值并发为 {execution['observed_peak_parallelism']}，"
            f"实际覆盖 GPU `{','.join(execution['observed_gpus'])}`。累计为 "
            f"`{execution['held_out_reused']} reused, {execution['held_out_evaluated']} evaluated`；"
            "恢复调用只允许复用已成功结果，不会再次执行同一 checkpoint。",
            "",
            f"并发日志：`{source['final_test_log']['resolved_path']}`"
            f"（SHA-256 `{source['final_test_log']['sha256']}`）；"
            f"提交日志：`{source['supervisor_log']['resolved_path']}`。",
            "",
            f"所有任务保留 `timeout={execution['task_timeout']}`。搜索与 test 都记录模型适配路由 "
            f"`{execution['api_routing']['upstream_base']}` → `{execution['api_routing']['served_model']}`；"
            "因此这是模型适配复现，不是论文原 provider 的严格同模型复现。",
            "",
            "## 严格终态",
            "",
            f"supervisor 状态为 `{execution['supervisor_terminal_stage']}`（return code "
            f"{execution['supervisor_return_code']}）；39 项 held-out 均只提交一次且全部成功。",
            "",
        ),
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(
            matrix_run=args.matrix_run.resolve(),
            results_path=args.results.resolve(),
            supervisor_state_path=args.supervisor_state.resolve(),
            supervisor_log_path=args.supervisor_log.resolve(),
            final_test_log_path=args.final_test_log.resolve(),
        )
        atomic_write(
            args.output_dir / "evidence_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        atomic_write(args.output_dir / "EXPERIMENT_AUDIT.md", render_markdown(manifest))
    except (ExperimentAuditError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"Cannot render experiment audit: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"Wrote experiment audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
