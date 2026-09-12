#!/usr/bin/env python3
"""Collect deterministic FT-Dojo results without using held-out test scores for selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

if __package__:
    from .final_test_protocol import (
        FINAL_TEST_SCHEMA_VERSION,
        TEST_RANGE,
        is_policy_compliant_selection,
        model_artifact_type,
        normalize_training_policy,
        selection_signature,
        trace_protocol_pollution,
    )
    from .paper_matrix import all_paper_experiments
    from .rslora_pairing import pairing_signature_from_status, validate_pairing_artifact
else:
    from final_test_protocol import (
        FINAL_TEST_SCHEMA_VERSION,
        TEST_RANGE,
        is_policy_compliant_selection,
        model_artifact_type,
        normalize_training_policy,
        selection_signature,
        trace_protocol_pollution,
    )
    from paper_matrix import all_paper_experiments
    from rslora_pairing import pairing_signature_from_status, validate_pairing_artifact

ROOT = Path(__file__).resolve().parents[2]
FT_ROOT = ROOT / "finetune_files"
MATRIX_LOG_ROOT = FT_ROOT / "logs" / "paper-matrix"
BASE_LOG_ROOT = FT_ROOT / "logs" / "paper-base"
DEFAULT_REFERENCE = Path(__file__).with_name("paper_results.json")
SCHEMA_VERSION = 1
SESSION_FILE_RE = re.compile(r"^(?P<step>\d+)_")
HISTORY_NODE_SIZE = 2

PAPER_METRICS = {
    "aime25": ("accuracy",),
    "panorama_par4pc": ("macro_f1",),
    "panorama_noc4pc": ("macro_f1",),
    "panorama_pi4pc": ("gold_hit_rate",),
    "tablebench_data_analysis": ("accuracy",),
    "tablebench_fact_checking": ("accuracy",),
    "tablebench_numerical_reasoning": ("accuracy",),
    "tablebench_visualization": ("pass_at_1",),
    "FinanceIQ_gen": ("accuracy",),
    "chemcotbench_mol_und": ("mae", "tanimoto_similarity", "accuracy"),
    "chemcotbench_mol_edit": ("accuracy",),
    "chemcotbench_mol_opt": ("success_rate", "valid_smiles_rate"),
    "chemcotbench_reaction": ("fingerprint_similarity", "accuracy"),
}

TASK_CSV_FIELDS = (
    "paper_experiment_id",
    "paper_group",
    "runner",
    "state",
    "benchmark",
    "target_model",
    "training_policy",
    "training_method",
    "planner",
    "data_limit",
    "run_index",
    "selection_source",
    "candidate_source",
    "selected_history_index",
    "selected_loop_id",
    "test_result_source",
    "validation_metrics",
    "test_metrics",
    "diagnostics",
)
METRIC_CSV_FIELDS = (
    "paper_experiment_id",
    "paper_group",
    "benchmark",
    "stage",
    "split",
    "history_index",
    "loop_id",
    "selected",
    "derived",
    "dataset",
    "metric",
    "value",
    "unit",
    "formula",
)
AGGREGATE_CSV_FIELDS = (
    "paper_group",
    "benchmark",
    "target_model",
    "planner",
    "data_limit",
    "split",
    "metric",
    "label",
    "unit",
    "higher_is_better",
    "n",
    "expected_n",
    "mean",
    "std",
    "experiment_ids",
)
COMPARISON_CSV_FIELDS = (
    "source_table",
    "paper_group",
    "benchmark",
    "split",
    "metric",
    "mean",
    "std",
    "status",
    "observed_n",
    "observed_mean",
    "observed_std",
    "mean_delta",
    "std_delta",
)
LORA_RSLORA_CSV_FIELDS = (
    "source_experiment_id",
    "paired_experiment_id",
    "benchmark",
    "run_index",
    "source_training_method",
    "paired_training_method",
    "split",
    "metric",
    "unit",
    "higher_is_better",
    "source_value",
    "paired_value",
    "raw_delta",
    "improvement_delta",
    "state",
)


class CollectionError(RuntimeError):
    """Raised when a run artifact is structurally ambiguous or corrupt."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-run",
        action="append",
        default=[],
        metavar="NAME_OR_PATH",
        help="FT-Agent run name below paper-matrix, or a run-root path; repeatable",
    )
    parser.add_argument(
        "--base-run",
        action="append",
        default=[],
        metavar="NAME_OR_PATH",
        help="Base run name below paper-base, or a run-root path; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON and CSV reports")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    return parser.parse_args()


def safe_id(experiment_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", experiment_id).strip("_")


def project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    """Convert common result types to strict, deterministic JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return project_path(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        result = float(value)
    else:
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return numeric(item_method())
            except (TypeError, ValueError):
                return None
        try:
            result = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    return result if math.isfinite(result) else None


def normalize_accuracy_summary(payload: Any) -> dict[str, dict[str, float]]:
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("accuracy_summary", payload)
    if not isinstance(summary, dict):
        return {}
    result: dict[str, dict[str, float]] = {}
    for dataset, metrics in sorted(summary.items(), key=lambda pair: str(pair[0])):
        if not isinstance(metrics, dict):
            continue
        normalized = {
            str(metric): number
            for metric, value in sorted(metrics.items(), key=lambda pair: str(pair[0]))
            if (number := numeric(value)) is not None
        }
        if normalized:
            result[str(dataset)] = normalized
    return result


def matching_values(
    summary: dict[str, dict[str, float]],
    metric: str,
    patterns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    result = []
    for dataset, metrics in sorted(summary.items()):
        if patterns is not None and not any(pattern.lower() in dataset.lower() for pattern in patterns):
            continue
        for candidate, value in metrics.items():
            if candidate.lower() == metric.lower():
                result.append({"dataset": dataset, "metric": candidate, "value": value})
    return result


def derived_metric(
    metric: str,
    label: str,
    components: list[dict[str, Any]],
    *,
    formula: str,
    higher_is_better: bool,
    unit: str,
    required_components: int | None = None,
) -> dict[str, Any] | None:
    if not components or (required_components is not None and len(components) != required_components):
        return None
    return {
        "metric": metric,
        "label": label,
        "value": statistics.fmean(component["value"] for component in components),
        "higher_is_better": higher_is_better,
        "unit": unit,
        "formula": formula,
        "components": components,
    }


def derive_paper_metrics(  # noqa: C901
    benchmark: str,
    summary: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Derive the exact task-level columns used by the paper tables."""
    specs: list[dict[str, Any] | None]
    if benchmark == "aime25":
        specs = [
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "accuracy"),
                formula="mean accuracy across configured views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark in {"panorama_par4pc", "panorama_noc4pc"}:
        specs = [
            derived_metric(
                "macro_f1",
                "F1",
                matching_values(summary, "macro_f1"),
                formula="mean macro_f1 across configured views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark == "panorama_pi4pc":
        specs = [
            derived_metric(
                "gold_hit_rate",
                "Acc",
                matching_values(summary, "gold_hit_rate"),
                formula="mean gold_hit_rate across configured views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark in {
        "tablebench_data_analysis",
        "tablebench_fact_checking",
        "tablebench_numerical_reasoning",
    }:
        specs = [
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "accuracy"),
                formula="unweighted mean accuracy across released dataset views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark == "tablebench_visualization":
        specs = [
            derived_metric(
                "pass_at_1",
                "P@1",
                matching_values(summary, "Pass@1"),
                formula="mean Pass@1 across configured views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark == "FinanceIQ_gen":
        specs = [
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "accuracy"),
                formula="unweighted mean accuracy across ten released subject views",
                higher_is_better=True,
                unit="percent",
            ),
        ]
    elif benchmark == "chemcotbench_mol_und":
        specs = [
            derived_metric(
                "mae",
                "MAE",
                matching_values(summary, "mae", ("fg_count", "ring_count")),
                formula="mean MAE of fg_count and ring_count",
                higher_is_better=False,
                unit="absolute",
                required_components=2,
            ),
            derived_metric(
                "tanimoto_similarity",
                "TMS",
                matching_values(summary, "tanimoto_similarity_larger_means_better", ("murcko_scaffold",)),
                formula="Murcko scaffold Tanimoto similarity",
                higher_is_better=True,
                unit="ratio",
                required_components=1,
            ),
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "accuracy", ("equivalence", "ring_system_scaffold")),
                formula="mean accuracy of equivalence and ring_system_scaffold",
                higher_is_better=True,
                unit="percent",
                required_components=2,
            ),
        ]
    elif benchmark == "chemcotbench_mol_edit":
        specs = [
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "correct_rate"),
                formula="unweighted mean correct_rate across add, delete, and substitute",
                higher_is_better=True,
                unit="percent",
                required_components=3,
            ),
        ]
    elif benchmark == "chemcotbench_mol_opt":
        specs = [
            derived_metric(
                "success_rate",
                "SR",
                matching_values(summary, "success_rate"),
                formula="unweighted mean success_rate across six optimization views",
                higher_is_better=True,
                unit="percent",
                required_components=6,
            ),
            derived_metric(
                "valid_smiles_rate",
                "VS",
                matching_values(summary, "valid_smiles_rate"),
                formula="unweighted mean valid_smiles_rate across six optimization views",
                higher_is_better=True,
                unit="percent",
                required_components=6,
            ),
        ]
    elif benchmark == "chemcotbench_reaction":
        specs = [
            derived_metric(
                "fingerprint_similarity",
                "FTS",
                matching_values(summary, "morgan_sims", ("reaction_fs", "reaction_retro", "reaction_nepp")),
                formula="mean Morgan fingerprint similarity of fs, retro, and nepp",
                higher_is_better=True,
                unit="percent",
                required_components=3,
            ),
            derived_metric(
                "accuracy",
                "Acc",
                matching_values(summary, "accuracy", ("reaction_mechsel",)),
                formula="mechanism-selection accuracy",
                higher_is_better=True,
                unit="percent",
                required_components=1,
            ),
        ]
    else:
        specs = []
    return [spec for spec in specs if spec is not None]


def result_view(payload: Any, benchmark: str) -> dict[str, Any] | None:
    summary = normalize_accuracy_summary(payload)
    if not summary:
        return None
    paper_metrics = derive_paper_metrics(benchmark, summary)
    return {
        "accuracy_summary": summary,
        "paper_metrics": paper_metrics,
    }


def session_sort_key(path: Path) -> tuple[int, int, str]:
    try:
        loop_id = int(path.parent.name)
    except ValueError as error:
        message = f"Non-numeric session loop directory: {path.parent}"
        raise CollectionError(message) from error
    match = SESSION_FILE_RE.match(path.name)
    if match is None:
        message = f"Unrecognized session filename: {path}"
        raise CollectionError(message)
    return loop_id, int(match.group("step")), path.name


def latest_session_path(trace_path: Path) -> Path:
    candidates = [path for path in (trace_path / "__session__").glob("*/*_*") if path.is_file()]
    if not candidates:
        message = f"No session snapshots below {trace_path / '__session__'}"
        raise CollectionError(message)
    return max(candidates, key=session_sort_key)


def load_latest_session(trace_path: Path) -> tuple[Path, Any]:
    path = latest_session_path(trace_path)
    try:
        with path.open("rb") as stream:
            return path, pickle.load(stream)
    except Exception as error:
        message = f"Unable to load session snapshot {path}: {type(error).__name__}: {error}"
        raise CollectionError(message) from error


def select_sota_index(trace: Any) -> int | None:  # noqa: C901
    """Mirror FTTrace.get_sota_experiment using only acceptance and DAG ancestry."""
    hist = list(getattr(trace, "hist", []))
    parents = list(getattr(trace, "dag_parent", []))
    if not hist:
        return None
    if len(parents) != len(hist):
        message = f"Trace has {len(hist)} history nodes but {len(parents)} parent records"
        raise CollectionError(message)
    selection = getattr(trace, "current_selection", (-1,))
    if selection is None or len(selection) == 0:
        return None
    node_id = int(selection[0])
    if node_id == -1:
        node_id = len(hist) - 1
    if not 0 <= node_id < len(hist):
        message = f"Trace selection {node_id} is outside history of size {len(hist)}"
        raise CollectionError(message)

    ancestry: list[int] = []
    seen: set[int] = set()
    current = node_id
    while True:
        if current in seen:
            message = f"Cycle in trace ancestry at history node {current}"
            raise CollectionError(message)
        seen.add(current)
        ancestry.append(current)
        parent = parents[current]
        if not parent or parent[0] == current:
            break
        current = int(parent[0])
        if not 0 <= current < len(hist):
            message = f"Trace parent {current} is outside history of size {len(hist)}"
            raise CollectionError(message)

    for index in ancestry:
        node = hist[index]
        if not isinstance(node, tuple) or len(node) != HISTORY_NODE_SIZE:
            message = f"Malformed history node {index}"
            raise CollectionError(message)
        if bool(getattr(node[1], "decision", False)):
            return index
    return None


def experiment_result(experiment: Any) -> dict[str, Any]:
    workspace = getattr(experiment, "experiment_workspace", None)
    running_info = getattr(workspace, "running_info", None)
    result = getattr(running_info, "result", None)
    return result if isinstance(result, dict) else {}


def diagnostic(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def metric_ids(view: dict[str, Any] | None) -> set[str]:
    if view is None:
        return set()
    return {metric["metric"] for metric in view["paper_metrics"]}


def _validated_selection_payload(
    path: Path,
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    benchmark_dataset_path: str,
    expected_samples: int,
    training_policy: str = "paper",
    pairing_artifact_signature: str | None = None,
    selection_profile: str = "main",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    if __package__:
        from .run_validation_sweep import (  # noqa: PLC0415
            SweepTarget,
            validate_current_selection_provenance,
        )
    else:
        from run_validation_sweep import (  # type: ignore[no-redef]  # noqa: PLC0415
            SweepTarget,
            validate_current_selection_provenance,
        )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        message = "Validation selection top-level value is not an object"
        raise TypeError(message)
    target = SweepTarget(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        benchmark_dataset_path=benchmark_dataset_path,
        task_root=path.parent.resolve(),
        expected_samples=expected_samples,
        training_policy=training_policy,
        pairing_artifact_signature=pairing_artifact_signature,
        selection_profile=selection_profile,
    )
    selection = validate_current_selection_provenance(target, artifact)
    view = result_view({"accuracy_summary": selection.get("accuracy_summary")}, benchmark)
    if view is None:
        message = "Selected validation result has no usable accuracy summary"
        raise CollectionError(message)
    selected_metrics = {
        str(item.get("metric")): numeric(item.get("value"))
        for item in selection.get("selection_metrics", [])
        if isinstance(item, dict) and isinstance(item.get("metric"), str)
    }
    derived_metrics = {item["metric"]: item["value"] for item in view["paper_metrics"]}
    if selected_metrics != derived_metrics:
        message = "Selected paper metrics do not match the signed accuracy summary"
        raise CollectionError(message)
    expected_signature = selection.get("selection_signature")
    if not isinstance(expected_signature, str) or not expected_signature:
        message = "Selected checkpoint signature is missing"
        raise CollectionError(message)
    candidate_id = selection.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        message = "Selected candidate id is missing"
        raise CollectionError(message)
    model_path = Path(str(selection.get("model_path", ""))).resolve()
    candidate_source = selection.get("source")
    if not isinstance(candidate_source, str) or not candidate_source:
        message = "Selected candidate source is missing"
        raise CollectionError(message)
    if not is_policy_compliant_selection(model_path, candidate_source, training_policy):
        message = f"Selected model does not comply with training policy {training_policy!r}"
        raise CollectionError(message)
    return artifact, selection, view, expected_signature, candidate_id


def resolve_validation_selection_view(
    task_root: Path,
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    benchmark_dataset_path: str,
    expected_samples: int,
    training_policy: str = "paper",
    pairing_artifact_signature: str | None = None,
    selection_profile: str = "main",
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    str | None,
    dict[str, Any],
    list[dict[str, str]],
]:
    """Return validation metrics and identity only from the signed sweep artifact."""
    if __package__:
        from .validation_selection import (  # noqa: PLC0415
            normalize_selection_profile,
            validation_selection_file,
        )
    else:
        from validation_selection import (  # type: ignore[no-redef]  # noqa: PLC0415
            normalize_selection_profile,
            validation_selection_file,
        )

    selection_profile = normalize_selection_profile(selection_profile)
    path = task_root / validation_selection_file(selection_profile)
    metadata: dict[str, Any] = {
        "artifact": project_path(path),
        "state": "missing",
        "artifact_signature": None,
        "candidate_id": None,
        "selection_profile": selection_profile,
    }
    diagnostics: list[dict[str, str]] = []
    try:
        artifact, selection, view, expected_signature, candidate_id = _validated_selection_payload(
            path,
            experiment_id=experiment_id,
            benchmark=benchmark,
            model=model,
            benchmark_dataset_path=benchmark_dataset_path,
            expected_samples=expected_samples,
            training_policy=training_policy,
            pairing_artifact_signature=pairing_artifact_signature,
            selection_profile=selection_profile,
        )
    except (CollectionError, json.JSONDecodeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        metadata["state"] = "invalid" if path.exists() else "missing"
        diagnostics.append(
            diagnostic(
                "invalid_validation_selection" if path.exists() else "missing_validation_selection",
                f"Unable to use {path}: {type(error).__name__}: {error}",
            ),
        )
        return None, None, None, metadata, diagnostics
    metadata.update(
        {
            "state": "succeeded",
            "artifact_signature": artifact.get("artifact_signature"),
            "candidate_id": candidate_id,
            "checkpoint_step": selection.get("checkpoint_step"),
            "model_path": selection.get("model_path"),
            "candidate_source": selection.get("source"),
            "training_policy": training_policy,
            "artifact_type": model_artifact_type(Path(str(selection.get("model_path", ""))).resolve()),
            "pairing_artifact_signature": pairing_artifact_signature,
            "selection_profile": selection_profile,
        },
    )
    selected = {
        "source": "validation_sweep",
        "selection_profile": selection_profile,
        "candidate_source": selection.get("source"),
        "history_index": None,
        "loop_id": None,
        "candidate_id": candidate_id,
        "checkpoint_step": selection.get("checkpoint_step"),
        "test_used_for_selection": False,
    }
    return view, selected, expected_signature, metadata, diagnostics


def resolve_final_test_view(  # noqa: C901, PLR0912, PLR0915
    task_root: Path,
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    expected_signature: str,
    expected_selection_artifact_signature: str,
    benchmark_dataset_path: str,
    expected_pairing_artifact_signature: str | None = None,
    selection_profile: str = "main",
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, str]]]:
    """Return only a valid, signature-matched post-selection test artifact."""
    if __package__:
        from .validation_selection import final_test_file, normalize_selection_profile  # noqa: PLC0415
    else:
        from validation_selection import (  # type: ignore[no-redef]  # noqa: PLC0415
            final_test_file,
            normalize_selection_profile,
        )

    selection_profile = normalize_selection_profile(selection_profile)
    path = task_root / final_test_file(selection_profile)
    metadata: dict[str, Any] = {
        "artifact": project_path(path),
        "expected_signature": expected_signature,
        "state": "missing",
        "evaluation_mode": None,
        "source": None,
        "selection_profile": selection_profile,
    }
    diagnostics: list[dict[str, str]] = []
    artifact: dict[str, Any] | None = None
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            diagnostics.append(
                diagnostic(
                    "invalid_final_test_artifact",
                    f"Unable to use {path}: {type(error).__name__}: {error}",
                ),
            )
        else:
            if isinstance(value, dict):
                artifact = value
            else:
                diagnostics.append(
                    diagnostic(
                        "invalid_final_test_artifact",
                        f"Unable to use {path}: top-level value is not an object",
                    ),
                )

    if artifact is not None:
        selection = artifact.get("selection")
        artifact_signature = selection.get("signature") if isinstance(selection, dict) else None
        validation_selection_signature = (
            selection.get("validation_selection_artifact_signature") if isinstance(selection, dict) else None
        )
        selection_pairing_signature = (
            selection.get("pairing_artifact_signature") if isinstance(selection, dict) else None
        )
        metadata.update(
            {
                "state": artifact.get("state", "unknown"),
                "evaluation_mode": artifact.get("evaluation_mode"),
                "artifact_signature": artifact_signature,
                "validation_selection_artifact_signature": validation_selection_signature,
                "pairing_artifact_signature": selection_pairing_signature,
            },
        )
        incompatibilities = []
        if artifact.get("schema_version") != FINAL_TEST_SCHEMA_VERSION:
            incompatibilities.append("schema version")
        if artifact.get("experiment_id") != experiment_id:
            incompatibilities.append("experiment id")
        if artifact.get("benchmark") != benchmark:
            incompatibilities.append("benchmark")
        if artifact.get("model") != model:
            incompatibilities.append("model")
        if artifact.get("test_range") != TEST_RANGE:
            incompatibilities.append("test range")
        try:
            artifact_profile = normalize_selection_profile(artifact.get("selection_profile"))
            selected_profile = normalize_selection_profile(
                selection.get("selection_profile") if isinstance(selection, dict) else None,
            )
        except ValueError:
            artifact_profile = None
            selected_profile = None
        if artifact_profile != selection_profile or selected_profile != selection_profile:
            incompatibilities.append("selection profile")
        if artifact_signature != expected_signature:
            incompatibilities.append("selection signature")
        if validation_selection_signature != expected_selection_artifact_signature:
            incompatibilities.append("validation selection artifact signature")
        if selection_pairing_signature != expected_pairing_artifact_signature:
            incompatibilities.append("selection pairing artifact signature")
        if artifact.get("pairing_artifact_signature") != expected_pairing_artifact_signature:
            incompatibilities.append("final-test pairing artifact signature")
        if artifact.get("benchmark_dataset_path") != benchmark_dataset_path:
            incompatibilities.append("benchmark dataset")
        if artifact.get("state") != "succeeded":
            incompatibilities.append("state")
        if artifact.get("evaluation_mode") != "post_selection":
            incompatibilities.append("evaluation mode")
        if incompatibilities:
            diagnostics.append(
                diagnostic(
                    "unusable_final_test_artifact",
                    "Final-test artifact does not match the current selected checkpoint: "
                    + ", ".join(incompatibilities),
                ),
            )
        else:
            view = result_view(artifact.get("result"), benchmark)
            if view is None:
                diagnostics.append(
                    diagnostic("invalid_final_test_result", "Final-test artifact has no usable accuracy summary"),
                )
            else:
                mode = str(artifact.get("evaluation_mode"))
                metadata["source"] = mode
                metadata["reused_from"] = artifact.get("reused_from")
                return view, metadata, diagnostics
    return None, metadata, diagnostics


def resolve_protocol_final_test(
    task_root: Path,
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    expected_signature: str | None,
    expected_selection_artifact_signature: str | None,
    benchmark_dataset_path: str | None,
    protocol_pollution: tuple[str, ...],
    pairing_artifact_signature: str | None = None,
    selection_profile: str = "main",
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, str]]]:
    if __package__:
        from .validation_selection import final_test_file, normalize_selection_profile  # noqa: PLC0415
    else:
        from validation_selection import (  # type: ignore[no-redef]  # noqa: PLC0415
            final_test_file,
            normalize_selection_profile,
        )

    selection_profile = normalize_selection_profile(selection_profile)
    if protocol_pollution:
        metadata = {
            "artifact": project_path(task_root / final_test_file(selection_profile)),
            "expected_signature": expected_signature,
            "state": "blocked_protocol_polluted",
            "evaluation_mode": None,
            "source": None,
            "protocol_pollution": list(protocol_pollution),
            "selection_profile": selection_profile,
        }
        diagnostics = [
            diagnostic(
                "protocol_polluted",
                "Search trace contains held-out result payloads and is excluded from clean test reporting: "
                + ", ".join(protocol_pollution),
            ),
        ]
        return None, metadata, diagnostics
    if (
        expected_signature is None
        or expected_selection_artifact_signature is None
        or benchmark_dataset_path is None
    ):
        return (
            None,
            {
                "artifact": project_path(task_root / final_test_file(selection_profile)),
                "expected_signature": None,
                "state": "blocked_validation_selection",
                "evaluation_mode": None,
                "source": None,
                "selection_profile": selection_profile,
            },
            [],
        )
    return resolve_final_test_view(
        task_root,
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        expected_signature=expected_signature,
        expected_selection_artifact_signature=expected_selection_artifact_signature,
        benchmark_dataset_path=benchmark_dataset_path,
        expected_pairing_artifact_signature=pairing_artifact_signature,
        selection_profile=selection_profile,
    )


def collect_ft_task(task_root: Path, status: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    benchmark = str(status["benchmark"])
    training_policy = normalize_training_policy(status.get("training_policy", "paper"))
    try:
        pairing_artifact_signature = pairing_signature_from_status(status)
        if pairing_artifact_signature is not None:
            pairing_artifact = validate_pairing_artifact(
                task_root,
                experiment_id=str(status["experiment_id"]),
                expected_samples=int(status["formal_expected_samples"]),
                expected_signature=pairing_artifact_signature,
            )
        else:
            pairing_artifact = None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        message = f"Invalid rsLoRA pairing contract: {error}"
        raise CollectionError(message) from error

    session_path: Path | None
    if pairing_artifact_signature is None:
        session_path, session = load_latest_session(task_root / "trace")
        trace = getattr(session, "trace", None)
        if trace is None:
            message = f"Latest session has no trace: {session_path}"
            raise CollectionError(message)
        protocol_pollution = trace_protocol_pollution(trace)
        hist = list(getattr(trace, "hist", []))
        idx2loop = dict(getattr(trace, "idx2loop_id", {}))
        selected_index = select_sota_index(trace)
    else:
        # The standalone paired trainer has no agent loop.  Its signed pairing
        # artifact and validation selection are the complete provenance chain.
        session_path = None
        trace = None
        protocol_pollution = ()
        hist = []
        idx2loop = {}
        selected_index = None
    diagnostics: list[dict[str, str]] = []
    loops = []
    for index, node in enumerate(hist):
        if not isinstance(node, tuple) or len(node) != HISTORY_NODE_SIZE:
            message = f"Malformed history node {index}"
            raise CollectionError(message)
        experiment, feedback = node
        result = experiment_result(experiment)
        validation = result_view(result.get("benchmark"), benchmark)
        loops.append(
            {
                "history_index": index,
                "loop_id": idx2loop.get(index),
                "accepted": bool(getattr(feedback, "decision", False)),
                "validation": validation,
                "test": None,
            },
        )

    scen = getattr(trace, "scen", None)
    baseline_validation = result_view(getattr(scen, "baseline_benchmark_score", None), benchmark)
    baseline = {"validation": baseline_validation, "test": None}

    experiment_id = str(status["experiment_id"])
    model = str(status["model"])
    trace_selected_source = "baseline" if selected_index is None else "accepted_loop"
    trace_selected_loop = None if selected_index is None else loops[selected_index]
    trace_selected_loop_id = trace_selected_loop["loop_id"] if trace_selected_loop is not None else None
    validation_selection_metadata: dict[str, Any]
    if protocol_pollution:
        if selected_index is None:
            trace_model_path = (FT_ROOT / "models" / model).resolve()
        else:
            workspace = getattr(hist[selected_index][0], "experiment_workspace", None)
            if workspace is None or getattr(workspace, "workspace_path", None) is None:
                message = f"Selected history node {selected_index} has no workspace"
                raise CollectionError(message)
            trace_model_path = (Path(workspace.workspace_path) / "output").resolve()
        expected_signature: str | None = selection_signature(
            trace_selected_source,
            selected_index,
            trace_selected_loop_id,
            trace_model_path,
        )
        final_validation = None
        selected = {
            "source": trace_selected_source,
            "history_index": selected_index,
            "loop_id": trace_selected_loop_id,
            "test_used_for_selection": False,
        }
        validation_selection_metadata = {
            "artifact": None,
            "state": "blocked_protocol_polluted",
            "artifact_signature": None,
            "candidate_id": None,
        }
    else:
        benchmark_dataset_path = status.get("benchmark_dataset_path")
        expected_samples = status.get("formal_expected_samples")
        sample_contract_valid = (
            isinstance(expected_samples, int)
            and not isinstance(expected_samples, bool)
            and expected_samples > 0
            and status.get("data_limit") == expected_samples
        )
        if not isinstance(benchmark_dataset_path, str) or not benchmark_dataset_path:
            final_validation = None
            selected = None
            expected_signature = None
            validation_selection_metadata = {
                "artifact": None,
                "state": "invalid",
                "artifact_signature": None,
                "candidate_id": None,
            }
            diagnostics.append(
                diagnostic(
                    "invalid_validation_selection",
                    "Task status has no pinned benchmark dataset path",
                ),
            )
        elif not sample_contract_valid:
            final_validation = None
            selected = None
            expected_signature = None
            validation_selection_metadata = {
                "artifact": None,
                "state": "invalid",
                "artifact_signature": None,
                "candidate_id": None,
            }
            diagnostics.append(
                diagnostic(
                    "invalid_validation_selection",
                    "Task status has no consistent formal sample contract",
                ),
            )
        else:
            (
                final_validation,
                selected,
                expected_signature,
                validation_selection_metadata,
                selection_diagnostics,
            ) = resolve_validation_selection_view(
                task_root,
                experiment_id=experiment_id,
                benchmark=benchmark,
                model=model,
                benchmark_dataset_path=benchmark_dataset_path,
                expected_samples=expected_samples,
                training_policy=training_policy,
                pairing_artifact_signature=pairing_artifact_signature,
            )
            diagnostics.extend(selection_diagnostics)
        if selected is None:
            selected = {
                "source": None,
                "history_index": None,
                "loop_id": None,
                "candidate_id": None,
                "checkpoint_step": None,
                "test_used_for_selection": False,
            }

    artifact_signature = validation_selection_metadata.get("artifact_signature")
    expected_selection_artifact_signature = (
        artifact_signature if isinstance(artifact_signature, str) and artifact_signature else None
    )
    final_test_dataset_path = status.get("benchmark_dataset_path")
    if not isinstance(final_test_dataset_path, str) or not final_test_dataset_path:
        final_test_dataset_path = None
    final_test, final_test_metadata, final_test_diagnostics = resolve_protocol_final_test(
        task_root,
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        expected_signature=expected_signature,
        expected_selection_artifact_signature=expected_selection_artifact_signature,
        benchmark_dataset_path=final_test_dataset_path,
        protocol_pollution=protocol_pollution,
        pairing_artifact_signature=pairing_artifact_signature,
    )
    diagnostics.extend(final_test_diagnostics)
    final = {"validation": final_validation, "test": final_test}
    if final_test is None:
        diagnostics.append(
            diagnostic(
                "missing_final_test",
                "No clean, successful, signature-matched post-selection final-test artifact",
            ),
        )

    expected_metrics = set(PAPER_METRICS.get(benchmark, ()))
    for split in ("validation", "test"):
        missing = expected_metrics - metric_ids(final[split])
        if missing:
            diagnostics.append(
                diagnostic(
                    f"missing_{split}_paper_metrics",
                    f"Missing derived paper metrics: {', '.join(sorted(missing))}",
                ),
            )

    lora_comparison: dict[str, Any] | None = None
    if (
        pairing_artifact_signature is None
        and not protocol_pollution
        and status.get("formal_training_method") == "lora"
    ):
        comparison_diagnostics: list[dict[str, str]] = []
        comparison_validation: dict[str, Any] | None = None
        comparison_selected: dict[str, Any] | None = None
        comparison_signature: str | None = None
        comparison_selection_metadata: dict[str, Any] = {
            "artifact": None,
            "state": "invalid",
            "artifact_signature": None,
            "candidate_id": None,
            "selection_profile": "lora_comparison",
        }
        comparison_expected_samples = status.get("formal_expected_samples")
        if (
            final_test_dataset_path is not None
            and isinstance(comparison_expected_samples, int)
            and not isinstance(comparison_expected_samples, bool)
            and comparison_expected_samples > 0
            and status.get("data_limit") == comparison_expected_samples
        ):
            (
                comparison_validation,
                comparison_selected,
                comparison_signature,
                comparison_selection_metadata,
                comparison_selection_diagnostics,
            ) = resolve_validation_selection_view(
                task_root,
                experiment_id=experiment_id,
                benchmark=benchmark,
                model=model,
                benchmark_dataset_path=final_test_dataset_path,
                expected_samples=comparison_expected_samples,
                training_policy=training_policy,
                pairing_artifact_signature=None,
                selection_profile="lora_comparison",
            )
            comparison_diagnostics.extend(comparison_selection_diagnostics)
        comparison_artifact_signature = comparison_selection_metadata.get("artifact_signature")
        comparison_test, comparison_test_metadata, comparison_test_diagnostics = resolve_protocol_final_test(
            task_root,
            experiment_id=experiment_id,
            benchmark=benchmark,
            model=model,
            expected_signature=comparison_signature,
            expected_selection_artifact_signature=(
                comparison_artifact_signature
                if isinstance(comparison_artifact_signature, str) and comparison_artifact_signature
                else None
            ),
            benchmark_dataset_path=final_test_dataset_path,
            protocol_pollution=(),
            pairing_artifact_signature=None,
            selection_profile="lora_comparison",
        )
        comparison_diagnostics.extend(comparison_test_diagnostics)
        comparison_final = {
            "validation": comparison_validation,
            "test": comparison_test,
        }
        for split in ("validation", "test"):
            missing = expected_metrics - metric_ids(comparison_final[split])
            if missing:
                comparison_diagnostics.append(
                    diagnostic(
                        f"missing_lora_comparison_{split}_paper_metrics",
                        f"Missing ordinary-LoRA comparison metrics: {', '.join(sorted(missing))}",
                    ),
                )
        lora_comparison = {
            "selection_profile": "lora_comparison",
            "selection": comparison_selected,
            "validation_selection": comparison_selection_metadata,
            "final_test": comparison_test_metadata,
            "final": comparison_final,
            "diagnostics": comparison_diagnostics,
        }

    return {
        "runner": "ft-agent",
        "run_root": project_path(task_root.parent),
        "task_root": project_path(task_root),
        "experiment_id": experiment_id,
        "paper_experiment_id": f"ft-agent/{experiment_id}",
        "state": "protocol_polluted" if protocol_pollution else status.get("state", "unknown"),
        "search_state": status.get("state", "unknown"),
        "benchmark": benchmark,
        "target_model": status.get("model"),
        "training_policy": training_policy,
        "training_method": status.get("formal_training_method"),
        "planner": status.get("planner"),
        "data_limit": status.get("data_limit"),
        "run_index": int(experiment_id.rsplit("run-", 1)[1]) if "/run-" in experiment_id else 1,
        "trace": (
            {
                "latest_session": project_path(session_path),
                "history_nodes": len(hist),
                "selection_rule": "latest accepted ancestor on current validation-selected branch",
                "protocol_pollution": list(protocol_pollution),
            }
            if session_path is not None
            else None
        ),
        "pairing": (
            {
                "run_kind": pairing_artifact.get("run_kind"),
                "source_experiment_id": pairing_artifact.get("source_experiment_id"),
                "source_matrix_root": project_path(Path(str(pairing_artifact.get("source_matrix_root")))),
                "artifact_signature": pairing_artifact_signature,
            }
            if pairing_artifact is not None
            else None
        ),
        "baseline": baseline,
        "loops": loops,
        "selection": selected,
        "validation_selection": validation_selection_metadata,
        "final_test": final_test_metadata,
        "final": final,
        "lora_comparison": lora_comparison,
        "diagnostics": diagnostics,
    }


def score_column(rows: list[dict[str, str]]) -> str:
    metadata = {"dataset", "version", "metric", "mode"}
    columns = sorted({column for row in rows for column in row} - metadata)
    if len(columns) != 1:
        message = f"Expected one OpenCompass score column, found {columns}"
        raise CollectionError(message)
    return columns[0]


def summary_from_rows(rows: Any) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        message = "OpenCompass status has no summary rows"
        raise CollectionError(message)
    typed_rows: list[dict[str, str]] = rows
    column = score_column(typed_rows)
    result: dict[str, dict[str, float]] = {}
    diagnostics: list[dict[str, str]] = []
    for row in typed_rows:
        dataset = str(row.get("dataset", "")).strip()
        metric = str(row.get("metric", "")).strip()
        value = numeric(row.get(column))
        if not dataset or not metric or value is None:
            continue
        metrics = result.setdefault(dataset, {})
        if metric in metrics and metrics[metric] != value:
            diagnostics.append(
                diagnostic(
                    "conflicting_summary_rows",
                    f"Keeping first {dataset}/{metric}={metrics[metric]}, ignoring {value}",
                ),
            )
            continue
        metrics.setdefault(metric, value)
    return dict(sorted(result.items())), diagnostics


def collect_base_task(task_root: Path, status: dict[str, Any]) -> dict[str, Any]:
    benchmark = str(status["benchmark"])
    diagnostics: list[dict[str, str]] = []
    final: dict[str, dict[str, Any] | None] = {}
    for split in ("validation", "test"):
        split_status = status.get("splits", {}).get(split, {})
        summary = split_status.get("summary", {})
        try:
            accuracy_summary, row_diagnostics = summary_from_rows(summary.get("rows"))
        except CollectionError as error:
            final[split] = None
            diagnostics.append(diagnostic(f"missing_{split}_summary", str(error)))
            continue
        diagnostics.extend(row_diagnostics)
        final[split] = result_view({"accuracy_summary": accuracy_summary}, benchmark)
        missing = set(PAPER_METRICS.get(benchmark, ())) - metric_ids(final[split])
        if missing:
            diagnostics.append(
                diagnostic(
                    f"missing_{split}_paper_metrics",
                    f"Missing derived paper metrics: {', '.join(sorted(missing))}",
                ),
            )
    experiment_id = str(status["experiment_id"])
    return {
        "runner": "base",
        "run_root": project_path(task_root.parent),
        "task_root": project_path(task_root),
        "experiment_id": experiment_id,
        "paper_experiment_id": experiment_id,
        "state": status.get("state", "unknown"),
        "benchmark": benchmark,
        "target_model": status.get("target_model"),
        "planner": None,
        "data_limit": None,
        "run_index": status.get("run_index", 1),
        "trace": None,
        "baseline": None,
        "loops": [],
        "selection": {
            "source": "base_model",
            "history_index": None,
            "loop_id": None,
            "test_used_for_selection": False,
        },
        "final_test": {
            "source": "base_runner_split",
            "state": status.get("state", "unknown"),
            "artifact": project_path(task_root / "status.json"),
        },
        "final": final,
        "diagnostics": diagnostics,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"Unable to read {path}: {type(error).__name__}: {error}"
        raise CollectionError(message) from error


def failed_task_record(
    run_root: Path,
    task: dict[str, Any],
    runner: str,
    error: str,
    status: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_id = str(task["experiment_id"])
    benchmark = str(task["benchmark"])
    return {
        "runner": runner,
        "run_root": project_path(run_root),
        "task_root": project_path(run_root / safe_id(experiment_id)),
        "experiment_id": experiment_id,
        "paper_experiment_id": f"ft-agent/{experiment_id}" if runner == "ft-agent" else experiment_id,
        "state": status.get("state", "missing") if status else "missing",
        "benchmark": benchmark,
        "target_model": task.get("model", task.get("target_model")),
        "planner": task.get("planner"),
        "data_limit": task.get("data_limit"),
        "run_index": task.get("run_index", 1),
        "trace": None,
        "baseline": None,
        "loops": [],
        "selection": None,
        "final_test": None,
        "final": {"validation": None, "test": None},
        "diagnostics": [diagnostic("collection_failed", error)],
    }


def collect_run(run_root: Path, runner: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_root / "matrix.json"
    manifest = read_json(manifest_path)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        message = f"Manifest tasks are not a list: {manifest_path}"
        raise CollectionError(message)
    records = []
    for task in sorted(tasks, key=lambda item: str(item["experiment_id"])):
        task_root = run_root / safe_id(str(task["experiment_id"]))
        status_path = task_root / "status.json"
        status = read_json(status_path) if status_path.is_file() else None
        if status is None:
            records.append(failed_task_record(run_root, task, runner, "status.json is missing", None))
            continue
        try:
            record = (
                collect_ft_task(task_root, status)
                if runner == "ft-agent"
                else collect_base_task(task_root, status)
            )
        except CollectionError as error:
            record = failed_task_record(run_root, task, runner, str(error), status)
        records.append(record)
    return records, {
        "runner": runner,
        "run_root": project_path(run_root),
        "manifest": project_path(manifest_path),
        "expected_tasks": len(tasks),
    }


def paper_inventory() -> list[dict[str, Any]]:
    return [experiment.to_dict() for experiment in all_paper_experiments()]


def attach_inventory(tasks: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> None:
    by_id = {item["experiment_id"]: item for item in inventory}
    seen: set[str] = set()
    for task in tasks:
        paper_id = task["paper_experiment_id"]
        if paper_id in seen:
            message = f"Duplicate supplied result for paper experiment {paper_id}"
            raise CollectionError(message)
        seen.add(paper_id)
        paper = by_id.get(paper_id)
        if paper is None:
            task["paper_group"] = None
            task["diagnostics"].append(diagnostic("not_in_paper_inventory", paper_id))
            continue
        task["paper_group"] = paper["paper_group"]
        task["artifact_status"] = paper["artifact_status"]


def final_metric_rows(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        for split in ("validation", "test"):
            view = task["final"].get(split)
            if view is None:
                continue
            rows.extend(
                {
                    "paper_group": task.get("paper_group"),
                    "benchmark": task["benchmark"],
                    "target_model": task.get("target_model"),
                    "planner": task.get("planner"),
                    "data_limit": task.get("data_limit"),
                    "split": split,
                    "metric": metric["metric"],
                    "label": metric["label"],
                    "unit": metric["unit"],
                    "higher_is_better": metric["higher_is_better"],
                    "value": metric["value"],
                    "paper_experiment_id": task["paper_experiment_id"],
                }
                for metric in view["paper_metrics"]
            )
    return rows


def aggregate_metrics(tasks: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = Counter((item["paper_group"], item["benchmark"]) for item in inventory)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in final_metric_rows(tasks):
        key = (
            row["paper_group"],
            row["benchmark"],
            row["target_model"],
            row["planner"],
            row["data_limit"],
            row["split"],
            row["metric"],
            row["label"],
            row["unit"],
            row["higher_is_better"],
        )
        grouped[key].append(row)
    aggregates = []
    for key, rows in sorted(grouped.items(), key=lambda pair: tuple(str(item) for item in pair[0])):
        values = [row["value"] for row in rows]
        (
            paper_group,
            benchmark,
            target_model,
            planner,
            data_limit,
            split,
            metric,
            label,
            unit,
            higher_is_better,
        ) = key
        aggregates.append(
            {
                "paper_group": paper_group,
                "benchmark": benchmark,
                "target_model": target_model,
                "planner": planner,
                "data_limit": data_limit,
                "split": split,
                "metric": metric,
                "label": label,
                "unit": unit,
                "higher_is_better": higher_is_better,
                "n": len(values),
                "expected_n": expected[(paper_group, benchmark)],
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "experiment_ids": sorted(row["paper_experiment_id"] for row in rows),
            },
        )
    return aggregates


def expand_reference_result(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        message = f"Reference result is not an object: {item!r}"
        raise CollectionError(message)
    if "split" in item:
        return [dict(item)]
    splits = item.get("splits")
    if not isinstance(splits, dict) or not splits:
        message = f"Reference result has no split or splits mapping: {item!r}"
        raise CollectionError(message)
    common = {key: value for key, value in item.items() if key != "splits"}
    expanded = []
    for split, split_result in sorted(splits.items()):
        if not isinstance(split_result, dict):
            message = f"Reference split {split!r} is not an object"
            raise CollectionError(message)
        expanded.append({**common, **split_result, "split": split})
    return expanded


def validate_reference_result(reference: dict[str, Any], seen: set[tuple[str, str, str, str]]) -> None:
    required = {"source_table", "paper_group", "benchmark", "split", "metric", "mean", "std"}
    missing = required - reference.keys()
    if missing:
        message = f"Reference result is missing fields {sorted(missing)}: {reference!r}"
        raise CollectionError(message)
    key = tuple(str(reference[field]) for field in ("paper_group", "benchmark", "split", "metric"))
    if key in seen:
        message = f"Duplicate reference result: {key}"
        raise CollectionError(message)
    seen.add(key)
    if numeric(reference["mean"]) is None or (
        reference["std"] is not None and numeric(reference["std"]) is None
    ):
        message = f"Reference result has a non-numeric mean or standard deviation: {reference!r}"
        raise CollectionError(message)


def load_references(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    results = payload.get("results")
    if not isinstance(results, list):
        message = f"Reference results are not a list: {path}"
        raise CollectionError(message)
    expanded = [reference for item in results for reference in expand_reference_result(item)]
    seen: set[tuple[str, str, str, str]] = set()
    for reference in expanded:
        validate_reference_result(reference, seen)
    return expanded


def compare_references(
    aggregates: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["paper_group"], row["benchmark"], row["split"], row["metric"]): row for row in aggregates
    }
    comparisons = []
    for reference in sorted(
        references,
        key=lambda row: (row["source_table"], row["paper_group"], row["benchmark"], row["split"], row["metric"]),
    ):
        key = (
            reference["paper_group"],
            reference["benchmark"],
            reference["split"],
            reference["metric"],
        )
        observed = by_key.get(key)
        comparisons.append(
            {
                **reference,
                "status": "matched" if observed is not None else "missing_observation",
                "observed_n": observed["n"] if observed else 0,
                "observed_mean": observed["mean"] if observed else None,
                "observed_std": observed["std"] if observed else None,
                "mean_delta": observed["mean"] - reference["mean"] if observed else None,
                "std_delta": (
                    observed["std"] - reference["std"]
                    if observed and observed["std"] is not None and reference.get("std") is not None
                    else None
                ),
            },
        )
    return comparisons


def coverage_report(
    tasks: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    task_by_id = {task["paper_experiment_id"]: task for task in tasks}
    paper_states = Counter()
    for item in inventory:
        task = task_by_id.get(item["experiment_id"])
        if task is not None:
            if task["final"].get("validation") is not None and task["final"].get("test") is not None:
                paper_states["collected"] += 1
            else:
                paper_states["supplied_but_incomplete"] += 1
        elif item["runner"] == "not-public":
            paper_states["blocked_unpublished"] += 1
        else:
            paper_states["runnable_not_supplied"] += 1
    task_states = Counter(task["state"] for task in tasks)
    diagnostic_counts = Counter(entry["code"] for task in tasks for entry in task["diagnostics"])
    return {
        "supplied_runs": runs,
        "supplied_tasks": len(tasks),
        "task_states": dict(sorted(task_states.items())),
        "diagnostics": dict(sorted(diagnostic_counts.items())),
        "paper_inventory_total": len(inventory),
        "paper_inventory_states": dict(sorted(paper_states.items())),
    }


def all_metric_rows(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        stages: list[tuple[str, int | None, int | None, bool, dict[str, Any] | None]] = []
        if task.get("baseline") is not None:
            stages.extend(
                (
                    f"baseline:{split}",
                    None,
                    None,
                    task["selection"]["source"] == "baseline",
                    task["baseline"][split],
                )
                for split in ("validation", "test")
            )
        for loop in task["loops"]:
            selected = (
                task["selection"] is not None
                and task["selection"]["history_index"] == loop["history_index"]
            )
            stages.extend(
                (f"loop:{split}", loop["history_index"], loop["loop_id"], selected, loop[split])
                for split in ("validation", "test")
            )
        stages.extend(
            (
                f"final:{split}",
                None,
                task["selection"]["loop_id"] if task["selection"] else None,
                True,
                task["final"][split],
            )
            for split in ("validation", "test")
        )
        for stage, history_index, loop_id, selected, view in stages:
            if view is None:
                continue
            split = stage.rsplit(":", 1)[1]
            for dataset, metrics in view["accuracy_summary"].items():
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "paper_experiment_id": task["paper_experiment_id"],
                            "paper_group": task.get("paper_group"),
                            "benchmark": task["benchmark"],
                            "stage": stage.split(":", 1)[0],
                            "split": split,
                            "history_index": history_index,
                            "loop_id": loop_id,
                            "selected": selected,
                            "derived": False,
                            "dataset": dataset,
                            "metric": metric,
                            "value": value,
                            "unit": None,
                            "formula": None,
                        },
                    )
            rows.extend(
                {
                    "paper_experiment_id": task["paper_experiment_id"],
                    "paper_group": task.get("paper_group"),
                    "benchmark": task["benchmark"],
                    "stage": stage.split(":", 1)[0],
                    "split": split,
                    "history_index": history_index,
                    "loop_id": loop_id,
                    "selected": selected,
                    "derived": True,
                    "dataset": "__paper_aggregate__",
                    "metric": metric["metric"],
                    "value": metric["value"],
                    "unit": metric["unit"],
                    "formula": metric["formula"],
                }
                for metric in view["paper_metrics"]
            )
    return sorted(
        rows,
        key=lambda row: (
            row["paper_experiment_id"],
            row["stage"],
            row["split"],
            -1 if row["history_index"] is None else row["history_index"],
            row["dataset"],
            row["metric"],
        ),
    )


def task_csv_rows(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        final_metrics = {}
        for split in ("validation", "test"):
            view = task["final"].get(split)
            final_metrics[split] = (
                {metric["metric"]: metric["value"] for metric in view["paper_metrics"]} if view else None
            )
        rows.append(
            {
                "paper_experiment_id": task["paper_experiment_id"],
                "paper_group": task.get("paper_group"),
                "runner": task["runner"],
                "state": task["state"],
                "benchmark": task["benchmark"],
                "target_model": task.get("target_model"),
                "training_policy": task.get("training_policy"),
                "training_method": task.get("training_method"),
                "planner": task.get("planner"),
                "data_limit": task.get("data_limit"),
                "run_index": task.get("run_index"),
                "selection_source": task["selection"]["source"] if task["selection"] else None,
                "candidate_source": task["selection"].get("candidate_source") if task["selection"] else None,
                "selected_history_index": task["selection"]["history_index"] if task["selection"] else None,
                "selected_loop_id": task["selection"]["loop_id"] if task["selection"] else None,
                "test_result_source": (task.get("final_test") or {}).get("source"),
                "validation_metrics": json.dumps(final_metrics["validation"], sort_keys=True),
                "test_metrics": json.dumps(final_metrics["test"], sort_keys=True),
                "diagnostics": "; ".join(entry["code"] for entry in task["diagnostics"]),
            },
        )
    return rows


def _task_paper_metrics(
    task: dict[str, Any] | None,
    split: str,
    *,
    selection_profile: str = "main",
) -> dict[str, dict[str, Any]]:
    if task is None:
        return {}
    result_root = task if selection_profile == "main" else task.get(selection_profile, {})
    if not isinstance(result_root, dict):
        return {}
    view = result_root.get("final", {}).get(split)
    if not isinstance(view, dict):
        return {}
    metrics = view.get("paper_metrics")
    if not isinstance(metrics, list):
        return {}
    return {
        str(item["metric"]): item
        for item in metrics
        if isinstance(item, dict) and isinstance(item.get("metric"), str)
    }


def lora_rslora_comparison_rows(  # noqa: C901, PLR0912
    tasks: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-run, per-metric rows from signed ordinary-LoRA/rsLoRA pairs."""
    task_list = list(tasks)
    ordinary: dict[tuple[str, str], dict[str, Any]] = {}
    for task in task_list:
        if task.get("pairing") is not None:
            continue
        key = (str(task.get("run_root")), str(task.get("experiment_id")))
        if key in ordinary:
            message = f"Duplicate ordinary task supplied for rsLoRA pairing: {key}"
            raise CollectionError(message)
        ordinary[key] = task

    rows: list[dict[str, Any]] = []
    for paired in task_list:
        pairing = paired.get("pairing")
        if not isinstance(pairing, dict):
            continue
        source_experiment_id = str(pairing.get("source_experiment_id", ""))
        source_root = str(pairing.get("source_matrix_root", ""))
        source = ordinary.get((source_root, source_experiment_id))
        for split in ("validation", "test"):
            source_metrics = _task_paper_metrics(
                source,
                split,
                selection_profile="lora_comparison",
            )
            paired_metrics = _task_paper_metrics(paired, split)
            for metric in sorted(set(source_metrics) | set(paired_metrics)):
                source_item = source_metrics.get(metric)
                paired_item = paired_metrics.get(metric)
                source_value = numeric(source_item.get("value")) if source_item else None
                paired_value = numeric(paired_item.get("value")) if paired_item else None
                descriptor = paired_item or source_item or {}
                higher_is_better = descriptor.get("higher_is_better")
                complete = source_value is not None and paired_value is not None
                raw_delta = paired_value - source_value if complete else None
                improvement_delta = (
                    raw_delta
                    if complete and higher_is_better is not False
                    else (-raw_delta if complete else None)
                )
                if source is None:
                    state = "missing_source_task"
                elif source.get("training_method") != "lora":
                    state = "invalid_source_training_method"
                elif paired.get("training_method") != "rslora":
                    state = "invalid_paired_training_method"
                elif source_value is None:
                    state = "missing_source_lora_comparison_metric"
                elif paired_value is None:
                    state = "missing_paired_metric"
                else:
                    state = "complete"
                rows.append(
                    {
                        "source_experiment_id": source_experiment_id,
                        "paired_experiment_id": paired.get("experiment_id"),
                        "benchmark": paired.get("benchmark"),
                        "run_index": paired.get("run_index"),
                        "source_training_method": source.get("training_method") if source else None,
                        "paired_training_method": paired.get("training_method"),
                        "split": split,
                        "metric": metric,
                        "unit": descriptor.get("unit"),
                        "higher_is_better": higher_is_better,
                        "source_value": source_value,
                        "paired_value": paired_value,
                        "raw_delta": raw_delta,
                        "improvement_delta": improvement_delta,
                        "state": state,
                    },
                )
    return sorted(
        rows,
        key=lambda row: (
            str(row["source_experiment_id"]),
            str(row["split"]),
            str(row["metric"]),
        ),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(json_safe(rows))
    temporary.replace(path)


def build_report(
    run_specs: Sequence[tuple[Path, str]],
    reference_path: Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    runs = []
    for run_root, runner in sorted(run_specs, key=lambda item: (item[1], project_path(item[0]))):
        run_tasks, run_info = collect_run(run_root, runner)
        tasks.extend(run_tasks)
        runs.append(run_info)
    inventory = paper_inventory()
    attach_inventory(tasks, inventory)
    tasks.sort(key=lambda task: task["paper_experiment_id"])
    aggregates = aggregate_metrics(tasks, inventory)
    references = load_references(reference_path)
    comparisons = compare_references(aggregates, references)
    lora_rslora_pairs = lora_rslora_comparison_rows(tasks)
    coverage = coverage_report(tasks, runs, inventory)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_policy": {
            "model_selection": "FTTrace acceptance on validation-visible feedback only",
            "test_pairing": "one post-selection test artifact whose signature matches the selected checkpoint",
            "independent_test_optimization": False,
            "legacy_compatibility": "none; search-time held-out payloads mark the trace protocol_polluted",
            "no_accepted_loop_fallback": "base model selected by validation, then evaluated once on test",
        },
        "reference_manifest": project_path(reference_path),
        "coverage": coverage,
        "tasks": tasks,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "lora_rslora_pairs": lora_rslora_pairs,
    }


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", report)
    write_json(output / "coverage.json", report["coverage"])
    write_csv(output / "tasks.csv", task_csv_rows(report["tasks"]), TASK_CSV_FIELDS)
    write_csv(output / "metrics.csv", all_metric_rows(report["tasks"]), METRIC_CSV_FIELDS)
    write_csv(output / "aggregates.csv", report["aggregates"], AGGREGATE_CSV_FIELDS)
    write_csv(output / "comparisons.csv", report["comparisons"], COMPARISON_CSV_FIELDS)
    lora_rslora_pairs = report.get("lora_rslora_pairs", [])
    write_json(output / "lora_rslora_pairs.json", lora_rslora_pairs)
    write_csv(output / "lora_rslora_pairs.csv", lora_rslora_pairs, LORA_RSLORA_CSV_FIELDS)


def collect_and_write_run(run_root: Path, runner: str, output: Path | None = None) -> dict[str, Any]:
    report = build_report([(run_root, runner)])
    write_report(output or run_root / "report", report)
    return report


def resolve_run(value: str, default_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.parent != Path():
        return path.resolve()
    return (default_root / path).resolve()


def main() -> int:
    args = parse_args()
    run_specs = [
        *((resolve_run(value, MATRIX_LOG_ROOT), "ft-agent") for value in args.matrix_run),
        *((resolve_run(value, BASE_LOG_ROOT), "base") for value in args.base_run),
    ]
    if not run_specs:
        message = "Provide at least one --matrix-run or --base-run"
        raise SystemExit(message)
    report = build_report(run_specs, args.reference)
    write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Collected {coverage['supplied_tasks']} tasks into {args.output}")
    print(json.dumps(coverage["paper_inventory_states"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
