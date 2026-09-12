# ruff: noqa: EM101, EM102, TRY003
"""Validation-only checkpoint ranking and signed selection artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__:
    from .collect_results import PAPER_METRICS, json_safe
    from .final_test_protocol import model_artifact_type, selection_signature
else:
    from collect_results import PAPER_METRICS, json_safe
    from final_test_protocol import model_artifact_type, selection_signature

VALIDATION_RANGE = "[:min(100, len(index_list)//2)]"
MAIN_SELECTION_PROFILE = "main"
LORA_COMPARISON_SELECTION_PROFILE = "lora_comparison"
SELECTION_PROFILES = (MAIN_SELECTION_PROFILE, LORA_COMPARISON_SELECTION_PROFILE)
VALIDATION_SELECTION_FILE = "validation_selection.json"
LORA_COMPARISON_VALIDATION_SELECTION_FILE = "lora_comparison_validation_selection.json"
VALIDATION_SWEEP_DIRECTORY = "validation_sweep"
LORA_COMPARISON_VALIDATION_SWEEP_DIRECTORY = "lora_comparison_validation_sweep"
VALIDATION_SWEEP_SNAPSHOT_FILE = "validation_sweep.json"
LORA_COMPARISON_VALIDATION_SWEEP_SNAPSHOT_FILE = "lora_comparison_validation_sweep.json"
FINAL_TEST_FILE = "final_test.json"
LORA_COMPARISON_FINAL_TEST_FILE = "lora_comparison_final_test.json"
VALIDATION_SELECTION_SCHEMA_VERSION = 1
SELECTION_RULE = "direction-aware mean average ordinal rank"


class ValidationSelectionError(RuntimeError):
    """Raised when checkpoint evidence cannot support a deterministic selection."""


def normalize_selection_profile(profile: str | None = None) -> str:
    """Return a controlled checkpoint-selection profile name."""
    value = MAIN_SELECTION_PROFILE if profile is None else str(profile).strip().lower().replace("-", "_")
    if value not in SELECTION_PROFILES:
        raise ValueError(f"Selection profile must be one of {', '.join(SELECTION_PROFILES)}; got {profile!r}")
    return value


def validation_selection_file(profile: str | None = None) -> str:
    return (
        VALIDATION_SELECTION_FILE
        if normalize_selection_profile(profile) == MAIN_SELECTION_PROFILE
        else LORA_COMPARISON_VALIDATION_SELECTION_FILE
    )


def validation_sweep_directory(profile: str | None = None) -> str:
    return (
        VALIDATION_SWEEP_DIRECTORY
        if normalize_selection_profile(profile) == MAIN_SELECTION_PROFILE
        else LORA_COMPARISON_VALIDATION_SWEEP_DIRECTORY
    )


def validation_sweep_snapshot_file(profile: str | None = None) -> str:
    return (
        VALIDATION_SWEEP_SNAPSHOT_FILE
        if normalize_selection_profile(profile) == MAIN_SELECTION_PROFILE
        else LORA_COMPARISON_VALIDATION_SWEEP_SNAPSHOT_FILE
    )


def final_test_file(profile: str | None = None) -> str:
    return (
        FINAL_TEST_FILE
        if normalize_selection_profile(profile) == MAIN_SELECTION_PROFILE
        else LORA_COMPARISON_FINAL_TEST_FILE
    )


def final_test_workspace_directory(profile: str | None = None) -> str:
    active = normalize_selection_profile(profile)
    return "final_test_workspaces" if active == MAIN_SELECTION_PROFILE else f"{active}_final_test_workspaces"


def final_test_spec_file(profile: str | None = None) -> str:
    active = normalize_selection_profile(profile)
    return "final_test_spec.json" if active == MAIN_SELECTION_PROFILE else f"{active}_final_test_spec.json"


def final_test_log_file(profile: str | None = None) -> str:
    active = normalize_selection_profile(profile)
    return "final_test.console.log" if active == MAIN_SELECTION_PROFILE else f"{active}_final_test.console.log"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metric_map(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in candidate.get("paper_metrics", []):
        if not isinstance(item, dict) or not isinstance(item.get("metric"), str):
            continue
        try:
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        result[item["metric"]] = {
            "metric": item["metric"],
            "label": item.get("label", item["metric"]),
            "value": value,
            "higher_is_better": bool(item.get("higher_is_better", True)),
            "unit": item.get("unit"),
            "formula": item.get("formula"),
        }
    return result


def _average_ordinal_ranks(values: list[float], *, higher_is_better: bool) -> list[float]:
    """Return 1-based average ranks, assigning tied values their mean position."""
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=higher_is_better)
    result = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        average_rank = ((offset + 1) + end) / 2.0
        for position in range(offset, end):
            result[order[position]] = average_rank
        offset = end
    return result


def rank_candidates(benchmark: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank complete validation candidates using only paper-primary metrics."""
    if not candidates:
        raise ValidationSelectionError("No successful validation candidates are available")
    expected = tuple(PAPER_METRICS.get(benchmark, ()))
    if not expected:
        raise ValidationSelectionError(f"No paper-primary metric specification for {benchmark!r}")

    maps = [_metric_map(candidate) for candidate in candidates]
    for index, metrics in enumerate(maps):
        missing = [metric for metric in expected if metric not in metrics]
        if missing:
            candidate_id = candidates[index].get("candidate_id", index)
            raise ValidationSelectionError(
                f"Candidate {candidate_id!r} is missing paper-primary metrics: {', '.join(missing)}",
            )

    metric_ranks: dict[str, list[float]] = {}
    metric_specs: dict[str, dict[str, Any]] = {}
    for metric in expected:
        directions = {metrics[metric]["higher_is_better"] for metrics in maps}
        if len(directions) != 1:
            raise ValidationSelectionError(f"Metric direction differs across candidates: {metric}")
        higher_is_better = directions.pop()
        metric_specs[metric] = {
            key: maps[0][metric].get(key)
            for key in ("metric", "label", "higher_is_better", "unit", "formula")
        }
        metric_ranks[metric] = _average_ordinal_ranks(
            [metrics[metric]["value"] for metrics in maps],
            higher_is_better=higher_is_better,
        )

    ranked: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        ranks = {metric: metric_ranks[metric][index] for metric in expected}
        ranked.append(
            {
                **candidate,
                "selection_metrics": [maps[index][metric] for metric in expected],
                "ordinal_ranks": ranks,
                "mean_ordinal_rank": statistics.fmean(ranks.values()),
                "worst_ordinal_rank": max(ranks.values()),
                "metric_specifications": [metric_specs[metric] for metric in expected],
            },
        )

    # The first two keys implement the declared joint rule.  candidate_id is a
    # stable, metric-independent tie break and cannot leak held-out information.
    ranked.sort(
        key=lambda item: (
            item["mean_ordinal_rank"],
            item["worst_ordinal_rank"],
            str(item["candidate_id"]),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["selection_rank"] = rank
    return ranked


def make_selection_artifact(
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    benchmark_dataset_path: str,
    search_status: dict[str, Any],
    candidate_set_signature: str,
    candidates: list[dict[str, Any]],
    expected_samples: int,
    include_baseline: bool,
    include_final_outputs: bool,
    candidate_filter: str | None = None,
    pairing_artifact_signature: str | None = None,
    selection_profile: str = MAIN_SELECTION_PROFILE,
) -> dict[str, Any]:
    """Build a signed artifact only after the search itself has succeeded."""
    selection_profile = normalize_selection_profile(selection_profile)
    if search_status.get("state") != "succeeded":
        raise ValidationSelectionError(
            f"Search is {search_status.get('state', 'unknown')!r}; final checkpoint selection is premature",
        )
    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int) or expected_samples < 1:
        raise ValidationSelectionError("Formal expected sample count must be a positive integer")
    if pairing_artifact_signature is not None and re.fullmatch(r"[0-9a-f]{64}", pairing_artifact_signature) is None:
        raise ValidationSelectionError("rsLoRA pairing artifact signature is invalid")
    if selection_profile == LORA_COMPARISON_SELECTION_PROFILE:
        if include_baseline:
            raise ValidationSelectionError("LoRA comparison selection must exclude the unchanged baseline")
        invalid = [
            str(candidate.get("candidate_id", "unknown"))
            for candidate in candidates
            if candidate.get("source") == "baseline"
            or model_artifact_type(Path(str(candidate.get("model_path", ""))).resolve()) != "lora"
        ]
        if invalid:
            raise ValidationSelectionError(
                "LoRA comparison selection contains a baseline or non-LoRA candidate: " + ", ".join(invalid),
            )
    ranked = rank_candidates(benchmark, candidates)
    selected = ranked[0]
    selected_model_path = Path(selected["model_path"]).resolve()
    expected_signature = selection_signature(
        "validation_sweep",
        None,
        selected.get("checkpoint_step"),
        selected_model_path,
    )
    if selected.get("selection_signature") != expected_signature:
        raise ValidationSelectionError("Selected checkpoint identity changed after validation")

    payload: dict[str, Any] = {
        "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
        "state": "succeeded",
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "model": model,
        "benchmark_dataset_path": benchmark_dataset_path,
        "expected_samples": expected_samples,
        "validation_range": VALIDATION_RANGE,
        "held_out_test_used": False,
        "selection_profile": selection_profile,
        "pairing_artifact_signature": pairing_artifact_signature,
        "search": {
            "state": search_status["state"],
            "started_at": search_status.get("started_at"),
            "finished_at": search_status.get("finished_at"),
        },
        "candidate_set_signature": candidate_set_signature,
        "candidate_count": len(ranked),
        "candidate_discovery": {
            "include_baseline": include_baseline,
            "include_final_outputs": include_final_outputs,
            "candidate_filter": candidate_filter,
        },
        "selection_rule": {
            "name": SELECTION_RULE,
            "primary_metrics": list(PAPER_METRICS[benchmark]),
            "tie_break": "lowest worst ordinal rank, then stable candidate_id",
        },
        "selection": {
            key: selected.get(key)
            for key in (
                "candidate_id",
                "source",
                "workspace_id",
                "checkpoint_step",
                "model_path",
                "selection_signature",
                "formal_training_evidence_signature",
                "accuracy_summary",
                "selection_metrics",
                "ordinal_ranks",
                "mean_ordinal_rank",
                "worst_ordinal_rank",
                "model_relocation",
            )
        },
        "candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "candidate_id",
                    "source",
                    "workspace_id",
                    "checkpoint_step",
                    "model_path",
                    "selection_signature",
                    "formal_training_evidence_signature",
                    "accuracy_summary",
                    "selection_metrics",
                    "ordinal_ranks",
                    "mean_ordinal_rank",
                    "worst_ordinal_rank",
                    "selection_rank",
                    "model_relocation",
                )
            }
            for candidate in ranked
        ],
    }
    signed_content = {key: value for key, value in payload.items() if key not in {"created_at", "artifact_signature"}}
    payload["artifact_signature"] = _canonical_sha256(signed_content)
    return json_safe(payload)


def validate_selection_artifact(
    artifact: dict[str, Any],
    *,
    experiment_id: str,
    benchmark: str,
    model: str,
    selection_profile: str = MAIN_SELECTION_PROFILE,
) -> dict[str, Any]:
    """Validate a selection artifact and return its selected checkpoint."""
    selection_profile = normalize_selection_profile(selection_profile)
    expected_fields = {
        "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
        "state": "succeeded",
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "model": model,
        "validation_range": VALIDATION_RANGE,
        "held_out_test_used": False,
        "selection_profile": selection_profile,
    }
    mismatches = [key for key, value in expected_fields.items() if artifact.get(key) != value]
    if mismatches:
        raise ValidationSelectionError("Incompatible validation selection fields: " + ", ".join(mismatches))
    if artifact.get("search", {}).get("state") != "succeeded":
        raise ValidationSelectionError("Validation selection was finalized before search success")
    pairing_signature = artifact.get("pairing_artifact_signature")
    if pairing_signature is not None and (
        not isinstance(pairing_signature, str) or re.fullmatch(r"[0-9a-f]{64}", pairing_signature) is None
    ):
        raise ValidationSelectionError("Validation selection has an invalid rsLoRA pairing signature")
    signed_content = {
        key: value for key, value in artifact.items() if key not in {"created_at", "artifact_signature"}
    }
    if artifact.get("artifact_signature") != _canonical_sha256(signed_content):
        raise ValidationSelectionError("Validation selection artifact signature does not match its contents")
    selection = artifact.get("selection")
    if not isinstance(selection, dict):
        raise ValidationSelectionError("Validation selection has no selected checkpoint")
    model_path = Path(str(selection.get("model_path", ""))).resolve()
    expected_signature = selection_signature(
        "validation_sweep",
        None,
        selection.get("checkpoint_step"),
        model_path,
    )
    if selection.get("selection_signature") != expected_signature:
        raise ValidationSelectionError("Selected checkpoint files changed after selection")
    if selection_profile == LORA_COMPARISON_SELECTION_PROFILE:
        discovery = artifact.get("candidate_discovery")
        if not isinstance(discovery, dict) or discovery.get("include_baseline") is not False:
            raise ValidationSelectionError("LoRA comparison selection did not lock baseline exclusion")
        if selection.get("source") == "baseline" or model_artifact_type(model_path) != "lora":
            raise ValidationSelectionError("LoRA comparison selected a baseline or non-LoRA artifact")
    return selection
