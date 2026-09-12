"""Regression tests for deterministic FT-Dojo result collection."""

from __future__ import annotations

import csv
import io
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from reproduction.ft_agent.collect_results import (
    AGGREGATE_CSV_FIELDS,
    COMPARISON_CSV_FIELDS,
    DEFAULT_REFERENCE,
    METRIC_CSV_FIELDS,
    TASK_CSV_FIELDS,
    aggregate_metrics,
    collect_base_task,
    collect_ft_task,
    derive_paper_metrics,
    latest_session_path,
    load_latest_session,
    load_references,
    lora_rslora_comparison_rows,
    select_sota_index,
    write_report,
)
from reproduction.ft_agent.final_test_protocol import (
    FINAL_TEST_SCHEMA_VERSION,
    TEST_RANGE,
)
from reproduction.ft_agent.run_validation_sweep import (
    SweepTarget,
    candidate_set_signature,
    discover_candidates,
)
from reproduction.ft_agent.validation_selection import make_selection_artifact

EXPECTED_REFERENCE_ROWS = 118
EXPECTED_REPEAT_COUNT = 3
EXPECTED_SAMPLE_MEAN = 20.0
EXPECTED_SAMPLE_STD = 10.0
SELECTED_HISTORY_INDEX = 2


def _score(value: float) -> dict[str, Any]:
    return {"accuracy_summary": {"synthetic": {"accuracy": value}}}


def _experiment(validation: float, test: float | None, workspace_path: Path = Path()) -> SimpleNamespace:
    result = {"benchmark": _score(validation)}
    if test is not None:
        result["benchmark_test"] = _score(test)
    return SimpleNamespace(
        experiment_workspace=SimpleNamespace(
            workspace_path=workspace_path,
            running_info=SimpleNamespace(result=result),
        ),
    )


def _node(validation: float, test: float | None, *, accepted: bool) -> tuple[SimpleNamespace, SimpleNamespace]:
    return _experiment(validation, test), SimpleNamespace(decision=accepted)


def _scenario(validation: float, test: float) -> SimpleNamespace:
    return SimpleNamespace(
        baseline_benchmark_score=_score(validation),
        baseline_benchmark_score_test=_score(test),
    )


def _write_session(trace_path: Path, loop_id: int, step: int, payload: Any) -> Path:
    path = trace_path / "__session__" / str(loop_id) / f"{step}_session.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def _ft_status() -> dict[str, Any]:
    return {
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "planner": "gpt-5.2",
        "data_limit": 2000,
        "formal_expected_samples": 2000,
        "formal_training_method": "lora",
        "training_policy": "paper",
        "state": "succeeded",
        "benchmark_dataset_path": "pinned/aime25",
    }


def _write_validation_selection(
    task_root: Path,
    model_path: Path,
    validation: float,
    formal_session_writer: Any,
    *,
    checkpoint_step: int | None = None,
) -> tuple[str, str]:
    status = _ft_status()
    workspace_id = model_path.parent.name
    formal_step = checkpoint_step if checkpoint_step is not None else 4
    formal_session_writer(task_root, {workspace_id: formal_step})
    target = SweepTarget(
        experiment_id=status["experiment_id"],
        benchmark=status["benchmark"],
        model=status["model"],
        benchmark_dataset_path=status["benchmark_dataset_path"],
        task_root=task_root,
        expected_samples=status["formal_expected_samples"],
        training_policy=status["training_policy"],
    )
    discovered = discover_candidates(
        target,
        include_baseline=False,
        include_final_outputs=False,
    )
    assert len(discovered) == 1
    discovered_candidate = discovered[0]
    signature = discovered_candidate.selection_signature
    candidate = {
        **discovered_candidate.identity(),
        "accuracy_summary": _score(validation)["accuracy_summary"],
        "paper_metrics": [
            {
                "metric": "accuracy",
                "label": "Accuracy",
                "value": validation,
                "higher_is_better": True,
                "unit": "%",
                "formula": "reported accuracy",
            },
        ],
    }
    artifact = make_selection_artifact(
        experiment_id=status["experiment_id"],
        benchmark="aime25",
        model=status["model"],
        benchmark_dataset_path=status["benchmark_dataset_path"],
        search_status={"state": "succeeded"},
        candidate_set_signature=candidate_set_signature(discovered),
        candidates=[candidate],
        expected_samples=status["formal_expected_samples"],
        include_baseline=False,
        include_final_outputs=False,
    )
    (task_root / "validation_selection.json").write_text(json.dumps(artifact), encoding="utf-8")
    return signature, artifact["artifact_signature"]


def _paper_values(record: dict[str, Any], split: str) -> dict[str, float]:
    view = record["final"][split]
    assert view is not None
    return {metric["metric"]: metric["value"] for metric in view["paper_metrics"]}


def _derived_values(benchmark: str, summary: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric["metric"]: metric["value"] for metric in derive_paper_metrics(benchmark, summary)}


def test_lora_rslora_comparison_uses_dedicated_lora_view_when_main_selected_base() -> None:
    def view(value: float) -> dict[str, Any]:
        return {
            "paper_metrics": [
                {
                    "metric": "accuracy",
                    "value": value,
                    "unit": "%",
                    "higher_is_better": True,
                },
            ],
        }

    source = {
        "run_root": "/matrix/source",
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "run_index": 1,
        "training_method": "lora",
        "pairing": None,
        # The main profile selected Base, while the comparison profile is
        # constrained to the ordinary-LoRA checkpoint.
        "final": {"validation": view(5.0), "test": view(6.0)},
        "lora_comparison": {"final": {"validation": view(70.0), "test": view(71.0)}},
    }
    paired = {
        "run_root": "/matrix/paired",
        "experiment_id": "main/aime25/run-1",
        "benchmark": "aime25",
        "run_index": 1,
        "training_method": "rslora",
        "pairing": {
            "source_matrix_root": "/matrix/source",
            "source_experiment_id": "main/aime25/run-1",
        },
        "final": {"validation": view(80.0), "test": view(82.0)},
    }

    rows = lora_rslora_comparison_rows([source, paired])

    assert [(row["split"], row["source_value"], row["paired_value"]) for row in rows] == [
        ("test", 71.0, 82.0),
        ("validation", 70.0, 80.0),
    ]
    assert all(row["state"] == "complete" for row in rows)


def test_latest_session_uses_numeric_loop_and_step_order(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace"
    _write_session(trace_path, 2, 100, SimpleNamespace(marker="loop-2"))
    _write_session(trace_path, 10, 2, SimpleNamespace(marker="step-2"))
    expected = _write_session(trace_path, 10, 10, SimpleNamespace(marker="step-10"))

    assert latest_session_path(trace_path) == expected
    loaded_path, session = load_latest_session(trace_path)
    assert loaded_path == expected
    assert session.marker == "step-10"


def test_ft_selection_marks_legacy_test_trace_as_protocol_polluted(tmp_path: Path) -> None:
    trace = SimpleNamespace(
        hist=[
            _node(10.0, 11.0, accepted=True),
            _node(90.0, 999.0, accepted=True),
            _node(40.0, 41.0, accepted=True),
            _node(100.0, 1000.0, accepted=False),
        ],
        dag_parent=[(0,), (0,), (0,), (2,)],
        current_selection=(-1,),
        idx2loop_id={0: 0, 1: 1, 2: 2, 3: 3},
        scen=_scenario(5.0, 6.0),
    )
    task_root = tmp_path / "task"
    _write_session(task_root / "trace", 3, 1, SimpleNamespace(trace=trace))

    assert select_sota_index(trace) == SELECTED_HISTORY_INDEX
    record = collect_ft_task(task_root, _ft_status())

    assert record["selection"] == {
        "source": "accepted_loop",
        "history_index": 2,
        "loop_id": 2,
        "test_used_for_selection": False,
    }
    assert record["loops"][SELECTED_HISTORY_INDEX]["validation"] is not None
    assert record["final"]["validation"] is None
    assert record["final"]["test"] is None
    assert record["state"] == "protocol_polluted"
    assert all(loop["test"] is None for loop in record["loops"])
    assert record["baseline"]["test"] is None
    assert record["final_test"]["state"] == "blocked_protocol_polluted"
    assert {entry["code"] for entry in record["diagnostics"]} >= {"protocol_polluted", "missing_final_test"}


def test_ft_collection_does_not_fall_back_when_validation_selection_is_missing(tmp_path: Path) -> None:
    trace = SimpleNamespace(
        hist=[_node(100.0, None, accepted=False)],
        dag_parent=[(0,)],
        current_selection=(-1,),
        idx2loop_id={0: 0},
        scen=SimpleNamespace(
            baseline_benchmark_score=_score(12.0),
            baseline_benchmark_score_test={},
        ),
    )
    task_root = tmp_path / "task"
    _write_session(task_root / "trace", 0, 1, SimpleNamespace(trace=trace))

    record = collect_ft_task(task_root, _ft_status())

    assert record["selection"] == {
        "source": None,
        "history_index": None,
        "loop_id": None,
        "candidate_id": None,
        "checkpoint_step": None,
        "test_used_for_selection": False,
    }
    assert record["final"]["validation"] is None
    assert record["final"]["test"] is None
    assert record["state"] == "succeeded"
    assert {entry["code"] for entry in record["diagnostics"]} >= {
        "missing_validation_selection",
        "missing_final_test",
    }


@pytest.mark.parametrize(
    ("evaluation_mode", "expected_accuracy"),
    [("post_selection", 77.0), ("legacy_same_node_reuse", None)],
)
def test_ft_collection_accepts_only_clean_post_selection_artifact(
    tmp_path: Path,
    evaluation_mode: str,
    expected_accuracy: float | None,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "task"
    workspace = task_root / "workspace" / "selected"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text(
        json.dumps({"r": 8, "peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    (output / "adapter_model.safetensors").write_bytes(b"selected model")
    experiment = SimpleNamespace(
        experiment_workspace=SimpleNamespace(
            workspace_path=workspace,
            running_info=SimpleNamespace(result={"benchmark": _score(44.0)}),
        ),
    )
    trace = SimpleNamespace(
        hist=[(experiment, SimpleNamespace(decision=True))],
        dag_parent=[(0,)],
        current_selection=(-1,),
        idx2loop_id={0: 7},
        scen=SimpleNamespace(
            baseline_benchmark_score=_score(5.0),
            baseline_benchmark_score_test={},
        ),
    )
    _write_session(task_root / "trace", 7, 1, SimpleNamespace(trace=trace))
    signature, validation_artifact_signature = _write_validation_selection(
        task_root,
        output,
        44.0,
        formal_session_writer,
    )
    (task_root / "final_test.json").write_text(
        json.dumps(
            {
                "schema_version": FINAL_TEST_SCHEMA_VERSION,
                "experiment_id": _ft_status()["experiment_id"],
                "benchmark": "aime25",
                "model": _ft_status()["model"],
                "state": "succeeded",
                "evaluation_mode": evaluation_mode,
                "selection": {
                    "signature": signature,
                    "validation_selection_artifact_signature": validation_artifact_signature,
                },
                "test_range": TEST_RANGE,
                "benchmark_dataset_path": _ft_status()["benchmark_dataset_path"],
                "result": _score(77.0),
            },
        ),
        encoding="utf-8",
    )

    record = collect_ft_task(task_root, _ft_status())

    assert _paper_values(record, "validation") == {"accuracy": 44.0}
    assert record["training_method"] == "lora"
    if expected_accuracy is None:
        assert record["final"]["test"] is None
        assert record["final_test"]["source"] is None
        assert {entry["code"] for entry in record["diagnostics"]} >= {
            "unusable_final_test_artifact",
            "missing_final_test",
        }
    else:
        assert _paper_values(record, "test") == {"accuracy": expected_accuracy}
        assert record["final_test"]["source"] == "post_selection"


def test_ft_collection_rejects_stale_final_test_artifact(
    tmp_path: Path,
    formal_session_writer: Any,
) -> None:
    task_root = tmp_path / "task"
    workspace = task_root / "workspace" / "selected"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "use_rslora": False, "use_dora": False}),
        encoding="utf-8",
    )
    (output / "adapter_model.safetensors").write_bytes(b"selected model")
    experiment = SimpleNamespace(
        experiment_workspace=SimpleNamespace(
            workspace_path=workspace,
            running_info=SimpleNamespace(result={"benchmark": _score(44.0)}),
        ),
    )
    trace = SimpleNamespace(
        hist=[(experiment, SimpleNamespace(decision=True))],
        dag_parent=[(0,)],
        current_selection=(-1,),
        idx2loop_id={0: 7},
        scen=SimpleNamespace(baseline_benchmark_score=_score(5.0)),
    )
    _write_session(task_root / "trace", 7, 1, SimpleNamespace(trace=trace))
    _, validation_artifact_signature = _write_validation_selection(
        task_root,
        output,
        44.0,
        formal_session_writer,
    )
    (task_root / "final_test.json").write_text(
        json.dumps(
            {
                "schema_version": FINAL_TEST_SCHEMA_VERSION,
                "experiment_id": _ft_status()["experiment_id"],
                "benchmark": "aime25",
                "model": _ft_status()["model"],
                "state": "succeeded",
                "evaluation_mode": "post_selection",
                "selection": {
                    "signature": "stale",
                    "validation_selection_artifact_signature": validation_artifact_signature,
                },
                "test_range": TEST_RANGE,
                "benchmark_dataset_path": _ft_status()["benchmark_dataset_path"],
                "result": _score(99.0),
            },
        ),
        encoding="utf-8",
    )

    record = collect_ft_task(task_root, _ft_status())

    assert record["final"]["test"] is None
    assert {entry["code"] for entry in record["diagnostics"]} >= {
        "unusable_final_test_artifact",
        "missing_final_test",
    }


def test_base_task_parses_opencompass_csv_rows(tmp_path: Path) -> None:
    def rows(value: float) -> list[dict[str, str]]:
        content = f"dataset,version,metric,mode,base-qwen\naime2025,1,accuracy,gen,{value}\n"
        return list(csv.DictReader(io.StringIO(content)))

    status = {
        "experiment_id": "base/7b/aime25",
        "benchmark": "aime25",
        "target_model": "Qwen/Qwen2.5-7B-Instruct",
        "state": "succeeded",
        "splits": {
            "validation": {"summary": {"rows": rows(21.5)}},
            "test": {"summary": {"rows": rows(43.0)}},
        },
    }

    record = collect_base_task(tmp_path / "base-task", status)

    assert record["selection"]["source"] == "base_model"
    assert _paper_values(record, "validation") == {"accuracy": 21.5}
    assert _paper_values(record, "test") == {"accuracy": 43.0}


def test_chemcot_molecule_understanding_formulas() -> None:
    summary = {
        "mol_und_fg_count": {"mae": 2.0},
        "mol_und_ring_count": {"mae": 4.0},
        "mol_und_murcko_scaffold": {"tanimoto_similarity_larger_means_better": 0.7},
        "mol_und_equivalence": {"accuracy": 80.0},
        "mol_und_ring_system_scaffold": {"accuracy": 60.0},
        "mol_und_unrelated": {"mae": 100.0, "accuracy": 100.0},
    }

    assert _derived_values("chemcotbench_mol_und", summary) == {
        "mae": 3.0,
        "tanimoto_similarity": 0.7,
        "accuracy": 70.0,
    }


def test_chemcot_molecule_edit_and_optimization_formulas() -> None:
    edit_summary = {
        "mol_edit_add": {"correct_rate": 10.0},
        "mol_edit_delete": {"correct_rate": 20.0},
        "mol_edit_substitute": {"correct_rate": 30.0},
    }
    optimization_summary = {
        f"mol_opt_view_{index}": {
            "success_rate": float(index),
            "valid_smiles_rate": float(index * 10),
        }
        for index in range(1, 7)
    }

    assert _derived_values("chemcotbench_mol_edit", edit_summary) == {"accuracy": 20.0}
    assert _derived_values("chemcotbench_mol_opt", optimization_summary) == {
        "success_rate": 3.5,
        "valid_smiles_rate": 35.0,
    }


def test_chemcot_reaction_formulas_exclude_unreported_views() -> None:
    summary = {
        "reaction_fs": {"morgan_sims": 10.0},
        "reaction_retro": {"morgan_sims": 20.0},
        "reaction_nepp": {"morgan_sims": 30.0},
        "reaction_mechsel": {"morgan_sims": 999.0, "accuracy": 75.0},
        "reaction_other": {"morgan_sims": 999.0, "accuracy": 100.0},
    }

    assert _derived_values("chemcotbench_reaction", summary) == {
        "fingerprint_similarity": 20.0,
        "accuracy": 75.0,
    }


def test_default_reference_expands_to_unique_published_rows() -> None:
    references = load_references(DEFAULT_REFERENCE)
    keys = {
        (row["paper_group"], row["benchmark"], row["split"], row["metric"])
        for row in references
    }

    assert len(references) == EXPECTED_REFERENCE_ROWS
    assert len(keys) == len(references)


def test_repeated_runs_use_sample_standard_deviation() -> None:
    def task(run_index: int, value: float) -> dict[str, Any]:
        return {
            "paper_group": "synthetic-group",
            "benchmark": "aime25",
            "target_model": "synthetic-model",
            "planner": "synthetic-planner",
            "data_limit": 2000,
            "paper_experiment_id": f"synthetic/run-{run_index}",
            "final": {
                "validation": {
                    "paper_metrics": [
                        {
                            "metric": "accuracy",
                            "label": "Acc",
                            "unit": "percent",
                            "higher_is_better": True,
                            "value": value,
                        },
                    ],
                },
                "test": None,
            },
        }

    tasks = [task(1, 10.0), task(2, 20.0), task(3, 30.0)]
    inventory = [
        {"paper_group": "synthetic-group", "benchmark": "aime25"}
        for _ in range(EXPECTED_REPEAT_COUNT)
    ]

    aggregates = aggregate_metrics(tasks, inventory)

    assert len(aggregates) == 1
    assert aggregates[0]["n"] == EXPECTED_REPEAT_COUNT
    assert aggregates[0]["expected_n"] == EXPECTED_REPEAT_COUNT
    assert aggregates[0]["mean"] == EXPECTED_SAMPLE_MEAN
    assert aggregates[0]["std"] == EXPECTED_SAMPLE_STD


@pytest.mark.parametrize(
    ("filename", "fieldnames"),
    [
        ("tasks.csv", TASK_CSV_FIELDS),
        ("metrics.csv", METRIC_CSV_FIELDS),
        ("aggregates.csv", AGGREGATE_CSV_FIELDS),
        ("comparisons.csv", COMPARISON_CSV_FIELDS),
    ],
)
def test_empty_report_csvs_retain_headers(tmp_path: Path, filename: str, fieldnames: tuple[str, ...]) -> None:
    output = tmp_path / "report"
    write_report(output, {"coverage": {}, "tasks": [], "aggregates": [], "comparisons": []})

    with (output / filename).open(encoding="utf-8", newline="") as stream:
        assert next(csv.reader(stream)) == list(fieldnames)
