"""Tests for strict final main-table rendering."""

from __future__ import annotations

from typing import Any

from reproduction.ft_agent.render_main_report import select_best_runs, validation_utility

EXPECTED_SELECTED_RUN = 2


def _view(mae: float, accuracy: float) -> dict[str, Any]:
    return {
        "paper_metrics": [
            {
                "metric": "mae",
                "label": "MAE",
                "value": mae,
                "higher_is_better": False,
                "unit": "absolute",
            },
            {
                "metric": "tanimoto_similarity",
                "label": "TS",
                "value": accuracy / 100,
                "higher_is_better": True,
                "unit": "ratio",
            },
            {
                "metric": "accuracy",
                "label": "Acc",
                "value": accuracy,
                "higher_is_better": True,
                "unit": "percent",
            },
        ],
    }


def _task(run_index: int, validation: tuple[float, float], test: tuple[float, float]) -> dict[str, Any]:
    return {
        "paper_experiment_id": f"ft-agent/main/chemcotbench_mol_und/run-{run_index}",
        "benchmark": "chemcotbench_mol_und",
        "run_index": run_index,
        "final": {"validation": _view(*validation), "test": _view(*test)},
    }


def test_best_run_uses_joint_validation_and_never_held_out_test() -> None:
    tasks = [
        _task(1, (0.10, 80.0), (0.01, 100.0)),
        _task(2, (0.20, 100.0), (0.50, 10.0)),
        _task(3, (0.40, 60.0), (0.60, 0.0)),
    ]

    utilities = validation_utility(tasks)
    assert utilities[tasks[1]["paper_experiment_id"]] > utilities[tasks[0]["paper_experiment_id"]]
    selected = select_best_runs(tasks)

    assert len(selected) == 1
    assert selected[0]["task"]["run_index"] == EXPECTED_SELECTED_RUN


def test_validation_utility_honors_lower_is_better_and_tie_breaks_by_run() -> None:
    tasks = [
        _task(2, (0.10, 100.0), (0.50, 0.0)),
        _task(1, (0.10, 100.0), (0.01, 100.0)),
        _task(3, (0.50, 0.0), (0.01, 100.0)),
    ]

    utilities = validation_utility(tasks)
    assert utilities[tasks[0]["paper_experiment_id"]] == 1.0
    assert utilities[tasks[1]["paper_experiment_id"]] == 1.0
    selected = select_best_runs(tasks)

    assert selected[0]["task"]["run_index"] == 1
