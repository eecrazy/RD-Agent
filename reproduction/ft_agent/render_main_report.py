#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Render a strict Chinese FT-Dojo main-table report from collected results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if __package__:
    from .collect_results import PAPER_METRICS
else:
    from collect_results import PAPER_METRICS

BASE_GROUP = "base-7b"
FT_GROUP = "ft-agent-main"
EXPECTED_TASK_COUNTS = {BASE_GROUP: 13, FT_GROUP: 39}
EXPECTED_AGGREGATE_COUNTS = {BASE_GROUP: 1, FT_GROUP: 3}
FT_RUNS = EXPECTED_AGGREGATE_COUNTS[FT_GROUP]
TOTAL_TASKS = sum(EXPECTED_TASK_COUNTS.values())
SPLITS = ("validation", "test")


class MainReportError(RuntimeError):
    """Raised when a collected report is incomplete or protocol-unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Collected results.json")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    return parser.parse_args()


def require(condition: object, message: str) -> None:
    if not condition:
        raise MainReportError(message)


def finite_number(value: Any, context: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Missing number: {context}")
    result = float(value)
    require(math.isfinite(result), f"Non-finite number: {context}")
    return result


def paper_metric_map(task: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    final = task.get("final")
    require(isinstance(final, dict), f"Missing final result: {task.get('paper_experiment_id')}")
    view = final.get(split)
    require(isinstance(view, dict), f"Missing {split}: {task.get('paper_experiment_id')}")
    metrics = view.get("paper_metrics")
    require(isinstance(metrics, list), f"Missing {split} paper metrics: {task.get('paper_experiment_id')}")
    result: dict[str, dict[str, Any]] = {}
    for item in metrics:
        require(isinstance(item, dict) and isinstance(item.get("metric"), str), "Malformed paper metric")
        metric = item["metric"]
        require(metric not in result, f"Duplicate paper metric {metric}: {task.get('paper_experiment_id')}")
        finite_number(item.get("value"), f"{task.get('paper_experiment_id')} {split} {metric}")
        require(isinstance(item.get("higher_is_better"), bool), f"Missing metric direction: {metric}")
        result[metric] = item
    return result


def validate_tasks(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tasks = report.get("tasks")
    require(isinstance(tasks, list), "Report has no task list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        require(isinstance(task, dict), "Malformed task record")
        group = task.get("paper_group")
        if group not in EXPECTED_TASK_COUNTS:
            continue
        experiment_id = str(task.get("paper_experiment_id"))
        require(task.get("state") == "succeeded", f"Task is not succeeded: {experiment_id}")
        require(task.get("diagnostics") == [], f"Task has diagnostics: {experiment_id}")
        selection = task.get("selection")
        require(isinstance(selection, dict), f"Task has no selection: {experiment_id}")
        require(selection.get("test_used_for_selection") is False, f"Test-informed selection: {experiment_id}")
        final_test = task.get("final_test")
        require(
            isinstance(final_test, dict) and final_test.get("state") == "succeeded",
            f"Task has no successful final test: {experiment_id}",
        )
        benchmark = task.get("benchmark")
        require(benchmark in PAPER_METRICS, f"Unknown main-table benchmark: {benchmark}")
        for split in SPLITS:
            found = paper_metric_map(task, split)
            require(
                tuple(found) == PAPER_METRICS[benchmark],
                f"Unexpected {split} metrics for {experiment_id}: {tuple(found)}",
            )
        grouped[group].append(task)
    for group, expected in EXPECTED_TASK_COUNTS.items():
        require(len(grouped[group]) == expected, f"Expected {expected} {group} tasks, found {len(grouped[group])}")
    for benchmark in PAPER_METRICS:
        base_count = sum(task["benchmark"] == benchmark for task in grouped[BASE_GROUP])
        ft_count = sum(task["benchmark"] == benchmark for task in grouped[FT_GROUP])
        require(base_count == 1, f"Expected one Base-7B task for {benchmark}, found {base_count}")
        require(ft_count == FT_RUNS, f"Expected three FT-Agent tasks for {benchmark}, found {ft_count}")
    return grouped


def keyed_rows(rows: Any, name: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    require(isinstance(rows, list), f"Report has no {name} list")
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("paper_group") not in EXPECTED_TASK_COUNTS:
            continue
        key = tuple(str(row.get(field)) for field in ("paper_group", "benchmark", "split", "metric"))
        require(key not in result, f"Duplicate {name} row: {key}")
        result[key] = row
    return result


def validate_summary_rows(
    report: dict[str, Any],
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    aggregates = keyed_rows(report.get("aggregates"), "aggregates")
    comparisons = keyed_rows(report.get("comparisons"), "comparisons")
    for group, expected_n in EXPECTED_AGGREGATE_COUNTS.items():
        for benchmark, metrics in PAPER_METRICS.items():
            for split in SPLITS:
                for metric in metrics:
                    key = (group, benchmark, split, metric)
                    require(key in aggregates, f"Missing aggregate: {key}")
                    aggregate = aggregates[key]
                    require(aggregate.get("n") == expected_n, f"Wrong aggregate n for {key}")
                    require(aggregate.get("expected_n") == expected_n, f"Wrong expected_n for {key}")
                    finite_number(aggregate.get("mean"), f"aggregate mean {key}")
                    if expected_n > 1:
                        finite_number(aggregate.get("std"), f"aggregate std {key}")
                    require(key in comparisons, f"Missing paper comparison: {key}")
                    comparison = comparisons[key]
                    require(comparison.get("status") == "matched", f"Unmatched paper comparison: {key}")
                    require(comparison.get("observed_n") == expected_n, f"Wrong comparison n for {key}")
                    finite_number(comparison.get("mean"), f"paper mean {key}")
                    finite_number(comparison.get("mean_delta"), f"paper delta {key}")
    return aggregates, comparisons


def validation_utility(tasks: list[dict[str, Any]]) -> dict[str, float]:
    """Return equal-weight min-max utilities using validation metrics only."""
    require(tasks, "Cannot rank an empty task list")
    benchmark = tasks[0]["benchmark"]
    require(all(task["benchmark"] == benchmark for task in tasks), "Cannot jointly rank different benchmarks")
    utilities = {str(task["paper_experiment_id"]): [] for task in tasks}
    for metric in PAPER_METRICS[benchmark]:
        records = [(task, paper_metric_map(task, "validation")[metric]) for task in tasks]
        directions = {record[1]["higher_is_better"] for record in records}
        require(len(directions) == 1, f"Inconsistent direction for {benchmark}/{metric}")
        higher_is_better = directions.pop()
        values = [finite_number(record[1]["value"], f"{benchmark}/{metric}") for record in records]
        low, high = min(values), max(values)
        for (task, _), value in zip(records, values, strict=True):
            if high == low:
                utility = 1.0
            elif higher_is_better:
                utility = (value - low) / (high - low)
            else:
                utility = (high - value) / (high - low)
            utilities[str(task["paper_experiment_id"])].append(utility)
    return {experiment_id: sum(values) / len(values) for experiment_id, values in utilities.items()}


def select_best_runs(ft_tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in ft_tasks:
        grouped[str(task["benchmark"])].append(task)
    selected = []
    for benchmark in (name for name in PAPER_METRICS if name in grouped):
        tasks = grouped[benchmark]
        require(len(tasks) == FT_RUNS, f"Expected three FT-Agent runs for {benchmark}")
        utilities = validation_utility(tasks)
        best = min(
            tasks,
            key=lambda task: (
                -utilities[str(task["paper_experiment_id"])],
                int(task.get("run_index") or 0),
                str(task["paper_experiment_id"]),
            ),
        )
        selected.append({"task": best, "validation_utility": utilities[str(best["paper_experiment_id"])]})
    return selected


def display_number(value: Any, unit: str | None = None, *, signed: bool = False) -> str:
    number = finite_number(value, "display value")
    precision = 2 if unit == "percent" or abs(number) >= 1 else 3
    return f"{number:+.{precision}f}" if signed else f"{number:.{precision}f}"


def display_aggregate(row: dict[str, Any]) -> str:
    mean = display_number(row["mean"], row.get("unit"))
    if row.get("std") is None:
        return mean
    return f"{mean} ± {display_number(row['std'], row.get('unit'))}"


def metric_text(task: dict[str, Any], split: str) -> str:
    items = paper_metric_map(task, split)
    return "; ".join(
        f"{items[metric].get('label') or metric}={display_number(items[metric]['value'], items[metric].get('unit'))}"
        for metric in PAPER_METRICS[task["benchmark"]]
    )


def table_lines(
    split: str,
    aggregates: dict[tuple[str, ...], dict[str, Any]],
    comparisons: dict[tuple[str, ...], dict[str, Any]],
) -> list[str]:
    lines = [
        "| Benchmark | 指标 | Base-7B 复现 | Base-7B 论文 | Base 差值 | "
        "FT-Agent 复现（3 次） | FT-Agent 论文 | FT 差值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, metrics in PAPER_METRICS.items():
        for metric in metrics:
            base_key = (BASE_GROUP, benchmark, split, metric)
            ft_key = (FT_GROUP, benchmark, split, metric)
            base = aggregates[base_key]
            ft = aggregates[ft_key]
            base_comparison = comparisons[base_key]
            ft_comparison = comparisons[ft_key]
            lines.append(
                "| "
                + " | ".join(
                    (
                        benchmark,
                        str(base.get("label") or metric),
                        display_aggregate(base),
                        display_number(base_comparison["mean"], base.get("unit")),
                        display_number(base_comparison["mean_delta"], base.get("unit"), signed=True),
                        display_aggregate(ft),
                        display_number(ft_comparison["mean"], ft.get("unit")),
                        display_number(ft_comparison["mean_delta"], ft.get("unit"), signed=True),
                    ),
                )
                + " |",
            )
    return lines


def render_report(report: dict[str, Any]) -> str:
    require(report.get("schema_version") == 1, "Unsupported results schema")
    grouped = validate_tasks(report)
    aggregates, comparisons = validate_summary_rows(report)
    best_runs = select_best_runs(grouped[FT_GROUP])
    coverage = report.get("coverage")
    require(isinstance(coverage, dict), "Report has no coverage object")
    require(coverage.get("supplied_tasks") == TOTAL_TASKS, "Expected exactly 52 supplied tasks")
    require(coverage.get("task_states") == {"succeeded": TOTAL_TASKS}, "Not all supplied tasks succeeded")

    lines = [
        "# FT-Dojo 主表复现实验报告",
        "",
        f"结果生成时间：{report.get('generated_at', 'unknown')}。差值均为“本次复现 − 论文”。",
        "FT-Agent 为 3 次独立运行的算术均值与样本标准差；Base-7B 为 1 次确定性评测。",
        "",
        "## Held-out test 主表",
        "",
        *table_lines("test", aggregates, comparisons),
        "",
        "## Validation 对照表",
        "",
        *table_lines("validation", aggregates, comparisons),
        "",
        "## Validation-only 选出的最佳单次运行",
        "",
        "每个 benchmark 先在三次运行之间，对全部论文 validation 指标按方向做 min-max 归一化，再等权平均；"
        "得分最高者被选中，完全不读取 test。若联合得分相同，以较小 run index 作确定性决胜。",
        "",
        "| Benchmark | 最佳运行 | 联合 validation 得分 | Validation 指标 | 对应的一次性 Test 指标 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in best_runs:
        task = item["task"]
        lines.append(
            f"| {task['benchmark']} | run-{task['run_index']} | {item['validation_utility']:.4f} | "
            f"{metric_text(task, 'validation')} | {metric_text(task, 'test')} |",
        )
    lines.extend(
        (
            "",
            "## 协议与覆盖审计",
            "",
            "- Base-7B：13/13；FT-Agent：39/39；全部状态为 succeeded，且任务诊断为空。",
            "- 每个 FT benchmark 恰有 3 次独立运行；主表所有 metric/split 均与论文引用行匹配。",
            "- 搜索与 checkpoint 选择只使用 validation；test 仅在选择签名冻结后执行一次。",
            "- 报告渲染器拒绝缺失结果、协议诊断、非有限数值、错误样本数或 test-informed selection。",
            "",
        ),
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.results.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), "results.json must contain an object")
        rendered = render_report(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, json.JSONDecodeError, MainReportError) as error:
        print(f"Cannot render strict main-table report: {error}", file=sys.stderr)
        return 2
    print(f"Wrote strict main-table report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
