#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Render a strict Chinese ordinary-LoRA/rsLoRA paired-comparison report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .collect_results import PAPER_METRICS
else:
    from collect_results import PAPER_METRICS

SPLITS = ("test", "validation")


class PairedReportError(RuntimeError):
    """Raised when a paired result is incomplete or violates the comparison contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Combined collector results.json")
    parser.add_argument("--paired-audit", type=Path, required=True, help="Strict paired-training audit JSON")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    return parser.parse_args()


def require(condition: object, message: str) -> None:
    if not condition:
        raise PairedReportError(message)


def finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Missing number: {label}")
    number = float(value)
    require(math.isfinite(number), f"Non-finite number: {label}")
    return number


def validate_rows(report: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, Any]]:
    require(report.get("schema_version") == 1, "Unsupported collector schema")
    require(audit.get("schema_version") == 1 and audit.get("state") == "succeeded", "Paired audit is incomplete")
    rows = report.get("lora_rslora_pairs")
    require(isinstance(rows, list) and rows, "No LoRA/rsLoRA comparison rows were collected")

    paired_tasks = {
        str(task.get("experiment_id")): task
        for task in report.get("tasks", [])
        if isinstance(task, dict) and isinstance(task.get("pairing"), dict)
    }
    expected_sources = {
        str(task["pairing"].get("source_experiment_id"))
        for task in paired_tasks.values()
    }
    require(len(paired_tasks) == audit.get("paired_task_count"), "Collected paired-task count differs from audit")
    require(len(expected_sources) == audit.get("ordinary_lora_task_count"), "Source LoRA task count differs from audit")

    seen: set[tuple[str, str, str]] = set()
    observed_sources: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        require(isinstance(raw, dict), "Malformed comparison row")
        source_id = str(raw.get("source_experiment_id"))
        paired_id = str(raw.get("paired_experiment_id"))
        split = str(raw.get("split"))
        metric = str(raw.get("metric"))
        benchmark = str(raw.get("benchmark"))
        key = (source_id, split, metric)
        require(key not in seen, f"Duplicate comparison row: {key}")
        seen.add(key)
        require(source_id in expected_sources, f"Unexpected ordinary-LoRA source: {source_id}")
        require(paired_id == f"rslora-paired/{source_id}", f"Invalid paired experiment id: {paired_id}")
        require(paired_id in paired_tasks, f"Missing paired task record: {paired_id}")
        require(raw.get("source_training_method") == "lora", f"Source is not ordinary LoRA: {source_id}")
        require(raw.get("paired_training_method") == "rslora", f"Target is not rsLoRA: {paired_id}")
        require(raw.get("state") == "complete", f"Incomplete paired metric: {key}")
        require(split in SPLITS, f"Unexpected split: {split}")
        require(benchmark in PAPER_METRICS and metric in PAPER_METRICS[benchmark], f"Unexpected metric: {key}")
        source_value = finite(raw.get("source_value"), f"{key} source")
        paired_value = finite(raw.get("paired_value"), f"{key} paired")
        raw_delta = finite(raw.get("raw_delta"), f"{key} delta")
        improvement = finite(raw.get("improvement_delta"), f"{key} improvement")
        require(math.isclose(raw_delta, paired_value - source_value, abs_tol=1e-9), f"Wrong raw delta: {key}")
        direction = raw.get("higher_is_better")
        require(isinstance(direction, bool), f"Missing metric direction: {key}")
        expected_improvement = raw_delta if direction else -raw_delta
        require(math.isclose(improvement, expected_improvement, abs_tol=1e-9), f"Wrong improvement delta: {key}")
        observed_sources.add(source_id)
        normalized.append(dict(raw))

    require(observed_sources == expected_sources, "Comparison rows do not cover every paired ordinary-LoRA task")
    for source_id in sorted(expected_sources):
        paired_id = f"rslora-paired/{source_id}"
        benchmark = str(paired_tasks[paired_id].get("benchmark"))
        expected_metrics = tuple(PAPER_METRICS[benchmark])
        for split in SPLITS:
            found = tuple(
                sorted(
                    row["metric"]
                    for row in normalized
                    if row["source_experiment_id"] == source_id and row["split"] == split
                ),
            )
            require(found == tuple(sorted(expected_metrics)), f"Metric coverage mismatch: {source_id}/{split}")
    return normalized


def display(value: float, unit: str | None = None, *, signed: bool = False) -> str:
    precision = 2 if unit == "percent" or abs(value) >= 1 else 3
    return f"{value:+.{precision}f}" if signed else f"{value:.{precision}f}"


def aggregate_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[(str(row["benchmark"]), str(row["metric"]), row.get("unit"))].append(row)
    result = []
    for (benchmark, metric, unit), items in sorted(grouped.items()):
        source = [float(item["source_value"]) for item in items]
        paired = [float(item["paired_value"]) for item in items]
        improvements = [float(item["improvement_delta"]) for item in items]
        epsilon = 1e-12
        result.append(
            {
                "benchmark": benchmark,
                "metric": metric,
                "unit": unit,
                "n": len(items),
                "source_mean": statistics.fmean(source),
                "paired_mean": statistics.fmean(paired),
                "improvement_mean": statistics.fmean(improvements),
                "wins": sum(value > epsilon for value in improvements),
                "ties": sum(abs(value) <= epsilon for value in improvements),
                "losses": sum(value < -epsilon for value in improvements),
            },
        )
    return result


def summary_table(rows: list[dict[str, Any]], split: str) -> list[str]:
    lines = [
        "| Benchmark | 指标 | n | 普通 LoRA 均值 | rsLoRA 均值 | 方向归一化改进 | 胜/平/负 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows(rows, split):
        unit = row["unit"]
        lines.append(
            f"| {row['benchmark']} | {row['metric']} | {row['n']} | "
            f"{display(row['source_mean'], unit)} | {display(row['paired_mean'], unit)} | "
            f"{display(row['improvement_mean'], unit, signed=True)} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |",
        )
    return lines


def detail_table(rows: list[dict[str, Any]], split: str) -> list[str]:
    lines = [
        "| Task | Run | 指标 | 普通 LoRA | rsLoRA | rsLoRA − LoRA | 方向归一化改进 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    selected = sorted(
        (row for row in rows if row["split"] == split),
        key=lambda row: (str(row["benchmark"]), int(row["run_index"]), str(row["metric"])),
    )
    for row in selected:
        unit = row.get("unit")
        lines.append(
            f"| {row['benchmark']} | {row['run_index']} | {row['metric']} | "
            f"{display(float(row['source_value']), unit)} | {display(float(row['paired_value']), unit)} | "
            f"{display(float(row['raw_delta']), unit, signed=True)} | "
            f"{display(float(row['improvement_delta']), unit, signed=True)} |",
        )
    return lines


def render(report: dict[str, Any], audit: dict[str, Any]) -> str:
    rows = validate_rows(report, audit)
    return "\n".join(
        (
            "# 普通 LoRA 与 rsLoRA 严格配对实验",
            "",
            f"覆盖普通 LoRA 任务 {audit['ordinary_lora_task_count']} 个、严格配对训练 workspace "
            f"{audit['paired_workspace_count']} 个；每个 workspace 均训练 2,000 样本。",
            "除 `use_rslora: false → true` 外，训练 YAML 与输入文件保持一致。checkpoint 由 validation-only "
            "规则独立选择，held-out test 不参与选模。",
            "",
            "方向归一化改进大于 0 表示 rsLoRA 更好；对越低越好的指标，其符号已反转。",
            "",
            "## Held-out test 汇总",
            "",
            *summary_table(rows, "test"),
            "",
            "## Held-out test 逐运行结果",
            "",
            *detail_table(rows, "test"),
            "",
            "## Validation 汇总",
            "",
            *summary_table(rows, "validation"),
            "",
            "## 协议审计",
            "",
            f"- 配对训练任务：{audit['paired_task_count']}/{audit['ordinary_lora_task_count']}。",
            f"- 配对 workspace：{audit['paired_workspace_count']}；每项样本数："
            f"{audit['expected_samples_per_workspace']}。",
            "- 所有比较行均同时具备普通 LoRA 与 rsLoRA 的 validation/test 指标，且配对签名通过重算。",
            "",
        ),
    )


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.results.read_text(encoding="utf-8"))
        audit = json.loads(args.paired_audit.read_text(encoding="utf-8"))
        require(isinstance(report, dict) and isinstance(audit, dict), "Inputs must contain JSON objects")
        content = render(report, audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, PairedReportError) as error:
        print(f"Cannot render strict LoRA/rsLoRA report: {error}", file=sys.stderr)
        return 2
    print(f"Wrote strict LoRA/rsLoRA report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
