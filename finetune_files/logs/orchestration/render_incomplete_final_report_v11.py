#!/usr/bin/env python3
# ruff: noqa: E402, RUF001
"""Render the audited final report while preserving one irrecoverable held-out failure.

This renderer is intentionally incident-specific.  It accepts exactly one incomplete
FT task (Molecule Editing run-1), verifies the raw failure logs, and marks every
partial aggregate with its observed sample count.  It does not modify experiment
artifacts and is not a replacement for the strict main-report renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data/github/RD-Agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction.ft_agent.collect_results import PAPER_METRICS
from reproduction.ft_agent.render_main_report import (
    BASE_GROUP,
    EXPECTED_AGGREGATE_COUNTS,
    EXPECTED_TASK_COUNTS,
    FT_GROUP,
    MainReportError,
    display_aggregate,
    display_number,
    finite_number,
    keyed_rows,
    metric_text,
    paper_metric_map,
    require,
    select_best_runs,
)

INCOMPLETE_EXPERIMENT = "ft-agent/main/chemcotbench_mol_edit/run-1"
EXPECTED_DIAGNOSTICS = {
    "missing_final_test",
    "missing_test_paper_metrics",
    "unusable_final_test_artifact",
}
EXPECTED_COVERAGE_DIAGNOSTICS = dict.fromkeys(EXPECTED_DIAGNOSTICS, 1)
EXPECTED_TOTAL_TASKS = sum(EXPECTED_TASK_COUNTS.values())
EXPECTED_COMPLETE_TASKS = EXPECTED_TOTAL_TASKS - 1
EXPECTED_FT_RUNS = EXPECTED_AGGREGATE_COUNTS[FT_GROUP]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--failure-infer-log", type=Path, required=True)
    parser.add_argument("--failure-eval-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_metric_view(task: dict[str, Any], split: str) -> None:
    found = paper_metric_map(task, split)
    benchmark = task.get("benchmark")
    require(benchmark in PAPER_METRICS, f"Unknown benchmark: {benchmark}")
    require(
        tuple(found) == PAPER_METRICS[benchmark],
        f"Unexpected {split} metrics for {task.get('paper_experiment_id')}: {tuple(found)}",
    )


def validate_tasks(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tasks = report.get("tasks")
    require(isinstance(tasks, list), "Report has no task list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incomplete_seen: list[str] = []

    for task in tasks:
        require(isinstance(task, dict), "Malformed task record")
        group = task.get("paper_group")
        if group not in EXPECTED_TASK_COUNTS:
            continue
        experiment_id = str(task.get("paper_experiment_id"))
        require(task.get("state") == "succeeded", f"Search task is not succeeded: {experiment_id}")
        selection = task.get("selection")
        require(isinstance(selection, dict), f"Task has no selection: {experiment_id}")
        require(selection.get("test_used_for_selection") is False, f"Test-informed selection: {experiment_id}")
        diagnostics = task.get("diagnostics")
        require(isinstance(diagnostics, list), f"Malformed diagnostics: {experiment_id}")
        final_test = task.get("final_test")
        require(isinstance(final_test, dict), f"Task has no final-test metadata: {experiment_id}")
        validate_metric_view(task, "validation")

        if experiment_id == INCOMPLETE_EXPERIMENT:
            incomplete_seen.append(experiment_id)
            codes = {str(item.get("code")) for item in diagnostics if isinstance(item, dict)}
            require(codes == EXPECTED_DIAGNOSTICS, f"Unexpected incomplete-task diagnostics: {codes}")
            require(final_test.get("state") == "failed", "Expected the audited final test to remain failed")
            require(task.get("final", {}).get("test") is None, "Failed task unexpectedly has a test result")
        else:
            require(diagnostics == [], f"Unexpected diagnostics: {experiment_id}")
            require(final_test.get("state") == "succeeded", f"Final test is not succeeded: {experiment_id}")
            validate_metric_view(task, "test")
        grouped[str(group)].append(task)

    require(incomplete_seen == [INCOMPLETE_EXPERIMENT], "Expected exactly one known incomplete task")
    for group, expected in EXPECTED_TASK_COUNTS.items():
        require(len(grouped[group]) == expected, f"Expected {expected} {group} tasks, found {len(grouped[group])}")
    for benchmark in PAPER_METRICS:
        require(
            sum(task["benchmark"] == benchmark for task in grouped[BASE_GROUP]) == 1,
            f"Expected one Base-7B task for {benchmark}",
        )
        require(
            sum(task["benchmark"] == benchmark for task in grouped[FT_GROUP]) == EXPECTED_FT_RUNS,
            f"Expected three FT-Agent tasks for {benchmark}",
        )
    return grouped


def expected_observed_n(group: str, benchmark: str, split: str) -> int:
    if group == FT_GROUP and benchmark == "chemcotbench_mol_edit" and split == "test":
        return 2
    return EXPECTED_AGGREGATE_COUNTS[group]


def validate_summary_rows(
    report: dict[str, Any],
) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[tuple[str, ...], dict[str, Any]]]:
    aggregates = keyed_rows(report.get("aggregates"), "aggregates")
    comparisons = keyed_rows(report.get("comparisons"), "comparisons")
    for group, expected_n in EXPECTED_AGGREGATE_COUNTS.items():
        for benchmark, metrics in PAPER_METRICS.items():
            for split in ("validation", "test"):
                observed_n = expected_observed_n(group, benchmark, split)
                for metric in metrics:
                    key = (group, benchmark, split, metric)
                    require(key in aggregates, f"Missing aggregate: {key}")
                    aggregate = aggregates[key]
                    require(aggregate.get("expected_n") == expected_n, f"Wrong expected n for {key}")
                    require(aggregate.get("n") == observed_n, f"Wrong observed n for {key}")
                    finite_number(aggregate.get("mean"), f"aggregate mean {key}")
                    if observed_n > 1:
                        finite_number(aggregate.get("std"), f"aggregate std {key}")
                    require(key in comparisons, f"Missing comparison: {key}")
                    comparison = comparisons[key]
                    require(comparison.get("status") == "matched", f"Unmatched paper comparison: {key}")
                    require(comparison.get("observed_n") == observed_n, f"Wrong comparison n for {key}")
                    finite_number(comparison.get("mean"), f"paper mean {key}")
                    finite_number(comparison.get("mean_delta"), f"paper delta {key}")
    return aggregates, comparisons


def validate_coverage(report: dict[str, Any]) -> None:
    coverage = report.get("coverage")
    require(isinstance(coverage, dict), "Report has no coverage object")
    require(coverage.get("supplied_tasks") == EXPECTED_TOTAL_TASKS, "Expected exactly 52 supplied tasks")
    require(coverage.get("task_states") == {"succeeded": EXPECTED_TOTAL_TASKS}, "Unexpected search states")
    require(
        coverage.get("diagnostics") == EXPECTED_COVERAGE_DIAGNOSTICS,
        f"Unexpected coverage diagnostics: {coverage.get('diagnostics')}",
    )
    inventory = coverage.get("paper_inventory_states")
    require(isinstance(inventory, dict), "Missing paper inventory states")
    require(inventory.get("collected") == EXPECTED_COMPLETE_TASKS, "Expected 51 complete supplied tasks")
    require(inventory.get("supplied_but_incomplete") == 1, "Expected one incomplete supplied task")


def validate_failure_logs(infer_log: Path, eval_log: Path) -> str:
    infer_text = infer_log.read_text(encoding="utf-8", errors="replace")
    eval_text = eval_log.read_text(encoding="utf-8", errors="replace")
    require("EADDRINUSE" in infer_text, "Inference log does not contain the audited EADDRINUSE failure")
    require("port: 46473" in infer_text, "Inference log does not contain the audited port")
    require("Prediction files not found" in eval_text, "Evaluation log does not prove prediction absence")
    return "vLLM 本地 rendezvous 端口 46473 冲突（EADDRINUSE），模型引擎在生成前退出"


def table_lines(
    split: str,
    aggregates: dict[tuple[str, ...], dict[str, Any]],
    comparisons: dict[tuple[str, ...], dict[str, Any]],
) -> list[str]:
    lines = [
        "| Benchmark | 指标 | Base-7B 复现 | Base-7B 论文 | Base 差值 | "
        "FT-Agent 复现 | FT-Agent 论文 | FT 差值 |",
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
            ft_value = display_aggregate(ft)
            if ft["n"] != ft["expected_n"]:
                ft_value += f" (n={ft['n']}/{ft['expected_n']})"
            lines.append(
                "| "
                + " | ".join(
                    (
                        benchmark,
                        str(base.get("label") or metric),
                        display_aggregate(base),
                        display_number(base_comparison["mean"], base.get("unit")),
                        display_number(base_comparison["mean_delta"], base.get("unit"), signed=True),
                        ft_value,
                        display_number(ft_comparison["mean"], ft.get("unit")),
                        display_number(ft_comparison["mean_delta"], ft.get("unit"), signed=True),
                    ),
                )
                + " |",
            )
    return lines


def render_report(report: dict[str, Any], failure_cause: str, infer_log: Path, eval_log: Path) -> str:
    require(report.get("schema_version") == 1, "Unsupported results schema")
    grouped = validate_tasks(report)
    aggregates, comparisons = validate_summary_rows(report)
    validate_coverage(report)
    best_runs = select_best_runs(grouped[FT_GROUP])
    failed = next(task for task in grouped[FT_GROUP] if task["paper_experiment_id"] == INCOMPLETE_EXPERIMENT)
    failure_artifact = failed["final_test"]["artifact"]

    def shown_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    lines = [
        "# FT-Dojo 主表复现实验报告（不完整终态审计）",
        "",
        f"结果生成时间：{report.get('generated_at', 'unknown')}。差值均为“本次复现 − 论文”。",
        "",
        "> 本报告不是 52/52 严格完整主表：Base-7B 13/13 完整，FT-Agent held-out 38/39 成功。",
        "> 唯一缺失为 Molecule Editing run-1；其 test 聚合只使用 run-2/3，并明确标记 n=2/3。",
        "",
        "## Held-out test 主表",
        "",
        "FT-Agent 通常为 3 次独立运行的算术均值与样本标准差；"
        "Molecule Editing 因唯一失败仅报告有效的 2 次，禁止解读为完整三次复现。",
        "",
        *table_lines("test", aggregates, comparisons),
        "",
        "## Validation 对照表",
        "",
        *table_lines("validation", aggregates, comparisons),
        "",
        "## Validation-only 选出的最佳单次运行",
        "",
        "每个 benchmark 对三次运行的全部论文 validation 指标按方向做 min-max 归一化后等权平均；选择过程不读取 test。",
        "",
        "| Benchmark | 最佳运行 | 联合 validation 得分 | Validation 指标 | 对应的一次性 Test 指标 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in best_runs:
        task = item["task"]
        test_view = task.get("final", {}).get("test")
        test_text = metric_text(task, "test") if isinstance(test_view, dict) else "N/A（held-out 失败，无预测）"
        lines.append(
            f"| {task['benchmark']} | run-{task['run_index']} | {item['validation_utility']:.4f} | "
            f"{metric_text(task, 'validation')} | {test_text} |",
        )
    lines.extend(
        (
            "",
            "## 唯一失败与不可恢复性审计",
            "",
            f"- 任务：`{INCOMPLETE_EXPERIMENT}`；原始 `final_test.state` 保持 `failed`。",
            f"- 根因：{failure_cause}。",
            "- `chemcotbench_mol_edit_add` 没有生成 prediction/result；评分阶段也明确报告 prediction file not found。",
            "- delete/sub 虽有结果，但不能据此构造包含 add 的 benchmark accuracy；本报告没有填补、外推或伪造该指标。",
            "- held-out 只提交过一次，失败后没有重跑。严格主表渲染器仍会拒绝这份不完整结果。",
            f"- 终态证据：`{failure_artifact}`。",
            f"- 推理日志：`{shown_path(infer_log)}`。",
            f"- 评分日志：`{shown_path(eval_log)}`。",
            "",
            "## 协议与覆盖审计",
            "",
            "- 搜索与 checkpoint 选择只使用 validation；test 在 39 个选择签名冻结后统一提交一次。",
            "- 39/39 任务均进入 held-out；38 succeeded、1 failed；没有第二次测试或复用其他运行的 test 输出。",
            "- 除上述唯一缺失外，其余 51 个 supplied task 无采集诊断，所有主表 metric/split 均可用。",
            "- Molecule Editing held-out 行的均值、标准差和差值均基于明确显示的 n=2，不代表 n=3。",
            "",
        ),
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.results.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), "results.json must contain an object")
        failure_cause = validate_failure_logs(args.failure_infer_log, args.failure_eval_log)
        rendered = render_report(payload, failure_cause, args.failure_infer_log, args.failure_eval_log)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, json.JSONDecodeError, MainReportError) as error:
        print(f"Cannot render audited incomplete report: {error}", file=sys.stderr)
        return 2
    print(f"Wrote audited incomplete report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
