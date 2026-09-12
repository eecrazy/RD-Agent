#!/usr/bin/env python3
"""Complete run-level inventory of experiments reported in the FT-Dojo paper."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass

if __package__:
    from .matrix import ABLATION_BENCHMARKS, MAIN_BENCHMARKS, all_experiments
else:
    from matrix import ABLATION_BENCHMARKS, MAIN_BENCHMARKS, all_experiments

MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_PAPER_JOB_COUNT = 125
EXPECTED_GROUP_COUNTS = {
    "base-3b": 5,
    "base-7b": 13,
    "claude-code": 5,
    "codex": 5,
    "ft-agent-main": 39,
    "ft-agent-planner-deepseek-v3.2": 5,
    "ft-agent-planner-gpt-4o": 5,
    "ft-agent-planner-qwen3.5-397b-a17b": 5,
    "ft-agent-scale-5k": 5,
    "ft-agent-target-3b": 5,
    "manual-sft": 13,
    "manual-sft-llm-assisted": 5,
    "openhands-12h": 13,
    "openhands-24h-aime": 2,
}


@dataclass(frozen=True)
class PaperExperiment:
    """One independent model-training or base-evaluation job reported by the paper."""

    experiment_id: str
    paper_group: str
    method: str
    benchmark: str
    target_model: str
    planner: str | None
    data_limit: int | None
    timeout: str | None
    run_index: int
    runner: str
    artifact_status: str
    source: str

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def _fixed_jobs(
    *,
    prefix: str,
    paper_group: str,
    method: str,
    benchmarks: tuple[str, ...],
    target_model: str = MODEL_7B,
    planner: str | None = None,
    data_limit: int | None = 2000,
    timeout: str | None = "12h",
    runner: str,
    artifact_status: str,
    source: str,
) -> list[PaperExperiment]:
    return [
        PaperExperiment(
            experiment_id=f"{prefix}/{benchmark}",
            paper_group=paper_group,
            method=method,
            benchmark=benchmark,
            target_model=target_model,
            planner=planner,
            data_limit=data_limit,
            timeout=timeout,
            run_index=1,
            runner=runner,
            artifact_status=artifact_status,
            source=source,
        )
        for benchmark in benchmarks
    ]


def base_experiments() -> list[PaperExperiment]:
    """Base-model evaluations in the main table and the 3B ablation table."""
    return _fixed_jobs(
        prefix="base/7b",
        paper_group="base-7b",
        method="Base Model",
        benchmarks=MAIN_BENCHMARKS,
        data_limit=None,
        timeout=None,
        runner="run_base_eval.py",
        artifact_status="reconstructed-from-released-evaluator",
        source="main-results",
    ) + _fixed_jobs(
        prefix="base/3b",
        paper_group="base-3b",
        method="Base Model",
        benchmarks=ABLATION_BENCHMARKS,
        target_model=MODEL_3B,
        data_limit=None,
        timeout=None,
        runner="run_base_eval.py",
        artifact_status="reconstructed-from-released-evaluator",
        source="ablation-val-test",
    )


def released_ft_agent_experiments() -> list[PaperExperiment]:
    """Translate the released 64-job FT-Agent matrix into the full inventory."""
    result = []
    for experiment in all_experiments():
        if experiment.suite == "main":
            paper_group = "ft-agent-main"
            source = "main-results"
        elif experiment.experiment_id.startswith("ablation/scale-5k/"):
            paper_group = "ft-agent-scale-5k"
            source = "ablation-val-test"
        elif experiment.experiment_id.startswith("ablation/planner-gpt-4o/"):
            paper_group = "ft-agent-planner-gpt-4o"
            source = "ablation-val-test-and-frontier-agent-baselines"
        elif experiment.experiment_id.startswith("ablation/target-3b/"):
            paper_group = "ft-agent-target-3b"
            source = "ablation-val-test"
        elif experiment.experiment_id.startswith("planner/deepseek-v3.2/"):
            paper_group = "ft-agent-planner-deepseek-v3.2"
            source = "frontier-agent-baselines"
        else:
            paper_group = "ft-agent-planner-qwen3.5-397b-a17b"
            source = "frontier-agent-baselines"
        run_index = int(experiment.experiment_id.rsplit("run-", 1)[1]) if "/run-" in experiment.experiment_id else 1
        result.append(
            PaperExperiment(
                experiment_id=f"ft-agent/{experiment.experiment_id}",
                paper_group=paper_group,
                method="FT-Agent",
                benchmark=experiment.benchmark,
                target_model=experiment.model,
                planner=experiment.planner,
                data_limit=experiment.data_limit,
                timeout=experiment.timeout,
                run_index=run_index,
                runner="run_matrix.py",
                artifact_status="released",
                source=source,
            ),
        )
    return result


def protocol_only_experiments() -> list[PaperExperiment]:
    """Reported runs whose exact task artifacts or harness are not public."""
    experiments = _fixed_jobs(
        prefix="manual-sft",
        paper_group="manual-sft",
        method="Manual SFT",
        benchmarks=MAIN_BENCHMARKS,
        runner="not-public",
        artifact_status="protocol-only",
        source="main-results-and-baseline-implementation",
    )
    experiments += _fixed_jobs(
        prefix="openhands/12h",
        paper_group="openhands-12h",
        method="Tool-Augmented OpenHands v0.14",
        benchmarks=MAIN_BENCHMARKS,
        planner="gpt-5.2",
        runner="not-public",
        artifact_status="custom-harness-not-released",
        source="main-results-and-openhands-baseline-details",
    )
    experiments += _fixed_jobs(
        prefix="manual-sft-llm-assisted",
        paper_group="manual-sft-llm-assisted",
        method="Manual SFT with LLM synthesis",
        benchmarks=ABLATION_BENCHMARKS,
        planner="human-with-llm-synthesis",
        runner="not-public",
        artifact_status="task-specific-human-artifacts-not-released",
        source="manual-sft-llm-synthesis",
    )
    experiments += _fixed_jobs(
        prefix="codex",
        paper_group="codex",
        method="Codex",
        benchmarks=ABLATION_BENCHMARKS,
        planner="gpt-5.2",
        runner="not-public",
        artifact_status="exact-agent-harness-not-released",
        source="frontier-agent-baselines",
    )
    experiments += _fixed_jobs(
        prefix="claude-code",
        paper_group="claude-code",
        method="Claude Code",
        benchmarks=ABLATION_BENCHMARKS,
        planner="Claude Sonnet-4.6-thinking",
        runner="not-public",
        artifact_status="exact-agent-harness-not-released",
        source="frontier-agent-baselines",
    )
    for run in range(1, 3):
        experiments.append(
            PaperExperiment(
                experiment_id=f"openhands/24h/aime25/run-{run}",
                paper_group="openhands-24h-aime",
                method="Tool-Augmented OpenHands v0.14",
                benchmark="aime25",
                target_model=MODEL_7B,
                planner="gpt-5.2",
                data_limit=2000,
                timeout="24h",
                run_index=run,
                runner="not-public",
                artifact_status="custom-harness-not-released",
                source="extended-budget-check",
            ),
        )
    return experiments


def all_paper_experiments() -> list[PaperExperiment]:
    experiments = base_experiments() + released_ft_agent_experiments() + protocol_only_experiments()
    ids = [experiment.experiment_id for experiment in experiments]
    if len(experiments) != EXPECTED_PAPER_JOB_COUNT:
        message = f"Expected {EXPECTED_PAPER_JOB_COUNT} reported jobs, found {len(experiments)}"
        raise AssertionError(message)
    if len(ids) != len(set(ids)):
        message = "Paper experiment ids are not unique"
        raise AssertionError(message)
    counts = Counter(experiment.paper_group for experiment in experiments)
    if counts != EXPECTED_GROUP_COUNTS:
        message = f"Unexpected paper-group counts: {dict(sorted(counts.items()))}"
        raise AssertionError(message)
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print all run-level records as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = all_paper_experiments()
    if args.json:
        print(json.dumps([experiment.to_dict() for experiment in experiments], indent=2))
        return 0
    counts = Counter(experiment.paper_group for experiment in experiments)
    for group, count in sorted(counts.items()):
        print(f"{group}: {count}")
    print(f"total: {len(experiments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
