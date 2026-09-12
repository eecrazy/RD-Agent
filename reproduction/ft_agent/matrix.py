"""Paper experiment matrix that is executable with the released FT-Agent code."""

from __future__ import annotations

from dataclasses import asdict, dataclass

MAIN_BENCHMARKS = (
    "aime25",
    "panorama_par4pc",
    "panorama_noc4pc",
    "panorama_pi4pc",
    "chemcotbench_mol_und",
    "chemcotbench_mol_edit",
    "chemcotbench_mol_opt",
    "chemcotbench_reaction",
    "FinanceIQ_gen",
    "tablebench_data_analysis",
    "tablebench_fact_checking",
    "tablebench_numerical_reasoning",
    "tablebench_visualization",
)

ABLATION_BENCHMARKS = (
    "aime25",
    "panorama_pi4pc",
    "chemcotbench_mol_edit",
    "FinanceIQ_gen",
    "tablebench_visualization",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    suite: str
    benchmark: str
    model: str
    planner: str
    data_limit: int
    timeout: str = "12h"

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def main_experiments() -> list[Experiment]:
    return [
        Experiment(
            experiment_id=f"main/{benchmark}/run-{run}",
            suite="main",
            benchmark=benchmark,
            model="Qwen/Qwen2.5-7B-Instruct",
            planner="gpt-5.2",
            data_limit=2000,
        )
        for run in range(1, 4)
        for benchmark in MAIN_BENCHMARKS
    ]


def ablation_experiments() -> list[Experiment]:
    settings = (
        ("scale-5k", "Qwen/Qwen2.5-7B-Instruct", "gpt-5.2", 5000),
        ("planner-gpt-4o", "Qwen/Qwen2.5-7B-Instruct", "gpt-4o", 2000),
        ("target-3b", "Qwen/Qwen2.5-3B-Instruct", "gpt-5.2", 2000),
    )
    return [
        Experiment(
            experiment_id=f"ablation/{setting}/{benchmark}",
            suite="ablation",
            benchmark=benchmark,
            model=model,
            planner=planner,
            data_limit=data_limit,
        )
        for setting, model, planner, data_limit in settings
        for benchmark in ABLATION_BENCHMARKS
    ]


def planner_experiments() -> list[Experiment]:
    planners = (
        ("deepseek-v3.2", "DeepSeek-V3.2"),
        ("qwen3.5-397b-a17b", "Qwen3.5-397B-A17B"),
    )
    return [
        Experiment(
            experiment_id=f"planner/{label}/{benchmark}",
            suite="planner",
            benchmark=benchmark,
            model="Qwen/Qwen2.5-7B-Instruct",
            planner=planner,
            data_limit=2000,
        )
        for label, planner in planners
        for benchmark in ABLATION_BENCHMARKS
    ]


def all_experiments() -> list[Experiment]:
    return main_experiments() + ablation_experiments() + planner_experiments()
