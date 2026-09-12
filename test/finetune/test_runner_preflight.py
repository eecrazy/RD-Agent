from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import yaml
from rdagent.app.finetune.llm import loop as finetune_loop
from rdagent.scenarios.finetune.train.formal_training import enforce_formal_training_method_lock
from reproduction.ft_agent import install_backends, resume_task, run_base_eval, run_matrix


def fail_on_gpu_query(*_args: object, **_kwargs: object) -> NoReturn:
    pytest.fail("preflight-only must not query GPUs")


def test_matrix_preflight_only_returns_before_gpu_query(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[int] = []
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--suite", "main", "--preflight-only"])
    monkeypatch.setattr(run_matrix, "visible_gpus", fail_on_gpu_query)
    monkeypatch.setattr(
        run_matrix,
        "preflight",
        lambda experiments, _manifest, _benchmarks: selected.append(len(experiments)),
    )

    assert run_matrix.main() == 0
    assert selected == [39]


def test_base_preflight_only_returns_before_gpu_query(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[int] = []
    monkeypatch.setattr(sys, "argv", ["run_base_eval.py", "--suite", "base-3b", "--preflight-only"])
    monkeypatch.setattr(run_base_eval, "visible_gpus", fail_on_gpu_query)
    monkeypatch.setattr(
        run_base_eval,
        "preflight",
        lambda experiments, _benchmarks, _models: selected.append(len(experiments)),
    )

    assert run_base_eval.main() == 0
    assert selected == [5]


def test_matrix_environment_allows_slow_responses_data_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FT_DEBUG_DATA_PROCESSING_TIMEOUT", raising=False)
    monkeypatch.delenv("FT_DATA_PROCESSING_TIMEOUT", raising=False)
    monkeypatch.delenv("FT_RESPONSES_DATA_PROCESSING_TIMEOUT", raising=False)

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_DEBUG_DATA_PROCESSING_TIMEOUT"] == "3600"
    assert run_matrix.os.environ["FT_DATA_PROCESSING_TIMEOUT"] == "21600"
    assert run_matrix.os.environ["FT_RESPONSES_DATA_PROCESSING_TIMEOUT"] == "360000"


def test_matrix_environment_forces_validation_only_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_EVALUATE_HELD_OUT_DURING_SEARCH", "true")

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_EVALUATE_HELD_OUT_DURING_SEARCH"] == "false"


def test_matrix_environment_uses_paper_b200_for_method_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(run_matrix.LOGICAL_GPU_COUNT_ENV, "99")
    monkeypatch.setenv(run_matrix.LOGICAL_GPU_MEMORY_ENV, "1")

    run_matrix.configure_project_environment("paper")

    assert run_matrix.os.environ[run_matrix.LOGICAL_GPU_COUNT_ENV] == "1"
    assert run_matrix.os.environ[run_matrix.LOGICAL_GPU_MEMORY_ENV] == "178"
    assert run_matrix.os.environ[run_matrix.LOGICAL_GPU_NAME_ENV] == "NVIDIA B200"
    assert "Full SFT uses BF16 ZeRO-3" in run_matrix.os.environ[
        run_matrix.PHYSICAL_EXECUTION_MAPPING_ENV
    ]


def test_matrix_environment_preserves_timeout_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_DEBUG_DATA_PROCESSING_TIMEOUT", "2400")
    monkeypatch.setenv("FT_DATA_PROCESSING_TIMEOUT", "14400")
    monkeypatch.setenv("FT_RESPONSES_DATA_PROCESSING_TIMEOUT", "360001")

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_DEBUG_DATA_PROCESSING_TIMEOUT"] == "2400"
    assert run_matrix.os.environ["FT_DATA_PROCESSING_TIMEOUT"] == "14400"
    assert run_matrix.os.environ["FT_RESPONSES_DATA_PROCESSING_TIMEOUT"] == "360001"


def test_matrix_environment_raises_stale_responses_data_processing_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT_RESPONSES_DATA_PROCESSING_TIMEOUT", "43200")

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_RESPONSES_DATA_PROCESSING_TIMEOUT"] == str(
        run_matrix.DEFAULT_RESPONSES_DATA_PROCESSING_TIMEOUT,
    )


@pytest.mark.parametrize("value", ["0", "-1", "100h"])
def test_matrix_environment_rejects_invalid_responses_data_processing_timeout(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("FT_RESPONSES_DATA_PROCESSING_TIMEOUT", value)

    with pytest.raises(ValueError, match="FT_RESPONSES_DATA_PROCESSING_TIMEOUT must be a positive integer"):
        run_matrix.configure_project_environment()


def test_matrix_environment_raises_stale_full_training_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT_FULL_TIMEOUT", "43200")

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_FULL_TIMEOUT"] == str(
        run_matrix.DEFAULT_FULL_TRAINING_TIMEOUT,
    )


def test_matrix_environment_preserves_larger_full_training_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    larger_timeout = run_matrix.DEFAULT_FULL_TRAINING_TIMEOUT + 1
    monkeypatch.setenv("FT_FULL_TIMEOUT", str(larger_timeout))

    run_matrix.configure_project_environment()

    assert run_matrix.os.environ["FT_FULL_TIMEOUT"] == str(larger_timeout)


def test_outer_process_timeout_covers_longer_formal_training() -> None:
    environment = {"FT_FULL_TIMEOUT": "360000"}

    assert run_matrix.outer_process_timeout_seconds(12 * 3600, environment) == (
        360000 + run_matrix.OUTER_TIMEOUT_GRACE_SECONDS
    )


def test_outer_process_timeout_keeps_longer_task_budget() -> None:
    environment = {"FT_FULL_TIMEOUT": "3600"}

    assert run_matrix.outer_process_timeout_seconds(48 * 3600, environment) == (
        48 * 3600 + run_matrix.OUTER_TIMEOUT_GRACE_SECONDS
    )


def test_outer_process_timeout_rejects_invalid_full_timeout() -> None:
    with pytest.raises(ValueError, match="FT_FULL_TIMEOUT must be an integer"):
        run_matrix.outer_process_timeout_seconds(12 * 3600, {"FT_FULL_TIMEOUT": "100h"})


def test_resume_environment_replaces_serialized_timer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace"
    command = [
        "python",
        "loop.py",
        "--benchmark-description",
        "A benchmark description",
        "--timeout",
        "75m",
    ]
    status = {
        "planner": "gpt-5.2",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "benchmark": "aime25",
        "data_limit": 2000,
        "trace_path": str(trace_path),
        "workspace_path": str(tmp_path / "workspace"),
        "experiment_id": "ablation/target-3b/aime25",
        "benchmark_dataset_path": str(tmp_path / "benchmark"),
        "formal_method_lock_path": str(tmp_path / "formal_training_method.json"),
    }
    environment = resume_task.task_environment(status, command, "0")

    assert environment[finetune_loop.RESUME_TIMER_BUDGET_ENV] == "75m"

    class Timer:
        def __init__(self) -> None:
            self.budget: str | None = None

        def reset(self, budget: str) -> None:
            self.budget = budget

    timer = Timer()
    resumed_loop = type("ResumedLoop", (), {"timer": timer})()
    monkeypatch.setenv(finetune_loop.RESUME_TIMER_BUDGET_ENV, "75m")

    finetune_loop.reset_resumed_timer(resumed_loop, str(trace_path / "__session__" / "1" / "0_step"))

    assert timer.budget == "75m"


def test_filtered_retry_preserves_existing_full_manifest(tmp_path: Path) -> None:
    run_root = tmp_path / "matrix-run"
    run_root.mkdir()
    full_manifest = {
        "suite": "main",
        "api_routing": {"protocol": "responses", "served_model": "gpt-5.6-sol", "adapter_base": "old"},
        "benchmark_assets": {
            "aime25": {"id": "benchmark-aime25", "revision": "pinned", "dataset_path": "pinned/aime25"},
            "FinanceIQ_gen": {
                "id": "benchmark-financeiq",
                "revision": "pinned",
                "dataset_path": "pinned/financeiq",
            },
        },
        "tasks": [
            {"experiment_id": "main/aime25/run-1", "model": "7b"},
            {"experiment_id": "main/FinanceIQ_gen/run-1", "model": "7b"},
        ],
    }
    path = run_root / "matrix.json"
    original = json.dumps(full_manifest, indent=2) + "\n"
    path.write_text(original, encoding="utf-8")
    filtered_manifest = {
        "suite": "main",
        "api_routing": {"protocol": "responses", "served_model": "gpt-5.6-sol", "adapter_base": "new"},
        "benchmark_assets": {"aime25": full_manifest["benchmark_assets"]["aime25"]},
        "tasks": [full_manifest["tasks"][0]],
    }

    run_matrix.write_or_validate_manifest(run_root, filtered_manifest)

    assert path.read_text(encoding="utf-8") == original


def test_filtered_retry_rejects_task_definition_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "matrix-run"
    run_root.mkdir()
    existing = {
        "suite": "main",
        "api_routing": {"protocol": "responses"},
        "benchmark_assets": {},
        "tasks": [{"experiment_id": "main/aime25/run-1", "model": "7b"}],
    }
    (run_root / "matrix.json").write_text(json.dumps(existing), encoding="utf-8")
    changed = {
        **existing,
        "tasks": [{"experiment_id": "main/aime25/run-1", "model": "3b"}],
    }

    with pytest.raises(SystemExit, match="task definition differs"):
        run_matrix.write_or_validate_manifest(run_root, changed)


def test_retry_refuses_to_reuse_existing_task_root(tmp_path: Path) -> None:
    experiment = run_matrix.selected_experiments("main")[0]
    task_root = tmp_path / run_matrix.safe_id(experiment.experiment_id)
    task_root.mkdir()

    with pytest.raises(SystemExit, match="Archive them before retrying"):
        run_matrix.refuse_existing_task_roots([experiment], tmp_path)


def test_retry_allows_only_a_matching_preserved_method_lock(tmp_path: Path) -> None:
    experiment = run_matrix.selected_experiments("main")[0]
    task_root = tmp_path / run_matrix.safe_id(experiment.experiment_id)
    enforce_formal_training_method_lock(
        task_root / run_matrix.FORMAL_METHOD_LOCK_FILE,
        experiment_id=experiment.experiment_id,
        training_policy="paper",
        training_method="lora",
    )

    run_matrix.refuse_existing_task_roots([experiment], tmp_path, "paper")

    (task_root / "console.log").write_text("stale", encoding="utf-8")
    with pytest.raises(SystemExit, match="Archive them before retrying"):
        run_matrix.refuse_existing_task_roots([experiment], tmp_path, "paper")


def test_retry_rejects_method_lock_for_another_experiment(tmp_path: Path) -> None:
    experiment = run_matrix.selected_experiments("main")[0]
    task_root = tmp_path / run_matrix.safe_id(experiment.experiment_id)
    enforce_formal_training_method_lock(
        task_root / run_matrix.FORMAL_METHOD_LOCK_FILE,
        experiment_id="main/another-task/run-1",
        training_policy="paper",
        training_method="full",
    )

    with pytest.raises(SystemExit, match="Archive them before retrying"):
        run_matrix.refuse_existing_task_roots([experiment], tmp_path, "paper")


def test_matrix_task_timeout_override_is_bounded() -> None:
    experiments = run_matrix.selected_experiments("main")[:2]

    overridden = run_matrix.override_task_timeout(experiments, "48h")

    assert [experiment.timeout for experiment in overridden] == ["48h", "48h"]
    assert [experiment.timeout for experiment in experiments] == ["12h", "12h"]
    with pytest.raises(SystemExit, match="at most 48h"):
        run_matrix.override_task_timeout(experiments, "49h")
    with pytest.raises(SystemExit, match="greater than 0"):
        run_matrix.override_task_timeout(experiments, "0h")


def test_matrix_workers_per_gpu_must_be_positive() -> None:
    run_matrix.validate_workers_per_gpu(1)
    with pytest.raises(SystemExit, match="must be positive"):
        run_matrix.validate_workers_per_gpu(0)


def test_matrix_worker_assignments_spread_first_lanes_and_honor_global_cap() -> None:
    assert run_matrix.worker_gpu_assignments(["0", "1", "2"], 2) == [
        "0",
        "1",
        "2",
        "0",
        "1",
        "2",
    ]
    assert run_matrix.worker_gpu_assignments(["0", "1", "2"], 2, 4) == [
        "0",
        "1",
        "2",
        "0",
    ]


def test_matrix_pipelines_multiple_tasks_per_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiments = run_matrix.selected_experiments("main")[:4]
    started = asyncio.Event()
    release = asyncio.Event()
    assignments: list[str] = []

    async def blocked_run_one(_experiment: object, gpu: str, *_args: object, **_kwargs: object) -> bool:
        assignments.append(gpu)
        if len(assignments) == len(experiments):
            started.set()
        await release.wait()
        return True

    async def exercise() -> bool:
        task = asyncio.create_task(
            run_matrix.run_workers(
                experiments,
                ["0", "1"],
                tmp_path,
                {},
                {},
                {},
                {},
                workers_per_gpu=run_matrix.DEFAULT_WORKERS_PER_GPU,
            ),
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        return await task

    monkeypatch.setattr(run_matrix, "run_one", blocked_run_one)

    assert asyncio.run(exercise()) is True
    assert assignments.count("0") == run_matrix.DEFAULT_WORKERS_PER_GPU
    assert assignments.count("1") == run_matrix.DEFAULT_WORKERS_PER_GPU


def test_gpu_lease_installer_and_pipeline_safety_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(run_matrix, "FT_ROOT", tmp_path)
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    entrypoints = (
        ("llm_finetune", "llamafactory-cli", "llamafactory.cli"),
        ("opencompass", "opencompass", "opencompass.cli.main"),
    )

    for environment_name, script_name, main_module in entrypoints:
        prefix = tmp_path / "conda_envs" / environment_name
        binary = prefix / "bin"
        binary.mkdir(parents=True)
        (binary / "python").touch()
        (binary / script_name).touch()
        install_backends.install_gpu_lease_entrypoint(
            prefix,
            script_name,
            main_module,
            dry_run=False,
        )

    training_wrapper = tmp_path / "conda_envs" / "llm_finetune" / "bin" / "llamafactory-cli"
    training_source = training_wrapper.read_text(encoding="utf-8")
    assert training_source.index("    _training_config = prepare_training_invocation()") < training_source.index(
        "    _gpu_leases = acquire_gpu_leases(_training_config)",
    )
    assert training_source.index("    _gpu_leases = acquire_gpu_leases(_training_config)") < training_source.index(
        "    configure_training_launch(_training_config)",
    )
    assert "def wait_for_gpu_memory_ready" in training_source
    assert "                wait_for_gpu_memory_ready(gpu)\n                return [lease]" in training_source
    assert "        wait_for_gpu_memory_ready(gpu)\n        leases.append(lease)" in training_source

    run_matrix.validate_gpu_lease_entrypoints(run_matrix.DEFAULT_WORKERS_PER_GPU)

    unsafe = tmp_path / "conda_envs" / "opencompass" / "bin" / "opencompass"
    unsafe.write_text("import fcntl\n# fcntl.flock without an exclusive lock\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="requires GPU lease wrappers"):
        run_matrix.validate_gpu_lease_entrypoints(run_matrix.DEFAULT_WORKERS_PER_GPU)


def test_generated_gpu_wrapper_waits_for_memory_reclamation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    prefix = tmp_path / "conda_envs" / "opencompass"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "opencompass"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "opencompass",
        "opencompass.cli.main",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    snapshot_outputs = ("45000, 100000\n", "96000, 100000\n")
    snapshots = iter(snapshot_outputs)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=next(snapshots), stderr="")

    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    monkeypatch.setattr(namespace["time"], "sleep", lambda _seconds: None)
    monkeypatch.setenv("FT_GPU_MEMORY_READY_FRACTION", "0.95")
    namespace["wait_for_gpu_memory_ready"]("3")

    assert len(calls) == len(snapshot_outputs)
    assert all("--id=3" in command for command in calls)
    output = capsys.readouterr().out
    assert "Waiting for GPU 3 memory reclamation" in output
    assert "GPU 3 memory ready" in output


def test_generated_opencompass_wrapper_opportunistically_leases_subtask_gpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "gpu_leases" / "locks"
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", lock_root)
    lock_root.mkdir(parents=True)
    (lock_root.parent / "dynamic_pool").write_text("0,1,2,3,4,5,6,7\n", encoding="utf-8")
    prefix = tmp_path / "conda_envs" / "opencompass"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "opencompass"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "opencompass",
        "opencompass.cli.main",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("FT_GPU_LEASE_LOCK_ROOT", str(lock_root))
    monkeypatch.setenv("FT_GPU_LEASE_POOL_FILE", str(lock_root.parent / "dynamic_pool"))
    monkeypatch.setenv("FT_GPU_MEMORY_READY_FRACTION", "0")
    monkeypatch.delenv("FT_EXPERIMENT_ID", raising=False)
    monkeypatch.setenv("FT_TARGET_BENCHMARK", "chemcotbench_reaction")

    leases = namespace["acquire_gpu_leases"]()
    expected_lease_count = 5
    try:
        assert len(leases) == expected_lease_count
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4"
    finally:
        for lease in reversed(leases):
            lease.close()


def test_generated_opencompass_wrapper_keeps_single_dataset_worker_single_gpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "gpu_leases" / "locks"
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", lock_root)
    lock_root.mkdir(parents=True)
    (lock_root.parent / "dynamic_pool").write_text("0,1,2,3\n", encoding="utf-8")
    prefix = tmp_path / "conda_envs" / "opencompass"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "opencompass"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "opencompass",
        "opencompass.cli.main",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("FT_GPU_LEASE_LOCK_ROOT", str(lock_root))
    monkeypatch.setenv("FT_GPU_LEASE_POOL_FILE", str(lock_root.parent / "dynamic_pool"))
    monkeypatch.setenv("FT_GPU_MEMORY_READY_FRACTION", "0")
    monkeypatch.delenv("FT_EXPERIMENT_ID", raising=False)
    monkeypatch.setenv("FT_TARGET_BENCHMARK", "tablebench_fact_checking")

    leases = namespace["acquire_gpu_leases"]()
    try:
        assert len(leases) == 1
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    finally:
        leases[0].close()


def test_generated_gpu_wrapper_uses_absolute_interpreter_for_relative_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    prefix = Path("conda_envs/opencompass")
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "opencompass"
    wrapper.touch()

    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "opencompass",
        "opencompass.cli.main",
        dry_run=False,
    )

    assert wrapper.read_text(encoding="utf-8").splitlines()[0] == (
        f"#!{tmp_path}/conda_envs/opencompass/bin/python"
    )


def test_generated_training_wrapper_prioritizes_own_torchrun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.touch()
    torchrun = binary / "torchrun"
    torchrun.touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    foreign_binary = tmp_path / "foreign" / "bin"
    foreign_binary.mkdir(parents=True)
    (foreign_binary / "torchrun").touch()
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", os.pathsep.join((str(foreign_binary), str(binary))))

    namespace["prioritize_active_environment_path"]()

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == str(binary.resolve())
    assert Path(path_entries[0], "torchrun") == torchrun.resolve()
    assert path_entries.count(str(binary.resolve())) == 1


def test_generated_training_wrapper_rejects_policy_mismatch_before_gpu_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102
    prepare = namespace["prepare_training_invocation"]
    assert callable(prepare)

    config = tmp_path / "train.yaml"
    config.write_text("finetuning_type: full\n", encoding="utf-8")
    monkeypatch.setenv("FT_EXPERIMENT_ID", "ablation/target-3b/example")
    monkeypatch.setenv("FT_TRAINING_POLICY", "lora")
    monkeypatch.setattr(sys, "argv", [str(wrapper), "train", str(config)])

    with pytest.raises(SystemExit) as error:
        prepare()

    policy_rejection_exit_code = 78
    assert error.value.code == policy_rejection_exit_code
    assert "before GPU lease" in capsys.readouterr().err

    config.write_text(
        "finetuning_type: lora\nuse_rslora: false\nuse_dora: false\n",
        encoding="utf-8",
    )
    assert prepare()["use_rslora"] is False


def test_generated_training_wrapper_rejects_quantization_before_gpu_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102
    prepare = namespace["prepare_training_invocation"]
    assert callable(prepare)

    config = tmp_path / "train.yaml"
    config.write_text(
        "finetuning_type: lora\nuse_rslora: false\nquantization_bit: 4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FT_EXPERIMENT_ID", "main/quantized-config")
    monkeypatch.setenv("FT_TRAINING_POLICY", "paper")
    monkeypatch.setattr(sys, "argv", [str(wrapper), "train", str(config)])

    with pytest.raises(SystemExit) as error:
        prepare()

    policy_rejection_exit_code = 78
    assert error.value.code == policy_rejection_exit_code
    rejection = capsys.readouterr().err
    assert "quantization_bit must be null or omitted" in rejection
    assert "before GPU lease" in rejection


def test_generated_training_wrapper_disables_debug_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fresh wrapper must protect workers with a cached old validator."""
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    config = tmp_path / "debug_train.yaml"
    config.write_text(
        """finetuning_type: lora
use_rslora: false
use_dora: false
save_strategy: steps
save_steps: 1
save_total_limit: 1
load_best_model_at_end: true
save_only_model: false
resume_from_checkpoint: ./output/checkpoint-1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FT_TRAINING_POLICY", "paper")
    monkeypatch.setattr(sys, "argv", [str(wrapper), "train", str(config)])

    prepared = namespace["prepare_training_invocation"]()
    rendered = yaml.safe_load(config.read_text(encoding="utf-8"))

    assert prepared == rendered
    assert rendered["save_strategy"] == "no"
    assert rendered["load_best_model_at_end"] is False
    assert rendered["save_only_model"] is True
    assert "save_steps" not in rendered
    assert "save_total_limit" not in rendered
    assert "resume_from_checkpoint" not in rendered


@pytest.mark.parametrize("command", ["help", "version", "--help", "-h", "--version"])
def test_generated_training_wrapper_skips_gpu_lease_for_metadata_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", command])

    assert namespace["invocation_requires_gpu_lease"]() is False


def test_generated_training_wrapper_requires_gpu_lease_for_train(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "train", "train.yaml"])

    assert namespace["invocation_requires_gpu_lease"]() is True


@pytest.mark.parametrize(
    ("cutoff_len", "batch", "accumulation", "world_size", "expected_batch", "expected_accumulation"),
    [
        (8192, 8, 4, 2, 8, 2),
        (8193, 4, 8, 4, 4, 2),
    ],
)
def test_generated_full_sft_wrapper_injects_zero3_and_preserves_global_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cutoff_len: int,
    batch: int,
    accumulation: int,
    world_size: int,
    expected_batch: int,
    expected_accumulation: int,
) -> None:
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", tmp_path / "gpu_leases" / "locks")
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    zero3 = tmp_path / "ds_z3_config.json"
    zero3.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(
        "\n".join(
            (
                "finetuning_type: full",
                f"cutoff_len: {cutoff_len}",
                f"per_device_train_batch_size: {batch}",
                f"gradient_accumulation_steps: {accumulation}",
                "",
            ),
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FT_TRAINING_POLICY", "full")
    monkeypatch.setenv("FT_DEEPSPEED_ZERO3_CONFIG", str(zero3))
    monkeypatch.delenv("FT_FULL_SFT_GPUS", raising=False)
    monkeypatch.setattr(sys, "argv", [str(wrapper), "train", str(config)])

    prepared = namespace["prepare_training_invocation"]()
    rendered = config.read_text(encoding="utf-8")
    parsed = yaml.safe_load(rendered)

    # Re-entering the wrapper (for example after a launcher retry) must read
    # the persisted logical-batch marker instead of shrinking the batch again.
    prepared_again = namespace["prepare_training_invocation"]()
    rendered_again = config.read_text(encoding="utf-8")

    assert namespace["full_sft_gpu_count"](prepared) == world_size
    assert namespace["full_sft_gpu_count"](prepared_again) == world_size
    assert namespace["desired_gpu_lease_count"](prepared) == world_size
    assert parsed["deepspeed"] == str(zero3)
    assert parsed["per_device_train_batch_size"] == expected_batch
    assert parsed["gradient_accumulation_steps"] == expected_accumulation
    assert expected_batch * expected_accumulation * world_size == batch * accumulation
    assert f"# rdagent_global_batch_size: {batch * accumulation}" in rendered
    assert f"# rdagent_world_size: {world_size}" in rendered
    assert rendered_again == rendered
    assert rendered.count("# rdagent_global_batch_size:") == 1
    assert rendered.count("# rdagent_world_size:") == 1


def test_generated_wrapper_leases_multiple_gpus_only_for_full_sft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "gpu_leases" / "locks"
    monkeypatch.setattr(install_backends, "GPU_LEASE_LOCK_ROOT", lock_root)
    lock_root.mkdir(parents=True)
    (lock_root.parent / "dynamic_pool").write_text("0,1,2,3\n", encoding="utf-8")
    prefix = tmp_path / "conda_envs" / "llm_finetune"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    (binary / "python").touch()
    wrapper = binary / "llamafactory-cli"
    wrapper.touch()
    install_backends.install_gpu_lease_entrypoint(
        prefix,
        "llamafactory-cli",
        "llamafactory.cli",
        dry_run=False,
    )
    namespace: dict[str, object] = {"__name__": "generated_wrapper"}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)  # noqa: S102

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("FT_GPU_LEASE_LOCK_ROOT", str(lock_root))
    monkeypatch.setenv("FT_GPU_LEASE_POOL_FILE", str(lock_root.parent / "dynamic_pool"))
    monkeypatch.setenv("FT_GPU_MEMORY_READY_FRACTION", "0")
    monkeypatch.delenv("FT_FULL_SFT_GPUS", raising=False)

    assert namespace["desired_gpu_lease_count"](
        {"finetuning_type": "lora", "use_rslora": False},
    ) == 1
    leases = namespace["acquire_gpu_leases"](
        {"finetuning_type": "full", "cutoff_len": 8192},
    )
    try:
        assert len(leases) == namespace["desired_gpu_lease_count"](
            {"finetuning_type": "full", "cutoff_len": 8192},
        )
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
        namespace["configure_training_launch"]({"finetuning_type": "full"})
        assert os.environ["FORCE_TORCHRUN"] == "1"
        assert os.environ["NPROC_PER_NODE"] == "2"
    finally:
        for lease in reversed(leases):
            lease.close()


def test_gpu_guard_parsers_ignore_headers_and_malformed_rows() -> None:
    gpu_output = "0, GPU-a\nindex, uuid\n1, GPU-b\n"
    process_output = "123, GPU-a, VLLM::EngineCore\npid, gpu_uuid, process_name\nbad\n"

    assert run_matrix.parse_gpu_inventory(gpu_output) == {"0": "GPU-a", "index": "uuid", "1": "GPU-b"}
    assert run_matrix.parse_compute_inventory(process_output) == [
        {"pid": 123, "gpu_uuid": "GPU-a", "process_name": "VLLM::EngineCore"},
    ]


def test_gpu_guard_only_reports_live_non_descendants() -> None:
    process_table = {
        100: (1, 100),
        110: (100, 110),
        111: (110, 110),
        200: (1, 200),
    }
    processes = [
        {"pid": 111, "gpu_uuid": "GPU-a", "process_name": "owned"},
        {"pid": 200, "gpu_uuid": "GPU-a", "process_name": "external"},
        {"pid": 300, "gpu_uuid": "GPU-a", "process_name": "already-gone"},
        {"pid": 200, "gpu_uuid": "GPU-b", "process_name": "unselected"},
    ]

    assert run_matrix.foreign_gpu_processes(
        100,
        {"GPU-a"},
        compute_processes=processes,
        process_table=process_table,
    ) == [{"pid": 200, "gpu_uuid": "GPU-a", "process_name": "external"}]


def test_gpu_guard_cancels_and_awaits_active_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    cancelled: list[bool] = []

    async def blocked_run_one(*_args: object, **_kwargs: object) -> bool:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return False

    async def failing_guard(_owner_pid: int, _gpu_uuids: set[str], _interval: float) -> None:
        await started.wait()
        raise run_matrix.GPUExclusivityError(
            [{"pid": 200, "gpu_uuid": "GPU-a", "process_name": "external"}],
        )

    monkeypatch.setattr(run_matrix, "run_one", blocked_run_one)
    monkeypatch.setattr(run_matrix, "monitor_gpu_exclusivity", failing_guard)

    experiment = run_matrix.selected_experiments("main")[0]
    with pytest.raises(run_matrix.GPUExclusivityError, match="external process"):
        asyncio.run(
            run_matrix.run_workers(
                [experiment],
                ["0"],
                tmp_path,
                {},
                {},
                {},
                {},
                {"GPU-a"},
                0.01,
            ),
        )

    assert cancelled == [True]
