"""Safety checks for the downstream-only terminal supervisor."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import pytest
from reproduction.ft_agent.post_main_experiment_supervisor import (
    TEMPORARY_FAILURE_EXIT_CODE,
    Supervisor,
    SupervisorError,
    parse_gpus,
    run_supervisor,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        main_matrix="unit-main",
        base_run="unit-base",
        paired_matrix="unit-paired",
        gpus="0,1,2,3,4,5,6,7",
        max_parallel=8,
        poll_interval=0.01,
        paired_attempts=1,
        validation_attempts=1,
        retry_delay=0.01,
        report_root=tmp_path / "report",
        state_file=tmp_path / "state.json",
        lock_file=tmp_path / "state.lock",
    )


def test_supervisor_requires_all_h20s() -> None:
    with pytest.raises(SupervisorError, match="all eight H20s"):
        parse_gpus("0,1", 2)


def test_final_test_command_never_allows_retest(tmp_path: Path) -> None:
    supervisor = Supervisor(_args(tmp_path))

    command = supervisor.final_command("unit-main", "main", preflight=False)

    assert command[0] == sys.executable
    assert "--resume" in command
    assert "--allow-retest" not in command
    assert command[-1] == "--resume"


def test_lock_contention_does_not_overwrite_active_supervisor_state(tmp_path: Path) -> None:
    args = _args(tmp_path)
    active_state = {"stage": "paired_training", "pid": 12345, "return_code": None}
    args.state_file.write_text(json.dumps(active_state), encoding="utf-8")
    supervisor = Supervisor(args)

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return_code = run_supervisor(supervisor)

    assert return_code == TEMPORARY_FAILURE_EXIT_CODE
    assert json.loads(args.state_file.read_text(encoding="utf-8")) == active_state
