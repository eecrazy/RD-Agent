from pathlib import Path

import pytest
from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.scenarios.finetune.benchmark.benchmark import get_pinned_dataset_path


def test_pinned_dataset_path_uses_absolute_host_path_for_conda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "finetune_files"
    monkeypatch.setattr(FT_RD_SETTING, "file_path", file_path)
    monkeypatch.setenv("FT_BENCHMARK_DATASET_PATH", "pinned/financeiq")

    assert get_pinned_dataset_path(".") == str((file_path / "benchmarks" / "pinned" / "financeiq").resolve())


def test_pinned_dataset_path_keeps_container_path_for_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FT_BENCHMARK_DATASET_PATH", "pinned/financeiq")

    assert get_pinned_dataset_path("/workspace") == "/workspace/benchmarks/pinned/financeiq"
