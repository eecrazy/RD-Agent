import json
from pathlib import Path
from typing import NoReturn

import pytest
from rdagent.app.finetune.llm.loop import PIPELINE_CLAIM_FILE, use_pipeline_claim


def _configure_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FT_PIPELINE_CLAIM", "1")
    monkeypatch.setenv("FT_EXPERIMENT_ID", "main/example/run-2")
    monkeypatch.setenv("LOG_TRACE_PATH", str(tmp_path / "task" / "trace"))


def test_overflow_pipeline_is_run_once_and_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_claim(monkeypatch, tmp_path)
    calls: list[str] = []

    @use_pipeline_claim
    def work() -> None:
        calls.append("run")

    work()
    monkeypatch.delenv("FT_PIPELINE_CLAIM")
    work()

    claim_path = tmp_path / "task" / PIPELINE_CLAIM_FILE
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert calls == ["run"]
    assert claim["experiment_id"] == "main/example/run-2"
    assert claim["state"] == "succeeded"


def test_failed_overflow_pipeline_requires_fresh_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_claim(monkeypatch, tmp_path)

    @use_pipeline_claim
    def fail() -> NoReturn:
        message = "synthetic failure"
        raise ValueError(message)

    with pytest.raises(ValueError, match="synthetic failure"):
        fail()
    monkeypatch.delenv("FT_PIPELINE_CLAIM")
    with pytest.raises(RuntimeError, match="fresh run directory"):
        fail()

    claim_path = tmp_path / "task" / PIPELINE_CLAIM_FILE
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["state"] == "failed"
    assert claim["error"] == "ValueError: synthetic failure"
