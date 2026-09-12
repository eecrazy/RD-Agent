"""Tests for the strict LoRA/rsLoRA Markdown report."""

from __future__ import annotations

import pytest
from reproduction.ft_agent.render_lora_rslora_report import PairedReportError, render


def _payload() -> tuple[dict, dict]:
    source = "main/aime25/run-1"
    paired = f"rslora-paired/{source}"
    rows = []
    for split, lora, rslora in (("validation", 10.0, 20.0), ("test", 30.0, 25.0)):
        rows.append(
            {
                "source_experiment_id": source,
                "paired_experiment_id": paired,
                "benchmark": "aime25",
                "run_index": 1,
                "source_training_method": "lora",
                "paired_training_method": "rslora",
                "split": split,
                "metric": "accuracy",
                "unit": "percent",
                "higher_is_better": True,
                "source_value": lora,
                "paired_value": rslora,
                "raw_delta": rslora - lora,
                "improvement_delta": rslora - lora,
                "state": "complete",
            },
        )
    report = {
        "schema_version": 1,
        "tasks": [
            {
                "experiment_id": paired,
                "benchmark": "aime25",
                "pairing": {"source_experiment_id": source},
            },
        ],
        "lora_rslora_pairs": rows,
    }
    audit = {
        "schema_version": 1,
        "state": "succeeded",
        "ordinary_lora_task_count": 1,
        "paired_task_count": 1,
        "paired_workspace_count": 1,
        "expected_samples_per_workspace": 2000,
    }
    return report, audit


def test_render_strict_paired_report() -> None:
    report, audit = _payload()

    rendered = render(report, audit)

    assert "普通 LoRA 与 rsLoRA 严格配对实验" in rendered
    assert "2,000 样本" in rendered
    assert "30.00" in rendered
    assert "25.00" in rendered
    assert "-5.00" in rendered


def test_render_rejects_incomplete_pair() -> None:
    report, audit = _payload()
    report["lora_rslora_pairs"][0]["state"] = "missing_paired_metric"

    with pytest.raises(PairedReportError, match="Incomplete paired metric"):
        render(report, audit)
