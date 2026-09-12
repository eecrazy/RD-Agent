from __future__ import annotations

import json
from pathlib import Path

from rdagent.components.coder.finetune.eval import FTDataEvaluator


class _WorkspaceStub:
    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = workspace_path
        self.file_dict: dict[str, str] = {}

    def inject_files(self, **files: str) -> None:
        for name, content in files.items():
            self.file_dict[name] = content
            (self.workspace_path / name).write_text(content, encoding="utf-8")


def test_debug_splits_are_available_through_stable_dataset_names(tmp_path: Path) -> None:
    debug_training = [{"instruction": "edit", "input": "C", "output": "CC"}]
    debug_validation = [{"instruction": "edit", "input": "N", "output": "CN"}]
    (tmp_path / "debug_data.json").write_text(json.dumps(debug_training), encoding="utf-8")
    (tmp_path / "debug_validation_data.json").write_text(json.dumps(debug_validation), encoding="utf-8")
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"processed_data": {"file_name": "data.json"}}),
        encoding="utf-8",
    )
    workspace = _WorkspaceStub(tmp_path)

    FTDataEvaluator._materialize_debug_dataset_aliases(workspace)
    evaluator = object.__new__(FTDataEvaluator)
    evaluator._update_dataset_info(workspace, sample_count=1)

    assert json.loads((tmp_path / "data.json").read_text(encoding="utf-8")) == debug_training
    assert json.loads((tmp_path / "validation_data.json").read_text(encoding="utf-8")) == debug_validation
    registry = json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))
    assert registry["processed_data"]["file_name"] == "data.json"
    assert registry["processed_data_validation"]["file_name"] == "validation_data.json"


def test_existing_task_specific_registrations_are_preserved(tmp_path: Path) -> None:
    custom_registry = {
        "processed_data": {"file_name": "task_train.json"},
        "processed_data_validation": {"file_name": "task_validation.json"},
    }
    (tmp_path / "dataset_info.json").write_text(json.dumps(custom_registry), encoding="utf-8")
    workspace = _WorkspaceStub(tmp_path)

    evaluator = object.__new__(FTDataEvaluator)
    evaluator._update_dataset_info(workspace, sample_count=3)

    assert json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8")) == custom_registry
