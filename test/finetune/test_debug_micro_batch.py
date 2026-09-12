"""Regression tests for fine-tuning micro-batch configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from rdagent.components.coder.finetune.conf import (
    FT_DEBUG_YAML_FILE_NAME,
    FT_TEST_PARAMS_FILE_NAME,
)
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator


class _Workspace:
    def __init__(self, workspace_path: Path, *, fail_run: bool = False) -> None:
        self.workspace_path = workspace_path
        self.workspace_path.mkdir(parents=True)
        self.fail_run = fail_run
        self.file_dict = {
            FT_TEST_PARAMS_FILE_NAME: yaml.safe_dump(
                {
                    "max_samples": 10,
                    "val_size": 0.2,
                    "packing": True,
                    "neat_packing": True,
                    "dataloader_num_workers": 0,
                },
            ),
        }
        self.injected: dict[str, str] = {}

    def inject_files(self, **files: str) -> None:
        self.injected.update(files)
        self.file_dict.update(files)

    def run(self, **_kwargs: Any) -> SimpleNamespace:
        config = yaml.safe_load(self.injected[FT_DEBUG_YAML_FILE_NAME])
        output = self.workspace_path / Path(config["output_dir"]).name
        output.mkdir()
        (output / "trainer_state.json").write_text("{}", encoding="utf-8")
        if self.fail_run:
            message = "synthetic micro-batch failure"
            raise RuntimeError(message)
        return SimpleNamespace(exit_code=0, stdout="train_loss: 1.0")

    def remove_files(self, file_names: list[str]) -> None:
        for file_name in file_names:
            self.file_dict.pop(file_name, None)


def test_micro_batch_disables_packing_and_isolates_formal_output(tmp_path: Path) -> None:
    """A tiny packed subset must not collapse before its validation split."""
    workspace = _Workspace(tmp_path / "workspace")

    result = LLMConfigValidator()._run_micro_batch_test(  # noqa: SLF001
        yaml.safe_dump(
            {
                "packing": True,
                "neat_packing": True,
                "tokenized_path": "./tokenized_cache",
                "dataloader_num_workers": 8,
                "dataloader_prefetch_factor": 2,
                "save_strategy": "steps",
                "save_steps": 1,
                "save_total_limit": 1,
                "load_best_model_at_end": True,
                "save_only_model": False,
            },
        ),
        workspace,
        object(),
    )

    debug_config = yaml.safe_load(workspace.injected[FT_DEBUG_YAML_FILE_NAME])
    assert result.success
    assert debug_config["packing"] is False
    assert debug_config["neat_packing"] is False
    assert debug_config["tokenized_path"] is None
    assert debug_config["dataloader_num_workers"] == 0
    assert "dataloader_prefetch_factor" not in debug_config
    assert debug_config["save_strategy"] == "no"
    assert debug_config["load_best_model_at_end"] is False
    assert debug_config["save_only_model"] is True
    assert "save_steps" not in debug_config
    assert "save_total_limit" not in debug_config
    assert debug_config["output_dir"].startswith("./.rdagent_micro_batch_output-")
    assert debug_config["output_dir"] != "./output"
    assert debug_config["overwrite_output_dir"] is True
    assert "resume_from_checkpoint" not in debug_config
    assert not (workspace.workspace_path / Path(debug_config["output_dir"]).name).exists()
    assert FT_DEBUG_YAML_FILE_NAME not in workspace.file_dict
    assert FT_TEST_PARAMS_FILE_NAME not in workspace.file_dict


def test_micro_batch_cleans_isolated_output_after_exception(tmp_path: Path) -> None:
    workspace = _Workspace(tmp_path / "workspace", fail_run=True)

    with pytest.raises(RuntimeError, match="synthetic micro-batch failure"):
        LLMConfigValidator()._run_micro_batch_test(  # noqa: SLF001
            yaml.safe_dump({"resume_from_checkpoint": "./output/checkpoint-1"}),
            workspace,
            object(),
        )

    debug_config = yaml.safe_load(workspace.injected[FT_DEBUG_YAML_FILE_NAME])
    assert not (workspace.workspace_path / Path(debug_config["output_dir"]).name).exists()
    assert FT_DEBUG_YAML_FILE_NAME not in workspace.file_dict
    assert FT_TEST_PARAMS_FILE_NAME not in workspace.file_dict
