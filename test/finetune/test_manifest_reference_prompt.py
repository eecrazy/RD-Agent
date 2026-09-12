"""Regression tests for manifest-backed training-label verification guidance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rdagent.utils.agent import tpl as tpl_module
from rdagent.utils.agent.tpl import T

if TYPE_CHECKING:
    import pytest


def test_data_coder_prompt_pins_schema_v4_row_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tpl_module.logger, "log_object", lambda *_args, **_kwargs: None)
    prompt = T("rdagent.components.coder.finetune.prompts:data_coder.system").r(
        scenario="molecule editing",
        task_desc="prepare ChemCoT data",
        dataset_info="schema_version: 4",
        queried_former_failed_knowledge=[],
        api_max_workers=1,
        datasets_path="/datasets/",
        workspace_path="./",
        force_think_token=False,
    )

    assert 'manifest["verification_references"][identity]' in prompt
    assert "ensure_ascii=False" in prompt
    assert "sort_keys=True" in prompt
    assert 'separators=(",", ":")' in prompt
    assert "before changing, normalizing, or augmenting" in prompt
    assert "ground truth" in prompt


def test_data_coder_prompt_does_not_invent_absent_curated_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tpl_module.logger, "log_object", lambda *_args, **_kwargs: None)
    prompt = T("rdagent.components.coder.finetune.prompts:data_coder.system").r(
        scenario="molecule editing",
        task_desc="prepare a hypothetical curated split",
        dataset_info="chemcotbench-cot/mol_edit/add.json",
        queried_former_failed_knowledge=[],
        api_max_workers=1,
        datasets_path="/datasets/",
        workspace_path="./",
        force_think_token=False,
    )

    assert "file tree as authoritative" in prompt
    assert "Do not require an input artifact, curated split, manifest, or schema version" in prompt
    assert "verification rule applies only when the selected dataset" in prompt
    assert "actually contains the stated manifest" in prompt


def test_mol_edit_scenario_pins_released_source_contract() -> None:
    scenarios_path = (
        Path(__file__).parents[2]
        / "rdagent"
        / "app"
        / "finetune"
        / "llm"
        / "job"
        / "scenarios.json"
    )
    description = json.loads(scenarios_path.read_text(encoding="utf-8"))[
        "chemcotbench_mol_edit"
    ]["benchmark_description"]

    for name in ("add.json", "delete.json", "sub.json"):
        assert f"chemcotbench-cot/mol_edit/{name}" in description
    assert "1,499 rows each" in description
    assert "do not invent or require one" in description
    assert "meta.reference" in description


def test_documented_identity_matches_pinned_chemcot_example() -> None:
    row = {
        "instruction": ("Modify the molecule Cc1ccc(-c2cccc(COc3ccc4cnccc4c3)c2)cc1C(=O)[O-] by adding a hydroxyl."),
        "input": "",
        "output": (
            "1. Interpret the instruction as a molecule-editing operation: add the requested hydroxyl "
            "functionality at the localized editable site.\n"
            "2. The structured analysis localizes the change as follows: Electrophilic aromatic "
            "substitution at the para position of the methoxy-substituted benzene ring is chemically "
            "viable.\n"
            "3. Apply only that local change and retain the remaining scaffold. The common retained "
            "structure contains 28 atoms with scaffold-retention ratio 1.000. The localized edit "
            "increases the heavy-atom count from 28 to 29.\n"
            "4. Check that the edited molecule sanitizes with valid valence and aromaticity; formal "
            "charge remains -1. Preserve stereochemical assignments outside the edited site.\n"
            '{"output":"Cc1ccc(-c2cccc(COc3cc4ccncc4cc3O)c2)cc1C(=O)[O-]"}'
        ),
    }
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == (
        "d158b110eaf4d44231cb1c39912e16c2cab5b39cc771805758089c4d321213fa"
    )
