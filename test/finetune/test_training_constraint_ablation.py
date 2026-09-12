from __future__ import annotations

# ruff: noqa: FBT001, PLR2004
import json
from pathlib import Path

import pytest
from reproduction.ft_agent import run_training_constraint_ablation as ablation


def test_eight_gpu_layout_uses_every_device_once() -> None:
    specs = ablation.build_specs([str(index) for index in range(8)])

    assigned = [gpu for spec in specs for gpu in spec.gpus]

    assert assigned == ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert specs[0].method == "full"
    assert specs[0].gpus == ("0", "1")
    assert [spec.method for spec in specs[1:]] == ["lora", "rslora"] * 3


def test_gpu_layout_rejects_duplicates_or_wrong_count() -> None:
    with pytest.raises(SystemExit, match="exactly eight distinct"):
        ablation.parse_gpus("0,1,2,3,4,5,6")
    with pytest.raises(SystemExit, match="exactly eight distinct"):
        ablation.parse_gpus("0,1,2,3,4,5,6,6")


def test_only_filter_supports_safe_prefill() -> None:
    specs = ablation.build_specs([str(index) for index in range(8)])

    selected = ablation.select_specs(
        specs,
        r"(?:tablebench_fact_checking__full_smoke|FinanceIQ_gen__lora)",
    )

    assert [spec.run_id for spec in selected] == [
        "tablebench_fact_checking__full_smoke",
        "FinanceIQ_gen__lora",
    ]


def test_only_filter_rejects_empty_selection() -> None:
    specs = ablation.build_specs([str(index) for index in range(8)])

    with pytest.raises(SystemExit, match="matched no controlled runs"):
        ablation.select_specs(specs, "missing")


def test_all_method_pairs_differ_only_in_rslora_switch() -> None:
    specs = ablation.build_specs([str(index) for index in range(8)])
    by_id = {spec.run_id: ablation.rendered_config(spec) for spec in specs}

    for dataset in ablation.DATASETS:
        ordinary = by_id[f"{dataset.benchmark}__lora"]
        scaled = by_id[f"{dataset.benchmark}__rslora"]
        assert ordinary["use_rslora"] is False
        assert scaled["use_rslora"] is True
        assert ablation.pair_contract_hash(ordinary, scaled)


def test_seed_override_changes_both_training_and_data_seed() -> None:
    spec = ablation.build_specs([str(index) for index in range(8)])[1]

    config = ablation.rendered_config(spec, seed=47)

    assert config["seed"] == 47
    assert config["data_seed"] == 47


def test_pair_contract_rejects_hidden_hyperparameter_drift() -> None:
    ordinary = {"finetuning_type": "lora", "use_rslora": False, "learning_rate": 1e-5}
    scaled = {"finetuning_type": "lora", "use_rslora": True, "learning_rate": 2e-5}

    with pytest.raises(SystemExit, match="fields other than use_rslora"):
        ablation.pair_contract_hash(ordinary, scaled)


def test_full_sft_config_has_no_peft_fields() -> None:
    full = ablation.rendered_config(ablation.build_specs([str(index) for index in range(8)])[0])

    assert full["finetuning_type"] == "full"
    assert "use_rslora" not in full
    assert "use_dora" not in full
    assert not any(field.startswith("lora_") for field in full)


@pytest.mark.parametrize(
    ("method", "use_rslora"),
    [("lora", False), ("rslora", True)],
)
def test_adapter_artifact_evidence_checks_actual_method(
    tmp_path: Path,
    method: str,
    use_rslora: bool,
) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "use_dora": False,
                "use_rslora": use_rslora,
            },
        ),
        encoding="utf-8",
    )

    evidence = ablation.artifact_evidence(tmp_path, method)

    assert evidence["valid"] is True
    assert evidence["rank"] == 32
