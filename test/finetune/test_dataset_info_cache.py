import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdagent.scenarios.finetune.scen.utils import (
    FinetuneDatasetDescriptor,
    _compute_column_stats,
    dataset_directory_is_eligible,
    dataset_info_cache_entry_is_valid,
    generate_dataset_info_config,
    valid_dataset_info_cache,
)


def test_column_stats_treat_tokenizer_sentinels_as_plain_text() -> None:
    stats = _compute_column_stats([{"text": "prefix <|endoftext|> suffix"}])

    assert stats["text"]["min_tokens"] > 0
    assert stats["text"]["max_tokens"] == stats["text"]["min_tokens"]


def test_column_stats_special_tokens_are_safe_under_concurrency() -> None:
    samples = [{"text": f"row {index} <|endoftext|>"} for index in range(16)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _compute_column_stats(samples), range(8)))

    serialized = {json.dumps(result, sort_keys=True) for result in results}
    assert len(serialized) == 1


def _cached_entry(*files: str) -> dict:
    return {
        "readme": "cached",
        "file_tree": "cached",
        "total_samples": 1,
        "total_size_mb": 0.01,
        "tasks": {"_root": {"files": list(files), "sample_count": 1}},
    }


def test_dataset_info_cache_entry_requires_all_referenced_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "example"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "present.json").write_text("[]", encoding="utf-8")

    assert dataset_info_cache_entry_is_valid(dataset_dir, _cached_entry("present.json"))
    assert not dataset_info_cache_entry_is_valid(
        dataset_dir,
        _cached_entry("present.json", "quarantined.json"),
    )


def test_stale_dataset_info_entry_is_regenerated_from_live_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "example"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "live.json").write_text('[{"instruction": "live", "output": "ok"}]', encoding="utf-8")
    stale = {"example": _cached_entry("removed-generated-file.json")}

    assert valid_dataset_info_cache(tmp_path, stale) == {}
    regenerated = generate_dataset_info_config(["example"], str(tmp_path), stale)

    assert regenerated["example"]["tasks"]["_root"]["files"] == ["live.json"]
    assert regenerated["example"]["total_samples"] == 1


def test_dataset_info_cache_rejects_paths_outside_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "example"
    dataset_dir.mkdir(parents=True)
    (tmp_path / "outside.json").write_text("[]", encoding="utf-8")

    assert not dataset_info_cache_entry_is_valid(dataset_dir, _cached_entry("../../outside.json"))


def test_dataset_description_ignores_cache_and_registration_json(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "example"
    cache_dir = dataset_dir / ".cache" / "provider" / "trees"
    cache_dir.mkdir(parents=True)
    (dataset_dir / "live.json").write_text('[{"instruction": "live", "output": "ok"}]', encoding="utf-8")
    (dataset_dir / "dataset_info.json").write_text('{"registration": true}', encoding="utf-8")
    (dataset_dir / "processing_manifest.json").write_text(
        '{"status": "complete"}',
        encoding="utf-8",
    )
    (dataset_dir / "processing_report.json").write_text('{"rows": 999}', encoding="utf-8")
    (dataset_dir / "data_stats.json").write_text('{"rows": 999}', encoding="utf-8")
    (cache_dir / "revision.json").write_text('{"metadata": true}', encoding="utf-8")

    description = FinetuneDatasetDescriptor().analyze_dataset(dataset_dir)

    assert description["tasks"]["_root"]["files"] == ["live.json"]
    assert set(description["tasks"]) == {"_root"}


def test_aborted_generated_dataset_is_neither_cached_nor_regenerated(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "aborted_generated"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "partial.json").write_text(
        '[{"instruction": "partial", "output": "unsafe"}]',
        encoding="utf-8",
    )
    (dataset_dir / "processing_manifest.json").write_text(
        json.dumps(
            {
                "status": "aborted",
                "execution_mode": "production",
                "production_eligible": False,
            },
        ),
        encoding="utf-8",
    )
    stale = {"aborted_generated": _cached_entry("partial.json")}

    assert not dataset_directory_is_eligible(dataset_dir)
    assert valid_dataset_info_cache(tmp_path, stale) == {}
    assert generate_dataset_info_config([], str(tmp_path), stale) == {}


def test_completed_generated_dataset_remains_eligible(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "completed_generated"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.json").write_text(
        '[{"instruction": "ready", "output": "ok"}]',
        encoding="utf-8",
    )
    (dataset_dir / "processing_manifest.json").write_text(
        json.dumps({"status": "complete", "production_eligible": True}),
        encoding="utf-8",
    )

    assert dataset_directory_is_eligible(dataset_dir)
    generated = generate_dataset_info_config([], str(tmp_path), {})
    assert generated["completed_generated"]["tasks"]["_root"]["files"] == ["data.json"]
