from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest
from rdagent.scenarios.finetune.benchmark.merge import merge


@pytest.mark.parametrize("use_dora", [False, None])
def test_check_if_merging_needed_keeps_regular_lora_unmerged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_dora: bool | None,
) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": None, "use_dora": use_dora}),
        encoding="utf-8",
    )
    monkeypatch.setattr(merge, "is_blackwell_gpu", lambda: False)

    assert merge.check_if_merging_needed(tmp_path) is False


def test_check_if_merging_needed_merges_dora_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"modules_to_save": None, "use_dora": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(merge, "is_blackwell_gpu", lambda: False)

    assert merge.check_if_merging_needed(tmp_path) is True


def test_gpu_lease_environment_uses_an_available_dynamic_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_root = tmp_path / "gpu_leases" / "locks"
    lock_root.mkdir(parents=True)
    pool_file = lock_root.parent / "dynamic_pool"
    pool_file.write_text("0,1\n", encoding="utf-8")
    monkeypatch.setenv("FT_GPU_LEASE_LOCK_ROOT", str(lock_root))
    monkeypatch.setenv("FT_GPU_LEASE_POOL_FILE", str(pool_file))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with (lock_root / "gpu-0.lock").open("a+") as busy:
        fcntl.flock(busy.fileno(), fcntl.LOCK_EX)
        with merge.gpu_lease_environment() as environment:
            assert environment == {
                "CUDA_VISIBLE_DEVICES": "1",
                "FT_REQUESTED_CUDA_VISIBLE_DEVICES": "0",
            }
            with (lock_root / "gpu-1.lock").open("a+") as probe, pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    with (lock_root / "gpu-1.lock").open("a+") as released:
        fcntl.flock(released.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
