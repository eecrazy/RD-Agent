#!/usr/bin/env python3
"""Download and verify the pinned FT-Dojo datasets and target models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("assets.json")
FT_ROOT = ROOT / "finetune_files"
LOCK_PATH = FT_ROOT / "asset-lock.json"
MARKER_NAME = ".rdagent-asset.json"
STATE_NAME = ".rdagent-asset-state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", "dataset", "model"), default="all")
    parser.add_argument("--asset", action="append", default=[], help="Asset id to process; repeat as needed")
    parser.add_argument("--verify-only", action="store_true", help="Do not download; verify existing inventories")
    parser.add_argument("--force", action="store_true", help="Download and post-process again")
    parser.add_argument("--max-workers", type=int, default=8, help="Per-repository Hugging Face workers")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def target_path(asset: dict[str, str]) -> Path:
    if target := asset.get("target"):
        relative = Path(target)
        if relative.is_absolute() or ".." in relative.parts:
            message = f"Asset target must stay below finetune_files: {target!r}"
            raise ValueError(message)
        return FT_ROOT / relative
    if asset["kind"] == "dataset":
        return FT_ROOT / "datasets" / asset["name"]
    return FT_ROOT / "models" / asset["name"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def checked_relative_path(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path():
        message = f"Mirror file must be a relative path: {value!r}"
        raise ValueError(message)
    return relative


def validate_verified_git_mirror(asset: dict[str, Any]) -> dict[str, Any]:
    mirror: dict[str, Any] = asset["verified_git_mirror"]
    if mirror.get("provider") != "github-git":
        message = f"Unknown verified mirror provider for {asset['id']}: {mirror.get('provider')!r}"
        raise ValueError(message)

    prefix = mirror.get("prefix", "").strip("/")
    checked_relative_path(prefix)
    expected_oids = mirror["files"]
    source_oids = mirror.get("source_oids", {})
    append_final_newline = set(mirror.get("append_final_newline", []))
    unknown_transforms = append_final_newline - set(expected_oids)
    unknown_sources = set(source_oids) - set(expected_oids)
    if unknown_transforms or unknown_sources:
        unknown = sorted(unknown_transforms | unknown_sources)
        message = f"Mirror metadata references unknown files: {', '.join(unknown)}"
        raise ValueError(message)
    return mirror


def fetch_verified_git_subtree(mirror: dict[str, Any], temporary_root: Path, asset_id: str) -> Path:
    repository_root = temporary_root / "repository"
    git = shutil.which("git")
    if git is None:
        message = "git is required to retrieve the verified asset mirror"
        raise RuntimeError(message)
    commands = (
        ("init", "--quiet", str(repository_root)),
        (
            "-C",
            str(repository_root),
            "remote",
            "add",
            "origin",
            f"https://github.com/{mirror['repository']}.git",
        ),
        (
            "-C",
            str(repository_root),
            "-c",
            "protocol.version=2",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            mirror["revision"],
        ),
        (
            "-C",
            str(repository_root),
            "checkout",
            "--quiet",
            "FETCH_HEAD",
            "--",
            mirror["prefix"],
        ),
    )
    for command in commands:
        result = subprocess.run(  # noqa: S603
            (git, *command),
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            message = f"Failed to retrieve verified mirror for {asset_id}: {detail}"
            raise RuntimeError(message)
    return repository_root


def download_verified_git_mirror(asset: dict[str, Any], output: Path) -> None:
    mirror = validate_verified_git_mirror(asset)
    prefix = mirror["prefix"]
    expected_oids = mirror["files"]
    source_oids = mirror.get("source_oids", {})
    append_final_newline = set(mirror.get("append_final_newline", []))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.mirror-", dir=output.parent))
    staging = temporary_root / "payload"
    try:
        repository_root = fetch_verified_git_subtree(mirror, temporary_root, asset["id"])
        staging.mkdir()
        for value, expected_oid in sorted(expected_oids.items()):
            relative = checked_relative_path(value)
            payload = (repository_root / prefix / relative).read_bytes()

            source_oid = git_blob_oid(payload)
            wanted_source_oid = source_oids.get(value, expected_oid)
            if source_oid != wanted_source_oid:
                message = f"Source Git blob mismatch for {value}: {source_oid} != {wanted_source_oid}"
                raise RuntimeError(message)
            if value in append_final_newline:
                payload += b"\n"
            output_oid = git_blob_oid(payload)
            if output_oid != expected_oid:
                message = f"Pinned Git blob mismatch for {value}: {output_oid} != {expected_oid}"
                raise RuntimeError(message)

            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

        if output.exists():
            print(f"Replacing incomplete asset directory: {output}", flush=True)
            shutil.rmtree(output)
        staging.replace(output)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name in {MARKER_NAME, STATE_NAME} or ".cache" in relative.parts:
            continue
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    return files


def verify_inventory(root: Path, expected: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_paths = {item["path"] for item in expected}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {MARKER_NAME, STATE_NAME}
        and ".cache" not in path.relative_to(root).parts
    }
    errors.extend(f"missing: {missing}" for missing in sorted(expected_paths - actual_paths))
    errors.extend(f"unexpected: {extra}" for extra in sorted(actual_paths - expected_paths))
    for item in expected:
        path = root / item["path"]
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size != item["size"]:
            errors.append(f"size mismatch: {item['path']} ({size} != {item['size']})")
        elif sha256(path) != item["sha256"]:
            errors.append(f"sha256 mismatch: {item['path']}")
    return errors


def postprocess_training_dataset(name: str, output: Path) -> None:
    # Import after .env is loaded so FT settings resolve to the project-local tree.
    from rdagent.scenarios.finetune.datasets import DATASETS  # noqa: PLC0415

    config = DATASETS[name]
    if config.post_download_fn:
        config.post_download_fn(str(output))
    custom_readme = ROOT / "rdagent" / "scenarios" / "finetune" / "datasets" / name / "README.md"
    if custom_readme.exists():
        shutil.copy2(custom_readme, output / "README.md")


def combine_aime2025(output: Path) -> None:
    combined = output / "aime2025.jsonl"
    with combined.open("wb") as destination:
        for filename in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
            source = output / filename
            payload = source.read_bytes()
            destination.write(payload)
            if payload and not payload.endswith(b"\n"):
                destination.write(b"\n")


def prepare_financeiq_eval(output: Path) -> None:
    from rdagent.scenarios.finetune.datasets.financeiq.split import split_financeiq_dataset  # noqa: PLC0415

    data_dir = output / "data"
    for folder in ("dev", "test"):
        source = data_dir / folder
        if source.exists():
            shutil.move(str(source), str(output / folder))
    if data_dir.exists():
        shutil.rmtree(data_dir)
    split_financeiq_dataset(str(output), split="test")


def postprocess_dataset(asset: dict[str, Any], output: Path) -> None:
    postprocessor = asset.get("postprocess")
    if postprocessor == "combine_aime2025":
        combine_aime2025(output)
    elif postprocessor == "financeiq_eval":
        prepare_financeiq_eval(output)
    elif postprocessor is not None:
        message = f"Unknown postprocessor for {asset['id']}: {postprocessor}"
        raise ValueError(message)
    elif asset.get("role", "training") == "training":
        postprocess_training_dataset(asset["name"], output)


def write_marker(asset: dict[str, str], output: Path) -> dict[str, Any]:
    files = inventory(output)
    marker = {
        "id": asset["id"],
        "kind": asset["kind"],
        "repo_id": asset["repo_id"],
        "revision": asset["revision"],
        "path": str(output),
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    for key in (
        "role",
        "target",
        "dataset_path",
        "benchmarks",
        "postprocess",
        "allow_patterns",
        "verified_git_mirror",
    ):
        if key in asset:
            marker[key] = asset[key]
    temporary = output / f"{MARKER_NAME}.tmp"
    temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / MARKER_NAME)
    return marker


def matching_marker(asset: dict[str, str], output: Path) -> dict[str, Any] | None:
    marker_path = output / MARKER_NAME
    if not marker_path.is_file():
        return None
    try:
        marker = read_json(marker_path)
    except (json.JSONDecodeError, OSError):
        return None
    identity = ("id", "kind", "repo_id", "revision")
    optional_identity = (
        "role",
        "target",
        "dataset_path",
        "benchmarks",
        "postprocess",
        "allow_patterns",
        "verified_git_mirror",
    )
    if all(marker.get(key) == asset[key] for key in identity) and all(
        marker.get(key) == asset.get(key) for key in optional_identity
    ):
        return marker
    return None


def write_state(asset: dict[str, str], output: Path) -> None:
    state = {"id": asset["id"], "revision": asset["revision"], "phase": "postprocessing"}
    temporary = output / f"{STATE_NAME}.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / STATE_NAME)


def download(asset: dict[str, str], output: Path, max_workers: int, *, reset: bool) -> dict[str, Any]:
    if reset and output.exists():
        print(f"Resetting incomplete or stale asset directory: {output}", flush=True)
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {asset['id']} at {asset['revision']} -> {output}", flush=True)
    if "verified_git_mirror" in asset:
        download_verified_git_mirror(asset, output)
    else:
        snapshot_download(
            repo_id=asset["repo_id"],
            repo_type=asset["kind"],
            revision=asset["revision"],
            local_dir=output,
            token=os.environ.get("HF_TOKEN"),
            max_workers=max_workers,
            allow_patterns=asset.get("allow_patterns"),
        )
    if asset["kind"] == "dataset":
        # A failed post-processing pass must not be applied twice (FinanceIQ's split is
        # intentionally destructive). The state file makes the next run reset first.
        write_state(asset, output)
        postprocess_dataset(asset, output)
    print(f"Hashing {asset['id']}...", flush=True)
    marker = write_marker(asset, output)
    (output / STATE_NAME).unlink(missing_ok=True)
    return marker


def select_assets(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, str]]:
    selected = []
    requested = set(args.asset)
    for asset in manifest["assets"]:
        if args.kind != "all" and asset["kind"] != args.kind:
            continue
        if requested and asset["id"] not in requested:
            continue
        selected.append(asset)
    unknown = requested - {asset["id"] for asset in manifest["assets"]}
    if unknown:
        message = f"Unknown asset ids: {', '.join(sorted(unknown))}"
        raise SystemExit(message)
    if requested and not selected:
        message = "No requested assets match --kind"
        raise SystemExit(message)
    return selected


def process_asset(asset: dict[str, str], args: argparse.Namespace) -> tuple[dict[str, Any] | None, bool]:
    output = target_path(asset)
    marker = matching_marker(asset, output)
    if args.verify_only:
        if marker is None:
            print(f"FAIL {asset['id']}: matching marker not found", file=sys.stderr)
            return None, False
        errors = verify_inventory(output, marker["files"])
        if errors:
            print(f"FAIL {asset['id']}: {'; '.join(errors[:10])}", file=sys.stderr)
            return marker, False
        print(f"OK   {asset['id']}: {marker['file_count']} files, {marker['total_bytes']} bytes")
        return marker, True

    if marker is not None and not args.force:
        print(f"Skip {asset['id']}: pinned revision already prepared")
        return marker, True
    state_exists = (output / STATE_NAME).is_file()
    stale_marker_exists = (output / MARKER_NAME).is_file() and marker is None
    marker = download(
        asset,
        output,
        args.max_workers,
        reset=args.force or state_exists or stale_marker_exists,
    )
    return marker, True


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    manifest = read_json(MANIFEST_PATH)
    selected = select_assets(args, manifest)

    failures = 0
    for asset in selected:
        _, succeeded = process_asset(asset, args)
        if not succeeded:
            failures += 1

    if not args.verify_only:
        prepared = []
        for asset in manifest["assets"]:
            marker = matching_marker(asset, target_path(asset))
            if marker is not None:
                prepared.append(marker)
        lock = {"schema_version": 1, "manifest": str(MANIFEST_PATH), "assets": prepared}
        FT_ROOT.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
