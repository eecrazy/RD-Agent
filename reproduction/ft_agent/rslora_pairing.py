# ruff: noqa: C901, EM101, EM102, PLC0415, PLR0912, PLR0915, TRY003
"""Signed provenance contract for ordinary-LoRA/rsLoRA paired runs.

The paper-policy matrix is allowed to choose Full SFT or ordinary LoRA.  An
rsLoRA comparison is meaningful only when it reuses an ordinary-LoRA formal
workspace byte-for-byte and changes the single ``use_rslora`` scalar.  This
module defines that contract independently of the training scheduler so that
validation, held-out testing, and result collection can revalidate it later.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from rdagent.scenarios.finetune.train.formal_training import (
    FORMAL_TRAINING_EVIDENCE_FILE,
    formal_training_provenance_files,
)

PAIRING_ARTIFACT_FILE = "rslora_pairing.json"
PAIRING_ARTIFACT_SCHEMA_VERSION = 1
PAIRING_MATRIX_SCHEMA_VERSION = 1
PAIRING_RUN_KIND = "ordinary_lora_to_rslora"
PAIRING_SOURCE_POLICY = "paper"
PAIRING_TARGET_POLICY = "rslora"
PAIRING_EXPECTED_SAMPLES = 2000
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
_RSLORA_LINE_RE = re.compile(
    rb"^(?P<prefix>use_rslora[ \t]*:[ \t]*)(?P<value>false)(?P<suffix>[ \t]*(?:\#[^\r\n]*)?)(?P<ending>\r?\n)?$",
    re.IGNORECASE,
)


class PairingContractError(RuntimeError):
    """Raised when an rsLoRA result is not a strict ordinary-LoRA pair."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairingContractError(f"Unable to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise PairingContractError(f"{path} must contain a JSON object")
    return value


def _read_yaml_bytes(source: bytes, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(source.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PairingContractError(f"Unable to parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise PairingContractError(f"{label} must contain a YAML mapping")
    return value


def paired_train_yaml(source: bytes) -> bytes:
    """Return the exact source YAML with only ``false`` changed to ``true``.

    Requiring an explicit top-level scalar makes the textual contract as strict
    as the semantic one: comments, ordering, quoting, and every hyperparameter
    remain byte-identical.
    """
    source_plan = _read_yaml_bytes(source, "source train.yaml")
    if str(source_plan.get("finetuning_type", "")).strip().lower() != "lora":
        raise PairingContractError("Pair source must use finetuning_type: lora")
    if "use_rslora" not in source_plan or source_plan.get("use_rslora") is not False:
        raise PairingContractError("Pair source must explicitly set top-level use_rslora: false")
    if source_plan.get("use_dora", False) is not False:
        raise PairingContractError("Pair source must disable DoRA")

    lines = source.splitlines(keepends=True)
    matches: list[int] = []
    for index, line in enumerate(lines):
        if _RSLORA_LINE_RE.fullmatch(line):
            matches.append(index)
    if len(matches) != 1:
        raise PairingContractError(
            "Pair source must contain exactly one unindented 'use_rslora: false' YAML line",
        )
    index = matches[0]
    match = _RSLORA_LINE_RE.fullmatch(lines[index])
    if match is None:  # pragma: no cover - guarded by the match collection above.
        raise AssertionError("rsLoRA line match disappeared")
    lines[index] = match.group("prefix") + b"true" + match.group("suffix") + (match.group("ending") or b"")
    paired = b"".join(lines)

    paired_plan = _read_yaml_bytes(paired, "paired train.yaml")
    if paired_plan.get("use_rslora") is not True:
        raise PairingContractError("Paired YAML did not enable rsLoRA")
    source_masked = deepcopy(source_plan)
    paired_masked = deepcopy(paired_plan)
    source_masked["use_rslora"] = "<paired-method-switch>"
    paired_masked["use_rslora"] = "<paired-method-switch>"
    if source_masked != paired_masked:
        raise PairingContractError("LoRA/rsLoRA YAML objects differ outside use_rslora")
    return paired


def masked_train_yaml_signature(source: bytes, paired: bytes) -> str:
    """Validate a pair and sign its YAML object with the method bit masked."""
    expected = paired_train_yaml(source)
    if paired != expected:
        raise PairingContractError("Paired train.yaml has byte changes outside use_rslora false -> true")
    source_plan = _read_yaml_bytes(source, "source train.yaml")
    source_plan["use_rslora"] = "<paired-method-switch>"
    return canonical_sha256(source_plan)


def paired_experiment_id(source_experiment_id: str) -> str:
    if not source_experiment_id.startswith("main/"):
        raise PairingContractError(f"Paired source is not a main-table experiment: {source_experiment_id!r}")
    return f"rslora-paired/{source_experiment_id}"


def paired_workspace_id(source_workspace_id: str, source_evidence_signature: str) -> str:
    if not source_workspace_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_workspace_id):
        raise PairingContractError(f"Unsafe source workspace id: {source_workspace_id!r}")
    if not _SIGNATURE_RE.fullmatch(source_evidence_signature):
        raise PairingContractError("Source formal evidence signature is invalid")
    return f"rslora__{source_workspace_id}__{source_evidence_signature[:12]}"


def _input_file_hashes(provenance: Path) -> list[dict[str, str]]:
    excluded = {"train.yaml", FORMAL_TRAINING_EVIDENCE_FILE}
    result: list[dict[str, str]] = []
    for relative in formal_training_provenance_files(provenance):
        if relative in excluded:
            continue
        result.append({"path": relative, "sha256": file_sha256(provenance / relative)})
    return sorted(result, key=lambda item: item["path"])


def build_pair_input_contract(
    *,
    source_experiment_id: str,
    paired_experiment_id_value: str,
    source_workspace_id: str,
    paired_workspace_id_value: str,
    source_evidence_signature: str,
    source_provenance: str | Path,
    paired_provenance: str | Path | None = None,
    expected_samples: int = PAIRING_EXPECTED_SAMPLES,
) -> dict[str, Any]:
    """Build the immutable pre-training input contract for one workspace."""
    source_root = Path(source_provenance)
    source_yaml = (source_root / "train.yaml").read_bytes()
    expected_paired_yaml = paired_train_yaml(source_yaml)
    source_inputs = _input_file_hashes(source_root)
    if paired_provenance is not None:
        paired_root = Path(paired_provenance)
        actual_paired_yaml = (paired_root / "train.yaml").read_bytes()
        masked_signature = masked_train_yaml_signature(source_yaml, actual_paired_yaml)
        paired_inputs = _input_file_hashes(paired_root)
        if paired_inputs != source_inputs:
            raise PairingContractError("Paired provenance data files are not byte-identical to ordinary LoRA")
    else:
        masked_signature = masked_train_yaml_signature(source_yaml, expected_paired_yaml)

    contract: dict[str, Any] = {
        "schema_version": 1,
        "source_experiment_id": source_experiment_id,
        "paired_experiment_id": paired_experiment_id_value,
        "source_workspace_id": source_workspace_id,
        "paired_workspace_id": paired_workspace_id_value,
        "source_training_policy": PAIRING_SOURCE_POLICY,
        "source_training_method": "lora",
        "paired_training_policy": PAIRING_TARGET_POLICY,
        "paired_training_method": "rslora",
        "expected_samples": expected_samples,
        "source_evidence_signature": source_evidence_signature,
        "source_train_yaml_sha256": hashlib.sha256(source_yaml).hexdigest(),
        "paired_train_yaml_sha256": hashlib.sha256(expected_paired_yaml).hexdigest(),
        "masked_train_yaml_signature": masked_signature,
        "input_files": source_inputs,
    }
    contract["input_contract_signature"] = canonical_sha256(contract)
    return contract


def pairing_signature_from_status(status: dict[str, Any]) -> str | None:
    """Return a strict paired-run declaration, or ``None`` for normal tasks."""
    kind = status.get("run_kind")
    signature = status.get("pairing_artifact_signature")
    has_any = kind is not None or signature is not None or status.get("source_experiment_id") is not None
    if not has_any:
        return None
    if kind != PAIRING_RUN_KIND:
        raise PairingContractError(f"Unsupported paired-run kind: {kind!r}")
    if status.get("training_policy") != PAIRING_TARGET_POLICY:
        raise PairingContractError("Paired task must use the rslora training policy")
    if status.get("formal_expected_samples") != PAIRING_EXPECTED_SAMPLES:
        raise PairingContractError("Paired task must retain the exact 2,000-sample contract")
    if not isinstance(signature, str) or not _SIGNATURE_RE.fullmatch(signature):
        raise PairingContractError("Paired task has no valid pairing artifact signature")
    return signature


def _validated_formal_records(
    task_root: Path,
    *,
    expected_samples: int,
    experiment_id: str,
    training_policy: str,
) -> list[dict[str, Any]]:
    # Import lazily to avoid making the standalone matrix runner part of this
    # module's import-time dependency graph.
    if __package__:
        from .run_matrix import validate_task_formal_training
    else:
        from run_matrix import validate_task_formal_training

    records, errors = validate_task_formal_training(
        task_root,
        expected_samples=expected_samples,
        experiment_id=experiment_id,
        training_policy=training_policy,
        require_visible_evidence=True,
    )
    if errors or not records:
        detail = "; ".join(errors) if errors else "no formal records"
        raise PairingContractError(f"Formal training evidence is not clean for {experiment_id}: {detail}")
    return records


def _task_by_id(matrix: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    matches = [task for task in matrix.get("tasks", []) if task.get("experiment_id") == experiment_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise PairingContractError(f"Matrix must contain exactly one task {experiment_id!r}")
    return matches[0]


def current_pair_entries(
    *,
    paired_task_root: Path,
    paired_task: dict[str, Any],
    source_matrix_root: Path,
) -> list[dict[str, str]]:
    """Recompute all source/target pair entries from durable formal evidence."""
    paired_id = str(paired_task["experiment_id"])
    pairing = paired_task.get("pairing")
    if not isinstance(pairing, dict):
        raise PairingContractError("Paired matrix task has no pairing contract")
    source_id = pairing.get("source_experiment_id")
    if not isinstance(source_id, str) or paired_experiment_id(source_id) != paired_id:
        raise PairingContractError("Paired/source experiment ids do not match")
    expected_samples = paired_task.get("formal_expected_samples")
    if expected_samples != PAIRING_EXPECTED_SAMPLES or paired_task.get("data_limit") != expected_samples:
        raise PairingContractError("Paired matrix task does not declare exactly 2,000 samples")

    source_task_root = source_matrix_root / re.sub(r"[^A-Za-z0-9_.-]+", "__", source_id).strip("_")
    source_records = _validated_formal_records(
        source_task_root,
        expected_samples=expected_samples,
        experiment_id=source_id,
        training_policy=PAIRING_SOURCE_POLICY,
    )
    paired_records = _validated_formal_records(
        paired_task_root,
        expected_samples=expected_samples,
        experiment_id=paired_id,
        training_policy=PAIRING_TARGET_POLICY,
    )
    if {record["training_method"] for record in source_records} != {"lora"}:
        raise PairingContractError("Pair source no longer consists solely of ordinary LoRA formal workspaces")
    if {record["training_method"] for record in paired_records} != {"rslora"}:
        raise PairingContractError("Paired task no longer consists solely of rsLoRA formal workspaces")

    source_by_id = {str(record["workspace_id"]): record for record in source_records}
    paired_by_id = {str(record["workspace_id"]): record for record in paired_records}
    expected_contracts = pairing.get("expected_pairs")
    if not isinstance(expected_contracts, list) or not expected_contracts:
        raise PairingContractError("Paired matrix task has no expected workspace pairs")
    if not all(isinstance(item, dict) for item in expected_contracts):
        raise PairingContractError("Paired matrix expected-pair entries must be objects")
    expected_by_source = {str(item.get("source_workspace_id")): item for item in expected_contracts}
    if len(expected_by_source) != len(expected_contracts) or set(expected_by_source) != set(source_by_id):
        raise PairingContractError("Paired matrix does not cover every ordinary-LoRA formal workspace exactly once")

    entries: list[dict[str, str]] = []
    seen_paired: set[str] = set()
    for source_workspace_id in sorted(source_by_id):
        source_record = source_by_id[source_workspace_id]
        expected = expected_by_source[source_workspace_id]
        expected_paired_id = paired_workspace_id(source_workspace_id, source_record["evidence_signature"])
        if expected.get("paired_workspace_id") != expected_paired_id:
            raise PairingContractError(f"Unexpected paired workspace id for {source_workspace_id}")
        if expected_paired_id in seen_paired or expected_paired_id not in paired_by_id:
            raise PairingContractError(f"Missing or duplicate rsLoRA pair for {source_workspace_id}")
        seen_paired.add(expected_paired_id)
        paired_record = paired_by_id[expected_paired_id]
        contract = build_pair_input_contract(
            source_experiment_id=source_id,
            paired_experiment_id_value=paired_id,
            source_workspace_id=source_workspace_id,
            paired_workspace_id_value=expected_paired_id,
            source_evidence_signature=source_record["evidence_signature"],
            source_provenance=source_record["provenance_path"],
            paired_provenance=paired_record["provenance_path"],
            expected_samples=expected_samples,
        )
        if expected != contract:
            raise PairingContractError(f"Recorded input contract changed for {source_workspace_id}")
        entries.append(
            {
                "source_workspace_id": source_workspace_id,
                "paired_workspace_id": expected_paired_id,
                "source_evidence_signature": str(source_record["evidence_signature"]),
                "paired_evidence_signature": str(paired_record["evidence_signature"]),
                "input_contract_signature": str(contract["input_contract_signature"]),
            },
        )
    if seen_paired != set(paired_by_id):
        raise PairingContractError("Paired task contains an unpaired rsLoRA formal workspace")
    return entries


def make_pairing_artifact(task_root: str | Path) -> dict[str, Any]:
    """Create the completed signed task artifact from the paired matrix contract."""
    task = Path(task_root).resolve()
    matrix_path = task.parent / "matrix.json"
    matrix = _read_json(matrix_path)
    comparison = matrix.get("paired_comparison")
    if not isinstance(comparison, dict):
        raise PairingContractError("Matrix has no paired-comparison declaration")
    paired_task = _task_by_id(matrix, str(_read_json(task / "status.json")["experiment_id"]))
    source_root = Path(str(comparison.get("source_matrix_root", ""))).resolve()
    entries = current_pair_entries(
        paired_task_root=task,
        paired_task=paired_task,
        source_matrix_root=source_root,
    )
    payload: dict[str, Any] = {
        "schema_version": PAIRING_ARTIFACT_SCHEMA_VERSION,
        "run_kind": PAIRING_RUN_KIND,
        "state": "succeeded",
        "source_matrix_root": str(source_root),
        "source_matrix_manifest_sha256": comparison.get("source_matrix_manifest_sha256"),
        "source_experiment_id": paired_task["pairing"]["source_experiment_id"],
        "paired_experiment_id": paired_task["experiment_id"],
        "expected_samples": paired_task["formal_expected_samples"],
        "source_training_policy": PAIRING_SOURCE_POLICY,
        "paired_training_policy": PAIRING_TARGET_POLICY,
        "pairs": entries,
    }
    payload["artifact_signature"] = canonical_sha256(payload)
    return payload


def validate_pairing_artifact(
    task_root: str | Path,
    *,
    experiment_id: str,
    expected_samples: int,
    expected_signature: str | None = None,
) -> dict[str, Any]:
    """Revalidate matrix, source LoRA, target rsLoRA, and the signed pair set."""
    task = Path(task_root).resolve()
    matrix_path = task.parent / "matrix.json"
    matrix = _read_json(matrix_path)
    if matrix.get("training_policy") != PAIRING_TARGET_POLICY:
        raise PairingContractError("Pair matrix training policy is not rslora")
    comparison = matrix.get("paired_comparison")
    if not isinstance(comparison, dict) or comparison.get("schema_version") != PAIRING_MATRIX_SCHEMA_VERSION:
        raise PairingContractError("Pair matrix comparison declaration is missing or unsupported")
    if comparison.get("run_kind") != PAIRING_RUN_KIND:
        raise PairingContractError("Pair matrix run kind is invalid")
    source_root = Path(str(comparison.get("source_matrix_root", ""))).resolve()
    source_matrix_path = source_root / "matrix.json"
    if not source_matrix_path.is_file():
        raise PairingContractError(f"Source matrix manifest is missing: {source_matrix_path}")
    source_hash = file_sha256(source_matrix_path)
    if source_hash != comparison.get("source_matrix_manifest_sha256"):
        raise PairingContractError("Source matrix manifest changed after pairing")
    source_matrix = _read_json(source_matrix_path)
    if source_matrix.get("suite") != "main" or source_matrix.get("training_policy", "paper") != "paper":
        raise PairingContractError("Pair source must be a main-suite paper-policy matrix")
    expected_source_tasks = comparison.get("source_task_count")
    if (
        isinstance(expected_source_tasks, bool)
        or not isinstance(expected_source_tasks, int)
        or expected_source_tasks < 1
        or len(source_matrix.get("tasks", [])) != expected_source_tasks
    ):
        raise PairingContractError("Source matrix task-count contract changed")

    paired_task = _task_by_id(matrix, experiment_id)
    if paired_task.get("formal_expected_samples") != expected_samples or expected_samples != PAIRING_EXPECTED_SAMPLES:
        raise PairingContractError("Paired task expected-sample contract changed")
    source_id = paired_task.get("pairing", {}).get("source_experiment_id")
    if not isinstance(source_id, str):
        raise PairingContractError("Paired task source experiment id is missing")
    _task_by_id(source_matrix, source_id)

    status = _read_json(task / "status.json")
    if status.get("state") != "succeeded" or status.get("experiment_id") != experiment_id:
        raise PairingContractError("Paired task status is not a matching success")
    status_signature = pairing_signature_from_status(status)
    if expected_signature is not None and status_signature != expected_signature:
        raise PairingContractError("Paired task status signature changed")

    artifact = _read_json(task / PAIRING_ARTIFACT_FILE)
    signature = artifact.get("artifact_signature")
    signed = {key: value for key, value in artifact.items() if key != "artifact_signature"}
    if not isinstance(signature, str) or signature != canonical_sha256(signed):
        raise PairingContractError("Pairing artifact signature is invalid")
    if status_signature != signature:
        raise PairingContractError("Task status and pairing artifact signatures differ")
    expected_fields = {
        "schema_version": PAIRING_ARTIFACT_SCHEMA_VERSION,
        "run_kind": PAIRING_RUN_KIND,
        "state": "succeeded",
        "source_matrix_root": str(source_root),
        "source_matrix_manifest_sha256": source_hash,
        "source_experiment_id": source_id,
        "paired_experiment_id": experiment_id,
        "expected_samples": expected_samples,
        "source_training_policy": PAIRING_SOURCE_POLICY,
        "paired_training_policy": PAIRING_TARGET_POLICY,
    }
    mismatches = [key for key, value in expected_fields.items() if artifact.get(key) != value]
    if mismatches:
        raise PairingContractError("Pairing artifact fields changed: " + ", ".join(mismatches))
    entries = current_pair_entries(
        paired_task_root=task,
        paired_task=paired_task,
        source_matrix_root=source_root,
    )
    if artifact.get("pairs") != entries:
        raise PairingContractError("Pairing artifact no longer matches formal source/target evidence")
    return artifact
