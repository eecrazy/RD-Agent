#!/usr/bin/env python3
"""Build audited effective-success views for the run-3 outer-timeout batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("/data/github/RD-Agent")
SOURCE_ROOT = (
    ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10"
)
ADOPTION_ROOT = (
    ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v6"
)
CONSOLIDATED_ROOT = (
    ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-consolidated"
)
POLICY = "isolated_terminal_candidate_failure_v1"

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from reproduction.ft_agent.final_test_protocol import has_model_weights  # noqa: E402
from reproduction.ft_agent.run_validation_sweep import (  # noqa: E402
    SweepTarget,
    _formal_training_max_steps,
    _matching_success,
    candidate_set_signature,
    discover_candidates,
)


@dataclass(frozen=True)
class AdoptionSpec:
    benchmark: str
    terminal_workspace_id: str
    terminal_trace_stage: str
    exclude_terminal_workspace: bool
    expected_excluded_terminal_durable: bool = False
    run_number: int = 3

    @property
    def experiment_id(self) -> str:
        return f"main/{self.benchmark}/run-{self.run_number}"


SPECS = (
    AdoptionSpec("FinanceIQ_gen", "62de9b9f43f4413d9cce926334382755", "4_record", False),
    AdoptionSpec("chemcotbench_mol_opt", "ba94ab8d93224f46a70bef0b94fd7db6", "0_direct_exp_gen", True),
    AdoptionSpec("chemcotbench_mol_und", "42c34bb5d56643da92ccf9ae69cd5824", "0_direct_exp_gen", True),
    AdoptionSpec("chemcotbench_reaction", "9d24b3c0ab6c4b9893dc9e71365975b9", "0_direct_exp_gen", True),
    AdoptionSpec("panorama_noc4pc", "cf1917135233464397d19eac5b8d2360", "0_direct_exp_gen", True),
    AdoptionSpec("panorama_par4pc", "4bcc78db35924d53b4bb6d7d7ad181f5", "0_direct_exp_gen", True),
    AdoptionSpec("panorama_pi4pc", "551a7658875a4f5c826483ab3e88d653", "1_coding", True),
    AdoptionSpec("tablebench_data_analysis", "32cbb8dd95c1473cabcfedc5dcc62373", "1_coding", True),
    AdoptionSpec("tablebench_fact_checking", "4a84b70a27fc4c29a8452e8223aeab1b", "0_direct_exp_gen", True),
    AdoptionSpec("tablebench_numerical_reasoning", "61faf7c81fe942918e3804244e7f3acb", "0_direct_exp_gen", True),
    AdoptionSpec("tablebench_visualization", "93900163e51342238414271c15ac36e9", "0_direct_exp_gen", True),
)


def timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def safe_id(experiment_id: str) -> str:
    return experiment_id.replace("/", "__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def latest_trace_snapshot(task_root: Path) -> tuple[int, Path]:
    snapshots: list[tuple[int, int, Path]] = []
    for path in (task_root / "trace/__session__").glob("*/*_*"):
        try:
            snapshots.append((int(path.parent.name), int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    if not snapshots:
        raise RuntimeError(f"No trace snapshots found: {task_root}")
    session, _step, path = max(snapshots)
    return session, path


def trace_workspace_id(snapshot: Path) -> str:
    with snapshot.open("rb") as handle:
        loop = pickle.load(handle)
    loop_index = max(loop.loop_prev_out)
    stage = loop.loop_prev_out[loop_index]
    experiment = stage.get("running") or stage.get("coding") or stage.get("direct_exp_gen")
    if experiment is None:
        raise RuntimeError(f"Trace has no experiment at latest stage: {snapshot}")
    workspace = experiment.experiment_workspace
    path = getattr(workspace, "workspace_path", None)
    if not isinstance(path, Path):
        path = Path(str(path))
    return path.name


def newest_running_snapshot(task_root: Path) -> Path:
    candidates: list[tuple[int, int, Path]] = []
    for path in (task_root / "trace/__session__").glob("*/*_running"):
        try:
            candidates.append((int(path.parent.name), int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    if not candidates:
        raise RuntimeError(f"No intact running-stage trace found: {task_root}")
    return max(candidates)[2]


def make_target(task_root: Path, status: dict[str, Any]) -> SweepTarget:
    return SweepTarget(
        experiment_id=str(status["experiment_id"]),
        benchmark=str(status["benchmark"]),
        model=str(status["model"]),
        benchmark_dataset_path=str(status["benchmark_dataset_path"]),
        task_root=task_root,
    )


def link_workspace_view(source_task: Path, adoption_task: Path, excluded: set[str]) -> tuple[int, int]:
    source_workspace = source_task / "workspace"
    adoption_workspace = adoption_task / "workspace"
    adoption_durable = adoption_workspace / ".ft_model_checkpoints"
    adoption_durable.mkdir(parents=True)

    live_count = 0
    for entry in sorted(source_workspace.iterdir()):
        if entry.name == ".ft_model_checkpoints" or not entry.is_dir():
            continue
        if entry.name in excluded:
            continue
        symlink(entry, adoption_workspace / entry.name)
        live_count += 1

    durable_count = 0
    source_durable = source_workspace / ".ft_model_checkpoints"
    if source_durable.is_dir():
        for entry in sorted(source_durable.iterdir()):
            if not entry.is_dir() or entry.name in excluded:
                continue
            symlink(entry, adoption_durable / entry.name)
            durable_count += 1
    return live_count, durable_count


def failure_class(spec: AdoptionSpec) -> str:
    if not spec.exclude_terminal_workspace:
        return "outer_timeout_after_completed_iteration"
    if spec.terminal_trace_stage == "0_direct_exp_gen":
        return "outer_timeout_during_direct_experiment_generation"
    return "outer_timeout_during_candidate_coding_or_debug_validation"


def build_task(spec: AdoptionSpec, staging_root: Path, adopted_at: str) -> dict[str, Any]:
    source_task = SOURCE_ROOT / safe_id(spec.experiment_id)
    source_status_path = source_task / "status.json"
    source_console_path = source_task / "console.log"
    status = json.loads(source_status_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": spec.experiment_id,
        "state": "failed",
        "return_code": -15,
        "outer_timeout": True,
    }
    mismatches = [key for key, value in expected.items() if status.get(key) != value]
    if mismatches:
        raise RuntimeError(f"{spec.experiment_id}: unexpected source status fields: {mismatches}")
    if not source_console_path.is_file():
        raise RuntimeError(f"{spec.experiment_id}: source console is missing")

    terminal_live = source_task / "workspace" / spec.terminal_workspace_id
    terminal_durable = source_task / "workspace/.ft_model_checkpoints" / spec.terminal_workspace_id
    if not terminal_live.is_dir():
        raise RuntimeError(f"{spec.experiment_id}: terminal workspace is missing: {terminal_live}")

    latest_session, latest_snapshot = latest_trace_snapshot(source_task)
    if latest_snapshot.name != spec.terminal_trace_stage:
        raise RuntimeError(
            f"{spec.experiment_id}: expected trace stage {spec.terminal_trace_stage}, "
            f"found {latest_snapshot.name}",
        )
    observed_workspace = trace_workspace_id(latest_snapshot)
    if observed_workspace != spec.terminal_workspace_id:
        raise RuntimeError(
            f"{spec.experiment_id}: trace workspace {observed_workspace} does not match "
            f"{spec.terminal_workspace_id}",
        )

    source_target = make_target(source_task, status)
    source_candidates = discover_candidates(
        source_target,
        include_baseline=False,
        include_final_outputs=True,
    )
    terminal_candidates = [
        candidate for candidate in source_candidates if candidate.workspace_id == spec.terminal_workspace_id
    ]
    if spec.exclude_terminal_workspace and terminal_candidates:
        raise RuntimeError(
            f"{spec.experiment_id}: excluded terminal workspace has formal candidates: "
            f"{[candidate.candidate_id for candidate in terminal_candidates]}",
        )
    if not spec.exclude_terminal_workspace and not terminal_candidates:
        raise RuntimeError(f"{spec.experiment_id}: completed terminal workspace has no formal candidate")
    if spec.exclude_terminal_workspace:
        durable_present = terminal_durable.is_dir()
        if durable_present != spec.expected_excluded_terminal_durable:
            raise RuntimeError(
                f"{spec.experiment_id}: excluded terminal durable workspace presence "
                f"is {durable_present}, expected {spec.expected_excluded_terminal_durable}",
            )

    adoption_task = staging_root / safe_id(spec.experiment_id)
    adoption_task.mkdir(parents=True)
    excluded = {spec.terminal_workspace_id} if spec.exclude_terminal_workspace else set()
    linked_live, linked_durable = link_workspace_view(source_task, adoption_task, excluded)
    symlink(source_console_path, adoption_task / "console.log")
    symlink(source_status_path, adoption_task / "source_status.json")
    symlink(source_task / "trace", adoption_task / "trace")
    if (source_task / "validation_sweep").is_dir():
        symlink(source_task / "validation_sweep", adoption_task / "validation_sweep")
    if (source_task / "validation_sweep.json").is_file():
        symlink(source_task / "validation_sweep.json", adoption_task / "source_validation_sweep.json")

    final_task = ADOPTION_ROOT / safe_id(spec.experiment_id)
    effective_status = dict(status)
    effective_status.update(
        {
            "state": "succeeded",
            "finished_at": adopted_at,
            "return_code": None,
            "outer_timeout": False,
            "trace_path": str(final_task / "trace"),
            "workspace_path": str(final_task / "workspace"),
            "completion_observation": {
                "kind": "late_failure_artifact_set_adoption",
                "process_succeeded": False,
                "artifact_set_accepted": True,
                "source_state": status["state"],
                "source_return_code": status["return_code"],
                "source_outer_timeout": status["outer_timeout"],
                "source_status_path": str(source_status_path),
                "source_status_sha256": sha256(source_status_path),
                "audit_path": str(final_task / "late_failure_adoption.json"),
                "policy": POLICY,
                "note": (
                    "Effective artifact-set success only; the original timeout failure, return code, "
                    "outer-timeout flag, status, and logs remain unchanged."
                ),
            },
        },
    )
    write_json(adoption_task / "status.json", effective_status)

    adoption_target = make_target(adoption_task, effective_status)
    adoption_candidates = discover_candidates(
        adoption_target,
        include_baseline=False,
        include_final_outputs=True,
    )
    source_identities = [candidate.identity() for candidate in source_candidates]
    adoption_identities = [candidate.identity() for candidate in adoption_candidates]
    if source_identities != adoption_identities:
        raise RuntimeError(f"{spec.experiment_id}: adoption changed the formal candidate set")

    newest_running = newest_running_snapshot(source_task)
    formal_max_steps = _formal_training_max_steps(source_target)
    source_counts = Counter(candidate.source for candidate in source_candidates)
    matching_success = sum(_matching_success(source_target, candidate) is not None for candidate in source_candidates)
    terminal_output = terminal_live / "output"
    status_hash = sha256(source_status_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "policy": POLICY,
        "adopted_at": adopted_at,
        "experiment_id": spec.experiment_id,
        "decision": "eligible_pre_timeout_candidate_set_adopted_as_effective_search_success",
        "held_out_test_used": False,
        "source": {
            "task_root": str(source_task),
            "state": status["state"],
            "return_code": status["return_code"],
            "outer_timeout": status["outer_timeout"],
            "finished_at": status.get("finished_at"),
            "status_sha256": status_hash,
            "console_sha256": sha256(source_console_path),
            "status_is_preserved": True,
            "return_code_is_preserved": True,
            "outer_timeout_is_preserved": True,
            "console_is_preserved": True,
        },
        "terminal_observation": {
            "workspace_id": spec.terminal_workspace_id,
            "failure_stage": spec.terminal_trace_stage,
            "failure_class": failure_class(spec),
            "process_return_code": status["return_code"],
            "outer_timeout": status["outer_timeout"],
            "trace_session": latest_session,
            "trace_snapshot": str(latest_snapshot),
            "trace_snapshot_sha256": sha256(latest_snapshot),
            "live_workspace_path": str(terminal_live),
            "durable_workspace_path": str(terminal_durable),
            "durable_workspace_present": terminal_durable.is_dir(),
            "durable_workspace_presence_expected": (
                spec.expected_excluded_terminal_durable if spec.exclude_terminal_workspace else None
            ),
            "output_model_weights_present": terminal_output.is_dir() and has_model_weights(terminal_output),
            "formal_candidate_count": len(terminal_candidates),
            "formal_candidate_ids": [candidate.candidate_id for candidate in terminal_candidates],
            "physically_excluded_from_adoption": spec.exclude_terminal_workspace,
            "completed_iteration_retained": not spec.exclude_terminal_workspace,
        },
        "formal_provenance_at_adoption": {
            "newest_intact_session": str(newest_running),
            "newest_intact_session_sha256": sha256(newest_running),
            "formal_workspace_count": len(formal_max_steps),
            "formal_training_max_steps": dict(sorted(formal_max_steps.items())),
            "formal_candidate_count": len(source_candidates),
            "candidate_source_counts": dict(sorted(source_counts.items())),
            "candidate_set_signature": candidate_set_signature(source_candidates),
            "matching_successful_validation_count": matching_success,
            "remaining_formal_validation_count": len(source_candidates) - matching_success,
            "held_out_test_used": False,
        },
        "adoption_view": {
            "linked_workspace_count": linked_live,
            "linked_durable_checkpoint_workspace_count": linked_durable,
            "excluded_workspace_ids": sorted(excluded),
            "formal_candidate_set_exactly_matches_source": True,
        },
        "guards": {
            "source_task_is_never_modified_by_adoption": True,
            "original_failure_evidence_is_retained": True,
            "interrupted_workspace_is_not_exposed": spec.exclude_terminal_workspace,
            "completed_terminal_iteration_has_formal_provenance": not spec.exclude_terminal_workspace,
            "all_eligible_candidates_still_require_formal_validation": True,
            "formal_selection_must_be_signed": True,
            "test_scores_may_not_influence_selection": True,
            "held_out_test_may_run_only_once_after_all_formal_selections": True,
            "allow_retest_is_forbidden": True,
        },
    }
    manifest["artifact_signature"] = canonical_signature(manifest)
    write_json(adoption_task / "late_failure_adoption.json", manifest)
    if sha256(source_status_path) != status_hash:
        raise RuntimeError(f"{spec.experiment_id}: source status changed during adoption")
    return manifest


def update_consolidated(manifests: dict[str, dict[str, Any]], adopted_at: str) -> None:
    provenance_path = CONSOLIDATED_ROOT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    overrides = provenance.setdefault("overrides", {})
    for spec in SPECS:
        manifest = manifests[spec.experiment_id]
        overrides[spec.experiment_id] = {
            "source_run": ADOPTION_ROOT.name,
            "source_process_run": SOURCE_ROOT.name,
            "composition_kind": "late_failure_artifact_set_adoption",
            "adoption_policy": POLICY,
            "adoption_artifact_signature": manifest["artifact_signature"],
            "source_process_state": "failed",
            "source_process_return_code": -15,
            "source_process_outer_timeout": True,
            "effective_artifact_state": "succeeded",
            "held_out_test_used": False,
            "replaces_primary_state": "failed",
            "replaces_primary_return_code": -15,
        }
    provenance["updated_at"] = adopted_at
    write_json(provenance_path, provenance)

    for spec in SPECS:
        name = safe_id(spec.experiment_id)
        destination = CONSOLIDATED_ROOT / name
        temporary = CONSOLIDATED_ROOT / f".{name}.adoption-v6-{os.getpid()}.tmp"
        relative = os.path.relpath(ADOPTION_ROOT / name, CONSOLIDATED_ROOT)
        temporary.symlink_to(relative, target_is_directory=True)
        temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Materialize and activate the audited views")
    args = parser.parse_args()

    if ADOPTION_ROOT.exists():
        raise RuntimeError(f"Adoption root already exists: {ADOPTION_ROOT}")
    adopted_at = timestamp()
    staging_root = ADOPTION_ROOT.with_name(f".{ADOPTION_ROOT.name}.staging-{os.getpid()}")
    if staging_root.exists():
        raise RuntimeError(f"Staging root already exists: {staging_root}")

    if not args.apply:
        print(f"Preflight target: {ADOPTION_ROOT}")
        for spec in SPECS:
            source = SOURCE_ROOT / safe_id(spec.experiment_id)
            status = json.loads((source / "status.json").read_text(encoding="utf-8"))
            target = make_target(source, status)
            candidates = discover_candidates(target, include_baseline=False, include_final_outputs=True)
            terminal = sum(candidate.workspace_id == spec.terminal_workspace_id for candidate in candidates)
            print(
                f"{spec.experiment_id}: state={status.get('state')} rc={status.get('return_code')} "
                f"outer_timeout={status.get('outer_timeout')} candidates={len(candidates)} "
                f"terminal_candidates={terminal} exclude={spec.exclude_terminal_workspace}",
            )
        return 0

    staging_root.mkdir(parents=True)
    manifests: dict[str, dict[str, Any]] = {}
    for spec in SPECS:
        manifest = build_task(spec, staging_root, adopted_at)
        manifests[spec.experiment_id] = manifest
        formal = manifest["formal_provenance_at_adoption"]
        print(
            f"BUILT {spec.experiment_id}: candidates={formal['formal_candidate_count']} "
            f"reusable={formal['matching_successful_validation_count']} "
            f"pending={formal['remaining_formal_validation_count']}",
            flush=True,
        )
    staging_root.replace(ADOPTION_ROOT)
    update_consolidated(manifests, adopted_at)

    for spec in SPECS:
        source_status = SOURCE_ROOT / safe_id(spec.experiment_id) / "status.json"
        manifest_path = ADOPTION_ROOT / safe_id(spec.experiment_id) / "late_failure_adoption.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(source_status) != manifest["source"]["status_sha256"]:
            raise RuntimeError(f"{spec.experiment_id}: source failure evidence changed after activation")
        effective = json.loads((CONSOLIDATED_ROOT / safe_id(spec.experiment_id) / "status.json").read_text())
        if effective.get("state") != "succeeded":
            raise RuntimeError(f"{spec.experiment_id}: consolidated effective state is not succeeded")
    print(f"ACTIVATED {len(SPECS)} audited adoption views at {ADOPTION_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
