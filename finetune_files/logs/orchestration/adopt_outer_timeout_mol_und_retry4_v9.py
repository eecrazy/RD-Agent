#!/usr/bin/env python3
# ruff: noqa: EXE001, FBT003
"""Adopt the audited Molecule Understanding retry-4 artifact set into consolidated v10."""

import adopt_outer_timeout_batch_v6 as adoption

adoption.SOURCE_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-retry4"
)
adoption.ADOPTION_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v9"
)
adoption.POLICY = "isolated_terminal_candidate_failure_v4"
adoption.SPECS = (
    adoption.AdoptionSpec(
        "chemcotbench_mol_und",
        "c60e4f18ac524843bb2bd8ebf35d8209",
        "0_direct_exp_gen",
        True,
        run_number=2,
    ),
)


if __name__ == "__main__":
    raise SystemExit(adoption.main())
