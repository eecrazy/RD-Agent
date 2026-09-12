#!/usr/bin/env python3
# ruff: noqa: EXE001, FBT003
"""Adopt the audited Reaction Prediction retry-5 artifact set into consolidated v10."""

import adopt_outer_timeout_batch_v6 as adoption

adoption.SOURCE_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-retry5"
)
adoption.ADOPTION_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v10"
)
adoption.POLICY = "isolated_terminal_candidate_failure_v5"
adoption.SPECS = (
    adoption.AdoptionSpec(
        "chemcotbench_reaction",
        "7453e1d52e2c4bde92991f7eb84a9ded",
        "0_direct_exp_gen",
        True,
        run_number=1,
    ),
)


if __name__ == "__main__":
    raise SystemExit(adoption.main())
