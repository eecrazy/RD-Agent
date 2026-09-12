#!/usr/bin/env python3
# ruff: noqa: EXE001, FBT003
"""Adopt the audited Data Analysis retry-6 artifact set into consolidated v10."""

import adopt_outer_timeout_batch_v6 as adoption

adoption.SOURCE_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-retry6"
)
adoption.ADOPTION_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v11"
)
adoption.POLICY = "isolated_terminal_candidate_failure_v6"
adoption.SPECS = (
    adoption.AdoptionSpec(
        "tablebench_data_analysis",
        "e8897a8102474cc4b1864a79a18f038d",
        "0_direct_exp_gen",
        True,
        run_number=1,
    ),
)


if __name__ == "__main__":
    raise SystemExit(adoption.main())
