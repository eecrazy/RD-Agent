#!/usr/bin/env python3
# ruff: noqa: EXE001, FBT003
"""Adopt the completed Fact Checking retry-3 artifact set into consolidated v10."""

import adopt_outer_timeout_batch_v6 as adoption

adoption.SOURCE_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-retry3"
)
adoption.ADOPTION_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v8"
)
adoption.POLICY = "completed_iteration_outer_timeout_v3"
adoption.SPECS = (
    adoption.AdoptionSpec(
        "tablebench_fact_checking",
        "ca94b52a37134991827ae29647e4e9ec",
        "4_record",
        False,
        run_number=2,
    ),
)


if __name__ == "__main__":
    raise SystemExit(adoption.main())
