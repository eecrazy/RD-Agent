#!/usr/bin/env python3
# ruff: noqa: EXE001, FBT003
"""Adopt the audited run-2 outer-timeout candidate sets into consolidated v10."""

import adopt_outer_timeout_batch_v6 as adoption

adoption.ADOPTION_ROOT = (
    adoption.ROOT
    / "finetune_files/logs/paper-matrix"
    / "h20-gpt56-main-rslora-48h-v10-late-failure-adoption-v7"
)
adoption.POLICY = "isolated_terminal_candidate_failure_v2"
adoption.SPECS = (
    adoption.AdoptionSpec(
        "FinanceIQ_gen",
        "2cd7842f3d694d98a6eeae640ac538b1",
        "0_direct_exp_gen",
        True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "chemcotbench_mol_edit",
        "82e7d9a8e28340ad8d26f51d2cd49b50",
        "1_coding",
        True,
        expected_excluded_terminal_durable=True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "chemcotbench_mol_opt",
        "f93b1e2e496143e98e0cd6142d106c6c",
        "1_coding",
        True,
        expected_excluded_terminal_durable=True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "chemcotbench_reaction",
        "8c581269d67d4b1c9f2c0c58dfea4017",
        "0_direct_exp_gen",
        True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "tablebench_data_analysis",
        "f544cc9796f9429399f8cfb73a8fce40",
        "0_direct_exp_gen",
        True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "tablebench_numerical_reasoning",
        "c8a77dc20bea4b66bd1ca49b90748349",
        "1_coding",
        True,
        expected_excluded_terminal_durable=True,
        run_number=2,
    ),
    adoption.AdoptionSpec(
        "tablebench_visualization",
        "1fd8d73ff61f44d18c482b310ca5b3f2",
        "0_direct_exp_gen",
        True,
        run_number=2,
    ),
)


if __name__ == "__main__":
    raise SystemExit(adoption.main())
