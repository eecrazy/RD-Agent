# H20 training-constraint ablation

This experiment isolates the training method from data and hyperparameters.
Each ordinary-LoRA/rsLoRA pair uses byte-identical train/validation files,
seed, batch contract, learning rate, schedule, and step count.  The only
pairwise configuration difference is `use_rslora`.

The default eight-GPU layout is:

| GPU(s) | Run |
| --- | --- |
| 0,1 | TableBench full-parameter SFT smoke test (ZeRO-3) |
| 2,3 | ChemCoT ordinary LoRA / rsLoRA |
| 4,5 | FinanceIQ ordinary LoRA / rsLoRA |
| 6,7 | TableBench ordinary LoRA / rsLoRA |

The full-SFT run is deliberately a two-step feasibility smoke test.  The six
adapter runs use the complete pinned training and internal-validation sets.
Outputs are written below
`finetune_files/logs/training-constraint-ablation/<run-name>`.

Run:

```bash
.venv/bin/python reproduction/ft_agent/run_training_constraint_ablation.py \
  --run-name h20-method-equivalence-v1 --gpus 0,1,2,3,4,5,6,7
```

Use `--resume` to collect or finish an interrupted run without overwriting a
successful workspace.  `manifest.json` records source hashes and a masked
pair-contract hash; `results.json` and `RESULTS.md` contain the measured loss,
runtime, GPU-memory, utilization, and artifact checks.
