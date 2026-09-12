# FT-Dojo / FT-Agent reproduction

This directory targets the complete run-level experiment set reported by the
ICML 2026 FT-Dojo paper. It inventories all 125 jobs, runs every job for which
the authors released enough implementation detail, and keeps the remaining
jobs visible as explicit publication blockers. Unreleased baselines are not
silently dropped or labeled as reproduced.

The checkout used for this reproduction is pinned to RD-Agent commit
`6762f84f9bc0f5c6486c50a00e128a57ac6c3683`. The downloaded datasets, target
models, OpenCompass fork, and Python packages are pinned or recorded as
described below.

## Reproduction scope

`paper_matrix.py` is the canonical inventory:

| Paper group | Jobs | Coverage in this checkout |
| --- | ---: | --- |
| Base Qwen2.5-7B | 13 | Runnable reconstruction using the released evaluator |
| Base Qwen2.5-3B | 5 | Runnable reconstruction using the released evaluator |
| FT-Agent main results | 39 | Released runner: 13 tasks x 3 independent runs |
| FT-Agent 5k, GPT-4o, and 3B ablations | 15 | Released runner: 5 tasks x 3 settings |
| FT-Agent DeepSeek and Qwen3.5 planners | 10 | Released runner: 5 tasks x 2 planners |
| Manual SFT | 13 | Protocol only; exact human task artifacts are unavailable |
| Tool-augmented OpenHands, 12 hours | 13 | Protocol only; custom harness is unavailable |
| LLM-assisted Manual SFT | 5 | Protocol only; task-specific human artifacts are unavailable |
| Codex | 5 | Protocol only; exact agent harness is unavailable |
| Claude Code | 5 | Protocol only; exact agent harness is unavailable |
| OpenHands AIME, 24 hours | 2 | Protocol only; custom harness is unavailable |
| **Total** | **125** | **82 executable; 43 blocked on unpublished artifacts** |

`matrix.py` defines the 64 released FT-Agent jobs. `run_base_eval.py` adds the
18 Base jobs and records them as reconstructions because the paper does not
publish a standalone Base launcher. It uses the released model configuration,
benchmark adapters, and validation/test range function. The range is disjoint
within one configured dataset view, but the released TableBench Data Analysis
configuration contains overlapping views; see the protocol audit below.

The total counts unique executions, not every appearance of a result in a
paper table. Deduplicating results reused by later tables gives:

| Paper accounting increment | New jobs | Composition and reuse |
| --- | ---: | --- |
| Main results | 78 | 13 Base + 13 Manual SFT + 13 OpenHands + 39 FT-Agent (13 tasks x 3 runs) |
| Ablation-only additions | 20 | Five tasks each for 5k, GPT-4o planner, Base 3B, and FT-Agent 3B; Base 7B and default FT-Agent entries are reused |
| LLM-assisted Manual SFT | 5 | Five new assisted-human jobs |
| Frontier-table-only additions | 20 | Five jobs each for Codex, Claude Code, DeepSeek-V3.2, and Qwen3.5; the table reuses Base, OpenHands, assisted Manual SFT, GPT-4o, and representative GPT-5.2 FT-Agent results |
| OpenHands AIME 24-hour check | 2 | Two new extended-budget runs |
| **Unique total** | **125** | Table reuse is counted once |

The appendix explicitly describes focused-comparison FT-Agent values as
representative single-run scores; they are not additional runs beyond the
inventory above.

Every FT-Agent job has the paper's 12-hour end-to-end limit. Their maximum
scheduled budget is 768 GPU-hours, so eight GPUs give an ideal lower bound of
about 96 hours for those 64 jobs. Initialization, uneven task lengths,
failures, and retries increase elapsed time.

An audit of the pinned checkout and official GitHub remote refs on 2026-08-27
found only prose for the remaining 43 jobs. In particular, the paper's
OpenHands `LlamaFactoryTool`, `OpenCompassTool`, and `main.py` scaffold, the
Manual SFT per-task work products, and the exact Codex/Claude harnesses are not
published. Those jobs cannot be reproduced exactly until the authors release
the missing artifacts; implementing a new approximation would be a different
experiment. The FT-Agent planner suite is not a substitute for the OpenHands,
Codex, or Claude Code baselines.

## Released protocol versus appendix prose

For reported-result fidelity, the runners preserve the executable behavior of
the pinned RD-Agent checkout and its pinned OpenCompass fork. The appendix
describes context-length filtering followed by a centralized random partition,
but the released code instead applies these exact slice expressions to every
configured OpenCompass dataset view:

```python
"[:min(100, len(index_list)//2)]"
"[-min(100, len(index_list)//2):]"
```

ChemCoTBench, TableBench, and PANORAMA loaders shuffle deterministically with
seed 42 before slicing. AIME and FinanceIQ retain source order. The released
implementation does not contain the appendix's centralized filtering and
partitioning step. Auditing the pinned assets and installed configuration gives
the following executed cardinalities:

| Benchmark | Released validation / test | Appendix validation / test |
| --- | ---: | ---: |
| AIME 2025 | 15 / 15 | 15 / 15 |
| PANORAMA PAR4PC, NOC4PC, PI4PC | 100 / 100 each | 100 / 100 each |
| ChemCoTBench Mol_Und | 160 / 160 | 160 / 160 |
| ChemCoTBench Mol_Edit | 50 / 50 | 50 / 50 |
| ChemCoTBench Mol_Opt | 300 / 300 | 300 / 300 |
| ChemCoTBench Reaction | 237 / 237 | 237 / 237 |
| TableBench Data Analysis | 200 / 200 emitted; 169 / 167 unique | 170 / 170 |
| TableBench Fact Checking | 48 / 48 | 48 / 48 |
| TableBench Numerical Reasoning | 100 / 100 | 197 / 197 |
| TableBench Visualization | 25 / 25 | 25 / 25 |
| FinanceIQ | 472 / 472 across ten subject views | holistic 100 / 100 |

FinanceIQ exposes ten subject configurations. Nine contain 100 test rows and
contribute 50 rows per split; actuarial finance mathematics contains 44 and
contributes 22, for `9 x 50 + 22 = 472` rows per split. The released FinanceIQ
validation and test records do not overlap, but this is a subject-stratified
evaluation rather than the appendix's holistic 100/100 evaluation.

TableBench Data Analysis exposes four subtype-specific views and a fifth
supposedly `other` view. Each subtype view contributes 25 rows, while `other`
filters only on `qtype='DataAnalysis'` and therefore contributes 100 rows that
include records from the four subtype views. The released split consequently
executes 200 rows on each side: validation contains 169 unique records and 31
duplicate executions, while test contains 167 unique records and 33 duplicate
executions. Across the overlapping views, 51 unique source records occur in
both validation and test. Thus, the first/last slices are disjoint within each
individual view but do not establish record-level disjointness for this
aggregate configuration.

These differences are documented rather than silently patched. Retrofitting
the appendix counts would change the released evaluation granularity and could
break comparability with reported results; an appendix-conformant partition
should be reported as a separate experiment.

The pinned OpenCompass fork is Jensen246/OpenCompass commit
`abc7c91a485bc1e3188e71e330a7140be385da29`. Its released AIME module,
`aime2025_cascade_eval_gen_5e9f4f`, has a prompt that asks for both `A`/`B` and
`CORRECT`/`INCORRECT`. `CascadeEvaluator` intentionally accepts either `A` or
a response beginning with `CORRECT`, so both instructions are compatible in
execution. Generated configs explicitly import this module. The prompt is
therefore retained even though the appendix presents a shorter, single-format
version.

Blind-test visibility is independently regression-tested. Iterative search now
runs only `benchmark` (validation); it neither evaluates nor persists
`benchmark_test`. A sentinel test verifies that held-out evaluation is not
called and that an injected legacy held-out payload is absent from runner
feedback, trace/SOTA information, proposal prompts, and experiment-feedback
prompts:

```bash
.venv/bin/python -m pytest -q test/finetune/test_prompt_blindness.py
```

The current result is `2 passed`. A separate post-selection runner evaluates
the validation-selected checkpoint once and records a selection signature.
This verifies the agent prompt boundary; it does not erase the separate
TableBench configuration overlap described above.

## Local layout

All mutable state is kept inside this checkout:

| Path | Contents |
| --- | --- |
| `.venv/` | RD-Agent Python 3.11 environment |
| `finetune_files/datasets/` | Pinned and post-processed Hugging Face datasets |
| `finetune_files/benchmarks/pinned/` | Pinned evaluation datasets used by OpenCompass |
| `finetune_files/models/` | Pinned Qwen target models |
| `finetune_files/conda_envs/` | Python 3.10 training and OpenCompass environments |
| `finetune_files/cache/` | Hugging Face, pip, conda, Torch, Triton, and vLLM caches |
| `finetune_files/environment-locks/` | Pip freezes, conda explicit locks, and install metadata |
| `finetune_files/logs/paper-matrix/` | Matrix manifests, status files, traces, workspaces, and console logs |
| `finetune_files/logs/paper-base/` | Base-model configs, split results, summaries, and status files |
| `finetune_files/logs/paper-report/` | Optional reports combining one or more FT-Agent and Base runs |

`assets.json` pins 13 assets: six training dataset revisions, five evaluation
dataset revisions, and the Qwen2.5 7B/3B model revisions. BioProBench is
prepared even though it is not in the paper's 13-task matrix because the
released scenario's startup path prepares every registered training dataset.

The evaluation assets for AIME 2025, PANORAMA, FinanceIQ, and TableBench are
public. Each matrix task passes its prepared path through
`FT_BENCHMARK_DATASET_PATH`; the generated OpenCompass configuration replaces
both the top-level dataset path and any nested cascade-evaluator `dataset_cfg`
paths. FinanceIQ also skips its legacy unpinned Git download hook when this
pinned path is present.

The gated ChemCoTBench evaluation payload is available byte-for-byte from the
public `little1d/MolAct` Git repository at commit
`7f39250c77bf6d3527f8bb433b685df22f2f0182`. The manifest records the Git blob
OID of every file at the pinned Hugging Face revision
`4cfab96c6f511a504519e2cc003521b6afbac338`; the downloader verifies both the
mirror inputs and the installed outputs against those OIDs. The mirror's
`mol_opt/logp.json` omits the source asset's final newline, so the downloader
restores that byte before verifying its output OID. This produces the exact
21-file benchmark, not an approximate substitute.

The AIME 2025 Hugging Face asset contains separate AIME-I and AIME-II JSONL
files. The downloader concatenates them in that order to produce
`aime2025.jsonl`. Its SHA-256 is
`dee94372847a43a5eb6b0a439047efa82f3804db221a55dc640053425315dd4b`, and its
records exactly match the OpenCompass AIME 2025 archive pinned by the released
configuration's MD5 `aa18cd5d2e2de246c5397f5eb1e61004`.

The asset downloader records an inventory with the size and SHA-256 hash of
every file. The backend installer records both `pip freeze --all` and
`conda list --explicit` after successful validation.

## Required API configuration

The FT-Agent loop requires an OpenAI-compatible provider. This checkout uses a
local Responses-only endpoint through `responses_adapter.py`:

```dotenv
OPENAI_API_KEY=local-no-key-required
OPENAI_API_BASE=http://127.0.0.1:8313/v1
FT_API_PROTOCOL=responses
FT_RESPONSES_MODEL=gpt-5.6-sol
```

The paper's model strings remain unchanged in task definitions and reports:

- `gpt-5.2` for the main planner;
- `gpt-4o` for the planner ablation;
- `DeepSeek-V3.2` and `Qwen3.5-397B-A17B` for the alternative planner suite;
- `gpt-5` and `gpt-4o-mini` for the current strong/weak data-processing pools
  and AIME judge configuration.

In Responses mode the adapter forces LiteLLM's `openai/` provider prefix,
translates Chat Completions requests (including streaming and JSON output), and
routes every logical model name to `gpt-5.6-sol`. Run manifests and task status
files record the sanitized route with `fidelity: model-adapted`. Results from
this configuration are therefore model-adapted experiments, not strict
model-provider reproductions of the paper. For a native Chat Completions
provider, omit `FT_API_PROTOCOL` and `FT_RESPONSES_MODEL`; that provider must
recognize the logical model strings itself.

`HF_TOKEN` is optional for public assets. Approved access to the canonical
`IDEA-AI4S/ChemCoTDataset` repository is required only when downloading the
gated ChemCoT training data. The scripts read a token from the environment and
never need it on the command line.

## Preparation status on 2026-08-27

- The uv environment and both project-local Conda backends are installed and
  locked.
- Both Qwen models and all 11 dataset assets are verified (13/13 assets).
  Approved ChemCoT access produced the pinned 22-file, 174,022,161-byte
  training asset at revision `d88a8af379ba4a7e79f58c04590ad037e79b06ba`.
- All 19 ChemCoTBench subtasks load through the four installed OpenCompass
  dataset classes from the pinned local path, yielding 1,495 rows in total.
- The local `gpt-5.6-sol` endpoint is reachable through `/v1/responses`.
  OpenAI SDK and LiteLLM ordinary/streaming calls pass through the local
  Responses adapter; the upstream Chat Completions endpoint is not used.
- A complete one-sample Qwen2.5-3B/OpenCompass smoke run succeeded at
  `finetune_files/logs/smoke/opencompass-aime2024/20260827_003310`. The model
  predicted 196 for gold answer 204, so the expected accuracy was 0%; config,
  inference, prediction, evaluation, and summary generation all completed.
- All 36 generated Base configs (18 jobs x validation/test) load through the
  installed MMEngine/OpenCompass stack. This includes synchronized AIME
  cascade-evaluator paths/ranges and pinned FinanceIQ paths.
- Full runner preflight passes for all 64 FT-Agent jobs and all 18 Base jobs
  without querying GPUs. This includes the local Responses model-discovery
  check and AIME judge routing.
- No reported paper job has been launched. All eight H20 GPUs were occupied by
  another workload during preparation.

## 1. Validate the project environment

Run all commands from the repository root:

```bash
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/ruff check reproduction/ft_agent \
  test/finetune/test_prompt_blindness.py \
  test/finetune/test_collect_results.py \
  test/finetune/test_final_test_runner.py \
  test/finetune/test_workspace_model_checkpoint.py \
  test/finetune/test_responses_adapter.py \
  test/finetune/test_runner_preflight.py
.venv/bin/python -m py_compile reproduction/ft_agent/*.py
.venv/bin/python -m pytest -q \
  test/finetune/test_prompt_blindness.py \
  test/finetune/test_collect_results.py \
  test/finetune/test_final_test_runner.py \
  test/finetune/test_workspace_model_checkpoint.py \
  test/finetune/test_responses_adapter.py \
  test/finetune/test_runner_preflight.py
```

The expected Git revision is
`6762f84f9bc0f5c6486c50a00e128a57ac6c3683`.

## 2. Download and verify assets

Downloads are resumable and are always requested at the revisions in
`assets.json`:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/download_assets.py

.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/download_assets.py --verify-only
```

A successful complete verification prints `OK` for all 13 assets. To process a
subset, repeat `--asset`, for example:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/download_assets.py \
  --asset model-qwen2.5-7b-instruct
```

Do not use `--force` unless a pinned asset really needs to be downloaded and
post-processed again.

If the machine-wide `HF_ENDPOINT` points at a public mirror, use the official
endpoint for the gated ChemCoT training asset after access has been approved:

```bash
env HF_ENDPOINT=https://huggingface.co \
  .venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/download_assets.py \
  --asset dataset-chemcot
```

Public mirrors can authenticate a Hugging Face token while still returning
403 for gated files, so a successful `whoami` response alone does not prove
repository access.

Both ChemCoT manifest repository IDs are gated aliases:
`OpenMol/ChemCoTDataset` redirects to `IDEA-AI4S/ChemCoTDataset`, and
`OpenMol/ChemCoTBench` redirects to `IDEA-AI4S/ChemCoTBench`. The approved
training payload and the benchmark payload are now both installed and verified
against their pinned manifests. If access is unavailable on another machine,
the downloader preserves a partial directory but does not write
`.rdagent-asset.json`, so matrix preflight reports the missing asset.

## 3. Install and verify the backends

Docker is not required. The installer creates the two named conda environments
under `finetune_files/conda_envs/` and validates imports, CLIs, CUDA access, and
`pip check`:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/install_backends.py
```

The training environment uses Torch 2.9.0 with CUDA 12.8. Because no compatible
prebuilt FlashAttention wheel is available for this Python/Torch/CUDA
combination, the installer adds a project-local CUDA 12.8 compiler and builds
`flash-attn==2.8.3.post1` for compute capability 9.0. The OpenCompass
environment uses `vllm==0.12.0` and the Jensen246/OpenCompass fork at commit
`abc7c91a485bc1e3188e71e330a7140be385da29`.

If the default conda-forge endpoint is unreliable on the local network, set
`FT_CONDA_FORGE_CHANNEL` to a trusted conda-forge mirror URL. The installer
still uses `--override-channels`, so unrelated global Conda channels are never
mixed into either environment.

Useful diagnostics are:

```bash
.venv/bin/python reproduction/ft_agent/install_backends.py --dry-run
finetune_files/conda_envs/llm_finetune/bin/llamafactory-cli version
finetune_files/conda_envs/opencompass/bin/opencompass --help
```

Before launching the paper matrix, exercise the complete local inference and
evaluation path on one AIME 2024 sample. The configuration uses the downloaded
Qwen2.5-3B model, one GPU, batch size 1, and 30% vLLM memory utilization. The
`.env` cache settings keep the OpenCompass dataset, vLLM, Torch, Triton, and
XDG state under `finetune_files/cache/`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/dotenv run -- \
  finetune_files/conda_envs/opencompass/bin/opencompass \
  reproduction/ft_agent/opencompass_smoke.py \
  --work-dir finetune_files/logs/smoke/opencompass-aime2024
```

OpenCompass writes the generated response under `predictions/`, the evaluator
output under `results/`, and CSV/JSON summaries under `summary/` in the
timestamped work directory. The verified run above demonstrates the entire
local path; its 0% score is an expected model answer mismatch, not a harness
failure.

## 4. Inspect the complete paper inventory

Print group counts or machine-readable run records:

```bash
.venv/bin/python reproduction/ft_agent/paper_matrix.py
.venv/bin/python reproduction/ft_agent/paper_matrix.py --json
```

The command asserts a total of 125 unique job IDs before printing anything.

## 5. Run the Base evaluations

Inspect all 18 Base jobs first:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_base_eval.py \
  --suite all --dry-run --gpus 0
```

Validate the selected assets, backend, and judge configuration without
querying GPUs or creating a run directory:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_base_eval.py \
  --suite all --preflight-only
```

Each job evaluates validation and test in sequence. The validation range is
the first `min(100, N/2)` examples and the test range is the last
`min(100, N/2)`, matching the released FT-Dojo evaluator. They do not overlap
within one dataset view. The released TableBench Data Analysis benchmark
aggregates overlapping views, so its underlying source records are not fully
disjoint; see "Released protocol versus appendix prose."

After the judge API and GPUs are available, run or resume the complete Base
matrix:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_base_eval.py \
  --suite all --gpus 0,1,2,3,4,5,6,7 --run-name paper-base

.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_base_eval.py \
  --suite all --gpus 0,1,2,3,4,5,6,7 \
  --run-name paper-base --resume
```

While the judge is unavailable, this selector isolates the non-AIME jobs. Do
not launch it until GPUs are free:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_base_eval.py \
  --suite all \
  --only '^base/(7b|3b)/(chemcotbench_.*|panorama_.*|FinanceIQ_gen|tablebench_.*)$' \
  --gpus 0,1,2,3,4,5,6,7 --run-name paper-base-public
```

Base state is written under
`finetune_files/logs/paper-base/<run-name>/`; each split retains its rendered
config, console log, timestamped OpenCompass output, parsed summary rows, and
atomically updated `status.json`.

## 6. Inspect and run the FT-Agent matrix

Always inspect the selected tasks first:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite all --dry-run
```

Run the complete readiness check without querying GPUs or creating a run
directory:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite all --preflight-only
```

Run the three main-paper seeds on all available GPUs. By default the runner
keeps two FT-Agent pipelines in flight per GPU, overlapping API/CPU data
preparation from one task with the GPU work of another. The project-local
`llamafactory-cli` and `opencompass` entrypoints serialize actual training and
evaluation through GPU leases. The explicit `paper` policy preserves the
released agent's choice between full-parameter SFT and LoRA instead of forcing
an adapter method:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite main --training-policy paper \
  --gpus 0,1,2,3,4,5,6,7 --run-name paper-main
```

Four method policies are available. `paper` follows the generated hypothesis
and permits Full SFT or LoRA; `full` forces Full SFT; `lora` forces ordinary
LoRA (`use_rslora: false`); and `rslora` forces rsLoRA
(`use_rslora: true`). All policies reject DoRA so it cannot become an
uncontrolled variable. Use the three controlled policies only for explicitly
reported ablations. In particular, a run that forces rsLoRA for every task is
not a paper-policy reproduction.

Ordinary LoRA and rsLoRA lease one GPU per training process. Full SFT leases an
atomic GPU group, injects the project ZeRO-3 configuration, and launches via
`torchrun`: two H20s for `cutoff_len <= 8192`, four for longer contexts. Set
`FT_FULL_SFT_GPUS` explicitly when a larger Full-SFT group is required. The
wrapper adjusts per-device batch size and gradient accumulation so that
`per_device_train_batch_size * gradient_accumulation_steps * world_size`
remains equal to the generated single-process global batch; it rejects a
configuration when that invariant cannot be represented exactly.

The 12-hour end-to-end search budget is enforced between FT-Agent steps. A
formal training step that has already started is allowed to finish all of its
configured epochs: the matrix runner raises `FT_FULL_TIMEOUT` to a 100-hour
safety floor while preserving any larger operator override. This guard does
not grant another search iteration; it only prevents slower H20 execution from
turning a complete 2,000-sample schedule into a partial training artifact.

Use `--workers-per-gpu 1` for the former strictly sequential behavior, or set a
higher value only when the planner/data API has enough capacity. Increasing
this value does not run two training jobs on one card; it fills otherwise idle
GPU windows by preparing later tasks ahead of time.

Resume the same run after an interruption. Tasks whose `status.json` says
`succeeded` are skipped:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite main --gpus 0,1,2,3,4,5,6,7 \
  --run-name paper-main --resume
```

Use a regular expression to smoke-test or select one task:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite main --only '^main/aime25/run-1$' --gpus 0 --dry-run
```

For a difficult task that exhausted the normal 12-hour budget, use a fresh
retry run and raise only that selected task to at most 48 hours. A fresh run
name keeps the changed budget and artifacts distinct from the original matrix
manifest:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_matrix.py \
  --suite main --only '^main/aime25/run-1$' --task-timeout 48h \
  --gpus 0 --run-name paper-main-retry48h
```

The preflight check refuses to start unless the selected target models,
datasets, conda backends, and `OPENAI_API_KEY` are available. Do not use
`--skip-preflight` for reported experiments.

Each task writes atomically updated state to:

```text
finetune_files/logs/paper-matrix/<run-name>/<experiment-id>/status.json
```

Its `console.log`, RD-Agent trace, and workspace are kept beside that file. A
successful task is not rerun with `--resume`. A selected failed or interrupted
task directory is deliberately **not** reused: the runner exits and asks you to
archive that directory first. This prevents a retry from restoring an old trace
or appending to its console log. When `--only` targets archived tasks in an
existing run, the original full `matrix.json` is validated and preserved, so a
subset retry cannot silently reduce the run's coverage contract.

For a terminal matrix, inspect and then apply that recovery operation with the
strict retry helper. It keeps only tasks whose current status and durable formal
evidence agree, archives every other existing task root, and restores only a
validated autonomous method lock. Explicitly named failed generated datasets
can be quarantined in the same auditable transaction:

```bash
.venv/bin/python reproduction/ft_agent/prepare_matrix_retry.py \
  --matrix-run paper-main \
  --quarantine-dataset failed_generated_dataset

.venv/bin/python reproduction/ft_agent/prepare_matrix_retry.py \
  --matrix-run paper-main \
  --quarantine-dataset failed_generated_dataset \
  --apply
```

Apply mode refuses to run while a task status or associated Python process is
still active. Archives and the retry journal are written below
`finetune_files/quarantine/matrix-retries/`; no old status, trace, workspace,
or console file is copied back into the live run root.

The search runner evaluates validation only. When an accepted workspace is
restored, its top-level inference artifact is hard-linked into the sibling
`.ft_model_checkpoints/` store and restored with the lightweight code
checkpoint. This keeps the validation-selected adapter available without
embedding hundreds of megabytes in session pickles.

## 7. Evaluate selected checkpoints on held-out test

Do not run final evaluation until selection is fixed. First inspect the exact
selection and then check all pending assets, model weights, and API routing;
neither command executes a benchmark:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_final_test.py \
  --matrix-run paper-main --dry-run

.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_final_test.py \
  --matrix-run paper-main --preflight-only
```

Evaluate each pending validation-selected checkpoint on the held-out range:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python reproduction/ft_agent/run_final_test.py \
  --matrix-run paper-main --gpus 0,1,2,3,4,5,6,7 --resume
```

Each task receives a `final_test.json` containing the selected history/loop,
model identity signature, fixed test range, result, and sanitized API route.
A matching successful artifact is immutable by default and is reused with
`--resume`. A failed prior attempt or changed selection also blocks another
held-out call; `--allow-retest` is an explicit protocol exception, not a normal
resume option.

Runs created before validation-only search can contain held-out results in the
scenario or a history node. Such traces are protocol-polluted and are rejected
from final evaluation. `--audit-reuse-legacy` can persist a same-selection
legacy result for provenance inspection only; it never executes another
held-out call, and the collector excludes the artifact from clean reports.

## 8. Collect and compare results

Both runners automatically regenerate `<run-root>/report/` after their workers
finish. Failed or incomplete tasks remain in the report with diagnostics. A
`--resume` invocation that finds every task already successful also regenerates
the report; `--dry-run` never writes one.

To combine an FT-Agent run and a Base run into one paper-level report, run:

```bash
.venv/bin/python reproduction/ft_agent/collect_results.py \
  --matrix-run paper-main \
  --base-run paper-base \
  --output finetune_files/logs/paper-report/paper-main
```

Each run option is repeatable and also accepts an explicit run-root path. The
collector writes:

| File | Contents |
| --- | --- |
| `results.json` | Complete task records, selection provenance, aggregates, and published-value comparisons |
| `coverage.json` | Supplied, incomplete, runnable-but-unsupplied, and unpublished-job counts |
| `tasks.csv` | One row per supplied job and its final validation/test metrics |
| `metrics.csv` | Raw and derived metrics for Base, every FT-Agent loop, and the selected result |
| `aggregates.csv` | Paper-group means and standard deviations across independent runs |
| `comparisons.csv` | Observed-minus-published deltas against `paper_results.json` |

FT-Agent model selection mirrors `FTTrace.get_sota_experiment()`: the collector
starts at the current node in the numerically latest session, walks first-parent
ancestry backward, and chooses the newest accepted node. Acceptance is based on
validation-visible feedback only. Held-out test scores are never searched or
ranked. The collector first requires a successful `final_test.json` whose
signature matches the current selection. Any search-time
`baseline_benchmark_score_test` or `benchmark_test` payload marks the task
`protocol_polluted`; the collector does not parse or report those legacy test
scores. If no accepted ancestor exists, the Base checkpoint is selected
from validation and handled by the same final-test protocol. Missing, stale,
failed, or malformed artifacts are reported rather than silently substituted.

ChemCoTBench task-level columns use the formulas printed in the paper: Mol_Und
combines its specified counting, scaffold, and classification views; Mol_Edit
averages the three edit views; Mol_Opt averages six views; and Reaction FTS uses
only forward synthesis, retrosynthesis, and NEPP while Reaction accuracy uses
mechanism selection. Repeated runs use the arithmetic mean and sample standard
deviation (`n - 1`); a single observation has a null standard deviation.

Published values retain the paper's printed precision, so comparison deltas can
include publication-rounding error. The focused tables' representative
single-run FT-Agent values are not mapped to any released run ID and are
therefore omitted from canonical comparisons; the identified three-run main
aggregates remain in the reference manifest.

## 9. Run the strict main/rsLoRA terminal protocol

For the formal 13-task x 3-run experiment, use the downstream supervisor after
starting the paper-policy matrix. The supervisor does not repair or launch the
main matrix. It waits until the read-only retry audit proves all 39 tasks are
strict successes, each with exactly 2,000 training samples, a completed epoch
schedule, a matching autonomous Full-SFT/ordinary-LoRA method lock, and no
associated live process:

```bash
.venv/bin/python reproduction/ft_agent/prepare_matrix_retry.py \
  --matrix-run h20-gpt56-paper-2k-v2
```

Start exactly one supervisor. It owns a non-blocking singleton lock, so a
duplicate invocation exits with temporary-failure code 75 without overwriting
the active supervisor's state:

```bash
.venv/bin/dotenv run -- \
  .venv/bin/python -u reproduction/ft_agent/post_main_experiment_supervisor.py \
  --main-matrix h20-gpt56-paper-2k-v2 \
  --base-run h20-gpt56-base-v1 \
  --paired-matrix h20-gpt56-paper-2k-v2-rslora-paired
```

After the main gate passes, the supervisor performs these phases in order:

1. It creates one rsLoRA counterpart for every ordinary-LoRA formal workspace.
   The signed pair reuses the same data and training configuration byte for
   byte except for `use_rslora: false -> true`. Full-SFT workspaces are not
   converted and rsLoRA never enters the paper-policy main result.
2. It runs three independent validation-only selections: the complete main
   view (Base, Full SFT, or ordinary LoRA), the ordinary-LoRA comparison view
   with Base and Full SFT excluded, and the paired-rsLoRA view with the unchanged
   Base excluded.
3. It preflights all three frozen selections before committing any held-out
   work, then evaluates their selected checkpoints with the one-shot final-test
   protocol. The supervisor never passes `--allow-retest`; an identical
   checkpoint result may be reused across views instead of being evaluated
   again.
4. It collects the main and paired results, renders the Chinese tables, and
   independently rebuilds the terminal evidence audit from the persisted
   artifacts.

The default terminal output is:

```text
finetune_files/logs/paper-report/h20-gpt56-paper-2k-v2-complete/
├── MAIN_TABLE.md
├── LORA_RSLORA_COMPARISON.md
├── main-results/results.json
├── paired-results/results.json
├── paired-results/paired_training_audit.json
└── audit/
    ├── EXPERIMENT_AUDIT.md
    └── evidence_manifest.json
```

Supervisor state and phase logs are stored under
`finetune_files/logs/orchestration/`. A terminal result is complete only when
the state reports `stage=complete`, both rendered tables exist, and the final
audit succeeds; the presence of a partial report directory is not sufficient.

## Hardware fidelity and interpretation

The paper used one NVIDIA B200 with 178 GB per experiment. This machine has
eight NVIDIA H20 GPUs, each reporting 97,871 MiB and compute capability 9.0.
An H20 is not a hardware-equivalent substitute for a B200. The scheduler
therefore preserves the paper's training-method and optimization constraints,
not its physical one-GPU topology: LoRA remains one task on one H20, while Full
SFT uses multi-H20 ZeRO-3 as described above. ZeRO-3 shards model, gradient,
and optimizer state without freezing parameters, so this is still
full-parameter SFT. Preserving the effective global batch avoids silently
changing the number of optimizer updates when the world size changes.

The paper permits Full SFT or LoRA and specifically notes that B200 memory
allows full-parameter updates. Rewriting every Full-SFT choice to LoRA or
forcing rsLoRA across all tasks changes the method search space and is not an
equivalent reproduction. The controlled `full`, `lora`, and `rslora` policies
make that effect measurable while holding the rest of each generated
configuration fixed. Multi-H20 Full SFT should be reported as an H20
protocol-equivalent run rather than as a hardware-identical reproduction.

Results can also differ because planner/provider implementations behind model
names may change, stochastic generation is not bit-reproducible, and H20/B200
training throughput changes how much exploration fits inside the fixed
wall-clock budget. Preserve the generated matrix manifest, environment locks,
asset lock, status files, and traces when reporting results.
