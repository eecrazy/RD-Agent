# FT-Dojo 主实验终态审计

生成时间：2026-09-10T11:28:07.212375+00:00。本页与 `evidence_manifest.json` 由原始终态文件重新计算。

> 结论：13 个 task × 3 次独立运行全部完成；搜索、严格 2k/full-epoch SFT 与 held-out 均为 39/39。
> 加上 13 项 Base-7B，对应主表输入为严格 52/52 完整。

## 覆盖与协议链

| 阶段 | 可核验结果 |
| --- | ---: |
| 恰好 2,000 条训练数据及注册证据 | 39/39 |
| 完整 epoch、trainer state、方法锁及模型权重（lora=39） | 39/39 |
| 初始 hypothesis 自主方法选择与最终方法锁一致 | 39/39 |
| 方法选择时看到 1×B200 178GB 逻辑资源 | 39/39 |
| 方法选择时看到物理 H20 | 0/39 |
| 搜索成功 | 39/39 |
| validation-only 签名选择（held_out_test_used=false） | 39/39 |
| 一次性 held-out 提交 | 39/39 |
| held-out 成功 | 39/39 |

流程为：任务工作区中的 `process_data.py` 产出并登记恰好 2,000 条训练数据；随后由 task 在论文策略下自主选择 Full SFT 或普通 LoRA。正式证据会重新计算数据条数、配置 epoch、trainer 完成步数、方法锁及权重类型，主实验不接受 rsLoRA/DoRA/QLoRA；搜索结束后只按 validation 冻结并签名 checkpoint；最终 test 的 spec 与 result 必须同时匹配该签名。

每项小型证据均记录 SHA-256；权重文件记录存在性与字节数，并由 checkpoint selection signature 绑定大小和 mtime。机器可读明细：`evidence_manifest.json`（39 项）。

## 自主方法选择审计

39 个 step-0 hypothesis 中，最终锁定的方法分布为 `lora=39`；其中 39/39 明确讨论 2,000 条小样本带来的过拟合、灾难性遗忘或能力漂移风险。

方法选择阶段统一注入论文的单卡 B200 逻辑资源包络，而物理 H20 只在之后由执行层动态租约。因此若 39 项最终均选择 LoRA，这一结果不能归因于 hypothesis 被 H20 显存硬性限制；每项原始 hypothesis 摘要、资源字典、方法锁与 SHA-256 均保存在机器清单中。

## 代表性清洗/SFT 证据

FinanceIQ run-1 所选 workspace `9afb3a9bff57034002cf1d182e0dbf47`：pre-LLM 6179 条、接受训练 2000 条、validation 100 条、保护 holdout 472 条。train/validation=0、train/holdout=0、validation/holdout=0；正式救援链已核验 6 个训练输入 hash 与 6 个源 workspace hash。

该 workspace 的训练方法为 lora（SFT/bf16）；trainer state 为 global step 189、epoch 3.0。其源处理审计 SHA-256 为 `51fb8b74ae54508bd96517f66ba30d5b2817a4d0b56334429198bb7f16e05e11`。其余 38 项逐项证据见机器清单。

## 8 GPU 并行

统一 COMMIT 请求 GPU `0,1,2,3,4,5,6,7`、`max-parallel=8`；从 START/DONE/FAIL 事件重建的实际峰值并发为 8，实际覆盖 GPU `0,1,2,3,4,5,6,7`。累计为 `117 reused, 39 evaluated`；恢复调用只允许复用已成功结果，不会再次执行同一 checkpoint。

并发日志：`finetune_files/logs/orchestration/h20-gpt56-paper-2k-v2-final-test-main.log`（SHA-256 `123d2dd5b546133ab9decdd6bd4dfe7b96af581dc754e7279f2faf28d3ef6bf4`）；提交日志：`finetune_files/logs/orchestration/h20-gpt56-paper-2k-v2-held-out-commit.log`。

所有任务保留 `timeout=12h`。搜索与 test 都记录模型适配路由 `http://127.0.0.1:8313/v1` → `gpt-5.6-sol`；因此这是模型适配复现，不是论文原 provider 的严格同模型复现。

## 严格终态

supervisor 状态为 `complete`（return code 0）；39 项 held-out 均只提交一次且全部成功。
