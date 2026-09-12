# FT-Dojo 主表复现实验报告

结果生成时间：2026-09-10T11:25:29.069872+00:00。差值均为“本次复现 − 论文”。
FT-Agent 为 3 次独立运行的算术均值与样本标准差；Base-7B 为 1 次确定性评测。

## Held-out test 主表

| Benchmark | 指标 | Base-7B 复现 | Base-7B 论文 | Base 差值 | FT-Agent 复现（3 次） | FT-Agent 论文 | FT 差值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aime25 | Acc | 0.00 | 0.00 | +0.00 | 2.22 ± 3.85 | 11.11 | -8.89 |
| panorama_par4pc | F1 | 60.77 | 57.78 | +2.99 | 73.18 ± 0.75 | 67.61 | +5.57 |
| panorama_noc4pc | F1 | 14.29 | 26.20 | -11.91 | 27.81 ± 3.37 | 36.81 | -9.00 |
| panorama_pi4pc | Acc | 43.00 | 34.00 | +9.00 | 48.33 ± 9.24 | 49.00 | -0.67 |
| tablebench_data_analysis | Acc | 33.24 | 32.78 | +0.46 | 32.59 ± 1.12 | 34.71 | -2.12 |
| tablebench_fact_checking | Acc | 77.08 | 75.00 | +2.08 | 77.08 ± 0.00 | 81.25 | -4.17 |
| tablebench_numerical_reasoning | Acc | 34.00 | 32.00 | +2.00 | 36.33 ± 4.04 | 45.33 | -9.00 |
| tablebench_visualization | P@1 | 8.00 | 4.00 | +4.00 | 12.00 ± 4.00 | 29.33 | -17.33 |
| FinanceIQ_gen | Acc | 65.33 | 65.10 | +0.23 | 65.19 ± 0.12 | 67.20 | -2.01 |
| chemcotbench_mol_und | MAE | 0.580 | 0.530 | +0.050 | 0.340 ± 0.226 | 0.230 | +0.110 |
| chemcotbench_mol_und | TMS | 0.130 | 0.090 | +0.040 | 0.177 ± 0.050 | 0.360 | -0.183 |
| chemcotbench_mol_und | Acc | 62.66 | 63.67 | -1.01 | 64.89 ± 2.72 | 67.55 | -2.66 |
| chemcotbench_mol_edit | Acc | 21.11 | 22.20 | -1.09 | 51.85 ± 6.32 | 54.44 | -2.59 |
| chemcotbench_mol_opt | SR | 23.00 | 23.33 | -0.33 | 33.11 ± 3.08 | 34.44 | -1.33 |
| chemcotbench_mol_opt | VS | 66.00 | 68.67 | -2.67 | 79.89 ± 4.02 | 90.00 | -10.11 |
| chemcotbench_reaction | FTS | 14.83 | 15.59 | -0.76 | 12.41 ± 4.36 | 30.92 | -18.51 |
| chemcotbench_reaction | Acc | 32.00 | 30.00 | +2.00 | 36.00 ± 8.72 | 35.33 | +0.67 |

## Validation 对照表

| Benchmark | 指标 | Base-7B 复现 | Base-7B 论文 | Base 差值 | FT-Agent 复现（3 次） | FT-Agent 论文 | FT 差值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aime25 | Acc | 0.00 | 0.00 | +0.00 | 2.22 ± 3.85 | 13.32 | -11.10 |
| panorama_par4pc | F1 | 64.59 | 60.63 | +3.96 | 74.51 ± 0.39 | 67.50 | +7.01 |
| panorama_noc4pc | F1 | 12.15 | 27.40 | -15.25 | 23.20 ± 4.05 | 34.44 | -11.24 |
| panorama_pi4pc | Acc | 51.00 | 29.00 | +22.00 | 53.67 ± 4.62 | 38.67 | +15.00 |
| tablebench_data_analysis | Acc | 33.51 | 28.87 | +4.64 | 34.33 ± 1.42 | 33.56 | +0.77 |
| tablebench_fact_checking | Acc | 81.25 | 83.30 | -2.05 | 81.25 ± 0.00 | 82.64 | -1.39 |
| tablebench_numerical_reasoning | Acc | 41.00 | 40.00 | +1.00 | 41.67 ± 1.15 | 46.67 | -5.00 |
| tablebench_visualization | P@1 | 8.00 | 8.00 | +0.00 | 14.67 ± 6.11 | 16.00 | -1.33 |
| FinanceIQ_gen | Acc | 74.34 | 73.00 | +1.34 | 74.34 ± 0.00 | 74.04 | +0.30 |
| chemcotbench_mol_und | MAE | 0.520 | 0.520 | +0.000 | 0.310 ± 0.182 | 0.240 | +0.070 |
| chemcotbench_mol_und | TMS | 0.090 | 0.090 | +0.000 | 0.180 ± 0.090 | 0.260 | -0.080 |
| chemcotbench_mol_und | Acc | 56.00 | 56.00 | +0.00 | 58.55 ± 3.86 | 62.78 | -4.23 |
| chemcotbench_mol_edit | Acc | 30.00 | 34.40 | -4.40 | 50.74 ± 16.19 | 57.04 | -6.30 |
| chemcotbench_mol_opt | SR | 21.00 | 21.00 | +0.00 | 33.00 ± 1.76 | 36.22 | -3.22 |
| chemcotbench_mol_opt | VS | 67.67 | 66.00 | +1.67 | 77.22 ± 2.50 | 88.78 | -11.56 |
| chemcotbench_reaction | FTS | 15.26 | 14.93 | +0.33 | 12.27 ± 5.10 | 30.64 | -18.37 |
| chemcotbench_reaction | Acc | 36.00 | 36.00 | +0.00 | 38.00 ± 3.46 | 39.33 | -1.33 |

## Validation-only 选出的最佳单次运行

每个 benchmark 先在三次运行之间，对全部论文 validation 指标按方向做 min-max 归一化，再等权平均；得分最高者被选中，完全不读取 test。若联合得分相同，以较小 run index 作确定性决胜。

| Benchmark | 最佳运行 | 联合 validation 得分 | Validation 指标 | 对应的一次性 Test 指标 |
| --- | --- | ---: | --- | --- |
| aime25 | run-2 | 1.0000 | Acc=6.67 | Acc=0.00 |
| panorama_par4pc | run-1 | 1.0000 | F1=74.82 | F1=72.34 |
| panorama_noc4pc | run-2 | 1.0000 | F1=26.10 | F1=30.15 |
| panorama_pi4pc | run-3 | 1.0000 | Acc=59.00 | Acc=59.00 |
| tablebench_data_analysis | run-3 | 1.0000 | Acc=35.97 | Acc=31.29 |
| tablebench_fact_checking | run-1 | 1.0000 | Acc=81.25 | Acc=77.08 |
| tablebench_numerical_reasoning | run-2 | 1.0000 | Acc=43.00 | Acc=41.00 |
| tablebench_visualization | run-3 | 1.0000 | P@1=20.00 | P@1=12.00 |
| FinanceIQ_gen | run-1 | 1.0000 | Acc=74.34 | Acc=65.33 |
| chemcotbench_mol_und | run-3 | 0.9896 | MAE=0.210; TMS=0.270; Acc=63.00 | MAE=0.130; TMS=0.230; Acc=68.00 |
| chemcotbench_mol_edit | run-2 | 1.0000 | Acc=62.22 | Acc=50.00 |
| chemcotbench_mol_opt | run-2 | 1.0000 | SR=35.00; VS=78.67 | SR=36.67; VS=83.67 |
| chemcotbench_reaction | run-1 | 0.5000 | FTS=15.26; Acc=36.00 | FTS=15.02; Acc=30.00 |

## 协议与覆盖审计

- Base-7B：13/13；FT-Agent：39/39；全部状态为 succeeded，且任务诊断为空。
- 每个 FT benchmark 恰有 3 次独立运行；主表所有 metric/split 均与论文引用行匹配。
- 搜索与 checkpoint 选择只使用 validation；test 仅在选择签名冻结后执行一次。
- 报告渲染器拒绝缺失结果、协议诊断、非有限数值、错误样本数或 test-informed selection。
