# 普通 LoRA 与 rsLoRA 严格配对实验

覆盖普通 LoRA 任务 39 个、严格配对训练 workspace 43 个；每个 workspace 均训练 2,000 样本。
除 `use_rslora: false → true` 外，训练 YAML 与输入文件保持一致。checkpoint 由 validation-only 规则独立选择，held-out test 不参与选模。

方向归一化改进大于 0 表示 rsLoRA 更好；对越低越好的指标，其符号已反转。

## Held-out test 汇总

| Benchmark | 指标 | n | 普通 LoRA 均值 | rsLoRA 均值 | 方向归一化改进 | 胜/平/负 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FinanceIQ_gen | accuracy | 3 | 64.01 | 61.35 | -2.67 | 0/0/3 |
| aime25 | accuracy | 3 | 0.00 | 2.22 | +2.22 | 1/2/0 |
| chemcotbench_mol_edit | accuracy | 3 | 52.59 | 60.00 | +7.41 | 2/0/1 |
| chemcotbench_mol_opt | success_rate | 3 | 31.67 | 21.33 | -10.33 | 1/0/2 |
| chemcotbench_mol_opt | valid_smiles_rate | 3 | 78.56 | 56.78 | -21.78 | 0/1/2 |
| chemcotbench_mol_und | accuracy | 3 | 60.11 | 65.89 | +5.78 | 3/0/0 |
| chemcotbench_mol_und | mae | 3 | 0.417 | 0.680 | -0.263 | 1/0/2 |
| chemcotbench_mol_und | tanimoto_similarity | 3 | 0.220 | 0.307 | +0.087 | 3/0/0 |
| chemcotbench_reaction | accuracy | 3 | 30.67 | 38.67 | +8.00 | 2/0/1 |
| chemcotbench_reaction | fingerprint_similarity | 3 | 11.04 | 9.71 | -1.33 | 1/0/2 |
| panorama_noc4pc | macro_f1 | 3 | 27.31 | 37.92 | +10.60 | 3/0/0 |
| panorama_par4pc | macro_f1 | 3 | 73.18 | 72.52 | -0.66 | 1/0/2 |
| panorama_pi4pc | gold_hit_rate | 3 | 55.33 | 54.00 | -1.33 | 1/1/1 |
| tablebench_data_analysis | accuracy | 3 | 28.91 | 29.79 | +0.88 | 3/0/0 |
| tablebench_fact_checking | accuracy | 3 | 76.39 | 61.11 | -15.28 | 0/0/3 |
| tablebench_numerical_reasoning | accuracy | 3 | 35.67 | 20.00 | -15.67 | 0/0/3 |
| tablebench_visualization | pass_at_1 | 3 | 9.33 | 9.33 | +0.00 | 1/1/1 |

## Held-out test 逐运行结果

| Task | Run | 指标 | 普通 LoRA | rsLoRA | rsLoRA − LoRA | 方向归一化改进 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| FinanceIQ_gen | 1 | accuracy | 65.84 | 60.42 | -5.42 | -5.42 |
| FinanceIQ_gen | 2 | accuracy | 62.53 | 61.69 | -0.84 | -0.84 |
| FinanceIQ_gen | 3 | accuracy | 63.67 | 61.93 | -1.75 | -1.75 |
| aime25 | 1 | accuracy | 0.00 | 0.00 | +0.00 | +0.00 |
| aime25 | 2 | accuracy | 0.00 | 6.67 | +6.67 | +6.67 |
| aime25 | 3 | accuracy | 0.00 | 0.00 | +0.00 | +0.00 |
| chemcotbench_mol_edit | 1 | accuracy | 51.11 | 61.11 | +10.00 | +10.00 |
| chemcotbench_mol_edit | 2 | accuracy | 60.00 | 54.44 | -5.56 | -5.56 |
| chemcotbench_mol_edit | 3 | accuracy | 46.67 | 64.44 | +17.78 | +17.78 |
| chemcotbench_mol_opt | 1 | success_rate | 31.33 | 15.33 | -16.00 | -16.00 |
| chemcotbench_mol_opt | 1 | valid_smiles_rate | 80.33 | 51.00 | -29.33 | -29.33 |
| chemcotbench_mol_opt | 2 | success_rate | 32.00 | 33.00 | +1.00 | +1.00 |
| chemcotbench_mol_opt | 2 | valid_smiles_rate | 79.67 | 79.67 | +0.00 | +0.00 |
| chemcotbench_mol_opt | 3 | success_rate | 31.67 | 15.67 | -16.00 | -16.00 |
| chemcotbench_mol_opt | 3 | valid_smiles_rate | 75.67 | 39.67 | -36.00 | -36.00 |
| chemcotbench_mol_und | 1 | accuracy | 49.34 | 56.00 | +6.66 | +6.66 |
| chemcotbench_mol_und | 1 | mae | 0.810 | 1.60 | +0.790 | -0.790 |
| chemcotbench_mol_und | 1 | tanimoto_similarity | 0.260 | 0.400 | +0.140 | +0.140 |
| chemcotbench_mol_und | 2 | accuracy | 63.00 | 68.34 | +5.34 | +5.34 |
| chemcotbench_mol_und | 2 | mae | 0.310 | 0.280 | -0.030 | +0.030 |
| chemcotbench_mol_und | 2 | tanimoto_similarity | 0.170 | 0.260 | +0.090 | +0.090 |
| chemcotbench_mol_und | 3 | accuracy | 68.00 | 73.34 | +5.34 | +5.34 |
| chemcotbench_mol_und | 3 | mae | 0.130 | 0.160 | +0.030 | -0.030 |
| chemcotbench_mol_und | 3 | tanimoto_similarity | 0.230 | 0.260 | +0.030 | +0.030 |
| chemcotbench_reaction | 1 | accuracy | 42.00 | 44.00 | +2.00 | +2.00 |
| chemcotbench_reaction | 1 | fingerprint_similarity | 6.85 | 5.44 | -1.41 | -1.41 |
| chemcotbench_reaction | 2 | accuracy | 4.00 | 2.00 | -2.00 | -2.00 |
| chemcotbench_reaction | 2 | fingerprint_similarity | 18.87 | 23.67 | +4.80 | +4.80 |
| chemcotbench_reaction | 3 | accuracy | 46.00 | 70.00 | +24.00 | +24.00 |
| chemcotbench_reaction | 3 | fingerprint_similarity | 7.38 | 0.01 | -7.37 | -7.37 |
| panorama_noc4pc | 1 | macro_f1 | 27.85 | 29.71 | +1.86 | +1.86 |
| panorama_noc4pc | 2 | macro_f1 | 30.15 | 45.71 | +15.56 | +15.56 |
| panorama_noc4pc | 3 | macro_f1 | 23.94 | 38.33 | +14.39 | +14.39 |
| panorama_par4pc | 1 | macro_f1 | 72.34 | 72.99 | +0.65 | +0.65 |
| panorama_par4pc | 2 | macro_f1 | 73.45 | 71.53 | -1.92 | -1.92 |
| panorama_par4pc | 3 | macro_f1 | 73.76 | 73.05 | -0.71 | -0.71 |
| panorama_pi4pc | 1 | gold_hit_rate | 49.00 | 51.00 | +2.00 | +2.00 |
| panorama_pi4pc | 2 | gold_hit_rate | 58.00 | 52.00 | -6.00 | -6.00 |
| panorama_pi4pc | 3 | gold_hit_rate | 59.00 | 59.00 | +0.00 | +0.00 |
| tablebench_data_analysis | 1 | accuracy | 28.89 | 30.78 | +1.90 | +1.90 |
| tablebench_data_analysis | 2 | accuracy | 28.02 | 28.28 | +0.26 | +0.26 |
| tablebench_data_analysis | 3 | accuracy | 29.82 | 30.31 | +0.49 | +0.49 |
| tablebench_fact_checking | 1 | accuracy | 70.83 | 62.50 | -8.33 | -8.33 |
| tablebench_fact_checking | 2 | accuracy | 79.17 | 68.75 | -10.42 | -10.42 |
| tablebench_fact_checking | 3 | accuracy | 79.17 | 52.08 | -27.09 | -27.09 |
| tablebench_numerical_reasoning | 1 | accuracy | 36.00 | 24.00 | -12.00 | -12.00 |
| tablebench_numerical_reasoning | 2 | accuracy | 41.00 | 28.00 | -13.00 | -13.00 |
| tablebench_numerical_reasoning | 3 | accuracy | 30.00 | 8.00 | -22.00 | -22.00 |
| tablebench_visualization | 1 | pass_at_1 | 16.00 | 8.00 | -8.00 | -8.00 |
| tablebench_visualization | 2 | pass_at_1 | 0.00 | 0.00 | +0.00 | +0.00 |
| tablebench_visualization | 3 | pass_at_1 | 12.00 | 20.00 | +8.00 | +8.00 |

## Validation 汇总

| Benchmark | 指标 | n | 普通 LoRA 均值 | rsLoRA 均值 | 方向归一化改进 | 胜/平/负 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FinanceIQ_gen | accuracy | 3 | 65.71 | 62.08 | -3.63 | 0/0/3 |
| aime25 | accuracy | 3 | 2.22 | 2.22 | +0.00 | 0/3/0 |
| chemcotbench_mol_edit | accuracy | 3 | 50.37 | 63.70 | +13.33 | 3/0/0 |
| chemcotbench_mol_opt | success_rate | 3 | 32.78 | 22.78 | -10.00 | 0/0/3 |
| chemcotbench_mol_opt | valid_smiles_rate | 3 | 78.89 | 56.22 | -22.67 | 1/0/2 |
| chemcotbench_mol_und | accuracy | 3 | 58.00 | 64.56 | +6.56 | 3/0/0 |
| chemcotbench_mol_und | mae | 3 | 0.383 | 0.590 | -0.207 | 0/0/3 |
| chemcotbench_mol_und | tanimoto_similarity | 3 | 0.207 | 0.273 | +0.067 | 3/0/0 |
| chemcotbench_reaction | accuracy | 3 | 31.33 | 45.33 | +14.00 | 3/0/0 |
| chemcotbench_reaction | fingerprint_similarity | 3 | 10.07 | 7.72 | -2.35 | 1/0/2 |
| panorama_noc4pc | macro_f1 | 3 | 21.73 | 37.67 | +15.93 | 3/0/0 |
| panorama_par4pc | macro_f1 | 3 | 73.17 | 74.96 | +1.79 | 3/0/0 |
| panorama_pi4pc | gold_hit_rate | 3 | 54.00 | 53.67 | -0.33 | 1/1/1 |
| tablebench_data_analysis | accuracy | 3 | 31.61 | 35.18 | +3.57 | 3/0/0 |
| tablebench_fact_checking | accuracy | 3 | 76.39 | 71.53 | -4.86 | 1/0/2 |
| tablebench_numerical_reasoning | accuracy | 3 | 39.67 | 29.67 | -10.00 | 0/0/3 |
| tablebench_visualization | pass_at_1 | 3 | 10.67 | 6.67 | -4.00 | 0/1/2 |

## 协议审计

- 配对训练任务：39/39。
- 配对 workspace：43；每项样本数：2000。
- 所有比较行均同时具备普通 LoRA 与 rsLoRA 的 validation/test 指标，且配对签名通过重算。
