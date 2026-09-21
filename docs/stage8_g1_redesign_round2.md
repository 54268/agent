# Stage-8 下一轮：A/B G1 重构与 information-isolation 审计

> 状态：2026-09-20。结论为 **STOP-AND-REDESIGN**。本轮未使用 formal Unknown，未解锁 Memory、Auditor、Router 或 forced communication。

## 1. 本轮完成内容

按 `Temp/Codex_Stage8_Next_Round_G1_Redesign_Plan.md` 完成了六项限定工作：

1. ORACLE `16/32/64` 样本每类 small-set overfit；
2. 删除 class-id embedding，改为只由 inner support Known 构造的 normalized prototype-conditioned verifier；
3. 加入 class reindex invariance；
4. 将 rescue 拆为 identity、proxy-Unknown rejection、candidate-pair，并分别预测 T→S、S→T、both-fail、no-query；
5. 先重跑原 observation v1，确认仍单向后才实现 lossy local temporal tokens 与 coarse spectral statistics 的 v2；
6. 在完全相同的方法字段下重跑 WiSig/ORACLE 四 inner folds 和 G1/G1.5。

Stage-7 未修改。所有 prototype、训练、校准、probe 和阈值均只使用 inner Known 或 inner held-out proxy-Unknown；formal Unknown 样本计数始终为零。

## 2. Local learnability

| Profile | 每类样本 | Temporal train acc | Spectral train acc | T/S pair AUC | 结果 |
|---|---:|---:|---:|---:|---|
| v1 | 16 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | PASS |
| v1 | 32 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | PASS |
| v1 | 64 | 1.0000 | 0.9609 | 1.0000 / 1.0000 | PASS |
| v2 | 16 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | PASS |
| v2 | 32 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | PASS |
| v2 | 64 | 1.0000 | 1.0000 | 1.0000 / 1.0000 | PASS |

v2 首次稳定化前的尝试也保留在 `local_learnability_initial_report.json`：A 在 64/class 的 Temporal 为 `0.8307`，B 在 16/class 为 `0.7708`。只做了一次预先声明的优化稳定化：overfit epoch `30→45`、classifier-only warm-up `15→25`、学习率 `0.003→0.0015`；随后 Candidate A 全部通过，停止容量扫描。完整 loss、positive/negative verifier loss、pair-margin loss 与 pair-accuracy 曲线均保存在两份 `local_learnability_report.json`。

这证明 v1/v2 均有表达能力；ORACLE 后续失败不能解释为“网络连训练集都学不会”。

## 3. v1/v2 四折主要结果

完整逐折表位于 `results/stage8_g1_redesign_v2/stage8_g1_redesign_summary.md`。关键 gate 汇总如下：

| Profile / Dataset | 最低 T/S Known Acc | 平均 T/S pair AUC | 平均 T→S / S→T identity rescue | G1 | G1.5 |
|---|---:|---:|---:|---|---|
| v1 / WiSig | 0.9198 / 0.8877 | 0.9908 / 0.9915 | 2.90% / 0.65% | FAIL | PASS |
| v1 / ORACLE | 0.5742 / 0.1536 | 0.9154 / 0.5047 | 59.90% / 4.88% | FAIL | FAIL |
| v2 / WiSig | 0.8750 / 0.9158 | 0.9874 / 0.9956 | 0.72% / 4.35% | FAIL | PASS |
| v2 / ORACLE | 0.3177 / 0.1602 | 0.7570 / 0.4927 | 28.65% / 10.97% | FAIL | FAIL |

WiSig-v1 两个 Agent 都很强，但 rescue 很少，属于重复能力。v2 没有创造稳定双向互补，只把略强一方从 Temporal 换成 Spectral；T→S identity rescue 下降到 `0.72%`，S→T 仍只有 `4.35%`，未超过 5%。

ORACLE-v1 的 Temporal 可泛化，Spectral 的训练准确率为 `0.91–0.94`，held-out 却只有 `0.15–0.19`，pair AUC 约 `0.50`。v2 更严重：两 Agent 训练准确率均为 `1.00`，held-out Temporal 仅 `0.32–0.38`，Spectral 仍为 `0.16–0.18`。因此 v2 的高 rescue 是两个弱模型的不同错误，不是可靠专长。

## 4. Directional rescue predictability

| Profile / Dataset | mean AUROC(T→S) | mean AUROC(S→T) | >0.5 folds | 结论 |
|---|---:|---:|---:|---|
| v1 / WiSig | 0.9528 | 0.9301 | 4/4, 4/4 | 可预测，但正例极少 |
| v1 / ORACLE | 0.6341 | 0.6379 | 4/4, 4/4 | 两方向均低于 0.65 |
| v2 / WiSig | 0.9586 | 0.9435 | 4/4, 4/4 | 可预测，但 T→S 正例仅 9–22/fold |
| v2 / ORACLE | 0.6103 | 0.5320 | 4/4, 4/4 | 两方向均低于 0.65 |

逐折 PR-AUC、support count 和 positive rate 已写入 JSON/Markdown 汇总。WiSig 的 AUROC 不能脱离低 positive support 解读；ORACLE 的方向预测器没有通过 gate。

## 5. No class-ID shortcut

- Temporal/Spectral 都不再含 `nn.Embedding(class_id)` 或任何 class-index parameter table；
- candidate descriptor 来自当前 inner support Known 的 fold-local normalized prototype；
- prototype 每个 inner fold 独立重算，held-out class 和 formal Unknown 不进入；
- v1/v2、WiSig/ORACLE 共 16 个 fold-level reindex audits 全部通过；
- 同步重排 label、classifier row 与 prototype 后，logit、top-1 和 candidate support 的最大绝对误差均为 `0.0`。

## 6. Observation leakage audit

轻量 ExtraTrees probe 只在 inner support Known 上训练，在 held-out Known 上评价。R² 越低表示越难从一方 observation 重建另一方的目标 descriptor，但本审计不设人为通过阈值。

| Dataset | Profile | T→spectral descriptor R² | S→temporal-order descriptor R² |
|---|---|---:|---:|
| WiSig | v1 | 0.3661 | 0.8406 |
| WiSig | v2 | 0.7711 | 0.8244 |
| ORACLE | v1 | -0.0231 | -0.0106 |
| ORACLE | v2 | 0.0008 | -0.0093 |

v2 在类型和结构上删除了 raw I/Q、完整长时序和完整频谱 phase sequence，但量化 probe 没有在两个方向、两个数据集上一致降低 R²；尤其 WiSig 的 T→spectral R² 反而上升。因此本轮不能宣称 information isolation 已经被经验验证，只能说完成了可审计的 lossy observation 候选并得到负结果。

## 7. 失败归因

| Profile / Dataset | fold 0–3 的一致诊断 |
|---|---|
| v1 / WiSig | local model 和 verifier 均学会；失败来自两个 Agent 能力重复、双向 identity/pair rescue 都不足；rescue 可预测，不是 selector 问题。 |
| v1 / ORACLE | Temporal 可用；Spectral 训练收敛但 held-out identity 与 verifier 近随机，是 spectral observation/task 泛化失败；S→T rescue 低于 5%，方向 AUROC 也低于 0.65。 |
| v2 / WiSig | local model 和 verifier 均学会；Spectral 略强，但只是交换主弱关系，T→S 与多数 S→T rescue 仍不足；directional prediction 通过但正例稀疏。 |
| v2 / ORACLE | small-set 与完整训练均可拟合，held-out 两边却明显下降；属于 information specialization 过度且仍未修好 spectral view。pair/rescue 数字不能按互补性解释，G1.5 也失败。 |

所以失败不是 prototype verifier 失效或 class-id shortcut，也不是训练容量不足。当前核心问题是：没有一个跨 WiSig/ORACLE 共同成立的信息边界——v1 在 WiSig 重复、在 ORACLE 单向；v2 在 WiSig交换主弱关系、在 ORACLE过度删除可泛化信息。

## 8. Gate 决策

| Profile | WiSig | ORACLE | 总决策 |
|---|---|---|---|
| v1 | G1 FAIL / G1.5 PASS | G1 FAIL / G1.5 FAIL | STOP-AND-REDESIGN |
| v2 | G1 FAIL / G1.5 PASS | G1 FAIL / G1.5 FAIL | STOP-AND-REDESIGN |

本轮最终决定：**不解锁 Memory Agent、Auditor、Router、VOI、forced communication、formal Unknown 或多种子正式实验。** v2 不替换 v1；下一轮若继续，应重新设计能跨数据集保留稳定专长的 private task/observation，而不是继续调 gate、损失或 Router。

## 9. 复现入口

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_probe_oracle_k10u6.json --phase learnability
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_probe_oracle_k10u6.json --phase g1
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_probe_wisig_k40u20.json --phase g1
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_isolated_v2_oracle_k10u6.json --phase learnability
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_isolated_v2_oracle_k10u6.json --phase g1
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_isolated_v2_wisig_k40u20.json --phase g1
conda run --no-capture-output -n pytorch python scripts/experiments/summarize_stage8_g1_redesign.py
```

详细结果：

- `results/stage8_g1_redesign_v2/stage8_g1_redesign_summary.md`
- `results/stage8_g1_redesign_v2/stage8_g1_redesign_summary.json`
- `results/stage8_g1_redesign/{wisig_k40u20,oracle_k10u6}/g1_probe_report.json`
- `results/stage8_g1_redesign_v2/{wisig_k40u20,oracle_k10u6}/g1_probe_report.json`
