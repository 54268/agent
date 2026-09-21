# Stage-8 Agent-B v3 快速验证

> 日期：2026-09-20。最终结论：`SPECTRAL-PRIVATE-TASK-REDESIGN-REQUIRED`。

## 执行边界

本轮只修改 Agent B 的 observation。Agent A 保持 `legacy_v1` observation、相同 encoder、state/hidden 维度、loss、prototype/verifier、30 epochs 和优化配置。B3 使用 `4×64` Hann 分段、segment mean/std PSD、energy-centroid circular recenter、median/IQR normalization 和六个纯 magnitude 通道；没有 FFT phase、phase step、phase roughness或 paired complex phase。

只执行了 ORACLE inner fold 0。B3 同时低于早停线 `Known Acc<0.25` 和 `pair AUC<0.60`，因此严格按计划没有运行 ORACLE 四折、WiSig、directional predictor、Router 或任何 formal Unknown。

## 表 1：Agent B 本地能力

| Profile | Dataset | Fold | Train Acc | Eval Acc | Pair AUC | Hard Pair Acc |
|---|---|---:|---:|---:|---:|---:|
| v1 | ORACLE | 0 | 0.9375 | 0.1719 | 0.4929 | 0.1771 |
| v2 | ORACLE | 0 | 0.9997 | 0.1732 | 0.5004 | 0.1784 |
| robust-v3 | ORACLE | 0 | 0.9196 | 0.1484 | 0.4870 | 0.1836 |

B3 不仅没有达到继续门槛 `0.35/0.65`，而且没有超过 v1/v2。train/eval gap 为 `0.7712`，仍是“训练可分、测试近随机”。

## 表 2：Agent B 不可替代性

| Profile | Dataset | Fold | S→T Identity Rescue | S→T Pair Rescue | Rescue AUROC | Rescue PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| v1 | ORACLE | 0 | 0.0326 | 0.1029 | 0.7007 | 0.0603 |
| v2 | ORACLE | 0 | 0.1055 | 0.3359 | 0.5299 | 0.1115 |
| robust-v3 | ORACLE | 0 | 0.0299 | 0.1042 | 未运行 | 未运行 |

B3 的本地准确率与 pair verifier 均近随机，所以其 rescue 不能解释为有效不可替代性。跨折 directional predictor 也因此没有运行。

## 表 3：Nuisance Stability

以下 state drift 使用同一组 B3 训练权重，仅切换 observation profile，以隔离 observation 的影响；越低越稳定。

| Profile | Time-shift drift | Small-CFO drift |
|---|---:|---:|
| v1 | 0.000281 | 0.004479 |
| v3-noalign | 0.010160 | 0.021881 |
| v3-align | 0.013947 | 0.016400 |

centroid alignment 将 CFO drift 从 `0.02188` 降到 `0.01640`，说明对齐本身有局部作用；但 v3-align 仍明显差于 v1 的 `0.00448`，time-shift drift 也更大。也就是说，预期的 nuisance robustness 没有在真实 ORACLE fold-0 样本上成立。合成 stationary-multitone 的机制测试通过，但不能替代真实数据诊断。

## 表 4：Prototype Generalization

| Dataset | Fold | Train PGR | Eval PGR | Eval Prototype Margin |
|---|---:|---:|---:|---:|
| ORACLE | 0 | 1.6277 | 0.1126 | -0.0549 |

进一步分解：

| Split | Intra-class distance | Inter-class distance | Prototype/sample distance | Prototype-centroid drift |
|---|---:|---:|---:|---:|
| Train | 0.1082 | 0.1761 | 0.1082 | 0.0020 |
| Eval | 0.1409 | 0.0159 | 0.1806 | 0.1170 |

主要故障不是只有类内方差增加，而是 eval 类间距离从 `0.1761` 塌到 `0.0159`，PGR 从 `1.63` 降到 `0.11`；同时 prototype margin 变为负数且 eval centroid 明显偏离训练 prototype。即：B3 在测试分布中失去了类别结构。

## 决策

`robust_spectral_v3` 已被快速否决。删除 phase、分段平均和 centroid alignment 不能让 ORACLE 上的独立频谱 K-way identity task 泛化，继续增加频谱通道或调 Conv 容量没有证据基础。

下一轮若继续，应把 Agent B 的 private task 从：

```text
frequency view -> full emitter identity
```

改成更窄的：

```text
candidate-conditioned spectral impairment verification
```

并弱化或取消 B 的独立 K-way 分类要求。当前不解锁 Memory、Auditor、Router、VOI、forced communication、formal Unknown 或多种子实验。

## 结果文件

- `results/stage8_agent_b_v3_quick/oracle_fold0/agent_b_v3_fold0_report.json`
- `results/stage8_agent_b_v3_quick/agent_b_v3_quick_summary.json`
