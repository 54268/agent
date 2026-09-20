# Stage-5 Unified V4：同一多智能体算法的 WiSig / ORACLE 验证

> 日期：2026-09-18  
> 当前证据级别：正式协议、单训练种子 `42`。这是跨数据集可行性结果，不是最终三种子论文结论。

## 1. 本轮要回答的问题

本轮不再为 WiSig 和 ORACLE 分别设计 Agent，而是使用同一个
`universal_multiview_maros_v4_matched_lco_training` 方法族回答两个问题：

1. 固定角色、固定候选动作和固定训练协议能否同时处理两个数据集；
2. Agent 能否根据当前数据和样本，自主选择传感视角、通信边、证据组合和拒识阈值，而不是把这些选择写成数据集特例。

两份配置去除数据路径、运行名称、输出目录、缓存复用路径和比较目标后，共有 **45 个方法字段，45 个全部相同**。
ORACLE 的最终专家权重复用了先前从头训练的同结构、同 `30 epoch`、同 seed 检查点，仅避免重复执行确定性训练；
LCO 选择和协调器仍按 V4 协议重新运行。

## 2. 固定的多智能体体系

### 2.1 Identity Agent

- 输入原始时域 I/Q；
- 独立输出 Known 类 logits、置信度、Energy、MSP、margin、entropy、Identity 原型距离和压缩消息；
- 即使关闭通信也能完成本地分类和开放集证据计算。

### 2.2 Geometry Agent

Geometry 是一个能够行动的 Agent，而不是五个按数据集硬切换的分支。它在两个数据集上固定拥有相同的五个传感动作：

1. `raw`；
2. `spectral`；
3. `envelope_phase`；
4. `difference_iq`；
5. `complex_iq`。

每个传感器独立给出局部类别提案和类原型距离。Geometry 的样本级 view gate 根据各传感器状态、置信度、
竞争 margin 和 entropy 产生凸组合权重，再输出融合类别 logits、原型证据、可靠性和消息。固定传感器集合与决策公式
在两个数据集上完全一致，只有学习到的动作随数据和样本变化。

### 2.3 Boundary Explorer / Boundary Auditor

- Explorer 在训练期从 Known 边界生成伪未知候选，动作空间固定为竞争边界外推、跨 Agent 分歧外推及混合外推；
- Auditor 接收 Identity 与 Geometry 的选择性消息、分歧和竞争边界特征，只修改未知风险，不产生第三套类别 logits；
- Reconstruction 只保留为专家表征约束，不作为推理 Agent 或独立拒识票。

### 2.4 Calibration、Threshold 与 Communication Coordinator

- Calibration Agent 只用 Known validation 拟合类条件经验 CDF，使不同证据可比较；
- Class-Conditional Threshold Agent 可按预测类估计阈值，并按类别样本数向全局阈值收缩；
- Coordinator 为每条发送边产生样本级门，类别输出严格是 Identity/Geometry logits 的凸组合；
- 反事实信用通过删除消息后的损失变化监督通信门；Boundary 消息只能影响未知风险。

## 3. “同一算法、自主决策”如何落地

自主性发生在三个层级，而且都不观察正式测试 Unknown：

1. **样本级传感动作**：Geometry 的 view gate 为每个样本选择五种固定传感器的权重；
2. **样本级通信动作**：Coordinator 决定 Identity/Geometry 是否向 Boundary 发送消息；
3. **任务级策略选择**：五折 Leave-Class-Out 用训练 Known 中的留出类别选择 PUG、证据规则、仲裁和阈值策略。

### 3.1 Geometry 学到的平均动作

| 数据集 / 样本 | raw | spectral | envelope-phase | difference-IQ | complex-IQ |
|---|---:|---:|---:|---:|---:|
| ORACLE Known | 65.57% | 0.00% | 4.52% | 0.01% | 29.90% |
| ORACLE Unknown | 39.65% | 0.00% | 15.55% | 0.00% | 44.80% |
| WiSig Known | 4.65% | 8.52% | 5.54% | 3.74% | 77.55% |
| WiSig Unknown | 5.69% | 13.65% | 5.61% | 9.14% | 65.92% |

同一个 Geometry Agent 在 ORACLE 上主要依赖 raw/complex-IQ，在 WiSig 上主要依赖 complex-IQ；
对 Unknown 又会重新分配权重。这正是“Agent 面对不同环境采取不同动作”，而不是为数据集写 `if/else`。

### 3.2 LCO 冻结的任务级动作

| 决策 | ORACLE `10/6` | WiSig `40/20` |
|---|---|---|
| PUG | `competition_eta1.5` | `competition_eta2` |
| 通信 | `audit_sparse_cf` | `audit_sparse_cf` |
| 通信预算 / 硬门 | `0.25 / 0.5` | `0.25 / 0.5` |
| 拒识证据 | `idproto_boundary` | `sensor_mean` |
| 仲裁 | `communication` | `communication` |
| Known 接受率 | `0.90` | `0.88` |
| 分类别阈值收缩先验 | `25` | `global` |

候选集合和排序规则相同，但 LCO 根据各自训练 Known 的留类代理任务选择了不同动作。ORACLE 选择了分类别自适应阈值，
WiSig 则判断全局阈值更稳健；这也说明阈值 Agent 不是被强制打开的固定模块。

## 4. 单 seed 正式结果

### 4.1 WiSig：五项主性能闸门全部通过

| 方法 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| 旧 Stage-5H，三 seed 均值 | 0.948 | 0.892 | 0.862 | 0.892 | 0.877 |
| V4 等容量无通信，seed42 | 0.9396 | 0.9017 | 0.8709 | **0.9252** | 0.8972 |
| V4 完整协作，seed42 | **0.9536** | **0.9065** | **0.8739** | 0.9221 | **0.8974** |
| V4 推理关闭消息，seed42 | 0.9490 | 0.9035 | 0.8698 | 0.9237 | 0.8960 |

完整协作达到 `Unknown Recall≥0.90`、`Known Accuracy≥0.85`、`H≥0.875`、`AUROC≥0.95` 和
`OSCR≥0.90`。相对旧 Stage-5H 三 seed 均值，五项指标分别变化约
`+0.56/+1.45/+1.19/+3.01/+2.04` 个百分点，但该比较是 V4 单 seed 对旧版三 seed 均值，不能替代 V4 三 seed 复验。

相对等容量无通信，完整协作提高 AUROC `+1.40`、OSCR `+0.48`、Known Accuracy `+0.30` 个百分点，
H-score 基本持平，Unknown Recall 下降 `0.31` 个百分点。按 Tx 分组 paired bootstrap 的通信增益为：

- AUROC：`+0.0142`，95% CI `[0.0102, 0.0185]`；
- OSCR：`+0.0048`，95% CI `[0.0025, 0.0075]`；
- H-score：`+0.0002`，95% CI `[-0.0041, 0.0042]`。

因此 WiSig 已证明统一算法的性能提升与排序增益，但尚未通过预设的 `ΔH≥0.01` 通信净增益门。

### 4.2 ORACLE：协作真实有效，但尚未追平上一方案

| 方法 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| V4 等容量无通信 | 0.9099 | 0.9095 | **0.8989** | 0.6611 | 0.7619 |
| V4 完整协作 | **0.9483** | **0.9478** | 0.8950 | **0.8845** | **0.8897** |
| V4 推理关闭消息 | 0.6619 | 0.6613 | 0.9024 | 0.1774 | 0.2965 |
| V4 OpenMax 单证据基线 | 0.9672 | 0.9666 | 0.8994 | 0.9477 | 0.9229 |
| 上一方案 PCBM | **0.9814** | 0.9664 | **0.9455** | **0.9652** | 0.9552* |

`*` PCBM 的 H-score 是由其正式 `Known Accuracy` 与 `Unknown Recall` 按调和平均计算，原结果文件未单列该字段。

完整协作相对等容量无通信提高 AUROC `+3.84`、OSCR `+3.83`、Unknown Recall `+22.34`、
H-score `+12.79` 个百分点，说明 ORACLE 上通信不是装饰；关闭同一模型的消息还会使 H-score 从 `0.8897`
降到 `0.2965`。但正式结果仍低于 PCBM，尤其 Known Accuracy、Unknown Recall 与 AUROC 分别落后约
`5.05/8.06/3.30` 个百分点。当前只能得出“统一算法可迁移且存在真实协作”，不能声称已经持平上一方案。

## 5. 正式结论与诊断上限必须分开

看到 ORACLE 正式测试结果后，对已保存逐样本证据进行的诊断发现：`complexproto` 在
`known_acceptance=0.97, prior=0` 时可达到 AUROC `0.9764`、OSCR `0.9753`、Known Accuracy `0.9656`、
Unknown Recall `0.9501`、H-score `0.9578`。它说明复数原型传感器具有接近或局部超过 PCBM 的潜力。

但是该规则是在观察真实 ORACLE Unknown 后挑出的，违反“真实 Unknown 不参与选模”的正式协议，
所以只能作为下一轮设计依据，**绝不能替换 V4 正式结果或写入论文主表**。下一轮必须把这类证据的选择能力
移入统一的 LCO 代理任务，再分别在两个数据集的未触碰 Unknown 上验证。

## 6. 当前结论

1. 同一 V4 算法已在 WiSig 和 ORACLE 上完整运行，跨数据集方法鲁棒性成立；
2. Agent 的自主决策已体现在传感器权重、通信边、PUG、证据规则和阈值策略上；
3. WiSig 单 seed 已通过五项主性能门，但 H-score 通信净增益门未通过；
4. ORACLE 的通信产生很大因果增益，但最终性能仍未追平上一方案，且当前 V4 还不及自身 OpenMax 单证据基线；
5. 两个数据集都必须完成 seeds `42/43/44`，当前不得把单 seed 结果包装成最终结论。

## 7. 下一轮统一改进方向

下一轮仍保持同一算法，不增加数据集特例：

- 将各局部传感器的原型距离、类条件尾部统计与 Boundary 消息交给一个监督式 Calibration Agent；
- 用重复 LCO 或开放程度匹配 LCO，减小 ORACLE 仅 10 个 Known 时单折留类估计的方差；
- 把“是否采用分类别阈值、使用哪类证据、是否发送哪条边”都保留为 Agent 动作；
- 在配置冻结后同时运行 WiSig/ORACLE seeds `42/43/44`，再判断均值、方差和 `2/3 seeds` 通信增益门。

## 8. 复现入口

```powershell
$env:PYTHONUTF8 = "1"
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage5.py --config configs/experiments/stage5_unified_wisig_k40u20.json
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage5.py --config configs/experiments/stage5_unified_oracle_k10u6.json
```

关键产物：

- WiSig 聚合：`results/stage5_unified_v4/wisig_k40u20/aggregate_stage5.md`；
- ORACLE 聚合：`results/stage5_unified_v4/oracle_k10u6/aggregate_stage5.md`；
- ORACLE 事后诊断：`results/stage5_unified_v3/oracle_k10u6/diagnostics/evidence_rule_audit_seed42.json`；
- 正式配置：`configs/experiments/stage5_unified_wisig_k40u20.json`、`configs/experiments/stage5_unified_oracle_k10u6.json`。
