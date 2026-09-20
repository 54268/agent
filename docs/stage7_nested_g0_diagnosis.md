# Stage-7 nested G0 诊断：互补存在，但公共选择不可迁移

> 状态日期：2026-09-20。本文只汇总 inner/outer 代理 Unknown 诊断；正式 WiSig 20 Unknown 与 ORACLE 6 Unknown 均未进入本轮训练、选择、阈值或评价。

## 结论

Stage-7 已证明两个异构 Agent 具有真实互补性，但尚未证明可部署的协作价值：

- WiSig outer G0a 的最佳单 Agent `H=0.47429`，逐样本 Agent oracle 仍有 `ΔH=+0.13863`、`ΔOSCR=+0.02919`；
- ORACLE outer G0a 的最佳单 Agent `H=0.80864`，Agent oracle 仍有 `ΔH=+0.07573`、`ΔOSCR=+0.08216`；
- 两数据集的双向 unique rescue 均超过 5%，所以不是“两个分支完全重复”；
- 但公共摘要无法跨 inner episode 稳定预测应信任哪个 Agent。WiSig 的安全选择器在 V4、修正版 V6 和 V7 均选择 100% 回退，`switch rate=0`、`ΔH=0`；ORACLE V4 的安全选择器反而 `ΔH=-0.08519`。

因此 G0 未通过，G1 条件消息、G2 路由、G3 多划分和正式 Unknown 全部保持锁定。不得把 Agent-oracle 上界或 outer 诊断写成多智能体协作结果。

## 公平 nested 协议

每个 outer fold 的 support 类再划成四个 inner 类别折。每个 inner fold 都从头训练自己的 Waveform/Prototype Agent；B2 只读 `LocalDecision.public_summary`，通过 leave-one-inner-producer-out 拟合和评价。阈值校准 Known、meta Known、meta proxy Unknown 与训练样本隔离。若 inner OOF 失败，流程在读取 outer test 前停止。

候选之间在同一 held-out inner fold 使用完全相同的初始化 seed。第一次候选脚本曾误用 candidate-specific seed，目录已保留并标记为 `INVALID_COMPARISON.md`，不参与任何选择。

## V4 默认 B2

| 数据集 | 最佳单 Agent H | B2 H | ΔH | 正增益 inner folds | 结论 |
|---|---:|---:|---:|---:|---|
| WiSig | 0.61634 | 0.63474 | +0.01840 | 2/4 | 稳定性失败 |
| ORACLE | 0.67770 | 0.68896 | +0.01126 | 3/4 | 单数据集通过 |

ORACLE V4 因 inner OOF 通过而进行了一次 outer fold 评价：B2 `H=0.81448`、`OSCR=0.90785`，略高于最佳单 Agent `H=0.80864`。该 outer 结果没有参与候选选择，也不是正式 Unknown 结果。

## 公平 B2 动作空间候选

所有候选只使用缓存的 inner 公共决策，没有重训 Agent，也没有读取 outer test。

### WiSig

| 候选 | H | ΔH | 正增益 folds | 结论 |
|---|---:|---:|---:|---|
| anchored reference | 0.63474 | +0.01840 | 2/4 | 失败 |
| free selector（强选择监督） | 0.62901 | +0.01266 | 2/4 | 失败 |
| free task balanced | **0.64632** | **+0.02997** | **3/4** | 通过 |
| free no-regret | 0.63586 | +0.01952 | 2/4 | 失败 |

### ORACLE

| 候选 | H | ΔH | 正增益 folds | 结论 |
|---|---:|---:|---:|---|
| anchored reference | **0.68896** | **+0.01126** | **3/4** | 通过 |
| free selector | 0.58653 | -0.09117 | 2/4 | 失败 |
| free task balanced | 0.57459 | -0.10311 | 2/4 | 失败 |
| free no-regret | 0.58368 | -0.09403 | 2/4 | 失败 |

两个数据集没有共同胜者。为 WiSig 放开到近离散专家切换会严重伤害 ORACLE，故没有冻结数据集专用 V5。

## 安全选择器与 competence 消融

安全选择器默认严格复现 fit-side 最佳锚点，仅在三个 fit inner 的 producer-level OOF 证明替代 Agent 有正损失收益时才切换。它没有使用 held-out 标签选择阈值。

1. **V4 原始可靠性**：WiSig 四折均无可验证正收益，100% 回退；ORACLE 的一次 22.25% 切换把对应 fold 的 H 从 Prototype `0.75797` 降至 `0.39037`。
2. **V5：competence 与 Unknown logit 共用**：WiSig 默认 B2 `H=0.63242`、2/4；安全选择器仍 100% 回退。该目标被拒识损失主导，未形成独立能力信号。
3. **V6：独立 competence head，Known 正确性/Unknown 目标为 0**：第一次运行因辅助头改变全局 RNG、进而改变 Prototype 初始化，被及时停止并标记无效。RNG 隔离后的公平运行精确复现 V4 分类和 Unknown 指标，但安全选择器仍 100% 回退。
4. **V7：独立 competence head，目标改为完整本地决策成功概率**：Known 使用 `p(true class)×p(accept)`，代理/PUG Unknown 使用 `p(reject)`；默认 B2 仍为 `H=0.63474`、2/4，安全选择器仍 100% 回退。

V6 的 competence 确实改变了公开可靠性，WiSig meta Known 上 Prototype 的正确性 Brier score 从 `0.06630` 改善到 `0.02736`；但“Waveform 是否比 Prototype 的完整开放集损失更低”仍不可跨 inner 预测。问题不再是可靠性头没有训练，而是当前救援模式对类别集合/episode 不稳定。

## 止损与下一步约束

以下工作暂停：B2 loss/阈值继续扫描、G1 通信头、Router、Adaptive Explorer、CommFormer、PPO/QMIX、ORACLE V7、正式 Unknown、多种子扩展。

下一轮若继续 Stage-7，只允许改变 **Agent 的私有任务或观察，使互补关系本身可预测**。最低要求是先在 WiSig 与 ORACLE 的缓存/新 inner 审计中同时满足：

- 公共或请求后条件证据相对最佳单 Agent `ΔH≥0.01`；
- 至少 3/4 inner folds 正增益；
- 不依赖数据集专用分支；
- 不读取 outer/formal Unknown；
- 同初始化、同损失和同选择规则。

在满足这些条件前，不得声称多智能体协作成功。
