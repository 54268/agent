# Stage-7 冻结基线

> 冻结日期：2026-09-20。Stage-8 开始后，`src/maros_stage7/`、Stage-7 配置和既有结果只作为历史基线；本轮未修改其中任何文件。

Stage-7 的最终诊断结论是：两个本地 Agent 存在真实互补，但“谁能救谁”不能跨类别划分和数据集稳定迁移，因此 G0 未通过，G1/G2/G3 与正式 Unknown 一直保持锁定。

| 数据集 | 最佳单 Agent H | Agent-oracle ΔH | Agent-oracle ΔOSCR | 结论 |
|---|---:|---:|---:|---|
| WiSig | 0.47429 | +0.13863 | +0.02919 | 有互补，无可迁移选择器 |
| ORACLE | 0.80864 | +0.07573 | +0.08216 | 有互补，无跨数据集共同候选 |

WiSig 的 `free_task_balanced` 在 inner OOF 上达到 `H=0.64632、ΔH=+0.02997、3/4`，但同一规则在 ORACLE 为 `H=0.57459、ΔH=-0.10311`。ORACLE 的通过候选也不能稳定迁移回 WiSig。因此 Stage-7 不再继续 B2、阈值、Router、communication-loss 或 competence-head 调参。

完整证据见 [Stage-7 nested G0 诊断](stage7_nested_g0_diagnosis.md)。Stage-8 只复用其中已经审计的 provenance、nested LCO 和 formal-Unknown lock；不会把 Stage-7 Prototype super-agent 复制到新实现中。

