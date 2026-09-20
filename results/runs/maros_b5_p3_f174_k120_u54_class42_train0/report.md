# MAROS-SEI F174 首轮完整实验报告

## 流程

`WiSig IQ → RMS归一化 → 时间/频相/域/开放集四Agent → Router生成有向Top-k通信图 → Agent接收消息更新 → 动态融合 → 已知分类与Unknown评分 → 验证集校准 → 测试拒识 → 被拒样本无监督细分`

真实未知 Tx 只在最终测试和离线聚类指标中使用。训练期 Unknown 监督由不同已知类信号混合、相位扰动、噪声和局部遮挡生成。

## 多智能体协作

- Time Agent：从原始 IQ 学习局部时间指纹。
- Frequency/Phase Agent：从 FFT 幅度和相邻频点相位差学习频相指纹。
- Domain Agent：从幅相、相关性和相位增量统计学习 Rx/date 域状态，并为消息提供可靠性依据。
- Open-Set Agent：读取前三个 Agent 的状态，形成未知风险私有证据。
- Dynamic Router：为每个样本生成 Agent 间有向 top-k 图、消息可靠性和最终信任权重。
- Fusion/Decision Agent：使用通信后的局部 logits 和 Router 权重联合输出 known class 与 unknown score。

## 数据

- Known/Unknown：120 / 54
- 训练 seed：0；类别 seed：42
- 拒识阈值：0.107731

## 拒识结果

| 指标 | 数值 |
|---|---:|
| Known Accuracy | 0.328833 |
| Closed-set Known Accuracy | 0.343083 |
| Unknown Precision | 0.333648 |
| Unknown Recall | 0.065556 |
| Unknown F1 | 0.109581 |
| Macro-F1 | 0.280724 |
| AUROC | 0.513702 |
| AUPR-Out | 0.321714 |
| FPR95 | 0.950083 |
| OSCR（旧方案兼容端点） | 0.182904 |
| OSCR-standard | 0.182904 |

## 未知类细分结果

| 指标 | 数值 |
|---|---:|
| 自动选择细分类数 | 12 |
| 真实未知类数（仅评价） | 45 |
| NMI | 0.390613 |
| ARI | 0.103277 |
| Purity | 0.274011 |
| Hungarian Accuracy | 0.257062 |
| Unknown-cache Precision | 0.333648 |
| Unknown-cache Recall | 0.065556 |
| Coverage of total test unknown | 0.065556 |

## Router 摘要

| Agent | Known平均权重 | Unknown平均权重 |
|---|---:|---:|
| time | 0.139684 | 0.154933 |
| frequency_phase | 0.420509 | 0.386407 |
| domain | 0.097380 | 0.092502 |
| open_set | 0.342427 | 0.366159 |

这是一轮单 seed 的端到端结果，用于验证完整流程；论文结论仍需按实验矩阵补齐重复 seed、基线和消融。
