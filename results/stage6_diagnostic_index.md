# Stage-6 V2–V7 诊断运行索引

> 状态冻结于 2026-09-19。下列目录全部是中途停止的 WiSig 单 fold 开发诊断，不是完整五折 LCO、跨数据集验证或正式 Unknown 结果；不得进入论文主表、均值/方差汇总或方法选择。

| 目录 | 可解释范围 | 完整性判定 |
|---|---|---|
| `stage6_screen_v2/` | partition 2026、fold0 候选诊断 | 不完整；无 selection、aggregate 或 COMPLETE |
| `stage6_screen_v3/` | partition 2026、fold0 候选诊断 | 不完整；无 selection、aggregate 或 COMPLETE |
| `stage6_screen_v4/` | partition 2026、fold0 候选诊断 | 不完整；无 selection、aggregate 或 COMPLETE |
| `stage6_screen_v5/` | partition 2026、fold0 部分候选诊断 | 不完整；无 selection、aggregate 或 COMPLETE |
| `stage6_screen_v6/` | partition 2026、fold0 候选诊断 | 不完整；fold1 仅有部分残留产物，无 selection、aggregate 或 COMPLETE |
| `stage6_screen_v7/` | partition 2026、fold0 部分候选诊断 | 不完整；无 selection、aggregate 或 COMPLETE |

统一解释规则：

- 目录及原始产物原样保留，供失败分析和复现审计；不删除、不改名、不补写完成标记。
- 空目录、部分 fold1 文件和缺少候选的 fold0 均不计作已完成 fold。
- V2–V7 只支持“Stage-6 的通用 latent message / query-cost 路线未形成稳定通信净增益”这一诊断，不支持任何正式性能结论。
- Stage-7 必须重新从严格隔离的 nested LCO 与公平 B2 开始；不得继承这些运行对真实 Unknown 的观察来选模型、阈值或停止轮次。

Stage-7 的设计与闸门见 [`../docs/stage7_conditional_consultation.md`](../docs/stage7_conditional_consultation.md)。
