# Stage-5B～H：通信、仲裁与拒识工作点筛选

## 协议

本轮始终使用 WiSig 同日同接收机 `40 Known / 20 Unknown`。所有结构、伪未知、通信门、证据融合、
仲裁规则和拒识工作点均由 40 个 Known 的五折 Leave-Class-Out 代理任务选择；当前 20 个真实 Unknown
只在配置冻结且每个训练 seed 完成后评价。正式结果使用 seeds `42/43/44`，代码启用 CUDA 确定性算法。

## 逐步筛选结论

1. **定向审计通信（Stage-5B/D）**：Identity/Geometry 不再相互改写类别状态，只向 Boundary Auditor
   发送按需消息。可选静默门能提高 Unknown Recall，但直接通信的种子方差大，且 OSCR 低于同模型关闭消息。
2. **非负残差审计（Stage-5E）**：消息只能增加未知风险，不能修改类别预测或降低本地未知风险。
   该候选在伪未知上近满分，但五折 LCO 没有选中，说明约束方向正确但消息内容仍未覆盖真实留类边界。
3. **Decision Arbitrator Agent（Stage-5F）**：本地审计与消息审计分别做 Known-only 校准，再在
   `communication/mean/max/noisy-or` 中由 LCO 选择。五折选择均值仲裁；三种子 H-score 从固定 0.95
   工作点下的 `0.726` 提高到 `0.774`，消息相对关闭消息的 H 增益三个 seed 均为正，但 OSCR 仍略降。
4. **LCO Risk Budget（Stage-5G）**：在 Known Accuracy≥85% 约束下，联合搜索
   Known 接受率 `{0.85,0.88,0.90,0.92,0.95}`。五折冻结 `mean4 + max 仲裁 + 0.88`，不使用真实
   Unknown 选阈值。
5. **Class-Conditional Threshold Agent（Stage-5H）**：按预测 Known 类估计分位数阈值，并按
   `n/(n+n_prior)` 向全局分位数收缩。比较 global 与 prior `0/10/25/50/100` 后，LCO 仍选择 global：
   `tau=.88, prior=100` 的 LCO H=`0.8615`，低于 global 的 `0.8649`；完全分类别阈值还会把 Known
   Accuracy 压到约 80%。因此保留实现和消融，但当前正式模型不启用分类别阈值。

## 当前三种子正式结果（Stage-5H 冻结选择）

冻结配置：`mixed_eta1.5 + audit_optional_cf(budget=.1, gate=.65) + mean4 + max arbitration + tau=.88`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Prototype | 0.938±0.011 | 0.893±0.002 | 0.880±0.004 | 0.815±0.045 | 0.845±0.023 |
| 等容量无通信 | 0.903±0.022 | 0.872±0.006 | 0.868±0.013 | 0.812±0.036 | 0.839±0.015 |
| 协作模型关闭消息 | 0.939±0.006 | **0.894±0.005** | **0.869±0.015** | 0.882±0.011 | 0.875±0.010 |
| 完整协作 | **0.948±0.005** | 0.892±0.007 | 0.862±0.013 | **0.892±0.010** | **0.877±0.012** |

按 Tx 分组 bootstrap 的完整协作 95% CI：AUROC `[0.941,0.955]`、OSCR `[0.879,0.906]`、
Known Accuracy `[0.844,0.880]`、Unknown Recall `[0.885,0.898]`、H-score `[0.867,0.886]`。

当前已经通过 Known Accuracy≥85% 与 H-score≥87.5%；Unknown Recall 距 90% 约 0.8 个百分点，
AUROC 距 95% 约 0.2 个百分点，OSCR 距 90% 约 0.8 个百分点。通信相对独立无通信模型有明显增益，
但相对同模型关闭消息仅 `ΔH≈+0.0011` 且 `ΔOSCR≈-0.0020`，尚未通过消息净增益闸门。

硬门平均只启用 `Identity → Boundary` 约 `0.207` 条/样本，其余边为零。下一轮不应继续放宽阈值，
而应提升 Geometry 消息的可用性和分数排序，使 AUROC/OSCR 与召回同时上升。

## 入口与产物

- 最新配置：`configs/experiments/stage5h_class_adaptive_threshold_wisig.json`
- 统一入口：`scripts/experiments/run_stage5.py`
- 聚合报告：`results/stage5h/class_adaptive_threshold_wisig_k40u20/aggregate_stage5.md`
- LCO 全候选：`results/stage5h/class_adaptive_threshold_wisig_k40u20/lco/lco_selection.json`
- 每 seed 的 checkpoint、逐样本分数、预测、通信门与实际阈值：对应 `multiseed/seed*/`

