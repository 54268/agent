# Stage-11 ABC 开放集拒识：首轮实施与失败复盘

## 结论先行

本轮保留 Stage-5 Unified V4 冻结的 A/B 专家，确实观察到强 Known 身份能力与双向互补；新增 C-lite、C-full、C 私有 prototype/tail、PCA 重构和竞争边界 PUG 均已实现并完成 ORACLE/WiSig 各 5 个 LCO fold。**但 C 的开放集能力远未达计划书成功标准，不能推进 Agent D，也不能宣称 ABC 优于 Stage-5 母体。**

最明显的失败是 ORACLE：正式 Unknown 上全量 C 的 Unknown Recall 为 **2.2%**、H-score **4.3%**，远低于同一次评估的 B-alone MSP（69.8%、80.7%）。WiSig 全量 C 为 9.5%、17.1%；第一版选择器选择的 disagreement-only 消融为 13.8%、24.0%，但它不符合计划中“高置信一致也必须经 C-lite”的约束，所以**不能作为合格 ABC 策略的成功证据**。修复后代码只允许 `all_full` 或 `disagreement_low_lite` 作为正式可选路线；已暴露的正式 Unknown 没有二次测试。

## 架构与隔离

- A：冻结 Stage-5 Identity 专家，Known top-1/top-2、置信度、margin、entropy；不刻意削弱。
- B：冻结 Stage-5 Geometry 专家，动态五视图 `raw / spectral / envelope_phase / difference_iq / complex_iq`；其完整 K 类判断独立于 A。
- C-lite：C 私有 I/Q 摘要的候选类 prototype/tail 风险，供一致高置信样本筛查。
- C-full：同一 C 私有支持域证据 + PCA 重构残差 + 从 Known 竞争边界外推的 PUG 判别器；只验证当前候选，不训练 K-way 分类头，也不读取 A/B hidden state。
- A/B 的身份候选采用固定等权 logits 融合。C 看公开 Evidence Report 和自己的原始 I/Q 描述子。报告有 top-1/top-2、margin、entropy、主要 B 视图、局部异常、冲突和审查建议。
- `disagreement_only`、`disagreement_low` 是诊断消融；修正后的部署路线必须有 C-lite 或全量 C。当前 C-full 离线预计算所有样本以实现配对反事实，故“full fraction”只是**逻辑审查率**，不是实测推理延迟或 FLOPs 降低。

## 协议与可复现性

方法配置：`configs/experiments/stage11_abc_*.json`，脚本：`scripts/experiments/run_stage11_abc.py`。seed 42、分区 seed 2026；每数据集 5 个外层 LCO fold。每 fold 使用冻结 Stage-5 fold checkpoint；C 仅在 support train Known 拟合，阈值和 C-lite 升级分位数由 support validation Known 定，策略评分用 heldout validation proxy-Unknown，独立 heldout test proxy-Unknown 只做评估。Known 训练/校准/测试分别来自 train/val/test_known，样本不重合。测试样本按类最多 64，训练每类最多 128。正式模型是冻结 Stage-5 seed42 final checkpoint，正式 Known/Unknown 测试使用完整 test 分区。

第一版先写 `results/stage11_abc/frozen_selection.json`，再触碰正式 Unknown；Stage-11 代码未将正式 Unknown 用于训练、阈值或选择。但是项目以前的 Stage-5 实验已评估同一正式 Unknown，因此这不是项目历史上从未见过的全新盲测。正式结果已有保护，脚本拒绝覆盖或重跑一次性结果。

第一版的选择器错误地允许纯 disagreement 消融进入候选，导致 WiSig 选中它。这个**计划符合性缺陷**是在正式测试后发现并修正的；修正后的策略未用同一正式 Unknown 重新评价，不能把下面 WiSig 的第一版所选路线称为修正版性能。

## 五折 LCO proxy 测试（H-score 均值）

| 方法 | ORACLE | WiSig |
|---|---:|---:|
| A-alone MSP | 0.57 | 0.10 |
| B-alone MSP | 0.63 | 0.18 |
| 静态均值融合 MSP | **0.72** | 0.10 |
| C-lite only | 0.22 | 0.15 |
| ABC 全量 C-full + PUG | 0.14 | 0.19 |
| ABC 全量 C-full、无 PUG | 0.12 | 0.21 |
| disagreement-only（诊断消融） | 0.00 | **0.29** |
| disagreement + low confidence + C-lite | 0.14 | 0.19 |

上述显示 PUG 对 ORACLE 的 5 fold 为 4 次正贡献、1 次基本持平；对 WiSig 为 **5/5 负贡献**，所以“可重复跨数据集正贡献”不成立。Known 身份方面，五折平均 ORACLE A/B/Fusion 准确率为 98.44%/99.14%/99.73%，A-only/B-only 正确样本共 22/40；WiSig 为 90.83%/91.86%/92.03%，A-only/B-only 共 81/186。B 未被故意削弱，双向互补真实存在，但 C 破坏了开放集表现。

## 正式 Unknown 一次性结果（第一版，已暴露）

| 数据集和方法 | Known Acc | Unknown Recall | H-score | AUROC | FPR95 |
|---|---:|---:|---:|---:|---:|
| ORACLE A-alone MSP | 0.941 | 0.482 | 0.638 | 0.760 | 0.953 |
| ORACLE B-alone MSP | 0.955 | 0.698 | **0.807** | **0.958** | 0.165 |
| ORACLE 静态均值 MSP | 0.944 | 0.612 | 0.742 | 0.796 | 0.951 |
| ORACLE ABC 全量 C | 0.946 | **0.022** | **0.043** | 0.399 | 0.975 |
| WiSig B-alone MSP | 0.920 | 0.065 | 0.121 | 0.875 | 0.386 |
| WiSig 静态均值 MSP | 0.918 | 0.041 | 0.079 | 0.865 | 0.353 |
| WiSig ABC 全量 C | 0.885 | 0.095 | 0.171 | 0.651 | 0.944 |
| WiSig ABC 含 C-lite 分流 | 0.885 | 0.095 | 0.171 | 0.769 | 1.000 |
| WiSig disagreement-only（不合计划） | 0.911 | 0.138 | 0.240 | 0.720 | 1.000 |

全量 C 与含 C-lite 分流的 H-score 恰好相同，但 AUROC 不同；各路线逐项准确值以 JSON 为准。

历史 Stage-5 Unified V4 seed42 正式结果：ORACLE Communication H=0.890 / OpenMax H=0.923；WiSig Communication H=0.897 / OpenMax H=0.764。它们是**历史参照**，不是本轮相同阈值/相同训练资源的严格配对消融。无论如何，当前 C 与母体仍相距甚远。

## 高置信一致 Unknown 与归因

ORACLE 正式 Unknown 有 13,062 个 A/B 高置信一致样本，C 最终仅拒识其中 337 个；WiSig 有 1,714 个，第一版 disagreement-only 选路拒识 **0** 个。C-lite 会扫描这批样本，但目前 C 私有 I/Q 摘要 + PCA 的开放空间统计无法可靠区分真实 Unknown。PUG 在 WiSig 上还持续降低 proxy H-score；不能仅凭它在 ORACLE 上偶有微弱收益宣布边界机制成功。

最可能的工程原因（推断，尚未用新盲测验证）：C 的手工 80 维摘要和 PCA 主要度量幅度/频谱分布，未充分利用 Stage-5 已学到的候选条件原型/多视图几何证据；C-full 的固定加权使 prototype 风险压过更可靠的 B 身份不确定度。WiSig 还有 C-lite 升级过多、全量 C 反而比 disagreement 消融差的问题。应先在**新 proxy 协议或未见测试集**上验证 C 的候选条件报告尾部与风险方向，不能用已看过的正式 Unknown 继续调权重再报告“盲测”。

## 状态

已完成实现、五折 proxy、四类基线、四种路由、PUG/prototype/reconstruction/C-lite 消融、报告打乱与候选替换干预、正式一次性评估和代码测试。**成功门槛未通过**；下一步应先修复 C 的表征/校准以及跨数据集 PUG 稳定性，然后申请新的未触碰正式 Unknown 切分验证，不建议现在继续扩充 Agent D。

原始数据：`results/stage11_abc/{oracle,wisig}/lco_selection_and_test.json`、`formal_one_shot.json`、`frozen_selection.json`、`paired_summary.json`。
