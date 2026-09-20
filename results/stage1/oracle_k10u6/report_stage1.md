# Stage-1 三证据互补性诊断 — stage1_oracle_k10u6_demod

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.8233 | 0.9313 | 0.6804 | 0.8204 | 0.9906 | 0.4477 | 0.7335 | 0.6094 |
| Prototype | 0.9235 | 0.9689 | 0.3493 | 0.9192 | 0.9904 | 0.6426 | 0.7819 | 0.7675 |
| Reconstruction | 0.7500 | 0.8988 | 0.8531 | 0.7447 | — | 0.2612 | 0.6480 | 0.4097 |
| 固定融合·均值 | 0.8981 | 0.9552 | 0.3476 | 0.8937 | 0.9908 | 0.2328 | 0.6611 | 0.3761 |
| 固定融合·max/OR | 0.9158 | 0.9645 | 0.3513 | 0.9118 | 0.9908 | 0.7478 | 0.7997 | 0.8177 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.448，Prototype=0.643，Reconstruction=0.261
- **三通道并集召回=0.748**（最强单通道=0.643）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.053，Prototype=0.191，Reconstruction=0.039
- 两两 Jaccard：identity__prototype=0.538，identity__reconstruction=0.272，prototype__reconstruction=0.301

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 13254 个：由Prototype抓住=0.473，由Reconstruction抓住=0.198；**任一其他通道救援=0.543**
- 被 Prototype 漏掉 8577 个：由Identity抓住=0.186，由Reconstruction抓住=0.146；**任一其他通道救援=0.294**
- 被 Reconstruction 漏掉 17732 个：由Identity抓住=0.401，由Prototype抓住=0.587；**任一其他通道救援=0.659**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.665
- 嵌入 CKA：I-P=0.514，I-R=0.482，P-R=0.381
- Known 上 Identity/Prototype 预测分歧率=0.010

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.8981，max/OR=0.9158；最强单证据=0.9235（最优融合规则：fused_max）
- 存在固定融合超过最强单证据：否（最优增益 -0.0076）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.543
- 并集召回 0.748 相对最强单通道 0.643 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`