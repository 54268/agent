# Stage-1 三证据互补性诊断 — stage1_wisig_k40u20_1day_1rx_seed42

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.8826 | 0.8356 | 0.3438 | 0.8554 | 0.9109 | 0.0698 | 0.6490 | 0.1296 |
| Prototype | 0.9048 | 0.8667 | 0.2717 | 0.8676 | 0.9062 | 0.1598 | 0.6784 | 0.2716 |
| Reconstruction | 0.8638 | 0.8751 | 0.5647 | 0.8071 | — | 0.3479 | 0.7109 | 0.5017 |
| 固定融合·均值 | 0.9236 | 0.9123 | 0.2883 | 0.8730 | 0.9106 | 0.0723 | 0.6431 | 0.1339 |
| 固定融合·max/OR | 0.8958 | 0.8706 | 0.3505 | 0.8637 | 0.9106 | 0.4792 | 0.7583 | 0.6242 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.070，Prototype=0.160，Reconstruction=0.348
- **三通道并集召回=0.479**（最强单通道=0.348）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.031，Prototype=0.080，Reconstruction=0.275
- 两两 Jaccard：identity__prototype=0.123，identity__reconstruction=0.047，prototype__reconstruction=0.134

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 4465 个：由Prototype抓住=0.145，由Reconstruction抓住=0.354；**任一其他通道救援=0.440**
- 被 Prototype 漏掉 4033 个：由Identity抓住=0.053，由Reconstruction抓住=0.343；**任一其他通道救援=0.380**
- 被 Reconstruction 漏掉 3130 个：由Identity抓住=0.078，由Prototype抓住=0.153；**任一其他通道救援=0.201**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.838
- 嵌入 CKA：I-P=0.354，I-R=0.619，P-R=0.261
- Known 上 Identity/Prototype 预测分歧率=0.075

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.9236，max/OR=0.8958；最强单证据=0.9048（最优融合规则：fused_mean）
- 存在固定融合超过最强单证据：是（最优增益 +0.0188）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.440
- 并集召回 0.479 相对最强单通道 0.348 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`