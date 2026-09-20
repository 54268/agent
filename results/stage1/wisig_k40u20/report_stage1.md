# Stage-1 三证据互补性诊断 — stage1_wisig_k40u20_1day_1rx

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.8655 | 0.8363 | 0.4399 | 0.8332 | 0.9117 | 0.0867 | 0.6623 | 0.1582 |
| Prototype | 0.8998 | 0.8835 | 0.4872 | 0.8551 | 0.9133 | 0.1910 | 0.7046 | 0.3158 |
| Reconstruction | 0.8565 | 0.8863 | 0.6505 | 0.7970 | — | 0.3704 | 0.7188 | 0.5243 |
| 固定融合·均值 | 0.9136 | 0.9162 | 0.4323 | 0.8573 | 0.9117 | 0.0965 | 0.6569 | 0.1744 |
| 固定融合·max/OR | 0.8949 | 0.8972 | 0.5003 | 0.8509 | 0.9117 | 0.5171 | 0.7661 | 0.6548 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.087，Prototype=0.191，Reconstruction=0.370
- **三通道并集召回=0.517**（最强单通道=0.370）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.039，Prototype=0.091，Reconstruction=0.269
- 两两 Jaccard：identity__prototype=0.119，identity__reconstruction=0.073，prototype__reconstruction=0.174

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 4384 个：由Prototype抓住=0.177，由Reconstruction抓住=0.372；**任一其他通道救援=0.471**
- 被 Prototype 漏掉 3883 个：由Identity抓住=0.071，由Reconstruction抓住=0.355；**任一其他通道救援=0.403**
- 被 Reconstruction 漏掉 3022 个：由Identity抓住=0.088，由Prototype抓住=0.171；**任一其他通道救援=0.233**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.806
- 嵌入 CKA：I-P=0.363，I-R=0.510，P-R=0.271
- Known 上 Identity/Prototype 预测分歧率=0.062

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.9136，max/OR=0.8949；最强单证据=0.8998（最优融合规则：fused_mean）
- 存在固定融合超过最强单证据：是（最优增益 +0.0138）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.471
- 并集召回 0.517 相对最强单通道 0.370 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`