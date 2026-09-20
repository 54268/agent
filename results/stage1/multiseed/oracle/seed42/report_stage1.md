# Stage-1 三证据互补性诊断 — stage1_oracle_k10u6_demod_seed42

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.8390 | 0.9380 | 0.6834 | 0.8370 | 0.9911 | 0.4768 | 0.7684 | 0.6362 |
| Prototype | 0.9386 | 0.9761 | 0.3204 | 0.9359 | 0.9925 | 0.6778 | 0.7952 | 0.7945 |
| Reconstruction | 0.6724 | 0.8629 | 0.9440 | 0.6687 | — | 0.2163 | 0.7013 | 0.3529 |
| 固定融合·均值 | 0.8849 | 0.9529 | 0.4386 | 0.8814 | 0.9911 | 0.2151 | 0.7099 | 0.3531 |
| 固定融合·max/OR | 0.9353 | 0.9707 | 0.2541 | 0.9315 | 0.9911 | 0.7947 | 0.8294 | 0.8465 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.477，Prototype=0.678，Reconstruction=0.216
- **三通道并集召回=0.795**（最强单通道=0.678）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.066，Prototype=0.243，Reconstruction=0.033
- 两两 Jaccard：identity__prototype=0.516，identity__reconstruction=0.256，prototype__reconstruction=0.226

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 12557 个：由Prototype抓住=0.544，由Reconstruction抓住=0.143；**任一其他通道救援=0.608**
- 被 Prototype 漏掉 7732 个：由Identity抓住=0.260，由Reconstruction抓住=0.159；**任一其他通道救援=0.363**
- 被 Reconstruction 漏掉 18809 个：由Identity抓住=0.428，由Prototype抓住=0.654；**任一其他通道救援=0.738**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.597
- 嵌入 CKA：I-P=0.448，I-R=0.323，P-R=0.462
- Known 上 Identity/Prototype 预测分歧率=0.012

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.8849，max/OR=0.9353；最强单证据=0.9386（最优融合规则：fused_max）
- 存在固定融合超过最强单证据：否（最优增益 -0.0034）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.608
- 并集召回 0.795 相对最强单通道 0.678 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`