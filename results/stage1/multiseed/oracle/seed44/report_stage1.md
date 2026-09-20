# Stage-1 三证据互补性诊断 — stage1_oracle_k10u6_demod_seed44

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.6545 | 0.8764 | 0.9909 | 0.6537 | 0.9941 | 0.3362 | 0.7239 | 0.4970 |
| Prototype | 0.8992 | 0.9636 | 0.5655 | 0.8983 | 0.9964 | 0.6411 | 0.7885 | 0.7684 |
| Reconstruction | 0.8037 | 0.9300 | 0.8919 | 0.7997 | — | 0.5099 | 0.7755 | 0.6661 |
| 固定融合·均值 | 0.8322 | 0.9383 | 0.6594 | 0.8296 | 0.9941 | 0.3388 | 0.7422 | 0.5044 |
| 固定融合·max/OR | 0.8916 | 0.9587 | 0.6099 | 0.8900 | 0.9941 | 0.7698 | 0.8179 | 0.8319 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.336，Prototype=0.641，Reconstruction=0.510
- **三通道并集召回=0.770**（最强单通道=0.641）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.014，Prototype=0.152，Reconstruction=0.106
- 两两 Jaccard：identity__prototype=0.473，identity__reconstruction=0.370，prototype__reconstruction=0.523

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 15932 个：由Prototype抓住=0.493，由Reconstruction抓住=0.424；**任一其他通道救援=0.653**
- 被 Prototype 漏掉 8613 个：由Identity抓住=0.062，由Reconstruction抓住=0.319；**任一其他通道救援=0.358**
- 被 Reconstruction 漏掉 11762 个：由Identity抓住=0.220，由Prototype抓住=0.502；**任一其他通道救援=0.530**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.663
- 嵌入 CKA：I-P=0.541，I-R=0.445，P-R=0.219
- Known 上 Identity/Prototype 预测分歧率=0.007

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.8322，max/OR=0.8916；最强单证据=0.8992（最优融合规则：fused_max）
- 存在固定融合超过最强单证据：否（最优增益 -0.0076）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.653
- 并集召回 0.770 相对最强单通道 0.641 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`