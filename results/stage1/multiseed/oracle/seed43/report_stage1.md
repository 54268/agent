# Stage-1 三证据互补性诊断 — stage1_oracle_k10u6_demod_seed43

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.9155 | 0.9655 | 0.4220 | 0.9111 | 0.9886 | 0.6076 | 0.7680 | 0.7421 |
| Prototype | 0.9108 | 0.9650 | 0.4675 | 0.9056 | 0.9880 | 0.6783 | 0.7876 | 0.7928 |
| Reconstruction | 0.4814 | 0.7751 | 0.9915 | 0.4775 | — | 0.1032 | 0.5870 | 0.1860 |
| 固定融合·均值 | 0.8431 | 0.9291 | 0.4756 | 0.8392 | 0.9886 | 0.1505 | 0.6191 | 0.2609 |
| 固定融合·max/OR | 0.9200 | 0.9667 | 0.3500 | 0.9145 | 0.9886 | 0.8153 | 0.8146 | 0.8508 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.608，Prototype=0.678，Reconstruction=0.103
- **三通道并集召回=0.815**（最强单通道=0.678）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.109，Prototype=0.197，Reconstruction=0.006
- 两两 Jaccard：identity__prototype=0.590，identity__reconstruction=0.150，prototype__reconstruction=0.107

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 9418 个：由Prototype抓住=0.513，由Reconstruction抓住=0.026；**任一其他通道救援=0.529**
- 被 Prototype 漏掉 7721 个：由Identity抓住=0.406，由Reconstruction抓住=0.085；**任一其他通道救援=0.426**
- 被 Reconstruction 漏掉 21523 个：由Identity抓住=0.574，由Prototype抓住=0.672；**任一其他通道救援=0.794**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.694
- 嵌入 CKA：I-P=0.410，I-R=0.231，P-R=0.256
- Known 上 Identity/Prototype 预测分歧率=0.012

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.8431，max/OR=0.9200；最强单证据=0.9155（最优融合规则：fused_max）
- 存在固定融合超过最强单证据：是（最优增益 +0.0045）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.529
- 并集召回 0.815 相对最强单通道 0.678 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`