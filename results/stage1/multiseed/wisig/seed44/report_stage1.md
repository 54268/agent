# Stage-1 三证据互补性诊断 — stage1_wisig_k40u20_1day_1rx_seed44

阈值 τ=0.95（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。

## 1. 各证据通道开放集能力

| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |
|---|---|---|---|---|---|---|---|---|
| Identity | 0.8886 | 0.8543 | 0.4005 | 0.8524 | 0.9101 | 0.1535 | 0.6527 | 0.2627 |
| Prototype | 0.9172 | 0.8959 | 0.2946 | 0.8474 | 0.8908 | 0.2160 | 0.6773 | 0.3475 |
| Reconstruction | 0.8184 | 0.8568 | 0.7193 | 0.7663 | — | 0.3329 | 0.6904 | 0.4858 |
| 固定融合·均值 | 0.9165 | 0.9151 | 0.2973 | 0.8634 | 0.9103 | 0.1865 | 0.6540 | 0.3094 |
| 固定融合·max/OR | 0.9083 | 0.8966 | 0.2576 | 0.8673 | 0.9103 | 0.4958 | 0.7489 | 0.6380 |

## 2. Unknown 检测集合重叠与独有贡献 @τ

- 单通道召回：Identity=0.154，Prototype=0.216，Reconstruction=0.333
- **三通道并集召回=0.496**（最强单通道=0.333）
- 独有贡献（仅该通道检出的 Unknown 占比）：Identity=0.047，Prototype=0.086，Reconstruction=0.185
- 两两 Jaccard：identity__prototype=0.189，identity__reconstruction=0.188，prototype__reconstruction=0.224

## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）

- 被 Identity 漏掉 4063 个：由Prototype抓住=0.186，由Reconstruction抓住=0.302；**任一其他通道救援=0.404**
- 被 Prototype 漏掉 3763 个：由Identity抓住=0.121，由Reconstruction抓住=0.297；**任一其他通道救援=0.357**
- 被 Reconstruction 漏掉 3202 个：由Identity抓住=0.115，由Prototype抓住=0.173；**任一其他通道救援=0.244**

## 4. 多样性 / 防 Collapse

- 原始未知分数两两 Spearman 最大绝对值=0.809
- 嵌入 CKA：I-P=0.365，I-R=0.355，P-R=0.248
- Known 上 Identity/Prototype 预测分歧率=0.084

## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）

- 固定融合 AUROC：均值=0.9165，max/OR=0.9083；最强单证据=0.9172（最优融合规则：fused_mean）
- 存在固定融合超过最强单证据：否（最优增益 -0.0007）
- 证据未完全相关（|Spearman|<0.95）：是
- Identity 漏检 Unknown 被几何/重构救援比例=0.404
- 并集召回 0.496 相对最强单通道 0.333 有增量：是

说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。

图：`figures/score_histograms.png`、`score_correlation.png`、`overlap_and_rescue.png`、`embedding_tsne.png`