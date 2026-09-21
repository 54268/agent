# Stage-8 下一轮 G1 重构结果

> 本报告只使用 Known 与 inner proxy-Unknown；formal Unknown 未进入训练、选模、阈值或评价。

## ORACLE small-set overfit

| Profile | 每类样本 | T train acc | S train acc | T pair AUC | S pair AUC | 结果 |
|---|---|---|---|---|---|---|
| legacy_v1 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| legacy_v1 | 32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| legacy_v1 | 64 | 1.0000 | 0.9609 | 1.0000 | 1.0000 | PASS |
| isolated_v2 | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| isolated_v2 | 32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |
| isolated_v2 | 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | PASS |

## Identity / pair / rescue

| Profile | Dataset | Fold | T Acc | S Acc | T pair AUC | S pair AUC | T→S id | S→T id | T→S pair | S→T pair |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy_v1 | wisig | 0 | 0.9248 | 0.8995 | 0.9869 | 0.9888 | 0.0294 | 0.0041 | 0.0249 | 0.0059 |
| legacy_v1 | wisig | 1 | 0.9312 | 0.9108 | 0.9915 | 0.9903 | 0.0267 | 0.0063 | 0.0177 | 0.0091 |
| legacy_v1 | wisig | 2 | 0.9198 | 0.8877 | 0.9917 | 0.9920 | 0.0389 | 0.0068 | 0.0186 | 0.0086 |
| legacy_v1 | wisig | 3 | 0.9343 | 0.9221 | 0.9933 | 0.9948 | 0.0208 | 0.0086 | 0.0131 | 0.0027 |
| legacy_v1 | oracle | 0 | 0.8190 | 0.1719 | 0.9743 | 0.4929 | 0.6797 | 0.0326 | 0.7943 | 0.1029 |
| legacy_v1 | oracle | 1 | 0.7096 | 0.1536 | 0.9493 | 0.4985 | 0.6055 | 0.0495 | 0.7188 | 0.1081 |
| legacy_v1 | oracle | 2 | 0.5742 | 0.1940 | 0.7684 | 0.5199 | 0.4622 | 0.0820 | 0.5964 | 0.2044 |
| legacy_v1 | oracle | 3 | 0.7852 | 0.1680 | 0.9698 | 0.5074 | 0.6484 | 0.0312 | 0.7643 | 0.1107 |
| isolated_v2 | wisig | 0 | 0.8750 | 0.9185 | 0.9790 | 0.9950 | 0.0086 | 0.0521 | 0.0095 | 0.0516 |
| isolated_v2 | wisig | 1 | 0.8881 | 0.9289 | 0.9897 | 0.9951 | 0.0063 | 0.0471 | 0.0145 | 0.0430 |
| isolated_v2 | wisig | 2 | 0.8750 | 0.9158 | 0.9884 | 0.9950 | 0.0100 | 0.0507 | 0.0113 | 0.0399 |
| isolated_v2 | wisig | 3 | 0.9162 | 0.9361 | 0.9927 | 0.9971 | 0.0041 | 0.0240 | 0.0145 | 0.0249 |
| isolated_v2 | oracle | 0 | 0.3177 | 0.1732 | 0.6905 | 0.5004 | 0.2500 | 0.1055 | 0.5456 | 0.3359 |
| isolated_v2 | oracle | 1 | 0.3385 | 0.1823 | 0.7502 | 0.4942 | 0.2799 | 0.1237 | 0.6120 | 0.2995 |
| isolated_v2 | oracle | 2 | 0.3581 | 0.1602 | 0.7962 | 0.4890 | 0.3086 | 0.1107 | 0.6406 | 0.3086 |
| isolated_v2 | oracle | 3 | 0.3763 | 0.1680 | 0.7909 | 0.4871 | 0.3073 | 0.0990 | 0.6250 | 0.2799 |

## Pair verification diagnostics

| Profile | Dataset | Fold | T pair acc | S pair acc | T hard-pair | S hard-pair | T Brier | S Brier |
|---|---|---|---|---|---|---|---|---|
| legacy_v1 | wisig | 0 | 0.9058 | 0.8827 | 0.9805 | 0.9606 | 0.1030 | 0.0490 |
| legacy_v1 | wisig | 1 | 0.9144 | 0.8904 | 0.9778 | 0.9728 | 0.0719 | 0.0478 |
| legacy_v1 | wisig | 2 | 0.9076 | 0.8909 | 0.9819 | 0.9719 | 0.0623 | 0.0482 |
| legacy_v1 | wisig | 3 | 0.9280 | 0.9189 | 0.9905 | 0.9796 | 0.0900 | 0.0319 |
| legacy_v1 | oracle | 0 | 0.8060 | 0.1589 | 0.8073 | 0.1771 | 0.0509 | 0.2317 |
| legacy_v1 | oracle | 1 | 0.7435 | 0.1758 | 0.7591 | 0.2070 | 0.0684 | 0.2313 |
| legacy_v1 | oracle | 2 | 0.5391 | 0.1836 | 0.6042 | 0.1979 | 0.1740 | 0.2131 |
| legacy_v1 | oracle | 3 | 0.7982 | 0.1641 | 0.7982 | 0.2070 | 0.0536 | 0.2325 |
| isolated_v2 | wisig | 0 | 0.8641 | 0.9158 | 0.9325 | 0.9701 | 0.0499 | 0.0319 |
| isolated_v2 | wisig | 1 | 0.8854 | 0.9212 | 0.9398 | 0.9710 | 0.0332 | 0.0340 |
| isolated_v2 | wisig | 2 | 0.8822 | 0.9144 | 0.9479 | 0.9751 | 0.0390 | 0.0424 |
| isolated_v2 | wisig | 3 | 0.9121 | 0.9303 | 0.9656 | 0.9774 | 0.0325 | 0.0261 |
| isolated_v2 | oracle | 0 | 0.3281 | 0.1693 | 0.3372 | 0.1784 | 0.1938 | 0.2441 |
| isolated_v2 | oracle | 1 | 0.3607 | 0.1771 | 0.3841 | 0.1797 | 0.1886 | 0.2282 |
| isolated_v2 | oracle | 2 | 0.3893 | 0.1576 | 0.3997 | 0.1719 | 0.1777 | 0.2452 |
| isolated_v2 | oracle | 3 | 0.3906 | 0.1576 | 0.3945 | 0.1693 | 0.1687 | 0.2298 |

## Directional rescue predictability

| Profile | Dataset | Fold | T→S AUROC | T→S PR-AUC | T→S n | S→T AUROC | S→T PR-AUC | S→T n |
|---|---|---|---|---|---|---|---|---|
| legacy_v1 | wisig | 0 | 0.9459 | 0.3200 | 65 | 0.9073 | 0.2476 | 9 |
| legacy_v1 | wisig | 1 | 0.9532 | 0.2852 | 59 | 0.9285 | 0.1072 | 14 |
| legacy_v1 | wisig | 2 | 0.9582 | 0.4345 | 86 | 0.9493 | 0.1545 | 15 |
| legacy_v1 | wisig | 3 | 0.9539 | 0.3003 | 46 | 0.9352 | 0.1436 | 19 |
| legacy_v1 | oracle | 0 | 0.6573 | 0.7781 | 522 | 0.7007 | 0.0603 | 25 |
| legacy_v1 | oracle | 1 | 0.6829 | 0.7659 | 465 | 0.6677 | 0.0840 | 38 |
| legacy_v1 | oracle | 2 | 0.5398 | 0.5181 | 355 | 0.5601 | 0.1048 | 63 |
| legacy_v1 | oracle | 3 | 0.6566 | 0.7772 | 498 | 0.6230 | 0.0448 | 24 |
| isolated_v2 | wisig | 0 | 0.9507 | 0.1016 | 19 | 0.9489 | 0.5128 | 115 |
| isolated_v2 | wisig | 1 | 0.9546 | 0.0715 | 14 | 0.9506 | 0.4064 | 104 |
| isolated_v2 | wisig | 2 | 0.9609 | 0.1542 | 22 | 0.9296 | 0.3833 | 112 |
| isolated_v2 | wisig | 3 | 0.9681 | 0.0767 | 9 | 0.9452 | 0.2397 | 53 |
| isolated_v2 | oracle | 0 | 0.6137 | 0.4054 | 192 | 0.5299 | 0.1115 | 81 |
| isolated_v2 | oracle | 1 | 0.6049 | 0.3838 | 215 | 0.5167 | 0.1344 | 95 |
| isolated_v2 | oracle | 2 | 0.6049 | 0.4305 | 237 | 0.5606 | 0.1246 | 85 |
| isolated_v2 | oracle | 3 | 0.6176 | 0.4532 | 236 | 0.5210 | 0.1067 | 76 |

## Unknown rejection rescue（辅助指标）

| Profile | Dataset | Fold | T→S unknown | S→T unknown |
|---|---|---|---|---|
| legacy_v1 | wisig | 0 | 0.0217 | 0.0068 |
| legacy_v1 | wisig | 1 | 0.0109 | 0.0340 |
| legacy_v1 | wisig | 2 | 0.0149 | 0.0747 |
| legacy_v1 | wisig | 3 | 0.0177 | 0.0883 |
| legacy_v1 | oracle | 0 | 0.0000 | 0.0352 |
| legacy_v1 | oracle | 1 | 0.0039 | 0.0781 |
| legacy_v1 | oracle | 2 | 0.2148 | 0.0508 |
| legacy_v1 | oracle | 3 | 0.1133 | 0.0352 |
| isolated_v2 | wisig | 0 | 0.0177 | 0.0149 |
| isolated_v2 | wisig | 1 | 0.0095 | 0.0149 |
| isolated_v2 | wisig | 2 | 0.0109 | 0.0245 |
| isolated_v2 | wisig | 3 | 0.0394 | 0.0177 |
| isolated_v2 | oracle | 0 | 0.0000 | 0.0742 |
| isolated_v2 | oracle | 1 | 0.0352 | 0.0430 |
| isolated_v2 | oracle | 2 | 0.0703 | 0.0352 |
| isolated_v2 | oracle | 3 | 0.0820 | 0.0430 |

## Gate 决策

| Profile | Dataset | G1 | G1.5 | 最终 |
|---|---|---|---|---|
| legacy_v1 | wisig | FAIL | PASS | STOP-AND-REDESIGN |
| legacy_v1 | oracle | FAIL | FAIL | STOP-AND-REDESIGN |
| isolated_v2 | wisig | FAIL | PASS | STOP-AND-REDESIGN |
| isolated_v2 | oracle | FAIL | FAIL | STOP-AND-REDESIGN |

## Observation leakage probe

| Dataset | Profile | T→spectral R² | ΔR² | S→temporal R² | ΔR² |
|---|---|---|---|---|---|
| wisig | legacy_v1 | 0.3661 | — | 0.8406 | — |
| wisig | isolated_v2 | 0.7711 | +0.4049 | 0.8244 | -0.0162 |
| oracle | legacy_v1 | -0.0231 | — | -0.0106 | — |
| oracle | isolated_v2 | 0.0008 | +0.0239 | -0.0093 | +0.0013 |

Leakage probe 的 held-out R² 越低，表示跨 observation 的可恢复信息越少；该审计不设人为通过阈值。
本轮 v2 没有在两个数据集上同时降低 R²，因此不能宣称量化信息隔离已经成立。
Class reindex 的逐折误差见同目录 JSON。

## 失败归因

- `legacy_v1/wisig`：G1 失败项 ['temporal_to_spectral_identity_rescue', 'spectral_to_temporal_identity_rescue', 'temporal_to_spectral_pair_rescue', 'spectral_to_temporal_pair_rescue']；G1.5 失败项 无。
- `legacy_v1/oracle`：G1 失败项 ['spectral_local_learnability', 'spectral_pair_auroc', 'spectral_hard_pair_accuracy', 'spectral_to_temporal_identity_rescue']；G1.5 失败项 ['auroc_temporal_rescues_spectral', 'auroc_spectral_rescues_temporal']。
- `isolated_v2/wisig`：G1 失败项 ['temporal_to_spectral_identity_rescue', 'spectral_to_temporal_identity_rescue', 'temporal_to_spectral_pair_rescue', 'spectral_to_temporal_pair_rescue']；G1.5 失败项 无。
- `isolated_v2/oracle`：G1 失败项 ['temporal_local_learnability', 'spectral_local_learnability', 'spectral_pair_auroc', 'temporal_hard_pair_accuracy', 'spectral_hard_pair_accuracy']；G1.5 失败项 ['auroc_temporal_rescues_spectral', 'auroc_spectral_rescues_temporal']。

只要任一数据集未同时通过 G1 与 G1.5，Memory、Auditor、Router 与 forced communication 继续锁定。
