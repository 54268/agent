# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'mixed_eta1.5', 'communication': 'audit_optional_cf_budget0.1_threshold0.65_mean4_arbmax_tau0.88_priorglobal', 'mode': 'audit_optional_cf', 'budget': 0.1, 'gate_threshold': 0.65, 'score_rule': 'mean4', 'arbitration_rule': 'max', 'known_acceptance': 0.88, 'adaptive_threshold_prior': None}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.879±0.003 | 0.834±0.011 | 0.875±0.004 | 0.580±0.083 | 0.694±0.062 |
| Prototype | 0.938±0.011 | 0.893±0.002 | 0.880±0.004 | 0.815±0.045 | 0.845±0.023 |
| OpenMax | 0.863±0.030 | 0.807±0.022 | 0.861±0.014 | 0.747±0.063 | 0.798±0.032 |
| 等容量无通信 | 0.903±0.022 | 0.872±0.006 | 0.868±0.013 | 0.812±0.036 | 0.839±0.015 |
| 完整协作 | 0.948±0.005 | 0.892±0.007 | 0.862±0.013 | 0.892±0.010 | 0.877±0.012 |
| 推理关闭消息 | 0.939±0.006 | 0.894±0.005 | 0.869±0.015 | 0.882±0.011 | 0.875±0.010 |
| 打乱消息 | 0.944±0.002 | 0.893±0.006 | 0.866±0.015 | 0.886±0.013 | 0.876±0.012 |
| 删除 Identity 发送边 | 0.936±0.007 | 0.891±0.003 | 0.869±0.012 | 0.875±0.018 | 0.872±0.010 |
| 删除 Geometry 发送边 | 0.939±0.008 | 0.892±0.003 | 0.869±0.014 | 0.883±0.011 | 0.876±0.006 |
| 均匀类别融合 | 0.948±0.005 | 0.894±0.007 | 0.862±0.014 | 0.892±0.010 | 0.877±0.012 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 40 个 Known Tx 与 20 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.941, 0.955] | [+0.037, +0.054] | [+0.007, +0.013] |
| oscr | [0.879, 0.906] | [+0.013, +0.029] | [-0.003, -0.000] |
| known_accuracy | [0.844, 0.880] | [-0.017, +0.005] | [-0.011, -0.004] |
| unknown_recall | [0.885, 0.898] | [+0.074, +0.085] | [+0.008, +0.012] |
| h_score | [0.867, 0.886] | [+0.032, +0.045] | [-0.001, +0.003] |

## 预注册成功闸门

- 未通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 通过：`h_score_ge_0_875`
- 未通过：`auroc_ge_0_95`
- 未通过：`oscr_ge_0_90`
- 通过：`communication_delta_h_ge_0_01`
- 通过：`communication_delta_oscr_ge_0_005`
- 通过：`positive_h_seeds_ge_2`
- 通过：`messages_improve_h_vs_off`
- 未通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`