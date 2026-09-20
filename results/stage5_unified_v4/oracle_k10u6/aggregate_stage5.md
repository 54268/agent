# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'competition_eta1.5', 'communication': 'audit_sparse_cf_budget0.25_threshold0.5_idproto_boundary_arbcommunication_tau0.9_prior25', 'mode': 'audit_sparse_cf', 'budget': 0.25, 'gate_threshold': 0.5, 'score_rule': 'idproto_boundary', 'arbitration_rule': 'communication', 'known_acceptance': 0.9, 'adaptive_threshold_prior': 25.0}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.906±0.000 | 0.909±0.000 | 0.902±0.000 | 0.714±0.000 | 0.797±0.000 |
| Prototype | 0.955±0.000 | 0.954±0.000 | 0.898±0.000 | 0.882±0.000 | 0.890±0.000 |
| OpenMax | 0.967±0.000 | 0.967±0.000 | 0.899±0.000 | 0.948±0.000 | 0.923±0.000 |
| 等容量无通信 | 0.910±0.000 | 0.909±0.000 | 0.899±0.000 | 0.661±0.000 | 0.762±0.000 |
| 完整协作 | 0.948±0.000 | 0.948±0.000 | 0.895±0.000 | 0.885±0.000 | 0.890±0.000 |
| 推理关闭消息 | 0.662±0.000 | 0.661±0.000 | 0.902±0.000 | 0.177±0.000 | 0.296±0.000 |
| 打乱消息 | 0.942±0.000 | 0.941±0.000 | 0.898±0.000 | 0.868±0.000 | 0.883±0.000 |
| 删除 Identity 发送边 | 0.821±0.000 | 0.821±0.000 | 0.904±0.000 | 0.491±0.000 | 0.636±0.000 |
| 删除 Geometry 发送边 | 0.940±0.000 | 0.940±0.000 | 0.895±0.000 | 0.888±0.000 | 0.891±0.000 |
| 均匀类别融合 | 0.948±0.000 | 0.948±0.000 | 0.895±0.000 | 0.885±0.000 | 0.890±0.000 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 10 个 Known Tx 与 6 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.887, 0.993] | [-0.036, +0.115] | [+0.179, +0.402] |
| oscr | [0.887, 0.992] | [-0.036, +0.114] | [+0.179, +0.402] |
| known_accuracy | [0.887, 0.902] | [-0.016, +0.010] | [-0.020, +0.005] |
| unknown_recall | [0.699, 0.999] | [+0.046, +0.460] | [+0.536, +0.851] |
| h_score | [0.786, 0.945] | [+0.024, +0.293] | [+0.468, +0.740] |

## 预注册成功闸门

- 未通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 通过：`h_score_ge_0_875`
- 未通过：`auroc_ge_0_95`
- 通过：`oscr_ge_0_90`
- 通过：`communication_delta_h_ge_0_01`
- 通过：`communication_delta_oscr_ge_0_005`
- 未通过：`positive_h_seeds_ge_2`
- 通过：`messages_improve_h_vs_off`
- 通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`