# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'mixed_eta1', 'communication': 'audit_optional_cf_budget0.1_threshold0.5_idproto_boundary_arbcommunication_tau0.88_prior0', 'mode': 'audit_optional_cf', 'budget': 0.1, 'gate_threshold': 0.5, 'score_rule': 'idproto_boundary', 'arbitration_rule': 'communication', 'known_acceptance': 0.88, 'adaptive_threshold_prior': 0.0}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.906±0.000 | 0.909±0.000 | 0.788±0.000 | 0.887±0.000 | 0.835±0.000 |
| Prototype | 0.955±0.000 | 0.954±0.000 | 0.877±0.000 | 0.901±0.000 | 0.889±0.000 |
| OpenMax | 0.967±0.000 | 0.967±0.000 | 0.881±0.000 | 0.958±0.000 | 0.918±0.000 |
| 等容量无通信 | 0.907±0.000 | 0.907±0.000 | 0.878±0.000 | 0.693±0.000 | 0.774±0.000 |
| 完整协作 | 0.903±0.000 | 0.903±0.000 | 0.884±0.000 | 0.902±0.000 | 0.893±0.000 |
| 推理关闭消息 | 0.650±0.000 | 0.650±0.000 | 0.877±0.000 | 0.182±0.000 | 0.301±0.000 |
| 打乱消息 | 0.893±0.000 | 0.892±0.000 | 0.886±0.000 | 0.886±0.000 | 0.886±0.000 |
| 删除 Identity 发送边 | 0.816±0.000 | 0.815±0.000 | 0.884±0.000 | 0.465±0.000 | 0.609±0.000 |
| 删除 Geometry 发送边 | 0.940±0.000 | 0.939±0.000 | 0.873±0.000 | 0.894±0.000 | 0.883±0.000 |
| 均匀类别融合 | 0.903±0.000 | 0.902±0.000 | 0.884±0.000 | 0.902±0.000 | 0.893±0.000 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 10 个 Known Tx 与 6 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.805, 0.974] | [-0.118, +0.085] | [+0.121, +0.377] |
| oscr | [0.805, 0.974] | [-0.117, +0.085] | [+0.121, +0.377] |
| known_accuracy | [0.871, 0.897] | [-0.004, +0.015] | [-0.008, +0.021] |
| unknown_recall | [0.749, 0.998] | [+0.043, +0.438] | [+0.578, +0.850] |
| h_score | [0.813, 0.939] | [+0.026, +0.267] | [+0.480, +0.730] |

## 预注册成功闸门

- 通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 通过：`h_score_ge_0_875`
- 未通过：`auroc_ge_0_95`
- 通过：`oscr_ge_0_90`
- 通过：`communication_delta_h_ge_0_01`
- 未通过：`communication_delta_oscr_ge_0_005`
- 未通过：`positive_h_seeds_ge_2`
- 通过：`messages_improve_h_vs_off`
- 通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`