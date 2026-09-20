# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'competition_eta2', 'communication': 'audit_optional_cf_budget0.1_threshold0.8_max4_arbcommunication_tau0.88_prior0', 'mode': 'audit_optional_cf', 'budget': 0.1, 'gate_threshold': 0.8, 'score_rule': 'max4', 'arbitration_rule': 'communication', 'known_acceptance': 0.88, 'adaptive_threshold_prior': 0.0}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.843±0.000 | 0.836±0.000 | 0.875±0.000 | 0.678±0.000 | 0.764±0.000 |
| Prototype | 0.623±0.000 | 0.537±0.000 | 0.744±0.000 | 0.257±0.000 | 0.382±0.000 |
| OpenMax | 0.597±0.000 | 0.519±0.000 | 0.744±0.000 | 0.220±0.000 | 0.340±0.000 |
| 等容量无通信 | 0.770±0.000 | 0.765±0.000 | 0.866±0.000 | 0.536±0.000 | 0.662±0.000 |
| 完整协作 | 0.776±0.000 | 0.772±0.000 | 0.863±0.000 | 0.554±0.000 | 0.675±0.000 |
| 推理关闭消息 | 0.796±0.000 | 0.792±0.000 | 0.863±0.000 | 0.540±0.000 | 0.664±0.000 |
| 打乱消息 | 0.777±0.000 | 0.772±0.000 | 0.863±0.000 | 0.554±0.000 | 0.675±0.000 |
| 删除 Identity 发送边 | 0.807±0.000 | 0.802±0.000 | 0.860±0.000 | 0.437±0.000 | 0.579±0.000 |
| 删除 Geometry 发送边 | 0.748±0.000 | 0.743±0.000 | 0.869±0.000 | 0.543±0.000 | 0.668±0.000 |
| 均匀类别融合 | 0.776±0.000 | 0.773±0.000 | 0.864±0.000 | 0.603±0.000 | 0.710±0.000 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 10 个 Known Tx 与 6 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.675, 0.885] | [-0.043, +0.053] | [-0.055, +0.015] |
| oscr | [0.671, 0.879] | [-0.042, +0.053] | [-0.055, +0.015] |
| known_accuracy | [0.852, 0.875] | [-0.012, +0.005] | [-0.001, +0.002] |
| unknown_recall | [0.407, 0.698] | [-0.007, +0.043] | [+0.004, +0.025] |
| h_score | [0.553, 0.772] | [-0.005, +0.037] | [+0.003, +0.020] |

## 预注册成功闸门

- 未通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 未通过：`h_score_ge_0_875`
- 未通过：`auroc_ge_0_95`
- 未通过：`oscr_ge_0_90`
- 通过：`communication_delta_h_ge_0_01`
- 通过：`communication_delta_oscr_ge_0_005`
- 未通过：`positive_h_seeds_ge_2`
- 通过：`messages_improve_h_vs_off`
- 未通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`