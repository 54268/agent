# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'competition_eta1', 'communication': 'audit_residual_cf_budget0.1_threshold0.65_idproto_arbcommunication_tau0.88_prior0', 'mode': 'audit_residual_cf', 'budget': 0.1, 'gate_threshold': 0.65, 'score_rule': 'idproto', 'arbitration_rule': 'communication', 'known_acceptance': 0.88, 'adaptive_threshold_prior': 0.0}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.843±0.000 | 0.836±0.000 | 0.875±0.000 | 0.678±0.000 | 0.764±0.000 |
| Prototype | 0.623±0.000 | 0.537±0.000 | 0.744±0.000 | 0.257±0.000 | 0.382±0.000 |
| OpenMax | 0.597±0.000 | 0.519±0.000 | 0.744±0.000 | 0.220±0.000 | 0.340±0.000 |
| 等容量无通信 | 0.942±0.000 | 0.935±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 完整协作 | 0.942±0.000 | 0.936±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 推理关闭消息 | 0.942±0.000 | 0.936±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 打乱消息 | 0.942±0.000 | 0.936±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 删除 Identity 发送边 | 0.942±0.000 | 0.936±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 删除 Geometry 发送边 | 0.942±0.000 | 0.936±0.000 | 0.879±0.000 | 0.869±0.000 | 0.874±0.000 |
| 均匀类别融合 | 0.942±0.000 | 0.936±0.000 | 0.878±0.000 | 0.869±0.000 | 0.874±0.000 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 10 个 Known Tx 与 6 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.887, 0.990] | [+0.000, +0.000] | [+0.000, +0.000] |
| oscr | [0.883, 0.983] | [+0.000, +0.001] | [+0.000, +0.000] |
| known_accuracy | [0.867, 0.892] | [-0.000, +0.001] | [+0.000, +0.000] |
| unknown_recall | [0.715, 0.998] | [+0.000, +0.001] | [+0.000, +0.000] |
| h_score | [0.789, 0.937] | [-0.000, +0.001] | [+0.000, +0.000] |

## 预注册成功闸门

- 未通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 未通过：`h_score_ge_0_875`
- 未通过：`auroc_ge_0_95`
- 通过：`oscr_ge_0_90`
- 未通过：`communication_delta_h_ge_0_01`
- 未通过：`communication_delta_oscr_ge_0_005`
- 未通过：`positive_h_seeds_ge_2`
- 未通过：`messages_improve_h_vs_off`
- 未通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`