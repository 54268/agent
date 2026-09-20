# Stage-5 多种子聚合

LCO 冻结选择：`{'pug': 'competition_eta2', 'communication': 'audit_sparse_cf_budget0.25_threshold0.5_sensor_mean_arbcommunication_tau0.88_priorglobal', 'mode': 'audit_sparse_cf', 'budget': 0.25, 'gate_threshold': 0.5, 'score_rule': 'sensor_mean', 'arbitration_rule': 'communication', 'known_acceptance': 0.88, 'adaptive_threshold_prior': None}`。

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Identity | 0.842±0.000 | 0.806±0.000 | 0.869±0.000 | 0.482±0.000 | 0.621±0.000 |
| Prototype | 0.935±0.000 | 0.895±0.000 | 0.891±0.000 | 0.790±0.000 | 0.838±0.000 |
| OpenMax | 0.811±0.000 | 0.757±0.000 | 0.887±0.000 | 0.670±0.000 | 0.764±0.000 |
| 等容量无通信 | 0.940±0.000 | 0.902±0.000 | 0.871±0.000 | 0.925±0.000 | 0.897±0.000 |
| 完整协作 | 0.954±0.000 | 0.906±0.000 | 0.874±0.000 | 0.922±0.000 | 0.897±0.000 |
| 推理关闭消息 | 0.949±0.000 | 0.903±0.000 | 0.870±0.000 | 0.924±0.000 | 0.896±0.000 |
| 打乱消息 | 0.951±0.000 | 0.905±0.000 | 0.873±0.000 | 0.916±0.000 | 0.894±0.000 |
| 删除 Identity 发送边 | 0.949±0.000 | 0.903±0.000 | 0.868±0.000 | 0.922±0.000 | 0.894±0.000 |
| 删除 Geometry 发送边 | 0.955±0.000 | 0.907±0.000 | 0.875±0.000 | 0.934±0.000 | 0.904±0.000 |
| 均匀类别融合 | 0.954±0.000 | 0.908±0.000 | 0.875±0.000 | 0.922±0.000 | 0.898±0.000 |

## 按 Tx 分组的 paired bootstrap 95% CI

分层重采样 40 个 Known Tx 与 20 个 Unknown Tx，共 1000 次。

| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |
|---|---:|---:|---:|
| auroc | [0.946, 0.961] | [+0.010, +0.018] | [+0.002, +0.007] |
| oscr | [0.894, 0.919] | [+0.002, +0.008] | [+0.001, +0.005] |
| known_accuracy | [0.853, 0.893] | [-0.003, +0.009] | [-0.003, +0.012] |
| unknown_recall | [0.915, 0.929] | [-0.010, +0.004] | [-0.008, +0.005] |
| h_score | [0.887, 0.908] | [-0.004, +0.004] | [-0.004, +0.006] |

## 预注册成功闸门

- 通过：`unknown_recall_ge_0_90`
- 通过：`known_accuracy_ge_0_85`
- 通过：`h_score_ge_0_875`
- 通过：`auroc_ge_0_95`
- 通过：`oscr_ge_0_90`
- 未通过：`communication_delta_h_ge_0_01`
- 未通过：`communication_delta_oscr_ge_0_005`
- 未通过：`positive_h_seeds_ge_2`
- 通过：`messages_improve_h_vs_off`
- 通过：`messages_improve_oscr_vs_off`
- 未通过：`messages_positive_h_seeds_ge_2`
- 未通过：`messages_positive_oscr_seeds_ge_2`