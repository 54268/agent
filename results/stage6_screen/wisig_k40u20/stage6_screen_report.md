# Stage-6 LCO 协作筛选

选择：`b2_no_communication`；协作晋级：`False`。

真实 Unknown 未参与训练、选模、阈值或停止轮次。

| 候选 | H-score | OSCR | Known Acc | Unknown Recall | Query rate | Mean bits |
|---|---:|---:|---:|---:|---:|---:|
| b1_early_fusion | 0.3006 | 0.3478 | 0.8040 | 0.1933 | 0.000 | 0.0 |
| b2_no_communication | 0.3665 | 0.4248 | 0.7553 | 0.2528 | 0.000 | 0.0 |
| c1_parallel_request | 0.7266 | 0.7588 | 0.7960 | 0.6722 | 1.000 | 765.9 |
| c2_sequential | 0.7326 | 0.7573 | 0.7950 | 0.6822 | 1.000 | 747.5 |
| c3_sequential_cf | 0.7511 | 0.7626 | 0.7893 | 0.7178 | 0.932 | 691.0 |
| c4_sequential_cf_adversarial | 0.6966 | 0.7425 | 0.7850 | 0.6289 | 0.965 | 708.6 |

## 协作晋级检查

- c2_sequential: 未通过；ΔH=+0.3662，ΔOSCR=+0.3326，正增益 folds=5/4。
- c3_sequential_cf: 未通过；ΔH=+0.3846，ΔOSCR=+0.3379，正增益 folds=5/4。
- c4_sequential_cf_adversarial: 未通过；ΔH=+0.3301，ΔOSCR=+0.3177，正增益 folds=5/4。

未通过时正式真实 Unknown 评价被自动阻断，不能强行选通信模型。