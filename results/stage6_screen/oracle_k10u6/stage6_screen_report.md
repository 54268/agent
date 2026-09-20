# Stage-6 LCO 协作筛选

选择：`b2_no_communication`；协作晋级：`False`。

真实 Unknown 未参与训练、选模、阈值或停止轮次。

| 候选 | H-score | OSCR | Known Acc | Unknown Recall | Query rate | Mean bits |
|---|---:|---:|---:|---:|---:|---:|
| b1_early_fusion | 0.1133 | 0.0803 | 0.1551 | 0.0935 | 0.000 | 0.0 |
| b2_no_communication | 0.1061 | 0.0794 | 0.1547 | 0.0823 | 0.000 | 0.0 |
| c1_parallel_request | 0.1272 | 0.0911 | 0.1624 | 0.1062 | 1.000 | 626.8 |
| c2_sequential | 0.1171 | 0.0872 | 0.1609 | 0.0947 | 1.000 | 696.3 |
| c3_sequential_cf | 0.1143 | 0.0833 | 0.1604 | 0.0937 | 0.998 | 717.1 |
| c4_sequential_cf_adversarial | 0.1179 | 0.0910 | 0.1656 | 0.0935 | 1.000 | 560.0 |

## 协作晋级检查

- c2_sequential: 未通过；ΔH=+0.0110，ΔOSCR=+0.0077，正增益 folds=5/4。
- c3_sequential_cf: 未通过；ΔH=+0.0081，ΔOSCR=+0.0039，正增益 folds=4/4。
- c4_sequential_cf_adversarial: 未通过；ΔH=+0.0118，ΔOSCR=+0.0116，正增益 folds=5/4。

未通过时正式真实 Unknown 评价被自动阻断，不能强行选通信模型。