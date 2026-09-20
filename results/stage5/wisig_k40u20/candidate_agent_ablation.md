# Stage-5 候选 Agent / 模块组合消融

所有候选均在同一五折 Leave-Class-Out 协议上比较；当前 20 个真实 Unknown 不参与排名。

## 感知与几何证据（正式三种子）

| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---: |
| identity | 0.886 | 0.837 | 0.903 | 0.065 | 0.122 |
| prototype | 0.936 | 0.889 | 0.917 | 0.310 | 0.460 |
| openmax | 0.847 | 0.787 | 0.891 | 0.643 | 0.747 |

## Boundary Explorer Agent（PUG-V2）

| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---: |
| disagreement_eta1 | 0.816 | 0.810 | 0.911 | 0.052 | 0.098 |
| mixed_eta1.5 | 0.835 | 0.829 | 0.911 | 0.052 | 0.098 |
| disagreement_eta1.5 | 0.828 | 0.821 | 0.911 | 0.052 | 0.098 |
| competition_eta1 | 0.829 | 0.823 | 0.911 | 0.052 | 0.097 |
| mixed_eta1 | 0.823 | 0.816 | 0.911 | 0.052 | 0.097 |
| disagreement_eta2 | 0.834 | 0.828 | 0.911 | 0.052 | 0.097 |
| mixed_eta2 | 0.844 | 0.838 | 0.911 | 0.052 | 0.097 |
| competition_eta1.5 | 0.842 | 0.835 | 0.911 | 0.051 | 0.096 |
| competition_eta2 | 0.848 | 0.841 | 0.911 | 0.051 | 0.096 |

## Calibration Agent（固定无通信）

| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---: |
| mean4 | 0.932 | 0.884 | 0.899 | 0.568 | 0.685 |
| geo_open_boundary | 0.919 | 0.870 | 0.897 | 0.569 | 0.679 |
| mean3 | 0.919 | 0.882 | 0.904 | 0.411 | 0.550 |
| max4 | 0.921 | 0.879 | 0.895 | 0.378 | 0.512 |
| boundary | 0.811 | 0.799 | 0.909 | 0.094 | 0.167 |

## Communication（固定入选校准规则）

| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score | 平均边数 |
|---|---:|---:|---:|---:|---:|---: |
| sparse_cf_budget0.5 | 0.921 | 0.862 | 0.892 | 0.691 | 0.775 | 2.71 |
| full_budget0.5 | 0.914 | 0.859 | 0.895 | 0.679 | 0.770 | 4.00 |
| sparse_budget0.75 | 0.923 | 0.868 | 0.894 | 0.669 | 0.760 | 3.06 |
| sparse_budget0.5 | 0.922 | 0.866 | 0.894 | 0.668 | 0.759 | 2.27 |
| sparse_cf_budget0.75 | 0.914 | 0.857 | 0.892 | 0.653 | 0.749 | 3.67 |
| none_budget0.5 | 0.919 | 0.870 | 0.897 | 0.569 | 0.679 | 0.00 |

## 正式消息反事实

| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---: |
| communication | 0.900 | 0.847 | 0.897 | 0.611 | 0.721 |
| messages_off | 0.936 | 0.881 | 0.894 | 0.679 | 0.769 |
| messages_shuffled | 0.908 | 0.847 | 0.886 | 0.561 | 0.680 |
| drop_identity_sender | 0.909 | 0.864 | 0.900 | 0.487 | 0.628 |
| drop_geometry_sender | 0.903 | 0.847 | 0.895 | 0.653 | 0.751 |
| no_communication | 0.882 | 0.842 | 0.900 | 0.365 | 0.508 |

## 当前筛选结论

- LCO 冻结组合：`{'pug': 'mixed_eta1.5', 'communication': 'sparse_cf_budget0.5_geo_open_boundary', 'mode': 'sparse_cf', 'budget': 0.5, 'score_rule': 'geo_open_boundary'}`。
- Prototype/Geometry 是最稳定的基础开放集证据；OpenMax 的工作点召回较高，但排序指标较弱。
- Geometry + OpenMax + Boundary 的校准协作优于仅 Boundary，校准必须作为独立角色保留。
- 正式推理关闭消息优于完整通信，现有消息边尚未通过因果净增益检验；下一轮只优化消息，不回退已经有效的角色与校准。
- Boundary Explorer 的不同生成策略差距小，下一轮应扩大动作空间（边界壳约束、竞争类方向和尺度），继续由 LCO 反馈选择。