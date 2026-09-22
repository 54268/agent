# MAROS-SEI：多智能体开放集辐射源识别

当前主线已经完成到 Stage 14，并冻结为 **A+C 异构通信 MAPPO 开放集拒识框架**。

## 从这里开始

1. 阅读根目录 [`多智能体开放集拒识_主流程.md`](多智能体开放集拒识_主流程.md)：最终架构、数据协议、训练、推理、指标和运行入口；
2. 阅读 [`tests/stage14/README.md`](tests/stage14/README.md)：测试、消融、诊断、历史计划和非最终产物索引；
3. 最终全量单折结果位于 `results/stage14/final_ac_fold0_seed42/`；
4. Stage 1–13 的旧结果保留在 `总结果汇总.md`、`docs/` 和 `results/`，只作为历史研究记录。

## 最终 Stage 14 入口

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage14_comm_mappo.py --config configs/experiments/stage14_comm_mappo_oracle_ac.json
```

测试：

```powershell
conda run --no-capture-output -n pytorch python -m pytest -q tests/stage14
conda run --no-capture-output -n pytorch python -m pytest -q
```

## 当前最终选择

- Agent A：已知身份假设；
- Agent C：开放集证据获取和最终接受/拒绝；
- 原 B Actor 删除，其有效几何原型证据并入 C 的按需工具；
- 独立 GRU Actor + 集中式训练 Critic；
- 推理模型只有 A、C Actor，不包含 Critic；
- 正式 Unknown 不进入训练、校准或当前 smoke 评估。

当前 Oracle 全量单折训练结果：Known Accuracy 93.94%、Unknown Recall 96.44%、H-score 95.17%、AUROC 98.64%、OSCR 98.33%。在仅用于曲线比较的 95% Known 匹配点，Unknown Recall 为 95.69%、H-score 为 95.34%。正式 5 折、多种子实验仍应在服务器运行。

## 目录职责

| 目录 | 内容 |
|---|---|
| `src/maros_stage14/` | 最终多智能体开放集算法 |
| `scripts/experiments/` | 正式训练入口 |
| `configs/experiments/` | 最终主配置 |
| `results/stage14/final_ac_fold0_seed42/` | 当前全量单折结果与 Actor-only 检查点 |
| `tests/stage14/` | 测试、对照、诊断、计划和非最终产物 |
| `docs/` | Stage 1–13 历史与长期设计资料 |

根目录只承担导航和最终主流程说明，不再堆放阶段测试报告。
