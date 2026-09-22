# Stage 14：早期 smoke 与拓扑消融报告

> 本报告已归档。最终全量单折结果见根目录 `多智能体开放集拒识_主流程.md` 和 `results/stage14/final_ac_fold0_seed42/`。

## 当前结论

Stage 14 已实现为可训练、可评估、可审计的多智能体开放集算法。当前主拓扑是 **A+C**，不再把 A/B/C 当作不可变的三智能体模板：

- A（identity）提出或修正已知类身份假设；
- C（open_set）按需调用身份原型、几何原型、OpenMax 和冻结边界工具，并作已知/未知终局裁决；
- 原 B 的几何感知能力保留为 C 的私有工具，不在主实验中保留独立 B Actor；
- 可选 ABC 拓扑只用于消融和后续研究，不是默认部署拓扑；
- 未来未知类细分可以新增独立智能体，不需要恢复固定 ABC 结构。

这不是把多个分类头命名成多个智能体。A 和 C 分别拥有独立的循环 Actor、局部观测、动作空间和消息预算，通过延迟一轮的结构化消息协作；训练期只有一个集中式状态价值 Critic，推理包不包含 Critic。

## 数据与泄漏约束

训练 episode 只来自：

1. 外层 fold 的 Known 训练样本；
2. 同一训练划分内构造的 LCO 未知代理；
3. 通过 Stage 13 认证的特征空间 PUG。

正式未知样本由 `TrainingEpisodeBank` 和数据协议共同禁止进入训练、校准和工具拟合。运行报告显式记录 `formal_unknown_used=false`，并只记录未访问的正式未知样本数量。

动作使用 TOP1/TOP2 排名语义，而不是随已知类数量变化的类别动作，因此不同 fold 的 Actor 输出维度保持固定。环境采用同步联合动作；本轮发出的查询只能在下一轮收到响应，避免同轮读写造成非因果通信。

## 算法组成

- 独立 GRU Actor：A 与 C 不共享参数；
- 集中式循环 `V(s)` Critic：只读训练期允许的全局状态，不读取真实标签、样本来源或当前联合动作；
- MAPPO：GAE、裁剪策略目标、裁剪价值目标、熵正则、梯度裁剪、按完整 episode 切分 minibatch；
- 结构化 Blackboard：消息仅包含候选、置信度、间隔、风险类型和原因码；
- 动作 mask：阻止无假设时接受、无请求时回应、未检查工具时终局裁决等非法动作；
- 审计轨迹：保存局部观测摘要、动作 mask、动作概率、消息、工具调用、公开证据、预算、奖励和终止原因；
- 因果消融：支持强制 B 等待和打乱全部 C 工具输出。

## 单折真实数据 smoke 结果

配置：Oracle，outer fold 0，seed 42，24 次更新，每次 64 个 episode；评估包含 256 个 Known 和 64 个 test-split LCO 未知代理。正式未知测试集没有被访问。

| 拓扑 | Known Acc. | Unknown Recall | H-score | AUROC | OSCR |
|---|---:|---:|---:|---:|---:|
| A+C（主拓扑） | 95.31% | 93.75% | **94.52%** | 96.91% | 96.72% |
| A+B+C（消融拓扑） | 86.72% | **98.44%** | 92.21% | **98.61%** | **97.80%** |

A+C 的 H-score 比 ABC 高 2.32 个百分点，且已知类准确率高 8.59 个百分点；ABC 更偏向拒识，未知召回、AUROC 和 OSCR 更高。当前阶段以 Known/Unknown 平衡的 H-score 作为拓扑选择依据，因此采用 A+C。

不能把“ABC 内强制 B 等待”的结果解释成 B 从未发挥作用：共同训练后的 C 会依赖 B，强制 B 等待时 H-score 会明显下降。这说明存在策略共适应。更有意义的比较是分别训练的完整拓扑；在相同数据、种子和训练预算下，合并几何功能的 A+C 获得更高 H-score。

在 A+C 上随机置换全部 C 工具的样本对应关系后，H-score 从 94.52% 降至 80.47%，下降 14.05 个百分点。这为“策略实际使用样本级开放集证据”提供了因果证据，而不仅是动作形式上的多智能体包装。

### Known Accuracy 95% 受控工作点

为避免默认 argmax 工作点偏向拒识，额外使用校准集 Known 样本选择拒识阈值。阈值选择不读取校准未知或测试未知标签：在满足目标 Known Accuracy 的阈值中选择最严格的一个，以尽量保留 Unknown Recall。另报告一个使用测试 Known 标签精确匹配 95% 的诊断曲线点；它只用于公平比较，不可作为部署阈值。

在测试 Known Accuracy 同为 95.31% 时：

| 拓扑 | Known Acc. | Unknown Recall | H-score | AUROC | OSCR |
|---|---:|---:|---:|---:|---:|
| A+C | 95.31% | **93.75%** | **94.52%** | 96.91% | 96.72% |
| A+B+C | 95.31% | 92.19% | 93.72% | **98.61%** | **97.80%** |

因此，在“Known Accuracy 不低于 95%”这个实际工作点约束下，A+C 的 Unknown Recall 高 1.56 个百分点、H-score 高 0.80 个百分点，继续作为最终主拓扑。ABC 的 AUROC/OSCR仍然领先，说明它的全阈值排序能力更强，但该优势没有转化为目标工作点上的更好拒识；ABC 代码和配置继续保留为研究消融，而不进入默认部署。

这些结果只证明真实数据链路、学习能力和协议约束已经成立，不能替代 5 折、多随机种子的正式统计结论。

## 运行方式

主拓扑：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage14_comm_mappo.py --config tests/stage14/configs/oracle_ac_smoke.json
```

ABC 消融拓扑：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage14_comm_mappo.py --config tests/stage14/configs/oracle_abc_smoke.json
```

Known Accuracy 工作点扫描：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage14_comm_mappo.py --config tests/stage14/configs/oracle_abc_known95.json
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage14_comm_mappo.py --config tests/stage14/configs/oracle_ac_known95.json
```

本报告对应的旧 smoke 结果位于 `tests/stage14/artifacts/oracle_fold0_ac_smoke/`：

- `stage14_metrics.json`：训练历史、主指标、泄漏声明和因果消融；
- `trajectories.jsonl`：逐样本完整协作轨迹；
- `actors_inference.pt`：仅含推理期 A、C Actor，不含 Critic。

本地 RTX 5070 12 GB 可以完成当前单折 smoke，当前没有显存不足。正式 5 折、多种子、多个消融组合建议放到服务器，原因是总训练时长和实验并发，而不是单个模型无法装入本地显存。
