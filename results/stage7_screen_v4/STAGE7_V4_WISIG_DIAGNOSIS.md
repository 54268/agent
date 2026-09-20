# Stage-7 V4 WiSig 单折诊断

## 结论

V4 已解决 V3 最关键的“本地 Agent 训练量不足”问题：Waveform 与 Prototype 均通过训练后 eval-mode 能力闸门，并且在外层代理开放集测试上同时达到 `Known Accuracy >= 0.85` 和 `AUROC >= 0.75`。

但这不等于 G0 通过。outer fold 0 的总体 sanity 结果为 **失败**：公平 B2 虽然在训练集拟合审计中被选中，外层 H-score 却只有 `0.322066`，比最佳单 Agent Prototype 低 `0.152222`；Waveform 对 B2 的 unique rescue 也只有 `0.2174%`。当前结果说明本地专家已经可用、公共摘要中存在选择信号，而且逐样本 oracle headroom 很大，但 B2 尚未把互补性稳定地转化为可部署增益。

因此本轮不得进入 G1、G2 或正式 Unknown 评价。下一步只应训练四个真实 inner folds，用 inner out-of-fold episode 数据学习公共信息 B2/meta-selector，再冻结后评价一次未参与训练的 outer fold。

## 协议与数据范围

- 数据集：WiSig，同日同接收机协议。
- partition seed：`2026`；训练 seed：`42`。
- 当前只运行 outer fold 0：`32 Known / 8 proxy Unknown`。
- outer proxy-unknown classes：`[9, 11, 12, 21, 24, 25, 30, 38]`。
- 训练 Known：`10,176`；校准 Known：`1,440`；测试 Known：`2,944`；测试 proxy Unknown：`736`。
- Known 校准集与 Known 测试集互斥；OSCR 使用拒识前类别预测；`STOP` 与公平 B2 输出严格一致。
- WiSig 当前 20 个正式 Unknown 本轮未用于训练、选择、校准或评价。
- 单折总耗时：`465.2954 s`。该结果只是 feasibility sanity，不能单独晋级 G1。

## 1. 本地能力闸门：通过

本轮取消每类样本上限，使用全部 `10,176` 个 outer-train Known 样本预训练 30 epochs；本地预训练 Unknown loss 为 `0`，Prototype SupCon 权重为 `0`。预训练完成后以 eval mode 在训练 Known 上独立检查：

| Agent | eval-mode train accuracy | 要求 | 结果 |
|---|---:|---:|---|
| Waveform | 0.966785 | 0.85 | 通过 |
| Prototype | 0.986832 | 0.85 | 通过 |

这项硬闸门只证明两个编码器具有合法的独立分类能力。它没有使用正式 Unknown，也不证明 B2、选择器或通信有效。

训练历史与能力恢复一致：第 30 个本地 epoch 的 batch-mode Known accuracy 为 Waveform `0.964131`、Prototype `0.982410`；随后 5 个 open-head epochs 中，主干精度保持在约 `0.966785 / 0.986930`。

## 2. 外层代理开放集指标

每个规则均独立使用 Known calibration 设定分类别经验 CDF 阈值，工作点目标约为 `95%` Known 接受率；测试集为 `2,944 Known + 736 proxy Unknown`。

| 方法 | AUROC | AUPR-Out | FPR95 | Known Acc. | Unknown Recall | H-score | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waveform | 0.833486 | 0.429051 | 0.277174 | 0.878397 | 0.203804 | 0.330846 | 0.805356 |
| Prototype | **0.921095** | **0.654072** | **0.144701** | **0.899796** | **0.322011** | **0.474288** | 0.887771 |
| 公平 B2 | 0.844544 | 0.440878 | 0.257812 | 0.881793 | 0.197011 | 0.322066 | 0.817025 |
| public selector（仅诊断） | 0.899363 | 0.614196 | 0.233696 | 0.896399 | 0.372283 | 0.526080 | 0.869790 |
| Agent oracle upper bound | — | — | — | 0.916780 | 0.463315 | 0.615549 | **0.917582** |

两个单 Agent 的必要能力检查均通过：Waveform/Prototype 的 Known Accuracy 分别为 `0.878397/0.899796`，AUROC 分别为 `0.833486/0.921095`。Prototype 在全部主要开放集指标上明显强于 Waveform，是当前最佳单 Agent。

分类别阈值在 Known 测试集上实际接受率为 Waveform `94.8030%`、Prototype `94.7690%`、B2 `94.9389%`，说明约 95% Known 接受率目标基本实现；但对应 proxy-unknown Recall 仅为 `20.38% / 32.20% / 19.70%`。这不是通过过度拒绝 Known 换来的高 Recall，反而表明当前未知风险分数在该工作点对难未知类仍然缺乏分离力。

## 3. B2 锚点历史与外层失败

B2 训练的最终锚点审计为：

- `waveform_objective = 0.480162`
- `prototype_objective = 0.481495`
- `candidate_fit_objective = 0.466864`
- `anchor_fit_objective = 0.480162`
- 候选相对 Waveform 锚点的拟合目标改善为 `0.013298`
- 要求的最小改善为 `0.005`
- `anchor_agent = waveform`
- `adaptive_selected = 1`

因此训练器在拟合数据上选择了自适应 B2，而不是回退为单 Agent。五个 B2 epochs 中，selector loss 从 `1.648110` 降至 `1.439115`，no-regret 项从 `0.084755` 降至 `0.081623`，Known 训练准确率从 `0.971403` 升至 `0.972681`。

然而这个拟合改进没有泛化到外层类别级 holdout：

- `B2 H - best local H = -0.152222`（`0.322066 - 0.474288`）；
- `B2 OSCR - best local OSCR = -0.070747`；
- B2 Unknown Recall `0.197011`，甚至略低于 Waveform 的 `0.203804`；
- G0 只容许 B2 相对最佳单 Agent 最多下降 `0.005`。

所以本轮 B2 是明确失败，而不是“接近通过”。训练集上的 no-regret 拟合审计不足以保证跨类别泛化，也不能通过放宽选择条件把它记为协作增益。

## 4. 互补性上限：很大，但尚未兑现

逐样本 Agent oracle 相对 B2 的上限为：

- `Delta H = +0.293483`（`0.322066 -> 0.615549`）；
- `Delta OSCR = +0.100558`（`0.817025 -> 0.917582`）；
- B2 样本级成功率 `0.744837`，oracle 成功率 `0.826087`，绝对 headroom 为 `0.081250`。

Prototype 相对 B2 产生 `286` 个 unique rescues：占全部测试样本 `7.7717%`，占 B2 失败样本 `30.4579%`。Waveform 只产生 `8` 个 unique rescues：占全部样本 `0.2174%`，占 B2 失败样本 `0.8520%`，未达到 G0 的 `5%` 要求。

oracle 需要事后知道每个样本上哪个 Agent 的决策正确，只是互补潜力上界，不是可部署方法成绩。巨大的 `Delta H` 主要也受当前 B2 很弱影响，不能直接解释为多智能体协作已产生收益；它只支持继续研究如何从 inner episode 学到跨类别可泛化的选择规则。

## 5. public selector 为什么仍只是诊断

当前 public selector 只读取两个 Agent 的公共摘要，并使用样本级 5-fold cross-fitting。它得到：

- action accuracy：`0.866304`；
- mean loss：`0.208424`；
- gain over STOP：`0.048098`；
- oracle gain recovery：`0.591973`；
- 相对 B2：`Delta H = +0.204014`、`Delta OSCR = +0.052766`；
- 相对最佳单 Agent：`Delta H = +0.051792`。

这些数值说明公共摘要中确实存在较强的样本选择信号，且诊断结果已超过最佳单 Agent。但它仍不能登记为可部署成绩：selector 的动作标签由当前 outer 测试池中两个 Agent 的真实正确/错误结果生成，交叉拟合训练折仍包含同一批 outer proxy-unknown 类的其他测试样本。它只做到了样本级 out-of-fold，没有做到类别级 outer holdout；真实部署时也不存在 outer 测试标签供其重新拟合。

因此该 selector 只能回答“公共信息是否包含可学习信号”，不能作为公平 B2，更不能作为通信收益。真正的检验必须把 selector 完全训练在 inner episode，再冻结到未见 outer 类别上评价。

## 6. G0 检查结果

通过 8 项：

- 两个 Agent 的 Known Accuracy 均不低于 `0.85`；
- 两个 Agent 的 proxy-unknown AUROC 均不低于 `0.75`；
- oracle `Delta H >= 0.03`；
- oracle `Delta OSCR >= 0.015`；
- Prototype unique rescue 不低于 `5%`；
- 诊断 public selector 相对最佳单 Agent 达到 `Delta H >= 0.01`。

失败 2 项：

- B2 未保持在最佳单 Agent H-score 的 `0.005` 范围内；
- Waveform unique rescue 未达到 `5%`。

所以 `sanity.passed = false`。当前正确表述是：**本地能力恢复成功，Prototype 明显更强，公共摘要具有诊断性选择信号，互补 headroom 充足，但公平、可部署的 G0 协同基线仍然失败。**

## 7. 下一步：真正的 nested-inner 训练

不得继续使用 outer fold 0 的测试标签调 B2 或 selector。下一步按已冻结 manifest 完成 outer fold 0 内的四个 inner folds：

1. 每个 inner fold 只用 `24` 个 inner-known 类训练独立的 Waveform/Prototype；另外 `8` 类作为该 inner episode 的类别级 proxy Unknown。
2. 每折使用 `7,632` train-known、`2,544` train-proxy-unknown、`1,080` calibration-known、`360` validation-proxy-unknown、`2,208` test-known 和 `736` test-proxy-unknown；不得把 outer 正式测试样本或 WiSig 正式 Unknown 放入训练调用链。
3. 汇总四折 out-of-fold 的公共摘要、局部决策和精确 action loss，训练共享、类别数无关的公共信息 B2/meta-selector；该决策器不得读取私有 embedding 或 outer proxy-unknown 标签。
4. 冻结 inner 学到的决策规则，再在 outer fold 0 的 `2,944 Known + 736 proxy Unknown` 上评价一次。
5. 优先确认 selector 的 `+0.051792` 诊断潜力能否跨类别保留，同时改善 Prototype 之外的样本而不把弱 Waveform 大面积选入。
6. 只有 nested 结果同时保持 B2 公平性、超过最佳单 Agent并满足 G0 全部条件，才运行其余四个 outer folds；五折 G0 通过前不实现或运行 G1 通信。

## 状态声明

- 结果级别：Stage-7 V4 WiSig、单 outer fold、开发期 G0 sanity。
- 正式 Unknown：**未评价**。
- G1 消息价值：**未运行**。
- G2 路由：**未运行**。
- 可声明结论：两个异构本地 Agent 已恢复独立开放集能力；Prototype 是当前强专家；公共摘要和逐样本 oracle 均显示尚未兑现的互补空间。
- 不可声明结论：公平 B2 已优于单 Agent、诊断 selector 可部署、通信带来净增益、Stage-7 已通过 G0。
