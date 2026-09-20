# Stage-7 V4 ORACLE 单折诊断

## 结论

V4 已解决 V3 最关键的“本地 Agent 没有被充分训练”问题：Waveform 与 Prototype 均通过训练后 eval-mode 能力闸门，并且在外层代理开放集测试上同时达到 `Known Accuracy >= 0.85` 和 `AUROC >= 0.75`。

但这不等于 G0 通过。外层 fold 0 的总体 sanity 结果为 **失败**：公平 B2 回退为 Waveform，H-score 比最佳单 Agent 低 `0.046021`；Waveform 对 B2 没有 unique rescue；当前 public selector 也没有超过最佳单 Agent。现阶段只能确认“两个局部专家均可用且存在明显互补上限”，尚未证明可部署的公共信息选择器，更未证明通信有价值。

因此本轮不得进入 G1、G2 或正式 Unknown 评价。下一步只做四个真实 inner folds 的嵌套训练，使用 inner out-of-fold episode 数据学习公共信息决策，再冻结后评价一次未参与训练的 outer fold。

## 协议与数据范围

- 数据集：ORACLE。
- partition seed：`2026`；训练 seed：`42`。
- 当前只运行 outer fold 0。
- outer known classes：`[0, 2, 3, 4, 5, 6, 7, 8]`。
- outer proxy-unknown classes：`[1, 9]`。
- 训练 Known：`22,400`；校准 Known：`3,200`；测试 Known：`6,400`；测试 proxy Unknown：`1,600`。
- Known 校准集与 Known 测试集互斥；OSCR 使用拒识前类别预测；`STOP` 与公平 B2 输出严格一致。
- ORACLE 正式 Unknown 共 `24,000` 个样本，本轮未用于训练、选择、校准或评价。
- 单折总耗时：`1814.0897 s`。该结果只是 feasibility sanity，不能单独晋级 G1。

## 1. 本地能力闸门：通过

本轮取消每类 `512` 样本上限，使用全部 `22,400` 个 outer-train Known 样本预训练 30 epochs；本地预训练的 Unknown loss 为 `0`，Prototype SupCon 权重为 `0`。预训练完成后以 eval mode 在训练 Known 上独立检查：

| Agent | eval-mode train accuracy | 要求 | 结果 |
|---|---:|---:|---|
| Waveform | 0.999420 | 0.85 | 通过 |
| Prototype | 0.984152 | 0.85 | 通过 |

这项硬闸门只证明两个编码器具有合法的独立分类能力。它不使用正式 Unknown，也不证明 B2、选择器或通信有效。

训练历史也与能力恢复一致：第 30 个本地 epoch 的 batch-mode Known accuracy 为 Waveform `0.998527`、Prototype `0.985848`；随后 5 个 open-head epochs 中，主干精度保持在约 `0.999420 / 0.984107`。

## 2. 外层代理开放集指标

每个规则均独立使用 Known calibration 设定阈值，测试集为 `6,400 Known + 1,600 proxy Unknown`。

| 方法 | AUROC | AUPR-Out | FPR95 | Known Acc. | Unknown Recall | H-score | OSCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waveform | 0.920506 | 0.788604 | 0.400156 | 0.937969 | 0.642500 | 0.762615 | 0.901045 |
| Prototype | 0.922483 | 0.823885 | 0.413125 | 0.924219 | 0.718750 | **0.808636** | 0.895593 |
| 公平 B2 | 0.920506 | 0.788604 | 0.400156 | 0.937969 | 0.642500 | 0.762615 | 0.901045 |
| public selector（仅诊断） | 0.932446 | 0.809153 | 0.350313 | 0.931719 | 0.705625 | 0.803062 | 0.906593 |
| Agent oracle upper bound | — | — | — | 0.982969 | 0.803750 | 0.884371 | 0.983208 |

两个单 Agent 的必要能力检查均通过：Waveform/Prototype 的 Known Accuracy 分别为 `0.937969/0.924219`，AUROC 分别为 `0.920506/0.922483`。Prototype 是当前 H-score 最佳单 Agent；Waveform 的 OSCR 略高。

## 3. B2 锚点历史与失败原因

B2 训练的最终锚点审计为：

- `waveform_objective = 0.384135`
- `prototype_objective = 0.430733`
- `candidate_fit_objective = 0.380537`
- `anchor_fit_objective = 0.384135`
- 候选相对 Waveform 锚点的拟合目标改善仅 `0.003598`
- 要求的最小改善为 `0.005`
- `anchor_agent = waveform`
- `adaptive_selected = 0`

因此训练器按 no-regret 规则拒绝自适应候选并回退到 Waveform。B2 在外层测试上的全部指标与 Waveform 完全相同，这也验证了 `STOP == B2`，但同时说明 B2 没有利用 Prototype 的互补信息。

五个 B2 epochs 中，selector loss 从 `1.863544` 降至 `1.680706`，no-regret 项从 `0.056912` 降至 `0.056067`，Known 训练准确率始终约为 `0.999598`。这些训练目标的轻微改善没有转化成外层 H-score 改善：

- `B2 H - best local H = -0.046021`
- 闸门容许的最差差值为 `-0.005`

所以本轮 B2 明确未通过，而不是“接近通过”。不能通过放宽锚点选择条件把候选强行记为协作增益。

## 4. 互补性上限：存在，但尚未兑现

逐样本 Agent oracle 相对 B2 的上限为：

- `Delta H = +0.121756`（`0.762615 -> 0.884371`）
- `Delta OSCR = +0.082163`（`0.901045 -> 0.983208`）
- B2 样本级成功率 `0.878875`，oracle 成功率 `0.947125`，绝对 headroom 为 `0.068250`

Prototype 相对以 Waveform 为锚的 B2 产生 `546` 个 unique rescues：占全部测试样本 `6.825%`，占 B2 失败样本 `56.3467%`。Waveform 的 unique rescue 为 `0`，原因是当前 B2 本身就是 Waveform；这不表示 Waveform 没有能力，而表示以当前基线定义它不可能“救援自己”。

上述 oracle 需要事后知道每个样本上哪个 Agent 的决策正确，只是互补潜力上界，不是可部署方法成绩。它支持继续做 inner-level selector 学习，但不能用来宣称多智能体协作已经成功。

## 5. public selector 为什么仍只是诊断

当前 public selector 只读取两个 Agent 的公共摘要，并使用 5-fold cross-fitting 避免同一样本同时进入该折的训练与预测。它得到：

- action accuracy：`0.867000`
- mean loss：`0.113500`
- gain over STOP：`0.007625`
- oracle gain recovery：`0.111722`
- 相对 B2：`Delta H = +0.040446`、`Delta OSCR = +0.005548`

但是它仍不能部署：其训练标签由当前 outer 测试池中各 Agent 的真实正确/错误结果生成；交叉拟合的训练折仍包含 outer proxy-unknown 类 `[1, 9]` 的其他测试样本。换言之，它只做到“样本级 out-of-fold”，没有做到“类别级 outer holdout”。部署时也不存在这些 outer 测试标签来重新拟合选择器。

它相对最佳单 Agent 的 `Delta H = -0.005575`，而 G0 要求至少 `+0.01`，实际还差 `0.015575`。因此该结果只能回答“公共摘要是否含有一些可学习信号”，不能登记为公平 B2 或通信收益。

## 6. G0 检查结果

通过 7 项：

- 两个 Agent 的 Known Accuracy 均不低于 `0.85`；
- 两个 Agent 的 proxy-unknown AUROC 均不低于 `0.75`；
- oracle `Delta H >= 0.03`；
- oracle `Delta OSCR >= 0.015`；
- Prototype unique rescue 不低于 `5%`。

失败 3 项：

- B2 未保持在最佳单 Agent H-score 的 `0.005` 范围内；
- Waveform unique rescue 未达到 `5%`；
- public selector 相对最佳单 Agent未达到 `Delta H >= 0.01`。

所以 `sanity.passed = false`。当前正确表述是：**本地能力恢复成功，互补 headroom 充足，但可部署 G0 协同基线失败。**

## 7. 下一步：真正的 nested-inner 训练

不得继续使用 outer fold 0 的测试标签调 B2 或选择器。下一步按已冻结 manifest 完成 outer fold 0 内的四个 inner folds：

1. 每个 inner fold 只用 6 个 inner-known 类训练独立的 Waveform/Prototype；另外 2 个类只作为该 inner episode 的类别级 proxy Unknown。
2. 每折使用 `16,800` train-known、`2,400` calibration-known、`4,800` test-known，以及独立的 `800` validation-proxy-unknown / `1,600` test-proxy-unknown；不得把 outer 正式测试样本或正式 Unknown 放入训练调用链。
3. 汇总四折 out-of-fold 的公共摘要、局部决策和真实 action loss，训练共享的公共信息 B2/meta-selector；该决策器不得读取私有 embedding 或 outer proxy-unknown 标签。
4. 冻结 inner 学到的决策规则，再在 outer fold 0 的 `6,400 Known + 1,600 proxy Unknown` 上评价一次。
5. 只有 nested 结果同时保持 B2 公平性、超过最佳单 Agent并满足 G0 全部条件，才运行其余四个 outer folds；五折 G0 通过前不实现或运行 G1 通信。

## 状态声明

- 结果级别：Stage-7 V4 ORACLE、单 outer fold、开发期 G0 sanity。
- 正式 Unknown：**未评价**。
- G1 消息价值：**未运行**。
- G2 路由：**未运行**。
- 可声明结论：两个异构本地 Agent 已恢复独立开放集能力，并存在明显但尚未实现的互补空间。
- 不可声明结论：B2 已优于单 Agent、public selector 可部署、通信带来净增益、Stage-7 已通过 G0。

