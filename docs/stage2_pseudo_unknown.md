# Stage-2：Disagreement-Guided Pseudo-Unknown + Open-Space Boundary Agent

## 目标

在不使用任何真实 Unknown 的条件下，为 Stage-1 的 Identity、Prototype、Reconstruction 三路证据增加
一个可单独评价的开空间边界证据。该阶段暂不引入 Communication，确保性能变化可归因于伪未知与边界学习。

## Pseudo-Unknown Agent

该 Agent 的观察包括：

- 三个基础 Agent 的冻结嵌入；
- 样本相对联合类原型的位置和竞争间隔；
- 三 Agent 最近类意见是否冲突；
- Identity 不置信、Prototype 距离和 Reconstruction 误差。

它对每类 Known 按“类内边缘性 + 原型竞争 + Agent 分歧 + 异常证据”选择前 20% 种子，在三 Agent
联合潜空间中沿“远离本类原型 + 排斥最近竞争类”方向行动。每个 Agent 的嵌入块随后重新单位化，避免模型
通过向量范数识别生成痕迹；仅使用落在 Known 最近联合原型距离 90%–99.9% 分位边界壳内的样本。

伪未知按源 Known 样本分组拆分训练/验证，同一源样本的多个变体不会跨集合。

## Boundary Agent

输入由三路单位化嵌入、各 Agent 的最近/次近原型距离、距离比、原型分布熵和三组意见冲突组成。
网络输出：

- `unknown_logit`：开空间边界证据；
- `class_logits`：Known 类辅助任务，防止边界网络只学习生成伪影。

训练使用等量 Known 与伪未知；选轮只看 Known validation 与留出伪未知。最终阈值只由 Known validation
Boundary 分数的经验 CDF 决定，固定为 `τ=.95`。真实 Unknown 只在模型、阈值和融合规则冻结后离线评价。

## WiSig 三随机种子结果

| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
每一种完整融合规则均再次用 Known validation CDF 校准；`τ=.95` 表示约 95% Known 接受率，避免
均值和 max/OR 因分布不同而处在不等价工作点。

| Stage-1 三证据均值 | 0.922±0.004 | 0.867±0.004 | 0.893±0.005 | 0.535±0.026 | 0.669±0.019 |
| Stage-1 三证据 max/OR | 0.907±0.009 | 0.866±0.001 | 0.901±0.003 | 0.348±0.094 | 0.494±0.099 |
| Boundary Agent | 0.855±0.020 | 0.798±0.020 | 0.882±0.002 | 0.559±0.052 | 0.683±0.038 |
| 四证据均值 | **0.929±0.006** | **0.873±0.006** | 0.886±0.003 | **0.632±0.044** | **0.737±0.029** |
| 四证据 max/OR | 0.914±0.002 | 0.866±0.005 | **0.892±0.004** | 0.498±0.062 | 0.637±0.051 |

## 判定

Stage-2 有效：在统一 Known 接受率后，四证据均值同时提高 AUROC、Unknown Recall 和 H-score。
但留出伪未知 AUROC 约 0.999、真实 Boundary AUROC 约 0.855，生成—真实未知差距仍明显，因此该阶段
不是最终方案。下一阶段加入 Communication，使三个基础 Agent 在样本级交换信息，并必须报告：

1. 关闭全部消息；
2. 打乱发送者消息；
3. 只保留通信但去掉 Boundary；
4. 完整 Communication + Boundary；
5. 每条边的消息门、预测翻转率和性能变化。

入口：`scripts/experiments/run_stage2_pseudo.py`；多种子入口：
`scripts/experiments/run_stage2_multiseed.py`；配置：`configs/experiments/stage2_pseudo_wisig.json`。
