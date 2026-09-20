# Stage-7 V1 单折 G0 sanity 诊断

> 状态：**未通过、不得晋级**。日期：2026-09-19。  
> 这两次运行只是 WiSig/ORACLE 各一个 outer-LCO fold 的可行性检查，既不是五折 G0，也不是正式 Unknown 结果。

## 协议边界

- WiSig：outer fold 0，`32 Known / 8 proxy Unknown`；
- ORACLE：outer fold 0，`8 Known / 2 proxy Unknown`；
- 阈值只使用独立 Known calibration；Known test 与 calibration 样本互斥；
- 数据集正式 Unknown 未用于训练、校准、选择或评价；
- OSCR 使用拒识前类别预测；
- `STOP` 与公平 B2 输出严格一致。

首次评价时发现 `collect_predictions` 将 `ProvenanceSubset` 错误解包为完整底层 split。该调用在开放集指标检查处失败，没有生成指标；修复并增加回归测试后，直接复用先前正确按支持类训练的 checkpoint 重新评价。下表均为修复后的有效结果。

## 结果

| 数据集 | 局部方法 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |
|---|---|---:|---:|---:|---:|---:|
| WiSig | Waveform | 0.8137 | 0.7582 | 0.8950 | 0.7174 | 0.7964 |
| WiSig | Prototype | 0.6288 | 0.5804 | 0.8631 | 0.3152 | 0.4618 |
| WiSig | B2 / STOP | 0.8642 | 0.8047 | 0.8648 | 0.3478 | 0.4961 |
| ORACLE | Waveform | 0.5272 | 0.4315 | 0.7578 | 0.1988 | 0.3149 |
| ORACLE | Prototype | 0.5503 | 0.2603 | 0.4389 | 0.0638 | 0.1113 |
| ORACLE | B2 / STOP | 0.5713 | 0.4621 | 0.7578 | 0.0781 | 0.1416 |

两份数据集的 sanity 均失败，因此没有运行 G1 消息价值、G2 路由或正式 Unknown。

## 可操作诊断

1. **开放集锚证据被弱学习头覆盖。** 固定公共证据审计显示，WiSig Waveform 的 `1-reliability` 经相同 Known-only 分类别校准后 AUROC 为 `0.9435`，Prototype 的负 margin 为 `0.8881`；ORACLE Waveform 的 Energy 为 `0.8613`。均明显高于 V1 学习式 open head。
2. **Prototype 私有观察跨数据集不稳。** 仅用 spectral/envelope 两视角，在 ORACLE 的闭集测试准确率约 `46.13%`，不能承担独立 Agent 角色。
3. **B2 过拟合训练期最强局部 Agent。** WiSig B2 的闭集表现接近 Prototype，而没有保留测试上更强的 Waveform；未知头同样被 PUG 目标带偏。
4. **仍存在真实互补上限。** WiSig 逐样本 oracle 为 H=`0.8698`、OSCR=`0.9288`；ORACLE 为 H=`0.4638`、OSCR=`0.8706`。这只说明角色重做有空间，不等同于通信收益。

## V2 只允许解决的三件事

- 使用同一跨数据集强分类/度量骨干，Prototype 恢复多视角注册记忆和度量约束；
- 未知风险以 Energy、margin、经验原型尾部为不可绕过的锚，PUG 只训练有界残差；
- 公平 B2 使用公共摘要的局部损失蒸馏与 no-regret 约束，不能再以一个自由中央未知头替代两个本地决策。

在 V2 再次通过两数据集单折 sanity 之前，继续禁止通信超参扫描和正式 Unknown 测试。
