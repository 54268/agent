# Stage-7 V3 ORACLE 单折 sanity 诊断

> 状态：**运行完整，但属于无效的欠训练 sanity，不是正式结果，也不得用于方法比较或晋级。**  
> 日期：2026-09-19。仅运行 ORACLE outer fold 0（`8 Known / 2 proxy Unknown`）；WiSig V3、G1/G2、正式 Unknown 均未运行。

## 表面结果（仅用于定位故障）

| 局部方法 | AUROC | OSCR | Known Acc | Proxy-Unknown Recall | H-score |
|---|---:|---:|---:|---:|---:|
| Waveform | 0.6107 | 0.1923 | 0.2706 | 0.0738 | 0.1159 |
| Prototype | 0.8716 | 0.6070 | 0.6536 | 0.5406 | 0.5918 |
| B2 / STOP | 0.8767 | 0.6100 | 0.6528 | 0.5744 | 0.6111 |

这些数值来自独立 Known calibration 与 outer proxy-Unknown，但训练预算不合法，故不能解释为 V3 架构的有效性能。公平 B2 也没有弥补两个局部 Agent 的能力缺口。

## 首要故障：sanity 截断把训练预算压到 18.2%

- V3 设置 `sanity_max_per_class=512`，8 个训练类只有 `4096` 个样本；batch size 为 256，因而每 epoch 仅 `16` 次更新，30 epochs 共 `480` 次更新。
- 可比的 Stage-5 ORACLE fold 使用全部 `22,400` 个训练样本，每 epoch `88` 次更新，30 epochs 共 `2640` 次更新。
- V3 只有 Stage-5 更新预算的 `480/2640=18.2%`。因此本次运行首先证明的是 sanity cap 不适用于本地骨干训练，而不是骨干无法学习 ORACLE。

## 移植核验证明 forward/架构具备能力

将同一 ORACLE fold 上已训练的 Stage-5 权重只读移植到对应 Stage-7 本地路径后：

- Identity logits 最大绝对差为 `0`；Prototype logits 最大绝对差约 `2.38e-6`；
- calibration Known 的 Waveform/Prototype 闭集准确率约为 `0.997/0.994`；
- outer Known test 的 Waveform/Prototype 闭集准确率约为 `0.995/0.994`。

这排除了“Stage-7 forward 本身不具备 ORACLE 判别能力”的解释，并把主要问题收敛到训练预算和训练目标。该核验只是诊断，不是 Stage-7 训练结果。

## Prototype 的 batch-dependent gate 捷径

V3 Prototype 的五视图 gate 在训练模式下约 `99.976%` 选择 `complex_iq`，形成由 BatchNorm/批组成驱动的单视图捷径。切到真正 eval-mode 后，训练集 Prototype 准确率约 `0.707`，outer test 约 `0.673`；这与训练历史末期显示的约 `0.991` batch-mode accuracy 明显矛盾。故训练日志中的高 Prototype accuracy 不能作为本地能力通过证据。

## 受控损失消融（同一截断数据，只作诊断）

15-epoch、无保存 ORACLE fold 消融的 eval-mode Known accuracy 如下：

| 变体 | epoch 10 W / P | epoch 15 W / P |
|---|---:|---:|
| A：V3 当前目标 | 0.1250 / 0.3363 | 0.1250 / 0.3534 |
| B：本地预训练 open loss=0 | 0.5541 / 0.5800 | 0.4025 / 0.2784 |
| C：B + Prototype SupCon=0 | **0.5541 / 0.6303** | 0.4025 / 0.4494 |
| D：B + classification weight=2 | 0.5038 / 0.4828 | **0.4778 / 0.2938** |

结论仅限于：Known-only 本地预训练不应同时承受 open BCE，Prototype SupCon 在当前批次/视图机制下有冲突；继续在截断数据上扫权重没有意义。原目标还对两个 Agent 的 `(CE+BCE)` 取均值，相对放大了度量辅助损失，V4 改为累加两个 Agent 的本地目标。

## V4 止损方案

V4 成对配置保持 WiSig/ORACLE 方法字段一致，并只做以下训练有效性修复：

1. `sanity_max_per_class=0`，ORACLE 使用该 fold 全部 `22,400` 个 Known 训练样本；
2. 本地预训练 `local_pretrain_open_loss_weight=0`，开放集头留到后续独立阶段；
3. `prototype_supcon_weight=0`，保留可审计的分类、视图和原型约束；
4. 两个 Agent 的本地目标相加，不再取均值；
5. 在 enrollment、PUG 和 B2 之前执行硬能力门：eval-mode 训练 Known accuracy 必须对 Waveform 和 Prototype **分别达到 0.85**；否则保存显式 incomplete checkpoint 并立即停止；
6. 先运行 ORACLE V4。若本地能力门失败，不运行 WiSig，不开发 inner 系统、通信器或路由器；只有 ORACLE 完整通过后才运行成对 WiSig sanity。

即使 V4 单折通过，也只能进入注册 G0/nested inner 系统验证；在 G0 完整通过前，仍禁止 G1/G2、正式 Unknown 和多种子长实验。

