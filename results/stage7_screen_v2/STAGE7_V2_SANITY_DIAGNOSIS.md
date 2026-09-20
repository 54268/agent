# Stage-7 V2 单折 G0 sanity 诊断

> 状态：**未通过、不得晋级**。日期：2026-09-19。  
> WiSig 与 ORACLE 各运行一个 outer-LCO fold；没有训练 G1/G2，也没有读取两份数据集的正式 Unknown。

## 协议边界

- WiSig：outer fold 0，`32 Known / 8 proxy Unknown`；
- ORACLE：outer fold 0，`8 Known / 2 proxy Unknown`；
- Agent、损失、训练轮次、动作空间和校准规则完全相同，差异仅为数据路径、类别和输出目录；
- 阈值只由独立 Known calibration 估计，Known test 与 calibration 样本互斥；
- OSCR 使用拒识前类别预测，`STOP` 与公平 B2 输出严格一致；
- 运行期间未使用正式 Unknown 训练、选模、调参、阈值或评价。

## 单折结果

| 数据集 | 局部方法 | AUROC | OSCR | Known Acc | Proxy-Unknown Recall | H-score |
|---|---|---:|---:|---:|---:|---:|
| WiSig | Waveform | 0.9392 | 0.6848 | 0.6967 | 0.7962 | 0.7431 |
| WiSig | Prototype | 0.9089 | 0.8831 | 0.9117 | 0.4158 | 0.5711 |
| WiSig | B2 / STOP | 0.9089 | 0.8831 | 0.9117 | 0.4158 | 0.5711 |
| ORACLE | Waveform | 0.5580 | 0.0827 | 0.1248 | 0.0450 | 0.0662 |
| ORACLE | Prototype | 0.4916 | 0.2811 | 0.5153 | 0.0444 | 0.0817 |
| ORACLE | B2 / STOP | 0.4916 | 0.2811 | 0.5153 | 0.0444 | 0.0817 |

两份 sanity 均失败。G1 消息价值、G2 路由、五折注册 G0 和正式 Unknown 均未运行。

## V2 得到的有效结论

1. **不可覆盖的开放集锚有效。** WiSig Waveform/Prototype 的代理 AUROC 分别从 V1 的
   `0.8137/0.6288` 提升到 `0.9392/0.9089`；学习式拒识头不再覆盖 Known-only 的可靠性、Energy、
   竞争间隔和经验尾部证据。
2. **Prototype 在 WiSig 已恢复独立能力。** Known Accuracy 为 `0.9117`，但固定 95% Known 接受率下
   代理 Unknown Recall 仍只有 `0.4158`，说明仅靠 PUG 拟合 B2 不能替代真实 inner 留类元训练。
3. **公平 B2 的 no-regret 回退生效。** 两个数据集都因自适应融合未在合法拟合目标上超过最强局部
   Agent 而精确回退到 Prototype；因此没有把一个弱中央头包装成多 Agent 增益。
4. **WiSig 仍有真实互补空间。** 逐样本 Agent-oracle 达到 H=`0.8974`、OSCR=`0.9271`；仅使用公共
   摘要的外层交叉拟合诊断上限达到 H=`0.8590`。后者只作可学习性上限，不能作为可部署结果。
5. **ORACLE 的本地骨干结构失败。** Waveform 的训练准确率从第 1 到第 30 epoch 始终约为随机水平
   `0.125`；Prototype 训练/评估准确率约 `0.864`，但 outer Known test 仅 `0.5153`。

## 已定位的结构原因与 V3 约束

- 同一 ORACLE outer fold 上，Stage-5 已验证的 raw-IQ Identity 与五传感 Universal Geometry checkpoint
  在相同 Stage-7 Known test 上分别达到约 `0.9948/0.9937` 闭集准确率，证明失败不是数据不可学。
- V2 Waveform 将稳定 raw-IQ 判别路径与不稳定 complex 分支直接做样本级状态混合，导致 ORACLE
  表征塌缩，并使 WiSig 训练准确率也只到约 `0.674`。
- V2 Prototype 使用 `spectral / fft_iq / envelope_phase / difference_iq`，删除了 Stage-5 在 ORACLE
  自动赋予最高权重的 `raw` 和 `complex_iq` 视图；这破坏了跨数据集鲁棒性。
- V3 必须使用同一套数据集无关架构：Waveform 以成熟 raw-IQ 判别骨干保证本地决策，complex temporal
  token 专供条件质询或受控残差；Prototype 恢复固定五传感注册原型库，由样本级 gate 自主选择视图。
- 单折本地必要条件通过之前，仍禁止通信器、路由器、正式 Unknown 和长时间 nested 训练。

## nested 协议修正

outer 模型已经见过所有 outer-support 类，不能把 inner 留类样本直接喂给它并称为 episode-unknown。
正式五折前必须训练四个真正排除对应 held-class 的 inner 系统，缓存其局部决策，再在匹配的 inner
标签空间内训练共享 B2/响应器/路由器。当前外层交叉拟合 public selector 仅是诊断上限，不参与部署。
