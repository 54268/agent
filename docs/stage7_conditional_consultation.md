# Stage-7 条件咨询式多智能体 OS-SEI

## 阶段定位

Stage-7 不继续对 Stage-6 的通用 latent message 和 query-cost 做局部调参。当前目标是在冻结的强开放集骨干上，分别验证“本地互补性”、“条件消息价值”和“按需调度价值”。任一前置闸门失败，都不开启后续长训练或正式 Unknown 评价。

Stage-6 代码保持只读；V2–V7 的未完整单 fold 运行只作诊断记录，不视为五折 LCO 或正式结果。

## 当前实际状态（2026-09-20）

一个 outer fold 的严格 nested G0 已在 WiSig 与 ORACLE 完成。两个 Agent 的本地能力、双向 unique rescue 和 Agent-oracle headroom 均通过，但没有一个公平 B2/安全选择候选能在两个数据集同时稳定胜过最佳单 Agent。WiSig 的最佳公共 B2 候选达到 `H=0.64632、ΔH=+0.02997、3/4`，却在 ORACLE 变为 `H=0.57459、ΔH=-0.10311`；ORACLE 唯一通过的受限锚点候选在 WiSig 只有 `2/4` folds 正增益。

因此当前 **G0 未通过，G1/G2/G3/formal 均未解锁**。完整诊断、无效候选边界和 competence V5–V7 止损结果见 [`stage7_nested_g0_diagnosis.md`](stage7_nested_g0_diagnosis.md)。

## 角色与通信

- **Waveform Identity Agent** 使用原始 complex I/Q，独立生成身份 logits、未知风险和 top-2 提案。收到 Prototype 提案后，它对指定类别对重新计算时域证据。
- **Enrollment/Prototype Agent** 使用频域表示及类原型记忆，可独立分类和拒识。收到 Identity 提案后，它仅检验提案类与竞争类。
- 路由动作只有 `STOP / W_FIRST / P_FIRST`。`STOP` 必须与公平 B2 的同一次计算逐位相同；通信仅进行一轮。
- 消息仅含类别提案、立场、成对证据、开尾风险、可靠性与 bit cost，不允许传输完整 state、embedding 或全 logits。
- 裁决器、OpenMax/EVT、PCBM、CDF 校准、阈值和 PUG 是服务，不计为 Agent。

## 数据隔离与训练顺序

1. 外层五折 LCO 将训练 Known 划分为 support 类和代理 Unknown 类。
2. 每个外层 support 集再做四折 inner LCO，路由监督来自真实类别级 episode-unknown，不以 PUG 代替。
3. Known 的 train/calibration/test 样本隔离；正式 Unknown 不能进入训练、阈值、选轮、调参或停止条件。
4. 先独立训练本地 Agent，然后冻结它们训练条件响应头。
5. 对 inner 样本精确遍历三个动作，用实际的分类损失、开放集损失与通信代价产生路由标签。
6. G0、G1、G2 全部通过后，才允许解冻最后一个特征块微调。

WiSig 与 ORACLE 的方法字段由入口强制相等，只允许数据路径、数据集名、输出目录和对比基线不同。

### Inner episode 的模型边界

Inner LCO 不能只把 inner 数据子集喂给已经看过全部 outer-support 类的 outer 模型。这样被留出的类并非模型意义上的 Unknown，而且 inner 压缩标签与 outer logits 的类别下标也不一致。每个 inner fold 必须拥有按该 fold 的 support 类从头训练的本地 Agent；其 `train_proxy_unknown` 才能作为真正的类别级代理 Unknown。

为避免在本地互补性不足时浪费四倍训练量，执行分为两层：

1. 单 fold sanity 以及 G0 的本地能力、unique rescue、Agent-oracle headroom 可以先用 outer 模型检查；这一层不能训练或宣称可部署的 selector。
2. 上述必要条件通过后，才为每个 outer fold 训练四个 inner 模型。B2、条件响应和 Router 的共享元参数只用配对的 inner Known/代理 Unknown 学习，然后冻结并迁移到 outer 模型；outer proxy Unknown 只用于最终 fold 评价。

B2、裁决器和 Router 的元参数必须与类别数无关。Waveform 的条件响应同样不得依赖 `num_classes` 大小的可学习类别表；类别对查询应由当前模型自己的注册原型、分类器权重或成对标量证据构造，使 inner 学到的响应规则可以迁移到 outer 类别空间。阈值仍只由 outer calibration-Known 估计。

用于 G0 的 public selector 若在 outer 测试行上做 sample-wise cross-fit，只能标注为诊断上界，不能作为可部署结果。正式 selector 应在 inner episode 的类互斥元数据上交叉拟合，冻结后再一次性作用于 outer fold。所有 inner checkpoint 必须保存原始类别集合、压缩映射和数据来源；加载时要求模型类别集合与 episode 完全一致。

## 闸门与执行

- **G0** 验证两个本地 Agent 的基本能力、unique rescue 和 action-oracle headroom。
- **G1** 冻结相同 Agent 比较 B2、等带宽并行通信和强制顺序通信。强制通信不能胜过 B2 时禁止训练 Router。
- **G2** 验证按需路由的 oracle-gain recovery、预测/实际收益相关性、动作覆盖和 rescue/harm ratio。
- **G3** 先执行 `3 partition seeds × 5 folds`，通过后才运行 seeds `42/43/44` 的正式 Unknown 评价。

建议命令：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage7.py --config configs/experiments/stage7_screen_v4_wisig_k40u20.json --phase nested_sanity --fold 0
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage7.py --config configs/experiments/stage7_screen_v4_oracle_k10u6.json --phase nested_sanity --fold 0
```

进入后续阶段时，入口必须读取并检查前置闸门产物，不提供强制跳过选项。每次运行至少保存冻结配置、nested protocol manifest、本地决策、语义消息、逐动作反事实损失、最终分数、通信代价和 gate report。

## 可选第三 Agent

只有两 Agent 通过 G1 且仍有明显 oracle headroom 时，才屏蔽当前配置中的 `enable_active_consistency_auditor=false` 并单独做探针安全性筛选。未通过已知样本 98% 一致性闸门的探针不得进入模型。
