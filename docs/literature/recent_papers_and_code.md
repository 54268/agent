# 近年论文与开源代码：MAROS-SEI 设计映射

更新日期：2026-09-15。这里区分“可做同协议基线”“只借鉴机制”和“尚需许可证/实现核查”，避免把相关工作误写成直接可复现代码。

## 1. 多智能体通信、路由与多样性

| 工作 | 发表 | 可借鉴点 | 代码/页面 | 在本项目中的角色 |
|---|---|---|---|---|
| CommFormer: Learning Multi-Agent Communication from Graph Modeling Perspective | ICLR 2024 | 把通信拓扑作为可学习图，连续松弛并联合优化 | [论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/37c6d0bc4d2917dcbea693b18504bd87-Abstract-Conference.html) / [代码](https://github.com/charleshsc/CommFormer) | Router 的有向稀疏图；不照搬其 RL 训练任务 |
| How2comm | NeurIPS 2023 | 互信息感知消息、通信效率与协作融合 | [论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/4f31327e046913c7238d5b671f5d820e-Paper-Conference.pdf) | 消息价值与通信预算损失 |
| DiCo: Controlling Behavioral Diversity in MARL | ICML 2024 | 显式控制 Agent 多样性而不是期待自然分工 | [代码](https://github.com/proroklab/ControllingBehavioralDiversity) | 专家多样性/塌缩诊断；任务不同，仅借鉴思想 |
| MASIA | NeurIPS 2022 | 自监督信息聚合与高效通信 | [代码](https://github.com/chenf-ai/MASIA) | 消息表征的自监督一致性参考 |
| ADMAC | AAAI 2024 | 关注通信消息可靠性 | [论文](https://doi.org/10.1609/aaai.v38i16.29708) | 每条 Agent 消息附带 reliability |
| SchedNet | ICLR 2019 | 学习何时通信和带宽受限调度 | [代码](https://github.com/rhoowd/sched_net) | hard top-k/预算门控基线；虽较早但本地已有代码 |

结论：CommFormer 支撑“谁向谁通信”，How2comm/SchedNet 支撑“是否值得通信”，ADMAC 支撑“消息是否可信”，DiCo 支撑“Agent 是否真正分工”。本项目使用监督式 OSR 目标，不应机械套用 PPO、QMIX 或其他环境回报优化。

## 2. 通用开放集与多专家方法

| 工作 | 发表 | 可借鉴点 | 代码/页面 | 使用边界 |
|---|---|---|---|---|
| MEDAF: Exploring Diverse Representations for Open Set Recognition | AAAI 2024 | 多专家互补注意力、diversity regularization、adaptive gate | [论文](https://ojs.aaai.org/index.php/AAAI/article/download/28385/28753) / [代码](https://github.com/Vanixxz/MEDAF) | 必做强基线；仅有 gating 不等于本项目的 Agent 通信 |
| Meta Evidential Transformer | ICML 2024 | evidential loss、evidence-to-variance、困难未知识别 | [论文](https://proceedings.mlr.press/v235/sapkota24a.html) | Open-Set Agent 的证据建模参考 |
| OpenOOD v1.5 | DMLR 2024 | 统一 OOD 协议、post-hoc/training-based 方法和完整指标 | [代码](https://github.com/Jingkang50/OpenOOD) / [论文](https://arxiv.org/abs/2306.09301) | 借鉴评估器与基线接口，不照搬图像预处理 |
| ARPL | CVPR 2021 | reciprocal points 与开放空间约束 | [论文](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Adversarial_Reciprocal_Points_Learning_for_Open_Set_Recognition_CVPR_2021_paper.html) | 公共 OSR 训练式基线 |

## 3. 开放集与跨域辐射源识别

| 工作 | 发表 | 可借鉴点 | 代码/页面 | 使用方式 |
|---|---|---|---|---|
| OpenRFI | AAAI 2025 | Roinformer、信号增强、实例相似度、局部熵、开放世界半监督 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/32003) / [代码](https://github.com/ShuaS2020/OpenRFI) | 额外未标注流轨道；不可与纯监督 OSR 不加说明地比较 |
| Dynamic Boundary Adversarial Model for Open-Set Radar SEI | IEEE TAES 2025 | 类原型的可训练闭边界和动态紧致约束 | [论文](https://doi.org/10.1109/TAES.2025.3581888) | Open Agent 边界基线/设计参考，需核查公开实现 |
| Open-Set Few-Shot Class Incremental SEI | IEEE TCCN 2025 | 持续原型、原型校准、距离拒识与未知聚类 | [论文](https://doi.org/10.1109/TCCN.2025.3565589) | few-shot/增量扩展，不作为首版主任务 |
| DM-MML | IEEE SPL 2026 | 动态 margin 元度量与距离/置信度协同判决 | [论文](https://doi.org/10.1109/LSP.2026.3675912) | few-shot 压力实验参考 |
| Receiver-Agnostic RFFI via Domain-Invariant Feature Learning | IEEE Communications Letters 2025 | 相位教师、跨域不变特征与统计对齐 | [代码](https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL-) | Domain Agent/跨 Rx 强基线；先核查非商业许可证 |
| IB-RFF | IEEE TIFS 2026 | Frequency-Aware Network、IB/HSIC、无 Rx 标签跨域学习 | [代码](https://github.com/BeechburgPieStar/IB-RFF) | Domain Agent 的 HSIC 约束；MIT 代码可做 backbone 基线 |
| ASKNet | IEEE Communications Letters，作者仓库标注已接收 | 频谱门控和 Koopman 相位校正 | [代码](https://github.com/BeechburgPieStar/ASKNet-RFF) | Frequency/Phase Agent 候选模块，正式引用前核对最终出版信息 |
| WiSig | IEEE Access 2022 | 174 Tx、41 Rx、4 日期的大规模跨接收机/跨信道数据 | [数据页](https://cores.ee.ucla.edu/downloads/datasets/wisig/) / [论文](https://arxiv.org/abs/2112.15363) / [示例](https://github.com/WiSig-dataset/wisig-examples) | 本项目唯一数据源 |

## 4. 对首版方法的直接映射

| MAROS-SEI 组件 | 主要研究依据 | 首版实现约束 |
|---|---|---|
| Time Agent | 本地 CVCNN/MFE-Net 的基础指纹表征 | 独立 logits、prototype distance、energy |
| Frequency/Phase Agent | IB-RFF、ASKNet、HyDRA | 与 Time Agent 使用不同输入视图并加入多样性约束 |
| Domain Agent | Receiver-Agnostic RFFI、IB-RFF | 输出 domain shift/reliability；指纹表征用 GRL/HSIC 去域化 |
| Open Agent | MET、MEDAF、动态边界、PCBM | 只用 Known validation 与伪未知训练/校准 |
| Communication Router | CommFormer、How2comm、ADMAC、SchedNet | 输出有向稀疏图和融合信任权；进入真实推理公式 |
| Agent specialization | DiCo、MEDAF | 防止所有 Agent 学成同一特征，监测表征相似度和路由塌缩 |

## 5. 创新边界

以下单独使用都不能作为核心创新：多视图输入、多专家、注意力加权、Prototype、OpenMax/EVT、动态阈值、HSIC 去域或频相分支。

当前可检验的创新假设是：**在同一个 WiSig 包上，由具有私有观察和局部目标的异构 Agent，通过面向开放集风险和域漂移的样本级可靠性稀疏通信图进行协作，并用反事实边际贡献约束 Router，使通信本身改善未知拒识与跨域已知分类。**

这仍是待实验和更完整 prior-art 检索验证的研究假设，现阶段不宣称“首次”。

