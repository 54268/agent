# 多智能体开放集辐射源识别：第一阶段审计与方案冻结

> 2026-09-15 更新：28 Tx 均衡集现降级为调试协议；论文主实验已经扩展为 WiSig 全部 174 Tx。当前完整实验定义见 `docs/design/experiment_plan_wisig_only.md`，机器可读定义见 `configs/protocols/wisig_only_experiment_matrix.json`。

更新日期：2026-09-15  
阶段状态：**仅完成数据、参考实现、文献与方法设计审计；尚未开始主模型编码。**

## 1. 结论先行

本项目适合继续做，但首版不应直接套用传统多智能体强化学习（MARL）。当前任务是单包监督识别，并不存在自然的时序状态、环境动作和长期回报；强行使用 PPO、QMIX 或 MAPPO，容易被认为只是给分类器增加不必要的 RL 外壳。

建议采用暂名 **MAROS-SEI（Multi-Agent Routed Open-Set SEI）** 的“异构感知 Agent + 显式消息通信 + 样本级动态路由 + 协同证据拒识”框架。它和普通集成／MoE 的区别必须由以下四点保证：

1. 每个 Agent 有不同的局部观测、独立状态、独立预测和专属辅助目标；
2. Agent 先独立形成局部信念，再通过可记录的消息进行一轮协同更新；
3. Router 为每个样本生成不同的通信图和融合权重，权重直接进入最终已知类 logits 与未知分数；
4. 保存通信前后各 Agent 的输出，并通过删除、打乱、固定路由等反事实消融证明协同不是装饰。

WiSig 的价值不在于复现旧 `SingleDay + 固定 Rx` 高分，而在于利用完整的 4 天和多接收机结构同时考验开放集拒识与域泛化。已有 PCBM 在受控单日固定接收机协议上达到 Known Accuracy 0.9836、Unknown Recall 1.0、Macro-F1 0.9920、OSCR 0.9971；这个协议已经接近饱和，不能有效证明多智能体的必要性。主实验必须升级为跨日期、跨接收机协议。

## 2. 路径与只读边界

- 当前真实 WiSig 根目录：`D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\data\WiSig`
- 说明文件中的旧 WiSig 路径 `D:\learn_pytorch\笔记\多智能体\code\data\WiSig` 不存在于当前项目结构，后续以当前真实路径为准。
- 参考目录 1：`E:\Project`，只读。
- 参考目录 2：`D:\learn_pytorch\笔记\方案\os_sei_code`，只读。
- 本地多智能体参考：当前项目的 `多智能体GitHub`，只作思想和接口参考。

## 3. WiSig 数据审计

### 3.1 数据身份与格式

这批文件是官方 Full WiSig 的**处理后、非均衡（non-equalized）识别信号**，不是旧代码使用的 compact `SingleDay.pkl`。官方说明表明 WiSig 包含 174 个 WiFi Tx、41 个 USRP Rx、4 次跨月采集，重点就是研究接收机和日期变化造成的指纹偏移。[WiSig 官方页面](https://cores.ee.ucla.edu/downloads/datasets/wisig/)、[数据集论文](https://arxiv.org/abs/2112.15363)、[官方示例代码](https://github.com/WiSig-dataset/wisig-examples)

本地每个 `dataset_YYYY_MM_DD_nodeRX.pkl` 对应一个日期和一个接收机，pickle 为：

```text
{
  "node_list": [tx_id_0, tx_id_1, ...],
  "data": [array(N_0, 256, 2), array(N_1, 256, 2), ...]
}
```

最后一维依次为 I/Q。所有已扫描数据块均为 `float64`、尾部形状均为 `(256, 2)`；训练缓存应转为 `float32 [N,2,256]`，但不能改写原始 pickle。

### 3.2 全量只读扫描结果

| 项目 | 结果 |
|---|---:|
| 物理 pickle 文件 | 158 |
| 按文件名去重后的有效 pickle | 152 |
| 重复副本 | 6 |
| 有效 pickle 体积 | 36.484 GiB |
| 重复 pickle 体积 | 1.629 GiB |
| 保留的 zip 体积 | 9.048 GiB |
| 数据目录总占用 | 47.161 GiB |
| Tx 并集 | 174 |
| Rx 并集 | 41 |
| 日期 | 4 |
| 有效识别包总数 | 9,563,785 |
| 最少／中位／最多 Tx 包数 | 2,026 / 32,242 / 201,117 |
| Tx 的非空 `(Rx,day)` 覆盖最少／中位／最多 | 43 / 137 / 151 |
| 载入或结构错误 | 0 |

按日期统计：

| 日期 | Rx 文件数 | 非空 Tx 数 | 包数 |
|---|---:|---:|---:|
| 2021-03-01 | 32 | 174 | 1,937,019 |
| 2021-03-08 | 40 | 174 | 2,478,993 |
| 2021-03-15 | 40 | 174 | 2,501,683 |
| 2021-03-23 | 40 | 174 | 2,646,090 |

41 个 Rx 中有 30 个在四天均出现。日期间 Rx 数不同是数据可用性差异，不能用“每一天都必须有 41 个文件”作为完整性标准。

### 3.3 重复与压缩包处理

6 组重复文件均位于 2021-03-01，重复副本不但同名同尺寸，而且 SHA-256 完全一致：`node13-13`、`node18-2`、`node19-19`、`node19-20`、`node24-5`、`node8-8`。

后续 manifest 构建规则必须是：

1. 忽略 `.zip`，只索引已解压 `.pkl`；
2. 以 `(date, rx)`／规范文件名为主键；
3. 同主键多份文件时校验 size + SHA-256，只保留词典序最小路径；
4. 若同主键哈希冲突，立即失败，不能静默任选；
5. 不删除重复文件或 zip，原始数据保持只读。

### 3.4 旧 WiSig 预处理代码不能直接复用

参考目录 2 的 `functions/data/prep_wisig.py` 面向已经打包好的 compact 字典，预期顶层同时存在 `tx_list/rx_list/capture_date_list/equalized_list/data`。当前 Full WiSig 单文件只有 `node_list/data`，因此直接传入会失败或错误解释维度。

另外，参考实现把多个域拼接后按样本随机分层切分，会把同一日期和接收机域同时放进训练、验证、测试。对于本项目，这会导致明显的域泄漏。必须先生成带 `tx_id/rx_id/date/packet_index/source_file` 的 manifest，再按 Tx、Rx、日期组级划分。

### 3.5 默认归一化

首选官方示例采用的**每包复信号 RMS 功率归一化**：

\[
x' = x / \sqrt{\operatorname{mean}_t(I_t^2+Q_t^2)+\epsilon}.
\]

参考 PCBM 的 I、Q 各通道独立 z-score 可保留为消融，但不建议作为唯一默认，因为它会改变 I/Q 相对尺度与复平面几何，也可能擦除部分幅度硬件指纹。

### 3.6 可用于 D28 调试的 28 Tx × 12 Rx 均衡核心集

对四天共同 Rx 做确定性贪心筛选，可以找到 28 个 Tx 和 12 个 Rx，使每个 `(Tx,Rx,day)` 至少有 400 包。每格固定抽取 400 包后，共有：

\[
28\times12\times4\times400=537,600
\]

个均衡样本，float32 IQ 约 1.03 GiB，适合首版训练。

候选 Rx：

```text
1-1, 1-19, 18-2, 19-19, 19-2, 19-20,
2-19, 20-1, 20-19, 3-19, 7-7, 8-8
```

候选 Tx：

```text
20-15, 14-7, 14-10, 8-20, 6-15, 8-3, 8-18,
16-16, 20-12, 10-11, 7-14, 15-1, 20-7, 7-11,
11-7, 3-18, 1-18, 16-1, 4-1, 1-16, 5-5,
17-11, 7-10, 10-7, 11-1, 3-13, 13-3, 17-10
```

该列表是当前扫描结果，不应只写进 Python 常量；下一阶段要连同筛选参数、源文件哈希和随机种子写进版本化 split JSON。

## 4. D28 调试协议（已被全量主协议取代）

本节保留第一轮审计时冻结的小规模协议，用于快速回归和排错。它不再承担论文主结论；主协议以 `docs/design/experiment_plan_wisig_only.md` 中的 F174/C152 为准。

### 4.1 D28 类别划分

在上述 28 个 Tx 集合上按字符串排序后使用 NumPy `default_rng(42)` 打乱，冻结为：

```text
Known (16):
14-10, 20-7, 7-11, 7-10, 3-13, 11-7, 8-3, 7-14,
16-1, 3-18, 4-1, 15-1, 13-3, 5-5, 10-7, 1-16

Unknown (12):
20-15, 6-15, 17-10, 16-16, 20-12, 8-20,
10-11, 11-1, 8-18, 1-18, 17-11, 14-7
```

真实 Unknown 不进入编码器训练、Open-Set Agent 训练、Router 训练、阈值选择或早停，只在冻结后测试。

### 4.2 D28 Rx 划分

同样按字符串排序后以独立的 `default_rng(42)` 打乱：

```text
Train Rx (8): 1-1, 20-1, 2-19, 3-19, 8-8, 19-19, 19-20, 18-2
Val Rx   (2): 19-2, 7-7
Test Rx  (2): 1-19, 20-19
```

### 4.3 四层实验协议

| 协议 | 目的 | 组级划分 |
|---|---|---|
| P0 IID-OSR | 冒烟与旧方法对齐 | 固定 `day=03-01, Rx=1-1`，仅 Known 内按 packet 切 train/val/test；Unknown 只测试 |
| P1 Cross-Day OSR | 单独测时间漂移 | 固定同一组 Rx；train=`03-01,03-08`，val=`03-15`，test=`03-23` |
| P2 Cross-Rx OSR | 单独测接收机漂移 | 固定同一天；Rx 按 8/2/2 组级切分 |
| P3 Joint Cross-Day/Cross-Rx OSR | **论文主协议** | train=8 Rx × 前两天；val=2 Rx × 第三天；test=2 Rx × 第四天 |

P3-D28 在每格抽 400 包时的调试拆分规模：

| Split | 类别 | 样本数 |
|---|---|---:|
| train_known | 16 Known × 8 Rx × 2 day × 400 | 102,400 |
| val_known | 16 Known × 2 Rx × 1 day × 400 | 12,800 |
| test_known | 16 Known × 2 Rx × 1 day × 400 | 12,800 |
| test_unknown | 12 Unknown × 2 Rx × 1 day × 400 | 9,600 |

P0 只能作 sanity check，不能作为最终创新结论。P3 才同时逼迫模型解决“未知 Tx”和“已知 Tx 的域偏移”之间的混淆；最终主结果应使用新增的 P3-F174（120 Known + 54 Unknown），而不是本节的 D28。

### 4.4 不能假装成物理多接收机同步 Agent

WiSig 的原始采集确实由多个接收机监听同一 Tx，但当前 pickle 没有可验证的跨 Rx 同一包 ID／同步索引。不能把不同 Rx 的第 `k` 个样本直接配成同一次发射，更不能声称做了同步多接收机协同。

首版 Agent 因此是**同一包的异构虚拟感知 Agent**。未来如果获得可靠的跨接收机包对应关系，再扩展为物理接收节点 Agent。多接收机协作本身已有工作通过独立推理后融合提升性能，也进一步说明同步与样本对应关系需要被明确交代。[Receiver-Agnostic and Collaborative RFFI](https://arxiv.org/abs/2207.02999)、[Multiple Receiver SEI](https://doi.org/10.1049/rsn2.12606)

## 5. 两个参考目录审计

### 5.1 `E:\Project`

| 模块 | 已确认做法 | 可以借鉴 | 不能作为本项目创新 |
|---|---|---|---|
| MFE-Net | 时频图、ASPP + RPB、Triplet + Center + Prototype CE、经验原型与局部支持拒识 | P-K 采样、独立 Agent 输出、原型半径、严格来源分组与指标保存 | 整套 MFE 网络、固定时频分支、Triplet/Center/Prototype 组合本身 |
| HyDRA | VMD 多模态、残差时序卷积、Transformer/Mamba 路径、温度缩放与类条件 ECDF | 把 VMD/频域表示作为一个 Agent 的局部观测；类条件校准 | 直接复制 CFRE+TDSE/MLFE；VMD 或 Mamba 本身不是多智能体创新 |
| HyperRSI | 复数 CNN、ArcFace 类角度损失、超球面嵌入、GPD 尾部拒识 | 归一化嵌入、角度间隔、GPD/EVT 作为 single-agent baseline | HyperRSI 全结构和“超球面 + GPD”主张；其代码是自定义非商业许可证，不能随意复制 |
| PCBM | CVCNN、原型头、边界样本、特征级伪未知、OpenMax + 原型距离校准、GMM 未知细分 | 复数卷积基础、原型统计、伪未知生成思想、LCO 校准、OSCR 实现 | 固定 `q_OM/q_PD` 融合、OpenMax、边界外推或 GMM 不能包装成新 Agent 后宣称创新 |

MFE-Net 的输入和任务还是无人机时频图，不等于 WiSig 的 256 点单包 IQ；只能借鉴实验设计，不能直接移植骨干。

### 5.2 `D:\learn_pytorch\笔记\方案\os_sei_code`

这是当前最有价值的统一基线库：

- `CVCNNBackbone`：3 层复值卷积，GAP 后投影为 128 维 L2 embedding；
- `PrototypeClassifierHead`：平方欧氏距离 logits；
- 训练损失：CE + 角度间隔 + 原型紧致，可固定或不确定性加权；
- 开集：OpenMax 分数、标准化原型距离、3→8→1 监督校准器、按类阈值；
- 伪未知：竞争原型边界附近样本的特征外推；
- 评估：Known Accuracy、Unknown Recall/Precision、Macro-F1、AUROC、FPR95、OSCR、混淆矩阵与逐样本输出；
- 未知细分：embedding + IQ 统计、PCA、GMM、自动 K 和 Coverage。

推荐复用或 clean-room 重写其**通用接口和经过修正的指标定义**，尤其是 OSCR 必须用拒识前的 closed-set prediction 扫描阈值，不能对已阈值化预测重复拒识。

但它的 WiSig 正式结果来自 `SingleDay.pkl + Rx=1-1 + eq=0`，与 P3 不可横向比较；其随机样本切分、整块 NPZ 载入和 compact loader 也不适用于当前 Full WiSig。建议下一阶段实现“manifest + float32 NPY shards + mmap”，避免整块压缩 NPZ 的内存复制。

## 6. 本地多智能体代码审计

| 本地仓库 | 年份／任务 | 价值 | 结论 |
|---|---|---|---|
| `codebase-master` | 2024 MARL 教材代码，IA2C/IPPO/MAPPO/VDN/QMIX | 配置、日志、parameter sharing 与 CTDE 概念 | 可参考工程组织；不直接引入 RL 训练循环 |
| `sched_net-master` | ICLR 2019，学习何时由哪些 Agent 发消息 | 重要性打分、Top-k 通信调度、路由可解释性 | **思想最相关**，实现较旧且任务不同，只重写核心概念 |
| `rfrl-gym-master` | ICMLA 2023 / CCNC 2025，RF 强化学习环境 | RF 场景建模、Gym 接口 | 当前检出的主分支仍以单 Agent 为主，且任务是频谱交互，不是 OS-SEI |
| `JRC-AoI-multi-main` | 2021，JRC 调度 | 多 Agent PPO/A2C 的训练范式 | 调度问题，不适合作为表征网络 |
| `MARLV2X-main` | 2019，V2X 频谱共享 | 独立 Agent 与共享环境 | 旧 PyTorch 重写，和分类／拒识无直接模块关系 |
| `Cooperative-Multi-Agent-*` | 2017，Tx/Rx 学习调制 | 发送端—接收端协作概念 | 代码与依赖过旧，不建议复用 |

## 7. 联网文献与开源代码筛选

### 7.1 与最终框架最直接相关

1. **MEDAF, AAAI 2024**：多专家学习互补注意力，门控网络按样本融合独立预测，是本项目最直接的 OSR 结构参考。我们必须在它之上增加 RF 特定局部观测、显式 Agent 消息、域 Agent、开放集证据 Agent 和动态通信图，不能只把 MEDAF 的三个分支换成 1D CNN。[论文](https://arxiv.org/abs/2401.06521)、[官方代码](https://github.com/Vanixxz/MEDAF)
2. **OpenRFI, AAAI 2025**：Roinformer、RF 时序增强、半监督新类发现，说明开放集之后继续发现新类是一条成熟路线。首版只把它作为对比／后续未知细分参考，不把测试未知数据用于预训练。[论文](https://neuqnlp.github.io/assets/papers/AAAI2025_OpenRFI.pdf)、[代码](https://github.com/ShuaS2020/OpenRFI)
3. **Receiver-Agnostic RFFI via Domain-Invariant Feature Learning, IEEE CL 2025**：相位教师蒸馏 + 跨接收机均值／协方差对齐，直接使用 WiSig。适合 Channel/Domain Agent 的辅助目标，但该仓库也是非商业许可，宜根据论文 clean-room 实现。[论文 DOI](https://doi.org/10.1109/LCOMM.2025.3598034)、[代码](https://github.com/Edith-xx/Receiver-agnostic-RFFI-CL-)
4. **IB-RFF, IEEE TIFS 2026**：频率感知骨干 + 信息瓶颈/HSIC，不要求 receiver label，并报告跨 Rx 及跨 Rx+day。它适合做域鲁棒分支或强闭集基线，代码为 MIT。[论文信息](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/2026/09/information)、[代码](https://github.com/BeechburgPieStar/IB-RFF)
5. **ASKNet, IEEE CL**：可学习频谱门控和酉 Koopman 相位补偿，使用 WiSig ManySig/ManyRx。可以借鉴“频域 Agent 只做轻量、物理可解释修正”的思路，不复制其完整网络。[代码与方法说明](https://github.com/BeechburgPieStar/ASKNet-RFF)

### 7.2 开放集证据与边界

- **Multi-Task Prototype Learning, Sensors 2025**：多任务原型表征 + EVT 距离尾部，为 Open-Set Agent 的 prototype evidence 提供对比基线。[论文](https://doi.org/10.3390/s25175415)
- **Dynamic Boundary Adversarial Model, IEEE TAES 2025**：为每个已知类学习动态闭合边界，说明类条件边界优于单全局阈值的研究趋势。[论文](https://doi.org/10.1109/TAES.2025.3581888)
- **DM-MML, IEEE SPL 2026**：动态 margin、元度量学习并协同使用距离和置信度，在 WiSig 上做 few-shot open-set SEI。它是 Router 输入“距离 + 置信度”组合的重要近期依据。[论文](https://doi.org/10.1109/LSP.2026.3675912)
- **Meta Evidential Transformer, ICML 2024**：evidence-to-variance 与证据引导注意力说明不确定性可以参与路由，而不只是最后阈值。[论文](https://proceedings.mlr.press/v235/sapkota24a.html)

### 7.3 多智能体通信与路由

- **SchedNet, ICLR 2019**：学习消息重要性并只调度 Top-k Agent，是本项目动态通信图的直接先驱。[代码](https://github.com/rhoowd/sched_net)
- **ADMAC, AAAI 2024**：估计消息可靠性并调整其对最终决策的影响，适合把分类置信、域偏移和未知证据都转成 message reliability。[论文](https://doi.org/10.1609/aaai.v38i16.29708)
- **Selective Information Communication, 2024**：强调“发什么”而不只“谁发”，使用分解表示和注意力选择紧凑信息。[论文](https://doi.org/10.1527/tjsai.39-6_B-NB1)
- **Comm-MADRL survey, 2024**：将学习何时、如何、向谁发送什么作为通信型多智能体的核心维度，可用于规范术语和相关工作。[综述](https://doi.org/10.1007/s10458-023-09633-6)

这些 MARL 论文只提供通信设计原则。V1 不宣称 MARL；只有未来引入“是否请求额外观测／额外 Agent、计算成本、连续包决策”等真正的序列动作与回报后，才考虑 PPO/QMIX。

## 8. MAROS-SEI 首版框架

### 8.1 文字结构图

```text
单个 WiSig IQ 包 x:[2,256]
        │
        ├─ 时间指纹 Agent A_T ───────────────┐
        │   raw complex IQ → h_T,z_T,d_T,E_T │
        │                                    │
        ├─ 频相指纹 Agent A_F ───────────────┤
        │   FFT/phase increment → h_F,z_F,d_F│
        │                                    │ local beliefs/messages
        ├─ 通道/域 Agent A_D ────────────────┤
        │   IQ statistics + shallow feature  │
        │   → h_D, Rx/day posterior, shift   │
        │                                    │
        └─ 开集 Agent A_O ───────────────────┘
            energy/prototype/disagreement scout
            → h_O,q_O
                         │
                         ▼
        Dynamic Router / Communication Coordinator
        input = {h_i, confidence_i, prototype distance_i,
                 energy_i, domain shift, estimated SNR}
        output = sample-wise sparse graph A(x), trust w(x)
                         │
                         ▼
        one round reliability-weighted message passing
        h'_i = GRU(h_i, Σ_j A_ji(x)·Message_j)
                         │
                         ▼
        Fusion / Decision Agent
        class evidence = dynamic fusion of z'_T,z'_F and domain correction
        unknown evidence = q_O + energy + prototype radius
                           + agent disagreement + domain shift + router state
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       accepted known class       Unknown
```

### 8.2 每个 Agent 的定义

| Agent | 局部输入 | 独立输出 | 专属训练目标 | 作用 |
|---|---|---|---|---|
| 时间指纹 `A_T` | RMS 归一化 raw IQ `[2,256]` | embedding、Known logits、原型距离、energy | CE + SupCon/Prototype compactness + 域对抗 | 捕获时域硬件瞬态与复数关系 |
| 频相指纹 `A_F` | complex FFT、谱幅、相位增量 | embedding、Known logits、原型距离、energy | CE + Prototype + 与 `A_T` 的互补性约束 | 捕获频偏、相位与谱形硬件差异 |
| 通道/域 `A_D` | 低阶 IQ/频谱统计与浅层特征 | Rx/day posterior、domain-shift score、correction message、reliability | Rx CE + day CE；对指纹 Agent 使用 GRL/HSIC 去域约束 | 显式识别“当前证据可能被哪个域污染” |
| 开集 `A_O` | 各 Agent 的局部能量、原型半径、置信与分歧，加便宜 scout feature | unknown probability／Dirichlet evidence、unknown message | Known vs LCO/pseudo-unknown BCE 或 evidential loss | 统一建模多源开集证据，而非单一 MSP |
| Router | 所有本地 message 与可靠性标量 | 动态邻接 `A(x)`、Agent trust `w(x)` | team loss + counterfactual route loss + load balance | 决定谁与谁通信、谁在本样本上可信 |
| Fusion | 通信更新后的各 Agent 输出与 Router 权重 | 最终 Known logits、unknown score、decision | team CE + open-set calibration | 做最终协同决策 |

所有 Agent 的 pre-communication 和 post-communication 输出都必须保存。`A_D` 的 Rx/day 标签只在训练源域作为辅助监督；部署到未知 Rx 时不需要真实 Rx 标签。

### 8.3 动态路由必须进入推理公式

先由 Router 产生样本级权重：

\[
w(x)=\operatorname{softmax}(g_\phi(r_T,r_F,r_D,r_O)),\qquad
A(x)=\operatorname{SparseTopK}(QK^\top+R).
\]

其中 `r_i` 包含 Agent 隐状态与置信／距离／域偏移，`R` 是消息可靠性修正。首版为稳定性可训练 soft graph，同时保存 hard top-k 评估模式。

最终 Known logits 必须显式依赖 Router：

\[
z(x)=w_Tz'_T+w_Fz'_F+w_D\Delta z_D.
\]

最终未知特征建议为：

\[
v=[q_O,\ E(z),\ \tilde d_{min},\ JS(p_T,p_F),\
s_{domain},\ H(w),\ w_T,w_F,w_D].
\]

由轻量校准头输出 `q_unknown=sigmoid(MLP(v))`。令 `c*=argmax z`，使用仅由 Known validation、LCO simulated unknown 和伪未知确定的类别阈值：

\[
\hat y=\begin{cases}
\text{Unknown}, & q_{unknown}>\tau_{c^*},\\
c^*, & \text{otherwise}.
\end{cases}
\]

OpenMax/GPD 可以作为 `A_O` 的可插拔证据或 baseline，但首版最终模型不应完全依赖它们，否则核心仍会退化成旧算法的静态后处理。

### 8.4 损失函数

首版总损失可写为：

\[
L=L_{team-cls}+\lambda_{local}\sum_{i\in\{T,F\}}L^i_{cls}
+\lambda_{proto}L_{proto}+\lambda_{open}L_{open}
+\lambda_{domain}L_{domain}+\lambda_{div}L_{div}
+\lambda_{route}L_{cf-route}+\lambda_{bal}L_{balance}.
\]

- `L_team-cls`：动态路由后的最终 Known CE；
- `L_local`：防止 Agent 退化为无意义分支；
- `L_proto`：已知类紧致与类间 margin；
- `L_open`：Known=0，LCO/pseudo-unknown=1；
- `L_domain`：`A_D` 预测 Rx/day，同时通过 GRL 或 HSIC 降低 `A_T/A_F` 的域信息；
- `L_div`：约束两类指纹 Agent 的注意区域／表示不要完全相同，但避免直接把 embedding 推到互斥；
- `L_cf-route`：用“移除某 Agent 后 team loss 的变化”监督 Router 的相对贡献；
- `L_balance`：防止 Router 永远只选同一个 Agent。

首轮实现不同时启用所有正则。推荐分三步训练：先独立预训练 `A_T/A_F/A_D`，再训练 Open-Set Agent 与 Router/Fusion，最后小学习率联合微调。

### 8.5 Open-Set Agent 的监督来源

不使用真实测试 Unknown。正样本来自两种来源：

1. **Leave-class-out episode**：在 16 个 Known 内轮流把一部分临时隐藏为 simulated unknown；
2. **边界伪未知**：在最近竞争原型之间做受约束插值／外推，并记录来源类别与生成方式。

Known validation 作为负样本。阈值与融合头必须在 LCO/pseudo-unknown 上选定，再一次性冻结后评估真实 12 Unknown。

## 9. 必做基线与消融

### 9.1 主基线

1. `B0 Closed-set`：单 CVCNN，仅输出 Known；
2. `B1 Single-agent OSR`：同骨干 + Energy/Prototype distance + 类阈值；
3. `B2 Multi-agent uniform`：全部 Agent，独立输出，均匀融合，无通信、无 Router；
4. `B3 Multi-agent + communication`：固定全连接／均匀 Router；
5. `B4 Multi-agent + dynamic Router`：动态权重，无消息更新；
6. `B5 MAROS-SEI full`：动态通信图 + Router + Fusion。

外部对比优先：Softmax/MSP、OpenMax、HyperRSI、HyDRA、PCBM；条件允许再适配 MEDAF、OpenRFI、IB-RFF。所有方法必须在同一个 P0–P3 split 和同一数据预算上重跑，不能把旧表成绩直接放进新表。

### 9.2 证明 Router 真有效的反事实消融

- `w(x)` 改成均匀；
- 用训练集平均权重冻结；
- batch 内随机打乱权重；
- Router 权重保留但禁止消息传递；
- 消息保留但打乱 sender；
- 分别移除 `A_T/A_F/A_D/A_O`；
- soft routing vs hard top-k；
- 去掉 confidence／prototype distance／domain shift 中的每一种 Router 输入。

除最终指标外，要报告 Router 权重方差、各 Agent 使用率、路由熵、按日期/Rx/known-vs-unknown 的权重分布，以及 Agent 删除后的样本级性能变化。如果权重几乎不随样本变化，就不能宣称动态协同成功。

## 10. 评估与保存产物

### 10.1 指标

最低要求：

- Known Accuracy；
- Unknown Precision / Recall / F1；
- Macro-F1（Unknown 作为一个拒识类）；
- AUROC、AUPR、FPR95；
- OSCR（使用拒识前 Known prediction）；
- 每 Rx、每日期和 worst-domain 指标；
- ECE／Brier score；
- 参数量、FLOPs、单包延迟。

若继续做未知类细分，再报告 NMI、ARI、Hungarian Acc. 与 Coverage；拒识和未知细分要作为两个阶段分别评价。

### 10.2 每次运行的必要文件

```text
resolved_config.yaml
dataset_manifest.csv
split_manifest.json
checkpoint.pt
metrics.json
metrics_by_rx.csv
metrics_by_day.csv
confusion_matrix.csv / .png
predictions.parquet
agent_outputs.parquet
router_weights.parquet
router_weight_histograms.png
agent_ablation.json
calibration.json
```

`agent_outputs` 至少保存 sample_id、每 Agent top-1/confidence/energy/prototype distance、通信前后 logits 摘要、domain shift、unknown score、Router 权重和最终结果。

## 11. 下一阶段建议目录

```text
multi_agent_os_sei/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/wisig_p0.yaml
│   ├── data/wisig_p3.yaml
│   └── model/{closed,single_osr,uniform,router,full}.yaml
├── src/maros_sei/
│   ├── data/{audit,manifest,shards,dataset,splits}.py
│   ├── models/{stem,agents,router,communication,fusion}.py
│   ├── openset/{prototypes,pseudo_unknown,calibration}.py
│   ├── training/{losses,trainer}.py
│   └── evaluation/{metrics,plots,export}.py
├── scripts/{audit_wisig,prepare_wisig,train,evaluate}.py
├── tests/
└── outputs/.gitkeep
```

## 12. 下一步编码门槛

开始模型编码前应先完成并验证：

1. Full WiSig manifest 构建器能复现本报告的 152 个有效文件、174 Tx、41 Rx、4 日期和 9,563,785 包；
2. split JSON 固化上述 28/16/12 Tx 与 8/2/2 Rx；
3. P3 四个 split 的样本数完全匹配 102,400 / 12,800 / 12,800 / 9,600；
4. train、val、test 的 Rx 和日期组无意外交集；
5.真实 Unknown 的 sample_id 不出现在训练、校准、阈值或早停产物；
6. 先跑通 B0 与 B1，再增加 Agent，避免一次性堆叠后无法定位增益来源。

达到这六项后再进入第二阶段实现。
