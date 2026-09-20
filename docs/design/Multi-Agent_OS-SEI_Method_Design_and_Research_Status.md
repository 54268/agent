# 多智能体协同开放集辐射源识别（Multi-Agent OS-SEI）方法设计与研究现状

> 用途：作为后续代码实现、实验设计和论文方法章节的统一设计文档。  
> 当前任务边界：**只做“开放集辐射源识别 + 多智能体协同”**，暂不把跨接收机、跨日期、域泛化作为主问题。跨域实验后续仅作为鲁棒性扩展。  
> 当前主数据集：  
> 1. **ORACLE KRI-16 demodulated**；  
> 2. **自建 WiSig_OpenSet_1Day_1Rx**：40 Known / 20 Unknown，固定同一天、同一接收机，避免把任务变成跨域识别。

---

# 1. 总体研究定位

## 1.1 要解决的问题

传统开放集辐射源识别通常依赖一种或少数几种证据，例如：

- SoftMax置信度；
- Prototype距离；
- OpenMax/EVT；
- 重构误差；
- 能量分数；
- 边界外推得到的伪未知样本。

问题在于：

> **某一种证据在开放集场景下往往不够可靠。**

例如一个真实Unknown样本可能出现：

- 分类器非常自信地认为它是Known Tx 7；
- 但它距离Tx 7原型很远；
- 同时它无法被Tx 7对应的已知特征空间很好地重构；
- 多种证据之间存在明显冲突。

因此，本工作不再让单一分类器独立决定Known/Unknown，而是将开放集识别拆解成多个互补的“认知任务”，分别交给不同专家Agent，然后通过**显式Agent通信、分歧建模和样本级动态路由**完成最终决策。

核心思想：

> **分类置信度不等于开放集可信度。**  
> 对同一个RF样本，从“已知类判别、原型几何、可重构性、开放边界风险”等不同视角建立专家Agent，让它们交换证据、暴露分歧，再由Router针对每个样本动态决定应该相信谁。

---

# 2. 当前研究现状与我们的位置

截至当前检索，没有找到一篇与下面定义完全一致的公开同行评审工作：

> “多个专职Agent分别产生不同类型的开放集证据，Agent之间显式通信，再通过样本级动态路由协同完成Open-Set SEI，并利用Agent分歧生成伪未知样本。”

这并不意味着绝对不存在遗漏，但在本次围绕以下关键词的检索中没有发现直接同类工作：

- multi-agent + open-set specific emitter identification
- multi-agent + open-set RFFI
- multi-agent + radio-frequency fingerprint identification
- multi-agent + emitter identification

目前最接近的工作主要分成四类。

---

## 2.1 最接近之一：多分类器融合OS-SEI

**Zhao et al., “Multi-Classifier Fusion for Open-Set Specific Emitter Identification,” Remote Sensing, 2022. DOI: 10.3390/rs14092226**

该工作：

- 构建三个独立SEI通道；
- 输入分别为：
  - I/Q采样点；
  - 频谱；
  - 幅度-相位；
- 三个通道分别使用ResNet；
- 各自先完成开放集判别；
- 最后使用mean、vote等分类器组合方式在决策层融合。

它已经证明：

> 单一特征空间容易出现Unknown与Known“feature coincidence”，融合不同证据有价值。

但它仍然属于：

**多分类器 / 多分支 + 决策层融合**

而不是我们的：

**任务专职Agent + 私有状态 + 显式通信 + 分歧驱动伪未知 + 样本级动态路由。**

因此，如果我们的系统只是：

```text
CNN Agent 1
CNN Agent 2
CNN Agent 3
     ↓
平均概率
```

创新会非常危险。

---

## 2.2 重构 + 多通道判别：MRDN

**Tan et al., “Dynamic open set specific emitter identification via multi-channel reconstructive discriminant network,” IET Radar, Sonar & Navigation, 2023. DOI: 10.1049/rsn2.12380**

核心包括：

- Reconstruction Network；
- Multi-channel discriminator；
- 异常检测通道；
- Known闭集分类通道；
- GAN式对抗训练思想；
- 通过重构质量判断未知。

该工作说明：

> “已知空间是否能够解释/重构当前样本”是一种有价值的开放集证据。

所以我们不能把“加重构分支”本身写成创新。

我们的区别必须在：

- Reconstruction是一个独立专家Agent；
- 它输出的不只是最终分数，而是发送message给其它Agent；
- 重构异常会参与Agent disagreement；
- disagreement进一步驱动Pseudo-Unknown Generation Agent；
- 最终由Router决定当前样本该不该信任重构证据。

---

## 2.3 多任务Prototype开放集RFFI

**Ma et al., “Open-Set Radio Frequency Fingerprint Identification Method Based on Multi-Task Prototype Learning,” Sensors, 2025. DOI: 10.3390/s25175415**

该工作联合优化：

- classification；
- reconstruction；
- prototype clustering；

再利用：

- Prototype最小距离；
- EVT尾部分布；

完成Known/Unknown判定。

这篇与我们的“Identity + Reconstruction + Prototype”三个Agent在功能上非常接近，因此需要特别注意。

如果我们的方案只是：

```text
分类分支
重构分支
Prototype分支
```

那很容易被认为只是把MTPL重新命名成Agent。

所以真正的创新必须建立在：

1. 每个Agent存在独立状态与消息；
2. Agent之间有显式通信；
3. 使用跨Agent disagreement定位危险边界；
4. Pseudo-Unknown Generator Agent根据多个Agent的意见冲突主动生成伪未知；
5. Router进行样本级动态证据选择，而不是固定融合；
6. 推理阶段不同样本会得到不同Agent权重。

---

## 2.4 Collaborative RFFI，但不是Open-Set

**Shen et al., “Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification,” IEEE Transactions on Mobile Computing, 2024（在线版本更早）. DOI: 10.1109/TMC.2023.3340039**

主要解决：

- receiver impairment；
- receiver-independent representation；
- 多接收机 collaborative inference。

它说明“协同推理”在RFFI领域本身是合理的。

但是它的协同主体是：

> 多个物理Receiver。

而我们的协同主体是：

> 对同一个RF样本从不同开放集证据空间进行判断的认知型Expert Agents。

并且我们解决的是OS-SEI，不是receiver-agnostic closed-set RFFI。

---

## 2.5 RF领域的真正Multi-Agent工作

RF领域已经大量使用Multi-Agent，但主要集中在：

- 认知无线电；
- 频谱共享；
- 干扰博弈；
- 资源分配；
- 通信调度。

典型工作：

**Liang et al., “Spectrum Sharing in Vehicular Networks Based on Multi-Agent Reinforcement Learning,” IEEE JSAC, 2019. DOI: 10.1109/JSAC.2019.2933962**

其中：

- 每条V2V链路作为Agent；
- 每个Agent拥有局部Observation；
- 多Agent共享整体目标；
- 通过协同完成频谱和功率分配。

这类工作支持一个重要概念：

> Agent不必是LLM，也不必是机器人；只要它有自己的Observation / State / Action / Objective，并且与其它Agent通过共享信息完成全局任务，就可以形成Multi-Agent系统。

---

## 2.6 RFRL Gym：RF多智能体环境

**Vangaru et al., “A Multi-Agent Reinforcement Learning Testbed for Cognitive Radio Applications,” 2025 CCNC相关工作 / arXiv版本。**

该工作将RF Reinforcement Learning环境扩展到Multi-Agent场景，用于：

- cognitive radio；
- jamming；
- spectrum interaction。

它不是SEI，但可以参考：

- Agent observation定义；
- 多Agent协作；
- RF场景下的Agent通信组织方式。

---

## 2.7 OOD领域的多智能体思想

**Song et al., “Multi-Agent Visual Reasoning for Out-of-Distribution Detection in Complex Road Environments,” IEEE Access, 2025. DOI: 10.1109/ACCESS.2025.3627794**

虽然是视觉OOD而不是RF，但它说明：

> OOD/Unknown判断可以被拆成多个专家推理任务，再进行Multi-Agent协作。

因此，我们把Unknown检测从“一个阈值”升级为“多专家协同审查”在研究逻辑上是合理的。

---

# 3. 与上一版Competition-Aware Boundary Extrapolation方案的关系

你上一版纯开放集方案的核心逻辑是：

1. Prototype-structured embedding；
2. 边界Known样本挖掘；
3. Prototype competition；
4. Feature-level pseudo-unknown extrapolation；
5. OpenMax + Prototype distance；
6. Supervised calibration；
7. Unknown rejection；
8. Rejected unknown subdivision。

其中最值得继承到当前工作的是：

> **不使用真实Unknown训练，而是从Known空间构造具有开放边界意义的pseudo-unknown。**

但不能直接照搬原方法，否则新工作会变成：

> 旧方案 + Multi-Agent外壳。

因此新的Pseudo-Unknown机制改成：

> **由Agent disagreement驱动，而不是主要由prototype competition驱动。**

旧方案：

```text
Local marginality
       +
Prototype competition
       ↓
Boundary samples
       ↓
Outward extrapolation
       ↓
Pseudo Unknown
```

新方案：

```text
Identity Agent
Prototype Agent
Reconstruction Agent
       ↓
跨Agent意见分歧
       ↓
Disagreement-sensitive Known samples
       ↓
Pseudo-Unknown Generation Agent
       ↓
最大化专家冲突的边界外样本
```

这样旧方案的核心经验自然继承，但研究问题和生成依据发生了变化。

---

# 4. 最终建议的Agent体系

建议采用：

## 5个专家Agent + 1个协调Agent

### 专家Agent

1. **Identity Discrimination Agent（身份判别Agent）**
2. **Prototype Geometry Agent（原型几何Agent）**
3. **Reconstruction Consistency Agent（重构一致性Agent）**
4. **Pseudo-Unknown Generation Agent（伪未知生成Agent）**
5. **Open-Space Boundary Agent（开放边界Agent）**

### 协调Agent

6. **Dynamic Router & Decision Agent（动态路由与决策Agent）**

其中：

- 前3个Agent负责观察Known空间的不同属性；
- 第4个Agent只在训练阶段主要工作，负责制造Unknown-like训练证据；
- 第5个Agent学习Known/Unknown开放边界；
- 第6个Agent负责通信后的证据选择和最终决策。

---

# 5. 整体网络结构

```text
                         Input I/Q
                          2 × 256
                             │
                             ▼
                    Shared Shallow Stem
                             │
                  Shared Feature h_s
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
 Identity Agent       Prototype Agent      Reconstruction Agent
 “它像哪一类？”       “它在已知空间吗？”      “已知空间能解释它吗？”
       │                     │                     │
       └──────────────┬──────┴──────┬──────────────┘
                      │             │
                      ▼             ▼
                 Agent Message Construction
                      │
                      ▼
               Agent Communication Module
                      │
       ┌──────────────┴────────────────┐
       │                               │
       ▼                               ▼
Disagreement Estimator        Contextualized Agent States
       │                               │
       ▼                               │
Pseudo-Unknown                     Open-Space
Generation Agent  ───────────────► Boundary Agent
(training mainly)                  │
                                   │
                    ┌──────────────┘
                    ▼
          Dynamic Router & Decision Agent
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Known Class            Unknown
```

---

# 6. Shared Stem：共享浅层特征提取

## 6.1 为什么需要Shared Stem

如果每个Agent都从原始2×256 IQ完全重新训练一套Backbone：

- 参数量大；
- Agent之间基础表示差异过大；
- 训练不稳定；
- 很难证明性能提升来自Multi-Agent协作而不是模型规模堆叠。

因此建议：

> 浅层共享 + 深层私有。

例如：

```text
IQ 2×256
  ↓
Conv1D
  ↓
BN + GELU
  ↓
Residual Conv Block
  ↓
Shared Feature h_s
```

推荐初始维度：

```text
h_s: [B, 128, 64]
```

随后各Agent有自己的private encoder。

---

## 6.2 Shared Stem不承担最终开放集决策

Shared Stem只负责学习：

- 局部I/Q变化；
- 基础相位/幅度模式；
- 短期相关结构。

它不应成为所有Agent共用的完整Feature Extractor。

否则最终又会退化成：

> 一个Backbone + 多Head。

我们需要每个专家后续存在真实的private representation。

---

# 7. Agent 1：Identity Discrimination Agent

## 7.1 Agent任务

回答：

> “假设这个样本来自已知类，它最像哪个发射机？”

它主要服务于：

- Known classification；
- 类间判别；
- 生成类别置信信息。

---

## 7.2 Observation

输入：

```text
Shared Feature h_s
```

---

## 7.3 Private Encoder

建议：

```text
h_s
 ↓
Private Residual 1D CNN
 ↓
Global Average Pooling
 ↓
FC
 ↓
z_I ∈ R^128
```

---

## 7.4 输出

### 类别概率

\[
p_I=\operatorname{Softmax}(W_I z_I)
\]

### Top-1置信度

\[
c_I=\max_k p_I(k)
\]

### Top1-Top2 margin

\[
m_I=p_{(1)}-p_{(2)}
\]

### Entropy

\[
H_I=-\sum_kp_I(k)\log p_I(k)
\]

### Message

\[
m_I=[z_I,\;p_I,\;c_I,\;m_I,\;H_I]
\]

然后通过独立Projection映射到统一message维度。

---

## 7.5 Loss

第一版建议：

\[
L_{id}=L_{CE}+\lambda_{sc}L_{SupCon}
\]

如果SupCon第一轮不稳定，可以先只使用：

\[
L_{id}=L_{CE}
\]

后续再加。

---

## 7.6 Identity Agent的局限

它最容易出现：

> Unknown → 高置信度Known。

因此它没有最终拒识权。

---

# 8. Agent 2：Prototype Geometry Agent

## 8.1 Agent任务

回答：

> “这个样本在Known feature geometry中到底处于什么位置？”

不是只看SoftMax。

---

## 8.2 Private Embedding

\[
z_P=E_P(h_s)
\]

建议：

```text
z_P ∈ R^128
```

每个Known类别维护Prototype：

\[
c_k=\frac{1}{N_k}\sum_{y_i=k}z_{P,i}
\]

Prototype可以：

- epoch-end重新计算；
- 或EMA更新。

第一版推荐：

> epoch-end全训练集均值 + batch内EMA辅助。

---

## 8.3 Prototype距离

\[
d_k=\|z_P-c_k\|_2^2
\]

最近类别：

\[
\hat y_P=\arg\min_k d_k
\]

最近和次近：

\[
d_1,\;d_2
\]

距离margin：

\[
m_P=d_2-d_1
\]

Prototype概率：

\[
p_P(k)=
\frac{\exp(-d_k/T)}
{\sum_j\exp(-d_j/T)}
\]

---

## 8.4 Prototype unknown evidence

为每类估计Known validation距离统计。

例如：

\[
u_P=\sigma\left(
\gamma_{\hat y}(d_1-\delta_{\hat y})
\right)
\]

其中：

- \(\delta_c\)：类别c已知验证样本距离中心位置；
- \(\gamma_c\)：尺度参数。

第一版不需要一上来就EVT，可以先使用：

- 类内距离quantile；
- z-score；
- logistic calibration。

后续再增加EVT baseline。

---

## 8.5 Message

\[
m_P=
[z_P,\;p_P,\;d_1,\;d_2,\;m_P,\;u_P]
\]

---

## 8.6 Loss

推荐：

\[
L_{proto}
=
L_{proto-cls}
+
\lambda_cL_{compact}
+
\lambda_mL_{margin}
\]

其中：

### Prototype classification

\[
L_{proto-cls}
=
-\log p_P(y)
\]

### 类内紧凑

\[
L_{compact}
=
\|z_P-c_y\|_2^2
\]

### 类间margin

要求当前样本到目标prototype明显小于最近竞争prototype。

---

# 9. Agent 3：Reconstruction Consistency Agent

## 9.1 Agent任务

回答：

> “如果Identity/Prototype认为你属于某个Known类别，那么Known feature manifold能不能解释你？”

重点不是高质量复原IQ，而是：

> 利用重构一致性形成另一种Unknown evidence。

---

## 9.2 为什么不直接重构原始IQ

第一版建议：

> 重构Shared latent feature，而不是直接重构2×256 IQ。

优势：

- 训练稳定；
- 参数少；
- 不需要生成真实RF波形；
- 更容易和其它Agent共享embedding信息。

---

## 9.3 Class-conditioned Reconstruction

训练时使用真实Known标签：

\[
\hat h_s=R(h_s,c_y)
\]

推理时使用候选类别：

\[
\hat y=
\arg\max
\left(
\beta p_I+(1-\beta)p_P
\right)
\]

然后：

\[
\hat h_s=R(h_s,c_{\hat y})
\]

---

## 9.4 Reconstruction Error

\[
e_R=
\frac{1}{D}
\|h_s-\hat h_s\|_2^2
\]

再根据Known validation统计进行归一化：

\[
u_R=
\operatorname{NormScore}(e_R)
\]

得到：

```text
u_R ≈ 0 → 很像Known
u_R ≈ 1 → Known manifold很难解释
```

---

## 9.5 Message

\[
m_R=
[z_R,\;e_R,\;u_R,\;\text{consistency}]
\]

其中consistency可以定义为：

- Identity候选类；
- Prototype候选类；
- Reconstruction最匹配类；

之间是否一致。

---

## 9.6 Loss

\[
L_{rec}
=
L_{reconstruction}
+
\lambda_{con}L_{class-consistency}
\]

第一版：

\[
L_{reconstruction}
=
\|h_s-\hat h_s\|_2^2
\]

即可。

---

# 10. Agent之间的分歧建模

这是本方法区别于普通多任务网络的重要部分。

---

## 10.1 Identity vs Prototype分歧

两个Agent都可以输出Known类别概率：

\[
p_I,\quad p_P
\]

定义：

\[
D_{IP}
=
JS(p_I\|p_P)
\]

即Jensen-Shannon Divergence。

---

## 10.2 类别意见冲突

\[
D_{label}
=
\mathbf{1}
[
\arg\max p_I
\neq
\arg\max p_P
]
\]

---

## 10.3 置信与几何矛盾

例如：

Identity非常自信：

\[
c_I\rightarrow1
\]

但Prototype unknown score也很高：

\[
u_P\rightarrow1
\]

这是很重要的开放集危险状态。

定义：

\[
D_{conf-geo}=c_I\cdot u_P
\]

---

## 10.4 置信与重构矛盾

\[
D_{conf-rec}=c_I\cdot u_R
\]

表示：

> 分类器越自信、重构越失败，越值得怀疑。

---

## 10.5 总Disagreement Score

第一版可以定义：

\[
D(x)=
\lambda_1D_{IP}
+\lambda_2D_{label}
+\lambda_3D_{conf-geo}
+\lambda_4D_{conf-rec}
\]

所有项先归一化到：

\[
[0,1]
\]

初始可令各项等权：

```text
λ1 = λ2 = λ3 = λ4 = 0.25
```

之后让Router或一个小MLP学习权重。

---

# 11. Agent 4：Pseudo-Unknown Generation Agent

这是当前方案非常值得主打的核心。

---

## 11.1 它是不是一个真正的Agent？

建议明确把它设计成Agent，而不是普通augmentation module。

它具有：

### Observation

来自：

- Identity Agent；
- Prototype Agent；
- Reconstruction Agent；
- Agent Communication Module；

的状态和消息。

### Internal State

包括：

\[
s_G=
[z,\;c_y,\;D(x),\;u_P,\;u_R,\;H_I,\;m_I]
\]

### Action

它决定：

1. 哪些Known样本适合作为boundary seed；
2. 对seed向哪个方向移动；
3. 移动多远；
4. 是否保留生成样本。

### Objective

构造：

> 距离Known manifold不远，但足以使多个Expert意见发生冲突的hard pseudo-unknown。

### Output

\[
\tilde z
\]

以及：

```text
pseudo_unknown_hardness
generation_confidence
seed_id
```

所以它具备一个完整Agent的基本语义。

---

# 12. Pseudo-Unknown Agent内部工作流程

## Step 1：寻找Disagreement-sensitive Known samples

对训练Known：

\[
D(x_i)
\]

按照类别分别排序。

每类取Top-q：

```text
q = 10% ~ 20%
```

得到：

\[
B_D
\]

这些不是普通的“距离中心最远Known”，而是：

> 多个Agent开始产生意见分裂的Known。

这与上一版“prototype competition边界挖掘”形成明显区别。

---

## Step 2：确定基础embedding

建议第一版在一个共享开放集latent空间中生成：

\[
z=Proj(h_s)
\]

而不要分别在每个Agent自己的空间都生成一套pseudo-unknown。

这样实现更简单。

---

## Step 3：生成方向

推荐把方向拆成三部分。

### A. Outward direction

\[
v_{out}
=
\frac{z-c_y}
{\|z-c_y\|_2+\epsilon}
\]

### B. Disagreement gradient direction

让当前样本往“专家意见更加冲突”的方向移动：

\[
v_{dis}
=
\frac{
\nabla_zD(z)
}{
\|\nabla_zD(z)\|_2+\epsilon
}
\]

### C. Learnable residual direction

Generator MLP：

\[
v_G=
G_\phi(
[m_I,m_P,m_R,D(x)]
)
\]

归一化：

\[
\bar v_G=
\frac{v_G}{\|v_G\|_2+\epsilon}
\]

---

## Step 4：最终生成方向

\[
v=
Norm(
\beta_1v_{out}
+
\beta_2v_{dis}
+
\beta_3\bar v_G
)
\]

第一轮简单实现可以先关闭learnable residual：

```text
β1 = 0.5
β2 = 0.5
β3 = 0
```

确认有效以后再开启：

```text
β3 > 0
```

---

## Step 5：Generator决定步长

\[
\alpha=
\alpha_{min}
+
(\alpha_{max}-\alpha_{min})
\sigma(g_\phi(s_G))
\]

生成：

\[
\tilde z=z+\alpha v
\]

---

# 13. 防止Pseudo-Unknown生成成“无意义远端噪声”

这是必须解决的问题。

我们不希望：

```text
Known cluster ------------------------------ pseudo unknown
```

太远的样本Boundary Agent很容易识别，对真正Unknown没有帮助。

因此要求pseudo-unknown处于：

> Known boundary附近的外壳区域（boundary shell）。

对类别c，从Known训练/验证计算高分位半径：

\[
r_c^{(q)}
=
Q_q(
\|z_i-c_c\|
)
\]

例如：

```text
q = 0.90 或 0.95
```

目标pseudo-unknown距离：

\[
r_c^{(q)}
<
\|\tilde z-c_c\|
<
r_c^{(q)}+\Delta
\]

即：

> 刚出Known高密度区域，但不要飞得过远。

---

# 14. Pseudo-Unknown Agent Loss

第一版建议：

\[
L_{PUG}
=
-\lambda_DD(\tilde z)
+
\lambda_sL_{shell}
+
\lambda_nL_{norm}
\]

其中：

### Disagreement maximization

\[
-D(\tilde z)
\]

使生成样本诱发更强Agent disagreement。

### Boundary shell constraint

\[
L_{shell}
=
(
\|\tilde z-c_y\|-r_y^{target}
)^2
\]

### Perturbation norm

\[
L_{norm}
=
\max(0,\|\tilde z-z\|-\epsilon_{max})^2
\]

---

# 15. Pseudo-Unknown Agent V2：与Boundary Agent形成对抗

第一版先不要做，等稳定后再加。

高级版：

- Generator Agent产生难pseudo-unknown；
- Boundary Agent尝试识别它；
- Generator继续生成更难的边界样本。

即：

```text
Pseudo-Unknown Agent
       ↓
hard unknown-like feature
       ↓
Boundary Agent
       ↓
是否成功拒识
       └──── feedback ────► Generator
```

这里可以形成轻量Min-Max训练。

但第一版代码务必先完成：

> Disagreement-guided deterministic/learnable extrapolation。

不要一开始就同时引入GAN、MARL等复杂机制。

---

# 16. Agent 5：Open-Space Boundary Agent

## 16.1 Agent任务

它专门回答：

> “结合其它Agent提供的证据，这个样本是不是Unknown？”

它不是Known类别分类器。

---

## 16.2 输入Evidence Vector

建议：

\[
e(x)=
[
c_I,
H_I,
m_I,
d_1,
d_2,
m_P,
u_P,
e_R,
u_R,
D_{IP},
D(x)
]
\]

再加communication后的context feature。

---

## 16.3 网络

第一版不用大模型。

例如：

```text
Evidence Vector
      ↓
Linear
      ↓
GELU
      ↓
Dropout
      ↓
Linear
      ↓
Sigmoid
```

输出：

\[
u_B=P(Unknown|x)
\]

---

## 16.4 Boundary训练数据

Known：

```text
label = 0
```

Pseudo-Unknown Generator产生的：

```text
label = 1
```

严格禁止：

> 使用真实Unknown test来训练Boundary Agent。

---

## 16.5 Loss

\[
L_B
=
BCE(u_B,r)
\]

其中：

\[
r=
\begin{cases}
0 & Known\\
1 & PseudoUnknown
\end{cases}
\]

---

# 17. Agent Communication Module

这是把“多个分支”真正升级成“多智能体”的关键。

---

## 17.1 每个Agent先形成message

不同Agent输出维度不同。

例如：

```text
Identity message:
[z_I, probability, entropy, margin]

Prototype message:
[z_P, distance, prototype probability, unknown score]

Reconstruction message:
[z_R, reconstruction error, consistency]
```

分别使用：

\[
\bar m_i=W_im_i
\]

投影为统一：

```text
message_dim = 64
```

---

## 17.2 Multi-Agent Attention

令：

\[
M=
[
\bar m_I,
\bar m_P,
\bar m_R,
\bar m_B
]
\]

采用Self-Attention：

\[
A=
Softmax
\left(
\frac{QK^T}{\sqrt d}
\right)
\]

然后：

\[
M'=AV
\]

---

## 17.3 Contextualized Agent State

每个Agent更新：

\[
m_i'
=
LN(
m_i+M_i'
)
\]

意味着：

Identity Agent不再只知道：

> “我认为Tx7概率0.94”。

它还知道：

- Prototype认为它离Tx7很远；
- Reconstruction认为它解释失败；
- Boundary认为unknown风险高。

---

## 17.4 为什么Communication不能被普通Concat替代

必须做消融：

### Variant A

```text
No communication
```

各Agent独立。

### Variant B

```text
Concat + MLP
```

证明简单拼接效果。

### Variant C

```text
Agent Attention Communication
```

证明显式communication价值。

---

# 18. Coordinator Agent：Dynamic Router & Decision Agent

Router不能只是固定权重。

它必须针对每一个输入样本：

\[
x
\]

输出不同专家权重。

---

# 19. 推荐使用“双路由”而不是一个统一权重

因为：

Known分类和Unknown判断依赖的Agent并不完全相同。

---

## 19.1 Known-class Router

主要在：

- Identity；
- Prototype；

之间动态融合。

\[
[\alpha_I,\alpha_P]
=
Softmax(
R_K(M')
)
\]

Known类别分布：

\[
p_K
=
\alpha_Ip_I
+
\alpha_Pp_P
\]

---

## 19.2 Unknown-evidence Router

主要在：

- Prototype；
- Reconstruction；
- Boundary；

之间动态融合：

\[
[\beta_P,\beta_R,\beta_B]
=
Softmax(
R_U(M')
)
\]

Unknown score：

\[
S_U
=
\beta_Pu_P
+
\beta_Ru_R
+
\beta_Bu_B
\]

可以额外加入：

\[
+\beta_DD(x)
\]

第一版先不加，避免复杂。

---

# 20. Router为什么是动态的

不同样本会得到不同权重。

例如Easy Known：

```text
Identity       0.62
Prototype      0.38

Prototype-U    0.15
Recon-U        0.18
Boundary-U     0.67
```

另一个危险Unknown：

```text
Identity:
Tx7 = 0.96

Prototype:
distance abnormal

Reconstruction:
error high

Boundary:
unknown = 0.91
```

Router可能产生：

```text
Known Router:
Identity       0.21
Prototype      0.79

Unknown Router:
Prototype      0.27
Reconstruction 0.31
Boundary       0.42
```

这就是：

> Sample-Adaptive Evidence Routing。

---

# 21. 最终Open-Set Decision

最终：

\[
\hat y=
\begin{cases}
Unknown,&S_U\ge\tau\\
\arg\max_kp_K(k),&S_U<\tau
\end{cases}
\]

其中：

\[
\tau
\]

只能利用：

- Known validation；
- Pseudo-Unknown；

进行确定。

不能看真实Unknown测试标签。

---

# 22. Router的训练目标

Known样本：

- 应正确分类；
- Unknown score应低。

Pseudo-Unknown：

- Unknown score应高。

因此：

\[
L_{router}
=
L_{known-cls}
+
\lambda_uL_{open}
+
\lambda_rL_{route-reg}
\]

---

## 22.1 Routing Regularization

防止Router永远只用一个Agent：

\[
L_{balance}
\]

可以控制batch级平均Agent使用率不要彻底坍塌。

但是不要强制每个样本平均使用所有Agent。

我们的目标是：

> batch级避免永久弃用Agent，sample级允许高度稀疏。

---

# 23. 完整训练流程

建议不要一上来end-to-end全部训练。

---

## Stage 0：数据固定

WiSig：

```text
40 Known
20 Unknown

train = 12720
val = 1800
known_test = 3680
unknown_test = 4800
open_test = 8480
```

Unknown：

> 不进入任何训练、阈值选择或超参数选择。

ORACLE同样严格按照Known/Unknown emitter划分。

---

# 24. Stage 1：独立训练三个基础Expert Agents

训练：

1. Identity Agent；
2. Prototype Agent；
3. Reconstruction Agent。

Loss：

\[
L_{stage1}
=
L_{id}
+
\lambda_PL_{proto}
+
\lambda_RL_{rec}
\]

目的：

让它们形成真正不同的专长。

---

# 25. Stage 2：Communication Warm-up

加入：

Agent Communication。

但暂时不训练Pseudo-Unknown和Router。

目的：

观察不同Agent的message是否形成互补。

---

## 可加入轻量Known Consistency Loss

对明显Easy Known：

\[
L_{cons}
=
JS(p_I,p_P)
\]

但权重不能太大。

否则Identity和Prototype会被训练成完全相同。

---

## 防止Agent Collapse

可以使用representation decorrelation：

\[
L_{div}
\]

鼓励：

\[
z_I,z_P,z_R
\]

保留互补特征。

注意：

> 不是要求它们互相越远越好，而是防止private encoder完全退化成同一个表示。

---

# 26. Stage 3：训练Pseudo-Unknown Generation Agent

使用Known train。

步骤：

```text
Known
 ↓
3 Experts
 ↓
Disagreement
 ↓
每类Top-q危险Known
 ↓
Pseudo-Unknown Agent
 ↓
Pseudo Unknown
```

第一版生成后可以建立：

```text
pseudo_unknown_cache
```

每N个epoch重新生成。

这样比每个batch实时生成稳定。

---

# 27. Stage 4：训练Boundary Agent

训练数据：

```text
Known Validation / Held-out Known
+
Pseudo Unknown
```

训练：

\[
Known=0,\quad PseudoUnknown=1
\]

注意：

最好不要让Boundary直接在它随后用于阈值选择的同一批Known上过拟合。

可以把Known validation进一步拆：

```text
calibration-train
threshold-val
```

如果样本量暂时不够，第一版可先保持现有val，后续论文正式实验再严格拆分。

---

# 28. Stage 5：训练Dynamic Router

此时：

- 基础Agent已稳定；
- Boundary Agent已具备Unknown evidence；
- Communication已建立。

Router输入：

```text
contextualized messages
```

训练：

- Known；
- PseudoUnknown。

第一版建议：

> freeze大部分expert，只训练communication后半段 + router。

---

# 29. Stage 6：小学习率Joint Fine-tuning（可选）

如果前5个阶段稳定，再使用：

```text
lr = base_lr × 0.1
```

联合微调。

不要第一轮直接joint training。

---

# 30. 总Loss

完整形式可以写成：

\[
L=
L_{id}
+\lambda_PL_{proto}
+\lambda_RL_{rec}
+\lambda_CL_{comm}
+\lambda_GL_{PUG}
+\lambda_BL_B
+\lambda_{RT}L_{router}
+\lambda_{bal}L_{balance}
\]

实际实现一定分阶段开启。

---

# 31. 推理阶段流程

Pseudo-Unknown Generation Agent主要是训练Agent。

正式推理时关闭它。

因此：

```text
IQ
 ↓
Shared Stem
 ↓
Identity / Prototype / Reconstruction
 ↓
Communication
 ↓
Boundary
 ↓
Router
 ↓
Known / Unknown
```

优点：

> Pseudo-Unknown Agent提升训练阶段的开放边界，但不会显著增加部署推理开销。

---

# 32. 为什么Pseudo-Unknown Generator Agent非常适合本工作

它把上一版工作的优势自然继承下来：

旧工作已经证明：

> Known边界附近构造未知样本，可以帮助学习Unknown rejection。

现在把它升级为：

> 多Agent主动协同寻找“专家意见开始分裂”的Known区域，再由专门Generator Agent探索这些边界外区域。

研究逻辑就变成：

### 旧

```text
几何边界 → Pseudo Unknown
```

### 新

```text
认知分歧边界 → Agent主动探索 → Pseudo Unknown
```

这是一个自然升级，而不是生硬添加。

---

# 33. 当前最值得主打的创新点

## 创新点1：Evidence-Specialized Multi-Agent OS-SEI

首次尝试将OS-SEI分解为多个专职认知任务：

- Identity discrimination；
- Prototype geometry；
- Reconstruction consistency；
- Pseudo-unknown exploration；
- Open-space boundary assessment。

不同Agent拥有：

- private representation；
- private objective；
- explicit message；
- 不同的开放集判断依据。

重点不是“多分支”，而是“多任务认知实体”。

---

## 创新点2：Confidence-Aware Agent Communication

不是各Agent最后才投票。

而是：

> 在最终判决前显式交换置信度、原型距离、重构异常和边界风险。

通过Agent Attention形成contextualized states。

---

## 创新点3：Disagreement-Guided Pseudo-Unknown Generation Agent

利用Agent之间的：

- 类别分布冲突；
- 分类置信与prototype geometry矛盾；
- 分类置信与reconstruction矛盾；

寻找开放空间最危险的Known boundary。

然后由Generator Agent主动生成：

> disagreement-maximizing boundary pseudo-unknown。

这可以作为本工作的核心创新之一。

---

## 创新点4：Sample-Adaptive Dual Dynamic Routing

不是mean/vote/fixed weight。

对每个样本动态输出：

- Known分类Agent权重；
- Unknown evidence Agent权重。

解决：

> 某个Agent在某个样本上明显失效，但固定融合仍然强行相信它的问题。

---

## 创新点5：No Real Unknown During Training

所有：

- Agent训练；
- pseudo-unknown生成；
- Boundary训练；
- Router训练；
- 阈值选择；

都不接触真实Unknown test。

保证严格开放集协议。

---

# 34. 与最相关现有方法的差异

| 方法 | 多证据 | 显式Agent通信 | 动态路由 | 伪未知 | 分歧驱动生成 |
|---|---:|---:|---:|---:|---:|
| Multi-Classifier Fusion 2022 | ✓ | × | × | × | × |
| MRDN 2023 | ✓ | × | × | 重构式 | × |
| MTPL 2025 | ✓ | × | × | × | × |
| Competition-Aware Boundary Extrapolation | ✓ | × | × | ✓ | × |
| Receiver Collaborative RFFI | 多Receiver | collaborative | 非本任务 | × | × |
| **Proposed Multi-Agent OS-SEI** | **✓** | **✓** | **✓** | **✓** | **✓** |

这张表后续可以扩展成论文Related Work核心表格。

---

# 35. 数据集设计

## WiSig

当前正式主实验：

```text
WiSig_OpenSet_1Day_1Rx
```

固定：

- 同一天；
- 同一Receiver；
- non-EQ；
- 不引入unseen domain。

规模：

```text
40 Known Tx
20 Unknown Tx

train        = 12720
val          = 1800
known_test   = 3680
unknown_test = 4800
open_test    = 8480
```

额外：

```text
open_test_full_unknown = 12780
```

仅作为Unknown-heavy压力测试。

---

## ORACLE

继续使用：

```text
KRI-16 demodulated
```

作为经典、较小规模OS-SEI基准。

它的作用不是提供最大规模，而是：

> 与上一版工作及现有OS-SEI方法进行直接比较。

---

# 36. 第一轮实验不要做Cross-Domain

暂时不要同时解决：

```text
Cross-Day
Cross-Rx
Domain Adaptation
Domain Generalization
```

当前论文任务始终保持：

> Multi-Agent Open-Set Specific Emitter Identification。

后续可以额外增加：

```text
Cross-Rx robustness
Cross-Day robustness
```

但作为附加鲁棒性实验，不改变主问题定义。

---

# 37. 必须做的Baseline

至少包括：

### Baseline 1：Closed-set SoftMax/MSP

Shared Encoder + classifier。

### Baseline 2：OpenMax

### Baseline 3：Prototype Distance / EVT

### Baseline 4：单一Open-Set Network

例如：

```text
Encoder + Prototype + threshold
```

### Baseline 5：Multi-Expert Fixed Fusion

```text
Identity
Prototype
Reconstruction
↓
固定平均
```

这是最重要的内部baseline。

### Baseline 6：Multi-Agent without Communication

有多个Agent，但不能相互通信。

### Baseline 7：Communication + Fixed Fusion

证明Communication本身的价值。

### Final

```text
Communication
+
Pseudo-Unknown Agent
+
Dynamic Router
```

---

# 38. 必须做的消融

建议按这个顺序。

| Variant | Identity | Prototype | Recon | PUG Agent | Communication | Router |
|---|---:|---:|---:|---:|---:|---:|
| A | ✓ | × | × | × | × | × |
| B | ✓ | ✓ | × | × | × | Fixed |
| C | ✓ | ✓ | ✓ | × | × | Fixed |
| D | ✓ | ✓ | ✓ | × | ✓ | Fixed |
| E | ✓ | ✓ | ✓ | ✓ | ✓ | Fixed |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | Dynamic |

这样可以回答：

1. Prototype Agent有没有价值；
2. Reconstruction Agent有没有价值；
3. Communication有没有价值；
4. Pseudo-Unknown Agent有没有价值；
5. Dynamic Router是不是最后的关键增益。

---

# 39. Pseudo-Unknown Agent专门消融

至少比较：

### Random perturbation

\[
z+\epsilon
\]

### Prototype outward

\[
z+\alpha\frac{z-c_y}{\|z-c_y\|}
\]

### High-distance boundary sample

只选离prototype较远Known。

### Disagreement seed + outward

### Disagreement seed + disagreement-gradient

### Final learnable PUG Agent

这组实验会非常重要。

因为它证明：

> 性能提升不是因为“生成了一点伪未知”，而是因为“Agent disagreement提供了更有意义的生成方向”。

---

# 40. Router专门消融

比较：

### Mean

固定平均。

### Handcrafted weight

人工设定。

### Global learned weight

整套数据使用同一个可学习权重。

### Sample-level Router

每个样本输出不同权重。

### Dual Router

Known和Unknown两套权重。

如果Dual Router最好，就可以很自然支撑论文设计。

---

# 41. Communication专门消融

比较：

```text
None
Concat + MLP
Mean message
Self-Attention
```

最终不要只证明Attention最好。

更重要的是可视化：

> Unknown出现时，各Agent之间attention关系发生了什么变化。

---

# 42. 评估指标

主指标：

- Known Accuracy；
- Unknown Recall / TUR；
- Macro-F1；
- AUROC；
- OSCR。

可增加：

- FPR@95TPR；
- Known Precision；
- calibration ECE；
- inference time；
- parameters；
- FLOPs。

---

# 43. 需要保存的可解释性结果

代码一定要保存：

## 1. Router weights

每个样本：

```text
alpha_identity
alpha_prototype
beta_prototype
beta_reconstruction
beta_boundary
```

---

## 2. Disagreement Score

Known vs Unknown分布。

如果方法合理：

```text
Known disagreement    低
Unknown disagreement  高
```

但不要假设一定如此，要实测。

---

## 3. Agent evidence

保存：

```text
softmax_confidence
entropy
prototype_distance
prototype_unknown
reconstruction_error
boundary_score
final_unknown_score
```

---

## 4. Pseudo-Unknown位置

t-SNE/UMAP：

```text
Known
Pseudo Unknown
Real Unknown（仅测试可视化）
```

必须明确：

Real Unknown只用于最终离线可视化和评估。

---

## 5. Router heatmap

横轴：

Agent。

纵轴：

样本。

可分：

```text
Easy Known
Hard Known
Unknown
```

观察Router是否真的表现不同。

---

# 44. 建议的代码目录

```text
Multi-Agent OS-SEI/
│
├── configs/
│   ├── wisig.yaml
│   ├── oracle.yaml
│   └── model.yaml
│
├── data/
│   ├── WiSig/
│   └── ORACLE/
│
├── datasets/
│   ├── wisig_dataset.py
│   ├── oracle_dataset.py
│   └── open_set_split.py
│
├── models/
│   ├── shared_stem.py
│   │
│   ├── agents/
│   │   ├── identity_agent.py
│   │   ├── prototype_agent.py
│   │   ├── reconstruction_agent.py
│   │   ├── pseudo_unknown_agent.py
│   │   └── boundary_agent.py
│   │
│   ├── communication.py
│   ├── disagreement.py
│   ├── router.py
│   ├── decision.py
│   └── multi_agent_ossei.py
│
├── losses/
│   ├── identity_loss.py
│   ├── prototype_loss.py
│   ├── reconstruction_loss.py
│   ├── pseudo_unknown_loss.py
│   ├── boundary_loss.py
│   └── router_loss.py
│
├── training/
│   ├── stage1_pretrain_experts.py
│   ├── stage2_train_communication.py
│   ├── stage3_generate_pseudo_unknown.py
│   ├── stage4_train_boundary.py
│   ├── stage5_train_router.py
│   └── stage6_joint_finetune.py
│
├── evaluation/
│   ├── metrics.py
│   ├── oscr.py
│   ├── open_set_eval.py
│   └── ablation_eval.py
│
├── visualization/
│   ├── plot_embedding.py
│   ├── plot_router_weights.py
│   ├── plot_agent_evidence.py
│   └── plot_disagreement.py
│
├── train.py
├── test.py
└── README.md
```

---

# 45. 推荐第一版网络参数

只是初始值，不作为最终论文固定参数。

```text
Input                    2 × 256
Shared channels          64 → 128
Shared feature           [B,128,64]

Identity embedding       128
Prototype embedding      128
Recon latent             128

Message dim              64
Communication heads      4

Boundary MLP             64 → 32 → 1
Router hidden            128

Known classes:
WiSig                    40
ORACLE                   根据固定split
```

---

# 46. 最小可行版本（MVP）

第一轮千万不要把所有创新一次性写完。

## V0

```text
Shared Stem
+
Identity Agent
+
Prototype Agent
```

先保证Known分类和prototype正常。

---

## V1

增加：

```text
Reconstruction Agent
```

输出三种Evidence。

---

## V2

增加：

```text
Communication
```

观察各Agent消息。

---

## V3

增加：

```text
Disagreement Score
+
简单Pseudo Unknown：
v_out + v_dis
```

暂时不用learnable Generator。

---

## V4

训练：

```text
Boundary Agent
```

---

## V5

增加：

```text
Dynamic Router
```

完整Multi-Agent版本。

---

## V6

再把Pseudo-Unknown Generator升级为learnable Agent。

这个开发顺序最稳。

---

# 47. 第一轮成功的判断标准

不要一开始要求SOTA。

先验证机制真的存在。

### 应看到：

1. Identity / Prototype / Reconstruction输出不是完全同质；
2. Known上的agent disagreement总体低于Unknown；
3. hard Known和Unknown拥有更高disagreement；
4. disagreement-driven pseudo-unknown比random perturbation有效；
5. fixed fusion < dynamic router；
6. 多Agent模型至少在Unknown Recall / OSCR / Macro-F1之一稳定优于single-agent；
7. Router weights对不同类型样本有明显差异。

如果这些都不存在：

> 应先修改Agent分工，而不是继续堆更复杂网络。

---

# 48. 可能失败的地方

## 48.1 Agent Collapse

三个Agent学成一样。

解决：

- private encoders；
- 不同loss；
- 不同Evidence；
- representation decorrelation。

---

## 48.2 Router Collapse

所有样本：

```text
Boundary = 0.99
其他Agent ≈ 0
```

解决：

- batch-level load balancing；
- warm-up；
- 先freeze experts；
- Router entropy regularization。

---

## 48.3 Pseudo Unknown过于简单

如果pseudo-unknown远离Known：

Boundary训练准确率100%，但real Unknown不好。

说明生成失败。

应观察：

```text
pseudo unknown距离Known是否适中
disagreement是否升高
real unknown是否与pseudo unknown overlap
```

---

## 48.4 Reconstruction对Unknown也重构很好

普通AE常出现。

解决：

- class-conditioned reconstruction；
- latent reconstruction；
- prototype-conditioned reconstruction；
- 限制decoder容量。

---

## 48.5 Communication没有作用

如果：

```text
communication on/off
```

几乎一样。

可能说明：

- Agent本身同质；
- message中没有真正互补证据；
- attention退化。

先检查Agent差异，再改Communication。

---

# 49. 论文叙事建议

不要把论文写成：

> “我们把Multi-Agent用于SEI。”

这太弱。

更好的动机：

### 第一层

OS-SEI存在多个不一致证据：

> 一个Unknown样本可能具有高分类置信度，但在原型几何和重构空间中明显异常。

### 第二层

现有方法通常：

> 单独使用某一种证据，或者做固定融合。

### 第三层

因此：

> 将开放集识别分解为多个认知角色，让不同Agent从互补空间审视同一RF信号。

### 第四层

尤其：

> Agent disagreement本身就是一种open-space risk signal。

### 第五层

因此设计：

> Disagreement-Guided Pseudo-Unknown Generation Agent主动探索多Agent意见分裂区域。

### 第六层

最后：

> Dynamic Router对每个样本自适应选择最可信证据。

这是整篇论文最完整的逻辑链。

---

# 50. 可以使用的论文标题方向

### 方向1

**Multi-Agent Collaborative Open-Set Specific Emitter Identification via Disagreement-Guided Pseudo-Unknown Learning**

中文：

**基于分歧引导伪未知学习的多智能体协同开放集辐射源识别**

---

### 方向2

**Evidence-Specialized Multi-Agent Learning for Open-Set Radio Frequency Fingerprint Identification**

中文：

**面向开放集射频指纹识别的多证据专家智能体协同学习**

---

### 方向3

**Dynamic Evidence Routing with Multi-Agent Disagreement for Open-Set Specific Emitter Identification**

中文：

**基于多智能体分歧与动态证据路由的开放集辐射源识别**

当前我更推荐第1或第3个方向。

---

# 51. 可写成论文Contribution的版本

后续根据实验结果再调整措辞。

### Contribution 1

提出一种面向OS-SEI的多智能体协同框架，将传统单模型开放集判断拆分为身份判别、原型几何、重构一致性、开放边界建模等互补认知任务，并通过显式Agent通信形成联合开放集证据。

### Contribution 2

提出Disagreement-Guided Pseudo-Unknown Generation Agent，以不同Expert Agent之间的类别、几何与重构意见分歧定位高风险Known边界，并主动生成具有高专家冲突度的pseudo-unknown表示，在不使用真实Unknown训练样本的条件下增强开放边界学习。

### Contribution 3

提出Sample-Adaptive Dual Dynamic Router，分别针对Known类别判别和Unknown风险评估动态选择最可信的Agent证据，避免固定平均或投票在某个专家失效时仍强制使用其输出。

### Contribution 4

在ORACLE和重新构造的纯开放集WiSig子集上验证方法，并通过Agent、Communication、Pseudo-Unknown Generation和Dynamic Routing的逐层消融分析各组件贡献。

---

# 52. 与“Mixture of Experts”的区别要提前准备

审稿人可能问：

> 这是不是MoE？

回答思路：

普通MoE通常是：

```text
Router
 ↓
选择不同Expert处理输入
 ↓
产生任务输出
```

我们的Agent系统进一步包含：

1. Agent具有不同任务目标，而非只是多个同构Expert；
2. Agent输出语义不同的开放集Evidence；
3. Agent之间在Router前显式通信；
4. Agent disagreement被显式计算；
5. disagreement用于驱动Pseudo-Unknown Agent；
6. Router融合的是多种开放集认知证据，而不是简单计算资源路由。

所以设计时必须真的做到这些，否则确实容易退化成MoE。

---

# 53. 是否需要强化学习？

第一版：

> **不建议。**

当前任务本质还是：

```text
supervised known classification
+
open-set rejection
```

强行使用PPO/Q-learning会大幅增加训练不稳定和工作量。

Multi-Agent不等于MARL。

当前每个Agent已经拥有：

- observation；
- private state；
- specialized objective；
- message；
- collaborative decision。

足以形成多智能体协同网络。

如果以后要增加MARL：

最适合放在：

> Pseudo-Unknown Generator Agent

让其Action是：

```text
direction
step size
seed selection
```

Reward来自：

```text
boundary hardness
agent disagreement
open-set validation performance
```

但这是后续扩展，不是第一篇必须内容。

---

# 54. 与现有代码迁移时的原则

你现有参考算法可以借：

- 数据加载；
- Backbone基础卷积；
- Prototype计算；
- OpenMax/EVT实现；
- 指标；
- 可视化；
- 训练框架。

但不要直接复制成创新：

- MFE-Net；
- 现有Triplet+Center完整流程；
- Competition-aware旧伪未知生成；
- 原OpenMax+Prototype fixed fusion。

真正新代码应集中在：

```text
agents/
communication.py
disagreement.py
pseudo_unknown_agent.py
router.py
multi_agent_ossei.py
```

---

# 55. 代码实现时每个Agent统一接口

建议所有Agent都实现统一API：

```python
class BaseAgent(nn.Module):
    def observe(self, shared_feature, context=None):
        ...

    def forward_private(self, obs):
        ...

    def build_message(self, state, evidence):
        ...

    def receive_messages(self, messages):
        ...

    def act(self, contextual_state):
        ...

    def get_loss(self, batch, outputs):
        ...
```

这样Codex实现会很清晰。

---

# 56. 每个Agent的统一返回结构

建议：

```python
{
    "embedding": ...,
    "evidence": {...},
    "message": ...,
    "local_pred": ...,
    "aux": {...}
}
```

例如Identity：

```python
{
    "embedding": z_i,
    "evidence": {
        "prob": p_i,
        "confidence": conf,
        "entropy": entropy,
        "margin": margin,
    },
    "message": msg_i,
    "local_pred": pred_i,
}
```

Prototype、Reconstruction都按同样格式。

这样Communication和Router不会依赖某个Agent内部实现。

---

# 57. 推荐保存的训练日志

每个epoch记录：

```text
loss_identity
loss_prototype
loss_reconstruction
loss_pug
loss_boundary
loss_router

known_acc
unknown_recall
macro_f1
auroc
oscr

mean_disagreement_known
mean_disagreement_pseudo_unknown

router_identity_weight
router_prototype_weight
router_reconstruction_weight
router_boundary_weight

pseudo_unknown_shell_distance
pseudo_unknown_disagreement
```

---

# 58. 当前方案最核心的一句话

> **不是让多个Agent对同一个问题重复投票，而是让不同Agent分别判断“像谁、离已知空间多远、能否被已知空间解释、是否处于开放边界”，再利用它们之间的分歧主动探索未知空间，并通过动态路由完成样本级协同开放集决策。**

如果代码实现最终真正体现了这句话，这个Multi-Agent框架就比较站得住。

---

# 59. 推荐的第一阶段编码目标

先实现：

```text
Shared Stem
Identity Agent
Prototype Agent
Reconstruction Agent
Disagreement Calculator
```

然后只验证：

```text
Known vs Unknown
```

的：

- Identity confidence；
- Prototype distance；
- Reconstruction error；
- Agent disagreement；

分布是否有差别。

如果这些基础Evidence没有互补性，暂时不要继续写Pseudo-Unknown Agent和Router。

只有确认：

> 不同Agent真的观察到了不同现象，

再继续实现：

```text
Pseudo-Unknown Generation Agent
Boundary Agent
Communication
Dynamic Router
```

这会显著减少无效开发。

---

# 60. 最终完整方法链

```text
                         RF I/Q
                           │
                           ▼
                      Shared Stem
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
 Identity Agent      Prototype Agent      Reconstruction Agent
      │                    │                    │
      ├──── confidence     ├──── geometry       ├──── consistency
      │                    │                    │
      └──────────────┬─────┴─────┬──────────────┘
                     │           │
                     ▼           ▼
                 Agent Messages
                     │
                     ▼
              Communication Module
                     │
                     ▼
             Cross-Agent Disagreement
                     │
            ┌────────┴─────────┐
            │                  │
            ▼                  ▼
 Pseudo-Unknown Agent      Context States
  (training mainly)             │
            │                   │
            ▼                   ▼
   Pseudo Unknown       Open-Space Boundary Agent
            │                   │
            └──────────┬────────┘
                       ▼
            Dynamic Dual Router Agent
                       │
               ┌───────┴───────┐
               ▼               ▼
           Known Tx          Unknown
```

---

# 61. 参考文献 / 必读工作

1. Zhao, Y., Wang, X., Lin, Z., Huang, Z.  
   **Multi-Classifier Fusion for Open-Set Specific Emitter Identification.**  
   Remote Sensing, 2022, 14(9), 2226.  
   DOI: 10.3390/rs14092226

2. Tan, K. et al.  
   **Dynamic open set specific emitter identification via multi-channel reconstructive discriminant network.**  
   IET Radar, Sonar & Navigation, 2023, 17(5): 813–829.  
   DOI: 10.1049/rsn2.12380

3. Shen, G. et al.  
   **Towards Receiver-Agnostic and Collaborative Radio Frequency Fingerprint Identification.**  
   IEEE Transactions on Mobile Computing.  
   DOI: 10.1109/TMC.2023.3340039

4. Ma, Z., Fang, S., Fan, Y.  
   **Open-Set Radio Frequency Fingerprint Identification Method Based on Multi-Task Prototype Learning.**  
   Sensors, 2025, 25(17), 5415.  
   DOI: 10.3390/s25175415

5. Liang, L., Ye, H., Li, G. Y.  
   **Spectrum Sharing in Vehicular Networks Based on Multi-Agent Reinforcement Learning.**  
   IEEE Journal on Selected Areas in Communications, 2019, 37(10): 2282–2292.  
   DOI: 10.1109/JSAC.2019.2933962

6. Vangaru, S. et al.  
   **A Multi-Agent Reinforcement Learning Testbed for Cognitive Radio Applications.**  
   Multi-Agent RFRL Gym相关工作，2025。

7. Song, J. et al.  
   **Multi-Agent Visual Reasoning for Out-of-Distribution Detection in Complex Road Environments.**  
   IEEE Access, 2025, 13: 188198–188216.  
   DOI: 10.1109/ACCESS.2025.3627794

8. 你上一版方案 / 论文：  
   **Competition-Aware Boundary Extrapolation for Open-Set Specific Emitter Identification.**  
   重点参考其中：prototype-structured embedding、boundary sample mining、feature-level pseudo-unknown、supervised calibration。  
   当前新方案应避免直接复用其“prototype competition → outward extrapolation”作为核心创新，而改为“cross-agent disagreement → Pseudo-Unknown Agent exploration”。

---

# 62. 当前建议的研究边界

本项目第一版明确不做：

```text
× Cross-domain OSR
× Domain adaptation
× Cross-day generalization
× Cross-receiver generalization
× Unknown clustering
× Incremental learning
× LLM Agent
× 全流程MARL
```

只集中做：

```text
✓ Open-Set SEI
✓ Evidence-specialized Agents
✓ Agent Communication
✓ Disagreement
✓ Pseudo-Unknown Generation Agent
✓ Dynamic Routing
```

把这一条主线做到扎实，比同时铺开多个问题更有价值。

---

# 63. 最终建议

当前最值得先实现的版本不是“六个Agent一次写全”，而是：

```text
Step 1
Identity + Prototype + Reconstruction

Step 2
验证三者Evidence是否互补

Step 3
显式Communication

Step 4
Disagreement Score

Step 5
Pseudo-Unknown Generator Agent

Step 6
Boundary Agent

Step 7
Dynamic Dual Router

Step 8
Full OS-SEI evaluation
```

**Pseudo-Unknown Generation Agent可以成为整个工作的核心Agent之一。**

它与上一版伪未知生成思想有天然继承关系，但新的关键不再是单纯几何边界，而是：

> **利用多个专家Agent之间的认知分歧主动寻找并探索开放空间风险区域。**

这也是当前整个方法最自然、最容易形成论文主线的创新方向。
