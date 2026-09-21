# Stage-8 多智能体开放集辐射源识别重构指导文档（交付 Codex）

> 仓库：`54268/agent`  
> 当前主线：Stage-7 Conditional Consultation  
> 本文目的：指导 Codex 停止继续对 Stage-7 的 B2、阈值、Router loss、communication loss 做局部调参，转而从 **Agent 的定义、私有观测、私有任务、通信语义和最终决策机制** 上重构下一阶段。  
> 目标不是把多视图/多分类器包装成多智能体，而是让多智能体在开放集辐射源识别中产生 **不可替代、可验证、可泛化的协作价值**。  
> 本路线明确 **不采用“多接收机 = 多 Agent”** 的方向；多接收机协同属于另一条研究路线，不是本阶段目标。

---

## 0. 给 Codex 的一句话任务定义

不要继续优化“两个都能独立分类的专家，最后看谁 confidence 高”。

下一阶段要构造这样的系统：

> **Agent A 在自己的信息域内遇到无法解决的问题 → 它知道自己缺什么证据 → 向另一个拥有不同私有信息/能力的 Agent 发起有语义的请求 → 对方返回 A 无法自行计算的证据 → 这条证据直接改变具体候选类别或 Known/Unknown 判断。**

我们需要证明的是：

> **没有任何单个 Agent 拥有完成最终开放集判决所需的全部信息。协作不是“加权平均”，而是“补齐缺失证据”。**

---

# 1. 当前项目已经做到什么

Stage-7 并不是完全失败。它已经证明了一件非常重要的事情：

**两个现有 Agent 之间存在真实互补性。**

当前严格 nested G0 诊断中：

- WiSig：
  - 最佳单 Agent `H = 0.47429`
  - Agent-oracle 相对最佳单 Agent `ΔH = +0.13863`
  - `ΔOSCR = +0.02919`
- ORACLE：
  - 最佳单 Agent `H = 0.80864`
  - Agent-oracle `ΔH = +0.07573`
  - `ΔOSCR = +0.08216`
- 两个数据集均存在超过 5% 的双向 unique rescue。

这说明：

> A 和 B 并不是完全重复，它们确实会在不同样本上互相“救回来”。

因此不应否定多 Agent 方向。

但是 Stage-7 同时证明：

> **当前“谁该救谁”的规律不能跨类别划分稳定迁移。**

例如：

- WiSig 某个 B2 候选：
  - `H = 0.64632`
  - `ΔH = +0.02997`
  - 3/4 inner folds 正增益
- 同一个策略放到 ORACLE：
  - `H = 0.57459`
  - `ΔH = -0.10311`

V6 的 competence head 甚至把 WiSig Prototype 的 correctness Brier score 从 `0.06630` 改善到了 `0.02736`，但仍无法稳定回答：

> “这个样本到底应该信 Waveform，还是信 Prototype？”

所以当前问题已经不应继续解释为：

- reliability 没校准好；
- B2 loss 不合适；
- 阈值不合适；
- Router 训练轮数不够；
- communication loss 权重没调好。

**当前核心问题是 Agent 本身的角色定义仍然不够异质，导致救援关系主要依赖具体类别/episode，而不是稳定的能力边界。**

---

# 2. 当前 Stage-7 的核心结构性问题

## 2.1 Prototype Agent 已经是一个“多视图超级 Agent”

当前 `src/maros_stage7/agents.py` 中的 `EnrollmentPrototypeAgent` 并不是一个狭义的“Prototype/Memory Agent”。

它内部已经同时处理多种 sensor/view：

- raw
- spectral
- envelope / phase
- difference-IQ
- complex-IQ

并且包含：

- 多套 stem / encoder；
- ComplexIQEncoder；
- `view_gate`；
- 每个 view 的 logits；
- 每个 view 的 prototypes；
- 每个 view 的 distances；
- per-view tail statistics；
- 全局 prototype/tail；
- unknown head；
- competence head。

也就是说，一个 Prototype Agent 内部已经自己做了：

> 多视图观察  
> → 多专家表示  
> → 动态 view selection  
> → 多视图融合  
> → 分类  
> → open-set rejection

这会产生一个非常严重的研究问题：

> **外层虽然叫两个 Agent，内层实际上已经有一个“超级多视图专家系统”。**

于是 Waveform Agent 再与它交互时，很容易变成：

> 一个时域分类器  
> VS  
> 一个更复杂的多视图分类器

最后再由 Router/B2 看谁更可信。

这本质上仍然很像：

> multi-view ensemble / mixture-of-experts / multi-classifier fusion

而不是必须通过通信才能完成任务的多智能体系统。

### Stage-8 必须改

Prototype Agent 不能再同时持有 raw / spectral / phase / difference / complex 等大多数信息源。

**一个 Agent 不得成为内部 multi-view super-agent。**

---

## 2.2 两个现有 Agent 虽然网络不同，但“工作内容”过于相同

目前 Waveform 与 Prototype 最终都在完成类似的全任务：

```text
输入
  ↓
K-way class logits
  ↓
unknown score
  ↓
reliability
  ↓
top1 / top2
```

也就是说：

- 两个 Agent 都做完整分类；
- 两个 Agent 都做 Unknown 判断；
- 两个 Agent 都输出置信度；
- 两个 Agent 都试图成为一个独立完整 OSR 系统。

所以它们最自然的协作方式就会退化为：

> A 分类一次  
> B 分类一次  
> 谁 confidence 高就多信谁

这就是目前最需要摆脱的范式。

### Stage-8 的原则

Agent 可以保留基本独立预测用于能力审计，但它的**核心私有任务必须不同**。

需要追求：

> Agent A 擅长的问题，从 Agent B 的允许观测中就无法直接解决；  
> Agent B 擅长的问题，从 Agent A 的允许观测中也无法直接解决。

这里的“无法解决”首先是**信息边界上的结构性无法访问**，而不是仅仅“某个网络暂时准确率低”。

---

## 2.3 当前 Router 主要看的是“结果置信度”，而不是“为什么这个 Agent 此时可靠”

`src/maros_stage7/router.py` 的 `public_team_features()` 当前主要使用：

- top1 probability
- top2 probability
- margin
- entropy
- energy
- unknown score
- reliability
- JS divergence
- top1 disagreement
- unknown gap

这些特征大部分是在描述：

> “这个 Agent 有多自信？”

而不是：

> “这个样本当前存在什么物理/信号学问题？”  
> “哪一种专长能解决这个问题？”

于是 Router 被迫从 confidence/uncertainty 反推：

> “Waveform 和 Prototype 到底谁真的对？”

这很容易依赖具体类别、具体 decision boundary、具体 episode。

这与 Stage-7 的实际失败完全一致：

> reliability 可以变得更 calibrated，  
> 但“谁比谁更值得信”仍不能跨类迁移。

### Stage-8 必须改

Coordinator/Router 后续不能主要学习：

```text
confidence A vs confidence B
```

而应该学习：

```text
当前缺失的是哪一种证据？
查询某个 Agent 后，预计能降低多少决策损失？
```

即：

> **Value of Information / Missing-Evidence Routing**

而不是 Expert Selection。

---

## 2.4 当前 SemanticAdjudicator 仍然会退化成“重新给两个专家加权”

Stage-7 的 `ChallengePacket` 已经有不错的语义：

- `stance`
- `alternative_class`
- `signed_pair_evidence`
- `open_tail_evidence`
- `reliability`

这比直接传 embedding 好。

但是当前 `SemanticAdjudicator.forward()` 中，分类侧的主要操作仍然是：

```text
收到 challenge
    ↓
调整 waveform/prototype mixture weight
    ↓
重新加权两个 Agent 的整套 class logits
```

也就是说，即使 responder 已经返回了：

```text
alternative_class = Tx_12
signed_pair_evidence = ...
```

最终 Known 分类仍主要是：

```text
w1 * waveform_logits + w2 * prototype_logits
```

**`alternative_class` 没有直接成为“候选类别分数更新”的核心对象。**

这会让“语义咨询”重新退化成：

> 动态 MoE 权重调整。

### Stage-8 必须改

如果 Agent B 回答：

> “我不支持 Tx7，我支持 Tx12，并且证据强度为 +e”

那么最终决策必须直接发生类似：

```text
score(Tx7)  -= λ * e
score(Tx12) += λ * e
```

而不是只做：

```text
减少 A 的整体权重
增加 B 的整体权重
```

通信必须改变**具体候选假设**。

---

## 2.5 当前系统证明的是“evidence complementarity”，还没有证明“multi-agent collaboration”

Stage-1 到 Stage-7 已经多次证明：

- 多种证据组合能提高 OSR；
- 单 Agent 错误并不完全重合；
- 某些数据集上通信会明显影响结果。

但这仍不足以回答审稿人最可能问的一句话：

> “为什么不用一个更大的多视图网络，把所有视图一次性输入，然后统一训练？”

下一版的实验设计必须主动回答这个问题。

---

# 3. Stage-8 的核心研究命题

Stage-8 不再把研究问题定义为：

> “怎么让多个 classifier 更好地融合？”

而定义为：

> **Heterogeneous Private-Task Agents for Open-Set SEI**

核心假设：

> 不同辐射源证据存在不同的物理来源、不同的失效模式和不同的可观测条件。  
> 将这些能力放入具有严格私有观测边界的 Agent 中，并通过候选条件化语义咨询交换不可替代证据，可以比等容量单体多视图网络和无通信多 Agent 系统获得更稳定的开放集增益。

---

# 4. Stage-8 建议使用 4 个 Agent

不要为了“多”而堆 6～8 个 Agent。

当前建议：

1. Temporal Fingerprint Agent
2. Spectral Hardware Fingerprint Agent
3. Enrollment Memory Verification Agent
4. Open-Space Consistency Auditor

外加：

5. Coordinator / Adjudicator

**第 5 个不是 Agent。**

---

# 5. Agent A：Temporal Fingerprint Agent

## 5.1 私有观测

只允许：

```text
raw complex I/Q temporal sequence
```

禁止直接访问：

- FFT/STFT；
- spectrum；
- frequency-domain handcrafted view；
- Prototype bank；
- tail memory；
- 其他 Agent embedding；
- 其他 Agent private state。

建议通过 API 类型和单元测试强制隔离，而不是靠注释约定。

例如：

```python
TemporalObservation(iq=...)
```

Temporal Agent 的 `forward()` 根本不接受 `SpectralObservation` 或 enrollment memory。

---

## 5.2 私有能力

Temporal Agent 专门建模：

- temporal transient；
- time-local amplitude dynamics；
- phase continuity；
- temporal nonlinear distortion；
- short-range waveform fingerprint；
- complex temporal correlation。

---

## 5.3 核心任务不能再只是 K-way classifier

应该增加/强化 candidate-conditioned verification：

```text
TemporalVerify(x, class_c)
    → support score
```

训练样本：

### Positive

```text
(sample x from Tx7, candidate Tx7)
```

### Hard negative

```text
(sample x from Tx7, candidate nearest confusing Tx12)
```

### Class-held-out unknown

在严格 inner LCO episode 中作为 Unknown。

Temporal Agent 最终应该能回答：

> “从时域证据看，当前样本是否支持候选 emitter c？”

而不只是：

> “我的 softmax top1 是 c。”

---

# 6. Agent B：Spectral Hardware Fingerprint Agent

## 6.1 私有观测

只允许频域/频率相关表征，例如：

- FFT / STFT；
- PSD；
- spectral envelope；
- frequency-local phase statistics；
- 频谱不对称；
- 频谱再生相关结构；
- CFO / phase-noise / IQ-imbalance 相关表征。

**不允许访问完整 raw temporal sequence。**

建议由 `Stage8System` 中的 observation builder 在 Agent 外部完成 view 构造：

```python
spectral_obs = build_spectral_observation(iq)
```

之后只把 `SpectralObservation` 交给 Agent B。

这样可以在代码层证明：

> Agent B 没有 raw I/Q 能力。

---

## 6.2 私有能力

Agent B 专门回答：

> “从频域硬件失真证据看，候选 emitter c 是否成立？”

同样采用 candidate-conditioned verifier：

```text
SpectralVerify(x_freq, class_c)
    → support score
```

---

## 6.3 A/B 应形成真正的互斥能力边界

理想案例：

```text
A：
Tx7 与 Tx12 的时域瞬态非常相似，我无法稳定区分。

A → B：
请验证 Tx7 vs Tx12 的频域硬件证据。

B：
Tx7 的 spectral asymmetry / impairment pattern 明显更匹配。
支持 Tx7。
```

反方向也必须存在：

```text
B：
Tx3 与 Tx9 频域非常接近。

B → A：
请验证 Tx3 vs Tx9 的 temporal transient。

A：
时域证据明确支持 Tx9。
```

只有出现这种双向“你有我没有”的 rescue，才值得继续训练通信和路由。

---

# 7. Agent C：Enrollment Memory Verification Agent

## 7.1 它的私有资源不是另一种普通 view

Agent C 私有持有：

- Known emitter enrollment prototypes；
- intra-class distribution；
- class-specific distance/tail statistics；
- nearest-competitor relations；
- robust center / covariance / tail；
- Known-only historical variability。

A/B 不允许直接读这些 memory。

---

## 7.2 它不负责“先猜一个类别”

它主要回答：

> “你说是 Tx7，那么它真的像一个正常的 Tx7 吗？”

这是开放集任务中特别重要的问题：

> 最近的 Known 类，不等于真正属于这个 Known 类。

示例：

```text
A：最像 Tx7。
B：也稍微支持 Tx7。
C：但是 query 与 Tx7 enrollment distribution 的 tail distance 极端异常，
   它只是“所有 Known 中最像 Tx7”，并不像“真正的 Tx7”。
C：UNKNOWN_SUSPECT。
```

---

## 7.3 Memory Agent 输入边界

建议它只接收：

- proposer 给出的 candidate IDs；
- 一个经过允许的 query registration descriptor；
- 它自己持有的 enrollment memory。

它不能访问 A/B 的完整 embedding、full logits 或 private state。

如果需要 query descriptor，必须单独定义：

```python
RegistrationObservation
```

并明确其来源、维度和信息权限。

不要简单把 A/B 的 private embedding 传给它。

---

# 8. Agent D：Open-Space Consistency Auditor

## 8.1 为什么需要第四 Agent

前三个 Agent 主要在回答：

> “候选身份是否得到某类证据支持？”

但开放集还需要一种不同能力：

> “当前判断是不是一个脆弱的、开放空间中的伪自信判断？”

因此建议第四个 Agent 不做 K-way 分类，而做：

> **identity-preserving perturbation consistency audit**

---

## 8.2 它的私有操作

对输入实施受控、身份保持的轻微 counterfactual probes，例如：

- 小幅 time shift；
- 合法 phase rotation；
- 轻微 additive noise；
- 轻微局部 masking；
- 受控频带扰动。

具体扰动必须经过已知样本安全性审计。

它观察：

```text
原输入判断
vs
轻微扰动后的身份/拒识稳定性
```

---

## 8.3 它的输出不是类别

建议只输出：

- consistency score；
- instability score；
- open-space suspicion；
- abstain / stable / unstable。

例如：

```text
A/B 都认为 Tx7
但轻微合法扰动后：
Tx7 → Tx18 → Tx3

D：
identity instability 很高
UNKNOWN_SUSPECT
```

这样 D 的任务与 A/B/C 完全不同。

---

# 9. Coordinator 不是第五个 Agent

以下都不要叫 Agent：

- Router
- B2
- threshold
- calibration
- OpenMax/EVT
- adjudicator
- fusion service

统一放在：

```text
Coordinator / Decision Service
```

它不拥有辐射源识别私有能力。

它只负责：

1. 当前是否缺证据；
2. 缺哪一种证据；
3. 是否值得支付通信成本；
4. 请求哪个 Agent；
5. 将返回的 evidence certificate 应用到当前候选假设。

---

# 10. 下一版通信协议：不要传“confidence”，要传 Evidence Certificate

建议 Stage-8 新建消息协议。

## 10.1 QueryPacket

至少包含：

```text
sender
candidate_a
candidate_b
query_type
reason_code
local_pair_support
local_open_risk
active_mask
bit_cost
```

`query_type` 示例：

```text
VERIFY_TEMPORAL_PAIR
VERIFY_SPECTRAL_PAIR
VERIFY_ENROLLMENT_COMPATIBILITY
AUDIT_OPEN_SPACE_STABILITY
```

---

## 10.2 EvidenceCertificate

建议包含：

```text
sender
receiver
candidate_a
candidate_b
stance
signed_candidate_evidence
alternative_class
domain_quality
open_risk
reliability
abstain
bit_cost
```

其中 `stance`：

```text
SUPPORT_A
SUPPORT_B
UNKNOWN_SUSPECT
ABSTAIN
```

---

# 11. 最关键改动：消息必须直接修改候选类别证据

Stage-8 禁止继续把 consultation 的核心效果写成：

```text
w_A * logits_A + w_B * logits_B
```

建议维护：

```text
Candidate Evidence Table
```

例如：

```text
Tx7:
  temporal   +1.8
  spectral   +2.3
  memory     +0.7
  audit      -0.1

Tx12:
  temporal   +1.2
  spectral   -1.9
  memory     +0.2
  audit      -0.3
```

最终类别分数可以是经过学习标定的 candidate-level evidence aggregation：

\[
S(c)=
\lambda_T E_T(c)
+\lambda_F E_F(c)
+\lambda_M E_M(c)
+\lambda_A E_A(c)
\]

重点不是公式必须固定为相加，而是：

> **响应 Agent 的证据必须绑定到具体 candidate。**

如果 responder 返回：

```text
alternative_class = Tx12
```

那么 Tx12 的分数必须被显式更新。

需要新增单元测试：

> 保持其它输入完全相同，只改变 `alternative_class`，最终 candidate score / prediction 必须有可解释变化。

如果这个测试不成立，说明系统仍然是假语义通信。

---

# 12. 不要马上训练 Router：先证明“救援关系可预测”

Stage-7 最大教训：

> 有 oracle headroom ≠ Router 能学会什么时候找谁。

因此 Stage-8 必须按顺序进行。

---

# 13. Stage-8 实验阶段设计

## G0：Capability Isolation

先不看最终 H。

必须验证代码层的能力隔离：

### Temporal Agent

- 不能调用 FFT/STFT；
- 不能读取 prototype memory；
- 不能读 Spectral private state。

### Spectral Agent

- API 不接收 raw I/Q；
- 只能接收 `SpectralObservation`。

### Memory Agent

- 不接收 raw waveform；
- 不接收 spectrum；
- 不接收 A/B full embedding；
- 只访问允许的 registration descriptor + private enrollment memory。

### Auditor

- 不输出 K-way logits；
- 只输出 consistency/open-space certificate。

### Coordinator

- 不访问任何 Agent private tensor；
- 不访问 hidden embedding/token sequence；
- 只能读取 public diagnostic + message。

必须增加自动化 capability isolation tests。

---

## G1：Local Specialization + Rescue Structure

不训练通信 Router。

先独立训练各 Agent。

必须报告：

- 各 Agent local Known capability；
- 各 Agent local Unknown capability；
- pairwise unique rescue；
- Agent-oracle H；
- `ΔH_oracle`；
- 哪一类失败由哪一种 Agent 能救；
- rescue 是否与物理/信号质量因素相关。

推荐开发闸门：

1. A↔B 双向 unique rescue 每个数据集均应明显非零，优先要求 ≥5%；
2. Best-single → Agent-oracle 有有意义的 headroom，目标 `ΔH_oracle ≥ 0.05`；
3. rescue pattern 不能只由 class ID 决定；
4. WiSig 与 ORACLE 使用同一方法字段，不允许 dataset-specific branch。

如果新的 Agent 分工反而没有互补性：

> 停止，不进入通信阶段。

---

## G1.5：Cross-Class Rescue Predictability

这是 Stage-8 新增的核心诊断。

定义真实标签：

```text
Temporal wins
Spectral wins
Memory rescue needed
Auditor rescue needed
none can rescue
```

这些标签必须由 inner held-out class episode 中的真实反事实损失得到。

然后只用：

- class-invariant domain quality descriptors；
- request-conditioned evidence；
- 不含 class ID 的信号质量描述；

预测：

> “当前应该向谁咨询？”

重点是：

> 在一些类别上训练，在未见类别划分上验证。

推荐 development gate：

- Agent-win / rescue prediction macro-AUROC 目标 ≥0.65；
- 至少 3/4 inner folds 为正迁移；
- 不能依赖 dataset name；
- 不能使用 outer/formal Unknown。

如果仍然接近随机：

> **不要训练 Router。重新设计 Agent 私有任务。**

---

## G2：Forced Consultation Value

此阶段仍然不要让 Router 学会挑动作。

对需要评估的样本强制执行：

```text
A → B
B → A
A/B → C
conflict → D
```

比较：

1. Same agents, no communication
2. Equal-bandwidth independent parallel message
3. Proposal-conditioned consultation

必须回答：

> 同样的 Agent，同样的参数，只因为 responder 看到了对方的具体问题并返回针对性证据，结果是否变好了？

核心指标：

\[
\Delta H_{comm}
=
H_{forced\ consultation}
-
H_{same\ agents,\ messages\ off}
\]

建议进入下一阶段的最低目标：

- `ΔH_comm ≥ +0.01`
- 至少 3/4 inner folds 为正
- WiSig 与 ORACLE 都成立
- OSCR 不应系统性下降

若 forced consultation 都不能稳定胜过 no-communication：

> 不训练 Coordinator。

---

## G3：Value-of-Information Coordinator

只有 G2 通过后再训练 Coordinator。

Coordinator 不预测：

```text
哪个 Agent confidence 最大？
```

而预测：

\[
VOI(j)
=
E[
L_{\text{before}}
-
L_{\text{after querying Agent }j}
]
-
\lambda C_j
\]

其中：

- `L`：完整 open-set decision loss；
- `C_j`：通信/计算代价。

动作空间可以是：

```text
STOP
ASK_TEMPORAL
ASK_SPECTRAL
ASK_MEMORY
ASK_AUDITOR
```

如果需要两轮咨询，再在 G3 后半段增加：

```text
ASK_SECOND_AGENT
```

不要一开始就上复杂 MARL / PPO / QMIX / CommFormer。

---

## G4：Formal Evaluation

通过前面所有 gate 后才运行：

```text
3 partition seeds × 5 folds
```

稳定后再解锁正式 Unknown。

保持 Stage-7 的优秀部分：

- strict nested LCO；
- inner Agents 从头训练；
- formal Unknown 不进入训练/选轮/阈值；
- 同初始化公平比较；
- provenance；
- counterfactual action loss；
- audit manifest；
- gate report。

---

# 14. 必须保留的 Stage-7 资产

不要全部推倒重来。

应该复用：

- 数据加载与 provenance；
- outer/inner LCO 逻辑；
- formal unknown leakage protection；
- same-seed comparison；
- PublicDecision / audit 思想；
- message bit-cost accounting；
- counterfactual action enumeration；
- H / OSCR / AUROC 统计；
- gate system；
- 结果 manifest；
- formal Unknown lock。

Stage-8 应新增：

```text
src/maros_stage8/
```

不要直接继续把 Stage-7 改成 V8/V9。

Stage-7 要保留为历史基线。

---

# 15. 建议 Stage-8 目录结构

```text
src/maros_stage8/
├── contracts.py
├── observations.py
├── temporal_agent.py
├── spectral_agent.py
├── memory_agent.py
├── consistency_agent.py
├── evidence.py
├── coordinator.py
├── dialogue.py
├── system.py
├── training.py
├── evaluation.py
├── gates.py
└── audit.py
```

测试：

```text
tests/stage8/
├── test_capability_isolation.py
├── test_observation_boundaries.py
├── test_no_private_state_leak.py
├── test_candidate_specific_message_effect.py
├── test_alternative_class_changes_decision.py
├── test_stop_identity.py
├── test_formal_unknown_lock.py
├── test_inner_lco_class_isolation.py
└── test_message_shuffle_intervention.py
```

---

# 16. H、ΔH 在 Stage-8 中如何使用

## 16.1 H-score

当前项目中的核心综合指标：

\[
H =
\frac{
2 \cdot KnownAcc \cdot UnknownRecall
}{
KnownAcc + UnknownRecall
}
\]

作用：

同时防止两种“作弊”：

- 所有东西都判 Unknown；
- 所有东西都硬分到 Known。

开放集 SEI 必须同时：

- Known 识别正确；
- Unknown 成功拒识。

---

## 16.2 ΔH 必须写清“相对谁”

Stage-8 至少统一报告以下三个：

### Collaboration Gain

\[
\Delta H_{collab}
=
H_{FullMultiAgent}
-
H_{BestSingleAgent}
\]

回答：

> 多智能体整体比最强单 Agent 好多少？

### Communication Gain

\[
\Delta H_{comm}
=
H_{MessagesOn}
-
H_{SameAgents,MessagesOff}
\]

回答：

> 真正通信本身贡献了多少？

### Oracle Headroom

\[
\Delta H_{oracle}
=
H_{AgentOracle}
-
H_{BestSingleAgent}
\]

回答：

> 理论上还有多少多 Agent 协作空间？

如果：

```text
ΔH_oracle 很大
ΔH_comm 很小
```

结论必须写成：

> Agent 有互补性，但通信机制没有把互补性利用出来。

不能把它包装成“多智能体成功”。

---

# 17. 新增的多智能体专属指标

不能再只看最终 AUROC/H。

至少增加：

## 17.1 Unique Rescue Rate

A 错而 B 对的比例。

## 17.2 Bidirectional Rescue

必须验证：

- A rescues B
- B rescues A

而不是永远一个强 Agent 救另一个弱 Agent。

## 17.3 Consultation Change Rate

发生通信后，最终判断改变的比例。

如果几乎从不改变：

> 通信可能只是装饰。

## 17.4 Beneficial Change Rate

改变的决策中，有多少由错变对。

## 17.5 Harm Rate

通信后由对变错的比例。

## 17.6 Rescue Precision

Coordinator 发起一次咨询后，真正带来正收益的比例。

## 17.7 Cross-Class Agent-Win Predictability

这是 Stage-8 的核心新指标。

必须评价：

> 在未见类别 episode 上，能否预测“哪个 Agent 的独特能力此时有价值”。

---

# 18. 必须加入的强基线

下一版如果只与单 Agent 比是不够的。

必须包含：

1. Best Single Agent
2. Equal-capacity No-Communication Multi-Agent
3. Equal-parameter Single-Body Multi-View Model
4. Static Ensemble / Mean Fusion
5. Stage-7 B2 / conditional consultation baseline
6. 强 open-set baseline（复用仓库中已有 OpenMax / PCBM / PUG 等公平实现）
7. Full Stage-8 Multi-Agent

最关键的是第 3 个：

> **Single-Body Multi-View Equal-Parameter Baseline**

因为这直接回答：

> “为什么不把所有视图塞进一个模型？”

---

# 19. 必须做的因果干预实验

只有最终指标提高，不足以证明 multi-agent collaboration。

正式结果至少要包含：

## Messages Off

保持 Agent 完全相同，只关闭通信。

预期：

```text
Full > Messages Off
```

## Message Shuffle

打乱 responder 的消息归属。

如果性能不下降：

> 模型可能没有真正使用消息语义。

## Wrong Responder

把本应问 Spectral 的请求故意交给不具备该能力的 Agent。

应该下降。

## Wrong Candidate Pair

保留 responder，但把查询 pair 换成错误 candidate pair。

应该下降。

## Alternative-Class Intervention

只改变 `alternative_class`。

最终具体类别分数必须随之改变。

## Drop One Specialist

删除某个 Agent 后，在它专属的 failure subset 上性能应明显下降。

这些实验比“通信层 attention 可视化”更能证明 Agent 不可替代性。

---

# 20. 不允许 Codex 继续做的事情

在 G0/G1/G2 未通过前，暂停：

- B2 loss 网格；
- threshold 网格；
- class-adaptive threshold；
- communication-loss 权重扫描；
- Router hidden size 扫描；
- Router LR 扫描；
- PPO；
- QMIX；
- CommFormer；
- 大规模 multi-seed formal Unknown；
- 为 WiSig 与 ORACLE 写不同逻辑；
- 为了提高数字把 forbidden view 偷回某个 Agent。

特别禁止这种“修复”：

> Spectral Agent 不够强 → 再把 raw IQ 加进去。  
> Temporal Agent 不够强 → 再把 spectrum 加进去。  
> Memory Agent 不够强 → 再给它 full logits/embedding。

这会再次回到 super-agent。

---

# 21. Codex 的执行顺序

请严格按以下顺序工作，不要跳步。

### Step 1：冻结 Stage-7

- 不删除；
- 不覆盖；
- 生成 Stage-7 final diagnostic summary；
- 作为 Stage-8 baseline。

### Step 2：建立 `maros_stage8`

优先迁移 protocol/audit/LCO/leakage-safe 基础设施。

### Step 3：先实现 Observation Contracts

先写：

```text
TemporalObservation
SpectralObservation
RegistrationObservation
AuditObservation
```

再写 Agent。

不要先复制 Stage-7 Agent 再慢慢删能力。

### Step 4：实现 Temporal / Spectral 两个硬隔离 Agent

先只做 A/B。

### Step 5：实现 candidate-conditioned verifier training

不要先做 Router。

### Step 6：跑 G1 complementarity

如果 A/B 没有稳定双向 rescue：

> 停止并重新设计 A/B。

### Step 7：实现 Memory Agent

验证它是否主要救“closest-known but actually unknown”的错误。

### Step 8：实现 Consistency Auditor

先在 Known 上做身份保持安全审计，确保扰动不会系统性破坏合法身份。

### Step 9：实现 EvidenceCertificate + Candidate Evidence Table

先不训练 Coordinator。

### Step 10：Forced Consultation

证明 request-conditioned communication 本身有效。

### Step 11：只有 G2 通过才实现 Coordinator

训练 VOI，而不是 confidence winner。

### Step 12：完成干预实验

Messages off / shuffle / wrong responder / wrong pair / drop specialist。

### Step 13：与 equal-parameter single-body multi-view baseline 比较

如果 Full Multi-Agent 仍不能稳定超过这个基线：

> 不得声称多智能体具有不可替代优势。

### Step 14：最后才进入 formal multi-seed

---

# 22. Codex 每轮输出必须回答的 8 个问题

每完成一轮，不要只给我 loss 和 AUROC。

必须回答：

1. 这轮改了哪个 Agent 的**能力边界**？
2. 该 Agent 新增了什么**别的 Agent 无法访问的信息**？
3. 它主要救了哪一种失败？
4. rescue 在 held-out classes 上是否仍成立？
5. Forced consultation 是否比 same-agent no-comm 更好？
6. 消息是否真正改变了具体 candidate score？
7. message shuffle / wrong responder 是否让结果下降？
8. 当前证据是否已经足够进入下一 gate？

---

# 23. Codex 的停止条件

如果出现下面任何情况，不要继续堆复杂模型。

## 情况 A

```text
Agent-oracle ΔH ≈ 0
```

说明 Agent 已经高度重复。

**回到 Agent 设计。**

## 情况 B

```text
Agent-oracle ΔH 很大
但 rescue predictor ≈ random
```

说明互补存在但不可预测。

**回到私有任务/观测设计。**

## 情况 C

```text
Forced communication ≤ no communication
```

说明 consultation 没有净价值。

**不要训练 Router。**

## 情况 D

```text
Full Multi-Agent ≈ equal-capacity single-body multi-view
```

说明多智能体仍没有不可替代性。

**不能靠改名字解决。**

## 情况 E

```text
message shuffle ≈ original message
```

说明消息内容可能没有被真正使用。

**回到 message-to-decision mechanism。**

---

# 24. 我们最终希望得到的系统行为

最终论文中应该可以展示这种真实案例：

```text
Temporal Agent:
“从 temporal transient 看，我倾向 Tx7，
但 Tx7 与 Tx12 的时域证据差距很小。
我缺少频域硬件证据。”

        ↓ ASK_SPECTRAL(Tx7, Tx12)

Spectral Agent:
“Tx7 的 spectral impairment 与当前样本一致，
Tx12 不一致。
SUPPORT Tx7，pair evidence +2.1。”

        ↓

Candidate evidence:
Tx7 ↑
Tx12 ↓

        ↓ ASK_MEMORY(Tx7)

Memory Agent:
“样本对 Tx7 的 enrollment-tail distance 正常，
没有明显越出 Known support。”

        ↓

Consistency Auditor:
“身份保持扰动前后 Tx7 判断稳定。”

        ↓

Final:
Known → Tx7
```

以及 Unknown 案例：

```text
Temporal:
“最像 Tx7，但不确定。”

Spectral:
“Tx7 只是相对最像，支持度弱。”

Memory:
“Tx7 是最近 Known，但距离已经落入异常 tail。”

Auditor:
“轻微身份保持扰动导致类别剧烈变化。”

Final:
Unknown
```

这时多智能体的逻辑才是：

> 我不知道  
> → 我知道自己缺哪类证据  
> → 我询问拥有该信息的 Agent  
> → 它返回我自己无法生成的证据  
> → 我的具体判断改变

而不是：

> A 0.73  
> B 0.78  
> 所以相信 B。

---

# 25. 最终研究成功标准

只有当以下链条全部成立时，才认为“多智能体开放集辐射源识别”真正成立：

```text
严格私有观测
    ↓
不同 Agent 形成稳定不同失败模式
    ↓
存在双向 unique rescue
    ↓
rescue 关系可跨类别预测
    ↓
proposal-conditioned consultation > parallel/no-comm
    ↓
消息直接改变 candidate-level evidence
    ↓
VOI Coordinator 能按需调用正确专家
    ↓
messages-off / shuffle / wrong-responder 消融显著退化
    ↓
Full Multi-Agent 稳定优于 best-single
    ↓
Full Multi-Agent 稳定优于 equal-capacity no-comm
    ↓
Full Multi-Agent 能与 equal-parameter single-body multi-view 拉开差距
    ↓
最终 H / OSCR / AUROC 在 WiSig 与 ORACLE 上稳定提升
```

---

# 26. 对下一轮工作的明确指令

Codex 下一步不要直接开始长训练。

首先完成：

1. 创建 `src/maros_stage8/`；
2. 给出 Stage-8 observation/capability contracts；
3. 将当前 Prototype super-agent 拆除，不允许一个 Agent 同时拥有五种 view；
4. 实现严格隔离的 Temporal 与 Spectral Agent；
5. 实现 candidate-conditioned verifier objective；
6. 添加 capability-isolation tests；
7. 添加 candidate-specific message-effect tests；
8. 保留并迁移 nested LCO / formal unknown lock；
9. 只跑小规模 G0/G1 probe；
10. 输出：
   - local capability；
   - unique rescue；
   - oracle headroom；
   - cross-class rescue predictability；
   - 是否值得进入 Memory/Auditor 与 forced communication 阶段。

**在这十项完成以前，不允许进入新的 Router 调参循环。**

---

# 27. 最重要的设计原则，再强调一次

下一版判断一个模块能不能叫 Agent，不看它有没有 neural network，不看它有没有单独的 class，不看代码是不是单独一个文件。

看三件事：

> **它有没有别人没有的私有信息？**  
> **它有没有别人无法直接完成的私有任务？**  
> **别人是否会因为缺少这项能力而必须向它询问？**

如果三个答案不是“是”，那它更可能只是：

- branch；
- head；
- expert；
- service；
- fusion module；

而不是我们这篇工作里真正需要的 Agent。

---

## 给 Codex 的最终目标

不要追求“Agent 数量更多”。

追求：

> **每一个 Agent 都有不可替代的存在理由；每一次通信都能解释为什么必须问这个 Agent；每一条消息都能证明它改变了接收方原本无法独立完成的判断。**

这才是 Stage-8 应该围绕的核心。
