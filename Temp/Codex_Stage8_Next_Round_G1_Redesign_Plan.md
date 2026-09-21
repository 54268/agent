# Stage-8 下一轮重构与验证规划（交付 Codex）

> 仓库：`54268/agent`  
> 上一轮交付：Stage-8 G0/G1 implementation + initial WiSig / ORACLE results  
> 本文定位：**Stage-8 下一轮研究与工程执行指令**  
> 目标：不进入 Router、不进入正式 Unknown、不进入大规模多种子，而是先把 Temporal / Spectral 两个基础 Agent 的“能力边界、可学习性、双向救援与跨类可预测性”真正做扎实。

---

# 0. 本轮总判断

上一轮 Stage-8 是一次**方向正确的结构性重构**，但还不是性能成功。

已经取得的进展：

- Stage-7 的 Prototype multi-view super-agent 已被实质拆除；
- Temporal / Spectral 已建立不同 observation contract；
- candidate-conditioned verification 已经进入代码；
- candidate-specific evidence update 已经实现；
- `alternative_class` 已经能直接改变具体候选类别分数；
- capability isolation / private-state boundary / candidate-specific message-effect 等关键机制测试已通过；
- WiSig 已出现较强的 cross-class rescue predictability 信号；
- Codex 没有在 G1 未通过时擅自继续开 Router / Memory / Auditor / formal Unknown。

但当前仍有四个核心问题：

1. **ORACLE 的两个本地 Agent 都没有学会基本任务，当前 rescue 结果不可解释。**
2. **WiSig 上 Temporal 明显强于 Spectral，双向 unique rescue 尚未形成。**
3. **candidate-conditioned verifier 仍依赖 `nn.Embedding(class_id)`，类无关泛化逻辑不够干净。**
4. **当前是 API isolation，但还没有充分形成 information isolation。Raw I/Q 理论上仍包含可恢复的频域信息。**

因此：

> **下一轮禁止进入 Router / VOI / forced communication / Memory Agent / Auditor 正式训练。**

先修 A/B。

---

# 1. 当前结果应如何解释

## 1.1 WiSig：有积极信号，但还不是双向异构 Agent

当前 Temporal Known Accuracy 约：

```text
0.829
0.817
0.883
0.912
```

Spectral Known Accuracy 约：

```text
0.561
0.644
0.692
0.784
```

说明：

> Temporal 已经是有效本地专家；Spectral 有一定能力，但明显更弱。

当前 unique rescue：

### Temporal rescues Spectral

```text
22.4%
16.1%
16.1%
10.7%
```

### Spectral rescues Temporal

```text
2.38%
9.78%
2.99%
2.07%
```

这说明当前系统更像：

```text
Temporal = 主专家
Spectral = 偶尔补充
```

而不是：

```text
Temporal 在某类问题上真正需要 Spectral
Spectral 在另一类问题上真正需要 Temporal
```

这一点必须修。

---

## 1.2 WiSig 的 rescue predictability 是本轮最重要的积极结果

当前 cross-class rescue macro-AUROC：

```text
0.714
0.751
0.795
0.870
mean ≈ 0.782
```

这比 Stage-7 有明显研究意义。

说明：

> 当前至少已经出现“救援关系具有一定可预测性”的信号。

但现在不能只看 macro-AUROC。

下一轮必须拆开：

```text
AUROC(Temporal rescues Spectral)
AUROC(Spectral rescues Temporal)
AUROC(no rescue / both fail)
```

尤其重点看：

> **Spectral rescues Temporal 是否可预测。**

否则 macro-AUROC 可能主要由“Temporal rescue”和“none”两个容易类别拉高。

---

# 2. ORACLE 当前不能用来判断 Agent 互补性

当前 ORACLE Known Accuracy 大约：

### Temporal

```text
18% ~ 22%
```

### Spectral

```text
13% ~ 16%
```

六分类随机水平约：

\[
1/6 \approx 16.7\%
\]

训练准确率也仅约：

```text
25% ~ 35%
```

所以当前 ORACLE 不是“泛化差”，而是：

> **连训练任务都没有充分学会。**

因此当前 ORACLE 中：

- 双向 rescue；
- Agent-oracle ΔH；
- rescue predictability；
- H-score；

都不能作为可靠的 Agent 设计判断。

Codex 下一轮首先必须回答：

> 当前 ORACLE 是训练预算太小，还是 observation / architecture 本身无法学习？

这个问题必须在任何进一步方法改动之前解决。

---

# 3. 下一轮最优先任务：ORACLE Local Learnability Sanity

## 3.1 禁止改变 Agent 信息边界

这一轮 ORACLE sanity 期间：

**禁止：**

- 给 Spectral Agent 加 raw IQ；
- 给 Temporal Agent 加 FFT/STFT；
- 给任一 Agent 加其他 Agent embedding；
- 加 Router；
- 加 Memory；
- 加 Auditor；
- 加 cross-agent fusion；
- 加 dataset-specific branch。

只能增加合理训练能力和优化稳定性。

---

## 3.2 做 small-set overfit test

为 Temporal / Spectral 分别做：

```text
每类选 16 / 32 / 64 个 Known 样本
只在这个小集合训练
```

目标不是泛化，而是验证：

> 模型能不能把一个小训练集学会。

建议判据：

```text
train accuracy ≥ 95%
```

如果做不到：

> 架构、loss、optimizer、normalization 或 observation 有基础问题。

如果能做到：

> 说明模型表达能力存在，继续做正常容量训练。

必须分别给出：

```text
Temporal small-set train acc
Spectral small-set train acc
loss curve
pair-verification loss curve
```

---

## 3.3 正常训练预算上调

在不改变 observation boundary 的前提下，允许：

```text
state_dim: 32 → 64 / 96
hidden_channels: 16 → 32 / 64
epochs: 12 → 25~40
```

优先做少量结构化候选，不做大网格。

推荐：

### Candidate A

```text
state_dim = 64
hidden = 32
epochs = 30
```

### Candidate B

```text
state_dim = 96
hidden = 64
epochs = 30
```

如果 A 已明显学会，不必继续 B。

停止标准：

> 只要 local learnability 已经正常，就停止容量扫描。

---

# 4. 第二优先任务：去掉 `nn.Embedding(class_id)` verifier

当前 candidate-conditioned verifier 中仍存在：

```python
nn.Embedding(num_classes, state_dim)
```

这会产生一个潜在问题：

> verifier 学到的是“类别编号对应的可训练表项”，而不是真正的 class-invariant compatibility。

这不符合 Stage-8 的核心目标。

---

# 5. 改成 Prototype/Descriptor-Conditioned Verification

## 5.1 Temporal Agent

从 Known enrollment/train 样本构造：

```text
Temporal Prototype p_T^c
```

然后：

\[
Verify_T(z_T(x), p_T^c)
\rightarrow e_T(x,c)
\]

其中：

- `z_T(x)`：Temporal query descriptor；
- `p_T^c`：candidate c 的 temporal prototype；
- verifier 参数对所有 class 共享。

---

## 5.2 Spectral Agent

同理：

```text
Spectral Prototype p_F^c
```

\[
Verify_F(z_F(x), p_F^c)
\rightarrow e_F(x,c)
\]

不能存在：

```text
candidate id → embedding table lookup → verifier
```

---

## 5.3 Candidate descriptor 必须来自真实类数据

建议至少包含：

```text
mean prototype
robust variance / scale
optional nearest-competitor relation
```

但第一轮不要复杂化。

先用：

```text
normalized class prototype
```

就够。

---

## 5.4 Inner LCO 要重新计算 prototype

非常重要：

每个 inner fold：

```text
inner support classes
    ↓
只用该 inner support Known
    ↓
构建 candidate prototype
```

held-out class 不能进入 prototype。

formal Unknown 更不能进入。

---

# 6. 第三优先任务：重新定义 G1 指标

上一轮 G1 对 A/B 使用 H-score 有一个问题：

Temporal / Spectral 当前主要应承担：

> identity evidence

而不是最终 Unknown rejection。

所以不能让一个很粗糙的 local unknown score 把 G1 结论完全支配。

---

# 7. Rescue 必须拆成三类

下一轮至少报告：

## 7.1 Known Identity Rescue

定义：

```text
Agent A Known identity wrong
Agent B Known identity correct
```

这是 A/B 当前最重要的互补性指标。

---

## 7.2 Unknown Rejection Rescue

定义：

```text
Agent A accepts proxy Unknown
Agent B rejects proxy Unknown
```

此项保留，但不作为 A/B 主要 gate。

未来主要由 Memory / Auditor 承担。

---

## 7.3 Candidate-Pair Rescue

对于真实类别 `y` 与最难竞争类 `c*`：

```text
Agent A pair verification wrong
Agent B pair verification correct
```

这应成为 Stage-8 最重要的新 rescue 指标之一。

因为未来通信本质上就是：

```text
“帮我验证 candidate A vs candidate B”
```

---

# 8. 新增 Pair Verification Metrics

对每个 Agent 报告：

```text
pair accuracy
pair AUROC
hard-negative pair accuracy
pair calibration
```

并按 held-out class inner fold 统计。

推荐 hard negative：

```text
当前 Agent 最容易混淆的 nearest competitor
```

而不是随机负类。

---

# 9. Rescue Predictability 必须按方向拆开

当前一个 macro-AUROC 不够。

必须输出：

```text
AUROC(T rescues S)
AUROC(S rescues T)
AUROC(T/S both fail)
AUROC(no query needed)
```

此外报告：

```text
PR-AUC
support count
positive rate
```

尤其 Spectral rescue 很少时：

> AUROC 可能看起来正常，但 PR-AUC 会暴露实际可用性。

---

# 10. 下一轮新核心指标：Bidirectional Useful Rescue

建议定义：

```text
BUR_T→S = P(Temporal uniquely fixes Spectral error)
BUR_S→T = P(Spectral uniquely fixes Temporal error)
```

进入下一阶段的最低要求建议改成：

```text
两方向都 > 5%
```

如果某方向持续 <3%：

> 说明那个 Agent 仍缺乏不可替代性。

不要靠 Router 解决。

---

# 11. 第四优先任务：把 API Isolation 推进为 Information Isolation

当前：

```text
TemporalObservation = raw I/Q
SpectralObservation = f(raw I/Q)
```

因此从信息角度：

> Temporal 理论上仍持有 Spectral 的源信息。

这会削弱论文中：

> “对方拥有我没有的信息”

这个核心命题。

下一轮需要开始做真正的 observation specialization。

---

# 12. Temporal Observation 建议改成“有损时域观察”

目标：

> 保留 temporal fingerprint，主动削弱可恢复的全局 spectral information。

候选方案不要一次全上。

优先做一个最简单、可解释版本：

```text
local amplitude dynamics
phase-increment sequence
short temporal patches
local complex correlation
```

并做：

- per-window normalization；
- 消除绝对幅度尺度；
- 避免给出完整长序列的全局频谱恢复能力。

第一轮可以：

```text
raw I/Q → short-window temporal tokens
```

然后只允许 Temporal Agent 访问这些 local tokens。

---

# 13. Spectral Observation 要主动破坏精细时间顺序

Spectral Agent 保留：

```text
PSD
spectral envelope
phase-related frequency statistics
spectral asymmetry
frequency-local impairment
```

但不要保留完整时序。

例如：

```text
frequency representation
+
coarse pooled spectral descriptors
```

不要再带可还原原始时间序列的完整 phase sequence。

---

# 14. 不要一次把 Temporal Observation 改得太激进

Observation specialization 必须做消融：

```text
Temporal-v1: current raw
Temporal-v2: local-window temporal only
```

比较：

```text
Known accuracy
pair verification
T→S rescue
S→T rescue
rescue predictability
```

如果 v2 让 Temporal 完全崩掉：

> 说明信息删太多。

目标不是故意削弱 Agent。

目标是：

> 保留其专长，同时让另一 Agent 的信息真正不可替代。

---

# 15. 下一轮不能做的事情

在 A/B G1.5 未通过前，禁止：

- Router；
- VOI Coordinator；
- PPO / QMIX；
- CommFormer；
- formal Unknown；
- multi-seed formal；
- Memory Agent 正式训练；
- Auditor 正式训练；
- Stage-8 full four-Agent result；
- dataset-specific branch；
- hidden-view leakage；
- 给 Spectral 加 raw；
- 给 Temporal 加 FFT；
- 为追 H 偷偷给 A/B 加 open-set super-head。

---

# 16. 下一轮允许做的事情

只允许：

```text
ORACLE local learnability
prototype-conditioned verification
A/B observation specialization
rescue metric decomposition
direction-specific rescue prediction
pair verification diagnostics
```

---

# 17. 下一轮 G1.5 新 Gate

建议重新定义 A/B gate。

## Gate A：Local Learnability

WiSig / ORACLE：

```text
Temporal Known accuracy 明显高于随机
Spectral Known accuracy 明显高于随机
small-set overfit 可达到高训练准确率
```

ORACLE 如果仍接近随机：

> 不进入后续 gate。

---

## Gate B：Pair Verification

两 Agent 在 held-out class：

```text
pair AUROC > 0.70
```

作为开发目标，不是论文最终标准。

hard-negative pair accuracy 需要明显高于随机 50%。

---

## Gate C：Bidirectional Rescue

```text
Temporal → Spectral unique identity/pair rescue > 5%
Spectral → Temporal unique identity/pair rescue > 5%
```

至少两个数据集都不能出现长期单向退化。

---

## Gate D：Rescue Predictability

分别要求：

```text
AUROC(T rescues S) > 0.65
AUROC(S rescues T) > 0.65
```

并且：

```text
≥ 3/4 inner folds > 0.5
```

同时看 PR-AUC。

---

## Gate E：No Class-ID Shortcut

验证：

```text
shuffle candidate class index
```

但保持 prototype 对应关系。

如果 verifier 性能不变：

> 说明没有 class-id shortcut。

也可以做：

```text
reindex classes
```

性能应该近似保持。

---

# 18. 新增必须做的 Reindex Invariance Test

由于我们要证明 verifier 是 class-invariant：

对同一个 inner episode：

```text
原类别映射:
TxA → 0
TxB → 1
TxC → 2
```

重新随机映射：

```text
TxA → 2
TxB → 0
TxC → 1
```

prototype 与 label 同步重排。

如果模型逻辑正确：

> pair verification 和最终结果应近似不变。

这个测试非常重要。

它可以直接证明：

> 模型不是依赖 class index embedding。

---

# 19. 新增 Observation Leakage Tests

## Temporal Leakage Probe

训练一个轻量 probe：

```text
TemporalObservation
    ↓
预测 spectral-only descriptor
```

如果能极高精度重建核心 spectral descriptor：

> Temporal observation 仍包含太多 spectral information。

---

## Spectral Leakage Probe

训练：

```text
SpectralObservation
    ↓
预测 temporal-order descriptor
```

如果能轻易恢复 temporal local ordering：

> Spectral observation 仍过度包含时间信息。

这不是要求完全不可预测。

而是作为：

> **信息隔离的量化审计。**

---

# 20. 对 H-score 的使用做阶段区分

## A/B G1 阶段

主看：

```text
Known accuracy
pair AUROC
unique identity rescue
candidate-pair rescue
directional rescue predictability
```

H-score只做辅助。

因为 A/B 目前不是最终 Unknown specialist。

---

## C/D 加入后

再把：

```text
Unknown Recall
H
OSCR
AUROC
```

提升为主指标。

这样职责更符合 Agent 分工。

---

# 21. 下一轮 Codex 推荐执行顺序

严格按以下顺序：

### Step 1

完成 ORACLE small-set overfit test。

### Step 2

在 observation 不变情况下，让 ORACLE local model 正常收敛。

### Step 3

删除 `nn.Embedding(class_id)` verifier。

### Step 4

实现 prototype-conditioned verifier。

### Step 5

加入 class reindex invariance test。

### Step 6

拆分：

```text
identity rescue
unknown rescue
pair rescue
```

### Step 7

拆分 directional rescue predictability。

### Step 8

先在现有 Temporal/Spectral observation 上重新跑 WiSig + ORACLE G1。

### Step 9

如果仍然强单向：

> 再做 information-isolated Temporal-v2 / Spectral-v2。

### Step 10

重新跑 G1.5。

### Step 11

只有双向 rescue + directional predictability 同时通过：

> 才允许解锁 Memory Agent。

---

# 22. 本轮不要一次实现四 Agent

下一轮只专注：

```text
Agent A = Temporal
Agent B = Spectral
```

理由：

如果 A/B 还不能形成稳定私有专长：

> 加 C/D 只会把问题复杂化。

等 A/B 成立后：

下一阶段才进入：

```text
Agent C = Enrollment Memory
Agent D = Open-Space Consistency
```

---

# 23. 什么时候可以解锁 Memory Agent

只有同时满足：

```text
WiSig local learnability PASS
ORACLE local learnability PASS

T→S rescue PASS
S→T rescue PASS

directional rescue predictability PASS

prototype-conditioned verifier PASS

class reindex invariance PASS
```

才进入下一阶段。

---

# 24. Memory Agent 下一阶段预告

暂时不要实现，但接口先保持兼容。

未来 Memory Agent 负责：

```text
“candidate c 是最近 Known”
vs
“query 真正属于 candidate c 的支持域”
```

也就是：

```text
nearest-known ≠ valid-known-member
```

它将主要负责：

```text
Unknown Rejection Rescue
```

这样 A/B/C 的职责会自然分开：

```text
A：temporal identity evidence
B：spectral identity evidence
C：Known support / open-space compatibility
```

---

# 25. Codex 每轮结果必须给出的表

下一轮不要再只给一个 summary。

必须输出至少：

| Dataset | Fold | T Known Acc | S Known Acc | T pair AUC | S pair AUC | T→S identity rescue | S→T identity rescue | T→S pair rescue | S→T pair rescue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

以及：

| Dataset | Fold | AUROC(T rescues S) | PR-AUC(T rescues S) | AUROC(S rescues T) | PR-AUC(S rescues T) |
|---|---:|---:|---:|---:|---:|

再给：

```text
small-set overfit result
class reindex invariance result
observation leakage probe result
```

---

# 26. Codex 必须解释失败原因，不能只给数

每个失败 fold 都回答：

1. 是 local model 没学会？
2. 还是两 Agent 学到重复能力？
3. 还是 Spectral 本身太弱？
4. 还是 observation boundary 不合理？
5. 还是 verifier 没泛化？
6. 还是 rescue 存在但不可预测？

必须区分。

---

# 27. 下一轮成功标准

我们不是追求：

```text
H 立刻变成最高
```

而是追求以下链条：

```text
ORACLE local agent 可正常学习
    ↓
candidate verifier 不依赖 class ID
    ↓
Temporal / Spectral 保持不同能力
    ↓
两个方向都有 unique identity / pair rescue
    ↓
两个方向 rescue 都可以跨 class 预测
    ↓
这种现象同时出现在 WiSig 和 ORACLE
```

如果这条链成立：

> Stage-8 A/B 才算真正成功。

---

# 28. 下一轮 Stop Conditions

## Stop A

ORACLE small-set 都 overfit 不了。

→ 先修 architecture / optimizer。

---

## Stop B

去掉 class embedding 后 verifier 完全失效。

→ 检查 prototype construction / pair objective。

---

## Stop C

Temporal 永远远强于 Spectral，S→T rescue <3%。

→ 不开 Router，重新设计 Spectral private task / observation。

---

## Stop D

双向 rescue 存在，但 directional AUROC ≈ 0.5。

→ 不开 Router，继续改 task/quality descriptors。

---

## Stop E

Temporal-v2 / Spectral-v2 信息隔离后，两边性能都严重下降。

→ observation specialization 过度，回滚，不继续堆损失。

---

# 29. 下一轮不要做“为了满足双向 rescue 故意削弱强 Agent”

非常重要。

不能为了让 Spectral rescue 变多，而人为把 Temporal 做差。

我们需要的是：

> 两个 Agent 对不同问题各有所长。

不是：

> 人为把强模型削弱，让另一个看起来有价值。

因此任何 observation specialization 必须同时满足：

```text
Agent 自己的专长指标仍然合理
```

并且通过：

```text
best-single
equal-capacity single-body
```

后续公平比较。

---

# 30. 最终给 Codex 的执行指令

下一轮请只完成以下六件事：

1. **修复 ORACLE local learnability，先完成 small-set overfit sanity。**
2. **把 `nn.Embedding(class_id)` verifier 改为 prototype-conditioned verifier。**
3. **加入 class reindex invariance test。**
4. **将 rescue 拆成 identity / unknown / candidate-pair，并将 predictability 拆成 T→S 和 S→T 两个方向。**
5. **在现有 observation 上重新跑 WiSig + ORACLE；只有仍然单向时再实现更严格的 information-isolated Temporal-v2 / Spectral-v2。**
6. **重新执行 G1.5 gate，并明确输出 PASS / FAIL / STOP-AND-REDESIGN。**

在这六项完成之前：

> **禁止开启 Memory Agent、Auditor、Router、forced communication、formal Unknown 或多种子正式实验。**

---

# 31. 本轮最重要的一句话

上一轮解决的是：

> **“两个 Agent 在代码上是不是两个独立模块？”**

下一轮必须解决的是：

> **“两个 Agent 是否真的拥有彼此不能替代的知识，并且这种不可替代性可以跨类别稳定预测？”**

只有这个问题通过，后面的多智能体通信才值得做。
