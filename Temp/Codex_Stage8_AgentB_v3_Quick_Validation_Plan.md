# Stage-8 Agent-B v3 快速验证规划（交付 Codex）

> 仓库：`54268/agent`  
> 当前结论：Stage-8 G1/G1.5 仍为 `STOP-AND-REDESIGN`。  
> 本轮目的：**只修改 Agent B（Spectral Hardware Fingerprint Agent），快速判断当前跨数据集失败是否主要来自 Spectral observation / private task 的设计。**  
> 本轮不是正式方法升级，不解锁 Memory、Auditor、Router、VOI、forced communication 或 formal Unknown。

---

# 0. 本轮只回答一个问题

当前 Stage-8 的主要矛盾已经非常集中：

## WiSig

Spectral Agent 表现很好：

```text
isolated_v2:
Known Acc ≈ 0.916 ~ 0.936
pair AUC ≈ 0.995 ~ 0.997
```

## ORACLE

Spectral Agent 接近随机：

```text
Known Acc ≈ 0.160 ~ 0.182
pair AUC ≈ 0.487 ~ 0.500
```

但 small-set overfit：

```text
16/class → 1.00
32/class → 1.00
64/class → 1.00
```

因此：

> **Agent B 不是容量不够，也不是优化器学不会。它能记住训练集，但当前频域 observation / representation 在 ORACLE 上没有形成可泛化的硬件指纹。**

本轮只验证：

> **把 Agent B 改成更抗时间对齐、频移、绝对相位和瞬时信道变化的频域专家后，ORACLE 是否明显改善，同时 WiSig 不明显退化。**

---

# 1. 为什么优先怀疑当前 Agent B

当前 `build_spectral_observation()` / `build_coarse_spectral_observation()` 中含有：

```text
FFT phase
phase step
phase roughness
paired-frequency complex coherence
asymmetry around fixed bin position
frequency-bin-local structure
```

这些量并不一定都是稳定的 emitter fingerprint。

它们可能强烈受到：

```text
packet temporal alignment
time shift
carrier-frequency offset
sampling offset
channel response
receiver response
capture-session variation
SNR / interference
```

影响。

尤其：

## 1.1 FFT 绝对/相对 phase 对 time shift 非常敏感

若：

\[
x[n] \rightarrow x[n-n_0]
\]

FFT 会出现：

\[
X[k]e^{-j2\pi kn_0/N}
\]

即使发射机完全没变，FFT phase 也会系统变化。

因此：

> 使用 phase-step / roughness 的 Agent B 可能学到“采集对齐方式”，而不是稳定硬件指纹。

---

## 1.2 固定频率 bin 位置容易受到 CFO / sampling offset 影响

同一 emitter 若频谱整体移动几个 bin：

```text
训练：
peak at bin 127

测试：
peak at bin 132
```

普通 Conv1d 虽有一定平移鲁棒性，但固定-bin asymmetry、局部 pattern 和 prototype geometry 仍可能明显漂移。

---

## 1.3 WiSig 与 ORACLE 都是 256 点，并不意味着统计条件相同

“256 samples”只表示：

```text
每个输入张量长度相同
```

它不意味着：

```text
采样率相同
中心频率相同
接收机相同
信道相同
包起始对齐相同
CFO 分布相同
SNR 分布相同
session 变化相同
设备间 separability 相同
```

所以完全可能：

```text
同样 [2,256] I/Q
```

但一个数据集的 spectral pattern 很稳定，另一个数据集的 spectral nuisance 比 emitter fingerprint 更强。

这正是当前结果最像的情况。

---

# 2. 本轮禁止修改 Agent A

Temporal Agent 完全冻结。

包括：

- observation；
- encoder；
- state_dim；
- loss；
- verifier；
- prototype；
- threshold；
- training epochs。

必须使用与当前基线完全相同的 Temporal checkpoint/config。

原因：

> 本轮必须能把结果变化归因到 Agent B。

---

# 3. 新 Agent B：Robust Spectral Fingerprint Agent v3

建议命名：

```text
SpectralHardwareFingerprintAgentV3
```

或：

```text
observation_profile = robust_spectral_v3
```

不要覆盖 v1/v2。

---

# 4. v3 的核心原则

Agent B 仍然只能看到频域信息。

但是：

> **去掉对绝对 FFT phase 和精确频率位置过度敏感的特征，改成“多段、归一化、相对频谱形状”的硬件指纹。**

---

# 5. Observation v3：Segment-Averaged Relative Spectrum

输入仍为 raw IQ，但 observation builder 在 Agent 外部完成频域转换。

Agent B 本身绝对不能访问 raw IQ。

---

# 6. Step A：分段频谱，而不是一次 256 点 FFT

将 256 点序列分成短段，例如：

```text
8 × 32
或
4 × 64
```

第一轮只选一个：

```text
4 × 64
```

对每段做：

```text
window
FFT
power spectrum
```

然后计算：

```text
mean PSD across segments
std PSD across segments
```

这样保留：

> 稳定的频谱硬件形状

同时弱化：

> 单一时间起点 / 瞬时相位 / 单段噪声的影响。

---

# 7. Step B：只保留 magnitude / power 路线

第一轮 v3 **完全删除 FFT phase**。

禁止：

```text
absolute FFT phase
phase step
phase roughness
paired complex phase
```

理由不是认定 phase 永远无用。

而是做一个快速因果诊断：

> ORACLE 的失败是否主要来自 phase-sensitive spectral features？

如果 v3 明显恢复 ORACLE，说明这个假设成立。

后续再考虑加入更稳健的 relative-phase fingerprint。

---

# 8. Step C：频谱中心对齐 / CFO nuisance reduction

根据 mean PSD 计算频谱能量中心：

\[
k_c =
\frac{\sum_k kP[k]}
{\sum_k P[k]}
\]

然后将每个样本的 mean/std spectrum 做 circular recenter：

```text
spectral centroid → center bin
```

也可以用：

```text
peak / robust centroid
```

但第一版只用 energy centroid。

目的：

> 弱化整体 CFO 导致的频谱平移。

注意：

不要把 CFO 本身完全删除后就不记录。

可以把：

```text
normalized centroid offset
```

保留为一个 `quality diagnostic`，但不要让 classifier 直接依赖其绝对值作为主 fingerprint。

---

# 9. Step D：每样本 robust normalization

对 recentered log-power 做：

```text
median centering
/
IQR or robust scale
```

例如：

\[
\tilde P =
\frac{P-\operatorname{median}(P)}
{Q_{0.75}(P)-Q_{0.25}(P)+\epsilon}
\]

相比 mean/std：

> 更不容易被少数强干扰 bin 拉偏。

---

# 10. v3 频域通道建议

保持简单。

第一轮只使用以下 6 个 channel：

```text
1. robust normalized mean log-PSD
2. segment-to-segment PSD std
3. smoothed spectral envelope
4. local spectral contrast = PSD - envelope
5. first spectral derivative
6. recentered asymmetry
```

不要再加 phase。

不要再增加十几个 handcrafted feature。

---

# 11. v3 quality diagnostics

可以输出：

```text
spectral entropy
spectral flatness
segment consistency
centroid offset
peak-to-average ratio
asymmetry energy
```

这些只用于：

```text
public diagnostics / rescue prediction
```

不要把它们作为 dataset-specific rule。

---

# 12. Agent B 网络本轮尽量不改

为保证因果归因：

继续使用当前 Spectral encoder：

```text
Conv1d
→ BN
→ GELU
→ Conv1d
→ BN
→ GELU
→ Conv1d
→ GAP
```

state_dim / hidden_channels 使用当前已验证配置。

不要同时修改：

```text
architecture
loss
observation
optimizer
```

本轮只改 observation。

---

# 13. Candidate verifier 继续使用 prototype-conditioned 版本

保留当前已经正确完成的：

```text
query state
candidate prototype
|query - prototype|
query * prototype
      ↓
shared verifier
```

严禁恢复：

```text
nn.Embedding(class_id)
```

class reindex invariance 必须继续通过。

---

# 14. 本轮只跑三个 Spectral profile

不要做大网格。

## B0：当前 v1 baseline

```text
legacy_v1
```

直接复用已有结果，不必重跑。

---

## B1：当前 v2 baseline

```text
isolated_v2
```

直接复用已有结果，不必重跑。

---

## B3：新 robust_spectral_v3

只训练这一个。

如果需要做一个最小消融，可以加：

## B3-noalign

和 B3 完全一样，但不做 spectral centroid recenter。

只用于判断：

> 改善主要来自 phase removal，还是 frequency alignment。

最多两个新候选。

---

# 15. 快速验证顺序

不要一上来跑全部四折。

---

## Phase 1：ORACLE fold-0 快速筛选

只跑：

```text
ORACLE inner fold 0
```

检查：

```text
Spectral Known Acc
Spectral pair AUC
Spectral hard-pair accuracy
Spectral train/eval gap
S→T identity rescue
S→T pair rescue
```

### Early-stop 失败条件

如果 B3 仍然：

```text
Known Acc < 0.25
pair AUC < 0.60
```

则：

> 立即停止，不跑四折。

说明单纯 robust spectrum 仍然没有解决 ORACLE。

---

# 16. Phase 1 的“值得继续”标准

只要达到：

```text
ORACLE fold0 Spectral Known Acc ≥ 0.35
AND
pair AUC ≥ 0.65
```

就认为：

> **Agent B 改动出现有效信号。**

然后进入 Phase 2。

这个门槛不是最终成功标准。

它只是为了快速判断：

> 从原来的 0.17 / 0.50 有没有实质跳升。

---

# 17. Phase 2：ORACLE 四 inner folds

如果 fold0 有明显改善，再跑 4 folds。

报告：

| Fold | S train acc | S eval acc | S pair AUC | hard-pair | S→T id rescue | S→T pair rescue |
|---|---:|---:|---:|---:|---:|---:|

---

# 18. Agent B v3 的快速成功标准

如果 ORACLE 四折满足：

```text
mean Spectral Known Acc ≥ 0.40
pair AUC ≥ 0.70 on ≥3/4 folds
hard-pair accuracy > 0.55 on ≥3/4 folds
```

同时：

```text
Spectral train acc 高
eval 不再接近随机
```

则认为：

> **Agent B 的跨样本泛化问题得到明显改善。**

注意这里先不要求最终 G1 全过。

---

# 19. Phase 3：只在 ORACLE 有效后才回测 WiSig

如果 ORACLE 没改善：

> 不跑 WiSig。

如果 ORACLE 改善，再跑 WiSig 4 folds。

要求：

```text
WiSig Spectral Known Acc 不低于 0.85
WiSig pair AUC 不低于 0.95
```

即：

> 修 ORACLE 不能以摧毁 WiSig 为代价。

---

# 20. 本轮最关键的 multi-agent 指标

如果 B v3 本地能力恢复，再看：

## S→T identity rescue

当前 WiSig v2 大约：

```text
2.4% ~ 5.2%
```

ORACLE v1/v2 虽有 rescue，但因为 B 很弱，不能解释。

本轮要求：

```text
WiSig mean S→T identity rescue ≥ 5%
ORACLE mean S→T identity rescue ≥ 5%
```

但必须建立在：

```text
B local accuracy 已有效
```

前提下。

---

# 21. S→T pair rescue 更重要

通信以后主要做：

```text
candidate A vs candidate B verification
```

所以优先看：

```text
Spectral rescues Temporal on candidate pair
```

目标：

```text
mean S→T pair rescue ≥ 5%
```

并且不是由随机错误产生。

---

# 22. Directional rescue predictability

只重点看：

```text
AUROC(Spectral rescues Temporal)
PR-AUC(Spectral rescues Temporal)
positive count
positive rate
```

快速目标：

```text
AUROC ≥ 0.65
≥ 3/4 folds > 0.5
```

如果 rescue >5% 但 AUROC≈0.5：

> Agent B 有独特能力，但系统仍不知道何时该问它。

此时仍不能开 Router。

---

# 23. 必须增加“train vs eval spectral diagnostic”

当前 ORACLE 最重要的问题就是：

```text
train ≈ 1.0
eval ≈ random
```

所以 B3 每折必须保存：

```text
train state intra-class distance
eval state intra-class distance

train inter-class distance
eval inter-class distance

prototype → eval sample distance
nearest-class margin
```

至少给出聚合统计。

目的是区分：

```text
A. representation 整体漂移
B. 类内方差突然增大
C. 类间 margin 消失
D. prototype 本身不稳定
```

---

# 24. 建议加一个简单 Prototype Generalization Ratio

定义：

\[
PGR =
\frac{
\text{mean inter-class distance}
}{
\text{mean intra-class eval distance}
}
\]

分别在 train/eval 报告。

不需要作为正式论文指标。

只做诊断。

如果：

```text
train PGR 很高
eval PGR 接近 1
```

说明：

> representation 在训练集分得开，但测试分布中类结构塌掉。

---

# 25. 加一个 Spectral Alignment Ablation

如果 B3 有效，请额外比较：

```text
B3-align
vs
B3-noalign
```

如果：

```text
B3-align >> B3-noalign
```

说明 ORACLE 的重要 nuisance 很可能包含：

> frequency offset / spectral position drift

如果二者差不多但都优于 v1：

说明主要问题更可能来自：

> phase-sensitive features / single-FFT instability。

---

# 26. 本轮不要加入 frequency-shift augmentation

第一轮先不加。

因为我们需要知道：

> observation 本身是否解决问题。

如果同时加 augmentation，很难判断改善来自哪里。

只有 B3 仍稍差但有明显信号时，下一轮才允许尝试：

```text
small random spectral roll
```

---

# 27. 本轮不要重新改 Temporal Observation

当前 v2 已经说明：

> 同时激进修改 A/B observation 容易让 ORACLE 两边一起掉。

所以这轮：

```text
Temporal = 固定强基线
Spectral = 唯一变量
```

---

# 28. 本轮不要追求 H-score

A/B 当前主要负责 identity evidence。

本轮只看：

```text
B Known Acc
B pair AUC
B hard-pair accuracy
S→T identity rescue
S→T pair rescue
S→T rescue predictability
```

Unknown/H 仅记录，不作为 Agent-B v3 成败依据。

---

# 29. 不允许做的事情

本轮禁止：

```text
Memory Agent
Auditor
Router
VOI
forced communication
formal Unknown
multi-seed
threshold search
B2
OpenMax tuning
PPO/QMIX
dataset-specific branch
```

也禁止：

```text
ORACLE 用 v3
WiSig 用 v1
```

最终方法必须同一套 Agent B。

---

# 30. Codex 实现建议

建议新增：

```text
CoarseRobustSpectralObservation
```

或：

```text
RobustSpectralObservation
```

builder：

```python
build_robust_spectral_observation_v3(iq)
```

Agent：

现有 `SpectralHardwareFingerprintAgent` 增加：

```text
observation_profile = robust_v3
```

不要复制一整套训练流程。

---

# 31. 必须补的测试

至少增加：

```text
test_robust_spectral_has_no_phase_channel
test_robust_spectral_has_no_raw_iq
test_robust_spectral_alignment_invariance
test_robust_spectral_time_shift_stability
test_robust_spectral_class_reindex_invariance
```

---

# 32. Time-shift stability test

这是本轮非常重要的机制测试。

对同一个 IQ：

```text
x
roll(x, +5 samples)
roll(x, +11 samples)
```

比较 Agent B observation/state。

要求：

> B3 的 state drift 显著小于 v1。

例如计算：

\[
D =
1-\cos(z(x),z(\operatorname{roll}(x)))
\]

报告：

```text
v1 mean drift
v3 mean drift
```

---

# 33. Frequency-shift diagnostic

不要训练 augmentation。

只做诊断：

人为给 IQ 乘：

\[
e^{j2\pi \Delta f n}
\]

产生小频偏。

比较：

```text
v1 state drift
v3-noalign state drift
v3-align state drift
```

预期：

```text
v3-align 最稳定
```

这能直接验证 v3 的设计是否真的减少 CFO nuisance。

---

# 34. 快速决策树

```text
ORACLE fold0 B3
        │
        ├─ Acc < .25 或 pair AUC < .60
        │      ↓
        │    STOP
        │    Agent B 仍需重新设计
        │
        └─ Acc ≥ .35 且 pair AUC ≥ .65
               ↓
           跑 ORACLE 4 folds
               │
               ├─ mean Acc < .40
               │      ↓
               │    STOP
               │
               └─ 达标
                      ↓
                  跑 WiSig
                      │
                      ├─ WiSig < .85
                      │      ↓
                      │   跨数据集不成立
                      │
                      └─ WiSig 保持
                             ↓
                        看 S→T rescue
                             ↓
                     决定是否继续 Stage-8
```

---

# 35. 本轮最终报告格式

Codex 最终必须输出：

## 表 1：Agent B 本地能力

| Profile | Dataset | Fold | Train Acc | Eval Acc | Pair AUC | Hard Pair Acc |
|---|---|---:|---:|---:|---:|---:|

包括：

```text
v1
v2
v3
```

v1/v2 可以读取已有结果。

---

## 表 2：Agent B 不可替代性

| Dataset | Fold | S→T Identity Rescue | S→T Pair Rescue | Rescue AUROC | Rescue PR-AUC |
|---|---:|---:|---:|---:|---:|

---

## 表 3：Nuisance Stability

| Profile | Time-shift drift | Small-CFO drift |
|---|---:|---:|

---

## 表 4：Prototype Generalization

| Dataset | Fold | Train PGR | Eval PGR | Eval Prototype Margin |
|---|---:|---:|---:|---:|

---

# 36. 本轮最终只允许三个结论

## 结论 A：B3 明显改善

满足：

```text
ORACLE mean Known Acc ≥ .40
pair AUC ≥ .70 on ≥3/4 folds
WiSig Known Acc ≥ .85
WiSig pair AUC ≥ .95
```

则输出：

```text
AGENT-B-REDESIGN-PROMISING
```

下一轮再继续提升双向 rescue。

---

## 结论 B：ORACLE 改善但 WiSig 明显下降

输出：

```text
DATASET-ROBUSTNESS-FAIL
```

说明 B3 找到了 ORACLE 有效统计，但还不是共同的 emitter fingerprint。

禁止写 dataset-specific branch。

---

## 结论 C：ORACLE 仍接近随机

输出：

```text
SPECTRAL-PRIVATE-TASK-REDESIGN-REQUIRED
```

这时不要继续调 Conv 层。

下一轮要考虑：

> Agent B 的角色本身是否应从“频谱身份分类”改成某种更具体的硬件失真验证任务。

---

# 37. 如果 B3 仍失败，下一步不要继续做“更多频谱通道”

如果 ORACLE 仍失败：

不要再加：

```text
更多 FFT channel
更多 phase feature
更多 handcrafted descriptor
```

下一阶段应改 Agent B 的任务：

例如从：

```text
frequency view → emitter identity
```

改成：

```text
candidate-conditioned spectral impairment verification
```

并弱化独立 K-way classifier。

也就是说 B 真正只负责：

> “A 提出的 Tx7 和 Tx12，谁的频域硬件失真更符合当前样本？”

而不是要求它自己完整识别全部 emitter。

这会更符合多智能体设计。

---

# 38. 给 Codex 的最终执行指令

本轮只做以下工作：

1. 保持 Agent A 完全冻结；
2. 新增 phase-free、segment-averaged、centroid-aligned `robust_spectral_v3`；
3. 保留 prototype-conditioned verifier；
4. 先跑 ORACLE fold0；
5. fold0 达到快速门槛后才跑四折；
6. ORACLE 四折有效后才回测 WiSig；
7. 增加 time-shift / small-CFO stability diagnostic；
8. 输出 B local capability、S→T rescue 和 directional predictability；
9. 不解锁任何后续 Agent/Router；
10. 最终只输出：
   - `AGENT-B-REDESIGN-PROMISING`
   - `DATASET-ROBUSTNESS-FAIL`
   - `SPECTRAL-PRIVATE-TASK-REDESIGN-REQUIRED`

---

# 39. 本轮最重要的一句话

现在不要问：

> “怎么让整个四 Agent 系统更强？”

只问：

> **“能不能先做出一个在 WiSig 和 ORACLE 上都真正有用、并且能救 Temporal Agent 某类错误的 Agent B？”**

如果这个最基本的问题都没有解决，后面的通信、Router 和多智能体都没有意义。
