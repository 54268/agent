# Stage-14：开放集辐射源识别的真正 Comm-MADRL 实施计划

## 1. 本阶段唯一目标

本阶段不再做“多专家 / 多分支 / 多分类器换名为 Agent”的结构，也暂不花精力做 Single-Agent PPO、完整消融或大规模基线。

当前唯一任务是：

> **先把开放集辐射源识别真正建模为一个多智能体强化学习任务，并让 A/B/C 成为具有局部观察、独立策略、动作选择、通信和状态更新的真实 Agent。**

Stage-5 / Stage-13 中已经验证过的识别与开放集能力全部作为底层专业工具复用，不重新设计。

---

# 2. 多智能体形式正式定为

## Cooperative Heterogeneous Comm-MADRL

三个 Agent 具有共同目标：

> 正确识别 Known 辐射源，并拒识 Unknown 辐射源。

不是对抗博弈。

形式上采用：

- **Dec-POMDP / Dec-POMDP-Comm**
- **CTDE：Centralized Training, Decentralized Execution**
- **MAPPO：Multi-Agent PPO**
- 三个异构 Actor
- 一个 centralized critic
- 显式结构化通信

核心闭环必须满足：

```text
局部观察
   ↓
Agent Policy
   ↓
自主选择 Action
   ↓
通信 / 请求证据 / 调用私有工具
   ↓
环境与 Blackboard 更新
   ↓
获得新的局部 Observation
   ↓
再次决策
```

只要还是：

```text
A 前向一次
→ B 前向一次
→ C 前向一次
→ 固定融合
```

就不算本阶段目标。

---

# 3. 已验证模块全部冻结为 Tools / Perception

## 必须复用

### Stage-5

`src/maros_stage5/agents.py`

保留：

- `IdentityAgent`
- Universal Geometry
- raw
- spectral
- envelope_phase
- difference_iq
- complex_iq
- sample-level dynamic view gate
- Identity / Geometry prototypes

保留 Stage-5 完整开放集证据：

- Prototype distance
- OpenMax / EVT
- Boundary evidence
- reconstruction 能力
- Known-only calibration / threshold 逻辑

### Stage-13

`src/maros_stage13/feature_pug.py`

保留：

- feature-space boundary extrapolation
- competition / disagreement / mixed seeds
- fresh A/B state-level evaluation
- Certified PUG
- Known support / rival-core / too-far 筛选

PUG 本阶段主要用于训练时构造困难 proxy Unknown，不默认作为在线推理动作。

---

# 4. 三个 Agent 的最终职责

## 4.1 Agent A：Identity Investigator

### 私有能力

- Stage-5 Identity backbone
- K 类身份 logits
- confidence
- margin
- entropy
- identity prototype distance
- 当前候选 Known 类

### A 的局部 Observation

只包含：

- A 自己的 Evidence Report
- 已收到的公开消息
- 当前公共 hypothesis
- 当前 step
- 剩余 communication / tool budget
- 自己的历史 action

不直接读取：

- B hidden state
- C private tool state
- centralized critic
- ground truth

### A 的动作

第一版：

```text
PROPOSE_KNOWN(k)
ASK_B
ASK_C
SEND_EVIDENCE_B
SEND_EVIDENCE_C
RECHECK
ABSTAIN
WAIT
```

A 不允许仅凭高 confidence 直接结束整个开放集决策。

---

## 4.2 Agent B：Geometry Verifier

### 私有能力

复用 Stage-5 Universal Geometry：

- raw
- spectral
- envelope_phase
- difference_iq
- complex_iq
- dynamic view weights
- geometry logits
- prototype distance
- view-specific anomaly evidence

### B 的职责

B 不是“第二个分类器”。

B 要回答：

> 当前 A/B 公共候选类，从几何与多视角特征空间看是否成立？

### B 的动作

```text
SUPPORT
CHALLENGE(k)
ASK_A
ASK_C
SEND_EVIDENCE_A
SEND_EVIDENCE_C
RECHECK_VIEW
ABSTAIN
WAIT
```

`RECHECK_VIEW` 不需要重新训练新模型。

它可以让 B 根据当前状态重新聚合已有 view evidence，或调用当前尚未暴露的 view-level evidence。

---

## 4.3 Agent C：Open-Set Auditor

C 不设置新的 K-way 分类头。

它只负责：

> 当前 Known hypothesis 是否位于 Known 支持域内？

### C 的私有 Tools

```text
CHECK_ID_PROTOTYPE
CHECK_GEO_PROTOTYPE
CHECK_OPENMAX
CHECK_BOUNDARY
CHECK_RECONSTRUCTION
```

后续需要时再增加，不在第一轮继续堆工具。

### C 的动作

```text
QUERY_A
QUERY_B
CHECK_ID_PROTOTYPE
CHECK_GEO_PROTOTYPE
CHECK_OPENMAX
CHECK_BOUNDARY
CHECK_RECONSTRUCTION
ACCEPT_CURRENT_KNOWN
REJECT_UNKNOWN
WAIT
```

Known 类身份来自 A/B 当前 hypothesis。

C 决定的是：

```text
ACCEPT / REJECT
```

而不是重新做一次 K 类分类。

---

# 5. Environment：一个样本就是一个 Episode

每个 RF 样本形成一个短 Episode。

建议第一版：

```text
Tmax = 4 ~ 6
```

不用设计很长。

## 初始状态

环境先完成一次冻结的 A/B perception：

```text
IQ
↓
Identity perception
Geometry perception
```

但并不是把所有结果一次性公开给所有 Agent。

每个 Agent 只看到自己的 private observation。

C 的重型开放集工具也不预先全部计算并公开。

---

# 6. 真正的状态转移来自“新证据被获取”

例如：

```text
A：ASK_B
```

环境：

```text
B 收到 query
↓
B policy 决定 SUPPORT / CHALLENGE / SEND_EVIDENCE
↓
消息进入 Blackboard
↓
A 下一时刻 observation 发生改变
```

又例如：

```text
C：CHECK_OPENMAX
```

环境：

```text
执行 OpenMax Tool
↓
该结果由 hidden → revealed
↓
C observation 更新
↓
C 再决定 ACCEPT / REJECT / QUERY
```

因此环境不是“同一个样本重复跑几次网络”。

它是：

> **Sequential Evidence Acquisition Environment**

---

# 7. Blackboard 只作为共享通信状态

Blackboard 不做决策，不是第四个 Agent。

保存：

```text
current_hypothesis
public_messages
sender
receiver
message_type
revealed_public_evidence
step
remaining_budget
```

不能保存并广播：

```text
A hidden state
B hidden state
完整 K-way logits
C private vector
```

除非该信息通过明确的通信动作被转换成允许公开的 message。

---

# 8. 第一版通信协议

根据 Comm-MADRL 综述，第一版采用：

| 维度 | 当前设计 |
|---|---|
| Controlled Goal | Cooperative |
| Communication Constraint | 有通信成本和预算 |
| Communicatee | 其他学习 Agent |
| Communication Policy | Individual learned control |
| Message Content | Existing knowledge |
| Message Combination | 发送者固定槽位 + mask |
| Inner Integration | Policy-level |
| Learning Method | Reinforcement Learning |
| Training Scheme | CTDE |

## 结构化 Message

第一版不要学习不可解释的 32D hidden message。

消息使用低维 Evidence Report：

```text
sender_id
message_type
candidate_class
confidence_bin / normalized_confidence
margin
support_risk
main_evidence_type
reason_code
```

例如：

```text
A → B:
candidate = Tx7
confidence = 0.91
margin = 0.34
reason = IDENTITY_STRONG

B → A:
message = CHALLENGE
candidate = Tx12
geometry_support = 0.42
reason = PROTOTYPE_CONFLICT
```

后续若第一版成功，再研究 learnable latent message。

---

# 9. Policy 网络

不要使用 LLM。

第一版使用小型神经策略网络即可。

因为 A/B/C：

- observation 不同
- action space 不同
- private tools 不同

所以使用：

> **三个独立 Actor 参数**

建议：

```text
Observation
↓
MLP
↓
GRU
↓
Action logits
```

GRU 用于记忆：

- 前几步自己做过什么
- 收到过什么消息
- 哪些工具已经调用过

Actor 不直接访问全局状态。

---

# 10. Action Mask 非常重要

不是所有动作每一步都有效。

例如：

- C 已调用 OpenMax 后，不应无限重复调用
- 没有 current_hypothesis 时不能 ACCEPT_CURRENT_KNOWN
- budget=0 后不能继续通信
- Episode 已 REJECT 后全部结束
- A 未收到新证据时是否允许重复 RECHECK，需要限制

所以每个 policy 输出后应用：

```text
valid_action_mask
```

防止 MAPPO 学大量无意义动作。

---

# 11. CTDE + MAPPO

## Actor

训练和推理都存在：

```text
πA(aA | oA, hA)
πB(aB | oB, hB)
πC(aC | oC, hC)
```

## Centralized Critic

只在训练阶段存在。

Critic 可以看到：

```text
A/B/C observation summary
Blackboard
joint action
revealed tool mask
remaining budget
current step
```

Ground Truth：

> 只用于计算 reward。

不要把 Ground Truth class / Known-Unknown flag 直接作为 critic 输入。

推理时：

```text
删除 Critic
只保留 A/B/C Actors + private tools + Blackboard
```

---

# 12. Reward 第一版只做简单共享奖励

A/B/C 共享 team reward。

终局奖励：

```text
Known 且类别正确       +1
Unknown 且正确拒识    +1
任何错误             -1
```

过程成本：

```text
每多一步           -λ_step
每发一次消息       -λ_msg
每调用重型 Tool    -λ_tool
```

初始可取很小：

```text
λ_step = 0.01
λ_msg  = 0.01
λ_tool = 0.02
```

这些值只作为初始训练设置。

第一轮不要人为设置：

```text
Unknown miss = -3.5
Unknown correct = +1.5
```

避免 reward 本身把系统推成过度拒识。

---

# 13. Unknown 训练协议

真实 Formal Unknown 绝不进入 MARL 训练。

继续使用现有协议：

```text
Known Classes
↓
Leave-Class-Out
↓
Proxy Unknown
```

再加入：

```text
Stage-13 Certified Feature PUG
```

训练 Episode 来自：

```text
Known
+
LCO Proxy Unknown
+
Certified Pseudo Unknown
```

尽量近似平衡 Known / Unknown Episode。

继续调用：

```text
build_nested_lco_protocol
assert_no_formal_unknown
```

禁止用 Formal Unknown 调：

- MAPPO reward 参数
- communication threshold
- tool cost
- early stopping
- PUG 参数
- policy architecture

---

# 14. 第一阶段代码结构

新增：

```text
src/maros_stage14/
    __init__.py

    actions.py
        AAction
        BAction
        CAction
        action masks

    messages.py
        EvidenceMessage
        Blackboard

    observations.py
        build_a_observation()
        build_b_observation()
        build_c_observation()

    env.py
        OSSEIMultiAgentEnv
        reset()
        step()
        terminal decision

    policies.py
        IdentityPolicy
        GeometryPolicy
        OpenSetPolicy

    critic.py
        CentralizedCritic

    buffer.py
        rollout buffer
        recurrent states
        masks

    mappo.py
        PPO clipped loss
        GAE
        value loss
        entropy

    runner.py
        collect episodes
        update MAPPO
        checkpoint

    evaluation.py
        OSR metrics
        trajectory statistics
```

底层直接 import：

```text
maros_stage5
maros_stage13
```

禁止复制并魔改 Stage-5 backbone。

---

# 15. 实现顺序

## Step 1：先做 Environment，不训练 MAPPO

先用随机 policy / scripted policy 测试：

```text
reset
→ observe
→ action
→ message/tool
→ new observation
→ terminal
```

必须确认：

- 三个 Agent observation 不相同
- query 会真正改变 receiver observation
- tool action 会揭示新 evidence
- action mask 正确
- Episode 能正常结束
- Formal Unknown 没进入训练环境

---

## Step 2：接 MAPPO

不要自己从零发明 PPO。

借鉴成熟实现完成：

- rollout collection
- recurrent hidden states
- GAE
- PPO clipping
- entropy
- centralized value
- gradient clipping
- minibatch update

---

## Step 3：只跑 LCO 小规模 Smoke Test

第一轮目的不是追最好 OSR 数字。

先看：

```text
episode reward 是否上升
policy loss 是否正常
critic loss 是否收敛
entropy 是否没有瞬间归零
```

以及行为：

```text
是否出现 ASK
是否出现 SEND
是否会调用 C Tool
是否存在提前结束
是否存在不同路径
```

---

# 16. 第一轮最重要的结构性检查

至少要观察到多种实际 trajectory：

```text
A → C → ACCEPT

A → B → C → REJECT

B → A → C

A → B → ACCEPT

C → QUERY_A → CHECK_OPENMAX → REJECT
```

不要求每一种都有。

但如果所有样本都退化成：

```text
A → B → C
```

说明仍然只是流水线。

如果所有样本：

```text
所有 Agent 每一步全部通信
所有 Tool 每次全部调用
```

说明 communication cost / action design 失败。

如果所有 Agent：

```text
永不通信
```

说明局部 observation 已经足够或者 reward / communication policy 没学起来，需要诊断。

---

# 17. 当前阶段只保留最必要的对照

本阶段暂不做大规模消融。

只保留两个工程基准用于确认没有把旧系统搞坏：

```text
Stage-5 V4 full
Stage-13 current rule-based ABC
```

作用仅是：

> 确认新增 MARL 层至少没有出现明显实现错误。

完整的：

- message off
- shuffled messages
- agent removal
- single-agent control
- 不同通信轮数
- 不同 tool ablation

全部放到系统跑通以后再做。

---

# 18. 推荐借鉴的 GitHub 开源代码

## 18.1 第一优先：官方 MAPPO

Repository：

`https://github.com/marlbenchmark/on-policy`

用途：

> 借 MAPPO 的算法核心，不直接照搬它的环境和 Agent 结构。

重点参考：

```text
onpolicy/algorithms/r_mappo/
onpolicy/algorithms/r_mappo/algorithm/
onpolicy/utils/shared_buffer.py
onpolicy/runner/
```

重点借：

- PPO clipped objective
- recurrent MAPPO
- GAE
- centralized critic
- rollout buffer
- mini-batch update
- value normalization
- gradient clipping

注意：

该仓库默认更偏向同构 Agent，并且 README 明确说明默认共享 policy。

我们的 A/B/C 是异构 Agent：

> **不要共享 Actor 参数。**

应改成：

```text
actor_A
actor_B
actor_C
```

Critic 可以集中。

---

## 18.2 第二优先：PyTorch TorchRL 的 Multi-Agent PPO Tutorial

Repository：

`https://github.com/pytorch/rl`

示例：

`tutorials/sphinx-tutorials/multiagent_ppo.py`

它非常适合我们参考：

- centralized critic 怎么写
- decentralized actors 怎么写
- multi-agent observation 怎么组织
- TensorDict
- rollout collector
- PPO loss

优点：

> 和我们现有 PyTorch 工程更接近，而且代码现代、结构较清晰。

注意：

教程里的 Agent 大多同构。

我们的三 Agent 异构，所以不要强行套 `shared_parameters=True`。

---

## 18.3 PettingZoo：主要借 Environment API 思想

Repository：

`https://github.com/Farama-Foundation/PettingZoo`

它不是 MAPPO 算法库。

主要借鉴：

- multi-agent reset / step
- agent-specific observation
- agent-specific action
- termination
- Parallel / AEC API

我们的第一版更建议：

> 使用 joint step + WAIT + action mask

这样更容易接 MAPPO。

不用为了“标准”强制把整个项目迁移到 PettingZoo。

---

## 18.4 BenchMARL：后期参考，不建议现在作为主工程

Repository：

`https://github.com/facebookresearch/BenchMARL`

它支持 MAPPO，并基于 TorchRL。

优点：

- 训练框架完整
- 配置规范
- MAPPO 已实现
- 统计与 benchmark 方便

缺点：

- 框架较重
- 我们的 A/B/C observation、action 和 private tools 都很异构
- 现在强行迁移会增加工程复杂度

所以当前建议：

> 看其 MAPPO 与 centralized critic 写法，不要第一轮把整个仓库接进项目。

---

## 18.5 EPyMARL：备用参考

Repository：

`https://github.com/uoe-agents/epymarl`

支持：

- MAPPO
- QMIX
- COMA
- VDN 等

可参考：

- 多智能体 batch
- episode runner
- common reward
- config 管理

但其 PyMARL 风格与我们当前代码差异较大。

不是第一选择。

---

# 19. 推荐实际借法

不要：

```text
clone 一个 MAPPO repo
→ 把我们的 RF 网络塞进去
→ 全部重写
```

建议：

```text
现有 agent repo
        +
Stage-5 / Stage-13 frozen tools
        +
自己实现 OSSEIMultiAgentEnv
        +
参考官方 MAPPO 的 PPO/GAE/buffer
        +
参考 TorchRL 的 centralized critic / rollout 组织
```

最合理的组合是：

```text
业务架构：
自己写

MAPPO 数学与训练器：
marlbenchmark/on-policy

PyTorch 工程写法：
TorchRL multiagent PPO tutorial

Environment 接口思想：
PettingZoo
```

---

# 20. 当前阶段成功标准

现在暂时不要求：

> MAPPO 一上来就超过所有 OS-SEI 基线。

第一阶段先证明系统确实成为真正的多智能体：

1. A/B/C 有不同 local observation
2. A/B/C 有独立 Actor policy
3. 每个 Agent 能自主选择多个 action
4. ASK / SEND / TOOL 会导致后续 observation 改变
5. trajectory 不是固定流水线
6. 训练时有 centralized critic
7. 推理时没有 critic
8. Formal Unknown 无泄露
9. reward 能稳定学习
10. 系统可以完整输出 Known / Unknown 决策

做到这一步后：

> 才开始第二阶段追求 OS-SEI 性能，并逐步做通信和 Agent 消融。

---

# 21. 下一步直接执行

建议 Codex 按以下顺序开始：

```text
1. 创建 src/maros_stage14/
2. 定义 A/B/C action enum
3. 定义 EvidenceMessage + Blackboard
4. 把 Stage-5 / Stage-13 包成只读 Tool API
5. 实现 OSSEIMultiAgentEnv
6. 写 scripted/random policy 做 environment smoke test
7. 接入三独立 Actor + centralized critic
8. 借官方 MAPPO 实现 PPO/GAE/buffer
9. 跑一个 LCO fold 小规模训练
10. 输出逐 Episode trajectory 日志
```

在第 6 步以前：

> 不进行正式 MAPPO 长时间训练。

先把“真正的 Agent 环境”验证正确，再开始学习。


---

# 22. 本地三个开源仓库的具体借鉴方案与注意事项

本地参考目录：

```text
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\
├─ on-policy-main
├─ PettingZoo-main
└─ rl-main
```

这三个仓库的职责不要混在一起。

最推荐的分工是：

```text
on-policy-main
→ 借 MAPPO 训练算法骨架

PettingZoo-main
→ 借多智能体 Environment / API 设计思想

rl-main
→ 借现代 PyTorch MARL 数据组织、centralized critic、collector、PPO 写法
```

第一版不要同时把三个仓库作为运行依赖接入项目。

> **把它们视为“参考实现库”，而不是把当前 OS-SEI 项目迁移进三个框架。**

Codex 在实现 Stage-14 时可以直接阅读这三个本地仓库，但默认只读，不修改它们。

---

## 22.1 `on-policy-main`：MAPPO 主算法参考

本地：

```text
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\on-policy-main
```

### 最值得看的文件

优先阅读：

```text
on-policy-main/
└─ onpolicy/
   ├─ algorithms/
   │  └─ r_mappo/
   │     ├─ r_mappo.py
   │     └─ algorithm/
   │        └─ rMAPPOPolicy.py
   │
   ├─ runner/
   │  ├─ separated/
   │  └─ shared/
   │
   └─ utils/
      ├─ separated_buffer.py
      └─ shared_buffer.py
```

以及：

```text
config.py
scripts/train/
```

### 我们主要借什么

主要借：

```text
PPO clipped objective
GAE / return 计算
actor loss
critic loss
entropy regularization
gradient clipping
recurrent state
rollout buffer
mini-batch update
value normalization
training / rollout 分离
```

### 对我们最重要：优先看 `runner/separated`

官方 README 默认强调：

```text
默认所有 Agent 共享一个 policy
```

这不适合我们的 A/B/C。

但仓库实际上提供：

```text
runner/shared/
runner/separated/
```

`separated/base_runner.py` 会针对每一个 Agent：

```text
创建自己的 Policy
创建自己的 Trainer
创建自己的 Buffer
```

并且每个 Agent 可以拥有自己的：

```text
observation_space[agent_id]
action_space[agent_id]
```

这和我们的：

```text
A：Identity action space
B：Geometry action space
C：Open-set action space
```

非常匹配。

因此第一版实现应：

> **优先参考 `separated`，不要照着 `shared` runner 改。**

---

### Centralized Critic 的实现注意

这里需要对之前方案做一个小修正：

> CTDE 并不要求“全系统只能有一个 critic 网络”。

更关键的是：

> **训练时 critic 能看到比 actor 更完整的 global/shared state，而执行时 actor 只看 local observation。**

对于我们的异构 A/B/C，第一版更容易实现成：

```text
Actor A ← oA
Critic A ← global state

Actor B ← oB
Critic B ← global state

Actor C ← oC
Critic C ← global state
```

也就是：

> **三个独立 Actor + 三个 centralized value heads / critics**

这与 `on-policy-main/runner/separated` 的组织方式更一致，也避免因为 A/B/C observation/action 维度完全不同而强行共享网络。

后续若需要再研究：

```text
一个共享 trunk + 三个 value heads
```

第一轮不必做。

---

### 不要直接照搬的地方

#### 1. 不要照搬默认参数共享

我们的 A/B/C 是异构角色。

禁止直接：

```text
one shared actor for all agents
```

#### 2. 不要照搬 MPE / SMAC 环境

这些环境只用于理解 runner 输入输出。

我们的环境必须自己实现：

```text
OSSEIMultiAgentEnv
```

#### 3. 不要照搬旧环境依赖

官方 README 中给出的安装示例非常老，例如旧 Python / PyTorch 版本。

因此：

> **不要为了运行 `on-policy-main` 而降级当前 OS-SEI 主环境。**

第一阶段把它作为源码参考最安全。

如果必须单独运行官方示例：

```text
建立独立 conda 环境
```

不要污染现有项目环境。

#### 4. 不要把 episode length 照搬

SMAC 可能几十到几百步。

我们的 OS-SEI：

```text
Tmax ≈ 4~6
```

是短时序证据交互。

#### 5. 不要照搬 reward normalization / active mask 语义

需要先确认：

```text
done
truncated
active mask
available actions
```

在我们的环境里具体代表什么。

特别是：

```text
available_actions
```

可以很好地借来实现 A/B/C 的 action mask。

---

### 建议实际复用的程度

推荐：

```text
算法逻辑复用  ★★★★★
Runner思想    ★★★★☆
Buffer思想    ★★★★★
Environment   ★☆☆☆☆
网络结构      ★★☆☆☆
```

不要把整个仓库复制进 `agent` 项目。

更合理的是：

```text
阅读 on-policy-main
↓
在 src/maros_stage14/ 中实现适合我们异构 Agent 的简化版 MAPPO
```

---

## 22.2 `PettingZoo-main`：环境规范参考

本地：

```text
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\PettingZoo-main
```

### 它不是 MAPPO 算法库

PettingZoo 主要解决：

> 多个 Agent 的 Environment 应该怎样规范组织。

重点看：

```text
PettingZoo-main/
└─ pettingzoo/
   ├─ utils/
   │  └─ env.py
   ├─ mpe/
   ├─ butterfly/
   └─ test/
```

还可以重点找：

```text
AECEnv
ParallelEnv
observe()
action_space()
observation_space()
state()
terminations
truncations
rewards
infos
```

---

### 我们主要借什么

#### 1. Agent-specific observation

借：

```text
observe(agent)
```

思想。

我们的环境应明确：

```text
observe("identity")
observe("geometry")
observe("open_set")
```

返回不同内容。

这是防止重新退化成“所有模块共享全部信息”的关键。

---

#### 2. Agent-specific action space

PettingZoo 支持：

```text
action_space(agent)
```

我们的三者应该天然不同：

```text
AAction
BAction
CAction
```

不要为了代码方便，把 A/B/C 强行做成完全一样的动作语义。

---

#### 3. `state()` 与 CTDE

PettingZoo 的 `state()` 表示：

> 给集中训练使用的 global view。

这非常适合我们的 centralized critic。

可以设计：

```text
env.state()
```

返回训练时全局状态，例如：

```text
A observation summary
B observation summary
C revealed evidence
Blackboard
tool mask
message history summary
remaining budget
current step
```

注意：

> `state()` 只能给 Critic / training pipeline 使用。

不能在执行时偷偷送进 A/B/C Actor。

---

#### 4. termination / truncation 要区分

我们可以定义：

```text
termination:
    正常 ACCEPT / REJECT

truncation:
    达到 Tmax 仍未决策
```

这样比只用一个 `done=True` 更清楚。

---

### AEC API 还是 Parallel API？

PettingZoo 的 AEC 是：

```text
Agent A 动
→ Agent B 动
→ Agent C 动
```

如果我们直接照搬 AEC，很容易人为制造：

```text
固定 A→B→C 流水线
```

这与我们当前目标冲突。

因此第一版不建议直接使用 AEC 的固定 agent cycle。

更推荐借其 API 思想，但我们自己的环境使用：

```text
joint decision round
```

每一轮：

```text
A action
B action
C action
```

同时提交。

没有实际动作的 Agent：

```text
WAIT
```

然后 Environment 统一处理：

```text
messages
queries
tool calls
terminal actions
```

再产生下一时刻三个 observation。

这更方便 MAPPO rollout。

如果后续确认任务更适合异步交互，再考虑 AEC。

---

### Windows 注意事项

你现在三个仓库下载在 Windows 本地。

PettingZoo 官方当前说明主要维护 Linux / macOS，Windows 并不是其主要官方支持平台。

因此：

> 本地 Windows 用于阅读源码没有任何问题。

但如果要安装其完整环境依赖：

```text
不要第一时间 pip install pettingzoo[all]
```

因为会拉很多完全无关的 Atari / SISL 等依赖。

如果我们的 Stage-14 只是自己写 environment：

> 甚至可以不把 PettingZoo 作为运行依赖。

只借接口和环境组织思想即可。

---

### 不要照搬的地方

不要照搬：

```text
Atari环境
Butterfly环境
MPE物理环境
游戏奖励设计
固定 agent iteration 顺序
```

我们真正要借的是：

```text
observation ownership
action ownership
state()
termination
truncation
info
environment validation
```

---

### 建议复用程度

```text
Environment API思想   ★★★★★
Agent生命周期思想     ★★★★☆
state() / CTDE思想    ★★★★★
直接运行依赖          ★★☆☆☆
环境代码直接复制      ★☆☆☆☆
```

---

## 22.3 `rl-main`：TorchRL 现代实现参考

本地：

```text
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\rl-main
```

这是 PyTorch 官方 TorchRL 仓库。

### 第一优先阅读文件

直接先看：

```text
rl-main/
└─ tutorials/
   └─ sphinx-tutorials/
      └─ multiagent_ppo.py
```

这是最值得看的文件。

它完整演示：

```text
multi-agent observation
decentralized actor
centralized critic
MAPPO / IPPO
parameter sharing
collector
rollout
PPO loss
GAE
```

---

### 我们主要借什么

#### 1. MAPPO 与 IPPO 的区别

教程明确区分：

```text
MAPPO:
critic 看 global / concatenated observations

IPPO:
critic 只看 local observation
```

我们当前明确选择：

```text
MAPPO / CTDE
```

所以借它 centralized critic 的组织思路。

---

#### 2. Actor 必须 decentralised

教程里的关键思想是：

```text
centralised=False
```

Actor 根据 local observation 做动作。

这与我们的定义一致：

```text
πA(a|oA)
πB(a|oB)
πC(a|oC)
```

---

#### 3. Parameter sharing 是选择，不是多智能体要求

TorchRL 教程默认示例为了同构机器人方便，会使用：

```text
share_parameters_policy = True
```

但教程本身也明确指出：

> 共享参数会使 Agent 更同质化。

我们的角色天然异构，所以：

```text
不要使用教程里的共享 Actor
```

第一版直接：

```text
actor_A
actor_B
actor_C
```

三个独立 module。

---

#### 4. TensorDict 数据组织值得参考

TorchRL 对 trajectory 的组织很成熟。

特别适合参考：

```text
time t
time t+1
observation
action
reward
done
log_prob
value
hidden state
```

如果后续自己写 buffer 时容易乱，可以参考它这种结构。

但第一版不要求必须全面迁移 TensorDict。

---

#### 5. Collector / GAE / ClipPPOLoss 值得参考

可以重点学习：

```text
Collector
ClipPPOLoss
GAE
ReplayBuffer
SamplerWithoutReplacement
```

但是我们已经决定算法骨架主要参考 `on-policy-main`。

因此不要同时：

```text
用 on-policy 的 buffer
+
用 TorchRL 的 collector
+
用另一个 PPO loss
```

混成一个难以排查的训练器。

---

### 对我们最大的直接限制：教程主要面向同构 Agent

TorchRL `multiagent_ppo.py` 中的示例 Agent：

- observation shape 相同
- action shape 相同
- 任务角色相同
- 可以 parameter sharing

而我们的：

```text
A/B/C observation维度可能不同
A/B/C action个数不同
角色完全不同
```

所以不要直接把：

```text
MultiAgentMLP
```

一套代码套给三个 Agent。

更合理：

```text
普通 MLP/GRU Actor A
普通 MLP/GRU Actor B
普通 MLP/GRU Actor C
```

需要 TorchRL 时，各自封成 TensorDictModule 即可。

---

### 版本注意

`rl-main` 是持续更新的开发仓库。

最新 TorchRL 对：

```text
Python
PyTorch
TensorDict
```

版本有明确配套要求。

因此：

> **不要直接在现有 OS-SEI 环境里 `pip install -e rl-main` 后让 pip 自动升级/降级整个 PyTorch 栈。**

第一阶段：

```text
只阅读源码
```

如果决定正式采用 TorchRL 组件，再单独确认：

```text
当前 Python
当前 torch
当前 tensordict
当前 CUDA
```

是否兼容。

---

### 建议复用程度

```text
MAPPO概念理解       ★★★★★
centralized critic  ★★★★★
数据组织            ★★★★★
GAE/PPO写法         ★★★★☆
直接照搬网络        ★★☆☆☆
整个框架迁移        ★★☆☆☆
```

---

# 23. 三个仓库不要“拼装式混用”

非常重要。

第一轮禁止出现：

```text
PettingZoo env
+ TorchRL collector
+ on-policy replay buffer
+ TorchRL PPO loss
+ on-policy runner
```

这种混搭。

这会导致：

- shape 对不上
- done 语义不一致
- recurrent state 生命周期混乱
- action mask 格式不一致
- return / GAE 计算重复
- 很难判断 bug 属于哪里

---

# 24. 当前推荐的唯一实现组合

第一轮建议：

## Environment

自己写：

```text
OSSEIMultiAgentEnv
```

参考：

```text
PettingZoo-main
```

但不必继承 PettingZoo 类。

---

## MAPPO Trainer

主要参考：

```text
on-policy-main/onpolicy/algorithms/r_mappo/
on-policy-main/onpolicy/runner/separated/
```

在 Stage-14 中写一个简化版本。

---

## Modern PyTorch cross-check

用：

```text
rl-main/tutorials/sphinx-tutorials/multiagent_ppo.py
```

检查：

- centralized critic 是否正确
- actor 是否 truly decentralized
- GAE / PPO 流程是否合理
- rollout tensor shape 是否合理

---

# 25. 对 Codex 的具体要求

Codex 开始实现前，应先读取：

```text
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\on-policy-main

D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\PettingZoo-main

D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\多智能体GitHub\新一轮多智能体\rl-main
```

但：

> 这三个目录只作为 Reference，不得直接修改。

同时读取当前项目：

```text
54268/agent
```

重点：

```text
src/maros_stage5/
src/maros_stage13/
```

然后再创建：

```text
src/maros_stage14/
```

---

# 26. 开始实验前的代码检查清单

必须确认：

- [ ] A/B/C 使用三个独立 Actor
- [ ] A/B/C observation 真的不同
- [ ] A/B/C action space 真的不同
- [ ] critic 输入包含 global state
- [ ] actor 看不到 global state
- [ ] Formal Unknown 没进训练
- [ ] `ASK` 会产生新的 receiver observation
- [ ] `CHECK_TOOL` 会从 hidden evidence 变成 revealed evidence
- [ ] action mask 生效
- [ ] 达到 `Tmax` 会 truncation
- [ ] ACCEPT / REJECT 会 termination
- [ ] trajectory buffer 保存 action/log_prob/value/reward/mask
- [ ] recurrent hidden state 在 episode reset 时正确清零
- [ ] Stage-5 backbone 默认冻结
- [ ] Stage-13 PUG 默认不在线调用
- [ ] 不存在固定 `A→B→C` 调度器

---

# 27. 第一轮实验输出必须保存

除了 OSR 指标，还必须保存：

```text
episode_id
sample_id
step
agent
local_observation_summary
valid_action_mask
selected_action
action_probability
message_sender
message_receiver
message_type
tool_called
revealed_evidence
current_hypothesis
remaining_budget
reward
termination_reason
```

建议额外保存一份可读 trajectory：

```text
Episode 103
t0 A: ASK_B
t0 B: SEND_CHALLENGE(Tx12)
t0 C: WAIT

t1 A: ASK_C
t1 B: SUPPORT(Tx12)
t1 C: CHECK_OPENMAX

t2 C: REJECT_UNKNOWN
END
```

第一轮测试阶段：

> **trajectory 是否像真正 Agent 在决策，比最终多提升 0.3% 更重要。**

先确认系统没有再次退化成“多个专家模块按固定流程跑一遍”。
