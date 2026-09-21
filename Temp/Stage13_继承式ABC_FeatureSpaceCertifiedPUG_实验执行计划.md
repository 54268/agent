# Stage-13：继承式 ABC + Feature-Space Certified PUG 实验计划

## 1. 本轮原则

本轮不再重新设计整套系统。

核心原则：

> **之前已经实验验证有效的组件直接复用，只替换当前明确有问题的部分，然后做严格配对实验。**

本轮唯一重点替换：

```text
Stage-12 I/Q interpolation PUG
        ↓
Stage-5 / 旧方案式 Feature-Space Boundary Extrapolation PUG
```

ABC 主体不推翻，Agent D 暂不实现。

---

## 2. 必须直接复用的组件

### A / B
直接复用 Stage-5 已训练并验证有效的：

- Identity Agent；
- Universal Geometry Agent；
- dynamic multi-view gate；
- raw / spectral / envelope / difference / complex 视图；
- prototype / candidate distance；
- 原有 Known 分类能力。

禁止为了本轮方便重新训练弱版本 A/B。

### Stage-5 Open-Set 母体
本轮必须恢复并复用**完整 Stage-5 V4 的 open-set baseline**，不能再只拿 OpenMax 单组件代替：

- Prototype；
- OpenMax / EVT；
- Boundary evidence；
- LCO evidence-rule selection；
- calibration；
- class-conditional threshold；
- Stage-5 已冻结的 communication / audit 逻辑。

C 必须建立在这个强母体上：

```text
final risk = Stage5_full_base_risk + C_delta_risk
```

而不是重新生成一个弱 risk 覆盖 Stage-5。

### Stage-12 已验证正确的部分
继续复用：

- A/B Evidence Report；
- pseudo 生成后 A/B fresh re-evaluation；
- 禁止复制原样本 report；
- C-lite / C-full；
- Certified PUG 闭环；
- PUG USE / IGNORE 思路。

---

# 3. 正常推理流程

```text
                    Input I/Q
                       │
              ┌────────┴────────┐
              ↓                 ↓
          Agent A           Agent B
          Identity          Evidence
              │                 │
              └────────┬────────┘
                       ↓
               Evidence Reports
                       │
        ┌──────────────┴──────────────┐
        │                             │
 disagreement / low confidence   high-confidence agreement
        │                             │
        │                         C-lite
        │                             │
        │                    low risk │ high risk
        │                       │     │
        │                     Known   │
        │                             │
        └────────────────────→ C-full
                              │
                    Stage-5 base evidence
                              +
                      C private evidence
                              │
                  KEEP / CHALLENGE / REJECT
                              │
                       Known / Unknown
```

---

# 4. A/B 给 C 的 Evidence Report

不能只交付类别和 confidence。

至少包括：

- top-1 / top-2；
- confidence；
- margin；
- entropy；
- top-1 / top-2 candidate support；
- prototype distance；
- candidate distance gap；
- 主要 view 权重；
- local anomaly；
- disagreement；
- reason code；
- 是否建议审查。

C 根据 A/B 的“判断 + 理由”审查，而不是自己重新做第三套弱分类。

---

# 5. C 的定位

C 是：

> **第三方 Open-Set Reviewer**

不是第三个 K-way 分类器。

C 拥有自己的开放集知识：

- Stage-5 Prototype；
- OpenMax / EVT；
- class tail；
- Known distribution；
- competition boundary；
- PUG；
- 必要时 reconstruction / consistency。

C 的输出：

```text
KEEP
CHALLENGE
REJECT
```

C 只做增量审查，不覆盖 A/B 已有的强身份能力。

---

# 6. 恢复正确的 Feature-Space PUG

本轮 PUG 直接参考：

```text
旧方案：
D:\learn_pytorch\笔记\方案\os_sei_code

Stage-5：
src/maros_stage5/pseudo_unknown.py
```

不再使用：

```text
(1-eta) * source_IQ + eta * rival_IQ
```

---

## 6.1 构造 Known 特征空间

使用真实训练 Known 的 A/B learned states：

```text
z_A = Identity state
z_B = Geometry state

z_joint = normalize([z_A, z_B])
```

在该空间建立：

- 每类 prototype；
- 类内距离分布；
- kNN local scale；
- nearest rival；
- class tail；
- competition gap；
- local density。

---

## 6.2 筛选边界样本

保留 Stage-5 已验证的候选方式：

```text
competition
disagreement
mixed
```

优先选择：

- 离 own prototype 较远；
- 与 nearest rival gap 较小；
- A/B disagreement；
- low margin；
- 位于类内尾部；
- local density 较低但仍属于正常 Known 的样本。

不随机生成。

---

## 6.3 正确外推

复用 Stage-5 PUG-V2 思路：

```text
outward = z - own_prototype
repel   = z - rival_prototype
```

得到归一化的外推方向，然后：

```text
z_pseudo
=
z
+
eta × local_scale × direction
+
small orthogonal noise
```

其中：

```text
competition_eta2
```

表示：

> 从竞争边界附近的 Known 样本出发，沿 Known 支持域之外外推约 `2 × local_scale`。

候选 `eta` 不针对数据集写 if/else，由 LCO 统一选择，例如：

```text
1.0 / 1.5 / 2.0 / 2.5
```

Stage-5 历史结果中：

- ORACLE 选择过 `competition_eta1.5`；
- WiSig 选择过 `competition_eta2`。

本轮把它们作为已知有效候选，而不是人工固定。

---

# 7. Pseudo 必须重新经过 A/B

生成 `z_pseudo` 后：

```text
A.report_from_state(z_A_pseudo)
B.report_from_state(z_B_pseudo)
```

重新计算：

- logits；
- top-1 / top-2；
- confidence；
- margin；
- prototype distance；
- Evidence Report。

严禁复制 source sample 的原报告。

如现有 A/B 没有 state-level 接口，则新增：

```python
forward_from_state(...)
report_from_state(...)
```

但不重写 A/B encoder。

---

# 8. C 二次认证

C 第一次生成的只能叫：

```text
Pseudo-Unknown Candidate
```

A/B 重新验证后，C 再决定是否进入 Certified Pool。

### Boundary Pseudo
A/B 出现：

- disagreement；
- low margin；
- strong candidate competition；

同时 C 判断已经越出正常 Known support。

### Hard Pseudo
重点保留：

```text
A/B 高置信一致地认成 Known
+
C 根据 Known distribution / tail / OpenMax / boundary trajectory
确认它已经处于支持域之外
```

这是最需要训练 C 的困难样本。

### 丢弃
以下直接丢弃：

- 仍在 Known 核心；
- 掉入另一个 Known 核心；
- 外推过远；
- 极端低密度、不真实；
- 不满足连续边界穿越；
- C 无法可靠确认。

---

# 9. PUG 必须自主判断是否值得使用

不能固定给 PUG 0.20 / 0.25 权重。

LCO 同时选择：

```text
PUG kind
eta
PUG weight
USE / DOWNWEIGHT / IGNORE
```

评价：

```text
ΔH
ΔUnknown Recall
ΔAUROC
ΔOSCR
ΔKnown Accuracy
```

如果 PUG 为负贡献：

```text
IGNORE
```

同时记录为什么负，不能只关闭不分析。

---

# 10. 本轮严格配对实验

所有方法使用：

- 同一 fold；
- 同一 A/B checkpoint；
- 同一 validation / proxy split；
- 同一 Known acceptance；
- 同一随机种子。

依次比较：

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | 完整 Stage-5 V4 | 强母体基线 |
| B1 | Stage-5 + C，无新 PUG | 看 C 审查本身是否有增益 |
| B2 | Stage-5 + C + Feature PUG（未认证） | 看正确外推本身 |
| B3 | Stage-5 + C + Certified Feature PUG | 看闭环认证价值 |
| B4 | B3 + PUG Utility Gate | 看 Agent 自主 USE/IGNORE |
| Old | Stage-12 I/Q interpolation PUG | 作为失败方案对照 |

---

# 11. 必看指标

### Known
- Known Accuracy；
- A-only correct；
- B-only correct；
- disagreement rate。

### Open-set
- Unknown Recall；
- H-score；
- AUROC；
- OSCR；
- FPR95。

### 特别关注
单独统计：

```text
A/B high-confidence agreement Unknown
```

记录：

- 总数；
- Stage-5 能拒掉多少；
- C 新救回多少；
- Certified PUG 新救回多少；
- 新增误杀 Known 数量。

这是本轮最重要的诊断指标。

---

# 12. PUG 质量诊断

每折必须保存：

- seed 数；
- generated 数；
- certified 数；
- boundary / hard 数量；
- certification rate；
- per-class coverage；
- own prototype distance；
- nearest rival distance；
- competition gap；
- local scale；
- boundary crossing ratio；
- nearest Known density；
- fell-into-rival-core ratio；
- too-far ratio；
- A/B confidence before/after；
- A/B disagreement before/after；
- 每个 eta 的生成数 / 认证数 / utility；
- PUG ON/OFF ΔH / ΔAUROC / ΔUR。

---

# 13. 本轮成功标准

本轮优先验证以下四件事：

1. **完整 Stage-5 V4 能力成功恢复**，不再用弱 OpenMax component 冒充母体；
2. Feature-Space PUG 明显优于 Stage-12 I/Q interpolation PUG；
3. Certified PUG 比未经认证的 Feature PUG 更稳定；
4. C 对 `A/B high-confidence agreement Unknown` 出现明确新增救援。

最终目标：

```text
Stage-5 full baseline + C + Certified Feature PUG
```

在 proxy 五折上至少做到：

- Unknown Recall / H-score 不低于 Stage-5；
- 多数 fold 有正增益；
- Known Accuracy 不出现明显损失；
- PUG 负贡献时能自动停用。

达不到则不进入 Agent D，也不继续堆新模块。

---

# 14. 本轮禁止事项

禁止：

- 重写 A/B；
- 把 B 限制成单一 Spectral；
- 删除 dynamic multi-view；
- 用 OpenMax 单组件代替完整 Stage-5 V4；
- 用 I/Q 两类线性混合作为主 PUG；
- pseudo 复制原 source report；
- C 从零覆盖 Stage-5 risk；
- PUG 固定权重；
- 正式 Unknown 参与选模；
- 为 ORACLE / WiSig 写数据集专用 if/else。

---

# 15. 实验完成后的判定

实验完成后只回答三个问题：

### Q1
完整 Stage-5 V4 + C 是否比 Stage-5 本身更好？

### Q2
Feature-Space Certified PUG 是否真正解决了 Stage-12 I/Q 插值造成的问题？

### Q3
C 是否开始救回 A/B “一致且自信地认错”的 Unknown？

如果三项中前两项都不成立：

> 停止继续调 eta / threshold，重新检查 C 的特征空间和边界定义。

如果成立：

> 固定 ABC 拒识体系，再进入下一阶段；Agent D 暂不提前开发。
