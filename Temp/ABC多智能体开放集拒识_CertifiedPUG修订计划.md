# ABC 多智能体开放集拒识方案（修订版）

## 1. 当前判断

这版整体逻辑已经基本闭合，可以进入下一轮实现与验证。

核心结构是：

> **A/B 负责身份判断并提交证据报告；C 负责第三方开放集审查；C 生成的伪未知必须重新经过 A/B/C 联合认证，只有认证通过的样本才能进入伪未知池。**

本阶段暂不实现 Agent D，先把 ABC 的 Known / Unknown 拒识做好。

---

## 2. 总体流程

```text
                    输入 I/Q
                       │
               ┌───────┴───────┐
               ↓               ↓
           Agent A          Agent B
           Identity         Evidence
               │               │
               └───────┬───────┘
                       │
                两份 Evidence Report
                       │
         ┌─────────────┴─────────────┐
         │                           │
  不一致 / 低置信度           一致 / 高置信度
         │                           │
         │                    C-lite 风险筛查
         │                           │
         │                   低风险 │ 高风险
         │                     │     │
         │                   Known   │
         │                           │
         └──────────────────→ Agent C-full
                        Review + Open-Set
                              │
                        ┌─────┴─────┐
                        ↓           ↓
                      Known       Unknown
```

---

# 3. Agent A：Identity Agent

目标：

> **判断“这个信号是谁”。**

要求：

- 保留完整 Known 分类能力；
- 不人为削弱；
- 保留 Stage-5 的动态多视图能力；
- 可自主使用 raw / complex / envelope / diff / spectral 等证据；
- 不单独负责最终 Unknown 拒识。

### A 输出 Evidence Report

至少包含：

- top-1 / top-2；
- confidence；
- margin；
- entropy；
- candidate-level support；
- 主要使用的证据类型；
- 是否存在明显竞争；
- 当前判断异常点；
- 是否建议第三方审查。

---

# 4. Agent B：Independent Evidence Agent

目标：

> **用与 A 不同的私有模型、表征和证据独立判断身份。**

要求：

- 保留完整分类能力；
- 不再固定成 Spectral Agent；
- 同样允许动态使用多种证据；
- 与 A 的失败模式不能完全重合；
- B 可以比 A 强，不为了制造协作而故意削弱 B。

B 同样提交完整的 `Evidence Report`。

---

# 5. A/B 如何触发 C

## 5.1 直接进入 C-full

出现以下任一情况：

- A/B top-1 不一致；
- confidence 低；
- margin 低；
- A/B Evidence Report 明显冲突；
- 候选处于明显竞争边界。

这部分样本天然适合做第三方审查。

---

## 5.2 高置信一致样本不能全部直接放行

原因：

> A/B 可能对 Unknown 一致且高置信度地判断成某个 Known。

因此高置信一致样本先经过 `C-lite`。

### C-lite

只进行低成本 Known-support 检查，例如：

- prototype distance；
- class-conditional tail risk；
- lightweight boundary risk；
- Known-support score。

低风险：

> 直接接受 Known。

高风险：

> 进入 C-full。

---

# 6. Agent C：Review & Open-Set Agent

C 的定位不是第三个弱分类器，而是：

> **读取 A/B 的判断和证据报告，再利用自己独有的开放集知识进行第三方审查。**

C 不重新从头做 K 类分类。

C 要回答：

> **“A/B 当前提出的 Known 假设是否真的成立，还是应该拒识为 Unknown？”**

---

# 7. C 的私有工具

C 可以调用：

- Prototype / Known distribution；
- 类内距离与 tail statistics；
- OpenMax / EVT；
- Competition Boundary；
- Pseudo-Unknown；
- Reconstruction；
- Consistency / stability。

这些是 C 的 Tool，不额外拆成新的 Agent。

C 根据当前样本自行决定：

```text
USE
IGNORE
ABSTAIN
```

而不是固定所有 Tool 都参与。

---

# 8. C 的决策形式

C 不应该覆盖 A/B 的完整判断，而是做增量审查。

建议动作：

```text
KEEP
CHALLENGE
REJECT
```

含义：

- `KEEP`：A/B 的 Known 判断可信；
- `CHALLENGE`：当前候选存在问题，需要重新审查或调整；
- `REJECT`：当前样本应判为 Unknown。

最终风险可以采用：

```text
Stage-5 baseline open-set risk
+
C 的增量审查证据 Δrisk
```

而不是重新从零训练一个弱拒识器替代 Stage-5。

---

# 9. Competition Boundary / PUG 的正确定位

Competition Boundary 不单独作为推理 Agent。

它属于 C 的训练能力。

目标：

> **生成靠近 Known 支持边界、但已经开始脱离 Known 的困难伪未知。**

不能追求“越远越 Unknown”。

---

# 10. Pseudo-Unknown 必须形成闭环认证

以前的问题：

```text
C 生成
→ 直接标 Unknown
→ 加入训练
```

这种做法质量没有保证。

下一版必须改成：

```text
Known 边界样本
      ↓
C 生成 Pseudo-Unknown Candidate
      ↓
重新送回完整流程
      ↓
┌──────────────┐
↓              ↓
A 重新判断     B 重新判断
↓              ↓
新的 Evidence Reports
      ↓
      C 再审查
      ↓
┌──────────────────┐
│                  │
认证通过            认证失败
│                  │
↓                  ↓
Certified          丢弃 /
Pseudo-Unknown     重新生成
```

---

# 11. 伪未知认证标准

重点保留两类高价值伪未知。

## 11.1 Boundary Pseudo-Unknown

A/B 出现：

- disagreement；
- low margin；
- candidate competition。

同时 C 判断：

> 已经脱离正常 Known 支持域。

保留。

---

## 11.2 Hard Pseudo-Unknown

这是最重要的一类。

A/B：

> 高置信、一致地判断成某个 Known。

但 C：

> 根据 prototype / tail / boundary / OpenMax 等证据确认其已超出 Known 支持域。

这种样本重点保留，因为它最接近：

> **“Unknown 被模型自信地认成 Known”**

这一真实开放集难点。

---

## 11.3 应丢弃的伪未知

- C 自己也无法确认；
- 仍明显处在 Known 核心区；
- 外推过远，成为不真实离群点；
- 与真实信号分布严重脱节；
- 只有生成器自己认为是 Unknown，但 A/B/C 联合验证不支持。

---

# 12. PUG 生成后必须重新跑 A/B

严禁：

> 生成 pseudo 后直接复制原 Known 样本的 A/B confidence、margin 或 report。

Pseudo 样本必须重新经过 A、B 前向推理，得到新的 Evidence Reports。

如果当前生成空间无法重新送入 A/B：

> 说明该空间不适合作为最终 PUG 生成空间。

优先考虑：

- 可重新构造的 I/Q 空间；
- 或可被 A/B 重新消费的共享可解码表示空间。

---

# 13. PUG 必须具备自我价值判断

当前 PUG 不能固定占某个权重。

需要使用 LCO 做 counterfactual utility：

```text
loss_without_PUG
vs
loss_with_PUG
```

如果：

```text
utility(PUG) <= 0
```

C 应该：

```text
降低 PUG 权重
或
直接 IGNORE
```

不能出现：

> WiSig 五折都负贡献，但正式流程仍固定使用 PUG。

---

# 14. PUG 负贡献必须追查原因

不能只简单关闭。

需要重点检查：

1. 生成空间是否与 A/B 判别空间一致；
2. 外推方向是否真的穿过竞争边界；
3. eta 是否导致样本外推过远；
4. pseudo 是否掉进其他 Known 类；
5. pseudo 与真实 Known 的 kNN 距离是否异常；
6. A/B 对 pseudo 的 confidence / margin 是否符合预期；
7. C 是否只是生成了“太容易”的假 Unknown；
8. ORACLE 与 WiSig 的边界几何是否存在明显差异。

---

# 15. Stage-5 能力必须继承

下一版不能丢掉 Stage-5 Unified V4 已经验证有效的开放集能力。

C 应优先继承：

- multi-view Geometry evidence；
- Prototype；
- OpenMax / EVT；
- Boundary；
- competition PUG；
- class-conditional calibration；
- LCO-selected evidence rule。

C 的定位是：

> **在强 Stage-5 拒识 baseline 上做第三方增量审查，而不是重新造一个更弱的 Unknown detector。**

---

# 16. 训练与验证协议

继续保持严格 LCO：

- 正式 Unknown 不参与训练；
- 不参与阈值选择；
- 不参与 PUG 策略选择；
- 不参与 Tool utility 选择；
- 不参与 C 的动作策略选择。

流程：

```text
Known 类
  ↓
Leave-Class-Out
  ↓
proxy-Unknown
  ↓
选择 / 冻结 ABC 与 PUG 策略
  ↓
最后一次性测试正式 Unknown
```

---

# 17. 必须比较的 Baseline

至少包含：

1. A alone；
2. B alone；
3. A/B static mean；
4. A/B confidence fusion；
5. Stage-5 Unified V4；
6. Stage-5 + C-lite；
7. Stage-5 + C-full；
8. Stage-5 + C + 原始 PUG；
9. Stage-5 + C + Certified PUG。

核心问题：

> **Certified PUG + C 是否真正优于原 Stage-5，而不是只是增加复杂度。**

---

# 18. 关键指标

### Known
- Known Accuracy；
- A/B disagreement；
- A-only correct；
- B-only correct；
- 双向 rescue。

### Open-set
- Unknown Recall；
- H-score；
- AUROC；
- OSCR；
- FPR95。

### PUG
- 生成数量；
- 认证通过率；
- Hard Pseudo-Unknown 比例；
- 过远离群比例；
- per-fold ΔH / ΔAUROC；
- PUG ON / OFF counterfactual utility。

---

# 19. 当前成功判据

ABC 阶段至少满足：

1. A、B 保持强 Known 分类能力；
2. A/B 具有真实互补；
3. C 明显提升 Unknown Recall / H-score；
4. C 不作为弱分类器覆盖 A/B；
5. 高置信一致 Unknown 能被 C 审查出来；
6. Certified PUG 相比未经认证 PUG 更稳定；
7. PUG 负贡献时 C 能自动停用；
8. PUG 生成逻辑在 WiSig / ORACLE 均经过独立诊断；
9. Stage-5 baseline + C 不低于原 Stage-5 的核心开放集能力；
10. 正式 Unknown 始终保持不可见，直到策略完全冻结。

---

# 20. 本阶段最终结构

```text
A：Identity Agent
    ↓
独立身份判断 + Evidence Report

B：Independent Evidence Agent
    ↓
独立身份判断 + Evidence Report

A/B
    ↓
冲突 / 低置信度 / 风险一致样本
    ↓
C：Review & Open-Set Agent
    ↓
读取 A/B 报告
    ↓
调用自己的开放集 Tool
    ↓
KEEP / CHALLENGE / REJECT
    ↓
Known / Unknown
```

训练阶段额外形成：

```text
C 发现边界薄弱区域
      ↓
生成 Pseudo-Unknown Candidate
      ↓
A/B 重新验证
      ↓
C 再审查
      ↓
Certified Pseudo-Unknown
      ↓
用于训练 / 校准 C
      ↓
LCO 检查真实净贡献
      ↓
正贡献：保留
负贡献：停用并诊断原因
```

---

# 21. 当前结论

目前这版逻辑已经比较完整。

相比前几轮，它解决了几个核心问题：

- 不再把 Agent 固定成单一 View；
- 不人为削弱 B；
- C 不再是第三个弱分类器；
- A/B 必须向 C 交付可解释的样本级 Evidence Report；
- C 的职责变成真正的第三方开放集审查；
- PUG 不再“自己生成、自己认定”；
- 每个 pseudo 都必须经过 A/B/C 闭环认证；
- PUG 本身也必须根据 LCO 实际贡献决定是否使用；
- 正式 Unknown 继续保持不可见。

因此下一步建议：

> **停止继续扩充 Agent 数量，直接实现 ABC + Certified PUG 闭环，并先验证是否能恢复并超过 Stage-5 Unified V4 的开放集能力。**
