# ABC 多智能体开放集拒识阶段计划书（简版）

## 1. 阶段目标

本阶段暂不实现 Unknown Discovery Agent D，先集中解决：

> **A/B 已知类身份判断 + C 开放集审查与未知拒识。**

核心目标不是继续提高 Known 分类精度，而是：

1. 保留 Stage-5 Unified V4 的动态多视图能力；
2. 让 A、B 独立形成身份判断与证据报告；
3. 将 A/B 冲突、低置信度和高风险一致样本交给 C 审查；
4. 利用竞争边界外推、Known 分布和开放空间证据提升 Unknown Recall；
5. 在不查看正式 Unknown 的前提下，用 LCO proxy-Unknown 完成结构与策略选择。

---

## 2. 总体架构

```text
                   输入 I/Q
                      │
              ┌───────┴───────┐
              ↓               ↓
        Agent A            Agent B
        Identity           Evidence
              │               │
              └───────┬───────┘
                      │
               两份 Evidence Report
                      │
          ┌───────────┴───────────┐
          │                       │
   不一致 / 低置信度       一致 / 高置信度
          │                       │
          │                 C-lite 风险筛查
          │                       │
          │               低风险 │ 高风险
          │                 │     │
          │               Known   │
          │                       │
          └──────────────→ Agent C-full
                    Review + Open-Set
                          │
                    ┌─────┴─────┐
                    ↓           ↓
                  Known       Unknown
```

---

## 3. Agent A：Identity Agent

### 目标
回答：

> **“当前信号最可能属于哪个 Known emitter？”**

### 设计原则
- 保留完整 Known 分类能力；
- 不人为削弱；
- 内部允许像 Stage-5 一样动态组合多种证据；
- 不负责最终 Unknown 拒识。

### 可用工具
- raw
- complex-IQ
- envelope
- difference
- spectral 等有限多视图

### 输出
形成简洁的 `Evidence Report`：

- top-1 / top-2；
- logits / probability；
- confidence、margin、entropy；
- 主要使用的证据类型；
- 是否存在候选竞争；
- 是否建议第三方审查。

---

## 4. Agent B：Independent Evidence Agent

### 目标
回答：

> **“从另一套独立表征和物理/几何证据看，它最可能是谁？”**

### 设计原则
- 同样保留完整 Known 分类能力；
- 不再限定为 Spectral Agent；
- 内部允许动态多视图；
- 与 A 使用不同的私有模型、私有表征或训练目标，保证失败模式不完全重合。

### 输出
与 A 相同格式的 `Evidence Report`。

### 关键要求
B 可以比 A 强，不能为了制造协作而故意削弱 B。

实验真正要验证：

> A/B 是否存在稳定的双向独有正确样本与互补错误模式。

---

## 5. A/B 进入 C 的触发规则

### 直接进入 C-full
出现以下任一情况：

1. A/B top-1 不一致；
2. A/B margin 过低；
3. 任一 Agent 置信度低；
4. A/B 证据报告明显冲突；
5. 候选处于明显竞争边界。

这些样本很可能正对应困难 Known 或竞争边界样本。

---

### 高置信一致样本不能直接全部放行

原因：

> Unknown 可能被 A/B 一致且高置信度地错误识别为某个 Known。

因此所有“高置信一致”样本先经过 **C-lite**。

---

## 6. Agent C：Review & Open-Set Agent

C 是本阶段重点。

目标：

> **“A/B 给出的 Known 假设是否真的位于 Known 支持区域内，还是应该拒识为 Unknown？”**

C 不重新从头做 K 类身份分类，而是审查当前候选是否可信。

---

## 7. C-lite：轻量风险筛查

### 作用
对 A/B 一致且高置信度的样本做低成本筛查，避免“自信地一起错”。

### 可使用
- nearest prototype distance；
- class-conditional tail risk；
- lightweight boundary score；
- Known-support score。

### 决策
- 低风险：直接接受为 Known；
- 高风险：进入 C-full。

---

## 8. C-full：深度开放集审查

C-full 重点整合以下私有能力：

### 8.1 Known 分布证据
- prototype；
- 类内距离；
- 类内方差/协方差；
- tail statistics；
- OpenMax / EVT 类证据。

### 8.2 Competitive Boundary / Pseudo-Unknown
恢复以前已经验证过的竞争边界外推思路：

- 挑选低 margin 样本；
- 挑选 A/B disagreement 样本；
- 挑选 Known 类边缘样本；
- 沿类别竞争方向外推；
- 生成 proxy / pseudo-Unknown。

用途：

> 帮助 C 学习“Known 支持域在哪里结束”。

### 8.3 Reconstruction / Consistency
可保留为 C 的工具，而不是独立 Agent：

- reconstruction residual；
- identity-preserving perturbation consistency；
- feature stability。

---

## 9. Competition Boundary 的定位

不额外增加一个正式推理 Agent。

### 训练阶段
作为 C 的训练组件 / Pseudo-Unknown Explorer：

```text
Known 边缘样本
   ↓
类别竞争区域
   ↓
竞争方向外推
   ↓
Pseudo-Unknown
   ↓
训练 C 学习 Known / Open-space 边界
```

### 推理阶段
C 使用已经学到的 boundary / pseudo-unknown knowledge 进行审查。

---

## 10. Evidence Report 与第三方审查

A/B 不只向 C 提交类别和 confidence。

报告至少包含：

- top-1 / top-2；
- margin / entropy；
- 主要证据来源；
- 候选竞争情况；
- 局部异常信息；
- 是否建议审查。

C 根据两份报告决定：

1. 当前冲突属于普通 Known 边界，还是 Open-space 风险；
2. 是否需要调用 prototype / tail / boundary / reconstruction 等工具；
3. 最终接受 Known 还是拒识 Unknown。

---

## 11. 训练与验证协议

继续保持严格 LCO：

- 正式 Unknown 不参与训练；
- 不参与阈值选择；
- 不参与工具选择；
- 不参与 C 的策略选择。

使用 Known 类进行 Leave-Class-Out：

```text
部分 Known 正常训练
+
留出的 Known 类作为 proxy-Unknown
```

先在 proxy-Unknown 上完成所有结构选择与阈值冻结，再一次性测试正式 Unknown。

---

## 12. 本阶段重点指标

### Known 身份能力
- A Known Acc；
- B Known Acc；
- A/B disagreement rate；
- A-only correct；
- B-only correct；
- 双向 rescue rate。

### C 开放集能力
- AUROC；
- OSCR；
- Unknown Recall；
- Known Accuracy；
- H-score；
- FPR95。

### 路由能力
分别比较：

- 全部样本都进 C；
- 只有 disagreement 进 C；
- disagreement + low confidence；
- disagreement + low confidence + C-lite 风险筛查。

目标：

> 在较低审查成本下尽量接近或超过全量 C。

---

## 13. 必须保留的 Baseline

至少比较：

1. A alone；
2. B alone；
3. A/B static mean fusion；
4. A/B confidence fusion；
5. Stage-5 Unified V4；
6. A/B + C-lite；
7. A/B + C-full；
8. A/B + C-lite + C-full + Boundary PUG。

这样才能回答：

> C 是否真的带来了开放集能力，而不是普通融合就足够。

---

## 14. 关键消融

必须做：

- 去掉 Boundary PUG；
- 去掉 prototype / tail；
- 去掉 reconstruction / consistency；
- 只处理 A/B disagreement；
- 高置信一致样本不经 C-lite；
- message / report shuffle；
- candidate pair 替换；
- C 不读取 A/B 私有 hidden state。

---

## 15. 成功判据

本阶段不要求 Agent D。

ABC 阶段成功至少满足：

1. A、B 均保持强 Known 分类能力；
2. A/B 有真实互补，而不是完全冗余；
3. C 明显提高 Unknown Recall / H-score；
4. Boundary PUG 对 C 有可重复正贡献；
5. 高置信一致 Unknown 能被 C-lite/C-full 部分救回；
6. A/B + C 明显优于单独 A、单独 B；
7. 与静态融合相比，至少在开放集指标或审查成本上体现优势；
8. 正式 Unknown 在全部策略冻结前保持不可见。

---

## 16. 当前阶段结论

下一阶段不再扩充 Agent 数量。

先完成：

> **A、B 两个身份判断 Agent + C 第三方开放集审查 Agent。**

其中：

- A/B 负责 Known 世界内部的身份判断；
- C 负责 Known / Unknown 世界边界；
- Competition Boundary / Pseudo-Unknown 作为 C 的核心训练能力；
- 高置信一致样本通过 C-lite 防止“自信地一起错”；
- 只有被 C 最终拒识的样本，未来才进入 Agent D 做 Unknown Discovery。

本阶段最重要的技术目标是：

> **把 WiSig 和 ORACLE 的 Unknown Recall 真正拉起来，同时证明提升来自 C 的开放集审查能力，而不是单纯增加一个更强分类器。**
