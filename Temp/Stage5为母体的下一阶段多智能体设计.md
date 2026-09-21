# 下一阶段多智能体开放集识别设计思路

## 1. 当前判断

Stage-5 Unified V4 不应该被推翻。

它最有价值的地方是：

> **同一个 Agent / 同一套算法可以根据不同数据和不同样本，自主改变证据组合。**

例如：

- ORACLE 更依赖 `raw + complex`，几乎不用 pure spectral；
- WiSig 主要依赖 `complex`，同时保留部分 spectral / envelope / diff。

这说明：

> **Agent 不应该被固定成某一种 View。**

之前把 Agent 强行拆成 Temporal / Spectral，虽然边界清楚，但把 Agent 变成了固定输入—固定输出的模块，反而削弱了智能体应有的自主性。

---

## 2. Stage-5 真正的问题

Stage-5 的问题不是“Agent 内部用了多个 View”。

真正的问题是：

1. 部分 Agent 能力过大，接近一个完整多视图算法；
2. Calibration、Threshold、Coordinator 等更像 Service，不应该都叫 Agent；
3. Agent 之间主要还是融合，没有充分证明“主动询问—获得新信息—改变判断”的协作价值。

因此需要改的是：

> **Agent 的职责边界和协作方式，而不是删除它内部的自适应多证据能力。**

---

## 3. 下一版 Agent 定义

下一版不再采用：

```text
一个 View = 一个 Agent
```

而采用：

```text
一个问题目标 = 一个 Agent
```

建议先保留 3 个核心 Agent。

### Agent A：Identity Agent

目标：

> 判断“当前信号是谁”。

内部可以自主使用：

- raw
- complex
- envelope
- difference
- 部分 spectral

但它只负责身份判断，不拥有完整开放集 memory 和最终 Unknown 决策权。

---

### Agent B：Independent Evidence / Geometry Agent

目标：

> 从另一套物理、几何或硬件证据验证 Identity Agent 的候选。

它也可以拥有多个工具，并根据样本自主选择，而不是被固定成 Spectral Agent。

重点是：

> A、B 的目标、私有模型和私有知识不同，但都允许内部自适应使用多种证据。

---

### Agent C：Open-Set Agent

目标：

> 判断“虽然最像某个 Known，但它是否真的属于这个 Known”。

它可以使用：

- prototype
- tail statistics
- reconstruction
- consistency
- open-space evidence

它不负责从头做完整身份分类。

---

## 4. Tool 和 Agent 必须分开

`raw / complex / spectral / envelope / diff / reconstruction / prototype`

这些都应该是 **Tool**，不是 Agent。

Agent 负责：

```text
选择工具
→ 分析当前样本
→ 形成判断
→ 判断是否还缺证据
→ 主动询问其他 Agent
→ 根据回复修改判断
```

---

## 5. 下一版协作方式

保留 Stage-5 的：

> **Agent 内部动态多证据选择**

结合 Stage-7 的：

> **条件咨询**

再结合 Stage-8 的：

> **candidate-level evidence update**

最终流程应类似：

```text
Identity Agent 先形成候选
        ↓
如果证据充分 → 直接输出
        ↓
如果 Tx7 / Tx12 不确定
        ↓
主动询问 Evidence Agent
        ↓
对方针对 Tx7 / Tx12 重新分析
        ↓
返回候选级证据
        ↓
更新 Tx7 / Tx12 判断
        ↓
再由 Open-Set Agent 检查该 Known 是否可信
        ↓
Known / Unknown
```

---

## 6. 下一轮代码原则

下一阶段建议：

> **以 Stage-5 Unified V4 为母体重构，而不是继续修固定 Temporal / Spectral Agent。**

重点保留：

- 动态多视图使用；
- LCO；
- PUG；
- Reconstruction；
- Boundary；
- strong OSR backbone。

重点重做：

- Agent 职责划分；
- Tool / Agent 分离；
- 主动询问机制；
- candidate-level evidence；
- Agent 间真正的信息交换。

Calibration、Threshold、Fusion、Coordinator 统一降级为 Service。

---

## 7. 最终目标

下一版需要证明的不是：

> “三个 Agent 都很强”。

而是：

> **每个 Agent 都有自己的目标和私有知识；内部可以自主使用多个工具；当自身证据不足时，会主动向另一个 Agent 请求自己没有的信息；收到回复后，具体身份或 Known/Unknown 判断发生改变。**

这才是下一步真正需要实现的多智能体开放集算法。
