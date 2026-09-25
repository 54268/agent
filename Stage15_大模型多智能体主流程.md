# Stage 15：大模型多智能体开放集辐射源识别

## 1. 当前正式主线

Stage 15 采用：

**Partially Observable Heterogeneous LLM Multi-Agent + RF Tools + Independent Memory**

Stage 14 的 Comm-MAPPO/CTDE 不删除，冻结为传统 MARL baseline。

## 2. 三个真正的 Agent

### Proposer Agent
只接收 identity-side 局部观测，例如 Top-1、Top-2、概率、margin、entropy。

私有工具：
- identity prototype
- identity energy / distance margin

任务：
- 形成已知类假设；
- 判断是否需要更多身份侧证据；
- 将必要证据通过显式消息发给其他 Agent。

### Critic Agent
只接收 geometry/open-set-side 局部观测。

私有工具：
- geometry prototype
- geometry margin
- OpenMax
- boundary
- multi-view geometry risk

任务：
- 主动寻找 Proposer 假设的反证；
- 分析开放集风险与域偏移迹象；
- 支持或质疑当前候选；
- 将必要证据显式发送给其他 Agent。

### Arbiter Agent
不直接读取原始 RF 数值，不拥有 RF 工具。

任务：
- 读取 Proposer / Critic 发来的消息；
- 证据不足时主动向指定 Agent 追问；
- 最终执行 Known / Unknown 裁决。

## 3. 为什么这一版属于实质多智能体

系统明确具备：

1. 独立 Agent 决策主体；
2. 角色异构；
3. 局部/部分可观测；
4. 私有工具；
5. 独立 Case Memory；
6. 显式 Agent-to-Agent 消息；
7. 信息不对称；
8. 联合开放集目标；
9. Agent 行为相互依赖；
10. Ground Truth / provenance / formal unknown 防泄漏。

共享同一个 Qwen 权重并不代表是同一个 Agent。三者不共享上下文、私有观测、私有工具结果和记忆；也支持未来为三个角色配置不同模型后端。

## 4. RF 模块不是 Agent

以下模块全部属于 Agent 可调用的 RF 工具或感知层：

- classifier
- prototype
- geometry
- OpenMax
- boundary
- energy / distance margin
- multi-view evidence

不再把“一个专家网络”包装成“一个 Agent”。

## 5. 单一运行入口

仓库根目录：

```powershell
python run_stage15.py
```

快速验证：

```powershell
python run_stage15.py --samples-per-class 4
```

默认配置：

`configs/experiments/stage15_llm_multiagent_oracle.json`

## 6. 大模型服务

默认：

- model: `Qwen/Qwen3-8B`
- endpoint: `http://127.0.0.1:8000/v1`

入口连接 OpenAI-compatible Chat Completions API。

因此运行前需要已经启动对应的本地 vLLM/SGLang 服务，或者把配置里的 model/base_url 换成你实际使用的服务。

## 7. 输出

默认目录：

`results/stage15/llm_multiagent_oracle_fold0/`

### summary.md
人工阅读的最终总结果，包括：

- Known Accuracy
- Unknown Recall
- H-score
- 平均协作轮数
- 平均工具调用
- 平均 Agent 消息数
- Unique trajectories
- 每个 Agent 的动作统计
- RF Tool 使用次数

### summary.json
完整机器可读汇总。

### agent_stats.json
专门统计多智能体是否真正发生协作：

- proposer / critic / arbiter action usage
- tool usage
- mean turns
- mean messages
- mean tool calls
- unique trajectories

### trajectories.jsonl
逐样本保存完整 Agent 行动、工具调用和通信轨迹，用于后续分析典型 Known / Unknown 案例。

## 8. 第一轮实验关注点

除了识别指标，更重要的是检查：

- 三个 Agent 是否产生差异化行为；
- Critic 是否真正改变部分样本的最终裁决；
- Arbiter 是否主动追问；
- 工具调用是否随样本变化；
- unique trajectories 是否明显高于 Stage 14；
- 是否出现退化成固定流程的情况。

下一阶段再正式做 Single LLM / no-communication / no-Critic / shared-global-observation 等消融。
