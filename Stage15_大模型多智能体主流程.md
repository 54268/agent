# Stage 15：基于大模型的多智能体开放集辐射源识别

Stage 15 将项目主线从传统 MAPPO/CTDE 多智能体强化学习切换为 **LLM-based Multi-Agent + RF Tools + Memory/Knowledge**。

## 1. 新主线

不再将“专家网络”本身定义为 Agent。

三个 Agent 都是大模型决策主体：

- **Proposer Agent**：根据当前 RF 摘要提出已知类候选，并主动决定是否调用额外 RF 工具；
- **Critic Agent**：专门寻找反证、域偏移和未知风险，不允许只做简单附和；
- **Arbiter Agent**：综合前两者意见和已揭示证据，决定继续调用工具、接受已知类或拒绝为未知。

## 2. RF 工具层

Stage 14 已有感知能力全部保留，并降级为可调用工具：

- identity prototype risk
- geometry prototype risk
- OpenMax risk
- frozen boundary risk
- energy / distance margins
- geometry multi-view risks

Agent 不直接读取 Ground Truth、数据来源、formal unknown 标志。

## 3. 大模型接口

默认通过 OpenAI-compatible Chat Completions 接口调用，因此可接：

- 本地 vLLM / SGLang 部署的 Qwen 等开源大模型；
- 任何 OpenAI-compatible 托管模型。

默认配置：

```json
{
  "model": "Qwen/Qwen3-8B",
  "base_url": "http://127.0.0.1:8000/v1"
}
```

## 4. 知识与记忆

当前第一版包含两层：

1. **Global RF Knowledge**：写入系统提示的领域先验；
2. **Case Memory**：只保存无标签的历史公开摘要、已调用工具结果和最终决策。

后续将把这一层升级为：
- 外部 RF 文献 / 方法知识库；
- 跨数据集 RF foundation model 表征；
- 检索增强 RAG；
- 反思与长期记忆。

## 5. 与 Stage 14 的关系

Stage 14 不删除，冻结为传统多智能体强化学习 baseline。

Stage 15 不再使用：
- GRU Actor 作为主决策器；
- centralized critic；
- PPO / GAE / policy clipping 作为主训练算法。

Stage 14 中以下部分继续复用：
- nested Leave-Class-Out 开放集协议；
- formal unknown 防泄漏约束；
- Stage 5 冻结感知模型；
- prototype / geometry / OpenMax / boundary 等证据；
- 评估数据划分。

## 6. 当前代码

| 内容 | 路径 |
|---|---|
| Agent 定义 | `src/maros_stage15/agents.py` |
| 严格动作协议 | `src/maros_stage15/schemas.py` |
| LLM Provider | `src/maros_stage15/providers.py` |
| RF Tools | `src/maros_stage15/tools.py` |
| Memory | `src/maros_stage15/memory.py` |
| Agentic 编排 | `src/maros_stage15/orchestrator.py` |
| 主配置 | `configs/experiments/stage15_llm_multiagent_oracle.json` |
| 运行入口 | `scripts/experiments/run_stage15_llm_multiagent.py` |
| 协议测试 | `tests/stage15/test_agentic_contracts.py` |

## 7. 运行

先启动一个 OpenAI-compatible 本地模型服务，例如 vLLM，然后运行：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage15_llm_multiagent.py --config configs/experiments/stage15_llm_multiagent_oracle.json
```

协议测试：

```powershell
conda run --no-capture-output -n pytorch python -m pytest -q tests/stage15
```

## 8. 下一阶段

第一轮只验证：

1. 单 LLM Agent；
2. Proposer + Arbiter；
3. Proposer + Critic + Arbiter；
4. Stage 14 MAPPO baseline；
5. 固定规则 / 单 MLP evidence fusion baseline。

验证重点不是只看 H-score，还要看：
- 是否真的产生多样化工具调用轨迹；
- Critic 是否改变最终决策；
- 多 Agent 相比单 Agent 是否有增益；
- 工具调用成本和推理延迟；
- 去掉 memory / knowledge / critic 后性能变化。

只有这些实验成立后，再考虑 LoRA / preference optimization / RL fine-tuning。
