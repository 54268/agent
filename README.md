# MAROS-SEI：大模型多智能体开放集辐射源识别

当前项目主线已切换到 **Stage 15：LLM-based Multi-Agent + RF Tools + Memory/Knowledge**。

传统 A+C Comm-MAPPO Stage 14 已冻结为 baseline，不再作为后续主线继续扩展。

## 从这里开始

1. 阅读根目录 [`Stage15_大模型多智能体主流程.md`](Stage15_大模型多智能体主流程.md)：新主线架构、Agent角色、RF工具、知识/记忆与运行方法；
2. Stage 15 代码位于 `src/maros_stage15/`；
3. 默认配置位于 `configs/experiments/stage15_llm_multiagent_oracle.json`；
4. 传统 MARL 主线保留在 [`多智能体开放集拒识_主流程.md`](多智能体开放集拒识_主流程.md)，作为 Stage 14 baseline。

## Stage 15 角色

- **Proposer Agent**：形成已知身份假设并主动调用 RF 工具；
- **Critic Agent**：主动寻找反证、域偏移与未知风险；
- **Arbiter Agent**：综合证据并执行 Known / Unknown 最终裁决。

Agent 不再等同于某个分类器。Identity classifier、Geometry、Prototype、OpenMax、Boundary 等模块全部作为 Agent 可调用的冻结 RF 工具。

## 运行

先启动 OpenAI-compatible 大模型服务，然后：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage15_llm_multiagent.py --config configs/experiments/stage15_llm_multiagent_oracle.json
```

测试：

```powershell
conda run --no-capture-output -n pytorch python -m pytest -q tests/stage15
```

## 研究定位

新主线目标不是“把 GRU Actor 换成 Qwen”，而是建立：

**RF 感知/基础模型 → 专业 RF 工具 → LLM 多智能体推理协作 → 开放集拒识 → 未知类发现**

Stage 14 的 PPO/MAPPO、CTDE、GRU Actor 与 centralized critic 只保留用于传统 MARL 对照。
