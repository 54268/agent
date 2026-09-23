# MAROS-SEI：大模型多智能体开放集辐射源识别

当前主线：**Stage 15 — Partially Observable Heterogeneous LLM Multi-Agent Open-Set SEI**。

Stage 14 Comm-MAPPO/CTDE 保留为传统 MARL baseline。

## 唯一运行入口

本地下载更新后，只运行仓库根目录：

```powershell
python run_stage15.py
```

快速烟雾测试：

```powershell
python run_stage15.py --samples-per-class 4
```

默认配置：

`configs/experiments/stage15_llm_multiagent_oracle.json`

默认结果目录：

`results/stage15/llm_multiagent_oracle_fold0/`

其中：

- `summary.md`：最方便人工查看的总结果；
- `summary.json`：完整结构化汇总；
- `agent_stats.json`：Agent动作、RF工具、通信与轨迹统计；
- `trajectories.jsonl`：逐样本完整多智能体推理轨迹。

## Stage 15 多智能体定义

- **Proposer Agent**：只看 identity-side 局部观测，拥有 identity 私有工具；
- **Critic Agent**：只看 geometry/open-set-side 局部观测，拥有 open-set 私有工具；
- **Arbiter Agent**：不直接读取底层 RF 数值，只根据显式 Agent 消息裁决；
- 每个 Agent 拥有独立上下文、独立私有工具结果、独立 Case Memory；
- 跨 Agent 私有信息只能通过显式消息传递；
- Ground Truth、数据来源和 formal unknown 标记禁止进入 Agent 上下文。

RF classifier、Prototype、OpenMax、Boundary 等均属于工具，不把专家网络伪装成 Agent。

## 大模型服务

当前入口连接 OpenAI-compatible Chat Completions 服务。默认配置使用：

`Qwen/Qwen3-8B @ http://127.0.0.1:8000/v1`

因此运行 `run_stage15.py` 前，本地需已有兼容服务，或将配置中的 `base_url/model` 修改为你的实际模型服务。

## 研究结构

**RF 感知/基础模型 → 私有 RF 观测与工具 → Proposer ↔ Critic ↔ Arbiter → Known/Unknown → Unknown Discovery**

详细设计见 `Stage15_大模型多智能体主流程.md`。
