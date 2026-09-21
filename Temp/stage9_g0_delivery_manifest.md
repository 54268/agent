# Stage-9 G0 交付清单

压缩包：`stage9_g0_autonomy_probe_20260920.zip`。压缩仅是额外副本；原始代码、文档、JSON 与模型均保留在项目目录。

内容：

- 规划来源：`Temp/下一阶段真正多智能体_代码重构规划.md`
- 新代码：`src/maros_stage9/`（不含 `__pycache__`）
- 配置：`configs/experiments/stage9_g0_*.json`
- 入口：`scripts/experiments/run_stage9_g0.py`
- 测试：`tests/stage9/`（不含 `__pycache__`）
- 分析：`docs/stage9_g0_autonomy_probe.md`
- 结果：`results/stage9_g0/` 下的 JSON 和模型 checkpoint

本轮 47 项 Stage-8/Stage-9 测试通过，但 G0 实验门槛未通过；按规划未实现 Agent C、Blackboard、主动通信或 formal Unknown 评估。请以分析文档和 `g0_pair_summary.json` 为准，不要把“跨数据集工具使用率不同”解释为“工具选择有效”。

