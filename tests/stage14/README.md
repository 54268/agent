# Stage 14 测试与研究归档

最终主流程请先阅读根目录 `多智能体开放集拒识_主流程.md`。本目录只保存测试、消融、诊断和历史计划，不定义最终主流程。

当前正式全量单折结果只保存在 `results/stage14/final_ac_fold0_seed42/`；本目录中的结果均为 smoke、对照或诊断。

## 目录

| 目录/文件 | 用途 |
|---|---|
| `test_env_contracts.py` | 环境、消息延迟、mask、泄漏锁和终止条件测试 |
| `test_mappo.py` | 独立 Actor、循环 MAPPO、两智能体拓扑和推理检查点测试 |
| `configs/` | ABC 对照、Known Accuracy 工作点扫描等测试配置 |
| `diagnostics/` | B 证据拆解等只读诊断脚本 |
| `reports/` | Stage 14 实验报告和诊断结论 |
| `plans/` | 实施前计划，只作历史参考 |
| `artifacts/` | 非最终运行结果；最终结果不放这里 |

## 快速命令

```powershell
conda run --no-capture-output -n pytorch python -m pytest -q tests/stage14
conda run --no-capture-output -n pytorch python tests/stage14/diagnostics/analyze_b_transfer.py
```

ABC 对照和工作点扫描必须显式传入 `tests/stage14/configs/` 中的配置，默认训练入口始终使用根主流程指定的 A+C 配置。
