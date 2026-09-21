# Stage-8 Agent-B v3 Quick Validation Delivery

日期：2026-09-20

## 最终结论

`SPECTRAL-PRIVATE-TASK-REDESIGN-REQUIRED`

- ORACLE fold-0 B3 train/eval accuracy：`0.9196 / 0.1484`；
- pair AUC：`0.4870`；
- hard-pair accuracy：`0.1836`；
- train/eval PGR：`1.6277 / 0.1126`；
- 同时触发 accuracy 与 pair-AUC 早停条件；
- 未运行 ORACLE 四折、WiSig、formal Unknown 或任何下游模块。

## 验证

- Stage-8 tests：36 passed；
- project tests：131 passed；
- robust-v3 phase-free/no-raw-IQ、alignment、time-shift 和 class-reindex 测试通过；
- WiSig/ORACLE v3 配置的方法字段完全相同；
- Agent A 固定为 legacy-v1 与原 v1 训练预算。

压缩包只是额外副本；源码、文档和结果仍保留在项目原位置。
