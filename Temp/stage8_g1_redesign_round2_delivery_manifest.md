# Stage-8 G1 Redesign Round-2 Delivery Manifest

日期：2026-09-20

## 结论

- v1 / WiSig：G1 FAIL，G1.5 PASS；
- v1 / ORACLE：G1 FAIL，G1.5 FAIL；
- information-isolated v2 / WiSig：G1 FAIL，G1.5 PASS；
- information-isolated v2 / ORACLE：G1 FAIL，G1.5 FAIL；
- 总决策：`STOP-AND-REDESIGN`；
- Memory、Auditor、Router、VOI、forced communication、formal Unknown 与多种子正式实验均未解锁。

## 验证

- `pytest tests/stage8 -q`：31 passed；
- `pytest tests -q`：126 passed；
- WiSig/ORACLE v1 与 v2 配置均通过 paired-method guard；
- 16 个 fold-level class reindex audits 全部通过，最大绝对误差 0；
- 所有结果均记录 `formal_unknown_used=false`。

## 压缩包范围

压缩包包含本轮新增或变动的：

- Stage-8 prototype-conditioned verifier、v2 observations、训练/评价/gate/probe 源码；
- 新增和修改的 Stage-8 测试；
- v1/v2 成对实验配置；
- ORACLE learnability、v1/v2 WiSig/ORACLE 四折 JSON 结果；
- 自动生成的完整逐折 JSON/Markdown 汇总；
- README、总结果汇总与 Stage-8 设计/结果文档；
- 本轮建议文档与本 manifest。

压缩只是额外副本；项目中的源码、文档和结果文件均保留原位。
