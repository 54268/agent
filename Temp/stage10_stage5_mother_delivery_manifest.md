# Stage-10 Stage-5 母体可行性探针交付清单

额外压缩包：`stage10_stage5_mother_feasibility_20260920.zip`。项目内原始文件均保留，不移动或删除。

压缩包包括：

- 本轮来源文档：`Temp/Stage5为母体的下一阶段多智能体设计.md`
- 新实现：`src/maros_stage10/`
- 配对配置：`configs/experiments/stage10_stage5_mother_*.json`
- 运行入口：`scripts/experiments/run_stage10_stage5_mother.py`
- 测试：`tests/stage10/`
- 分析：`docs/stage10_stage5_mother_feasibility.md`
- 原始结果：`results/stage10_stage5_mother/` 下的 JSON

注意：ZIP **不包含** 项目已有的大型 Stage-5 V4 fold0 `experts.pt`。复现需要本项目原有的 ORACLE/WiSig 数据及冻结 checkpoint；各 checkpoint 的 SHA-256 已记录在结果 JSON 中。

结论为“候选级问答、消息因果性和 C 的 proxy-Unknown 检查具备组件可行性；多智能体不可替代性尚未证明”。不要把 WiSig/ORACLE 的单 fold、proxy-Unknown 结果当正式 Unknown 或多 seed 最终性能。

