# Stage-10：以 Stage-5 Unified V4 为母体的多智能体可行性验证

## 结论

**组件可行，但尚未证明多智能体不可替代性。** 冻结的 Stage-5 专家提供了强身份/几何母体；A 发起候选对询问、B 返回候选级语义证据、A 的候选分数随回复改变、C 用私有 Known 尾部分布审查开放空间，这条一轮链路已在 ORACLE 与 WiSig 上实际运行。消息打乱和错误候选对会削弱结果，说明不是单纯给 B 整体加权。

然而，ORACLE 上 Agent 发起询问与简单低置信度询问打平；WiSig 只领先 1/2048 个 Known 样本。静态 A/B 均值融合的 Known 准确率在两个数据集都高于主动咨询，WiSig 的 proxy-Unknown H-score 也更高。故本轮不能宣称“真正多智能体比强多视图融合更好”。

这是一轮 **seed 42 × outer fold 0** 的开发性可行性探针，未使用正式 Unknown，也未做多 seed。Stage-5 原代码和 checkpoint 没有修改。

## 方法与边界

- A：冻结的 Stage-5 Identity 专家，先提出 top-2 候选；A 自己的 `IdentityQueryPolicy` 只看 A 的公开置信度、margin、entropy、原型距离标量，预测询问 B 的反事实收益。该策略在校准 Known 上学习，没有 dataset-name 分支。
- B：冻结的 Stage-5 Universal Geometry 专家，保留 raw、spectral、envelope/phase、difference-I/Q、complex-I/Q 五视图样本级门控。新加入的 pair verifier 只用 **B 私有 Geometry 视图证据 + A 发来的两个候选 ID** 训练和回答，回复只有候选 ID、支持方向、签名标量证据、可靠度与通信成本；B 的 API 不接收 A logits、标签或隐藏状态。
- A 收到回复后，经 `CandidateEvidenceTable` 只改被点名的两个候选分数；没有 central winner router。
- C：独立的 Open-Set Examiner，仅接收 A 的 5 个公开标量和 B 发布的 3 个候选级支持标量；持有按 Known 类组织的经验尾部分布，并输出风险与结构化主因。C 没有 raw-I/Q 身份分类头、A/B embedding 或完整 logits。Known 分位阈值由单独的 Service 负责，不叫 Agent。
- `SharedBlackboard` 只存候选、询问掩码、语义证书、公开支持标量和 C 风险。真实单轮运行入口与用于干预比较的缓存推理在每个数据集的首个 128 样本上精确复现一致。
- B 的 Geometry backbone 为给 C 发布支持标量，在本原型中对全部样本运行；**条件询问节省的是候选级消息，而不是 B 编码器计算**。这点不能包装成计算效率收益。

母体 checkpoint 与新的 LCO fold 类集合在加载时强制核对。B pair verifier 在 support-train Known 上拟合；A query policy 在 calibration Known 上拟合；C 的私有 Known 分布用 support-train 拟合、阈值由 calibration Known 拟合；评估只看 capped test-Known 与外层 held-out Known 类构成的 proxy-Unknown。所有跨数据集方法字段一致，数据路径和母体 checkpoint 不同。

## 母体是否保留自适应多视图

下表是 **fold0 checkpoint、test-Known** 平均权重，不能与 Stage-5 最终全 Known 模型的 65.57% raw / 29.90% complex 直接混为一个数字；趋势一致：ORACLE 几乎不用 pure spectral，WiSig 主要用 complex。

| Dataset | raw | spectral | envelope/phase | diff-I/Q | complex-I/Q |
|---|---:|---:|---:|---:|---:|
| ORACLE | 23.68% | 0.007% | 8.02% | 13.03% | 55.26% |
| WiSig | 1.82% | 0.45% | 3.88% | 2.42% | 91.44% |

这避免了上一轮 Stage-9 在 ORACLE 上把 B 困在弱频域工具箱里的错误。这里 B 可以选择 raw/complex，但它没有 enrollment/open-set memory 与最终 Unknown 决策权。

## 候选级协作：有因果效果，但优势很小

Known 测试样本 ORACLE 512、WiSig 2048（每个 support 类最多 64 个）。随机、低置信度和主动询问使用相同数量的询问；NLL 越低越好。

| Dataset | 方法 | Known top-1 | NLL | 询问数 | Rescue / Harm |
|---|---|---:|---:|---:|---:|
| ORACLE | A 不询问 | 99.02% | 0.03410 | 0 | — |
| ORACLE | 随机询问 | 99.02% | 0.03405 | 81 | 0 / 0 |
| ORACLE | 低置信度询问 | **99.41%** | **0.01425** | 81 | 2 / 0 |
| ORACLE | A 主动询问 | **99.41%** | 0.01425 | 81 | 2 / 0 |
| ORACLE | 静态 A/B 均值融合 | **100.00%** | **0.00339** | 非询问基线 | — |
| WiSig | A 不询问 | 90.82% | 0.37332 | 0 | — |
| WiSig | 随机询问 | 91.21% | 0.36836 | 529 | 11 / 3 |
| WiSig | 低置信度询问 | 91.70% | 0.35195 | 529 | 27 / 9 |
| WiSig | A 主动询问 | 91.75% | 0.35178 | 529 | 25 / 6 |
| WiSig | 静态 A/B 均值融合 | **92.53%** | **0.29497** | 非询问基线 | — |

ORACLE 的 A 已达 99.02%（B 99.80%），几乎是天花板：B 独有正确样本只有 5/512，主动询问救回 2 个；不能从 2 个样本推断策略优越。WiSig 的 A 为 90.82%、B 为 92.43%，B 独有正确 48/2048；A 的 top-2 包含真类 92.63%，B 在“真类位于候选对内”时的 pair verifier 准确率 98.84%。主动询问改变了 73 个候选身份，其中 25 次救回、6 次伤害；净增加 19/2048 个正确样本。

消息干预：ORACLE 正确消息 99.41%，shuffle 99.22%，错误候选对 99.02%；WiSig 正确消息 91.75%，shuffle 90.87%，错误候选对 90.67%。这支持“回复内容与被问的候选有关”，但并不消除静态融合更强的反例。

## C 的 proxy-Unknown 验证

阈值只由 calibration Known 的 95% 分位数决定。下表中的 Unknown 是 **外层 LCO 留出的原 Known 类**，不是数据集正式 Unknown；各行的阈值各自在同一 calibration Known 上拟合。H-score 是 Known 正确接纳率与 proxy-Unknown 拒识率的调和平均。

| Dataset | 方法 | AUROC | Known 正确接纳 | Proxy-U 召回 | H-score |
|---|---|---:|---:|---:|---:|
| ORACLE | A MSP，不咨询 | 0.8471 | 92.77% | 58.59% | 0.7182 |
| ORACLE | C，不咨询 | 0.9797 | 92.97% | 92.97% | 0.9297 |
| ORACLE | C + A 主动咨询 | 0.9821 | 93.95% | 93.75% | **0.9385** |
| ORACLE | C + 静态 A/B 融合 | **0.9878** | 92.58% | 93.75% | 0.9316 |
| WiSig | A MSP，不咨询 | 0.8203 | 90.09% | 4.88% | 0.0926 |
| WiSig | C，不咨询 | 0.9065 | 89.79% | 40.43% | 0.5576 |
| WiSig | C + A 主动咨询 | 0.9073 | 90.48% | 44.53% | 0.5969 |
| WiSig | C + 静态 A/B 融合 | **0.9280** | **90.67%** | **56.64%** | **0.6973** |

C 的私有 Known 尾部分布相对单一 MSP 有明显价值，且询问确实改变部分 Open-Set 风险（ORACLE Known 73 / Proxy-U 40；WiSig Known 491 / Proxy-U 354）。不过 WiSig 的静态融合 + C 明显更好；ORACLE 主动咨询 H 略高、AUROC 反而低于静态融合。不能据此宣称多智能体最终优于单体自适应多视图。

## 本轮未完成、不得提前宣称的事项

1. A 目前仍是 Stage-5 raw Identity 专家，尚未实现文档建议的 A 内部 raw/complex/envelope 等自主工具选择。
2. C 只有 prototype-distance / public-evidence tail memory；尚未加入 Reconstruction、consistency probe，也没有证明 C 相对等参数单体的独有贡献。
3. Stage-5 的 PUG、Boundary、Reconstruction 权重及模型参数保留在冻结母体/历史实现里，但本快速探针没有重新启动完整 PUG/Boundary 协调链路；各视图也尚未重写成独立 ToolRegistry 插件。这是从母体取证的可行性重构，不是 Stage-5 完整最终算法的替代实现。
4. B 虽拥有多视图私有证据，但 Geometry backbone 对全部样本运行。本轮没有证明按需计算节省。
5. 只有 1 seed、outer fold0，未做统计置信区间、跨 fold 稳定性、正式 Unknown、Stage-5 正式同协议最终比较或等参数单体基线。
6. 这是研究性 feasibility probe；当前 `paired_gate_passed=false`。下一步不应直接扩大模型，应先在新 outer fold 检验询问策略是否稳定超过简单低置信度规则，并设计能与静态融合公平竞争的独立知识与成本收益指标。

## 复现

```powershell
D:\Anaconda3\envs\pytorch\python.exe -m pytest tests\stage10 tests\stage9 tests\stage8 -q
D:\Anaconda3\envs\pytorch\python.exe scripts\experiments\run_stage10_stage5_mother.py
```

测试：58 passed。原始 JSON：`results/stage10_stage5_mother/paired_consultation_summary.json` 与两个数据集的 `consultation_report.json`。运行依赖项目内原有的 Stage-5 V4 fold0 `experts.pt`，不是重新训练出的新母体。
