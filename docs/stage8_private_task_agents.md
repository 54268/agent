# Stage-8：Heterogeneous Private-Task Agents

## 当前范围

Stage-8 是新包 `src/maros_stage8/`，没有覆盖 Stage-7。当前只实现并评价两个硬隔离 Agent：

- `TemporalFingerprintAgent`：v1 只接收 `TemporalObservation(iq)`；v2 只接收不含 raw-I/Q 字段的 `TemporalPatchObservation(features, quality)`；
- `SpectralHardwareFingerprintAgent`：v1/v2 分别只接收 `SpectralObservation` / `CoarseSpectralObservation`，均没有 raw-I/Q 参数或字段；FFT 只存在于 Agent 外部的 observation builder。

Memory Agent、Consistency Auditor 和 Value-of-Information Coordinator 均为 fail-closed 门控接口。G1/G1.5 未共同通过前不能实例化，也没有训练 Router。

## 能力边界

| 模块 | 可访问 | 禁止访问 | 私有任务 |
|---|---|---|---|
| Temporal v1/v2 | raw complex I/Q / 有损局部时域 token | FFT/STFT、频谱 observation、enrollment memory | 候选时域支持验证 |
| Spectral v1/v2 | 六通道频域特征 / 八通道粗粒度频域统计 | raw I/Q、Temporal observation、enrollment memory | 候选频域硬件证据验证 |
| Candidate Evidence Table | public logits、Evidence Certificate | embedding、token、prototype、private state | 对具体 candidate 加减证据 |
| Coordinator | 当前未实现 | 所有 Agent private tensor | G2 通过后才定义 VOI |

`LocalEvidenceContext` 对外是两个 `LocalDecision` 的只读映射。私有 state 只能由 `Stage8System` 持有的对象身份 capability 读取；把 `public_dict()` 交给咨询接口会直接失败。

## Candidate-conditioned verifier

每个 Agent 除本地 K-way 预测外，还训练：

```text
Verify(private_observation, candidate_c) -> support logit
```

`candidate_c` 不再经过 class-id embedding。每个 inner fold 只用 support Known 的 Agent-private state 构造 normalized class prototype；共享 verifier 比较 query state 与 candidate prototype。每个 Known 样本使用真实类作为 positive，并从当前本地 logits 中选最高分错误类作为 hard negative。目标由独立分类损失、positive BCE、hard-negative BCE 和 pair-margin loss 构成。两个 Agent 的损失按单位尺度相加，不交换 observation 或 private state。同步重排 label、classifier row 与 prototype 的 reindex invariance 测试在 v1/v2 均为零误差。

## 通信语义

`QueryPacket` 明确携带 sender/receiver、candidate A/B、query type、reason、local pair support、open risk、active mask 和 bit cost。响应为 `EvidenceCertificate`，包含 stance、signed candidate evidence、alternative class、domain quality、open risk、reliability、abstention 和成本。

通信的核心效果不是 `w_A logits_A + w_B logits_B`。`CandidateEvidenceTable.apply()` 直接执行候选更新：

```text
SUPPORT_A: candidate_a += evidence; candidate_b -= evidence
SUPPORT_B: candidate_a -= evidence; alternative_class += evidence
```

只改变 `alternative_class` 会改变被加分的列和最终预测；单元测试已固定这一因果约束。STOP/no-communication 对锚 Agent 的 candidate scores、open risk 与 prediction 是 bit-exact identity。

## 数据与协议安全

Stage-8 复用 Stage-7 已审计的 `ProvenanceSubset`、nested LCO、sample-disjoint 检查和 formal-Unknown lock。复用通过 `maros_stage8.splits` 的显式 re-export 完成，Stage-7 源码保持不变。

当前重构探针只使用一个 outer fold 内的四个 inner 类别折：

- train Known：每类最多 512 个样本；
- calibration/test：每原始类最多 128 个样本；
- 30 epochs，状态维 64，hidden 32；
- Known calibration 只用于 95% 接受率阈值；
- formal Unknown 只记录样本数，从未进入训练、校准、选择或评价。

运行入口：

```powershell
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_probe_wisig_k40u20.json
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_probe_oracle_k10u6.json
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_isolated_v2_wisig_k40u20.json
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage8_probe.py --config configs/experiments/stage8_g1_isolated_v2_oracle_k10u6.json
```

两份配置除数据路径、名称、输出目录和 paired metadata 外的方法字段完全相同。
