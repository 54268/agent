# Stage-8 G0/G1 小规模探针报告

> 状态：2026-09-20。结果是架构/方向筛选，不是正式性能；WiSig 20 个与 ORACLE 6 个 formal Unknown 完全未使用。
>
> 本文是首轮历史结果。prototype-conditioned verifier、v1/v2 observation 与新 G1/G1.5 的后续结果见
> [`stage8_g1_redesign_round2.md`](stage8_g1_redesign_round2.md)。

## 结论

当前 A/B 设计不能进入 Memory/Auditor 或 forced consultation：

- WiSig 的本地 Known 能力与跨类 rescue 可预测性已出现，但 Spectral 的独特救援不足，oracle H headroom 也太小，说明 A/B 仍然过度重复；
- ORACLE 在当前轻量训练预算下两个 Agent 都没有达到最低本地 Known 能力，且跨类 rescue 预测接近随机，现有 rescue 数值不能作为有效能力证据；
- 因此两数据集没有同时通过 G1 与 G1.5。Memory Agent、Consistency Auditor、forced consultation 与 VOI Coordinator 全部保持锁定。

## 四折结果

### WiSig

| inner fold | Temporal Known Acc | Spectral Known Acc | Temporal H | Spectral H | oracle ΔH | T 救 S | S 救 T | rescue AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.82880 | 0.56114 | 0.04751 | 0.04936 | +0.02100 | 0.22418 | 0.02378 | 0.71435 |
| 1 | 0.81703 | 0.64357 | 0.09952 | 0.42683 | +0.05357 | 0.16135 | 0.09783 | 0.75069 |
| 2 | 0.88270 | 0.69203 | 0.06793 | 0.14586 | +0.01955 | 0.16135 | 0.02989 | 0.79459 |
| 3 | 0.91168 | 0.78351 | 0.11221 | 0.17608 | +0.01407 | 0.10700 | 0.02072 | 0.86957 |

汇总：最低 Temporal/Spectral Known Acc=`0.81703/0.56114`；最小双向 unique rescue=`10.70%/2.07%`；平均 oracle `ΔH=+0.02705`；cross-class rescue macro-AUROC=`0.78230`（4/4 folds > 0.5）。

G1 失败项：Spectral unique rescue `<5%`，平均 oracle `ΔH<0.05`。G1.5 独立诊断通过，但不能越过失败的 G1 解锁下游。

### ORACLE

| inner fold | Temporal Known Acc | Spectral Known Acc | Temporal H | Spectral H | oracle ΔH | T 救 S | S 救 T | rescue AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.22135 | 0.16406 | 0.11116 | 0.09844 | +0.08973 | 0.15820 | 0.11426 | 0.52257 |
| 1 | 0.20964 | 0.15755 | 0.04216 | 0.07225 | +0.04144 | 0.11035 | 0.07715 | 0.46244 |
| 2 | 0.18620 | 0.13411 | 0.10208 | 0.10760 | +0.09435 | 0.13965 | 0.10547 | 0.45116 |
| 3 | 0.20312 | 0.16406 | 0.07617 | 0.05790 | +0.05013 | 0.13477 | 0.10254 | 0.55409 |

汇总：最低 Temporal/Spectral Known Acc=`0.18620/0.13411`；最小双向 unique rescue=`11.04%/7.71%`；平均 oracle `ΔH=+0.06891`；cross-class rescue macro-AUROC=`0.49757`（2/4 folds > 0.5）。

G1 因两个本地能力检查失败而失败；G1.5 也失败。ORACLE 的高 rescue 主要来自两个弱模型的不同错误，不能解释为稳定专长。

## 本轮必须回答的 8 个问题

1. **改了哪个能力边界？** 拆除了 Stage-7 Prototype super-agent；Temporal 只有 raw I/Q，Spectral 只有外部构造的频域 observation。
2. **新增了什么不可共享信息？** Temporal 的时序样本与 Spectral 的频域硬件特征在类型/API 上互斥，双方都不能读取对方 observation 或 private state。
3. **主要救了哪种失败？** 当前只证明 A/B 在部分样本上有双向 unique rescue；轻量 probe 尚不足以把 rescue 归因到稳定物理失效类型。
4. **held-out classes 是否成立？** WiSig 的 rescue 归属可预测（AUROC `0.782`），但互补量不足；ORACLE 约随机（`0.498`）且本地能力失败。
5. **forced consultation 是否优于 no-comm？** 未运行；G1/G1.5 没有在两个数据集共同通过，按协议禁止进入。
6. **消息是否改变具体 candidate score？** 是。candidate-specific 与 alternative-class intervention 单测均通过；这是机制测试，不是性能收益声明。
7. **shuffle/wrong responder 是否下降？** shuffle 机制测试确认同成本打乱会改变 candidate evidence；性能下降尚未评价。Wrong-responder 属于 G2，未解锁。
8. **是否足够进入下一 gate？** 否。当前决定是 `stop_and_redesign_ab`。

## 下一步允许做什么

只允许回到 A/B 本地专长设计与轻量训练充分性诊断：

- ORACLE：先让两个硬隔离 Agent 在相同小样本预算下达到有效 Known 能力；不能通过把 raw I/Q 偷回 Spectral 来修复；
- WiSig：增强 Spectral 私有任务的不可替代性，目标是提高 Spectral→Temporal unique rescue 与 oracle headroom，而不是调 Router；
- 两边重新通过 G1/G1.5 后，才实现 Memory/Auditor 并做 forced consultation。

机器可读结果：

- `results/stage8_g1_probe/wisig_k40u20/g1_probe_report.json`
- `results/stage8_g1_probe/oracle_k10u6/g1_probe_report.json`
