# MAROS-SEI：多智能体开放集辐射源识别

> **2026-09 起按方法设计文档第 46/59 节改为分阶段实现（当前主线）。**
> 旧的六 Agent + 稀疏通信图 + 域对抗一次性全量跨域模型保留在 `src/maros_sei/`（首轮 AUROC≈0.51，仅作归档，见 `总结果汇总.md`）。
> 新主线在 **`src/maros_staged/`** 与 **`src/maros_stage5/`**，已依次完成三证据 Agent、Pseudo-Unknown + Boundary、
> 显式 Communication，以及 Reliability Router。各阶段都保留独立入口、三随机种子聚合和反事实对照，
> 不把未通过净增益闸门的模块包装成正结果。

**当前主线为 Stage-7 条件咨询式多智能体。** Stage-6 V2–V7 已停止继续调参，其运行仅是未完成的
WiSig 单 fold 开发诊断，完整性登记见 [`results/stage6_diagnostic_index.md`](results/stage6_diagnostic_index.md)。
Stage-7 只保留 Waveform Identity 与 Enrollment/Prototype 两个具有私有观察、独立决策和条件响应能力的
推理 Agent，通过 `STOP / W_FIRST / P_FIRST` 一轮低带宽语义通信验证真实协作价值。裁决、校准、阈值、
OpenMax/EVT、PCBM 与 PUG 均作为服务，不因是模块就命名为 Agent。

**Stage-1 已于 2026-09-17 完成修正版三随机种子复验。** 修正版统一使用同一校准未知分数计算
AUROC/OSCR，并用参与重构训练的 decoder bottleneck 计算 Reconstruction CKA。WiSig 均值融合
AUROC=`0.922±0.004`；ORACLE max/OR 未知召回=`0.793±0.019`；两数据集均为 3/3 seeds
出现三证据并集召回增益，允许进入 Pseudo-Unknown + Communication 阶段。

**Stage-2 已完成 Pseudo-Unknown + Boundary Agent 的 WiSig 三随机种子实验。** 在每种完整规则均由
Known validation CDF 统一到约 95% Known 接受率后，四证据均值 AUROC=`0.929±0.006`、
Unknown Recall=`0.632±0.044`、H-score=`0.737±0.029`，优于 Stage-1 均值的
`0.922 / 0.535 / 0.669`。真实 Unknown 不参与生成、训练、选轮或阈值。

**Stage-3 已完成显式 Communication 三随机种子及消息反事实实验。** 消息确实进入决策：推理关闭消息
平均使 AUROC 下降 0.022；打乱消息使 OSCR 下降 0.230、类别翻转率达到 0.455。但当前直接通信
AUROC=`0.867±0.013`，低于同构无通信模型 `0.877±0.025`；通信+三证据均值也略低于 Stage-2
四证据均值。因此当前通信被判定为“生效但尚未产生稳定净增益”，不直接升级为最终主模型。

**Stage-4 已完成 Reliability Router 三随机种子实验。** Router 严格限制为无通信/通信系统输出的凸组合，
并与全局、单路由、双路由、均匀和打乱权重比较。样本级双路由 AUROC=`0.877±0.025`，没有优于
全局权重 `0.881±0.022` 或均匀权重 `0.881±0.022`；双路由与三基础证据融合达到
AUROC=`0.928±0.008`、OSCR=`0.874±0.006`、Unknown Recall=`0.602±0.055`、
H-score=`0.716±0.039`，仍未超过 Stage-2 四证据均值的召回与 H-score。下一阶段重点转向更贴近真实
开空间边界的 PUG-V2 / 代理任务，而不是继续堆叠 Router 容量。

**Stage-5B～H 已完成角色化重构后的通信、仲裁、风险预算与分类别阈值筛选。** 五折 LCO 冻结
`mixed PUG + 可选反事实审计通信 + 四证据 + max 仲裁 + 0.88 Known 接受率`；三种子达到
AUROC=`0.948±0.005`、OSCR=`0.892±0.007`、Known Acc=`0.862±0.013`、Unknown Recall=`0.892±0.010`、
H-score=`0.877±0.012`。分类别自适应阈值没有胜过全局阈值，作为负消融保留。完整协作相对同模型关闭
消息仅 `ΔH≈+0.0011`、`ΔOSCR≈-0.0020`，因此 H-score 已过闸，但消息净增益仍未通过。

**Stage-5 Unified V4 已完成 WiSig / ORACLE 同算法单 seed 验证。** 两份配置除数据和运行元数据外的
45 个方法字段完全相同；固定五传感动作的 Geometry Agent、PUG、稀疏反事实通信、证据候选和阈值候选
全部共用。Agent 根据数据自主选择不同动作：ORACLE 选择 `competition_eta1.5 + idproto_boundary +
分类别阈值 prior=25`，WiSig 选择 `competition_eta2 + sensor_mean + 全局阈值`，没有数据集专用分支。

WiSig seed42 达到 AUROC=`0.9536`、OSCR=`0.9065`、Known Acc=`0.8739`、Unknown Recall=`0.9221`、
H-score=`0.8974`，五项主性能门全部通过。ORACLE seed42 达到 `0.9483/0.9478/0.8950/0.8845/0.8897`；
相对等容量无通信模型 H-score 提升 `0.1279`，证明通信真实有效，但仍未追平上一方案 PCBM，不能包装成
持平结果。当前两组都是单 seed，下一步必须完成 `42/43/44` 后再形成论文结论。

本项目当前用 **WiSig Full（non-equalized）** 作为主要充分实验数据，同时用 **ORACLE** 检验同一算法的
跨数据集鲁棒性。目标是让 Agent 在固定角色和固定动作空间内自主适应环境，而不是针对数据集定制模块。

当前阶段：**Stage-7 的一个 outer fold 严格 nested G0 已在 WiSig 与 ORACLE 完成。两个本地 Agent
具有显著互补余量，但公共 B2/安全选择没有跨数据集共同胜者：WiSig 最佳候选
`H=0.64632、ΔH=+0.02997、3/4`，同候选在 ORACLE 为 `H=0.57459、ΔH=-0.10311`；ORACLE
唯一通过的锚点候选在 WiSig 只有 `2/4` 正增益。V5–V7 competence 消融仍不能产生可迁移切换。
因此 G0 未通过，G1/G2/G3 与正式 Unknown 全部保持锁定。**

当前正式训练入口：

```powershell
$env:PYTHONUTF8 = "1"
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage7.py --config configs/experiments/stage7_screen_v4_wisig_k40u20.json --phase nested_sanity --fold 0
conda run --no-capture-output -n pytorch python scripts/experiments/run_stage7.py --config configs/experiments/stage7_screen_v4_oracle_k10u6.json --phase nested_sanity --fold 0
```

后续阶段分别使用 `--phase g0`、`g1`、`g2`、`g3`；入口会读取前置 gate report，不提供跳过闸门选项。
Stage-7 协议见 [`docs/stage7_conditional_consultation.md`](docs/stage7_conditional_consultation.md)。Stage-5H、
Unified V4 和旧 F174 结果继续保留为历史基线，不能替代 Stage-7 尚未产生的结果。

## 项目入口

| 内容 | 位置 |
|---|---|
| 所有实验的唯一总表 | [`总结果汇总.md`](总结果汇总.md) |
| WiSig-only 完整实验计划 | [`docs/design/experiment_plan_wisig_only.md`](docs/design/experiment_plan_wisig_only.md) |
| 分阶段多智能体方法总设计 | [`docs/design/Multi-Agent_OS-SEI_Method_Design_and_Research_Status.md`](docs/design/Multi-Agent_OS-SEI_Method_Design_and_Research_Status.md) |
| Stage-2 伪未知与边界 Agent | [`docs/stage2_pseudo_unknown.md`](docs/stage2_pseudo_unknown.md) |
| Stage-3 显式通信与反事实 | [`docs/stage3_communication.md`](docs/stage3_communication.md) |
| Stage-4 可靠性路由与对照 | [`docs/stage4_reliability_router.md`](docs/stage4_reliability_router.md) |
| Stage-5 角色化 Agent、LCO 与候选消融 | [`docs/stage5_role_agents.md`](docs/stage5_role_agents.md) |
| Stage-5B～H 通信、仲裁与阈值筛选 | [`docs/stage5_performance_screening.md`](docs/stage5_performance_screening.md) |
| Unified V4 跨数据集同算法验证 | [`docs/stage5_unified_cross_dataset_v4.md`](docs/stage5_unified_cross_dataset_v4.md) |
| Stage-6 V2–V7 未完成诊断索引 | [`results/stage6_diagnostic_index.md`](results/stage6_diagnostic_index.md) |
| Stage-7 条件咨询、nested LCO 与晋级闸门 | [`docs/stage7_conditional_consultation.md`](docs/stage7_conditional_consultation.md) |
| Stage-7 nested G0 实测失败诊断 | [`docs/stage7_nested_g0_diagnosis.md`](docs/stage7_nested_g0_diagnosis.md) |
| 第一阶段审计与框架 | [`docs/phase1_audit_and_framework.md`](docs/phase1_audit_and_framework.md) |
| 近年论文与代码映射 | [`docs/literature/recent_papers_and_code.md`](docs/literature/recent_papers_and_code.md) |
| 机器可读实验矩阵 | [`configs/protocols/wisig_only_experiment_matrix.json`](configs/protocols/wisig_only_experiment_matrix.json) |
| WiSig 基础审计 | [`docs/wisig_audit_summary.json`](docs/wisig_audit_summary.json) |
| Tx×Rx×日期支持度审计 | [`artifacts/audits/wisig_support_by_cell.json`](artifacts/audits/wisig_support_by_cell.json) |
| 原始任务约束 | [`docs/requirements/codex_multi_agent_os_sei_prompt.md`](docs/requirements/codex_multi_agent_os_sei_prompt.md) |

## 当前冻结的 Unified V4 验证

- 数据协议：WiSig 同日同接收机 `40 Known / 20 Unknown`；ORACLE `10 Known / 6 Unknown`。
- 两个数据集共用 Identity、五传感 Geometry、Boundary Explorer/Auditor、Calibration/Threshold Agent
  与 Communication Coordinator；方法配置的 45 个字段完全相同。
- 五折 LCO 只在训练 Known 中构造留出类代理 Unknown；真实 Unknown 不参与训练、选模、调参或阈值。
- 当前只完成 seed `42`；所有结论按单 seed 标记，不能代替最终三种子均值。

## 后续 WiSig 大类别扩展计划

- 数据：WiSig 的 174 个 Tx、30 个四天均存在的公共 Rx、4 个采集日期。
- 主开放集划分：`120 Known + 54 Unknown`，类别划分种子为 `13/42/87`。
- 主域划分：前两天与 20 Rx 训练，第 3 天与 5 Rx 验证，第 4 天与另外 5 Rx 测试。
- 所有真实 Unknown 只允许出现在测试集；阈值只能用 Known validation、leave-class-out 和训练期伪未知校准。
- 主方法：异构 Agent 私有观察与私有损失、显式可学习消息、样本级稀疏通信图、可靠性路由和开放集协同决策。

## 为什么这不是普通多分支网络

每个 Agent 必须能独立输出预测和不确定性；Router 必须决定谁向谁发送什么消息；接收消息后 Agent 状态必须发生更新；最终预测必须对消息删除、发送者打乱和路由反事实敏感。只有同时满足这些条件，实验中才称为多智能体模型。

## 根目录约定

根目录只保留两个长期入口文件：本文件和 `总结果汇总.md`。数据、配置、文档、脚本、源码、测试、过程产物与详细结果分别进入对应目录。现有 `data/` 和 `多智能体GitHub/` 是目录级资源，不在根目录散放过程文件。

参考目录 `E:/Project` 与 `D:/learn_pytorch/笔记/方案/os_sei_code` 始终只读。
