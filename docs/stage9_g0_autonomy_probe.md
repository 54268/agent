# Stage-9 G0：自主工具选择快速重构与验证（seed 42，outer fold 0）

## 判定

**G0 未通过。** 当前实现有明确的 Tool/Agent 边界和逐样本离散工具选择，WiSig/ORACLE 的选择分布也不同；但两个 Agent 的策略在两个数据集上均未超过校准集选出的最佳固定动作。因此按《下一阶段真正多智能体_代码重构规划.md》的开发顺序，**停在 Step 6，不实现 Agent C、Blackboard、QueryPolicy 或正式 Unknown 评测**。

本轮是 **1 seed × 1 outer LCO fold** 的小诊断，不是 Stage-9 完整算法或多 seed 结论。

## 实现边界

- 新建 `src/maros_stage9/`；Stage-8 保持历史基线，未修改。
- 复用冻结的 nested LCO、sample provenance、formal Unknown lock 和 CandidateEvidenceTable。所有训练、校准、选择、测试只使用 support-Known / LCO proxy-Unknown；正式 Unknown 只记录数量，不进入模型路径。
- Signal Identity Investigator 持有 `raw`、`complex`、`envelope`、`difference` 四个私有工具；Hardware Impairment Investigator 持有 `robust_spectrum`、`frequency_difference`、`iq_imbalance`、`phase_noise` 四个私有工具。B 没有完整 raw temporal encoder，也没有 Known enrollment/open-set memory。
- `CapabilityManifest` + `ToolRegistry` 对工具调用做 fail-closed 检查。Tool 是被动网络；Agent 才有目标、策略、信念包与弃权决定。公开 `BeliefPacket` 只含 top-2 候选、标量证据、不确定性、工具名、原因码和建议，不含 embedding、完整 logits 或 prototype。
- 先独立监督训练 8 个工具；冻结工具后，在 **calibration Known** 上计算每个样本 Top-1/Top-2/STOP 动作的反事实效用（负 NLL 减额外工具成本），监督本地 ToolPolicy。推理时 scout 先选动作，只执行选中的工具编码器。策略代码不读 dataset name。
- 预注册 G0 条件：两数据集上均非恒定选择、均优于均匀随机动作和校准集最佳固定动作，且动作分布跨数据集有差异。G0 无通信。

配置完全配对，仅数据路径/元数据不同。训练：seed 42、outer fold 0、每个 support 类最多 128 个训练样本、8 expert epochs、25 policy epochs；Known test 每类最多 64 个样本。ORACLE support 8 类，WiSig support 32 类。WiSig/ORACLE 的条件/SNR 元数据未由当前数据集契约统一提供，故未生成该分组表；没有凭空推断条件。

## 第一张诊断表：工具使用与能力

Known/Proxy-U 使用率表示该工具被选入 Top-1 或 Top-2 的样本比例，**同一行 Agent 的各工具比例可合计超过 100%**。工具单独准确率是不经 Agent 策略、直接用该工具的 Known top-1。条件增益是“选到该工具的样本上，策略动作的负 NLL − 校准集最佳固定动作的负 NLL”；极低使用率时不可解释。

| Dataset | Agent | Tool | Usage Known | Usage Proxy-U | 单工具 Known top-1 | 条件增益 vs 固定 |
|---|---|---|---:|---:|---:|---:|
| ORACLE | Identity | raw | 0.0% | 0.0% | 20.7% | — |
| ORACLE | Identity | complex | 0.0% | 0.0% | 13.1% | — |
| ORACLE | Identity | envelope | 100.0% | 100.0% | 43.4% | 0.0000 |
| ORACLE | Identity | difference | 0.0% | 0.0% | 18.4% | — |
| ORACLE | Impairment | robust_spectrum | 94.3% | 93.0% | 10.7% | −0.0457 |
| ORACLE | Impairment | frequency_difference | 0.0% | 0.0% | 15.6% | — |
| ORACLE | Impairment | iq_imbalance | 5.7% | 7.0% | 15.0% | −0.0477 |
| ORACLE | Impairment | phase_noise | 0.0% | 0.0% | 13.1% | — |
| WiSig | Identity | raw | 82.1% | 91.6% | 90.4% | −0.0015 |
| WiSig | Identity | complex | 0.0% | 0.0% | 89.2% | — |
| WiSig | Identity | envelope | 52.5% | 36.7% | 85.3% | −0.0042 |
| WiSig | Identity | difference | 0.05% | 0.20% | 89.4% | +0.1740（仅 1 个 Known 样本） |
| WiSig | Impairment | robust_spectrum | 3.4% | 3.1% | 12.1% | −0.1200 |
| WiSig | Impairment | frequency_difference | 86.6% | 85.9% | 75.2% | −0.0082 |
| WiSig | Impairment | iq_imbalance | 70.4% | 85.7% | 77.9% | −0.0078 |
| WiSig | Impairment | phase_noise | 14.1% | 0.4% | 48.3% | +0.1032 |

**不要误读 ORACLE 的 94.3% robust_spectrum：** Stage-5 Unified V4 的 `spectral` 是完整 Geometry 工具箱里一个全局 FFT 幅度/相位增量视图，当时在 ORACLE 权重约 0%，因为 raw/complex 可用。Stage-9 的 `robust_spectrum` 是 B 的四段平均、去绝对相位统计，**不是同一特征，也不能与 Stage-5 权重直接比较**。这里 B 不能选 raw/complex；它在受限选项里频繁选 robust，并不意味着它有效。实际 ORACLE B 准确率接近随机，选 robust 的反事实效用还低于固定 `frequency_difference`。

## 第二张诊断表：本地能力与失败重叠

`proposal top-1` 不把弃权算错，`accepted correct` 则把弃权算错；二者不能混用。unique-only 是本轮单 fold、把弃权视作错误的 A/B 正确集合差，不是稳定的 G1 unique rescue 结论。

| Dataset | Agent | Proposal top-1 | Accepted correct | Coverage | Unique-only correct | 主要失败类型 |
|---|---|---:|---:|---:|---:|---|
| ORACLE | Identity | 43.4% | 27.7% | 46.9% | 138 / 512 | 100% 固定选 envelope；高弃权；raw/complex 未学好 |
| ORACLE | Impairment | 11.1% | 1.6% | 18.9% | 4 / 512 | 近随机、81.1% 弃权；误选 robust spectrum |
| WiSig | Identity | 90.4% | 89.7% | 93.9% | 240 / 2048 | raw 主导；相对固定 raw+envelope 无增益 |
| WiSig | Impairment | 80.9% | 78.8% | 89.4% | 16 / 2048 | 与 A 高度重叠，独有正确样本少 |

ORACLE 两者都失败 366/512；WiSig 两者都失败 194/2048。这里不能把 B 的低独有贡献硬说成互补。

| Dataset | Agent | 策略效用 − 随机动作 | 策略效用 − 最佳固定动作 | 是否非恒定 |
|---|---|---:|---:|---|
| ORACLE | Identity | +0.5236 | **0.0000** | 否 |
| ORACLE | Impairment | +0.0011 | **−0.0458** | 是，但集中在 robust |
| WiSig | Identity | +0.5539 | **−0.0033** | 是 |
| WiSig | Impairment | +0.7958 | **−0.0186** | 是 |

随机基线均匀抽取所有 Top-1、Top-2、STOP 动作，因此是较弱对照；真正关键的是最佳固定动作。两个 Agent 在两个数据集上都未取得严格正的固定基线增益，G0 失败。

## 限制、判断与下一步

本轮规模显著小于 Stage-5 Unified V4（本轮 8 expert epochs、每类最多 128 样本，仅 outer fold 0；Stage-5 正式报告为 30 epochs 及完整开集管线）。因此不能根据当前 ORACLE raw/complex 的低准确率推翻 Stage-5 的 65.57% raw、29.90% complex、0% spectral 发现。当前结果仅说明 **这份 Stage-9 v1 小配置还没学出有价值的自主工具选择**。

尤其需要修正 B 的能力边界：这版四个工具仍主要围绕频域/频域派生统计，缺少规划中明确提到的、又不等于完整 raw temporal encoder 的 `complex-frequency descriptor`。若设计 v2，应先冻结新的物理工具定义与训练预算，并使用未参与本轮诊断的 outer fold 检验，不能对 fold 0 的 test 结果反复调工具。还需区分“工具本身学不会”和“策略选错工具”；当前 ORACLE B 两种问题同时存在。

G0 未通过前，不新增 Agent C、Blackboard、主动查询、Router、PPO、多轮对话或 formal Unknown 评测。Stage-9 代码和失败结果保留为可复现负基线。

## 复现与文件

```powershell
D:\Anaconda3\envs\pytorch\python.exe -m pytest tests\stage9 tests\stage8 -q
D:\Anaconda3\envs\pytorch\python.exe scripts\experiments\run_stage9_g0.py
```

测试结果：Stage-9 + Stage-8 合计 47 passed。数值原始文件：`results/stage9_g0/g0_pair_summary.json`、两个数据集的 `g0_report.json`；同目录保留模型 checkpoint。当前 G0 实现不包含真实 query/response，关于 blackboard、wrong-responder、消息干预等测试明确延后到 G1/G2/G3，不能把已有 CandidateEvidenceTable 单元测试冒充消息因果验证。

