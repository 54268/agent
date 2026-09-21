# Stage-13 继承式 ABC + Feature-Space Certified PUG 实验报告

## 三个问题的直接答案

### Q1：完整 Stage-5 V4 + C 是否比 Stage-5 本身更好？

**成立，但只是小幅提升。** 本轮真正恢复了每折的完整 Stage‑5 V4 open-set 路径：冻结 A/B 专家、Universal Geometry 动态五视图、原 prototype/OpenMax、原 feature PUG、communication/audit coordinator、LCO 冻结 evidence rule、calibration 与 class-conditional threshold。每折 coordinator 按冻结配置重新训练，B0 不再是 OpenMax 单组件。

- ORACLE：B0 H=0.9062，B1 Stage‑5+C H=0.9083（+0.0021）。
- WiSig：B0 H=0.8931，B1 H=0.8965（+0.0033）。

B4 utility gate 相对 B0：ORACLE H +0.0040、Unknown Recall +0.0071、Known Accuracy +0.0012；WiSig H +0.0034、Unknown Recall +0.0052、Known Accuracy +0.0019。ORACLE 5/5 折 B4 不低于 B0；WiSig 4/5 折不低于 B0。

### Q2：Feature-Space Certified PUG 是否解决了 Stage-12 I/Q 插值问题？

**不成立。** 严格配对的旧 I/Q PUG 仍略好于 Feature Certified PUG：

- ORACLE：Feature Certified H=0.9102，旧 I/Q H=0.9106。
- WiSig：Feature Certified H=0.8965，旧 I/Q H=0.8971。

Certified Feature PUG 对未经认证 Feature PUG 的平均优势也只有 ORACLE +0.00013、WiSig +0.00004；按逐折 H-score 只在两数据集各 2/5 折更好，不能认为认证显著提高稳定性。

### Q3：C 是否开始救回 A/B 一致且自信地认错的 Unknown？

**ORACLE 有有限新增救回；WiSig 几乎没有。**

- ORACLE：高置信一致 proxy Unknown 共 4,063，Stage‑5 已拒识 3,495；B4 新增拒识 53、丢失原拒识 6，净增 47。C 本身新增 28，Certified PUG 相对 C 又新增 26。
- WiSig：共 1,164，Stage‑5 已拒识 887；B4 新增 9、丢失 6，净增仅 3。C 本身新增 9，Certified PUG 相对 C 仅新增 1。

更关键的是，所有 12 组 `kind × eta`、全部 5 折中，认证出的 **Hard Pseudo-Unknown 数量为 0**。当前 Feature PUG 只产生了 Boundary 型候选，没有形成计划最需要的“高置信一致但超出支持域”的训练信号。

因此本轮不满足进入 Agent D 的条件。根据计划中的停止规则，不应继续围绕 eta/threshold 微调；应重新检查 C 的 feature-space 边界和 Hard PUG 定义。

## 严格配对结果

以下均为 seed 42、5 个外层 LCO fold 的独立 `test_known + test_proxy_unknown` 均值。结构、PUG 类型/eta/权重只用样本不重合的 `calibration_known + validation_proxy_unknown` 选择，test proxy 不参与选择。正式 Unknown 没有加载。

| 方法 | ORACLE Known / UR / H / AUROC | WiSig Known / UR / H / AUROC |
|---|---:|---:|
| B0 完整 Stage‑5 V4 | 0.8957 / 0.9184 / 0.9062 / 0.9579 | 0.8766 / 0.9114 / 0.8931 / 0.9471 |
| B1 Stage‑5 + C，无新 PUG | 0.8965 / 0.9219 / 0.9083 / 0.9599 | 0.8781 / 0.9168 / 0.8965 / 0.9489 |
| B2 + 未认证 Feature PUG | 0.8970 / 0.9251 / 0.9101 / 0.9613 | 0.8781 / 0.9168 / 0.8965 / 0.9486 |
| B3 + Certified Feature PUG | 0.8969 / 0.9255 / 0.9102 / 0.9616 | 0.8785 / 0.9166 / 0.8965 / 0.9486 |
| B4 + Utility Gate | 0.8969 / 0.9255 / 0.9102 / 0.9616 | 0.8785 / 0.9166 / 0.8965 / 0.9486 |
| Old 配对 I/Q PUG | **0.8970 / 0.9263 / 0.9106 / 0.9617** | **0.8790 / 0.9171 / 0.8971 / 0.9502** |

ORACLE B0/B4 FPR95 从 0.1555 降至 0.1383，OSCR 从 0.9571 升至 0.9607；WiSig FPR95 从 0.1463 降至 0.1388，OSCR 从 0.9023 升至 0.9037。提升方向一致，但量级不足以支持“Feature PUG 明显优于旧方案”。

## 完整 Stage-5 母体如何恢复

- 从原 `lco_selection.json` 的冻结 `selected` 区块恢复，而非重新在当前 test proxy 上挑 Stage‑5 规则。
- ORACLE：`competition_eta1.5`、`audit_sparse_cf`、`idproto_boundary`、通信仲裁、Known acceptance 0.90、class threshold prior 25。
- WiSig：`competition_eta2`、`audit_sparse_cf`、`sensor_mean`、通信仲裁、Known acceptance 0.88、global threshold。
- 每个 fold 使用原 Stage‑5 V4 expert checkpoint，重新刷新训练 Known prototype；训练样本最多 4,096，coordinator 使用原 LCO 10 epoch 和原损失/门控配置。
- B0 风险来自 `DecisionArbitratorAgent` 的完整 communication/local 分支和冻结 evidence rule，不是 OpenMax 单项。

B0 proxy 结果与历史正式 Stage‑5 数值接近但不是同一个评估对象：历史正式 ORACLE/WiSig communication H 为 0.8897/0.8974；本轮是跨类 proxy test 的 0.9062/0.8931。不可把两者当重复测量。

## Feature PUG 与认证实现

- 直接复用 Stage‑5 `BoundaryExplorerAgent/PUGConfig`：joint normalized Identity/Geometry state，competition/disagreement/mixed 种子，own-prototype outward + rival repulsion，local scale，正交小噪声。
- 候选统一搜索 kind=`competition/disagreement/mixed`、eta=`1.0/1.5/2.0/2.5`，没有 ORACLE/WiSig 专用分支。
- 新增 `IdentityAgent.forward_from_state` 和 `GeometryAgent.forward_from_state`。每个 pseudo 都重新计算 A/B logits、confidence、margin、prototype distance 和消息；没有复制 source report。Geometry fused state 无法真实恢复原五个 view state/gate，因此 state-level report 明确不伪造 view 权重。
- C 二次证书检查 source-class 95% tail 穿越、向外移动、最近 Known 密度、too-far、是否落入 rival 核心、OpenMax + prototype 双证据，并要求 boundary 或 hard 类型。
- C/PUG 只对完整 Stage‑5 风险作非负增量。C-lite 负责高置信一致筛查；冲突、低置信或 lite 高风险进入逻辑 C-full。
- PUG 权重从 `0/0.05/0.1/0.2` 由 validation proxy utility 选择。两数据集均 4/5 折 USE、1/5 折 IGNORE；没有数据集硬编码。

所有候选合计：ORACLE 生成 197,120、认证 65,294（33.1%）；WiSig 生成 201,792、认证 63,025（31.2%）。最终各折所选候选合计认证通过约为 ORACLE 4,806/17,248（27.9%）、WiSig 4,913/15,132（32.5%），但 hard 数始终为 0。

## 为什么尚未成功

1. Feature PUG 的 state-level 样本几乎都表现为 disagreement/low-margin Boundary 类型；C 的 hard 条件与“超出支持域”的双证据没有交集。
2. Fused Geometry state 无法重新构造真实 dynamic multi-view gate 和各 view 局部证据；虽然 A/B 分类与 prototype 报告是 fresh 的，但 PUG 认证缺少 B 最有价值的 view-level物理证据。这是 Feature-space 路线的结构性限制，不应伪造补齐。
3. Utility Gate 在 validation proxy 上选 USE 后，WiSig 的独立 test proxy 有 3/5 折 B3 低于 B1，说明 utility 估计仍会过拟合；它只保证负贡献可在选择集停用，不能保证未知 test 类泛化。
4. 旧 I/Q PUG 虽是简单插值，却能完整重新经过五个 Geometry views；在本轮严格配对下略胜 Feature PUG。这说明“生成空间更漂亮”不等于开放集证据更有用。
5. B4 新增误拒原本被 B0 接受的 Known：ORACLE 107/32,000，WiSig 36/14,720。平均 Known Accuracy没有下降，但样本级交换仍存在，不能只看均值。

## 协议、测试与结论

- 正式 Unknown 本轮完全未使用；结果文件记录 `formal_unknown_used=false` 和未访问数量。
- 项目 `tests/` 共 167 个测试通过。直接在仓库根运行无约束 `pytest` 会额外收集第三方样例 `JRC-AoI-multi`，因其未安装 `gym` 在收集阶段失败；这与本项目代码无关。
- 本轮结论：**完整母体恢复成功；C 有小幅、跨数据集正增益；Feature Certified PUG 没有胜过 I/Q 对照，认证稳定性不足，Hard PUG 目标失败。** 暂停 Agent D，也停止继续调 eta/threshold。下一次修改应针对可保留真实 multi-view 证据的边界生成空间，或重新定义可验证的 hard boundary，而不是继续堆模块。

原始结果：`results/stage13_feature_pug/{oracle,wisig}/proxy_fivefold.json` 与 `paired_proxy_summary.json`。
