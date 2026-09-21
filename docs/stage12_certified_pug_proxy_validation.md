# ABC Certified PUG 修订计划：五折 proxy 验证报告

## 结论

这次闭环**方向有改善，但尚未达到计划书的成功门槛**。冻结 Stage‑5 多视图 B 专家的 OpenMax 组件作拒识母体后，C 的增量审查避免了上一轮“弱 C 覆盖强母体”的严重退化。经认证 PUG 的五折独立 proxy 测试 H-score 在 ORACLE 从 0.701 升到 0.717，在 WiSig 从 0.757 升到 0.760；认证 PUG 对比未认证的同源 I/Q PUG，H-score 也分别从 0.711/0.759 微升到 0.717/0.760。**增益很小，不能称为明显提升，更不能宣称已超过完整 Stage‑5 Unified V4。**

正式 Unknown 本轮没有读取、训练、选策略或评估。此前项目已经评估过同一正式 Unknown 切分，重测不能算新盲测；本轮先停在 proxy gate。

## 本轮实际实现

1. 冻结 Stage‑5 V4 fold A/B：A 为 Identity，B 为独立的动态五视图 Geometry；均保留完整 Known 类判断。A/B 分别重新输出 top‑1/top‑2、confidence、margin、entropy 和候选异常证据。
2. C 从已保存的 Stage‑5 OpenMax/EVT 和 A/B prototype 距离建立 Known-only、候选条件校准。C-lite 先检查支持域，冲突/低置信和 lite 高风险才逻辑上调用 C-full。C 风险只作非负增量；不会以新的弱分数替换 OpenMax 母体。
3. 从训练 Known 边界样本及最近竞争类构造**可重新消费的 I/Q 候选**。对每个候选实际重新运行 A/B 完整前向，绝不复制原 Known 的报告。
4. C 用 OpenMax + prototype 双证据确认离开 Known 支持域，并检查 I/Q 能量、到 Known 的最近邻距离，剔除明显过远候选；分别保留 A/B 冲突边界型和 A/B 高置信一致的 hard 型。通过认证者才进入 Certified PUG 池。
5. 用同一批候选的“全部未经认证”与“仅认证通过”分别训练 C 的二分类边界工具。工具的输出先用 validation Known 做类条件校准，再叠加到 Stage‑5 风险；PUG 启用/停用由该 fold 的 validation proxy-Unknown 的 ΔH、ΔAUROC 决定，测试 proxy-Unknown 只评估。

需要准确界定母体范围：这轮实际继承的是 Stage‑5 冻结专家、动态几何视图、prototype 和 OpenMax/EVT，**没有重新训练/重建完整 Stage‑5 communication coordinator 和其 LCO 选出的复合 evidence rule**。因此表中的 `Stage‑5 OpenMax component` 不是完整 Stage‑5 Unified V4 baseline。历史完整 Stage‑5 正式结果仅能作背景，不能直接与此 proxy 表格对比。I/Q 候选沿竞争类方向作有界插值，尚不是经过验证的“穿越边界外推”生成器。

## 五折独立 test proxy 结果（均值，seed 42）

| 方法 | ORACLE H / AUROC / Unknown Recall | WiSig H / AUROC / Unknown Recall |
|---|---:|---:|
| Stage‑5 OpenMax 组件 | 0.701 / 0.845 / 0.606 | 0.757 / 0.835 / 0.659 |
| + C-lite | 0.704 / 0.852 / 0.609 | 0.755 / 0.845 / 0.657 |
| + C-full 路由，无 PUG | 0.705 / 0.855 / 0.611 | 0.757 / 0.848 / 0.659 |
| + 未认证同源 I/Q PUG | 0.711 / **0.865** / 0.617 | 0.759 / **0.856** / 0.661 |
| + Certified PUG（始终 ON 的反事实） | **0.717** / 0.864 / **0.622** | **0.760** / 0.852 / **0.663** |
| Validation utility 控制 USE/IGNORE | 0.713 / 0.863 / 0.619 | 0.760 / 0.850 / 0.663 |

认证 PUG 的 H-score 高于未认证版本，但 AUROC 在两个数据集都略低；“更稳定”只在部分指标成立。按 validation utility，ORACLE 3/5 折启用 PUG，WiSig 4/5 折启用。这个选择也并非完美：WiSig 至少有一折 validation 判为 USE，但独立 test proxy 上 Certified PUG 低于无 PUG，说明收益估计仍不稳。所有对照使用同一个 Known-only 95% 接受率目标，阈值由 validation Known 计算。

ORACLE 逻辑 C-full 审查率为 test Known 22.7%、proxy Unknown 83.1%；WiSig 为 22.4%、86.0%。这只是离线配对反事实的**逻辑路由率**，当前仍预计算了所有样本的工具输出，不是实测运行时计算节省。

## PUG 认证和高置信 Unknown

| 诊断 | ORACLE | WiSig |
|---|---:|---:|
| I/Q 候选生成总数 | 1,440 | 5,760 |
| 联合认证通过 | 326（22.6%） | 2,450（42.5%） |
| 其中 hard 型 | 147 | 437 |
| 高置信 A/B 一致 test proxy Unknown | 334 | 816 |
| 相比 OpenMax 组件新增拒识 | **6** | **2** |
| 相比 OpenMax 组件丢失的拒识 | 0 | 3 |

证书过滤确实产生了可观的 hard 候选，但对目标难例的**实际增量救回极少**。这比平均 H-score 更能说明当前方案尚未解决“共同自信地认错 Unknown”的核心难题。逐折证书通过率、不同 η、过远候选、掉入其他 Known 核心、A/B 重新前向后的置信变化及逐折 ΔH/ΔAUROC 均保存在原始 JSON。

## 协议与局限

- 数据集各 5 个外层 LCO fold，seed 42，分区 seed 2026；每类训练最多 128、校准和测试最多 64。训练 Known、校准 Known、验证 proxy、测试 Known、测试 proxy 严格分离；正式 Unknown 只读取协议中的数量，不加载样本。
- `Stage‑5 + C-full` 只比 OpenMax 组件小幅改善，不能证明优于完整 Stage‑5。上一轮弱 C 的灾难性退化被修复，主要来自**恢复强母体**，不能将整段提升归功于认证 PUG。
- 当前代理 Unknown 也在反复研究中使用，方法级超参数尚需新的独立 proxy/盲测分割复核。此前正式 Unknown 已暴露，不能把它当全新 holdout。
- A 仍沿用 Stage‑5 Identity 专家而非 A 内部动态多视图；B 承担动态五视图。C 未实现 identity-preserving consistency；C 的 USE/IGNORE 是工具增益动作，推理样本的 KEEP/CHALLENGE/REJECT 由路由和拒识阈值体现，尚无独立训练的动作策略。
- 本轮没有重训完整 Stage‑5 coordinator，所以计划书第 9 条“Stage‑5 baseline + C 不低于原 Stage‑5 核心能力”**尚未验证**；正式 Unknown 测试也尚未执行。

结论：Certified PUG 的闭环认证实现与方向性正贡献可以保留，但**先不推进 Agent D 或正式发布**。下一步应在新 proxy 切分上重点提高高置信共同误判的净救回，并补齐完整 Stage‑5 communication 母体的严格配对增量对照；达标后再申请新的未触碰正式 Unknown 切分。

原始结果：`results/stage12_certified_pug/{oracle,wisig}/proxy_fivefold.json` 和 `paired_proxy_summary.json`。入口：`scripts/experiments/run_stage12_certified_pug.py`。
