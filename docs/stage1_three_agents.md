# Stage-1：三证据 Agent（Identity + Prototype + Reconstruction）互补性验证

本目录对应方法设计文档 **第 46 节（分阶段路线）的 V0→V1** 与 **第 59 节"第一阶段编码目标"**。
旧的 `src/maros_sei/` 是"六个 Agent + 稀疏通信图 + 域对抗"一次性上全量跨域 WiSig 的尝试，
指标差时无法定位是哪个 Agent 的问题（见根目录 `总结果汇总.md`，AUROC≈0.51）。
本次按文档建议**退回纯开放集设定，只先实现三个证据专职 Agent，验证证据确实互补后，再写
Communication 与 Pseudo-Unknown Agent。**

## 1. 范围与红线

- 只做 **Shared Stem + Identity + Prototype + Reconstruction**，外加一个**无通信、无路由**的
  固定融合/分歧诊断。
- 训练只用 Known；**真实 Unknown 仅在离线诊断时启用，绝不参与训练、阈值选择或超参选择**。
- 证据校准只用 Known 验证集（类条件经验 CDF + 全局收缩）。
- 本阶段**没有**：Communication、Pseudo-Unknown、Boundary、Router、跨域判别器。

## 2. 数据（纯开放集，去掉跨域干扰）

| 数据集 | 划分 | 归一化 | 规模 |
|---|---|---|---|
| WiSig `WiSig_OpenSet_1Day_1Rx` | 固定日期 2021_03_23、固定 Rx 18-2、non-EQ；40 Known / 20 Unknown | 逐样本平均功率归一化 | 训练 12720 / 验证 1800 / Known 测试 3680 / Unknown 测试 4800 |
| ORACLE KRI-16 解调后 | Known=1,2,3,4,5,7,9,13,14,15；Unknown=17,18,19,25,26,32；窗长/步长 256；每发射机≤4000 | 逐窗逐通道均值/方差（与上一方案一致） | 训练 28000 / 验证 4000 / Known 测试 8000 / Unknown 测试 24000 |

- WiSig 子集已在 `data/WiSig/WiSig_OpenSet_1Day_1Rx/wisig_openset_subset.pkl`。
- ORACLE 由原始 SigMF 复现上一方案协议：`python scripts/data/build_oracle_stage1.py`
  （原始数据实际为 cf64/complex128，脚本按字节数自动推断，输出到 `data/oracle/processed_k10u6`）。

## 3. 模型（`src/maros_staged/`）

- `stem.SharedStem`：2×256 IQ → 共享特征图 `h_s[B,128,64]`（两层步长卷积下采样 + 残差块）。
- `agents.IdentityAgent`：私有编码器 + 线性分类头；证据 `c_I`、熵、top1-2 margin；损失 CE。
- `agents.PrototypeAgent`：私有编码器（嵌入 L2 归一化）+ 类原型（归一化嵌入的类均值，**不二次归一化**，
  原型范数编码类紧致度）；证据 `d1/d2`、距离 margin、`p_P`；损失 proto-CE + compact + margin；
  epoch 初/末全量刷新原型，batch 内 EMA 更新。
- `agents.ReconstructionAgent`：**类条件原始 IQ 重构**（文档 §9.2 首选"重构 x，证据更直接"）。
  窄瓶颈 + FiLM 类别调制 + 上采样解码器；证据为融合预测类下的重构误差 `e_R` 与 top-k 最优误差。
  损失 = 正确类重构 MSE + **最难负类（最近竞争原型）重构裕度**，迫使类别条件承重。
  > 注：最初实现重构学习特征 `h_s`，在 Oracle 上出现"未知重构得比已知还好"（AUROC≈0.23，文档 §48.4
  > 预判的"特征过于容易重构/方向不稳定"），故改为重构原始 IQ。

## 4. 证据校准与分歧（`evidence.py`）

- 每个原始异常分（`1-c_I`、`d1`、`e_R`）经 **Known 验证集类条件经验 CDF（带全局收缩）**映射到 `u∈[0,1]`，
  `u=0.95` 即"比 95% 的 Known 验证样本更异常"。
- 分歧（文档 §10）：`D_IP=JS(p_I,p_P)`、标签冲突、`D_confgeo=c_I·u_P`、`D_confrec=c_I·u_R`。
- 两种**无需未知标签**的固定融合：均值（共识）与 max/OR（取最大异常分，覆盖互补未知）。

## 5. 互补性诊断闸门（`diagnose.py`，决定是否进入 Stage 2）

1. 各通道单独 AUROC/AUPR-Out/FPR95/OSCR 与两种固定融合；
2. **条件救援**：Identity 高置信但错误/漏检的 Unknown，被几何/重构抓住的比例（项目立项的核心动机）；
3. τ=0.95 下三通道 Unknown 检测集合的并集召回 vs 最强单通道、独有贡献、两两 Jaccard；
4. 原始分数 Spearman 相关、三嵌入线性 CKA、Known 上预测分歧率（查 Agent collapse）；
5. 图：分数分布直方图、相关热图、重叠/救援柱状、t-SNE。

## 6. 运行

```bash
# 单次（训练 + 诊断，输出 results/stage1/<dataset>/）
python scripts/experiments/run_stage1.py --config configs/experiments/stage1_wisig.json
python scripts/experiments/run_stage1.py --config configs/experiments/stage1_oracle.json

# 多种子（推荐，输出 results/stage1/multiseed/<dataset>/aggregate_stage1.md）
python scripts/experiments/run_stage1_multiseed.py --config configs/experiments/stage1_wisig.json
python scripts/experiments/run_stage1_multiseed.py --config configs/experiments/stage1_oracle.json
```

环境：`D:\Anaconda3\envs\pytorch\python.exe`（Python 3.9 / torch 2.8+cu128 / RTX 5070）。
单次 WiSig≈65s、Oracle≈115s。

## 7. 进入 Stage 2 的判据

- 三通道单独有效（AUROC 明显 >0.5）且跨种子稳定；
- 最大 |Spearman| 明显 <1、嵌入 CKA 不接近 1（没有 collapse 成一个 Agent）；
- Identity 漏检 Unknown 被几何/重构救援的比例、并集召回相对最强单通道有稳定增量；
- 固定融合（均值或 max/OR）至少其一在 AUROC/OSCR 上不劣于、最好超过最强单证据。

满足后再实现 Communication（消息门控 m_i=σ(MLP([z_i,e_i,D_i]))）与 Pseudo-Unknown Agent
（文档 §11/§14/§15）；Oracle 上"硬未知"（18/19/32 极贴近已知）预计主要由 Pseudo-Unknown 解决，
这与上一方案必须依赖伪未知+监督校准才能拿到高未知召回的经验一致。
