"""Stage-1 evidence-complementarity diagnostic (the gate before Stage 2).

Real unknowns are used here *offline only* - nothing feeds back into training or
threshold selection.  The diagnostic answers five questions:

1. How strong is each evidence channel alone (AUROC / AUPR / FPR95 / OSCR)?
2. Does a fixed equal-weight fusion beat the strongest single channel?
3. Do geometry / reconstruction catch unknowns the confident classifier misses
   (conditional catch) - the motivating scenario of the whole project?
4. How much do the three unknown-detection sets overlap (unique contribution)?
5. Are the three agents collapsing to one representation (score correlation and
   embedding CKA)?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from .metrics_osr import detection_metrics, oscr, threshold_decision

CHANNELS = ["identity", "prototype", "reconstruction"]
FUSIONS = ["fused_mean", "fused_max"]
FUSION_SCORE_KEY = {"fused_mean": "u_fused_mean", "fused_max": "u_fused_max"}
TAU = 0.95


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(x.T @ y, ord="fro") ** 2
    den = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(num / max(den, 1e-12))


def _channel_view(data, ch: str):
    # OSCR must sweep the same unknown score used by AUROC/AUPR.  Using raw
    # softmax confidence for Identity or probability margin for Prototype
    # silently changed the sample ranking after class-conditional CDF
    # calibration and made cross-channel OSCR comparisons inconsistent.
    if ch == "identity":
        return data["pred_id"], 1.0 - data["u_identity"]
    if ch == "prototype":
        return data["pred_proto"], 1.0 - data["u_prototype"]
    if ch in FUSION_SCORE_KEY:
        return data["closed_pred"], 1.0 - data[FUSION_SCORE_KEY[ch]]
    return data["closed_pred"], 1.0 - data["u_reconstruction"]


def evaluate_channels(data) -> Dict[str, Dict[str, float]]:
    is_unknown = (data["y"] == -1).astype(np.int32)
    results = {}
    for ch in CHANNELS + FUSIONS:
        if ch in FUSION_SCORE_KEY:
            score = data[FUSION_SCORE_KEY[ch]]
        else:
            score = data[f"u_{ch}"]
        closed_pred, known_conf = _channel_view(data, ch)
        det = detection_metrics(is_unknown, score)
        det["oscr"] = oscr(is_unknown, closed_pred, data["y"], known_conf)
        det.update(threshold_decision(is_unknown, data["y"], closed_pred, score, TAU))
        results[ch] = det
    return results


def conditional_catch(data, tau: float = TAU) -> Dict[str, Dict[str, float]]:
    is_unknown = data["y"] == -1
    u = {ch: data[f"u_{ch}"] for ch in CHANNELS}
    out = {}
    for missed in CHANNELS:
        idx = np.where(is_unknown & (u[missed] < tau))[0]
        if len(idx) == 0:
            out[missed] = {"n_missed": 0}
            continue
        row = {"n_missed": int(len(idx))}
        for helper in CHANNELS:
            if helper == missed:
                continue
            row[f"caught_by_{helper}"] = float((u[helper][idx] >= tau).mean())
        row["caught_by_either_helper"] = float(
            ((u["prototype"][idx] >= tau) | (u["reconstruction"][idx] >= tau)
             | (u["identity"][idx] >= tau)).mean()
        ) if False else float(
            np.any(np.stack([u[h][idx] >= tau for h in CHANNELS if h != missed]), axis=0).mean()
        )
        out[missed] = row
    return out


def detection_overlap(data, tau: float = TAU) -> Dict:
    is_unknown = data["y"] == -1
    flags = {ch: set(np.where(is_unknown & (data[f"u_{ch}"] >= tau))[0].tolist())
             for ch in CHANNELS}
    n_unknown = int(is_unknown.sum())
    union = set().union(*flags.values())
    pairwise = {}
    for i, a in enumerate(CHANNELS):
        for b in CHANNELS[i + 1:]:
            inter = len(flags[a] & flags[b])
            union_ab = len(flags[a] | flags[b])
            pairwise[f"{a}__{b}"] = {
                "jaccard": inter / max(union_ab, 1),
                "intersection": inter,
            }
    unique = {ch: len(flags[ch] - set().union(*[flags[o] for o in CHANNELS if o != ch]))
              for ch in CHANNELS}
    return {
        "tau": tau,
        "n_unknown": n_unknown,
        "recall_per_channel": {ch: len(flags[ch]) / n_unknown for ch in CHANNELS},
        "union_recall": len(union) / n_unknown,
        "unique_contribution": {ch: unique[ch] / n_unknown for ch in CHANNELS},
        "pairwise_jaccard": pairwise,
    }


def correlations(data) -> Dict:
    raw = {
        "1-conf": 1.0 - data["conf"],
        "d1": data["d1"],
        "rec_err": data["rec_err"],
    }
    names = list(raw)
    score_corr = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = spearmanr(raw[names[i]], raw[names[j]]).statistic
            score_corr[i, j] = score_corr[j, i] = r
    cka = {
        "identity__prototype": linear_cka(data["z_id"], data["z_proto"]),
        "identity__reconstruction": linear_cka(data["z_id"], data["z_rec"]),
        "prototype__reconstruction": linear_cka(data["z_proto"], data["z_rec"]),
    }
    known = data["y"] != -1
    pred_disagree = float((data["pred_id"][known] != data["pred_proto"][known]).mean())
    return {"raw_score_spearman": score_corr.tolist(), "raw_names": names,
            "embedding_cka": cka, "id_proto_pred_disagree_on_known": pred_disagree}


def _fig_histograms(data, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [("identity", "Identity  u (from 1-max softmax)"),
             ("prototype", "Prototype  u (nearest-prototype distance)"),
             ("reconstruction", "Reconstruction  u (best-case recon error)")]
    known_mask = data["y"] != -1
    for ax, (ch, title) in zip(axes, specs):
        u = data[f"u_{ch}"]
        bins = np.linspace(0, 1, 41)
        ax.hist(u[known_mask], bins=bins, density=True, alpha=0.55, label="known", color="#3b7dd8")
        ax.hist(u[~known_mask], bins=bins, density=True, alpha=0.55, label="unknown", color="#d85a3b")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("calibrated unknown evidence u")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_correlation(corr: Dict, path: Path):
    mat = np.array(corr["raw_score_spearman"])
    names = corr["raw_names"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Spearman corr of raw unknown scores")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_overlap(overlap: Dict, cond: Dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ch = CHANNELS
    recall = [overlap["recall_per_channel"][c] for c in ch] + [overlap["union_recall"]]
    axes[0].bar(ch + ["union"], recall, color=["#3b7dd8", "#3bb273", "#e0a32e", "#444444"])
    for i, v in enumerate(recall):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(f"unknown recall @tau={TAU}")
    axes[0].set_title("Per-channel vs union unknown detection")

    # conditional catch: unknowns missed by each channel rescued by helpers
    width = 0.35
    x = np.arange(len(ch))
    helper_names = [c for c in ch]
    for k, helper in enumerate([c for c in ch]):
        vals = []
        for missed in ch:
            row = cond[missed]
            vals.append(row.get(f"caught_by_{helper}", 0.0) if row["n_missed"] else 0.0)
        axes[1].bar(x + (k - 1) * width, vals, width, label=f"rescued by {helper}")
    axes[1].set_xticks(x, [f"missed by\n{c}" for c in ch])
    axes[1].set_ylabel("rescue fraction")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Conditional rescue of missed unknowns")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_tsne(data, path: Path):
    rng = np.random.default_rng(0)
    n = len(data["y"])
    take = rng.choice(n, size=min(3000, n), replace=False)
    y = data["y"][take]
    is_unk = y == -1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes, ["z_id", "z_proto", "z_rec"],
                              ["Identity embedding", "Prototype embedding", "Reconstruction embedding"]):
        z = data[key][take]
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(z)
        known_idx = ~is_unk
        sc = ax.scatter(emb[known_idx, 0], emb[known_idx, 1], c=y[known_idx], cmap="tab20",
                        s=6, alpha=0.6)
        ax.scatter(emb[is_unk, 0], emb[is_unk, 1], c="black", marker="x", s=18, alpha=0.7,
                   label="unknown")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_diagnostic(open_data, out_dir: str | Path, dataset_name: str) -> Dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    channel_metrics = evaluate_channels(open_data)
    cond = conditional_catch(open_data)
    overlap = detection_overlap(open_data)
    corr = correlations(open_data)
    known = open_data["y"] != -1
    known_acc = {
        "identity": float((open_data["pred_id"][known] == open_data["y"][known]).mean()),
        "prototype": float((open_data["pred_proto"][known] == open_data["y"][known]).mean()),
        "fused_closed": float((open_data["closed_pred"][known] == open_data["y"][known]).mean()),
    }

    best_single = max(channel_metrics[c]["auroc"] for c in CHANNELS)
    fusion_aurocs = {f: channel_metrics[f]["auroc"] for f in FUSIONS}
    best_fusion = max(fusion_aurocs, key=fusion_aurocs.get)
    fused_auroc = fusion_aurocs[best_fusion]
    max_pair_corr = np.abs(np.array(corr["raw_score_spearman"])[np.triu_indices(3, 1)]).max()
    id_missed = cond["identity"].get("caught_by_either_helper", 0.0)
    verdict = {
        "best_fusion_rule": best_fusion,
        "fusion_aurocs": fusion_aurocs,
        "best_single_auroc": float(best_single),
        "fused_beats_best_single_auroc": bool(fused_auroc > best_single + 1e-4),
        "fused_auroc_gain": float(fused_auroc - best_single),
        "max_abs_score_correlation": float(max_pair_corr),
        "not_collapsed_corr_lt_0_95": bool(max_pair_corr < 0.95),
        "identity_missed_unknown_rescued_fraction": float(id_missed),
        "union_recall": float(overlap["union_recall"]),
        "best_single_channel_recall": float(max(overlap["recall_per_channel"].values())),
        "union_adds_recall": bool(overlap["union_recall"]
                                  > max(overlap["recall_per_channel"].values()) + 1e-3),
    }

    report = {
        "dataset": dataset_name, "tau": TAU,
        "known_closed_set_accuracy": known_acc,
        "channel_metrics": channel_metrics,
        "conditional_catch": cond,
        "detection_overlap": overlap,
        "correlations": corr,
        "gate": verdict,
    }
    (out_dir / "complementarity_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    _fig_histograms(open_data, fig_dir / "score_histograms.png")
    _fig_correlation(corr, fig_dir / "score_correlation.png")
    _fig_overlap(overlap, cond, fig_dir / "overlap_and_rescue.png")
    try:
        _fig_tsne(open_data, fig_dir / "embedding_tsne.png")
        tsne_ok = True
    except Exception as exc:  # t-SNE is diagnostic only
        tsne_ok = False
        report["tsne_error"] = str(exc)

    _write_markdown(report, out_dir / "report_stage1.md", tsne_ok)
    return report


def _write_markdown(r: Dict, path: Path, tsne_ok: bool) -> None:
    g = r["gate"]
    lines = [f"# Stage-1 三证据互补性诊断 — {r['dataset']}", "",
             f"阈值 τ={r['tau']}（Known 验证集经验 CDF，真实 Unknown 不参与训练/校准）。", ""]
    lines += ["## 1. 各证据通道开放集能力", "",
              "| 通道 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall | Macro-F1 | H-score |",
              "|---|---|---|---|---|---|---|---|---|"]
    name_map = {"identity": "Identity", "prototype": "Prototype",
                "reconstruction": "Reconstruction",
                "fused_mean": "固定融合·均值", "fused_max": "固定融合·max/OR"}
    ka = r["known_closed_set_accuracy"]
    for ch in CHANNELS + FUSIONS:
        m = r["channel_metrics"][ch]
        if ch == "fused_mean" or ch == "fused_max":
            kacc = ka["fused_closed"]
        elif ch in ka:
            kacc = ka[ch]
        else:
            kacc = float("nan")  # reconstruction has no closed-set prediction
        kacc_str = "—" if np.isnan(kacc) else f"{kacc:.4f}"
        lines.append(f"| {name_map[ch]} | {m['auroc']:.4f} | {m['aupr_out']:.4f} | "
                     f"{m['fpr95']:.4f} | {m['oscr']:.4f} | {kacc_str} | "
                     f"{m['unknown_recall']:.4f} | {m['macro_f1']:.4f} | {m['h_score']:.4f} |")
    lines += ["", "## 2. Unknown 检测集合重叠与独有贡献 @τ", ""]
    ov = r["detection_overlap"]
    lines += [f"- 单通道召回：" + "，".join(f"{name_map[c]}={ov['recall_per_channel'][c]:.3f}" for c in CHANNELS),
              f"- **三通道并集召回={ov['union_recall']:.3f}**（最强单通道="
              f"{max(ov['recall_per_channel'].values()):.3f}）",
              "- 独有贡献（仅该通道检出的 Unknown 占比）：" +
              "，".join(f"{name_map[c]}={ov['unique_contribution'][c]:.3f}" for c in CHANNELS),
              "- 两两 Jaccard：" + "，".join(f"{k}={v['jaccard']:.3f}"
                                            for k, v in ov["pairwise_jaccard"].items())]
    lines += ["", "## 3. 条件救援（被某通道漏掉的 Unknown 被其他通道抓住的比例）", ""]
    for missed in CHANNELS:
        row = r["conditional_catch"][missed]
        if row["n_missed"] == 0:
            lines.append(f"- {name_map[missed]} 无漏检 Unknown。")
            continue
        helpers = "，".join(f"由{name_map[h]}抓住={row[f'caught_by_{h}']:.3f}"
                           for h in CHANNELS if h != missed)
        lines.append(f"- 被 {name_map[missed]} 漏掉 {row['n_missed']} 个：{helpers}；"
                     f"**任一其他通道救援={row['caught_by_either_helper']:.3f}**")
    lines += ["", "## 4. 多样性 / 防 Collapse", ""]
    c = r["correlations"]
    lines += [f"- 原始未知分数两两 Spearman 最大绝对值={g['max_abs_score_correlation']:.3f}",
              f"- 嵌入 CKA：I-P={c['embedding_cka']['identity__prototype']:.3f}，"
              f"I-R={c['embedding_cka']['identity__reconstruction']:.3f}，"
              f"P-R={c['embedding_cka']['prototype__reconstruction']:.3f}",
              f"- Known 上 Identity/Prototype 预测分歧率={c['id_proto_pred_disagree_on_known']:.3f}"]
    lines += ["", "## 5. 闸门判定（是否值得进入 Stage 2：Communication + Pseudo-Unknown）", ""]
    fa = g["fusion_aurocs"]
    lines += [f"- 固定融合 AUROC：均值={fa['fused_mean']:.4f}，max/OR={fa['fused_max']:.4f}；"
              f"最强单证据={g['best_single_auroc']:.4f}（最优融合规则：{g['best_fusion_rule']}）",
              f"- 存在固定融合超过最强单证据：{'是' if g['fused_beats_best_single_auroc'] else '否'}"
              f"（最优增益 {g['fused_auroc_gain']:+.4f}）",
              f"- 证据未完全相关（|Spearman|<0.95）：{'是' if g['not_collapsed_corr_lt_0_95'] else '否'}",
              f"- Identity 漏检 Unknown 被几何/重构救援比例={g['identity_missed_unknown_rescued_fraction']:.3f}",
              f"- 并集召回 {g['union_recall']:.3f} 相对最强单通道 {g['best_single_channel_recall']:.3f} "
              f"有增量：{'是' if g['union_adds_recall'] else '否'}",
              "", "说明：等权均值对强弱不一的通道不友好；max/OR 固定融合与并集召回更能体现互补性。",
              "", "图：`figures/score_histograms.png`、`score_correlation.png`、"
              "`overlap_and_rescue.png`" + ("、`embedding_tsne.png`" if tsne_ok else "（t-SNE 未生成）")]
    path.write_text("\n".join(lines), encoding="utf-8")
