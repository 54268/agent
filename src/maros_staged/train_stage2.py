"""Stage-2 training: Pseudo-Unknown Agent + Open-Space Boundary Agent.

No real unknown is used for generation, training, checkpoint selection or
threshold selection.  It is evaluated exactly once after the model and the
known-validation operating point have been frozen.
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .boundary_agent import (
    FeatureStandardizer,
    OpenSpaceBoundaryAgent,
    boundary_features,
    make_prototypes,
)
from .datasets import load_oracle_npz, load_wisig_subset
from .evidence import EmpiricalCdfCalibrator, collect_evidence
from .metrics_osr import detection_metrics, oscr, threshold_decision
from .model import Stage1ModelConfig, ThreeAgentModel
from .pseudo_unknown import AGENTS, DisagreementGuidedPseudoUnknownAgent, PseudoUnknownConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_splits(cfg: Dict):
    if cfg["dataset"] == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=False)
    if cfg["dataset"] == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=False)
    raise ValueError(f"unknown dataset {cfg['dataset']}")


def _load_stage1(path: str | Path, device: torch.device) -> ThreeAgentModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Stage1ModelConfig(**payload["config"])
    model = ThreeAgentModel(cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def extract_agent_features(model: ThreeAgentModel, loader: DataLoader,
                           device: torch.device) -> Dict[str, np.ndarray]:
    rows = {"y": [], "z_identity": [], "z_prototype": [], "z_reconstruction": [],
            "raw_evidence": []}
    model.eval()
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        id_ev = out["identity"]["evidence"]
        pr_ev = out["prototype"]["evidence"]
        fused_prob = 0.5 * id_ev["prob"] + 0.5 * pr_ev["prob"]
        candidate = fused_prob.argmax(dim=1)
        rec = model.reconstruction(x, out["h_s"], candidate)
        raw = torch.stack([
            1.0 - id_ev["confidence"],
            pr_ev["d1"],
            rec["evidence"]["error"],
        ], dim=1)
        rows["y"].append(y.numpy())
        rows["z_identity"].append(out["identity"]["embedding"].cpu().numpy())
        rows["z_prototype"].append(out["prototype"]["embedding"].cpu().numpy())
        rows["z_reconstruction"].append(rec["embedding"].cpu().numpy())
        rows["raw_evidence"].append(raw.cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def _embeddings(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: features[f"z_{name}"] for name in AGENTS}


@torch.no_grad()
def _predict(agent: OpenSpaceBoundaryAgent, x: np.ndarray, device: torch.device,
             batch_size: int = 2048) -> Dict[str, np.ndarray]:
    agent.eval()
    logits, pred = [], []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size]).to(device)
        out = agent(batch)
        logits.append(out["unknown_logit"].cpu().numpy())
        pred.append(out["class_logits"].argmax(dim=1).cpu().numpy())
    logit = np.concatenate(logits)
    return {"score": 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))),
            "class_pred": np.concatenate(pred)}


def _train_boundary(x_known: np.ndarray, y_known: np.ndarray,
                    x_pseudo: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                    x_pseudo_val: np.ndarray, num_classes: int, cfg: Dict,
                    device: torch.device):
    agent = OpenSpaceBoundaryAgent(
        x_known.shape[1], num_classes,
        hidden_dim=int(cfg.get("boundary_hidden_dim", 192)),
        dropout=float(cfg.get("boundary_dropout", 0.15)),
    ).to(device)
    opt = torch.optim.AdamW(agent.parameters(), lr=float(cfg.get("boundary_lr", 1e-3)),
                            weight_decay=float(cfg.get("boundary_weight_decay", 1e-4)))
    epochs = int(cfg.get("boundary_epochs", 35))
    batch_size = int(cfg.get("boundary_batch_size", 512))
    aux_weight = float(cfg.get("class_aux_weight", 0.25))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    history, best_state, best_score, best_epoch = [], None, -np.inf, -1

    for epoch in range(1, epochs + 1):
        # Equal known/pseudo sampling prevents the boundary prior from being
        # determined by the generator's arbitrary variation count.
        n = max(len(x_known), len(x_pseudo))
        ik = rng.choice(len(x_known), size=n, replace=len(x_known) < n)
        ip = rng.choice(len(x_pseudo), size=n, replace=len(x_pseudo) < n)
        x_epoch = np.concatenate([x_known[ik], x_pseudo[ip]], axis=0)
        u_epoch = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
        y_epoch = np.concatenate([y_known[ik], np.full(n, -1, dtype=np.int64)])
        order = rng.permutation(len(x_epoch))
        dataset = TensorDataset(
            torch.from_numpy(x_epoch[order]),
            torch.from_numpy(u_epoch[order]),
            torch.from_numpy(y_epoch[order]),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        agent.train()
        running = 0.0
        for xb, ub, yb in loader:
            xb, ub, yb = xb.to(device), ub.to(device), yb.to(device)
            out = agent(xb)
            bce = F.binary_cross_entropy_with_logits(out["unknown_logit"], ub)
            known = yb >= 0
            aux = F.cross_entropy(out["class_logits"][known], yb[known])
            loss = bce + aux_weight * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)

        val_known = _predict(agent, x_val, device)
        val_pseudo = _predict(agent, x_pseudo_val, device)
        threshold = float(np.quantile(val_known["score"], 0.95))
        pseudo_recall = float((val_pseudo["score"] >= threshold).mean())
        auc = float(roc_auc_score(
            np.concatenate([np.zeros(len(x_val)), np.ones(len(x_pseudo_val))]),
            np.concatenate([val_known["score"], val_pseudo["score"]]),
        ))
        class_acc = float((val_known["class_pred"] == y_val).mean())
        score = auc + 0.5 * pseudo_recall + 0.1 * class_acc
        row = {"epoch": epoch, "loss": running / len(dataset),
               "val_pseudo_auroc": auc, "val_pseudo_recall_at_known95": pseudo_recall,
               "val_class_acc": class_acc, "known95_threshold": threshold}
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  boundary ep{epoch:03d} loss={row['loss']:.4f} "
                  f"pseudo_auc={auc:.4f} pseudo_rec={pseudo_recall:.4f} "
                  f"known_cls={class_acc:.4f}")

    agent.load_state_dict(best_state)
    return agent, history, best_epoch


def _evaluate(open_data: Dict[str, np.ndarray], u_boundary: np.ndarray,
              val_data: Dict[str, np.ndarray], u_boundary_val: np.ndarray) -> Dict:
    y = open_data["y"]
    is_unknown = (y == -1).astype(np.int32)
    closed = open_data["closed_pred"]
    raw_open = {
        "stage1_mean3": open_data["u_fused_mean"],
        "stage1_max3": open_data["u_fused_max"],
        "boundary": u_boundary,
        "mean4": np.stack([
            open_data["u_identity"], open_data["u_prototype"],
            open_data["u_reconstruction"], u_boundary,
        ], axis=1).mean(axis=1),
        "max4": np.stack([
            open_data["u_identity"], open_data["u_prototype"],
            open_data["u_reconstruction"], u_boundary,
        ], axis=1).max(axis=1),
    }
    raw_val = {
        "stage1_mean3": val_data["u_fused_mean"],
        "stage1_max3": val_data["u_fused_max"],
        "boundary": u_boundary_val,
        "mean4": np.stack([
            val_data["u_identity"], val_data["u_prototype"],
            val_data["u_reconstruction"], u_boundary_val,
        ], axis=1).mean(axis=1),
        "max4": np.stack([
            val_data["u_identity"], val_data["u_prototype"],
            val_data["u_reconstruction"], u_boundary_val,
        ], axis=1).max(axis=1),
    }
    result = {}
    for name, open_score in raw_open.items():
        # A shared numeric threshold on mean and max is not a shared operating
        # point.  Recalibrate each complete rule on known validation so tau=.95
        # means approximately the same 5% known rejection budget for all rules.
        rule_calibrator = EmpiricalCdfCalibrator(raw_val[name])
        score = rule_calibrator.score(open_score)
        metrics = detection_metrics(is_unknown, score)
        metrics["oscr"] = oscr(is_unknown, closed, y, 1.0 - score)
        metrics.update(threshold_decision(is_unknown, y, closed, score, 0.95))
        result[name] = metrics
    return result


def _plot_scores(val_score: np.ndarray, pseudo_score: np.ndarray,
                 open_score: np.ndarray, open_y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(val_score, bins=bins, density=True, alpha=0.45, label="validation known")
    ax.hist(pseudo_score, bins=bins, density=True, alpha=0.45, label="held-out pseudo unknown")
    ax.hist(open_score[open_y == -1], bins=bins, density=True, histtype="step", linewidth=2,
            label="real unknown (offline only)")
    ax.set_xlabel("Boundary Agent calibrated unknown evidence")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_stage2(config: Dict) -> Dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    splits = _load_splits(config)
    loader_bs = int(config.get("extract_batch_size", 512))
    train_loader = DataLoader(splits.train, batch_size=loader_bs, shuffle=False)
    val_loader = DataLoader(splits.val, batch_size=loader_bs, shuffle=False)
    open_loader = DataLoader(splits.open_test, batch_size=loader_bs, shuffle=False)
    stage1 = _load_stage1(config["stage1_checkpoint"], device)
    print(f"[stage2] dataset={config['dataset']} device={device} extracting frozen Stage-1 features")
    train_f = extract_agent_features(stage1, train_loader, device)
    val_f = extract_agent_features(stage1, val_loader, device)
    open_f = extract_agent_features(stage1, open_loader, device)

    pug_cfg = PseudoUnknownConfig(**config.get("pseudo_unknown", {}), seed=seed)
    pug = DisagreementGuidedPseudoUnknownAgent(pug_cfg)
    pseudo = pug.generate(_embeddings(train_f), train_f["y"], train_f["raw_evidence"])
    eligible = pseudo["in_shell"]
    min_required = splits.num_known * 10
    if int(eligible.sum()) < min_required:
        eligible = np.ones(len(pseudo["source_indices"]), dtype=bool)
        shell_fallback = True
    else:
        shell_fallback = False
    pseudo_embeddings = {
        name: pseudo[f"z_{name}"][eligible] for name in AGENTS
    }
    pseudo_sources = pseudo["source_indices"][eligible]

    rng = np.random.default_rng(seed + 991)
    unique_sources = np.unique(pseudo_sources)
    rng.shuffle(unique_sources)
    n_val_sources = max(1, int(round(len(unique_sources) * float(config.get("pseudo_val_ratio", 0.2)))))
    val_sources = set(unique_sources[:n_val_sources].tolist())
    pseudo_val_mask = np.asarray([int(s) in val_sources for s in pseudo_sources])
    pseudo_train_mask = ~pseudo_val_mask

    prototypes = make_prototypes(_embeddings(train_f), train_f["y"], splits.num_known)
    x_train, _ = boundary_features(_embeddings(train_f), prototypes)
    x_val, _ = boundary_features(_embeddings(val_f), prototypes)
    x_open, _ = boundary_features(_embeddings(open_f), prototypes)
    x_pseudo, _ = boundary_features(pseudo_embeddings, prototypes)
    standardizer = FeatureStandardizer().fit(x_train)
    x_train = standardizer.transform(x_train)
    x_val = standardizer.transform(x_val)
    x_open = standardizer.transform(x_open)
    x_pseudo = standardizer.transform(x_pseudo)

    print(f"[stage2] generated={len(pseudo['source_indices'])} shell={int(pseudo['in_shell'].sum())} "
          f"used={len(x_pseudo)} sources={len(unique_sources)} fallback={shell_fallback}")
    boundary, history, best_epoch = _train_boundary(
        x_train, train_f["y"], x_pseudo[pseudo_train_mask],
        x_val, val_f["y"], x_pseudo[pseudo_val_mask], splits.num_known,
        config, device,
    )

    pred_val = _predict(boundary, x_val, device)
    pred_pseudo = _predict(boundary, x_pseudo[pseudo_val_mask], device)
    pred_open = _predict(boundary, x_open, device)
    boundary_cal = EmpiricalCdfCalibrator(pred_val["score"])
    u_val = boundary_cal.score(pred_val["score"])
    u_pseudo = boundary_cal.score(pred_pseudo["score"])
    u_open = boundary_cal.score(pred_open["score"])

    # Recompute Stage-1 calibrated evidence from the exact checkpoint so array
    # ordering and all fusion scores are guaranteed to match Stage 2.
    val_evidence, calibrators = collect_evidence(stage1, val_loader, device,
                                                 top_k=stage1.config.recon_top_k)
    open_evidence, _ = collect_evidence(stage1, open_loader, device,
                                        top_k=stage1.config.recon_top_k,
                                        calibrators=calibrators)
    if not np.array_equal(open_evidence["y"], open_f["y"]):
        raise RuntimeError("Stage-1 evidence and Stage-2 features are misaligned")
    metrics = _evaluate(open_evidence, u_open, val_evidence, u_val)
    pseudo_validation = {
        "auroc": float(roc_auc_score(
            np.concatenate([np.zeros(len(u_val)), np.ones(len(u_pseudo))]),
            np.concatenate([u_val, u_pseudo]),
        )),
        "recall_at_tau_0.95": float((u_pseudo >= 0.95).mean()),
        "num_validation_known": int(len(u_val)),
        "num_validation_pseudo": int(len(u_pseudo)),
    }

    torch.save({
        "state_dict": boundary.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(splits.num_known),
        "best_epoch": int(best_epoch),
        "standardizer_mean": standardizer.mean,
        "standardizer_std": standardizer.std,
        "prototypes": prototypes,
        "config": config,
    }, out_dir / "boundary_agent.pt")
    np.savez_compressed(
        out_dir / "pseudo_unknown_artifacts.npz",
        source_indices=pseudo_sources,
        source_classes=pseudo["source_classes"][eligible],
        eta=pseudo["eta"][eligible],
        in_shell=pseudo["in_shell"][eligible],
        pseudo_val_mask=pseudo_val_mask,
        seed_score=pseudo["seed_score"],
        selected_source_indices=pseudo["selected_source_indices"],
    )
    np.savez_compressed(out_dir / "boundary_scores.npz", y=open_evidence["y"],
                        boundary_raw=pred_open["score"], u_boundary=u_open,
                        closed_pred=open_evidence["closed_pred"])
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "dataset": config["dataset"], "seed": seed, "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - t0,
        "pseudo_generation": {
            "num_candidates": int(len(pseudo["source_indices"])),
            "num_in_shell": int(pseudo["in_shell"].sum()),
            "num_used": int(len(x_pseudo)),
            "num_sources": int(len(unique_sources)),
            "shell_fallback": shell_fallback,
            "shell_low": float(pseudo["shell_low"]),
            "shell_high": float(pseudo["shell_high"]),
            "config": dataclasses.asdict(pug_cfg),
        },
        "pseudo_validation": pseudo_validation,
        "open_test_metrics": metrics,
        "protocol": {
            "real_unknown_used_for_training": False,
            "real_unknown_used_for_threshold": False,
            "checkpoint_selection": "known validation + held-out generated pseudo-unknown only",
        },
    }
    (out_dir / "stage2_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_scores(u_val, u_pseudo, u_open, open_evidence["y"], out_dir / "boundary_score_histogram.png")
    _write_report(summary, out_dir / "report_stage2.md")
    print(f"[stage2] best_epoch={best_epoch} elapsed={summary['elapsed_seconds']:.1f}s")
    return summary


def _write_report(summary: Dict, path: Path) -> None:
    pg, pv, metrics = (summary["pseudo_generation"], summary["pseudo_validation"],
                       summary["open_test_metrics"])
    names = {
        "stage1_mean3": "Stage-1 三证据均值",
        "stage1_max3": "Stage-1 三证据 max/OR",
        "boundary": "Pseudo-Unknown + Boundary Agent",
        "mean4": "四证据均值",
        "max4": "四证据 max/OR",
    }
    lines = [
        f"# Stage-2 Pseudo-Unknown + Boundary Agent — {summary['dataset']}", "",
        "真实 Unknown 不参与生成、训练、选轮或阈值；只用于本报告最终离线评价。", "",
        "## 伪未知质量控制", "",
        f"- 候选 {pg['num_candidates']}，已知边界壳内 {pg['num_in_shell']}，实际使用 {pg['num_used']}；"
        f"独立源样本 {pg['num_sources']}。",
        f"- 边界壳：最近联合原型距离 [{pg['shell_low']:.4f}, {pg['shell_high']:.4f}]；"
        f"回退使用全部候选：{'是' if pg['shell_fallback'] else '否'}。",
        f"- 留出伪未知 AUROC={pv['auroc']:.4f}，Recall@Known-CDF τ=.95={pv['recall_at_tau_0.95']:.4f}。",
        "", "## WiSig 真实开放集结果（离线评价）", "",
        "每一种完整融合规则均再次用 Known validation CDF 校准；τ=.95 表示约 95% Known 接受率。", "",
        "| 规则 | AUROC↑ | AUPR-Out↑ | FPR95↓ | OSCR↑ | Known Acc↑ | Unknown Recall↑ | Macro-F1↑ | H-score↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("stage1_mean3", "stage1_max3", "boundary", "mean4", "max4"):
        m = metrics[key]
        lines.append(
            f"| {names[key]} | {m['auroc']:.4f} | {m['aupr_out']:.4f} | {m['fpr95']:.4f} "
            f"| {m['oscr']:.4f} | {m['known_accuracy']:.4f} | {m['unknown_recall']:.4f} "
            f"| {m['macro_f1']:.4f} | {m['h_score']:.4f} |"
        )
    lines += ["", "图：`boundary_score_histogram.png`。",
              "下一闸门：只有边界证据在真实 Unknown 上不是反向、且不以大幅牺牲 Known Accuracy 换召回，"
              "才接入 Communication；否则先调整生成策略，不用测试 Unknown 反向调参。"]
    path.write_text("\n".join(lines), encoding="utf-8")
