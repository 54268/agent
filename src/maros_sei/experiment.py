from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CombinedTestDataset, WiSigArrayDataset, load_json, prepare_dataset, save_json
from .metrics import evaluate_open_set
from .model import MAROSSEI, ModelConfig
from .subdivision import run_unknown_subdivision


AGENT_NAMES = ["time", "frequency_phase", "domain", "open_set"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_pseudo_unknown(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch = len(x)
    permutation = torch.randperm(batch, device=x.device)
    if batch > 1:
        same = labels[permutation] == labels
        fallback = torch.roll(permutation, shifts=1)
        permutation = torch.where(same, fallback, permutation)
    mix = torch.empty(batch, 1, 1, device=x.device).uniform_(0.35, 0.65)
    pseudo = mix * x + (1.0 - mix) * x[permutation]
    angle = torch.empty(batch, 1, device=x.device).uniform_(-math.pi, math.pi)
    cos_a, sin_a = angle.cos(), angle.sin()
    i, q = pseudo[:, 0].clone(), pseudo[:, 1].clone()
    pseudo[:, 0] = cos_a * i - sin_a * q
    pseudo[:, 1] = sin_a * i + cos_a * q
    pseudo = pseudo + 0.06 * torch.randn_like(pseudo)
    if pseudo.shape[-1] >= 32:
        start = int(torch.randint(0, pseudo.shape[-1] - 16, (1,), device=x.device))
        pseudo[:, :, start : start + 16] *= 0.25
    power = pseudo.square().sum(dim=1).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return pseudo / power[:, None]


def diversity_loss(states: torch.Tensor) -> torch.Tensor:
    similarity = torch.einsum("bid,bjd->bij", F.normalize(states, dim=-1), F.normalize(states, dim=-1))
    mask = ~torch.eye(states.shape[1], device=states.device, dtype=torch.bool)[None]
    return similarity.square().masked_select(mask.expand_as(similarity)).mean()


def counterfactual_router_loss(
    logits: torch.Tensor,
    agent_logits: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    full_loss = F.cross_entropy(logits, labels, reduction="none")
    utilities = []
    for agent in range(weights.shape[1]):
        remaining = weights.clone()
        remaining[:, agent] = 0.0
        remaining = remaining / remaining.sum(dim=1, keepdim=True).clamp_min(1e-6)
        leave_logits = (remaining[..., None] * agent_logits).sum(dim=1)
        utilities.append(F.cross_entropy(leave_logits, labels, reduction="none") - full_loss)
    target = torch.softmax(torch.stack(utilities, dim=1).detach() / 0.20, dim=1)
    return -(target * torch.log(weights + 1e-8)).sum(dim=1).mean()


def compute_losses(
    model: MAROSSEI,
    known: dict[str, torch.Tensor],
    pseudo: dict[str, torch.Tensor],
    labels: torch.Tensor,
    rx_ids: torch.Tensor,
    date_ids: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    expanded = labels[:, None].expand(-1, 4).reshape(-1)
    team = F.cross_entropy(known["fused_logits"], labels)
    local = 0.5 * (
        F.cross_entropy(known["pre_logits"].reshape(-1, model.config.num_classes), expanded)
        + F.cross_entropy(known["post_logits"].reshape(-1, model.config.num_classes), expanded)
    )
    prototype = model.prototype_loss(known["post_states"], labels)
    domain = F.cross_entropy(known["rx_logits"], rx_ids) + F.cross_entropy(known["date_logits"], date_ids)
    for index in range(2):
        domain = domain + 0.5 * F.cross_entropy(known["adv_rx_logits"][:, index], rx_ids)
        domain = domain + 0.5 * F.cross_entropy(known["adv_date_logits"][:, index], date_ids)
    diverse = diversity_loss(known["pre_states"])
    zeros = torch.zeros_like(known["unknown_logit"])
    ones = torch.ones_like(pseudo["unknown_logit"])
    open_loss = (
        F.binary_cross_entropy_with_logits(known["unknown_logit"], zeros)
        + F.binary_cross_entropy_with_logits(pseudo["unknown_logit"], ones)
        + 0.5 * F.binary_cross_entropy_with_logits(known["private_open_logit"], zeros)
        + 0.5 * F.binary_cross_entropy_with_logits(pseudo["private_open_logit"], ones)
    )
    route = counterfactual_router_loss(
        known["fused_logits"], known["post_logits"], known["fusion_weights"], labels
    )
    balance = (known["fusion_weights"].mean(dim=0) - 0.25).square().mean()
    budget = known["reliability"].mean()
    parts = {
        "team": team,
        "local": local,
        "prototype": prototype,
        "domain": domain,
        "diversity": diverse,
        "open": open_loss,
        "route": route,
        "balance": balance,
        "budget": budget,
    }
    total = sum(float(weights[name]) * value for name, value in parts.items())
    return total, {name: float(value.detach()) for name, value in parts.items()}


@torch.no_grad()
def collect_outputs(
    model: MAROSSEI,
    loader: DataLoader,
    device: torch.device,
    include_pseudo: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    buffers: dict[str, list[np.ndarray]] = {
        key: []
        for key in [
            "labels", "rx_ids", "date_ids", "indices", "fused_logits", "post_logits",
            "pre_logits", "embedding", "open_features", "unknown_logit", "fusion_weights",
            "reliability", "adjacency", "pseudo_open_features", "pseudo_unknown_logit",
        ]
    }
    for batch in loader:
        x = batch["iq"].to(device, non_blocking=True)
        out = model(x, grl_scale=0.0)
        buffers["labels"].append(batch["label"].numpy())
        buffers["rx_ids"].append(batch["rx_id"].numpy())
        buffers["date_ids"].append(batch["date_id"].numpy())
        buffers["indices"].append(batch["index"].numpy())
        for source, target in [
            ("fused_logits", "fused_logits"), ("post_logits", "post_logits"),
            ("pre_logits", "pre_logits"), ("fused_embedding", "embedding"),
            ("open_features", "open_features"), ("unknown_logit", "unknown_logit"),
            ("fusion_weights", "fusion_weights"), ("reliability", "reliability"),
            ("adjacency", "adjacency"),
        ]:
            buffers[target].append(out[source].float().cpu().numpy())
        if include_pseudo:
            pseudo_x = make_pseudo_unknown(x, batch["label"].to(device))
            pseudo = model(pseudo_x, grl_scale=0.0)
            buffers["pseudo_open_features"].append(pseudo["open_features"].float().cpu().numpy())
            buffers["pseudo_unknown_logit"].append(pseudo["unknown_logit"].float().cpu().numpy())
    result = {}
    for key, values in buffers.items():
        if values:
            result[key] = np.concatenate(values, axis=0)
    return result


def calibrator_features(outputs: dict[str, np.ndarray], pseudo: bool = False) -> np.ndarray:
    prefix = "pseudo_" if pseudo else ""
    return np.column_stack([outputs[f"{prefix}open_features"], outputs[f"{prefix}unknown_logit"]])


def select_threshold(
    y_true: np.ndarray,
    closed_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
) -> tuple[float, dict[str, float]]:
    known_mask = y_true != unknown_label
    closed_accuracy = float((closed_pred[known_mask] == y_true[known_mask]).mean())
    minimum_known = 0.90 * closed_accuracy
    candidates = np.unique(np.quantile(unknown_score, np.linspace(0.01, 0.99, 399)))
    best: tuple[float, float, dict[str, float]] | None = None
    fallback: tuple[float, float, dict[str, float]] | None = None
    for threshold in candidates:
        predicted = closed_pred.copy()
        predicted[unknown_score >= threshold] = unknown_label
        metrics, _ = evaluate_open_set(y_true, predicted, closed_pred, unknown_score, unknown_label)
        objective = (
            0.45 * metrics["known_accuracy"]
            + 0.35 * metrics["unknown_recall"]
            + 0.15 * metrics["macro_f1"]
            + 0.05 * metrics["auroc"]
        )
        row = (objective, float(threshold), metrics)
        if fallback is None or row[0] > fallback[0]:
            fallback = row
        if metrics["known_accuracy"] >= minimum_known and (best is None or row[0] > best[0]):
            best = row
    chosen = best or fallback
    assert chosen is not None
    metrics = dict(chosen[2])
    metrics["selection_objective"] = float(chosen[0])
    metrics["minimum_known_accuracy"] = float(minimum_known)
    return chosen[1], metrics


def fit_calibrator(
    outputs: dict[str, np.ndarray], unknown_label: int, seed: int
) -> tuple[Pipeline, float, dict[str, Any]]:
    labels = outputs["labels"]
    known_features = calibrator_features(outputs)
    pseudo_features = calibrator_features(outputs, pseudo=True)
    rng = np.random.default_rng(seed)
    calibration_indices: list[int] = []
    selection_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        midpoint = max(1, len(indices) // 2)
        calibration_indices.extend(indices[:midpoint].tolist())
        selection_indices.extend(indices[midpoint:].tolist())
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    selection_indices = np.asarray(selection_indices, dtype=np.int64)

    x_fit = np.concatenate([known_features[calibration_indices], pseudo_features[calibration_indices]])
    y_fit = np.concatenate(
        [np.zeros(len(calibration_indices), dtype=np.int64), np.ones(len(calibration_indices), dtype=np.int64)]
    )
    calibrator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    ).fit(x_fit, y_fit)

    x_select = np.concatenate([known_features[selection_indices], pseudo_features[selection_indices]])
    score = calibrator.predict_proba(x_select)[:, 1]
    closed_known = outputs["fused_logits"][selection_indices].argmax(axis=1)
    closed_pseudo = outputs["fused_logits"][selection_indices].argmax(axis=1)
    closed = np.concatenate([closed_known, closed_pseudo])
    y_true = np.concatenate(
        [labels[selection_indices], np.full(len(selection_indices), unknown_label, dtype=np.int64)]
    )
    threshold, selection_metrics = select_threshold(y_true, closed, score, unknown_label)
    return calibrator, threshold, {
        "threshold": threshold,
        "fit_known": int(len(calibration_indices)),
        "fit_pseudo_unknown": int(len(calibration_indices)),
        "selection_known": int(len(selection_indices)),
        "selection_pseudo_unknown": int(len(selection_indices)),
        "selection_metrics": selection_metrics,
        "features": [
            "private_open_logit", "one_minus_max_softmax", "fused_entropy", "agent_js_divergence",
            "domain_shift", "router_entropy", "normalized_energy", "network_unknown_logit",
        ],
    }


def _validation_score(outputs: dict[str, np.ndarray]) -> dict[str, float]:
    closed_accuracy = float((outputs["fused_logits"].argmax(1) == outputs["labels"]).mean())
    known_score = 1.0 / (1.0 + np.exp(-outputs["unknown_logit"]))
    pseudo_score = 1.0 / (1.0 + np.exp(-outputs["pseudo_unknown_logit"]))
    from sklearn.metrics import roc_auc_score

    target = np.concatenate([np.zeros(len(known_score)), np.ones(len(pseudo_score))])
    score = np.concatenate([known_score, pseudo_score])
    pseudo_auroc = float(roc_auc_score(target, score))
    selection = closed_accuracy + 0.25 * pseudo_auroc - 0.10 * float(known_score.mean())
    return {
        "closed_accuracy": closed_accuracy,
        "pseudo_auroc": pseudo_auroc,
        "known_unknown_score_mean": float(known_score.mean()),
        "selection_score": selection,
    }


def _save_plots(
    run_dir: Path,
    history: list[dict[str, float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
    curves: dict[str, np.ndarray],
    weights: np.ndarray,
) -> None:
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["val_closed_accuracy"] for row in history], label="Known accuracy")
    axes[1].plot(epochs, [row["val_pseudo_auroc"] for row in history], label="Pseudo AUROC")
    axes[1].legend()
    axes[1].set(title="Validation", xlabel="Epoch", ylabel="Score")
    fig.tight_layout()
    fig.savefig(figures / "training_curves.png", dpi=180)
    plt.close(fig)

    binary = (y_true == unknown_label).astype(np.int32)
    fpr, tpr, _ = roc_curve(binary, unknown_score)
    precision, recall, _ = precision_recall_curve(binary, unknown_score)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Known vs unknown ROC", xlabel="Known FPR", ylabel="Unknown TPR")
    axes[1].plot(recall, precision)
    axes[1].set(title="Unknown precision-recall", xlabel="Recall", ylabel="Precision")
    axes[2].plot(curves["oscr_fpr"], curves["oscr_ccr"])
    axes[2].set(title="OSCR curve", xlabel="Unknown false accept rate", ylabel="Correct classification rate")
    fig.tight_layout()
    fig.savefig(figures / "open_set_curves.png", dpi=180)
    plt.close(fig)

    collapsed_true = binary
    collapsed_pred = (y_pred == unknown_label).astype(np.int32)
    matrix = confusion_matrix(collapsed_true, collapsed_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["Known", "Unknown"])
    ax.set_yticks([0, 1], ["Known", "Unknown"])
    ax.set(xlabel="Predicted", ylabel="True", title="Rejection confusion")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "rejection_confusion.png", dpi=180)
    plt.close(fig)

    known = y_true != unknown_label
    means = np.stack([weights[known].mean(0), weights[~known].mean(0)])
    x = np.arange(len(AGENT_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, means[0], 0.36, label="Known")
    ax.bar(x + 0.18, means[1], 0.36, label="Unknown")
    ax.set_xticks(x, AGENT_NAMES, rotation=15)
    ax.set(ylabel="Mean fusion weight", title="Dynamic Router usage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "router_weights_known_unknown.png", dpi=180)
    plt.close(fig)


def _write_report(run_dir: Path, config: dict[str, Any], metrics: dict[str, Any], subdivision: dict[str, Any], router: dict[str, Any]) -> None:
    rows = [
        ("Known Accuracy", metrics["known_accuracy"]),
        ("Closed-set Known Accuracy", metrics["closed_set_known_accuracy"]),
        ("Unknown Precision", metrics["unknown_precision"]),
        ("Unknown Recall", metrics["unknown_recall"]),
        ("Unknown F1", metrics["unknown_f1"]),
        ("Macro-F1", metrics["macro_f1"]),
        ("AUROC", metrics["auroc"]),
        ("AUPR-Out", metrics["aupr_out"]),
        ("FPR95", metrics["fpr95"]),
        ("OSCR（旧方案兼容端点）", metrics["oscr"]),
        ("OSCR-standard", metrics["oscr_standard"]),
    ]
    lines = [
        "# MAROS-SEI F174 首轮完整实验报告",
        "",
        "## 流程",
        "",
        "`WiSig IQ → RMS归一化 → 时间/频相/域/开放集四Agent → Router生成有向Top-k通信图 → Agent接收消息更新 → 动态融合 → 已知分类与Unknown评分 → 验证集校准 → 测试拒识 → 被拒样本无监督细分`",
        "",
        "真实未知 Tx 只在最终测试和离线聚类指标中使用。训练期 Unknown 监督由不同已知类信号混合、相位扰动、噪声和局部遮挡生成。",
        "",
        "## 多智能体协作",
        "",
        "- Time Agent：从原始 IQ 学习局部时间指纹。",
        "- Frequency/Phase Agent：从 FFT 幅度和相邻频点相位差学习频相指纹。",
        "- Domain Agent：从幅相、相关性和相位增量统计学习 Rx/date 域状态，并为消息提供可靠性依据。",
        "- Open-Set Agent：读取前三个 Agent 的状态，形成未知风险私有证据。",
        "- Dynamic Router：为每个样本生成 Agent 间有向 top-k 图、消息可靠性和最终信任权重。",
        "- Fusion/Decision Agent：使用通信后的局部 logits 和 Router 权重联合输出 known class 与 unknown score。",
        "",
        "## 数据",
        "",
        f"- Known/Unknown：{config['protocol']['known_classes']} / {config['protocol']['total_classes'] - config['protocol']['known_classes']}",
        f"- 训练 seed：{config['training']['seed']}；类别 seed：{config['protocol']['class_partition_seed']}",
        f"- 拒识阈值：{metrics['threshold']:.6f}",
        "",
        "## 拒识结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        *[f"| {name} | {value:.6f} |" for name, value in rows],
        "",
        "## 未知类细分结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 自动选择细分类数 | {subdivision['resolved_num_clusters']} |",
        f"| 真实未知类数（仅评价） | {subdivision['num_true_unknown_classes']} |",
        f"| NMI | {subdivision['nmi']:.6f} |",
        f"| ARI | {subdivision['ari']:.6f} |",
        f"| Purity | {subdivision['purity']:.6f} |",
        f"| Hungarian Accuracy | {subdivision['hungarian_accuracy']:.6f} |",
        f"| Unknown-cache Precision | {subdivision['unknown_cache_precision']:.6f} |",
        f"| Unknown-cache Recall | {subdivision['unknown_cache_recall']:.6f} |",
        f"| Coverage of total test unknown | {subdivision['coverage_of_total_test_unknown']:.6f} |",
        "",
        "## Router 摘要",
        "",
        "| Agent | Known平均权重 | Unknown平均权重 |",
        "|---|---:|---:|",
    ]
    for index, name in enumerate(AGENT_NAMES):
        lines.append(f"| {name} | {router['known_mean_weights'][index]:.6f} | {router['unknown_mean_weights'][index]:.6f} |")
    lines.extend(
        [
            "",
            "这是一轮单 seed 的端到端结果，用于验证完整流程；论文结论仍需按实验矩阵补齐重复 seed、基线和消融。",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config_path: str | Path, force_prepare: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    config = load_json(config_path)
    config["project_root"] = str(project_root)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    processed = prepare_dataset(config, force=force_prepare)
    manifest = load_json(processed / "manifest.json")
    unknown_label = int(manifest["unknown_label"])

    run_dir = project_root / config["output"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "data_manifest.json", manifest)

    train_dataset = WiSigArrayDataset(processed, "train", augment=True, seed=seed)
    val_dataset = WiSigArrayDataset(processed, "validation", augment=False, seed=seed)
    test_known = WiSigArrayDataset(processed, "test_known", augment=False, seed=seed)
    test_unknown = WiSigArrayDataset(processed, "test_unknown", augment=False, seed=seed)
    test_dataset = CombinedTestDataset(test_known, test_unknown)
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAROSSEI(
        ModelConfig(
            num_classes=unknown_label,
            num_receivers=len(manifest["rx_names"]),
            num_dates=len(manifest["date_names"]),
            embedding_dim=int(config["model"]["embedding_dim"]),
            top_k=int(config["model"]["top_k"]),
        )
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["training"]["epochs"]))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    patience = int(config["training"].get("patience", 5))
    started = time.time()

    print(f"[train] device={device} parameters={parameter_count:,} train={len(train_dataset):,}", flush=True)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        totals: dict[str, float] = {"loss": 0.0}
        steps = 0
        grl_scale = min(1.0, epoch / max(int(config["training"]["epochs"]) * 0.4, 1.0))
        for batch in train_loader:
            x = batch["iq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            rx_ids = batch["rx_id"].to(device, non_blocking=True)
            date_ids = batch["date_id"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                known = model(x, grl_scale=grl_scale)
                pseudo = model(make_pseudo_unknown(x, labels), grl_scale=0.0)
                loss, parts = compute_losses(
                    model, known, pseudo, labels, rx_ids, date_ids, config["loss_weights"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 5.0)))
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
        scheduler.step()
        torch.manual_seed(seed + epoch)
        validation_outputs = collect_outputs(model, val_loader, device, include_pseudo=True)
        validation = _validation_score(validation_outputs)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(steps, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "val_closed_accuracy": validation["closed_accuracy"],
            "val_pseudo_auroc": validation["pseudo_auroc"],
            "val_selection_score": validation["selection_score"],
            **{f"loss_{key}": value / max(steps, 1) for key, value in totals.items() if key != "loss"},
        }
        history.append(row)
        print(
            f"[epoch {epoch:02d}] loss={row['train_loss']:.4f} val_acc={row['val_closed_accuracy']:.4f} "
            f"pseudo_auc={row['val_pseudo_auroc']:.4f} score={row['val_selection_score']:.4f}",
            flush=True,
        )
        if validation["selection_score"] > best_score:
            best_score = validation["selection_score"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": best_score}, run_dir / "best.pt")
        elif epoch - best_epoch >= patience:
            print(f"[train] early stopping at epoch={epoch}, best={best_epoch}", flush=True)
            break

    torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"]}, run_dir / "last.pt")
    save_json(run_dir / "training_history.json", {"epochs": history, "best_epoch": best_epoch, "best_score": best_score})
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])

    torch.manual_seed(seed + 10000)
    validation_outputs = collect_outputs(model, val_loader, device, include_pseudo=True)
    calibrator, threshold, calibration = fit_calibrator(validation_outputs, unknown_label, seed)
    joblib.dump(calibrator, run_dir / "unknown_calibrator.joblib")
    save_json(run_dir / "calibration.json", calibration)

    test_outputs = collect_outputs(model, test_loader, device, include_pseudo=False)
    unknown_score = calibrator.predict_proba(calibrator_features(test_outputs))[:, 1]
    closed_pred = test_outputs["fused_logits"].argmax(axis=1).astype(np.int64)
    y_true = test_outputs["labels"].astype(np.int64)
    y_pred = closed_pred.copy()
    y_pred[unknown_score >= threshold] = unknown_label
    metrics, curves = evaluate_open_set(y_true, y_pred, closed_pred, unknown_score, unknown_label)
    metrics["threshold"] = float(threshold)
    metrics["best_epoch"] = int(best_epoch)
    metrics["parameter_count"] = int(parameter_count)
    metrics["training_seconds"] = float(time.time() - started)

    true_names = np.concatenate([np.asarray(test_known.tx_names), np.asarray(test_unknown.tx_names)])
    rx_ids = np.concatenate([np.asarray(test_known.rx_ids), np.asarray(test_unknown.rx_ids)]).astype(np.int64)
    date_ids = np.concatenate([np.asarray(test_known.date_ids), np.asarray(test_unknown.date_ids)]).astype(np.int64)
    order = test_outputs["indices"].astype(np.int64)
    true_names, rx_ids, date_ids = true_names[order], rx_ids[order], date_ids[order]
    rejected = y_pred == unknown_label
    true_unknown = y_true == unknown_label
    cache_precision = float(true_unknown[rejected].mean()) if np.any(rejected) else 0.0
    cache_recall = float((rejected & true_unknown).sum() / max(int(true_unknown.sum()), 1))
    subdivision, cluster_labels, reduced = run_unknown_subdivision(
        embeddings=test_outputs["embedding"][rejected],
        cache_true_names=true_names[rejected],
        cache_is_true_unknown=true_unknown[rejected],
        total_test_unknown=int(true_unknown.sum()),
        unknown_cache_precision=cache_precision,
        unknown_cache_recall=cache_recall,
        seed=seed,
        config=config["subdivision"],
    )

    per_rx: dict[str, Any] = {}
    for rx_id in np.unique(rx_ids):
        mask = rx_ids == rx_id
        rx_metrics, _ = evaluate_open_set(y_true[mask], y_pred[mask], closed_pred[mask], unknown_score[mask], unknown_label)
        per_rx[manifest["rx_names"][int(rx_id)]] = rx_metrics
    metrics["worst_rx_known_accuracy"] = float(min(row["known_accuracy"] for row in per_rx.values()))
    metrics["worst_rx_unknown_recall"] = float(min(row["unknown_recall"] for row in per_rx.values()))
    metrics["worst_rx_auroc"] = float(min(row["auroc"] for row in per_rx.values()))

    agent_diagnostics: dict[str, Any] = {}
    for agent, name in enumerate(AGENT_NAMES):
        for stage in ["pre", "post"]:
            logits = test_outputs[f"{stage}_logits"][:, agent]
            prediction = logits.argmax(1)
            score = 1.0 - logits.softmax(1).max(1) if hasattr(logits, "softmax") else None
            if score is None:
                exp = np.exp(logits - logits.max(1, keepdims=True))
                probability = exp / exp.sum(1, keepdims=True)
                score = 1.0 - probability.max(1)
            from sklearn.metrics import roc_auc_score

            agent_diagnostics[f"{stage}_{name}"] = {
                "known_closed_accuracy": float((prediction[~true_unknown] == y_true[~true_unknown]).mean()),
                "unknown_auroc_one_minus_msp": float(roc_auc_score(true_unknown.astype(int), score)),
            }

    router = {
        "agent_names": AGENT_NAMES,
        "known_mean_weights": test_outputs["fusion_weights"][~true_unknown].mean(0).tolist(),
        "unknown_mean_weights": test_outputs["fusion_weights"][true_unknown].mean(0).tolist(),
        "known_mean_reliability": test_outputs["reliability"][~true_unknown].mean(0).tolist(),
        "unknown_mean_reliability": test_outputs["reliability"][true_unknown].mean(0).tolist(),
        "mean_adjacency_known": test_outputs["adjacency"][~true_unknown].mean(0).tolist(),
        "mean_adjacency_unknown": test_outputs["adjacency"][true_unknown].mean(0).tolist(),
    }

    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "per_receiver_metrics.json", per_rx)
    save_json(run_dir / "agent_diagnostics.json", agent_diagnostics)
    save_json(run_dir / "router_summary.json", router)
    save_json(run_dir / "subdivision_metrics.json", subdivision)
    np.savez_compressed(
        run_dir / "model_outputs.npz",
        fusion_weights=test_outputs["fusion_weights"],
        reliability=test_outputs["reliability"],
        adjacency=test_outputs["adjacency"],
        fused_embedding=test_outputs["embedding"],
        unknown_score=unknown_score,
    )
    predictions = pd.DataFrame(
        {
            "sample_index": order,
            "true_tx": true_names,
            "y_true": y_true,
            "y_pred": y_pred,
            "closed_set_pred": closed_pred,
            "unknown_score": unknown_score,
            "rx": [manifest["rx_names"][int(value)] for value in rx_ids],
            "date": [manifest["date_names"][int(value)] for value in date_ids],
            **{f"weight_{name}": test_outputs["fusion_weights"][:, index] for index, name in enumerate(AGENT_NAMES)},
            **{f"reliability_{name}": test_outputs["reliability"][:, index] for index, name in enumerate(AGENT_NAMES)},
        }
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    cache_frame = predictions.loc[rejected].copy()
    cache_frame["cluster"] = cluster_labels
    cache_frame.to_csv(run_dir / "unknown_subdivision_assignments.csv", index=False)
    np.save(run_dir / "unknown_subdivision_features.npy", reduced, allow_pickle=False)
    full_confusion = confusion_matrix(y_true, y_pred, labels=list(range(unknown_label + 1)))
    np.savetxt(run_dir / "open_set_confusion_matrix.csv", full_confusion, delimiter=",", fmt="%d")
    _save_plots(run_dir, history, y_true, y_pred, unknown_score, unknown_label, curves, test_outputs["fusion_weights"])
    _write_report(run_dir, config, metrics, subdivision, router)
    save_json(run_dir / "COMPLETE.json", {"status": "complete", "best_epoch": best_epoch, "metrics": metrics})
    print(f"[complete] {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    run_experiment(args.config, force_prepare=args.force_prepare)


if __name__ == "__main__":
    main()

