"""Stage-3 explicit Communication training and counterfactual evaluation."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .boundary_agent import make_prototypes
from .communication import MultiAgentCommunicationModel
from .evidence import EmpiricalCdfCalibrator, collect_evidence
from .metrics_osr import detection_metrics, oscr, threshold_decision
from .pseudo_unknown import (
    AGENTS,
    DisagreementGuidedPseudoUnknownAgent,
    PseudoUnknownConfig,
    normalise_agents,
)
from .train_stage2 import (
    _embeddings,
    _load_splits,
    _load_stage1,
    extract_agent_features,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def communication_inputs(embeddings: Dict[str, np.ndarray],
                         prototypes: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    z = normalise_agents(embeddings)
    local_evidence = []
    for name in AGENTS:
        distance = np.maximum(1.0 - z[name] @ prototypes[name].T, 0.0)
        order = np.argsort(distance, axis=1)
        d1 = distance[np.arange(len(distance)), order[:, 0]]
        d2 = distance[np.arange(len(distance)), order[:, 1]]
        probability = np.exp(-distance - (-distance).max(axis=1, keepdims=True))
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
        entropy /= np.log(probability.shape[1])
        local_evidence.append(np.stack([
            d1, d2 - d1, d1 / np.maximum(d2, 1e-6), entropy,
        ], axis=1))
    return z, np.stack(local_evidence, axis=1).astype(np.float32)


class EvidenceStandardizer:
    def fit(self, evidence: np.ndarray) -> "EvidenceStandardizer":
        self.mean = evidence.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = np.maximum(evidence.std(axis=0, keepdims=True), 1e-5).astype(np.float32)
        return self

    def transform(self, evidence: np.ndarray) -> np.ndarray:
        return ((evidence - self.mean) / self.std).astype(np.float32)


def _dataset(z: Dict[str, np.ndarray], evidence: np.ndarray,
             is_unknown: np.ndarray, labels: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(z["identity"].astype(np.float32)),
        torch.from_numpy(z["prototype"].astype(np.float32)),
        torch.from_numpy(z["reconstruction"].astype(np.float32)),
        torch.from_numpy(evidence.astype(np.float32)),
        torch.from_numpy(is_unknown.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )


def _forward_batch(model, batch, device, communicate=True, shuffle=False,
                   drop_sender=None):
    zi, zp, zr, evidence, unknown, labels = [item.to(device) for item in batch]
    embeddings = {"identity": zi, "prototype": zp, "reconstruction": zr}
    output = model(embeddings, evidence, communicate=communicate,
                   shuffle_messages=shuffle, drop_sender=drop_sender)
    return output, unknown, labels


@torch.no_grad()
def _predict(model, z, evidence, labels, device, communicate=True,
             shuffle=False, drop_sender=None, batch_size=2048):
    ds = _dataset(z, evidence, np.zeros(len(labels), dtype=np.float32), labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores, predictions, adjacency = [], [], []
    model.eval()
    for batch in loader:
        out, _, _ = _forward_batch(model, batch, device, communicate=communicate,
                                   shuffle=shuffle, drop_sender=drop_sender)
        scores.append(torch.sigmoid(out["unknown_logit"]).cpu().numpy())
        predictions.append(out["fused_logits"].argmax(dim=1).cpu().numpy())
        adjacency.append(out["adjacency"].cpu().numpy())
    return {"raw_score": np.concatenate(scores), "class_pred": np.concatenate(predictions),
            "adjacency": np.concatenate(adjacency)}


def _train_variant(name: str, communicate: bool, z_known, e_known, y_known,
                   z_pseudo, e_pseudo, pseudo_train_mask, z_val, e_val, y_val,
                   pseudo_val_mask, input_dims, num_classes, cfg, device,
                   init_state=None, teacher=None):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)  # identical initial parameters for the two controlled variants
    model = MultiAgentCommunicationModel(
        input_dims, num_classes, evidence_dim=e_known.shape[2],
        message_dim=int(cfg.get("message_dim", 64)),
        hidden_dim=int(cfg.get("communication_hidden_dim", 128)),
        dropout=float(cfg.get("communication_dropout", 0.10)),
        update_mode=str(cfg.get("communication_update_mode", "gru")),
    ).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    lr_key = "communication_finetune_lr" if communicate and init_state is not None else "communication_lr"
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get(lr_key, 8e-4)),
                            weight_decay=float(cfg.get("communication_weight_decay", 1e-4)))
    epochs = int(cfg.get("communication_epochs", 40))
    batch_size = int(cfg.get("communication_batch_size", 512))
    class_weight = float(cfg.get("communication_class_weight", 0.7))
    agent_aux_weight = float(cfg.get("agent_class_aux_weight", 0.15))
    gate_weight = float(cfg.get("gate_sparsity_weight", 0.001))
    distill_weight = float(cfg.get("known_distill_weight", 0.20)) if teacher is not None else 0.0
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    rng = np.random.default_rng(seed + (0 if communicate else 10000))
    history, best_score, best_state, best_epoch = [], -np.inf, None, -1

    pseudo_indices = np.where(pseudo_train_mask)[0]
    z_pseudo_train = {key: value[pseudo_indices] for key, value in z_pseudo.items()}
    e_pseudo_train = e_pseudo[pseudo_indices]
    pseudo_val_indices = np.where(pseudo_val_mask)[0]
    z_pseudo_val = {key: value[pseudo_val_indices] for key, value in z_pseudo.items()}
    e_pseudo_val = e_pseudo[pseudo_val_indices]
    pseudo_val_y = np.full(len(pseudo_val_indices), -1, dtype=np.int64)

    for epoch in range(1, epochs + 1):
        n = max(len(y_known), len(pseudo_indices))
        ik = rng.choice(len(y_known), size=n, replace=len(y_known) < n)
        ip = rng.choice(len(pseudo_indices), size=n, replace=len(pseudo_indices) < n)
        z_epoch = {key: np.concatenate([z_known[key][ik], z_pseudo_train[key][ip]])
                   for key in AGENTS}
        e_epoch = np.concatenate([e_known[ik], e_pseudo_train[ip]])
        u_epoch = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
        y_epoch = np.concatenate([y_known[ik], np.full(n, -1, dtype=np.int64)])
        order = rng.permutation(2 * n)
        z_epoch = {key: value[order] for key, value in z_epoch.items()}
        ds = _dataset(z_epoch, e_epoch[order], u_epoch[order], y_epoch[order])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        model.train()
        total_loss = 0.0
        for batch in loader:
            out, unknown, labels = _forward_batch(model, batch, device, communicate=communicate)
            boundary_loss = F.binary_cross_entropy_with_logits(out["unknown_logit"], unknown)
            known = labels >= 0
            fused_ce = F.cross_entropy(out["fused_logits"][known], labels[known])
            agent_ce = torch.stack([
                F.cross_entropy(out["agent_logits"][known, i], labels[known])
                for i in range(len(AGENTS))
            ]).mean()
            gate_penalty = out["adjacency"].mean() if communicate else out["unknown_logit"].new_zeros(())
            distill = out["unknown_logit"].new_zeros(())
            if teacher is not None:
                with torch.no_grad():
                    teacher_out, _, _ = _forward_batch(
                        teacher, batch, device, communicate=False
                    )
                distill = F.kl_div(
                    F.log_softmax(out["fused_logits"][known], dim=1),
                    F.softmax(teacher_out["fused_logits"][known], dim=1),
                    reduction="batchmean",
                )
            loss = (boundary_loss + class_weight * fused_ce + agent_aux_weight * agent_ce
                    + gate_weight * gate_penalty + distill_weight * distill)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach()) * len(labels)

        val_known = _predict(model, z_val, e_val, y_val, device, communicate=communicate)
        val_pseudo = _predict(model, z_pseudo_val, e_pseudo_val, pseudo_val_y, device,
                              communicate=communicate)
        threshold = float(np.quantile(val_known["raw_score"], 0.95))
        pseudo_recall = float((val_pseudo["raw_score"] >= threshold).mean())
        auc = float(roc_auc_score(
            np.concatenate([np.zeros(len(y_val)), np.ones(len(pseudo_val_y))]),
            np.concatenate([val_known["raw_score"], val_pseudo["raw_score"]]),
        ))
        class_acc = float((val_known["class_pred"] == y_val).mean())
        score = auc + 0.5 * pseudo_recall + 0.2 * class_acc
        row = {"epoch": epoch, "loss": total_loss / len(ds), "val_pseudo_auroc": auc,
               "val_pseudo_recall": pseudo_recall, "val_class_acc": class_acc,
               "known95_threshold": threshold}
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  {name} ep{epoch:03d} loss={row['loss']:.4f} pseudo_auc={auc:.4f} "
                  f"pseudo_rec={pseudo_recall:.4f} known_cls={class_acc:.4f}")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _calibrated_metrics(val_score, open_score, open_y, closed_pred):
    calibrator = EmpiricalCdfCalibrator(val_score)
    u_open = calibrator.score(open_score)
    is_unknown = (open_y == -1).astype(np.int32)
    result = detection_metrics(is_unknown, u_open)
    result["oscr"] = oscr(is_unknown, closed_pred, open_y, 1.0 - u_open)
    result.update(threshold_decision(is_unknown, open_y, closed_pred, u_open, 0.95))
    return result, u_open


def _fused_rule(val_components, open_components, open_y, closed_pred):
    val_score = np.asarray(val_components).mean(axis=0)
    open_score = np.asarray(open_components).mean(axis=0)
    return _calibrated_metrics(val_score, open_score, open_y, closed_pred)


def train_stage3(config: Dict) -> Dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    splits = _load_splits(config)
    bs = int(config.get("extract_batch_size", 512))
    train_loader = DataLoader(splits.train, batch_size=bs, shuffle=False)
    val_loader = DataLoader(splits.val, batch_size=bs, shuffle=False)
    open_loader = DataLoader(splits.open_test, batch_size=bs, shuffle=False)
    stage1 = _load_stage1(config["stage1_checkpoint"], device)
    print(f"[stage3] dataset={config['dataset']} device={device} extracting frozen agent states")
    train_f = extract_agent_features(stage1, train_loader, device)
    val_f = extract_agent_features(stage1, val_loader, device)
    open_f = extract_agent_features(stage1, open_loader, device)

    pug_cfg = PseudoUnknownConfig(**config.get("pseudo_unknown", {}), seed=seed)
    pseudo = DisagreementGuidedPseudoUnknownAgent(pug_cfg).generate(
        _embeddings(train_f), train_f["y"], train_f["raw_evidence"]
    )
    eligible = pseudo["in_shell"]
    if int(eligible.sum()) < splits.num_known * 10:
        eligible = np.ones(len(pseudo["source_indices"]), dtype=bool)
    pseudo_z_raw = {name: pseudo[f"z_{name}"][eligible] for name in AGENTS}
    pseudo_sources = pseudo["source_indices"][eligible]
    rng = np.random.default_rng(seed + 991)
    sources = np.unique(pseudo_sources)
    rng.shuffle(sources)
    n_val_source = max(1, int(round(len(sources) * float(config.get("pseudo_val_ratio", 0.2)))))
    val_source = set(sources[:n_val_source].tolist())
    pseudo_val_mask = np.asarray([int(source) in val_source for source in pseudo_sources])
    pseudo_train_mask = ~pseudo_val_mask

    prototypes = make_prototypes(_embeddings(train_f), train_f["y"], splits.num_known)
    z_train, e_train = communication_inputs(_embeddings(train_f), prototypes)
    z_val, e_val = communication_inputs(_embeddings(val_f), prototypes)
    z_open, e_open = communication_inputs(_embeddings(open_f), prototypes)
    z_pseudo, e_pseudo = communication_inputs(pseudo_z_raw, prototypes)
    evidence_scaler = EvidenceStandardizer().fit(e_train)
    e_train, e_val = evidence_scaler.transform(e_train), evidence_scaler.transform(e_val)
    e_open, e_pseudo = evidence_scaler.transform(e_open), evidence_scaler.transform(e_pseudo)
    input_dims = {name: z_train[name].shape[1] for name in AGENTS}
    print(f"[stage3] pseudo_used={len(e_pseudo)} sources={len(sources)} train/val="
          f"{int(pseudo_train_mask.sum())}/{int(pseudo_val_mask.sum())}")

    no_comm, hist_no, epoch_no = _train_variant(
        "no_comm", False, z_train, e_train, train_f["y"], z_pseudo, e_pseudo,
        pseudo_train_mask, z_val, e_val, val_f["y"], pseudo_val_mask,
        input_dims, splits.num_known, config, device,
    )
    warm_start = bool(config.get("communication_warm_start", False))
    comm, hist_comm, epoch_comm = _train_variant(
        "communication", True, z_train, e_train, train_f["y"], z_pseudo, e_pseudo,
        pseudo_train_mask, z_val, e_val, val_f["y"], pseudo_val_mask,
        input_dims, splits.num_known, config, device,
        init_state=({key: value.detach().cpu().clone()
                    for key, value in no_comm.state_dict().items()} if warm_start else None),
        teacher=(no_comm if warm_start else None),
    )

    modes = {
        "no_comm_trained": (no_comm, False, False, None),
        "communication": (comm, True, False, None),
        "messages_off": (comm, False, False, None),
        "messages_shuffled": (comm, True, True, None),
        "drop_identity_sender": (comm, True, False, 0),
        "drop_prototype_sender": (comm, True, False, 1),
        "drop_reconstruction_sender": (comm, True, False, 2),
    }
    predictions, metrics, u_by_mode = {}, {}, {}
    for name, (model, enabled, shuffled, drop_sender) in modes.items():
        val_pred = _predict(model, z_val, e_val, val_f["y"], device,
                            communicate=enabled, shuffle=shuffled, drop_sender=drop_sender)
        open_pred = _predict(model, z_open, e_open, open_f["y"], device,
                             communicate=enabled, shuffle=shuffled, drop_sender=drop_sender)
        metric, u_open = _calibrated_metrics(
            val_pred["raw_score"], open_pred["raw_score"], open_f["y"], open_pred["class_pred"]
        )
        predictions[name] = {"val": val_pred, "open": open_pred}
        metrics[name], u_by_mode[name] = metric, u_open

    val_evidence, calibrators = collect_evidence(stage1, val_loader, device,
                                                 top_k=stage1.config.recon_top_k)
    open_evidence, _ = collect_evidence(stage1, open_loader, device,
                                        top_k=stage1.config.recon_top_k,
                                        calibrators=calibrators)
    # Fairly calibrated fixed fusion using the normal Communication score as a
    # fourth item.  The full rule, not each component alone, is CDF calibrated.
    comm_val_cal = EmpiricalCdfCalibrator(predictions["communication"]["val"]["raw_score"])
    u_comm_val = comm_val_cal.score(predictions["communication"]["val"]["raw_score"])
    u_comm_open = comm_val_cal.score(predictions["communication"]["open"]["raw_score"])
    mean_metric, mean_u = _fused_rule(
        [val_evidence["u_identity"], val_evidence["u_prototype"],
         val_evidence["u_reconstruction"], u_comm_val],
        [open_evidence["u_identity"], open_evidence["u_prototype"],
         open_evidence["u_reconstruction"], u_comm_open],
        open_evidence["y"], predictions["communication"]["open"]["class_pred"],
    )
    max_val = np.stack([val_evidence["u_identity"], val_evidence["u_prototype"],
                        val_evidence["u_reconstruction"], u_comm_val], axis=1).max(axis=1)
    max_open = np.stack([open_evidence["u_identity"], open_evidence["u_prototype"],
                         open_evidence["u_reconstruction"], u_comm_open], axis=1).max(axis=1)
    max_metric, max_u = _calibrated_metrics(
        max_val, max_open, open_evidence["y"], predictions["communication"]["open"]["class_pred"]
    )
    metrics["communication_mean4"] = mean_metric
    metrics["communication_max4"] = max_metric
    u_by_mode["communication_mean4"], u_by_mode["communication_max4"] = mean_u, max_u

    normal = predictions["communication"]["open"]
    interventions = {}
    for name in ("messages_off", "messages_shuffled", "drop_identity_sender",
                 "drop_prototype_sender", "drop_reconstruction_sender"):
        changed = predictions[name]["open"]
        interventions[name] = {
            "class_prediction_flip_rate": float(
                (changed["class_pred"] != normal["class_pred"]).mean()),
            "unknown_score_mean_abs_change": float(
                np.abs(changed["raw_score"] - normal["raw_score"]).mean()),
            "auroc_delta": float(metrics[name]["auroc"] - metrics["communication"]["auroc"]),
            "oscr_delta": float(metrics[name]["oscr"] - metrics["communication"]["oscr"]),
        }
    known = open_f["y"] != -1
    adjacency = normal["adjacency"]
    adjacency_summary = {
        "known_mean": adjacency[known].mean(axis=0).tolist(),
        "unknown_mean": adjacency[~known].mean(axis=0).tolist(),
        "all_mean": adjacency.mean(axis=0).tolist(),
    }

    model_meta = {"input_dims": input_dims, "num_classes": splits.num_known,
                  "evidence_mean": evidence_scaler.mean,
                  "evidence_std": evidence_scaler.std, "prototypes": prototypes,
                  "config": config}
    torch.save({**model_meta, "state_dict": no_comm.state_dict(), "best_epoch": epoch_no},
               out_dir / "no_communication.pt")
    torch.save({**model_meta, "state_dict": comm.state_dict(), "best_epoch": epoch_comm},
               out_dir / "communication.pt")
    (out_dir / "history.json").write_text(
        json.dumps({"no_communication": hist_no, "communication": hist_comm}, indent=2),
        encoding="utf-8")
    summary = {
        "dataset": config["dataset"], "seed": seed,
        "best_epoch": {"no_communication": epoch_no, "communication": epoch_comm},
        "elapsed_seconds": time.time() - t0,
        "metrics": metrics,
        "interventions": interventions,
        "adjacency": adjacency_summary,
        "pseudo": {"num_used": int(len(e_pseudo)), "num_sources": int(len(sources))},
        "protocol": {"real_unknown_used_for_training": False,
                     "real_unknown_used_for_threshold": False,
                     "operating_point": "each complete rule CDF-calibrated on known validation"},
    }
    (out_dir / "stage3_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "stage3_scores.npz", y=open_f["y"],
                        **{f"u_{name}": score for name, score in u_by_mode.items()})
    _plot_adjacency(adjacency_summary, out_dir / "communication_adjacency.png")
    _write_report(summary, out_dir / "report_stage3.md")
    print(f"[stage3] completed in {summary['elapsed_seconds']:.1f}s")
    return summary


def _plot_adjacency(summary: Dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(axes, ("known_mean", "unknown_mean"), ("Known", "Unknown")):
        matrix = np.asarray(summary[key])
        image = ax.imshow(matrix, vmin=0, vmax=max(float(matrix.max()), 1e-4), cmap="viridis")
        ax.set_xticks(range(3), ["I", "P", "R"])
        ax.set_yticks(range(3), ["I", "P", "R"])
        ax.set_xlabel("sender"); ax.set_ylabel("receiver"); ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _write_report(summary: Dict, path: Path) -> None:
    names = {
        "no_comm_trained": "同构模型·无通信训练",
        "communication": "显式 Communication",
        "messages_off": "通信模型·推理关闭消息",
        "messages_shuffled": "通信模型·打乱消息内容",
        "drop_identity_sender": "删除 Identity 发送边",
        "drop_prototype_sender": "删除 Prototype 发送边",
        "drop_reconstruction_sender": "删除 Reconstruction 发送边",
        "communication_mean4": "Communication + 三证据均值",
        "communication_max4": "Communication + 三证据 max/OR",
    }
    order = list(names)
    lines = [f"# Stage-3 显式多智能体 Communication — {summary['dataset']}", "",
             "真实 Unknown 不参与训练、选轮、阈值或规则校准。每条完整规则均用 Known validation CDF "
             "统一到约 95% Known 接受率。", "",
             "| 规则 | AUROC↑ | OSCR↑ | Known Acc↑ | Unknown Recall↑ | H-score↑ |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in order:
        m = summary["metrics"][key]
        lines.append(f"| {names[key]} | {m['auroc']:.4f} | {m['oscr']:.4f} | "
                     f"{m['known_accuracy']:.4f} | {m['unknown_recall']:.4f} | {m['h_score']:.4f} |")
    lines += ["", "## 反事实消息验证", ""]
    for key, row in summary["interventions"].items():
        lines.append(f"- {names[key]}：类别翻转率={row['class_prediction_flip_rate']:.4f}，"
                     f"未知分数平均变化={row['unknown_score_mean_abs_change']:.4f}，"
                     f"ΔAUROC={row['auroc_delta']:+.4f}，ΔOSCR={row['oscr_delta']:+.4f}。")
    lines += ["", "消息矩阵图：`communication_adjacency.png`。接收者为行、发送者为列。"]
    path.write_text("\n".join(lines), encoding="utf-8")
