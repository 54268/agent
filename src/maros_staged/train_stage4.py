"""Stage-4 reliability routing over isolated and communicated agent systems."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .boundary_agent import make_prototypes
from .communication import MultiAgentCommunicationModel
from .evidence import EmpiricalCdfCalibrator, collect_evidence
from .metrics_osr import detection_metrics, oscr, threshold_decision
from .pseudo_unknown import AGENTS, DisagreementGuidedPseudoUnknownAgent, PseudoUnknownConfig
from .router import ReliabilityRouter, routing_observation
from .train_stage2 import _embeddings, _load_splits, _load_stage1, extract_agent_features
from .train_stage3 import EvidenceStandardizer, communication_inputs


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _load_communication_model(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = MultiAgentCommunicationModel(
        payload["input_dims"], int(payload["num_classes"]), evidence_dim=4,
        message_dim=int(cfg.get("message_dim", 64)),
        hidden_dim=int(cfg.get("communication_hidden_dim", 128)),
        dropout=float(cfg.get("communication_dropout", 0.10)),
        update_mode=str(cfg.get("communication_update_mode", "gru")),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


@torch.no_grad()
def _precompute(no_model, comm_model, z, evidence, labels, device, batch_size=2048):
    arrays = {key: [] for key in ("observation", "no_class", "comm_class",
                                   "no_unknown", "comm_unknown", "labels")}
    dataset = TensorDataset(
        torch.from_numpy(z["identity"].astype(np.float32)),
        torch.from_numpy(z["prototype"].astype(np.float32)),
        torch.from_numpy(z["reconstruction"].astype(np.float32)),
        torch.from_numpy(evidence.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    for zi, zp, zr, local, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        embeddings = {"identity": zi.to(device), "prototype": zp.to(device),
                      "reconstruction": zr.to(device)}
        local = local.to(device)
        no = no_model(embeddings, local, communicate=False)
        comm = comm_model(embeddings, local, communicate=True)
        observation = routing_observation(no, comm, local)
        arrays["observation"].append(observation.cpu().numpy())
        arrays["no_class"].append(no["fused_logits"].cpu().numpy())
        arrays["comm_class"].append(comm["fused_logits"].cpu().numpy())
        arrays["no_unknown"].append(no["unknown_logit"].cpu().numpy())
        arrays["comm_unknown"].append(comm["unknown_logit"].cpu().numpy())
        arrays["labels"].append(y.numpy())
    return {key: np.concatenate(value) for key, value in arrays.items()}


def _routing_dataset(data, unknown_target):
    return TensorDataset(
        torch.from_numpy(data["observation"].astype(np.float32)),
        torch.from_numpy(data["no_class"].astype(np.float32)),
        torch.from_numpy(data["comm_class"].astype(np.float32)),
        torch.from_numpy(data["no_unknown"].astype(np.float32)),
        torch.from_numpy(data["comm_unknown"].astype(np.float32)),
        torch.from_numpy(np.asarray(unknown_target, dtype=np.float32)),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )


def _router_forward(router, batch, device, override=None):
    observation, no_class, comm_class, no_unknown, comm_unknown, target, labels = [x.to(device) for x in batch]
    no = {"fused_logits": no_class, "unknown_logit": no_unknown}
    comm = {"fused_logits": comm_class, "unknown_logit": comm_unknown}
    out = router(observation, no, comm, override=override)
    return out, target, labels


@torch.no_grad()
def _predict_router(router, data, device, override=None, batch_size=4096):
    ds = _routing_dataset(data, np.zeros(len(data["labels"]), dtype=np.float32))
    scores, pred, kw, uw = [], [], [], []
    router.eval()
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out, _, _ = _router_forward(router, batch, device, override=override)
        scores.append(torch.sigmoid(out["unknown_logit"]).cpu().numpy())
        pred.append(out["known_logits"].argmax(dim=1).cpu().numpy())
        kw.append(out["known_weights"].cpu().numpy())
        uw.append(out["unknown_weights"].cpu().numpy())
    return {"score": np.concatenate(scores), "pred": np.concatenate(pred),
            "known_weights": np.concatenate(kw), "unknown_weights": np.concatenate(uw)}


def _train_router(mode, train_known, train_pseudo, pseudo_train_mask,
                  val_known, val_pseudo, pseudo_val_mask, cfg, device):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    router = ReliabilityRouter(
        train_known["observation"].shape[1], mode=mode,
        hidden_dim=int(cfg.get("router_hidden_dim", 64)),
    ).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=float(cfg.get("router_lr", 5e-4)),
                            weight_decay=float(cfg.get("router_weight_decay", 1e-4)))
    epochs = int(cfg.get("router_epochs", 35))
    batch_size = int(cfg.get("router_batch_size", 512))
    class_weight = float(cfg.get("router_class_weight", 0.7))
    balance_weight = float(cfg.get("router_balance_weight", 0.02))
    entropy_floor = float(cfg.get("router_entropy_floor", 0.35))
    entropy_weight = float(cfg.get("router_entropy_weight", 0.01))
    rng = np.random.default_rng(seed + {"global": 0, "single": 1, "dual": 2}[mode] * 1000)
    ptr = np.where(pseudo_train_mask)[0]
    pval = np.where(pseudo_val_mask)[0]
    history, best, best_state, best_epoch = [], -np.inf, None, -1

    for epoch in range(1, epochs + 1):
        n = max(len(train_known["labels"]), len(ptr))
        ik = rng.choice(len(train_known["labels"]), n, replace=len(train_known["labels"]) < n)
        ip = rng.choice(ptr, n, replace=len(ptr) < n)
        merged = {}
        for key in train_known:
            merged[key] = np.concatenate([train_known[key][ik], train_pseudo[key][ip]])
        unknown = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
        order = rng.permutation(2 * n)
        merged = {key: value[order] for key, value in merged.items()}
        ds = _routing_dataset(merged, unknown[order])
        router.train(); running = 0.0
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            out, target, labels = _router_forward(router, batch, device)
            bce = F.binary_cross_entropy_with_logits(out["unknown_logit"], target)
            known = labels >= 0
            ce = F.cross_entropy(out["known_logits"][known], labels[known])
            weights = torch.cat([out["known_weights"], out["unknown_weights"]], dim=0)
            balance = ((weights.mean(dim=0) - 0.5) ** 2).sum()
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
            entropy_penalty = F.relu(entropy_floor - entropy)
            loss = bce + class_weight * ce + balance_weight * balance + entropy_weight * entropy_penalty
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0); opt.step()
            running += float(loss.detach()) * len(labels)

        known_pred = _predict_router(router, val_known, device)
        pseudo_subset = {key: value[pval] for key, value in val_pseudo.items()}
        pseudo_pred = _predict_router(router, pseudo_subset, device)
        threshold = float(np.quantile(known_pred["score"], 0.95))
        recall = float((pseudo_pred["score"] >= threshold).mean())
        auc = float(roc_auc_score(
            np.concatenate([np.zeros(len(known_pred["score"])), np.ones(len(pseudo_pred["score"]))]),
            np.concatenate([known_pred["score"], pseudo_pred["score"]])))
        accuracy = float((known_pred["pred"] == val_known["labels"]).mean())
        selection = auc + 0.5 * recall + 0.2 * accuracy
        row = {"epoch": epoch, "loss": running / len(ds), "val_pseudo_auroc": auc,
               "val_pseudo_recall": recall, "val_known_accuracy": accuracy}
        history.append(row)
        if selection > best:
            best, best_epoch = selection, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  router-{mode} ep{epoch:03d} loss={row['loss']:.4f} "
                  f"pseudo_auc={auc:.4f} pseudo_rec={recall:.4f} known_acc={accuracy:.4f}")
    router.load_state_dict(best_state)
    return router, history, best_epoch


def _metrics(val_pred, open_pred, y):
    u = EmpiricalCdfCalibrator(val_pred["score"]).score(open_pred["score"])
    is_unknown = (y == -1).astype(np.int32)
    metric = detection_metrics(is_unknown, u)
    metric["oscr"] = oscr(is_unknown, open_pred["pred"], y, 1.0 - u)
    metric.update(threshold_decision(is_unknown, y, open_pred["pred"], u, 0.95))
    return metric, u


def train_stage4(config: Dict) -> Dict:
    seed = int(config.get("seed", 42)); set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(config["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); splits = _load_splits(config); bs = int(config.get("extract_batch_size", 512))
    train_loader = DataLoader(splits.train, batch_size=bs, shuffle=False)
    val_loader = DataLoader(splits.val, batch_size=bs, shuffle=False)
    open_loader = DataLoader(splits.open_test, batch_size=bs, shuffle=False)
    stage1 = _load_stage1(config["stage1_checkpoint"], device)
    no_model, no_payload = _load_communication_model(config["no_communication_checkpoint"], device)
    comm_model, comm_payload = _load_communication_model(config["communication_checkpoint"], device)
    train_f = extract_agent_features(stage1, train_loader, device)
    val_f = extract_agent_features(stage1, val_loader, device)
    open_f = extract_agent_features(stage1, open_loader, device)
    prototypes = comm_payload["prototypes"]
    scaler = EvidenceStandardizer(); scaler.mean = comm_payload["evidence_mean"]; scaler.std = comm_payload["evidence_std"]
    z_train, e_train = communication_inputs(_embeddings(train_f), prototypes)
    z_val, e_val = communication_inputs(_embeddings(val_f), prototypes)
    z_open, e_open = communication_inputs(_embeddings(open_f), prototypes)
    e_train, e_val, e_open = scaler.transform(e_train), scaler.transform(e_val), scaler.transform(e_open)

    pug = DisagreementGuidedPseudoUnknownAgent(PseudoUnknownConfig(
        **config.get("pseudo_unknown", {}), seed=seed))
    pseudo = pug.generate(_embeddings(train_f), train_f["y"], train_f["raw_evidence"])
    eligible = pseudo["in_shell"]
    if int(eligible.sum()) < splits.num_known * 10:
        eligible = np.ones(len(pseudo["source_indices"]), dtype=bool)
    z_pseudo_raw = {name: pseudo[f"z_{name}"][eligible] for name in AGENTS}
    sources = pseudo["source_indices"][eligible]
    z_pseudo, e_pseudo = communication_inputs(z_pseudo_raw, prototypes)
    e_pseudo = scaler.transform(e_pseudo)
    rng = np.random.default_rng(seed + 991); unique = np.unique(sources); rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * float(config.get("pseudo_val_ratio", 0.2)))))
    val_sources = set(unique[:n_val].tolist())
    pseudo_val_mask = np.asarray([int(source) in val_sources for source in sources])
    pseudo_train_mask = ~pseudo_val_mask

    print(f"[stage4] precomputing frozen no-communication/communication states on {device}")
    train_data = _precompute(no_model, comm_model, z_train, e_train, train_f["y"], device)
    val_data = _precompute(no_model, comm_model, z_val, e_val, val_f["y"], device)
    open_data = _precompute(no_model, comm_model, z_open, e_open, open_f["y"], device)
    pseudo_labels = np.full(len(sources), -1, dtype=np.int64)
    pseudo_data = _precompute(no_model, comm_model, z_pseudo, e_pseudo, pseudo_labels, device)

    routers, histories, epochs = {}, {}, {}
    for mode in ("global", "single", "dual"):
        routers[mode], histories[mode], epochs[mode] = _train_router(
            mode, train_data, pseudo_data, pseudo_train_mask, val_data, pseudo_data,
            pseudo_val_mask, config, device)

    metrics, predictions, u_scores = {}, {}, {}
    for mode, router in routers.items():
        v = _predict_router(router, val_data, device); o = _predict_router(router, open_data, device)
        metrics[mode], u_scores[mode] = _metrics(v, o, open_f["y"])
        predictions[mode] = {"val": v, "open": o}
    dual = routers["dual"]
    for name, override in (("dual_uniform", "uniform"), ("dual_shuffled", "shuffle")):
        v = _predict_router(dual, val_data, device, override=override)
        o = _predict_router(dual, open_data, device, override=override)
        metrics[name], u_scores[name] = _metrics(v, o, open_f["y"])
        predictions[name] = {"val": v, "open": o}

    # Fixed source baselines under the identical operating-point calibration.
    for name, source in (("no_communication", "no"), ("communication", "comm")):
        val_pred = {"score": 1.0 / (1.0 + np.exp(-np.clip(val_data[f"{source}_unknown"], -30, 30))),
                    "pred": val_data[f"{source}_class"].argmax(axis=1)}
        open_pred = {"score": 1.0 / (1.0 + np.exp(-np.clip(open_data[f"{source}_unknown"], -30, 30))),
                     "pred": open_data[f"{source}_class"].argmax(axis=1)}
        metrics[name], u_scores[name] = _metrics(val_pred, open_pred, open_f["y"])

    val_ev, calibrators = collect_evidence(stage1, val_loader, device, top_k=stage1.config.recon_top_k)
    open_ev, _ = collect_evidence(stage1, open_loader, device, top_k=stage1.config.recon_top_k,
                                  calibrators=calibrators)
    dual_val_u = EmpiricalCdfCalibrator(predictions["dual"]["val"]["score"]).score(
        predictions["dual"]["val"]["score"])
    raw_val = np.stack([val_ev["u_identity"], val_ev["u_prototype"],
                        val_ev["u_reconstruction"], dual_val_u], axis=1).mean(axis=1)
    raw_open = np.stack([open_ev["u_identity"], open_ev["u_prototype"],
                         open_ev["u_reconstruction"], u_scores["dual"]], axis=1).mean(axis=1)
    fused_u = EmpiricalCdfCalibrator(raw_val).score(raw_open)
    is_unknown = (open_f["y"] == -1).astype(np.int32)
    fused_pred = predictions["dual"]["open"]["pred"]
    fused_metric = detection_metrics(is_unknown, fused_u)
    fused_metric["oscr"] = oscr(is_unknown, fused_pred, open_f["y"], 1.0 - fused_u)
    fused_metric.update(threshold_decision(is_unknown, open_f["y"], fused_pred, fused_u, 0.95))
    metrics["dual_plus_three_mean"] = fused_metric; u_scores["dual_plus_three_mean"] = fused_u

    known = open_f["y"] != -1; dual_open = predictions["dual"]["open"]
    route_summary = {
        "known_known_weights": dual_open["known_weights"][known].mean(axis=0).tolist(),
        "unknown_known_weights": dual_open["known_weights"][~known].mean(axis=0).tolist(),
        "known_unknown_weights": dual_open["unknown_weights"][known].mean(axis=0).tolist(),
        "unknown_unknown_weights": dual_open["unknown_weights"][~known].mean(axis=0).tolist(),
    }
    for mode, router in routers.items():
        torch.save({"state_dict": router.state_dict(), "mode": mode,
                    "observation_dim": train_data["observation"].shape[1], "config": config,
                    "best_epoch": epochs[mode]}, out_dir / f"router_{mode}.pt")
    (out_dir / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    summary = {"dataset": config["dataset"], "seed": seed, "metrics": metrics,
               "route_summary": route_summary, "best_epoch": epochs,
               "elapsed_seconds": time.time() - t0,
               "protocol": {"real_unknown_used_for_training": False,
                            "real_unknown_used_for_threshold": False}}
    (out_dir / "stage4_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    np.savez_compressed(out_dir / "stage4_scores.npz", y=open_f["y"],
                        **{f"u_{key}": value for key, value in u_scores.items()})
    _write_report(summary, out_dir / "report_stage4.md")
    print(f"[stage4] completed in {summary['elapsed_seconds']:.1f}s")
    return summary


def _write_report(summary, path):
    names = {"no_communication": "固定无通信", "communication": "固定通信",
             "global": "全局学习权重", "single": "样本级单路由", "dual": "样本级双路由",
             "dual_uniform": "双路由·强制均匀", "dual_shuffled": "双路由·打乱权重",
             "dual_plus_three_mean": "双路由 + 三基础证据均值"}
    order = list(names)
    lines = [f"# Stage-4 Reliability Router — {summary['dataset']}", "",
             "Router 只能在无通信/通信系统输出之间做凸组合，不能绕过 Agent 自行预测。", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in order:
        m = summary["metrics"][key]
        lines.append(f"| {names[key]} | {m['auroc']:.4f} | {m['oscr']:.4f} | "
                     f"{m['known_accuracy']:.4f} | {m['unknown_recall']:.4f} | {m['h_score']:.4f} |")
    r = summary["route_summary"]
    lines += ["", "## 双路由平均权重 [无通信, 通信]", "",
              f"- Known 样本的 Known 路由：{r['known_known_weights']}",
              f"- Unknown 样本的 Known 路由：{r['unknown_known_weights']}",
              f"- Known 样本的 Unknown 路由：{r['known_unknown_weights']}",
              f"- Unknown 样本的 Unknown 路由：{r['unknown_unknown_weights']}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

