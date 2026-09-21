"""Training and evaluation primitives for Stage-5."""
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from maros_staged.evidence import ClassConditionalCdf, EmpiricalCdfCalibrator
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision

from .agents import RoleStructuredExperts, role_decorrelation_loss, supervised_contrastive_loss
from .calibration_agent import (CalibrationAgent, ClassConditionalThresholdAgent,
                                DecisionArbitratorAgent)
from .coordinator import (CommunicationDecisionCoordinator, EDGE_NAMES,
                          counterfactual_gate_loss, per_sample_task_loss)
from .openmax import OpenMaxEVT
from .pseudo_unknown import PseudoUnknownBatch


def set_seed(seed: int) -> None:
    # cuBLAS workspace configuration must be present before the first
    # deterministic CUDA GEMM.  This project uses a single-process loader, so
    # these switches make a repeated seed a meaningful reproducibility test.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@dataclass
class ExpertRecords:
    identity_state: np.ndarray
    geometry_state: np.ndarray
    identity_logits: np.ndarray
    geometry_logits: np.ndarray
    evidence: np.ndarray
    labels: np.ndarray
    geometry_view_pred: np.ndarray | None = None

    def subset(self, mask) -> "ExpertRecords":
        return ExpertRecords(**{
            key: (None if getattr(self, key) is None else getattr(self, key)[mask])
            for key in self.__dataclass_fields__
        })


class EvidenceStandardizer:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "EvidenceStandardizer":
        self.mean = np.asarray(x, dtype=np.float32).mean(axis=0)
        self.std = np.asarray(x, dtype=np.float32).std(axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("EvidenceStandardizer is not fitted")
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


def _expert_loss(model, inputs, outputs, labels, cfg):
    identity, geometry = outputs["identity"], outputs["geometry"]
    id_ce = F.cross_entropy(identity.class_logits, labels)
    geo_ce = F.cross_entropy(geometry.class_logits, labels)
    view_logits = geometry.evidence.get("view_logits")
    if view_logits is not None and view_logits.shape[1] > 1:
        expanded_labels = labels[:, None].expand(-1, view_logits.shape[1]).reshape(-1)
        view_aux = F.cross_entropy(view_logits.flatten(0, 1), expanded_labels)
    else:
        view_aux = geo_ce.new_zeros(())
    id_supcon = supervised_contrastive_loss(identity.state, labels,
                                             float(cfg.get("supcon_temperature", 0.1)))
    # Geometry distances are already squared Euclidean distances to normalized prototypes.
    distances = geometry.evidence["distances"]
    pos = distances.gather(1, labels[:, None]).squeeze(1)
    masked = distances.clone(); masked.scatter_(1, labels[:, None], float("inf"))
    neg = masked.min(dim=1).values
    compact = pos.mean()
    margin = F.relu(float(cfg.get("prototype_margin", 0.5)) + pos - neg).mean()
    diversity = role_decorrelation_loss(identity.state, geometry.state)
    reconstruction = F.mse_loss(model.reconstruct(outputs), F.avg_pool1d(inputs, kernel_size=4))
    loss = (id_ce + geo_ce
            + float(cfg.get("view_aux_weight", 0.25)) * view_aux
            + float(cfg.get("supcon_weight", 0.1)) * id_supcon
            + float(cfg.get("compact_weight", 0.1)) * compact
            + float(cfg.get("margin_weight", 0.1)) * margin
            + float(cfg.get("diversity_weight", 0.01)) * diversity
            + float(cfg.get("reconstruction_weight", 0.05)) * reconstruction)
    return loss, {"id_ce": float(id_ce.detach()), "geo_ce": float(geo_ce.detach()),
                  "view_aux": float(view_aux.detach()),
                  "supcon": float(id_supcon.detach()), "compact": float(compact.detach()),
                  "margin": float(margin.detach()), "diversity": float(diversity.detach()),
                  "reconstruction": float(reconstruction.detach())}


@torch.no_grad()
def _expert_accuracy(model, dataset, device, batch_size=512):
    model.eval(); correct_i = correct_g = total = 0
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct_i += int((out["identity"].class_logits.argmax(1) == y).sum())
        correct_g += int((out["geometry"].class_logits.argmax(1) == y).sum())
        total += len(y)
    return correct_i / max(total, 1), correct_g / max(total, 1)


def train_experts(train_set, val_set, num_classes: int, cfg: Dict,
                  device: torch.device, seed: int):
    set_seed(seed)
    model = RoleStructuredExperts(
        num_classes,
        state_dim=int(cfg.get("state_dim", 64)),
        stem_channels=int(cfg.get("stem_channels", 64)),
        message_dim=int(cfg.get("message_dim", 32)),
        prototype_temperature=float(cfg.get("prototype_temperature", 0.15)),
        geometry_view=str(cfg.get("geometry_view", "spectral")),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("expert_lr", 1e-3)),
                                  weight_decay=float(cfg.get("expert_weight_decay", 1e-4)))
    epochs = int(cfg.get("expert_epochs", 30)); batch_size = int(cfg.get("batch_size", 256))
    best, best_state, best_epoch, history = -np.inf, None, -1, []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train(); totals = {key: 0.0 for key in ("loss", "id_ce", "geo_ce", "view_aux", "supcon",
                                                      "compact", "margin", "diversity",
                                                      "reconstruction")}; count = 0
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss, pieces = _expert_loss(model, x, out, y, cfg)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            totals["loss"] += float(loss.detach()) * len(y)
            for key, value in pieces.items(): totals[key] += value * len(y)
            count += len(y)
        id_acc, geo_acc = _expert_accuracy(model, val_set, device, batch_size)
        row = {key: value / max(count, 1) for key, value in totals.items()}
        row.update({"epoch": epoch, "identity_val_accuracy": id_acc,
                    "geometry_val_accuracy": geo_acc})
        history.append(row); score = 0.5 * (id_acc + geo_acc)
        if score > best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    experts ep{epoch:03d} loss={row['loss']:.4f} "
                  f"id_acc={id_acc:.4f} geo_acc={geo_acc:.4f}")
    model.load_state_dict(best_state)
    # Replace learned prototype directions by exact training means.
    identity_states, geometry_states, geometry_view_states, labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in DataLoader(train_set, batch_size=batch_size, shuffle=False):
            outputs = model(x.to(device)); identity, geometry = outputs["identity"], outputs["geometry"]
            identity_states.append(identity.state)
            geometry_states.append(geometry.state)
            geometry_view_states.append(geometry.evidence["view_states"])
            labels.append(y.to(device))
    all_labels = torch.cat(labels)
    model.identity.set_empirical_prototypes(torch.cat(identity_states), all_labels)
    model.geometry.set_empirical_prototypes(
        torch.cat(geometry_states), all_labels, torch.cat(geometry_view_states))
    return model, history, best_epoch


@torch.no_grad()
def refresh_empirical_prototypes(model: RoleStructuredExperts, train_set,
                                 device: torch.device, batch_size: int = 512) -> None:
    """Rebuild all local prototype memories without changing encoder weights."""
    identity_states, geometry_states, view_states, labels = [], [], [], []
    model.eval()
    for x, y in DataLoader(train_set, batch_size=batch_size, shuffle=False):
        outputs = model(x.to(device)); identity, geometry = outputs["identity"], outputs["geometry"]
        identity_states.append(identity.state); geometry_states.append(geometry.state)
        view_states.append(geometry.evidence["view_states"]); labels.append(y.to(device))
    all_labels = torch.cat(labels)
    model.identity.set_empirical_prototypes(torch.cat(identity_states), all_labels)
    model.geometry.set_empirical_prototypes(
        torch.cat(geometry_states), all_labels, torch.cat(view_states))


@torch.no_grad()
def extract_records(model: RoleStructuredExperts, dataset, device: torch.device,
                    openmax: OpenMaxEVT | None = None, batch_size: int = 1024) -> ExpertRecords:
    rows = {key: [] for key in ("identity_state", "geometry_state", "identity_logits",
                                 "geometry_logits", "evidence", "labels",
                                 "geometry_view_pred")}
    model.eval()
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        out = model(x.to(device)); i, g = out["identity"], out["geometry"]
        base = torch.stack([
            1.0 - i.evidence["confidence"], i.evidence["entropy"], i.evidence["energy"],
            g.evidence["d1"], -g.evidence["distance_margin"],
            1.0 - g.evidence["confidence"],
        ], dim=1)
        local = torch.cat([
            i.evidence["d1"][:, None], -i.evidence["distance_margin"][:, None],
            (i.evidence["pred"] != g.evidence["pred"]).float()[:, None],
            g.evidence["view_d1"], -g.evidence["view_distance_margin"],
        ], dim=1)
        rows["identity_state"].append(i.state.cpu().numpy())
        rows["geometry_state"].append(g.state.cpu().numpy())
        rows["identity_logits"].append(i.class_logits.cpu().numpy())
        rows["geometry_logits"].append(g.class_logits.cpu().numpy())
        rows["geometry_view_pred"].append(
            g.evidence["view_distances"].argmin(dim=-1).cpu().numpy())
        rows["evidence"].append(torch.cat([base, local], dim=1).cpu().numpy())
        rows["labels"].append(y.numpy())
    data = {key: np.concatenate(value) for key, value in rows.items()}
    geometry_prob = torch.softmax(torch.from_numpy(data["geometry_logits"]), dim=1).numpy()
    evt = (openmax.score(geometry_prob) if openmax is not None
           else np.zeros(len(data["labels"]), dtype=np.float32))
    # Preserve the legacy OpenMax position at column 6.  Extended local
    # evidence follows it: Identity prototype (d1, -margin), disagreement,
    # per-view d1 values, then per-view negative margins.
    data["evidence"] = np.column_stack([
        data["evidence"][:, :6], evt, data["evidence"][:, 6:]]).astype(np.float32)
    return ExpertRecords(**data)


@torch.no_grad()
def extract_geometry_view_weights(model: RoleStructuredExperts, dataset,
                                  device: torch.device,
                                  batch_size: int = 1024) -> np.ndarray:
    """Return the Geometry Agent's sample-level sensor actions in data order."""
    model.eval(); rows = []
    for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        geometry = model.geometry(x.to(device))
        rows.append(geometry.evidence["view_weights"].cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def fit_openmax(model, train_set, device, alpha_rank=3, tail_size=25) -> OpenMaxEVT:
    records = extract_records(model, train_set, device, openmax=None)
    activations = torch.softmax(torch.from_numpy(records.geometry_logits), dim=1).numpy()
    return OpenMaxEVT(alpha_rank, tail_size).fit(
        activations, records.labels, records.geometry_logits.argmax(axis=1))


@torch.no_grad()
def pseudo_to_records(model: RoleStructuredExperts, pseudo: PseudoUnknownBatch,
                      openmax: OpenMaxEVT, device: torch.device) -> ExpertRecords:
    zi = torch.from_numpy(pseudo.identity_state).to(device)
    zg = torch.from_numpy(pseudo.geometry_state).to(device)
    # Generated states must be freshly evaluated by both agents.  This is a
    # real state-level forward path, not a copy of the source Known report.
    identity = model.identity.forward_from_state(zi)
    geometry = model.geometry.forward_from_state(zg)
    identity_logits, geometry_logits = identity.class_logits, geometry.class_logits
    ip = F.softmax(identity_logits, dim=1); gp = F.softmax(geometry_logits, dim=1)
    iconf = identity.evidence["confidence"]
    ientropy = identity.evidence["entropy"]
    ienergy = identity.evidence["energy"]
    near = torch.stack([geometry.evidence["d1"], geometry.evidence["d2"]], dim=1)
    gconf = geometry.evidence["confidence"]
    identity_near = torch.stack(
        [identity.evidence["d1"], identity.evidence["d2"]], dim=1)
    disagreement = (identity_logits.argmax(1) != geometry_logits.argmax(1)).float()
    base = torch.stack([1.0 - iconf, ientropy, ienergy, near[:, 0],
                        -(near[:, 1] - near[:, 0]), 1.0 - gconf], dim=1)
    gl = geometry_logits.cpu().numpy()
    geometry_prob = F.softmax(geometry_logits, dim=1).cpu().numpy()
    # PUG proposals live in the fused specialist state spaces, so sensor-local
    # Geometry evidence is conservatively represented by the fused distance.
    # Real Known/LCO-heldout samples retain their true per-sensor evidence.
    view_count = model.geometry.view_prototypes.shape[0]
    local = torch.cat([
        identity_near[:, :1], -(identity_near[:, 1:2] - identity_near[:, :1]),
        disagreement[:, None], near[:, :1].repeat(1, view_count),
        -(near[:, 1:2] - near[:, :1]).repeat(1, view_count),
    ], dim=1)
    evidence = np.column_stack([
        base.cpu().numpy(), openmax.score(geometry_prob), local.cpu().numpy()
    ]).astype(np.float32)
    geometry_view_pred = geometry_logits.argmax(1).cpu().numpy()[:, None].repeat(
        model.geometry.view_prototypes.shape[0], axis=1)
    return ExpertRecords(zi.cpu().numpy(), zg.cpu().numpy(),
                         identity_logits.cpu().numpy(), gl, evidence,
                         np.full(len(zi), -1, dtype=np.int64), geometry_view_pred)


def _tensor_dataset(records: ExpertRecords, evidence: np.ndarray,
                    unknown: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(records.identity_state.astype(np.float32)),
        torch.from_numpy(records.geometry_state.astype(np.float32)),
        torch.from_numpy(records.identity_logits.astype(np.float32)),
        torch.from_numpy(records.geometry_logits.astype(np.float32)),
        torch.from_numpy(evidence.astype(np.float32)),
        torch.from_numpy(unknown.astype(np.float32)),
        torch.from_numpy(records.labels.astype(np.int64)),
    )


def _coordinator_inputs(batch, device):
    zi, zg, li, lg, evidence, unknown, labels = [item.to(device) for item in batch]
    inputs = {"identity_state": zi, "geometry_state": zg,
              "identity_logits": li, "geometry_logits": lg, "evidence": evidence}
    return inputs, unknown, labels


@torch.no_grad()
def predict_coordinator(model, records: ExpertRecords, standardizer: EvidenceStandardizer,
                        device, *, hard=True, intervention: str | None = None,
                        batch_size=2048):
    evidence = standardizer.transform(records.evidence)
    ds = _tensor_dataset(records, evidence, np.zeros(len(records.labels), dtype=np.float32))
    output = {key: [] for key in ("score", "pred", "gates", "known_weights")}
    model.eval()
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        inputs, _, _ = _coordinator_inputs(batch, device)
        kwargs = {"hard": hard}
        if intervention == "messages_off":
            kwargs["edge_override"] = torch.zeros((len(batch[0]), len(EDGE_NAMES)), device=device)
        elif intervention == "messages_shuffled":
            kwargs["shuffle_messages"] = True
        elif intervention == "drop_identity_sender":
            override = torch.ones((len(batch[0]), len(EDGE_NAMES)), device=device)
            override[:, [1, 2]] = 0; kwargs["edge_override"] = override
        elif intervention == "drop_geometry_sender":
            override = torch.ones((len(batch[0]), len(EDGE_NAMES)), device=device)
            override[:, [0, 3]] = 0; kwargs["edge_override"] = override
        elif intervention == "uniform_weights":
            kwargs["uniform_weights"] = True
        out = model(**inputs, **kwargs)
        output["score"].append(out.unknown_score.cpu().numpy())
        output["pred"].append(out.fused_class_logits.argmax(1).cpu().numpy())
        output["gates"].append(out.edge_gates.cpu().numpy())
        output["known_weights"].append(out.known_weights.cpu().numpy())
    return {key: np.concatenate(value) for key, value in output.items()}


def train_coordinator(train_known: ExpertRecords, val_known: ExpertRecords,
                      pseudo: ExpertRecords, pseudo_sources: np.ndarray,
                      num_classes: int, cfg: Dict, device: torch.device, seed: int,
                      mode: str, budget: float, gate_threshold: float = 0.5):
    set_seed(seed)
    standardizer = EvidenceStandardizer().fit(train_known.evidence)
    rng = np.random.default_rng(seed + 730)
    unique_sources = np.unique(pseudo_sources); rng.shuffle(unique_sources)
    count_val = max(1, int(round(0.2 * len(unique_sources))))
    val_sources = set(unique_sources[:count_val].tolist())
    pseudo_val_mask = np.asarray([int(x) in val_sources for x in pseudo_sources])
    pseudo_train_mask = ~pseudo_val_mask
    pseudo_train, pseudo_val = pseudo.subset(pseudo_train_mask), pseudo.subset(pseudo_val_mask)
    model = CommunicationDecisionCoordinator(
        num_classes, train_known.identity_state.shape[1], train_known.evidence.shape[1],
        message_dim=int(cfg.get("message_dim", 32)),
        hidden_dim=int(cfg.get("coordinator_hidden_dim", 128)),
        mode=mode, budget_target=budget, gate_threshold=gate_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("coordinator_lr", 5e-4)),
                                  weight_decay=float(cfg.get("coordinator_weight_decay", 1e-4)))
    epochs = int(cfg.get("coordinator_epochs", 20)); batch_size = int(cfg.get("coordinator_batch_size", 512))
    history = []
    for epoch in range(1, epochs + 1):
        n = max(len(train_known.labels), len(pseudo_train.labels))
        ik = rng.choice(len(train_known.labels), n, replace=len(train_known.labels) < n)
        ip = rng.choice(len(pseudo_train.labels), n, replace=len(pseudo_train.labels) < n)
        merged = ExpertRecords(**{
            key: (None if getattr(train_known, key) is None else np.concatenate([
                getattr(train_known, key)[ik], getattr(pseudo_train, key)[ip]]))
            for key in train_known.__dataclass_fields__
        })
        unknown = np.r_[np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)]
        order = rng.permutation(2 * n); merged = merged.subset(order); unknown = unknown[order]
        ds = _tensor_dataset(merged, standardizer.transform(merged.evidence), unknown)
        model.train(); total_loss = total_cf = total_budget = 0.0
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            inputs, target, labels = _coordinator_inputs(batch, device)
            out = model(**inputs, hard=False)
            task = per_sample_task_loss(out, target, labels,
                                        float(cfg.get("coordinator_class_weight", 0.5))).mean()
            budget_loss = model.budget_loss(out.edge_gates)
            cf_loss = task.new_zeros(())
            if mode in {"sparse_cf", "audit_sparse_cf", "audit_optional_cf",
                        "audit_residual_cf"}:
                cf_loss, _ = counterfactual_gate_loss(model, inputs, out, target, labels,
                                                       float(cfg.get("cf_temperature", 0.25)))
            loss = (task + float(cfg.get("budget_weight", 0.02)) * budget_loss
                    + float(cfg.get("cf_weight", 0.10)) * cf_loss)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); total_loss += float(loss.detach()) * len(labels)
            total_cf += float(cf_loss.detach()) * len(labels); total_budget += float(budget_loss.detach()) * len(labels)
        vk = predict_coordinator(model, val_known, standardizer, device, hard=True)
        vp = predict_coordinator(model, pseudo_val, standardizer, device, hard=True)
        tau = float(np.quantile(vk["score"], 0.95))
        auc = float(roc_auc_score(np.r_[np.zeros(len(vk["score"])), np.ones(len(vp["score"]))],
                                 np.r_[vk["score"], vp["score"]]))
        recall = float((vp["score"] >= tau).mean())
        acc = float((vk["pred"] == val_known.labels).mean())
        row = {"epoch": epoch, "loss": total_loss / len(ds), "cf_loss": total_cf / len(ds),
               "budget_loss": total_budget / len(ds), "pseudo_auroc": auc,
               "pseudo_recall": recall, "known_accuracy": acc,
               "mean_edge_gate": float(vk["gates"].mean())}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"      coord-{mode} ep{epoch:03d} loss={row['loss']:.4f} "
                  f"p_auc={auc:.4f} p_rec={recall:.4f} k_acc={acc:.4f} gate={row['mean_edge_gate']:.3f}")
    # The epoch count is selected by the LCO protocol and then frozen.  Do not
    # silently choose a checkpoint by near-perfect synthetic-unknown metrics.
    return model, standardizer, history, epochs


def evaluate_rule(val_records: ExpertRecords, open_records: ExpertRecords,
                  val_prediction: Dict[str, np.ndarray], open_prediction: Dict[str, np.ndarray],
                  score_rule: str = "boundary", known_acceptance: float = 0.95,
                  adaptive_threshold_prior: float | None = None,
                  return_thresholds: bool = False):
    calibrator = CalibrationAgent(score_rule).fit(val_records, val_prediction)
    val_calibrated = calibrator.transform(val_records, val_prediction).unknown_score
    calibrated = calibrator.transform(open_records, open_prediction).unknown_score
    if adaptive_threshold_prior is None:
        thresholds = np.full(len(calibrated), known_acceptance, dtype=np.float64)
    else:
        threshold_agent = ClassConditionalThresholdAgent(
            known_acceptance, adaptive_threshold_prior).fit(
                val_calibrated, val_prediction["pred"])
        thresholds = threshold_agent.thresholds(open_prediction["pred"])
    is_unknown = (open_records.labels == -1).astype(np.int32)
    metrics = detection_metrics(is_unknown, calibrated)
    metrics["oscr"] = oscr(is_unknown, open_prediction["pred"], open_records.labels, 1.0 - calibrated)
    metrics.update(threshold_decision(is_unknown, open_records.labels, open_prediction["pred"],
                                      calibrated, thresholds))
    metrics["threshold_min"] = float(thresholds.min())
    metrics["threshold_mean"] = float(thresholds.mean())
    metrics["threshold_max"] = float(thresholds.max())
    return (metrics, calibrated, thresholds) if return_thresholds else (metrics, calibrated)


def evaluate_arbitrated_rule(val_records: ExpertRecords, open_records: ExpertRecords,
                             val_communicated: Dict[str, np.ndarray],
                             open_communicated: Dict[str, np.ndarray],
                             val_local: Dict[str, np.ndarray],
                             open_local: Dict[str, np.ndarray],
                             score_rule: str = "boundary",
                             arbitration_rule: str = "communication",
                             known_acceptance: float = 0.95,
                             adaptive_threshold_prior: float | None = None,
                             return_thresholds: bool = False):
    """Evaluate a Known-only calibrated arbitration between two audit branches."""
    arbitrator = DecisionArbitratorAgent(score_rule, arbitration_rule).fit(
        val_records, val_communicated, val_local)
    val_calibrated = arbitrator.transform(
        val_records, val_communicated, val_local).unknown_score
    calibrated = arbitrator.transform(
        open_records, open_communicated, open_local).unknown_score
    if adaptive_threshold_prior is None:
        thresholds = np.full(len(calibrated), known_acceptance, dtype=np.float64)
    else:
        threshold_agent = ClassConditionalThresholdAgent(
            known_acceptance, adaptive_threshold_prior).fit(
                val_calibrated, val_communicated["pred"])
        thresholds = threshold_agent.thresholds(open_communicated["pred"])
    is_unknown = (open_records.labels == -1).astype(np.int32)
    metrics = detection_metrics(is_unknown, calibrated)
    # The communication modes used by this agent only audit unknown risk, so
    # their class prediction is the communicated branch's convex expert vote.
    metrics["oscr"] = oscr(is_unknown, open_communicated["pred"],
                           open_records.labels, 1.0 - calibrated)
    metrics.update(threshold_decision(
        is_unknown, open_records.labels, open_communicated["pred"], calibrated,
        thresholds))
    metrics["threshold_min"] = float(thresholds.min())
    metrics["threshold_mean"] = float(thresholds.mean())
    metrics["threshold_max"] = float(thresholds.max())
    return (metrics, calibrated, thresholds) if return_thresholds else (metrics, calibrated)
