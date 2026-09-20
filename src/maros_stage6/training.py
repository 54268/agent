"""Training, prediction and Known-only evaluation for Stage-6."""
from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from maros_staged.evidence import ClassConditionalCdf
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision

from .contracts import DialogueOutput, LocalDecision
from .dialogue import EDGE_NAMES
from .explorer import AdaptiveBoundaryExplorer, static_boundary_challenge
from .model import Stage6System


ROLES = ("waveform", "prototype", "verifier")


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _local_loss(decisions: Dict[str, LocalDecision], labels: torch.Tensor,
                open_target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    per_role = {}
    known = labels >= 0
    for role in ROLES:
        decision = decisions[role]
        loss = F.binary_cross_entropy_with_logits(
            decision.unknown_logit, open_target, reduction="none")
        if bool(known.any()):
            loss = loss.clone()
            loss[known] += F.cross_entropy(
                decision.class_logits[known], labels[known], reduction="none")
        per_role[role] = loss
    stacked = torch.stack([per_role[role] for role in ROLES], dim=1)
    return stacked.mean(), per_role


def dialogue_sample_loss(output: DialogueOutput, labels: torch.Tensor,
                         open_target: torch.Tensor,
                         class_weight: float = 1.0) -> torch.Tensor:
    result = F.binary_cross_entropy_with_logits(
        output.unknown_logit, open_target, reduction="none")
    known = labels >= 0
    if bool(known.any()):
        result = result.clone()
        result[known] += float(class_weight) * F.cross_entropy(
            output.fused_class_logits[known], labels[known], reduction="none")
    return result


def _protocol_action_loss(output: DialogueOutput, labels: torch.Tensor,
                          pseudo_unknown: bool) -> torch.Tensor:
    """Supervise support/challenge/unknown-suspect packet intents."""
    if pseudo_unknown:
        target = labels.new_full(labels.shape, 2)
    else:
        waveform_proposal = output.local_decisions["waveform"].class_logits.argmax(1)
        prototype_proposal = output.local_decisions["prototype"].class_logits.argmax(1)
        proposal = torch.where(
            output.leader_id == 0, waveform_proposal, prototype_proposal)
        target = torch.where(proposal == labels,
                             labels.new_zeros(labels.shape),
                             labels.new_ones(labels.shape))
    losses = []
    for packet in output.messages:
        action_probability = packet.payload[:, :3].clamp_min(1e-8)
        losses.append(F.nll_loss(action_probability.log(), target))
    return torch.stack(losses).mean()


def _dialogue_objective(system: Stage6System,
                        known_decisions: Dict[str, LocalDecision],
                        pseudo_decisions: Dict[str, LocalDecision],
                        labels: torch.Tensor, cfg: Dict,
                        use_counterfactual: bool) -> tuple[torch.Tensor, dict]:
    zeros = labels.new_zeros(len(labels), dtype=torch.float32)
    ones = labels.new_ones(len(labels), dtype=torch.float32)
    negative_labels = labels.new_full(labels.shape, -1)
    known_output = system.deliberate(known_decisions, hard=False)
    pseudo_output = system.deliberate(pseudo_decisions, hard=False)
    known_team = dialogue_sample_loss(
        known_output, labels, zeros, float(cfg.get("team_class_weight", 1.0)))
    pseudo_team = dialogue_sample_loss(
        pseudo_output, negative_labels, ones,
        float(cfg.get("team_class_weight", 1.0)))
    team = torch.cat([known_team, pseudo_team])
    # A communication policy must retain a competent fall-back.  Train the
    # exact same candidate with both message edges disabled, then penalise
    # communicated decisions that are worse than their own silent decision.
    # This is deliberately in-model: it does not borrow the separately
    # trained B2 baseline and therefore remains a valid causal ablation.
    if system.mode not in {"no_comm", "early_fusion"}:
        silent_known = system.deliberate(
            known_decisions, hard=False,
            edge_override=known_output.edge_gates.new_zeros(
                known_output.edge_gates.shape))
        silent_pseudo = system.deliberate(
            pseudo_decisions, hard=False,
            edge_override=pseudo_output.edge_gates.new_zeros(
                pseudo_output.edge_gates.shape))
        silent = torch.cat([
            dialogue_sample_loss(
                silent_known, labels, zeros,
                float(cfg.get("team_class_weight", 1.0))),
            dialogue_sample_loss(
                silent_pseudo, negative_labels, ones,
                float(cfg.get("team_class_weight", 1.0))),
        ])
        communication_regret = F.relu(team - silent.detach()).mean()
        silent_anchor = silent.mean()
    else:
        communication_regret = team.new_zeros(())
        silent_anchor = team.new_zeros(())
    local_known, known_role = _local_loss(known_decisions, labels, zeros)
    local_pseudo, pseudo_role = _local_loss(pseudo_decisions, negative_labels, ones)
    best_known = torch.stack([known_role[r] for r in ROLES], 1).min(1).values
    best_pseudo = torch.stack([pseudo_role[r] for r in ROLES], 1).min(1).values
    best_local = torch.cat([best_known, best_pseudo])
    no_regret_margin = float(cfg.get("no_regret_margin", 0.05))
    no_regret = F.relu(team - best_local + no_regret_margin).mean()
    local = 0.5 * (local_known + local_pseudo)
    gates = torch.cat([known_output.edge_gates, pseudo_output.edge_gates])
    communication = gates.mean()
    masks = torch.cat([
        known_output.messages[0].feature_mask,
        known_output.messages[1].feature_mask,
        pseudo_output.messages[0].feature_mask,
        pseudo_output.messages[1].feature_mask,
    ])
    bandwidth = masks.mean()
    cf_loss = team.new_zeros(())
    delta_rows = []
    if use_counterfactual:
        for edge in range(len(EDGE_NAMES)):
            known_override = known_output.edge_gates.detach().clone()
            pseudo_override = pseudo_output.edge_gates.detach().clone()
            known_override[:, edge] = 0.0
            pseudo_override[:, edge] = 0.0
            dropped_known = system.deliberate(
                known_decisions, hard=False, edge_override=known_override)
            dropped_pseudo = system.deliberate(
                pseudo_decisions, hard=False, edge_override=pseudo_override)
            dropped_loss = torch.cat([
                dialogue_sample_loss(dropped_known, labels, zeros,
                                     float(cfg.get("team_class_weight", 1.0))),
                dialogue_sample_loss(dropped_pseudo, negative_labels, ones,
                                     float(cfg.get("team_class_weight", 1.0))),
            ])
            # Positive means the received message reduced task loss.
            delta_rows.append((dropped_loss - team).detach())
        delta = torch.stack(delta_rows, dim=1)
        predicted = torch.cat([
            known_output.query_actions.expected_gain,
            pseudo_output.query_actions.expected_gain,
        ])
        cf_loss = F.smooth_l1_loss(predicted, delta)
        known_output.counterfactual_gains = delta[:len(labels)]
        pseudo_output.counterfactual_gains = delta[len(labels):]
    if system.mode.startswith("sequential"):
        protocol_action = 0.5 * (
            _protocol_action_loss(known_output, labels, False)
            + _protocol_action_loss(pseudo_output, labels, True))
    else:
        protocol_action = team.new_zeros(())
    loss = (team.mean()
            + float(cfg.get("silent_loss_weight", 0.5)) * silent_anchor
            + float(cfg.get("communication_regret_weight", 0.5))
            * communication_regret
            + float(cfg.get("local_loss_weight", 0.35)) * local
            + float(cfg.get("no_regret_weight", 0.5)) * no_regret
            + float(cfg.get("communication_weight", 0.02)) * communication
            + float(cfg.get("bandwidth_weight", 0.005)) * bandwidth
            + float(cfg.get("counterfactual_weight", 0.25)) * cf_loss)
    loss = loss + float(cfg.get("protocol_action_weight", 0.2)) * protocol_action
    return loss, {
        "team": float(team.mean().detach()),
        "silent_anchor": float(silent_anchor.detach()),
        "communication_regret": float(communication_regret.detach()),
        "local": float(local.detach()),
        "no_regret": float(no_regret.detach()),
        "communication": float(communication.detach()),
        "bandwidth": float(bandwidth.detach()),
        "counterfactual": float(cf_loss.detach()),
        "protocol_action": float(protocol_action.detach()),
    }


@torch.no_grad()
def local_accuracy(agents: nn.Module, dataset, device: torch.device,
                   batch_size: int = 512) -> dict:
    agents.eval()
    correct = {role: 0 for role in ROLES}
    total = 0
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        decisions = agents(x)
        for role in ROLES:
            correct[role] += int((decisions[role].class_logits.argmax(1) == y).sum())
        total += len(y)
    return {role: correct[role] / max(total, 1) for role in ROLES}


def pretrain_local_agents(system: Stage6System, train_set, val_set, cfg: Dict,
                          device: torch.device, seed: int) -> tuple[list[dict], int]:
    set_seed(seed)
    agents = system.agents.to(device)
    optimizer = torch.optim.AdamW(
        agents.parameters(), lr=float(cfg.get("local_lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    epochs = int(cfg.get("local_pretrain_epochs", 30))
    batch_size = int(cfg.get("batch_size", 256))
    generator = torch.Generator().manual_seed(seed)
    history, best_score, best_state, best_epoch = [], -float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        agents.train()
        running = 0.0; total = 0
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            decisions = agents(x)
            zeros = y.new_zeros(len(y), dtype=torch.float32)
            local, _ = _local_loss(decisions, y, zeros)
            proto = decisions["prototype"].private_evidence
            distances = proto["distances"]
            own = distances.gather(1, y[:, None]).squeeze(1)
            rival = distances.clone(); rival.scatter_(1, y[:, None], float("inf"))
            geometry = own.mean() + F.relu(
                float(cfg.get("prototype_margin", 0.5)) + own - rival.min(1).values).mean()
            loss = local + float(cfg.get("prototype_loss_weight", 0.1)) * geometry
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agents.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(y); total += len(y)
        accuracy = local_accuracy(agents, val_set, device, batch_size)
        score = float(np.mean(list(accuracy.values())))
        row = {"epoch": epoch, "loss": running / max(total, 1),
               **{f"val_{k}_accuracy": v for k, v in accuracy.items()}}
        history.append(row)
        if score > best_score:
            best_score = score; best_epoch = epoch
            best_state = copy.deepcopy(agents.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"      local ep{epoch:03d} loss={row['loss']:.4f} "
                  f"w={accuracy['waveform']:.3f} p={accuracy['prototype']:.3f} "
                  f"v={accuracy['verifier']:.3f}")
    if best_state is not None:
        agents.load_state_dict(best_state)
    return history, best_epoch


def _set_agent_trainability(system: Stage6System, joint: bool) -> None:
    for parameter in system.agents.parameters():
        parameter.requires_grad_(False)
    if joint:
        for role in ROLES:
            decision = getattr(system.agents, role).decision
            for parameter in decision.parameters():
                parameter.requires_grad_(True)
        # The metric prototypes are policy parameters, not a signal encoder.
        system.agents.prototype.decision.prototypes.requires_grad_(True)
    if system.mode not in {"no_comm", "early_fusion"}:
        # These two modules define the exact silent fallback learned before
        # protocol training.  Communication is restricted to a residual on
        # top of it and cannot rewrite the comparator it must beat.
        for module in (system.dialogue.no_comm_weights,
                       system.dialogue.no_comm_open):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def _detach_decisions(decisions: Dict[str, LocalDecision]) -> Dict[str, LocalDecision]:
    return {
        role: LocalDecision(
            row.state.detach(), row.class_logits.detach(), row.unknown_logit.detach(),
            row.reliability.detach(), row.intent.detach(),
            {key: value.detach() for key, value in row.private_evidence.items()},
        )
        for role, row in decisions.items()
    }


def train_dialogue_candidate(system: Stage6System, train_set, val_set, cfg: Dict,
                             device: torch.device, seed: int,
                             adaptive_explorer: bool = False) -> tuple[list[dict], object | None]:
    set_seed(seed)
    system.to(device)
    explorer = (AdaptiveBoundaryExplorer(
        int(cfg.get("intent_dim", 16)), int(cfg.get("explorer_hidden_dim", 64)),
        float(cfg.get("pug_eta_min", 1.0)), float(cfg.get("pug_eta_max", 2.0)),
    ).to(device) if adaptive_explorer else None)
    explorer_optimizer = (torch.optim.AdamW(
        explorer.parameters(), lr=float(cfg.get("explorer_lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4))) if explorer else None)
    history = []
    phases = (("frozen", int(cfg.get("dialogue_frozen_epochs", 10)), False),
              ("joint", int(cfg.get("joint_finetune_epochs", 20)), True))
    batch_size = int(cfg.get("batch_size", 256))
    generator = torch.Generator().manual_seed(seed + 17)
    for phase_name, epochs, joint in phases:
        _set_agent_trainability(
            system, joint and bool(cfg.get("joint_agent_top", True)))
        parameters = [p for p in system.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(cfg.get(
                "joint_lr" if joint else "dialogue_lr", 3e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)))
        for epoch in range(1, epochs + 1):
            system.train(); totals = {}; count = 0
            loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                generator=generator, num_workers=0, drop_last=True)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                decisions = system.local(x)
                if explorer is not None:
                    # One adversarial update: find a plausible challenge that
                    # the current team is least able to reject.
                    explorer.train(); explorer_optimizer.zero_grad()
                    detached = _detach_decisions(decisions)
                    pseudo_adv = explorer(system.agents, detached, y)
                    adversarial_output = system.deliberate(pseudo_adv, hard=False)
                    target = y.new_ones(len(y), dtype=torch.float32)
                    adversarial_bce = F.binary_cross_entropy_with_logits(
                        adversarial_output.unknown_logit, target)
                    explorer_loss = -adversarial_bce
                    explorer_loss.backward(); explorer_optimizer.step()
                    for parameter in explorer.parameters():
                        parameter.requires_grad_(False)
                    pseudo = explorer(system.agents, decisions, y)
                else:
                    pseudo = static_boundary_challenge(
                        system.agents, decisions, y,
                        float(cfg.get("pug_eta", 1.5)),
                        float(cfg.get("pug_repel_weight", 0.35)))
                use_cf = system.mode == "sequential_cf"
                loss, parts = _dialogue_objective(
                    system, decisions, pseudo, y, cfg, use_cf)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                if explorer is not None:
                    for parameter in explorer.parameters():
                        parameter.requires_grad_(True)
                count += 1
                for key, value in {"loss": float(loss.detach()), **parts}.items():
                    totals[key] = totals.get(key, 0.0) + value
            accuracy = local_accuracy(system.agents, val_set, device, batch_size)
            row = {"phase": phase_name, "epoch": epoch,
                   **{key: value / max(count, 1) for key, value in totals.items()},
                   **{f"val_{key}_accuracy": value for key, value in accuracy.items()}}
            history.append(row)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"      {system.mode}/{phase_name} ep{epoch:03d} "
                      f"loss={row.get('loss', 0):.4f} team={row.get('team', 0):.4f} "
                      f"cf={row.get('counterfactual', 0):.4f} "
                      f"gate={row.get('communication', 0):.3f}")
    return history, explorer


@torch.no_grad()
def predict_system(system: Stage6System, dataset, device: torch.device, *,
                   intervention: str | None = None,
                   batch_size: int = 1024) -> dict[str, np.ndarray]:
    system.eval()
    rows = {key: [] for key in (
        "y", "pred", "raw_unknown", "gates", "leader", "bits", "known_weights",
        "expected_gain", "transcript", "message_waveform", "message_prototype",
        "mask_waveform", "mask_prototype",
        "pred_waveform", "pred_prototype", "pred_verifier",
        "raw_waveform", "raw_prototype", "raw_verifier",
    )}
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = x.to(device)
        out = system(x, hard=True, intervention=intervention)
        rows["y"].append(y.numpy())
        rows["pred"].append(out.fused_class_logits.argmax(1).cpu().numpy())
        rows["raw_unknown"].append(out.unknown_logit.cpu().numpy())
        rows["gates"].append(out.edge_gates.cpu().numpy())
        rows["leader"].append(out.leader_id.cpu().numpy())
        rows["bits"].append(out.query_actions.communication_cost.cpu().numpy())
        rows["known_weights"].append(out.known_weights.cpu().numpy())
        rows["expected_gain"].append(
            out.query_actions.expected_gain.cpu().numpy())
        rows["transcript"].append(out.transcript.cpu().numpy())
        rows["message_waveform"].append(out.messages[0].payload.cpu().numpy())
        rows["message_prototype"].append(out.messages[1].payload.cpu().numpy())
        rows["mask_waveform"].append(out.messages[0].feature_mask.cpu().numpy())
        rows["mask_prototype"].append(out.messages[1].feature_mask.cpu().numpy())
        for role in ROLES:
            rows[f"pred_{role}"].append(
                out.local_decisions[role].class_logits.argmax(1).cpu().numpy())
            rows[f"raw_{role}"].append(
                out.local_decisions[role].unknown_logit.cpu().numpy())
    return {key: np.concatenate(value) for key, value in rows.items()}


@dataclass
class KnownOnlyCalibrationService:
    known_acceptance: float = 0.90
    n_prior: float = 25.0
    calibrator: ClassConditionalCdf | None = None
    threshold: float | None = None

    def fit(self, raw_score: np.ndarray, predicted_class: np.ndarray):
        self.calibrator = ClassConditionalCdf(
            raw_score, predicted_class, n_prior=self.n_prior)
        calibrated = self.calibrator.score(raw_score, predicted_class)
        self.threshold = float(np.quantile(calibrated, self.known_acceptance))
        return self

    def transform(self, raw_score: np.ndarray, predicted_class: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("calibration service is not fitted")
        return self.calibrator.score(raw_score, predicted_class)


def evaluate_open_set(val: dict[str, np.ndarray], known_open: dict[str, np.ndarray],
                      unknown_open: dict[str, np.ndarray], *, method: str = "team",
                      known_acceptance: float = 0.90,
                      n_prior: float = 25.0) -> tuple[dict, np.ndarray, float]:
    if method == "team":
        pred_key, raw_key = "pred", "raw_unknown"
    else:
        pred_key, raw_key = f"pred_{method}", f"raw_{method}"
    service = KnownOnlyCalibrationService(known_acceptance, n_prior).fit(
        val[raw_key], val[pred_key])
    y = np.r_[known_open["y"], np.full(len(unknown_open["y"]), -1, dtype=np.int64)]
    pred = np.r_[known_open[pred_key], unknown_open[pred_key]]
    raw = np.r_[known_open[raw_key], unknown_open[raw_key]]
    classes = pred
    score = service.transform(raw, classes)
    is_unknown = (y == -1).astype(np.int32)
    metrics = detection_metrics(is_unknown, score)
    metrics["oscr"] = oscr(is_unknown, pred, y, 1.0 - score)
    metrics.update(threshold_decision(
        is_unknown, y, pred, score, float(service.threshold)))
    return metrics, score, float(service.threshold)


def lco_proxy_metrics(system: Stage6System, val_set, heldout_set,
                      device: torch.device, cfg: Dict) -> tuple[dict, dict]:
    val = predict_system(system, val_set, device)
    held = predict_system(system, heldout_set, device)
    methods = (*ROLES, "team")
    metrics = {}
    for method in methods:
        metrics[method], _, _ = evaluate_open_set(
            val, val, held, method=method,
            known_acceptance=float(cfg.get("known_acceptance", 0.90)),
            n_prior=float(cfg.get("calibration_prior", 25.0)))
    behaviour = {
        "query_rate": float(held["gates"].max(axis=1).mean()),
        "mean_edges": float(held["gates"].sum(1).mean()),
        "mean_bits": float(held["bits"].mean()),
        "leader_waveform_rate": float((held["leader"] == 0).mean()),
    }
    return metrics, behaviour
