"""G0-only supervised expert and counterfactual tool-policy training."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .policies.tool_policy import counterfactual_utility
from .splits import assert_no_formal_unknown


def cap_per_class(dataset: Dataset, limit: int, seed: int) -> Dataset:
    assert_no_formal_unknown(dataset)
    labels = getattr(dataset, "y", None)
    if labels is None:
        raise TypeError("G0 data must expose provenance-backed y labels")
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0):
        raise ValueError("G0 Known training/calibration cannot contain proxy Unknown")
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        rng.shuffle(candidates)
        indices.extend(candidates[:limit].tolist())
    return Subset(dataset, sorted(indices))


def train_experts(system, train_known: Dataset, cfg: Mapping,
                  device: torch.device) -> list[dict]:
    """Train each passive tool on support-Known only; no policy gradients yet."""
    assert_no_formal_unknown(train_known)
    train = cap_per_class(train_known, int(cfg["train_samples_per_class"]),
                          int(cfg["seed"]))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader = DataLoader(train, batch_size=int(cfg["batch_size"]),
                        shuffle=True, generator=generator)
    expert_parameters = [parameter for agent in system.agents()
                         for parameter in agent.registry.parameters()]
    optimizer = torch.optim.AdamW(
        expert_parameters, lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]))
    history = []
    for epoch in range(int(cfg["expert_epochs"])):
        system.train()
        totals = {agent.role: {"loss": 0.0, "correct": 0, "seen": 0}
                  for agent in system.agents()}
        for iq, labels in loader:
            iq, labels = iq.to(device), labels.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = {agent.role: agent.all_tool_logits(iq)
                       for agent in system.agents()}
            losses = []
            for agent in system.agents():
                logits = outputs[agent.role]
                losses.append(torch.stack([
                    F.cross_entropy(logits[:, index], labels)
                    for index in range(logits.shape[1])]).mean())
                tally = totals[agent.role]
                tally["loss"] += float(losses[-1].detach()) * len(labels)
                tally["correct"] += int(logits.argmax(2).eq(labels[:, None]).sum())
                tally["seen"] += len(labels) * logits.shape[1]
            torch.stack(losses).sum().backward()
            optimizer.step()
        history.append({
            "epoch": epoch + 1,
            **{agent.role: {
                "mean_tool_loss": totals[agent.role]["loss"] /
                                  max(totals[agent.role]["seen"] /
                                      len(agent.tool_names), 1),
                "mean_tool_accuracy": totals[agent.role]["correct"] /
                                      max(totals[agent.role]["seen"], 1),
            } for agent in system.agents()},
        })
    return history


@torch.no_grad()
def _calibration_targets(agent, calibration_known: Dataset, cfg: Mapping,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    assert_no_formal_unknown(calibration_known)
    calibration = cap_per_class(
        calibration_known, int(cfg["policy_samples_per_class"]),
        int(cfg["seed"]) + 101)
    loader = DataLoader(calibration, batch_size=int(cfg["batch_size"]),
                        shuffle=False)
    agent.eval()
    observations, utilities = [], []
    for iq, labels in loader:
        iq_device = iq.to(device)
        logits = agent.all_tool_logits(iq_device)
        utility = counterfactual_utility(
            logits, labels.long().to(device), agent.tool_policy.actions,
            float(cfg["pair_cost"]), float(cfg["stop_cost"]))
        observations.append(iq.cpu())
        utilities.append(utility.cpu())
    return torch.cat(observations), torch.cat(utilities)


def train_tool_policies(system, calibration_known: Dataset, cfg: Mapping,
                        device: torch.device) -> tuple[dict, dict]:
    """Only local policies train, using frozen expert counterfactual targets."""
    histories, fixed_actions = {}, {}
    for agent in system.agents():
        iq, utility = _calibration_targets(agent, calibration_known, cfg, device)
        fixed_actions[agent.role] = int(utility.mean(0).argmax())
        teacher = (utility / float(cfg["teacher_temperature"])).softmax(1)
        dataset = TensorDataset(iq, teacher)
        loader = DataLoader(
            dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 202))
        optimizer = torch.optim.AdamW(
            agent.tool_policy.parameters(), lr=float(cfg["policy_learning_rate"]),
            weight_decay=float(cfg["weight_decay"]))
        history = []
        for epoch in range(int(cfg["policy_epochs"])):
            agent.tool_policy.train()
            total = seen = 0
            for batch_iq, target in loader:
                batch_iq, target = batch_iq.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = agent.tool_policy(batch_iq)
                loss = -(target * F.log_softmax(output, dim=1)).sum(1).mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch_iq)
                seen += len(batch_iq)
            history.append({"epoch": epoch + 1, "soft_target_loss": total / max(seen, 1)})
        histories[agent.role] = history
    return histories, fixed_actions

