"""Known-only boundary challenges for Stage-6 dialogue training."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .agents import RoleAgents
from .contracts import LocalDecision


def _different_class_partner(labels: torch.Tensor) -> torch.Tensor:
    n = len(labels)
    if n < 2:
        return torch.arange(n, device=labels.device)
    partner = torch.roll(torch.arange(n, device=labels.device), 1)
    for shift in range(2, n + 1):
        conflict = labels[partner] == labels
        if not bool(conflict.any()):
            break
        candidate = torch.roll(torch.arange(n, device=labels.device), shift)
        partner = torch.where(conflict, candidate, partner)
    return partner


def _class_centres(state: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    centres = torch.empty_like(state)
    for cls in torch.unique(labels):
        mask = labels == cls
        centre = F.normalize(state[mask].mean(0), dim=0)
        centres[mask] = centre
    return centres


def _challenge_state(state: torch.Tensor, labels: torch.Tensor, eta: torch.Tensor,
                     repel_weight: torch.Tensor) -> torch.Tensor:
    centre = _class_centres(state, labels)
    partner = _different_class_partner(labels)
    outward = F.normalize(state - centre, dim=-1)
    repel = F.normalize(state - state[partner], dim=-1)
    direction = F.normalize((1.0 - repel_weight) * outward
                            + repel_weight * repel, dim=-1)
    # Batch-local radius keeps the challenge close to the current manifold.
    radius = (state - centre).norm(dim=-1, keepdim=True).detach().clamp_min(0.05)
    return F.normalize(state + eta * radius * direction, dim=-1)


class AdaptiveBoundaryExplorer(nn.Module):
    """Training-only adversary selecting extrapolation scale and direction."""

    def __init__(self, intent_dim: int = 16, hidden_dim: int = 64,
                 eta_min: float = 1.0, eta_max: float = 2.0):
        super().__init__()
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        observation_dim = 3 * intent_dim + 5
        self.policy = nn.Sequential(
            nn.LayerNorm(observation_dim), nn.Linear(observation_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 6),
        )

    def _observation(self, decisions: Dict[str, LocalDecision]) -> torch.Tensor:
        w, p, v = decisions["waveform"], decisions["prototype"], decisions["verifier"]
        pw, pp = F.softmax(w.class_logits, -1), F.softmax(p.class_logits, -1)
        mean = 0.5 * (pw + pp)
        js = 0.5 * ((pw * (pw.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
                    + (pp * (pp.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1))
        disagree = (w.class_logits.argmax(1) != p.class_logits.argmax(1)).float()
        return torch.cat([
            w.intent, p.intent, v.intent,
            torch.stack([w.reliability, p.reliability, v.reliability, js, disagree], 1),
        ], 1)

    def actions(self, decisions: Dict[str, LocalDecision]) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.policy(self._observation(decisions))
        eta = self.eta_min + (self.eta_max - self.eta_min) * torch.sigmoid(action[:, :3])
        repel = torch.sigmoid(action[:, 3:])
        return eta, repel

    def forward(self, agents: RoleAgents, decisions: Dict[str, LocalDecision],
                labels: torch.Tensor) -> Dict[str, LocalDecision]:
        eta, repel = self.actions(decisions)
        states = {}
        for index, role in enumerate(("waveform", "prototype", "verifier")):
            states[role] = _challenge_state(
                decisions[role].state, labels, eta[:, index:index + 1],
                repel[:, index:index + 1])
        return agents.from_states(states)


def static_boundary_challenge(agents: RoleAgents,
                              decisions: Dict[str, LocalDecision],
                              labels: torch.Tensor, eta: float = 1.5,
                              repel_weight: float = 0.35) -> Dict[str, LocalDecision]:
    states = {
        role: _challenge_state(
            decisions[role].state, labels,
            decisions[role].state.new_full((len(labels), 1), float(eta)),
            decisions[role].state.new_full((len(labels), 1), float(repel_weight)),
        )
        for role in ("waveform", "prototype", "verifier")
    }
    return agents.from_states(states)
