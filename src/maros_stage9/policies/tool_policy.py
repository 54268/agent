"""Counterfactual-utility-supervised discrete Top-1/Top-2 tool selection."""
from __future__ import annotations

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from ..tools.observations import policy_observation


def tool_actions(names: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple((i,) for i in range(len(names))) + tuple(
        combinations(range(len(names)), 2)) + ((),)


class ToolPolicy(nn.Module):
    def __init__(self, role: str, tool_names: tuple[str, ...],
                 width: int = 24) -> None:
        super().__init__()
        if role not in {"identity", "impairment"} or len(tool_names) < 2:
            raise ValueError("unsupported local policy")
        self.role = role
        self.tool_names = tool_names
        self.actions = tool_actions(tool_names)
        self.scout = nn.Sequential(
            nn.Conv1d(2, width, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, len(self.actions)),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.scout(policy_observation(iq, self.role))


def action_logits(tool_logits: torch.Tensor,
                  actions: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """Logit proposals for all legal local actions [B,A,K]."""
    if tool_logits.ndim != 3:
        raise ValueError("tool logits must have shape [B,T,K]")
    return torch.stack([
        (tool_logits[:, action].mean(1) if action else
         torch.zeros_like(tool_logits[:, 0])) for action in actions], dim=1)


def counterfactual_utility(tool_logits: torch.Tensor, labels: torch.Tensor,
                           actions: tuple[tuple[int, ...], ...],
                           pair_cost: float, stop_cost: float = 0.25) -> torch.Tensor:
    """Per-sample negative NLL of each action, minus extra-tool cost."""
    proposals = action_logits(tool_logits, actions)
    batch, count, classes = proposals.shape
    nll = F.cross_entropy(
        proposals.reshape(batch * count, classes),
        labels[:, None].expand(-1, count).reshape(-1),
        reduction="none").reshape(batch, count)
    costs = nll.new_tensor([
        float(stop_cost) if not action else float(pair_cost) * (len(action) - 1)
        for action in actions])
    return -nll - costs[None]
