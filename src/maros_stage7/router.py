"""Public-evidence-only value-of-information routing for Stage-7."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import LocalDecision, RouteAction


def validate_public_decisions(
    decisions: Mapping[str, LocalDecision],
) -> tuple[int, int, int]:
    """Validate the complete public boundary without accessing private state."""

    if set(decisions) != {"waveform", "prototype"}:
        raise ValueError("public decisions must contain waveform and prototype")
    reference = decisions["waveform"]
    if reference.class_logits.ndim != 2:
        raise ValueError("LocalDecision.class_logits must be [batch, classes]")
    batch, classes = reference.class_logits.shape
    if classes < 2:
        raise ValueError("Stage-7 requires at least two class logits")
    summary_dim = reference.public_summary.shape[1] \
        if reference.public_summary.ndim == 2 else -1
    if summary_dim <= 0:
        raise ValueError("LocalDecision.public_summary must be [batch, features]")
    reference_device = reference.class_logits.device
    for agent_name in ("waveform", "prototype"):
        decision = decisions[agent_name]
        if decision.class_logits.shape != (batch, classes):
            raise ValueError("Agent class logits must have identical shape")
        if decision.public_summary.shape != (batch, summary_dim):
            raise ValueError("Agent public summaries must have identical shape")
        for field_name in ("unknown_score", "reliability", "top1", "top2"):
            value = getattr(decision, field_name)
            if not torch.is_tensor(value) or value.shape != (batch,):
                raise ValueError(
                    f"LocalDecision.{field_name} must have shape [batch]")
        continuous = (
            decision.class_logits, decision.unknown_score,
            decision.reliability, decision.public_summary)
        all_fields = (*continuous, decision.top1, decision.top2)
        if any(value.device != reference_device for value in all_fields):
            raise ValueError("all public decision tensors must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in continuous):
            raise ValueError(f"{agent_name} public decision contains non-finite values")
        if bool(((decision.reliability < 0)
                 | (decision.reliability > 1)).any()):
            raise ValueError("LocalDecision.reliability must lie in [0, 1]")
        for field_name in ("top1", "top2"):
            value = getattr(decision, field_name)
            if value.dtype.is_floating_point or value.dtype.is_complex:
                raise TypeError(f"LocalDecision.{field_name} must be integral")
            if bool(((value < 0) | (value >= classes)).any()):
                raise ValueError(f"LocalDecision.{field_name} is outside class range")
        if bool(decision.top1.eq(decision.top2).any()):
            raise ValueError("LocalDecision top1 and top2 must be distinct")
    return batch, classes, summary_dim


def public_team_features(decisions: Mapping[str, LocalDecision]) -> torch.Tensor:
    """Build router/B2 features without touching either Agent's private state."""

    validate_public_decisions(decisions)
    waveform = decisions["waveform"]
    prototype = decisions["prototype"]
    wp = F.softmax(waveform.class_logits, dim=-1)
    pp = F.softmax(prototype.class_logits, dim=-1)
    midpoint = 0.5 * (wp + pp)
    js = 0.5 * (
        (wp * (wp.clamp_min(1e-8).log()
               - midpoint.clamp_min(1e-8).log())).sum(1)
        + (pp * (pp.clamp_min(1e-8).log()
                 - midpoint.clamp_min(1e-8).log())).sum(1)
    )
    disagreement = waveform.top1.ne(prototype.top1).to(wp)
    unknown_gap = (torch.sigmoid(waveform.unknown_score)
                   - torch.sigmoid(prototype.unknown_score)).abs()
    return torch.cat([
        waveform.public_summary, prototype.public_summary,
        torch.stack([js, disagreement, unknown_gap], dim=1),
    ], dim=1)


class ValueOfInformationRouter(nn.Module):
    """Choose STOP unless a communication direction predicts positive value."""

    def __init__(self, public_summary_dim: int = 7, hidden_dim: int = 96):
        super().__init__()
        input_dim = 2 * int(public_summary_dim) + 3
        self.gain_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 2),
        )
        # A new/untrained router conservatively stops.  Counterfactual action
        # labels, rather than an entropy regularizer, must justify querying.
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.constant_(self.gain_head[-1].bias, -0.1)

    def forward(self, decisions: Mapping[str, LocalDecision]) -> torch.Tensor:
        gains = self.gain_head(public_team_features(decisions))
        stop = torch.zeros_like(gains[:, :1])
        return torch.cat([stop, gains], dim=1)

    def choose(self, decisions: Mapping[str, LocalDecision]) -> torch.Tensor:
        return self(decisions).argmax(1)

    @staticmethod
    def validate_action(action: torch.Tensor, batch_size: int) -> torch.Tensor:
        if action.ndim != 1 or len(action) != batch_size:
            raise ValueError("route_action must have shape [batch]")
        action = action.long()
        if bool(((action < int(RouteAction.STOP))
                 | (action > int(RouteAction.P_FIRST))).any()):
            raise ValueError("route_action contains an unsupported action id")
        return action
