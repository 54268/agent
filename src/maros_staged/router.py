"""Reliability-aware routing between isolated and communicated agent states."""
from __future__ import annotations

from typing import Dict, Optional
import math

import torch
from torch import nn
from torch.nn import functional as F


def _prediction_summary(logits: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(logits, dim=1)
    top2 = probability.topk(2, dim=1).values
    entropy = -(probability * F.log_softmax(logits, dim=1)).sum(dim=1)
    entropy = entropy / max(math.log(logits.shape[1]), 1e-6)
    return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy], dim=1)


def routing_observation(no_comm: Dict[str, torch.Tensor],
                        communication: Dict[str, torch.Tensor],
                        local_evidence: torch.Tensor) -> torch.Tensor:
    """Observable reliability state; base model outputs are treated as frozen."""
    p_no = F.softmax(no_comm["fused_logits"], dim=1)
    p_comm = F.softmax(communication["fused_logits"], dim=1)
    midpoint = 0.5 * (p_no + p_comm)
    js = 0.5 * (
        (p_no * (torch.log(p_no.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(1)
        + (p_comm * (torch.log(p_comm.clamp_min(1e-8)) - torch.log(midpoint.clamp_min(1e-8)))).sum(1)
    )
    state_delta = torch.linalg.vector_norm(
        communication["tokens_post"] - communication["tokens_pre"], dim=2
    )
    return torch.cat([
        _prediction_summary(no_comm["fused_logits"]),
        _prediction_summary(communication["fused_logits"]),
        no_comm["unknown_logit"][:, None],
        communication["unknown_logit"][:, None],
        js[:, None],
        state_delta,
        communication["adjacency"].flatten(1),
        local_evidence.flatten(1),
    ], dim=1)


class ReliabilityRouter(nn.Module):
    """Global, sample-shared, or dual sample-adaptive routing.

    The Router cannot create a new class decision.  It only convex-combines
    logits produced by the isolated and communicated systems.
    """

    def __init__(self, observation_dim: int, mode: str = "dual", hidden_dim: int = 64):
        super().__init__()
        if mode not in {"global", "single", "dual"}:
            raise ValueError(f"unknown router mode={mode}")
        self.mode = mode
        if mode == "global":
            self.known_logits = nn.Parameter(torch.zeros(2))
            self.unknown_logits = nn.Parameter(torch.zeros(2))
        else:
            self.encoder = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
            self.known_head = nn.Linear(hidden_dim, 2)
            self.unknown_head = self.known_head if mode == "single" else nn.Linear(hidden_dim, 2)
        self.unknown_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.unknown_bias = nn.Parameter(torch.tensor(0.0))

    def weights(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.mode == "global":
            batch = observation.shape[0]
            known = F.softmax(self.known_logits, dim=0)[None].expand(batch, -1)
            unknown = F.softmax(self.unknown_logits, dim=0)[None].expand(batch, -1)
        else:
            state = self.encoder(observation)
            known = F.softmax(self.known_head(state), dim=1)
            unknown = F.softmax(self.unknown_head(state), dim=1)
        return {"known": known, "unknown": unknown}

    def forward(self, observation: torch.Tensor, no_comm: Dict[str, torch.Tensor],
                communication: Dict[str, torch.Tensor],
                override: Optional[str] = None) -> Dict[str, torch.Tensor]:
        route = self.weights(observation)
        if override == "uniform":
            route = {key: torch.full_like(value, 0.5) for key, value in route.items()}
        elif override == "shuffle":
            index = torch.arange(observation.shape[0] - 1, -1, -1, device=observation.device)
            route = {key: value[index] for key, value in route.items()}
        known_sources = torch.stack([no_comm["fused_logits"], communication["fused_logits"]], dim=1)
        unknown_sources = torch.stack([no_comm["unknown_logit"], communication["unknown_logit"]], dim=1)
        known_logits = (route["known"][:, :, None] * known_sources).sum(dim=1)
        unknown_logit = (route["unknown"] * unknown_sources).sum(dim=1)
        unknown_logit = F.softplus(self.unknown_scale_raw) * unknown_logit + self.unknown_bias
        return {"known_logits": known_logits, "unknown_logit": unknown_logit,
                "known_weights": route["known"], "unknown_weights": route["unknown"]}
