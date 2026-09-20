"""Typed contracts shared by the Stage-5 agents and coordinator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class AgentOutput:
    state: torch.Tensor
    class_logits: torch.Tensor
    unknown_evidence: torch.Tensor
    reliability: torch.Tensor
    message: torch.Tensor
    evidence: Dict[str, torch.Tensor]


@dataclass
class MultiAgentOutput:
    fused_class_logits: torch.Tensor
    unknown_logit: torch.Tensor
    edge_gates: torch.Tensor
    agent_outputs: Dict[str, AgentOutput]
    known_weights: torch.Tensor
    counterfactual_deltas: torch.Tensor | None = None

    @property
    def unknown_score(self) -> torch.Tensor:
        return torch.sigmoid(self.unknown_logit)
