"""End-to-end Stage-6 system wrapper."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .agents import RoleAgents
from .contracts import DialogueOutput, LocalDecision
from .dialogue import SequentialDialogue


class Stage6System(nn.Module):
    def __init__(self, num_classes: int, *, state_dim: int = 128,
                 stem_channels: int = 128, intent_dim: int = 16,
                 message_dim: int = 32, query_dim: int = 32,
                 hidden_dim: int = 192, mode: str = "sequential_cf",
                 prototype_temperature: float = 0.15,
                 gate_temperature: float = 0.5,
                 query_cost: float = 0.05):
        super().__init__()
        self.agents = RoleAgents(
            num_classes, state_dim, stem_channels, intent_dim,
            prototype_temperature)
        self.dialogue = SequentialDialogue(
            num_classes, state_dim, intent_dim, message_dim, query_dim,
            hidden_dim=hidden_dim, mode=mode,
            gate_temperature=gate_temperature,
            query_cost=query_cost)
        self.num_classes = int(num_classes)
        self.mode = mode

    def local(self, iq: torch.Tensor) -> Dict[str, LocalDecision]:
        return self.agents(iq)

    def deliberate(self, decisions: Dict[str, LocalDecision], *, hard: bool = False,
                   intervention: str | None = None,
                   edge_override: torch.Tensor | None = None) -> DialogueOutput:
        return self.dialogue(decisions, hard=hard, intervention=intervention,
                             edge_override=edge_override)

    def forward(self, iq: torch.Tensor, *, hard: bool = False,
                intervention: str | None = None) -> DialogueOutput:
        return self.deliberate(self.local(iq), hard=hard, intervention=intervention)
