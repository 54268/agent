"""Public contracts for the Stage-6 communicating agents.

The contracts deliberately distinguish private evidence from transmitted
messages.  A verifier may consume ``intent`` and ``MessagePacket.payload`` but
must never inspect another agent's ``private_evidence``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class LocalDecision:
    state: torch.Tensor
    class_logits: torch.Tensor
    unknown_logit: torch.Tensor
    reliability: torch.Tensor
    intent: torch.Tensor
    private_evidence: Dict[str, torch.Tensor]


@dataclass
class QueryAction:
    leader_id: torch.Tensor
    responder_mask: torch.Tensor
    query_token: torch.Tensor
    expected_gain: torch.Tensor
    communication_cost: torch.Tensor


@dataclass
class MessagePacket:
    sender: str
    receiver: str
    message_type: str
    payload: torch.Tensor
    feature_mask: torch.Tensor
    bit_cost: torch.Tensor


@dataclass
class DialogueOutput:
    local_decisions: Dict[str, LocalDecision]
    leader_id: torch.Tensor
    query_actions: QueryAction
    transcript: torch.Tensor
    messages: Tuple[MessagePacket, ...]
    fused_class_logits: torch.Tensor
    unknown_logit: torch.Tensor
    edge_gates: torch.Tensor
    known_weights: torch.Tensor
    counterfactual_gains: torch.Tensor | None = None

    @property
    def unknown_score(self) -> torch.Tensor:
        return torch.sigmoid(self.unknown_logit)
