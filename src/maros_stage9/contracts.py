"""Stage-9 capability and local-belief contracts.

Public packets contain candidate-specific scalar evidence, never embeddings,
feature maps, full class logits, prototypes, or enrollment memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class ReasonCode(IntEnum):
    CLEAR_LOCAL_EVIDENCE = 0
    LOW_MARGIN = 1
    TOOL_DISAGREEMENT = 2
    MISSING_IMPAIRMENT_EVIDENCE = 3
    OUT_OF_SUPPORT = 4


class RequestSuggestion(IntEnum):
    REPORT = 0
    ASK_IDENTITY = 1
    ASK_IMPAIRMENT = 2
    ASK_OPEN_SET = 3
    ABSTAIN = 4


@dataclass(frozen=True)
class CapabilityManifest:
    owner: str
    allowed_tools: frozenset[str]

    def __post_init__(self) -> None:
        if not self.owner or not self.allowed_tools:
            raise ValueError("capability manifest requires an owner and tools")


@dataclass(frozen=True)
class BeliefPacket:
    """Only the top candidate pair and public semantic scalars leave an Agent."""

    sender: str
    candidate_a: torch.Tensor
    candidate_b: torch.Tensor
    evidence_a: torch.Tensor
    evidence_b: torch.Tensor
    uncertainty: torch.Tensor
    open_risk_opinion: torch.Tensor | None
    used_tools: tuple[tuple[str, ...], ...]
    reason_code: torch.Tensor
    request_suggestion: torch.Tensor
    abstain: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.candidate_a.shape[0]
        if not self.sender or len(self.used_tools) != batch:
            raise ValueError("belief sender and per-sample used tools are required")
        for name in ("candidate_a", "candidate_b", "evidence_a", "evidence_b",
                     "uncertainty", "reason_code", "request_suggestion", "abstain"):
            field = getattr(self, name)
            if not torch.is_tensor(field) or field.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if self.open_risk_opinion is not None and self.open_risk_opinion.shape != (batch,):
            raise ValueError("open-risk opinion must have shape [batch]")

