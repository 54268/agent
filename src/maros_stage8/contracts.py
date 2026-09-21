"""Public, low-bandwidth contracts for Stage-8.

No object in this module contains an encoder state, token sequence, prototype
bank, enrollment tail, or raw observation.  These are the only values allowed
to cross an Agent boundary or enter a future Coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class QueryType(IntEnum):
    VERIFY_TEMPORAL_PAIR = 0
    VERIFY_SPECTRAL_PAIR = 1
    VERIFY_ENROLLMENT_COMPATIBILITY = 2
    AUDIT_OPEN_SPACE_STABILITY = 3


class EvidenceStance(IntEnum):
    SUPPORT_A = 0
    SUPPORT_B = 1
    UNKNOWN_SUSPECT = 2
    ABSTAIN = 3


@dataclass(frozen=True)
class LocalDecision:
    """An independently auditable local prediction with public scalars only."""

    sender: str
    class_logits: torch.Tensor
    unknown_score: torch.Tensor
    top1: torch.Tensor
    top2: torch.Tensor
    public_diagnostics: torch.Tensor

    def __post_init__(self) -> None:
        if self.sender not in {"temporal", "spectral"}:
            raise ValueError("sender must be temporal or spectral")
        if self.class_logits.ndim != 2 or self.class_logits.shape[1] < 2:
            raise ValueError("class_logits must have shape [batch, classes>=2]")
        batch = self.class_logits.shape[0]
        for name in ("unknown_score", "top1", "top2"):
            value = getattr(self, name)
            if not torch.is_tensor(value) or value.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if (not torch.is_tensor(self.public_diagnostics)
                or self.public_diagnostics.ndim != 2
                or self.public_diagnostics.shape[0] != batch):
            raise ValueError("public_diagnostics must have shape [batch, fields]")


@dataclass(frozen=True)
class QueryPacket:
    """A proposal-conditioned request for one unavailable evidence domain."""

    sender: str
    receiver: str
    candidate_a: torch.Tensor
    candidate_b: torch.Tensor
    query_type: QueryType
    reason_code: torch.Tensor
    local_pair_support: torch.Tensor
    local_open_risk: torch.Tensor
    active_mask: torch.Tensor
    bit_cost: torch.Tensor


@dataclass(frozen=True)
class EvidenceCertificate:
    """Candidate-bound semantic evidence returned by a specialist."""

    sender: str
    receiver: str
    candidate_a: torch.Tensor
    candidate_b: torch.Tensor
    stance: torch.Tensor
    signed_candidate_evidence: torch.Tensor
    alternative_class: torch.Tensor
    domain_quality: torch.Tensor
    open_risk: torch.Tensor
    reliability: torch.Tensor
    abstain: torch.Tensor
    active_mask: torch.Tensor
    bit_cost: torch.Tensor


@dataclass(frozen=True)
class CandidateDecision:
    """Auditable candidate table after zero or more certificates."""

    candidate_scores: torch.Tensor
    open_risk: torch.Tensor
    prediction: torch.Tensor
    contributions: tuple[torch.Tensor, ...]
    certificates: tuple[EvidenceCertificate, ...]

