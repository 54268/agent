"""Typed public protocol for Stage-7 conditional consultation.

Only the values defined in this module may cross an Agent boundary.  In
particular, :class:`LocalDecision` deliberately contains neither an encoder
state nor an embedding.  The private state needed to answer a consultation is
kept by ``LocalContext`` (in :mod:`maros_stage7.agents`) behind a capability
owned by ``Stage7System``.

``unknown_score`` is an *unbounded rejection logit*.  This keeps the contract
compatible with ``binary_cross_entropy_with_logits`` and avoids silently
applying sigmoid twice during exact counterfactual supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple, Union

import torch


class RouteAction(IntEnum):
    """The complete Stage-7 routing action space."""

    STOP = 0
    W_FIRST = 1
    P_FIRST = 2


class ChallengeStance(IntEnum):
    """Two-bit semantic answer emitted by a conditional challenger."""

    SUPPORT = 0
    CHALLENGE = 1
    UNKNOWN_SUSPECT = 2


@dataclass(frozen=True)
class LocalDecision:
    """An Agent's independently valid, public local decision.

    ``public_summary`` contains calibrated scalar diagnostics only.  It must
    never contain a latent state, token sequence, or class prototype.
    """

    class_logits: torch.Tensor
    unknown_score: torch.Tensor
    reliability: torch.Tensor
    top1: torch.Tensor
    top2: torch.Tensor
    public_summary: torch.Tensor

    @property
    def unknown_logit(self) -> torch.Tensor:
        """Alias retained for callers that explicitly name the logit."""

        return self.unknown_score

    @property
    def unknown_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.unknown_score)


@dataclass(frozen=True)
class ProposalPacket:
    """Low-bandwidth identity hypothesis sent by the route leader."""

    sender: str
    top1: torch.Tensor
    top2: torch.Tensor
    log_odds: torch.Tensor
    local_unknown: torch.Tensor
    reliability: torch.Tensor
    active_mask: torch.Tensor
    bit_cost: torch.Tensor


@dataclass(frozen=True)
class ChallengePacket:
    """Semantic response to one concrete ``(top1, top2)`` proposal."""

    sender: str
    receiver: str
    stance: torch.Tensor
    alternative_class: torch.Tensor
    signed_pair_evidence: torch.Tensor
    open_tail_evidence: torch.Tensor
    reliability: torch.Tensor
    active_mask: torch.Tensor
    bit_cost: torch.Tensor


@dataclass(frozen=True)
class IndependentSummaryPacket:
    """Equal-bandwidth parallel response produced without seeing a proposal.

    The two-bit ``certainty_bin`` occupies the same wire budget as the stance
    in :class:`ChallengePacket`.  The adjudicator may compare this independently
    generated class summary with a leader proposal, but the responding Agent
    cannot condition its computation on the leader's class pair.
    """

    sender: str
    receiver: str
    predicted_class: torch.Tensor
    certainty_bin: torch.Tensor
    confidence_margin: torch.Tensor
    local_unknown: torch.Tensor
    reliability: torch.Tensor
    active_mask: torch.Tensor
    bit_cost: torch.Tensor


ResponsePacket = Union[ChallengePacket, IndependentSummaryPacket]
TranscriptPacket = Union[
    ProposalPacket, ChallengePacket, IndependentSummaryPacket]


@dataclass
class DialogueOutput:
    """Result of one STOP or one-round proposal--challenge dialogue."""

    route_action: torch.Tensor
    local_decisions: Dict[str, LocalDecision]
    transcript: Tuple[TranscriptPacket, ...]
    fused_class_logits: torch.Tensor
    unknown_score: torch.Tensor
    known_weights: torch.Tensor
    bit_cost: torch.Tensor
    route_logits: torch.Tensor | None = None
    counterfactual_action_losses: torch.Tensor | None = None

    @property
    def unknown_logit(self) -> torch.Tensor:
        return self.unknown_score

    @property
    def unknown_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.unknown_score)

    @property
    def communicated(self) -> torch.Tensor:
        return self.route_action.ne(int(RouteAction.STOP))
