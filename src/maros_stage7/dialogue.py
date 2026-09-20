"""Fair B2 baseline and semantic proposal--challenge adjudication."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (ChallengePacket, ChallengeStance, DialogueOutput,
                        IndependentSummaryPacket, LocalDecision,
                        ProposalPacket, ResponsePacket, RouteAction)
from .router import public_team_features, validate_public_decisions


def _require_vector(name: str, value: torch.Tensor, batch: int,
                    device: torch.device) -> None:
    if not torch.is_tensor(value) or value.shape != (batch,):
        raise ValueError(f"{name} must have shape [batch]")
    if value.device != device:
        raise ValueError(f"{name} must be on the route_action device")


def _require_inactive_zero(name: str, value: torch.Tensor,
                           active: torch.Tensor) -> None:
    inactive = ~active
    if bool(inactive.any()) and bool(value[inactive].ne(0).any()):
        raise ValueError(f"inactive {name} must carry a zero payload")


def _validate_protocol(
    route_action: torch.Tensor,
    proposals: Tuple[ProposalPacket, ...],
    responses: Tuple[ResponsePacket, ...],
    *,
    batch: int,
    num_classes: int,
) -> None:
    """Fail closed on malformed, misdirected, or privacy-leaking packets."""

    if route_action.shape != (batch,):
        raise ValueError("route_action must have shape [batch]")
    if bool(((route_action < int(RouteAction.STOP))
             | (route_action > int(RouteAction.P_FIRST))).any()):
        raise ValueError("route_action contains an unsupported action id")
    if len(proposals) != len(responses):
        raise ValueError("each proposal must have exactly one response")
    if len(proposals) > 2:
        raise ValueError("one-round dialogue permits at most two routed packets")

    covered = torch.zeros(batch, dtype=torch.bool, device=route_action.device)
    seen_senders: set[str] = set()
    direction = {
        "waveform": ("prototype", RouteAction.W_FIRST),
        "prototype": ("waveform", RouteAction.P_FIRST),
    }
    for proposal, response in zip(proposals, responses):
        if proposal.sender not in direction:
            raise ValueError(f"unsupported proposal sender {proposal.sender!r}")
        expected_challenger, expected_action = direction[proposal.sender]
        if proposal.sender in seen_senders:
            raise ValueError("duplicate proposal direction in one-round dialogue")
        seen_senders.add(proposal.sender)
        if response.sender != expected_challenger:
            raise ValueError("response sender does not match proposal direction")
        if response.receiver != proposal.sender:
            raise ValueError("response receiver does not match proposal sender")

        proposal_fields = {
            "proposal.top1": proposal.top1,
            "proposal.top2": proposal.top2,
            "proposal.log_odds": proposal.log_odds,
            "proposal.local_unknown": proposal.local_unknown,
            "proposal.reliability": proposal.reliability,
            "proposal.active_mask": proposal.active_mask,
            "proposal.bit_cost": proposal.bit_cost,
        }
        if isinstance(response, ChallengePacket):
            response_fields = {
                "challenge.stance": response.stance,
                "challenge.alternative_class": response.alternative_class,
                "challenge.signed_pair_evidence": response.signed_pair_evidence,
                "challenge.open_tail_evidence": response.open_tail_evidence,
                "challenge.reliability": response.reliability,
                "challenge.active_mask": response.active_mask,
                "challenge.bit_cost": response.bit_cost,
            }
        elif isinstance(response, IndependentSummaryPacket):
            response_fields = {
                "summary.predicted_class": response.predicted_class,
                "summary.certainty_bin": response.certainty_bin,
                "summary.confidence_margin": response.confidence_margin,
                "summary.local_unknown": response.local_unknown,
                "summary.reliability": response.reliability,
                "summary.active_mask": response.active_mask,
                "summary.bit_cost": response.bit_cost,
            }
        else:
            raise TypeError("response must be conditional or independent packet")
        for name, value in {**proposal_fields, **response_fields}.items():
            _require_vector(name, value, batch, route_action.device)

        active = proposal.active_mask.bool()
        if not bool(active.any()):
            raise ValueError("empty message packets are not part of a transcript")
        if not torch.equal(active, response.active_mask.bool()):
            raise ValueError("proposal and response masks must be identical")
        expected_mask = route_action.eq(int(expected_action))
        if not torch.equal(active, expected_mask):
            raise ValueError("message activity does not match the routed direction")
        if bool((covered & active).any()):
            raise ValueError("a sample may execute only one communication direction")
        covered |= active

        active_top1, active_top2 = proposal.top1[active], proposal.top2[active]
        response_class = (response.alternative_class
                          if isinstance(response, ChallengePacket)
                          else response.predicted_class)
        for name, value in (
                ("proposal.top1", active_top1),
                ("proposal.top2", active_top2),
                ("response class", response_class[active])):
            if bool(((value < 0) | (value >= num_classes)).any()):
                raise ValueError(f"active {name} is outside class range")
        if bool(active_top1.eq(active_top2).any()):
            raise ValueError("active proposal top1 and top2 must be distinct")
        if isinstance(response, ChallengePacket):
            active_stance = response.stance[active]
            if bool(((active_stance < 0)
                     | (active_stance >= len(ChallengeStance))).any()):
                raise ValueError("active challenge stance is invalid")
            response_continuous = (
                response.signed_pair_evidence, response.open_tail_evidence,
                response.reliability, response.bit_cost)
            if bool(((response.open_tail_evidence[active] < 0)
                     | (response.open_tail_evidence[active] > 1)).any()):
                raise ValueError("open-tail evidence must lie in [0, 1]")
        else:
            certainty = response.certainty_bin[active]
            if bool(((certainty < 0) | (certainty > 3)).any()):
                raise ValueError("parallel certainty bin must lie in [0, 3]")
            if bool(((response.confidence_margin[active] < 0)
                     | (response.confidence_margin[active] > 1)).any()):
                raise ValueError("parallel confidence margin must lie in [0, 1]")
            response_continuous = (
                response.confidence_margin, response.local_unknown,
                response.reliability, response.bit_cost)

        finite_fields = (
            proposal.log_odds, proposal.local_unknown, proposal.reliability,
            proposal.bit_cost, *response_continuous)
        if any(not bool(torch.isfinite(value[active]).all())
               for value in finite_fields):
            raise ValueError("active message payload contains non-finite values")
        if bool(((proposal.reliability[active] < 0)
                 | (proposal.reliability[active] > 1)).any()):
            raise ValueError("proposal reliability must lie in [0, 1]")
        if bool(((response.reliability[active] < 0)
                 | (response.reliability[active] > 1)).any()):
            raise ValueError("response reliability must lie in [0, 1]")
        if bool((proposal.bit_cost[active] < 0).any()) or bool(
                (response.bit_cost[active] < 0).any()):
            raise ValueError("message bit cost cannot be negative")

        for name, value in {**proposal_fields, **response_fields}.items():
            _require_inactive_zero(name, value, active)

    if not torch.equal(covered, route_action.ne(int(RouteAction.STOP))):
        raise ValueError("every query action requires exactly one dialogue edge")


class FairB2Fusion(nn.Module):
    """Anchored late fusion restricted to public local decisions.

    The original Stage-7 implementation learned an unconstrained ``open_head``
    and therefore discarded both Agents' independently trained rejection
    scores.  A head fitted to synthetic PUG samples could look excellent on the
    training objective while catastrophically re-ordering real class-held-out
    unknowns.  Here both the class logits *and* the unknown logit are convex
    mixtures of local decisions.

    The best global local Agent on the leakage-safe B2 fitting episode is used
    as an anchor.  A learned public-summary selector may spend only
    ``max_adaptive_mass`` away from that anchor.  Training may finally disable
    adaptation, in which case this module is exactly the anchored local Agent;
    this gives the caller an auditable no-regret fallback rather than merely a
    loss penalty that can still fail at inference.
    """

    AGENT_NAMES = ("waveform", "prototype")

    def __init__(self, public_summary_dim: int = 7, hidden_dim: int = 96,
                 max_adaptive_mass: float = 0.35,
                 initial_adaptive_fraction: float = 0.2):
        super().__init__()
        if not 0.0 <= max_adaptive_mass <= 1.0:
            raise ValueError("max_adaptive_mass must lie in [0, 1]")
        if not 0.0 < initial_adaptive_fraction < 1.0:
            raise ValueError("initial_adaptive_fraction must lie in (0, 1)")
        input_dim = 2 * int(public_summary_dim) + 3
        self.weight_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 2),
        )
        # Kept under the historical name for checkpoint/API readability.  It
        # now predicts a *mixture gate*, never a replacement unknown score.
        self.open_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        initial_logit = math.log(
            float(initial_adaptive_fraction)
            / (1.0 - float(initial_adaptive_fraction)))
        self.class_adaptive_logit = nn.Parameter(torch.tensor(initial_logit))
        self.open_adaptive_logit = nn.Parameter(torch.tensor(initial_logit))
        self.max_adaptive_mass = float(max_adaptive_mass)
        self.initial_adaptive_fraction = float(initial_adaptive_fraction)
        self.register_buffer("anchor_distribution", torch.tensor([0.5, 0.5]))
        self.register_buffer("anchor_index_tensor", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("adaptive_enabled", torch.tensor(True, dtype=torch.bool))

    @property
    def anchor_index(self) -> int | None:
        value = int(self.anchor_index_tensor.item())
        return value if value >= 0 else None

    @property
    def anchor_name(self) -> str | None:
        index = self.anchor_index
        return None if index is None else self.AGENT_NAMES[index]

    @torch.no_grad()
    def set_anchor(self, agent: str | int) -> None:
        """Set the leakage-safe global fallback expert.

        This is deliberately a single expert for both identity and rejection:
        disabling adaptation then reproduces one independently valid local
        system exactly, rather than constructing an unvalidated hybrid.
        """

        if isinstance(agent, str):
            if agent not in self.AGENT_NAMES:
                raise ValueError(f"unsupported B2 anchor Agent: {agent!r}")
            index = self.AGENT_NAMES.index(agent)
        else:
            index = int(agent)
            if index not in (0, 1):
                raise ValueError("B2 anchor index must be 0 or 1")
        self.anchor_distribution.zero_()
        self.anchor_distribution[index] = 1.0
        self.anchor_index_tensor.fill_(index)

    @torch.no_grad()
    def set_adaptive_enabled(self, enabled: bool) -> None:
        """Enable the learned residual or execute the exact local fallback."""

        self.adaptive_enabled.fill_(bool(enabled))

    def mixture_weights(
        self, decisions: Mapping[str, LocalDecision]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return public-only convex weights for class and open evidence."""

        feature = public_team_features(decisions)
        candidate_class = F.softmax(self.weight_head(feature), dim=1)
        waveform_open = torch.sigmoid(self.open_head(feature).squeeze(1))
        candidate_open = torch.stack([waveform_open, 1.0 - waveform_open], dim=1)
        anchor = self.anchor_distribution.to(feature).expand(len(feature), -1)
        enabled = self.adaptive_enabled.to(feature)
        class_mass = (
            self.max_adaptive_mass * torch.sigmoid(self.class_adaptive_logit)
            * enabled)
        open_mass = (
            self.max_adaptive_mass * torch.sigmoid(self.open_adaptive_logit)
            * enabled)
        class_weights = (1.0 - class_mass) * anchor + class_mass * candidate_class
        open_weights = (1.0 - open_mass) * anchor + open_mass * candidate_open
        return class_weights, open_weights

    def forward(self, decisions: Mapping[str, LocalDecision]
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, open_weights = self.mixture_weights(decisions)
        waveform = decisions["waveform"].class_logits
        prototype = decisions["prototype"].class_logits
        logits = weights[:, :1] * waveform + weights[:, 1:] * prototype
        unknown = (
            open_weights[:, 0] * decisions["waveform"].unknown_score
            + open_weights[:, 1] * decisions["prototype"].unknown_score)
        return logits, unknown, weights


class SemanticAdjudicator(nn.Module):
    """Update B2 solely from transmitted semantic packet fields.

    The module has no argument through which an Agent state, token, embedding,
    prototype, or distance vector could be supplied.
    """

    def __init__(self, public_summary_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        public_dim = 2 * int(public_summary_dim) + 3
        # The semantic fields are: signed pair evidence, tail evidence,
        # responder reliability, proposer reliability, and local unknown.
        semantic_dim = 5 + len(ChallengeStance)
        self.weight_delta = nn.Sequential(
            nn.LayerNorm(public_dim + semantic_dim),
            nn.Linear(public_dim + semantic_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.tail_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.stance_scale_raw = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def independent_to_challenge(
        proposal: ProposalPacket,
        summary: IndependentSummaryPacket,
    ) -> ChallengePacket:
        """Interpret two independently produced packets after transmission.

        This conversion lives in the non-Agent adjudication service.  The
        responder has already committed to its class, uncertainty and margin,
        so it cannot condition those quantities on the proposal pair.
        """

        agreement = summary.predicted_class.eq(proposal.top1)
        signed = torch.where(
            agreement, summary.confidence_margin, -summary.confidence_margin)
        tail = torch.sigmoid(summary.local_unknown)
        unknown_suspect = summary.certainty_bin.ge(2)
        stance = torch.where(
            unknown_suspect,
            torch.full_like(summary.certainty_bin,
                            int(ChallengeStance.UNKNOWN_SUSPECT)),
            torch.where(
                agreement,
                torch.full_like(summary.certainty_bin,
                                int(ChallengeStance.SUPPORT)),
                torch.full_like(summary.certainty_bin,
                                int(ChallengeStance.CHALLENGE)),
            ),
        )
        active = summary.active_mask.bool()
        return ChallengePacket(
            sender=summary.sender,
            receiver=summary.receiver,
            stance=torch.where(active, stance, torch.zeros_like(stance)),
            alternative_class=torch.where(
                active, summary.predicted_class,
                torch.zeros_like(summary.predicted_class)),
            signed_pair_evidence=torch.where(
                active, signed, torch.zeros_like(signed)),
            open_tail_evidence=torch.where(
                active, tail, torch.zeros_like(tail)),
            reliability=summary.reliability,
            active_mask=active,
            bit_cost=summary.bit_cost,
        )

    def forward(self, decisions: Mapping[str, LocalDecision],
                proposal: ProposalPacket, challenge: ChallengePacket,
                route_action: torch.Tensor, base_logits: torch.Tensor,
                base_unknown: torch.Tensor, base_weights: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        public = public_team_features(decisions)
        stance = F.one_hot(
            challenge.stance.long(), num_classes=len(ChallengeStance)).to(public)
        semantic = torch.cat([
            challenge.signed_pair_evidence[:, None],
            challenge.open_tail_evidence[:, None],
            challenge.reliability[:, None],
            proposal.reliability[:, None],
            proposal.local_unknown[:, None], stance,
        ], dim=1)
        learned_delta = self.weight_delta(torch.cat([public, semantic], dim=1)).squeeze(1)
        # Positive signed evidence supports the leader.  Convert this into a
        # logit offset for the Waveform mixture weight, reversing the sign when
        # Prototype leads.  The result remains a convex two-Agent combination.
        support = challenge.signed_pair_evidence * challenge.reliability
        leader_delta = learned_delta + support
        waveform_delta = torch.where(
            route_action.eq(int(RouteAction.W_FIRST)), leader_delta, -leader_delta)
        # The safe B2 fallback may be exactly one-hot.  Keep STOP exact, but
        # give an *active, paid* consultation a finite prior for the other
        # expert; otherwise log(1e-8) makes the responder's gradient and any
        # realistic semantic evidence effectively unable to overturn the
        # anchor.  This smoothing exists only inside the communicated branch.
        communication_prior = base_weights.add(0.02)
        communication_prior = communication_prior / communication_prior.sum(
            dim=1, keepdim=True)
        base_weight_logits = communication_prior.log()
        adjusted_weight_logits = base_weight_logits + torch.stack([
            waveform_delta, -waveform_delta,
        ], dim=1)
        weights = F.softmax(adjusted_weight_logits, dim=1)
        candidate_logits = (
            weights[:, :1] * decisions["waveform"].class_logits
            + weights[:, 1:] * decisions["prototype"].class_logits)

        pair_scale = F.softplus(self.pair_scale_raw)
        tail_scale = F.softplus(self.tail_scale_raw)
        stance_scale = F.softplus(self.stance_scale_raw)
        stance_risk = torch.where(
            challenge.stance.eq(int(ChallengeStance.SUPPORT)),
            -torch.ones_like(base_unknown),
            torch.where(
                challenge.stance.eq(int(ChallengeStance.CHALLENGE)),
                torch.ones_like(base_unknown),
                2.0 * torch.ones_like(base_unknown),
            ),
        )
        # This auditable certificate explicitly permits support to lower risk
        # and challenge/tail evidence to raise it.
        unknown_delta = (
            -pair_scale * challenge.signed_pair_evidence
            + tail_scale * challenge.open_tail_evidence
            + stance_scale * stance_risk
        ) * challenge.reliability
        candidate_unknown = base_unknown + unknown_delta

        active = challenge.active_mask.bool()
        logits = torch.where(active[:, None], candidate_logits, base_logits)
        unknown = torch.where(active, candidate_unknown, base_unknown)
        weights = torch.where(active[:, None], weights, base_weights)
        return logits, unknown, weights


class ConditionalDialogue(nn.Module):
    """Resolve a one-round transcript while preserving exact STOP identity."""

    def __init__(self, public_summary_dim: int = 7, hidden_dim: int = 96,
                 b2_max_adaptive_mass: float = 0.35,
                 b2_initial_adaptive_fraction: float = 0.2):
        super().__init__()
        self.b2_fusion = FairB2Fusion(
            public_summary_dim, hidden_dim,
            max_adaptive_mass=b2_max_adaptive_mass,
            initial_adaptive_fraction=b2_initial_adaptive_fraction)
        self.adjudicator = SemanticAdjudicator(
            public_summary_dim, max(hidden_dim // 2, 32))

    def baseline(self, decisions: Mapping[str, LocalDecision]
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.b2_fusion(decisions)

    def forward(self, decisions: Mapping[str, LocalDecision],
                route_action: torch.Tensor,
                proposals: Tuple[ProposalPacket, ...] = (),
                challenges: Tuple[ResponsePacket, ...] = (),
                route_logits: torch.Tensor | None = None) -> DialogueOutput:
        batch, num_classes, _ = validate_public_decisions(decisions)
        route_action = route_action.to(
            device=decisions["waveform"].class_logits.device, dtype=torch.long)
        _validate_protocol(
            route_action, proposals, challenges,
            batch=batch, num_classes=num_classes)
        base_logits, base_unknown, base_weights = self.baseline(decisions)
        final_logits = base_logits
        final_unknown = base_unknown
        final_weights = base_weights
        total_bits = base_unknown.new_zeros(len(base_unknown))

        for proposal, response in zip(proposals, challenges):
            challenge = (response if isinstance(response, ChallengePacket)
                         else self.adjudicator.independent_to_challenge(
                             proposal, response))
            candidate_logits, candidate_unknown, candidate_weights = self.adjudicator(
                decisions, proposal, challenge, route_action,
                base_logits, base_unknown, base_weights)
            active = challenge.active_mask.bool()
            final_logits = torch.where(active[:, None], candidate_logits, final_logits)
            final_unknown = torch.where(active, candidate_unknown, final_unknown)
            final_weights = torch.where(active[:, None], candidate_weights, final_weights)
            total_bits = total_bits + proposal.bit_cost + response.bit_cost

        # STOP is the exact fair B2 parent, including a literal zero bit cost.
        stop = route_action.eq(int(RouteAction.STOP))
        final_logits = torch.where(stop[:, None], base_logits, final_logits)
        final_unknown = torch.where(stop, base_unknown, final_unknown)
        final_weights = torch.where(stop[:, None], base_weights, final_weights)
        total_bits = torch.where(stop, torch.zeros_like(total_bits), total_bits)
        public = dict(decisions)
        transcript = tuple(
            packet for pair in zip(proposals, challenges) for packet in pair)
        return DialogueOutput(
            route_action=route_action,
            local_decisions=public,
            transcript=transcript,
            fused_class_logits=final_logits,
            unknown_score=final_unknown,
            known_weights=final_weights,
            bit_cost=total_bits,
            route_logits=route_logits,
        )
