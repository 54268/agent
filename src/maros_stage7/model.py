"""End-to-end wrapper for the Stage-7 conditional consultation system."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Tuple

import torch
from torch import nn

from .agents import LocalContext, RoleAgents
from .contracts import (ChallengePacket, DialogueOutput,
                        IndependentSummaryPacket, LocalDecision,
                        ProposalPacket, ResponsePacket, RouteAction)
from .dialogue import ConditionalDialogue
from .router import ValueOfInformationRouter


class Stage7System(nn.Module):
    """Two heterogeneous Agents plus a non-Agent router/adjudication service."""

    def __init__(self, num_classes: int, *, state_dim: int = 128,
                 stem_channels: int = 128, class_embed_dim: int = 24,
                 complex_hidden: int = 64, hidden_dim: int = 96,
                 prototype_temperature: float = 0.15,
                 scalar_bits: int = 16,
                 b2_max_adaptive_mass: float = 0.35,
                 b2_initial_adaptive_fraction: float = 0.2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Stage-7 requires at least two Known classes")
        if scalar_bits <= 0:
            raise ValueError("scalar_bits must be positive")
        self._private_capability = object()
        self.agents = RoleAgents(
            num_classes=num_classes,
            capability=self._private_capability,
            state_dim=state_dim,
            stem_channels=stem_channels,
            class_embed_dim=class_embed_dim,
            complex_hidden=complex_hidden,
            prototype_temperature=prototype_temperature,
            scalar_bits=scalar_bits,
        )
        self.router = ValueOfInformationRouter(7, hidden_dim)
        self.dialogue = ConditionalDialogue(
            7, hidden_dim,
            b2_max_adaptive_mass=b2_max_adaptive_mass,
            b2_initial_adaptive_fraction=b2_initial_adaptive_fraction)
        self.num_classes = int(num_classes)
        self.scalar_bits = int(scalar_bits)

    def local(self, iq: torch.Tensor) -> LocalContext:
        return self.agents(iq)

    def _validate_class_count(
        self, decisions: Mapping[str, LocalDecision]
    ) -> None:
        if decisions["waveform"].class_logits.shape[-1] != self.num_classes:
            raise ValueError("local decisions do not match this system's class count")

    def route_logits(self, decisions: Mapping[str, LocalDecision]) -> torch.Tensor:
        self._validate_class_count(decisions)
        return self.router(decisions)

    def _proposal(self, decision: LocalDecision, sender: str,
                  active: torch.Tensor) -> ProposalPacket:
        probability = torch.softmax(decision.class_logits, dim=-1)
        p1 = probability.gather(1, decision.top1[:, None]).squeeze(1)
        p2 = probability.gather(1, decision.top2[:, None]).squeeze(1)
        log_odds = torch.log(p1.clamp_min(1e-8)) - torch.log(p2.clamp_min(1e-8))
        class_bits = max(1, (self.num_classes - 1).bit_length())
        fixed_bits = 2 * class_bits + 3 * self.scalar_bits
        bit_cost = active.to(log_odds) * float(fixed_bits)
        # A batched packet is a sparse wire representation.  Samples which did
        # not take this edge must carry no payload, rather than merely claiming
        # zero cost while exposing their local decision.
        active = active.bool()
        top1 = torch.where(active, decision.top1, torch.zeros_like(decision.top1))
        top2 = torch.where(active, decision.top2, torch.zeros_like(decision.top2))
        log_odds = torch.where(active, log_odds, torch.zeros_like(log_odds))
        local_unknown = torch.where(
            active, decision.unknown_score,
            torch.zeros_like(decision.unknown_score))
        reliability = torch.where(
            active, decision.reliability,
            torch.zeros_like(decision.reliability))
        return ProposalPacket(
            sender=sender, top1=top1, top2=top2,
            log_odds=log_odds, local_unknown=local_unknown,
            reliability=reliability, active_mask=active,
            bit_cost=bit_cost,
        )

    @staticmethod
    def _shuffle_response(packet: ResponsePacket) -> ResponsePacket:
        active_indices = packet.active_mask.bool().nonzero(as_tuple=False).flatten()
        if len(active_indices) < 2:
            return packet
        order = torch.arange(
            len(packet.active_mask), device=packet.active_mask.device)
        order[active_indices] = torch.roll(active_indices, shifts=1)
        if isinstance(packet, ChallengePacket):
            return ChallengePacket(
                sender=packet.sender, receiver=packet.receiver,
                stance=packet.stance[order],
                alternative_class=packet.alternative_class[order],
                signed_pair_evidence=packet.signed_pair_evidence[order],
                open_tail_evidence=packet.open_tail_evidence[order],
                reliability=packet.reliability[order],
                active_mask=packet.active_mask,
                bit_cost=packet.bit_cost,
            )
        return IndependentSummaryPacket(
            sender=packet.sender, receiver=packet.receiver,
            predicted_class=packet.predicted_class[order],
            certainty_bin=packet.certainty_bin[order],
            confidence_margin=packet.confidence_margin[order],
            local_unknown=packet.local_unknown[order],
            reliability=packet.reliability[order],
            active_mask=packet.active_mask,
            bit_cost=packet.bit_cost,
        )

    def _independent_summary(
        self,
        decision: LocalDecision,
        *,
        sender: str,
        receiver: str,
        active: torch.Tensor,
    ) -> IndependentSummaryPacket:
        """Create a parallel response without reading the leader proposal."""

        active = active.bool()
        margin = decision.public_summary[:, 2].clamp(0.0, 1.0)
        unknown_probability = torch.sigmoid(decision.unknown_score)
        certainty = torch.where(
            unknown_probability >= 0.75,
            torch.full_like(decision.top1, 3),
            torch.where(
                unknown_probability >= 0.5,
                torch.full_like(decision.top1, 2),
                torch.where(
                    margin < 0.15,
                    torch.ones_like(decision.top1),
                    torch.zeros_like(decision.top1))),
        )
        class_bits = max(1, (self.num_classes - 1).bit_length())
        fixed_bits = 2 + class_bits + 3 * self.scalar_bits
        bit_cost = active.to(margin) * float(fixed_bits)
        return IndependentSummaryPacket(
            sender=sender,
            receiver=receiver,
            predicted_class=torch.where(
                active, decision.top1, torch.zeros_like(decision.top1)),
            certainty_bin=torch.where(
                active, certainty, torch.zeros_like(certainty)),
            confidence_margin=torch.where(
                active, margin, torch.zeros_like(margin)),
            local_unknown=torch.where(
                active, decision.unknown_score,
                torch.zeros_like(decision.unknown_score)),
            reliability=torch.where(
                active, decision.reliability,
                torch.zeros_like(decision.reliability)),
            active_mask=active,
            bit_cost=bit_cost,
        )

    def _wrong_pair(self, proposal: ProposalPacket) -> ProposalPacket:
        wrong_top1 = (proposal.top1 + 1) % self.num_classes
        wrong_top2 = (proposal.top2 + 2) % self.num_classes
        collision = wrong_top1.eq(wrong_top2)
        wrong_top2 = torch.where(collision,
                                 (wrong_top2 + 1) % self.num_classes,
                                 wrong_top2)
        active = proposal.active_mask.bool()
        wrong_top1 = torch.where(active, wrong_top1, torch.zeros_like(wrong_top1))
        wrong_top2 = torch.where(active, wrong_top2, torch.zeros_like(wrong_top2))
        return ProposalPacket(
            sender=proposal.sender, top1=wrong_top1, top2=wrong_top2,
            log_odds=proposal.log_odds,
            local_unknown=proposal.local_unknown,
            reliability=proposal.reliability,
            active_mask=active,
            bit_cost=proposal.bit_cost,
        )

    def _messages(self, decisions: Mapping[str, LocalDecision], action: torch.Tensor,
                  intervention: str | None,
                  protocol: str = "sequential",
                  ) -> tuple[Tuple[ProposalPacket, ...], Tuple[ResponsePacket, ...]]:
        waveform_active = action.eq(int(RouteAction.W_FIRST))
        prototype_active = action.eq(int(RouteAction.P_FIRST))
        proposals = []
        responses = []
        if bool(waveform_active.any()):
            proposal = self._proposal(decisions["waveform"], "waveform", waveform_active)
            if protocol == "sequential":
                response_proposal = (self._wrong_pair(proposal)
                                     if intervention == "wrong_class_pair"
                                     else proposal)
                private = decisions._private_for(
                    "prototype", self._private_capability)
                response = self.agents.prototype.challenge(
                    private, response_proposal)
            else:
                response = self._independent_summary(
                    decisions["prototype"], sender="prototype",
                    receiver="waveform", active=waveform_active)
            proposals.append(proposal)
            responses.append(response)
        if bool(prototype_active.any()):
            proposal = self._proposal(decisions["prototype"], "prototype", prototype_active)
            if protocol == "sequential":
                response_proposal = (self._wrong_pair(proposal)
                                     if intervention == "wrong_class_pair"
                                     else proposal)
                private = decisions._private_for(
                    "waveform", self._private_capability)
                response = self.agents.waveform.challenge(
                    private, response_proposal)
            else:
                response = self._independent_summary(
                    decisions["waveform"], sender="waveform",
                    receiver="prototype", active=prototype_active)
            proposals.append(proposal)
            responses.append(response)
        if intervention in {"messages_shuffled", "cross_sample_messages"}:
            responses = [self._shuffle_response(packet) for packet in responses]
        return tuple(proposals), tuple(responses)

    def b2(self, decisions: Mapping[str, LocalDecision]) -> DialogueOutput:
        self._validate_class_count(decisions)
        batch = len(decisions["waveform"].unknown_score)
        device = decisions["waveform"].unknown_score.device
        action = torch.full((batch,), int(RouteAction.STOP),
                            dtype=torch.long, device=device)
        return self.dialogue(decisions, action)

    def deliberate(self, decisions: Mapping[str, LocalDecision], *,
                   route_action: torch.Tensor | int | RouteAction | None = None,
                   hard: bool = True,
                   intervention: str | None = None,
                   protocol: str = "sequential") -> DialogueOutput:
        batch = len(decisions["waveform"].unknown_score)
        route_logits = self.route_logits(decisions)
        if route_action is None:
            if not hard:
                # The semantic protocol is discrete by construction.  Soft
                # probabilities are exposed in route_logits for the router
                # loss, while the executed action remains auditable.
                action = route_logits.argmax(1)
            else:
                action = route_logits.argmax(1)
        elif isinstance(route_action, (int, RouteAction)):
            action = torch.full(
                (batch,), int(route_action), dtype=torch.long,
                device=route_logits.device)
        else:
            action = route_action.to(device=route_logits.device)
        action = self.router.validate_action(action, batch)

        valid_interventions = {
            None, "messages_off", "messages_shuffled", "cross_sample_messages",
            "wrong_class_pair", "drop_challenger",
        }
        if intervention not in valid_interventions:
            raise ValueError(f"unsupported Stage-7 intervention: {intervention}")
        if protocol not in {"sequential", "parallel"}:
            raise ValueError("protocol must be 'sequential' or 'parallel'")
        if intervention in {"messages_off", "drop_challenger"}:
            action = torch.full_like(action, int(RouteAction.STOP))

        if bool(action.ne(int(RouteAction.STOP)).any()):
            if protocol == "sequential" and not isinstance(decisions, LocalContext):
                raise PermissionError(
                    "communication requires the capability-bearing LocalContext "
                    "returned by Stage7System.local")
            proposals, challenges = self._messages(
                decisions, action, intervention, protocol)
        else:
            proposals, challenges = (), ()
        return self.dialogue(
            decisions, action, proposals, challenges, route_logits=route_logits)

    def parallel(self, decisions: Mapping[str, LocalDecision], *,
                 route_action: torch.Tensor | int | RouteAction | None = None,
                 intervention: str | None = None) -> DialogueOutput:
        """Run the equal-bandwidth, proposal-independent response control."""

        return self.deliberate(
            decisions, route_action=route_action,
            intervention=intervention, protocol="parallel")

    def forward(self, iq: torch.Tensor, *,
                route_action: torch.Tensor | int | RouteAction | None = None,
                hard: bool = True,
                intervention: str | None = None,
                protocol: str = "sequential") -> DialogueOutput:
        local = self.local(iq)
        return self.deliberate(
            local, route_action=route_action, hard=hard,
            intervention=intervention, protocol=protocol)
