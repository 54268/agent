"""Hypothesis-conditioned proposal--challenge--verification communication."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (DialogueOutput, LocalDecision, MessagePacket,
                        QueryAction)


COMMUNICATION_MODES = (
    "no_comm", "early_fusion", "parallel_request", "sequential",
    "sequential_cf",
)
EDGE_NAMES = ("waveform_to_verifier", "prototype_to_verifier")


class SequentialDialogue(nn.Module):
    """A verifier-led dialogue with no cross-agent evidence bypass.

    The query scheduler only sees compact public intents and the verifier's
    own state.  Sender states enter the verifier solely through a gated and
    bandwidth-masked MessagePacket.  Full local logits are used only by the
    final convex known-class vote, never by the unknown-risk network.
    """

    def __init__(self, num_classes: int, state_dim: int = 128,
                 intent_dim: int = 16, message_dim: int = 32,
                 query_dim: int = 32, class_embed_dim: int = 8,
                 hidden_dim: int = 192, mode: str = "sequential_cf",
                 gate_temperature: float = 0.5,
                 query_cost: float = 0.05):
        super().__init__()
        if mode not in COMMUNICATION_MODES:
            raise ValueError(f"unsupported Stage-6 mode: {mode}")
        if message_dim < 4:
            raise ValueError("message_dim must reserve 3 intent actions plus a payload")
        self.mode = mode
        self.num_classes = int(num_classes)
        self.state_dim = int(state_dim)
        self.intent_dim = int(intent_dim)
        self.message_dim = int(message_dim)
        self.query_dim = int(query_dim)
        self.gate_temperature = float(gate_temperature)
        self.query_cost = float(query_cost)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        # intent + two class embeddings + two probabilities + reliability,
        # local unknown score, confidence, margin
        self.public_dim = intent_dim + 2 * class_embed_dim + 6
        negotiation_dim = 3 * self.public_dim + state_dim
        self.leader_head = nn.Sequential(
            nn.LayerNorm(negotiation_dim), nn.Linear(negotiation_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 2),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(negotiation_dim + 2),
            nn.Linear(negotiation_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, query_dim), nn.Tanh(),
        )
        self.value_critic = nn.Sequential(
            nn.LayerNorm(negotiation_dim + query_dim + 2),
            nn.Linear(negotiation_dim + query_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # A sequential responder is explicitly conditioned on the leader's
        # class proposal and four role-private, class-conditional statistics.
        response_dim = state_dim + query_dim + 2 + class_embed_dim + 4
        self.waveform_response = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, message_dim - 3), nn.Tanh(),
        )
        self.prototype_response = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, message_dim - 3), nn.Tanh(),
        )
        self.waveform_action = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 3),
        )
        self.prototype_action = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 3),
        )
        self.waveform_mask = nn.Sequential(
            nn.Linear(response_dim, message_dim - 3), nn.Sigmoid())
        self.prototype_mask = nn.Sequential(
            nn.Linear(response_dim, message_dim - 3), nn.Sigmoid())
        transcript_dim = state_dim + 3 * self.public_dim + 2 * message_dim + 2
        self.audit_head = nn.Sequential(
            nn.LayerNorm(transcript_dim), nn.Linear(transcript_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, 1),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(transcript_dim), nn.Linear(transcript_dim, 64),
            nn.GELU(), nn.Linear(64, 2),
        )
        self.no_comm_weights = nn.Sequential(
            nn.LayerNorm(3 * self.public_dim), nn.Linear(3 * self.public_dim, 64),
            nn.GELU(), nn.Linear(64, 2),
        )
        self.no_comm_open = nn.Sequential(
            nn.LayerNorm(3 * self.public_dim), nn.Linear(3 * self.public_dim, hidden_dim),
            nn.GELU(), nn.Dropout(0.10), nn.Linear(hidden_dim, 1),
        )
        # Centralised early-fusion baseline.  It is never used by a
        # communication candidate.
        early_dim = 3 * state_dim
        self.early_class = nn.Sequential(
            nn.LayerNorm(early_dim), nn.Linear(early_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.early_open = nn.Sequential(
            nn.LayerNorm(early_dim), nn.Linear(early_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _public(self, decision: LocalDecision) -> torch.Tensor:
        probability = F.softmax(decision.class_logits, dim=-1)
        top = probability.topk(2, dim=-1)
        embeddings = self.class_embedding(top.indices).flatten(1)
        margin = top.values[:, 0] - top.values[:, 1]
        scalars = torch.stack([
            top.values[:, 0], top.values[:, 1], decision.reliability,
            torch.sigmoid(decision.unknown_logit), top.values[:, 0], margin,
        ], dim=1)
        return torch.cat([decision.intent, embeddings, scalars], dim=1)

    def _gates(self, expected_gain: torch.Tensor, hard: bool) -> torch.Tensor:
        if self.mode in {"no_comm", "early_fusion"}:
            return expected_gain.new_zeros(expected_gain.shape)
        # The critic predicts task-loss reduction before communication.  A
        # message is worthwhile only when that gain also pays its explicit
        # query cost; this avoids the degenerate all-communicate policy.
        net_gain = expected_gain - self.query_cost
        soft = torch.sigmoid(net_gain / max(self.gate_temperature, 1e-6))
        if hard:
            return (net_gain > 0.0).to(expected_gain.dtype)
        return soft

    @staticmethod
    def _conditional_private(decision: LocalDecision,
                             proposal: torch.Tensor,
                             prototype: bool) -> torch.Tensor:
        """Private sender evidence; it can leave the Agent only in a packet."""
        probability = F.softmax(decision.class_logits, dim=-1)
        proposed_probability = probability.gather(1, proposal[:, None]).squeeze(1)
        top = probability.topk(2, dim=-1).values
        if prototype and "distances" in decision.private_evidence:
            distances = decision.private_evidence["distances"]
            proposed_distance = distances.gather(1, proposal[:, None]).squeeze(1)
            d1 = decision.private_evidence.get("d1", distances.min(1).values)
            distance_margin = decision.private_evidence.get(
                "distance_margin", distances.topk(2, largest=False, dim=-1).values.diff().squeeze(1))
            return torch.stack([
                proposed_probability, torch.tanh(proposed_distance),
                torch.tanh(d1), torch.tanh(distance_margin),
            ], 1)
        entropy = decision.private_evidence.get("entropy")
        if entropy is None:
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
        energy = decision.private_evidence.get("energy")
        if energy is None:
            energy = -torch.logsumexp(decision.class_logits, dim=-1)
        return torch.stack([
            proposed_probability, top[:, 0] - top[:, 1],
            torch.tanh(entropy), torch.tanh(energy),
        ], 1)

    def forward(self, decisions: Dict[str, LocalDecision], *, hard: bool = False,
                intervention: str | None = None,
                edge_override: torch.Tensor | None = None) -> DialogueOutput:
        waveform = decisions["waveform"]
        prototype = decisions["prototype"]
        verifier = decisions["verifier"]
        pub_w = self._public(waveform)
        pub_p = self._public(prototype)
        pub_v = self._public(verifier)
        public = torch.cat([pub_w, pub_p, pub_v], dim=1)
        negotiation = torch.cat([public, verifier.state], dim=1)
        leader_logits = self.leader_head(negotiation)
        if intervention == "reverse_leader":
            leader_logits = leader_logits.flip(1)
        elif intervention == "random_leader":
            random_id = torch.randint(0, 2, (len(leader_logits),), device=leader_logits.device)
            leader_logits = F.one_hot(random_id, 2).to(leader_logits) * 20.0 - 10.0
        leader_prob = F.softmax(leader_logits, dim=1)
        leader_id = leader_prob.argmax(1)
        leader_action = (F.one_hot(leader_id, 2).to(leader_prob)
                         if hard else leader_prob)
        query_token = self.query_head(torch.cat([negotiation, leader_action], dim=1))
        value_input = torch.cat([negotiation, query_token, leader_action], dim=1)
        expected_gain = self.value_critic(value_input)
        gates = self._gates(expected_gain, hard)
        if intervention == "messages_off":
            gates = torch.zeros_like(gates)
        elif intervention == "drop_waveform_sender":
            gates = gates.clone(); gates[:, 0] = 0.0
        elif intervention == "drop_prototype_sender":
            gates = gates.clone(); gates[:, 1] = 0.0
        if edge_override is not None:
            gates = edge_override.to(gates)

        # SeqComm-style conditioning: in the sequential modes each response
        # knows which agent has already proposed.  The parallel baseline gets
        # no ordering token.
        sequential = self.mode.startswith("sequential")
        response_order = leader_action if sequential else torch.zeros_like(leader_action)
        waveform_proposal = waveform.class_logits.argmax(1)
        prototype_proposal = prototype.class_logits.argmax(1)
        proposal = torch.where(leader_id == 0, waveform_proposal, prototype_proposal)
        if sequential:
            proposal_embedding = self.class_embedding(proposal)
            private_w = self._conditional_private(waveform, proposal, prototype=False)
            private_p = self._conditional_private(prototype, proposal, prototype=True)
        else:
            proposal_embedding = waveform.state.new_zeros(
                len(waveform.state), self.class_embedding.embedding_dim)
            private_w = waveform.state.new_zeros(len(waveform.state), 4)
            private_p = prototype.state.new_zeros(len(prototype.state), 4)
        response_w_input = torch.cat([
            waveform.state, query_token, response_order,
            proposal_embedding, private_w,
        ], dim=1)
        response_p_input = torch.cat([
            prototype.state, query_token, response_order,
            proposal_embedding, private_p,
        ], dim=1)
        action_w = F.softmax(self.waveform_action(response_w_input), dim=1)
        action_p = F.softmax(self.prototype_action(response_p_input), dim=1)
        residual_w = self.waveform_response(response_w_input)
        residual_p = self.prototype_response(response_p_input)
        residual_mask_w = self.waveform_mask(response_w_input)
        residual_mask_p = self.prototype_mask(response_p_input)
        mandatory = residual_mask_w.new_ones(len(residual_mask_w), 3)
        mask_w = torch.cat([mandatory, residual_mask_w], dim=1)
        mask_p = torch.cat([mandatory, residual_mask_p], dim=1)
        payload_w = torch.cat([action_w, residual_w * residual_mask_w], dim=1)
        payload_p = torch.cat([action_p, residual_p * residual_mask_p], dim=1)
        if intervention in {"messages_shuffled", "cross_sample_messages"} and len(payload_w) > 1:
            order = torch.randperm(len(payload_w), device=payload_w.device)
            payload_w, payload_p = payload_w[order], payload_p[order]
        gated_w = gates[:, 0:1] * payload_w
        gated_p = gates[:, 1:2] * payload_p
        bit_w = gates[:, 0] * mask_w.sum(1) * 32.0
        bit_p = gates[:, 1] * mask_p.sum(1) * 32.0
        packets = (
            MessagePacket("waveform", "verifier", "proposal_or_response",
                          payload_w, mask_w, bit_w),
            MessagePacket("prototype", "verifier", "challenge_or_support",
                          payload_p, mask_p, bit_p),
        )
        transcript = torch.cat([
            verifier.state, public, gated_w, gated_p, leader_action,
        ], dim=1)
        base_weights = F.softmax(self.no_comm_weights(public), dim=1)
        base_unknown = self.no_comm_open(public).squeeze(1)

        if self.mode == "early_fusion":
            early = torch.cat([waveform.state, prototype.state, verifier.state], dim=1)
            fused_logits = self.early_class(early)
            unknown_logit = self.early_open(early).squeeze(1)
            known_weights = leader_prob
        elif self.mode == "no_comm":
            known_weights = base_weights
            fused_logits = (known_weights[:, :1] * waveform.class_logits
                            + known_weights[:, 1:] * prototype.class_logits)
            unknown_logit = base_unknown
        else:
            active = gates.amax(1, keepdim=True)
            negotiated_weights = F.softmax(self.fusion_head(transcript), dim=1)
            known_weights = (1.0 - active) * base_weights + active * negotiated_weights
            # Convexity is retained even with soft training gates.
            known_weights = known_weights / known_weights.sum(1, keepdim=True).clamp_min(1e-8)
            fused_logits = (known_weights[:, :1] * waveform.class_logits
                            + known_weights[:, 1:] * prototype.class_logits)
            # Communication is an additional rejection certificate, not a
            # licence to erase the competent silent detector.  Its residual
            # is therefore non-negative.  Sequential messages further scale
            # it by their supervised support/challenge/unknown-suspect act.
            audit_strength = F.softplus(self.audit_head(transcript).squeeze(1))
            if sequential:
                suspicion_w = (0.10 * action_w[:, 0]
                               + 0.50 * action_w[:, 1] + action_w[:, 2])
                suspicion_p = (0.10 * action_p[:, 0]
                               + 0.50 * action_p[:, 1] + action_p[:, 2])
                suspicion = ((gates[:, 0] * suspicion_w
                              + gates[:, 1] * suspicion_p)
                             / gates.sum(1).clamp_min(1e-6))
                audit_delta = audit_strength * suspicion
            else:
                audit_delta = audit_strength
            unknown_logit = base_unknown + active.squeeze(1) * audit_delta

        cost = bit_w + bit_p
        action = QueryAction(leader_id, gates, query_token, expected_gain, cost)
        return DialogueOutput(
            decisions, leader_id, action, transcript, packets, fused_logits,
            unknown_logit, gates, known_weights,
        )
