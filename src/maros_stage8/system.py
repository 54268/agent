"""Stage-8 A/B capability shell and forced consultation service.

There is deliberately no learned Router here.  G1/G1.5 must pass before a
Value-of-Information Coordinator is allowed to execute.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Dict

import torch
from torch import nn

from .contracts import (EvidenceCertificate, LocalDecision, QueryPacket,
                        QueryType)
from .evidence import CandidateEvidenceTable
from .observations import (CoarseSpectralObservation, RobustSpectralObservation,
                           SpectralObservation,
                           TemporalObservation, TemporalPatchObservation,
                           build_coarse_spectral_observation,
                           build_robust_spectral_observation_v3,
                           build_spectral_observation,
                           build_temporal_observation,
                           build_temporal_patch_observation)
from .spectral_agent import SpectralHardwareFingerprintAgent
from .temporal_agent import TemporalFingerprintAgent


class LocalEvidenceContext(Mapping[str, LocalDecision]):
    """Public decisions plus owner-capability-protected Agent states."""

    __slots__ = ("_public", "__private", "__capability")

    def __init__(self, public: Dict[str, LocalDecision],
                 private: Dict[str, object], capability: object):
        if set(public) != {"temporal", "spectral"} or set(private) != set(public):
            raise ValueError("context requires exactly temporal and spectral Agents")
        self._public = public
        self.__private = private
        self.__capability = capability

    def __getitem__(self, key: str) -> LocalDecision:
        return self._public[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._public)

    def __len__(self) -> int:
        return len(self._public)

    def public_dict(self) -> Dict[str, LocalDecision]:
        return dict(self._public)

    def _private_for(self, name: str, capability: object) -> object:
        if capability is not self.__capability:
            raise PermissionError("private Agent state requires the owner capability")
        return self.__private[name]


class Stage8System(nn.Module):
    """Two hard-isolated specialists and candidate-bound communication."""

    def __init__(self, num_classes: int, *, state_dim: int = 96,
                 hidden_channels: int = 64, scalar_bits: int = 16,
                 observation_profile: str = "legacy_v1",
                 temporal_observation_profile: str | None = None,
                 spectral_observation_profile: str | None = None):
        super().__init__()
        temporal_profile = (
            observation_profile if temporal_observation_profile is None
            else temporal_observation_profile)
        spectral_profile = (
            observation_profile if spectral_observation_profile is None
            else spectral_observation_profile)
        if temporal_profile not in {"legacy_v1", "isolated_v2"}:
            raise ValueError("unsupported temporal observation profile")
        if spectral_profile not in {
                "legacy_v1", "isolated_v2",
                "robust_v3", "robust_v3_noalign"}:
            raise ValueError("unsupported spectral observation profile")
        self.temporal = TemporalFingerprintAgent(
            num_classes, state_dim, hidden_channels, scalar_bits,
            observation_profile=temporal_profile)
        self.spectral = SpectralHardwareFingerprintAgent(
            num_classes, state_dim, hidden_channels, scalar_bits,
            observation_profile=spectral_profile)
        self.num_classes = int(num_classes)
        self.scalar_bits = int(scalar_bits)
        self.temporal_observation_profile = temporal_profile
        self.spectral_observation_profile = spectral_profile
        self.observation_profile = (
            temporal_profile if temporal_profile == spectral_profile
            else f"temporal={temporal_profile};spectral={spectral_profile}")
        self._private_capability = object()

    def local(self, temporal: TemporalObservation | TemporalPatchObservation,
              spectral: (SpectralObservation | CoarseSpectralObservation
                         | RobustSpectralObservation),
              ) -> LocalEvidenceContext:
        temporal_decision, temporal_private = self.temporal.forward_private(temporal)
        spectral_decision, spectral_private = self.spectral.forward_private(spectral)
        return LocalEvidenceContext(
            {"temporal": temporal_decision, "spectral": spectral_decision},
            {"temporal": temporal_private, "spectral": spectral_private},
            self._private_capability)

    def observe(self, iq: torch.Tensor) -> LocalEvidenceContext:
        """Build both views in the system-level observation service."""

        temporal, spectral = self.build_observations(iq)
        return self.local(temporal, spectral)

    def build_observations(self, iq: torch.Tensor):
        """Apply the registered external observation profile."""

        temporal = (
            build_temporal_observation(iq)
            if self.temporal_observation_profile == "legacy_v1"
            else build_temporal_patch_observation(iq))
        if self.spectral_observation_profile == "legacy_v1":
            spectral = build_spectral_observation(iq)
        elif self.spectral_observation_profile == "isolated_v2":
            spectral = build_coarse_spectral_observation(iq)
        else:
            spectral = build_robust_spectral_observation_v3(
                iq, align=self.spectral_observation_profile == "robust_v3")
        return temporal, spectral

    def make_query(self, context: Mapping[str, LocalDecision], *,
                   sender: str, receiver: str,
                   candidate_a: torch.Tensor | None = None,
                   candidate_b: torch.Tensor | None = None,
                   active_mask: torch.Tensor | None = None,
                   reason_code: int = 0) -> QueryPacket:
        if sender == receiver or {sender, receiver} != {"temporal", "spectral"}:
            raise ValueError("A/B query must cross the temporal/spectral boundary")
        decision = context[sender]
        batch = len(decision.top1)
        candidate_a = decision.top1 if candidate_a is None else candidate_a
        candidate_b = decision.top2 if candidate_b is None else candidate_b
        if candidate_a.shape != (batch,) or candidate_b.shape != (batch,):
            raise ValueError("candidate pair must have shape [batch]")
        active = (torch.ones(batch, dtype=torch.bool, device=decision.top1.device)
                  if active_mask is None else active_mask.bool())
        if active.shape != (batch,):
            raise ValueError("active_mask must have shape [batch]")
        if bool(candidate_a[active].eq(candidate_b[active]).any()):
            raise ValueError("an active candidate pair must contain two hypotheses")
        pair_probability = decision.class_logits.softmax(1)
        support = (
            pair_probability.gather(1, candidate_a[:, None]).squeeze(1)
            - pair_probability.gather(1, candidate_b[:, None]).squeeze(1))
        query_type = (QueryType.VERIFY_TEMPORAL_PAIR if receiver == "temporal"
                      else QueryType.VERIFY_SPECTRAL_PAIR)
        class_bits = max(1, (self.num_classes - 1).bit_length())
        bit_cost = active.to(support) * float(2 * class_bits + 2 * self.scalar_bits + 4)
        zeros_long = torch.zeros_like(candidate_a)
        zeros = torch.zeros_like(support)
        return QueryPacket(
            sender=sender, receiver=receiver,
            candidate_a=torch.where(active, candidate_a, zeros_long),
            candidate_b=torch.where(active, candidate_b, zeros_long),
            query_type=query_type,
            reason_code=torch.where(active, torch.full_like(candidate_a, reason_code), zeros_long),
            local_pair_support=torch.where(active, support, zeros),
            local_open_risk=torch.where(active, decision.unknown_score, zeros),
            active_mask=active, bit_cost=bit_cost)

    def answer(self, context: LocalEvidenceContext,
               query: QueryPacket) -> EvidenceCertificate:
        if not isinstance(context, LocalEvidenceContext):
            raise PermissionError("consultation requires capability-bearing local context")
        private = context._private_for(query.receiver, self._private_capability)
        if query.receiver == "temporal":
            return self.temporal.verify(private, query)
        if query.receiver == "spectral":
            return self.spectral.verify(private, query)
        raise ValueError("unknown Stage-8 responder")

    def all_candidate_support(
        self, context: LocalEvidenceContext, agent: str,
    ) -> torch.Tensor:
        """Expose candidate support, never the underlying private descriptor."""

        if not isinstance(context, LocalEvidenceContext):
            raise PermissionError(
                "candidate verification requires capability-bearing context")
        private = context._private_for(agent, self._private_capability)
        if agent == "temporal":
            return self.temporal.all_verification_scores(private)
        if agent == "spectral":
            return self.spectral.all_verification_scores(private)
        raise ValueError("agent must be temporal or spectral")

    @torch.no_grad()
    def reindex_classes(self, old_to_new: torch.Tensor) -> None:
        """Relabel both Agents without changing any class semantics."""

        self.temporal.reindex_classes(old_to_new)
        self.spectral.reindex_classes(old_to_new)

    def no_communication(self, context: Mapping[str, LocalDecision], *,
                         anchor: str = "temporal"):
        """Return one Agent's unchanged table; STOP is an exact identity."""

        if anchor not in {"temporal", "spectral"}:
            raise ValueError("anchor must be temporal or spectral")
        decision = context[anchor]
        return CandidateEvidenceTable(
            decision.class_logits, decision.unknown_score).decision()

    def forced_consultation(self, context: LocalEvidenceContext, *,
                            sender: str, receiver: str,
                            candidate_a: torch.Tensor | None = None,
                            candidate_b: torch.Tensor | None = None,
                            active_mask: torch.Tensor | None = None,
                            evidence_scale: float = 1.0):
        """Execute exactly one request; no routing policy is learned."""

        query = self.make_query(
            context, sender=sender, receiver=receiver,
            candidate_a=candidate_a, candidate_b=candidate_b,
            active_mask=active_mask)
        certificate = self.answer(context, query)
        base = context[sender]
        table = CandidateEvidenceTable(base.class_logits, base.unknown_score)
        table.apply(certificate, scale=evidence_scale)
        return table.decision(), query, certificate
