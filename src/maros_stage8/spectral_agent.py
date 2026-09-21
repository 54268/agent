"""Frequency-domain-only hardware fingerprint and candidate verifier."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (EvidenceCertificate, EvidenceStance, LocalDecision,
                        QueryPacket, QueryType)
from .observations import (CoarseSpectralObservation, RobustSpectralObservation,
                           SpectralObservation)


@dataclass(frozen=True)
class _SpectralPrivateState:
    state: torch.Tensor
    quality: torch.Tensor


class SpectralHardwareFingerprintAgent(nn.Module):
    """Spectral impairment specialist; raw I/Q is not accepted by this API."""

    name = "spectral"

    def __init__(self, num_classes: int, state_dim: int = 96,
                 hidden_channels: int = 64, scalar_bits: int = 16,
                 observation_profile: str = "legacy_v1"):
        super().__init__()
        if num_classes < 2:
            raise ValueError("at least two Known classes are required")
        if observation_profile not in {
                "legacy_v1", "isolated_v2",
                "robust_v3", "robust_v3_noalign"}:
            raise ValueError("unsupported spectral observation profile")
        input_channels = 8 if observation_profile == "isolated_v2" else 6
        quality_dim = 8 if observation_profile == "isolated_v2" else 6
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, 9, stride=2, padding=4),
            nn.BatchNorm1d(hidden_channels), nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, 7,
                      stride=2, padding=3),
            nn.BatchNorm1d(hidden_channels * 2), nn.GELU(),
            nn.Conv1d(hidden_channels * 2, state_dim, 5,
                      stride=2, padding=2),
            nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.quality_projection = nn.Linear(quality_dim, state_dim)
        self.classifier = nn.Linear(state_dim, num_classes)
        self.register_buffer(
            "candidate_prototypes", torch.zeros(num_classes, state_dim))
        self.register_buffer(
            "prototype_counts", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("prototypes_ready", torch.tensor(False))
        self.verifier = nn.Sequential(
            nn.LayerNorm(4 * state_dim), nn.Linear(4 * state_dim, state_dim),
            nn.GELU(), nn.Linear(state_dim, 1),
        )
        self.num_classes = int(num_classes)
        self.scalar_bits = int(scalar_bits)
        self.observation_profile = observation_profile

    @torch.no_grad()
    def set_candidate_prototypes(
        self, prototypes: torch.Tensor, counts: torch.Tensor,
    ) -> None:
        """Install descriptors computed exclusively from this Agent's Known data."""

        if prototypes.shape != self.candidate_prototypes.shape:
            raise ValueError("spectral prototypes have the wrong shape")
        if counts.shape != self.prototype_counts.shape:
            raise ValueError("spectral prototype counts have the wrong shape")
        if bool((counts <= 0).any()):
            raise ValueError("every spectral candidate requires Known samples")
        self.candidate_prototypes.copy_(F.normalize(
            prototypes.to(self.candidate_prototypes), dim=1))
        self.prototype_counts.copy_(counts.to(self.prototype_counts))
        self.prototypes_ready.fill_(True)

    def _candidate_descriptors(self, candidates: torch.Tensor) -> torch.Tensor:
        if not bool(self.prototypes_ready):
            raise RuntimeError(
                "spectral candidate prototypes must be refreshed from Known data")
        return self.candidate_prototypes.index_select(0, candidates)

    def _encode(
        self, observation: (SpectralObservation | CoarseSpectralObservation
                            | RobustSpectralObservation),
    ) -> _SpectralPrivateState:
        if self.observation_profile == "legacy_v1":
            expected = SpectralObservation
        elif self.observation_profile == "isolated_v2":
            expected = CoarseSpectralObservation
        else:
            expected = RobustSpectralObservation
        if not isinstance(observation, expected):
            raise TypeError(
                "SpectralHardwareFingerprintAgent requires its configured "
                "SpectralObservation contract")
        state = self.encoder(observation.features).squeeze(-1)
        # V3 diagnostics must stay public and cannot become a centroid or
        # capture-condition shortcut for the identity classifier.
        if self.observation_profile in {"legacy_v1", "isolated_v2"}:
            state = state + 0.1 * torch.tanh(
                self.quality_projection(observation.quality))
        return _SpectralPrivateState(state=state, quality=observation.quality)

    def _decision(self, private: _SpectralPrivateState) -> LocalDecision:
        logits = self.classifier(private.state)
        probability = logits.softmax(1)
        top = probability.topk(2, dim=1)
        entropy = -(probability * probability.clamp_min(1e-9).log()).sum(1)
        margin = top.values[:, 0] - top.values[:, 1]
        unknown = entropy - logits.amax(1)
        public = torch.cat([
            top.values, margin[:, None], entropy[:, None],
            unknown[:, None], private.quality,
        ], dim=1)
        return LocalDecision(
            sender=self.name, class_logits=logits, unknown_score=unknown,
            top1=top.indices[:, 0], top2=top.indices[:, 1],
            public_diagnostics=public)

    def forward_private(
        self, observation: (SpectralObservation | CoarseSpectralObservation
                            | RobustSpectralObservation),
    ) -> tuple[LocalDecision, _SpectralPrivateState]:
        private = self._encode(observation)
        return self._decision(private), private

    def forward(
        self, observation: (SpectralObservation | CoarseSpectralObservation
                            | RobustSpectralObservation),
    ) -> LocalDecision:
        return self.forward_private(observation)[0]

    def verification_scores(
        self, private: _SpectralPrivateState, candidates: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(private, _SpectralPrivateState):
            raise TypeError("Spectral verifier requires its own private state")
        candidates = candidates.long().to(private.state.device)
        if candidates.shape != (len(private.state),):
            raise ValueError("candidates must have shape [batch]")
        if bool(((candidates < 0) | (candidates >= self.num_classes)).any()):
            raise ValueError("candidate is outside the Known class range")
        descriptor = self._candidate_descriptors(candidates)
        state = F.normalize(private.state, dim=1)
        values = torch.cat([
            state, descriptor,
            (state - descriptor).abs(), state * descriptor,
        ], dim=1)
        return self.verifier(values).squeeze(1)

    def all_verification_scores(
        self, private: _SpectralPrivateState,
    ) -> torch.Tensor:
        """Score every candidate with one class-shared compatibility network."""

        if not isinstance(private, _SpectralPrivateState):
            raise TypeError("Spectral verifier requires its own private state")
        if not bool(self.prototypes_ready):
            raise RuntimeError(
                "spectral candidate prototypes must be refreshed from Known data")
        state = F.normalize(private.state, dim=1)
        query = state[:, None, :].expand(-1, self.num_classes, -1)
        descriptor = self.candidate_prototypes[None, :, :].expand(
            len(state), -1, -1)
        values = torch.cat([
            query, descriptor, (query - descriptor).abs(), query * descriptor,
        ], dim=2)
        return self.verifier(values).squeeze(2)

    @torch.no_grad()
    def reindex_classes(self, old_to_new: torch.Tensor) -> None:
        """Apply a pure class relabeling to classifier rows and prototypes."""

        permutation = old_to_new.long().to(self.classifier.weight.device)
        if (permutation.shape != (self.num_classes,)
                or sorted(permutation.cpu().tolist()) != list(range(self.num_classes))):
            raise ValueError("old_to_new must be a class permutation")
        weight = self.classifier.weight.detach().clone()
        bias = self.classifier.bias.detach().clone()
        prototypes = self.candidate_prototypes.detach().clone()
        counts = self.prototype_counts.detach().clone()
        self.classifier.weight[permutation] = weight
        self.classifier.bias[permutation] = bias
        self.candidate_prototypes[permutation] = prototypes
        self.prototype_counts[permutation] = counts

    def verify(self, private: _SpectralPrivateState,
               query: QueryPacket) -> EvidenceCertificate:
        if query.receiver != self.name or query.query_type != QueryType.VERIFY_SPECTRAL_PAIR:
            raise ValueError("Spectral Agent only answers spectral-pair queries")
        score_a = self.verification_scores(private, query.candidate_a)
        score_b = self.verification_scores(private, query.candidate_b)
        signed = score_a - score_b
        reliability = torch.sigmoid(signed.abs())
        stance = torch.where(
            signed >= 0, torch.full_like(query.candidate_a, int(EvidenceStance.SUPPORT_A)),
            torch.full_like(query.candidate_a, int(EvidenceStance.SUPPORT_B)))
        alternative = torch.where(signed >= 0, query.candidate_a, query.candidate_b)
        active = query.active_mask.bool()
        abstain = (~active) | reliability.lt(0.55)
        stance = torch.where(abstain, torch.full_like(stance, int(EvidenceStance.ABSTAIN)), stance)
        bits = active.to(signed) * float(2 + 3 * self.scalar_bits)
        zeros = torch.zeros_like(signed)
        return EvidenceCertificate(
            sender=self.name, receiver=query.sender,
            candidate_a=query.candidate_a, candidate_b=query.candidate_b,
            stance=torch.where(active, stance, torch.zeros_like(stance)),
            signed_candidate_evidence=torch.where(active, signed, zeros),
            alternative_class=torch.where(active, alternative, torch.zeros_like(alternative)),
            domain_quality=torch.where(active, private.quality.mean(1), zeros),
            open_risk=zeros, reliability=torch.where(active, reliability, zeros),
            abstain=abstain, active_mask=active, bit_cost=bits)
