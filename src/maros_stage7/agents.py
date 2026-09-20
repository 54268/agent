"""The two Stage-7 inference Agents and their capability-isolated state.

The Agents share a semantic protocol, not an encoder.  Waveform observes the
complex waveform and retains temporal CVCNN tokens.  Prototype observes a
spectral/envelope view and owns enrollment prototypes plus per-class tail
memory.  Their conditional ``challenge`` methods cannot be called without a
proposal packet.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from maros_stage5.agents import (ComplexBlock, ComplexIQEncoder,
                                 specialist_view)
from maros_staged.stem import PrivateEncoder, SharedStem

from .contracts import (ChallengePacket, ChallengeStance, LocalDecision,
                        ProposalPacket)


_PUBLIC_SUMMARY_DIM = 7


def _make_competence_head(input_dim: int) -> nn.Sequential:
    """Create a private head without perturbing the experiment RNG stream.

    Adding an auxiliary head must not silently change the initialisation of
    modules constructed after it (notably the Prototype Agent).  A local RNG
    fork makes V6 a paired architectural comparison rather than a different
    random initialisation.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7103 + int(input_dim))
        head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 64),
            nn.GELU(), nn.Linear(64, 1),
        )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _validate_iq(iq: torch.Tensor) -> None:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("Stage-7 Agents require I/Q shaped [batch, 2, length]")
    if iq.shape[-1] < 8:
        raise ValueError("I/Q length must be at least 8 samples")
    if not iq.is_floating_point():
        raise TypeError("I/Q input must be a floating-point tensor")


def _decision_summary(logits: torch.Tensor, unknown_logit: torch.Tensor,
                      reliability: torch.Tensor) -> tuple[torch.Tensor, ...]:
    log_probability = F.log_softmax(logits, dim=-1)
    probability = log_probability.exp()
    top = probability.topk(2, dim=-1)
    entropy = -(probability * log_probability).sum(1)
    entropy_scale = max(float(torch.log(torch.tensor(logits.shape[1]))), 1e-6)
    energy = -torch.logsumexp(logits, dim=-1)
    margin = top.values[:, 0] - top.values[:, 1]
    public = torch.stack([
        top.values[:, 0], top.values[:, 1], margin,
        entropy / entropy_scale, torch.tanh(energy),
        torch.sigmoid(unknown_logit), reliability,
    ], dim=1)
    return top.indices[:, 0], top.indices[:, 1], public


def _local_decision(logits: torch.Tensor, unknown_logit: torch.Tensor,
                    reliability: torch.Tensor) -> LocalDecision:
    top1, top2, public = _decision_summary(logits, unknown_logit, reliability)
    return LocalDecision(
        class_logits=logits,
        unknown_score=unknown_logit,
        reliability=reliability.clamp(0.0, 1.0),
        top1=top1,
        top2=top2,
        public_summary=public,
    )


def _masked_bits(active: torch.Tensor, fixed_bits: int) -> torch.Tensor:
    return active.to(dtype=torch.float32) * float(fixed_bits)


@torch.no_grad()
def _robust_enrollment_statistics(
    states: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return class centres and robust high-tail location/scale.

    The memory is deliberately empirical and Known-only.  A 95th-percentile
    location matches the fixed Known-acceptance operating point; the scale is
    estimated from the central-to-tail spread rather than the standard
    deviation of a handful of extreme points (which was unstable in the
    first Stage-7 screen).
    """

    states = F.normalize(states, dim=-1)
    labels = labels.long().to(states.device)
    centers, locations, scales = [], [], []
    for class_index in range(int(num_classes)):
        selected = states[labels == class_index]
        if len(selected) == 0:
            raise ValueError(f"class {class_index} has no enrollment samples")
        center = F.normalize(selected.mean(0), dim=0)
        distance = (selected - center).square().sum(1)
        median = torch.quantile(distance, 0.50)
        location = torch.quantile(distance, 0.95)
        mad = torch.quantile((distance - median).abs(), 0.50) * 1.4826
        scale = torch.maximum(location - median, mad).clamp_min(1e-3)
        centers.append(center)
        locations.append(location)
        scales.append(scale)
    return torch.stack(centers), torch.stack(locations), torch.stack(scales)


def _validate_proposal(proposal: ProposalPacket, *, expected_sender: str,
                       batch_size: int, num_classes: int) -> torch.Tensor:
    if not isinstance(proposal, ProposalPacket):
        raise TypeError("conditional challenge requires a ProposalPacket")
    if proposal.sender != expected_sender:
        raise ValueError(
            f"expected a {expected_sender} proposal, got {proposal.sender!r}")
    fields = {
        "top1": proposal.top1,
        "top2": proposal.top2,
        "log_odds": proposal.log_odds,
        "local_unknown": proposal.local_unknown,
        "reliability": proposal.reliability,
        "active_mask": proposal.active_mask,
        "bit_cost": proposal.bit_cost,
    }
    for name, value in fields.items():
        if not torch.is_tensor(value) or value.ndim != 1 or len(value) != batch_size:
            raise ValueError(f"ProposalPacket.{name} must have shape [batch]")
    active = proposal.active_mask.bool()
    for name, value in (("top1", proposal.top1), ("top2", proposal.top2)):
        selected = value[active]
        if bool(((selected < 0) | (selected >= num_classes)).any()):
            raise ValueError(f"active ProposalPacket.{name} is outside class range")
    if bool(proposal.top1[active].eq(proposal.top2[active]).any()):
        raise ValueError("active proposal top1 and top2 must be distinct")
    return active


def _mask_challenge_payload(*, active: torch.Tensor, stance: torch.Tensor,
                            alternative: torch.Tensor, signed: torch.Tensor,
                            tail: torch.Tensor, reliability: torch.Tensor,
                            bit_cost: torch.Tensor
                            ) -> tuple[torch.Tensor, ...]:
    """Remove all payload from samples for which no message was sent."""

    zeros = torch.zeros_like(signed)
    stance = torch.where(
        active, stance,
        torch.full_like(stance, int(ChallengeStance.SUPPORT)))
    alternative = torch.where(active, alternative, torch.zeros_like(alternative))
    signed = torch.where(active, signed, zeros)
    tail = torch.where(active, tail, zeros)
    reliability = torch.where(active, reliability, zeros)
    bit_cost = torch.where(active, bit_cost, torch.zeros_like(bit_cost))
    return stance, alternative, signed, tail, reliability, bit_cost


@dataclass(frozen=True)
class _WaveformPrivate:
    state: torch.Tensor
    tokens: torch.Tensor
    distances: torch.Tensor
    tail_z: torch.Tensor


@dataclass(frozen=True)
class _PrototypePrivate:
    state: torch.Tensor
    distances: torch.Tensor
    tail_z: torch.Tensor
    view_states: torch.Tensor
    view_logits: torch.Tensor
    view_distances: torch.Tensor
    view_weights: torch.Tensor


class LocalContext(Mapping[str, LocalDecision]):
    """Mapping of public decisions with capability-protected private state.

    Training and evaluation code can treat this object as a normal mapping.
    Only the owning :class:`Stage7System` has the identity capability required
    by ``_private_for``.  The dialogue adjudicator is never given that
    capability and therefore cannot bypass a transmitted message.
    """

    __slots__ = ("_decisions", "__private", "__capability")

    def __init__(self, decisions: Dict[str, LocalDecision],
                 private: Dict[str, object], capability: object):
        if set(decisions) != {"waveform", "prototype"}:
            raise ValueError("LocalContext requires waveform and prototype decisions")
        if set(private) != set(decisions):
            raise ValueError("private Agent state must match public Agent decisions")
        self._decisions = decisions
        self.__private = private
        self.__capability = capability

    def __getitem__(self, key: str) -> LocalDecision:
        return self._decisions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decisions)

    def __len__(self) -> int:
        return len(self._decisions)

    def public_dict(self) -> Dict[str, LocalDecision]:
        return dict(self._decisions)

    def _private_for(self, name: str, capability: object) -> object:
        if capability is not self.__capability:
            raise PermissionError("private Agent state requires the owner capability")
        return self.__private[name]


class _ComplexTemporalEncoder(nn.Module):
    """Stable raw-I/Q identity state plus phase-coherent consultation tokens.

    G0-V2 normalised two independently learned branch states and let a softmax
    gate mix them.  On ORACLE the gate selected the weak branch, the complex
    representation collapsed to one point, and all eight classes received the
    same prediction even though every gradient path was live.  The identity
    classifier therefore returns to the proven Stage-5 contract: it consumes
    the *unnormalised* raw ``SharedStem -> PrivateEncoder`` state.  The CVCNN
    cannot starve that state; it contributes only a small bounded residual and
    the private temporal tokens later used for conditional consultation.

    The bounded residual also gives the complex blocks and token projection a
    local classification gradient before communication training.  Normalising
    for supervised contrastive learning or enrollment distance remains the
    responsibility of those objectives, not of this encoder.
    """

    COMPLEX_RESIDUAL_BOUND = 0.1

    def __init__(self, state_dim: int, hidden: int = 64,
                 dropout: float = 0.15,
                 raw_channels: int | None = None):
        super().__init__()
        widths = (hidden, hidden * 2, hidden * 4)
        self.blocks = nn.ModuleList([
            ComplexBlock(1, widths[0], 7, dropout * 0.25),
            ComplexBlock(widths[0], widths[1], 5, dropout * 0.25),
            ComplexBlock(widths[1], widths[2], 3, dropout * 0.25),
        ])
        token_input = 2 * widths[-1]
        self.token_project = nn.Linear(token_input, state_dim)
        self.complex_residual = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim),
            nn.Tanh(),
        )
        raw_channels = max(8, int(
            hidden if raw_channels is None else raw_channels))
        self.raw_channels = raw_channels
        self.raw_stem = SharedStem(raw_channels)
        self.raw_encoder = PrivateEncoder(
            raw_channels, state_dim, hidden=raw_channels)

    def forward(self, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_iq(iq)
        z = torch.complex(iq[:, :1], iq[:, 1:2])
        for block in self.blocks:
            z = block(z)
        token_source = torch.cat([z.real, z.imag], dim=1).transpose(1, 2)
        token_values = self.token_project(token_source)
        tokens = F.normalize(token_values, dim=-1)
        raw_state = self.raw_encoder(self.raw_stem(iq))
        complex_delta = self.complex_residual(token_values.mean(1))
        state = raw_state + self.COMPLEX_RESIDUAL_BOUND * complex_delta
        return state, tokens


class WaveformIdentityAgent(nn.Module):
    """Raw-I/Q identity proposer with pair-conditioned temporal attention."""

    name = "waveform"

    def __init__(self, num_classes: int, state_dim: int = 128,
                 class_embed_dim: int = 24, complex_hidden: int = 64,
                 scalar_bits: int = 16,
                 raw_channels: int | None = None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Stage-7 requires at least two Known classes")
        self.encoder = _ComplexTemporalEncoder(
            state_dim, complex_hidden, raw_channels=raw_channels)
        self.classifier = nn.Linear(state_dim, num_classes)
        self.register_buffer(
            "enrollment_prototypes", torch.zeros(num_classes, state_dim))
        self.register_buffer("tail_location", torch.ones(num_classes))
        self.register_buffer("tail_scale", torch.ones(num_classes))
        self.register_buffer("risk_location", torch.zeros(num_classes, 2))
        self.register_buffer("risk_scale", torch.ones(num_classes, 2))
        self.register_buffer("enrollment_ready", torch.tensor(False))
        self.open_head = nn.Sequential(
            nn.LayerNorm(state_dim + 4), nn.Linear(state_dim + 4, 64),
            nn.GELU(), nn.Linear(64, 1),
        )
        self.competence_head = _make_competence_head(state_dim + 4)
        self.register_buffer("competence_enabled", torch.tensor(False))
        # The local rejection score must retain a strong, interpretable
        # Known-only anchor.  PUG/inner-episode training may learn only a
        # bounded residual around it; it cannot replace the anchor with a
        # shortcut.  Zero initialisation makes pre-open-head behaviour exact.
        nn.init.zeros_(self.open_head[-1].weight)
        nn.init.zeros_(self.open_head[-1].bias)
        # Class descriptors are computed from the local classifier and
        # Known-only enrollment memory.  Unlike ``nn.Embedding(num_classes)``,
        # the response-head parameters are independent of the fold's class
        # count and can be transferred from an inner episode to its outer
        # model without remapping arbitrary embedding rows.
        self.class_descriptor = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, class_embed_dim), nn.Tanh(),
        )
        self.pair_query = nn.Sequential(
            nn.Linear(2 * class_embed_dim, state_dim), nn.Tanh(),
        )
        response_dim = 2 * state_dim + 2 * class_embed_dim + 2
        self.pair_residual = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, 96),
            nn.GELU(), nn.Linear(96, 1),
        )
        self.tail_head = nn.Sequential(
            nn.LayerNorm(response_dim), nn.Linear(response_dim, 64),
            nn.GELU(), nn.Linear(64, 1),
        )
        self.num_classes = int(num_classes)
        self.state_dim = int(state_dim)
        self.scalar_bits = int(scalar_bits)

    def _describe_classes(self, class_indices: torch.Tensor) -> torch.Tensor:
        classifier_memory = F.normalize(
            self.classifier.weight.index_select(0, class_indices), dim=1)
        if bool(self.enrollment_ready):
            enrollment_memory = self.enrollment_prototypes.index_select(
                0, class_indices)
        else:
            # Before Known-only enrollment is refreshed the classifier itself
            # is the only legal class memory.  Reusing it in both slots keeps
            # the descriptor well-defined without fabricating unknown data.
            enrollment_memory = classifier_memory
        descriptor = torch.cat(
            [classifier_memory, F.normalize(enrollment_memory, dim=1)], dim=1)
        return self.class_descriptor(descriptor)

    def forward_private(self, iq: torch.Tensor) -> tuple[LocalDecision, _WaveformPrivate]:
        state, tokens = self.encoder(iq)
        logits = self.classifier(state)
        probability = F.softmax(logits, dim=-1)
        top = probability.topk(2, dim=-1).values
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
        normalized_state = F.normalize(state, dim=-1)
        if bool(self.enrollment_ready):
            distances = torch.cdist(
                normalized_state, self.enrollment_prototypes, p=2.0).square()
            nearest_class = distances.argmin(1)
            nearest_distance = distances.gather(
                1, nearest_class[:, None]).squeeze(1)
            tail_z = (
                nearest_distance - self.tail_location.gather(0, nearest_class)
            ) / self.tail_scale.gather(0, nearest_class).clamp_min(1e-4)
        else:
            distances = state.new_zeros((len(state), self.num_classes))
            tail_z = state.new_zeros(len(state))
        # One-minus-reliability was substantially stronger than the first
        # learned Stage-7 score on the registered WiSig sanity fold.  Use its
        # logit as the immutable anchor and add empirical tail evidence once
        # enrollment memory is ready.
        intrinsic_reliability = (
            top[:, 0] * (1.0 - entropy / max(float(torch.log(
                torch.tensor(self.num_classes, device=state.device))), 1e-6))
        ).clamp(1e-5, 1.0 - 1e-5)
        reliability_risk = torch.logit(1.0 - intrinsic_reliability)
        energy_risk = -torch.logsumexp(logits, dim=-1)
        intrinsic_unknown = reliability_risk
        if bool(self.enrollment_ready):
            predicted = logits.argmax(1)
            locations = self.risk_location.index_select(0, predicted)
            scales = self.risk_scale.index_select(0, predicted).clamp_min(1e-4)
            standardized = (
                torch.stack([reliability_risk, energy_risk], dim=1)
                - locations) / scales
            # A Known-only standardized evidence pool lets the same Agent use
            # confidence geometry on WiSig and energy geometry on ORACLE,
            # without a dataset branch or any outer Unknown.
            components = torch.cat([
                standardized, tail_z.clamp(-6.0, 6.0)[:, None]], dim=1)
            intrinsic_unknown = torch.logsumexp(components, dim=1) - float(
                torch.log(torch.tensor(components.shape[1])))
        evidence = torch.stack([
            1.0 - top[:, 0], top[:, 0] - top[:, 1],
            entropy / max(float(torch.log(torch.tensor(self.num_classes))), 1e-6),
            torch.tanh(tail_z),
        ], dim=1)
        public_input = torch.cat([state, evidence], dim=1)
        residual = 0.5 * torch.tanh(
            self.open_head(public_input).squeeze(1))
        unknown = intrinsic_unknown + residual
        if bool(self.competence_enabled):
            reliability = torch.sigmoid(
                self.competence_head(public_input).squeeze(1))
        else:
            reliability = torch.sigmoid(-unknown)
        reliability = reliability.clamp(0.0, 1.0)
        private = _WaveformPrivate(state, tokens, distances, tail_z)
        return _local_decision(logits, unknown, reliability), private

    @torch.no_grad()
    def set_enrollment_memory(self, states: torch.Tensor,
                              logits: torch.Tensor,
                              labels: torch.Tensor) -> None:
        """Fit waveform prototypes/tails using Known enrollment only."""

        states = F.normalize(states, dim=-1)
        centers, locations, scales = _robust_enrollment_statistics(
            states, labels, self.num_classes)
        self.enrollment_prototypes.copy_(centers)
        self.tail_location.copy_(locations)
        self.tail_scale.copy_(scales)
        probability = F.softmax(logits, dim=-1)
        top = probability.topk(2, dim=-1).values
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
        entropy_scale = max(float(torch.log(torch.tensor(
            self.num_classes, device=states.device))), 1e-6)
        reliability = (top[:, 0] * (1.0 - entropy / entropy_scale)).clamp(
            1e-5, 1.0 - 1e-5)
        risks = torch.stack([
            torch.logit(1.0 - reliability),
            -torch.logsumexp(logits, dim=-1),
        ], dim=1)
        risk_locations, risk_scales = [], []
        for class_index in range(self.num_classes):
            selected = risks[labels == class_index]
            median = torch.quantile(selected, 0.50, dim=0)
            location = torch.quantile(selected, 0.95, dim=0)
            mad = torch.quantile(
                (selected - median).abs(), 0.50, dim=0) * 1.4826
            risk_locations.append(location)
            risk_scales.append(torch.maximum(
                location - median, mad).clamp_min(1e-3))
        self.risk_location.copy_(torch.stack(risk_locations))
        self.risk_scale.copy_(torch.stack(risk_scales))
        self.enrollment_ready.fill_(True)

    def challenge(self, private: _WaveformPrivate,
                  proposal: ProposalPacket) -> ChallengePacket:
        if not isinstance(private, _WaveformPrivate):
            raise TypeError("Waveform challenge requires Waveform private state")
        active = _validate_proposal(
            proposal, expected_sender="prototype",
            batch_size=len(private.state), num_classes=self.num_classes)
        emb1 = self._describe_classes(proposal.top1)
        emb2 = self._describe_classes(proposal.top2)
        query = self.pair_query(torch.cat([emb1, emb2], dim=1))
        attention = torch.softmax(
            torch.einsum("btd,bd->bt", private.tokens, query)
            / max(self.state_dim ** 0.5, 1.0), dim=1)
        attended = torch.einsum("bt,btd->bd", attention, private.tokens)
        probability = F.softmax(self.classifier(private.state), dim=-1)
        p1 = probability.gather(1, proposal.top1[:, None]).squeeze(1)
        p2 = probability.gather(1, proposal.top2[:, None]).squeeze(1)
        features = torch.cat([
            private.state, attended, emb1, emb2, p1[:, None], p2[:, None],
        ], dim=1)
        local_pair = torch.logit(p1.clamp(1e-5, 1 - 1e-5)) - torch.logit(
            p2.clamp(1e-5, 1 - 1e-5))
        signed = local_pair + self.pair_residual(features).squeeze(1)
        tail = torch.sigmoid(self.tail_head(features).squeeze(1))
        alternative = probability.argmax(1)
        stance = torch.where(
            tail >= 0.5,
            torch.full_like(alternative, int(ChallengeStance.UNKNOWN_SUSPECT)),
            torch.where(
                signed >= 0.0,
                torch.full_like(alternative, int(ChallengeStance.SUPPORT)),
                torch.full_like(alternative, int(ChallengeStance.CHALLENGE)),
            ),
        )
        class_bits = max(1, (self.num_classes - 1).bit_length())
        bit_cost = _masked_bits(
            active, 2 + class_bits + 3 * self.scalar_bits).to(signed)
        reliability = (1.0 - tail) * torch.sigmoid(signed.abs())
        (stance, alternative, signed, tail,
         reliability, bit_cost) = _mask_challenge_payload(
            active=active, stance=stance, alternative=alternative,
            signed=signed, tail=tail, reliability=reliability,
            bit_cost=bit_cost)
        return ChallengePacket(
            sender=self.name, receiver=proposal.sender, stance=stance,
            alternative_class=alternative, signed_pair_evidence=signed,
            open_tail_evidence=tail, reliability=reliability,
            active_mask=active, bit_cost=bit_cost,
        )


_PROTOTYPE_REAL_VIEW_NAMES = (
    "raw", "spectral", "envelope_phase", "difference_iq")
_PROTOTYPE_VIEW_NAMES = (*_PROTOTYPE_REAL_VIEW_NAMES, "complex_iq")


class EnrollmentPrototypeAgent(nn.Module):
    """Multi-view metric Agent with private enrollment and tail memory.

    All datasets use the same five-sensor bank already validated by the
    Stage-5 universal Geometry Agent: raw I/Q, spectral, envelope/phase,
    difference I/Q and a phase-coherent complex-I/Q encoder.  These are
    private sensors inside one metric Agent, not five independently voting
    Agents.  A learned per-sample gate chooses how much evidence each sensor
    contributes.  The Agent exports only one local decision; individual view
    states and memories stay private and are used by the conditional
    challenge operation.
    """

    name = "prototype"

    def __init__(self, num_classes: int, state_dim: int = 128,
                 stem_channels: int = 128, temperature: float = 0.15,
                 scalar_bits: int = 16):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Stage-7 requires at least two Known classes")
        if temperature <= 0:
            raise ValueError("prototype temperature must be positive")
        self.view_names = _PROTOTYPE_VIEW_NAMES
        self.real_view_names = _PROTOTYPE_REAL_VIEW_NAMES
        # Match the proven Stage-5 universal bank: real-valued sensors start
        # from an identical architecture/initialization and then specialize
        # independently.  No dataset name or dataset-specific switch enters
        # this construction.
        base_stem = SharedStem(stem_channels)
        base_encoder = PrivateEncoder(
            stem_channels, state_dim, hidden=stem_channels)
        self.view_stems = nn.ModuleList([
            copy.deepcopy(base_stem) for _ in self.real_view_names])
        self.view_encoders = nn.ModuleList([
            copy.deepcopy(base_encoder) for _ in self.real_view_names])
        # Stage-5 uses hidden=64.  Preserve that capacity for real runs while
        # allowing the deliberately tiny unit-test model to stay lightweight.
        complex_hidden = min(64, max(8, int(stem_channels)))
        self.complex_encoder = ComplexIQEncoder(
            state_dim, hidden=complex_hidden)
        descriptor_dim = len(self.view_names) * (state_dim + 3)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(descriptor_dim), nn.Linear(descriptor_dim, state_dim),
            nn.GELU(), nn.Linear(state_dim, len(self.view_names)),
        )
        nn.init.zeros_(self.view_gate[-1].weight)
        nn.init.zeros_(self.view_gate[-1].bias)
        self.prototypes = nn.Parameter(F.normalize(
            torch.randn(num_classes, state_dim), dim=-1))
        # Classification prototypes remain trainable decision parameters.
        # Enrollment centres are separate Known-only memory: refreshing the
        # latter must never silently replace the classifier used in local
        # pretraining (the V2 implementation made exactly that train/eval
        # swap, which was especially destructive on ORACLE).
        self.register_buffer("enrollment_prototypes", torch.zeros(
            num_classes, state_dim))
        self.register_buffer("view_prototypes", torch.zeros(
            len(self.view_names), num_classes, state_dim))
        self.register_buffer("tail_location", torch.ones(num_classes))
        self.register_buffer("tail_scale", torch.ones(num_classes))
        self.register_buffer("margin_location", torch.zeros(num_classes))
        self.register_buffer("margin_scale", torch.ones(num_classes))
        self.register_buffer("view_tail_location", torch.ones(
            len(self.view_names), num_classes))
        self.register_buffer("view_tail_scale", torch.ones(
            len(self.view_names), num_classes))
        self.register_buffer("enrollment_ready", torch.tensor(False))
        self.open_head = nn.Sequential(
            nn.LayerNorm(state_dim + 4), nn.Linear(state_dim + 4, 64),
            nn.GELU(), nn.Linear(64, 1),
        )
        self.competence_head = _make_competence_head(state_dim + 4)
        self.register_buffer("competence_enabled", torch.tensor(False))
        nn.init.zeros_(self.open_head[-1].weight)
        nn.init.zeros_(self.open_head[-1].bias)
        self.temperature = float(temperature)
        self.num_classes = int(num_classes)
        self.state_dim = int(state_dim)
        self.scalar_bits = int(scalar_bits)

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return (F.normalize(state, dim=-1)
                @ F.normalize(self.prototypes, dim=-1).T
                / max(self.temperature, 1e-6))

    def forward_private(self, iq: torch.Tensor) -> tuple[LocalDecision, _PrototypePrivate]:
        _validate_iq(iq)
        real_states = [
            F.normalize(encoder(stem(specialist_view(iq, name))), dim=-1)
            for name, stem, encoder in zip(
                self.real_view_names, self.view_stems, self.view_encoders)
        ]
        view_states = torch.stack(
            [*real_states, self.complex_encoder(iq)], dim=1)
        learned_centers = F.normalize(self.prototypes, dim=-1)
        # Class logits keep the exact pre-enrollment classifier.  Empirical
        # sensor centres below are reserved for metric distance, tails and
        # conditional challenges, as in the proven universal Geometry Agent.
        view_logits = torch.einsum(
            "bvd,cd->bvc", view_states, learned_centers
        ) / max(self.temperature, 1e-6)
        if bool(self.enrollment_ready):
            global_centers = F.normalize(
                self.enrollment_prototypes, dim=-1)
            view_centers = F.normalize(self.view_prototypes, dim=-1)
        else:
            global_centers = learned_centers
            view_centers = learned_centers[None].expand(
                len(self.view_names), -1, -1)
        view_probability = F.softmax(view_logits, dim=-1)
        view_top = view_probability.topk(2, dim=-1).values
        view_entropy = -(view_probability * view_probability.clamp_min(
            1e-8).log()).sum(-1)
        entropy_scale = max(float(torch.log(torch.tensor(
            self.num_classes, device=iq.device))), 1e-6)
        diagnostics = torch.stack([
            view_top[..., 0], view_top[..., 0] - view_top[..., 1],
            1.0 - view_entropy / entropy_scale,
        ], dim=-1)
        descriptors = torch.cat([view_states, diagnostics], dim=-1).flatten(1)
        weights = F.softmax(self.view_gate(descriptors), dim=1)
        state = F.normalize(
            (weights[:, :, None] * view_states).sum(1), dim=-1)
        # Per-view prototypes preserve evidence that a single averaged state
        # would erase.  The global prototype term stabilises early training.
        logits = (weights[:, :, None] * view_logits).sum(1)
        global_distances = torch.cdist(
            state, global_centers, p=2.0).square()
        view_distances = (
            view_states[:, :, None, :] - view_centers[None]
        ).square().sum(-1)
        distances = 0.5 * global_distances + 0.5 * (
            weights[:, :, None] * view_distances).sum(1)
        nearest = distances.topk(2, dim=1, largest=False).values
        nearest_class = distances.argmin(1)
        location = self.tail_location.gather(0, nearest_class)
        scale = self.tail_scale.gather(0, nearest_class).clamp_min(1e-4)
        if bool(self.enrollment_ready):
            global_tail_z = (nearest[:, 0] - location) / scale
            view_nearest_class = view_distances.argmin(-1)
            view_nearest = view_distances.gather(
                2, view_nearest_class[..., None]).squeeze(-1)
            view_location = self.view_tail_location[None].expand(
                len(iq), -1, -1).gather(
                    2, view_nearest_class[..., None]).squeeze(-1)
            view_scale = self.view_tail_scale[None].expand(
                len(iq), -1, -1).gather(
                    2, view_nearest_class[..., None]).squeeze(-1).clamp_min(1e-4)
            view_tail_z = (view_nearest - view_location) / view_scale
            tail_z = 0.5 * global_tail_z + 0.5 * (
                weights * view_tail_z).sum(1)
        else:
            # Learned prototype geometry supplies the competition margin
            # during pretraining; empirical tail evidence only becomes legal
            # after Known enrollment has been explicitly refreshed.
            tail_z = nearest[:, 0].new_zeros(len(nearest))
        probability = F.softmax(logits, dim=-1)
        top = probability.topk(2, dim=1).values
        competition_margin = nearest[:, 1] - nearest[:, 0]
        evidence = torch.stack([
            nearest[:, 0], competition_margin,
            1.0 - top[:, 0], torch.tanh(tail_z),
        ], dim=1)
        public_input = torch.cat([state, evidence], dim=1)
        residual = 0.5 * torch.tanh(
            self.open_head(public_input).squeeze(1))
        negative_margin = -competition_margin
        if bool(self.enrollment_ready):
            margin_z = (
                negative_margin - self.margin_location.gather(0, nearest_class)
            ) / self.margin_scale.gather(0, nearest_class).clamp_min(1e-4)
            components = torch.stack([
                margin_z.clamp(-6.0, 6.0), tail_z.clamp(-6.0, 6.0)], dim=1)
            intrinsic_unknown = torch.logsumexp(components, dim=1) - float(
                torch.log(torch.tensor(components.shape[1])))
        else:
            intrinsic_unknown = negative_margin / max(self.temperature, 1e-6)
        unknown = intrinsic_unknown + residual
        if bool(self.competence_enabled):
            reliability = torch.sigmoid(
                self.competence_head(public_input).squeeze(1))
        else:
            reliability = torch.sigmoid(-unknown)
        reliability = reliability.clamp(0.0, 1.0)
        private = _PrototypePrivate(
            state, distances, tail_z, view_states, view_logits,
            view_distances, weights)
        return _local_decision(logits, unknown, reliability), private

    def challenge(self, private: _PrototypePrivate,
                  proposal: ProposalPacket) -> ChallengePacket:
        if not isinstance(private, _PrototypePrivate):
            raise TypeError("Prototype challenge requires Prototype private state")
        active = _validate_proposal(
            proposal, expected_sender="waveform",
            batch_size=len(private.state), num_classes=self.num_classes)
        d1 = private.distances.gather(1, proposal.top1[:, None]).squeeze(1)
        d2 = private.distances.gather(1, proposal.top2[:, None]).squeeze(1)
        # Positive means the proposed top-1 is geometrically better supported.
        signed = (d2 - d1) / max(self.temperature, 1e-6)
        location = self.tail_location.gather(0, proposal.top1)
        scale = self.tail_scale.gather(0, proposal.top1).clamp_min(1e-4)
        tail_z = (d1 - location) / scale
        tail = torch.sigmoid(tail_z)
        alternative = private.distances.argmin(1)
        stance = torch.where(
            tail >= 0.5,
            torch.full_like(alternative, int(ChallengeStance.UNKNOWN_SUSPECT)),
            torch.where(
                signed >= 0.0,
                torch.full_like(alternative, int(ChallengeStance.SUPPORT)),
                torch.full_like(alternative, int(ChallengeStance.CHALLENGE)),
            ),
        )
        class_bits = max(1, (self.num_classes - 1).bit_length())
        bit_cost = _masked_bits(
            active, 2 + class_bits + 3 * self.scalar_bits).to(signed)
        reliability = torch.exp(-d1) * torch.sigmoid(signed.abs()) * (1.0 - tail)
        (stance, alternative, signed, tail,
         reliability, bit_cost) = _mask_challenge_payload(
            active=active, stance=stance, alternative=alternative,
            signed=signed, tail=tail,
            reliability=reliability.clamp(0.0, 1.0), bit_cost=bit_cost)
        return ChallengePacket(
            sender=self.name, receiver=proposal.sender, stance=stance,
            alternative_class=alternative, signed_pair_evidence=signed,
            open_tail_evidence=tail, reliability=reliability,
            active_mask=active, bit_cost=bit_cost,
        )

    @torch.no_grad()
    def set_enrollment_memory(self, states: torch.Tensor,
                              view_states: torch.Tensor,
                              view_weights: torch.Tensor,
                              labels: torch.Tensor) -> None:
        """Refresh prototypes and robust class-tail statistics from Known only."""

        states = F.normalize(states, dim=-1)
        view_states = F.normalize(view_states, dim=-1)
        if view_states.ndim != 3 or view_states.shape[1] != len(self.view_names):
            raise ValueError("view enrollment states have the wrong shape")
        if view_weights.shape != view_states.shape[:2]:
            raise ValueError("view enrollment weights have the wrong shape")
        centers, _, _ = _robust_enrollment_statistics(
            states, labels, self.num_classes)
        view_centers, view_locations, view_scales = [], [], []
        for view_index in range(len(self.view_names)):
            center, location, scale = _robust_enrollment_statistics(
                view_states[:, view_index], labels, self.num_classes)
            view_centers.append(center)
            view_locations.append(location)
            view_scales.append(scale)
        stacked_views = torch.stack(view_centers)
        global_distance = (
            states - centers[labels]
        ).square().sum(1)
        per_view_distance = (
            view_states - stacked_views[:, labels].permute(1, 0, 2)
        ).square().sum(-1)
        aggregate_distance = (
            0.5 * global_distance
            + 0.5 * (view_weights * per_view_distance).sum(1))
        global_all = torch.cdist(states, centers, p=2.0).square()
        view_all = (
            view_states[:, :, None, :] - stacked_views[None]
        ).square().sum(-1)
        aggregate_all = 0.5 * global_all + 0.5 * (
            view_weights[:, :, None] * view_all).sum(1)
        nearest_all = aggregate_all.topk(2, dim=1, largest=False).values
        negative_margin = nearest_all[:, 0] - nearest_all[:, 1]
        locations, scales = [], []
        margin_locations, margin_scales = [], []
        for class_index in range(self.num_classes):
            selected = aggregate_distance[labels == class_index]
            median = torch.quantile(selected, 0.50)
            location = torch.quantile(selected, 0.95)
            mad = torch.quantile((selected - median).abs(), 0.50) * 1.4826
            locations.append(location)
            scales.append(torch.maximum(
                location - median, mad).clamp_min(1e-3))
            selected_margin = negative_margin[labels == class_index]
            margin_median = torch.quantile(selected_margin, 0.50)
            margin_location = torch.quantile(selected_margin, 0.95)
            margin_mad = torch.quantile(
                (selected_margin - margin_median).abs(), 0.50) * 1.4826
            margin_locations.append(margin_location)
            margin_scales.append(torch.maximum(
                margin_location - margin_median, margin_mad).clamp_min(1e-3))
        self.enrollment_prototypes.copy_(centers)
        self.view_prototypes.copy_(stacked_views)
        self.tail_location.copy_(torch.stack(locations))
        self.tail_scale.copy_(torch.stack(scales))
        self.margin_location.copy_(torch.stack(margin_locations))
        self.margin_scale.copy_(torch.stack(margin_scales))
        self.view_tail_location.copy_(torch.stack(view_locations))
        self.view_tail_scale.copy_(torch.stack(view_scales))
        self.enrollment_ready.fill_(True)


class RoleAgents(nn.Module):
    """Container that constructs one isolated local context per batch."""

    def __init__(self, num_classes: int, capability: object,
                 state_dim: int = 128, stem_channels: int = 128,
                 class_embed_dim: int = 24, complex_hidden: int = 64,
                 prototype_temperature: float = 0.15,
                 scalar_bits: int = 16):
        super().__init__()
        self.waveform = WaveformIdentityAgent(
            num_classes, state_dim, class_embed_dim, complex_hidden,
            scalar_bits, raw_channels=stem_channels)
        self.prototype = EnrollmentPrototypeAgent(
            num_classes, state_dim, stem_channels, prototype_temperature,
            scalar_bits)
        self._capability = capability

    @torch.no_grad()
    def set_competence_enabled(self, enabled: bool) -> None:
        """Switch both Agents between legacy risk and learned competence.

        The flag is shared by role but each competence head remains private
        and is trained only from its own local decision loss.
        """

        self.waveform.competence_enabled.fill_(bool(enabled))
        self.prototype.competence_enabled.fill_(bool(enabled))

    def forward(self, iq: torch.Tensor) -> LocalContext:
        waveform, waveform_private = self.waveform.forward_private(iq)
        prototype, prototype_private = self.prototype.forward_private(iq)
        return LocalContext(
            {"waveform": waveform, "prototype": prototype},
            {"waveform": waveform_private, "prototype": prototype_private},
            self._capability,
        )

    def training_private(self, context: LocalContext) -> Dict[str, object]:
        """Expose private tensors only to the owning training service.

        The dialogue receives a Mapping of public decisions and has neither
        this container nor its identity capability.  Keeping this accessor on
        the owner lets supervised metric objectives train the private spaces
        without adding embeddings to ``LocalDecision`` or the wire protocol.
        """

        if not isinstance(context, LocalContext):
            raise TypeError("private training evidence requires LocalContext")
        return {
            name: context._private_for(name, self._capability)
            for name in ("waveform", "prototype")
        }
