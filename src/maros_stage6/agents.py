"""Three heterogeneous agents with private RF observations.

The agents share an output protocol, not an encoder.  Waveform sees complex
I/Q, Prototype sees spectral/envelope observations, and Verifier sees
difference/phase-transition observations.  Every agent has a valid local
classification and rejection proposal before communication.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from maros_staged.stem import PrivateEncoder, SharedStem
from maros_stage5.agents import ComplexIQEncoder, specialist_view

from .contracts import LocalDecision


def _summary(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    log_prob = F.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    top = prob.topk(2, dim=-1)
    return {
        "prob": prob,
        "pred": top.indices[:, 0],
        "runner_up": top.indices[:, 1],
        "confidence": top.values[:, 0],
        "margin": top.values[:, 0] - top.values[:, 1],
        "entropy": -(prob * log_prob).sum(-1),
        "energy": -torch.logsumexp(logits, dim=-1),
    }


class DecisionHead(nn.Module):
    """Role-local action head shared in shape, never in parameters."""

    def __init__(self, num_classes: int, state_dim: int, intent_dim: int):
        super().__init__()
        self.classifier = nn.Linear(state_dim, num_classes)
        summary_dim = 4
        self.open_head = nn.Sequential(
            nn.LayerNorm(state_dim + summary_dim),
            nn.Linear(state_dim + summary_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(state_dim + summary_dim),
            nn.Linear(state_dim + summary_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.intent_head = nn.Sequential(
            nn.LayerNorm(state_dim + summary_dim),
            nn.Linear(state_dim + summary_dim, intent_dim), nn.Tanh(),
        )
        self.num_classes = int(num_classes)

    def from_state(self, state: torch.Tensor, logits: torch.Tensor | None = None,
                   extra: Dict[str, torch.Tensor] | None = None) -> LocalDecision:
        logits = self.classifier(state) if logits is None else logits
        ev = _summary(logits)
        scale = max(float(torch.log(torch.tensor(self.num_classes))), 1e-6)
        features = torch.stack([
            1.0 - ev["confidence"], ev["margin"], ev["entropy"] / scale,
            torch.tanh(ev["energy"]),
        ], dim=1)
        joined = torch.cat([state, features], dim=1)
        unknown = self.open_head(joined).squeeze(1)
        reliability = torch.sigmoid(self.reliability_head(joined)).squeeze(1)
        intent = self.intent_head(joined)
        private = dict(ev)
        private["summary_features"] = features
        if extra:
            private.update(extra)
        return LocalDecision(state, logits, unknown, reliability, intent, private)


class PrototypeDecisionHead(DecisionHead):
    def __init__(self, num_classes: int, state_dim: int, intent_dim: int,
                 temperature: float):
        super().__init__(num_classes, state_dim, intent_dim)
        del self.classifier
        self.prototypes = nn.Parameter(F.normalize(
            torch.randn(num_classes, state_dim), dim=-1))
        self.temperature = float(temperature)

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return F.normalize(state, dim=-1) @ F.normalize(
            self.prototypes, dim=-1).T / max(self.temperature, 1e-6)

    def from_state(self, state: torch.Tensor, logits: torch.Tensor | None = None,
                   extra: Dict[str, torch.Tensor] | None = None) -> LocalDecision:
        logits = self.logits_from_state(state) if logits is None else logits
        distances = torch.cdist(F.normalize(state, dim=-1),
                                F.normalize(self.prototypes, dim=-1)) ** 2
        nearest = distances.topk(2, largest=False, dim=-1).values
        geometric = {
            "distances": distances,
            "d1": nearest[:, 0],
            "d2": nearest[:, 1],
            "distance_margin": nearest[:, 1] - nearest[:, 0],
        }
        if extra:
            geometric.update(extra)
        result = super().from_state(state, logits, geometric)
        # Geometry reliability must reflect both confidence and prototype fit.
        result.reliability = result.reliability * torch.exp(-nearest[:, 0]).clamp(0, 1)
        return result


class WaveformIdentityAgent(nn.Module):
    name = "waveform"

    def __init__(self, num_classes: int, state_dim: int = 128,
                 intent_dim: int = 16, complex_hidden: int = 64):
        super().__init__()
        self.encoder = ComplexIQEncoder(state_dim, hidden=complex_hidden)
        self.decision = DecisionHead(num_classes, state_dim, intent_dim)

    def forward(self, iq: torch.Tensor) -> LocalDecision:
        return self.decision.from_state(self.encoder(iq))

    def from_state(self, state: torch.Tensor) -> LocalDecision:
        return self.decision.from_state(state)


class SpectralPrototypeAgent(nn.Module):
    name = "prototype"

    def __init__(self, num_classes: int, state_dim: int = 128,
                 stem_channels: int = 128, intent_dim: int = 16,
                 temperature: float = 0.15):
        super().__init__()
        self.spectral_stem = SharedStem(stem_channels)
        self.spectral_encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
        self.envelope_stem = SharedStem(stem_channels)
        self.envelope_encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(2 * state_dim), nn.Linear(2 * state_dim, 64),
            nn.GELU(), nn.Linear(64, 2),
        )
        self.decision = PrototypeDecisionHead(
            num_classes, state_dim, intent_dim, temperature)

    def forward(self, iq: torch.Tensor) -> LocalDecision:
        spectral = F.normalize(self.spectral_encoder(
            self.spectral_stem(specialist_view(iq, "spectral"))), dim=-1)
        envelope = F.normalize(self.envelope_encoder(
            self.envelope_stem(specialist_view(iq, "envelope_phase"))), dim=-1)
        states = torch.stack([spectral, envelope], dim=1)
        weights = F.softmax(self.view_gate(torch.cat([spectral, envelope], dim=1)), dim=1)
        state = F.normalize((weights[:, :, None] * states).sum(1), dim=-1)
        return self.decision.from_state(state, extra={
            "view_weights": weights, "view_states": states,
        })

    def from_state(self, state: torch.Tensor) -> LocalDecision:
        return self.decision.from_state(state)


class BoundaryVerifierAgent(nn.Module):
    name = "verifier"

    def __init__(self, num_classes: int, state_dim: int = 128,
                 stem_channels: int = 128, intent_dim: int = 16):
        super().__init__()
        self.diff_stem = SharedStem(stem_channels)
        self.diff_encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
        self.phase_stem = SharedStem(stem_channels)
        self.phase_encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
        self.combine = nn.Sequential(
            nn.LayerNorm(2 * state_dim), nn.Linear(2 * state_dim, state_dim), nn.GELU(),
        )
        self.decision = DecisionHead(num_classes, state_dim, intent_dim)

    def forward(self, iq: torch.Tensor) -> LocalDecision:
        difference = self.diff_encoder(self.diff_stem(
            specialist_view(iq, "difference_iq")))
        phase = self.phase_encoder(self.phase_stem(
            specialist_view(iq, "envelope_phase")))
        state = F.normalize(self.combine(torch.cat([difference, phase], dim=1)), dim=-1)
        return self.decision.from_state(state)

    def from_state(self, state: torch.Tensor) -> LocalDecision:
        return self.decision.from_state(state)


class RoleAgents(nn.Module):
    """Container for the three independently useful Stage-6 agents."""

    def __init__(self, num_classes: int, state_dim: int = 128,
                 stem_channels: int = 128, intent_dim: int = 16,
                 prototype_temperature: float = 0.15):
        super().__init__()
        self.waveform = WaveformIdentityAgent(num_classes, state_dim, intent_dim)
        self.prototype = SpectralPrototypeAgent(
            num_classes, state_dim, stem_channels, intent_dim, prototype_temperature)
        self.verifier = BoundaryVerifierAgent(
            num_classes, state_dim, stem_channels, intent_dim)
        self.num_classes = int(num_classes)
        self.state_dim = int(state_dim)
        self.intent_dim = int(intent_dim)

    def forward(self, iq: torch.Tensor) -> Dict[str, LocalDecision]:
        return {
            "waveform": self.waveform(iq),
            "prototype": self.prototype(iq),
            "verifier": self.verifier(iq),
        }

    def from_states(self, states: Dict[str, torch.Tensor]) -> Dict[str, LocalDecision]:
        return {
            "waveform": self.waveform.from_state(states["waveform"]),
            "prototype": self.prototype.from_state(states["prototype"]),
            "verifier": self.verifier.from_state(states["verifier"]),
        }
