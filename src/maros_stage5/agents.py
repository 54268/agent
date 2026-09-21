"""Identity and Geometry/OpenMax specialist agents for Stage-5."""
from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from maros_staged.stem import PrivateEncoder, SharedStem

from .contracts import AgentOutput


def spectral_view(x: torch.Tensor) -> torch.Tensor:
    """Build a differentiable log-magnitude/phase-increment view [B,2,L]."""
    xc = torch.complex(x[:, 0], x[:, 1])
    spectrum = torch.fft.fft(xc, dim=-1)
    magnitude = torch.log1p(torch.abs(spectrum))
    phase_step = torch.angle(spectrum * torch.roll(spectrum.conj(), shifts=1, dims=-1))
    view = torch.stack([magnitude, phase_step], dim=1)
    mean = view.mean(dim=-1, keepdim=True)
    std = view.std(dim=-1, keepdim=True).clamp_min(1e-5)
    return (view - mean) / std


def specialist_view(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a two-channel private observation for the Geometry Agent."""
    if mode == "raw":
        return x
    if mode == "spectral":
        return spectral_view(x)
    xc = torch.complex(x[:, 0], x[:, 1])
    if mode == "fft_iq":
        spectrum = torch.fft.fft(xc, dim=-1)
        view = torch.stack([spectrum.real, spectrum.imag], dim=1)
    elif mode == "envelope_phase":
        amplitude = torch.log1p(torch.abs(xc))
        phase_step = torch.angle(xc * torch.roll(xc.conj(), shifts=1, dims=-1))
        view = torch.stack([amplitude, phase_step], dim=1)
    elif mode == "difference_iq":
        view = x - torch.roll(x, shifts=1, dims=-1)
    else:
        raise ValueError(f"unsupported geometry view {mode}")
    mean = view.mean(dim=-1, keepdim=True)
    std = view.std(dim=-1, keepdim=True).clamp_min(1e-5)
    return (view - mean) / std


REAL_VIEW_NAMES = ("raw", "spectral", "envelope_phase", "difference_iq")
UNIVERSAL_VIEW_NAMES = (*REAL_VIEW_NAMES, "complex_iq")


def universal_specialist_views(x: torch.Tensor) -> list[torch.Tensor]:
    """Fixed cross-dataset view bank used by the unified Geometry Agent."""
    return [specialist_view(x, name) for name in REAL_VIEW_NAMES]


class ComplexConv1d(nn.Module):
    """Phase-coherent complex convolution for the universal sensor bank."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int = 0):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)
        self.real_bias = nn.Parameter(torch.zeros(out_channels))
        self.imag_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real(z.real) - self.imag(z.imag) + self.real_bias[None, :, None]
        imag = self.real(z.imag) + self.imag(z.real) + self.imag_bias[None, :, None]
        return torch.complex(real, imag)


class ComplexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size // 2)
        self.real_norm = nn.BatchNorm1d(out_channels)
        self.imag_norm = nn.BatchNorm1d(out_channels)
        self.dropout = float(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(z)
        z = torch.complex(F.relu(self.real_norm(convolved.real)),
                          F.relu(self.imag_norm(convolved.imag)))
        if self.dropout:
            z = torch.complex(F.dropout(z.real, self.dropout, self.training),
                              F.dropout(z.imag, self.dropout, self.training))
        return torch.complex(F.avg_pool1d(z.real, 2), F.avg_pool1d(z.imag, 2))


class ComplexIQEncoder(nn.Module):
    """Dataset-agnostic complex-I/Q sensor producing one prototype state."""

    def __init__(self, state_dim: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        widths = (hidden, hidden * 2, hidden * 4)
        self.blocks = nn.ModuleList([
            ComplexBlock(1, widths[0], 7, dropout * 0.25),
            ComplexBlock(widths[0], widths[1], 5, dropout * 0.25),
            ComplexBlock(widths[1], widths[2], 3, dropout * 0.25),
        ])
        self.project = nn.Sequential(
            nn.Linear(widths[-1] * 2, widths[-1] * 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(widths[-1] * 2, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, :1], x[:, 1:2])
        for block in self.blocks:
            z = block(z)
        pooled = torch.cat([z.real.mean(-1), z.imag.mean(-1)], dim=1)
        return F.normalize(self.project(pooled), dim=-1)


def _summary(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    top2 = p.topk(2, dim=-1).values
    return {
        "prob": p,
        "pred": logits.argmax(dim=-1),
        "confidence": top2[:, 0],
        "margin": top2[:, 0] - top2[:, 1],
        "entropy": -(p * log_p).sum(dim=-1),
        "energy": -torch.logsumexp(logits, dim=-1),
    }


class IdentityAgent(nn.Module):
    name = "identity"

    def __init__(self, num_classes: int, state_dim: int = 64,
                 stem_channels: int = 64, message_dim: int = 32):
        super().__init__()
        self.stem = SharedStem(stem_channels)
        self.encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
        self.classifier = nn.Linear(state_dim, num_classes)
        self.register_buffer("prototypes", torch.zeros(num_classes, state_dim))
        self.register_buffer("prototypes_ready", torch.tensor(False))
        self.message_head = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, message_dim), nn.Tanh()
        )
        self.state_dim = state_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> AgentOutput:
        state = self.encoder(self.stem(x))
        return self.forward_from_state(state)

    def forward_from_state(self, state: torch.Tensor) -> AgentOutput:
        """Re-evaluate a generated Identity state; no source report is reused."""
        logits = self.classifier(state)
        ev = _summary(logits)
        normalized_state = F.normalize(state, dim=-1)
        distances = torch.cdist(normalized_state, F.normalize(
            self.prototypes, dim=-1), p=2.0) ** 2
        if not bool(self.prototypes_ready):
            distances = torch.zeros_like(distances)
        d2 = distances.topk(2, dim=-1, largest=False).values
        ev.update({"distances": distances, "d1": d2[:, 0], "d2": d2[:, 1],
                   "distance_margin": d2[:, 1] - d2[:, 0]})
        reliability = ev["confidence"] * (1.0 - ev["entropy"] / max(float(torch.log(
            torch.tensor(self.num_classes, device=state.device))), 1e-6))
        unknown = torch.stack([1.0 - ev["confidence"], ev["entropy"], ev["energy"]], dim=1)
        return AgentOutput(state, logits, unknown, reliability.clamp(0, 1),
                           self.message_head(state), ev)

    @torch.no_grad()
    def set_empirical_prototypes(self, states: torch.Tensor, labels: torch.Tensor) -> None:
        normalized = F.normalize(states, dim=-1)
        rows = []
        for cls in range(self.num_classes):
            mask = labels == cls
            if not bool(mask.any()):
                raise ValueError(f"class {cls} has no Identity states for prototype refresh")
            rows.append(F.normalize(normalized[mask].mean(dim=0), dim=0))
        self.prototypes.copy_(torch.stack(rows))
        self.prototypes_ready.fill_(True)


class GeometryAgent(nn.Module):
    name = "geometry"

    def __init__(self, num_classes: int, state_dim: int = 64,
                 stem_channels: int = 64, message_dim: int = 32,
                 temperature: float = 0.15, view_mode: str = "spectral"):
        super().__init__()
        if view_mode == "universal":
            # Fixed sensor bank: identical architecture and initialization for
            # every dataset, but each sensor can specialize during training.
            base_stem = SharedStem(stem_channels)
            base_encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
            self.view_stems = nn.ModuleList([
                copy.deepcopy(base_stem) for _ in REAL_VIEW_NAMES])
            self.view_encoders = nn.ModuleList([
                copy.deepcopy(base_encoder) for _ in REAL_VIEW_NAMES])
            self.complex_encoder = ComplexIQEncoder(state_dim, hidden=64)
            self.stem = None
            self.encoder = None
        else:
            self.stem = SharedStem(stem_channels)
            self.encoder = PrivateEncoder(stem_channels, state_dim, hidden=stem_channels)
            self.view_stems = None
            self.view_encoders = None
            self.complex_encoder = None
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_classes, state_dim), dim=-1))
        view_count = len(UNIVERSAL_VIEW_NAMES) if view_mode == "universal" else 1
        self.register_buffer("view_prototypes", torch.zeros(
            view_count, num_classes, state_dim))
        self.register_buffer("view_prototypes_ready", torch.tensor(False))
        self.message_head = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, message_dim), nn.Tanh()
        )
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.view_mode = view_mode
        if view_mode == "universal":
            gate_feature_dim = state_dim + 3
            self.view_gate = nn.Sequential(
                nn.LayerNorm(len(UNIVERSAL_VIEW_NAMES) * gate_feature_dim),
                nn.Linear(len(UNIVERSAL_VIEW_NAMES) * gate_feature_dim, state_dim),
                nn.GELU(), nn.Linear(state_dim, len(UNIVERSAL_VIEW_NAMES)),
            )
            nn.init.zeros_(self.view_gate[-1].weight)
            nn.init.zeros_(self.view_gate[-1].bias)
        else:
            self.view_gate = None

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        z = F.normalize(state, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return z @ p.T / max(self.temperature, 1e-6)

    def forward(self, x: torch.Tensor) -> AgentOutput:
        if self.view_mode == "universal":
            # Each fixed sensor produces a local proposal inside one Geometry
            # Agent.  The sensor bank, prototype space and decision rule are
            # identical across datasets; only the learned per-sample action
            # changes.  It is therefore not a dataset switch or four agents.
            real_states = [
                F.normalize(encoder(stem(view)), dim=-1)
                for view, stem, encoder in zip(
                    universal_specialist_views(x), self.view_stems, self.view_encoders)
            ]
            view_states = torch.stack([*real_states, self.complex_encoder(x)], dim=1)
            flat_states = view_states.flatten(0, 1)
            view_logits = self.logits_from_state(flat_states).reshape(
                len(x), len(UNIVERSAL_VIEW_NAMES), self.num_classes)
            view_prob = F.softmax(view_logits, dim=-1)
            top2 = view_prob.topk(2, dim=-1).values
            entropy = -(view_prob * view_prob.clamp_min(1e-8).log()).sum(dim=-1)
            diagnostics = torch.stack([
                top2[..., 0], top2[..., 0] - top2[..., 1],
                1.0 - entropy / max(float(torch.log(torch.tensor(
                    self.num_classes, device=x.device))), 1e-6),
            ], dim=-1)
            descriptors = torch.cat([view_states, diagnostics], dim=-1).flatten(1)
            view_weights = F.softmax(self.view_gate(descriptors), dim=1)
            state = F.normalize((view_weights[:, :, None] * view_states).sum(dim=1), dim=-1)
            logits = (view_weights[:, :, None] * view_logits).sum(dim=1)
        else:
            feature = self.stem(specialist_view(x, self.view_mode))
            view_weights = x.new_ones((len(x), 1))
            state = F.normalize(self.encoder(feature), dim=-1)
            view_states = state[:, None, :]
            view_logits = self.logits_from_state(state)[:, None, :]
            logits = view_logits[:, 0, :]
        p = F.normalize(self.prototypes, dim=-1)
        distances = torch.cdist(state, p, p=2.0) ** 2
        d2, idx = distances.topk(2, dim=-1, largest=False)
        ev = _summary(logits)
        view_proto = F.normalize(self.view_prototypes, dim=-1)
        view_distances = ((view_states[:, :, None, :] - view_proto[None]) ** 2).sum(dim=-1)
        if not bool(self.view_prototypes_ready):
            view_distances = torch.zeros_like(view_distances)
        view_d2 = view_distances.topk(2, dim=-1, largest=False).values
        ev.update({"distances": distances, "d1": d2[:, 0], "d2": d2[:, 1],
                   "distance_margin": d2[:, 1] - d2[:, 0],
                   "view_weights": view_weights, "view_states": view_states,
                   "view_logits": view_logits, "view_distances": view_distances,
                   "view_d1": view_d2[:, :, 0],
                   "view_distance_margin": view_d2[:, :, 1] - view_d2[:, :, 0]})
        reliability = torch.exp(-d2[:, 0]) * ev["confidence"]
        unknown = torch.stack([d2[:, 0], -ev["distance_margin"],
                               1.0 - ev["confidence"]], dim=1)
        return AgentOutput(state, logits, unknown, reliability.clamp(0, 1),
                           self.message_head(state), ev)

    def forward_from_state(self, state: torch.Tensor) -> AgentOutput:
        """Re-evaluate a fused Geometry state generated in its learned space.

        View-local evidence is intentionally absent: a fused state cannot
        truthfully reconstruct the private sensor states or their gate action.
        """
        state = F.normalize(state, dim=-1)
        logits = self.logits_from_state(state)
        p = F.normalize(self.prototypes, dim=-1)
        distances = torch.cdist(state, p, p=2.0) ** 2
        d2 = distances.topk(2, dim=-1, largest=False).values
        ev = _summary(logits)
        ev.update({"distances": distances, "d1": d2[:, 0], "d2": d2[:, 1],
                   "distance_margin": d2[:, 1] - d2[:, 0]})
        reliability = torch.exp(-d2[:, 0]) * ev["confidence"]
        unknown = torch.stack([d2[:, 0], -ev["distance_margin"],
                               1.0 - ev["confidence"]], dim=1)
        return AgentOutput(state, logits, unknown, reliability.clamp(0, 1),
                           self.message_head(state), ev)

    @torch.no_grad()
    def set_empirical_prototypes(self, states: torch.Tensor, labels: torch.Tensor,
                                 view_states: torch.Tensor | None = None) -> None:
        rows = []
        for cls in range(self.num_classes):
            mask = labels == cls
            if not bool(mask.any()):
                raise ValueError(f"class {cls} has no states for prototype refresh")
            rows.append(F.normalize(states[mask].mean(dim=0), dim=0))
        self.prototypes.copy_(torch.stack(rows))
        if view_states is None:
            view_states = states[:, None, :]
        for view_index in range(view_states.shape[1]):
            for cls in range(self.num_classes):
                mask = labels == cls
                center = F.normalize(view_states[mask, view_index].mean(dim=0), dim=0)
                self.view_prototypes[view_index, cls].copy_(center)
        self.view_prototypes_ready.fill_(True)


class RoleStructuredExperts(nn.Module):
    """Two independent evidence agents; the Boundary Agent lives in the coordinator."""

    def __init__(self, num_classes: int, state_dim: int = 64,
                 stem_channels: int = 64, message_dim: int = 32,
                 prototype_temperature: float = 0.15,
                 geometry_view: str = "spectral"):
        super().__init__()
        self.identity = IdentityAgent(num_classes, state_dim, stem_channels, message_dim)
        self.geometry = GeometryAgent(num_classes, state_dim, stem_channels, message_dim,
                                      prototype_temperature, geometry_view)
        # Reconstruction is deliberately an auxiliary objective, not a third
        # inference-time expert.  Joint specialist states reconstruct a
        # downsampled IQ waveform and thereby retain signal information without
        # creating another set of class logits or a direct rejection vote.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(2 * state_dim, 128), nn.GELU(), nn.Linear(128, 2 * 64)
        )
        self.num_classes = num_classes
        self.state_dim = state_dim
        self.message_dim = message_dim
        self.geometry_view = geometry_view

    def forward(self, x: torch.Tensor) -> Dict[str, AgentOutput]:
        return {"identity": self.identity(x), "geometry": self.geometry(x)}

    def reconstruct(self, outputs: Dict[str, AgentOutput]) -> torch.Tensor:
        state = torch.cat([outputs["identity"].state, outputs["geometry"].state], dim=1)
        return self.reconstruction_head(state).reshape(len(state), 2, 64)


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    count = positive.sum(dim=1)
    valid = count > 0
    if not bool(valid.any()):
        return logits.new_zeros(())
    return -(log_prob * positive).sum(dim=1)[valid].div(count[valid]).mean()


def role_decorrelation_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = (a - a.mean(0)) / a.std(0).clamp_min(1e-5)
    b = (b - b.mean(0)) / b.std(0).clamp_min(1e-5)
    cross = a.T @ b / max(len(a) - 1, 1)
    return cross.square().mean()
