"""Passive local evidence networks; tools have no communication authority."""
from __future__ import annotations

import torch
from torch import nn

from maros_stage5.agents import ComplexIQEncoder

from ..tool_registry import EvidenceTool
from .observations import identity_view, impairment_view


_CHANNELS = {
    "raw": 2, "complex": 2, "envelope": 2, "difference": 2,
    "robust_spectrum": 6, "frequency_difference": 3,
    "iq_imbalance": 4, "phase_noise": 3,
}


class EncoderTool(EvidenceTool):
    """One independently trained local proposal, not an Agent."""

    def __init__(self, name: str, role: str, num_classes: int,
                 state_dim: int = 48, width: int = 24) -> None:
        super().__init__()
        if name not in _CHANNELS:
            raise KeyError(name)
        if (role == "identity") != (name in {"raw", "complex", "envelope", "difference"}):
            raise ValueError("tool is outside the role's private capability family")
        self.name = name
        self.role = role
        if name == "complex":
            self.encoder = ComplexIQEncoder(state_dim, hidden=width)
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(_CHANNELS[name], width, 9, stride=2, padding=4),
                nn.BatchNorm1d(width), nn.GELU(),
                nn.Conv1d(width, width * 2, 7, stride=2, padding=3),
                nn.BatchNorm1d(width * 2), nn.GELU(),
                nn.Conv1d(width * 2, state_dim, 5, stride=2, padding=2),
                nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
        self.classifier = nn.Linear(state_dim, num_classes)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        view = (identity_view(iq, self.name) if self.role == "identity"
                else impairment_view(iq, self.name))
        return self.classifier(self.encoder(view))

