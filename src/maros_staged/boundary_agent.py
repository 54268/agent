"""Open-space boundary Agent trained on known and generated pseudo-unknowns."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .pseudo_unknown import AGENTS, class_prototypes, concatenate_agents, normalise_agents


def boundary_features(embeddings: Dict[str, np.ndarray],
                      prototypes: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Create a common input with embeddings + comparable geometry evidence."""
    z = normalise_agents(embeddings)
    fused = concatenate_agents(z)
    geom, opinions = [], []
    for name in AGENTS:
        p = prototypes[name]
        distance = np.maximum(1.0 - z[name] @ p.T, 0.0)
        order = np.argsort(distance, axis=1)
        d1 = distance[np.arange(len(fused)), order[:, 0]]
        d2 = distance[np.arange(len(fused)), order[:, 1]]
        prob = np.exp(-distance - (-distance).max(axis=1, keepdims=True))
        prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(prob * np.log(np.maximum(prob, 1e-12))).sum(axis=1) / np.log(prob.shape[1])
        geom.extend([d1, d2 - d1, d1 / np.maximum(d2, 1e-6), entropy])
        opinions.append(order[:, 0])
    opinions = np.stack(opinions, axis=1)
    geom.extend([
        (opinions[:, 0] != opinions[:, 1]).astype(np.float32),
        (opinions[:, 0] != opinions[:, 2]).astype(np.float32),
        (opinions[:, 1] != opinions[:, 2]).astype(np.float32),
    ])
    geometry = np.stack(geom, axis=1).astype(np.float32)
    return np.concatenate([fused, geometry], axis=1).astype(np.float32), opinions


class FeatureStandardizer:
    def fit(self, x: np.ndarray) -> "FeatureStandardizer":
        self.mean = np.asarray(x, dtype=np.float32).mean(axis=0)
        self.std = np.asarray(x, dtype=np.float32).std(axis=0)
        self.std = np.maximum(self.std, 1e-5)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


class OpenSpaceBoundaryAgent(nn.Module):
    """Binary open-space critic plus an auxiliary known-class head."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 192,
                 dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.unknown_head = nn.Linear(hidden_dim // 2, 1)
        self.class_head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "state": h,
            "unknown_logit": self.unknown_head(h).squeeze(-1),
            "class_logits": self.class_head(h),
        }


def make_prototypes(embeddings: Dict[str, np.ndarray], labels: np.ndarray,
                    num_classes: int) -> Dict[str, np.ndarray]:
    return class_prototypes(embeddings, labels, num_classes)

