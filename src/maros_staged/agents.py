"""The three Stage-1 evidence-specialized agents (design doc Sections 7-9).

Every agent follows the same contract: it observes the shared feature map
``h_s`` through a *private* encoder and returns an ``evidence`` dict plus an
embedding.  The three agents deliberately answer different questions:

* :class:`IdentityAgent`        - "which known emitter does it look like?"
* :class:`PrototypeAgent`       - "where is it relative to known geometry?"
* :class:`ReconstructionAgent` - "can the known manifold explain it?"
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .stem import PrivateEncoder


class BaseAgent(nn.Module):
    name = "base"

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def observe(self, h_s: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class IdentityAgent(BaseAgent):
    name = "identity"

    def __init__(self, in_channels: int, num_classes: int, dim: int = 128):
        super().__init__(dim)
        self.encoder = PrivateEncoder(in_channels, dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.num_classes = num_classes

    def forward(self, h_s: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(h_s)
        logits = self.classifier(z)
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        top2 = p.topk(2, dim=-1).values
        evidence = {
            "logits": logits,
            "prob": p,
            "pred": logits.argmax(dim=-1),
            "confidence": top2[:, 0],
            "margin": top2[:, 0] - top2[:, 1],
            "entropy": -(p * log_p).sum(dim=-1),
        }
        return {"embedding": z, "evidence": evidence, "local_pred": evidence["pred"]}


class PrototypeAgent(BaseAgent):
    name = "prototype"

    def __init__(self, in_channels: int, num_classes: int, dim: int = 128,
                 temperature: float = 1.0, momentum: float = 0.9):
        super().__init__(dim)
        self.encoder = PrivateEncoder(in_channels, dim)
        self.num_classes = num_classes
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.register_buffer("prototypes", torch.zeros(num_classes, dim))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, h_s: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = F.normalize(self.encoder(h_s), dim=-1)
        distances = torch.cdist(z, self.prototypes, p=2.0) ** 2
        logits = -distances / max(self.temperature, 1e-6)
        p = F.softmax(logits, dim=-1)
        d1, idx1 = distances.min(dim=-1)
        d2 = distances.topk(2, dim=-1, largest=False).values[:, 1]
        evidence = {
            "distances": distances,
            "logits": logits,
            "prob": p,
            "pred": idx1,
            "d1": d1,
            "d2": d2,
            "margin": d2 - d1,
        }
        return {"embedding": z, "evidence": evidence, "local_pred": idx1}

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        self.prototypes.copy_(prototypes.to(self.prototypes))
        self.initialized.fill_(True)

    @torch.no_grad()
    def update_ema(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        if not bool(self.initialized):
            return
        for cls in labels.unique():
            mask = labels == cls
            cls_mean = z[mask].mean(dim=0)
            self.prototypes[int(cls)].mul_(self.momentum).add_(
                cls_mean, alpha=1.0 - self.momentum
            )


class ClassConditionedInputReconstructor(nn.Module):
    """Class-conditioned decoder that reconstructs the *raw IQ* input.

    Design doc Section 9.2 prefers reconstructing ``x`` ("evidence is more
    direct") over reconstructing learned features (which can become trivially
    easy).  A narrow bottleneck + FiLM class modulation + the hard-negative
    reconstruction margin make the class identity load-bearing: an unseen
    emitter cannot be explained by any known-class decoder.
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 40,
                 latent_channels: int = 16, out_length: int = 256):
        super().__init__()
        self.out_length = out_length
        self.bottleneck = nn.Conv1d(in_channels, latent_channels, kernel_size=1)
        self.class_embedding = nn.Embedding(num_classes, 32)
        self.film = nn.Linear(32, 2 * latent_channels)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 32, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),
        )
        self.latent_channels = latent_channels

    def forward(self, h_s: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(h_s)
        gamma_beta = self.film(self.class_embedding(classes))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        b = b * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        x_hat = self.up(b)
        if x_hat.shape[-1] != self.out_length:
            x_hat = F.interpolate(x_hat, size=self.out_length, mode="linear", align_corners=False)
        return x_hat


class ReconstructionAgent(BaseAgent):
    name = "reconstruction"

    def __init__(self, in_channels: int, num_classes: int, dim: int = 128,
                 latent_channels: int = 16, signal_length: int = 256):
        super().__init__(dim)
        self.decoder = ClassConditionedInputReconstructor(
            in_channels, num_classes, latent_channels, signal_length
        )

    def reconstruct(self, h_s: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        return self.decoder(h_s, classes)

    def reconstruction_error(self, x: torch.Tensor, h_s: torch.Tensor,
                             classes: torch.Tensor) -> torch.Tensor:
        """Mean per-element squared IQ reconstruction error."""
        x_hat = self.reconstruct(h_s, classes)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))

    def forward(self, x: torch.Tensor, h_s: torch.Tensor,
                classes: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = self.decoder.bottleneck(h_s)
        x_hat = self.reconstruct(h_s, classes)
        error = ((x - x_hat) ** 2).mean(dim=(1, 2))
        # Use the trained decoder bottleneck itself as the reconstruction
        # representation.  The previous implementation passed it through an
        # otherwise unused MLP.  That MLP received no gradient from any loss,
        # so CKA/t-SNE on its random output was not valid evidence of agent
        # diversity.  ``b`` is directly load-bearing for IQ reconstruction.
        z_r = F.normalize(b.mean(dim=-1), dim=-1)
        return {"embedding": z_r, "evidence": {"error": error, "x_hat": x_hat},
                "local_pred": torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)}

    @torch.no_grad()
    def best_case_error(self, x: torch.Tensor, h_s: torch.Tensor,
                        candidate_classes: torch.Tensor) -> torch.Tensor:
        errors = []
        for k in range(candidate_classes.shape[1]):
            errors.append(self.reconstruction_error(x, h_s, candidate_classes[:, k]))
        return torch.stack(errors, dim=1).min(dim=1).values
