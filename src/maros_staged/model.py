"""Stage-1 three-agent model and its losses (design doc Sections 6-9, 24)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .agents import IdentityAgent, PrototypeAgent, ReconstructionAgent
from .stem import SharedStem


@dataclass
class Stage1ModelConfig:
    num_classes: int
    embedding_dim: int = 128
    stem_channels: int = 128
    proto_temperature: float = 1.0
    proto_momentum: float = 0.9
    compact_weight: float = 0.1
    margin_weight: float = 0.1
    margin: float = 1.0
    recon_weight: float = 1.0
    proto_weight: float = 1.0
    recon_top_k: int = 3
    latent_channels: int = 16
    signal_length: int = 256
    rec_wrong_weight: float = 1.0
    rec_margin: float = 0.1


class ThreeAgentModel(nn.Module):
    """Shared stem + Identity / Prototype / Reconstruction agents."""

    agent_names = ("identity", "prototype", "reconstruction")

    def __init__(self, config: Stage1ModelConfig):
        super().__init__()
        self.config = config
        self.stem = SharedStem(config.stem_channels)
        c = config.stem_channels
        self.identity = IdentityAgent(c, config.num_classes, config.embedding_dim)
        self.prototype = PrototypeAgent(
            c, config.num_classes, config.embedding_dim,
            temperature=config.proto_temperature, momentum=config.proto_momentum,
        )
        self.reconstruction = ReconstructionAgent(
            c, config.num_classes, config.embedding_dim,
            latent_channels=config.latent_channels, signal_length=config.signal_length,
        )

    def forward(self, x: torch.Tensor, recon_classes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h_s = self.stem(x)
        id_out = self.identity(h_s)
        proto_out = self.prototype(h_s)
        out = {"x": x, "h_s": h_s, "identity": id_out, "prototype": proto_out}
        if recon_classes is not None:
            out["reconstruction"] = self.reconstruction(x, h_s, recon_classes)
        return out

    # ------------------------------------------------------------------ losses
    def losses(self, out: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.config
        id_logits = out["identity"]["evidence"]["logits"]
        l_id = F.cross_entropy(id_logits, labels)

        distances = out["prototype"]["evidence"]["distances"]
        l_proto_cls = F.cross_entropy(-distances / max(cfg.proto_temperature, 1e-6), labels)
        d_own = distances[torch.arange(labels.size(0), device=labels.device), labels]
        d_competitor = distances.masked_fill(
            F.one_hot(labels, cfg.num_classes).bool(), float("inf")
        ).min(dim=1).values
        l_compact = d_own.mean()
        l_margin = F.relu(cfg.margin + d_own - d_competitor).mean()
        l_proto = l_proto_cls + cfg.compact_weight * l_compact + cfg.margin_weight * l_margin

        l_rec_true = out["reconstruction"]["evidence"]["error"]
        # Hard-negative wrong-class reconstruction margin.  The wrong target is
        # the nearest *competing* prototype (where an unknown would actually be
        # routed), not a random far class, so the decoder cannot reconstruct
        # nearest-neighbour unknowns equally well.
        one_hot = F.one_hot(labels, cfg.num_classes).bool()
        hard_wrong = distances.masked_fill(one_hot, float("inf")).argmin(dim=1)
        l_rec_wrong = self.reconstruction.reconstruction_error(
            out["x"], out["h_s"], hard_wrong
        )
        l_rec_hinge = F.relu(cfg.rec_margin - (l_rec_wrong - l_rec_true)).mean()
        l_rec = l_rec_true.mean() + cfg.rec_wrong_weight * l_rec_hinge

        total = l_id + cfg.proto_weight * l_proto + cfg.recon_weight * l_rec
        return {
            "total": total,
            "identity_ce": l_id.detach(),
            "proto_cls": l_proto_cls.detach(),
            "proto_compact": l_compact.detach(),
            "proto_margin": l_margin.detach(),
            "recon_mse": l_rec_true.mean().detach(),
            "recon_wrong_gap": (l_rec_wrong - l_rec_true).mean().detach(),
        }

    @torch.no_grad()
    def refresh_prototypes(self, loader: DataLoader, device: torch.device) -> None:
        self.eval()
        zs, ys = [], []
        for x, y in loader:
            h_s = self.stem(x.to(device))
            z = F.normalize(self.prototype.encoder(h_s), dim=-1).cpu()
            zs.append(z)
            ys.append(y)
        z = torch.cat(zs).numpy()
        y = torch.cat(ys).numpy()
        protos = np.zeros((self.config.num_classes, z.shape[1]), dtype=np.float32)
        for cls in range(self.config.num_classes):
            mask = y == cls
            if mask.any():
                protos[cls] = z[mask].mean(axis=0)
        self.prototype.set_prototypes(torch.from_numpy(protos))
