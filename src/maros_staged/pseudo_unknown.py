"""Disagreement-guided pseudo-unknown generation for staged MAROS-SEI.

The agent never observes a real unknown.  It observes the three frozen Stage-1
embeddings, their raw unknown evidence and their nearest-class opinions, then
selects difficult known boundary samples and acts by moving them through the
joint multi-agent latent space.  Per-agent unit-norm constraints and a known
boundary shell prevent trivial far-away noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors


AGENTS = ("identity", "prototype", "reconstruction")


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def normalise_agents(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: l2_normalize(embeddings[name]) for name in AGENTS}


def concatenate_agents(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([embeddings[name] for name in AGENTS], axis=1).astype(np.float32)


def split_agents(fused: np.ndarray, dimensions: Dict[str, int]) -> Dict[str, np.ndarray]:
    out, start = {}, 0
    for name in AGENTS:
        end = start + dimensions[name]
        out[name] = l2_normalize(fused[:, start:end])
        start = end
    if start != fused.shape[1]:
        raise ValueError("agent dimensions do not match fused embedding width")
    return out


def class_prototypes(embeddings: Dict[str, np.ndarray], labels: np.ndarray,
                     num_classes: int) -> Dict[str, np.ndarray]:
    result = {}
    for name in AGENTS:
        z = l2_normalize(embeddings[name])
        p = np.zeros((num_classes, z.shape[1]), dtype=np.float32)
        for cls in range(num_classes):
            mask = labels == cls
            if not np.any(mask):
                raise ValueError(f"class {cls} has no training samples")
            p[cls] = z[mask].mean(axis=0)
        result[name] = l2_normalize(p)
    return result


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float32) / max(len(values) - 1, 1)


def _per_class_rank(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        out[mask] = _rank01(values[mask])
    return out


@dataclass
class PseudoUnknownConfig:
    seed_ratio: float = 0.20
    variations: int = 6
    eta_min: float = 0.45
    eta_max: float = 1.35
    repel_weight: float = 0.35
    jitter: float = 0.10
    noise: float = 0.04
    knn_k: int = 10
    shell_low_quantile: float = 0.90
    shell_high_quantile: float = 0.999
    seed: int = 42


class DisagreementGuidedPseudoUnknownAgent:
    """A deterministic policy Agent that mines and perturbs hard knowns."""

    def __init__(self, config: PseudoUnknownConfig):
        self.config = config

    def generate(self, embeddings: Dict[str, np.ndarray], labels: np.ndarray,
                 raw_evidence: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        labels = np.asarray(labels, dtype=np.int64)
        z = normalise_agents(embeddings)
        dims = {name: z[name].shape[1] for name in AGENTS}
        fused = concatenate_agents(z)
        num_classes = int(labels.max()) + 1
        proto_by_agent = class_prototypes(z, labels, num_classes)
        fused_proto = concatenate_agents(proto_by_agent)

        dist = np.linalg.norm(fused[:, None, :] - fused_proto[None, :, :], axis=2)
        own = dist[np.arange(len(labels)), labels]
        foreign_dist = dist.copy()
        foreign_dist[np.arange(len(labels)), labels] = np.inf
        nearest_foreign = foreign_dist.argmin(axis=1)
        competition_gap = foreign_dist.min(axis=1) - own

        opinions = []
        for name in AGENTS:
            d = 1.0 - z[name] @ proto_by_agent[name].T
            opinions.append(d.argmin(axis=1))
        opinions = np.stack(opinions, axis=1)
        disagreement = (
            (opinions[:, 0] != opinions[:, 1]).astype(np.float32)
            + (opinions[:, 0] != opinions[:, 2]).astype(np.float32)
            + (opinions[:, 1] != opinions[:, 2]).astype(np.float32)
        ) / 3.0

        raw_evidence = np.asarray(raw_evidence, dtype=np.float32)
        if raw_evidence.shape != (len(labels), 3):
            raise ValueError("raw_evidence must have shape [N, 3]")
        evidence_rank = np.stack(
            [_per_class_rank(raw_evidence[:, i], labels) for i in range(3)], axis=1
        ).mean(axis=1)
        # High own distance, small competition gap, cross-agent disagreement
        # and high raw anomaly evidence all make a known sample a useful seed.
        seed_score = (
            0.30 * _per_class_rank(own, labels)
            + 0.25 * (1.0 - _per_class_rank(competition_gap, labels))
            + 0.25 * disagreement
            + 0.20 * evidence_rank
        )

        local_scale = np.empty(len(labels), dtype=np.float32)
        selected = []
        for cls in range(num_classes):
            idx = np.where(labels == cls)[0]
            use_k = min(cfg.knn_k + 1, len(idx))
            nn = NearestNeighbors(n_neighbors=use_k, metric="euclidean").fit(fused[idx])
            distances = nn.kneighbors(fused[idx], return_distance=True)[0]
            local_scale[idx] = distances[:, 1:].mean(axis=1) if use_k > 1 else 1.0
            count = max(1, int(np.ceil(len(idx) * cfg.seed_ratio)))
            selected.extend(idx[np.argsort(-seed_score[idx])[:count]].tolist())
        selected = np.asarray(sorted(selected), dtype=np.int64)

        known_nearest = dist.min(axis=1)
        shell_low = float(np.quantile(known_nearest, cfg.shell_low_quantile))
        shell_high = float(np.quantile(known_nearest, cfg.shell_high_quantile))
        etas = np.linspace(cfg.eta_min, cfg.eta_max, cfg.variations, dtype=np.float32)

        generated, sources, source_classes, eta_used, in_shell = [], [], [], [], []
        for index in selected:
            cls = int(labels[index])
            foreign = int(nearest_foreign[index])
            outward = fused[index] - fused_proto[cls]
            repel = fused[index] - fused_proto[foreign]
            outward /= max(float(np.linalg.norm(outward)), 1e-8)
            repel /= max(float(np.linalg.norm(repel)), 1e-8)
            direction = (1.0 - cfg.repel_weight) * outward + cfg.repel_weight * repel
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            for eta in etas:
                actual_eta = float(eta * rng.uniform(1.0 - cfg.jitter, 1.0 + cfg.jitter))
                noise = rng.normal(size=fused.shape[1]).astype(np.float32)
                noise -= float(noise @ direction) * direction
                noise /= max(float(np.linalg.norm(noise)), 1e-8)
                candidate = (
                    fused[index]
                    + actual_eta * float(local_scale[index]) * direction
                    + cfg.noise * float(local_scale[index]) * noise
                )[None, :]
                # Eliminate the easiest generator artefact: every agent block
                # remains on the same unit sphere as real Stage-1 embeddings.
                candidate_agents = split_agents(candidate, dims)
                candidate = concatenate_agents(candidate_agents)[0]
                nearest = float(np.linalg.norm(fused_proto - candidate[None, :], axis=1).min())
                generated.append(candidate)
                sources.append(int(index))
                source_classes.append(cls)
                eta_used.append(actual_eta)
                in_shell.append(shell_low <= nearest <= shell_high)

        pseudo_fused = np.asarray(generated, dtype=np.float32)
        pseudo_agents = split_agents(pseudo_fused, dims)
        result = {
            **{f"z_{name}": pseudo_agents[name] for name in AGENTS},
            "fused": pseudo_fused,
            "source_indices": np.asarray(sources, dtype=np.int64),
            "source_classes": np.asarray(source_classes, dtype=np.int64),
            "eta": np.asarray(eta_used, dtype=np.float32),
            "in_shell": np.asarray(in_shell, dtype=bool),
            "seed_score": seed_score,
            "selected_source_indices": selected,
            "known_nearest_distance": known_nearest.astype(np.float32),
            "shell_low": np.asarray(shell_low, dtype=np.float32),
            "shell_high": np.asarray(shell_high, dtype=np.float32),
            "config": asdict(cfg),
        }
        return result

