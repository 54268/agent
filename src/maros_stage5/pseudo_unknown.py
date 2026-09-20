"""PUG-V2: competition-aware and disagreement-aware proxy unknowns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float32) / max(len(values) - 1, 1)


def _class_rank(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        out[mask] = _rank(values[mask])
    return out


def _prototypes(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = []
    for cls in range(int(labels.max()) + 1):
        rows.append(_l2(z[labels == cls].mean(axis=0, keepdims=True))[0])
    return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class PUGConfig:
    kind: str = "mixed"
    eta: float = 1.5
    seed_ratio: float = 0.15
    variations: int = 4
    repel_weight: float = 0.35
    jitter: float = 0.10
    noise: float = 0.03
    knn_k: int = 8
    seed: int = 42


@dataclass
class PseudoUnknownBatch:
    identity_state: np.ndarray
    geometry_state: np.ndarray
    source_indices: np.ndarray
    source_classes: np.ndarray
    kinds: np.ndarray


@dataclass
class ExplorerOutput:
    proposals: PseudoUnknownBatch
    reliability: float
    evidence: dict


class BoundaryExplorerAgent:
    """Training-time agent that actively proposes open-space boundary samples."""

    name = "boundary_explorer"

    def __init__(self, config: PUGConfig):
        self.config = config

    def act(self, identity_state: np.ndarray, geometry_state: np.ndarray,
            labels: np.ndarray, identity_pred: np.ndarray,
            geometry_pred: np.ndarray, raw_evidence: np.ndarray) -> ExplorerOutput:
        proposals = generate_pug_v2(identity_state, geometry_state, labels,
                                    identity_pred, geometry_pred, raw_evidence,
                                    self.config)
        counts = {str(key): int(value) for key, value in zip(
            *np.unique(proposals.kinds, return_counts=True))}
        source_coverage = len(np.unique(proposals.source_indices)) / max(len(labels), 1)
        class_coverage = len(np.unique(proposals.source_classes)) / max(len(np.unique(labels)), 1)
        # A local confidence about proposal coverage, not a claim that the
        # proposals resemble unseen test devices.  LCO supplies that feedback.
        reliability = float(np.sqrt(source_coverage * class_coverage))
        return ExplorerOutput(proposals, reliability, {
            "kind_counts": counts, "source_coverage": float(source_coverage),
            "class_coverage": float(class_coverage),
        })


def generate_pug_v2(identity_state: np.ndarray, geometry_state: np.ndarray,
                    labels: np.ndarray, identity_pred: np.ndarray,
                    geometry_pred: np.ndarray, raw_evidence: np.ndarray,
                    config: PUGConfig) -> PseudoUnknownBatch:
    if config.kind not in {"competition", "disagreement", "mixed"}:
        raise ValueError(f"unsupported PUG kind: {config.kind}")
    labels = np.asarray(labels, dtype=np.int64)
    zi, zg = _l2(identity_state), _l2(geometry_state)
    di, dg = zi.shape[1], zg.shape[1]
    fused = _l2(np.concatenate([zi, zg], axis=1))
    proto = _prototypes(fused, labels)
    distances = np.linalg.norm(fused[:, None] - proto[None], axis=2)
    own = distances[np.arange(len(labels)), labels]
    foreign = distances.copy()
    foreign[np.arange(len(labels)), labels] = np.inf
    nearest_foreign = foreign.argmin(axis=1)
    gap = foreign.min(axis=1) - own

    local_scale = np.ones(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        k = min(config.knn_k + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=k).fit(fused[idx])
        d = nn.kneighbors(fused[idx], return_distance=True)[0]
        local_scale[idx] = d[:, 1:].mean(axis=1) if k > 1 else 1.0

    disagreement = (np.asarray(identity_pred) != np.asarray(geometry_pred)).astype(np.float32)
    evidence = np.asarray(raw_evidence, dtype=np.float32)
    if evidence.ndim != 2 or len(evidence) != len(labels):
        raise ValueError("raw_evidence must be [N,E]")
    anomaly = np.stack([_class_rank(evidence[:, col], labels)
                        for col in range(evidence.shape[1])], axis=1).mean(axis=1)
    competition_score = 0.55 * _class_rank(own, labels) + 0.45 * (
        1.0 - _class_rank(gap, labels))
    disagreement_score = 0.45 * disagreement + 0.30 * anomaly + 0.25 * competition_score
    rng = np.random.default_rng(config.seed)

    def seeds_for(score: np.ndarray) -> np.ndarray:
        selected = []
        for cls in np.unique(labels):
            idx = np.where(labels == cls)[0]
            count = max(1, int(np.ceil(len(idx) * config.seed_ratio)))
            selected.extend(idx[np.argsort(-score[idx])[:count]])
        return np.asarray(sorted(selected), dtype=np.int64)

    kinds = ("competition", "disagreement") if config.kind == "mixed" else (config.kind,)
    generated, sources, classes, labels_kind = [], [], [], []
    for kind in kinds:
        score = competition_score if kind == "competition" else disagreement_score
        for index in seeds_for(score):
            cls = int(labels[index]); other = int(nearest_foreign[index])
            outward = fused[index] - proto[cls]
            repel = fused[index] - proto[other]
            outward /= max(float(np.linalg.norm(outward)), 1e-8)
            repel /= max(float(np.linalg.norm(repel)), 1e-8)
            if kind == "competition":
                direction = (1.0 - config.repel_weight) * outward + config.repel_weight * repel
            else:
                # Disagreement samples move more strongly toward a competing boundary.
                weight = 0.55 + 0.25 * float(disagreement[index])
                direction = (1.0 - weight) * outward + weight * repel
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            for _ in range(config.variations):
                eta = config.eta * rng.uniform(1.0 - config.jitter, 1.0 + config.jitter)
                noise = rng.normal(size=fused.shape[1]).astype(np.float32)
                noise -= float(noise @ direction) * direction
                noise /= max(float(np.linalg.norm(noise)), 1e-8)
                candidate = fused[index] + eta * local_scale[index] * direction
                candidate += config.noise * local_scale[index] * noise
                generated.append(candidate)
                sources.append(int(index)); classes.append(cls); labels_kind.append(kind)
    z = np.asarray(generated, dtype=np.float32)
    # Normalize each agent block separately so the generator cannot be detected
    # from a trivial norm artefact.
    pseudo_i, pseudo_g = _l2(z[:, :di]), _l2(z[:, di:di + dg])
    if config.kind == "mixed":
        # Preserve an exact 1:1 generator balance for every class.
        keep = []
        kinds_arr, cls_arr = np.asarray(labels_kind), np.asarray(classes)
        for cls in np.unique(cls_arr):
            a = np.where((cls_arr == cls) & (kinds_arr == "competition"))[0]
            b = np.where((cls_arr == cls) & (kinds_arr == "disagreement"))[0]
            count = min(len(a), len(b))
            keep.extend(a[:count]); keep.extend(b[:count])
        keep = np.asarray(sorted(keep), dtype=np.int64)
        pseudo_i, pseudo_g = pseudo_i[keep], pseudo_g[keep]
        sources = np.asarray(sources, dtype=np.int64)[keep]
        classes = np.asarray(classes, dtype=np.int64)[keep]
        labels_kind = np.asarray(labels_kind)[keep]
    return PseudoUnknownBatch(
        pseudo_i, pseudo_g,
        np.asarray(sources, dtype=np.int64),
        np.asarray(classes, dtype=np.int64),
        np.asarray(labels_kind),
    )
