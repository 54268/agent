"""Open-Set Examiner with private Known-support tail memory.

C receives only A/B public scalar packets. It has neither a raw-I/Q encoder
nor a K-way identity head and cannot reconstruct either Agent's hidden state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .consultation import PublicSupportPacket


def identity_public_after_scores(scores: np.ndarray,
                                 original_public: np.ndarray) -> np.ndarray:
    """A publishes updated confidence, margin, entropy and two proto scalars."""
    scores = np.asarray(scores, dtype=np.float64)
    original = np.asarray(original_public, dtype=np.float64)
    if scores.ndim != 2 or original.shape != (len(scores), 5):
        raise ValueError("A public packet requires [N,K] private scores and [N,5] public")
    shifted = scores - scores.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    top2 = np.sort(probability, axis=1)[:, -2:]
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(1)
    return np.column_stack([
        top2[:, 1], top2[:, 1] - top2[:, 0], entropy,
        original[:, 3], original[:, 4],
    ]).astype(np.float32)


@dataclass(frozen=True)
class OpenSetOpinion:
    risk: np.ndarray
    dominant_reason: np.ndarray


class OpenSetExaminer:
    """A class-conditional empirical tail memory over public evidence."""

    feature_names = (
        "low_identity_confidence", "identity_entropy",
        "identity_prototype_distance", "geometry_candidate_distance",
        "low_geometry_candidate_confidence", "low_view_agreement",
    )

    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("Known support requires at least two classes")
        self.num_classes = int(num_classes)
        self.memory: dict[int, tuple[np.ndarray, ...]] = {}
        self.global_memory: tuple[np.ndarray, ...] | None = None

    @staticmethod
    def public_features(identity_public: np.ndarray,
                        support: PublicSupportPacket) -> np.ndarray:
        values = np.asarray(identity_public, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("C accepts only A's five public scalars")
        if support.candidate.shape != (len(values),):
            raise ValueError("B support packet and A packet have different sample counts")
        return np.column_stack([
            1.0 - values[:, 0], values[:, 2], values[:, 3],
            support.prototype_distance, 1.0 - support.candidate_confidence,
            1.0 - support.view_agreement,
        ]).astype(np.float32)

    def fit_memory(self, identity_public: np.ndarray,
                   support: PublicSupportPacket, known_labels: np.ndarray):
        labels = np.asarray(known_labels, dtype=np.int64)
        features = self.public_features(identity_public, support)
        if labels.shape != (len(features),) or np.any(labels < 0):
            raise ValueError("Open-Set memory cannot contain Unknown labels")
        self.global_memory = tuple(np.sort(features[:, index])
                                   for index in range(features.shape[1]))
        for class_index in range(self.num_classes):
            mask = labels == class_index
            if not mask.any():
                raise ValueError("every Known class requires enrollment support")
            self.memory[class_index] = tuple(
                np.sort(features[mask, index]) for index in range(features.shape[1]))
        return self

    def risk(self, identity_public: np.ndarray,
             support: PublicSupportPacket) -> tuple[np.ndarray, np.ndarray]:
        if self.global_memory is None:
            raise RuntimeError("C memory is not enrolled")
        features = self.public_features(identity_public, support)
        ranks = np.zeros_like(features, dtype=np.float32)
        for row, candidate in enumerate(support.candidate):
            memory = self.memory.get(int(candidate), self.global_memory)
            for column, sorted_known in enumerate(memory):
                ranks[row, column] = np.searchsorted(
                    sorted_known, features[row, column], side="right") / len(sorted_known)
        # Mean tail rank is fixed before seeing proxy Unknown. A single noisy
        # sensor cannot dominate as it would with a max rule.
        return ranks.mean(1), ranks.argmax(1).astype(np.int64)

    def examine(self, identity_public: np.ndarray,
                support: PublicSupportPacket) -> OpenSetOpinion:
        risk, reason = self.risk(identity_public, support)
        return OpenSetOpinion(risk=risk, dominant_reason=reason)

