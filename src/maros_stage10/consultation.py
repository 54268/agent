"""Agent-initiated, candidate-specific one-turn consultation.

The verifier and query policy are deliberately separate: B owns multi-view
geometry evidence; A owns the decision to ask. No central winner router exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from maros_stage8.contracts import EvidenceCertificate, EvidenceStance
from maros_stage8.evidence import CandidateEvidenceTable

from .mother import GeometryPrivateEvidence


def top_pair(identity_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(identity_logits, axis=1)
    return order[:, -1].astype(np.int64), order[:, -2].astype(np.int64)


def pair_features(evidence: GeometryPrivateEvidence, candidate_a: np.ndarray,
                  candidate_b: np.ndarray) -> np.ndarray:
    """Anti-symmetric local evidence, never a full-logit message."""
    n = len(evidence)
    a, b = np.asarray(candidate_a, dtype=np.int64), np.asarray(candidate_b, dtype=np.int64)
    classes = evidence.geometry_logits.shape[1]
    if a.shape != (n,) or b.shape != (n,) or np.any(a == b):
        raise ValueError("candidate query requires two distinct IDs per sample")
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= classes) or np.any(b >= classes):
        raise ValueError("candidate query is outside support classes")
    rows = np.arange(n)[:, None]
    views = np.arange(evidence.view_logits.shape[1])[None, :]
    a_view = a[:, None]
    b_view = b[:, None]
    logit_delta = (evidence.view_logits[rows, views, a_view]
                   - evidence.view_logits[rows, views, b_view])
    # Smaller prototype distance means more compatible.
    distance_delta = (evidence.view_distances[rows, views, b_view]
                      - evidence.view_distances[rows, views, a_view])
    weighted = evidence.view_weights
    return np.column_stack([
        logit_delta, distance_delta,
        (weighted * logit_delta).sum(1),
        (weighted * distance_delta).sum(1),
    ]).astype(np.float32)


class IndependentEvidenceInvestigator:
    """B answers a named pair with its private Stage-5 Geometry model."""

    def __init__(self):
        self.verifier = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, fit_intercept=False,
                               class_weight="balanced", random_state=42),
        )
        self.fitted = False

    def fit(self, train: GeometryPrivateEvidence, known_labels: np.ndarray,
            identity_top1: np.ndarray, identity_top2: np.ndarray
            ) -> "IndependentEvidenceInvestigator":
        labels = np.asarray(known_labels, dtype=np.int64)
        if labels.shape != (len(train),) or np.any(labels < 0):
            raise ValueError("B verifier cannot train on proxy or formal Unknown")
        top1 = np.asarray(identity_top1, dtype=np.int64)
        top2 = np.asarray(identity_top2, dtype=np.int64)
        if top1.shape != labels.shape or top2.shape != labels.shape:
            raise ValueError("training pair proposal has wrong shape")
        true = labels
        rival = np.where(top1 == true, top2, top1)
        positive = pair_features(train, true, rival)
        negative = pair_features(train, rival, true)
        self.verifier.fit(np.concatenate([positive, negative]),
                          np.concatenate([np.ones(len(train)), np.zeros(len(train))]))
        self.fitted = True
        return self

    def signed_support(self, evidence: GeometryPrivateEvidence,
                       candidate_a: np.ndarray, candidate_b: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("B pair verifier must be fit on support-Known")
        return np.asarray(self.verifier.decision_function(
            pair_features(evidence, candidate_a, candidate_b)), dtype=np.float32)

    def answer(self, evidence: GeometryPrivateEvidence, candidate_a: np.ndarray,
               candidate_b: np.ndarray, active: np.ndarray) -> EvidenceCertificate:
        """Release only a semantic scalar certificate for the requested pair."""
        signed = self.signed_support(evidence, candidate_a, candidate_b)
        active = np.asarray(active, dtype=bool)
        if active.shape != (len(evidence),):
            raise ValueError("active query mask has wrong shape")
        device = torch.device("cpu")
        signed_t = torch.from_numpy(signed).to(device)
        a = torch.from_numpy(np.asarray(candidate_a, dtype=np.int64)).to(device)
        b = torch.from_numpy(np.asarray(candidate_b, dtype=np.int64)).to(device)
        active_t = torch.from_numpy(active).to(device)
        stance = torch.where(
            signed_t >= 0,
            torch.full_like(a, int(EvidenceStance.SUPPORT_A)),
            torch.full_like(a, int(EvidenceStance.SUPPORT_B)))
        reliability = torch.sigmoid(signed_t.abs())
        return EvidenceCertificate(
            sender="independent_evidence", receiver="identity",
            candidate_a=a, candidate_b=b, stance=stance,
            signed_candidate_evidence=signed_t,
            alternative_class=b,
            domain_quality=torch.from_numpy(
                evidence.view_weights.max(1).astype(np.float32)),
            open_risk=torch.zeros_like(signed_t), reliability=reliability,
            abstain=~active_t, active_mask=active_t,
            bit_cost=active_t.float() * 80.0)


def apply_certificate(identity_logits: np.ndarray,
                      certificate: EvidenceCertificate,
                      scale: float) -> np.ndarray:
    table = CandidateEvidenceTable(torch.from_numpy(
        np.asarray(identity_logits, dtype=np.float32)))
    table.apply(certificate, scale=scale)
    return table.decision().candidate_scores.detach().cpu().numpy()


def per_sample_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return torch.nn.functional.cross_entropy(values, y, reduction="none").numpy()


@dataclass
class IdentityQueryPolicy:
    """A-owned counterfactual-utility regressor using A diagnostics only."""

    budget: float
    cost: float
    seed: int

    def __post_init__(self):
        if not 0 < self.budget <= 1 or self.cost < 0:
            raise ValueError("query budget/cost out of range")
        self.regressor = ExtraTreesRegressor(
            n_estimators=96, min_samples_leaf=20, max_features=1.0,
            random_state=self.seed, n_jobs=1)
        self.threshold = None

    def fit(self, identity_public: np.ndarray,
            counterfactual_gain: np.ndarray) -> "IdentityQueryPolicy":
        self.regressor.fit(identity_public, counterfactual_gain)
        predicted = self.regressor.predict(identity_public)
        self.threshold = max(float(self.cost),
                             float(np.quantile(predicted, 1.0 - self.budget)))
        return self

    def ask(self, identity_public: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("query policy must be trained on calibration Known")
        predicted = self.regressor.predict(identity_public)
        return predicted > self.threshold, predicted


@dataclass(frozen=True)
class PublicSupportPacket:
    """B publishes three candidate-specific scalars, not private view tables."""

    candidate: np.ndarray
    prototype_distance: np.ndarray
    candidate_confidence: np.ndarray
    view_agreement: np.ndarray


def publish_candidate_support(evidence: GeometryPrivateEvidence,
                              candidate: np.ndarray) -> PublicSupportPacket:
    """Executed by B; callers receive only the public packet."""
    candidate = np.asarray(candidate, dtype=np.int64)
    if candidate.shape != (len(evidence),):
        raise ValueError("candidate must have shape [samples]")
    classes = evidence.geometry_logits.shape[1]
    if np.any(candidate < 0) or np.any(candidate >= classes):
        raise ValueError("candidate is outside support classes")
    rows = np.arange(len(evidence))[:, None]
    views = np.arange(evidence.view_logits.shape[1])[None, :]
    distances = evidence.view_distances[rows, views, candidate[:, None]]
    weighted_distance = (evidence.view_weights * distances).sum(1)
    logits = evidence.geometry_logits.astype(np.float64)
    shifted = logits - logits.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(1, keepdims=True)
    confidence = probabilities[np.arange(len(evidence)), candidate]
    agreement = (evidence.view_logits.argmax(2) == candidate[:, None]).mean(1)
    return PublicSupportPacket(
        candidate=candidate.copy(),
        prototype_distance=weighted_distance.astype(np.float32),
        candidate_confidence=confidence.astype(np.float32),
        view_agreement=agreement.astype(np.float32))
