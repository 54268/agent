"""Feature-space PUG certification and incremental C tools."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from maros_stage5.pseudo_unknown import (BoundaryExplorerAgent, PUGConfig,
                                         PseudoUnknownBatch)
from maros_staged.evidence import ClassConditionalCdf


def generate_candidates(records, kind: str, eta: float, seed: int):
    config = PUGConfig(kind=kind, eta=float(eta), seed_ratio=0.15,
                       variations=4, seed=int(seed))
    output = BoundaryExplorerAgent(config).act(
        records.identity_state, records.geometry_state, records.labels,
        records.identity_logits.argmax(1), records.geometry_logits.argmax(1),
        records.evidence)
    return output.proposals, output.evidence


class CSupport:
    """Known-only calibration for Stage-5 private open-set evidence."""

    def fit(self, calibration):
        a = calibration.identity_logits.argmax(1)
        b = calibration.geometry_logits.argmax(1)
        self.openmax = ClassConditionalCdf(calibration.evidence[:, 6], b)
        self.geometry = ClassConditionalCdf(calibration.evidence[:, 3], b)
        self.identity = ClassConditionalCdf(calibration.evidence[:, 7], a)
        return self

    def transform(self, records):
        a = records.identity_logits.argmax(1)
        b = records.geometry_logits.argmax(1)
        return {
            "openmax": self.openmax.score(records.evidence[:, 6], b),
            "geometry": self.geometry.score(records.evidence[:, 3], b),
            "identity": self.identity.score(records.evidence[:, 7], a),
        }


def _probability_summary(logits):
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    top = np.sort(p, axis=1)[:, -2:]
    return p.argmax(1), top[:, 1], top[:, 1] - top[:, 0]


def certify(train, pseudo_batch: PseudoUnknownBatch, pseudo_records,
            support: CSupport) -> tuple[np.ndarray, dict]:
    """C certifies fresh A/B state reports plus geometric boundary quality."""
    if len(pseudo_batch.identity_state) != len(pseudo_records.labels):
        raise ValueError("each feature candidate requires a fresh state-level A/B report")
    if np.any(pseudo_records.labels != -1):
        raise ValueError("feature candidates must carry Unknown labels")
    zi = train.identity_state / np.maximum(
        np.linalg.norm(train.identity_state, axis=1, keepdims=True), 1e-8)
    zg = train.geometry_state / np.maximum(
        np.linalg.norm(train.geometry_state, axis=1, keepdims=True), 1e-8)
    joint = np.concatenate([zi, zg], axis=1)
    joint /= np.maximum(np.linalg.norm(joint, axis=1, keepdims=True), 1e-8)
    pi = pseudo_batch.identity_state / np.maximum(
        np.linalg.norm(pseudo_batch.identity_state, axis=1, keepdims=True), 1e-8)
    pg = pseudo_batch.geometry_state / np.maximum(
        np.linalg.norm(pseudo_batch.geometry_state, axis=1, keepdims=True), 1e-8)
    pjoint = np.concatenate([pi, pg], axis=1)
    pjoint /= np.maximum(np.linalg.norm(pjoint, axis=1, keepdims=True), 1e-8)
    labels = np.asarray(train.labels, dtype=int)
    classes = np.unique(labels)
    prototypes = np.stack([joint[labels == cls].mean(0) for cls in classes])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    train_distance = np.linalg.norm(joint - prototypes[labels], axis=1)
    tail = {int(cls): float(np.quantile(train_distance[labels == cls], 0.95))
            for cls in classes}
    core = {int(cls): float(np.quantile(train_distance[labels == cls], 0.50))
            for cls in classes}
    source = pseudo_batch.source_indices
    source_class = pseudo_batch.source_classes
    source_distance = train_distance[source]
    distances = np.linalg.norm(pjoint[:, None] - prototypes[None], axis=2)
    own = distances[np.arange(len(distances)), source_class]
    nearest_class = distances.argmin(1)
    nearest_proto = distances.min(1)
    outside_tail = own >= np.asarray([tail[int(c)] for c in source_class])
    outward = own > source_distance
    fell_core = ((nearest_class != source_class) &
                 (nearest_proto <= np.asarray([core[int(c)] for c in nearest_class])))
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=1).fit(joint)
    train_near = nn.kneighbors(joint, return_distance=True)[0][:, 1]
    pseudo_near = nn.kneighbors(pjoint, n_neighbors=1,
                                return_distance=True)[0][:, 0]
    far_cap = float(np.quantile(train_near, 0.99) * 2.0)
    too_far = pseudo_near > far_cap
    c = support.transform(pseudo_records)
    c_out = (c["openmax"] >= 0.90) & (
        (c["geometry"] >= 0.90) | (c["identity"] >= 0.90))
    ap, ac, am = _probability_summary(pseudo_records.identity_logits)
    bp, bc, bm = _probability_summary(pseudo_records.geometry_logits)
    conflict = (ap != bp) | (am < 0.15) | (bm < 0.15)
    hard = (ap == bp) & (ac >= 0.60) & (bc >= 0.60) & (
        am >= 0.15) & (bm >= 0.15)
    accepted = outside_tail & outward & ~fell_core & ~too_far & c_out & (conflict | hard)
    source_ap, source_ac, _ = _probability_summary(train.identity_logits[source])
    source_bp, source_bc, _ = _probability_summary(train.geometry_logits[source])
    per_class = {str(int(cls)): {"generated": int(np.sum(source_class == cls)),
                                 "certified": int(np.sum(accepted & (source_class == cls)))}
                 for cls in classes}
    diagnostic = {
        "generated": int(len(accepted)), "certified": int(accepted.sum()),
        "certification_rate": float(accepted.mean()),
        "boundary": int(np.sum(accepted & conflict)),
        "hard": int(np.sum(accepted & hard)),
        "per_class": per_class,
        "own_prototype_distance_mean": float(own.mean()),
        "nearest_rival_distance_mean": float(np.partition(distances, 1, axis=1)[:, 1].mean()),
        "competition_gap_mean": float((np.partition(distances, 1, axis=1)[:, 1] - own).mean()),
        "local_scale_mean": float(train_near[source].mean()),
        "boundary_crossing_ratio": float((outside_tail & outward).mean()),
        "nearest_known_distance_mean": float(pseudo_near.mean()),
        "fell_into_rival_core_ratio": float(fell_core.mean()),
        "too_far_ratio": float(too_far.mean()),
        "c_support_outside_ratio": float(c_out.mean()),
        "a_confidence_before": float(source_ac.mean()),
        "a_confidence_after": float(ac.mean()),
        "b_confidence_before": float(source_bc.mean()),
        "b_confidence_after": float(bc.mean()),
        "disagreement_before": float((source_ap != source_bp).mean()),
        "disagreement_after": float((ap != bp).mean()),
    }
    return accepted, diagnostic


def tool_features(records, support: CSupport):
    c = support.transform(records)
    a, ac, am = _probability_summary(records.identity_logits)
    b, bc, bm = _probability_summary(records.geometry_logits)
    return np.column_stack([c["openmax"], c["geometry"], c["identity"],
                            1-ac, 1-bc, 1-am, 1-bm, (a != b).astype(float),
                            records.evidence[:, 2], records.evidence[:, 4]])


class FeaturePUGTool:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.enabled = False

    def fit(self, known, pseudo, support: CSupport, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 16:
            return self
        rng = np.random.default_rng(self.seed)
        count = min(len(known.labels), int(mask.sum()) * 2)
        index = rng.choice(len(known.labels), count, replace=False)
        x_known = tool_features(known, support)[index]
        x_pseudo = tool_features(pseudo, support)[mask]
        x = np.vstack([x_known, x_pseudo])
        y = np.r_[np.zeros(len(x_known)), np.ones(len(x_pseudo))]
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(max_iter=500, class_weight="balanced",
                                        random_state=self.seed).fit(
            self.scaler.transform(x), y)
        self.enabled = True
        return self

    def raw_score(self, records, support):
        if not self.enabled:
            return np.zeros(len(records.labels), dtype=float)
        return self.model.predict_proba(
            self.scaler.transform(tool_features(records, support)))[:, 1]


def incremental(base, evidence, weight):
    base = np.asarray(base, dtype=float)
    evidence = np.asarray(evidence, dtype=float)
    return np.clip(base + float(weight) * np.maximum(evidence - base, 0), 0, 1)
