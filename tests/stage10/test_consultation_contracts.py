import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage10.consultation import (IdentityQueryPolicy,  # noqa: E402
    IndependentEvidenceInvestigator, apply_certificate, pair_features, top_pair)
from maros_stage10.mother import FrozenEvidence  # noqa: E402


def fake_evidence(n=8, classes=3):
    rng = np.random.default_rng(37)
    weights = rng.random((n, 5)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    return FrozenEvidence(
        labels=np.arange(n) % classes,
        identity_logits=rng.normal(size=(n, classes)).astype(np.float32),
        identity_public=rng.random((n, 5)).astype(np.float32),
        geometry_logits=rng.normal(size=(n, classes)).astype(np.float32),
        view_logits=rng.normal(size=(n, 5, classes)).astype(np.float32),
        view_distances=rng.random((n, 5, classes)).astype(np.float32),
        view_weights=weights)


def test_pair_features_change_sign_when_candidate_order_swaps():
    rows = fake_evidence()
    a = np.zeros(len(rows), dtype=np.int64)
    b = np.ones(len(rows), dtype=np.int64)
    private = rows.geometry_private()
    assert np.allclose(pair_features(private, a, b),
                       -pair_features(private, b, a), atol=1e-6)


def test_verifier_never_trains_on_unknown():
    rows = fake_evidence()
    rows.labels[0] = -1
    top1, top2 = top_pair(rows.identity_logits)
    with pytest.raises(ValueError, match="Unknown"):
        IndependentEvidenceInvestigator().fit(
            rows.geometry_private(), rows.labels, top1, top2)


def test_query_policy_uses_identity_features_only():
    source = inspect.getsource(IdentityQueryPolicy)
    assert "geometry" not in source and "view_logits" not in source
    assert "oracle" not in source.lower() and "wisig" not in source.lower()


def test_b_private_contract_has_no_identity_logits_or_labels():
    private = fake_evidence().geometry_private()
    assert not hasattr(private, "identity_logits")
    assert not hasattr(private, "identity_public")
    assert not hasattr(private, "labels")


def test_candidate_pair_evidence_is_class_reindex_invariant():
    rows = fake_evidence(9, 4)
    mapping = np.array([2, 0, 3, 1], dtype=np.int64)
    old_a = np.zeros(len(rows), dtype=np.int64)
    old_b = np.ones(len(rows), dtype=np.int64)
    expected = pair_features(rows.geometry_private(), old_a, old_b)
    new_logits = np.empty_like(rows.geometry_logits)
    new_views = np.empty_like(rows.view_logits)
    new_distances = np.empty_like(rows.view_distances)
    new_logits[:, mapping] = rows.geometry_logits
    new_views[:, :, mapping] = rows.view_logits
    new_distances[:, :, mapping] = rows.view_distances
    changed = replace(rows, geometry_logits=new_logits,
                      view_logits=new_views, view_distances=new_distances)
    actual = pair_features(changed.geometry_private(), mapping[old_a], mapping[old_b])
    assert np.allclose(actual, expected, atol=1e-6)


def test_candidate_certificate_updates_only_named_candidates():
    rows = fake_evidence(12, 4)
    top1, top2 = top_pair(rows.identity_logits)
    verifier = IndependentEvidenceInvestigator().fit(
        rows.geometry_private(), rows.labels, top1, top2)
    a = np.zeros(len(rows), dtype=np.int64)
    b = np.ones(len(rows), dtype=np.int64)
    active = np.arange(len(rows)) % 2 == 0
    certificate = verifier.answer(rows.geometry_private(), a, b, active)
    updated = apply_certificate(rows.identity_logits, certificate, 0.25)
    assert np.allclose(updated[~active], rows.identity_logits[~active])
    assert np.allclose(updated[:, 2:], rows.identity_logits[:, 2:])
    assert np.any(np.abs(updated[active, :2] - rows.identity_logits[active, :2]) > 0)
