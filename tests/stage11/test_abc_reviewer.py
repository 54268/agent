import numpy as np
import pytest

from maros_stage11.reviewer import CReviewer, reports, route, signal_descriptor
from maros_stage11.experiment import ELIGIBLE_ROUTES, formal_dataset


def _public(n):
    a = np.tile(np.array([[3.0, 1.0, 0.0]]), (n, 1))
    b = np.tile(np.array([[2.0, 0.0, 1.0]]), (n, 1))
    weights = np.tile(np.array([[0.5, 0.3, 0.2]]), (n, 1))
    distances = np.ones((n, 3, 3))
    return reports(a, b, weights, distances)


def test_report_has_competition_and_view_evidence():
    public = _public(4)
    assert public["a_top1"].tolist() == [0] * 4
    assert public["b_top1"].tolist() == [0] * 4
    assert public["b_view_source"].tolist() == [0] * 4
    assert np.isfinite(public["b_view_anomaly"]).all()
    assert "a_margin" in public and "b_entropy" in public


def test_c_is_candidate_validator_and_rejects_unknown_enrollment():
    rng = np.random.default_rng(42)
    iq = rng.normal(size=(90, 2, 256)).astype(np.float32)
    x = signal_descriptor(iq)
    y = np.repeat(np.arange(3), 30)
    x[y == 1] += 0.3
    x[y == 2] -= 0.3
    public = _public(90)
    reviewer = CReviewer().fit(x, y, public)
    assert reviewer.pug_training["pseudo_count"] > 0
    scores = reviewer.scores(x, np.zeros(90, dtype=int), public)
    assert set(("lite", "full", "no_pug", "no_proto", "no_reconstruction")) <= set(scores)
    assert all(np.isfinite(values).all() for values in scores.values())
    changed = reviewer.scores(x, np.ones(90, dtype=int), public)
    assert np.any(np.abs(changed["prototype"] - scores["prototype"]) > 1e-6)
    with pytest.raises(ValueError, match="Known"):
        CReviewer().fit(x, np.where(y == 0, -1, y), public)


def test_high_confidence_agreement_gets_lite_not_bypass():
    assert "disagreement_only" not in ELIGIBLE_ROUTES
    assert "disagreement_low" not in ELIGIBLE_ROUTES
    public = _public(4)
    scores = {"lite": np.array([0.1, 0.3, 0.8, 0.9]),
              "full": np.array([0.2, 0.4, 0.7, 0.95])}
    risk, full = route(public, scores, "disagreement_low_lite", 0.5)
    assert full.tolist() == [False, False, True, True]
    assert risk.tolist() == [0.0, 0.0, 0.7, 0.95]
    lite, reviewed = route(public, scores, "lite_only", 0.5)
    assert lite.tolist() == scores["lite"].tolist()
    assert not reviewed.any()


def test_formal_requires_frozen_selection(tmp_path):
    p = tmp_path / "selection.json"
    p.write_text('{"state":"not_frozen","routes":{}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen"):
        formal_dataset({"dataset": "oracle"}, p, tmp_path)
