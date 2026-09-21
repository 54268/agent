import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.evaluation import (pair_verification_metrics,  # noqa: E402
                                     rescue_decomposition)
from maros_stage8.gates import (evaluate_ab_g1_gate,  # noqa: E402
                                evaluate_directional_g15_gate)


def test_pair_metrics_use_true_and_hard_negative_candidates():
    scores = np.asarray([
        [3.0, 1.0, -1.0],
        [0.0, 2.0, 1.0],
        [1.5, 0.0, 1.0],
    ])
    metrics = pair_verification_metrics(
        scores, np.asarray([0, 1, 2]), np.asarray([1, 2, 0]))
    assert metrics["pair_accuracy"] == pytest.approx(2 / 3)
    assert metrics["hard_negative_pair_accuracy"] == pytest.approx(2 / 3)
    assert 0.5 < metrics["pair_auroc"] <= 1.0
    assert 0.0 <= metrics["pair_brier"] <= 1.0


def test_rescue_decomposition_keeps_identity_unknown_and_pair_separate():
    labels = np.asarray([0, 1, 2, 3, -1, -1])
    result = rescue_decomposition(
        labels,
        temporal_prediction=np.asarray([0, 0, 2, 0, -1, 0]),
        spectral_prediction=np.asarray([1, 1, 0, 3, 0, -1]),
        temporal_reject=np.asarray([0, 0, 0, 0, 1, 0]),
        spectral_reject=np.asarray([0, 0, 0, 0, 0, 1]),
        temporal_pair_correct=np.asarray([1, 0, 1, 0, 0, 0]),
        spectral_pair_correct=np.asarray([0, 1, 0, 1, 0, 0]),
    )
    assert result["temporal_to_spectral_identity_rescue"] == pytest.approx(0.5)
    assert result["spectral_to_temporal_identity_rescue"] == pytest.approx(0.5)
    assert result["temporal_unknown_rejection_rescue"] == pytest.approx(0.5)
    assert result["spectral_unknown_rejection_rescue"] == pytest.approx(0.5)
    assert result["temporal_to_spectral_pair_rescue"] == pytest.approx(0.5)
    assert result["spectral_to_temporal_pair_rescue"] == pytest.approx(0.5)


def _passing_fold():
    return {
        "temporal_identity_accuracy": 0.75,
        "spectral_identity_accuracy": 0.70,
        "temporal_pair_metrics": {
            "pair_auroc": 0.80, "hard_negative_pair_accuracy": 0.65},
        "spectral_pair_metrics": {
            "pair_auroc": 0.78, "hard_negative_pair_accuracy": 0.62},
        "rescue_decomposition": {
            "temporal_to_spectral_identity_rescue": 0.10,
            "spectral_to_temporal_identity_rescue": 0.08,
            "temporal_to_spectral_hard_pair_rescue": 0.11,
            "spectral_to_temporal_hard_pair_rescue": 0.09,
        },
    }


def test_new_gates_require_both_rescue_directions():
    folds = [_passing_fold() for _ in range(4)]
    assert evaluate_ab_g1_gate(folds).passed
    folds[0]["rescue_decomposition"][
        "spectral_to_temporal_identity_rescue"] = 0.0
    for row in folds[1:]:
        row["rescue_decomposition"][
            "spectral_to_temporal_identity_rescue"] = 0.01
    assert not evaluate_ab_g1_gate(folds).passed

    good = {
        "temporal_rescues_spectral": [0.7, 0.8, 0.7, 0.6],
        "spectral_rescues_temporal": [0.7, 0.68, 0.72, 0.6],
    }
    assert evaluate_directional_g15_gate(good).passed
    weak = {**good, "spectral_rescues_temporal": [0.51, 0.49, 0.50, 0.48]}
    assert not evaluate_directional_g15_gate(weak).passed
