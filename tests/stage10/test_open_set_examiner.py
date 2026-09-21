import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage10.consultation import PublicSupportPacket  # noqa: E402
from maros_stage10.open_set import OpenSetExaminer  # noqa: E402
from maros_stage10.services import KnownQuantileThresholdService  # noqa: E402


def packet(candidates, distances):
    candidates = np.asarray(candidates, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    return PublicSupportPacket(
        candidate=candidates, prototype_distance=distances,
        candidate_confidence=np.full(len(candidates), 0.8, dtype=np.float32),
        view_agreement=np.full(len(candidates), 0.6, dtype=np.float32))


def test_open_set_examiner_owns_tail_memory_but_no_threshold():
    public = np.tile(np.array([[0.8, 0.5, 0.4, 0.2, 0.1]], dtype=np.float32), (8, 1))
    public[:, 3] += np.linspace(0, 0.1, 8)
    support = packet([0, 0, 0, 0, 1, 1, 1, 1], np.linspace(0.1, 0.2, 8))
    examiner = OpenSetExaminer(2).fit_memory(
        public, support, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    opinion = examiner.examine(public, support)
    assert opinion.risk.shape == (8,)
    assert not hasattr(examiner, "threshold")
    threshold = KnownQuantileThresholdService(0.75).fit(opinion.risk)
    assert threshold.reject(opinion.risk).shape == (8,)


def test_open_set_examiner_rejects_private_logit_matrix():
    examiner = OpenSetExaminer(2)
    with pytest.raises(ValueError, match="five public scalars"):
        examiner.public_features(np.zeros((4, 10), dtype=np.float32),
                                 packet([0, 0, 1, 1], [0.2] * 4))


def test_open_set_memory_rejects_unknown_training_labels():
    examiner = OpenSetExaminer(2)
    with pytest.raises(ValueError, match="Unknown"):
        examiner.fit_memory(np.zeros((4, 5), dtype=np.float32),
                            packet([0, 0, 1, 1], [0.2] * 4),
                            np.array([0, 0, 1, -1]))

