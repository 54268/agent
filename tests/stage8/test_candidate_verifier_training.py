import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.observations import (build_spectral_observation,  # noqa: E402
                                       build_temporal_observation)
from maros_stage8.system import Stage8System  # noqa: E402
from maros_stage8.training import local_specialization_objective  # noqa: E402


def test_candidate_objective_uses_true_positive_and_nearest_hard_negative():
    torch.manual_seed(11)
    system = Stage8System(4, state_dim=8, hidden_channels=4)
    prototypes = torch.randn(4, 8)
    counts = torch.full((4,), 2, dtype=torch.long)
    system.temporal.set_candidate_prototypes(prototypes, counts)
    system.spectral.set_candidate_prototypes(prototypes.flip(0), counts)
    iq = torch.randn(6, 2, 64)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    total, pieces = local_specialization_objective(
        system, build_temporal_observation(iq),
        build_spectral_observation(iq), labels)

    assert torch.isfinite(total)
    assert set(pieces) == {"temporal", "spectral", "decisions"}
    for name in ("temporal", "spectral"):
        objective = pieces[name]
        assert torch.isfinite(objective.loss)
        assert objective.hard_negative.shape == labels.shape
        assert torch.all(objective.hard_negative != labels)
        assert objective.positive_loss >= 0
        assert objective.hard_negative_loss >= 0


def test_verifiers_have_no_class_id_embedding_table():
    system = Stage8System(4, state_dim=8, hidden_channels=4)
    assert not hasattr(system.temporal, "class_embedding")
    assert not hasattr(system.spectral, "class_embedding")
    assert all("embedding" not in name for name, _ in system.named_parameters())
