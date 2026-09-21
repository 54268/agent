import copy
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.system import Stage8System  # noqa: E402


@pytest.mark.parametrize("observation_profile,spectral_profile", [
    ("legacy_v1", None),
    ("isolated_v2", None),
    ("legacy_v1", "robust_v3"),
])
def test_class_reindex_preserves_predictions_and_prototype_verification(
        observation_profile, spectral_profile):
    torch.manual_seed(37)
    system = Stage8System(
        4, state_dim=8, hidden_channels=4,
        observation_profile=observation_profile,
        spectral_observation_profile=spectral_profile).eval()
    prototypes = torch.randn(4, 8)
    counts = torch.tensor([3, 4, 5, 6])
    system.temporal.set_candidate_prototypes(prototypes, counts)
    system.spectral.set_candidate_prototypes(prototypes.roll(1, 0), counts)
    reindexed = copy.deepcopy(system)
    old_to_new = torch.tensor([2, 0, 3, 1])
    reindexed.reindex_classes(old_to_new)

    iq = torch.randn(7, 2, 256)
    original = system.observe(iq)
    changed = reindexed.observe(iq)
    for name in ("temporal", "spectral"):
        expected_logits = torch.empty_like(original[name].class_logits)
        expected_logits[:, old_to_new] = original[name].class_logits
        assert torch.allclose(changed[name].class_logits, expected_logits, atol=1e-6)
        assert torch.equal(changed[name].top1, old_to_new[original[name].top1])

        expected_support = torch.empty_like(
            system.all_candidate_support(original, name))
        expected_support[:, old_to_new] = system.all_candidate_support(
            original, name)
        actual_support = reindexed.all_candidate_support(changed, name)
        assert torch.allclose(actual_support, expected_support, atol=1e-6)
