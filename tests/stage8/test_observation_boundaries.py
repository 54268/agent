import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.observations import (CoarseSpectralObservation,  # noqa: E402
                                       ROBUST_SPECTRAL_V3_CHANNELS,
                                       RobustSpectralObservation,
                                       SpectralObservation,
                                       TemporalObservation,
                                       TemporalPatchObservation,
                                       build_coarse_spectral_observation,
                                       build_robust_spectral_observation_v3,
                                       build_spectral_observation,
                                       build_temporal_patch_observation)
from maros_stage8.system import Stage8System  # noqa: E402


def test_observation_shapes_are_strict_and_immutable():
    iq = torch.randn(3, 2, 64)
    temporal = TemporalObservation(iq)
    spectral = build_spectral_observation(iq)
    assert temporal.iq.shape == (3, 2, 64)
    assert spectral.features.shape == (3, 6, 64)
    assert spectral.quality.shape == (3, 6)
    with pytest.raises(Exception):
        temporal.iq = torch.zeros_like(iq)


@pytest.mark.parametrize("bad", [
    torch.randn(3, 64),
    torch.randn(3, 3, 64),
    torch.ones(3, 2, 8),
])
def test_temporal_contract_rejects_invalid_iq(bad):
    with pytest.raises((TypeError, ValueError)):
        TemporalObservation(bad)


def test_spectral_contract_rejects_raw_shaped_payload():
    with pytest.raises(ValueError, match="spectral features"):
        SpectralObservation(torch.randn(3, 2, 64), torch.randn(3, 6))


def test_information_isolated_v2_observations_are_lossy_and_typed():
    iq = torch.randn(3, 2, 256)
    temporal = build_temporal_patch_observation(iq)
    spectral = build_coarse_spectral_observation(iq)
    assert isinstance(temporal, TemporalPatchObservation)
    assert isinstance(spectral, CoarseSpectralObservation)
    assert temporal.features.shape == (3, 8, 32)
    assert temporal.quality.shape == (3, 3)
    assert spectral.features.shape == (3, 8, 64)
    assert spectral.quality.shape == (3, 8)
    assert not hasattr(temporal, "iq")
    assert not hasattr(spectral, "iq")
    assert set(temporal.__dataclass_fields__) == {"features", "quality"}
    assert set(spectral.__dataclass_fields__) == {"features", "quality"}


def test_isolated_v2_system_rejects_legacy_observations():
    iq = torch.randn(3, 2, 256)
    system = Stage8System(
        3, state_dim=8, hidden_channels=4,
        observation_profile="isolated_v2")
    context = system.observe(iq)
    assert context["temporal"].class_logits.shape == (3, 3)
    assert context["spectral"].class_logits.shape == (3, 3)
    with pytest.raises(TypeError, match="TemporalPatchObservation"):
        system.temporal(TemporalObservation(iq))
    with pytest.raises(TypeError, match="configured SpectralObservation"):
        system.spectral(build_spectral_observation(iq))


def _cosine_drift(left, right):
    left = left.flatten(1)
    right = right.flatten(1)
    return float((1.0 - torch.nn.functional.cosine_similarity(
        left, right, dim=1)).mean())


def _stationary_multitone(batch=8):
    n = torch.arange(256, dtype=torch.float32)
    z = (torch.exp(1j * 2.0 * torch.pi * 3.0 * n / 64.0)
         + 0.55 * torch.exp(-1j * 2.0 * torch.pi * 7.0 * n / 64.0))
    z = z[None, :].repeat(batch, 1)
    return torch.stack([z.real, z.imag], dim=1)


def test_robust_spectral_v3_is_phase_free_and_has_no_raw_iq():
    observation = build_robust_spectral_observation_v3(
        torch.randn(4, 2, 256))
    assert isinstance(observation, RobustSpectralObservation)
    assert observation.features.shape == (4, 6, 64)
    assert observation.quality.shape == (4, 6)
    assert not hasattr(observation, "iq")
    assert all("phase" not in name for name in ROBUST_SPECTRAL_V3_CHANNELS)


def test_robust_spectral_v3_centroid_alignment_handles_integer_cfo():
    iq = _stationary_multitone(5)
    n = torch.arange(256, dtype=iq.dtype)
    rotation = torch.exp(1j * 2.0 * torch.pi * 4.0 * n / 64.0)
    z = torch.complex(iq[:, 0], iq[:, 1]) * rotation
    shifted = torch.stack([z.real, z.imag], dim=1)
    original = build_robust_spectral_observation_v3(iq, align=True)
    changed = build_robust_spectral_observation_v3(shifted, align=True)
    assert _cosine_drift(original.features, changed.features) < 1e-5


def test_robust_spectral_v3_is_more_time_shift_stable_than_v1():
    iq = _stationary_multitone(16)
    shifted = torch.roll(iq, shifts=11, dims=-1)
    legacy = build_spectral_observation(iq)
    legacy_shifted = build_spectral_observation(shifted)
    robust = build_robust_spectral_observation_v3(iq)
    robust_shifted = build_robust_spectral_observation_v3(shifted)
    assert _cosine_drift(
        robust.features, robust_shifted.features) < _cosine_drift(
            legacy.features, legacy_shifted.features)
