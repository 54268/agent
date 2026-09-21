import inspect
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.audit import capability_audit  # noqa: E402
from maros_stage8.coordinator import ValueOfInformationCoordinator  # noqa: E402
from maros_stage8.memory_agent import EnrollmentMemoryVerificationAgent  # noqa: E402
from maros_stage8.observations import (SpectralObservation,  # noqa: E402
                                       build_spectral_observation,
                                       build_temporal_observation)
from maros_stage8.spectral_agent import SpectralHardwareFingerprintAgent  # noqa: E402
from maros_stage8.temporal_agent import TemporalFingerprintAgent  # noqa: E402


def _iq(batch=4):
    return torch.randn(batch, 2, 64)


def test_fft_is_confined_to_external_observation_builder():
    audit = capability_audit()
    assert audit.passed
    assert audit.fft_confined_to_observation_builder
    assert "torch.fft" not in inspect.getsource(TemporalFingerprintAgent)
    assert "torch.fft" not in inspect.getsource(SpectralHardwareFingerprintAgent)


def test_agents_reject_each_others_observation_types():
    temporal = TemporalFingerprintAgent(3, state_dim=8, hidden_channels=4)
    spectral = SpectralHardwareFingerprintAgent(3, state_dim=8, hidden_channels=4)
    temporal_observation = build_temporal_observation(_iq())
    spectral_observation = build_spectral_observation(_iq())

    with pytest.raises(TypeError, match="TemporalObservation"):
        temporal(spectral_observation)
    with pytest.raises(TypeError, match="SpectralObservation"):
        spectral(temporal_observation)
    with pytest.raises((TypeError, AttributeError)):
        spectral(_iq())


def test_future_agents_and_coordinator_are_gate_locked():
    with pytest.raises(RuntimeError, match="G1.5"):
        EnrollmentMemoryVerificationAgent()
    with pytest.raises(RuntimeError, match="G2"):
        ValueOfInformationCoordinator()


def test_spectral_observation_has_no_raw_iq_field():
    observation = build_spectral_observation(_iq())
    assert isinstance(observation, SpectralObservation)
    assert not hasattr(observation, "iq")
    assert set(observation.__dataclass_fields__) == {"features", "quality"}

