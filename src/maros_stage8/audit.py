"""Machine-readable capability and privacy audit for Stage-8 G0."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import inspect

from . import observations
from .contracts import LocalDecision
from .spectral_agent import SpectralHardwareFingerprintAgent
from .temporal_agent import TemporalFingerprintAgent


@dataclass(frozen=True)
class CapabilityAudit:
    passed: bool
    temporal_accepts_only_temporal_observation: bool
    spectral_accepts_only_spectral_observation: bool
    fft_confined_to_observation_builder: bool
    public_contract_has_no_private_state: bool
    coordinator_present: bool

    def as_dict(self) -> dict:
        return asdict(self)


def capability_audit() -> CapabilityAudit:
    temporal_source = inspect.getsource(TemporalFingerprintAgent)
    spectral_source = inspect.getsource(SpectralHardwareFingerprintAgent)
    observation_source = inspect.getsource(observations.build_spectral_observation)
    public_fields = {field.name.lower() for field in fields(LocalDecision)}
    forbidden_fragments = ("state", "embedding", "token", "prototype", "memory")
    checks = {
        "temporal_accepts_only_temporal_observation": (
            "TemporalObservation" in temporal_source
            and "SpectralObservation" not in temporal_source),
        "spectral_accepts_only_spectral_observation": (
            "SpectralObservation" in spectral_source
            and "TemporalObservation" not in spectral_source),
        "fft_confined_to_observation_builder": (
            "torch.fft" not in temporal_source
            and "torch.fft" not in spectral_source
            and "torch.fft" in observation_source),
        "public_contract_has_no_private_state": not any(
            fragment in name for name in public_fields
            for fragment in forbidden_fragments),
        "coordinator_present": False,
    }
    required = {key: value for key, value in checks.items()
                if key != "coordinator_present"}
    return CapabilityAudit(passed=all(required.values()), **checks)
