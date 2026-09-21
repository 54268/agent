"""Capability-separated observations built outside every Stage-8 Agent."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _validate_iq(iq: torch.Tensor) -> None:
    if not torch.is_tensor(iq):
        raise TypeError("I/Q must be a torch.Tensor")
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[-1] < 16:
        raise ValueError("I/Q must have shape [batch, 2, length>=16]")
    if not iq.is_floating_point():
        raise TypeError("I/Q must be floating point")
    if not torch.isfinite(iq).all():
        raise ValueError("I/Q contains non-finite values")


@dataclass(frozen=True)
class TemporalObservation:
    """The only observation accepted by TemporalFingerprintAgent."""

    iq: torch.Tensor

    def __post_init__(self) -> None:
        _validate_iq(self.iq)


@dataclass(frozen=True)
class TemporalPatchObservation:
    """Lossy local-time tokens with no recoverable raw-I/Q field."""

    features: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if (not torch.is_tensor(self.features) or self.features.ndim != 3
                or self.features.shape[1] != 8):
            raise ValueError(
                "temporal patch features must have shape [batch, 8, patches]")
        if self.features.shape[-1] < 8 or not self.features.is_floating_point():
            raise ValueError(
                "temporal patch features require at least 8 floating patches")
        if (not torch.is_tensor(self.quality) or self.quality.ndim != 2
                or self.quality.shape != (self.features.shape[0], 3)):
            raise ValueError("temporal patch quality must have shape [batch, 3]")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.quality).all():
            raise ValueError("temporal patch observation contains non-finite values")


@dataclass(frozen=True)
class SpectralObservation:
    """Frequency-domain features with no recoverable raw-I/Q field."""

    features: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if (not torch.is_tensor(self.features) or self.features.ndim != 3
                or self.features.shape[1] != 6):
            raise ValueError("spectral features must have shape [batch, 6, bins]")
        if self.features.shape[-1] < 16 or not self.features.is_floating_point():
            raise ValueError("spectral features require at least 16 floating bins")
        if (not torch.is_tensor(self.quality) or self.quality.ndim != 2
                or self.quality.shape != (self.features.shape[0], 6)):
            raise ValueError("spectral quality must have shape [batch, 6]")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.quality).all():
            raise ValueError("spectral observation contains non-finite values")


@dataclass(frozen=True)
class CoarseSpectralObservation:
    """Coarse spectral statistics with fine phase/time order removed."""

    features: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if (not torch.is_tensor(self.features) or self.features.ndim != 3
                or self.features.shape[1] != 8):
            raise ValueError(
                "coarse spectral features must have shape [batch, 8, bins]")
        if self.features.shape[-1] < 8 or not self.features.is_floating_point():
            raise ValueError(
                "coarse spectral features require at least 8 floating bins")
        if (not torch.is_tensor(self.quality) or self.quality.ndim != 2
                or self.quality.shape != (self.features.shape[0], 8)):
            raise ValueError("coarse spectral quality must have shape [batch, 8]")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.quality).all():
            raise ValueError("coarse spectral observation contains non-finite values")


ROBUST_SPECTRAL_V3_CHANNELS = (
    "robust_mean_log_psd",
    "segment_log_psd_std",
    "spectral_envelope",
    "local_spectral_contrast",
    "first_spectral_derivative",
    "recentered_asymmetry",
)


@dataclass(frozen=True)
class RobustSpectralObservation:
    """Phase-free, segment-averaged relative spectrum for Agent B v3."""

    features: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if (not torch.is_tensor(self.features) or self.features.ndim != 3
                or self.features.shape[1] != len(ROBUST_SPECTRAL_V3_CHANNELS)):
            raise ValueError(
                "robust spectral features must have shape [batch, 6, bins]")
        if self.features.shape[-1] < 16 or not self.features.is_floating_point():
            raise ValueError(
                "robust spectral features require at least 16 floating bins")
        if (not torch.is_tensor(self.quality) or self.quality.ndim != 2
                or self.quality.shape != (self.features.shape[0], 6)):
            raise ValueError("robust spectral quality must have shape [batch, 6]")
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.quality).all():
            raise ValueError("robust spectral observation contains non-finite values")


@dataclass(frozen=True)
class RegistrationObservation:
    """Future Memory-Agent input; never aliases an A/B private embedding."""

    descriptor: torch.Tensor

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.descriptor) or self.descriptor.ndim != 2:
            raise ValueError("registration descriptor must have shape [batch, dim]")


@dataclass(frozen=True)
class AuditObservation:
    """Future consistency-audit input owned by the perturbation service."""

    iq: torch.Tensor
    perturbation_budget: torch.Tensor

    def __post_init__(self) -> None:
        _validate_iq(self.iq)
        if (not torch.is_tensor(self.perturbation_budget)
                or self.perturbation_budget.shape != (self.iq.shape[0],)):
            raise ValueError("perturbation_budget must have shape [batch]")


def build_temporal_observation(iq: torch.Tensor) -> TemporalObservation:
    """Register raw I/Q for the temporal capability without transforming it."""

    return TemporalObservation(iq=iq)


def _patch_mean(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    usable = (values.shape[-1] // patch_size) * patch_size
    if usable < 8 * patch_size:
        raise ValueError("observation requires at least eight temporal patches")
    return values[..., :usable].unfold(-1, patch_size, patch_size).mean(-1)


def _patch_std(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    usable = (values.shape[-1] // patch_size) * patch_size
    return values[..., :usable].unfold(-1, patch_size, patch_size).std(
        -1, unbiased=False)


def build_temporal_patch_observation(
    iq: torch.Tensor, *, patch_size: int = 8,
) -> TemporalPatchObservation:
    """Compress I/Q into local, order-aware tokens before Agent access.

    The transform keeps amplitude dynamics, phase increments and local complex
    correlation.  Non-overlapping pooling removes the complete long sequence,
    absolute phase and absolute amplitude needed for exact global-spectrum
    recovery.
    """

    _validate_iq(iq)
    z = torch.complex(iq[:, 0], iq[:, 1])
    amplitude = z.abs().clamp_min(1e-7)
    normalized_amplitude = (
        amplitude - amplitude.mean(1, keepdim=True)
    ) / amplitude.std(1, keepdim=True).clamp_min(1e-5)
    amplitude_delta = F.pad(
        normalized_amplitude[:, 1:] - normalized_amplitude[:, :-1], (1, 0))
    adjacent = z[:, 1:] * z[:, :-1].conj()
    phase_step = F.pad(torch.angle(adjacent), (1, 0))
    phase_delta = F.pad(
        torch.atan2(
            torch.sin(phase_step[:, 1:] - phase_step[:, :-1]),
            torch.cos(phase_step[:, 1:] - phase_step[:, :-1])),
        (1, 0))
    normalized_adjacent = adjacent / (
        z[:, 1:].abs() * z[:, :-1].abs()).clamp_min(1e-7)
    local_correlation = F.pad(normalized_adjacent.real, (1, 0))
    transient = torch.linalg.vector_norm(
        iq[:, :, 1:] - iq[:, :, :-1], dim=1)
    transient = F.pad(transient, (1, 0))
    features = torch.stack([
        _patch_mean(normalized_amplitude, patch_size),
        _patch_std(normalized_amplitude, patch_size),
        _patch_mean(amplitude_delta.abs(), patch_size),
        _patch_mean(torch.cos(phase_step), patch_size),
        _patch_mean(torch.sin(phase_step), patch_size),
        _patch_std(phase_step, patch_size),
        _patch_mean(local_correlation, patch_size),
        _patch_mean(transient, patch_size),
    ], dim=1).to(dtype=iq.dtype)
    quality = torch.stack([
        features[:, 0].abs().mean(1),
        features[:, 2].mean(1),
        features[:, 5].mean(1),
    ], dim=1)
    return TemporalPatchObservation(features=features, quality=quality)


def build_spectral_observation(iq: torch.Tensor) -> SpectralObservation:
    """Convert raw I/Q into the Spectral Agent's complete private view.

    FFT is deliberately confined to this external builder.  The returned type
    exposes only frequency-domain channels; SpectralFingerprintAgent has no
    method accepting raw I/Q or TemporalObservation.
    """

    _validate_iq(iq)
    z = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fftshift(
        torch.fft.fft(z, dim=-1, norm="ortho"), dim=-1)
    magnitude = spectrum.abs().clamp_min(1e-7)
    power = magnitude.square()
    log_power = torch.log1p(power)
    log_power = (log_power - log_power.mean(1, keepdim=True)) / (
        log_power.std(1, keepdim=True).clamp_min(1e-5))
    delta = F.pad(log_power[:, 1:] - log_power[:, :-1], (1, 0))
    asymmetry = log_power - torch.flip(log_power, dims=(-1,))
    centered_power = power / power.mean(1, keepdim=True).clamp_min(1e-7)
    envelope = F.avg_pool1d(
        log_power[:, None], kernel_size=9, stride=1, padding=4).squeeze(1)
    phase = torch.angle(spectrum)
    phase_step = torch.atan2(
        torch.sin(phase[:, 1:] - phase[:, :-1]),
        torch.cos(phase[:, 1:] - phase[:, :-1]))
    phase_magnitude = F.pad(phase_step.abs(), (1, 0))
    phase_roughness = F.pad(
        (phase_step[:, 1:] - phase_step[:, :-1]).abs(), (2, 0))
    features = torch.stack([
        log_power,
        envelope,
        delta,
        asymmetry,
        phase_magnitude,
        phase_roughness,
    ], dim=1).to(dtype=iq.dtype)

    probability = power / power.sum(1, keepdim=True).clamp_min(1e-7)
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(1)
    entropy = entropy / torch.log(torch.tensor(
        power.shape[1], device=power.device, dtype=power.dtype))
    quality = torch.stack([
        log_power.abs().mean(1),
        log_power.std(1),
        centered_power.amax(1).log1p(),
        asymmetry.abs().mean(1),
        phase_step.abs().mean(1),
        entropy,
    ], dim=1).to(dtype=iq.dtype)
    return SpectralObservation(features=features, quality=quality)


def build_coarse_spectral_observation(
    iq: torch.Tensor, *, pool_size: int = 4,
) -> CoarseSpectralObservation:
    """Build a lossy frequency view without a complete phase sequence.

    Only coarse log-power, envelope/asymmetry and pooled paired-frequency
    coherence survive.  Pooling is performed outside the Agent and prevents
    an inverse FFT from reconstructing the original temporal ordering.
    """

    _validate_iq(iq)
    z = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fftshift(
        torch.fft.fft(z, dim=-1, norm="ortho"), dim=-1)
    power = spectrum.abs().square().clamp_min(1e-8)
    log_power = torch.log1p(power)
    log_power = (log_power - log_power.mean(1, keepdim=True)) / (
        log_power.std(1, keepdim=True).clamp_min(1e-5))
    coarse_power = _patch_mean(log_power, pool_size)
    envelope = F.avg_pool1d(
        coarse_power[:, None], kernel_size=5, stride=1, padding=2).squeeze(1)
    contrast = coarse_power - envelope
    asymmetry = coarse_power - torch.flip(coarse_power, dims=(-1,))

    paired = spectrum * torch.flip(spectrum, dims=(-1,)).conj()
    paired = paired / paired.abs().clamp_min(1e-7)
    paired_real = _patch_mean(paired.real, pool_size)
    paired_imag = _patch_mean(paired.imag, pool_size)
    paired_strength = torch.sqrt(
        paired_real.square() + paired_imag.square() + 1e-8)
    slope = F.pad(coarse_power[:, 1:] - coarse_power[:, :-1], (1, 0))
    curvature = F.pad(slope[:, 1:] - slope[:, :-1], (1, 0))
    features = torch.stack([
        coarse_power, envelope, contrast, asymmetry,
        paired_strength, paired_real, paired_imag, curvature,
    ], dim=1).to(dtype=iq.dtype)

    probability = power / power.sum(1, keepdim=True).clamp_min(1e-7)
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(1)
    entropy = entropy / torch.log(torch.tensor(
        power.shape[1], device=power.device, dtype=power.dtype))
    quality = torch.stack([
        coarse_power.abs().mean(1), coarse_power.std(1),
        envelope.std(1), contrast.abs().mean(1),
        asymmetry.abs().mean(1), paired_strength.mean(1),
        curvature.abs().mean(1), entropy,
    ], dim=1).to(dtype=iq.dtype)
    return CoarseSpectralObservation(features=features, quality=quality)


def _circular_recenter(
    values: torch.Tensor, shifts: torch.Tensor,
) -> torch.Tensor:
    bins = values.shape[-1]
    source = (
        torch.arange(bins, device=values.device)[None, :]
        - shifts[:, None]
    ).remainder(bins)
    return values.gather(1, source)


def build_robust_spectral_observation_v3(
    iq: torch.Tensor, *, align: bool = True,
    segments: int = 4,
) -> RobustSpectralObservation:
    """Build phase-free segment spectra with optional centroid recentering.

    The Agent receives only relative magnitude statistics.  Absolute FFT
    phase, phase increments and paired complex coherence never enter the
    returned observation.
    """

    _validate_iq(iq)
    if segments != 4 or iq.shape[-1] % segments:
        raise ValueError("robust spectral v3 requires four equal segments")
    segment_length = iq.shape[-1] // segments
    if segment_length < 16:
        raise ValueError("robust spectral v3 segments are too short")
    z = torch.complex(iq[:, 0], iq[:, 1]).reshape(
        len(iq), segments, segment_length)
    window = torch.hann_window(
        segment_length, periodic=True, device=iq.device, dtype=iq.dtype)
    spectrum = torch.fft.fftshift(
        torch.fft.fft(z * window, dim=-1, norm="ortho"), dim=-1)
    power = spectrum.abs().square().clamp_min(1e-8)
    mean_power = power.mean(1)
    log_segments = torch.log1p(power)
    mean_log = log_segments.mean(1)
    std_log = log_segments.std(1, unbiased=False)

    indices = torch.arange(
        segment_length, device=iq.device, dtype=iq.dtype)
    centroid = (
        mean_power * indices[None, :]).sum(1)
    centroid = centroid / mean_power.sum(1).clamp_min(1e-8)
    centroid_bin = centroid.round().long()
    shifts = segment_length // 2 - centroid_bin
    if align:
        mean_power = _circular_recenter(mean_power, shifts)
        mean_log = _circular_recenter(mean_log, shifts)
        std_log = _circular_recenter(std_log, shifts)

    median = mean_log.median(1, keepdim=True).values
    q1 = torch.quantile(mean_log, 0.25, dim=1, keepdim=True)
    q3 = torch.quantile(mean_log, 0.75, dim=1, keepdim=True)
    robust_scale = (q3 - q1).clamp_min(1e-5)
    normalized = (mean_log - median) / robust_scale
    segment_std = std_log / robust_scale
    envelope = F.avg_pool1d(
        normalized[:, None], kernel_size=7, stride=1, padding=3).squeeze(1)
    contrast = normalized - envelope
    derivative = F.pad(normalized[:, 1:] - normalized[:, :-1], (1, 0))
    asymmetry = normalized - torch.flip(normalized, dims=(-1,))
    features = torch.stack([
        normalized, segment_std, envelope, contrast, derivative, asymmetry,
    ], dim=1).to(dtype=iq.dtype)

    probability = mean_power / mean_power.sum(1, keepdim=True).clamp_min(1e-8)
    entropy = -(probability * probability.clamp_min(1e-9).log()).sum(1)
    entropy = entropy / torch.log(torch.tensor(
        segment_length, device=iq.device, dtype=iq.dtype))
    flatness = torch.exp(mean_power.clamp_min(1e-8).log().mean(1)) / (
        mean_power.mean(1).clamp_min(1e-8))
    consistency = 1.0 / (1.0 + segment_std.mean(1))
    centroid_offset = (
        centroid - float(segment_length // 2)) / float(segment_length)
    peak_to_average = mean_power.amax(1) / mean_power.mean(1).clamp_min(1e-8)
    asymmetry_energy = asymmetry.abs().mean(1)
    quality = torch.stack([
        entropy, flatness, consistency, centroid_offset,
        peak_to_average.log1p(), asymmetry_energy,
    ], dim=1).to(dtype=iq.dtype)
    return RobustSpectralObservation(features=features, quality=quality)
