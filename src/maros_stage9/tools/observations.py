"""Dataset-agnostic, capability-bounded views of one I/Q sample."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from maros_stage8.observations import build_robust_spectral_observation_v3


IDENTITY_TOOLS = ("raw", "complex", "envelope", "difference")
IMPAIRMENT_TOOLS = (
    "robust_spectrum", "frequency_difference", "iq_imbalance", "phase_noise")


def _validate(iq: torch.Tensor) -> None:
    if (not torch.is_tensor(iq) or iq.ndim != 3 or iq.shape[1] != 2
            or iq.shape[-1] < 64 or not iq.is_floating_point()):
        raise ValueError("expected floating I/Q [batch, 2, length>=64]")
    if not bool(torch.isfinite(iq).all()):
        raise ValueError("I/Q contains non-finite values")


def _normalize(view: torch.Tensor) -> torch.Tensor:
    return (view - view.mean(-1, keepdim=True)) / view.std(
        -1, keepdim=True, unbiased=False).clamp_min(1e-5)


def identity_view(iq: torch.Tensor, name: str) -> torch.Tensor:
    _validate(iq)
    if name == "raw" or name == "complex":
        return iq
    z = torch.complex(iq[:, 0], iq[:, 1])
    if name == "envelope":
        amplitude = torch.log1p(z.abs())
        phase_step = torch.angle(z[:, 1:] * z[:, :-1].conj())
        return _normalize(torch.stack([
            amplitude, F.pad(phase_step, (1, 0))], dim=1))
    if name == "difference":
        return _normalize(iq - torch.roll(iq, 1, dims=-1))
    raise KeyError(name)


def _complex_fft(iq: torch.Tensor) -> torch.Tensor:
    z = torch.complex(iq[:, 0], iq[:, 1])
    return torch.fft.fftshift(torch.fft.fft(z, norm="ortho"), dim=-1)


def impairment_view(iq: torch.Tensor, name: str) -> torch.Tensor:
    """No returned view preserves the original time-ordered I/Q sequence."""
    _validate(iq)
    if name == "robust_spectrum":
        return build_robust_spectral_observation_v3(iq).features
    if name == "frequency_difference":
        power = torch.log1p(_complex_fft(iq).abs().square())
        power = _normalize(power[:, None])[:, 0]
        smooth = F.avg_pool1d(power[:, None], 9, stride=1, padding=4)[:, 0]
        first = power - torch.roll(power, 1, dims=-1)
        second = first - torch.roll(first, 1, dims=-1)
        return _normalize(torch.stack([power - smooth, first, second], dim=1))
    if name == "iq_imbalance":
        fi = torch.fft.fftshift(torch.fft.fft(iq[:, 0], norm="ortho"), dim=-1)
        fq = torch.fft.fftshift(torch.fft.fft(iq[:, 1], norm="ortho"), dim=-1)
        pi = torch.log1p(fi.abs().square())
        pq = torch.log1p(fq.abs().square())
        cross = fi * fq.conj() / (fi.abs() * fq.abs()).clamp_min(1e-6)
        return _normalize(torch.stack([
            pi - pq, pi + pq, cross.real, cross.imag], dim=1))
    if name == "phase_noise":
        z = torch.complex(iq[:, 0], iq[:, 1])
        step = torch.angle(z[:, 1:] * z[:, :-1].conj())
        step = step - step.mean(-1, keepdim=True)
        spectrum = torch.fft.fftshift(
            torch.fft.fft(torch.complex(step, torch.zeros_like(step)),
                          n=iq.shape[-1], norm="ortho"), dim=-1)
        power = torch.log1p(spectrum.abs().square())
        smooth = F.avg_pool1d(power[:, None], 9, stride=1, padding=4)[:, 0]
        rough = power - smooth
        return _normalize(torch.stack([power, smooth, rough], dim=1))
    raise KeyError(name)


def policy_observation(iq: torch.Tensor, role: str) -> torch.Tensor:
    """Cheap local scout view; no expert encoder or hidden state is consulted."""
    _validate(iq)
    if role == "identity":
        # A sees a low-resolution raw observation; the expensive expert tools
        # are still selected *after* this scout runs.
        return F.adaptive_avg_pool1d(iq, 64)
    if role == "impairment":
        # B's scout has no reconstructable time-domain signal.
        spectrum = _complex_fft(iq)
        log_power = torch.log1p(spectrum.abs().square())
        contrast = log_power - F.avg_pool1d(
            log_power[:, None], 9, stride=1, padding=4)[:, 0]
        return _normalize(F.adaptive_avg_pool1d(
            torch.stack([log_power, contrast], dim=1), 64))
    raise KeyError(role)

