"""Shared shallow stem (Section 6 of the design doc).

A compact real-valued 1-D CNN maps IQ ``[B, 2, 256]`` to the shared feature map
``h_s [B, 128, 64]``.  The stem only learns low-level I/Q structure; every agent
adds its own private encoder afterwards so the model does not collapse into a
single backbone with multiple heads.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class SharedStem(nn.Module):
    """2 x 256 -> h_s [B, out_channels, 64]."""

    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),  # 256 -> 128
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualConv1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),  # 128 -> 64
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualConv1d(64),
            nn.Conv1d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            ResidualConv1d(out_channels),
        )
        self.out_channels = out_channels
        self.feature_length = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PrivateEncoder(nn.Module):
    """h_s [B, C, T] -> private embedding z [B, dim] (one per agent)."""

    def __init__(self, in_channels: int, dim: int = 128, hidden: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            ResidualConv1d(hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim),
        )
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        feat = self.block(h)
        pooled = torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)
        return self.proj(pooled)
