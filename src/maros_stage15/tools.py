"""Leakage-safe RF tools exposed to LLM agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from maros_stage14.data import EpisodeSnapshot
from maros_stage14.observations import ranked

from .schemas import ToolResult


def public_rf_summary(snapshot: EpisodeSnapshot) -> dict:
    """Build the always-visible RF summary without label/provenance leakage."""
    probability, top1, top2, p1, p2, entropy = ranked(snapshot.identity_logits)
    g_probability, g_top1, g_top2, gp1, gp2, g_entropy = ranked(
        snapshot.geometry_logits)
    return {
        "num_known_classes": snapshot.num_classes,
        "identity_top1": int(top1),
        "identity_top2": int(top2),
        "identity_top1_probability": float(p1),
        "identity_top2_probability": float(p2),
        "identity_margin": float(p1 - p2),
        "identity_entropy": float(entropy),
        "geometry_top1": int(g_top1),
        "geometry_top2": int(g_top2),
        "geometry_top1_probability": float(gp1),
        "geometry_top2_probability": float(gp2),
        "geometry_margin": float(gp1 - gp2),
        "geometry_entropy": float(g_entropy),
        "identity_geometry_agree": bool(top1 == g_top1),
    }


@dataclass(frozen=True)
class RFTool:
    name: str
    description: str
    fn: Callable[[EpisodeSnapshot], ToolResult]


class RFToolRegistry:
    """Frozen perception evidence presented as explicit on-demand tools."""

    def __init__(self):
        self._tools = {
            "identity_prototype": RFTool(
                "identity_prototype",
                "Known-class identity prototype risk in [0,1]; higher means less compatible.",
                lambda s: ToolResult(
                    "identity_prototype",
                    float(s.identity_prototype_risk),
                    f"identity prototype risk={s.identity_prototype_risk:.4f}")),
            "geometry_prototype": RFTool(
                "geometry_prototype",
                "Geometry prototype risk in [0,1]; higher means less compatible.",
                lambda s: ToolResult(
                    "geometry_prototype",
                    float(s.geometry_prototype_risk),
                    f"geometry prototype risk={s.geometry_prototype_risk:.4f}")),
            "openmax": RFTool(
                "openmax",
                "OpenMax unknown risk in [0,1]; higher supports unknown rejection.",
                lambda s: ToolResult(
                    "openmax", float(s.openmax_risk),
                    f"OpenMax risk={s.openmax_risk:.4f}")),
            "boundary": RFTool(
                "boundary",
                "Frozen open-set boundary risk in [0,1]; higher supports unknown rejection.",
                lambda s: ToolResult(
                    "boundary", float(s.boundary_risk),
                    f"boundary risk={s.boundary_risk:.4f}")),
            "energy_margin": RFTool(
                "energy_margin",
                "Energy and identity/geometry distance margins for uncertainty analysis.",
                lambda s: ToolResult(
                    "energy_margin",
                    {
                        "identity_energy": float(np.tanh(s.identity_energy)),
                        "identity_distance_margin": float(np.tanh(
                            s.identity_distance_margin)),
                        "geometry_distance_margin": float(np.tanh(
                            s.geometry_distance_margin)),
                    },
                    "normalized energy and distance margins")),
            "geometry_views": RFTool(
                "geometry_views",
                "Per-view geometry risks for checking consistency across RF views.",
                lambda s: ToolResult(
                    "geometry_views",
                    [float(v) for v in s.view_risks],
                    "per-view geometry risks")),
        }

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def descriptions(self) -> dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def call(self, name: str, snapshot: EpisodeSnapshot) -> ToolResult:
        if name not in self._tools:
            raise ValueError(f"unknown RF tool: {name}")
        return self._tools[name].fn(snapshot)
