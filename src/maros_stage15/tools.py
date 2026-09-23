"""Role-scoped, leakage-safe RF observations and tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from maros_stage14.data import EpisodeSnapshot
from maros_stage14.observations import ranked

from .schemas import Role, ToolResult


def role_private_summary(snapshot: EpisodeSnapshot, role: Role) -> dict:
    """Return genuinely local RF observations for one agent.

    Ground truth, source provenance and formal-unknown flags are never exposed.
    Proposer and Critic receive different sensor views; Arbiter receives no raw RF
    score and must reason from communicated evidence.
    """
    if role == Role.PROPOSER:
        _, top1, top2, p1, p2, entropy = ranked(snapshot.identity_logits)
        return {
            "identity_top1": int(top1),
            "identity_top2": int(top2),
            "identity_top1_probability": float(p1),
            "identity_top2_probability": float(p2),
            "identity_margin": float(p1 - p2),
            "identity_entropy": float(entropy),
        }
    if role == Role.CRITIC:
        _, top1, top2, p1, p2, entropy = ranked(snapshot.geometry_logits)
        return {
            "geometry_top1": int(top1),
            "geometry_top2": int(top2),
            "geometry_top1_probability": float(p1),
            "geometry_top2_probability": float(p2),
            "geometry_margin": float(p1 - p2),
            "geometry_entropy": float(entropy),
        }
    if role == Role.ARBITER:
        return {}
    raise ValueError(f"unsupported role: {role}")


@dataclass(frozen=True)
class RFTool:
    name: str
    description: str
    allowed_roles: frozenset[Role]
    fn: Callable[[EpisodeSnapshot], ToolResult]


class RFToolRegistry:
    """Frozen perception modules used as private tools, not disguised agents."""

    def __init__(self):
        proposer = frozenset({Role.PROPOSER})
        critic = frozenset({Role.CRITIC})
        self._tools = {
            "identity_prototype": RFTool(
                "identity_prototype",
                "Identity prototype risk in [0,1]; higher means less compatible "
                "with the proposed known identity.",
                proposer,
                lambda s: ToolResult(
                    "identity_prototype",
                    float(s.identity_prototype_risk),
                    f"identity prototype risk={s.identity_prototype_risk:.4f}")),
            "identity_energy_margin": RFTool(
                "identity_energy_margin",
                "Identity-only normalized energy and distance margin.",
                proposer,
                lambda s: ToolResult(
                    "identity_energy_margin",
                    {
                        "identity_energy": float(np.tanh(s.identity_energy)),
                        "identity_distance_margin": float(np.tanh(
                            s.identity_distance_margin)),
                    },
                    "identity energy/distance evidence")),
            "geometry_prototype": RFTool(
                "geometry_prototype",
                "Geometry prototype risk in [0,1]; higher means less compatible.",
                critic,
                lambda s: ToolResult(
                    "geometry_prototype",
                    float(s.geometry_prototype_risk),
                    f"geometry prototype risk={s.geometry_prototype_risk:.4f}")),
            "geometry_margin": RFTool(
                "geometry_margin",
                "Geometry-only normalized distance margin.",
                critic,
                lambda s: ToolResult(
                    "geometry_margin",
                    float(np.tanh(s.geometry_distance_margin)),
                    "geometry distance-margin evidence")),
            "openmax": RFTool(
                "openmax",
                "OpenMax unknown risk in [0,1]; higher supports unknown rejection.",
                critic,
                lambda s: ToolResult(
                    "openmax", float(s.openmax_risk),
                    f"OpenMax risk={s.openmax_risk:.4f}")),
            "boundary": RFTool(
                "boundary",
                "Frozen open-set boundary risk in [0,1]; higher supports unknown rejection.",
                critic,
                lambda s: ToolResult(
                    "boundary", float(s.boundary_risk),
                    f"boundary risk={s.boundary_risk:.4f}")),
            "geometry_views": RFTool(
                "geometry_views",
                "Per-view geometry risks for checking RF-view consistency.",
                critic,
                lambda s: ToolResult(
                    "geometry_views",
                    [float(v) for v in s.view_risks],
                    "per-view geometry risks")),
        }

    def names_for(self, role: Role) -> tuple[str, ...]:
        return tuple(
            name for name, tool in self._tools.items()
            if role in tool.allowed_roles)

    def descriptions_for(self, role: Role) -> dict[str, str]:
        return {
            name: tool.description for name, tool in self._tools.items()
            if role in tool.allowed_roles
        }

    def call(self, role: Role, name: str,
             snapshot: EpisodeSnapshot) -> ToolResult:
        if name not in self._tools:
            raise ValueError(f"unknown RF tool: {name}")
        tool = self._tools[name]
        if role not in tool.allowed_roles:
            raise PermissionError(f"{role.value} cannot call RF tool {name}")
        return tool.fn(snapshot)

    def permission_map(self) -> dict[str, tuple[str, ...]]:
        return {
            role.value: self.names_for(role)
            for role in (Role.PROPOSER, Role.CRITIC, Role.ARBITER)
        }
