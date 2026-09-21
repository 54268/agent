"""Machine-readable Stage-9 G0 capability and privacy audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import inspect

from .contracts import BeliefPacket
from .policies.tool_policy import ToolPolicy
from .system import Stage9G0System
from .tool_registry import EvidenceTool
from .tools.observations import IDENTITY_TOOLS, IMPAIRMENT_TOOLS


@dataclass(frozen=True)
class G0Audit:
    passed: bool
    separate_capability_families: bool
    policy_has_no_dataset_name: bool
    public_packet_has_no_private_state: bool
    tools_are_not_agents: bool
    no_central_router: bool

    def as_dict(self) -> dict:
        return asdict(self)


def audit_g0() -> G0Audit:
    system = Stage9G0System(3, 8, 4)
    public_names = {field.name.lower() for field in fields(BeliefPacket)}
    source = inspect.getsource(ToolPolicy).lower()
    checks = {
        "separate_capability_families": (
            not set(IDENTITY_TOOLS) & set(IMPAIRMENT_TOOLS)
            and system.identity.manifest.allowed_tools == frozenset(IDENTITY_TOOLS)
            and system.impairment.manifest.allowed_tools == frozenset(IMPAIRMENT_TOOLS)),
        "policy_has_no_dataset_name": "oracle" not in source and "wisig" not in source,
        "public_packet_has_no_private_state": not any(
            term in name for name in public_names
            for term in ("embedding", "hidden", "prototype", "memory", "logits")),
        "tools_are_not_agents": not any(
            hasattr(tool, "decide") or not isinstance(tool, EvidenceTool)
            for agent in system.agents() for tool in agent.registry.tools.values()),
        "no_central_router": not hasattr(system, "router"),
    }
    return G0Audit(passed=all(checks.values()), **checks)
