"""Private tool registry with fail-closed manifest enforcement."""
from __future__ import annotations

from torch import nn

from .contracts import CapabilityManifest


class EvidenceTool(nn.Module):
    """Passive callable module. It has no goal, query policy, or dialogue API."""


class ToolRegistry(nn.Module):
    def __init__(self, manifest: CapabilityManifest) -> None:
        super().__init__()
        self.manifest = manifest
        self.tools = nn.ModuleDict()

    def register(self, name: str, tool: EvidenceTool) -> None:
        if name not in self.manifest.allowed_tools:
            raise PermissionError(f"{self.manifest.owner} cannot register {name}")
        if not isinstance(tool, EvidenceTool):
            raise TypeError("tools must be passive EvidenceTool modules")
        if getattr(tool, "name", None) != name or getattr(tool, "role", None) != self.manifest.owner:
            raise PermissionError("tool identity/owner conflicts with capability manifest")
        if name in self.tools:
            raise ValueError(f"duplicate tool: {name}")
        self.tools[name] = tool

    def execute(self, name: str, observation):
        if name not in self.manifest.allowed_tools or name not in self.tools:
            raise PermissionError(f"{self.manifest.owner} cannot call {name}")
        return self.tools[name](observation)
