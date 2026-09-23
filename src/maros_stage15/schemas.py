"""Strict schemas for LLM-agent actions and RF evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    ARBITER = "arbiter"


class Action(str, Enum):
    REQUEST_TOOL = "request_tool"
    PROPOSE = "propose"
    SUPPORT = "support"
    CHALLENGE = "challenge"
    ABSTAIN = "abstain"
    FINAL = "final"


@dataclass(frozen=True)
class ToolResult:
    name: str
    value: Any
    summary: str
    cost: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentAction:
    action: Action
    reason: str
    tool: str | None = None
    candidate_class: int | None = None
    decision: str | None = None
    confidence: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentAction":
        if not isinstance(payload, dict):
            raise ValueError("agent response must be a JSON object")
        action = Action(str(payload.get("action", "")).strip())
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise ValueError("agent action needs a non-empty reason")
        tool = payload.get("tool")
        candidate = payload.get("candidate_class")
        decision = payload.get("decision")
        confidence = payload.get("confidence")
        if candidate is not None:
            candidate = int(candidate)
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
        if decision is not None:
            decision = str(decision).strip().lower()
            if decision not in {"known", "unknown"}:
                raise ValueError("decision must be known or unknown")
        return cls(
            action=action,
            reason=reason,
            tool=None if tool is None else str(tool),
            candidate_class=candidate,
            decision=decision,
            confidence=confidence,
        )


@dataclass(frozen=True)
class TranscriptItem:
    role: Role
    action: Action
    reason: str
    candidate_class: int | None = None
    tool: str | None = None
    tool_result: ToolResult | None = None
    decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["role"] = self.role.value
        row["action"] = self.action.value
        return row


@dataclass
class SharedState:
    public_summary: dict[str, Any]
    current_candidate: int | None = None
    revealed_tools: dict[str, ToolResult] = field(default_factory=dict)
    transcript: list[TranscriptItem] = field(default_factory=list)
    remaining_tool_budget: int = 4
    turn: int = 0

    def safe_dict(self) -> dict[str, Any]:
        return {
            "public_summary": self.public_summary,
            "current_candidate": self.current_candidate,
            "revealed_tools": {
                name: result.to_dict()
                for name, result in self.revealed_tools.items()
            },
            "transcript": [item.to_dict() for item in self.transcript],
            "remaining_tool_budget": self.remaining_tool_budget,
            "turn": self.turn,
        }
