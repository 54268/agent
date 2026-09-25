"""Schemas for autonomous LLM agents, private observations and messages."""
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
    SHARE_EVIDENCE = "share_evidence"
    ASK_PEER = "ask_peer"
    ABSTAIN = "abstain"
    FINAL = "final"


class MessageKind(str, Enum):
    PROPOSAL = "proposal"
    EVIDENCE = "evidence"
    QUERY = "query"
    SUPPORT = "support"
    CHALLENGE = "challenge"
    ABSTAIN = "abstain"
    SYSTEM = "system"


@dataclass(frozen=True)
class ToolResult:
    name: str
    value: Any
    summary: str
    cost: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentMessage:
    sender: Role | str
    recipient: Role | str
    kind: MessageKind
    content: str
    candidate_class: int | None = None
    confidence: float | None = None
    evidence_name: str | None = None
    evidence_value: Any = None
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["sender"] = self.sender.value if isinstance(self.sender, Role) else self.sender
        row["recipient"] = (
            self.recipient.value if isinstance(self.recipient, Role) else self.recipient)
        row["kind"] = self.kind.value
        return row


@dataclass(frozen=True)
class AgentAction:
    action: Action
    reason: str
    tool: str | None = None
    recipient: Role | None = None
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
        recipient_raw = payload.get("recipient")
        recipient = None if recipient_raw in (None, "", "null") else Role(
            str(recipient_raw).strip())
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
            recipient=recipient,
            candidate_class=candidate,
            decision=decision,
            confidence=confidence,
        )


@dataclass
class AgentContext:
    """What one agent is actually allowed to know at one decision step."""

    role: Role
    num_known_classes: int
    private_summary: dict[str, Any]
    current_candidate: int | None
    inbox: list[AgentMessage] = field(default_factory=list)
    private_tools: dict[str, ToolResult] = field(default_factory=dict)
    own_history: list[dict[str, Any]] = field(default_factory=list)
    remaining_tool_budget: int = 0
    turn: int = 0

    def safe_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "num_known_classes": self.num_known_classes,
            "private_observation": dict(self.private_summary),
            "current_candidate": self.current_candidate,
            "messages_received": [message.to_dict() for message in self.inbox[-12:]],
            "private_tool_results": {
                name: result.to_dict()
                for name, result in self.private_tools.items()
            },
            "own_recent_actions": list(self.own_history[-8:]),
            "remaining_tool_budget": self.remaining_tool_budget,
            "turn": self.turn,
        }
