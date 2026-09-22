"""Public, structured communication contracts and the Stage-14 blackboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class MessageType(IntEnum):
    QUERY = 0
    PROPOSAL = 1
    SUPPORT = 2
    CHALLENGE = 3
    EVIDENCE = 4
    ABSTAIN = 5


class EvidenceType(IntEnum):
    NONE = 0
    IDENTITY = 1
    GEOMETRY = 2
    OPEN_SET = 3
    VIEW = 4


class ReasonCode(IntEnum):
    NONE = 0
    IDENTITY_STRONG = 1
    IDENTITY_UNCERTAIN = 2
    GEOMETRY_SUPPORT = 3
    PROTOTYPE_CONFLICT = 4
    OPEN_SET_RISK = 5
    REQUESTED = 6


@dataclass(frozen=True)
class EvidenceMessage:
    sender: str
    receiver: str
    message_type: MessageType
    candidate_class: int | None = None
    confidence: float = 0.0
    margin: float = 0.0
    support_risk: float = 0.0
    evidence_type: EvidenceType = EvidenceType.NONE
    reason_code: ReasonCode = ReasonCode.NONE
    step: int = 0

    def __post_init__(self) -> None:
        if self.sender not in {"identity", "geometry", "open_set"}:
            raise ValueError("unknown message sender")
        if self.receiver not in {"identity", "geometry", "open_set", "all"}:
            raise ValueError("unknown message receiver")
        if self.receiver == self.sender:
            raise ValueError("an agent cannot message itself")
        for value, name in ((self.confidence, "confidence"),
                            (self.support_risk, "support_risk")):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class Blackboard:
    """Only public protocol state; private model vectors are never stored here."""

    current_hypothesis: int | None = None
    public_messages: list[EvidenceMessage] = field(default_factory=list)
    pending_queries: dict[str, set[str]] = field(default_factory=lambda: {
        "identity": set(), "geometry": set(), "open_set": set()})
    revealed_public_evidence: dict[str, float] = field(default_factory=dict)
    step: int = 0
    remaining_message_budget: dict[str, int] = field(default_factory=dict)
    remaining_tool_budget: int = 0

    def append(self, message: EvidenceMessage, history_limit: int = 12) -> None:
        self.public_messages.append(message)
        if len(self.public_messages) > history_limit:
            del self.public_messages[:-history_limit]

    def last_from(self, sender: str) -> EvidenceMessage | None:
        return next((message for message in reversed(self.public_messages)
                     if message.sender == sender), None)
