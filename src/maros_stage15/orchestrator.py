"""Partially observable LLM multi-agent coordination for open-set SEI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from maros_stage14.data import EpisodeSnapshot

from .agents import LLMAgent
from .memory import Case
from .schemas import (
    Action, AgentContext, AgentMessage, MessageKind, Role,
)
from .tools import RFToolRegistry, role_private_summary


@dataclass(frozen=True)
class AgenticResult:
    prediction: int
    decision: str
    candidate_class: int | None
    confidence: float | None
    turns: int
    tool_calls: int
    message_count: int
    transcript: tuple[dict, ...]


class AgenticOpenSetSystem:
    """A heterogeneous partially-observable multi-agent decision system.

    Information contract:
    - Proposer sees only identity-side observations/tools.
    - Critic sees only geometry/open-set observations/tools.
    - Arbiter sees no raw RF scores/tools.
    - Cross-agent information moves only through explicit messages.
    """

    def __init__(self, agents: Mapping[Role, LLMAgent],
                 tools: RFToolRegistry | None = None,
                 *, max_turns: int = 12, tool_budget: int = 4,
                 message_budget: int = 8):
        required = set(Role)
        if set(agents) != required:
            raise ValueError(
                f"agents must be exactly {sorted(role.value for role in required)}")
        if max_turns < 3:
            raise ValueError("max_turns must be at least 3")
        if tool_budget < 1:
            raise ValueError("tool_budget must be positive")
        if message_budget < 2:
            raise ValueError("message_budget must be at least two")
        self.agents = dict(agents)
        self.tools = tools or RFToolRegistry()
        self.max_turns = int(max_turns)
        self.tool_budget = int(tool_budget)
        self.message_budget = int(message_budget)

    @staticmethod
    def _deliver(inboxes, message: AgentMessage) -> None:
        recipient = Role(message.recipient)
        inboxes[recipient].append(message)

    def _broadcast(self, inboxes, sender: Role, kind: MessageKind,
                   content: str, *, candidate_class=None, confidence=None,
                   evidence_name=None, evidence_value=None, turn=0) -> int:
        count = 0
        for recipient in Role:
            if recipient == sender:
                continue
            self._deliver(inboxes, AgentMessage(
                sender=sender,
                recipient=recipient,
                kind=kind,
                content=content,
                candidate_class=candidate_class,
                confidence=confidence,
                evidence_name=evidence_name,
                evidence_value=evidence_value,
                turn=turn,
            ))
            count += 1
        return count

    def run(self, snapshot: EpisodeSnapshot) -> AgenticResult:
        summaries = {
            role: role_private_summary(snapshot, role)
            for role in Role
        }
        inboxes = {role: [] for role in Role}
        private_tools = {role: {} for role in Role}
        histories = {role: [] for role in Role}
        transcript: list[dict] = []
        current_candidate: int | None = None
        remaining_tools = self.tool_budget
        remaining_messages = self.message_budget
        tool_calls = 0
        message_count = 0
        role = Role.PROPOSER

        for turn in range(self.max_turns):
            context = AgentContext(
                role=role,
                num_known_classes=snapshot.num_classes,
                private_summary=summaries[role],
                current_candidate=current_candidate,
                inbox=list(inboxes[role]),
                private_tools=dict(private_tools[role]),
                own_history=list(histories[role]),
                remaining_tool_budget=remaining_tools,
                turn=turn,
            )
            action = self.agents[role].act(
                context, self.tools.descriptions_for(role))
            action_row = {
                "turn": turn,
                "agent": role.value,
                "action": action.action.value,
                "reason": action.reason,
                "tool": action.tool,
                "recipient": (
                    action.recipient.value if action.recipient is not None else None),
                "candidate_class": action.candidate_class,
                "decision": action.decision,
                "confidence": action.confidence,
            }
            histories[role].append(action_row)
            transcript.append(dict(action_row))

            if action.action == Action.REQUEST_TOOL:
                if action.tool in private_tools[role]:
                    raise ValueError(
                        f"{role.value} requested private tool {action.tool} twice")
                result = self.tools.call(role, action.tool, snapshot)
                if result.cost > remaining_tools:
                    raise ValueError("RF tool request exceeds remaining budget")
                private_tools[role][action.tool] = result
                remaining_tools -= result.cost
                tool_calls += 1
                transcript[-1]["private_tool_result"] = result.to_dict()
                # Tool output remains private, so the same agent reasons again.
                continue

            sent = 0
            if action.action == Action.PROPOSE:
                current_candidate = action.candidate_class
                if remaining_messages <= 0:
                    raise RuntimeError("message budget exhausted before proposal")
                sent = self._broadcast(
                    inboxes, role, MessageKind.PROPOSAL, action.reason,
                    candidate_class=action.candidate_class,
                    confidence=action.confidence, turn=turn)
                role = Role.CRITIC

            elif action.action in {Action.SUPPORT, Action.CHALLENGE}:
                if remaining_messages <= 0:
                    raise RuntimeError("message budget exhausted before critique")
                kind = (
                    MessageKind.SUPPORT if action.action == Action.SUPPORT
                    else MessageKind.CHALLENGE)
                sent = self._broadcast(
                    inboxes, role, kind, action.reason,
                    candidate_class=action.candidate_class,
                    confidence=action.confidence, turn=turn)
                role = Role.ARBITER

            elif action.action == Action.SHARE_EVIDENCE:
                if remaining_messages <= 0:
                    raise RuntimeError("message budget exhausted before evidence share")
                evidence = private_tools[role][action.tool]
                self._deliver(inboxes, AgentMessage(
                    sender=role,
                    recipient=action.recipient,
                    kind=MessageKind.EVIDENCE,
                    content=action.reason,
                    candidate_class=action.candidate_class,
                    confidence=action.confidence,
                    evidence_name=evidence.name,
                    evidence_value=evidence.value,
                    turn=turn,
                ))
                sent = 1
                role = action.recipient

            elif action.action == Action.ASK_PEER:
                if remaining_messages <= 0:
                    raise RuntimeError("message budget exhausted before peer query")
                self._deliver(inboxes, AgentMessage(
                    sender=role,
                    recipient=action.recipient,
                    kind=MessageKind.QUERY,
                    content=action.reason,
                    candidate_class=current_candidate,
                    confidence=action.confidence,
                    turn=turn,
                ))
                sent = 1
                role = action.recipient

            elif action.action == Action.ABSTAIN:
                if role == Role.PROPOSER:
                    role = Role.CRITIC
                elif role == Role.CRITIC:
                    role = Role.ARBITER
                else:
                    role = Role.PROPOSER

            elif action.action == Action.FINAL:
                candidate = (
                    action.candidate_class
                    if action.candidate_class is not None
                    else current_candidate)
                if action.decision == "known" and candidate is None:
                    raise ValueError("known final has no available candidate")
                prediction = -1 if action.decision == "unknown" else int(candidate)
                result = AgenticResult(
                    prediction=prediction,
                    decision=str(action.decision),
                    candidate_class=candidate,
                    confidence=action.confidence,
                    turns=turn + 1,
                    tool_calls=tool_calls,
                    message_count=message_count,
                    transcript=tuple(transcript),
                )
                self._remember(
                    summaries, inboxes, private_tools, result)
                return result

            # One communication action consumes one logical message budget unit,
            # even if proposal/critique is delivered to both peers.
            if sent:
                remaining_messages -= 1
                message_count += 1
                transcript[-1]["message_deliveries"] = sent

        # Fail closed when the dialogue does not reach a valid final decision.
        result = AgenticResult(
            prediction=-1,
            decision="unknown",
            candidate_class=current_candidate,
            confidence=None,
            turns=self.max_turns,
            tool_calls=tool_calls,
            message_count=message_count,
            transcript=tuple(transcript),
        )
        self._remember(summaries, inboxes, private_tools, result)
        return result

    def _remember(self, summaries, inboxes, private_tools,
                  result: AgenticResult) -> None:
        for role, agent in self.agents.items():
            agent.memory.add(Case(
                local_summary=dict(summaries[role]),
                private_tools={
                    name: tool.value
                    for name, tool in private_tools[role].items()
                },
                observed_messages=[
                    message.to_dict() for message in inboxes[role][-12:]
                ],
                final_decision=result.decision,
                candidate_class=result.candidate_class,
                note="Past agentic outcome; no ground truth or provenance stored.",
            ))
