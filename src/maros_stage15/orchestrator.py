"""Round-robin LLM multi-agent orchestration for open-set SEI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from maros_stage14.data import EpisodeSnapshot

from .agents import LLMAgent
from .memory import Case, CaseMemory
from .schemas import Action, Role, SharedState, TranscriptItem
from .tools import RFToolRegistry, public_rf_summary


@dataclass(frozen=True)
class AgenticResult:
    prediction: int
    decision: str
    candidate_class: int | None
    turns: int
    tool_calls: int
    transcript: tuple[dict, ...]


class AgenticOpenSetSystem:
    """Three role agents sharing evidence but not hidden labels/provenance."""

    def __init__(self, agents: Mapping[Role, LLMAgent],
                 tools: RFToolRegistry | None = None,
                 *, max_turns: int = 9, tool_budget: int = 4,
                 memory: CaseMemory | None = None):
        required = {Role.PROPOSER, Role.CRITIC, Role.ARBITER}
        if set(agents) != required:
            raise ValueError(f"agents must be exactly {sorted(r.value for r in required)}")
        if max_turns < 3:
            raise ValueError("max_turns must be at least 3")
        if tool_budget < 1:
            raise ValueError("tool_budget must be positive")
        self.agents = dict(agents)
        self.tools = tools or RFToolRegistry()
        self.max_turns = int(max_turns)
        self.tool_budget = int(tool_budget)
        self.memory = memory

    def run(self, snapshot: EpisodeSnapshot) -> AgenticResult:
        state = SharedState(
            public_summary=public_rf_summary(snapshot),
            remaining_tool_budget=self.tool_budget,
        )
        roles = (Role.PROPOSER, Role.CRITIC, Role.ARBITER)
        tool_calls = 0

        for turn in range(self.max_turns):
            role = roles[turn % len(roles)]
            state.turn = turn
            action = self.agents[role].act(
                state, self.tools.descriptions())

            if action.action == Action.REQUEST_TOOL:
                if action.tool in state.revealed_tools:
                    raise ValueError(
                        f"tool {action.tool} was requested more than once")
                result = self.tools.call(action.tool, snapshot)
                state.revealed_tools[action.tool] = result
                state.remaining_tool_budget -= result.cost
                tool_calls += 1
                state.transcript.append(TranscriptItem(
                    role=role, action=action.action, reason=action.reason,
                    tool=action.tool, tool_result=result))
                continue

            if action.action in {Action.PROPOSE, Action.CHALLENGE}:
                if action.candidate_class is None:
                    raise ValueError(
                        f"{action.action.value} requires candidate_class")
                state.current_candidate = action.candidate_class

            state.transcript.append(TranscriptItem(
                role=role, action=action.action, reason=action.reason,
                candidate_class=action.candidate_class,
                decision=action.decision))

            if action.action == Action.FINAL:
                candidate = (
                    action.candidate_class
                    if action.candidate_class is not None
                    else state.current_candidate)
                prediction = -1 if action.decision == "unknown" else int(candidate)
                result = AgenticResult(
                    prediction=prediction,
                    decision=str(action.decision),
                    candidate_class=candidate,
                    turns=turn + 1,
                    tool_calls=tool_calls,
                    transcript=tuple(item.to_dict() for item in state.transcript),
                )
                self._remember(state, result)
                return result

        # Fail closed: unresolved episodes become Unknown, never silent Known.
        result = AgenticResult(
            prediction=-1,
            decision="unknown",
            candidate_class=state.current_candidate,
            turns=self.max_turns,
            tool_calls=tool_calls,
            transcript=tuple(item.to_dict() for item in state.transcript),
        )
        self._remember(state, result)
        return result

    def _remember(self, state: SharedState, result: AgenticResult) -> None:
        if self.memory is None:
            return
        self.memory.add(Case(
            public_summary=dict(state.public_summary),
            revealed_tools={
                name: tool.value for name, tool in state.revealed_tools.items()},
            decision=result.decision,
            candidate_class=result.candidate_class,
            note="Stage15 agentic decision; no ground truth stored.",
        ))
