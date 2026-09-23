"""Role-specialized LLM agents for open-set RF reasoning."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .memory import DEFAULT_GLOBAL_KNOWLEDGE, CaseMemory
from .providers import ChatModel
from .schemas import Action, AgentAction, Role, SharedState


ROLE_PROMPTS = {
    Role.PROPOSER: (
        "You are the Proposer Agent in open-set specific-emitter identification. "
        "Form a known-class hypothesis from structured RF evidence. You may request "
        "one RF tool at a time before proposing. Do not invent measurements. "
        "Prefer a candidate already supported by the identity/geometry rankings."
    ),
    Role.CRITIC: (
        "You are the Critic Agent. Your job is to actively search for counter-"
        "evidence, domain-shift symptoms and open-set risk. You may request RF "
        "tools, support the current candidate, challenge it with another ranked "
        "candidate, or abstain. Do not simply agree with the Proposer."
    ),
    Role.ARBITER: (
        "You are the Arbiter Agent. Integrate the Proposer and Critic arguments "
        "with revealed RF tools. You alone may make FINAL known/unknown decisions. "
        "If evidence is insufficient and budget remains, request another tool. "
        "A known decision must name the accepted candidate class."
    ),
}

JSON_CONTRACT = """
Return exactly one JSON object, no markdown:
{
  "action": "request_tool|propose|support|challenge|abstain|final",
  "reason": "brief evidence-grounded reason",
  "tool": "tool name or null",
  "candidate_class": "integer or null",
  "decision": "known|unknown|null",
  "confidence": "number in [0,1] or null"
}
"""


@dataclass
class LLMAgent:
    role: Role
    model: ChatModel
    memory: CaseMemory | None = None
    global_knowledge: str = DEFAULT_GLOBAL_KNOWLEDGE

    def act(self, state: SharedState, tool_descriptions: dict[str, str]) -> AgentAction:
        memory_hits = (
            self.memory.retrieve(state.public_summary)
            if self.memory is not None else [])
        payload = {
            "role": self.role.value,
            "global_rf_knowledge": self.global_knowledge,
            "available_tools": tool_descriptions,
            "shared_state": state.safe_dict(),
            "similar_past_cases": memory_hits,
            "constraints": [
                "Never use or infer ground-truth labels or dataset provenance.",
                "Never fabricate tool outputs.",
                "Request only tools listed in available_tools.",
                "Only arbiter may use action=final.",
                "Use candidate IDs only within [0, num_known_classes-1].",
            ],
        }
        response = self.model.complete([
            {"role": "system", "content": ROLE_PROMPTS[self.role] + "\n" + JSON_CONTRACT},
            {"role": "user", "content": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"))},
        ])
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{self.role.value} returned invalid JSON: {response}") from exc
        action = AgentAction.from_dict(parsed)
        self._validate(action, state, tool_descriptions)
        return action

    def _validate(self, action: AgentAction, state: SharedState,
                  tools: dict[str, str]) -> None:
        if action.action == Action.REQUEST_TOOL:
            if action.tool not in tools:
                raise ValueError(f"unknown requested tool: {action.tool}")
            if state.remaining_tool_budget <= 0:
                raise ValueError("tool requested with exhausted budget")
        if action.candidate_class is not None:
            k = int(state.public_summary["num_known_classes"])
            if not 0 <= action.candidate_class < k:
                raise ValueError("candidate class is outside known-class range")
        if action.action == Action.FINAL and self.role != Role.ARBITER:
            raise ValueError("only Arbiter may issue FINAL")
        if self.role == Role.ARBITER and action.action == Action.FINAL:
            if action.decision is None:
                raise ValueError("FINAL requires known/unknown decision")
            if action.decision == "known" and action.candidate_class is None:
                raise ValueError("known FINAL requires candidate_class")


def build_default_agents(model: ChatModel, memory: CaseMemory | None = None):
    return {
        role: LLMAgent(role=role, model=model, memory=memory)
        for role in (Role.PROPOSER, Role.CRITIC, Role.ARBITER)
    }
