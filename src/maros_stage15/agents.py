"""Autonomous role-specialized LLM agents for open-set RF reasoning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from .memory import DEFAULT_GLOBAL_KNOWLEDGE, CaseMemory
from .providers import ChatModel
from .schemas import Action, AgentAction, AgentContext, Role


ROLE_PROMPTS = {
    Role.PROPOSER: (
        "You are the Proposer Agent for open-set specific-emitter identification. "
        "You only observe identity-side RF evidence. Form a known-class hypothesis, "
        "request identity-side tools when needed, and communicate evidence you "
        "actually observed. You do not know the Critic's geometry/open-set evidence."
    ),
    Role.CRITIC: (
        "You are the Critic Agent. You only observe geometry/open-set-side RF "
        "evidence. Actively search for counter-evidence, unknown risk and "
        "domain-shift symptoms. Do not simply agree with the Proposer. Request "
        "your private RF tools and communicate evidence when it is decision-relevant."
    ),
    Role.ARBITER: (
        "You are the Arbiter Agent. You do not directly observe raw classifier, "
        "geometry or open-set scores and you have no RF tools. Integrate only the "
        "messages supplied by the other agents. If evidence is insufficient, ask "
        "a specific peer for more evidence. You alone make the final Known/Unknown "
        "decision."
    ),
}

ALLOWED_ACTIONS = {
    Role.PROPOSER: {
        Action.REQUEST_TOOL, Action.PROPOSE, Action.SHARE_EVIDENCE,
        Action.ASK_PEER, Action.ABSTAIN,
    },
    Role.CRITIC: {
        Action.REQUEST_TOOL, Action.SUPPORT, Action.CHALLENGE,
        Action.SHARE_EVIDENCE, Action.ASK_PEER, Action.ABSTAIN,
    },
    Role.ARBITER: {Action.ASK_PEER, Action.ABSTAIN, Action.FINAL},
}

JSON_CONTRACT = """
Return exactly one JSON object and no markdown:
{
  "action": "request_tool|propose|support|challenge|share_evidence|ask_peer|abstain|final",
  "reason": "brief evidence-grounded reason",
  "tool": "tool name or null",
  "recipient": "proposer|critic|arbiter|null",
  "candidate_class": "integer or null",
  "decision": "known|unknown|null",
  "confidence": "number in [0,1] or null"
}
"""


@dataclass
class LLMAgent:
    role: Role
    model: ChatModel
    memory: CaseMemory
    global_knowledge: str = DEFAULT_GLOBAL_KNOWLEDGE

    def act(self, context: AgentContext,
            tool_descriptions: dict[str, str]) -> AgentAction:
        payload = {
            "role": self.role.value,
            "global_rf_knowledge": self.global_knowledge,
            "available_private_tools": tool_descriptions,
            "local_context": context.safe_dict(),
            "similar_private_cases": self.memory.retrieve(
                context.private_summary, k=3),
            "constraints": [
                "Ground truth, dataset provenance and formal-unknown flags are hidden.",
                "Never invent RF measurements or another agent's observations.",
                "A tool result remains private until explicitly shared.",
                "Use only actions allowed by your role.",
                "Use candidate IDs only within the known-class range.",
                "Only the Arbiter may make a final decision.",
            ],
        }
        response = self.model.complete([
            {
                "role": "system",
                "content": ROLE_PROMPTS[self.role] + "\n" + JSON_CONTRACT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")),
            },
        ])
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{self.role.value} returned invalid JSON: {response}") from exc
        action = AgentAction.from_dict(parsed)
        self._validate(action, context, tool_descriptions)
        return action

    def _validate(self, action: AgentAction, context: AgentContext,
                  tools: dict[str, str]) -> None:
        if action.action not in ALLOWED_ACTIONS[self.role]:
            raise ValueError(
                f"{self.role.value} cannot use action {action.action.value}")

        if action.action == Action.REQUEST_TOOL:
            if action.tool not in tools:
                raise ValueError(
                    f"{self.role.value} cannot request tool {action.tool}")
            if context.remaining_tool_budget <= 0:
                raise ValueError("tool requested with exhausted global budget")

        if action.action == Action.SHARE_EVIDENCE:
            if action.tool not in context.private_tools:
                raise ValueError(
                    "share_evidence requires one previously observed private tool")
            if action.recipient is None or action.recipient == self.role:
                raise ValueError("share_evidence requires another recipient")

        if action.action == Action.ASK_PEER:
            if action.recipient is None or action.recipient == self.role:
                raise ValueError("ask_peer requires another recipient")

        if action.candidate_class is not None:
            if not 0 <= action.candidate_class < context.num_known_classes:
                raise ValueError("candidate class is outside known-class range")

        if action.action == Action.PROPOSE and action.candidate_class is None:
            raise ValueError("propose requires candidate_class")
        if action.action == Action.CHALLENGE and action.candidate_class is None:
            raise ValueError("challenge requires candidate_class")

        if action.action == Action.FINAL:
            if action.decision is None:
                raise ValueError("final requires known/unknown decision")
            if action.decision == "known" and action.candidate_class is None:
                raise ValueError("known final requires candidate_class")


def build_default_agents(
        model: ChatModel | Mapping[Role, ChatModel],
        memory_size: int = 256) -> dict[Role, LLMAgent]:
    """Build three independent agents.

    The agents may share one foundation-model backend, but they never share
    conversation context, private observations, private tools or memory. A mapping
    can be supplied when different LLM backends are desired for different roles.
    """
    if isinstance(model, Mapping):
        models = dict(model)
    else:
        models = {role: model for role in Role}
    if set(models) != set(Role):
        raise ValueError("model mapping must define proposer, critic and arbiter")
    return {
        role: LLMAgent(
            role=role,
            model=models[role],
            memory=CaseMemory(max_cases=int(memory_size)),
        )
        for role in Role
    }
