"""Role-specific observation builders with no ground-truth or hidden-state path."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import AAction, AGENT_IDS, BAction, CAction, action_count
from .data import EpisodeSnapshot
from .messages import Blackboard, EvidenceMessage


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    value = np.exp(shifted)
    return value / value.sum()


def ranked(logits: np.ndarray):
    probability = _softmax(logits)
    order = np.argsort(probability)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    entropy = float(-(probability * np.log(probability.clip(1e-12))).sum()
                    / np.log(len(probability)))
    return probability, top1, top2, float(probability[top1]), \
        float(probability[top2]), entropy


@dataclass
class PrivateRuntime:
    a_rechecked: bool = False
    a_secondary: float = 0.0
    a_abstained: bool = False
    b_view_index: int = 0
    b_last_view_risk: float = 0.0
    b_last_statement: tuple[str, int] | None = None
    b_abstained: bool = False
    c_revealed: dict[str, float] = field(default_factory=dict)
    last_actions: dict[str, int] = field(default_factory=lambda: {
        "identity": int(AAction.WAIT), "geometry": int(BAction.WAIT),
        "open_set": int(CAction.WAIT)})


def _message_vector(message: EvidenceMessage | None, blackboard: Blackboard,
                    snapshot: EpisodeSnapshot, agent: str) -> list[float]:
    if message is None:
        return [0.0] * 11
    logits = (snapshot.identity_logits if agent == "identity" else
              snapshot.geometry_logits if agent == "geometry" else None)
    matches_top1 = matches_top2 = 0.0
    if logits is not None and message.candidate_class is not None:
        _, top1, top2, *_ = ranked(logits)
        matches_top1 = float(message.candidate_class == top1)
        matches_top2 = float(message.candidate_class == top2)
    receiver_code = {"all": 0.0, "identity": 1/3, "geometry": 2/3,
                     "open_set": 1.0}[message.receiver]
    return [
        1.0, receiver_code,
        float(int(message.message_type)) / 5.0,
        float(message.candidate_class is not None),
        float(message.candidate_class == blackboard.current_hypothesis
              if message.candidate_class is not None else 0.0),
        matches_top1, matches_top2, float(message.confidence),
        float(np.tanh(message.margin)), float(message.support_risk),
        (float(int(message.evidence_type)) / 4.0
         + float(int(message.reason_code)) / 30.0),
    ]


def _common(agent: str, snapshot: EpisodeSnapshot, blackboard: Blackboard,
            runtime: PrivateRuntime, max_steps: int,
            initial_message_budget: int, initial_tool_budget: int) -> list[float]:
    pending = blackboard.pending_queries[agent]
    values = [
        float(blackboard.current_hypothesis is not None),
        float(blackboard.step) / max(max_steps, 1),
        float(blackboard.remaining_message_budget[agent]) /
        max(initial_message_budget, 1),
        float(blackboard.remaining_tool_budget) / max(initial_tool_budget, 1),
        float("identity" in pending), float("geometry" in pending),
        float("open_set" in pending),
    ]
    for sender in AGENT_IDS:
        values.extend(_message_vector(
            blackboard.last_from(sender), blackboard, snapshot, agent))
    last = np.zeros(action_count(agent), dtype=np.float32)
    last[int(runtime.last_actions[agent])] = 1.0
    values.extend(last.tolist())
    return values


def build_a_observation(snapshot: EpisodeSnapshot, blackboard: Blackboard,
                        runtime: PrivateRuntime, *, max_steps: int,
                        initial_message_budget: int,
                        initial_tool_budget: int) -> np.ndarray:
    p, top1, top2, p1, p2, entropy = ranked(snapshot.identity_logits)
    hypothesis = blackboard.current_hypothesis
    private = [
        p1, p2, p1-p2, entropy,
        float(snapshot.identity_prototype_risk),
        float(np.tanh(snapshot.identity_distance_margin)),
        float(hypothesis == top1 if hypothesis is not None else 0.0),
        float(hypothesis == top2 if hypothesis is not None else 0.0),
        float(p[hypothesis] if hypothesis is not None else 0.0),
        float(runtime.a_rechecked), float(runtime.a_secondary),
    ]
    return np.asarray(private + _common(
        "identity", snapshot, blackboard, runtime, max_steps,
        initial_message_budget, initial_tool_budget), dtype=np.float32)


def build_b_observation(snapshot: EpisodeSnapshot, blackboard: Blackboard,
                        runtime: PrivateRuntime, *, max_steps: int,
                        initial_message_budget: int,
                        initial_tool_budget: int) -> np.ndarray:
    p, top1, top2, p1, p2, entropy = ranked(snapshot.geometry_logits)
    hypothesis = blackboard.current_hypothesis
    private = [
        p1, p2, p1-p2, entropy,
        float(snapshot.geometry_prototype_risk),
        float(np.tanh(snapshot.geometry_distance_margin)),
        float(hypothesis == top1 if hypothesis is not None else 0.0),
        float(hypothesis == top2 if hypothesis is not None else 0.0),
        float(p[hypothesis] if hypothesis is not None else 0.0),
        float(runtime.b_view_index) / max(len(snapshot.view_risks), 1),
        float(runtime.b_last_view_risk),
    ]
    return np.asarray(private + _common(
        "geometry", snapshot, blackboard, runtime, max_steps,
        initial_message_budget, initial_tool_budget), dtype=np.float32)


def build_c_observation(snapshot: EpisodeSnapshot, blackboard: Blackboard,
                        runtime: PrivateRuntime, *, max_steps: int,
                        initial_message_budget: int,
                        initial_tool_budget: int) -> np.ndarray:
    private = []
    for name in ("id_prototype", "geo_prototype", "openmax", "boundary"):
        private.extend([float(name in runtime.c_revealed),
                        float(runtime.c_revealed.get(name, 0.0))])
    return np.asarray(private + _common(
        "open_set", snapshot, blackboard, runtime, max_steps,
        initial_message_budget, initial_tool_budget), dtype=np.float32)
