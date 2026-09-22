"""Sequential evidence-acquisition Dec-POMDP for open-set SEI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .actions import (AAction, AGENT_IDS, BAction, CAction, action_count,
                      action_name)
from .data import EpisodeSnapshot
from .messages import (Blackboard, EvidenceMessage, EvidenceType, MessageType,
                       ReasonCode)
from .observations import (PrivateRuntime, build_a_observation,
                           build_b_observation, build_c_observation, ranked)


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 6
    message_budget: int = 4
    tool_budget: int = 3
    step_cost: float = 0.01
    message_cost: float = 0.01
    tool_cost: float = 0.02
    min_tools_before_decision: int = 1
    use_geometry_actor: bool = True

    def __post_init__(self) -> None:
        if self.max_steps < 2:
            raise ValueError("max_steps must be at least two")
        if self.message_budget < 1 or self.tool_budget < 1:
            raise ValueError("communication and tool budgets must be positive")
        if not 0 <= self.min_tools_before_decision <= self.tool_budget:
            raise ValueError("min_tools_before_decision must fit the tool budget")


class OSSEIMultiAgentEnv:
    """Joint-step environment with one-round-delayed structured messages."""

    def __init__(self, config: EnvConfig | None = None,
                 *, allow_formal_unknown: bool = False,
                 tool_overrides: Mapping[str, Mapping[str, float]] | None = None):
        self.config = config or EnvConfig()
        self.agents = (AGENT_IDS if self.config.use_geometry_actor else
                       ("identity", "open_set"))
        self.allow_formal_unknown = bool(allow_formal_unknown)
        self.tool_overrides = dict(tool_overrides or {})
        self.snapshot: EpisodeSnapshot | None = None
        self.blackboard: Blackboard | None = None
        self.runtime: PrivateRuntime | None = None
        self.terminated = False
        self.truncated = False
        self.prediction: int | None = None
        self.trajectory: list[dict] = []

    def reset(self, snapshot: EpisodeSnapshot):
        if snapshot.formal_unknown and not self.allow_formal_unknown:
            raise RuntimeError("formal Unknown cannot enter a training environment")
        self.snapshot = snapshot
        self.blackboard = Blackboard(
            remaining_message_budget={agent: self.config.message_budget
                                      for agent in AGENT_IDS},
            remaining_tool_budget=self.config.tool_budget)
        self.runtime = PrivateRuntime()
        self.terminated = False
        self.truncated = False
        self.prediction = None
        self.trajectory = []
        return self.observations(), {"sample_id": snapshot.sample_id}

    def _require_active(self) -> None:
        if self.snapshot is None or self.blackboard is None or self.runtime is None:
            raise RuntimeError("environment must be reset before use")
        if self.terminated or self.truncated:
            raise RuntimeError("episode has already ended")

    def observations(self) -> dict[str, np.ndarray]:
        if self.snapshot is None or self.blackboard is None or self.runtime is None:
            raise RuntimeError("environment is not initialised")
        kw = dict(max_steps=self.config.max_steps,
                  initial_message_budget=self.config.message_budget,
                  initial_tool_budget=self.config.tool_budget)
        observations = {
            "identity": build_a_observation(
                self.snapshot, self.blackboard, self.runtime, **kw),
            "geometry": build_b_observation(
                self.snapshot, self.blackboard, self.runtime, **kw),
            "open_set": build_c_observation(
                self.snapshot, self.blackboard, self.runtime, **kw),
        }
        return {agent: observations[agent] for agent in self.agents}

    def action_masks(self) -> dict[str, np.ndarray]:
        self._require_active()
        board, runtime = self.blackboard, self.runtime
        assert board is not None and runtime is not None and self.snapshot is not None
        a = np.zeros(action_count("identity"), dtype=bool)
        b = np.zeros(action_count("geometry"), dtype=bool)
        c = np.zeros(action_count("open_set"), dtype=bool)
        a[AAction.WAIT] = b[BAction.WAIT] = c[CAction.WAIT] = True
        if board.remaining_message_budget["identity"] > 0:
            if (self.config.use_geometry_actor and
                    "identity" not in board.pending_queries["geometry"]):
                a[AAction.ASK_B] = True
            if not runtime.a_abstained:
                a[AAction.ABSTAIN] = True
            if board.current_hypothesis is None:
                a[[AAction.PROPOSE_TOP1, AAction.PROPOSE_TOP2]] = True
            if board.current_hypothesis is not None:
                if "identity" not in board.pending_queries["open_set"]:
                    a[AAction.ASK_C] = True
            if "geometry" in board.pending_queries["identity"]:
                a[AAction.SEND_EVIDENCE_B] = True
            if "open_set" in board.pending_queries["identity"]:
                a[AAction.SEND_EVIDENCE_C] = True
        if not runtime.a_rechecked:
            a[AAction.RECHECK] = True
        if (self.config.use_geometry_actor and
                board.remaining_message_budget["geometry"] > 0):
            if "geometry" not in board.pending_queries["identity"]:
                b[BAction.ASK_A] = True
            if not runtime.b_abstained:
                b[BAction.ABSTAIN] = True
            if board.current_hypothesis is not None:
                _, b_top1, b_top2, *_ = ranked(self.snapshot.geometry_logits)
                if (b_top1 == board.current_hypothesis and
                        runtime.b_last_statement != ("support", b_top1)):
                    b[BAction.SUPPORT] = True
                if (b_top1 != board.current_hypothesis and
                        runtime.b_last_statement != ("challenge", b_top1)):
                    b[BAction.CHALLENGE_TOP1] = True
                if (b_top2 != board.current_hypothesis and
                        runtime.b_last_statement != ("challenge", b_top2)):
                    b[BAction.CHALLENGE_TOP2] = True
                if "geometry" not in board.pending_queries["open_set"]:
                    b[BAction.ASK_C] = True
            if "identity" in board.pending_queries["geometry"]:
                b[BAction.SEND_EVIDENCE_A] = True
            if "open_set" in board.pending_queries["geometry"]:
                b[BAction.SEND_EVIDENCE_C] = True
        if runtime.b_view_index < len(self.snapshot.view_risks):
            b[BAction.RECHECK_VIEW] = True
        if board.remaining_message_budget["open_set"] > 0:
            if "open_set" not in board.pending_queries["identity"]:
                c[CAction.QUERY_A] = True
            if (self.config.use_geometry_actor and
                    "open_set" not in board.pending_queries["geometry"]):
                c[CAction.QUERY_B] = True
        if board.current_hypothesis is not None:
            if len(runtime.c_revealed) >= self.config.min_tools_before_decision:
                c[[CAction.ACCEPT_CURRENT_KNOWN, CAction.REJECT_UNKNOWN]] = True
            if board.remaining_tool_budget > 0:
                for action, key in (
                    (CAction.CHECK_ID_PROTOTYPE, "id_prototype"),
                    (CAction.CHECK_GEO_PROTOTYPE, "geo_prototype"),
                    (CAction.CHECK_OPENMAX, "openmax"),
                    (CAction.CHECK_BOUNDARY, "boundary"),
                ):
                    if key not in runtime.c_revealed:
                        c[action] = True
        masks = {"identity": a, "geometry": b, "open_set": c}
        return {agent: masks[agent] for agent in self.agents}

    def state(self) -> np.ndarray:
        """Centralized critic state: all local observations and public masks.

        Ground truth, source provenance and unrevealed C tool values are absent.
        """
        observations = self.observations()
        masks = self.action_masks()
        return np.concatenate([
            *(observations[agent] for agent in self.agents),
            *(masks[agent].astype(np.float32) for agent in self.agents),
        ]).astype(np.float32)

    def _spend_message(self, sender: str) -> None:
        assert self.blackboard is not None
        self.blackboard.remaining_message_budget[sender] -= 1

    def _append(self, message: EvidenceMessage) -> None:
        assert self.blackboard is not None
        self._spend_message(message.sender)
        self.blackboard.append(message)

    def step(self, actions: Mapping[str, int]):
        self._require_active()
        if set(actions) != set(self.agents):
            raise ValueError("joint action must contain exactly the active agents")
        masks = self.action_masks()
        chosen = {agent: int(actions[agent]) for agent in self.agents}
        for agent, action in chosen.items():
            if action < 0 or action >= len(masks[agent]) or not masks[agent][action]:
                raise ValueError(f"invalid masked action {agent}:{action}")
        assert self.snapshot is not None and self.blackboard is not None
        assert self.runtime is not None
        snapshot, board, runtime = self.snapshot, self.blackboard, self.runtime
        current_observations = self.observations()
        pending_at_round_start = {
            receiver: set(senders)
            for receiver, senders in board.pending_queries.items()}
        old_hypothesis = board.current_hypothesis
        emitted_before = len(board.public_messages)
        tool_calls = 0
        a_action = AAction(chosen["identity"])
        b_action = BAction(chosen.get("geometry", int(BAction.WAIT)))
        c_action = CAction(chosen["open_set"])
        _, a_top1, a_top2, a_p1, _, _ = ranked(snapshot.identity_logits)
        _, b_top1, b_top2, b_p1, _, _ = ranked(snapshot.geometry_logits)

        # A owns new identity proposals. B challenges only the hypothesis it saw;
        # a simultaneous fresh A proposal therefore has deterministic priority.
        a_proposal = None
        if a_action in {AAction.PROPOSE_TOP1, AAction.PROPOSE_TOP2}:
            a_proposal = a_top1 if a_action == AAction.PROPOSE_TOP1 else a_top2
            board.current_hypothesis = a_proposal
            self._append(EvidenceMessage(
                "identity", "all", MessageType.PROPOSAL, a_proposal,
                a_p1, float(np.tanh(snapshot.identity_distance_margin)),
                snapshot.identity_prototype_risk, EvidenceType.IDENTITY,
                ReasonCode.IDENTITY_STRONG, board.step))
        if (b_action in {BAction.CHALLENGE_TOP1, BAction.CHALLENGE_TOP2}
                and a_proposal is None):
            candidate = b_top1 if b_action == BAction.CHALLENGE_TOP1 else b_top2
            board.current_hypothesis = candidate
            runtime.b_last_statement = ("challenge", candidate)
            self._append(EvidenceMessage(
                "geometry", "all", MessageType.CHALLENGE, candidate,
                b_p1, float(np.tanh(snapshot.geometry_distance_margin)),
                snapshot.geometry_prototype_risk, EvidenceType.GEOMETRY,
                ReasonCode.PROTOTYPE_CONFLICT, board.step))
        elif b_action == BAction.SUPPORT:
            runtime.b_last_statement = ("support", int(old_hypothesis))
            self._append(EvidenceMessage(
                "geometry", "all", MessageType.SUPPORT, old_hypothesis,
                b_p1, float(np.tanh(snapshot.geometry_distance_margin)),
                snapshot.geometry_prototype_risk, EvidenceType.GEOMETRY,
                ReasonCode.GEOMETRY_SUPPORT, board.step))

        query_actions = (
            (a_action == AAction.ASK_B, "identity", "geometry"),
            (a_action == AAction.ASK_C, "identity", "open_set"),
            (b_action == BAction.ASK_A, "geometry", "identity"),
            (b_action == BAction.ASK_C, "geometry", "open_set"),
            (c_action == CAction.QUERY_A, "open_set", "identity"),
            (c_action == CAction.QUERY_B, "open_set", "geometry"),
        )
        for active, sender, receiver in query_actions:
            if active:
                board.pending_queries[receiver].add(sender)
                self._append(EvidenceMessage(
                    sender, receiver, MessageType.QUERY,
                    old_hypothesis, evidence_type=EvidenceType.NONE,
                    reason_code=ReasonCode.REQUESTED, step=board.step))

        send_actions = (
            (a_action == AAction.SEND_EVIDENCE_B, "identity", "geometry"),
            (a_action == AAction.SEND_EVIDENCE_C, "identity", "open_set"),
            (b_action == BAction.SEND_EVIDENCE_A, "geometry", "identity"),
            (b_action == BAction.SEND_EVIDENCE_C, "geometry", "open_set"),
        )
        for active, sender, receiver in send_actions:
            if active:
                board.pending_queries[sender].discard(receiver)
                is_a = sender == "identity"
                self._append(EvidenceMessage(
                    sender, receiver, MessageType.EVIDENCE,
                    a_top1 if is_a else b_top1,
                    a_p1 if is_a else b_p1,
                    float(np.tanh(snapshot.identity_distance_margin if is_a else
                                  snapshot.geometry_distance_margin)),
                    snapshot.identity_prototype_risk if is_a else
                    snapshot.geometry_prototype_risk,
                    EvidenceType.IDENTITY if is_a else EvidenceType.GEOMETRY,
                    ReasonCode.REQUESTED, board.step))

        for active, sender, confidence, risk, evidence in (
            (a_action == AAction.ABSTAIN, "identity", a_p1,
             snapshot.identity_prototype_risk, EvidenceType.IDENTITY),
            (b_action == BAction.ABSTAIN, "geometry", b_p1,
             snapshot.geometry_prototype_risk, EvidenceType.GEOMETRY),
        ):
            if active:
                if sender == "identity":
                    runtime.a_abstained = True
                else:
                    runtime.b_abstained = True
                self._append(EvidenceMessage(
                    sender, "all", MessageType.ABSTAIN, old_hypothesis,
                    confidence, support_risk=risk, evidence_type=evidence,
                    reason_code=ReasonCode.IDENTITY_UNCERTAIN, step=board.step))

        called_tools = []
        newly_revealed = {}
        if a_action == AAction.RECHECK:
            runtime.a_rechecked = True
            runtime.a_secondary = float(np.tanh(snapshot.identity_energy))
            tool_calls += 1
            called_tools.append("identity_recheck")
        if b_action == BAction.RECHECK_VIEW:
            runtime.b_last_view_risk = float(
                snapshot.view_risks[runtime.b_view_index])
            runtime.b_view_index += 1
            tool_calls += 1
            called_tools.append("geometry_view_recheck")
        c_tools = {
            CAction.CHECK_ID_PROTOTYPE: ("id_prototype", snapshot.identity_prototype_risk),
            CAction.CHECK_GEO_PROTOTYPE: ("geo_prototype", snapshot.geometry_prototype_risk),
            CAction.CHECK_OPENMAX: ("openmax", snapshot.openmax_risk),
            CAction.CHECK_BOUNDARY: ("boundary", snapshot.boundary_risk),
        }
        if c_action in c_tools:
            key, value = c_tools[c_action]
            value = float(self.tool_overrides.get(snapshot.sample_id, {}).get(key, value))
            runtime.c_revealed[key] = float(value)
            board.remaining_tool_budget -= 1
            tool_calls += 1
            called_tools.append(key)
            newly_revealed[key] = float(value)
            # ASK_C is a delayed request. A learned CHECK action is C's
            # response, converted to the same bounded public evidence schema.
            requesters = tuple(pending_at_round_start["open_set"])
            for requester in requesters:
                if board.remaining_message_budget["open_set"] <= 0:
                    break
                self._append(EvidenceMessage(
                    "open_set", requester, MessageType.EVIDENCE,
                    old_hypothesis, confidence=float(1.0-value),
                    support_risk=float(value), evidence_type=EvidenceType.OPEN_SET,
                    reason_code=ReasonCode.OPEN_SET_RISK, step=board.step))
                board.pending_queries["open_set"].discard(requester)

        terminal_reward = 0.0
        termination_reason = None
        # C acts on the hypothesis present in o_t, never a simultaneous update.
        if c_action == CAction.ACCEPT_CURRENT_KNOWN:
            self.prediction = int(old_hypothesis)
            self.terminated = True
            correct = not snapshot.is_unknown and self.prediction == snapshot.label
            terminal_reward = 1.0 if correct else -1.0
            termination_reason = "accepted_known"
        elif c_action == CAction.REJECT_UNKNOWN:
            self.prediction = -1
            self.terminated = True
            terminal_reward = 1.0 if snapshot.is_unknown else -1.0
            termination_reason = "rejected_unknown"

        runtime.last_actions.update(chosen)
        board.step += 1
        message_count = len(board.public_messages) - emitted_before
        cost = (self.config.step_cost + self.config.message_cost * message_count
                + self.config.tool_cost * tool_calls)
        reward = terminal_reward - cost
        if not self.terminated and board.step >= self.config.max_steps:
            self.truncated = True
            self.prediction = -1
            reward += -1.0
            termination_reason = "timeout_failure"
        trace = {
            "sample_id": snapshot.sample_id, "step": board.step - 1,
            "actions": {agent: action_name(agent, value)
                        for agent, value in chosen.items()},
            "valid_action_masks": {agent: masks[agent].astype(int).tolist()
                                   for agent in self.agents},
            "local_observation_summary": {
                agent: {"mean": float(value.mean()),
                        "std": float(value.std()),
                        "l2": float(np.linalg.norm(value))}
                for agent, value in current_observations.items()},
            "current_hypothesis": board.current_hypothesis,
            "message_count": message_count, "tool_calls": tool_calls,
            "messages": [{
                "sender": message.sender, "receiver": message.receiver,
                "message_type": message.message_type.name,
                "candidate_class": message.candidate_class,
                "confidence": message.confidence, "margin": message.margin,
                "support_risk": message.support_risk,
                "evidence_type": message.evidence_type.name,
                "reason_code": message.reason_code.name,
            } for message in board.public_messages[-message_count:]]
            if message_count else [],
            "tool_called": called_tools,
            "revealed_evidence": newly_revealed,
            "remaining_budget": dict(board.remaining_message_budget),
            "remaining_tool_budget": board.remaining_tool_budget,
            "reward": float(reward), "termination_reason": termination_reason,
        }
        self.trajectory.append(trace)
        info = dict(trace)
        info.update({"prediction": self.prediction,
                     "decision_hypothesis": old_hypothesis,
                     "correct": bool(terminal_reward > 0) if self.terminated else False,
                     "trajectory": list(self.trajectory)})
        observations = self.observations()
        return observations, float(reward), self.terminated, self.truncated, info
