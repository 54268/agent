import json

import numpy as np

from maros_stage14.data import EpisodeSnapshot, with_label
from maros_stage15.agents import build_default_agents
from maros_stage15.orchestrator import AgenticOpenSetSystem
from maros_stage15.providers import ScriptedChatModel
from maros_stage15.schemas import Role
from maros_stage15.tools import role_private_summary


def snapshot(label=0):
    return EpisodeSnapshot(
        sample_id="x", label=label, source_kind="known",
        identity_logits=np.array([4.0, 1.0, 0.2], dtype=np.float32),
        geometry_logits=np.array([3.0, 1.2, 0.1], dtype=np.float32),
        identity_prototype_risk=0.1,
        geometry_prototype_risk=0.2,
        openmax_risk=0.15,
        boundary_risk=0.2,
        identity_energy=0.3,
        identity_distance_margin=1.0,
        geometry_distance_margin=0.8,
        view_risks=np.array([0.1, 0.2], dtype=np.float32),
    )


def test_private_observations_have_no_label_or_provenance():
    a = role_private_summary(snapshot(label=0), Role.PROPOSER)
    b = role_private_summary(with_label(snapshot(label=0), 2), Role.PROPOSER)
    assert a == b
    text = json.dumps(a)
    assert "label" not in text
    assert "source_kind" not in text
    assert "formal_unknown" not in text


def test_roles_receive_different_observations():
    s = snapshot()
    proposer = role_private_summary(s, Role.PROPOSER)
    critic = role_private_summary(s, Role.CRITIC)
    arbiter = role_private_summary(s, Role.ARBITER)
    assert proposer != critic
    assert arbiter == {}
    assert all(key.startswith("identity_") for key in proposer)
    assert all(key.startswith("geometry_") for key in critic)


def test_three_agents_can_reach_known_final():
    model = ScriptedChatModel([
        '{"action":"propose","reason":"identity top1 dominates","tool":null,'
        '"recipient":null,"candidate_class":0,"decision":null,"confidence":0.9}',
        '{"action":"support","reason":"geometry does not contradict candidate",'
        '"tool":null,"recipient":null,"candidate_class":0,'
        '"decision":null,"confidence":0.8}',
        '{"action":"final","reason":"independent identity and geometry agree",'
        '"tool":null,"recipient":null,"candidate_class":0,'
        '"decision":"known","confidence":0.88}',
    ])
    system = AgenticOpenSetSystem(
        build_default_agents(model), max_turns=3, tool_budget=2,
        message_budget=4)
    result = system.run(snapshot())
    assert result.prediction == 0
    assert result.decision == "known"
    assert result.message_count == 2


def test_private_tool_is_not_shared_automatically():
    model = ScriptedChatModel([
        '{"action":"request_tool","reason":"verify identity prototype","tool":"identity_prototype",'
        '"recipient":null,"candidate_class":null,"decision":null,"confidence":null}',
        '{"action":"propose","reason":"identity evidence supports class 0","tool":null,'
        '"recipient":null,"candidate_class":0,"decision":null,"confidence":0.9}',
        '{"action":"support","reason":"geometry supports current candidate","tool":null,'
        '"recipient":null,"candidate_class":0,"decision":null,"confidence":0.8}',
        '{"action":"final","reason":"sufficient communicated evidence","tool":null,'
        '"recipient":null,"candidate_class":0,"decision":"known","confidence":0.8}',
    ])
    system = AgenticOpenSetSystem(
        build_default_agents(model), max_turns=4, tool_budget=2,
        message_budget=4)
    result = system.run(snapshot())
    assert result.tool_calls == 1
    # Raw private tool result stays only in the owner's action log.
    proposal_rows = [r for r in result.transcript if r["agent"] == "proposer"]
    assert "private_tool_result" in proposal_rows[0]
    assert "identity_prototype" not in json.dumps(
        [r for r in result.transcript if r["agent"] != "proposer"])


def test_timeout_fails_closed_to_unknown():
    model = ScriptedChatModel([
        '{"action":"abstain","reason":"need more evidence","tool":null,'
        '"recipient":null,"candidate_class":null,"decision":null,"confidence":0.2}',
        '{"action":"abstain","reason":"insufficient evidence","tool":null,'
        '"recipient":null,"candidate_class":null,"decision":null,"confidence":0.2}',
        '{"action":"abstain","reason":"cannot finalize","tool":null,'
        '"recipient":null,"candidate_class":null,"decision":null,"confidence":0.2}',
    ])
    system = AgenticOpenSetSystem(
        build_default_agents(model), max_turns=3, message_budget=4)
    result = system.run(snapshot())
    assert result.prediction == -1
    assert result.decision == "unknown"
