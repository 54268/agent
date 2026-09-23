import json

import numpy as np

from maros_stage14.data import EpisodeSnapshot, with_label
from maros_stage15.agents import build_default_agents
from maros_stage15.orchestrator import AgenticOpenSetSystem
from maros_stage15.providers import ScriptedChatModel
from maros_stage15.schemas import Role
from maros_stage15.tools import public_rf_summary


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


def test_public_summary_has_no_label_or_provenance():
    a = public_rf_summary(snapshot(label=0))
    b = public_rf_summary(with_label(snapshot(label=0), 2))
    assert a == b
    text = json.dumps(a)
    assert "label" not in text
    assert "source_kind" not in text
    assert "formal_unknown" not in text


def test_three_llm_roles_can_reach_known_final():
    model = ScriptedChatModel([
        '{"action":"propose","reason":"top1 dominates","tool":null,'
        '"candidate_class":0,"decision":null,"confidence":0.9}',
        '{"action":"request_tool","reason":"check open-set risk","tool":"openmax",'
        '"candidate_class":null,"decision":null,"confidence":null}',
        '{"action":"final","reason":"proposal strong and no high unknown risk",'
        '"tool":null,"candidate_class":0,"decision":"known","confidence":0.88}',
    ])
    agents = build_default_agents(model)
    system = AgenticOpenSetSystem(agents, max_turns=3, tool_budget=2)
    result = system.run(snapshot())
    assert result.prediction == 0
    assert result.decision == "known"
    assert result.tool_calls == 1


def test_timeout_fails_closed_to_unknown():
    model = ScriptedChatModel([
        '{"action":"abstain","reason":"need more evidence","tool":null,'
        '"candidate_class":null,"decision":null,"confidence":0.2}',
        '{"action":"abstain","reason":"insufficient counter-evidence","tool":null,'
        '"candidate_class":null,"decision":null,"confidence":0.2}',
        '{"action":"abstain","reason":"not enough evidence to finalize","tool":null,'
        '"candidate_class":null,"decision":null,"confidence":0.2}',
    ])
    system = AgenticOpenSetSystem(build_default_agents(model), max_turns=3)
    result = system.run(snapshot())
    assert result.prediction == -1
    assert result.decision == "unknown"
