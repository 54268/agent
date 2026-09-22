import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage14.actions import AAction, BAction, CAction  # noqa: E402
from maros_stage14.data import (EpisodeSnapshot, TrainingEpisodeBank,
                                with_label)  # noqa: E402
from maros_stage14.env import EnvConfig, OSSEIMultiAgentEnv  # noqa: E402
from maros_stage14.messages import Blackboard  # noqa: E402


def snapshot(label=1, source="known", formal=False):
    return EpisodeSnapshot(
        sample_id="sample-1", label=label, source_kind=source,
        identity_logits=np.array([0.1, 3.0, 0.2]),
        geometry_logits=np.array([0.2, 2.5, 0.1]),
        identity_prototype_risk=0.2, geometry_prototype_risk=0.15,
        openmax_risk=0.1, boundary_risk=0.12, identity_energy=-0.5,
        identity_distance_margin=0.4, geometry_distance_margin=0.5,
        view_risks=np.array([0.1, 0.2, 0.15, 0.12, 0.18]),
        formal_unknown=formal)


def wait_actions():
    return {"identity": int(AAction.WAIT),
            "geometry": int(BAction.WAIT),
            "open_set": int(CAction.WAIT)}


def test_observations_are_role_specific_and_ground_truth_free():
    env = OSSEIMultiAgentEnv()
    observations, _ = env.reset(snapshot())
    assert len({len(value) for value in observations.values()}) == 3
    state_known = env.state().copy()
    env.reset(with_label(snapshot(), 2))
    np.testing.assert_array_equal(state_known, env.state())


def test_query_is_delayed_and_enables_receiver_send_next_round():
    env = OSSEIMultiAgentEnv()
    before, _ = env.reset(snapshot())
    actions = wait_actions(); actions["identity"] = int(AAction.ASK_B)
    after, _, terminated, truncated, _ = env.step(actions)
    assert not terminated and not truncated
    assert not np.array_equal(before["geometry"], after["geometry"])
    assert env.action_masks()["geometry"][BAction.SEND_EVIDENCE_A]


def test_proposal_then_tool_reveal_then_terminal_decision():
    env = OSSEIMultiAgentEnv()
    env.reset(snapshot())
    assert not env.action_masks()["open_set"][CAction.ACCEPT_CURRENT_KNOWN]
    actions = wait_actions(); actions["identity"] = int(AAction.PROPOSE_TOP1)
    env.step(actions)
    assert env.action_masks()["open_set"][CAction.CHECK_OPENMAX]
    assert not env.action_masks()["open_set"][CAction.ACCEPT_CURRENT_KNOWN]
    before = env.observations()["open_set"].copy()
    actions = wait_actions()
    actions["geometry"] = int(BAction.SUPPORT)
    actions["open_set"] = int(CAction.CHECK_OPENMAX)
    after, _, _, _, _ = env.step(actions)
    assert not np.array_equal(before, after["open_set"])
    assert not env.action_masks()["open_set"][CAction.CHECK_OPENMAX]
    actions = wait_actions(); actions["open_set"] = int(CAction.ACCEPT_CURRENT_KNOWN)
    _, reward, terminated, truncated, info = env.step(actions)
    assert terminated and not truncated and reward > 0
    assert info["prediction"] == 1


def test_ask_c_receives_delayed_structured_tool_response():
    env = OSSEIMultiAgentEnv()
    env.reset(snapshot())
    actions = wait_actions(); actions["identity"] = int(AAction.PROPOSE_TOP1)
    env.step(actions)
    actions = wait_actions(); actions["identity"] = int(AAction.ASK_C)
    env.step(actions)
    assert "identity" in env.blackboard.pending_queries["open_set"]
    actions = wait_actions(); actions["open_set"] = int(CAction.CHECK_OPENMAX)
    _, _, _, _, info = env.step(actions)
    assert "identity" not in env.blackboard.pending_queries["open_set"]
    response = [message for message in info["messages"]
                if message["sender"] == "open_set"]
    assert response and response[0]["message_type"] == "EVIDENCE"


def test_same_round_query_cannot_be_consumed_by_simultaneous_tool_action():
    env = OSSEIMultiAgentEnv()
    env.reset(snapshot())
    actions = wait_actions(); actions["identity"] = int(AAction.PROPOSE_TOP1)
    env.step(actions)
    actions = wait_actions()
    actions["identity"] = int(AAction.ASK_C)
    actions["open_set"] = int(CAction.CHECK_OPENMAX)
    _, _, _, _, info = env.step(actions)
    assert "identity" in env.blackboard.pending_queries["open_set"]
    assert not any(message["sender"] == "open_set"
                   for message in info["messages"])


def test_masked_action_is_rejected_and_timeout_is_failure():
    env = OSSEIMultiAgentEnv(EnvConfig(max_steps=2))
    env.reset(snapshot())
    actions = wait_actions(); actions["open_set"] = int(CAction.ACCEPT_CURRENT_KNOWN)
    with pytest.raises(ValueError, match="invalid masked action"):
        env.step(actions)
    env.reset(snapshot())
    env.step(wait_actions())
    _, reward, terminated, truncated, info = env.step(wait_actions())
    assert not terminated and truncated and reward < -1
    assert info["termination_reason"] == "timeout_failure"


def test_formal_unknown_is_fail_closed_for_bank_and_environment():
    formal = snapshot(-1, "formal_unknown", True)
    with pytest.raises(RuntimeError, match="formal"):
        TrainingEpisodeBank([formal])
    with pytest.raises(RuntimeError, match="formal"):
        OSSEIMultiAgentEnv().reset(formal)
    OSSEIMultiAgentEnv(allow_formal_unknown=True).reset(formal)


def test_blackboard_contract_has_no_private_model_payloads():
    names = {field.name.lower() for field in fields(Blackboard)}
    assert not any(token in name for name in names for token in (
        "hidden", "logits", "embedding", "raw_iq", "prototype_table"))
