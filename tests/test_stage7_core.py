import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.contracts import (  # noqa: E402
    ChallengePacket,
    ChallengeStance,
    IndependentSummaryPacket,
    LocalDecision,
    ProposalPacket,
    RouteAction,
)
from maros_stage7.agents import _make_competence_head  # noqa: E402
from maros_stage7.dialogue import ConditionalDialogue, FairB2Fusion  # noqa: E402
from maros_stage7.model import Stage7System  # noqa: E402


def _system(seed: int = 12, classes: int = 5) -> Stage7System:
    torch.manual_seed(seed)
    return Stage7System(
        classes, state_dim=16, stem_channels=16, class_embed_dim=4,
        complex_hidden=4, hidden_dim=16, scalar_bits=16).eval()


def _iq(batch: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(99)
    return torch.randn(batch, 2, 32, generator=generator)


def _public_decisions(batch: int = 3, classes: int = 4):
    waveform_logits = torch.tensor(
        [[3.0, 1.0, 0.0, -1.0]] * batch)
    prototype_logits = torch.tensor(
        [[2.0, 1.5, 0.0, -1.0]] * batch)

    def decision(logits):
        probability = logits.softmax(1)
        top = probability.topk(2, dim=1)
        margin = top.values[:, 0] - top.values[:, 1]
        summary = torch.stack([
            top.values[:, 0], top.values[:, 1], margin,
            torch.full((batch,), 0.2), torch.zeros(batch),
            torch.full((batch,), 0.4), torch.full((batch,), 0.8),
        ], dim=1)
        return LocalDecision(
            class_logits=logits,
            unknown_score=torch.zeros(batch),
            reliability=torch.full((batch,), 0.8),
            top1=top.indices[:, 0], top2=top.indices[:, 1],
            public_summary=summary)

    return {
        "waveform": decision(waveform_logits),
        "prototype": decision(prototype_logits),
    }


def test_private_state_requires_capability_and_parallel_needs_no_private_state():
    system = _system()
    local = system.local(_iq())
    public = local.public_dict()

    with pytest.raises(PermissionError, match="capability"):
        local._private_for("waveform", object())
    with pytest.raises(PermissionError, match="capability-bearing"):
        system.deliberate(public, route_action=RouteAction.W_FIRST)

    # The fair parallel control is independently generated from public local
    # decisions, so it must remain executable after private state is removed.
    parallel = system.parallel(public, route_action=RouteAction.W_FIRST)
    assert parallel.communicated.all()
    assert any(isinstance(packet, IndependentSummaryPacket)
               for packet in parallel.transcript)


def test_stop_is_bit_exact_b2_and_every_class_fusion_is_convex():
    system = _system()
    local = system.local(_iq())
    baseline = system.b2(local)
    stopped = system.deliberate(local, route_action=RouteAction.STOP)

    assert torch.equal(stopped.fused_class_logits, baseline.fused_class_logits)
    assert torch.equal(stopped.unknown_score, baseline.unknown_score)
    assert torch.equal(stopped.known_weights, baseline.known_weights)
    assert torch.equal(stopped.bit_cost, torch.zeros_like(stopped.bit_cost))

    actions = torch.tensor([0, 1, 2, 1, 2, 0])
    for output in (
            system.deliberate(local, route_action=actions),
            system.parallel(local, route_action=actions)):
        assert torch.all(output.known_weights >= 0)
        assert torch.allclose(
            output.known_weights.sum(1), torch.ones(6), atol=1e-7)
        expected = (
            output.known_weights[:, :1] * local["waveform"].class_logits
            + output.known_weights[:, 1:] * local["prototype"].class_logits)
        assert torch.allclose(output.fused_class_logits, expected, atol=1e-7)


def test_fair_b2_unknown_is_convex_and_anchor_fallback_is_exact():
    decisions = _public_decisions()
    decisions["waveform"] = replace(
        decisions["waveform"], unknown_score=torch.tensor([-3.0, -1.0, 2.0]))
    decisions["prototype"] = replace(
        decisions["prototype"], unknown_score=torch.tensor([4.0, 2.0, -2.0]))
    fusion = FairB2Fusion(public_summary_dim=7, hidden_dim=16)
    fusion.set_anchor("waveform")

    _, adaptive_unknown, _ = fusion(decisions)
    local_unknown = torch.stack([
        decisions["waveform"].unknown_score,
        decisions["prototype"].unknown_score,
    ], dim=1)
    assert torch.all(adaptive_unknown >= local_unknown.min(1).values)
    assert torch.all(adaptive_unknown <= local_unknown.max(1).values)

    fusion.set_adaptive_enabled(False)
    logits, unknown, weights = fusion(decisions)
    assert fusion.anchor_name == "waveform"
    assert torch.equal(logits, decisions["waveform"].class_logits)
    assert torch.equal(unknown, decisions["waveform"].unknown_score)
    assert torch.equal(weights, torch.tensor([[1.0, 0.0]]).expand(3, -1))


def test_fair_b2_can_be_configured_for_real_expert_switching():
    decisions = _public_decisions()
    fusion = FairB2Fusion(
        public_summary_dim=7, hidden_dim=16,
        max_adaptive_mass=1.0, initial_adaptive_fraction=0.9)
    fusion.set_anchor("prototype")
    with torch.no_grad():
        fusion.weight_head[-1].weight.zero_()
        fusion.weight_head[-1].bias.copy_(torch.tensor([10.0, -10.0]))
    class_weights, open_weights = fusion.mixture_weights(decisions)

    assert fusion.max_adaptive_mass == 1.0
    assert fusion.initial_adaptive_fraction == 0.9
    assert torch.allclose(class_weights.sum(1), torch.ones(3), atol=1e-7)
    assert torch.allclose(open_weights.sum(1), torch.ones(3), atol=1e-7)
    # The old 0.35 cap guaranteed at least 65% anchor mass.  A 0.9 initial
    # adaptive fraction leaves enough action range for a genuine W/P switch.
    assert torch.all(class_weights[:, 1] < 0.65)

    with pytest.raises(ValueError, match="max_adaptive_mass"):
        FairB2Fusion(max_adaptive_mass=1.01)
    with pytest.raises(ValueError, match="initial_adaptive_fraction"):
        FairB2Fusion(initial_adaptive_fraction=1.0)


def test_competence_reliability_is_private_and_decoupled_from_unknown_risk():
    system = _system()
    x = _iq()
    legacy = system.local(x)
    for role in ("waveform", "prototype"):
        assert torch.allclose(
            legacy[role].reliability,
            torch.sigmoid(-legacy[role].unknown_score), atol=1e-7)

    system.agents.set_competence_enabled(True)
    with torch.no_grad():
        system.agents.waveform.competence_head[-1].bias.fill_(2.0)
        system.agents.prototype.competence_head[-1].bias.fill_(-2.0)
    calibrated = system.local(x)

    assert torch.equal(
        calibrated["waveform"].unknown_score,
        legacy["waveform"].unknown_score)
    assert torch.equal(
        calibrated["prototype"].unknown_score,
        legacy["prototype"].unknown_score)
    assert torch.all(calibrated["waveform"].reliability > 0.8)
    assert torch.all(calibrated["prototype"].reliability < 0.2)


def test_auxiliary_competence_heads_do_not_shift_base_agent_initialisation():
    torch.manual_seed(431)
    expected_next = torch.rand(8)
    torch.manual_seed(431)
    _make_competence_head(20)
    actual_next = torch.rand(8)

    assert torch.equal(actual_next, expected_next)


def test_sparse_packets_hide_inactive_payload_and_parallel_has_equal_bits():
    system = _system()
    local = system.local(_iq())
    actions = torch.tensor([0, 1, 2, 1, 2, 0])
    sequential = system.deliberate(local, route_action=actions)
    parallel = system.parallel(local, route_action=actions)

    assert torch.equal(sequential.bit_cost, parallel.bit_cost)
    assert torch.equal(
        sequential.bit_cost.eq(0), actions.eq(int(RouteAction.STOP)))

    for output in (sequential, parallel):
        for packet in output.transcript:
            inactive = ~packet.active_mask.bool()
            if isinstance(packet, ProposalPacket):
                payload = (
                    packet.top1, packet.top2, packet.log_odds,
                    packet.local_unknown, packet.reliability, packet.bit_cost)
            elif isinstance(packet, ChallengePacket):
                payload = (
                    packet.stance, packet.alternative_class,
                    packet.signed_pair_evidence, packet.open_tail_evidence,
                    packet.reliability, packet.bit_cost)
            else:
                payload = (
                    packet.predicted_class, packet.certainty_bin,
                    packet.confidence_margin, packet.local_unknown,
                    packet.reliability, packet.bit_cost)
            assert all(torch.equal(value[inactive], torch.zeros_like(value[inactive]))
                       for value in payload)


def test_support_lowers_and_challenge_raises_unknown_risk():
    decisions = _public_decisions()
    dialogue = ConditionalDialogue(public_summary_dim=7, hidden_dim=16).eval()
    active = torch.ones(3, dtype=torch.bool)
    proposal = ProposalPacket(
        sender="waveform", top1=torch.zeros(3, dtype=torch.long),
        top2=torch.ones(3, dtype=torch.long), log_odds=torch.ones(3),
        local_unknown=torch.zeros(3), reliability=torch.ones(3),
        active_mask=active, bit_cost=torch.full((3,), 52.0))

    def response(stance, signed):
        return ChallengePacket(
            sender="prototype", receiver="waveform",
            stance=torch.full((3,), int(stance), dtype=torch.long),
            alternative_class=torch.zeros(3, dtype=torch.long),
            signed_pair_evidence=torch.full((3,), signed),
            open_tail_evidence=torch.zeros(3), reliability=torch.ones(3),
            active_mask=active, bit_cost=torch.full((3,), 52.0))

    route = torch.full((3,), int(RouteAction.W_FIRST), dtype=torch.long)
    base = dialogue(decisions, torch.zeros_like(route))
    support = dialogue(
        decisions, route, (proposal,),
        (response(ChallengeStance.SUPPORT, 1.0),))
    challenge = dialogue(
        decisions, route, (proposal,),
        (response(ChallengeStance.CHALLENGE, -1.0),))

    assert torch.all(support.unknown_score < base.unknown_score)
    assert torch.all(challenge.unknown_score > base.unknown_score)


def test_wrong_pair_and_cross_sample_shuffle_have_measurable_effects():
    system = _system()
    local = system.local(_iq())
    route = torch.full((6,), int(RouteAction.W_FIRST), dtype=torch.long)
    clean = system.deliberate(local, route_action=route)
    wrong = system.deliberate(
        local, route_action=route, intervention="wrong_class_pair")
    shuffled = system.deliberate(
        local, route_action=route, intervention="messages_shuffled")

    assert not torch.allclose(clean.unknown_score, wrong.unknown_score)
    assert not torch.allclose(clean.unknown_score, shuffled.unknown_score)
    assert torch.equal(clean.bit_cost, wrong.bit_cost)
    assert torch.equal(clean.bit_cost, shuffled.bit_cost)


def test_protocol_rejects_payload_on_an_inactive_edge():
    system = _system()
    local = system.local(_iq())
    actions = torch.tensor([0, 1, 0, 1, 0, 1])
    valid = system.deliberate(local, route_action=actions)
    proposal = valid.transcript[0]
    response = valid.transcript[1]
    assert isinstance(proposal, ProposalPacket)
    leaked = replace(
        proposal,
        log_odds=torch.where(
            proposal.active_mask, proposal.log_odds,
            torch.ones_like(proposal.log_odds)))
    with pytest.raises(ValueError, match="inactive proposal.log_odds"):
        system.dialogue(local.public_dict(), actions, (leaked,), (response,))


def test_eval_mode_is_exactly_reproducible_for_same_seed():
    first = _system(seed=71)
    second = _system(seed=71)
    x = _iq()
    actions = torch.tensor([0, 1, 2, 1, 2, 0])
    first_output = first(x, route_action=actions)
    second_output = second(x, route_action=actions)

    assert torch.equal(
        first_output.fused_class_logits, second_output.fused_class_logits)
    assert torch.equal(first_output.unknown_score, second_output.unknown_score)
    assert torch.equal(first_output.known_weights, second_output.known_weights)
    assert torch.equal(first_output.bit_cost, second_output.bit_cost)
