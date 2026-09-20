import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage6.contracts import LocalDecision
from maros_stage6.dialogue import SequentialDialogue
from maros_stage6.experiment import _select_candidate
from maros_stage6.model import Stage6System
from maros_stage6.training import _dialogue_objective, _set_agent_trainability


def _system(mode="sequential_cf"):
    return Stage6System(
        4, state_dim=16, stem_channels=32, intent_dim=8,
        message_dim=6, query_dim=7, hidden_dim=32, mode=mode)


def test_three_private_agents_have_independent_local_decisions():
    torch.manual_seed(1)
    model = _system()
    local = model.local(torch.randn(5, 2, 256))
    assert set(local) == {"waveform", "prototype", "verifier"}
    assert len({id(model.agents.waveform.encoder),
                id(model.agents.prototype.spectral_encoder),
                id(model.agents.verifier.diff_encoder)}) == 3
    for decision in local.values():
        assert decision.state.shape == (5, 16)
        assert decision.class_logits.shape == (5, 4)
        assert decision.unknown_logit.shape == (5,)
        assert decision.intent.shape == (5, 8)


def test_private_evidence_reaches_verifier_only_through_messages():
    torch.manual_seed(2)
    model = _system().eval()
    local = model.local(torch.randn(4, 2, 256))
    first_off = model.deliberate(local, hard=True, intervention="messages_off")
    changed = {}
    for role, row in local.items():
        private = {
            key: (torch.randn_like(value) if value.is_floating_point()
                  else torch.zeros_like(value))
            for key, value in row.private_evidence.items()
        }
        changed[role] = LocalDecision(
            row.state, row.class_logits, row.unknown_logit,
            row.reliability, row.intent, private)
    second_off = model.deliberate(changed, hard=True, intervention="messages_off")
    assert torch.equal(first_off.edge_gates, second_off.edge_gates)
    assert torch.allclose(first_off.unknown_logit, second_off.unknown_logit)
    assert torch.allclose(first_off.fused_class_logits, second_off.fused_class_logits)
    first_on = model.deliberate(
        local, hard=True, edge_override=torch.ones(4, 2))
    second_on = model.deliberate(
        changed, hard=True, edge_override=torch.ones(4, 2))
    assert not torch.allclose(first_on.messages[0].payload,
                              second_on.messages[0].payload)


def test_messages_off_removes_cross_agent_gradient_from_unknown_risk():
    torch.manual_seed(3)
    dialogue = SequentialDialogue(
        4, state_dim=8, intent_dim=5, message_dim=4,
        query_dim=6, hidden_dim=24, mode="sequential_cf").eval()

    def decision():
        state = torch.randn(3, 8, requires_grad=True)
        return LocalDecision(
            state, torch.randn(3, 4, requires_grad=True),
            torch.randn(3, requires_grad=True), torch.rand(3),
            torch.randn(3, 5, requires_grad=True), {})

    local = {role: decision() for role in ("waveform", "prototype", "verifier")}
    out = dialogue(local, hard=True, intervention="messages_off")
    assert torch.all(out.edge_gates == 0)
    gradient = torch.autograd.grad(
        out.unknown_logit.sum(), local["waveform"].state,
        allow_unused=True)[0]
    assert gradient is None or torch.allclose(gradient, torch.zeros_like(gradient))


def test_enabled_messages_change_dialogue_but_class_vote_remains_convex():
    torch.manual_seed(4)
    model = _system().eval()
    local = model.local(torch.randn(6, 2, 256))
    on = model.deliberate(local, hard=True,
                          edge_override=torch.ones(6, 2))
    off = model.deliberate(local, hard=True,
                           edge_override=torch.zeros(6, 2))
    assert torch.allclose(on.known_weights.sum(1), torch.ones(6), atol=1e-6)
    assert torch.all(on.known_weights >= 0)
    assert not torch.allclose(on.unknown_logit, off.unknown_logit)
    assert torch.all(on.unknown_logit >= off.unknown_logit)
    expected = (on.known_weights[:, :1] * local["waveform"].class_logits
                + on.known_weights[:, 1:] * local["prototype"].class_logits)
    assert torch.allclose(on.fused_class_logits, expected, atol=1e-6)
    for packet in on.messages:
        assert torch.allclose(packet.payload[:, :3].sum(1), torch.ones(6), atol=1e-6)


def test_no_comm_uses_public_late_fusion_not_private_states():
    torch.manual_seed(5)
    dialogue = SequentialDialogue(
        4, state_dim=8, intent_dim=5, message_dim=4,
        query_dim=6, hidden_dim=24, mode="no_comm").eval()

    def decision():
        return LocalDecision(
            torch.randn(3, 8), torch.randn(3, 4), torch.randn(3),
            torch.rand(3), torch.randn(3, 5), {"secret": torch.randn(3, 9)})

    local = {role: decision() for role in ("waveform", "prototype", "verifier")}
    first = dialogue(local, hard=True)
    hidden_changed = {
        role: LocalDecision(row.state, row.class_logits, row.unknown_logit,
                            row.reliability, row.intent,
                            {"secret": torch.randn_like(row.private_evidence["secret"])})
        for role, row in local.items()
    }
    assert torch.allclose(first.unknown_logit,
                          dialogue(hidden_changed, hard=True).unknown_logit)
    row = local["waveform"]
    public_changed = dict(local)
    public_changed["waveform"] = LocalDecision(
        row.state, row.class_logits + torch.tensor([0.0, 2.0, -1.0, 0.5]),
        row.unknown_logit + 2.0, row.reliability, row.intent,
        row.private_evidence)
    assert not torch.allclose(first.unknown_logit,
                              dialogue(public_changed, hard=True).unknown_logit)


def test_query_cost_controls_hard_communication_decision():
    torch.manual_seed(6)
    low = SequentialDialogue(
        4, state_dim=8, intent_dim=5, message_dim=4, query_dim=6,
        hidden_dim=24, mode="sequential_cf", query_cost=-1e6)
    high = SequentialDialogue(
        4, state_dim=8, intent_dim=5, message_dim=4, query_dim=6,
        hidden_dim=24, mode="sequential_cf", query_cost=1e6)
    expected_gain = torch.zeros(3, 2)
    assert torch.all(low._gates(expected_gain, hard=True) == 1)
    assert torch.all(high._gates(expected_gain, hard=True) == 0)


def test_communication_objective_trains_exact_silent_fallback():
    torch.manual_seed(7)
    model = _system("sequential")
    local = model.local(torch.randn(4, 2, 256))
    pseudo = {
        role: model.agents.from_states({
            key: value.state + 0.1 for key, value in local.items()
        })[role]
        for role in local
    }
    loss, parts = _dialogue_objective(
        model, local, pseudo, torch.tensor([0, 1, 2, 3]),
        {"silent_loss_weight": 0.5, "communication_regret_weight": 0.5},
        use_counterfactual=False)
    loss.backward()
    assert parts["silent_anchor"] > 0
    gradient = model.dialogue.no_comm_open[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_protocol_training_cannot_rewrite_silent_fusion_parent():
    model = _system("sequential_cf")
    _set_agent_trainability(model, joint=True)
    assert any(parameter.requires_grad
               for parameter in model.agents.waveform.decision.parameters())
    assert all(not parameter.requires_grad
               for parameter in model.dialogue.no_comm_weights.parameters())
    assert all(not parameter.requires_grad
               for parameter in model.dialogue.no_comm_open.parameters())


def test_no_communication_is_a_legal_lco_winner():
    def metrics(h, oscr=0.80, known=0.88):
        return {"team": {"auroc": 0.9, "oscr": oscr,
                         "known_accuracy": known, "unknown_recall": 0.8,
                         "h_score": h},
                "waveform": {"auroc": 0.8, "oscr": 0.7,
                             "known_accuracy": 0.88, "unknown_recall": 0.7,
                             "h_score": 0.78},
                "prototype": {"auroc": 0.8, "oscr": 0.7,
                              "known_accuracy": 0.88, "unknown_recall": 0.7,
                              "h_score": 0.78},
                "verifier": {"auroc": 0.8, "oscr": 0.7,
                             "known_accuracy": 0.88, "unknown_recall": 0.7,
                             "h_score": 0.78}}

    rows = []
    for fold in range(5):
        for candidate in ("b2_no_communication", "c2_sequential",
                          "c3_sequential_cf", "c4_sequential_cf_adversarial"):
            h = 0.90 if candidate == "b2_no_communication" else 0.895
            rows.append({"candidate": candidate, "partition_seed": 2026,
                         "fold": fold, "metrics": metrics(h),
                         "behaviour": {"query_rate": 0.5, "mean_edges": 1.0,
                                       "mean_bits": 32.0}})
    summary = {}
    for candidate in {row["candidate"] for row in rows}:
        chosen = [row for row in rows if row["candidate"] == candidate]
        summary[candidate] = {
            "metrics": {key: {"mean": float(np.mean([
                row["metrics"]["team"][key] for row in chosen]))}
                for key in ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")},
            "role_metrics": {role: chosen[0]["metrics"][role]
                             for role in ("waveform", "prototype", "verifier")},
            "query_rate": 0.5, "mean_bits": 32.0,
        }
    selected = _select_candidate(rows, summary, {})
    assert selected["selected"] == "b2_no_communication"
    assert selected["collaboration_promoted"] is False
