import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.agents import (RoleStructuredExperts, specialist_view,
                                 spectral_view, universal_specialist_views)
from maros_stage5.calibration_agent import (CalibrationAgent,
                                            ClassConditionalThresholdAgent,
                                            DecisionArbitratorAgent)
from maros_stage5.coordinator import CommunicationDecisionCoordinator, EDGE_NAMES
from maros_stage5.protocol import build_lco_folds
from maros_stage5.pseudo_unknown import BoundaryExplorerAgent, PUGConfig, generate_pug_v2
from maros_stage5.training import ExpertRecords, set_seed
from maros_staged.datasets import load_oracle_npz


def test_lco_folds_are_disjoint_and_cover_every_class_once():
    folds = build_lco_folds(40, 5, 2026)
    counts = np.zeros(40, dtype=int)
    for fold in folds:
        assert not set(fold.support_classes) & set(fold.heldout_classes)
        assert set(fold.support_classes) | set(fold.heldout_classes) == set(range(40))
        counts[list(fold.heldout_classes)] += 1
    assert np.all(counts == 1)


def test_agent_contract_and_spectral_view():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 256)
    model = RoleStructuredExperts(5, state_dim=16, stem_channels=32, message_dim=8)
    outputs = model(x)
    assert spectral_view(x).shape == x.shape
    for mode in ("raw", "spectral", "fft_iq", "envelope_phase", "difference_iq"):
        assert specialist_view(x, mode).shape == x.shape
    assert len(universal_specialist_views(x)) == 4
    reconstruction = model.reconstruct(outputs)
    assert reconstruction.shape == (4, 2, 64)
    reconstruction.square().mean().backward()
    assert model.reconstruction_head[-1].weight.grad is not None
    for name in ("identity", "geometry"):
        out = outputs[name]
        assert out.state.shape == (4, 16)
        assert out.class_logits.shape == (4, 5)
        assert out.message.shape == (4, 8)
        assert torch.all((out.reliability >= 0) & (out.reliability <= 1))


def test_unified_geometry_uses_a_convex_sample_level_view_gate():
    torch.manual_seed(7)
    x = torch.randn(5, 2, 256)
    model = RoleStructuredExperts(4, state_dim=16, stem_channels=32,
                                  message_dim=8, geometry_view="universal")
    geometry = model(x)["geometry"]
    weights = geometry.evidence["view_weights"]
    assert weights.shape == (5, 5)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(1), torch.ones(5), atol=1e-6)


def test_coordinator_class_action_is_a_convex_fusion_when_messages_off():
    torch.manual_seed(1)
    batch, classes, dim, evidence_dim = 6, 4, 8, 7
    coordinator = CommunicationDecisionCoordinator(classes, dim, evidence_dim,
                                                    message_dim=6, mode="none")
    li, lg = torch.randn(batch, classes), torch.randn(batch, classes)
    out = coordinator(torch.randn(batch, dim), torch.randn(batch, dim), li, lg,
                      torch.randn(batch, evidence_dim), hard=True)
    expected = out.known_weights[:, :1] * li + out.known_weights[:, 1:] * lg
    assert torch.allclose(out.fused_class_logits, expected, atol=1e-6)
    assert torch.allclose(out.edge_gates, torch.zeros_like(out.edge_gates))
    assert torch.allclose(out.known_weights.sum(1), torch.ones(batch), atol=1e-6)


def test_hard_sparse_routing_keeps_at_least_one_boundary_edge():
    torch.manual_seed(2)
    coordinator = CommunicationDecisionCoordinator(4, 8, 7, message_dim=6, mode="sparse")
    out = coordinator(torch.randn(5, 8), torch.randn(5, 8), torch.randn(5, 4),
                      torch.randn(5, 4), torch.randn(5, 7), hard=True)
    assert torch.all(out.edge_gates[:, 2:4].sum(1) >= 1)
    assert set(torch.unique(out.edge_gates).tolist()).issubset({0.0, 1.0})


def test_audit_only_routing_never_changes_specialist_states_or_logits():
    torch.manual_seed(3)
    coordinator = CommunicationDecisionCoordinator(4, 8, 7, message_dim=6,
                                                    mode="audit_sparse_cf",
                                                    budget_target=0.25)
    zi, zg = torch.randn(5, 8), torch.randn(5, 8)
    li, lg = torch.randn(5, 4), torch.randn(5, 4)
    out = coordinator(zi, zg, li, lg, torch.randn(5, 7), hard=True)
    assert torch.all(out.edge_gates[:, :2] == 0)
    assert torch.all(out.edge_gates[:, 2:].sum(1) >= 1)
    assert torch.allclose(out.agent_outputs["identity"].state, zi)
    assert torch.allclose(out.agent_outputs["geometry"].state, zg)


def test_optional_audit_routing_can_remain_silent():
    torch.manual_seed(4)
    coordinator = CommunicationDecisionCoordinator(4, 8, 7, message_dim=6,
                                                    mode="audit_optional",
                                                    budget_target=0.15)
    with torch.no_grad():
        for head in coordinator.gate_heads:
            head[-1].weight.zero_()
            head[-1].bias.fill_(-10.0)
    out = coordinator(torch.randn(5, 8), torch.randn(5, 8), torch.randn(5, 4),
                      torch.randn(5, 4), torch.randn(5, 7), hard=True)
    assert torch.all(out.edge_gates == 0)


def test_residual_audit_messages_can_only_increase_unknown_logit():
    torch.manual_seed(5)
    coordinator = CommunicationDecisionCoordinator(4, 8, 7, message_dim=6,
                                                    mode="audit_residual",
                                                    budget_target=0.1)
    inputs = (torch.randn(7, 8), torch.randn(7, 8), torch.randn(7, 4),
              torch.randn(7, 4), torch.randn(7, 7))
    off = coordinator(*inputs, hard=True,
                      edge_override=torch.zeros(7, len(EDGE_NAMES)))
    on = coordinator(*inputs, hard=True,
                     edge_override=torch.tensor([[0., 0., 1., 0.]]).expand(7, -1))
    assert torch.all(on.unknown_logit >= off.unknown_logit)
    assert torch.allclose(on.agent_outputs["identity"].class_logits, inputs[2])
    assert torch.allclose(on.agent_outputs["geometry"].class_logits, inputs[3])


def test_same_seed_recreates_identical_agent_parameters_and_output():
    x = torch.randn(3, 2, 256)
    set_seed(123)
    first = RoleStructuredExperts(4, state_dim=16, stem_channels=32, message_dim=8)
    first_output = first(x)["identity"].class_logits.detach().clone()
    first_state = {key: value.detach().clone() for key, value in first.state_dict().items()}
    set_seed(123)
    second = RoleStructuredExperts(4, state_dim=16, stem_channels=32, message_dim=8)
    second_output = second(x)["identity"].class_logits.detach().clone()
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first_state.items())
    assert torch.equal(first_output, second_output)


def test_pug_v2_is_balanced_and_preserves_agent_norms():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 12)
    zi = rng.normal(size=(36, 8)).astype(np.float32)
    zg = rng.normal(size=(36, 8)).astype(np.float32)
    pred_i = labels.copy(); pred_g = labels.copy(); pred_g[::5] = (pred_g[::5] + 1) % 3
    pseudo = generate_pug_v2(zi, zg, labels, pred_i, pred_g,
                             rng.normal(size=(36, 7)).astype(np.float32),
                             PUGConfig(kind="mixed", eta=1.5, seed_ratio=0.2,
                                       variations=2, seed=5))
    assert len(pseudo.identity_state) == len(pseudo.geometry_state) > 0
    assert np.allclose(np.linalg.norm(pseudo.identity_state, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pseudo.geometry_state, axis=1), 1.0, atol=1e-5)
    unique, counts = np.unique(pseudo.kinds, return_counts=True)
    assert set(unique) == {"competition", "disagreement"}
    assert counts[0] == counts[1]


def test_boundary_explorer_and_calibration_are_independent_agents():
    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(3), 10)
    zi = rng.normal(size=(30, 8)).astype(np.float32)
    zg = rng.normal(size=(30, 8)).astype(np.float32)
    explorer = BoundaryExplorerAgent(PUGConfig(kind="competition", variations=1, seed=9))
    decision = explorer.act(zi, zg, labels, labels, labels,
                            rng.normal(size=(30, 7)).astype(np.float32))
    assert len(decision.proposals.source_indices) > 0
    assert 0.0 < decision.reliability <= 1.0

    records = ExpertRecords(zi, zg, rng.normal(size=(30, 3)).astype(np.float32),
                            rng.normal(size=(30, 3)).astype(np.float32),
                            rng.normal(size=(30, 7)).astype(np.float32), labels)
    prediction = {"score": rng.normal(size=30).astype(np.float32)}
    calibration = CalibrationAgent("mean4").fit(records, prediction)
    output = calibration.transform(records, prediction)
    assert output.unknown_score.shape == (30,)
    assert set(output.component_scores) == {"identity", "prototype", "openmax", "boundary"}
    assert np.all((output.unknown_score >= 0) & (output.unknown_score <= 1))


def test_decision_arbitrator_uses_known_only_branches_and_max_risk():
    rng = np.random.default_rng(10)
    labels = np.repeat(np.arange(3), 12)
    records = ExpertRecords(
        rng.normal(size=(36, 8)).astype(np.float32),
        rng.normal(size=(36, 8)).astype(np.float32),
        rng.normal(size=(36, 3)).astype(np.float32),
        rng.normal(size=(36, 3)).astype(np.float32),
        rng.normal(size=(36, 7)).astype(np.float32), labels)
    communicated = {"score": rng.normal(size=36).astype(np.float32)}
    local = {"score": rng.normal(size=36).astype(np.float32)}
    agent = DecisionArbitratorAgent("geo_open_boundary", "max").fit(
        records, communicated, local)
    output = agent.transform(records, communicated, local)
    assert output.unknown_score.shape == (36,)
    assert np.all((output.unknown_score >= 0) & (output.unknown_score <= 1))
    assert output.rule == "max"
    # Before the final monotone known-CDF calibration, max arbitration cannot
    # be lower than either independently calibrated branch.
    fused = agent._fuse(output.communicated_score, output.local_score)
    assert np.all(fused >= output.communicated_score)
    assert np.all(fused >= output.local_score)


def test_class_conditional_thresholds_adapt_and_shrink_to_global():
    scores = np.r_[np.linspace(0.0, 0.4, 20), np.linspace(0.5, 0.9, 20)]
    prediction = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    adaptive = ClassConditionalThresholdAgent(0.9, n_prior=0).fit(scores, prediction)
    shrunk = ClassConditionalThresholdAgent(0.9, n_prior=1000).fit(scores, prediction)
    tau = adaptive.thresholds(np.array([0, 1, 2]))
    shrunk_tau = shrunk.thresholds(np.array([0, 1]))
    assert tau[0] < tau[1]
    assert tau[2] == adaptive.global_threshold
    assert abs(shrunk_tau[0] - shrunk.global_threshold) < abs(tau[0] - adaptive.global_threshold)
    assert abs(shrunk_tau[1] - shrunk.global_threshold) < abs(tau[1] - adaptive.global_threshold)


def test_oracle_loader_recovers_unknown_transmitter_groups(tmp_path):
    rng = np.random.default_rng(12)
    def save(name, count, labels):
        np.savez_compressed(tmp_path / name,
                            x=rng.normal(size=(count, 2, 16)).astype(np.float32),
                            y=np.asarray(labels, dtype=np.int64))
    save("train_known.npz", 4, [0, 0, 1, 1])
    save("val_known.npz", 2, [0, 1])
    save("test_known.npz", 2, [0, 1])
    save("test_unknown.npz", 6, [-1] * 6)
    (tmp_path / "dataset_summary.json").write_text(
        '{"unknown_classes":["7","9"],"max_per_label":3,'
        '"records":[{"label":"7","num_windows":5},'
        '{"label":"9","num_windows":4}]}', encoding="utf-8")
    splits = load_oracle_npz(tmp_path)
    assert splits.num_known == 2
    assert np.array_equal(splits.open_test.group_ids, np.array([0, 1, 2, 2, 2, 3, 3, 3]))
