import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.contracts import RouteAction  # noqa: E402
from maros_stage7.model import Stage7System  # noqa: E402
from maros_stage7.splits import ProvenanceSubset  # noqa: E402
from maros_stage7.training import (  # noqa: E402
    collect_predictions,
    export_b2_transfer_bundle,
    import_b2_transfer_bundle,
    local_agent_objective,
    make_competition_pug,
    refresh_enrollment,
    train_fair_b2,
    train_local_agents,
)


class TaggedIQDataset(Dataset):
    def __init__(self, labels, *, purpose, source_split="train", seed=0,
                 formal_unknown=False):
        self.y = np.asarray(labels, dtype=np.int64)
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(len(self.y), 2, 32)).astype(np.float32)
        known = self.y >= 0
        if known.any():
            x[known, 0] += self.y[known, None] * 0.75
        self.x = torch.from_numpy(x)
        self.purpose = purpose
        self.source_split = source_split
        self.formal_unknown = bool(formal_unknown)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], torch.tensor(int(self.y[index]), dtype=torch.long)


def _system():
    return Stage7System(
        2, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)


def _known(seed=1):
    return TaggedIQDataset(
        [0, 0, 0, 0, 1, 1, 1, 1],
        purpose="outer0_train_known", seed=seed)


def _cfg():
    return {
        "local_pretrain_epochs": 1,
        "b2_epochs": 1,
        "batch_size": 4,
        "local_lr": 1e-3,
        "b2_lr": 1e-3,
        "weight_decay": 0.0,
        "class_loss_weight": 1.0,
        "open_loss_weight": 1.0,
        "pug_auxiliary_weight": 0.25,
        "b2_no_regret_weight": 1.0,
        "b2_selector_distillation_weight": 0.25,
        # Make this unit test exercise the exact empirical fallback branch.
        "b2_required_fit_gain": 1e6,
    }


def test_local_objective_sums_full_per_agent_losses_without_halving():
    """Each independent Agent keeps unit classification/rejection scale."""

    system = _system()
    known = _known()
    x, labels = zip(*[known[index] for index in range(4)])
    decisions = system.local(torch.stack(x))
    labels = torch.stack(labels)

    total, pieces = local_agent_objective(
        decisions, labels, class_weight=1.3, open_weight=0.4)

    assert set(pieces) == {"waveform", "prototype"}
    assert torch.allclose(total, pieces["waveform"] + pieces["prototype"])
    assert not torch.allclose(
        total, (pieces["waveform"] + pieces["prototype"]) / 2.0)


def test_local_competence_calibration_is_independent_and_non_negative():
    system = _system()
    known = _known()
    x, labels = zip(*[known[index] for index in range(4)])
    decisions = system.local(torch.stack(x))
    labels = torch.stack(labels)

    base, _ = local_agent_objective(
        decisions, labels, class_weight=1.0, open_weight=0.0,
        competence_weight=0.0)
    calibrated, pieces = local_agent_objective(
        decisions, labels, class_weight=1.0, open_weight=0.0,
        competence_weight=0.25)

    assert calibrated > base
    assert set(pieces) == {"waveform", "prototype"}
    with pytest.raises(ValueError, match="competence_weight"):
        local_agent_objective(decisions, labels, competence_weight=-0.1)


def test_open_decision_competence_rewards_correct_unknown_rejection():
    system = _system()
    decisions = system.local(torch.stack([_known()[index][0]
                                          for index in range(4)]))
    decisions = {
        name: replace(
            decision,
            unknown_score=torch.full((4,), 3.0),
            reliability=torch.full((4,), 0.95))
        for name, decision in decisions.items()
    }
    labels = torch.full((4,), -1, dtype=torch.long)

    legacy, _ = local_agent_objective(
        decisions, labels, class_weight=0.0, open_weight=0.0,
        competence_weight=1.0, competence_target_mode="identity_or_zero")
    aligned, _ = local_agent_objective(
        decisions, labels, class_weight=0.0, open_weight=0.0,
        competence_weight=1.0,
        competence_target_mode="local_decision_success")

    assert aligned < legacy
    with pytest.raises(ValueError, match="competence_target_mode"):
        local_agent_objective(
            decisions, labels, competence_weight=1.0,
            competence_target_mode="unsupported")


def test_b2_transfer_across_class_counts_changes_only_public_fusion():
    """An inner B2 can move 2->3 classes without moving private services."""

    torch.manual_seed(31)
    source = _system().eval()
    target = Stage7System(
        3, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8).eval()
    with torch.no_grad():
        for index, parameter in enumerate(
                source.dialogue.b2_fusion.parameters(), start=1):
            parameter.fill_(index / 100.0)
    source.dialogue.b2_fusion.set_anchor("prototype")
    source.dialogue.b2_fusion.set_adaptive_enabled(True)

    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    bundle = export_b2_transfer_bundle(source)
    audit = import_b2_transfer_bundle(target, bundle)
    after = target.state_dict()
    changed = {
        name for name, value in after.items()
        if not torch.equal(value, before[name])
    }

    assert changed
    assert all(name.startswith("dialogue.b2_fusion.") for name in changed)
    for prefix in ("agents.", "router.", "dialogue.adjudicator."):
        assert all(torch.equal(value, before[name])
                   for name, value in after.items() if name.startswith(prefix))
    assert audit["source_num_classes"] == 2
    assert audit["target_num_classes"] == 3
    assert audit["class_count_changed"] is True
    assert audit["strict_public_b2_only"] is True
    assert audit["state_sha256"] == bundle["manifest"]["state_sha256"]

    local = target.local(torch.randn(5, 2, 32))
    b2 = target.b2(local)
    stop = target.deliberate(local, route_action=RouteAction.STOP)
    assert torch.equal(stop.fused_class_logits, b2.fused_class_logits)
    assert torch.equal(stop.unknown_score, b2.unknown_score)
    assert torch.equal(stop.known_weights, b2.known_weights)
    assert torch.equal(stop.bit_cost, torch.zeros_like(stop.bit_cost))


@pytest.mark.parametrize("forbidden_key", [
    "agents.waveform.classifier.weight",
    "router.net.0.weight",
    "response_head.weight",
    "private_evidence",
])
def test_b2_transfer_rejects_non_b2_state_before_mutation(forbidden_key):
    source = _system()
    target = Stage7System(
        3, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)
    bundle = export_b2_transfer_bundle(source)
    bundle["state_dict"][forbidden_key] = torch.ones(1)
    before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }

    with pytest.raises(ValueError, match="forbidden"):
        import_b2_transfer_bundle(target, bundle)

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_b2_transfer_rejects_incompatible_fusion_shape_before_mutation():
    source = Stage7System(
        2, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=16)
    target = _system()
    bundle = export_b2_transfer_bundle(source)
    before = {
        name: value.detach().clone()
        for name, value in target.dialogue.b2_fusion.state_dict().items()
    }

    with pytest.raises(ValueError, match="shape/dtype"):
        import_b2_transfer_bundle(target, bundle)

    for name, value in target.dialogue.b2_fusion.state_dict().items():
        assert torch.equal(value, before[name])


def test_local_training_refresh_pug_and_fair_b2_are_composable():
    device = torch.device("cpu")
    system = _system()
    known = _known()
    pug = make_competition_pug(
        system, known, device, eta=1.0, max_per_sample=1,
        batch_size=4, seed=7)
    assert len(pug) == len(known)
    assert np.all(pug.y == -1)
    assert pug.purpose == "competition_pug_train"
    assert torch.isfinite(pug.x).all()

    history, selected_epoch = train_local_agents(
        system, known, None, _cfg(), device, 7, pug_dataset=pug)
    assert selected_epoch == 1
    assert len(history) == 1
    assert set(("waveform_known_accuracy", "prototype_known_accuracy")).issubset(
        history[0])

    enrollment = refresh_enrollment(system, known, device, batch_size=4)
    assert bool(system.agents.prototype.enrollment_ready)
    assert enrollment["num_classes"] == 2.0
    assert enrollment["num_samples"] == float(len(known))

    agent_before = {
        name: value.detach().clone()
        for name, value in system.agents.state_dict().items()
    }
    b2_before = {
        name: value.detach().clone()
        for name, value in system.dialogue.b2_fusion.state_dict().items()
    }
    b2_history = train_fair_b2(
        system, known, None, _cfg(), device, 11, pug_dataset=pug)
    assert len(b2_history) == 1
    assert b2_history[-1]["anchor_agent"] in {"waveform", "prototype"}
    assert b2_history[-1]["adaptive_selected"] == 0.0
    assert b2_history[-1]["no_regret"] >= 0.0
    assert b2_history[-1]["selector_loss"] >= 0.0
    for name, value in system.agents.state_dict().items():
        assert torch.equal(value, agent_before[name])
    assert any(
        not torch.equal(value, b2_before[name])
        for name, value in system.dialogue.b2_fusion.state_dict().items())

    prediction = collect_predictions(
        system, known, device, route_action=RouteAction.STOP, batch_size=3)
    assert prediction["pred"].shape == (len(known),)
    assert prediction["logits_waveform"].shape == (len(known), 2)
    assert prediction["summary_prototype"].shape == (len(known), 7)
    assert prediction["public_features"].shape == (len(known), 17)
    assert np.all(prediction["route"] == int(RouteAction.STOP))
    assert np.all(prediction["bits"] == 0)
    anchor = b2_history[-1]["anchor_agent"]
    assert np.array_equal(prediction["pred"], prediction[f"pred_{anchor}"])
    assert np.array_equal(prediction["raw_unknown"], prediction[f"raw_{anchor}"])


def test_open_head_fit_keeps_frozen_batchnorm_state_exactly_fixed():
    """The PUG rejection phase must not silently mutate frozen encoders."""

    device = torch.device("cpu")
    system = _system()
    known = _known(seed=13)
    for parameter in system.agents.parameters():
        parameter.requires_grad_(False)
    open_heads = (
        system.agents.waveform.open_head,
        system.agents.prototype.open_head,
    )
    for head in open_heads:
        for parameter in head.parameters():
            parameter.requires_grad_(True)

    batchnorms = [
        module for module in system.agents.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batchnorms, "the regression needs a stateful frozen normalizer"
    normalizer_state = [
        (module.running_mean.detach().clone(),
         module.running_var.detach().clone(),
         module.num_batches_tracked.detach().clone())
        for module in batchnorms
    ]
    open_before = [
        parameter.detach().clone()
        for head in open_heads for parameter in head.parameters()
    ]

    train_local_agents(system, known, None, _cfg(), device, 17)

    for module, before in zip(batchnorms, normalizer_state):
        assert module.training is False
        assert torch.equal(module.running_mean, before[0])
        assert torch.equal(module.running_var, before[1])
        assert torch.equal(module.num_batches_tracked, before[2])
    open_after = [
        parameter.detach()
        for head in open_heads for parameter in head.parameters()
    ]
    assert any(not torch.equal(before, after)
               for before, after in zip(open_before, open_after))


def test_enrollment_refresh_restores_mixed_module_modes_exactly():
    system = _system()
    system.agents.eval()
    # Reproduce the mixed mode used during rejection-residual fitting: the
    # frozen Agent tree is in eval mode while only open-head leaves are active.
    system.agents.waveform.open_head.train()
    system.agents.prototype.open_head.train()
    before = {
        name: module.training for name, module in system.agents.named_modules()
    }

    refresh_enrollment(
        system, _known(seed=23), torch.device("cpu"), batch_size=4)

    after = {
        name: module.training for name, module in system.agents.named_modules()
    }
    assert after == before


def test_outer_test_proxy_is_rejected_from_every_training_stage():
    system = _system()
    device = torch.device("cpu")
    known = _known()
    outer_test_proxy = TaggedIQDataset(
        [-1, -1, -1, -1], purpose="outer0_test_proxy_unknown",
        source_split="test_known", seed=4)
    with pytest.raises(RuntimeError, match="test/proxy-test"):
        train_local_agents(system, known, outer_test_proxy, _cfg(), device, 3)
    with pytest.raises(RuntimeError, match="test/proxy-test"):
        train_fair_b2(system, known, outer_test_proxy, _cfg(), device, 3)


def test_only_inner_train_proxy_is_accepted_and_formal_prediction_is_explicit():
    system = _system()
    device = torch.device("cpu")
    known = _known()
    inner_proxy = TaggedIQDataset(
        [-1, -1, -1, -1],
        purpose="outer0_inner0_train_proxy_unknown", seed=5)
    history, _ = train_local_agents(
        system, known, inner_proxy, _cfg(), device, 5)
    assert history[0]["proxy_open_loss"] > 0.0

    formal = TaggedIQDataset(
        [-1, -1], purpose="formal_unknown_evaluation",
        source_split="test_unknown", seed=6, formal_unknown=True)
    with pytest.raises(RuntimeError, match="formal unknown"):
        collect_predictions(system, formal, device)
    allowed = collect_predictions(
        system, formal, device, allow_formal_unknown=True)
    assert np.all(allowed["y"] == -1)


def test_prediction_preserves_registered_subset_and_forced_unknown_labels():
    """Regression: never unwrap a ProvenanceSubset to its full source split."""
    system = _system()
    device = torch.device("cpu")
    source = TaggedIQDataset(
        [0, 0, 1, 1, 0, 1], purpose="source", source_split="test_known",
        seed=9)
    known_subset = ProvenanceSubset(
        source, [0, 1, 4], source_split="test_known",
        purpose="outer0_test_known", class_mapping={0: 0})
    proxy_subset = ProvenanceSubset(
        source, [2, 3, 5], source_split="test_known",
        purpose="outer0_test_proxy_unknown", force_unknown=True)

    known = collect_predictions(system, known_subset, device, batch_size=2)
    proxy = collect_predictions(system, proxy_subset, device, batch_size=2)

    assert known["y"].tolist() == [0, 0, 0]
    assert proxy["y"].tolist() == [-1, -1, -1]
    assert len(known["pred"]) == len(known_subset)
    assert len(proxy["pred"]) == len(proxy_subset)
    assert known["sample_keys"].tolist() == list(known_subset.sample_keys)
    assert proxy["sample_keys"].tolist() == list(proxy_subset.sample_keys)
