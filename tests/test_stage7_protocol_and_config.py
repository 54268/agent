import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.splits import (  # noqa: E402
    FORMAL_UNKNOWN_PURPOSE,
    ProvenanceSubset,
    assert_no_formal_unknown,
    build_class_folds,
    build_nested_lco_protocol,
    protocol_manifest,
    validate_protocol_isolation,
)
from maros_stage7.experiment import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_config_contract,
    _load_fold_model,
    _system,
    _train_g0_model,
    _write_or_validate_frozen_config,
)
import maros_stage7.experiment as stage7_experiment  # noqa: E402
from scripts.experiments.run_stage7 import (  # noqa: E402
    assert_paired_method_config,
    method_config,
    resolve_config,
)


class TinyIQDataset(Dataset):
    def __init__(self, labels, seed=0):
        self.y = np.asarray(labels, dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(len(self.y), 2, 16)).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.tensor(int(self.y[index]))


def _splits(num_classes=10):
    labels = np.repeat(np.arange(num_classes), 3)
    return SimpleNamespace(
        train=TinyIQDataset(labels, 1),
        val=TinyIQDataset(labels, 2),
        test_known=TinyIQDataset(labels, 3),
        test_unknown=TinyIQDataset(np.full(12, -1), 4),
        num_known=num_classes,
    )


def test_nested_lco_is_class_and_sample_disjoint_and_reproducible():
    first = build_nested_lco_protocol(
        _splits(), outer_folds=5, inner_folds=4, seed=2026)
    second = build_nested_lco_protocol(
        _splits(), outer_folds=5, inner_folds=4, seed=2026)
    validate_protocol_isolation(first)

    first_manifest = protocol_manifest(first)
    second_manifest = protocol_manifest(second)
    assert first_manifest == second_manifest
    assert first_manifest["formal_unknown_used_for_training_selection_or_calibration"] is False
    assert first_manifest["formal_unknown_sample_count"] == 12

    outer_count = np.zeros(10, dtype=np.int64)
    for outer in first.outer_folds:
        known, held = set(outer.known_classes), set(outer.proxy_unknown_classes)
        assert known.isdisjoint(held)
        outer_count[list(held)] += 1
        assert set(outer.train_known.sample_keys).isdisjoint(
            outer.calibration_known.sample_keys)
        assert set(outer.calibration_known.sample_keys).isdisjoint(
            outer.test_known.sample_keys)

        inner_count = {value: 0 for value in known}
        for inner in outer.inner_folds:
            assert set(inner.known_classes).isdisjoint(inner.proxy_unknown_classes)
            assert set(inner.known_classes) | set(inner.proxy_unknown_classes) == known
            for value in inner.proxy_unknown_classes:
                inner_count[value] += 1
            assert np.all(inner.train_proxy_unknown.y == -1)
            assert np.all(inner.validation_proxy_unknown.y == -1)
            assert np.all(inner.test_proxy_unknown.y == -1)
        assert set(inner_count.values()) == {1}
    assert np.all(outer_count == 1)


def test_formal_unknown_guard_fails_closed():
    dataset = TinyIQDataset(np.full(5, -1), 9)
    formal = ProvenanceSubset(
        dataset, np.arange(5), source_split="test_unknown",
        purpose=FORMAL_UNKNOWN_PURPOSE, force_unknown=True, formal_unknown=True)
    with pytest.raises(RuntimeError, match="formal unknown"):
        assert_no_formal_unknown(formal)


def test_balanced_folds_support_non_divisible_class_counts():
    folds = build_class_folds(11, 5, seed=7)
    sizes = [len(fold.heldout_classes) for fold in folds]
    assert max(sizes) - min(sizes) <= 1
    assert sorted(value for fold in folds for value in fold.heldout_classes) == list(range(11))


def test_wisig_and_oracle_configs_have_identical_method_fields():
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_oracle_k10u6.json")
    assert wisig["dataset"] == "wisig"
    assert oracle["dataset"] == "oracle"
    assert method_config(wisig) == method_config(oracle)
    assert_paired_method_config(wisig)
    assert_paired_method_config(oracle)


def test_v4_configs_match_and_freeze_local_pretrain_safety_contract():
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v4_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v4_oracle_k10u6.json")

    assert method_config(wisig) == method_config(oracle)
    assert_paired_method_config(wisig)
    assert_paired_method_config(oracle)
    for cfg in (wisig, oracle):
        assert cfg["sanity_max_per_class"] == 0
        assert cfg["local_pretrain_open_loss_weight"] == 0.0
        assert cfg["pretrain_min_train_accuracy"] == 0.85
        assert cfg["prototype_supcon_weight"] == 0.0


def test_v5_inherited_configs_expand_and_remain_method_identical():
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v5_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v5_oracle_k10u6.json")

    assert wisig["competence_weight"] == 0.25
    assert oracle["competence_weight"] == 0.25
    assert wisig["local_pretrain_epochs"] == 30
    assert method_config(wisig) == method_config(oracle)
    assert_paired_method_config(wisig)
    assert_paired_method_config(oracle)


def test_v6_independent_competence_configs_are_method_identical():
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v6_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v6_oracle_k10u6.json")

    assert wisig["competence_mode"] == "independent_head"
    assert oracle["competence_mode"] == "independent_head"
    assert method_config(wisig) == method_config(oracle)
    assert_paired_method_config(wisig)
    assert_paired_method_config(oracle)


def test_v7_open_decision_competence_configs_are_method_identical():
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v7_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_v7_oracle_k10u6.json")

    assert wisig["competence_target_mode"] == "local_decision_success"
    assert oracle["competence_target_mode"] == "local_decision_success"
    assert method_config(wisig) == method_config(oracle)
    assert_paired_method_config(wisig)
    assert_paired_method_config(oracle)


def test_g0_pretrain_capability_gate_stops_before_pug_and_b2(
    tmp_path, monkeypatch,
):
    """An invalid local Agent must fail cheaply and leave an audit artifact."""

    seen = {}

    def fake_train(system, known, proxy, train_cfg, device, seed, **kwargs):
        seen["open_weight"] = train_cfg["open_loss_weight"]
        return ([{"epoch": 1.0}], 1)

    def fake_predictions(system, dataset, device, **kwargs):
        y = np.asarray(dataset.y, dtype=np.int64)
        return {
            "y": y,
            "pred_waveform": y.copy(),
            # Three of six are wrong: deliberately below the 0.85 hard floor.
            "pred_prototype": np.zeros_like(y),
        }

    def forbidden_downstream(*args, **kwargs):
        raise AssertionError("enrollment/PUG/B2 ran after local gate failure")

    monkeypatch.setattr(stage7_experiment, "train_local_agents", fake_train)
    monkeypatch.setattr(stage7_experiment, "collect_predictions", fake_predictions)
    monkeypatch.setattr(
        stage7_experiment, "refresh_enrollment", forbidden_downstream)
    monkeypatch.setattr(stage7_experiment, "make_competition_pug", forbidden_downstream)
    monkeypatch.setattr(stage7_experiment, "train_fair_b2", forbidden_downstream)

    episode = SimpleNamespace(
        train_known=TinyIQDataset([0, 0, 0, 1, 1, 1], seed=21),
        known_classes=(0, 1),
        proxy_unknown_classes=(2,),
    )
    cfg = {
        "dataset": "oracle",
        "method_profile": "stage7_g0_v4_gate_test",
        "state_dim": 8,
        "stem_channels": 8,
        "class_embed_dim": 4,
        "complex_hidden": 4,
        "hidden_dim": 8,
        "local_pretrain_open_loss_weight": 0.0,
        "pretrain_min_train_accuracy": 0.85,
        "evaluation_batch_size": 4,
    }
    run_dir = tmp_path / "fold0"

    with pytest.raises(RuntimeError, match="prototype=0.5000"):
        _train_g0_model(
            cfg, episode, torch.device("cpu"), 42, run_dir, sanity=True)

    assert seen["open_weight"] == 0.0
    audit = json.loads(
        (run_dir / "pretrain_capability.json").read_text(encoding="utf-8"))
    assert audit["passed"] is False
    assert audit["eval_mode_train_accuracy"] == {
        "waveform": 1.0, "prototype": 0.5}
    failure = torch.load(
        run_dir / "local_pretrain_failed.pt", map_location="cpu",
        weights_only=False)
    assert failure["status"] == "failed_local_capability_gate"
    assert failure["config_contract"] == _checkpoint_config_contract(cfg)


def test_paired_config_guard_rejects_dataset_specific_method_changes(tmp_path):
    wisig = resolve_config(
        ROOT / "configs/experiments/stage7_screen_wisig_k40u20.json")
    oracle = resolve_config(
        ROOT / "configs/experiments/stage7_screen_oracle_k10u6.json")
    oracle.pop("config_path", None)
    oracle["communication_cost"] = 0.02
    peer = tmp_path / "oracle_changed.json"
    peer.write_text(json.dumps(oracle), encoding="utf-8")
    wisig["paired_config"] = str(peer)
    with pytest.raises(ValueError, match="communication_cost"):
        assert_paired_method_config(wisig)


def test_frozen_config_cannot_be_silently_overwritten(tmp_path):
    path = tmp_path / "frozen_config.json"
    cfg = {"dataset": "wisig", "state_dim": 8, "output_dir": str(tmp_path)}
    _write_or_validate_frozen_config(path, cfg)
    _write_or_validate_frozen_config(path, dict(cfg))
    with pytest.raises(ValueError, match="state_dim"):
        _write_or_validate_frozen_config(path, {**cfg, "state_dim": 16})


def test_checkpoint_resume_requires_exact_config_and_both_class_partitions(tmp_path):
    cfg = {
        "dataset": "wisig", "wisig_pkl": "dummy.pkl",
        "method_profile": "stage7_conditional_consultation_g0_v2",
        "state_dim": 8, "stem_channels": 8, "class_embed_dim": 4,
        "complex_hidden": 4, "hidden_dim": 8,
        "prototype_temperature": 0.15,
    }
    episode = SimpleNamespace(
        known_classes=(0, 1), proxy_unknown_classes=(2,))
    model = _system(cfg, 2)
    path = tmp_path / "g0_model.pt"
    torch.save({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_contract": _checkpoint_config_contract(cfg),
        "known_classes": list(episode.known_classes),
        "proxy_unknown_classes": list(episode.proxy_unknown_classes),
        "model": model.state_dict(),
    }, path)

    restored = _load_fold_model(path, cfg, episode, torch.device("cpu"))
    assert restored.num_classes == 2
    with pytest.raises(ValueError, match="state_dim"):
        _load_fold_model(
            path, {**cfg, "state_dim": 16}, episode, torch.device("cpu"))
    wrong_proxy = SimpleNamespace(
        known_classes=(0, 1), proxy_unknown_classes=(3,))
    with pytest.raises(ValueError, match="proxy partition"):
        _load_fold_model(path, cfg, wrong_proxy, torch.device("cpu"))
