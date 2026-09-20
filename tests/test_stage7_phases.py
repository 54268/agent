import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.gates import GateCheck, GateResult  # noqa: E402
from maros_stage7.model import Stage7System  # noqa: E402
from maros_stage7.phases import (  # noqa: E402
    build_g1_gate_metrics,
    build_g2_gate_metrics,
    dataset_direction_rates,
    evaluate_g1_fold,
    evaluate_g2_fold,
    train_g1_responses,
    train_g2_router,
)
from maros_stage7.training import ExactCounterfactualSupervisor  # noqa: E402


class TaggedDataset(Dataset):
    def __init__(self, labels, *, purpose, source_split="train", seed=0,
                 formal_unknown=False):
        self.y = np.asarray(labels, dtype=np.int64)
        generator = np.random.default_rng(seed)
        values = generator.normal(size=(len(self.y), 2, 16)).astype(np.float32)
        known = self.y >= 0
        values[known, 0] += self.y[known, None] * 0.4
        self.x = torch.from_numpy(values)
        self.purpose = purpose
        self.source_split = source_split
        self.formal_unknown = bool(formal_unknown)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], torch.tensor(int(self.y[index]), dtype=torch.long)


def _system(seed=4):
    torch.manual_seed(seed)
    return Stage7System(
        2, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)


def _partitions():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    return {
        "train": TaggedDataset(
            labels, purpose="outer0_inner0_train_known", seed=1),
        "proxy_train": TaggedDataset(
            [-1] * 6, purpose="outer0_inner0_train_proxy_unknown", seed=2),
        "calibration": TaggedDataset(
            labels, purpose="outer0_calibration_known",
            source_split="calibration", seed=3),
        "test": TaggedDataset(
            labels, purpose="outer0_test_known",
            source_split="test_known", seed=4),
        "proxy_test": TaggedDataset(
            [-1] * 6, purpose="outer0_test_proxy_unknown",
            source_split="test_known", seed=5),
    }


def _cfg():
    return {
        "response_epochs": 1,
        "router_epochs": 1,
        "batch_size": 4,
        "response_lr": 1e-3,
        "router_lr": 1e-3,
        "weight_decay": 0.0,
        "proxy_unknown_weight": 1.0,
        "pug_auxiliary_weight": 0.25,
        "communication_cost": 0.0,
        "reference_bits": 64.0,
    }


def _passing_artifact(name):
    return GateResult(
        gate=name, passed=True,
        checks=(GateCheck("unit", True, True, "is", True),),
        measurements={}, claims={})


def test_g1_freezes_local_decision_core_b2_and_router_and_rejects_leakage():
    parts = _partitions()
    system = _system()
    prerequisites = {"g0": _passing_artifact("g0")}
    with pytest.raises(RuntimeError, match="prerequisite"):
        train_g1_responses(
            _system(), parts["train"], _cfg(), torch.device("cpu"), 10,
            prerequisites=None,
            train_proxy_unknown=parts["proxy_train"])
    result = train_g1_responses(
        system, parts["train"], _cfg(), torch.device("cpu"), 11,
        prerequisites=prerequisites,
        train_proxy_unknown=parts["proxy_train"])
    assert result["audit"]["local_decision_core_identical"]
    assert result["audit"]["b2_identical"]
    assert result["audit"]["router_identical"]
    assert result["history"][0]["w_first"] >= 0
    assert result["history"][0]["p_first"] >= 0
    assert all(
        name.startswith((
            "agents.waveform.class_descriptor",
            "agents.waveform.pair_query",
            "agents.waveform.pair_residual",
            "agents.waveform.tail_head",
            "dialogue.adjudicator",
        ))
        for name in result["audit"]["trainable_parameters"])

    outer_proxy = TaggedDataset(
        [-1, -1], purpose="outer0_test_proxy_unknown",
        source_split="test_known", seed=8)
    with pytest.raises(RuntimeError, match="test/proxy-test"):
        train_g1_responses(
            _system(), parts["train"], _cfg(), torch.device("cpu"), 12,
            prerequisites=prerequisites,
            train_proxy_unknown=outer_proxy)
    wrong_inner = TaggedDataset(
        [-1, -1], purpose="outer0_inner1_train_proxy_unknown", seed=9)
    with pytest.raises(RuntimeError, match="different inner episodes"):
        train_g1_responses(
            _system(), parts["train"], _cfg(), torch.device("cpu"), 12,
            prerequisites=prerequisites,
            train_proxy_unknown=wrong_inner)
    formal = TaggedDataset(
        [-1, -1], purpose="formal_unknown", source_split="test_unknown",
        formal_unknown=True, seed=10)
    with pytest.raises(RuntimeError, match="formal unknown"):
        train_g1_responses(
            _system(), parts["train"], _cfg(), torch.device("cpu"), 12,
            prerequisites=prerequisites,
            train_proxy_unknown=formal)


def test_g1_fold_has_independent_rules_equal_bits_and_gate_schema():
    parts = _partitions()
    system = _system()
    prerequisites = {"g0": _passing_artifact("g0")}
    training = train_g1_responses(
        system, parts["train"], _cfg(), torch.device("cpu"), 13,
        prerequisites=prerequisites,
        train_proxy_unknown=parts["proxy_train"])
    fold = evaluate_g1_fold(
        system, parts["calibration"], parts["test"], parts["proxy_test"],
        torch.device("cpu"), fold_index=0,
        training_audit=training["audit"], prerequisites=prerequisites,
        batch_size=32)
    assert fold["parallel_equal_bandwidth"]
    assert fold["frozen_agents_identical"]
    assert {"b2", "w_first", "p_first", "sequential", "parallel"}.issubset(
        fold["rules"])
    assert set(fold["ablation_h_drop"]) == {
        "messages_off", "messages_shuffled", "cross_sample_messages",
        "wrong_class_pair", "drop_challenger",
    }
    # Every rule owns a distinct Known-only calibration artifact.
    assert all("global_reference" in row["calibrator"]
               for row in fold["rules"].values())
    assert fold["rules"]["sequential"]["transcript"]["message_active"].all()
    assert not fold["rules"]["b2"]["transcript"]["message_active"].any()
    reports = []
    for index in range(5):
        row = copy.deepcopy(fold)
        row["fold_index"] = index
        reports.append(row)
    metrics = build_g1_gate_metrics(reports)
    assert metrics["b2_h_score"].shape == (5,)
    assert metrics["sequential_oscr"].shape == (5,)
    assert metrics["receiver_gain_ci_low"].shape == (1,)
    assert all(value.shape == (5,)
               for value in metrics["ablation_h_drop"].values())


def test_g2_requires_passing_artifacts_and_trains_only_router_from_exact_labels():
    parts = _partitions()
    with pytest.raises(RuntimeError, match="prerequisite"):
        train_g2_router(
            _system(), parts["train"], parts["proxy_train"], _cfg(),
            torch.device("cpu"), 14, prerequisites=None)

    system = _system()
    prerequisites = {
        "g0": _passing_artifact("g0"),
        "g1": _passing_artifact("g1"),
    }
    before_router = {
        name: value.detach().clone()
        for name, value in system.router.state_dict().items()
    }
    result = train_g2_router(
        system, parts["train"], parts["proxy_train"], _cfg(),
        torch.device("cpu"), 15, prerequisites=prerequisites)
    assert result["audit"]["exact_action_enumeration"]
    assert result["audit"]["non_router_state_identical"]
    assert all(name.startswith("router.")
               for name in result["audit"]["trainable_parameters"])
    assert any(not torch.equal(value, before_router[name])
               for name, value in system.router.state_dict().items())


def test_g2_fold_exposes_exact_diagnostics_and_gate_schema():
    parts = _partitions()
    system = _system()
    prerequisites = {
        "g0": _passing_artifact("g0"),
        "g1": _passing_artifact("g1"),
    }
    train_g2_router(
        system, parts["train"], parts["proxy_train"], _cfg(),
        torch.device("cpu"), 16, prerequisites=prerequisites)
    supervisor = ExactCounterfactualSupervisor(communication_cost=0.0)
    fold = evaluate_g2_fold(
        system, parts["calibration"], parts["test"], parts["proxy_test"],
        torch.device("cpu"), fold_index=0, supervisor=supervisor,
        prerequisites=prerequisites,
        batch_size=32)
    assert fold["counterfactual"]["action_losses"].shape == (
        len(parts["test"]) + len(parts["proxy_test"]), 3)
    assert 0 <= fold["diagnostics"]["query_rate"] <= 1
    reports = []
    for index in range(5):
        row = copy.deepcopy(fold)
        row["fold_index"] = index
        reports.append(row)
    rates = dataset_direction_rates(reports)
    metrics = build_g2_gate_metrics(
        reports,
        direction_rates_by_dataset={"wisig": rates, "oracle": rates})
    assert metrics["query_rate"].shape == (5,)
    assert metrics["oracle_gain_recovery"].shape == (5,)
    assert metrics["exact_action_enumeration"] is True
    assert set(metrics["direction_rates_by_dataset"]) == {"wisig", "oracle"}
