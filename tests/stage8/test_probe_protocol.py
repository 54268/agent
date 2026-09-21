import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.evaluation import open_set_rescue_metrics  # noqa: E402
from maros_stage8.gates import evaluate_g1_gate, evaluate_g15_gate  # noqa: E402


def _runner_module():
    path = ROOT / "scripts" / "experiments" / "run_stage8_probe.py"
    spec = importlib.util.spec_from_file_location("run_stage8_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_probe_configs_have_identical_method_fields():
    runner = _runner_module()
    wisig = runner.resolve_config(
        "configs/experiments/stage8_g1_probe_wisig_k40u20.json")
    runner.assert_paired_method_config(wisig)


def test_information_isolated_v2_configs_are_paired_without_dataset_branch():
    runner = _runner_module()
    wisig = runner.resolve_config(
        "configs/experiments/stage8_g1_isolated_v2_wisig_k40u20.json")
    oracle = runner.resolve_config(
        "configs/experiments/stage8_g1_isolated_v2_oracle_k10u6.json")
    runner.assert_paired_method_config(wisig)
    runner.assert_paired_method_config(oracle)
    assert runner.method_config(wisig) == runner.method_config(oracle)
    assert wisig["observation_profile"] == "isolated_v2"


def test_agent_b_v3_configs_freeze_temporal_and_pair_datasets():
    runner = _runner_module()
    oracle = runner.resolve_config(
        "configs/experiments/stage8_agent_b_v3_oracle_k10u6.json")
    wisig = runner.resolve_config(
        "configs/experiments/stage8_agent_b_v3_wisig_k40u20.json")
    runner.assert_paired_method_config(oracle)
    assert runner.method_config(oracle) == runner.method_config(wisig)
    assert oracle["temporal_observation_profile"] == "legacy_v1"
    assert oracle["spectral_observation_profile"] == "robust_v3"
    assert oracle["epochs"] == 30
    assert oracle["state_dim"] == 64
    assert oracle["hidden_channels"] == 32


def test_paired_guard_rejects_dataset_specific_method_branch(tmp_path):
    runner = _runner_module()
    wisig = runner.resolve_config(
        "configs/experiments/stage8_g1_probe_wisig_k40u20.json")
    oracle = runner.resolve_config(
        "configs/experiments/stage8_g1_probe_oracle_k10u6.json")
    oracle["epochs"] = oracle["epochs"] + 1
    oracle.pop("config_path", None)
    peer = tmp_path / "changed.json"
    import json
    peer.write_text(json.dumps(oracle), encoding="utf-8")
    wisig["paired_config"] = str(peer)
    with pytest.raises(ValueError, match="epochs"):
        runner.assert_paired_method_config(wisig)


def test_open_set_oracle_headroom_and_fail_closed_gates():
    labels = np.asarray([0, 1, 0, 1, -1, -1, -1, -1])
    temporal = np.asarray([0, 0, 0, 1, -1, 0, -1, 0])
    spectral = np.asarray([1, 1, 0, 0, 0, -1, 0, -1])
    metrics = open_set_rescue_metrics(labels, temporal, spectral)
    assert metrics.temporal_rescues_spectral > 0
    assert metrics.spectral_rescues_temporal > 0
    assert metrics.oracle_h > metrics.best_single_h
    assert not evaluate_g1_gate([metrics]).passed
    assert not evaluate_g15_gate([0.64, 0.66, 0.50, 0.51]).passed
