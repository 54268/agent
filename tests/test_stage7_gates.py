import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.gates import (  # noqa: E402
    GateResult,
    evaluate_g0,
    evaluate_g1,
    evaluate_g2,
    evaluate_g3,
    load_gate_result,
    render_gate_report,
    require_prerequisite_artifacts,
    save_gate_result,
    select_passing_candidate,
)


def _five(value):
    return np.full(5, value, dtype=np.float64)


def _passing_g0():
    return evaluate_g0({
        "nested_inner_oof_selector": True,
        "outer_test_used_for_selection": False,
        "public_only_b2_transfer": True,
        "formal_unknown_used": False,
        "local_known_accuracy": {
            "waveform": _five(0.88), "prototype": _five(0.86)},
        "local_auroc": {
            "waveform": _five(0.79), "prototype": _five(0.81)},
        "single_h_score": {
            "waveform": _five(0.80), "prototype": _five(0.81)},
        "unique_rescue": {
            "waveform": _five(0.06), "prototype": _five(0.055)},
        "b2_h_score": _five(0.806),
        "b2_oscr": _five(0.80),
        "oracle_h_score": _five(0.84),
        "oracle_oscr": _five(0.82),
        "public_selector_h_score": _five(0.821),
    })


def _passing_g1(g0=None):
    g0 = _passing_g0() if g0 is None else g0
    return evaluate_g1({
        "frozen_agents_identical": True,
        "parallel_equal_bandwidth": True,
        "b2_h_score": _five(0.80),
        "sequential_h_score": _five(0.812),
        "parallel_h_score": _five(0.805),
        "b2_oscr": _five(0.80),
        "sequential_oscr": _five(0.807),
        "parallel_oscr": _five(0.804),
        "b2_known_accuracy": _five(0.90),
        "sequential_known_accuracy": _five(0.897),
        "rescue_harm_ratio": _five(2.5),
        "receiver_gain_ci_low": [0.001],
        "ablation_h_drop": {
            "messages_off": _five(0.012),
            "cross_sample": _five(0.006),
        },
    }, prerequisites={"g0": g0})


def _passing_g2(g0=None, g1=None):
    g0 = _passing_g0() if g0 is None else g0
    g1 = _passing_g1(g0) if g1 is None else g1
    return evaluate_g2({
        "exact_action_enumeration": True,
        "query_rate": _five(0.45),
        "oracle_gain_recovery": _five(0.61),
        "gain_spearman": _five(0.37),
        "rescue_harm_ratio": _five(2.3),
        "direction_rates_by_dataset": {
            "wisig": {"w_first": 0.02, "p_first": 0.43},
            "oracle": {"w_first": 0.04, "p_first": 0.41},
        },
    }, prerequisites={"g0": g0, "g1": g1})


def test_g0_passes_only_with_all_registered_capability_evidence():
    result = _passing_g0()
    assert result.passed
    assert result.measurements["oracle_delta_h_vs_b2"] >= 0.03
    assert result.measurements["public_selector_delta_h_vs_best_single"] >= 0.01

    missing = evaluate_g0({})
    assert not missing.passed
    assert any("missing metric" in reason for reason in missing.failure_reasons)

    weak = evaluate_g0({
        "nested_inner_oof_selector": True,
        "outer_test_used_for_selection": False,
        "public_only_b2_transfer": True,
        "formal_unknown_used": False,
        "local_known_accuracy": {
            "waveform": _five(0.84), "prototype": _five(0.86)},
        "local_auroc": {
            "waveform": _five(0.79), "prototype": _five(0.81)},
        "single_h_score": {
            "waveform": _five(0.80), "prototype": _five(0.81)},
        "unique_rescue": {
            "waveform": _five(0.06), "prototype": _five(0.055)},
        "b2_h_score": _five(0.806), "b2_oscr": _five(0.80),
        "oracle_h_score": _five(0.84), "oracle_oscr": _five(0.82),
        "public_selector_h_score": _five(0.821),
    })
    assert not weak.passed
    assert "waveform_known_accuracy_ge_0_85" in {
        check.name for check in weak.checks if not check.passed}


def test_g1_requires_persisted_g0_and_checks_causal_message_gain():
    result = _passing_g1()
    assert result.passed
    assert result.claims["sequential_ordering_supported"]
    assert result.measurements["positive_h_folds"] == 5

    without_prerequisite = evaluate_g1({
        "frozen_agents_identical": True,
        "parallel_equal_bandwidth": True,
    }, prerequisites=None)
    assert not without_prerequisite.passed
    assert "prerequisite_g0" in {
        check.name for check in without_prerequisite.checks if not check.passed}

    failed_g0 = replace(_passing_g0(), passed=False)
    with pytest.raises(RuntimeError, match="pass flag disagrees"):
        # An inconsistent in-memory artifact is rejected before it can unlock G1.
        require_prerequisite_artifacts("g1", {"g0": failed_g0})


def test_g1_drops_seqcomm_claim_without_failing_useful_semantic_dialogue():
    g0 = _passing_g0()
    metrics = {
        "frozen_agents_identical": True,
        "parallel_equal_bandwidth": True,
        "b2_h_score": _five(0.80),
        "sequential_h_score": _five(0.812),
        "parallel_h_score": _five(0.810),
        "b2_oscr": _five(0.80),
        "sequential_oscr": _five(0.807),
        "parallel_oscr": _five(0.806),
        "b2_known_accuracy": _five(0.90),
        "sequential_known_accuracy": _five(0.897),
        "rescue_harm_ratio": _five(2.5),
        "receiver_gain_ci_low": [0.001],
        "ablation_h_drop": {"wrong_pair": _five(0.011)},
    }
    result = evaluate_g1(metrics, prerequisites={"g0": g0})
    assert result.passed
    assert not result.claims["sequential_ordering_supported"]
    assert any("Do not claim" in row for row in result.recommendations)


def test_g2_checks_coverage_recovery_correlation_and_prunes_unused_direction():
    result = _passing_g2()
    assert result.passed
    assert any("W_FIRST" in row for row in result.recommendations)

    g0, g1 = _passing_g0(), _passing_g1()
    collapsed = evaluate_g2({
        "exact_action_enumeration": True,
        "query_rate": _five(0.95),
        "oracle_gain_recovery": _five(0.61),
        "gain_spearman": _five(0.37),
        "rescue_harm_ratio": _five(2.3),
    }, prerequisites={"g0": g0, "g1": g1})
    assert not collapsed.passed
    assert "stop_rate_ge_0_10" in {
        check.name for check in collapsed.checks if not check.passed}


def test_g3_checks_15_fold_replication_and_all_formal_gates():
    g0, g1 = _passing_g0(), _passing_g1()
    g2 = _passing_g2(g0, g1)
    result = evaluate_g3({
        "lco_h_delta": np.r_[np.full(10, 0.01), np.full(5, -0.001)],
        "formal_seeds": [42, 43, 44],
        "formal_h_score": [0.90, 0.91, 0.89],
        "formal_oscr": [0.92, 0.91, 0.90],
        "b2_h_score": [0.885, 0.895, 0.875],
        "b2_oscr": [0.913, 0.903, 0.893],
        "best_single_h_score": [0.89, 0.90, 0.88],
        "messages_off_h_score": [0.88, 0.89, 0.87],
        "rescue_harm_ratio": [2.5, 2.2, 2.4],
        "bootstrap_h_ci_low": [0.001],
        "bootstrap_oscr_ci_low": [-0.001],
    }, prerequisites={"g0": g0, "g1": g1, "g2": g2})
    assert result.passed
    assert result.measurements["positive_lco_h_folds"] == 10
    assert result.measurements["positive_h_seeds"] == 3

    wrong_seed = evaluate_g3({
        "lco_h_delta": np.ones(15),
        "formal_seeds": [41, 42, 43],
    }, prerequisites={"g0": g0, "g1": g1, "g2": g2})
    assert not wrong_seed.passed
    assert "formal_seeds_are_42_43_44" in {
        check.name for check in wrong_seed.checks if not check.passed}


def test_gate_artifact_round_trip_is_deterministic_and_selection_fails_closed(tmp_path):
    first = _passing_g1()
    path = save_gate_result(tmp_path / "g1.json", first)
    text = path.read_text(encoding="utf-8")
    loaded = load_gate_result(path, expected_gate="g1", require_passed=True)
    save_gate_result(tmp_path / "g1_again.json", loaded)
    assert text == (tmp_path / "g1_again.json").read_text(encoding="utf-8")
    assert "Overall: **PASS**" in render_gate_report(loaded)

    better = replace(first, measurements={
        **first.measurements, "delta_h_mean": 0.02, "delta_oscr_mean": 0.006})
    worse = replace(first, measurements={
        **first.measurements, "delta_h_mean": 0.01, "delta_oscr_mean": 0.02})
    failed = replace(first, passed=False, checks=tuple([
        replace(first.checks[0], passed=False, detail="forced failure"),
        *first.checks[1:],
    ]))
    selection = select_passing_candidate({
        "z_worse": worse, "a_better": better, "failed": failed})
    assert selection["selected"] == "a_better"
    assert selection["rejected"]["failed"] == "gate failed"

    payload = json.loads(text)
    payload["passed"] = not payload["passed"]
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pass flag disagrees"):
        load_gate_result(corrupt)

    payload = json.loads(text)
    payload["measurements"]["delta_h_mean"] = float("nan")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_gate_result(nonfinite)
