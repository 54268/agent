"""Fail-closed Stage-8 G0/G1/G1.5 development gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .evaluation import OpenSetRescueMetrics, RescueMetrics


@dataclass(frozen=True)
class Stage8GateReport:
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]
    next_gate: str

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_g1_gate(folds: Sequence[RescueMetrics | OpenSetRescueMetrics], *,
                     min_unique_rescue: float = 0.05,
                     min_oracle_headroom: float = 0.05,
                     min_local_known_accuracy: float = 0.50) -> Stage8GateReport:
    if not folds:
        raise ValueError("G1 requires at least one held-out-class fold")
    minimum_temporal = min(row.temporal_rescues_spectral for row in folds)
    minimum_spectral = min(row.spectral_rescues_temporal for row in folds)
    mean_headroom = sum(row.oracle_headroom for row in folds) / len(folds)
    open_rows = all(isinstance(row, OpenSetRescueMetrics) for row in folds)
    minimum_temporal_known = (min(row.temporal_known_accuracy for row in folds)
                              if open_rows else 1.0)
    minimum_spectral_known = (min(row.spectral_known_accuracy for row in folds)
                              if open_rows else 1.0)
    checks = {
        "four_inner_folds": len(folds) == 4,
        "temporal_local_known_capability": (
            minimum_temporal_known >= min_local_known_accuracy),
        "spectral_local_known_capability": (
            minimum_spectral_known >= min_local_known_accuracy),
        "temporal_unique_rescue": minimum_temporal >= min_unique_rescue,
        "spectral_unique_rescue": minimum_spectral >= min_unique_rescue,
        "oracle_headroom": mean_headroom >= min_oracle_headroom,
        "bidirectional_every_fold": all(row.bidirectional_rescue for row in folds),
    }
    passed = all(checks.values())
    return Stage8GateReport(
        passed=passed, checks=checks,
        values={
            "minimum_temporal_unique_rescue": minimum_temporal,
            "minimum_spectral_unique_rescue": minimum_spectral,
            "minimum_temporal_known_accuracy": minimum_temporal_known,
            "minimum_spectral_known_accuracy": minimum_spectral_known,
            "mean_oracle_headroom": mean_headroom,
            "num_folds": float(len(folds)),
        },
        next_gate="g1.5_rescue_predictability" if passed else "redesign_agents")


def evaluate_g15_gate(macro_aurocs: Sequence[float], *,
                      min_macro_auroc: float = 0.65,
                      min_positive_folds: int = 3) -> Stage8GateReport:
    if not macro_aurocs:
        raise ValueError("G1.5 requires held-out-class AUROC values")
    mean_value = sum(float(x) for x in macro_aurocs) / len(macro_aurocs)
    positives = sum(float(x) > 0.5 for x in macro_aurocs)
    checks = {
        "macro_auroc": mean_value >= min_macro_auroc,
        "positive_folds": positives >= min(min_positive_folds, len(macro_aurocs)),
    }
    passed = all(checks.values())
    return Stage8GateReport(
        passed=passed, checks=checks,
        values={"macro_auroc": mean_value,
                "positive_folds": float(positives),
                "num_folds": float(len(macro_aurocs))},
        next_gate="memory_agent" if passed else "redesign_agents")


def evaluate_ab_g1_gate(
    folds: Sequence[Mapping], *,
    min_local_known_accuracy: float = 0.50,
    min_pair_auroc: float = 0.70,
    min_hard_pair_accuracy: float = 0.50,
    min_bidirectional_rescue: float = 0.05,
) -> Stage8GateReport:
    """Identity-first A/B gate; H-score is deliberately not a primary check."""

    if len(folds) != 4:
        raise ValueError("A/B G1 requires exactly four inner folds")

    def minimum(key: str) -> float:
        return min(float(row[key]) for row in folds)

    def mean_nested(agent: str, key: str) -> float:
        return sum(float(row[f"{agent}_pair_metrics"][key])
                   for row in folds) / len(folds)

    def mean_rescue(key: str) -> float:
        return sum(float(row["rescue_decomposition"][key])
                   for row in folds) / len(folds)

    values = {
        "minimum_temporal_identity_accuracy": minimum(
            "temporal_identity_accuracy"),
        "minimum_spectral_identity_accuracy": minimum(
            "spectral_identity_accuracy"),
        "mean_temporal_pair_auroc": mean_nested("temporal", "pair_auroc"),
        "mean_spectral_pair_auroc": mean_nested("spectral", "pair_auroc"),
        "mean_temporal_hard_pair_accuracy": mean_nested(
            "temporal", "hard_negative_pair_accuracy"),
        "mean_spectral_hard_pair_accuracy": mean_nested(
            "spectral", "hard_negative_pair_accuracy"),
        "mean_temporal_to_spectral_identity_rescue": mean_rescue(
            "temporal_to_spectral_identity_rescue"),
        "mean_spectral_to_temporal_identity_rescue": mean_rescue(
            "spectral_to_temporal_identity_rescue"),
        "mean_temporal_to_spectral_pair_rescue": mean_rescue(
            "temporal_to_spectral_hard_pair_rescue"),
        "mean_spectral_to_temporal_pair_rescue": mean_rescue(
            "spectral_to_temporal_hard_pair_rescue"),
        "num_folds": float(len(folds)),
    }
    checks = {
        "temporal_local_learnability": (
            values["minimum_temporal_identity_accuracy"]
            >= min_local_known_accuracy),
        "spectral_local_learnability": (
            values["minimum_spectral_identity_accuracy"]
            >= min_local_known_accuracy),
        "temporal_pair_auroc": (
            values["mean_temporal_pair_auroc"] >= min_pair_auroc),
        "spectral_pair_auroc": (
            values["mean_spectral_pair_auroc"] >= min_pair_auroc),
        "temporal_hard_pair_accuracy": (
            values["mean_temporal_hard_pair_accuracy"]
            > min_hard_pair_accuracy),
        "spectral_hard_pair_accuracy": (
            values["mean_spectral_hard_pair_accuracy"]
            > min_hard_pair_accuracy),
        "temporal_to_spectral_identity_rescue": (
            values["mean_temporal_to_spectral_identity_rescue"]
            > min_bidirectional_rescue),
        "spectral_to_temporal_identity_rescue": (
            values["mean_spectral_to_temporal_identity_rescue"]
            > min_bidirectional_rescue),
        "temporal_to_spectral_pair_rescue": (
            values["mean_temporal_to_spectral_pair_rescue"]
            > min_bidirectional_rescue),
        "spectral_to_temporal_pair_rescue": (
            values["mean_spectral_to_temporal_pair_rescue"]
            > min_bidirectional_rescue),
    }
    passed = all(checks.values())
    return Stage8GateReport(
        passed=passed, checks=checks, values=values,
        next_gate="g1.5_directional_predictability" if passed
        else "redesign_agents")


def evaluate_directional_g15_gate(
    directional_aurocs: Mapping[str, Sequence[float]], *,
    min_directional_auroc: float = 0.65,
    min_positive_folds: int = 3,
) -> Stage8GateReport:
    required = ("temporal_rescues_spectral", "spectral_rescues_temporal")
    if set(directional_aurocs) != set(required):
        raise ValueError("directional G1.5 requires both rescue directions")
    values: dict[str, float] = {}
    checks: dict[str, bool] = {}
    for name in required:
        scores = [float(value) for value in directional_aurocs[name]]
        if len(scores) != 4:
            raise ValueError("directional G1.5 requires four inner-fold scores")
        mean_value = sum(scores) / len(scores)
        positives = sum(value > 0.5 for value in scores)
        values[f"mean_auroc_{name}"] = mean_value
        values[f"positive_folds_{name}"] = float(positives)
        checks[f"auroc_{name}"] = mean_value > min_directional_auroc
        checks[f"positive_folds_{name}"] = positives >= min_positive_folds
    passed = all(checks.values())
    return Stage8GateReport(
        passed=passed, checks=checks, values=values,
        next_gate="memory_agent" if passed else "redesign_agents")
