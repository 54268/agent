"""Known-only calibration and reproducible OSR diagnostics for Stage-7."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision

from .splits import ProvenanceSubset, assert_no_formal_unknown


CDF_PRIOR = 25.0
KNOWN_ACCEPTANCE = 0.95


class _EmpiricalCdf:
    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("CDF reference must be non-empty and finite")
        self.reference = np.sort(values)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.searchsorted(self.reference, values, side="right") / len(self.reference)


@dataclass(frozen=True)
class CalibratedDecision:
    unknown_score: np.ndarray
    threshold: np.ndarray
    reject: np.ndarray


class ClasswiseCalibrationService:
    """Class-conditional empirical CDF and threshold shrinkage, Known-only.

    The prior is deliberately fixed at 25 as part of the registered Stage-7
    protocol.  Calibration consumes predicted classes, not ground-truth
    classes, so inference follows exactly the same path as fitting.
    """

    def __init__(self, known_acceptance: float = KNOWN_ACCEPTANCE):
        if not 0.0 < known_acceptance < 1.0:
            raise ValueError("known_acceptance must lie in (0, 1)")
        self.known_acceptance = float(known_acceptance)
        self.n_prior = CDF_PRIOR
        self.global_cdf: _EmpiricalCdf | None = None
        self.class_cdfs: dict[int, _EmpiricalCdf] = {}
        self.global_threshold: float | None = None
        self.class_thresholds: dict[int, float] = {}
        self.class_counts: dict[int, int] = {}

    @staticmethod
    def _arrays(raw_unknown_score, predicted_class) -> tuple[np.ndarray, np.ndarray]:
        score = np.asarray(raw_unknown_score, dtype=np.float64).reshape(-1)
        prediction = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
        if score.shape != prediction.shape or not len(score):
            raise ValueError("score and predicted_class must be non-empty matching vectors")
        if not np.isfinite(score).all():
            raise ValueError("unknown scores must be finite")
        return score, prediction

    def fit(
        self,
        raw_unknown_score: np.ndarray,
        predicted_class: np.ndarray,
        *,
        labels: np.ndarray | None = None,
        dataset: ProvenanceSubset | None = None,
    ) -> "ClasswiseCalibrationService":
        if dataset is not None:
            assert_no_formal_unknown(dataset)
            if "calibration_known" not in dataset.purpose:
                raise ValueError("calibration must use the dedicated Known calibration split")
        if labels is not None and np.any(np.asarray(labels) < 0):
            raise RuntimeError("unknown data entered Known-only calibration")
        raw, prediction = self._arrays(raw_unknown_score, predicted_class)
        self.global_cdf = _EmpiricalCdf(raw)
        self.class_cdfs = {
            int(cls): _EmpiricalCdf(raw[prediction == cls])
            for cls in np.unique(prediction)
        }
        calibrated = self.transform(raw, prediction)
        self.global_threshold = float(
            np.quantile(calibrated, self.known_acceptance))
        self.class_thresholds = {}
        self.class_counts = {}
        for cls in np.unique(prediction):
            values = calibrated[prediction == cls]
            count = int(len(values))
            local = float(np.quantile(values, self.known_acceptance))
            weight = count / (count + self.n_prior)
            self.class_thresholds[int(cls)] = (
                weight * local + (1.0 - weight) * self.global_threshold)
            self.class_counts[int(cls)] = count
        return self

    def transform(
        self, raw_unknown_score: np.ndarray, predicted_class: np.ndarray
    ) -> np.ndarray:
        if self.global_cdf is None:
            raise RuntimeError("ClasswiseCalibrationService must be fitted first")
        raw, prediction = self._arrays(raw_unknown_score, predicted_class)
        global_score = self.global_cdf(raw)
        result = global_score.copy()
        for cls in np.unique(prediction):
            mask = prediction == cls
            local_cdf = self.class_cdfs.get(int(cls))
            if local_cdf is None:
                continue
            count = len(local_cdf.reference)
            weight = count / (count + self.n_prior)
            result[mask] = (
                weight * local_cdf(raw[mask])
                + (1.0 - weight) * global_score[mask])
        return result

    def thresholds(self, predicted_class: np.ndarray) -> np.ndarray:
        if self.global_threshold is None:
            raise RuntimeError("ClasswiseCalibrationService must be fitted first")
        prediction = np.asarray(predicted_class, dtype=np.int64).reshape(-1)
        return np.asarray([
            self.class_thresholds.get(int(cls), self.global_threshold)
            for cls in prediction
        ], dtype=np.float64)

    def predict(
        self, raw_unknown_score: np.ndarray, predicted_class: np.ndarray
    ) -> CalibratedDecision:
        score = self.transform(raw_unknown_score, predicted_class)
        threshold = self.thresholds(predicted_class)
        return CalibratedDecision(score, threshold, score >= threshold)

    def state_dict(self) -> dict:
        if self.global_cdf is None or self.global_threshold is None:
            raise RuntimeError("cannot serialise an unfitted calibrator")
        return {
            "known_acceptance": self.known_acceptance,
            "n_prior": self.n_prior,
            "global_reference": self.global_cdf.reference.tolist(),
            "class_references": {
                str(cls): cdf.reference.tolist() for cls, cdf in self.class_cdfs.items()
            },
            "global_threshold": self.global_threshold,
            "class_thresholds": {
                str(cls): value for cls, value in self.class_thresholds.items()
            },
            "class_counts": {str(cls): value for cls, value in self.class_counts.items()},
        }


def evaluate_open_set(
    y: np.ndarray,
    closed_pred: np.ndarray,
    unknown_score: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float]:
    """Evaluate with the unrejected class prediction, as OSCR requires."""
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    prediction = np.asarray(closed_pred, dtype=np.int64).reshape(-1)
    score = np.asarray(unknown_score, dtype=np.float64).reshape(-1)
    if not (y.shape == prediction.shape == score.shape):
        raise ValueError("y, closed_pred and unknown_score must match")
    is_unknown = (y < 0).astype(np.int32)
    if len(np.unique(is_unknown)) != 2:
        raise ValueError("open-set evaluation requires both Known and Unknown samples")
    result = detection_metrics(is_unknown, score)
    result.update(threshold_decision(
        is_unknown, y, prediction, score, threshold))
    result["oscr"] = oscr(is_unknown, prediction, y, 1.0 - score)
    result["num_known"] = int((is_unknown == 0).sum())
    result["num_unknown"] = int((is_unknown == 1).sum())
    return result


def _correct_open_set(
    y: np.ndarray, closed_pred: np.ndarray, unknown_score: np.ndarray,
    threshold: float | np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    prediction = np.asarray(closed_pred, dtype=np.int64)
    reject = np.asarray(unknown_score) >= np.asarray(threshold)
    return np.where(y < 0, reject, (~reject) & (prediction == y))


def complementarity_report(
    y: np.ndarray,
    systems: Mapping[str, Mapping[str, np.ndarray | float]],
    *,
    baseline: str,
) -> dict:
    """Measure unique rescue and label-using per-sample oracle headroom."""
    if baseline not in systems or len(systems) < 2:
        raise ValueError("baseline and at least one other system are required")
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    success = {}
    for name, row in systems.items():
        success[name] = _correct_open_set(
            y, row["closed_pred"], row["unknown_score"], row["threshold"])
    base = success[baseline]
    baseline_failures = max(int((~base).sum()), 1)
    unique = {}
    for name, values in success.items():
        if name == baseline:
            continue
        others = [row for key, row in success.items() if key not in {baseline, name}]
        only = values & ~base
        if others:
            only &= ~np.logical_or.reduce(others)
        unique[name] = {
            "absolute_rate": float(only.mean()),
            "baseline_failure_rate": float(only.sum() / baseline_failures),
            "count": int(only.sum()),
        }
    oracle_success = np.logical_or.reduce(list(success.values()))
    known = y >= 0
    known_accuracy = float(oracle_success[known].mean())
    unknown_recall = float(oracle_success[~known].mean())
    h_score = (2.0 * known_accuracy * unknown_recall /
               max(known_accuracy + unknown_recall, 1e-12))

    # Label-using upper-bound OSCR: select a correct class prediction where one
    # exists, and the most favourable risk across local systems.  This is an
    # audit-only ceiling and must never be deployed.
    names = list(systems)
    predictions = np.stack([
        np.asarray(systems[name]["closed_pred"], dtype=np.int64) for name in names
    ], axis=1)
    risks = np.stack([
        np.asarray(systems[name]["unknown_score"], dtype=np.float64) for name in names
    ], axis=1)
    oracle_prediction = predictions[:, 0].copy()
    for column in range(predictions.shape[1]):
        mask = known & (predictions[:, column] == y)
        oracle_prediction[mask] = y[mask]
    oracle_risk = np.where(known, risks.min(1), risks.max(1))
    oracle_oscr = oscr((~known).astype(np.int32), oracle_prediction, y, 1.0 - oracle_risk)
    return {
        "unique_rescue": unique,
        "oracle": {
            "known_accuracy": known_accuracy,
            "unknown_recall": unknown_recall,
            "h_score": float(h_score),
            "oscr": float(oracle_oscr),
        },
        "baseline_success_rate": float(base.mean()),
        "oracle_success_rate": float(oracle_success.mean()),
    }


def communication_effect_report(
    before_correct: np.ndarray,
    after_correct: np.ndarray,
    queried: np.ndarray,
    *,
    receiver_before_loss: np.ndarray | None = None,
    receiver_after_loss: np.ndarray | None = None,
) -> dict[str, float]:
    before = np.asarray(before_correct, dtype=bool).reshape(-1)
    after = np.asarray(after_correct, dtype=bool).reshape(-1)
    queried = np.asarray(queried, dtype=bool).reshape(-1)
    if not (before.shape == after.shape == queried.shape):
        raise ValueError("communication diagnostic arrays must match")
    rescue = queried & ~before & after
    harm = queried & before & ~after
    queried_count = max(int(queried.sum()), 1)
    report = {
        "query_rate": float(queried.mean()),
        "rescue_rate": float(rescue.sum() / queried_count),
        "harm_rate": float(harm.sum() / queried_count),
        "rescue_count": int(rescue.sum()),
        "harm_count": int(harm.sum()),
        "rescue_harm_ratio": float(rescue.sum() / max(int(harm.sum()), 1)),
    }
    if receiver_before_loss is not None and receiver_after_loss is not None:
        delta = (np.asarray(receiver_before_loss, dtype=np.float64)
                 - np.asarray(receiver_after_loss, dtype=np.float64))
        if delta.shape != before.shape:
            raise ValueError("receiver loss arrays must match decisions")
        selected = delta[queried]
        report["receiver_gain_mean"] = float(selected.mean()) if len(selected) else 0.0
        if len(selected) > 1:
            standard_error = selected.std(ddof=1) / np.sqrt(len(selected))
            report["receiver_gain_ci_low"] = float(selected.mean() - 1.96 * standard_error)
        else:
            report["receiver_gain_ci_low"] = 0.0
    return report


def paired_group_bootstrap_gain(
    baseline: np.ndarray,
    candidate: np.ndarray,
    group_ids: np.ndarray,
    *,
    statistic=None,
    repetitions: int = 2000,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired bootstrap over Tx groups for any per-sample statistic."""
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    groups = np.asarray(group_ids).reshape(-1)
    if baseline.shape[0] != len(groups) or candidate.shape[0] != len(groups):
        raise ValueError("bootstrap arrays and group_ids must share the first dimension")
    if statistic is None:
        statistic = lambda values: float(np.mean(values))
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    gains = np.empty(repetitions, dtype=np.float64)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique}
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([group_indices[group] for group in sampled])
        gains[index] = statistic(candidate[positions]) - statistic(baseline[positions])
    return {
        "mean": float(statistic(candidate) - statistic(baseline)),
        "ci_low": float(np.quantile(gains, 0.025)),
        "ci_high": float(np.quantile(gains, 0.975)),
        "repetitions": int(repetitions),
    }
