"""G1 complementarity and G1.5 rescue-predictability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_h(known_accuracy: float, unknown_recall: float) -> float:
    total = known_accuracy + unknown_recall
    return 0.0 if total <= 0 else 2.0 * known_accuracy * unknown_recall / total


@dataclass(frozen=True)
class RescueMetrics:
    temporal_accuracy: float
    spectral_accuracy: float
    best_single_accuracy: float
    oracle_accuracy: float
    oracle_headroom: float
    temporal_rescues_spectral: float
    spectral_rescues_temporal: float
    bidirectional_rescue: bool


@dataclass(frozen=True)
class OpenSetRescueMetrics:
    temporal_known_accuracy: float
    spectral_known_accuracy: float
    temporal_unknown_recall: float
    spectral_unknown_recall: float
    temporal_h: float
    spectral_h: float
    best_single_h: float
    oracle_h: float
    oracle_headroom: float
    temporal_rescues_spectral: float
    spectral_rescues_temporal: float
    bidirectional_rescue: bool


def rescue_metrics(labels, temporal_prediction,
                   spectral_prediction) -> RescueMetrics:
    labels = np.asarray(labels).reshape(-1)
    temporal = np.asarray(temporal_prediction).reshape(-1)
    spectral = np.asarray(spectral_prediction).reshape(-1)
    if not len(labels) or labels.shape != temporal.shape or labels.shape != spectral.shape:
        raise ValueError("labels and predictions must be matching non-empty vectors")
    temporal_ok = temporal == labels
    spectral_ok = spectral == labels
    temporal_accuracy = float(temporal_ok.mean())
    spectral_accuracy = float(spectral_ok.mean())
    oracle = temporal_ok | spectral_ok
    temporal_rescue = float((temporal_ok & ~spectral_ok).mean())
    spectral_rescue = float((spectral_ok & ~temporal_ok).mean())
    best = max(temporal_accuracy, spectral_accuracy)
    return RescueMetrics(
        temporal_accuracy=temporal_accuracy,
        spectral_accuracy=spectral_accuracy,
        best_single_accuracy=best,
        oracle_accuracy=float(oracle.mean()),
        oracle_headroom=float(oracle.mean()) - best,
        temporal_rescues_spectral=temporal_rescue,
        spectral_rescues_temporal=spectral_rescue,
        bidirectional_rescue=temporal_rescue > 0 and spectral_rescue > 0)


def rescue_labels(labels, temporal_prediction, spectral_prediction) -> np.ndarray:
    """0 none/both, 1 temporal uniquely wins, 2 spectral uniquely wins."""

    labels = np.asarray(labels).reshape(-1)
    temporal_ok = np.asarray(temporal_prediction).reshape(-1) == labels
    spectral_ok = np.asarray(spectral_prediction).reshape(-1) == labels
    result = np.zeros(len(labels), dtype=np.int64)
    result[temporal_ok & ~spectral_ok] = 1
    result[spectral_ok & ~temporal_ok] = 2
    return result


def open_set_rescue_metrics(labels, temporal_prediction,
                            spectral_prediction) -> OpenSetRescueMetrics:
    """Measure rescue and oracle H from final predictions (``-1`` = reject)."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    temporal = np.asarray(temporal_prediction, dtype=np.int64).reshape(-1)
    spectral = np.asarray(spectral_prediction, dtype=np.int64).reshape(-1)
    if not len(labels) or labels.shape != temporal.shape or labels.shape != spectral.shape:
        raise ValueError("labels and predictions must be matching non-empty vectors")
    known = labels >= 0
    unknown = ~known
    if not known.any() or not unknown.any():
        raise ValueError("open-set rescue metrics require Known and proxy Unknown samples")
    temporal_ok = temporal == labels
    spectral_ok = spectral == labels
    temporal_known = float(temporal_ok[known].mean())
    spectral_known = float(spectral_ok[known].mean())
    temporal_unknown = float(temporal_ok[unknown].mean())
    spectral_unknown = float(spectral_ok[unknown].mean())
    temporal_h = _safe_h(temporal_known, temporal_unknown)
    spectral_h = _safe_h(spectral_known, spectral_unknown)
    oracle_ok = temporal_ok | spectral_ok
    oracle_h = _safe_h(
        float(oracle_ok[known].mean()), float(oracle_ok[unknown].mean()))
    temporal_rescue = float((temporal_ok & ~spectral_ok).mean())
    spectral_rescue = float((spectral_ok & ~temporal_ok).mean())
    best = max(temporal_h, spectral_h)
    return OpenSetRescueMetrics(
        temporal_known_accuracy=temporal_known,
        spectral_known_accuracy=spectral_known,
        temporal_unknown_recall=temporal_unknown,
        spectral_unknown_recall=spectral_unknown,
        temporal_h=temporal_h, spectral_h=spectral_h,
        best_single_h=best, oracle_h=oracle_h,
        oracle_headroom=oracle_h - best,
        temporal_rescues_spectral=temporal_rescue,
        spectral_rescues_temporal=spectral_rescue,
        bidirectional_rescue=temporal_rescue > 0 and spectral_rescue > 0)


def open_set_h_metrics(labels, prediction, reject) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    reject = np.asarray(reject, dtype=bool).reshape(-1)
    known = labels >= 0
    unknown = ~known
    known_accuracy = float(((prediction[known] == labels[known]) & ~reject[known]).mean())
    unknown_recall = float(reject[unknown].mean()) if unknown.any() else 0.0
    return {
        "known_accuracy": known_accuracy,
        "unknown_recall": unknown_recall,
        "h_score": _safe_h(known_accuracy, unknown_recall),
    }


def pair_verification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    hard_negative: np.ndarray,
) -> dict[str, float]:
    """Class-shared prototype verifier diagnostics on Known samples."""

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    hard_negative = np.asarray(hard_negative, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("verification scores must have shape [samples, classes]")
    if hard_negative.shape != labels.shape:
        raise ValueError("hard_negative must have shape [samples]")
    rows = np.arange(len(labels))
    if (np.any(labels < 0) or np.any(labels >= scores.shape[1])
            or np.any(hard_negative < 0)
            or np.any(hard_negative >= scores.shape[1])
            or np.any(hard_negative == labels)):
        raise ValueError("pair metric class indices are invalid")
    target = np.zeros_like(scores, dtype=np.int64)
    target[rows, labels] = 1
    probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    true_score = scores[rows, labels]
    hard_score = scores[rows, hard_negative]
    return {
        "pair_accuracy": float((scores.argmax(1) == labels).mean()),
        "pair_auroc": float(roc_auc_score(target.reshape(-1), scores.reshape(-1))),
        "hard_negative_pair_accuracy": float((true_score > hard_score).mean()),
        "pair_brier": float(np.square(probability - target).mean()),
    }


def rescue_decomposition(
    labels: np.ndarray,
    temporal_prediction: np.ndarray,
    spectral_prediction: np.ndarray,
    temporal_reject: np.ndarray,
    spectral_reject: np.ndarray,
    temporal_pair_correct: np.ndarray,
    spectral_pair_correct: np.ndarray,
) -> dict[str, float]:
    """Separate identity, proxy-Unknown, and candidate-pair rescue."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    temporal_prediction = np.asarray(temporal_prediction).reshape(-1)
    spectral_prediction = np.asarray(spectral_prediction).reshape(-1)
    temporal_reject = np.asarray(temporal_reject, dtype=bool).reshape(-1)
    spectral_reject = np.asarray(spectral_reject, dtype=bool).reshape(-1)
    temporal_pair_correct = np.asarray(temporal_pair_correct, dtype=bool).reshape(-1)
    spectral_pair_correct = np.asarray(spectral_pair_correct, dtype=bool).reshape(-1)
    expected = labels.shape
    if any(value.shape != expected for value in (
            temporal_prediction, spectral_prediction, temporal_reject,
            spectral_reject, temporal_pair_correct, spectral_pair_correct)):
        raise ValueError("rescue decomposition vectors must have matching shapes")
    known = labels >= 0
    unknown = ~known
    if not known.any() or not unknown.any():
        raise ValueError("rescue decomposition requires Known and proxy Unknown")
    temporal_identity = temporal_prediction == labels
    spectral_identity = spectral_prediction == labels
    return {
        "temporal_to_spectral_identity_rescue": float(
            (temporal_identity[known] & ~spectral_identity[known]).mean()),
        "spectral_to_temporal_identity_rescue": float(
            (spectral_identity[known] & ~temporal_identity[known]).mean()),
        "temporal_unknown_rejection_rescue": float(
            (temporal_reject[unknown] & ~spectral_reject[unknown]).mean()),
        "spectral_unknown_rejection_rescue": float(
            (spectral_reject[unknown] & ~temporal_reject[unknown]).mean()),
        "temporal_to_spectral_pair_rescue": float(
            (temporal_pair_correct[known] & ~spectral_pair_correct[known]).mean()),
        "spectral_to_temporal_pair_rescue": float(
            (spectral_pair_correct[known] & ~temporal_pair_correct[known]).mean()),
    }
