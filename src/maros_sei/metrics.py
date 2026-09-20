from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fpr95_known_vs_unknown(y_true: np.ndarray, unknown_score: np.ndarray, unknown_label: int) -> float:
    binary = (y_true == unknown_label).astype(np.int32)
    positive = unknown_score[binary == 1]
    negative = unknown_score[binary == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.0
    threshold = np.quantile(positive, 0.05)
    return float((negative >= threshold).mean())


def _oscr_curve(
    y_true: np.ndarray,
    closed_set_pred: np.ndarray,
    known_confidence: np.ndarray,
    unknown_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    known_mask = y_true != unknown_label
    unknown_mask = ~known_mask
    order = np.argsort(-known_confidence, kind="mergesort")
    correct_known = ((closed_set_pred == y_true) & known_mask)[order]
    accepted_unknown = unknown_mask[order]
    ccr = np.concatenate([[0.0], np.cumsum(correct_known) / max(int(known_mask.sum()), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(accepted_unknown) / max(int(unknown_mask.sum()), 1)])
    return fpr.astype(np.float64), ccr.astype(np.float64)


def oscr_scores(
    y_true: np.ndarray,
    closed_set_pred: np.ndarray,
    known_confidence: np.ndarray,
    unknown_label: int,
) -> dict[str, Any]:
    fpr, ccr = _oscr_curve(y_true, closed_set_pred, known_confidence, unknown_label)
    order = np.argsort(fpr, kind="mergesort")
    standard = float(np.trapz(ccr[order], fpr[order]))
    # Compatibility value follows the preceding project's endpoint convention.
    fpr_legacy = np.concatenate([fpr, [1.0]])
    ccr_legacy = np.concatenate([ccr, [1.0]])
    legacy_order = np.argsort(fpr_legacy, kind="mergesort")
    compatible = float(np.trapz(ccr_legacy[legacy_order], fpr_legacy[legacy_order]))
    return {
        "oscr": compatible,
        "oscr_standard": standard,
        "curve_fpr": fpr,
        "curve_ccr": ccr,
    }


def binary_ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(target), 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1.0 else probability <= upper)
        if np.any(mask):
            value += mask.sum() / total * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def evaluate_open_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    closed_set_pred: np.ndarray,
    unknown_score: np.ndarray,
    unknown_label: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    binary = (y_true == unknown_label).astype(np.int32)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {unknown_label})
    known_mask = y_true != unknown_label
    unknown_mask = ~known_mask
    unknown_precision = float(
        (y_true[y_pred == unknown_label] == unknown_label).mean()
    ) if np.any(y_pred == unknown_label) else 0.0
    unknown_recall = float((y_pred[unknown_mask] == unknown_label).mean()) if np.any(unknown_mask) else 0.0
    unknown_f1 = (
        2.0 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall)
        if unknown_precision + unknown_recall > 0
        else 0.0
    )
    oscr = oscr_scores(y_true, closed_set_pred, 1.0 - unknown_score, unknown_label)
    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "known_accuracy": float((y_pred[known_mask] == y_true[known_mask]).mean()),
        "closed_set_known_accuracy": float((closed_set_pred[known_mask] == y_true[known_mask]).mean()),
        "unknown_precision": unknown_precision,
        "unknown_recall": unknown_recall,
        "unknown_f1": float(unknown_f1),
        "known_fpr_as_unknown": float((y_pred[known_mask] == unknown_label).mean()),
        "unknown_false_accept_rate": float((y_pred[unknown_mask] != unknown_label).mean()),
        "auroc": float(roc_auc_score(binary, unknown_score)),
        "aupr_out": float(average_precision_score(binary, unknown_score)),
        "fpr95": fpr95_known_vs_unknown(y_true, unknown_score, unknown_label),
        "oscr": float(oscr["oscr"]),
        "oscr_standard": float(oscr["oscr_standard"]),
        "open_set_ece": binary_ece(binary, unknown_score),
        "open_set_brier": float(np.mean((unknown_score - binary) ** 2)),
        "num_test": int(len(y_true)),
        "num_test_known": int(known_mask.sum()),
        "num_test_unknown": int(unknown_mask.sum()),
    }
    curves = {"oscr_fpr": oscr["curve_fpr"], "oscr_ccr": oscr["curve_ccr"]}
    return metrics, curves


def purity_score(true_encoded: np.ndarray, predicted: np.ndarray) -> float:
    if len(true_encoded) == 0:
        return 0.0
    total = 0
    for cluster in np.unique(predicted):
        counts = np.bincount(true_encoded[predicted == cluster])
        total += int(counts.max())
    return float(total / len(true_encoded))


def hungarian_accuracy(true_encoded: np.ndarray, predicted: np.ndarray) -> float:
    if len(true_encoded) == 0:
        return 0.0
    true_ids = np.unique(true_encoded)
    cluster_ids = np.unique(predicted)
    matrix = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_map = {int(value): index for index, value in enumerate(true_ids)}
    cluster_map = {int(value): index for index, value in enumerate(cluster_ids)}
    for true_value, cluster in zip(true_encoded, predicted):
        matrix[true_map[int(true_value)], cluster_map[int(cluster)]] += 1
    row_index, column_index = linear_sum_assignment(-matrix)
    return float(matrix[row_index, column_index].sum() / len(true_encoded))


def evaluate_subdivision(true_names: np.ndarray, clusters: np.ndarray) -> dict[str, float | int]:
    unique_names = sorted(np.unique(true_names).tolist())
    mapping = {name: index for index, name in enumerate(unique_names)}
    encoded = np.asarray([mapping[name] for name in true_names], dtype=np.int64)
    return {
        "num_evaluated_unknown": int(len(true_names)),
        "num_true_unknown_classes": int(len(unique_names)),
        "num_predicted_clusters": int(len(np.unique(clusters))) if len(clusters) else 0,
        "nmi": float(normalized_mutual_info_score(encoded, clusters)) if len(encoded) else 0.0,
        "ari": float(adjusted_rand_score(encoded, clusters)) if len(encoded) else 0.0,
        "purity": purity_score(encoded, clusters),
        "hungarian_accuracy": hungarian_accuracy(encoded, clusters),
    }

