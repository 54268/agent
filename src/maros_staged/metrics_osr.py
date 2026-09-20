"""Open-set detection metrics used by the Stage-1 complementarity diagnostic."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def detection_metrics(is_unknown: np.ndarray, unknown_score: np.ndarray) -> Dict[str, float]:
    """unknown_score oriented so that larger = more unknown."""
    is_unknown = is_unknown.astype(np.int32)
    auroc = roc_auc_score(is_unknown, unknown_score)
    aupr_out = average_precision_score(is_unknown, unknown_score)
    # FPR95: false-positive rate on known when unknown recall is fixed at 95%
    pos = unknown_score[is_unknown == 1]
    neg = unknown_score[is_unknown == 0]
    threshold = np.quantile(pos, 0.05)
    fpr95 = float((neg >= threshold).mean())
    return {"auroc": float(auroc), "aupr_out": float(aupr_out), "fpr95": fpr95}


def threshold_decision(is_unknown: np.ndarray, y: np.ndarray, closed_pred: np.ndarray,
                       unknown_score: np.ndarray,
                       tau: float | np.ndarray) -> Dict[str, float]:
    tau_array = np.asarray(tau)
    predict_unknown = unknown_score >= tau_array
    y_pred = np.where(predict_unknown, -1, closed_pred)
    known_mask = is_unknown == 0
    known_acc = float((y_pred[known_mask] == y[known_mask]).mean())
    unknown_recall = float(predict_unknown[is_unknown == 1].mean())
    labels = sorted(set(y[known_mask].tolist()) | {-1})
    macro_f1 = float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0))
    h_score = (
        2.0 * known_acc * unknown_recall / (known_acc + unknown_recall)
        if known_acc + unknown_recall > 0 else 0.0
    )
    tau_summary = float(tau_array) if tau_array.ndim == 0 else float(tau_array.mean())
    return {"tau": tau_summary, "known_accuracy": known_acc, "unknown_recall": unknown_recall,
            "macro_f1": macro_f1, "h_score": float(h_score)}


def oscr(is_unknown: np.ndarray, closed_pred: np.ndarray, y: np.ndarray,
         known_confidence: np.ndarray) -> float:
    """Area under the CCR-vs-FPR curve (larger better). known_confidence in [0,1]."""
    known_mask = is_unknown == 0
    order = np.argsort(-known_confidence, kind="mergesort")
    correct_known = ((closed_pred == y) & known_mask)[order]
    accepted_unknown = (is_unknown == 1)[order]
    ccr = np.concatenate([[0.0], np.cumsum(correct_known) / max(int(known_mask.sum()), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(accepted_unknown) / max(int((~known_mask).sum()), 1)])
    order2 = np.argsort(fpr, kind="mergesort")
    return float(np.trapz(ccr[order2], fpr[order2]))
