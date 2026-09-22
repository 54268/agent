"""Open-set metrics and trajectory diagnostics for Stage-14."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from .actions import CAction
from .buffer import RolloutBuffer
from .data import EvaluationEpisodeBank
from .env import EnvConfig, OSSEIMultiAgentEnv
from .runner import collect_episode


def evaluate(trainer, bank: EvaluationEpisodeBank,
             env_config: EnvConfig | None = None,
             forced_actions: dict[str, int] | None = None,
             tool_overrides: dict | None = None) -> tuple[dict, list]:
    env = OSSEIMultiAgentEnv(
        env_config, allow_formal_unknown=True, tool_overrides=tool_overrides)
    episodes = [collect_episode(trainer, env, snapshot, deterministic=True,
                                forced_actions=forced_actions)
                for snapshot in bank]
    labels = np.asarray([episode.label for episode in episodes], dtype=int)
    prediction = np.asarray([episode.prediction for episode in episodes], dtype=int)
    known = labels >= 0
    unknown = ~known
    known_accuracy = float(np.mean(prediction[known] == labels[known])) if known.any() else 0.0
    unknown_recall = float(np.mean(prediction[unknown] == -1)) if unknown.any() else 0.0
    h_score = (2*known_accuracy*unknown_recall/(known_accuracy+unknown_recall)
               if known_accuracy+unknown_recall else 0.0)
    unknown_score = np.asarray([
        episode.steps[-1].action_distributions["open_set"][
            int(CAction.REJECT_UNKNOWN)] for episode in episodes], dtype=float)
    auroc = (float(roc_auc_score(unknown.astype(int), unknown_score))
             if known.any() and unknown.any() else float("nan"))
    oscr = _oscr(labels, np.asarray([
        episode.final_hypothesis if episode.final_hypothesis is not None else -1
        for episode in episodes]), unknown_score)
    paths = Counter("|".join(
        "+".join(step["actions"].values()) for step in episode.trajectory)
                    for episode in episodes)
    action_usage = Counter()
    for episode in episodes:
        for step in episode.trajectory:
            action_usage.update(step["actions"].values())
    metrics = {
        "known_accuracy": known_accuracy, "unknown_recall": unknown_recall,
        "h_score": h_score, "auroc": auroc, "oscr": oscr,
        "mean_reward": float(np.mean([episode.total_reward for episode in episodes])),
        "mean_steps": float(np.mean([len(episode.steps) for episode in episodes])),
        "unique_trajectories": len(paths), "action_usage": dict(action_usage),
        "timeout_count": int(sum(
            episode.trajectory[-1]["termination_reason"] == "timeout_failure"
            for episode in episodes)),
    }
    return metrics, episodes


def _episode_arrays(episodes):
    labels = np.asarray([episode.label for episode in episodes], dtype=int)
    hypotheses = np.asarray([
        episode.final_hypothesis if episode.final_hypothesis is not None else -1
        for episode in episodes], dtype=int)
    unknown_score = np.asarray([
        episode.steps[-1].action_distributions["open_set"][
            int(CAction.REJECT_UNKNOWN)] for episode in episodes], dtype=float)
    return labels, hypotheses, unknown_score


def calibrate_known_accuracy_threshold(episodes, target: float = 0.95) -> dict:
    """Choose the smallest known-only threshold meeting an accuracy target.

    Lower reject scores are accepted as Known.  Using the smallest feasible
    threshold preserves as much Unknown recall as the calibration constraint
    allows.  Unknown samples are neither required nor consulted.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target known accuracy must be in (0, 1]")
    labels, hypotheses, unknown_score = _episode_arrays(episodes)
    known = labels >= 0
    if not known.any():
        raise ValueError("threshold calibration requires Known episodes")
    correct_scores = np.sort(unknown_score[known & (hypotheses == labels)])
    known_count = int(known.sum())
    required = int(np.ceil(target * known_count - 1e-12))
    feasible = len(correct_scores) >= required
    threshold = float(correct_scores[required-1]) if feasible else float("inf")
    attained = float(np.mean(
        known & (hypotheses == labels) & (unknown_score <= threshold)) /
        np.mean(known))
    return {
        "target_known_accuracy": float(target),
        "threshold": threshold,
        "calibration_known_accuracy": attained,
        "calibration_known_count": known_count,
        "feasible": bool(feasible),
    }


def metrics_at_reject_threshold(episodes, threshold: float) -> dict:
    """Evaluate a fixed reject threshold without changing score rankings."""
    labels, hypotheses, unknown_score = _episode_arrays(episodes)
    known = labels >= 0
    unknown = ~known
    prediction = np.where(unknown_score > threshold, -1, hypotheses)
    known_accuracy = float(np.mean(prediction[known] == labels[known])) if known.any() else 0.0
    unknown_recall = float(np.mean(prediction[unknown] == -1)) if unknown.any() else 0.0
    h_score = (2*known_accuracy*unknown_recall/(known_accuracy+unknown_recall)
               if known_accuracy+unknown_recall else 0.0)
    auroc = (float(roc_auc_score(unknown.astype(int), unknown_score))
             if known.any() and unknown.any() else float("nan"))
    return {
        "reject_threshold": float(threshold),
        "known_accuracy": known_accuracy,
        "unknown_recall": unknown_recall,
        "h_score": h_score,
        "auroc": auroc,
        "oscr": _oscr(labels, hypotheses, unknown_score),
    }


def _oscr(labels: np.ndarray, class_prediction: np.ndarray,
          unknown_score: np.ndarray) -> float:
    known = labels >= 0
    unknown = ~known
    if not known.any() or not unknown.any():
        return float("nan")
    thresholds = np.r_[-np.inf, np.unique(unknown_score), np.inf]
    fpr, ccr = [], []
    correct = class_prediction == labels
    for threshold in thresholds:
        accepted = unknown_score <= threshold
        fpr.append(float(np.mean(accepted[unknown])))
        ccr.append(float(np.mean(accepted[known] & correct[known])))
    order = np.argsort(fpr)
    return float(np.trapz(np.asarray(ccr)[order], np.asarray(fpr)[order]))
