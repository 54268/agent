"""Conservative public-only expert selection for the Stage-7 fair B2.

The selector is trained on exact local action losses from inner meta episodes.
It has an auditable anchor fallback: an alternative Agent is selected only
when its fit-side cross-fitted probability exceeds a threshold that produced
positive realised loss reduction.  No private feature or test row is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .nested_g0 import SelectorRowBatch


def _classifier(seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=int(seed)))


def _probability(model: Pipeline | None, features: np.ndarray,
                 alternative: int) -> np.ndarray:
    if model is None:
        return np.zeros(len(features), dtype=np.float64)
    classes = np.asarray(model.named_steps["logisticregression"].classes_)
    where = np.flatnonzero(classes == int(alternative))
    if len(where) != 1:
        return np.zeros(len(features), dtype=np.float64)
    return model.predict_proba(features)[:, int(where[0])]


def _fit_or_none(features: np.ndarray, targets: np.ndarray,
                 seed: int) -> Pipeline | None:
    if len(np.unique(targets)) < 2:
        return None
    model = _classifier(seed)
    model.fit(features, targets)
    return model


@dataclass(frozen=True)
class SafeSelectorAudit:
    anchor_index: int
    alternative_index: int
    threshold: float
    fit_oof_gain: float
    fit_oof_switch_rate: float
    fit_oof_accuracy: float
    producer_indices: tuple[int, ...]
    public_only: bool = True
    outer_test_used: bool = False
    formal_unknown_used: bool = False


@dataclass
class SafePublicSelector:
    model: Pipeline | None
    audit: SafeSelectorAudit

    def choose(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("safe-selector features must be a finite matrix")
        probability = _probability(
            self.model, values, self.audit.alternative_index)
        choice = np.full(len(values), self.audit.anchor_index, dtype=np.int64)
        choice[probability >= self.audit.threshold] = self.audit.alternative_index
        return choice

    def alternative_probability(self, features: np.ndarray) -> np.ndarray:
        return _probability(
            self.model, np.asarray(features, dtype=np.float64),
            self.audit.alternative_index)


def fit_safe_public_selector(
    batches: Sequence[SelectorRowBatch],
    *,
    seed: int = 2026,
    probability_grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    minimum_switch_rate: float = 0.05,
    maximum_switch_rate: float = 0.50,
) -> SafePublicSelector:
    """Fit a conservative two-Agent selector using producer-level OOF tuning."""

    ordered = tuple(sorted(batches, key=lambda row: row.producer_inner_index))
    if len(ordered) < 2:
        raise ValueError("safe selector requires at least two inner producers")
    producers = tuple(int(row.producer_inner_index) for row in ordered)
    if len(set(producers)) != len(producers):
        raise ValueError("safe selector producer indices must be unique")
    for row in ordered:
        if row.public_features.ndim != 2 or row.action_losses.ndim != 2:
            raise ValueError("safe selector batches must contain matrices")
        if row.action_losses.shape[1] != 2:
            raise ValueError("safe selector supports exactly two local Agents")
        if len(row.public_features) != len(row.action_losses):
            raise ValueError("safe selector feature/loss rows are misaligned")
        if not (np.isfinite(row.public_features).all()
                and np.isfinite(row.action_losses).all()):
            raise ValueError("safe selector rows contain non-finite values")

    all_losses = np.concatenate([row.action_losses for row in ordered])
    anchor = int(all_losses.mean(0).argmin())
    alternative = 1 - anchor
    oof_probability, oof_losses, oof_targets = [], [], []
    for heldout in ordered:
        fit = [row for row in ordered if row is not heldout]
        x_fit = np.concatenate([row.public_features for row in fit])
        losses_fit = np.concatenate([row.action_losses for row in fit])
        target_fit = losses_fit.argmin(1)
        model = _fit_or_none(
            x_fit, target_fit, int(seed) + 101 * heldout.producer_inner_index)
        oof_probability.append(_probability(
            model, heldout.public_features, alternative))
        oof_losses.append(heldout.action_losses)
        oof_targets.append(heldout.action_losses.argmin(1))
    probability = np.concatenate(oof_probability)
    losses = np.concatenate(oof_losses)
    targets = np.concatenate(oof_targets)
    thresholds = tuple(float(value) for value in probability_grid)
    if (not thresholds or any(not 0.5 <= value <= 1.0 for value in thresholds)
            or tuple(sorted(set(thresholds))) != thresholds):
        raise ValueError("probability_grid must be sorted unique values in [0.5, 1]")
    if not 0.0 <= minimum_switch_rate <= maximum_switch_rate <= 1.0:
        raise ValueError("safe-selector switch-rate bounds are invalid")

    rows = np.arange(len(losses))
    anchor_loss = losses[:, anchor]
    candidates = []
    for threshold in thresholds:
        switch = probability >= threshold
        rate = float(switch.mean())
        choice = np.where(switch, alternative, anchor)
        gain = float((anchor_loss - losses[rows, choice]).mean())
        if minimum_switch_rate <= rate <= maximum_switch_rate and gain > 0:
            candidates.append((gain, threshold, rate, choice))
    if candidates:
        # Gain is selected exclusively on producer-level OOF fit rows.  A
        # higher threshold wins exact ties, preserving the safer fallback.
        gain, threshold, rate, oof_choice = max(
            candidates, key=lambda row: (row[0], row[1]))
    else:
        gain, threshold, rate = 0.0, 1.0, 0.0
        oof_choice = np.full(len(losses), anchor, dtype=np.int64)

    x_all = np.concatenate([row.public_features for row in ordered])
    target_all = all_losses.argmin(1)
    model = _fit_or_none(x_all, target_all, int(seed) + 9001)
    audit = SafeSelectorAudit(
        anchor_index=anchor, alternative_index=alternative,
        threshold=float(threshold), fit_oof_gain=float(gain),
        fit_oof_switch_rate=float(rate),
        fit_oof_accuracy=float((oof_choice == targets).mean()),
        producer_indices=producers)
    return SafePublicSelector(model=model, audit=audit)


__all__ = ["SafePublicSelector", "SafeSelectorAudit", "fit_safe_public_selector"]
