"""Leakage-safe episode snapshots backed by frozen Stage-5/13 evidence."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import torch


TRAINING_SOURCES = frozenset({"known", "lco_proxy", "certified_pug"})


def _as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class EpisodeSnapshot:
    sample_id: str
    label: int
    source_kind: str
    identity_logits: np.ndarray
    geometry_logits: np.ndarray
    identity_prototype_risk: float
    geometry_prototype_risk: float
    openmax_risk: float
    boundary_risk: float
    identity_energy: float
    identity_distance_margin: float
    geometry_distance_margin: float
    view_risks: np.ndarray
    formal_unknown: bool = False

    def __post_init__(self) -> None:
        identity = _as_vector(self.identity_logits, "identity_logits")
        geometry = _as_vector(self.geometry_logits, "geometry_logits")
        views = _as_vector(self.view_risks, "view_risks")
        if len(identity) < 2 or identity.shape != geometry.shape:
            raise ValueError("A/B logits must have the same K >= 2")
        if len(views) < 1:
            raise ValueError("at least one Geometry view is required")
        object.__setattr__(self, "identity_logits", identity.copy())
        object.__setattr__(self, "geometry_logits", geometry.copy())
        object.__setattr__(self, "view_risks", np.clip(views, 0.0, 1.0).copy())
        for name in ("identity_prototype_risk", "geometry_prototype_risk",
                     "openmax_risk", "boundary_risk"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(np.clip(value, 0.0, 1.0)))
        if self.label >= len(identity):
            raise ValueError("Known label is outside the local class mapping")
        if self.source_kind in {"lco_proxy", "certified_pug", "formal_unknown"} \
                and self.label != -1:
            raise ValueError("Unknown episodes must use label=-1")

    @property
    def num_classes(self) -> int:
        return int(len(self.identity_logits))

    @property
    def is_unknown(self) -> bool:
        return self.label < 0


class TrainingEpisodeBank(Sequence[EpisodeSnapshot]):
    """A fail-closed bank: formal Unknown can never enter MAPPO training."""

    def __init__(self, snapshots: Iterable[EpisodeSnapshot]):
        self._items = tuple(snapshots)
        if not self._items:
            raise ValueError("training episode bank cannot be empty")
        for snapshot in self._items:
            if snapshot.formal_unknown or snapshot.source_kind not in TRAINING_SOURCES:
                raise RuntimeError(
                    f"formal or unapproved Unknown entered training: {snapshot.sample_id}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def balanced_indices(self, rng: np.random.Generator, count: int) -> np.ndarray:
        known = np.asarray([i for i, item in enumerate(self) if not item.is_unknown])
        unknown = np.asarray([i for i, item in enumerate(self) if item.is_unknown])
        if not len(known) or not len(unknown):
            raise ValueError("balanced sampling requires Known and proxy Unknown episodes")
        half = count // 2
        result = np.r_[rng.choice(known, count - half, replace=len(known) < count-half),
                       rng.choice(unknown, half, replace=len(unknown) < half)]
        rng.shuffle(result)
        return result


class EvaluationEpisodeBank(Sequence[EpisodeSnapshot]):
    def __init__(self, snapshots: Iterable[EpisodeSnapshot]):
        self._items = tuple(snapshots)
        if not self._items:
            raise ValueError("evaluation episode bank cannot be empty")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def freeze_perception(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _probability_summary(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    top = np.sort(probability, axis=1)[:, -2:]
    entropy = -(probability * np.log(probability.clip(1e-12))).sum(axis=1)
    return probability, top[:, 1], top[:, 1] - top[:, 0], entropy


def snapshots_from_records(records, *, source_kind: str, prefix: str,
                           support=None, boundary_risk=None,
                           certified_mask=None) -> list[EpisodeSnapshot]:
    """Convert Stage-5 ExpertRecords (including Stage-13 PUG records) to episodes.

    ``support`` may be a fitted ``maros_stage13.feature_pug.CSupport``. Its
    Known-only empirical CDF values are preferred over raw distance surrogates.
    """
    count = len(records.labels)
    if source_kind not in TRAINING_SOURCES and source_kind != "formal_unknown":
        raise ValueError("unrecognised episode provenance")
    if certified_mask is None:
        selected = np.arange(count)
    else:
        mask = np.asarray(certified_mask, dtype=bool)
        if mask.shape != (count,):
            raise ValueError("certified mask shape mismatch")
        selected = np.flatnonzero(mask)
    if support is not None:
        calibrated = support.transform(records)
        id_risk = calibrated["identity"]
        geo_risk = calibrated["geometry"]
        open_risk = calibrated["openmax"]
    else:
        id_risk = 1.0 / (1.0 + np.exp(-(records.evidence[:, 7] - 1.0)))
        geo_risk = 1.0 / (1.0 + np.exp(-(records.evidence[:, 3] - 1.0)))
        open_risk = np.clip(records.evidence[:, 6], 0.0, 1.0)
    if boundary_risk is None:
        boundary = np.maximum.reduce([id_risk, geo_risk, open_risk])
    else:
        boundary = np.asarray(boundary_risk, dtype=np.float32).reshape(-1)
        if boundary.shape != (count,):
            raise ValueError("boundary risk shape mismatch")
    evidence = np.asarray(records.evidence, dtype=np.float32)
    # Stage-5 Universal Geometry places its five per-view distances after
    # base(6), OpenMax(1), id-prototype(2), and disagreement(1).
    view_count = (records.geometry_view_pred.shape[1]
                  if records.geometry_view_pred is not None else 1)
    start = 10
    if evidence.shape[1] >= start + view_count:
        raw_views = evidence[:, start:start + view_count]
        view_risk = 1.0 / (1.0 + np.exp(-(raw_views - 1.0)))
    else:
        view_risk = np.repeat(geo_risk[:, None], view_count, axis=1)
    _, _, _, id_entropy = _probability_summary(np.asarray(records.identity_logits))
    snapshots = []
    for index in selected:
        label = int(records.labels[index])
        if source_kind in {"lco_proxy", "certified_pug", "formal_unknown"}:
            label = -1
        snapshots.append(EpisodeSnapshot(
            sample_id=f"{prefix}:{int(index)}", label=label,
            source_kind=source_kind,
            identity_logits=records.identity_logits[index],
            geometry_logits=records.geometry_logits[index],
            identity_prototype_risk=float(id_risk[index]),
            geometry_prototype_risk=float(geo_risk[index]),
            openmax_risk=float(open_risk[index]),
            boundary_risk=float(boundary[index]),
            identity_energy=float(evidence[index, 2]),
            identity_distance_margin=float(-evidence[index, 8]),
            geometry_distance_margin=float(-evidence[index, 4]),
            view_risks=view_risk[index],
            formal_unknown=source_kind == "formal_unknown",
        ))
    return snapshots


def with_label(snapshot: EpisodeSnapshot, label: int) -> EpisodeSnapshot:
    """Test helper used to prove labels never enter observations or critic state."""
    return replace(snapshot, label=int(label))
