"""Deterministic synthetic evidence used only for fast algorithm smoke tests."""
from __future__ import annotations

import numpy as np

from .data import EpisodeSnapshot, EvaluationEpisodeBank, TrainingEpisodeBank


def make_synthetic_snapshots(count: int = 256, num_classes: int = 4,
                             seed: int = 42):
    if count < 8 or num_classes < 2:
        raise ValueError("synthetic smoke requires at least 8 samples and 2 classes")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        unknown = index % 2 == 1
        label = -1 if unknown else int(rng.integers(num_classes))
        candidate = int(rng.integers(num_classes)) if unknown else label
        a_candidate = candidate
        # A is deliberately wrong on part of Known; B can causally rescue it.
        if not unknown and index % 6 == 0:
            a_candidate = (label + 1) % num_classes
        a_logits = np.full(num_classes, -1.5, dtype=np.float32)
        b_logits = np.full(num_classes, -1.5, dtype=np.float32)
        a_logits[a_candidate] = 2.5
        b_logits[candidate if unknown else label] = 2.5
        open_risk = float(rng.uniform(0.78, 0.95) if unknown
                          else rng.uniform(0.05, 0.22))
        boundary = float(np.clip(open_risk + rng.normal(0, 0.02), 0, 1))
        rows.append(EpisodeSnapshot(
            sample_id=f"synthetic:{index}", label=label,
            source_kind="lco_proxy" if unknown else "known",
            identity_logits=a_logits, geometry_logits=b_logits,
            # Proposal messages cannot solve open-set status directly.
            identity_prototype_risk=0.4,
            geometry_prototype_risk=float(0.75 if unknown else 0.15),
            openmax_risk=open_risk, boundary_risk=boundary,
            identity_energy=0.7 if unknown else -0.2,
            identity_distance_margin=0.4,
            geometry_distance_margin=0.5,
            view_risks=np.asarray([
                np.clip(open_risk + rng.normal(0, 0.04), 0, 1)
                for _ in range(5)], dtype=np.float32)))
    return rows


def make_synthetic_banks(count: int = 256, num_classes: int = 4,
                         seed: int = 42):
    rows = make_synthetic_snapshots(count, num_classes, seed)
    cut = int(0.75 * len(rows))
    return TrainingEpisodeBank(rows[:cut]), EvaluationEpisodeBank(rows[cut:])
