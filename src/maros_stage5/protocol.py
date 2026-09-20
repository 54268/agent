"""Class-disjoint leave-class-out protocol and leakage guards."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LeaveClassOutFold:
    index: int
    support_classes: tuple[int, ...]
    heldout_classes: tuple[int, ...]


def build_lco_folds(num_classes: int = 40, num_folds: int = 5,
                    seed: int = 2026) -> list[LeaveClassOutFold]:
    if num_classes % num_folds:
        raise ValueError("num_classes must be divisible by num_folds")
    classes = np.arange(num_classes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(classes)
    groups = np.split(classes, num_folds)
    folds = []
    for index, group in enumerate(groups):
        held = tuple(sorted(int(x) for x in group))
        support = tuple(int(x) for x in range(num_classes) if int(x) not in held)
        folds.append(LeaveClassOutFold(index, support, held))
    validate_lco_folds(folds, num_classes)
    return folds


def validate_lco_folds(folds: list[LeaveClassOutFold], num_classes: int) -> None:
    held_counts = np.zeros(num_classes, dtype=np.int64)
    universe = set(range(num_classes))
    for fold in folds:
        support, held = set(fold.support_classes), set(fold.heldout_classes)
        if support & held or support | held != universe:
            raise ValueError(f"invalid class partition in fold {fold.index}")
        for cls in held:
            held_counts[cls] += 1
    if not np.all(held_counts == 1):
        raise ValueError("every class must be held out exactly once")


class IndexedClassSubset(Dataset):
    """Index a source dataset without bypassing its online augmentation."""

    def __init__(self, dataset, indices: np.ndarray, mapping: dict[int, int] | None):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        x, y = self.dataset[int(self.indices[index])]
        value = int(y)
        if self.mapping is not None:
            value = self.mapping[value]
        return x, torch.tensor(value, dtype=torch.long)


def class_subset(dataset, classes: tuple[int, ...], remap: bool) -> IndexedClassSubset:
    labels = np.asarray(dataset.y, dtype=np.int64)
    mask = np.isin(labels, np.asarray(classes, dtype=np.int64))
    indices = np.where(mask)[0]
    mapping = ({cls: local for local, cls in enumerate(classes)} if remap else None)
    return IndexedClassSubset(dataset, indices, mapping)


def assert_no_real_unknown(labels: np.ndarray | torch.Tensor) -> None:
    values = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if np.any(values < 0):
        raise RuntimeError("real unknown labels entered a training/selection path")
