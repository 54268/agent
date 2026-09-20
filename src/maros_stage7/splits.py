"""Leakage-safe nested class hold-outs for Stage-7.

The important distinction in this module is *provenance*, not merely the
numeric value of a label.  Inner/outer held-out classes are legitimate proxy
unknowns, whereas the dataset's formal unknown partition is never legal in a
training, calibration, model-selection, or early-stopping call.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAL_UNKNOWN_PURPOSE = "formal_unknown"


@dataclass(frozen=True)
class ClassHoldoutFold:
    index: int
    support_classes: tuple[int, ...]
    heldout_classes: tuple[int, ...]


class ProvenanceSubset(Dataset):
    """Subset that keeps immutable sample provenance and class identity.

    ``sample_keys`` are namespaced by the source split so equal integer
    indices from train/validation/test are not mistaken for the same sample.
    Proxy unknowns retain their original class in ``original_labels`` while
    returning ``-1`` from ``__getitem__``.
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int] | np.ndarray,
        *,
        source_split: str,
        purpose: str,
        class_mapping: Mapping[int, int] | None = None,
        force_unknown: bool = False,
        formal_unknown: bool = False,
    ) -> None:
        if not hasattr(dataset, "y"):
            raise TypeError("Stage-7 datasets must expose a 1-D 'y' array")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.source_split = str(source_split)
        self.purpose = str(purpose)
        self.class_mapping = (None if class_mapping is None else
                              {int(k): int(v) for k, v in class_mapping.items()})
        self.force_unknown = bool(force_unknown)
        self.formal_unknown = bool(formal_unknown)
        source_labels = np.asarray(dataset.y, dtype=np.int64).reshape(-1)
        if np.any(self.indices < 0) or np.any(self.indices >= len(source_labels)):
            raise IndexError("subset indices are outside the source dataset")
        self.original_labels = source_labels[self.indices].copy()
        if self.class_mapping is not None:
            missing = set(int(x) for x in np.unique(self.original_labels)) - set(self.class_mapping)
            if missing:
                raise ValueError(f"class mapping is missing labels {sorted(missing)}")
        self.sample_keys = tuple(
            f"{self.source_split}:{int(index)}" for index in self.indices)

    @property
    def y(self) -> np.ndarray:
        if self.force_unknown:
            return np.full(len(self), -1, dtype=np.int64)
        if self.class_mapping is None:
            return self.original_labels.copy()
        return np.asarray(
            [self.class_mapping[int(label)] for label in self.original_labels],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        x, _ = self.dataset[int(self.indices[index])]
        return x, torch.tensor(int(self.y[index]), dtype=torch.long)


@dataclass(frozen=True)
class InnerEpisode:
    outer_index: int
    inner_index: int
    known_classes: tuple[int, ...]
    proxy_unknown_classes: tuple[int, ...]
    train_known: ProvenanceSubset
    train_proxy_unknown: ProvenanceSubset
    calibration_known: ProvenanceSubset
    validation_proxy_unknown: ProvenanceSubset
    test_known: ProvenanceSubset
    test_proxy_unknown: ProvenanceSubset


@dataclass(frozen=True)
class OuterEpisode:
    index: int
    known_classes: tuple[int, ...]
    proxy_unknown_classes: tuple[int, ...]
    train_known: ProvenanceSubset
    calibration_known: ProvenanceSubset
    test_known: ProvenanceSubset
    test_proxy_unknown: ProvenanceSubset
    inner_folds: tuple[InnerEpisode, ...]


@dataclass(frozen=True)
class NestedLCOProtocol:
    seed: int
    num_classes: int
    outer_folds: tuple[OuterEpisode, ...]
    formal_unknown_sample_count: int


def build_class_folds(
    classes: int | Iterable[int],
    num_folds: int,
    seed: int,
) -> list[ClassHoldoutFold]:
    """Build deterministic, balanced class folds without requiring divisibility."""
    values = (np.arange(classes, dtype=np.int64) if isinstance(classes, int)
              else np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64))
    if num_folds < 2:
        raise ValueError("num_folds must be at least two")
    if len(values) < num_folds:
        raise ValueError("num_folds cannot exceed the number of classes")
    rng = np.random.default_rng(int(seed))
    shuffled = values.copy()
    rng.shuffle(shuffled)
    groups = np.array_split(shuffled, num_folds)
    universe = set(int(x) for x in values)
    folds = []
    for index, group in enumerate(groups):
        heldout = tuple(sorted(int(x) for x in group))
        support = tuple(sorted(universe - set(heldout)))
        folds.append(ClassHoldoutFold(index, support, heldout))
    validate_class_folds(folds, tuple(sorted(universe)))
    return folds


def build_inner_folds(
    outer_support_classes: Iterable[int],
    num_folds: int = 4,
    seed: int = 2026,
) -> list[ClassHoldoutFold]:
    """Public helper requested by the experiment runner."""
    return build_class_folds(outer_support_classes, num_folds, seed)


def validate_class_folds(
    folds: Sequence[ClassHoldoutFold], universe: Sequence[int]
) -> None:
    expected = set(int(x) for x in universe)
    held_counts = {value: 0 for value in expected}
    for fold in folds:
        support, heldout = set(fold.support_classes), set(fold.heldout_classes)
        if support & heldout or support | heldout != expected:
            raise ValueError(f"invalid class partition in fold {fold.index}")
        for value in heldout:
            held_counts[value] += 1
    if any(count != 1 for count in held_counts.values()):
        raise ValueError("each class must be held out exactly once")


def _subset(
    dataset: Dataset,
    classes: Iterable[int] | None,
    *,
    source_split: str,
    purpose: str,
    mapping: Mapping[int, int] | None = None,
    force_unknown: bool = False,
    formal_unknown: bool = False,
) -> ProvenanceSubset:
    labels = np.asarray(dataset.y, dtype=np.int64).reshape(-1)
    if classes is None:
        indices = np.arange(len(labels), dtype=np.int64)
    else:
        indices = np.flatnonzero(np.isin(labels, list(classes)))
    return ProvenanceSubset(
        dataset, indices, source_split=source_split, purpose=purpose,
        class_mapping=mapping, force_unknown=force_unknown,
        formal_unknown=formal_unknown,
    )


def split_known_calibration_test(
    splits,
    known_classes: Iterable[int],
    *,
    mapping: Mapping[int, int] | None = None,
    prefix: str = "outer",
) -> tuple[ProvenanceSubset, ProvenanceSubset, ProvenanceSubset]:
    """Return sample-disjoint train/calibration/test partitions for classes."""
    classes = tuple(sorted(int(x) for x in known_classes))
    train = _subset(
        splits.train, classes, source_split="train", purpose=f"{prefix}_train_known",
        mapping=mapping,
    )
    calibration = _subset(
        splits.val, classes, source_split="calibration",
        purpose=f"{prefix}_calibration_known", mapping=mapping,
    )
    test = _subset(
        splits.test_known, classes, source_split="test_known",
        purpose=f"{prefix}_test_known", mapping=mapping,
    )
    assert_sample_disjoint(train, calibration, test)
    return train, calibration, test


def _validate_source_splits(splits) -> None:
    required = ("train", "val", "test_known", "test_unknown", "num_known")
    missing = [name for name in required if not hasattr(splits, name)]
    if missing:
        raise TypeError(f"open-set splits missing fields: {missing}")
    for name in ("train", "val", "test_known"):
        labels = np.asarray(getattr(splits, name).y, dtype=np.int64)
        if np.any(labels < 0):
            raise ValueError(f"formal unknown label found in known source split '{name}'")
    if len({id(splits.train), id(splits.val), id(splits.test_known)}) != 3:
        raise ValueError("train, calibration and known-test must be distinct datasets")


def build_nested_lco_protocol(
    splits,
    *,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 2026,
) -> NestedLCOProtocol:
    """Construct the complete outer-LCO/inner-episodic Stage-7 protocol.

    Formal unknown data are deliberately counted for the manifest but never
    wrapped or returned by any training/selection episode.
    """
    _validate_source_splits(splits)
    num_classes = int(splits.num_known)
    outer_defs = build_class_folds(num_classes, outer_folds, seed)
    outer_episodes = []
    for outer in outer_defs:
        outer_mapping = {value: index for index, value in enumerate(outer.support_classes)}
        train, calibration, test = split_known_calibration_test(
            splits, outer.support_classes, mapping=outer_mapping,
            prefix=f"outer{outer.index}",
        )
        proxy_test = _subset(
            splits.test_known, outer.heldout_classes,
            source_split="test_known", purpose=f"outer{outer.index}_proxy_unknown",
            force_unknown=True,
        )
        inner_defs = build_inner_folds(
            outer.support_classes, inner_folds,
            seed=int(seed) + 1009 * (outer.index + 1),
        )
        episodes = []
        for inner in inner_defs:
            mapping = {value: index for index, value in enumerate(inner.support_classes)}
            prefix = f"outer{outer.index}_inner{inner.index}"
            ik_train, ik_cal, ik_test = split_known_calibration_test(
                splits, inner.support_classes, mapping=mapping, prefix=prefix)
            train_proxy = _subset(
                splits.train, inner.heldout_classes, source_split="train",
                purpose=f"{prefix}_train_proxy_unknown", force_unknown=True)
            val_proxy = _subset(
                splits.val, inner.heldout_classes, source_split="calibration",
                purpose=f"{prefix}_validation_proxy_unknown", force_unknown=True)
            test_proxy = _subset(
                splits.test_known, inner.heldout_classes, source_split="test_known",
                purpose=f"{prefix}_test_proxy_unknown", force_unknown=True)
            episode = InnerEpisode(
                outer.index, inner.index, inner.support_classes,
                inner.heldout_classes, ik_train, train_proxy, ik_cal,
                val_proxy, ik_test, test_proxy,
            )
            validate_episode_isolation(episode)
            episodes.append(episode)
        outer_episode = OuterEpisode(
            outer.index, outer.support_classes, outer.heldout_classes,
            train, calibration, test, proxy_test, tuple(episodes),
        )
        validate_episode_isolation(outer_episode)
        outer_episodes.append(outer_episode)
    protocol = NestedLCOProtocol(
        int(seed), num_classes, tuple(outer_episodes),
        int(len(splits.test_unknown)),
    )
    validate_protocol_isolation(protocol)
    return protocol


def _walk_datasets(value) -> Iterator[ProvenanceSubset]:
    if isinstance(value, ProvenanceSubset):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _walk_datasets(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_datasets(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_datasets(item)


def assert_no_formal_unknown(*values) -> None:
    """Fail closed if a formal-unknown partition enters a non-final path."""
    for value in values:
        if isinstance(value, ProvenanceSubset) and value.formal_unknown:
            raise RuntimeError(
                f"formal unknown data entered '{value.purpose}'")
        for dataset in _walk_datasets(value):
            if dataset.formal_unknown or dataset.purpose == FORMAL_UNKNOWN_PURPOSE:
                raise RuntimeError(
                    f"formal unknown data entered '{dataset.purpose}'")


def assert_sample_disjoint(*datasets: ProvenanceSubset) -> None:
    seen: set[str] = set()
    for dataset in datasets:
        overlap = seen.intersection(dataset.sample_keys)
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"sample leakage across partitions at {example}")
        seen.update(dataset.sample_keys)


def validate_episode_isolation(episode: InnerEpisode | OuterEpisode) -> None:
    known, heldout = set(episode.known_classes), set(episode.proxy_unknown_classes)
    if known & heldout:
        raise ValueError("known and proxy-unknown classes overlap")
    # Validate only the datasets owned by this episode.  An ``OuterEpisode``
    # contains nested InnerEpisodes whose proxy-unknown classes are, by
    # construction, members of the *outer* known set.  Recursively comparing
    # those inner datasets with the outer held-out set incorrectly reports a
    # leak even though the two class roles are intentionally different.
    if isinstance(episode, InnerEpisode):
        datasets = [
            episode.train_known, episode.train_proxy_unknown,
            episode.calibration_known, episode.validation_proxy_unknown,
            episode.test_known, episode.test_proxy_unknown,
        ]
    else:
        datasets = [
            episode.train_known, episode.calibration_known,
            episode.test_known, episode.test_proxy_unknown,
        ]
    assert_no_formal_unknown(*datasets)
    for dataset in datasets:
        original = set(int(x) for x in np.unique(dataset.original_labels))
        expected = heldout if dataset.force_unknown else known
        if not original.issubset(expected):
            raise ValueError(
                f"{dataset.purpose} contains classes outside its declared role")
    if isinstance(episode, InnerEpisode):
        assert_sample_disjoint(episode.train_known, episode.train_proxy_unknown)
        assert_sample_disjoint(
            episode.calibration_known, episode.validation_proxy_unknown)
        assert_sample_disjoint(episode.test_known, episode.test_proxy_unknown)
    else:
        assert_sample_disjoint(episode.test_known, episode.test_proxy_unknown)


def validate_protocol_isolation(protocol: NestedLCOProtocol) -> None:
    assert_no_formal_unknown(protocol)
    if len(protocol.outer_folds) < 2:
        raise ValueError("nested LCO requires at least two outer folds")
    outer_held_counts = {value: 0 for value in range(protocol.num_classes)}
    for outer in protocol.outer_folds:
        validate_episode_isolation(outer)
        for value in outer.proxy_unknown_classes:
            outer_held_counts[value] += 1
        if len(outer.inner_folds) != 4:
            raise ValueError("Stage-7 requires exactly four inner folds")
        inner_held_counts = {value: 0 for value in outer.known_classes}
        for inner in outer.inner_folds:
            validate_episode_isolation(inner)
            for value in inner.proxy_unknown_classes:
                inner_held_counts[value] += 1
        if any(count != 1 for count in inner_held_counts.values()):
            raise ValueError("each outer-support class must be inner-held exactly once")
    if any(count != 1 for count in outer_held_counts.values()):
        raise ValueError("each known class must be outer-held exactly once")


def protocol_manifest(protocol: NestedLCOProtocol) -> dict:
    """JSON-serialisable audit trail (never includes formal unknown IDs)."""
    return {
        "seed": protocol.seed,
        "num_classes": protocol.num_classes,
        "formal_unknown_sample_count": protocol.formal_unknown_sample_count,
        "formal_unknown_used_for_training_selection_or_calibration": False,
        "outer_folds": [
            {
                "index": outer.index,
                "known_classes": list(outer.known_classes),
                "proxy_unknown_classes": list(outer.proxy_unknown_classes),
                "counts": {
                    "train_known": len(outer.train_known),
                    "calibration_known": len(outer.calibration_known),
                    "test_known": len(outer.test_known),
                    "test_proxy_unknown": len(outer.test_proxy_unknown),
                },
                "inner_folds": [
                    {
                        "index": inner.inner_index,
                        "known_classes": list(inner.known_classes),
                        "proxy_unknown_classes": list(inner.proxy_unknown_classes),
                        "counts": {
                            name: len(getattr(inner, name)) for name in (
                                "train_known", "train_proxy_unknown",
                                "calibration_known", "validation_proxy_unknown",
                                "test_known", "test_proxy_unknown",
                            )
                        },
                    }
                    for inner in outer.inner_folds
                ],
            }
            for outer in protocol.outer_folds
        ],
    }
