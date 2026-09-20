"""Fail-closed orchestration for the registered Stage-7 nested G0 audit.

This module intentionally contains no neural-network training code.  It
defines the ownership and provenance contract that an experiment runner must
satisfy before a public-summary selector can count toward registered G0.

The outer model is first used only for the cheap G0a local-capability audit.
If that audit fails, no inner system is constructed.  If it passes, four
fresh, class-count-specific systems are trained.  Their selector examples are
drawn exclusively from the validation source; the outer known test set and
the formal unknown set are never exposed to the factory, trainer, collector,
or selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .splits import (InnerEpisode, OuterEpisode, ProvenanceSubset,
                     assert_no_formal_unknown, assert_sample_disjoint)


REGISTERED_INNER_FOLDS = 4


class SelectorEvidenceSource(str, Enum):
    """Whether selector evidence is eligible for the registered G0 gate."""

    INNER_EPISODE_OOF = "inner_episode_oof"
    OUTER_TEST_DIAGNOSTIC = "outer_test_diagnostic"


@dataclass(frozen=True)
class InnerRegistrationPartition:
    """The only datasets an inner registration callback is allowed to see.

    ``threshold_calibration_known`` and the two ``meta_*`` datasets are
    deterministic, sample-disjoint partitions of the validation source.
    Test-source datasets are deliberately not fields of this object.
    """

    outer_index: int
    inner_index: int
    known_classes: tuple[int, ...]
    proxy_unknown_classes: tuple[int, ...]
    train_known: ProvenanceSubset
    train_proxy_unknown: ProvenanceSubset
    threshold_calibration_known: ProvenanceSubset
    meta_known: ProvenanceSubset
    meta_proxy_unknown: ProvenanceSubset

    @property
    def num_classes(self) -> int:
        return len(self.known_classes)

    @property
    def selector_sample_keys(self) -> tuple[str, ...]:
        return self.meta_known.sample_keys + self.meta_proxy_unknown.sample_keys

    @property
    def active_sample_keys(self) -> tuple[str, ...]:
        return (
            self.train_known.sample_keys
            + self.train_proxy_unknown.sample_keys
            + self.threshold_calibration_known.sample_keys
            + self.selector_sample_keys
        )


@dataclass(frozen=True)
class SelectorRowBatch:
    """Public selector rows produced by exactly one inner system.

    ``public_features`` and ``action_losses`` are class-count-independent;
    their second dimensions may vary with the public feature contract but
    their row counts must match ``sample_keys``.  No private state is carried.
    """

    producer_inner_index: int
    sample_keys: tuple[str, ...]
    public_features: np.ndarray
    action_losses: np.ndarray


@dataclass(frozen=True)
class SelectorRowLocation:
    producer_inner_index: int
    row_index: int
    sample_key: str


@dataclass(frozen=True)
class SelectorCrossFitFold:
    """One leave-one-inner-episode-out selector fold."""

    heldout_inner_index: int
    fit_rows: tuple[SelectorRowLocation, ...]
    evaluation_rows: tuple[SelectorRowLocation, ...]


@dataclass(frozen=True)
class SelectorEvidence:
    source: SelectorEvidenceSource
    folds: tuple[SelectorCrossFitFold, ...]
    sample_keys: tuple[str, ...]
    note: str

    @property
    def registrable(self) -> bool:
        return self.source is SelectorEvidenceSource.INNER_EPISODE_OOF


@dataclass(frozen=True)
class RegisteredG0Decision:
    passed: bool
    reason: str


@dataclass(frozen=True)
class NestedG0Run:
    """Protocol audit returned to the experiment orchestration layer."""

    outer_index: int
    stopped_after_g0a: bool
    partitions: tuple[InnerRegistrationPartition, ...]
    systems: tuple[Any, ...]
    row_batches: tuple[SelectorRowBatch, ...]
    selector_evidence: SelectorEvidence | None
    registered_decision: RegisteredG0Decision


def _subset_from_source_indices(
    parent: ProvenanceSubset,
    source_indices: Iterable[int],
    *,
    purpose: str,
) -> ProvenanceSubset:
    allowed = set(int(value) for value in source_indices)
    mask = np.asarray(
        [int(value) in allowed for value in parent.indices], dtype=bool)
    return ProvenanceSubset(
        parent.dataset,
        parent.indices[mask],
        source_split=parent.source_split,
        purpose=purpose,
        class_mapping=parent.class_mapping,
        force_unknown=parent.force_unknown,
        formal_unknown=parent.formal_unknown,
    )


def _calibration_and_meta_indices(
    outer: OuterEpisode,
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[set[int], dict[int, tuple[int, str]]]:
    """Assign every active validation sample one globally unique role.

    The split is made once per outer fold, not independently per inner fold.
    This is essential: an identical validation sample must not become a
    threshold-calibration row for one inner system and a selector row for
    another.  Meta samples are divided between the class's held-out episode
    (proxy-unknown role) and one support episode (known role).
    """

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    heldout_owner: dict[int, int] = {}
    support_owners: dict[int, list[int]] = {
        int(value): [] for value in outer.known_classes}
    for inner in outer.inner_folds:
        for value in inner.proxy_unknown_classes:
            value = int(value)
            if value in heldout_owner:
                raise ValueError(f"class {value} is held out by multiple inner episodes")
            heldout_owner[value] = int(inner.inner_index)
        for value in inner.known_classes:
            support_owners[int(value)].append(int(inner.inner_index))

    labels = np.asarray(outer.calibration_known.original_labels, dtype=np.int64)
    indices = np.asarray(outer.calibration_known.indices, dtype=np.int64)
    calibration: set[int] = set()
    # source index -> (owning inner episode, "known" | "proxy_unknown")
    meta_owner: dict[int, tuple[int, str]] = {}
    for class_id in outer.known_classes:
        class_id = int(class_id)
        class_indices = indices[labels == class_id].copy()
        if len(class_indices) < 2:
            raise ValueError(
                "nested G0 needs at least two validation samples per outer-support class")
        rng = np.random.default_rng(
            int(seed) + 104729 * (int(outer.index) + 1) + 1009 * (class_id + 1))
        rng.shuffle(class_indices)
        cut = int(round(len(class_indices) * float(calibration_fraction)))
        cut = min(max(cut, 1), len(class_indices) - 1)
        calibration.update(int(value) for value in class_indices[:cut])
        meta = class_indices[cut:]

        proxy_count = 1 if len(meta) == 1 else max(1, len(meta) // 2)
        proxy_owner = heldout_owner[class_id]
        for value in meta[:proxy_count]:
            meta_owner[int(value)] = (proxy_owner, "proxy_unknown")
        known_owners = sorted(support_owners[class_id])
        if not known_owners:
            raise ValueError(f"class {class_id} has no inner support episode")
        for offset, value in enumerate(meta[proxy_count:]):
            owner = known_owners[offset % len(known_owners)]
            meta_owner[int(value)] = (owner, "known")
    return calibration, meta_owner


def _validate_registration_partition(
    partition: InnerRegistrationPartition,
    *,
    forbidden_sample_keys: set[str],
) -> None:
    datasets = (
        partition.train_known,
        partition.train_proxy_unknown,
        partition.threshold_calibration_known,
        partition.meta_known,
        partition.meta_proxy_unknown,
    )
    assert_no_formal_unknown(*datasets)
    assert_sample_disjoint(*datasets)
    if partition.train_known.source_split != "train":
        raise ValueError("inner Known training data must come from the train source")
    if partition.train_proxy_unknown.source_split != "train":
        raise ValueError("inner proxy training data must come from the train source")
    for dataset in datasets[2:]:
        if dataset.source_split != "calibration":
            raise ValueError("inner calibration/meta data must come from validation")
    leaked = forbidden_sample_keys.intersection(partition.active_sample_keys)
    if leaked:
        raise RuntimeError(
            f"outer-test or formal-unknown sample entered nested G0: {sorted(leaked)[0]}")
    if any(key.startswith(("test_known:", "test_unknown:"))
           for key in partition.active_sample_keys):
        raise RuntimeError("test-source sample entered registered nested G0")


def build_inner_registration_partitions(
    outer: OuterEpisode,
    *,
    seed: int = 2026,
    calibration_fraction: float = 0.5,
    formal_unknown_sample_keys: Iterable[str] = (),
) -> tuple[InnerRegistrationPartition, ...]:
    """Build four leakage-safe registration partitions for one outer fold.

    Inner ``test_known`` and ``test_proxy_unknown`` wrappers are intentionally
    ignored.  They may be used later for a clearly-labelled diagnostic, but
    never for model/selector choice or registered G0.
    """

    if len(outer.inner_folds) != REGISTERED_INNER_FOLDS:
        raise ValueError(
            f"registered Stage-7 G0 requires exactly {REGISTERED_INNER_FOLDS} inner folds")
    calibration_indices, meta_owner = _calibration_and_meta_indices(
        outer, seed=int(seed), calibration_fraction=float(calibration_fraction))
    outer_test_keys = set(outer.test_known.sample_keys) | set(
        outer.test_proxy_unknown.sample_keys)
    forbidden = outer_test_keys | set(str(value) for value in formal_unknown_sample_keys)

    partitions = []
    for inner in sorted(outer.inner_folds, key=lambda item: item.inner_index):
        inner_index = int(inner.inner_index)
        meta_known_indices = {
            index for index, (owner, role) in meta_owner.items()
            if owner == inner_index and role == "known"}
        meta_proxy_indices = {
            index for index, (owner, role) in meta_owner.items()
            if owner == inner_index and role == "proxy_unknown"}
        prefix = f"outer{outer.index}_inner{inner_index}_registered"
        partition = InnerRegistrationPartition(
            outer_index=int(outer.index),
            inner_index=inner_index,
            known_classes=tuple(int(value) for value in inner.known_classes),
            proxy_unknown_classes=tuple(
                int(value) for value in inner.proxy_unknown_classes),
            train_known=inner.train_known,
            train_proxy_unknown=inner.train_proxy_unknown,
            threshold_calibration_known=_subset_from_source_indices(
                inner.calibration_known, calibration_indices,
                purpose=f"{prefix}_threshold_calibration_known"),
            meta_known=_subset_from_source_indices(
                inner.calibration_known, meta_known_indices,
                purpose=f"{prefix}_meta_known"),
            meta_proxy_unknown=_subset_from_source_indices(
                inner.validation_proxy_unknown, meta_proxy_indices,
                purpose=f"{prefix}_meta_proxy_unknown"),
        )
        _validate_registration_partition(
            partition, forbidden_sample_keys=forbidden)
        partitions.append(partition)

    # A raw validation sample receives exactly one selector owner, which makes
    # the leave-one-episode-out fit/evaluation sides sample-disjoint.
    selector_keys = [
        key for partition in partitions for key in partition.selector_sample_keys]
    if len(selector_keys) != len(set(selector_keys)):
        raise RuntimeError("a selector sample is owned by multiple inner episodes")
    return tuple(partitions)


def _validate_row_batch(
    batch: SelectorRowBatch,
    partition: InnerRegistrationPartition,
) -> None:
    if int(batch.producer_inner_index) != int(partition.inner_index):
        raise RuntimeError(
            "OOF selector rows were attributed to a system other than their producer")
    expected = tuple(partition.selector_sample_keys)
    if tuple(batch.sample_keys) != expected:
        raise RuntimeError(
            "selector collector must return every assigned meta row exactly once and in order")
    features = np.asarray(batch.public_features)
    losses = np.asarray(batch.action_losses)
    if features.ndim != 2 or losses.ndim != 2:
        raise ValueError("selector public_features and action_losses must be matrices")
    if len(features) != len(expected) or len(losses) != len(expected):
        raise ValueError("selector matrices and sample keys have different row counts")
    if losses.shape[1] < 2:
        raise ValueError("selector action_losses must contain at least two actions")
    if not np.isfinite(features).all() or not np.isfinite(losses).all():
        raise ValueError("selector rows contain non-finite values")


def build_selector_crossfit_folds(
    row_batches: Sequence[SelectorRowBatch],
) -> tuple[SelectorCrossFitFold, ...]:
    """Leave out one *inner episode*, never an arbitrary row-level split."""

    ordered = tuple(sorted(row_batches, key=lambda item: item.producer_inner_index))
    indices = [int(batch.producer_inner_index) for batch in ordered]
    if len(ordered) != REGISTERED_INNER_FOLDS or len(set(indices)) != len(indices):
        raise ValueError("OOF selector evidence requires four distinct inner producers")
    all_locations = {
        index: tuple(
            SelectorRowLocation(index, row, key)
            for row, key in enumerate(batch.sample_keys))
        for index, batch in zip(indices, ordered)
    }
    folds = []
    for heldout in indices:
        evaluation = all_locations[heldout]
        fit = tuple(
            row for producer in indices if producer != heldout
            for row in all_locations[producer])
        fit_keys = {row.sample_key for row in fit}
        evaluation_keys = {row.sample_key for row in evaluation}
        if fit_keys & evaluation_keys:
            raise RuntimeError("OOF selector fit and evaluation samples overlap")
        if any(row.producer_inner_index == heldout for row in fit):
            raise RuntimeError("held-out inner episode leaked into selector fitting")
        if any(row.producer_inner_index != heldout for row in evaluation):
            raise RuntimeError("selector evaluation contains a non-held-out producer")
        folds.append(SelectorCrossFitFold(heldout, fit, evaluation))
    return tuple(folds)


def inner_oof_selector_evidence(
    row_batches: Sequence[SelectorRowBatch],
) -> SelectorEvidence:
    folds = build_selector_crossfit_folds(row_batches)
    keys = tuple(
        row.sample_key for fold in folds for row in fold.evaluation_rows)
    if len(keys) != len(set(keys)):
        raise RuntimeError("a raw selector sample was evaluated more than once")
    return SelectorEvidence(
        source=SelectorEvidenceSource.INNER_EPISODE_OOF,
        folds=folds,
        sample_keys=keys,
        note="four fresh inner systems; leave-one-inner-episode-out evaluation",
    )


def outer_diagnostic_selector_evidence(
    sample_keys: Iterable[str],
) -> SelectorEvidence:
    """Label a selector measured on the outer test partition as diagnostic."""

    return SelectorEvidence(
        source=SelectorEvidenceSource.OUTER_TEST_DIAGNOSTIC,
        folds=(),
        sample_keys=tuple(str(value) for value in sample_keys),
        note=("outer-test cross-fitting is a complementarity ceiling only; "
              "it cannot satisfy registered G0"),
    )


def assess_registered_g0(
    *,
    outer_g0a_passed: bool,
    selector_evidence: SelectorEvidence | None,
) -> RegisteredG0Decision:
    """Fail closed unless G0a passed and genuine inner OOF evidence exists."""

    if not bool(outer_g0a_passed):
        return RegisteredG0Decision(
            False, "outer G0a local capability/complementarity audit failed")
    if selector_evidence is None:
        return RegisteredG0Decision(False, "inner OOF selector evidence is missing")
    if not selector_evidence.registrable:
        return RegisteredG0Decision(
            False, "outer-test diagnostic selector is not registered G0 evidence")
    if len(selector_evidence.folds) != REGISTERED_INNER_FOLDS:
        return RegisteredG0Decision(False, "inner OOF evidence is incomplete")
    expected = {fold.heldout_inner_index for fold in selector_evidence.folds}
    if len(expected) != REGISTERED_INNER_FOLDS:
        return RegisteredG0Decision(False, "inner OOF held-out episodes are not unique")
    return RegisteredG0Decision(True, "nested inner OOF protocol is complete")


def run_nested_g0_registration(
    outer: OuterEpisode,
    *,
    outer_g0a_passed: bool,
    system_factory: Callable[[int, int, int], Any],
    train_inner_system: Callable[[Any, InnerRegistrationPartition], Any],
    collect_selector_rows: Callable[
        [Any, InnerRegistrationPartition], SelectorRowBatch],
    seed: int = 2026,
    calibration_fraction: float = 0.5,
    formal_unknown_sample_keys: Iterable[str] = (),
) -> NestedG0Run:
    """Run only the protocol-level part of nested G0.

    Callback signature for ``system_factory`` is
    ``(num_classes, outer_index, inner_index)``.  Returning the same object
    twice, returning a system with a mismatched ``num_classes`` attribute, or
    producing selector rows with the wrong owner fails immediately.
    """

    if not bool(outer_g0a_passed):
        decision = assess_registered_g0(
            outer_g0a_passed=False, selector_evidence=None)
        return NestedG0Run(
            int(outer.index), True, (), (), (), None, decision)

    partitions = build_inner_registration_partitions(
        outer, seed=int(seed), calibration_fraction=float(calibration_fraction),
        formal_unknown_sample_keys=formal_unknown_sample_keys)
    systems: list[Any] = []
    batches: list[SelectorRowBatch] = []
    system_ids: set[int] = set()
    for partition in partitions:
        system = system_factory(
            partition.num_classes, partition.outer_index, partition.inner_index)
        if id(system) in system_ids:
            raise RuntimeError("each inner episode must use a fresh system instance")
        system_ids.add(id(system))
        observed_classes = getattr(system, "num_classes", None)
        if observed_classes is None or int(observed_classes) != partition.num_classes:
            raise RuntimeError(
                "inner system class count does not match its support classes")
        train_inner_system(system, partition)
        batch = collect_selector_rows(system, partition)
        if not isinstance(batch, SelectorRowBatch):
            raise TypeError("selector collector must return SelectorRowBatch")
        _validate_row_batch(batch, partition)
        systems.append(system)
        batches.append(batch)
    evidence = inner_oof_selector_evidence(batches)
    decision = assess_registered_g0(
        outer_g0a_passed=True, selector_evidence=evidence)
    return NestedG0Run(
        int(outer.index), False, partitions, tuple(systems), tuple(batches),
        evidence, decision)


__all__ = [
    "InnerRegistrationPartition", "NestedG0Run", "RegisteredG0Decision",
    "REGISTERED_INNER_FOLDS", "SelectorCrossFitFold", "SelectorEvidence",
    "SelectorEvidenceSource", "SelectorRowBatch", "SelectorRowLocation",
    "assess_registered_g0", "build_inner_registration_partitions",
    "build_selector_crossfit_folds", "inner_oof_selector_evidence",
    "outer_diagnostic_selector_evidence", "run_nested_g0_registration",
]
