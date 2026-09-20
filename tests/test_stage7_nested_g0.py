import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.model import Stage7System  # noqa: E402
from maros_stage7.nested_b2 import (  # noqa: E402
    PublicDecisionTable,
    evaluate_public_b2_table,
    fit_nested_public_b2,
)
from maros_stage7.nested_g0 import (  # noqa: E402
    SelectorEvidenceSource,
    SelectorRowBatch,
    assess_registered_g0,
    build_inner_registration_partitions,
    outer_diagnostic_selector_evidence,
    run_nested_g0_registration,
)
from maros_stage7.splits import build_nested_lco_protocol  # noqa: E402
from maros_stage7.training import import_b2_transfer_bundle  # noqa: E402


class TinyIQ(Dataset):
    def __init__(self, labels, seed):
        self.y = np.asarray(labels, dtype=np.int64)
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(len(self.y), 2, 24)).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.tensor(int(self.y[index]))


def _protocol():
    # Eight validation samples per class leave enough rows for a disjoint
    # threshold/meta allocation in every inner episode.
    labels = np.repeat(np.arange(8), 8)
    splits = SimpleNamespace(
        train=TinyIQ(labels, 1), val=TinyIQ(labels, 2),
        test_known=TinyIQ(labels, 3),
        test_unknown=TinyIQ(np.full(10, -1), 4), num_known=8)
    return build_nested_lco_protocol(
        splits, outer_folds=2, inner_folds=4, seed=2026)


def _system(classes):
    return Stage7System(
        classes, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)


def _public_table(inner, role, classes, *, rows=8, source="train", seed=0):
    rng = np.random.default_rng(1000 + 17 * inner + seed)
    waveform = rng.normal(size=(rows, classes)).astype(np.float32)
    prototype = rng.normal(size=(rows, classes)).astype(np.float32)
    waveform_order = np.argsort(waveform, axis=1)
    prototype_order = np.argsort(prototype, axis=1)
    unknown = role.endswith("proxy_unknown")
    labels = (np.full(rows, -1, dtype=np.int64) if unknown else
              np.arange(rows, dtype=np.int64) % classes)
    # Raw train indices may overlap across producers.  The producer id is part
    # of a nested public row's identity; within one producer they remain unique.
    offset = 100 if unknown else 0
    keys = tuple(f"{source}:{offset + index}" for index in range(rows))
    kwargs = {}
    for name, logits, order in (
        ("waveform", waveform, waveform_order),
        ("prototype", prototype, prototype_order),
    ):
        kwargs.update({
            f"{name}_logits": logits,
            f"{name}_unknown": rng.normal(size=rows).astype(np.float32),
            f"{name}_reliability": rng.random(rows).astype(np.float32),
            f"{name}_top1": order[:, -1].astype(np.int64),
            f"{name}_top2": order[:, -2].astype(np.int64),
            f"{name}_summary": rng.normal(size=(rows, 7)).astype(np.float32),
        })
    return PublicDecisionTable(
        outer_index=0, inner_index=inner, split_role=role,
        source_split=source, purpose=f"outer0_inner{inner}_{role}",
        num_classes=classes, labels=labels,
        original_labels=np.arange(rows, dtype=np.int64) % max(classes, 1),
        sample_keys=keys, **kwargs)


def test_nested_partitions_use_validation_meta_and_never_outer_test():
    outer = _protocol().outer_folds[0]
    partitions = build_inner_registration_partitions(
        outer, seed=19, calibration_fraction=0.5)

    assert len(partitions) == 4
    selector_keys = []
    outer_test = set(outer.test_known.sample_keys) | set(
        outer.test_proxy_unknown.sample_keys)
    for partition in partitions:
        assert partition.threshold_calibration_known.source_split == "calibration"
        assert partition.meta_known.source_split == "calibration"
        assert partition.meta_proxy_unknown.source_split == "calibration"
        assert not outer_test.intersection(partition.active_sample_keys)
        assert not any(key.startswith("test_known:")
                       for key in partition.active_sample_keys)
        selector_keys.extend(partition.selector_sample_keys)
    assert len(selector_keys) == len(set(selector_keys))


def test_nested_registration_stops_before_inner_work_and_outer_diagnostic_cannot_pass():
    outer = _protocol().outer_folds[0]
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("inner work ran after failed G0a")

    stopped = run_nested_g0_registration(
        outer, outer_g0a_passed=False, system_factory=forbidden,
        train_inner_system=forbidden, collect_selector_rows=forbidden)
    assert stopped.stopped_after_g0a
    assert not stopped.registered_decision.passed
    assert calls == []

    diagnostic = outer_diagnostic_selector_evidence(
        outer.test_proxy_unknown.sample_keys)
    assert diagnostic.source is SelectorEvidenceSource.OUTER_TEST_DIAGNOSTIC
    assert not assess_registered_g0(
        outer_g0a_passed=True, selector_evidence=diagnostic).passed


def test_nested_registration_creates_four_fresh_class_specific_systems():
    outer = _protocol().outer_folds[0]
    created, trained = [], []

    def factory(num_classes, outer_index, inner_index):
        system = SimpleNamespace(
            num_classes=num_classes, outer_index=outer_index,
            inner_index=inner_index)
        created.append(system)
        return system

    def train(system, partition):
        trained.append((id(system), partition.inner_index))

    def collect(system, partition):
        rows = len(partition.selector_sample_keys)
        return SelectorRowBatch(
            producer_inner_index=partition.inner_index,
            sample_keys=partition.selector_sample_keys,
            public_features=np.ones((rows, 3), dtype=np.float32),
            action_losses=np.tile(
                np.asarray([[0.2, 0.1]], dtype=np.float32), (rows, 1)))

    run = run_nested_g0_registration(
        outer, outer_g0a_passed=True, system_factory=factory,
        train_inner_system=train, collect_selector_rows=collect,
        seed=29)
    assert len(created) == len(trained) == len(run.partitions) == 4
    assert len({id(system) for system in created}) == 4
    assert [system.num_classes for system in created] == [
        len(partition.known_classes) for partition in run.partitions]
    assert run.selector_evidence is not None
    assert run.selector_evidence.registrable
    assert run.registered_decision.passed
    for fold in run.selector_evidence.folds:
        assert all(row.producer_inner_index != fold.heldout_inner_index
                   for row in fold.fit_rows)
        assert all(row.producer_inner_index == fold.heldout_inner_index
                   for row in fold.evaluation_rows)


def test_public_table_roundtrip_and_nested_b2_transfer_across_class_counts(tmp_path):
    tables = []
    for inner, classes in ((0, 2), (1, 3)):
        tables.extend([
            _public_table(inner, "train_known", classes, seed=1),
            _public_table(inner, "train_proxy_unknown", classes, seed=2),
        ])
    path = tables[0].save(tmp_path / "public.npz")
    restored = PublicDecisionTable.load(path)
    assert restored.sample_keys == tables[0].sample_keys
    assert np.array_equal(restored.waveform_logits, tables[0].waveform_logits)

    carrier = _system(2)
    fit = fit_nested_public_b2(
        carrier, tables,
        {"nested_b2_epochs": 1, "batch_size": 4, "b2_lr": 1e-3,
         "weight_decay": 0.0, "proxy_unknown_weight": 1.0},
        torch.device("cpu"), 41)
    assert fit.audit["public_only"] is True
    assert fit.audit["contains_private_state"] is False
    assert fit.audit["inner_indices"] == [0, 1]

    target = _system(4)
    transfer = import_b2_transfer_bundle(target, fit.bundle)
    assert transfer["class_count_changed"] is True
    meta = _public_table(
        2, "meta_known", 4, source="calibration", seed=3)
    prediction = evaluate_public_b2_table(
        target, meta, torch.device("cpu"), batch_size=3)
    assert prediction["pred"].shape == (len(meta),)
    assert prediction["known_weights"].shape == (len(meta), 2)
    assert prediction["public_features"].shape == (len(meta), 17)


def test_nested_public_table_rejects_test_source():
    with pytest.raises(RuntimeError, match="test/formal"):
        _public_table(
            0, "meta_known", 2, source="test_known", seed=9)
