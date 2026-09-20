"""Public-only meta fitting for the Stage-7 nested G0 baseline.

The outer class hold-out must not teach the no-communication parent which
local Agent to trust.  This module therefore learns the class-count-invariant
``FairB2Fusion`` service from cached public decisions produced by inner LCO
systems.  Cached tables contain no encoder state, embedding, prototype or
private evidence, and fitting tables are accepted only from the train source.

Threshold calibration and meta evaluation tables may come from the dedicated
validation source.  Test-source and formal-unknown rows are rejected here;
the experiment runner owns the single final outer evaluation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import LocalDecision
from .dialogue import FairB2Fusion
from .router import public_team_features
from .splits import ProvenanceSubset, assert_no_formal_unknown
from .training import (LOCAL_AGENT_NAMES, collect_predictions,
                       export_b2_transfer_bundle, local_agent_sample_losses,
                       set_seed)


FIT_ROLES = frozenset({"train_known", "train_proxy_unknown"})
VALIDATION_ROLES = frozenset({
    "threshold_calibration_known", "meta_known", "meta_proxy_unknown",
})
ALL_ROLES = FIT_ROLES | VALIDATION_ROLES


def _array(value: Any, dtype, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class PublicDecisionTable:
    """Serializable public decisions from one inner system and one split."""

    outer_index: int
    inner_index: int
    split_role: str
    source_split: str
    purpose: str
    num_classes: int
    labels: np.ndarray
    original_labels: np.ndarray
    sample_keys: tuple[str, ...]
    waveform_logits: np.ndarray
    waveform_unknown: np.ndarray
    waveform_reliability: np.ndarray
    waveform_top1: np.ndarray
    waveform_top2: np.ndarray
    waveform_summary: np.ndarray
    prototype_logits: np.ndarray
    prototype_unknown: np.ndarray
    prototype_reliability: np.ndarray
    prototype_top1: np.ndarray
    prototype_top2: np.ndarray
    prototype_summary: np.ndarray

    def __post_init__(self) -> None:
        role = str(self.split_role)
        if role not in ALL_ROLES:
            raise ValueError(f"unsupported public-table role: {role}")
        source = str(self.source_split).lower()
        if source.startswith("test") or "formal_unknown" in str(self.purpose):
            raise RuntimeError("test/formal rows cannot enter nested B2 tables")
        if role in FIT_ROLES and source != "train":
            raise RuntimeError("nested B2 fitting tables must come from train")
        if role in VALIDATION_ROLES and source != "calibration":
            raise RuntimeError("nested B2 validation tables must come from validation")
        classes = int(self.num_classes)
        if classes < 2:
            raise ValueError("nested B2 requires at least two Known classes")

        one_dimensional = {
            "labels": (self.labels, np.int64),
            "original_labels": (self.original_labels, np.int64),
            "waveform_unknown": (self.waveform_unknown, np.float32),
            "waveform_reliability": (self.waveform_reliability, np.float32),
            "waveform_top1": (self.waveform_top1, np.int64),
            "waveform_top2": (self.waveform_top2, np.int64),
            "prototype_unknown": (self.prototype_unknown, np.float32),
            "prototype_reliability": (self.prototype_reliability, np.float32),
            "prototype_top1": (self.prototype_top1, np.int64),
            "prototype_top2": (self.prototype_top2, np.int64),
        }
        matrices = {
            "waveform_logits": (self.waveform_logits, np.float32),
            "waveform_summary": (self.waveform_summary, np.float32),
            "prototype_logits": (self.prototype_logits, np.float32),
            "prototype_summary": (self.prototype_summary, np.float32),
        }
        normalised: dict[str, np.ndarray] = {}
        for name, (value, dtype) in one_dimensional.items():
            normalised[name] = _array(value, dtype, ndim=1, name=name)
        for name, (value, dtype) in matrices.items():
            normalised[name] = _array(value, dtype, ndim=2, name=name)
        rows = len(normalised["labels"])
        if rows < 1 or len(self.sample_keys) != rows:
            raise ValueError("public table must be non-empty and align sample keys")
        if len(set(self.sample_keys)) != rows:
            raise ValueError("public table sample keys must be unique")
        for name, value in normalised.items():
            if len(value) != rows:
                raise ValueError(f"{name} row count does not match labels")
            object.__setattr__(self, name, value)
        if normalised["waveform_logits"].shape[1] != classes or \
                normalised["prototype_logits"].shape[1] != classes:
            raise ValueError("public logits do not match num_classes")
        if normalised["waveform_summary"].shape != \
                normalised["prototype_summary"].shape:
            raise ValueError("Agent public-summary shapes must match")
        if normalised["waveform_summary"].shape[1] < 1:
            raise ValueError("public summaries must expose scalar evidence")
        labels = normalised["labels"]
        expect_unknown = role in {"train_proxy_unknown", "meta_proxy_unknown"}
        if expect_unknown and np.any(labels >= 0):
            raise ValueError("proxy table must expose label -1")
        if not expect_unknown and np.any(labels < 0):
            raise ValueError("Known table cannot expose unknown labels")
        if np.any(labels >= classes):
            raise ValueError("Known labels exceed the inner class count")
        for agent in LOCAL_AGENT_NAMES:
            reliability = normalised[f"{agent}_reliability"]
            if np.any((reliability < 0.0) | (reliability > 1.0)):
                raise ValueError("public reliability must lie in [0,1]")
            top1, top2 = (normalised[f"{agent}_top1"],
                          normalised[f"{agent}_top2"])
            if np.any((top1 < 0) | (top1 >= classes)) or \
                    np.any((top2 < 0) | (top2 >= classes)):
                raise ValueError("public class proposal is outside class range")
            if np.any(top1 == top2):
                raise ValueError("public top1 and top2 must differ")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def summary_dim(self) -> int:
        return int(self.waveform_summary.shape[1])

    def decisions(
        self, indices: np.ndarray | Sequence[int], device: torch.device,
    ) -> dict[str, LocalDecision]:
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(index < 0) or np.any(index >= len(self)):
            raise IndexError("public-table batch index is out of range")

        def tensor(name: str, dtype=None) -> torch.Tensor:
            value = torch.from_numpy(getattr(self, name)[index])
            return value.to(device=device, dtype=dtype)

        return {
            agent: LocalDecision(
                class_logits=tensor(f"{agent}_logits", torch.float32),
                unknown_score=tensor(f"{agent}_unknown", torch.float32),
                reliability=tensor(f"{agent}_reliability", torch.float32),
                top1=tensor(f"{agent}_top1", torch.long),
                top2=tensor(f"{agent}_top2", torch.long),
                public_summary=tensor(f"{agent}_summary", torch.float32),
            ) for agent in LOCAL_AGENT_NAMES
        }

    def label_tensor(
        self, indices: np.ndarray | Sequence[int], device: torch.device,
    ) -> torch.Tensor:
        index = np.asarray(indices, dtype=np.int64).reshape(-1)
        return torch.from_numpy(self.labels[index]).to(device=device, dtype=torch.long)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "outer_index": int(self.outer_index),
            "inner_index": int(self.inner_index),
            "split_role": self.split_role,
            "source_split": self.source_split,
            "purpose": self.purpose,
            "num_classes": int(self.num_classes),
            "sample_keys": list(self.sample_keys),
            "contains_private_state": False,
        }
        arrays = {
            name: getattr(self, name) for name in (
                "labels", "original_labels", "waveform_logits",
                "waveform_unknown", "waveform_reliability", "waveform_top1",
                "waveform_top2", "waveform_summary", "prototype_logits",
                "prototype_unknown", "prototype_reliability", "prototype_top1",
                "prototype_top2", "prototype_summary",
            )
        }
        np.savez_compressed(
            path, metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PublicDecisionTable":
        payload = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata.get("contains_private_state") is not False:
            raise RuntimeError("public table metadata does not deny private state")
        arrays = {
            name: payload[name] for name in (
                "labels", "original_labels", "waveform_logits",
                "waveform_unknown", "waveform_reliability", "waveform_top1",
                "waveform_top2", "waveform_summary", "prototype_logits",
                "prototype_unknown", "prototype_reliability", "prototype_top1",
                "prototype_top2", "prototype_summary",
            )
        }
        return cls(
            outer_index=int(metadata["outer_index"]),
            inner_index=int(metadata["inner_index"]),
            split_role=str(metadata["split_role"]),
            source_split=str(metadata["source_split"]),
            purpose=str(metadata["purpose"]),
            num_classes=int(metadata["num_classes"]),
            sample_keys=tuple(str(value) for value in metadata["sample_keys"]),
            **arrays,
        )


@torch.no_grad()
def collect_public_decision_table(
    system: nn.Module,
    dataset: ProvenanceSubset,
    device: torch.device,
    *,
    outer_index: int,
    inner_index: int,
    split_role: str,
    batch_size: int = 1024,
) -> PublicDecisionTable:
    """Collect an auditable table without retaining any private Agent value."""

    if not isinstance(dataset, ProvenanceSubset):
        raise TypeError("nested public decisions require ProvenanceSubset metadata")
    assert_no_formal_unknown(dataset)
    if dataset.formal_unknown or dataset.source_split.startswith("test"):
        raise RuntimeError("test/formal data cannot enter nested public tables")
    predictions = collect_predictions(
        system, dataset, device, batch_size=batch_size)
    return PublicDecisionTable(
        outer_index=int(outer_index), inner_index=int(inner_index),
        split_role=str(split_role), source_split=str(dataset.source_split),
        purpose=str(dataset.purpose), num_classes=int(system.num_classes),
        labels=predictions["y"], original_labels=dataset.original_labels,
        sample_keys=tuple(dataset.sample_keys),
        waveform_logits=predictions["logits_waveform"],
        waveform_unknown=predictions["raw_waveform"],
        waveform_reliability=predictions["reliability_waveform"],
        waveform_top1=predictions["top1_waveform"],
        waveform_top2=predictions["top2_waveform"],
        waveform_summary=predictions["summary_waveform"],
        prototype_logits=predictions["logits_prototype"],
        prototype_unknown=predictions["raw_prototype"],
        prototype_reliability=predictions["reliability_prototype"],
        prototype_top1=predictions["top1_prototype"],
        prototype_top2=predictions["top2_prototype"],
        prototype_summary=predictions["summary_prototype"],
    )


def _team_sample_loss(
    fusion: FairB2Fusion,
    decisions: Mapping[str, LocalDecision],
    labels: torch.Tensor,
    *,
    class_weight: float,
    open_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, unknown, _ = fusion(decisions)
    is_unknown = labels.lt(0)
    open_loss = float(open_weight) * F.binary_cross_entropy_with_logits(
        unknown, is_unknown.float(), reduction="none")
    class_loss = open_loss.new_zeros(len(labels))
    known = ~is_unknown
    if bool(known.any()):
        class_loss[known] = float(class_weight) * F.cross_entropy(
            logits[known], labels[known], reduction="none")
    return class_loss + open_loss, logits, unknown


def _fusion_objective(
    fusion: FairB2Fusion,
    decisions: Mapping[str, LocalDecision],
    labels: torch.Tensor,
    *,
    class_weight: float,
    open_weight: float,
    no_regret_weight: float,
    selector_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    team, _, _ = _team_sample_loss(
        fusion, decisions, labels,
        class_weight=class_weight, open_weight=open_weight)
    local_total, local_class, local_open = local_agent_sample_losses(
        decisions, labels, class_weight=class_weight, open_weight=open_weight)
    best_local = local_total.min(1).values.detach()
    regret = F.relu(team - best_local)
    class_weights, open_weights = fusion.mixture_weights(decisions)
    known = labels.ge(0)
    class_selector = team.new_zeros(())
    if bool(known.any()):
        class_target = local_class[known].argmin(1)
        class_selector = F.nll_loss(
            class_weights[known].clamp_min(1e-8).log(), class_target)
    open_target = local_open.argmin(1)
    open_selector = F.nll_loss(
        open_weights.clamp_min(1e-8).log(), open_target)
    selector = class_selector + open_selector
    objective = (
        team.mean() + float(no_regret_weight) * regret.mean()
        + float(selector_weight) * selector)
    return objective, {
        "task_loss": team.mean(), "regret": regret.mean(),
        "selector_loss": selector, "best_local_loss": best_local.mean(),
    }


@torch.no_grad()
def _mean_local_loss(
    table: PublicDecisionTable,
    device: torch.device,
    *,
    batch_size: int,
    class_weight: float,
    open_weight: float,
) -> torch.Tensor:
    total = torch.zeros(2, dtype=torch.float64)
    for start in range(0, len(table), batch_size):
        index = np.arange(start, min(start + batch_size, len(table)))
        decisions = table.decisions(index, device)
        labels = table.label_tensor(index, device)
        losses, _, _ = local_agent_sample_losses(
            decisions, labels, class_weight=class_weight,
            open_weight=open_weight)
        total += losses.double().sum(0).cpu()
    return total / len(table)


def _validate_fit_tables(
    tables: Sequence[PublicDecisionTable],
) -> dict[int, dict[str, PublicDecisionTable]]:
    if not tables:
        raise ValueError("nested B2 fitting requires public decision tables")
    grouped: dict[int, dict[str, PublicDecisionTable]] = {}
    outer_indices = {int(table.outer_index) for table in tables}
    if len(outer_indices) != 1:
        raise RuntimeError("one nested B2 fit cannot mix outer folds")
    summary_dims = {table.summary_dim for table in tables}
    if len(summary_dims) != 1:
        raise ValueError("public-summary dimensions differ across inner systems")
    # Different inner models legitimately observe overlapping train-Known
    # samples.  Their public decisions are distinct producer/sample rows.
    # Duplicates are forbidden only within one producer; validation/meta rows
    # use the stricter globally unique ownership contract in ``nested_g0``.
    producer_keys: set[tuple[int, str]] = set()
    for table in tables:
        if table.split_role not in FIT_ROLES or table.source_split != "train":
            raise RuntimeError("only inner train Known/proxy tables may fit nested B2")
        tagged = {(int(table.inner_index), key) for key in table.sample_keys}
        overlap = producer_keys.intersection(tagged)
        if overlap:
            raise RuntimeError(
                "a producer/sample row entered nested B2 fitting twice: "
                f"{sorted(overlap)[0]}")
        producer_keys.update(tagged)
        episode = grouped.setdefault(int(table.inner_index), {})
        if table.split_role in episode:
            raise ValueError("duplicate table role for one inner episode")
        episode[table.split_role] = table
    for index, episode in grouped.items():
        if set(episode) != FIT_ROLES:
            raise ValueError(
                f"inner episode {index} needs train_known and train_proxy_unknown")
        known, proxy = episode["train_known"], episode["train_proxy_unknown"]
        if known.num_classes != proxy.num_classes:
            raise ValueError("one inner episode has inconsistent class counts")
    return grouped


@dataclass(frozen=True)
class NestedB2Fit:
    bundle: Mapping[str, Any]
    history: tuple[Mapping[str, float], ...]
    audit: Mapping[str, Any]


def fit_nested_public_b2(
    carrier_system: nn.Module,
    tables: Sequence[PublicDecisionTable],
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> NestedB2Fit:
    """Fit one transferable B2 from one or more inner training episodes."""

    grouped = _validate_fit_tables(tables)
    first = next(iter(tables))
    set_seed(seed)
    current = carrier_system.dialogue.b2_fusion
    hidden_dim = int(current.weight_head[1].out_features)
    fusion = FairB2Fusion(
        public_summary_dim=first.summary_dim, hidden_dim=hidden_dim,
        max_adaptive_mass=float(current.max_adaptive_mass),
        initial_adaptive_fraction=float(
            current.initial_adaptive_fraction)).to(device)
    carrier_system.dialogue.b2_fusion = fusion

    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("nested_b2_epochs", cfg.get("b2_epochs", 5)))
    if batch_size < 1 or epochs < 1:
        raise ValueError("nested B2 requires positive batch size and epochs")
    class_weight = float(cfg.get("class_loss_weight", 1.0))
    open_weight = float(cfg.get("open_loss_weight", 1.0))
    proxy_weight = float(cfg.get("proxy_unknown_weight", 1.0))
    no_regret_weight = float(cfg.get("b2_no_regret_weight", 1.0))
    selector_weight = float(cfg.get("b2_selector_distillation_weight", 0.25))
    if min(class_weight, open_weight, proxy_weight, no_regret_weight,
           selector_weight) < 0:
        raise ValueError("nested B2 loss weights must be non-negative")

    known_losses, proxy_losses = [], []
    for episode in grouped.values():
        known_losses.append(_mean_local_loss(
            episode["train_known"], device, batch_size=batch_size,
            class_weight=class_weight, open_weight=open_weight))
        proxy_losses.append(_mean_local_loss(
            episode["train_proxy_unknown"], device, batch_size=batch_size,
            class_weight=class_weight, open_weight=open_weight))
    aggregate = torch.stack(known_losses).mean(0)
    denominator = 1.0
    if proxy_weight:
        aggregate += proxy_weight * torch.stack(proxy_losses).mean(0)
        denominator += proxy_weight
    aggregate /= denominator
    anchor_index = int(aggregate.argmin().item())
    fusion.set_anchor(anchor_index)
    fusion.set_adaptive_enabled(True)

    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=float(cfg.get("b2_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    generator = np.random.default_rng(int(seed) + 911)
    history: list[dict[str, float]] = []
    ordered = [grouped[index] for index in sorted(grouped)]
    for epoch in range(1, epochs + 1):
        fusion.train()
        totals = {"loss": 0.0, "task_loss": 0.0, "regret": 0.0,
                  "selector_loss": 0.0, "best_local_loss": 0.0}
        steps = 0
        for episode in ordered:
            known = episode["train_known"]
            proxy = episode["train_proxy_unknown"]
            known_order = generator.permutation(len(known))
            proxy_order = generator.permutation(len(proxy))
            proxy_cursor = 0
            for start in range(0, len(known), batch_size):
                known_index = known_order[start:start + batch_size]
                take = min(batch_size, len(proxy))
                if proxy_cursor + take > len(proxy_order):
                    proxy_order = generator.permutation(len(proxy))
                    proxy_cursor = 0
                proxy_index = proxy_order[proxy_cursor:proxy_cursor + take]
                proxy_cursor += take
                known_loss, known_parts = _fusion_objective(
                    fusion, known.decisions(known_index, device),
                    known.label_tensor(known_index, device),
                    class_weight=class_weight, open_weight=open_weight,
                    no_regret_weight=no_regret_weight,
                    selector_weight=selector_weight)
                value = known_loss
                parts = dict(known_parts)
                divisor = 1.0
                if proxy_weight:
                    proxy_loss, proxy_parts = _fusion_objective(
                        fusion, proxy.decisions(proxy_index, device),
                        proxy.label_tensor(proxy_index, device),
                        class_weight=class_weight, open_weight=open_weight,
                        no_regret_weight=no_regret_weight,
                        selector_weight=selector_weight)
                    value = value + proxy_weight * proxy_loss
                    divisor += proxy_weight
                    for name in parts:
                        parts[name] = (
                            parts[name] + proxy_weight * proxy_parts[name]) / divisor
                value = value / divisor
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                torch.nn.utils.clip_grad_norm_(
                    fusion.parameters(), float(cfg.get("clip_grad_norm", 5.0)))
                optimizer.step()
                totals["loss"] += float(value.detach())
                for name, part in parts.items():
                    totals[name] += float(part.detach())
                steps += 1
        history.append({
            "epoch": float(epoch), "steps": float(steps),
            **{name: value / max(steps, 1) for name, value in totals.items()},
        })
    fusion.eval()
    fusion.set_adaptive_enabled(True)
    bundle = export_b2_transfer_bundle(carrier_system)
    audit = {
        "outer_index": int(first.outer_index),
        "inner_indices": sorted(grouped),
        "num_inner_episodes": len(grouped),
        "anchor_agent": LOCAL_AGENT_NAMES[anchor_index],
        "waveform_fit_objective": float(aggregate[0]),
        "prototype_fit_objective": float(aggregate[1]),
        "fit_sample_count": int(sum(len(table) for table in tables)),
        "fit_sample_keys_unique_by_producer": True,
        "fit_sources": ["train"],
        "public_only": True,
        "contains_private_state": False,
        "formal_unknown_used": False,
        "outer_test_used": False,
        "state_sha256": bundle["manifest"]["state_sha256"],
    }
    return NestedB2Fit(bundle, tuple(history), audit)


@torch.no_grad()
def evaluate_public_b2_table(
    system: nn.Module,
    table: PublicDecisionTable,
    device: torch.device,
    *,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Evaluate installed B2 and both local rules from a public table."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    fusion = system.dialogue.b2_fusion.to(device).eval()
    rows: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "y", "pred", "raw_unknown", "known_weights",
            "pred_waveform", "raw_waveform",
            "pred_prototype", "raw_prototype", "public_features",
        )}
    for start in range(0, len(table), batch_size):
        index = np.arange(start, min(start + batch_size, len(table)))
        decisions = table.decisions(index, device)
        logits, unknown, weights = fusion(decisions)
        rows["y"].append(table.labels[index])
        rows["pred"].append(logits.argmax(1).cpu().numpy())
        rows["raw_unknown"].append(unknown.cpu().numpy())
        rows["known_weights"].append(weights.cpu().numpy())
        rows["public_features"].append(
            public_team_features(decisions).cpu().numpy())
        for agent in LOCAL_AGENT_NAMES:
            rows[f"pred_{agent}"].append(
                decisions[agent].class_logits.argmax(1).cpu().numpy())
            rows[f"raw_{agent}"].append(
                decisions[agent].unknown_score.cpu().numpy())
    result = {name: np.concatenate(values, axis=0)
              for name, values in rows.items()}
    result["original_labels"] = table.original_labels.copy()
    result["sample_keys"] = np.asarray(table.sample_keys, dtype=str)
    return result


@torch.no_grad()
def public_local_action_losses(
    table: PublicDecisionTable,
    device: torch.device,
    *,
    batch_size: int = 1024,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed public features and exact W/P local losses for audit."""

    features, losses = [], []
    for start in range(0, len(table), batch_size):
        index = np.arange(start, min(start + batch_size, len(table)))
        decisions = table.decisions(index, device)
        labels = table.label_tensor(index, device)
        local, _, _ = local_agent_sample_losses(
            decisions, labels, class_weight=class_weight,
            open_weight=open_weight)
        features.append(public_team_features(decisions).cpu().numpy())
        losses.append(local.cpu().numpy())
    return np.concatenate(features), np.concatenate(losses)


__all__ = [
    "ALL_ROLES", "FIT_ROLES", "VALIDATION_ROLES", "NestedB2Fit",
    "PublicDecisionTable", "collect_public_decision_table",
    "evaluate_public_b2_table", "fit_nested_public_b2",
    "public_local_action_losses",
]
