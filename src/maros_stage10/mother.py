"""Read-only Stage-5 Unified V4 experts and leakage-safe fold extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from maros_stage5.agents import RoleStructuredExperts
from maros_stage7.splits import assert_no_formal_unknown


@dataclass(frozen=True)
class GeometryPrivateEvidence:
    geometry_logits: np.ndarray
    view_logits: np.ndarray
    view_distances: np.ndarray
    view_weights: np.ndarray

    def __len__(self) -> int:
        return len(self.geometry_logits)


@dataclass(frozen=True)
class FrozenEvidence:
    labels: np.ndarray
    identity_logits: np.ndarray
    identity_public: np.ndarray
    geometry_logits: np.ndarray
    view_logits: np.ndarray
    view_distances: np.ndarray
    view_weights: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def geometry_private(self) -> GeometryPrivateEvidence:
        """Boundary: B receives no A logits, labels, or hidden state."""
        return GeometryPrivateEvidence(
            geometry_logits=self.geometry_logits,
            view_logits=self.view_logits,
            view_distances=self.view_distances,
            view_weights=self.view_weights)


def load_mother(checkpoint_path: str | Path, cfg: dict,
                support_classes: tuple[int, ...],
                heldout_classes: tuple[int, ...], device: torch.device):
    """Never update or rewrite the frozen Stage-5 source checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (tuple(checkpoint["support_classes"]) != tuple(support_classes)
            or tuple(checkpoint["heldout_classes"]) != tuple(heldout_classes)):
        raise ValueError("Stage-5 checkpoint and LCO fold class partitions differ")
    if cfg["geometry_view"] != "universal":
        raise ValueError("Stage-5 mother must preserve the universal view bank")
    model = RoleStructuredExperts(
        len(support_classes), state_dim=int(cfg["state_dim"]),
        stem_channels=int(cfg["stem_channels"]),
        message_dim=int(cfg["message_dim"]),
        prototype_temperature=float(cfg["prototype_temperature"]),
        geometry_view="universal").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def cap_dataset(dataset: Dataset, per_class: int, seed: int,
                *, proxy_unknown: bool = False) -> Dataset:
    assert_no_formal_unknown(dataset)
    labels = np.asarray(dataset.original_labels if proxy_unknown else dataset.y,
                        dtype=np.int64)
    if not proxy_unknown and np.any(labels < 0):
        raise ValueError("Known cap received Unknown labels")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        chosen.extend(indices[:per_class].tolist())
    return Subset(dataset, sorted(chosen))


@torch.no_grad()
def collect_evidence(model: RoleStructuredExperts, dataset: Dataset,
                     device: torch.device, batch_size: int) -> FrozenEvidence:
    assert_no_formal_unknown(dataset)
    model.eval()
    rows = {name: [] for name in FrozenEvidence.__dataclass_fields__}
    for iq, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        iq = iq.to(device)
        identity = model.identity(iq)
        geometry = model.geometry(iq)
        id_ev, geo_ev = identity.evidence, geometry.evidence
        public = torch.stack([
            id_ev["confidence"], id_ev["margin"], id_ev["entropy"],
            id_ev["d1"], id_ev["distance_margin"],
        ], dim=1)
        values = {
            "labels": labels, "identity_logits": identity.class_logits,
            "identity_public": public, "geometry_logits": geometry.class_logits,
            "view_logits": geo_ev["view_logits"],
            "view_distances": geo_ev["view_distances"],
            "view_weights": geo_ev["view_weights"],
        }
        for name, value in values.items():
            rows[name].append(value.detach().cpu().numpy())
    return FrozenEvidence(**{name: np.concatenate(parts) for name, parts in rows.items()})
