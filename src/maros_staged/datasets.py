"""Datasets for the staged Multi-Agent OS-SEI Stage-1 experiments.

Two corpora, each returning ``(iq[2, L], label)`` tensors:

* :func:`load_wisig_subset` - the fixed day / fixed receiver pure open-set
  WiSig pickle (40 known / 20 unknown).  Per-sample average-power
  normalisation, matching the rest of this project.
* :func:`load_oracle_npz` - demodulated ORACLE npz produced by
  :mod:`maros_staged.prepare_oracle`.  Per-window per-channel mean/std
  normalisation, matching the previous single-model scheme.

Real unknowns are only ever exposed through the open-test loader; they never
enter train/val.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def _power_normalise(iq: np.ndarray) -> np.ndarray:
    """iq: (..., L, 2) -> divide by the per-sample RMS of I^2+Q^2."""
    power = np.mean(np.sum(iq * iq, axis=-1), axis=-1, keepdims=True)
    return iq / np.sqrt(np.maximum(power, 1e-12))[..., None]


class IQDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, channel_last: bool,
                 normalize: str, augment: bool = False):
        self.y = np.asarray(y, dtype=np.int64)
        x = np.asarray(x, dtype=np.float32)
        if channel_last:  # (N, L, 2) -> (N, 2, L)
            x = np.transpose(x, (0, 2, 1))
        if normalize == "power":
            x = _power_normalise(np.transpose(x, (0, 2, 1)))
            x = np.transpose(x, (0, 2, 1))
        elif normalize == "per_channel":
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True)
            x = (x - mean) / (std + 1e-6)
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        iq = self.x[index].copy()
        if self.augment:
            # random phase rotation (power preserving), small time roll, AWGN
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            i, q = iq[0].copy(), iq[1].copy()
            iq[0] = c * i - s * q
            iq[1] = s * i + c * q
            iq = np.roll(iq, np.random.randint(-3, 4), axis=-1)
            iq = iq + np.random.normal(0.0, 0.01, size=iq.shape).astype(np.float32)
        return torch.from_numpy(iq), torch.tensor(int(self.y[index]), dtype=torch.long)


@dataclass
class OpenSetSplits:
    train: IQDataset
    val: IQDataset
    test_known: IQDataset
    test_unknown: IQDataset
    open_test: IQDataset
    num_known: int
    signal_length: int
    meta: dict


def load_wisig_subset(pkl_path: str | Path, augment_train: bool = False) -> OpenSetSplits:
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    def ds(split: str, augment: bool = False):
        return IQDataset(data[split]["x"], data[split]["y"], channel_last=True,
                         normalize="power", augment=augment)

    train = ds("train", augment_train)
    val = ds("val")
    test_known = ds("known_test")
    test_unknown = IQDataset(data["unknown_test"]["x"], data["unknown_test"]["y"],
                             channel_last=True, normalize="power")
    open_test = ds("open_test")
    num_known = len(data["meta"]["known_tx_list"])
    # Keep transmitter identities strictly as evaluation metadata.  Known
    # labels already identify their Tx; unknown identities are offset so the
    # two strata can never collide during grouped bootstrap resampling.
    unknown_mask = np.asarray(data["open_test"]["y"]) == -1
    unknown_identity = np.asarray(data["unknown_test"]["unknown_identity_y"], dtype=np.int64)
    if int(unknown_mask.sum()) != len(unknown_identity):
        raise ValueError("open_test and unknown_test contain different unknown sample counts")
    group_ids = np.asarray(data["open_test"]["y"], dtype=np.int64).copy()
    group_ids[unknown_mask] = num_known + unknown_identity
    open_test.group_ids = group_ids
    return OpenSetSplits(train, val, test_known, test_unknown, open_test,
                         num_known=num_known, signal_length=256,
                         meta={"dataset": "wisig_1day_1rx", **data["meta"]})


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {k: payload[k] for k in payload.files}


def load_oracle_npz(root: str | Path, augment_train: bool = False) -> OpenSetSplits:
    root = Path(root)
    train_p = _load_npz(root / "train_known.npz")
    val_p = _load_npz(root / "val_known.npz")
    tk_p = _load_npz(root / "test_known.npz")
    tu_p = _load_npz(root / "test_unknown.npz")

    meta_path = root / "dataset_summary.json"
    import json
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    norm = "per_channel"
    train = IQDataset(train_p["x"], train_p["y"], channel_last=False, normalize=norm,
                      augment=augment_train)
    val = IQDataset(val_p["x"], val_p["y"], channel_last=False, normalize=norm)
    test_known = IQDataset(tk_p["x"], tk_p["y"], channel_last=False, normalize=norm)
    test_unknown = IQDataset(tu_p["x"], tu_p["y"], channel_last=False, normalize=norm)
    open_x = np.concatenate([tk_p["x"], tu_p["x"]], axis=0)
    open_y = np.concatenate([tk_p["y"], tu_p["y"]], axis=0)
    open_test = IQDataset(open_x, open_y, channel_last=False, normalize=norm)
    num_known = int(train_p["y"].max()) + 1
    if "unknown_identity_y" in tu_p:
        unknown_identity = np.asarray(tu_p["unknown_identity_y"], dtype=np.int64)
    else:
        # Backward-compatible recovery for the already prepared KRI-16 files.
        # prepare_oracle concatenates capped windows in unknown_classes order.
        records = {str(row["label"]): int(row["num_windows"])
                   for row in meta.get("records", [])}
        max_per_label = int(meta.get("max_per_label", 0))
        counts = [min(records.get(str(label), 0), max_per_label)
                  if max_per_label else records.get(str(label), 0)
                  for label in meta.get("unknown_classes", [])]
        if counts and sum(counts) == len(tu_p["y"]):
            unknown_identity = np.concatenate([
                np.full(count, index, dtype=np.int64)
                for index, count in enumerate(counts)
            ])
        elif len(meta.get("unknown_classes", [])) and (
                len(tu_p["y"]) % len(meta["unknown_classes"]) == 0):
            per_class = len(tu_p["y"]) // len(meta["unknown_classes"])
            unknown_identity = np.repeat(
                np.arange(len(meta["unknown_classes"]), dtype=np.int64), per_class)
        else:
            raise ValueError("cannot reconstruct ORACLE unknown transmitter group IDs")
    if len(unknown_identity) != len(tu_p["y"]):
        raise ValueError("ORACLE unknown identity metadata has the wrong length")
    open_test.group_ids = np.concatenate([
        np.asarray(tk_p["y"], dtype=np.int64), num_known + unknown_identity,
    ])
    test_unknown.group_ids = num_known + unknown_identity
    return OpenSetSplits(train, val, test_known, test_unknown, open_test,
                         num_known=num_known, signal_length=train_p["x"].shape[-1],
                         meta={"dataset": "oracle_kri16_demod", **meta})
