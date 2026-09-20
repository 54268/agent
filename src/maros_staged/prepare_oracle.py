"""Prepare the ORACLE KRI-16 demodulated SigMF corpus into train/val/test npz.

Adapted (simplified) from the previous single-model scheme's ``prep_oracle.py``
so the staged multi-agent code uses the identical, directly comparable protocol:

* known emitters   : 1, 2, 3, 4, 5, 7, 9, 13, 14, 15  (10 classes)
* unknown emitters : 17, 18, 19, 25, 26, 32           (6 classes, test only)
* segment length / stride = 256 / 256
* at most 4000 windows per emitter (random, seed 42)
* known split 70 / 10 / 20 -> 28000 / 4000 / 8000, unknown test = 24000

Raw windows are stored unnormalised; per-sample normalisation happens in the
Dataset, matching the previous scheme's ``normalize='per_sample'`` convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


KNOWN_LABELS = ["1", "2", "3", "4", "5", "7", "9", "13", "14", "15"]
UNKNOWN_LABELS = ["17", "18", "19", "25", "26", "32"]

SEGMENT_LEN = 256
MAX_PER_LABEL = 4000
SEED = 42


def _read_sigmf_complex(path: Path, datatype: str) -> np.ndarray:
    if datatype in ("ci16", "ci16_le"):
        raw = np.fromfile(path, dtype=np.dtype([("r", "<i2"), ("i", "<i2")]))
        return raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)
    if datatype in ("ci8", "ci8_le"):
        raw = np.fromfile(path, dtype=np.dtype([("r", "i1"), ("i", "i1")]))
        return raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)
    if datatype in ("cf64", "cf64_le"):
        return np.fromfile(path, dtype=np.complex128).astype(np.complex64)
    # default cf32
    return np.fromfile(path, dtype=np.complex64)


def _resolve_datatype(path: Path, declared: str, sample_count: int | None) -> str:
    declared = (declared or "cf32").lower()
    if sample_count:
        bps = path.stat().st_size / sample_count
        inferred = {2: "ci8", 4: "ci16_le", 8: "cf32_le", 16: "cf64_le"}
        rounded = int(round(bps))
        if abs(bps - rounded) < 1e-6 and rounded in inferred:
            return inferred[rounded]
    return {"cf32": "cf32_le", "cf64": "cf64_le", "ci16": "ci16_le", "ci8_le": "ci8"}.get(
        declared, declared
    )


def _load_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "global" in meta:
        return meta
    return meta.get("_metadata", {})


def _emit_label(meta_path: Path) -> str:
    return meta_path.stem.split("IQ#")[-1].split("_")[0]


def _window(signal: np.ndarray, length: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(signal) - length + 1, stride):
        seg = signal[start : start + length]
        windows.append(np.stack([seg.real, seg.imag], axis=0).astype(np.float32))
    return np.asarray(windows, dtype=np.float32)


def _stratified_indices(y: np.ndarray, rng: np.random.Generator,
                        train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    train, val, test = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(len(idx) * train_ratio)
        n_va = int(len(idx) * val_ratio)
        train.append(idx[:n_tr])
        val.append(idx[n_tr : n_tr + n_va])
        test.append(idx[n_tr + n_va :])
    return {
        "train": np.concatenate(train),
        "val": np.concatenate(val),
        "test": np.concatenate(test),
    }


def prepare_oracle(raw_root: str | Path, output_root: str | Path,
                   segment_len: int = SEGMENT_LEN, max_per_label: int = MAX_PER_LABEL,
                   seed: int = SEED) -> dict:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    by_label: Dict[str, np.ndarray] = {}
    records = []
    for meta_path in sorted(raw_root.rglob("*.sigmf-meta")):
        data_path = meta_path.with_suffix(".sigmf-data")
        if not data_path.exists():
            continue
        meta = _load_meta(meta_path)
        global_meta = meta.get("global", {})
        annotations = meta.get("annotations", [])
        sample_count = int(annotations[0].get("core:sample_count", 0)) if annotations else None
        datatype = _resolve_datatype(
            data_path, str(global_meta.get("core:datatype", "cf32")), sample_count
        )
        signal = _read_sigmf_complex(data_path, datatype)
        windows = _window(signal, segment_len, segment_len)
        if len(windows) == 0:
            continue
        label = _emit_label(meta_path)
        by_label[label] = windows
        records.append({"label": label, "datatype": datatype, "num_windows": int(len(windows))})

    missing = [lab for lab in KNOWN_LABELS + UNKNOWN_LABELS if lab not in by_label]
    if missing:
        raise RuntimeError(f"Missing Oracle emitters: {missing}; found {sorted(by_label)}")

    known_x, known_y = [], []
    unknown_x, unknown_y, unknown_identity_y = [], [], []
    for label in KNOWN_LABELS:
        w = by_label[label]
        if max_per_label and len(w) > max_per_label:
            w = w[rng.choice(len(w), size=max_per_label, replace=False)]
        known_x.append(w)
        known_y.append(np.full(len(w), KNOWN_LABELS.index(label), dtype=np.int64))
    for unknown_index, label in enumerate(UNKNOWN_LABELS):
        w = by_label[label]
        if max_per_label and len(w) > max_per_label:
            w = w[rng.choice(len(w), size=max_per_label, replace=False)]
        unknown_x.append(w)
        unknown_y.append(np.full(len(w), -1, dtype=np.int64))
        unknown_identity_y.append(np.full(len(w), unknown_index, dtype=np.int64))

    x_known = np.concatenate(known_x)
    y_known = np.concatenate(known_y)
    x_unknown = np.concatenate(unknown_x)
    y_unknown = np.concatenate(unknown_y)
    identity_unknown = np.concatenate(unknown_identity_y)

    idx = _stratified_indices(y_known, rng, 0.7, 0.1)
    np.savez_compressed(output_root / "train_known.npz", x=x_known[idx["train"]], y=y_known[idx["train"]])
    np.savez_compressed(output_root / "val_known.npz", x=x_known[idx["val"]], y=y_known[idx["val"]])
    np.savez_compressed(output_root / "test_known.npz", x=x_known[idx["test"]], y=y_known[idx["test"]])
    np.savez_compressed(output_root / "test_unknown.npz", x=x_unknown, y=y_unknown,
                        unknown_identity_y=identity_unknown)

    summary = {
        "dataset": "oracle_kri16_demod",
        "known_classes": KNOWN_LABELS,
        "unknown_classes": UNKNOWN_LABELS,
        "segment_len": segment_len,
        "max_per_label": max_per_label,
        "seed": seed,
        "num_train_known": int(len(idx["train"])),
        "num_val_known": int(len(idx["val"])),
        "num_test_known": int(len(idx["test"])),
        "num_test_unknown": int(len(x_unknown)),
        "signal_shape": [2, segment_len],
        "records": records,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    info = prepare_oracle(args.raw_root, args.output_root)
    print(json.dumps({k: v for k, v in info.items() if k != "records"}, indent=2, ensure_ascii=False))
