from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


FILENAME_RE = re.compile(
    r"^dataset_(?P<date>\d{4}_\d{2}_\d{2})_node(?P<rx>.+)\.pkl$"
)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def discover_canonical_pickles(data_root: Path) -> dict[tuple[str, str], Path]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(data_root.rglob("*.pkl")):
        match = FILENAME_RE.match(path.name)
        if match:
            groups[(match["date"], match["rx"])].append(path)
    return {key: paths[0] for key, paths in sorted(groups.items())}


def _aggregate_support(
    audit: dict[str, Any], dates: set[str], receivers: set[str]
) -> dict[str, int]:
    totals = {str(tx): 0 for tx in audit["transmitters"]}
    for cell in audit["cells"]:
        if cell["date"] in dates and cell["rx"] in receivers:
            totals[str(cell["tx"])] += int(cell["packets"])
    return totals


def realize_protocol(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"]).resolve()
    audit = load_json(project_root / config["data"]["audit"])
    domain = config["protocol"]["domain_split"]
    train_support = _aggregate_support(
        audit, set(domain["train_dates"]), set(domain["train_receivers"])
    )
    val_support = _aggregate_support(
        audit, set(domain["validation_dates"]), set(domain["validation_receivers"])
    )
    test_support = _aggregate_support(
        audit, set(domain["test_dates"]), set(domain["test_receivers"])
    )
    minimum = int(config["protocol"].get("min_packets_each_split", 1))
    eligible = sorted(
        tx
        for tx in audit["transmitters"]
        if min(train_support[tx], val_support[tx], test_support[tx]) >= minimum
    )
    expected = int(config["protocol"]["total_classes"])
    if len(eligible) != expected:
        raise ValueError(f"Expected {expected} eligible Tx, found {len(eligible)}")
    rng = np.random.default_rng(int(config["protocol"]["class_partition_seed"]))
    class_order = rng.permutation(eligible).tolist()
    num_known = int(config["protocol"]["known_classes"])
    known = class_order[:num_known]
    unknown = class_order[num_known:]
    return {
        "class_order": class_order,
        "known_tx": known,
        "unknown_tx": unknown,
        "known_to_label": {tx: index for index, tx in enumerate(known)},
        "unknown_label": num_known,
        "support": {
            tx: {
                "train": train_support[tx],
                "validation": val_support[tx],
                "test": test_support[tx],
            }
            for tx in class_order
        },
    }


def _balanced_quotas(items: list[tuple[tuple[str, str], int]], target: int) -> dict[tuple[str, str], int]:
    quotas = {key: 0 for key, _ in items}
    capacity = {key: int(count) for key, count in items}
    remaining = min(int(target), sum(capacity.values()))
    while remaining > 0:
        active = [key for key in quotas if quotas[key] < capacity[key]]
        if not active:
            break
        share = max(1, math.ceil(remaining / len(active)))
        progressed = 0
        for key in active:
            take = min(share, capacity[key] - quotas[key], remaining)
            quotas[key] += take
            remaining -= take
            progressed += take
            if remaining == 0:
                break
        if progressed == 0:
            break
    return {key: value for key, value in quotas.items() if value > 0}


def _build_quota_lookup(
    audit: dict[str, Any],
    tx_names: list[str],
    dates: set[str],
    receivers: set[str],
    target_per_tx: int,
) -> dict[tuple[str, str, str], int]:
    by_tx: dict[str, list[tuple[tuple[str, str], int]]] = defaultdict(list)
    tx_set = set(tx_names)
    for cell in audit["cells"]:
        tx = str(cell["tx"])
        if (
            tx in tx_set
            and cell["date"] in dates
            and cell["rx"] in receivers
            and int(cell["packets"]) > 0
        ):
            by_tx[tx].append(((cell["date"], cell["rx"]), int(cell["packets"])))
    lookup: dict[tuple[str, str, str], int] = {}
    for tx in tx_names:
        quotas = _balanced_quotas(by_tx[tx], target_per_tx)
        if sum(quotas.values()) < target_per_tx:
            raise ValueError(
                f"Tx {tx} only supports {sum(quotas.values())}/{target_per_tx} requested packets"
            )
        for (date, rx), value in quotas.items():
            lookup[(date, rx, tx)] = value
    return lookup


def _extract_split(
    *,
    name: str,
    file_map: dict[tuple[str, str], Path],
    quota_lookup: dict[tuple[str, str, str], int],
    known_to_label: dict[str, int],
    unknown_label: int,
    seed: int,
    rx_to_id: dict[str, int],
    date_to_id: dict[str, int],
) -> dict[str, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    tx_names: list[np.ndarray] = []
    rx_ids: list[np.ndarray] = []
    date_ids: list[np.ndarray] = []

    cells_by_file: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (date, rx, tx), quota in quota_lookup.items():
        cells_by_file[(date, rx)].append((tx, quota))

    for file_index, ((date, rx), requests) in enumerate(sorted(cells_by_file.items()), start=1):
        path = file_map[(date, rx)]
        print(
            f"[prepare:{name}] {file_index:02d}/{len(cells_by_file):02d} {date} rx={rx}",
            flush=True,
        )
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        node_to_index = {str(tx): index for index, tx in enumerate(payload["node_list"])}
        for tx, quota in sorted(requests):
            block = np.asarray(payload["data"][node_to_index[tx]])
            if len(block) < quota:
                raise ValueError(f"Audit mismatch for {date}/{rx}/{tx}: {len(block)} < {quota}")
            rng = np.random.default_rng(stable_seed(seed, name, date, rx, tx))
            indices = rng.choice(len(block), size=quota, replace=False)
            arrays.append(np.asarray(block[indices], dtype=np.float32))
            label = known_to_label.get(tx, unknown_label)
            labels.append(np.full(quota, label, dtype=np.int64))
            tx_names.append(np.full(quota, tx, dtype="U16"))
            rx_ids.append(np.full(quota, rx_to_id[rx], dtype=np.int16))
            date_ids.append(np.full(quota, date_to_id[date], dtype=np.int8))
        del payload

    result = {
        "iq": np.concatenate(arrays, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "tx_names": np.concatenate(tx_names, axis=0),
        "rx_ids": np.concatenate(rx_ids, axis=0),
        "date_ids": np.concatenate(date_ids, axis=0),
    }
    rng = np.random.default_rng(stable_seed(seed, name, "final_shuffle"))
    order = rng.permutation(len(result["labels"]))
    return {key: value[order] for key, value in result.items()}


def prepare_dataset(config: dict[str, Any], force: bool = False) -> Path:
    project_root = Path(config["project_root"]).resolve()
    output_dir = project_root / config["data"]["processed_dir"]
    complete = output_dir / "COMPLETE.json"
    if complete.exists() and not force:
        print(f"[prepare] reusing {output_dir}", flush=True)
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = load_json(project_root / config["data"]["audit"])
    realized = realize_protocol(config)
    domain = config["protocol"]["domain_split"]
    data_root = project_root / config["data"]["root"]
    file_map = discover_canonical_pickles(data_root)

    rx_names = sorted(set(audit["common_receivers_all_dates"]))
    date_names = sorted(audit["capture_dates"])
    rx_to_id = {name: index for index, name in enumerate(rx_names)}
    date_to_id = {name: index for index, name in enumerate(date_names)}
    known = realized["known_tx"]
    unknown = realized["unknown_tx"]
    caps = config["data"]["caps_per_tx"]

    split_specs = {
        "train": (
            known,
            set(domain["train_dates"]),
            set(domain["train_receivers"]),
            int(caps["train_known"]),
        ),
        "validation": (
            known,
            set(domain["validation_dates"]),
            set(domain["validation_receivers"]),
            int(caps["validation_known"]),
        ),
        "test_known": (
            known,
            set(domain["test_dates"]),
            set(domain["test_receivers"]),
            int(caps["test_known"]),
        ),
        "test_unknown": (
            unknown,
            set(domain["test_dates"]),
            set(domain["test_receivers"]),
            int(caps["test_unknown"]),
        ),
    }

    manifest_splits: dict[str, Any] = {}
    for split_name, (tx_names, dates, receivers, target) in split_specs.items():
        quota = _build_quota_lookup(audit, tx_names, dates, receivers, target)
        split = _extract_split(
            name=split_name,
            file_map=file_map,
            quota_lookup=quota,
            known_to_label=realized["known_to_label"],
            unknown_label=realized["unknown_label"],
            seed=int(config["protocol"]["class_partition_seed"]),
            rx_to_id=rx_to_id,
            date_to_id=date_to_id,
        )
        for key, value in split.items():
            np.save(output_dir / f"{split_name}_{key}.npy", value, allow_pickle=False)
        manifest_splits[split_name] = {
            "samples": int(len(split["labels"])),
            "classes": int(len(np.unique(split["tx_names"]))),
            "target_per_tx": target,
            "source_cells": len(quota),
        }

    manifest = {
        **realized,
        "rx_names": rx_names,
        "date_names": date_names,
        "splits": manifest_splits,
        "normalization": config["data"]["normalization"],
        "source_data_root": str(data_root.resolve()),
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(complete, {"status": "complete", "splits": manifest_splits})
    return output_dir


class WiSigArrayDataset(Dataset):
    def __init__(self, root: str | Path, split: str, augment: bool = False, seed: int = 0):
        root = Path(root)
        self.iq = np.load(root / f"{split}_iq.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.tx_names = np.load(root / f"{split}_tx_names.npy", mmap_mode="r")
        self.rx_ids = np.load(root / f"{split}_rx_ids.npy", mmap_mode="r")
        self.date_ids = np.load(root / f"{split}_date_ids.npy", mmap_mode="r")
        self.augment = augment
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        iq = np.asarray(self.iq[index], dtype=np.float32).copy()
        power = np.mean(np.sum(iq * iq, axis=-1))
        iq /= np.sqrt(max(float(power), 1e-12))
        if self.augment:
            rng = np.random.default_rng(stable_seed(self.seed, index))
            angle = rng.uniform(-math.pi, math.pi)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            i = iq[:, 0].copy()
            q = iq[:, 1].copy()
            iq[:, 0] = cos_a * i - sin_a * q
            iq[:, 1] = sin_a * i + cos_a * q
            iq = np.roll(iq, int(rng.integers(-3, 4)), axis=0)
            iq += rng.normal(0.0, 0.01, size=iq.shape).astype(np.float32)
        return {
            "iq": torch.from_numpy(iq.T.copy()),
            "label": int(self.labels[index]),
            "rx_id": int(self.rx_ids[index]),
            "date_id": int(self.date_ids[index]),
            "index": int(index),
        }


class CombinedTestDataset(Dataset):
    def __init__(self, known: WiSigArrayDataset, unknown: WiSigArrayDataset):
        self.known = known
        self.unknown = unknown

    def __len__(self) -> int:
        return len(self.known) + len(self.unknown)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < len(self.known):
            item = dict(self.known[index])
            item["source_split"] = 0
            return item
        item = dict(self.unknown[index - len(self.known)])
        item["source_split"] = 1
        item["index"] = index
        return item

