# -*- coding: utf-8 -*-
"""
从 Full WiSig 中自动筛选一个“纯开放集”子集：
- 仅使用 non-EQ 数据
- 固定 1 个日期 + 1 个接收机，避免跨域
- 自动寻找覆盖 Tx 数量最多的 Day/Rx 组合
- 默认 60 个 Tx：40 Known + 20 Unknown
- Known: 70% train / 10% val / 20% known_test
- Unknown 主测试集：每类固定取 240 条
- 主 open_test ≈ train 的 2/3
- 额外保留 full-unknown 压力测试集

输出目录：
D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\data\WiSig\WiSig_OpenSet_1Day_1Rx
"""

from __future__ import annotations

import csv
import gc
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# 1. 配置
# ============================================================

SOURCE_ROOT = Path(
    r"D:\learn_pytorch\笔记\多智能体\Multi-Agent OS-SEI\data\WiSig"
)
OUTPUT_DIR = SOURCE_ROOT / "WiSig_OpenSet_1Day_1Rx"

TARGET_TOTAL_TX = 60
TARGET_KNOWN_TX = 40
TARGET_UNKNOWN_TX = 20

PREFERRED_MIN_SAMPLES_PER_TX = 500
FALLBACK_MIN_SAMPLE_LEVELS = [500, 400, 300, 200, 100, 50]

# 每个已选 Tx 最多使用多少条
MAX_SAMPLES_PER_TX = 800

# Known 划分
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# 剩余自动作为 known_test

# 主开放集：每个 Unknown Tx 固定取 240 条
UNKNOWN_TEST_PER_TX = 240

RANDOM_SEED = 2026
OVERWRITE = True

SCAN_CACHE_FILE = OUTPUT_DIR / "wisig_scan_cache.json"

FILE_RE = re.compile(
    r"dataset_(\d{4}_\d{2}_\d{2})_node(.+)\.pkl$",
    re.IGNORECASE,
)


# ============================================================
# 2. 工具函数
# ============================================================

def is_non_eq_file(path: Path) -> bool:
    parent_text = " / ".join(p.name.lower() for p in path.parents)
    return "pkl_wifi_eq_" not in parent_text


def discover_full_wisig_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("dataset_*_node*.pkl"):
        try:
            p.relative_to(OUTPUT_DIR)
            continue
        except ValueError:
            pass

        if is_non_eq_file(p) and FILE_RE.match(p.name):
            files.append(p)

    return sorted(set(files))


def load_cache() -> dict:
    if not SCAN_CACHE_FILE.exists():
        return {}
    try:
        with SCAN_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SCAN_CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def scan_receiver_file(path: Path) -> dict:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"无法解析文件名: {path.name}")

    day = match.group(1)
    rx = match.group(2)

    with path.open("rb") as f:
        dataset = pickle.load(f)

    if "node_list" not in dataset or "data" not in dataset:
        raise KeyError(
            f"{path} 缺少 node_list/data 字段，keys={list(dataset.keys())}"
        )

    node_list = list(dataset["node_list"])
    data = dataset["data"]

    tx_counts = {}
    shape_examples = []

    for tx, arr in zip(node_list, data):
        arr = np.asarray(arr)
        n = int(arr.shape[0])
        tx_counts[str(tx)] = n

        if len(shape_examples) < 3 and n > 0:
            shape_examples.append(list(arr.shape))

    info = {
        "path": str(path.resolve()),
        "day": day,
        "rx": rx,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "num_tx": len(node_list),
        "total_samples": int(sum(tx_counts.values())),
        "tx_counts": tx_counts,
        "shape_examples": shape_examples,
    }

    del dataset
    gc.collect()
    return info


def scan_all_files(files: list[Path]) -> list[dict]:
    cache = load_cache()
    results = []

    print(f"\n发现 {len(files)} 个 non-EQ receiver 文件。")
    print("首次扫描会较慢；扫描结果会缓存。\n")

    for i, path in enumerate(files, start=1):
        key = str(path.resolve())
        stat = path.stat()
        cached = cache.get(key)

        if (
            cached
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            info = cached
            source = "cache"
        else:
            t0 = time.time()
            info = scan_receiver_file(path)
            cache[key] = info
            save_cache(cache)
            source = f"read {time.time() - t0:.1f}s"

        results.append(info)

        print(
            f"[{i:>3}/{len(files)}] "
            f"day={info['day']}  rx={info['rx']}  "
            f"Tx={info['num_tx']:>3}  samples={info['total_samples']:>8} "
            f"({source})"
        )

    return results


def choose_threshold_and_receiver(scan_results: list[dict]):
    levels = []
    for x in [PREFERRED_MIN_SAMPLES_PER_TX] + FALLBACK_MIN_SAMPLE_LEVELS:
        if x not in levels:
            levels.append(x)
    levels = sorted(levels, reverse=True)

    best_overall = None

    for threshold in levels:
        candidates = []

        for info in scan_results:
            eligible = {
                tx: int(n)
                for tx, n in info["tx_counts"].items()
                if int(n) >= threshold
            }

            top_counts = sorted(eligible.values(), reverse=True)[:TARGET_TOTAL_TX]
            score = (
                min(len(eligible), TARGET_TOTAL_TX),
                len(eligible),
                sum(top_counts),
            )

            candidates.append((score, info, eligible))

        candidates.sort(key=lambda x: x[0], reverse=True)
        score, info, eligible = candidates[0]

        if best_overall is None or score > best_overall[0]:
            best_overall = (score, threshold, info, eligible)

        if len(eligible) >= TARGET_TOTAL_TX:
            return threshold, info, eligible

    _, threshold, info, eligible = best_overall
    return threshold, info, eligible


def load_selected_receiver(info: dict) -> dict:
    path = Path(info["path"])
    print(f"\n读取最终选中的 receiver 文件：\n{path}\n")
    with path.open("rb") as f:
        return pickle.load(f)


def balanced_select_tx(
    eligible: dict[str, int],
    rng: np.random.Generator,
):
    eligible_tx = sorted(eligible.keys())

    if len(eligible_tx) < 2:
        raise RuntimeError("可用 Tx 太少，无法构造开放集。")

    perm = rng.permutation(len(eligible_tx))
    eligible_tx = [eligible_tx[i] for i in perm]

    total = min(TARGET_TOTAL_TX, len(eligible_tx))
    selected = eligible_tx[:total]

    if total >= TARGET_TOTAL_TX:
        known_n = TARGET_KNOWN_TX
    else:
        known_n = max(1, int(round(total * TARGET_KNOWN_TX / TARGET_TOTAL_TX)))
        known_n = min(known_n, total - 1)

    known = selected[:known_n]
    unknown = selected[known_n:]

    return selected, known, unknown


def get_tx_array(dataset: dict, tx_name: str) -> np.ndarray:
    node_list = [str(x) for x in dataset["node_list"]]
    index_map = {name: i for i, name in enumerate(node_list)}

    if tx_name not in index_map:
        raise KeyError(f"Tx {tx_name} 不在 node_list 中。")

    arr = np.asarray(dataset["data"][index_map[tx_name]])

    if arr.ndim != 3:
        raise ValueError(
            f"Tx {tx_name} shape={arr.shape}，预期类似 (N, 256, 2)。"
        )

    return arr


def split_known_tx(
    arr: np.ndarray,
    class_id: int,
    rng: np.random.Generator,
    samples_per_tx: int,
):
    idx = rng.permutation(arr.shape[0])[:samples_per_tx]
    arr = np.asarray(arr[idx], dtype=np.float32)

    n_train = int(samples_per_tx * TRAIN_FRAC)
    n_val = int(samples_per_tx * VAL_FRAC)
    n_test = samples_per_tx - n_train - n_val

    train_x = arr[:n_train]
    val_x = arr[n_train:n_train + n_val]
    test_x = arr[n_train + n_val:]

    train_y = np.full(n_train, class_id, dtype=np.int64)
    val_y = np.full(n_val, class_id, dtype=np.int64)
    test_y = np.full(n_test, class_id, dtype=np.int64)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def shuffle_xy(x, y, rng):
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def main():
    print("=" * 72)
    print("Full WiSig -> Pure Open-Set Subset")
    print("=" * 72)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"找不到 WiSig 根目录：{SOURCE_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_pkl = OUTPUT_DIR / "wisig_openset_subset.pkl"
    manifest_json = OUTPUT_DIR / "manifest.json"
    counts_csv = OUTPUT_DIR / "tx_counts.csv"
    readme_txt = OUTPUT_DIR / "README_subset.txt"

    if out_pkl.exists() and not OVERWRITE:
        print(f"输出已存在：{out_pkl}")
        print("如需覆盖，请将 OVERWRITE=True。")
        return

    files = discover_full_wisig_files(SOURCE_ROOT)
    if not files:
        raise FileNotFoundError(
            "未找到 non-EQ Full WiSig 文件。"
        )

    scan_results = scan_all_files(files)

    threshold, chosen_info, eligible = choose_threshold_and_receiver(scan_results)

    rng = np.random.default_rng(RANDOM_SEED)

    selected_tx, known_tx, unknown_tx = balanced_select_tx(eligible, rng)

    selected_counts = {tx: int(eligible[tx]) for tx in selected_tx}

    samples_per_tx = min(
        MAX_SAMPLES_PER_TX,
        min(selected_counts.values())
    )

    dataset = load_selected_receiver(chosen_info)

    first_arr = get_tx_array(dataset, selected_tx[0])
    sample_shape = tuple(first_arr.shape[1:])

    print("\n" + "=" * 72)
    print("自动选择结果")
    print("=" * 72)
    print(f"日期               : {chosen_info['day']}")
    print(f"接收机             : {chosen_info['rx']}")
    print(f"有效门槛           : >= {threshold} / Tx")
    print(f"实际选中 Tx        : {len(selected_tx)}")
    print(f"Known Tx           : {len(known_tx)}")
    print(f"Unknown Tx         : {len(unknown_tx)}")
    print(f"Known 每 Tx 总样本 : {samples_per_tx}")
    print(f"Unknown 主测试/Tx  : {UNKNOWN_TEST_PER_TX}")
    print(f"单条 IQ shape      : {sample_shape}")

    if UNKNOWN_TEST_PER_TX > samples_per_tx:
        raise ValueError(
            f"UNKNOWN_TEST_PER_TX={UNKNOWN_TEST_PER_TX} "
            f"大于统一样本数 {samples_per_tx}"
        )

    train_x_list, train_y_list = [], []
    val_x_list, val_y_list = [], []
    kt_x_list, kt_y_list = [], []

    unknown_main_x_list = []
    unknown_main_y_list = []
    unknown_main_identity_list = []

    unknown_full_x_list = []
    unknown_full_y_list = []
    unknown_full_identity_list = []

    known_label_map = {}
    unknown_identity_map = {}

    # ---------------- Known ----------------
    for class_id, tx in enumerate(known_tx):
        arr = get_tx_array(dataset, tx)
        known_label_map[tx] = class_id

        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_known_tx(
            arr, class_id, rng, samples_per_tx
        )

        train_x_list.append(tr_x)
        train_y_list.append(tr_y)
        val_x_list.append(va_x)
        val_y_list.append(va_y)
        kt_x_list.append(te_x)
        kt_y_list.append(te_y)

    # ---------------- Unknown ----------------
    for uid, tx in enumerate(unknown_tx):
        arr = get_tx_array(dataset, tx)
        unknown_identity_map[tx] = uid

        idx = rng.permutation(arr.shape[0])[:samples_per_tx]
        arr_full = np.asarray(arr[idx], dtype=np.float32)

        # 主测试：固定每类 240
        arr_main = arr_full[:UNKNOWN_TEST_PER_TX]

        unknown_main_x_list.append(arr_main)
        unknown_main_y_list.append(
            np.full(len(arr_main), -1, dtype=np.int64)
        )
        unknown_main_identity_list.append(
            np.full(len(arr_main), uid, dtype=np.int64)
        )

        # 压力测试：保留该 Unknown Tx 的全部统一样本
        unknown_full_x_list.append(arr_full)
        unknown_full_y_list.append(
            np.full(len(arr_full), -1, dtype=np.int64)
        )
        unknown_full_identity_list.append(
            np.full(len(arr_full), uid, dtype=np.int64)
        )

    # ---------------- 拼接 ----------------
    train_x = np.concatenate(train_x_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)

    val_x = np.concatenate(val_x_list, axis=0)
    val_y = np.concatenate(val_y_list, axis=0)

    known_test_x = np.concatenate(kt_x_list, axis=0)
    known_test_y = np.concatenate(kt_y_list, axis=0)

    unknown_test_x = np.concatenate(unknown_main_x_list, axis=0)
    unknown_test_y = np.concatenate(unknown_main_y_list, axis=0)
    unknown_identity_y = np.concatenate(unknown_main_identity_list, axis=0)

    unknown_full_x = np.concatenate(unknown_full_x_list, axis=0)
    unknown_full_y = np.concatenate(unknown_full_y_list, axis=0)
    unknown_full_identity_y = np.concatenate(
        unknown_full_identity_list, axis=0
    )

    # ---------------- 打乱 ----------------
    train_x, train_y = shuffle_xy(train_x, train_y, rng)
    val_x, val_y = shuffle_xy(val_x, val_y, rng)
    known_test_x, known_test_y = shuffle_xy(
        known_test_x, known_test_y, rng
    )

    p = rng.permutation(len(unknown_test_x))
    unknown_test_x = unknown_test_x[p]
    unknown_test_y = unknown_test_y[p]
    unknown_identity_y = unknown_identity_y[p]

    p = rng.permutation(len(unknown_full_x))
    unknown_full_x = unknown_full_x[p]
    unknown_full_y = unknown_full_y[p]
    unknown_full_identity_y = unknown_full_identity_y[p]

    # ---------------- 主 Open Test ----------------
    open_test_x = np.concatenate(
        [known_test_x, unknown_test_x], axis=0
    )
    open_test_y = np.concatenate(
        [known_test_y, unknown_test_y], axis=0
    )

    p = rng.permutation(len(open_test_x))
    open_test_x = open_test_x[p]
    open_test_y = open_test_y[p]

    # ---------------- Full Unknown 压力测试 ----------------
    open_test_full_unknown_x = np.concatenate(
        [known_test_x, unknown_full_x], axis=0
    )
    open_test_full_unknown_y = np.concatenate(
        [known_test_y, unknown_full_y], axis=0
    )

    p = rng.permutation(len(open_test_full_unknown_x))
    open_test_full_unknown_x = open_test_full_unknown_x[p]
    open_test_full_unknown_y = open_test_full_unknown_y[p]

    compact = {
        "meta": {
            "source_root": str(SOURCE_ROOT),
            "source_file": chosen_info["path"],
            "day": chosen_info["day"],
            "rx": chosen_info["rx"],
            "equalized": False,
            "random_seed": RANDOM_SEED,
            "selection_threshold": threshold,
            "samples_per_tx": samples_per_tx,
            "unknown_test_per_tx": UNKNOWN_TEST_PER_TX,
            "sample_shape": sample_shape,
            "known_tx_list": known_tx,
            "unknown_tx_list": unknown_tx,
            "known_label_map": known_label_map,
            "unknown_identity_map": unknown_identity_map,
            "label_convention": {
                "known": "0..K-1",
                "unknown": -1,
            },
            "split": {
                "known_train_frac": TRAIN_FRAC,
                "known_val_frac": VAL_FRAC,
                "known_test_frac": 1.0 - TRAIN_FRAC - VAL_FRAC,
                "unknown_main_test_per_tx": UNKNOWN_TEST_PER_TX,
                "unknown_training": False,
            },
        },
        "train": {
            "x": train_x,
            "y": train_y,
        },
        "val": {
            "x": val_x,
            "y": val_y,
        },
        "known_test": {
            "x": known_test_x,
            "y": known_test_y,
        },
        "unknown_test": {
            "x": unknown_test_x,
            "y": unknown_test_y,
            "unknown_identity_y": unknown_identity_y,
        },
        "open_test": {
            "x": open_test_x,
            "y": open_test_y,
            "is_unknown": open_test_y == -1,
        },
        "unknown_test_full": {
            "x": unknown_full_x,
            "y": unknown_full_y,
            "unknown_identity_y": unknown_full_identity_y,
        },
        "open_test_full_unknown": {
            "x": open_test_full_unknown_x,
            "y": open_test_full_unknown_y,
            "is_unknown": open_test_full_unknown_y == -1,
        },
    }

    # ---------------- 保存 ----------------
    with out_pkl.open("wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = compact["meta"].copy()
    manifest["sizes"] = {
        "train": int(len(train_x)),
        "val": int(len(val_x)),
        "known_test": int(len(known_test_x)),
        "unknown_test": int(len(unknown_test_x)),
        "open_test": int(len(open_test_x)),
        "unknown_test_full": int(len(unknown_full_x)),
        "open_test_full_unknown": int(len(open_test_full_unknown_x)),
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with counts_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["tx", "raw_sample_count", "role", "class_id"]
        )

        for tx in selected_tx:
            if tx in known_label_map:
                role = "known"
                class_id = known_label_map[tx]
            else:
                role = "unknown"
                class_id = unknown_identity_map[tx]

            writer.writerow(
                [tx, selected_counts[tx], role, class_id]
            )

    readme = f"""WiSig Pure Open-Set Subset
==========================

固定条件
--------
Day: {chosen_info['day']}
Rx: {chosen_info['rx']}
Equalized: False

类别
----
Known Tx: {len(known_tx)}
Unknown Tx: {len(unknown_tx)}

样本
----
Known 每 Tx 总样本: {samples_per_tx}
Unknown 主测试每 Tx: {UNKNOWN_TEST_PER_TX}
单条 IQ shape: {sample_shape}

Known 划分
----------
Train: {TRAIN_FRAC:.0%}
Val: {VAL_FRAC:.0%}
Known Test: {1.0 - TRAIN_FRAC - VAL_FRAC:.0%}

主开放集测试
------------
Known Test: {len(known_test_x)}
Unknown Test: {len(unknown_test_x)}
Open Test: {len(open_test_x)}

压力测试
--------
Unknown Full: {len(unknown_full_x)}
Open Test Full Unknown: {len(open_test_full_unknown_x)}

标签
----
Known: 0..{len(known_tx)-1}
Unknown: -1
"""

    readme_txt.write_text(readme, encoding="utf-8")

    del dataset
    gc.collect()

    print("\n" + "=" * 72)
    print("完成")
    print("=" * 72)
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"主数据文件：{out_pkl}")
    print()
    print(f"train                  = {len(train_x)}")
    print(f"val                    = {len(val_x)}")
    print(f"known_test             = {len(known_test_x)}")
    print(f"unknown_test           = {len(unknown_test_x)}")
    print(f"open_test              = {len(open_test_x)}")
    print(f"unknown_test_full      = {len(unknown_full_x)}")
    print(f"open_test_full_unknown = {len(open_test_full_unknown_x)}")
    print()
    print(
        f"主 Open Test / Train = "
        f"{len(open_test_x) / len(train_x):.3f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。扫描缓存会保留。")
        sys.exit(130)
    except Exception as e:
        print("\n运行失败：")
        print(repr(e))
        raise
