"""Read-only WiSig support audit for reproducible protocol design.

The script loads one canonical pickle per (capture date, receiver), records the
number of packets available for every transmitter, and writes a compact JSON
report outside the source dataset. It never modifies WiSig files.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


FILENAME_RE = re.compile(
    r"^dataset_(?P<date>\d{4}_\d{2}_\d{2})_node(?P<rx>.+)\.pkl$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def discover_canonical_files(data_root: Path) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(data_root.rglob("*.pkl")):
        match = FILENAME_RE.match(path.name)
        if match:
            groups[(match["date"], match["rx"])].append(path)

    canonical: list[dict] = []
    duplicates: list[dict] = []
    for (date, rx), paths in sorted(groups.items()):
        canonical.append({"date": date, "rx": rx, "path": paths[0]})
        if len(paths) > 1:
            duplicates.append(
                {
                    "date": date,
                    "rx": rx,
                    "kept": str(paths[0]),
                    "ignored": [str(path) for path in paths[1:]],
                }
            )
    return canonical, duplicates


def audit(data_root: Path) -> dict:
    canonical, duplicates = discover_canonical_files(data_root)
    if not canonical:
        raise FileNotFoundError(f"No canonical WiSig pickle files found under {data_root}")

    cells: list[dict] = []
    dates_to_rx: dict[str, set[str]] = defaultdict(set)
    tx_total: dict[str, int] = defaultdict(int)
    tx_positive_cells: dict[str, int] = defaultdict(int)
    all_tx: set[str] = set()

    for index, item in enumerate(canonical, start=1):
        path = item["path"]
        print(f"[{index:03d}/{len(canonical):03d}] {item['date']} rx={item['rx']}", flush=True)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        if set(payload) != {"data", "node_list"}:
            raise ValueError(f"Unexpected keys in {path}: {sorted(payload)}")
        if len(payload["data"]) != len(payload["node_list"]):
            raise ValueError(f"data/node_list length mismatch in {path}")

        dates_to_rx[item["date"]].add(item["rx"])
        for tx, block in zip(payload["node_list"], payload["data"]):
            tx = str(tx)
            count = int(len(block))
            all_tx.add(tx)
            tx_total[tx] += count
            if count > 0:
                tx_positive_cells[tx] += 1
            cells.append(
                {
                    "date": item["date"],
                    "rx": item["rx"],
                    "tx": tx,
                    "packets": count,
                }
            )
        del payload

    common_rx = sorted(set.intersection(*(set(v) for v in dates_to_rx.values())))
    common_rx_set = set(common_rx)
    common_cells = [cell for cell in cells if cell["rx"] in common_rx_set]

    common_stats: dict[str, dict] = {}
    for tx in sorted(all_tx):
        counts = [cell["packets"] for cell in common_cells if cell["tx"] == tx]
        positive = [count for count in counts if count > 0]
        common_stats[tx] = {
            "positive_cells": len(positive),
            "total_cells": len(counts),
            "min_positive_packets": min(positive) if positive else 0,
            "median_positive_packets": sorted(positive)[len(positive) // 2] if positive else 0,
            "total_packets": sum(counts),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "canonical_file_count": len(canonical),
        "duplicate_groups": duplicates,
        "capture_dates": sorted(dates_to_rx),
        "receivers_by_date": {
            date: sorted(receivers) for date, receivers in sorted(dates_to_rx.items())
        },
        "common_receivers_all_dates": common_rx,
        "transmitters": sorted(all_tx),
        "transmitter_summary": {
            tx: {
                "total_packets": tx_total[tx],
                "positive_cells_all_files": tx_positive_cells[tx],
            }
            for tx in sorted(all_tx)
        },
        "common_receiver_support": common_stats,
        "cells": cells,
    }


def main() -> None:
    args = parse_args()
    report = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
