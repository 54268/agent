"""Build the processed ORACLE KRI-16 npz splits used by Stage 1.

Usage:  python scripts/data/build_oracle_stage1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.prepare_oracle import prepare_oracle  # noqa: E402


def main() -> None:
    raw_root = ROOT / "data/oracle/KRI-16IQImbalances-DemodulatedData"
    output_root = ROOT / "data/oracle/processed_k10u6"
    summary = prepare_oracle(raw_root, output_root)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
