"""Run Stage 2: disagreement-guided pseudo-unknown + boundary Agent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage2 import train_stage2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "stage1_checkpoint", "output_dir"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    summary = train_stage2(config)
    print("\n================ Stage-2 result ================")
    for name, metric in summary["open_test_metrics"].items():
        print(f"{name:14s} AUROC={metric['auroc']:.4f} OSCR={metric['oscr']:.4f} "
              f"Kacc={metric['known_accuracy']:.4f} Urec={metric['unknown_recall']:.4f} "
              f"H={metric['h_score']:.4f}")
    print("report:", Path(config["output_dir"]) / "report_stage2.md")


if __name__ == "__main__":
    main()

