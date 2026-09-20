"""Train and evaluate the explicit Communication stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage3 import train_stage3  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "stage1_checkpoint", "output_dir"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    result = train_stage3(config)
    print("\n================ Stage-3 result ================")
    for name, metric in result["metrics"].items():
        print(f"{name:27s} AUROC={metric['auroc']:.4f} OSCR={metric['oscr']:.4f} "
              f"Kacc={metric['known_accuracy']:.4f} Urec={metric['unknown_recall']:.4f} "
              f"H={metric['h_score']:.4f}")
    print("report:", Path(config["output_dir"]) / "report_stage3.md")


if __name__ == "__main__":
    main()

