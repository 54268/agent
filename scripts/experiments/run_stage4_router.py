"""Train reliability routers over formal Stage-3 checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage4 import train_stage4  # noqa: E402


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "stage1_checkpoint",
                "no_communication_checkpoint", "communication_checkpoint", "output_dir"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    result = train_stage4(config)
    print("\n================ Stage-4 result ================")
    for name, m in result["metrics"].items():
        print(f"{name:24s} AUROC={m['auroc']:.4f} OSCR={m['oscr']:.4f} "
              f"Kacc={m['known_accuracy']:.4f} Urec={m['unknown_recall']:.4f} H={m['h_score']:.4f}")
    print("report:", Path(config["output_dir"]) / "report_stage4.md")


if __name__ == "__main__":
    main()

