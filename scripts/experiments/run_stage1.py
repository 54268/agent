"""Train the Stage-1 three-agent model and run the complementarity diagnostic.

Usage:
    python scripts/experiments/run_stage1.py --config configs/experiments/stage1_wisig.json
    python scripts/experiments/run_stage1.py --config configs/experiments/stage1_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.diagnose import run_diagnostic  # noqa: E402
from maros_staged.train_stage1 import train_stage1  # noqa: E402


def resolve_paths(config: dict) -> dict:
    for key in ("wisig_pkl", "oracle_root", "output_dir"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config = resolve_paths(config)

    result = train_stage1(config)
    report = run_diagnostic(result["open_data"], result["out_dir"],
                            config.get("name", config["dataset"]))

    print("\n================ Stage-1 complementarity ================")
    for ch, m in report["channel_metrics"].items():
        print(f"{ch:15s} AUROC={m['auroc']:.4f} AUPR={m['aupr_out']:.4f} "
              f"FPR95={m['fpr95']:.4f} OSCR={m['oscr']:.4f} "
              f"Kacc={m['known_accuracy']:.4f} Urec={m['unknown_recall']:.4f}")
    print("gate:", json.dumps(report["gate"], indent=2, ensure_ascii=False))
    print("report:", Path(result["out_dir"]) / "report_stage1.md")


if __name__ == "__main__":
    main()
