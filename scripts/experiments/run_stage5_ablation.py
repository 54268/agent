"""Run a frozen Stage-5 component ablation without repeating LCO selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.experiment import run_final_seed  # noqa: E402
from maros_staged.datasets import load_wisig_subset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", choices=["no_reconstruction"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "output_dir"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    result_root = Path(cfg["output_dir"])
    selection = json.loads((result_root / "lco" / "lco_selection.json").read_text(
        encoding="utf-8"))["selected"]
    if args.ablation == "no_reconstruction":
        cfg["reconstruction_weight"] = 0.0
    splits = load_wisig_subset(cfg["wisig_pkl"], augment_train=bool(cfg.get("augment_train", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparisons = []
    for seed in args.seeds:
        output = result_root / "ablations" / args.ablation / f"seed{seed}"
        result = run_final_seed(splits, cfg, selection, device, output, seed)
        reference_path = result_root / "multiseed" / f"seed{seed}" / "stage5_metrics.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        deltas = {metric: result["metrics"]["communication"][metric]
                  - reference["metrics"]["communication"][metric]
                  for metric in ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")}
        comparisons.append({"seed": seed, "ablation": result["metrics"]["communication"],
                            "reference": reference["metrics"]["communication"], "delta": deltas})
    payload = {"ablation": args.ablation, "selection_frozen": selection,
               "changed": {"reconstruction_weight": cfg["reconstruction_weight"]},
               "comparisons": comparisons}
    out_dir = result_root / "ablations" / args.ablation
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    lines = [f"# Stage-5 消融：{args.ablation}", "",
             "LCO 选择与其他训练配置保持冻结。", "",
             "| seed | ΔAUROC | ΔOSCR | ΔKnown Acc | ΔUnknown Recall | ΔH |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        d = row["delta"]
        lines.append(f"| {row['seed']} | {d['auroc']:+.4f} | {d['oscr']:+.4f} | "
                     f"{d['known_accuracy']:+.4f} | {d['unknown_recall']:+.4f} | "
                     f"{d['h_score']:+.4f} |")
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
