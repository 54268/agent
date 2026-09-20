"""Run Stage 2 on matched Stage-1 seeds and aggregate without test-time tuning."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage2 import train_stage2  # noqa: E402


METRICS = ("auroc", "aupr_out", "fpr95", "oscr", "known_accuracy",
           "unknown_recall", "macro_f1", "h_score")
RULES = ("stage1_mean3", "stage1_max3", "boundary", "mean4", "max4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    base = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root"):
        if key in base and not Path(base[key]).is_absolute():
            base[key] = str((ROOT / base[key]).resolve())
    out_root = ROOT / "results" / "stage2" / "multiseed" / base["dataset"]
    reports = []
    for seed in args.seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["stage1_checkpoint"] = str(
            ROOT / "results" / "stage1" / "multiseed" / base["dataset"]
            / f"seed{seed}" / "model.pt"
        )
        cfg["output_dir"] = str(out_root / f"seed{seed}")
        reports.append(train_stage2(cfg))

    aggregate = {"dataset": base["dataset"], "seeds": args.seeds, "rules": {},
                 "pseudo_validation": {}}
    for rule in RULES:
        aggregate["rules"][rule] = {}
        for metric in METRICS:
            values = [r["open_test_metrics"][rule][metric] for r in reports]
            aggregate["rules"][rule][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    for metric in ("auroc", "recall_at_tau_0.95"):
        values = [r["pseudo_validation"][metric] for r in reports]
        aggregate["pseudo_validation"][metric] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "aggregate_stage2.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    names = {"stage1_mean3": "Stage-1 均值", "stage1_max3": "Stage-1 max/OR",
             "boundary": "Boundary", "mean4": "四证据均值", "max4": "四证据 max/OR"}
    lines = [f"# Stage-2 多种子聚合 — {base['dataset']}（{args.seeds}）", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for rule in RULES:
        row = aggregate["rules"][rule]
        value = lambda key: f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
        lines.append(f"| {names[rule]} | {value('auroc')} | {value('oscr')} | "
                     f"{value('known_accuracy')} | {value('unknown_recall')} | {value('h_score')} |")
    pv = aggregate["pseudo_validation"]
    lines += ["", f"留出伪未知 AUROC={pv['auroc']['mean']:.3f}±{pv['auroc']['std']:.3f}；"
              f"Recall@τ=.95={pv['recall_at_tau_0.95']['mean']:.3f}±"
              f"{pv['recall_at_tau_0.95']['std']:.3f}。",
              "真实 Unknown 仅用于每个 seed 冻结后的最终离线评价。"]
    (out_root / "aggregate_stage2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
