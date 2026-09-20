"""Run Stage-4 reliability routing on matched Stage-1/Stage-3 seeds."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage4 import train_stage4  # noqa: E402


METRICS = ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")
RULES = (
    "no_communication",
    "communication",
    "global",
    "single",
    "dual",
    "dual_uniform",
    "dual_shuffled",
    "dual_plus_three_mean",
)
ROUTE_KEYS = (
    "known_known_weights",
    "unknown_known_weights",
    "known_unknown_weights",
    "unknown_unknown_weights",
)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean(axis=0)) if values.ndim == 1 else values.mean(axis=0).tolist(),
        "std": float(values.std(axis=0)) if values.ndim == 1 else values.std(axis=0).tolist(),
        "values": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    base = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if "wisig_pkl" in base and not Path(base["wisig_pkl"]).is_absolute():
        base["wisig_pkl"] = str((ROOT / base["wisig_pkl"]).resolve())
    out_root = ROOT / "results" / "stage4" / "multiseed" / base["dataset"]

    results = []
    for seed in args.seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["stage1_checkpoint"] = str(
            ROOT / "results" / "stage1" / "multiseed" / base["dataset"]
            / f"seed{seed}" / "model.pt"
        )
        stage3 = ROOT / "results" / "stage3" / "multiseed" / base["dataset"] / f"seed{seed}"
        cfg["no_communication_checkpoint"] = str(stage3 / "no_communication.pt")
        cfg["communication_checkpoint"] = str(stage3 / "communication.pt")
        cfg["output_dir"] = str(out_root / f"seed{seed}")
        print(f"\n[stage4-multiseed] seed={seed}")
        results.append(train_stage4(cfg))

    aggregate = {
        "dataset": base["dataset"],
        "seeds": args.seeds,
        "protocol": {
            "real_unknown_used_for_training": False,
            "real_unknown_used_for_threshold": False,
            "matched_stage1_stage3_seeds": True,
        },
        "rules": {},
        "route_summary": {},
    }
    for rule in RULES:
        aggregate["rules"][rule] = {
            metric: _stats([result["metrics"][rule][metric] for result in results])
            for metric in METRICS
        }
    for key in ROUTE_KEYS:
        aggregate["route_summary"][key] = _stats(
            [result["route_summary"][key] for result in results]
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "aggregate_stage4.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    names = {
        "no_communication": "固定无通信",
        "communication": "固定通信",
        "global": "全局学习权重",
        "single": "样本级单路由",
        "dual": "样本级双路由",
        "dual_uniform": "双路由·强制均匀",
        "dual_shuffled": "双路由·打乱权重",
        "dual_plus_three_mean": "双路由 + 三基础证据均值",
    }
    lines = [
        f"# Stage-4 Reliability Router 多种子聚合 — {base['dataset']}（{args.seeds}）",
        "",
        "所有完整规则都分别由 Known validation CDF 校准，并在约 95% Known 接受率下评价；真实 Unknown 不参与训练、模型选择或阈值设定。",
        "",
        "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule in RULES:
        row = aggregate["rules"][rule]
        value = lambda key: f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
        lines.append(
            f"| {names[rule]} | {value('auroc')} | {value('oscr')} | "
            f"{value('known_accuracy')} | {value('unknown_recall')} | {value('h_score')} |"
        )

    lines += ["", "## 双路由平均权重 [无通信, 通信]", ""]
    route_names = {
        "known_known_weights": "Known 样本的 Known 路由",
        "unknown_known_weights": "Unknown 样本的 Known 路由",
        "known_unknown_weights": "Known 样本的 Unknown 路由",
        "unknown_unknown_weights": "Unknown 样本的 Unknown 路由",
    }
    for key in ROUTE_KEYS:
        item = aggregate["route_summary"][key]
        mean = ", ".join(f"{x:.3f}" for x in item["mean"])
        std = ", ".join(f"{x:.3f}" for x in item["std"])
        lines.append(f"- {route_names[key]}：mean=[{mean}]，std=[{std}]。")
    report = "\n".join(lines)
    (out_root / "aggregate_stage4.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
