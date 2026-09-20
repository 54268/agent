"""Run explicit Communication on matched Stage-1 seeds and aggregate."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.train_stage3 import train_stage3  # noqa: E402


METRICS = ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")
RULES = ("no_comm_trained", "communication", "messages_off", "messages_shuffled",
         "drop_identity_sender", "drop_prototype_sender", "drop_reconstruction_sender",
         "communication_mean4", "communication_max4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    base = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root"):
        if key in base and not Path(base[key]).is_absolute():
            base[key] = str((ROOT / base[key]).resolve())
    out_root = ROOT / "results" / "stage3" / "multiseed" / base["dataset"]
    results = []
    for seed in args.seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["stage1_checkpoint"] = str(
            ROOT / "results" / "stage1" / "multiseed" / base["dataset"]
            / f"seed{seed}" / "model.pt"
        )
        cfg["output_dir"] = str(out_root / f"seed{seed}")
        results.append(train_stage3(cfg))

    aggregate = {"dataset": base["dataset"], "seeds": args.seeds,
                 "rules": {}, "interventions": {}}
    for rule in RULES:
        aggregate["rules"][rule] = {}
        for metric in METRICS:
            values = [result["metrics"][rule][metric] for result in results]
            aggregate["rules"][rule][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(value) for value in values],
            }
    for intervention in ("messages_off", "messages_shuffled", "drop_identity_sender",
                         "drop_prototype_sender", "drop_reconstruction_sender"):
        aggregate["interventions"][intervention] = {}
        for metric in ("class_prediction_flip_rate", "unknown_score_mean_abs_change",
                       "auroc_delta", "oscr_delta"):
            values = [result["interventions"][intervention][metric] for result in results]
            aggregate["interventions"][intervention][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(value) for value in values],
            }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "aggregate_stage3.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    names = {"no_comm_trained": "同构无通信", "communication": "显式通信",
             "messages_off": "推理关闭消息", "messages_shuffled": "打乱消息",
             "drop_identity_sender": "删除 I 发送边", "drop_prototype_sender": "删除 P 发送边",
             "drop_reconstruction_sender": "删除 R 发送边",
             "communication_mean4": "通信+三证据均值", "communication_max4": "通信+三证据 max"}
    lines = [f"# Stage-3 Communication 多种子聚合 — {base['dataset']}（{args.seeds}）", "",
             "所有规则各自使用 Known validation CDF，在约 95% Known 接受率下评价。", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for rule in RULES:
        row = aggregate["rules"][rule]
        value = lambda key: f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
        lines.append(f"| {names[rule]} | {value('auroc')} | {value('oscr')} | "
                     f"{value('known_accuracy')} | {value('unknown_recall')} | {value('h_score')} |")
    lines += ["", "## 消息因果干预（相对正常通信）", ""]
    for intervention in aggregate["interventions"]:
        row = aggregate["interventions"][intervention]
        lines.append(
            f"- {names[intervention]}：类别翻转率 {row['class_prediction_flip_rate']['mean']:.3f}±"
            f"{row['class_prediction_flip_rate']['std']:.3f}；ΔAUROC "
            f"{row['auroc_delta']['mean']:+.3f}±{row['auroc_delta']['std']:.3f}；ΔOSCR "
            f"{row['oscr_delta']['mean']:+.3f}±{row['oscr_delta']['std']:.3f}。"
        )
    (out_root / "aggregate_stage3.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
