"""Run Stage 1 over several seeds and aggregate the complementarity diagnostics.

This avoids drawing conclusions from a single run.  Per-seed artifacts go under
``<output_root>/seed<seed>/`` and an aggregate ``aggregate_stage1.md/json`` is
written at ``<output_root>``.

Usage:
    python scripts/experiments/run_stage1_multiseed.py --config configs/experiments/stage1_wisig.json
    python scripts/experiments/run_stage1_multiseed.py --config configs/experiments/stage1_oracle.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from maros_staged.diagnose import CHANNELS, FUSIONS, run_diagnostic  # noqa: E402
from maros_staged.train_stage1 import train_stage1  # noqa: E402


METRICS = ["auroc", "aupr_out", "fpr95", "oscr", "unknown_recall", "known_accuracy", "h_score"]
GATE_KEYS = ["fused_auroc_gain", "max_abs_score_correlation",
             "identity_missed_unknown_rescued_fraction", "union_recall",
             "best_single_channel_recall"]


def resolve_paths(config: dict) -> dict:
    for key in ("wisig_pkl", "oracle_root"):
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str((ROOT / config[key]).resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    base = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    base = resolve_paths(base)
    out_root = ROOT / "results/stage1/multiseed" / base["dataset"]
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        cfg["output_dir"] = str(out_root / f"seed{seed}")
        result = train_stage1(cfg)
        report = run_diagnostic(result["open_data"], result["out_dir"],
                                f"{base.get('name', base['dataset'])}_seed{seed}")
        per_seed.append(report)
        del result  # free model/embeddings before the next run

    # aggregate
    rules = CHANNELS + FUSIONS
    agg = {"dataset": base["dataset"], "seeds": args.seeds, "channels": {}, "gate": {}}
    for rule in rules:
        agg["channels"][rule] = {}
        for metric in METRICS:
            vals = [r["channel_metrics"][rule][metric] for r in per_seed]
            agg["channels"][rule][metric] = {"mean": float(np.mean(vals)),
                                             "std": float(np.std(vals, ddof=0)),
                                             "values": [float(v) for v in vals]}
    for key in GATE_KEYS:
        vals = [r["gate"][key] for r in per_seed]
        agg["gate"][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
                            "values": [float(v) for v in vals]}
    for bool_key in ("fused_beats_best_single_auroc", "not_collapsed_corr_lt_0_95",
                     "union_adds_recall"):
        agg["gate"][bool_key] = {"fraction_true": float(np.mean([r["gate"][bool_key] for r in per_seed])),
                                 "values": [bool(r["gate"][bool_key]) for r in per_seed]}

    (out_root / "aggregate_stage1.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Stage-1 多种子聚合 — {base['dataset']}（seeds={args.seeds}）", "",
             "## 各证据通道（均值±标准差）", "",
             "| 规则 | AUROC | AUPR-Out | FPR95 | OSCR | Unknown Recall | H-score |",
             "|---|---|---|---|---|---|---|"]
    name = {"identity": "Identity", "prototype": "Prototype", "reconstruction": "Reconstruction",
            "fused_mean": "融合·均值", "fused_max": "融合·max/OR"}
    for rule in rules:
        a = agg["channels"][rule]
        lines.append(
            f"| {name[rule]} | {a['auroc']['mean']:.3f}±{a['auroc']['std']:.3f} "
            f"| {a['aupr_out']['mean']:.3f}±{a['aupr_out']['std']:.3f} "
            f"| {a['fpr95']['mean']:.3f}±{a['fpr95']['std']:.3f} "
            f"| {a['oscr']['mean']:.3f}±{a['oscr']['std']:.3f} "
            f"| {a['unknown_recall']['mean']:.3f}±{a['unknown_recall']['std']:.3f} "
            f"| {a['h_score']['mean']:.3f}±{a['h_score']['std']:.3f} |")
    g = agg["gate"]
    lines += ["", "## 互补性闸门（跨种子）", ""]
    for key, title in [
        ("fused_auroc_gain", "最优融合相对最强单证据 AUROC 增益"),
        ("max_abs_score_correlation", "三证据最大 |Spearman| 相关"),
        ("identity_missed_unknown_rescued_fraction", "Identity 漏检被其他证据救援比例"),
        ("union_recall", "三证据并集 Unknown 召回"),
        ("best_single_channel_recall", "最强单证据 Unknown 召回"),
    ]:
        s = g[key]
        lines.append(f"- {title}：{s['mean']:.3f}±{s['std']:.3f}  各次 {[round(v,3) for v in s['values']]}")
    for key, title in [
        ("fused_beats_best_single_auroc", "存在固定融合 AUROC 超过最强单证据"),
        ("not_collapsed_corr_lt_0_95", "证据未完全相关（<0.95）"),
        ("union_adds_recall", "并集召回超过最强单证据"),
    ]:
        s = g[key]
        lines.append(f"- {title}：{s['fraction_true']*100:.0f}% 种子成立  {s['values']}")
    (out_root / "aggregate_stage1.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("aggregate:", out_root / "aggregate_stage1.md")


if __name__ == "__main__":
    main()
