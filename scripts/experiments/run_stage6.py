"""Run Stage-6 collaboration screening and guarded formal evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage6.experiment import CANDIDATES, METRIC_NAMES, run_stage6  # noqa: E402
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision  # noqa: E402


CONFIG_METADATA = {
    "dataset", "name", "wisig_pkl", "oracle_root", "output_dir",
    "paired_config", "comparison_target",
}


def _resolve_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "output_dir", "paired_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def assert_paired_method_config(cfg: dict) -> None:
    if not cfg.get("paired_config"):
        return
    peer_path = Path(cfg["paired_config"])
    peer = _resolve_config(peer_path)
    left = {key: value for key, value in cfg.items() if key not in CONFIG_METADATA}
    right = {key: value for key, value in peer.items() if key not in CONFIG_METADATA}
    if left != right:
        different = sorted(set(left) | set(right))
        different = [key for key in different if left.get(key) != right.get(key)]
        raise ValueError(f"WiSig/ORACLE Stage-6 method configs differ: {different}")


def render_lco_report(payload: dict) -> str:
    summary = payload["summary"]
    selection = payload["selection"]
    lines = ["# Stage-6 LCO 协作筛选", "",
             f"选择：`{selection['selected']}`；协作晋级："
             f"`{selection['collaboration_promoted']}`。", "",
             "真实 Unknown 未参与训练、选模、阈值或停止轮次。", "",
             "| 候选 | H-score | OSCR | Known Acc | Unknown Recall | Query rate | Mean bits |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CANDIDATES:
        if name not in summary:
            continue
        row = summary[name]
        metric = row["metrics"]
        lines.append(
            f"| {name} | {metric['h_score']['mean']:.4f} | "
            f"{metric['oscr']['mean']:.4f} | {metric['known_accuracy']['mean']:.4f} | "
            f"{metric['unknown_recall']['mean']:.4f} | {row['query_rate']:.3f} | "
            f"{row['mean_bits']:.1f} |")
    lines += ["", "## 协作晋级检查", ""]
    for name, row in selection["checks"].items():
        lines.append(
            f"- {name}: {'通过' if row['passes'] else '未通过'}；"
            f"ΔH={row['delta_h']:+.4f}，ΔOSCR={row['delta_oscr']:+.4f}，"
            f"正增益 folds={row['positive_h_folds']}/{row['required_positive_h_folds']}。")
    if not selection["collaboration_promoted"]:
        lines += ["", "未通过时正式真实 Unknown 评价被自动阻断，不能强行选通信模型。"]
    return "\n".join(lines)


def _metrics(y, pred, score, tau):
    unknown = (y == -1).astype(np.int32)
    result = detection_metrics(unknown, score)
    result["oscr"] = oscr(unknown, pred, y, 1.0 - score)
    result.update(threshold_decision(unknown, y, pred, score, tau))
    return result


def grouped_bootstrap(out_root: Path, seeds: list[int], repetitions: int,
                      rng_seed: int = 2026) -> dict:
    payloads = []
    for seed in seeds:
        path = out_root / "multiseed" / f"seed{seed}" / "scores_and_transcript.npz"
        with np.load(path) as data:
            payloads.append({key: data[key].copy() for key in data.files})
    y, groups = payloads[0]["y"], payloads[0]["group_id"]
    known_groups = np.unique(groups[y != -1]); unknown_groups = np.unique(groups[y == -1])
    indices = {int(group): np.flatnonzero(groups == group)
               for group in np.r_[known_groups, unknown_groups]}
    rng = np.random.default_rng(rng_seed)
    values = {metric: [] for metric in METRIC_NAMES}
    for _ in range(repetitions):
        chosen = np.r_[rng.choice(known_groups, len(known_groups), True),
                       rng.choice(unknown_groups, len(unknown_groups), True)]
        sample = np.concatenate([indices[int(group)] for group in chosen])
        per_seed = {metric: [] for metric in METRIC_NAMES}
        for row in payloads:
            full = _metrics(row["y"][sample], row["pred_communication"][sample],
                            row["u_communication"][sample], row["tau_communication"][sample])
            base = _metrics(row["y"][sample], row["pred_no_communication"][sample],
                            row["u_no_communication"][sample],
                            row["tau_no_communication"][sample])
            for metric in METRIC_NAMES:
                per_seed[metric].append(full[metric] - base[metric])
        for metric in METRIC_NAMES:
            values[metric].append(float(np.mean(per_seed[metric])))
    return {metric: {"mean": float(np.mean(row)),
                     "ci95": [float(x) for x in np.quantile(row, [0.025, 0.975])]}
            for metric, row in values.items()}


def aggregate_formal(cfg: dict, results: list[dict], seeds: list[int], elapsed: float) -> dict:
    rules = sorted(results[0]["metrics"])
    aggregate = {"seeds": seeds, "elapsed_seconds": elapsed, "rules": {},
                 "collaboration": {}, "protocol": results[0]["protocol"]}
    for rule in rules:
        aggregate["rules"][rule] = {}
        for metric in (*METRIC_NAMES, "aupr_out", "fpr95", "macro_f1"):
            values = [row["metrics"][rule][metric] for row in results]
            aggregate["rules"][rule][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(value) for value in values],
            }
    for key in results[0]["collaboration"]:
        values = [row["collaboration"][key] for row in results]
        aggregate["collaboration"][key] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)),
            "values": values,
        }
    aggregate["paired_bootstrap_delta_vs_no_comm"] = grouped_bootstrap(
        Path(cfg["output_dir"]), seeds, int(cfg.get("bootstrap_repetitions", 1000)))
    full = aggregate["rules"]["communication"]
    no = aggregate["rules"]["no_communication"]
    best_single_h = max(aggregate["rules"][role]["h_score"]["mean"]
                        for role in ("waveform", "prototype", "verifier"))
    ci = aggregate["paired_bootstrap_delta_vs_no_comm"]
    aggregate["success_gates"] = {
        "delta_h_ge_0_01": full["h_score"]["mean"] - no["h_score"]["mean"] >= 0.01,
        "delta_oscr_ge_0_005": full["oscr"]["mean"] - no["oscr"]["mean"] >= 0.005,
        "positive_h_seeds_ge_2": sum(a > b for a, b in zip(
            full["h_score"]["values"], no["h_score"]["values"])) >= 2,
        "beats_best_single_h_by_0_005": full["h_score"]["mean"] - best_single_h >= 0.005,
        "messages_off_drop_h_ge_0_01": full["h_score"]["mean"]
        - aggregate["rules"]["messages_off"]["h_score"]["mean"] >= 0.01,
        "rescue_harm_ratio_ge_2": aggregate["collaboration"]["rescue_harm_ratio"]["mean"] >= 2.0,
        "paired_h_or_oscr_ci_lower_gt_0": (
            ci["h_score"]["ci95"][0] > 0 or ci["oscr"]["ci95"][0] > 0),
    }
    return aggregate


def render_formal(aggregate: dict) -> str:
    lines = ["# Stage-6 三种子正式聚合", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for rule, row in aggregate["rules"].items():
        fmt = lambda key: f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
        lines.append(f"| {rule} | {fmt('auroc')} | {fmt('oscr')} | "
                     f"{fmt('known_accuracy')} | {fmt('unknown_recall')} | {fmt('h_score')} |")
    lines += ["", "## 多智能体成功闸门", ""]
    for key, value in aggregate["success_gates"].items():
        lines.append(f"- {'通过' if value else '未通过'}：`{key}`")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-lco", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _resolve_config(config_path)
    assert_paired_method_config(cfg)
    results, selection, elapsed = run_stage6(
        cfg, run_lco=not args.skip_lco, seeds=args.seeds)
    out_root = Path(cfg["output_dir"])
    lco = json.loads((out_root / "lco" / "stage6_lco_selection.json").read_text(
        encoding="utf-8"))
    lco_report = render_lco_report(lco)
    (out_root / "stage6_screen_report.md").write_text(lco_report, encoding="utf-8")
    print(lco_report)
    if results:
        aggregate = aggregate_formal(cfg, results, args.seeds, elapsed)
        (out_root / "aggregate_stage6.json").write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        report = render_formal(aggregate)
        (out_root / "aggregate_stage6.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
