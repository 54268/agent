"""Run Stage-5 LCO selection and formal WiSig seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.experiment import run_stage5  # noqa: E402
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision  # noqa: E402


RULES = ("identity", "prototype", "openmax", "no_communication", "communication",
         "messages_off", "messages_shuffled", "drop_identity_sender",
         "drop_geometry_sender", "uniform_weights")
METRICS = ("auroc", "aupr_out", "fpr95", "oscr", "known_accuracy",
           "unknown_recall", "macro_f1", "h_score")
CI_METRICS = ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")


def _metrics_from_scores(y, pred, unknown_score, threshold):
    is_unknown = (y == -1).astype(np.int32)
    row = detection_metrics(is_unknown, unknown_score)
    row["oscr"] = oscr(is_unknown, pred, y, 1.0 - unknown_score)
    row.update(threshold_decision(is_unknown, y, pred, unknown_score, threshold))
    return row


def grouped_paired_bootstrap(out_root: Path, seeds, repetitions: int, seed: int = 2026):
    """Paired, class-stratified transmitter bootstrap over the three runs.

    The exact same sampled transmitter multiset is applied to communication
    and no-communication predictions, preserving the paired comparison.  A
    replicate's statistic is the mean across training seeds.
    """
    payloads = []
    selection = json.loads((out_root / "lco" / "lco_selection.json").read_text(
        encoding="utf-8"))["selected"]
    fallback_threshold = float(selection.get("known_acceptance", 0.95))
    for run_seed in seeds:
        path = out_root / "multiseed" / f"seed{run_seed}" / "scores.npz"
        with np.load(path) as data:
            payloads.append({key: data[key].copy() for key in data.files})
    y = payloads[0]["y"]
    groups = payloads[0]["group_id"]
    for payload in payloads[1:]:
        if not np.array_equal(payload["y"], y) or not np.array_equal(payload["group_id"], groups):
            raise ValueError("Stage-5 seeds do not share the same ordered evaluation manifest")
    known_groups = np.unique(groups[y != -1])
    unknown_groups = np.unique(groups[y == -1])
    if set(known_groups.tolist()) & set(unknown_groups.tolist()):
        raise ValueError("known and unknown transmitter group IDs overlap")
    group_indices = {int(group): np.flatnonzero(groups == group)
                     for group in np.r_[known_groups, unknown_groups]}
    rng = np.random.default_rng(seed)
    method_names = ("communication", "no_communication", "messages_off")
    samples = {name: {metric: [] for metric in CI_METRICS} for name in method_names}
    samples["paired_delta"] = {metric: [] for metric in CI_METRICS}
    samples["paired_delta_vs_messages_off"] = {metric: [] for metric in CI_METRICS}
    for _ in range(int(repetitions)):
        chosen = np.r_[rng.choice(known_groups, len(known_groups), replace=True),
                       rng.choice(unknown_groups, len(unknown_groups), replace=True)]
        indices = np.concatenate([group_indices[int(group)] for group in chosen])
        per_method = {name: {metric: [] for metric in CI_METRICS} for name in method_names}
        for payload in payloads:
            for name in per_method:
                threshold_key = f"tau_{name}"
                threshold = (payload[threshold_key][indices]
                             if threshold_key in payload else fallback_threshold)
                row = _metrics_from_scores(payload["y"][indices], payload[f"pred_{name}"][indices],
                                           payload[f"u_{name}"][indices], threshold)
                for metric in CI_METRICS:
                    per_method[name][metric].append(row[metric])
        for metric in CI_METRICS:
            comm = float(np.mean(per_method["communication"][metric]))
            no_comm = float(np.mean(per_method["no_communication"][metric]))
            samples["communication"][metric].append(comm)
            samples["no_communication"][metric].append(no_comm)
            messages_off = float(np.mean(per_method["messages_off"][metric]))
            samples["messages_off"][metric].append(messages_off)
            samples["paired_delta"][metric].append(comm - no_comm)
            samples["paired_delta_vs_messages_off"][metric].append(comm - messages_off)
    result = {"unit": "transmitter", "stratified_known_unknown": True,
              "paired_methods": True, "repetitions": int(repetitions),
              "known_transmitters": int(len(known_groups)),
              "unknown_transmitters": int(len(unknown_groups))}
    for section in (*method_names, "paired_delta", "paired_delta_vs_messages_off"):
        result[section] = {}
        for metric in CI_METRICS:
            values = np.asarray(samples[section][metric])
            result[section][metric] = {
                "mean": float(values.mean()),
                "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-lco", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="rebuild aggregate reports from completed seed artifacts")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "output_dir", "reuse_lco_experts_from",
                "reuse_final_experts_from"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    if args.aggregate_only:
        out_root = Path(cfg["output_dir"])
        selection = json.loads((out_root / "lco" / "lco_selection.json").read_text(
            encoding="utf-8"))["selected"]
        results = [json.loads((out_root / "multiseed" / f"seed{seed}" /
                               "stage5_metrics.json").read_text(encoding="utf-8"))
                   for seed in args.seeds]
        elapsed = 0.0
    else:
        results, selection, elapsed = run_stage5(
            cfg, run_lco=not args.skip_lco, seeds=args.seeds)
    aggregate = {"seeds": args.seeds, "selection": selection, "elapsed_seconds": elapsed,
                 "rules": {}, "causal_interventions": {},
                 "protocol": {"real_unknown_used_for_training": False,
                              "real_unknown_used_for_selection": False,
                              "real_unknown_used_for_threshold": False}}
    for rule in RULES:
        aggregate["rules"][rule] = {}
        for metric in METRICS:
            values = [row["metrics"][rule][metric] for row in results]
            aggregate["rules"][rule][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(x) for x in values],
            }
    for rule in ("messages_off", "messages_shuffled", "drop_identity_sender",
                 "drop_geometry_sender", "uniform_weights"):
        aggregate["causal_interventions"][rule] = {}
        for metric in ("class_flip_rate", "unknown_score_mean_abs_change", "auroc_delta", "oscr_delta"):
            values = [row["causal_interventions"][rule][metric] for row in results]
            aggregate["causal_interventions"][rule][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": [float(x) for x in values],
            }
    gate_values = np.asarray([row["mean_edge_gates"] for row in results])
    aggregate["mean_edge_gates"] = {"names": list(results[0]["edge_names"]),
                                    "mean": gate_values.mean(0).tolist(),
                                    "std": gate_values.std(0).tolist()}
    out_root = Path(cfg["output_dir"]); out_root.mkdir(parents=True, exist_ok=True)
    aggregate["grouped_paired_bootstrap"] = grouped_paired_bootstrap(
        out_root, args.seeds, int(cfg.get("bootstrap_repetitions", 1000)))
    full = aggregate["rules"]["communication"]
    no = aggregate["rules"]["no_communication"]
    aggregate["success_gates"] = {
        "unknown_recall_ge_0_90": full["unknown_recall"]["mean"] >= 0.90,
        "known_accuracy_ge_0_85": full["known_accuracy"]["mean"] >= 0.85,
        "h_score_ge_0_875": full["h_score"]["mean"] >= 0.875,
        "auroc_ge_0_95": full["auroc"]["mean"] >= 0.95,
        "oscr_ge_0_90": full["oscr"]["mean"] >= 0.90,
        "communication_delta_h_ge_0_01": full["h_score"]["mean"] - no["h_score"]["mean"] >= 0.01,
        "communication_delta_oscr_ge_0_005": full["oscr"]["mean"] - no["oscr"]["mean"] >= 0.005,
        "positive_h_seeds_ge_2": sum(a > b for a, b in zip(
            full["h_score"]["values"], no["h_score"]["values"])) >= 2,
        "messages_improve_h_vs_off": full["h_score"]["mean"]
        > aggregate["rules"]["messages_off"]["h_score"]["mean"],
        "messages_improve_oscr_vs_off": full["oscr"]["mean"]
        > aggregate["rules"]["messages_off"]["oscr"]["mean"],
        "messages_positive_h_seeds_ge_2": sum(a > b for a, b in zip(
            full["h_score"]["values"],
            aggregate["rules"]["messages_off"]["h_score"]["values"])) >= 2,
        "messages_positive_oscr_seeds_ge_2": sum(a > b for a, b in zip(
            full["oscr"]["values"],
            aggregate["rules"]["messages_off"]["oscr"]["values"])) >= 2,
    }
    (out_root / "aggregate_stage5.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    report = render_report(aggregate)
    (out_root / "aggregate_stage5.md").write_text(report, encoding="utf-8")
    print(report)


def render_report(aggregate):
    labels = {"identity": "Identity", "prototype": "Prototype", "openmax": "OpenMax",
              "no_communication": "等容量无通信", "communication": "完整协作",
              "messages_off": "推理关闭消息", "messages_shuffled": "打乱消息",
              "drop_identity_sender": "删除 Identity 发送边",
              "drop_geometry_sender": "删除 Geometry 发送边", "uniform_weights": "均匀类别融合"}
    lines = ["# Stage-5 多种子聚合", "", f"LCO 冻结选择：`{aggregate['selection']}`。", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for rule in RULES:
        row = aggregate["rules"][rule]
        fmt = lambda key: f"{row[key]['mean']:.3f}±{row[key]['std']:.3f}"
        lines.append(f"| {labels[rule]} | {fmt('auroc')} | {fmt('oscr')} | "
                     f"{fmt('known_accuracy')} | {fmt('unknown_recall')} | {fmt('h_score')} |")
    ci = aggregate["grouped_paired_bootstrap"]
    lines += ["", "## 按 Tx 分组的 paired bootstrap 95% CI", "",
              f"分层重采样 {ci['known_transmitters']} 个 Known Tx 与 "
              f"{ci['unknown_transmitters']} 个 Unknown Tx，共 {ci['repetitions']} 次。", "",
              "| 指标 | 完整协作 95% CI | 协作-无通信 95% CI | 协作-关闭消息 95% CI |",
              "|---|---:|---:|---:|"]
    for metric in CI_METRICS:
        full_ci = ci["communication"][metric]["ci95"]
        delta_ci = ci["paired_delta"][metric]["ci95"]
        off_ci = ci["paired_delta_vs_messages_off"][metric]["ci95"]
        lines.append(f"| {metric} | [{full_ci[0]:.3f}, {full_ci[1]:.3f}] | "
                     f"[{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}] | "
                     f"[{off_ci[0]:+.3f}, {off_ci[1]:+.3f}] |")
    lines += ["", "## 预注册成功闸门", ""]
    for key, value in aggregate["success_gates"].items():
        lines.append(f"- {'通过' if value else '未通过'}：`{key}`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
