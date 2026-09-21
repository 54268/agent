"""Build the Stage-8 A/B redesign comparison tables from persisted probes."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = {
    "legacy_v1": {
        "wisig": ROOT / "results/stage8_g1_redesign/wisig_k40u20/g1_probe_report.json",
        "oracle": ROOT / "results/stage8_g1_redesign/oracle_k10u6/g1_probe_report.json",
    },
    "isolated_v2": {
        "wisig": ROOT / "results/stage8_g1_redesign_v2/wisig_k40u20/g1_probe_report.json",
        "oracle": ROOT / "results/stage8_g1_redesign_v2/oracle_k10u6/g1_probe_report.json",
    },
}
LEARNABILITY = {
    "legacy_v1": ROOT / "results/stage8_g1_redesign/oracle_k10u6/"
    "local_learnability_report.json",
    "isolated_v2": ROOT / "results/stage8_g1_redesign_v2/oracle_k10u6/"
    "local_learnability_report.json",
}
OUTPUT = ROOT / "results/stage8_g1_redesign_v2"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required Stage-8 result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value, digits=6):
    return round(float(value), digits)


def _fold_row(profile: str, dataset: str, row: dict) -> dict:
    rescue = row["rescue_decomposition"]
    temporal_pair = row["temporal_pair_metrics"]
    spectral_pair = row["spectral_pair_metrics"]
    return {
        "profile": profile,
        "dataset": dataset,
        "fold": int(row["inner_fold"]),
        "temporal_known_accuracy": _round(row["temporal_identity_accuracy"]),
        "spectral_known_accuracy": _round(row["spectral_identity_accuracy"]),
        "temporal_pair_auroc": _round(temporal_pair["pair_auroc"]),
        "spectral_pair_auroc": _round(spectral_pair["pair_auroc"]),
        "temporal_pair_accuracy": _round(temporal_pair["pair_accuracy"]),
        "spectral_pair_accuracy": _round(spectral_pair["pair_accuracy"]),
        "temporal_hard_pair_accuracy": _round(
            temporal_pair["hard_negative_pair_accuracy"]),
        "spectral_hard_pair_accuracy": _round(
            spectral_pair["hard_negative_pair_accuracy"]),
        "temporal_pair_brier": _round(temporal_pair["pair_brier"]),
        "spectral_pair_brier": _round(spectral_pair["pair_brier"]),
        "temporal_to_spectral_identity_rescue": _round(
            rescue["temporal_to_spectral_identity_rescue"]),
        "spectral_to_temporal_identity_rescue": _round(
            rescue["spectral_to_temporal_identity_rescue"]),
        "temporal_to_spectral_pair_rescue": _round(
            rescue["temporal_to_spectral_hard_pair_rescue"]),
        "spectral_to_temporal_pair_rescue": _round(
            rescue["spectral_to_temporal_hard_pair_rescue"]),
        "temporal_unknown_rejection_rescue": _round(
            rescue["temporal_unknown_rejection_rescue"]),
        "spectral_unknown_rejection_rescue": _round(
            rescue["spectral_unknown_rejection_rescue"]),
    }


def _direction_row(profile: str, dataset: str, row: dict) -> dict:
    temporal = row["temporal_rescues_spectral"]
    spectral = row["spectral_rescues_temporal"]
    return {
        "profile": profile,
        "dataset": dataset,
        "fold": int(row["inner_fold"]),
        "temporal_to_spectral_auroc": _round(temporal["auroc"]),
        "temporal_to_spectral_pr_auc": _round(temporal["pr_auc"]),
        "temporal_to_spectral_support": int(temporal["support"]),
        "spectral_to_temporal_auroc": _round(spectral["auroc"]),
        "spectral_to_temporal_pr_auc": _round(spectral["pr_auc"]),
        "spectral_to_temporal_support": int(spectral["support"]),
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    reports = {
        profile: {dataset: _load(path) for dataset, path in datasets.items()}
        for profile, datasets in REPORTS.items()
    }
    learnability = {
        profile: _load(path) for profile, path in LEARNABILITY.items()}
    fold_rows = []
    direction_rows = []
    gate_rows = []
    failure_rows = []
    reindex_rows = []
    leakage = {}
    leakage_rows = []
    for profile, datasets in reports.items():
        leakage[profile] = {}
        for dataset, report in datasets.items():
            if report.get("formal_unknown_used"):
                raise RuntimeError("formal Unknown entered a Stage-8 redesign probe")
            fold_rows.extend(
                _fold_row(profile, dataset, row) for row in report["folds"])
            direction_rows.extend(
                _direction_row(profile, dataset, row)
                for row in report["directional_rescue_predictability"])
            gate_rows.append({
                "profile": profile,
                "dataset": dataset,
                "g1": "PASS" if report["g1_gate"]["passed"] else "FAIL",
                "g15": "PASS" if report["g15_gate"]["passed"] else "FAIL",
                "outcome": "PASS" if report.get(
                    "memory_auditor_unlocked", False) else "STOP-AND-REDESIGN",
            })
            failure_rows.append({
                "profile": profile,
                "dataset": dataset,
                "folds": report.get("failure_diagnosis", []),
                "g1_failed_checks": [
                    key for key, value in report["g1_gate"]["checks"].items()
                    if not value],
                "g15_failed_checks": [
                    key for key, value in report["g15_gate"]["checks"].items()
                    if not value],
            })
            reindex_rows.append({
                "profile": profile,
                "dataset": dataset,
                "passed": bool(report["class_reindex_invariance_passed"]),
                "maximum_absolute_error": _round(max(
                    row["class_reindex_invariance"]["maximum_absolute_error"]
                    for row in report["folds"])),
            })
            leakage[profile][dataset] = report.get(
                "observation_leakage_audit")

    for dataset, report in reports["isolated_v2"].items():
        audit = report["observation_leakage_audit"]
        for profile, values in audit["profiles"].items():
            temporal = values["temporal_to_spectral_descriptor"]
            spectral = values["spectral_to_temporal_order_descriptor"]
            leakage_rows.append({
                "dataset": dataset, "profile": profile,
                "temporal_to_spectral_r2": _round(temporal["r2"]),
                "temporal_to_spectral_normalized_rmse": _round(
                    temporal["normalized_rmse"]),
                "spectral_to_temporal_r2": _round(spectral["r2"]),
                "spectral_to_temporal_normalized_rmse": _round(
                    spectral["normalized_rmse"]),
                "temporal_to_spectral_r2_change_vs_v1": (
                    None if profile == "legacy_v1" else _round(
                        temporal["r2_change_vs_legacy_v1"])),
                "spectral_to_temporal_r2_change_vs_v1": (
                    None if profile == "legacy_v1" else _round(
                        spectral["r2_change_vs_legacy_v1"])),
            })

    learnability_rows = []
    for profile, report in learnability.items():
        selected = next(
            row for row in report["candidates"]
            if row["name"] == report["selected_candidate"])
        learnability_rows.extend({
            "profile": profile,
            "samples_per_class": int(row["samples_per_class"]),
            "temporal_train_accuracy": _round(row["temporal_train_accuracy"]),
            "spectral_train_accuracy": _round(row["spectral_train_accuracy"]),
            "temporal_pair_auroc": _round(
                row["temporal_pair_metrics"]["pair_auroc"]),
            "spectral_pair_auroc": _round(
                row["spectral_pair_metrics"]["pair_auroc"]),
            "passed": bool(row["passed"]),
        } for row in selected["runs"])

    summary = {
        "stage": "stage8_next_round_g1_redesign",
        "formal_unknown_used": False,
        "memory_auditor_router_forced_communication_unlocked": all(
            row["outcome"] == "PASS" for row in gate_rows),
        "oracle_small_set_learnability": {
            "selected_candidates": {
                profile: report["selected_candidate"]
                for profile, report in learnability.items()},
            "passed": all(report["passed"] for report in learnability.values()),
            "rows": learnability_rows,
        },
        "identity_pair_rescue_by_fold": fold_rows,
        "directional_predictability_by_fold": direction_rows,
        "class_reindex_invariance": reindex_rows,
        "observation_leakage": leakage,
        "observation_leakage_comparison": leakage_rows,
        "gates": gate_rows,
        "failure_diagnosis": failure_rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "stage8_g1_redesign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Stage-8 下一轮 G1 重构结果", "",
        "> 本报告只使用 Known 与 inner proxy-Unknown；formal Unknown 未进入训练、选模、阈值或评价。",
        "", "## ORACLE small-set overfit", "",
    ]
    lines.extend(_md_table(
        ["Profile", "每类样本", "T train acc", "S train acc", "T pair AUC", "S pair AUC", "结果"],
        [[row["profile"], str(row["samples_per_class"]), f'{row["temporal_train_accuracy"]:.4f}',
          f'{row["spectral_train_accuracy"]:.4f}',
          f'{row["temporal_pair_auroc"]:.4f}',
          f'{row["spectral_pair_auroc"]:.4f}',
         "PASS" if row["passed"] else "FAIL"] for row in learnability_rows]))
    lines.extend(["", "## Identity / pair / rescue", ""])
    lines.extend(_md_table(
        ["Profile", "Dataset", "Fold", "T Acc", "S Acc", "T pair AUC",
         "S pair AUC", "T→S id", "S→T id", "T→S pair", "S→T pair"],
        [[row["profile"], row["dataset"], str(row["fold"]),
          f'{row["temporal_known_accuracy"]:.4f}',
          f'{row["spectral_known_accuracy"]:.4f}',
          f'{row["temporal_pair_auroc"]:.4f}',
          f'{row["spectral_pair_auroc"]:.4f}',
          f'{row["temporal_to_spectral_identity_rescue"]:.4f}',
          f'{row["spectral_to_temporal_identity_rescue"]:.4f}',
          f'{row["temporal_to_spectral_pair_rescue"]:.4f}',
          f'{row["spectral_to_temporal_pair_rescue"]:.4f}']
         for row in fold_rows]))
    lines.extend(["", "## Pair verification diagnostics", ""])
    lines.extend(_md_table(
        ["Profile", "Dataset", "Fold", "T pair acc", "S pair acc",
         "T hard-pair", "S hard-pair", "T Brier", "S Brier"],
        [[row["profile"], row["dataset"], str(row["fold"]),
          f'{row["temporal_pair_accuracy"]:.4f}',
          f'{row["spectral_pair_accuracy"]:.4f}',
          f'{row["temporal_hard_pair_accuracy"]:.4f}',
          f'{row["spectral_hard_pair_accuracy"]:.4f}',
          f'{row["temporal_pair_brier"]:.4f}',
          f'{row["spectral_pair_brier"]:.4f}']
         for row in fold_rows]))
    lines.extend(["", "## Directional rescue predictability", ""])
    lines.extend(_md_table(
        ["Profile", "Dataset", "Fold", "T→S AUROC", "T→S PR-AUC",
         "T→S n", "S→T AUROC", "S→T PR-AUC", "S→T n"],
        [[row["profile"], row["dataset"], str(row["fold"]),
          f'{row["temporal_to_spectral_auroc"]:.4f}',
          f'{row["temporal_to_spectral_pr_auc"]:.4f}',
          str(row["temporal_to_spectral_support"]),
          f'{row["spectral_to_temporal_auroc"]:.4f}',
          f'{row["spectral_to_temporal_pr_auc"]:.4f}',
          str(row["spectral_to_temporal_support"])]
         for row in direction_rows]))
    lines.extend(["", "## Unknown rejection rescue（辅助指标）", ""])
    lines.extend(_md_table(
        ["Profile", "Dataset", "Fold", "T→S unknown", "S→T unknown"],
        [[row["profile"], row["dataset"], str(row["fold"]),
          f'{row["temporal_unknown_rejection_rescue"]:.4f}',
          f'{row["spectral_unknown_rejection_rescue"]:.4f}']
         for row in fold_rows]))
    lines.extend(["", "## Gate 决策", ""])
    lines.extend(_md_table(
        ["Profile", "Dataset", "G1", "G1.5", "最终"],
        [[row["profile"], row["dataset"], row["g1"], row["g15"],
          row["outcome"]] for row in gate_rows]))
    lines.extend(["", "## Observation leakage probe", ""])
    lines.extend(_md_table(
        ["Dataset", "Profile", "T→spectral R²", "ΔR²", "S→temporal R²", "ΔR²"],
        [[row["dataset"], row["profile"],
          f'{row["temporal_to_spectral_r2"]:.4f}',
          "—" if row["temporal_to_spectral_r2_change_vs_v1"] is None
          else f'{row["temporal_to_spectral_r2_change_vs_v1"]:+.4f}',
          f'{row["spectral_to_temporal_r2"]:.4f}',
          "—" if row["spectral_to_temporal_r2_change_vs_v1"] is None
          else f'{row["spectral_to_temporal_r2_change_vs_v1"]:+.4f}']
         for row in leakage_rows]))
    lines.extend([
        "", "Leakage probe 的 held-out R² 越低，表示跨 observation 的可恢复信息越少；该审计不设人为通过阈值。",
        "本轮 v2 没有在两个数据集上同时降低 R²，因此不能宣称量化信息隔离已经成立。",
        "Class reindex 的逐折误差见同目录 JSON。", "", "## 失败归因", "",
    ])
    for row in failure_rows:
        lines.append(
            f'- `{row["profile"]}/{row["dataset"]}`：G1 失败项 '
            f'{row["g1_failed_checks"] or "无"}；G1.5 失败项 '
            f'{row["g15_failed_checks"] or "无"}。')
    lines.extend([
        "", "只要任一数据集未同时通过 G1 与 G1.5，Memory、Auditor、Router 与 forced communication 继续锁定。",
    ])
    (OUTPUT / "stage8_g1_redesign_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
