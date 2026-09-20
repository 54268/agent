"""Build the Stage-5 candidate-Agent and module ablation ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(row):
    return {key: float(row["metrics"][key]["mean"])
            for key in ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")}


def build(result_dir: Path):
    lco = json.loads((result_dir / "lco" / "lco_selection.json").read_text(encoding="utf-8"))
    formal = json.loads((result_dir / "aggregate_stage5.json").read_text(encoding="utf-8"))
    meta = {row["communication"]: row for row in lco["communication_rows"]}

    explorer = []
    for row in sorted(lco["generator_summary"],
                      key=lambda item: item["metrics"]["h_score"]["mean"], reverse=True):
        explorer.append({"candidate": row["pug"], **compact(row)})

    calibration = []
    for row in lco["communication_summary"]:
        info = meta[row["communication"]]
        if info["mode"] == "none":
            calibration.append({"candidate": info["score_rule"], **compact(row)})
    calibration.sort(key=lambda item: item["h_score"], reverse=True)

    selected_rule = lco["selected"]["score_rule"]
    communication = []
    for row in lco["communication_summary"]:
        info = meta[row["communication"]]
        if info["score_rule"] == selected_rule:
            communication.append({"candidate": f"{info['mode']}_budget{info['budget']:g}",
                                  "gate_threshold": float(info.get("gate_threshold", 0.5)),
                                  "mean_edges": float(row["mean_edges"]["mean"]),
                                  **compact(row)})
    communication.sort(key=lambda item: item["h_score"], reverse=True)

    joint = []
    for row in sorted(lco["communication_summary"],
                      key=lambda item: item["metrics"]["h_score"]["mean"], reverse=True)[:15]:
        info = meta[row["communication"]]
        joint.append({"candidate": row["communication"], "mode": info["mode"],
                      "budget": float(info["budget"]), "score_rule": info["score_rule"],
                      "mean_edges": float(row["mean_edges"]["mean"]), **compact(row)})

    perception = []
    for name in ("identity", "prototype", "openmax"):
        values = formal["rules"][name]
        perception.append({"candidate": name, **{metric: float(values[metric]["mean"])
                          for metric in ("auroc", "oscr", "known_accuracy",
                                         "unknown_recall", "h_score")}})

    interventions = []
    for name in ("communication", "messages_off", "messages_shuffled",
                 "drop_identity_sender", "drop_geometry_sender", "no_communication"):
        values = formal["rules"][name]
        interventions.append({"candidate": name, **{metric: float(values[metric]["mean"])
                              for metric in ("auroc", "oscr", "known_accuracy",
                                             "unknown_recall", "h_score")}})

    return {
        "selection_source": "five-fold leave-class-out; no real test unknown used",
        "selected": lco["selected"], "perception_agents_formal": perception,
        "boundary_explorer_lco": explorer, "calibration_agent_lco_no_communication": calibration,
        "communication_lco_at_selected_calibration": communication,
        "top_joint_lco_candidates": joint, "formal_message_interventions": interventions,
    }


def table(lines, title, rows, include_edges=False):
    lines += [f"## {title}", ""]
    header = "| 候选 | AUROC | OSCR | Known Acc | Unknown Recall | H-score"
    divider = "|---|---:|---:|---:|---:|---:"
    if include_edges:
        header += " | 平均边数"
        divider += "|---:"
    lines += [header + " |", divider + " |"]
    for row in rows:
        values = (f"| {row['candidate']} | {row['auroc']:.3f} | {row['oscr']:.3f} | "
                  f"{row['known_accuracy']:.3f} | {row['unknown_recall']:.3f} | {row['h_score']:.3f}")
        if include_edges:
            values += f" | {row['mean_edges']:.2f}"
        lines.append(values + " |")
    lines.append("")


def render(report):
    lines = ["# Stage-5 候选 Agent / 模块组合消融", "",
             "所有候选均在同一五折 Leave-Class-Out 协议上比较；当前 20 个真实 Unknown 不参与排名。", ""]
    table(lines, "感知与几何证据（正式三种子）", report["perception_agents_formal"])
    table(lines, "Boundary Explorer Agent（PUG-V2）", report["boundary_explorer_lco"])
    table(lines, "Calibration Agent（固定无通信）", report["calibration_agent_lco_no_communication"])
    table(lines, "Communication（固定入选校准规则）",
          report["communication_lco_at_selected_calibration"], include_edges=True)
    table(lines, "正式消息反事实", report["formal_message_interventions"])
    lines += ["## 当前筛选结论", "",
              f"- LCO 冻结组合：`{report['selected']}`。",
              "- Prototype/Geometry 是最稳定的基础开放集证据；OpenMax 的工作点召回较高，但排序指标较弱。",
              "- Geometry + OpenMax + Boundary 的校准协作优于仅 Boundary，校准必须作为独立角色保留。",
              "- 正式推理关闭消息优于完整通信，现有消息边尚未通过因果净增益检验；下一轮只优化消息，不回退已经有效的角色与校准。",
              "- Boundary Explorer 的不同生成策略差距小，下一轮应扩大动作空间（边界壳约束、竞争类方向和尺度），继续由 LCO 反馈选择。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results/stage5/wisig_k40u20")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    report = build(result_dir)
    (result_dir / "candidate_agent_ablation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text = render(report)
    (result_dir / "candidate_agent_ablation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
