"""Run the complete Stage-15 LLM multi-agent open-set SEI experiment."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.openmax import OpenMaxEVT  # noqa: E402
from maros_stage5.training import extract_records, refresh_empirical_prototypes  # noqa: E402
from maros_stage7.splits import build_nested_lco_protocol  # noqa: E402
from maros_stage10.experiment import _load_dataset  # noqa: E402
from maros_stage10.mother import cap_dataset, load_mother  # noqa: E402
from maros_stage13.experiment import load_selection_tail, _base_system, _base_outputs  # noqa: E402
from maros_stage13.feature_pug import CSupport  # noqa: E402
from maros_stage14.data import EvaluationEpisodeBank, snapshots_from_records  # noqa: E402
from maros_stage15.agents import build_default_agents  # noqa: E402
from maros_stage15.orchestrator import AgenticOpenSetSystem  # noqa: E402
from maros_stage15.providers import OpenAICompatibleChatModel  # noqa: E402


DEFAULT_CONFIG = "configs/experiments/stage15_llm_multiagent_oracle.json"


def resolve_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _trajectory_signature(transcript: tuple[dict, ...]) -> tuple[str, ...]:
    return tuple(
        f"{row['agent']}:{row['action']}:{row.get('tool') or '-'}:"
        f"{row.get('recipient') or '-'}"
        for row in transcript
    )


def _write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# Stage 15 大模型多智能体实验汇总",
        "",
        "## 核心识别结果",
        "",
        f"- Known Accuracy: **{summary['known_accuracy']:.4f}**",
        f"- Unknown Recall: **{summary['unknown_recall']:.4f}**",
        f"- H-score: **{summary['h_score']:.4f}**",
        f"- 样本数: **{summary['samples']}**",
        "",
        "## 多智能体协作统计",
        "",
        f"- 平均协作轮数: **{summary['mean_turns']:.3f}**",
        f"- 平均工具调用数: **{summary['mean_tool_calls']:.3f}**",
        f"- 平均消息数: **{summary['mean_messages']:.3f}**",
        f"- Unique trajectories: **{summary['unique_trajectories']}**",
        f"- Known final 数: **{summary['known_final_count']}**",
        f"- Unknown final 数: **{summary['unknown_final_count']}**",
        "",
        "## Agent 动作统计",
        "",
    ]
    for agent, counts in summary["agent_action_counts"].items():
        lines.append(f"### {agent}")
        for action, count in counts.items():
            lines.append(f"- {action}: {count}")
        lines.append("")
    lines.extend(["## RF 工具调用统计", ""])
    if summary["tool_usage"]:
        for tool, count in summary["tool_usage"].items():
            lines.append(f"- {tool}: {count}")
    else:
        lines.append("- 无")
    lines.extend([
        "",
        "## 说明",
        "",
        "- Proposer 只接收 identity-side 局部观测和私有工具。",
        "- Critic 只接收 geometry/open-set-side 局部观测和私有工具。",
        "- Arbiter 不直接读取底层 RF 分数，只通过 Agent 消息完成裁决。",
        "- formal Unknown 不参与本次评估/适配；本次 unknown 为 LCO proxy unknown。",
        "",
        "详细逐样本过程见 trajectories.jsonl。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Single-entry Stage-15 LLM multi-agent OS-SEI experiment")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--samples-per-class", type=int, default=None,
        help="Override evaluation_samples_per_class for a faster smoke run.")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    if args.samples_per_class is not None:
        cfg["evaluation_samples_per_class"] = int(args.samples_per_class)

    base = resolve_config(cfg["base_stage14_config"])
    stage13 = resolve_config(base["stage13_config"])
    mother_cfg = resolve_config(stage13["mother_config"])

    seed = int(cfg["seed"])
    np.random.seed(seed)

    print("[Stage15] 1/5 Loading nested LCO protocol...")
    splits = _load_dataset(stage13)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=5, inner_folds=4,
        seed=int(stage13["partition_seed"]))
    fold = protocol.outer_folds[int(cfg["outer_fold"])]
    n = int(cfg["evaluation_samples_per_class"])
    test_known = cap_dataset(fold.test_known, n, seed + 1)
    test_unknown = cap_dataset(
        fold.test_proxy_unknown, n, seed + 2, proxy_unknown=True)

    print("[Stage15] 2/5 Loading frozen RF perception and evidence models...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / stage13["fold_checkpoint_pattern"].format(fold=fold.index)
    model, checkpoint_payload = load_mother(
        checkpoint, mother_cfg, fold.known_classes,
        fold.proxy_unknown_classes, device)
    refresh_empirical_prototypes(
        model, fold.train_known, device, int(stage13["batch_size"]))
    evt = OpenMaxEVT.from_state_dict(checkpoint_payload["openmax"])

    train_records = extract_records(
        model, fold.train_known, device, evt,
        batch_size=int(stage13["batch_size"]))
    cal_records = extract_records(
        model, fold.calibration_known, device, evt,
        batch_size=int(stage13["batch_size"]))
    known_records = extract_records(
        model, test_known, device, evt,
        batch_size=int(stage13["batch_size"]))
    unknown_records = extract_records(
        model, test_unknown, device, evt,
        batch_size=int(stage13["batch_size"]))

    support = CSupport().fit(cal_records)
    selection = load_selection_tail(ROOT / stage13["stage5_selection"])
    coordinator, scaler, arbitrator, _, _, _ = _base_system(
        model, evt, train_records, cal_records, selection,
        mother_cfg, device, seed + fold.index * 100)
    known_boundary = _base_outputs(
        known_records, coordinator, scaler, arbitrator, device)["risk"]
    unknown_boundary = _base_outputs(
        unknown_records, coordinator, scaler, arbitrator, device)["risk"]

    rows = snapshots_from_records(
        known_records, source_kind="known", prefix="stage15-known",
        support=support, boundary_risk=known_boundary)
    rows.extend(snapshots_from_records(
        unknown_records, source_kind="lco_proxy", prefix="stage15-unknown",
        support=support, boundary_risk=unknown_boundary))
    bank = EvaluationEpisodeBank(rows)

    print("[Stage15] 3/5 Connecting LLM agents...")
    llm_cfg = cfg["llm"]
    chat = OpenAICompatibleChatModel(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        temperature=float(llm_cfg.get("temperature", 0.1)),
        timeout_seconds=float(llm_cfg.get("timeout_seconds", 120)),
    )
    agents = build_default_agents(
        chat, memory_size=int(cfg["agentic"].get("memory_size", 256)))
    system = AgenticOpenSetSystem(
        agents,
        max_turns=int(cfg["agentic"]["max_turns"]),
        tool_budget=int(cfg["agentic"]["tool_budget"]),
        message_budget=int(cfg["agentic"].get("message_budget", 8)),
    )

    print(f"[Stage15] 4/5 Running {len(bank)} multi-agent episodes...")
    records = []
    known_correct = known_count = 0
    unknown_rejected = unknown_count = 0
    turns_total = tool_total = message_total = 0
    tool_usage = collections.Counter()
    agent_action_counts = {
        "proposer": collections.Counter(),
        "critic": collections.Counter(),
        "arbiter": collections.Counter(),
    }
    signatures = set()

    for index, snapshot in enumerate(bank, start=1):
        result = system.run(snapshot)
        if snapshot.is_unknown:
            unknown_count += 1
            unknown_rejected += int(result.prediction == -1)
        else:
            known_count += 1
            known_correct += int(result.prediction == snapshot.label)

        turns_total += result.turns
        tool_total += result.tool_calls
        message_total += result.message_count
        signatures.add(_trajectory_signature(result.transcript))
        for row in result.transcript:
            agent_action_counts[row["agent"]][row["action"]] += 1
            if row.get("tool"):
                tool_usage[row["tool"]] += 1

        records.append({
            "sample_id": snapshot.sample_id,
            "target": snapshot.label,
            "prediction": result.prediction,
            "decision": result.decision,
            "candidate_class": result.candidate_class,
            "confidence": result.confidence,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "message_count": result.message_count,
            "transcript": result.transcript,
        })
        if index == 1 or index % 50 == 0 or index == len(bank):
            print(f"[Stage15] episode {index}/{len(bank)}")

    known_acc = known_correct / max(known_count, 1)
    unknown_recall = unknown_rejected / max(unknown_count, 1)
    h_score = (
        2 * known_acc * unknown_recall / (known_acc + unknown_recall)
        if known_acc + unknown_recall > 0 else 0.0
    )
    count = max(len(records), 1)
    summary = {
        "stage": "stage15_llm_multiagent",
        "architecture": "partially_observable_heterogeneous_llm_mas",
        "dataset": cfg["dataset"],
        "outer_fold": fold.index,
        "model": llm_cfg["model"],
        "known_accuracy": known_acc,
        "unknown_recall": unknown_recall,
        "h_score": h_score,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "samples": len(records),
        "mean_turns": turns_total / count,
        "mean_tool_calls": tool_total / count,
        "mean_messages": message_total / count,
        "unique_trajectories": len(signatures),
        "known_final_count": sum(r["decision"] == "known" for r in records),
        "unknown_final_count": sum(r["decision"] == "unknown" for r in records),
        "tool_usage": dict(sorted(tool_usage.items())),
        "agent_action_counts": {
            agent: dict(sorted(counter.items()))
            for agent, counter in agent_action_counts.items()
        },
        "formal_unknown_used": False,
        "leakage_guard": {
            "proposer_observation": "identity-side only",
            "critic_observation": "geometry/open-set-side only",
            "arbiter_raw_rf_access": False,
            "cross_agent_private_tool_visibility": False,
        },
    }

    print("[Stage15] 5/5 Writing consolidated outputs...")
    output = ROOT / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "agent_stats.json").write_text(
        json.dumps({
            "agent_action_counts": summary["agent_action_counts"],
            "tool_usage": summary["tool_usage"],
            "mean_turns": summary["mean_turns"],
            "mean_tool_calls": summary["mean_tool_calls"],
            "mean_messages": summary["mean_messages"],
            "unique_trajectories": summary["unique_trajectories"],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(output / "summary.md", summary)

    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n========== Stage 15 Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults: {output}")
    print(f"Human-readable summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
