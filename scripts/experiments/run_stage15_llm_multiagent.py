"""Run Stage-15 LLM multi-agent open-set SEI on frozen RF evidence."""
from __future__ import annotations

import argparse
import json
import os
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
from maros_stage15.memory import CaseMemory  # noqa: E402
from maros_stage15.orchestrator import AgenticOpenSetSystem  # noqa: E402
from maros_stage15.providers import OpenAICompatibleChatModel  # noqa: E402


def resolve_config(path: str | Path) -> dict:
    path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/stage15_llm_multiagent_oracle.json")
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    base = resolve_config(cfg["base_stage14_config"])
    stage13 = resolve_config(base["stage13_config"])
    mother_cfg = resolve_config(stage13["mother_config"])

    seed = int(cfg["seed"])
    np.random.seed(seed)

    splits = _load_dataset(stage13)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=5, inner_folds=4,
        seed=int(stage13["partition_seed"]))
    fold = protocol.outer_folds[int(cfg["outer_fold"])]
    n = int(cfg["evaluation_samples_per_class"])
    test_known = cap_dataset(fold.test_known, n, seed + 1)
    test_unknown = cap_dataset(fold.test_proxy_unknown, n, seed + 2, proxy_unknown=True)

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

    llm_cfg = cfg["llm"]
    chat = OpenAICompatibleChatModel(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        temperature=float(llm_cfg.get("temperature", 0.1)),
        timeout_seconds=float(llm_cfg.get("timeout_seconds", 120)),
    )
    memory = CaseMemory(max_cases=int(cfg["agentic"].get("memory_size", 256)))
    agents = build_default_agents(chat, memory)
    system = AgenticOpenSetSystem(
        agents,
        max_turns=int(cfg["agentic"]["max_turns"]),
        tool_budget=int(cfg["agentic"]["tool_budget"]),
        memory=memory,
    )

    records = []
    known_correct = 0
    known_count = 0
    unknown_rejected = 0
    unknown_count = 0
    for snapshot in bank:
        result = system.run(snapshot)
        if snapshot.is_unknown:
            unknown_count += 1
            unknown_rejected += int(result.prediction == -1)
        else:
            known_count += 1
            known_correct += int(result.prediction == snapshot.label)
        records.append({
            "sample_id": snapshot.sample_id,
            "target": snapshot.label,
            "prediction": result.prediction,
            "decision": result.decision,
            "candidate_class": result.candidate_class,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "transcript": result.transcript,
        })

    known_acc = known_correct / max(known_count, 1)
    unknown_recall = unknown_rejected / max(unknown_count, 1)
    h_score = (
        2 * known_acc * unknown_recall / (known_acc + unknown_recall)
        if known_acc + unknown_recall > 0 else 0.0
    )
    summary = {
        "stage": "stage15_llm_multiagent",
        "dataset": cfg["dataset"],
        "outer_fold": fold.index,
        "model": llm_cfg["model"],
        "known_accuracy": known_acc,
        "unknown_recall": unknown_recall,
        "h_score": h_score,
        "samples": len(records),
        "formal_unknown_used": false if False else False
    }
    output = ROOT / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
