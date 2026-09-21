"""CLI for the registered small Stage-8 G0/G1 probe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def resolve_config(path: str | Path, stack=()) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path in stack:
        raise ValueError("cyclic Stage-8 base_config inheritance")
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_name = raw.get("base_config")
    if base_name:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = ROOT / base_path
        base = resolve_config(base_path, (*stack, path))
        cfg = {**base, **raw}
    else:
        cfg = raw
    for key in ("wisig_pkl", "oracle_root", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def method_config(cfg: dict) -> dict:
    metadata = {
        "base_config", "config_path", "dataset", "name", "wisig_pkl",
        "oracle_root", "output_dir", "paired_config",
    }
    return {key: value for key, value in cfg.items() if key not in metadata}


def assert_paired_method_config(cfg: dict) -> None:
    peer = resolve_config(cfg["paired_config"])
    if cfg["dataset"] == peer["dataset"]:
        raise ValueError("paired Stage-8 configs must target different datasets")
    different = sorted(
        key for key in set(method_config(cfg)) | set(method_config(peer))
        if method_config(cfg).get(key) != method_config(peer).get(key))
    if different:
        raise ValueError(
            "WiSig/ORACLE Stage-8 method configs differ: " + ", ".join(different))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-8 G0/G1 small probe")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", choices=("learnability", "g1", "agent_b_v3_quick"),
        default="g1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    assert_paired_method_config(cfg)
    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return
    from maros_stage8.probe import (run_agent_b_v3_quick, run_g1_probe,
                                    run_local_learnability)
    if args.phase == "learnability":
        report = run_local_learnability(cfg)
    elif args.phase == "agent_b_v3_quick":
        report = run_agent_b_v3_quick(cfg)
    else:
        report = run_g1_probe(cfg)
    summary = {
        "dataset": report["dataset"], "phase": args.phase,
        "output_dir": cfg["output_dir"],
    }
    if args.phase == "agent_b_v3_quick":
        summary.update({
            "conclusion": report["conclusion"],
            "next_action": report["next_action"],
            "fold0_gate": report["fold0_gate"],
            "spectral_eval_accuracy": report["spectral_eval_accuracy"],
            "spectral_pair_auroc": report["spectral_pair_metrics"]["pair_auroc"],
        })
    elif args.phase == "g1":
        summary.update({
            "outcome": report["outcome"], "decision": report["decision"],
        })
        summary.update({
            "g1_gate": report["g1_gate"],
            "g15_gate": report["g15_gate"],
        })
    else:
        summary.update({
            "outcome": report["outcome"], "decision": report["decision"],
            "passed": report["passed"],
            "selected_candidate": report["selected_candidate"],
        })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
