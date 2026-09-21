"""Paired frozen-Stage-5 candidate-consultation feasibility probe."""
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
        raise ValueError("cyclic Stage-10 config inheritance")
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = (resolve_config(raw["base_config"], (*stack, path))
            if "base_config" in raw else {})
    cfg = {**base, **raw}
    for key in ("oracle_root", "wisig_pkl", "mother_config",
                "mother_checkpoint", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=
                        "configs/experiments/stage10_stage5_mother_oracle.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first = resolve_config(args.config)
    second = resolve_config(first["paired_config"])
    oracle, wisig = ((first, second) if first["dataset"] == "oracle"
                     else (second, first))
    if args.dry_run:
        print(json.dumps({"oracle": oracle, "wisig": wisig},
                         indent=2, ensure_ascii=False))
        return
    from maros_stage10.experiment import run_pair
    output = ROOT / "results" / "stage10_stage5_mother"
    summary = run_pair(oracle, wisig, output)
    print(json.dumps({
        "paired_gate_passed": summary["paired_gate_passed"],
        "datasets": {name: {
            "gate": report["consultation_gate"],
            "identity_accuracy": report["identity_local_accuracy"],
            "geometry_accuracy": report["geometry_local_accuracy"],
            "no_comm": report["conditions"]["no_communication"],
            "active_query": report["conditions"]["agent_initiated_query"],
        } for name, report in summary["datasets"].items()},
        "summary_path": str(output / "paired_consultation_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

