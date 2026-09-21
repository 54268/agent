"""Run the paired, same-method Stage-9 G0 autonomy diagnostic only."""
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
        raise ValueError("cyclic Stage-9 config inheritance")
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = (resolve_config(raw["base_config"], (*stack, path))
            if "base_config" in raw else {})
    cfg = {**base, **raw}
    for key in ("wisig_pkl", "oracle_root", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=
                        "configs/experiments/stage9_g0_oracle_k10u6.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first = resolve_config(args.config)
    second = resolve_config(first["paired_config"])
    oracle, wisig = ((first, second) if first["dataset"] == "oracle"
                     else (second, first))
    if args.dry_run:
        print(json.dumps({"oracle": oracle, "wisig": wisig},
                         ensure_ascii=False, indent=2))
        return
    from maros_stage9.experiment import run_pair
    output = ROOT / "results" / "stage9_g0"
    report = run_pair(oracle, wisig, output)
    print(json.dumps({
        "stage": "stage9_g0", "g0_gate": report["g0_gate"],
        "summary_path": str(output / "g0_pair_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

