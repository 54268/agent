"""Run Certified PUG five-fold proxy test without touching formal Unknown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def resolve(path, stack=()):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if p in stack:
        raise ValueError("cyclic Certified PUG config")
    raw = json.loads(p.read_text(encoding="utf-8"))
    base = resolve(raw["base_config"], (*stack, p)) if "base_config" in raw else {}
    cfg = {**base, **raw}
    for key in ("oracle_root", "wisig_pkl", "mother_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    if "fold_checkpoint_pattern" in cfg:
        pattern = Path(cfg["fold_checkpoint_pattern"])
        if not pattern.is_absolute():
            cfg["fold_checkpoint_pattern"] = str(ROOT / pattern)
    cfg["config_path"] = str(p)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=
                        "configs/experiments/stage12_certified_pug_oracle.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first = resolve(args.config)
    second = resolve(first["paired_config"])
    oracle, wisig = ((first, second) if first["dataset"] == "oracle"
                     else (second, first))
    if args.dry_run:
        print(json.dumps({"oracle": oracle, "wisig": wisig},
                         indent=2, ensure_ascii=False))
        return
    from maros_stage12.experiment import run_pair
    result = run_pair(oracle, wisig, ROOT / "results" / "stage12_certified_pug")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
