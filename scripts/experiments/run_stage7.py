"""Run leakage-safe Stage-7 conditional-consultation experiments.

The entry point intentionally owns only configuration resolution and the
cross-dataset method-equivalence guard.  The experiment module owns all data
access, training, gates, and artifact writing, so importing this file cannot
accidentally expose formal unknown data to a training call.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


CONFIG_METADATA = {
    "dataset",
    "name",
    "wisig_pkl",
    "oracle_root",
    "output_dir",
    "paired_config",
    "comparison_target",
    "base_config",
}


def resolve_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path in _stack:
        raise ValueError("cyclic Stage-7 base_config inheritance")
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_name = raw.get("base_config")
    if base_name:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = ROOT / base_path
        base = resolve_config(base_path, (*_stack, path))
        base.pop("config_path", None)
        cfg = {**base, **raw}
    else:
        cfg = raw
    cfg["config_path"] = str(path.resolve())
    for key in ("wisig_pkl", "oracle_root", "output_dir", "paired_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def method_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the fields that must match across datasets."""
    ignored = CONFIG_METADATA | {"config_path"}
    return {key: value for key, value in cfg.items() if key not in ignored}


def assert_paired_method_config(cfg: dict[str, Any]) -> None:
    peer_name = cfg.get("paired_config")
    if not peer_name:
        raise ValueError("Stage-7 requires a paired WiSig/ORACLE config")
    peer = resolve_config(peer_name)
    if cfg.get("dataset") == peer.get("dataset"):
        raise ValueError("paired Stage-7 configs must target different datasets")
    left, right = method_config(cfg), method_config(peer)
    different = sorted(key for key in set(left) | set(right)
                       if left.get(key) != right.get(key))
    if different:
        raise ValueError(
            "WiSig/ORACLE Stage-7 method configs differ: "
            + ", ".join(different))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-7 conditional-consultation OS-SEI")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", choices=(
            "sanity", "nested_sanity", "g0", "g1", "g2", "g3", "formal"),
        default="sanity",
        help="A later phase is blocked unless all preceding persisted gates pass.")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Optional outer-fold restriction for the two-dataset sanity run.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate paired configs and print the resolved run without training.")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    assert_paired_method_config(cfg)
    request = {"phase": args.phase, "fold": args.fold,
               "config": cfg["config_path"], "output_dir": cfg["output_dir"]}
    if args.dry_run:
        print(json.dumps(request, indent=2, ensure_ascii=False))
        return

    # Delayed import keeps config validation usable before heavyweight data or
    # CUDA modules are imported and gives tests a side-effect-free dry run.
    from maros_stage7.experiment import run_stage7

    result = run_stage7(cfg, phase=args.phase, outer_fold=args.fold)
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
