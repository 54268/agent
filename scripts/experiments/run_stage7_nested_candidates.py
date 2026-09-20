"""Screen a few pre-registered public-only B2 candidates from cached inner tables.

This diagnostic never evaluates an outer test fold or a formal Unknown.  It
exists to decide whether the no-communication parent has enough action space
to exploit the already demonstrated Waveform/Prototype complementarity before
any Stage-7 communication component is trained.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.experiment import (  # noqa: E402
    _evaluate_public_table_fold,
    _pool_inner_oof_reports,
    _save_oof_summary,
    _system,
    _write_json,
    load_splits,
)
from maros_stage7.nested_b2 import (  # noqa: E402
    PublicDecisionTable,
    fit_nested_public_b2,
    public_local_action_losses,
)
from maros_stage7.nested_g0 import (  # noqa: E402
    SelectorRowBatch,
    build_inner_registration_partitions,
    inner_oof_selector_evidence,
)
from maros_stage7.splits import build_nested_lco_protocol  # noqa: E402


CANDIDATES = {
    # Reproduce the constrained V4 reference under the same runner.
    "anchored_reference": {
        "b2_max_adaptive_mass": 0.35,
        "b2_initial_adaptive_fraction": 0.2,
        "nested_b2_epochs": 5,
        "b2_selector_distillation_weight": 0.25,
        "b2_no_regret_weight": 1.0,
    },
    # Hypothesis A: the selector signal is useful, but the anchor constraint
    # prevented it from performing a near-discrete expert switch.
    "free_selector": {
        "b2_max_adaptive_mass": 1.0,
        "b2_initial_adaptive_fraction": 0.9,
        "nested_b2_epochs": 20,
        "b2_selector_distillation_weight": 1.0,
        "b2_no_regret_weight": 1.0,
    },
    # Hypothesis B: freeing the action is sufficient and the stronger selector
    # supervision itself may overfit the inner training episodes.
    "free_task_balanced": {
        "b2_max_adaptive_mass": 1.0,
        "b2_initial_adaptive_fraction": 0.9,
        "nested_b2_epochs": 20,
        "b2_selector_distillation_weight": 0.25,
        "b2_no_regret_weight": 1.0,
    },
    # Hypothesis C: a stronger sample-wise no-regret penalty may reduce the two
    # negative held-out folds observed by the V4 reference.
    "free_no_regret": {
        "b2_max_adaptive_mass": 1.0,
        "b2_initial_adaptive_fraction": 0.9,
        "nested_b2_epochs": 20,
        "b2_selector_distillation_weight": 1.0,
        "b2_no_regret_weight": 2.0,
    },
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "output_dir", "paired_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg["config_path"] = str(path.resolve())
    return cfg


def _load_tables(source: Path, partitions, device, cfg):
    roles = (
        "train_known", "train_proxy_unknown",
        "threshold_calibration_known", "meta_known", "meta_proxy_unknown",
    )
    all_tables = {}
    batches = []
    for partition in partitions:
        inner = int(partition.inner_index)
        tables = {}
        for role in roles:
            table = PublicDecisionTable.load(
                source / f"inner{inner}" / f"public_{role}.npz")
            expected = getattr(partition, role)
            if (table.outer_index != partition.outer_index
                    or table.inner_index != inner
                    or table.split_role != role
                    or tuple(table.sample_keys) != tuple(expected.sample_keys)):
                raise RuntimeError(
                    f"cached public table does not match inner protocol: {inner}/{role}")
            tables[role] = table
        all_tables[inner] = tables
        features, losses = [], []
        for role in ("meta_known", "meta_proxy_unknown"):
            x, y = public_local_action_losses(
                tables[role], device,
                batch_size=int(cfg.get("evaluation_batch_size", 1024)),
                class_weight=float(cfg.get("class_loss_weight", 1.0)),
                open_weight=float(cfg.get("open_loss_weight", 1.0)))
            features.append(x)
            losses.append(y)
        batches.append(SelectorRowBatch(
            producer_inner_index=inner,
            sample_keys=tuple(partition.selector_sample_keys),
            public_features=np.concatenate(features),
            action_losses=np.concatenate(losses)))
    return all_tables, inner_oof_selector_evidence(batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-fold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(_resolve(args.config))
    source = _resolve(args.source_fold_dir)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "public-only nested B2 action-space diagnosis",
        "source_fold_dir": str(source.resolve()),
        "outer_fold": int(args.fold),
        "candidates": CANDIDATES,
        "ranking": [
            "positive_inner_folds>=3", "delta_h_vs_best_local>=0.01",
            "pooled_h_score", "pooled_oscr"],
        "candidate_initialization": "shared_within_heldout_fold",
        "outer_test_evaluated": False,
        "formal_unknown_used": False,
    }
    manifest_path = output / "candidate_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("candidate screen manifest changed in-place")
    else:
        _write_json(manifest_path, manifest)

    splits = load_splits(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg.get("outer_folds", 5)),
        inner_folds=int(cfg.get("inner_folds", 4)),
        seed=int(cfg.get("partition_seeds", [2026])[0]))
    episode = protocol.outer_folds[int(args.fold)]
    partitions = build_inner_registration_partitions(
        episode, seed=int(cfg.get("partition_seeds", [2026])[0]),
        calibration_fraction=float(cfg.get("nested_calibration_fraction", 0.5)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_tables, evidence = _load_tables(
        source, partitions, device, cfg)

    summaries = {}
    for name, overrides in CANDIDATES.items():
        candidate_cfg = copy.deepcopy(cfg)
        candidate_cfg.update(overrides)
        candidate_dir = output / name
        reports = []
        print(f"[stage7:nested-candidate] {name}", flush=True)
        for partition in partitions:
            heldout = int(partition.inner_index)
            fit_tables = [
                all_tables[index][role]
                for index in sorted(all_tables) if index != heldout
                for role in ("train_known", "train_proxy_unknown")]
            carrier = _system(candidate_cfg, fit_tables[0].num_classes).to(device)
            # All candidates receive the exact same initialization for a
            # given held-out producer.  Candidate-specific seeds would
            # confound architecture/loss changes with optimiser noise.
            seed = int(cfg.get("seed", 42)) + 40000 + 211 * heldout
            fit = fit_nested_public_b2(
                carrier, fit_tables, candidate_cfg, device, seed)
            heldout_dir = candidate_dir / f"heldout_inner{heldout}"
            heldout_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(fit.bundle), heldout_dir / "b2_transfer.pt")
            _write_json(heldout_dir / "fit_audit.json", {
                "overrides": overrides, "audit": fit.audit,
                "history": fit.history})
            report = _evaluate_public_table_fold(
                carrier, partition, {
                    role: all_tables[heldout][role]
                    for role in ("threshold_calibration_known", "meta_known",
                                 "meta_proxy_unknown")},
                candidate_cfg, device, heldout_dir)
            reports.append(report)
            carrier.to("cpu")
            del carrier
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pooled = _pool_inner_oof_reports(
            reports, evidence=evidence, cfg=candidate_cfg)
        _save_oof_summary(candidate_dir, pooled)
        summaries[name] = {
            key: value for key, value in pooled.items() if key != "_systems"}
        print(
            f"  H={pooled['metrics']['b2']['h_score']:.6f} "
            f"dH={pooled['delta_h_vs_best_local']:+.6f} "
            f"positive={pooled['positive_inner_folds']}/4 "
            f"passed={pooled['passed']}", flush=True)

    eligible = [name for name, row in summaries.items() if row["passed"]]
    eligible.sort(key=lambda name: (
        summaries[name]["metrics"]["b2"]["h_score"],
        summaries[name]["metrics"]["b2"]["oscr"]), reverse=True)
    selection = {
        "promotable": bool(eligible),
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "candidates": summaries,
        "stopped_before_outer_evaluation": True,
        "outer_test_evaluated": False,
        "formal_unknown_used": False,
    }
    _write_json(output / "candidate_selection.json", selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
