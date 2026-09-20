"""Evaluate the conservative public-only selector on cached inner episodes."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.experiment import _write_json, load_splits  # noqa: E402
from maros_stage7.metrics import (  # noqa: E402
    ClasswiseCalibrationService,
    evaluate_open_set,
)
from maros_stage7.nested_b2 import (  # noqa: E402
    PublicDecisionTable,
    public_local_action_losses,
)
from maros_stage7.nested_g0 import (  # noqa: E402
    SelectorRowBatch,
    build_inner_registration_partitions,
)
from maros_stage7.safe_selector import fit_safe_public_selector  # noqa: E402
from maros_stage7.splits import build_nested_lco_protocol  # noqa: E402


ROLES = ("waveform", "prototype")
TABLE_ROLES = (
    "threshold_calibration_known", "meta_known", "meta_proxy_unknown")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wisig_pkl", "oracle_root", "output_dir", "paired_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def _load(source: Path, partitions, device, cfg):
    all_tables, batches = {}, []
    for partition in partitions:
        inner = int(partition.inner_index)
        tables = {}
        for role in (
                "meta_known", "meta_proxy_unknown",
                "threshold_calibration_known"):
            table = PublicDecisionTable.load(
                source / f"inner{inner}" / f"public_{role}.npz")
            expected = getattr(partition, role)
            if (table.inner_index != inner or table.outer_index != partition.outer_index
                    or tuple(table.sample_keys) != tuple(expected.sample_keys)):
                raise RuntimeError(f"cached table/protocol mismatch: {inner}/{role}")
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
    return all_tables, batches


def _table_arrays(table, selector, device, cfg):
    features, _ = public_local_action_losses(
        table, device,
        batch_size=int(cfg.get("evaluation_batch_size", 1024)),
        class_weight=float(cfg.get("class_loss_weight", 1.0)),
        open_weight=float(cfg.get("open_loss_weight", 1.0)))
    choice = selector.choose(features)
    local_pred = np.stack([
        table.waveform_logits.argmax(1), table.prototype_logits.argmax(1)], 1)
    local_raw = np.stack([
        table.waveform_unknown, table.prototype_unknown], 1)
    rows = np.arange(len(table))
    return {
        "y": np.asarray(table.labels),
        "choice": choice,
        "pred": local_pred[rows, choice],
        "raw": local_raw[rows, choice],
        "pred_waveform": local_pred[:, 0],
        "raw_waveform": local_raw[:, 0],
        "pred_prototype": local_pred[:, 1],
        "raw_prototype": local_raw[:, 1],
    }


def _evaluate_fold(selector, partition, tables, device, cfg):
    rows = {role: _table_arrays(table, selector, device, cfg)
            for role, table in tables.items()}
    systems, metrics = {}, {}
    for role in (*ROLES, "safe_selector"):
        pred_key = "pred" if role == "safe_selector" else f"pred_{role}"
        raw_key = "raw" if role == "safe_selector" else f"raw_{role}"
        service = ClasswiseCalibrationService(float(cfg.get("known_acceptance", 0.95)))
        calibration = rows["threshold_calibration_known"]
        service.fit(
            calibration[raw_key], calibration[pred_key],
            labels=calibration["y"],
            dataset=partition.threshold_calibration_known)
        known, proxy = rows["meta_known"], rows["meta_proxy_unknown"]
        y = np.r_[known["y"], proxy["y"]]
        pred = np.r_[known[pred_key], proxy[pred_key]]
        raw = np.r_[known[raw_key], proxy[raw_key]]
        decision = service.predict(raw, pred)
        systems[role] = {
            "y": y, "closed_pred": pred,
            "unknown_score": decision.unknown_score,
            "threshold": decision.threshold,
        }
        metrics[role] = evaluate_open_set(
            y, pred, decision.unknown_score, decision.threshold)
    choices = np.r_[rows["meta_known"]["choice"],
                    rows["meta_proxy_unknown"]["choice"]]
    return systems, metrics, float((choices != selector.audit.anchor_index).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-fold-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    cfg = _config(_resolve(args.config))
    source, output = _resolve(args.source_fold_dir), _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "fit-side producer-OOF conservative public selector",
        "probability_grid": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "minimum_switch_rate": 0.05,
        "maximum_switch_rate": 0.5,
        "source_fold_dir": str(source.resolve()),
        "outer_test_evaluated": False,
        "formal_unknown_used": False,
    }
    _write_json(output / "screen_manifest.json", manifest)

    protocol = build_nested_lco_protocol(
        load_splits(cfg), outer_folds=int(cfg.get("outer_folds", 5)),
        inner_folds=int(cfg.get("inner_folds", 4)),
        seed=int(cfg.get("partition_seeds", [2026])[0]))
    episode = protocol.outer_folds[int(args.fold)]
    partitions = build_inner_registration_partitions(
        episode, seed=int(cfg.get("partition_seeds", [2026])[0]),
        calibration_fraction=float(cfg.get("nested_calibration_fraction", 0.5)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_tables, batches = _load(source, partitions, device, cfg)
    reports = []
    for partition in partitions:
        heldout = int(partition.inner_index)
        fit_batches = [row for row in batches
                       if row.producer_inner_index != heldout]
        selector = fit_safe_public_selector(
            fit_batches, seed=int(cfg.get("seed", 42)) + 51000 + 211 * heldout)
        systems, metrics, switch_rate = _evaluate_fold(
            selector, partition, all_tables[heldout], device, cfg)
        report = {
            "heldout_inner": heldout, "audit": asdict(selector.audit),
            "metrics": metrics, "heldout_switch_rate": switch_rate,
        }
        reports.append({**report, "_systems": systems})
        _write_json(output / f"heldout_inner{heldout}.json", report)

    pooled = {}
    for role in (*ROLES, "safe_selector"):
        values = {key: np.concatenate([
            report["_systems"][role][key] for report in reports])
            for key in ("y", "closed_pred", "unknown_score", "threshold")}
        pooled[role] = evaluate_open_set(
            values["y"], values["closed_pred"],
            values["unknown_score"], values["threshold"])
    best_local = max(pooled[role]["h_score"] for role in ROLES)
    delta = float(pooled["safe_selector"]["h_score"] - best_local)
    fold_deltas = [
        float(report["metrics"]["safe_selector"]["h_score"]
              - max(report["metrics"][role]["h_score"] for role in ROLES))
        for report in reports]
    result = {
        "passed": delta >= 0.01 and sum(value > 0 for value in fold_deltas) >= 3,
        "metrics": pooled, "delta_h_vs_best_local": delta,
        "fold_h_deltas": fold_deltas,
        "positive_inner_folds": sum(value > 0 for value in fold_deltas),
        "mean_heldout_switch_rate": float(np.mean([
            report["heldout_switch_rate"] for report in reports])),
        "fold_audits": [{key: value for key, value in report.items()
                         if key != "_systems"} for report in reports],
        "stopped_before_outer_evaluation": True,
        "outer_test_evaluated": False,
        "formal_unknown_used": False,
    }
    _write_json(output / "safe_selector_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
