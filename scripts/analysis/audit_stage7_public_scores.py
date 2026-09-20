"""Audit Stage-7 local rejection evidence on an outer-LCO proxy fold.

This utility never opens the dataset's formal Unknown split.  It compares
fixed, class-count-independent evidence definitions using a dedicated Known
calibration split and the registered outer held-out classes.  Results are
diagnostic only: they cannot replace the five-fold G0 gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments.run_stage7 import resolve_config  # noqa: E402
from maros_stage7.contracts import RouteAction  # noqa: E402
from maros_stage7.experiment import (  # noqa: E402
    _load_fold_model,
    load_splits,
)
from maros_stage7.metrics import (  # noqa: E402
    ClasswiseCalibrationService,
    evaluate_open_set,
)
from maros_stage7.splits import build_nested_lco_protocol  # noqa: E402
from maros_stage7.training import collect_predictions  # noqa: E402


def _candidates(rows: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    offset = 0 if role == "waveform" else 7
    summary = np.asarray(rows["public_features"], dtype=np.float64)
    one_minus_top1 = 1.0 - summary[:, offset]
    negative_margin = -summary[:, offset + 2]
    entropy = summary[:, offset + 3]
    energy = summary[:, offset + 4]
    return {
        "learned_open": np.asarray(rows[f"raw_{role}"], dtype=np.float64),
        "one_minus_top1": one_minus_top1,
        "negative_margin": negative_margin,
        "entropy": entropy,
        "energy": energy,
        "one_minus_reliability": 1.0 - summary[:, offset + 6],
        "entropy_energy_mean": 0.5 * (entropy + energy),
        "confidence_energy_mean": 0.5 * (one_minus_top1 + energy),
        "three_evidence_mean": (
            one_minus_top1 + entropy + energy) / 3.0,
    }


def audit(config: Path, checkpoint: Path, fold: int) -> dict:
    cfg = resolve_config(config)
    protocol = build_nested_lco_protocol(
        load_splits(cfg), outer_folds=int(cfg.get("outer_folds", 5)),
        inner_folds=int(cfg.get("inner_folds", 4)),
        seed=int(cfg.get("partition_seeds", [2026])[0]))
    episode = protocol.outer_folds[int(fold)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_fold_model(checkpoint, cfg, episode, device)
    collect = lambda dataset: collect_predictions(  # noqa: E731
        model, dataset, device, route_action=RouteAction.STOP,
        batch_size=int(cfg.get("evaluation_batch_size", 1024)))
    calibration = collect(episode.calibration_known)
    known = collect(episode.test_known)
    proxy = collect(episode.test_proxy_unknown)
    y = np.r_[known["y"], proxy["y"]]
    result = {
        "dataset": cfg["dataset"], "fold": int(fold),
        "formal_unknown_used": False, "roles": {},
    }
    for role in ("waveform", "prototype"):
        prediction_cal = calibration[f"pred_{role}"]
        prediction = np.r_[known[f"pred_{role}"], proxy[f"pred_{role}"]]
        rows = {}
        cal_scores = _candidates(calibration, role)
        test_scores = {
            name: np.r_[_candidates(known, role)[name],
                        _candidates(proxy, role)[name]]
            for name in cal_scores
        }
        for name, raw in test_scores.items():
            service = ClasswiseCalibrationService(
                float(cfg.get("known_acceptance", 0.95)))
            service.fit(
                cal_scores[name], prediction_cal, labels=calibration["y"],
                dataset=episode.calibration_known)
            decision = service.predict(raw, prediction)
            metrics = evaluate_open_set(
                y, prediction, decision.unknown_score, decision.threshold)
            rows[name] = {
                "raw_auroc": float(roc_auc_score(y < 0, raw)),
                **{key: float(metrics[key]) for key in (
                    "auroc", "oscr", "known_accuracy", "unknown_recall",
                    "h_score")},
            }
        result["roles"][role] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.checkpoint, args.fold)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
