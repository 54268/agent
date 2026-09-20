"""Diagnostic-only open-test audit of frozen local evidence rules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.agents import RoleStructuredExperts  # noqa: E402
from maros_stage5.calibration_agent import CalibrationAgent, ClassConditionalThresholdAgent  # noqa: E402
from maros_stage5.openmax import OpenMaxEVT  # noqa: E402
from maros_stage5.training import extract_records  # noqa: E402
from maros_staged.datasets import load_oracle_npz, load_wisig_subset  # noqa: E402
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config); cfg_path = cfg_path if cfg_path.is_absolute() else ROOT / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    source_key = "oracle_root" if cfg["dataset"] == "oracle" else "wisig_pkl"
    source = Path(cfg[source_key]); source = source if source.is_absolute() else ROOT / source
    splits = (load_oracle_npz(source, False) if cfg["dataset"] == "oracle"
              else load_wisig_subset(source, False))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute(): checkpoint_path = ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoleStructuredExperts(
        splits.num_known, int(cfg["state_dim"]), int(cfg["stem_channels"]),
        int(cfg["message_dim"]), float(cfg["prototype_temperature"]),
        str(cfg["geometry_view"])).to(device)
    model.load_state_dict(checkpoint["experts"])
    evt = OpenMaxEVT.from_state_dict(checkpoint["openmax"])
    val = extract_records(model, splits.val, device, evt)
    opened = extract_records(model, splits.open_test, device, evt)
    val_pred = (0.5 * val.identity_logits + 0.5 * val.geometry_logits).argmax(1)
    open_pred = (0.5 * opened.identity_logits + 0.5 * opened.geometry_logits).argmax(1)
    dummy_val = {"score": np.zeros(len(val.labels)), "pred": val_pred}
    dummy_open = {"score": np.zeros(len(opened.labels)), "pred": open_pred}
    rules = [name for name, components in CalibrationAgent.RULES.items()
             if "boundary" not in components]
    unknown = (opened.labels == -1).astype(np.int32)
    rows = []
    for rule in rules:
        calibrator = CalibrationAgent(rule).fit(val, dummy_val)
        val_score = calibrator.transform(val, dummy_val).unknown_score
        score = calibrator.transform(opened, dummy_open).unknown_score
        base = detection_metrics(unknown, score)
        base["oscr"] = oscr(unknown, open_pred, opened.labels, 1.0 - score)
        for acceptance in cfg["known_acceptance_grid"]:
            for prior in cfg["adaptive_threshold_priors"]:
                if prior is None:
                    thresholds = np.full(len(score), acceptance)
                else:
                    agent = ClassConditionalThresholdAgent(acceptance, prior).fit(
                        val_score, val_pred)
                    thresholds = agent.thresholds(open_pred)
                metrics = dict(base)
                metrics.update(threshold_decision(
                    unknown, opened.labels, open_pred, score, thresholds))
                rows.append({"rule": rule, "known_acceptance": acceptance,
                             "prior": prior, "metrics": metrics})
    rows.sort(key=lambda row: (row["metrics"]["h_score"], row["metrics"]["oscr"]),
              reverse=True)
    report = {"warning": "uses real unknown for diagnosis only; forbidden for selection",
              "rows": rows, "best": rows[0]}
    output = Path(args.output); output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
