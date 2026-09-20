from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_sei.data import CombinedTestDataset, WiSigArrayDataset, load_json, save_json
from maros_sei.experiment import calibrator_features, collect_outputs
from maros_sei.metrics import evaluate_open_set
from maros_sei.model import MAROSSEI, ModelConfig


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def candidate_scores(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    probability = softmax(outputs["fused_logits"])
    entropy = -(probability * np.log(probability + 1e-12)).sum(1) / math.log(probability.shape[1])
    post_probability = softmax(outputs["post_logits"])
    return {
        "network_sigmoid": 1.0 / (1.0 + np.exp(-outputs["unknown_logit"])),
        "one_minus_fused_msp": 1.0 - probability.max(1),
        "fused_entropy": entropy,
        "mean_agent_uncertainty": (1.0 - post_probability.max(2)).mean(1),
        "agent_js_divergence": outputs["open_features"][:, 3],
        "domain_shift": outputs["open_features"][:, 4],
    }


def benchmark(model: MAROSSEI, dataset: WiSigArrayDataset, device: torch.device, batch_size: int) -> dict[str, float]:
    count = min(batch_size, len(dataset))
    x = torch.stack([dataset[index]["iq"] for index in range(count)]).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(15):
            model(x, grl_scale=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(60):
                model(x, grl_scale=0.0)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            started = time.perf_counter()
            for _ in range(20):
                model(x, grl_scale=0.0)
            total_ms = (time.perf_counter() - started) * 1000.0 * 3.0
    batch_ms = total_ms / 60.0
    return {
        "batch_size": int(count),
        "milliseconds_per_batch": float(batch_ms),
        "milliseconds_per_sample": float(batch_ms / count),
        "samples_per_second": float(1000.0 * count / batch_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = (ROOT / args.run_dir).resolve()
    config = load_json(run_dir / "resolved_config.json")
    manifest = load_json(run_dir / "data_manifest.json")
    processed = ROOT / config["data"]["processed_dir"]
    batch_size = int(config["training"]["batch_size"])
    kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    validation = WiSigArrayDataset(processed, "validation", augment=False)
    test_known = WiSigArrayDataset(processed, "test_known", augment=False)
    test_unknown = WiSigArrayDataset(processed, "test_unknown", augment=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAROSSEI(
        ModelConfig(
            num_classes=int(manifest["unknown_label"]),
            num_receivers=len(manifest["rx_names"]),
            num_dates=len(manifest["date_names"]),
            embedding_dim=int(config["model"]["embedding_dim"]),
            top_k=int(config["model"]["top_k"]),
        )
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation_outputs = collect_outputs(model, DataLoader(validation, **kwargs), device)
    test_outputs = collect_outputs(model, DataLoader(CombinedTestDataset(test_known, test_unknown), **kwargs), device)
    unknown_label = int(manifest["unknown_label"])
    y_true = test_outputs["labels"].astype(np.int64)
    closed = test_outputs["fused_logits"].argmax(1).astype(np.int64)
    diagnostics = {}
    val_candidates = candidate_scores(validation_outputs)
    test_candidates = candidate_scores(test_outputs)
    for name, val_score in val_candidates.items():
        threshold = float(np.quantile(val_score, 0.95))
        score = test_candidates[name]
        predicted = closed.copy()
        predicted[score >= threshold] = unknown_label
        metrics, _ = evaluate_open_set(y_true, predicted, closed, score, unknown_label)
        diagnostics[name] = {"validation_known_q95_threshold": threshold, **metrics}

    calibrator = joblib.load(run_dir / "unknown_calibrator.joblib")
    calibrated_score = calibrator.predict_proba(calibrator_features(test_outputs))[:, 1]
    official_threshold = float(load_json(run_dir / "calibration.json")["threshold"])
    predicted = closed.copy()
    predicted[calibrated_score >= official_threshold] = unknown_label
    official_metrics, _ = evaluate_open_set(y_true, predicted, closed, calibrated_score, unknown_label)
    payload = {
        "note": "All diagnostic thresholds use only the known validation distribution. True unknown labels are used only below for final metric evaluation.",
        "official_pseudo_unknown_calibrator": {"threshold": official_threshold, **official_metrics},
        "known_validation_q95_scores": diagnostics,
        "latency": benchmark(model, test_known, device, batch_size),
    }
    save_json(run_dir / "score_diagnostics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
