from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_sei.metrics import evaluate_open_set
from maros_sei.model import MAROSSEI, ModelConfig


def test_model_shapes() -> None:
    model = MAROSSEI(ModelConfig(num_classes=10, num_receivers=6, num_dates=4, embedding_dim=32, top_k=2))
    output = model(torch.randn(5, 2, 256))
    assert output["fused_logits"].shape == (5, 10)
    assert output["post_logits"].shape == (5, 4, 10)
    assert output["adjacency"].shape == (5, 4, 4)
    assert output["fusion_weights"].shape == (5, 4)
    assert torch.allclose(output["adjacency"].sum(-1), torch.ones(5, 4), atol=1e-5)
    assert torch.allclose(output["fusion_weights"].sum(-1), torch.ones(5), atol=1e-5)


def test_open_set_metrics_perfect() -> None:
    unknown = 2
    y_true = np.asarray([0, 1, 0, 1, unknown, unknown])
    closed = np.asarray([0, 1, 0, 1, 0, 1])
    score = np.asarray([0.01, 0.02, 0.03, 0.04, 0.95, 0.99])
    predicted = closed.copy()
    predicted[score >= 0.5] = unknown
    metrics, _ = evaluate_open_set(y_true, predicted, closed, score, unknown)
    assert metrics["known_accuracy"] == 1.0
    assert metrics["unknown_recall"] == 1.0
    assert metrics["unknown_precision"] == 1.0
    assert metrics["auroc"] == 1.0

