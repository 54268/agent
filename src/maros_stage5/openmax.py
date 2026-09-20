"""Small self-contained OpenMax/EVT calibrator used by the Geometry Agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import softmax
from scipy.stats import weibull_min


@dataclass
class WeibullTail:
    shape: float
    scale: float


class OpenMaxEVT:
    def __init__(self, alpha_rank: int = 3, tail_size: int = 25):
        self.alpha_rank = int(alpha_rank)
        self.tail_size = int(tail_size)
        self.mavs: np.ndarray | None = None
        self.tails: dict[int, WeibullTail] = {}

    def fit(self, activations: np.ndarray, labels: np.ndarray,
            predictions: np.ndarray | None = None) -> "OpenMaxEVT":
        activations = np.asarray(activations, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if predictions is None:
            predictions = activations.argmax(axis=1)
        num_classes = activations.shape[1]
        self.mavs = np.zeros((num_classes, num_classes), dtype=np.float64)
        self.tails = {}
        for cls in range(num_classes):
            mask = (labels == cls) & (predictions == cls)
            rows = activations[mask]
            if len(rows) < 3:
                rows = activations[labels == cls]
            if len(rows) == 0:
                raise ValueError(f"class {cls} has no activations")
            mav = rows.mean(axis=0)
            self.mavs[cls] = mav
            distances = np.linalg.norm(rows - mav, axis=1)
            tail = np.sort(distances)[-min(self.tail_size, len(distances)):]
            tail = np.maximum(tail, 1e-8)
            try:
                shape, _, scale = weibull_min.fit(tail, floc=0.0)
            except (ValueError, FloatingPointError):
                shape, scale = 1.0, float(np.mean(tail))
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                shape, scale = 1.0, max(float(np.mean(tail)), 1e-6)
            self.tails[cls] = WeibullTail(float(shape), float(scale))
        return self

    def score(self, activations: np.ndarray) -> np.ndarray:
        if self.mavs is None:
            raise RuntimeError("OpenMaxEVT must be fitted before score")
        activations = np.asarray(activations, dtype=np.float64)
        scores = np.empty(len(activations), dtype=np.float32)
        for row_index, activation in enumerate(activations):
            revised = activation.copy()
            unknown_activation = 0.0
            ranked = np.argsort(activation)[::-1][:self.alpha_rank]
            for rank, cls in enumerate(ranked):
                tail = self.tails[int(cls)]
                distance = np.linalg.norm(activation - self.mavs[int(cls)])
                wscore = weibull_min.cdf(distance, tail.shape, loc=0.0, scale=tail.scale)
                omega = float(self.alpha_rank - rank) / max(self.alpha_rank, 1)
                reduced = revised[cls] * (1.0 - omega * wscore)
                unknown_activation += max(float(revised[cls] - reduced), 0.0)
                revised[cls] = reduced
            scores[row_index] = softmax(np.r_[revised, unknown_activation])[-1]
        return scores

    def state_dict(self) -> dict:
        return {
            "alpha_rank": self.alpha_rank,
            "tail_size": self.tail_size,
            "mavs": self.mavs,
            "tails": {str(key): asdict(value) for key, value in self.tails.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "OpenMaxEVT":
        result = cls(int(state["alpha_rank"]), int(state["tail_size"]))
        result.mavs = np.asarray(state["mavs"], dtype=np.float64)
        result.tails = {
            int(key): WeibullTail(**value) for key, value in state["tails"].items()
        }
        return result
