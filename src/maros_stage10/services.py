"""Calibration and threshold are services, not decision-making Agents."""
from __future__ import annotations

import numpy as np


class KnownQuantileThresholdService:
    def __init__(self, acceptance: float = 0.95):
        if not 0 < acceptance < 1:
            raise ValueError("Known acceptance must lie strictly between zero and one")
        self.acceptance = float(acceptance)
        self.threshold: float | None = None

    def fit(self, calibration_known_risk: np.ndarray):
        risk = np.asarray(calibration_known_risk, dtype=np.float64)
        if risk.ndim != 1 or len(risk) == 0 or not np.isfinite(risk).all():
            raise ValueError("threshold calibration requires finite Known risks")
        self.threshold = float(np.quantile(risk, self.acceptance))
        return self

    def reject(self, risk: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("threshold service must be calibrated")
        return np.asarray(risk) >= self.threshold
