"""Known-only calibration agent for heterogeneous open-set evidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maros_staged.evidence import ClassConditionalCdf, EmpiricalCdfCalibrator


@dataclass
class CalibrationOutput:
    unknown_score: np.ndarray
    component_scores: dict[str, np.ndarray]
    rule: str


@dataclass
class ArbitrationOutput:
    """Known-calibrated risk from local and message-informed audit branches."""

    unknown_score: np.ndarray
    local_score: np.ndarray
    communicated_score: np.ndarray
    rule: str


class CalibrationAgent:
    """Turn specialist evidence into a comparable, fused unknown risk.

    This agent is fitted exclusively on known validation data.  Its local
    decision is a calibrated unknown score; it neither changes class logits
    nor observes real unknowns.  The fusion rule itself is selected by LCO.
    """

    RULES = {
        "boundary": ("boundary",),
        "mean3": ("identity", "prototype", "boundary"),
        "mean4": ("identity", "prototype", "openmax", "boundary"),
        "geo_open_boundary": ("prototype", "openmax", "boundary"),
        "max4": ("identity", "prototype", "openmax", "boundary"),
        "idproto_boundary": ("identity_prototype", "boundary"),
        "mean5": ("identity", "identity_prototype", "prototype", "openmax", "boundary"),
        "max5": ("identity", "identity_prototype", "prototype", "openmax", "boundary"),
        "sensor_mean": ("identity_prototype", "geometry_raw", "geometry_spectral",
                        "geometry_envelope_phase", "geometry_difference_iq",
                        "geometry_complex_iq", "boundary"),
        "sensor_max": ("identity_prototype", "geometry_raw", "geometry_spectral",
                       "geometry_envelope_phase", "geometry_difference_iq",
                       "geometry_complex_iq", "boundary"),
        "idproto": ("identity_prototype",),
        "rawproto": ("geometry_raw",),
        "spectralproto": ("geometry_spectral",),
        "envelopeproto": ("geometry_envelope_phase",),
        "differenceproto": ("geometry_difference_iq",),
        "complexproto": ("geometry_complex_iq",),
        "proto_mean": ("identity_prototype", "geometry_raw", "geometry_spectral",
                       "geometry_envelope_phase", "geometry_difference_iq",
                       "geometry_complex_iq"),
        "proto_max": ("identity_prototype", "geometry_raw", "geometry_spectral",
                      "geometry_envelope_phase", "geometry_difference_iq",
                      "geometry_complex_iq"),
    }
    MAX_RULES = {"max4", "max5", "sensor_max", "proto_max"}

    def __init__(self, rule: str, n_prior: float = 50.0):
        if rule not in self.RULES:
            raise ValueError(f"unsupported calibration rule {rule}")
        self.rule = rule
        self.n_prior = float(n_prior)
        self.component_calibrators = {}
        self.rule_calibrator: EmpiricalCdfCalibrator | None = None

    @staticmethod
    def _classes(records):
        return (records.identity_logits.argmax(1), records.geometry_logits.argmax(1))

    def _components(self, records, prediction):
        if not self.component_calibrators:
            raise RuntimeError("CalibrationAgent must be fitted before transform")
        id_class, geo_class = self._classes(records)
        view_class = (records.geometry_view_pred if records.geometry_view_pred is not None
                      else np.repeat(geo_class[:, None], 5, axis=1))
        if view_class.shape[1] < 5:
            view_class = np.repeat(geo_class[:, None], 5, axis=1)
        idproto = records.evidence[:, 7] if records.evidence.shape[1] > 7 else records.evidence[:, 0]
        view_values = (records.evidence[:, 10:15] if records.evidence.shape[1] >= 15
                       else np.repeat(records.evidence[:, 3:4], 5, axis=1))
        components = {
            "identity": self.component_calibrators["identity"].score(
                records.evidence[:, 0], id_class),
            "prototype": self.component_calibrators["prototype"].score(
                records.evidence[:, 3], geo_class),
            "openmax": self.component_calibrators["openmax"].score(
                records.evidence[:, 6], geo_class),
            "boundary": self.component_calibrators["boundary"].score(prediction["score"]),
            "identity_prototype": self.component_calibrators["identity_prototype"].score(
                idproto, id_class),
            "geometry_raw": self.component_calibrators["geometry_raw"].score(
                view_values[:, 0], view_class[:, 0]),
            "geometry_spectral": self.component_calibrators["geometry_spectral"].score(
                view_values[:, 1], view_class[:, 1]),
            "geometry_envelope_phase": self.component_calibrators[
                "geometry_envelope_phase"].score(view_values[:, 2], view_class[:, 2]),
            "geometry_difference_iq": self.component_calibrators[
                "geometry_difference_iq"].score(view_values[:, 3], view_class[:, 3]),
            "geometry_complex_iq": self.component_calibrators[
                "geometry_complex_iq"].score(view_values[:, 4], view_class[:, 4]),
        }
        return {name: components[name] for name in self.RULES[self.rule]}

    def _fuse(self, components):
        values = np.stack([components[name] for name in self.RULES[self.rule]], axis=1)
        return values.max(axis=1) if self.rule in self.MAX_RULES else values.mean(axis=1)

    def fit(self, known_records, known_prediction):
        id_class, geo_class = self._classes(known_records)
        view_class = (known_records.geometry_view_pred
                      if known_records.geometry_view_pred is not None
                      else np.repeat(geo_class[:, None], 5, axis=1))
        if view_class.shape[1] < 5:
            view_class = np.repeat(geo_class[:, None], 5, axis=1)
        idproto = (known_records.evidence[:, 7] if known_records.evidence.shape[1] > 7
                   else known_records.evidence[:, 0])
        view_values = (known_records.evidence[:, 10:15]
                       if known_records.evidence.shape[1] >= 15
                       else np.repeat(known_records.evidence[:, 3:4], 5, axis=1))
        self.component_calibrators = {
            "identity": ClassConditionalCdf(known_records.evidence[:, 0], id_class,
                                             n_prior=self.n_prior),
            "prototype": ClassConditionalCdf(known_records.evidence[:, 3], geo_class,
                                              n_prior=self.n_prior),
            "openmax": ClassConditionalCdf(known_records.evidence[:, 6], geo_class,
                                            n_prior=self.n_prior),
            "boundary": EmpiricalCdfCalibrator(known_prediction["score"]),
            "identity_prototype": ClassConditionalCdf(
                idproto, id_class, n_prior=self.n_prior),
            "geometry_raw": ClassConditionalCdf(
                view_values[:, 0], view_class[:, 0], n_prior=self.n_prior),
            "geometry_spectral": ClassConditionalCdf(
                view_values[:, 1], view_class[:, 1], n_prior=self.n_prior),
            "geometry_envelope_phase": ClassConditionalCdf(
                view_values[:, 2], view_class[:, 2], n_prior=self.n_prior),
            "geometry_difference_iq": ClassConditionalCdf(
                view_values[:, 3], view_class[:, 3], n_prior=self.n_prior),
            "geometry_complex_iq": ClassConditionalCdf(
                view_values[:, 4], view_class[:, 4], n_prior=self.n_prior),
        }
        known_components = self._components(known_records, known_prediction)
        self.rule_calibrator = EmpiricalCdfCalibrator(self._fuse(known_components))
        return self

    def transform(self, records, prediction) -> CalibrationOutput:
        if self.rule_calibrator is None:
            raise RuntimeError("CalibrationAgent must be fitted before transform")
        components = self._components(records, prediction)
        score = self.rule_calibrator.score(self._fuse(components))
        return CalibrationOutput(score, components, self.rule)


class DecisionArbitratorAgent:
    """Arbitrate local and message-informed Boundary Auditor decisions.

    Both branches are calibrated independently using known validation data.
    The arbitration output is then calibrated once more on the same known-only
    validation split.  No real or proxy unknown is required to fit this agent;
    leave-class-out evaluation is responsible for selecting the rule.
    """

    RULES = {"communication", "messages_off", "mean", "max", "noisy_or"}

    def __init__(self, evidence_rule: str, arbitration_rule: str):
        if arbitration_rule not in self.RULES:
            raise ValueError(f"unsupported arbitration rule {arbitration_rule}")
        self.evidence_rule = evidence_rule
        self.arbitration_rule = arbitration_rule
        self.communicated = CalibrationAgent(evidence_rule)
        self.local = CalibrationAgent(evidence_rule)
        self.final_calibrator: EmpiricalCdfCalibrator | None = None

    def _fuse(self, communicated: np.ndarray, local: np.ndarray) -> np.ndarray:
        if self.arbitration_rule == "communication":
            return communicated
        if self.arbitration_rule == "messages_off":
            return local
        if self.arbitration_rule == "mean":
            return 0.5 * (communicated + local)
        if self.arbitration_rule == "max":
            return np.maximum(communicated, local)
        return 1.0 - (1.0 - communicated) * (1.0 - local)

    def fit(self, known_records, communicated_prediction, local_prediction):
        self.communicated.fit(known_records, communicated_prediction)
        self.local.fit(known_records, local_prediction)
        communicated = self.communicated.transform(
            known_records, communicated_prediction).unknown_score
        local = self.local.transform(known_records, local_prediction).unknown_score
        self.final_calibrator = EmpiricalCdfCalibrator(self._fuse(communicated, local))
        return self

    def transform(self, records, communicated_prediction, local_prediction) -> ArbitrationOutput:
        if self.final_calibrator is None:
            raise RuntimeError("DecisionArbitratorAgent must be fitted before transform")
        communicated = self.communicated.transform(
            records, communicated_prediction).unknown_score
        local = self.local.transform(records, local_prediction).unknown_score
        score = self.final_calibrator.score(self._fuse(communicated, local))
        return ArbitrationOutput(score, local, communicated, self.arbitration_rule)


class ClassConditionalThresholdAgent:
    """Estimate per-predicted-class rejection thresholds from known validation.

    A class quantile is shrunk toward the global quantile according to its
    validation support.  This prevents an accidentally extreme threshold for
    a rare routed class while preserving adaptation for consistently hard or
    easy classes.
    """

    def __init__(self, known_acceptance: float, n_prior: float = 25.0):
        if not 0.0 < known_acceptance < 1.0:
            raise ValueError("known_acceptance must lie in (0, 1)")
        if n_prior < 0.0:
            raise ValueError("n_prior must be non-negative")
        self.known_acceptance = float(known_acceptance)
        self.n_prior = float(n_prior)
        self.global_threshold: float | None = None
        self.class_thresholds: dict[int, float] = {}
        self.class_counts: dict[int, int] = {}

    def fit(self, known_score: np.ndarray, predicted_class: np.ndarray):
        score = np.asarray(known_score, dtype=np.float64)
        prediction = np.asarray(predicted_class, dtype=np.int64)
        if score.ndim != 1 or prediction.shape != score.shape:
            raise ValueError("known_score and predicted_class must be matching 1-D arrays")
        self.global_threshold = float(np.quantile(score, self.known_acceptance))
        self.class_thresholds = {}
        self.class_counts = {}
        for cls in np.unique(prediction):
            values = score[prediction == cls]
            count = int(len(values))
            local = float(np.quantile(values, self.known_acceptance))
            weight = count / (count + self.n_prior) if count else 0.0
            self.class_thresholds[int(cls)] = (
                weight * local + (1.0 - weight) * self.global_threshold)
            self.class_counts[int(cls)] = count
        return self

    def thresholds(self, predicted_class: np.ndarray) -> np.ndarray:
        if self.global_threshold is None:
            raise RuntimeError("ClassConditionalThresholdAgent must be fitted before use")
        prediction = np.asarray(predicted_class, dtype=np.int64)
        return np.asarray([
            self.class_thresholds.get(int(cls), self.global_threshold)
            for cls in prediction
        ], dtype=np.float64)
