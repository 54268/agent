"""Registered, fail-closed decision gates for Stage-7 experiments.

The functions in this module deliberately consume plain Python mappings and
NumPy-compatible arrays.  They do not know about a model or a dataset and can
therefore be reused by the WiSig and ORACLE runners without dataset-specific
branches.

Later gates require *persisted, passing* results from every preceding gate.
Missing, malformed, non-finite, or failed prerequisite artifacts make the
current gate fail; they never silently turn into a warning.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
GATE_ORDER = ("g0", "g1", "g2", "g3")
LOCAL_ROLES = ("waveform", "prototype")

# This registry is the single source of truth for the Stage-7 protocol.
GATE_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "g0": {
        "outer_folds": 5,
        "local_known_accuracy": 0.85,
        "local_auroc": 0.75,
        "b2_max_deficit_vs_best_single_h": 0.005,
        "oracle_delta_h": 0.03,
        "oracle_delta_oscr": 0.015,
        "unique_rescue": 0.05,
        "public_selector_delta_h": 0.01,
    },
    "g1": {
        "outer_folds": 5,
        "delta_h": 0.01,
        "delta_oscr": 0.005,
        "max_known_accuracy_drop": 0.005,
        "positive_h_folds": 4,
        "rescue_harm_ratio": 2.0,
        "ablation_h_drop": 0.01,
        "seqcomm_delta_h": 0.005,
        "seqcomm_delta_oscr": 0.002,
    },
    "g2": {
        "outer_folds": 5,
        "minimum_query_rate": 0.10,
        "minimum_stop_rate": 0.10,
        "oracle_gain_recovery": 0.50,
        "gain_spearman": 0.30,
        "rescue_harm_ratio": 2.0,
        "direction_prune_rate": 0.05,
    },
    "g3": {
        "lco_folds": 15,
        "positive_lco_h_folds": 10,
        "formal_seeds": 3,
        "delta_h": 0.01,
        "delta_oscr": 0.005,
        "positive_h_seeds": 2,
        "delta_h_vs_best_single": 0.005,
        "messages_off_h_drop": 0.01,
        "rescue_harm_ratio": 2.0,
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _assert_finite_numbers(value: Any, location: str = "artifact") -> None:
    """Reject JSON's non-standard NaN/Infinity values recursively."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_numbers(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_numbers(item, f"{location}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"non-finite number in {location}")


@dataclass(frozen=True)
class GateCheck:
    """One auditable boolean condition in a gate result."""

    name: str
    passed: bool
    observed: Any
    operator: str
    threshold: Any
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "observed": _jsonable(self.observed),
            "operator": self.operator,
            "threshold": _jsonable(self.threshold),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateResult:
    """Deterministic result persisted between registered experiment phases."""

    gate: str
    passed: bool
    checks: tuple[GateCheck, ...]
    measurements: Mapping[str, Any]
    claims: Mapping[str, bool]
    recommendations: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            check.detail or check.name for check in self.checks if not check.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "gate": self.gate,
            "passed": bool(self.passed),
            "checks": [check.to_dict() for check in self.checks],
            "measurements": _jsonable(self.measurements),
            "claims": {str(key): bool(value) for key, value in self.claims.items()},
            "recommendations": list(self.recommendations),
            "failure_reasons": list(self.failure_reasons),
        }


def gate_result_from_dict(payload: Mapping[str, Any]) -> GateResult:
    """Validate and reconstruct a persisted result.

    ``passed`` is recomputed from the checks.  A payload that claims to pass
    while containing a failed check is rejected as corrupt rather than being
    trusted as a prerequisite.
    """

    _assert_finite_numbers(payload)
    required = {"schema_version", "gate", "passed", "checks", "measurements"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"gate artifact is missing fields: {sorted(missing)}")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported gate artifact schema")
    gate = str(payload["gate"]).lower()
    if gate not in GATE_ORDER:
        raise ValueError(f"unknown gate '{gate}'")
    raw_checks = payload["checks"]
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError("gate artifact must contain non-empty checks")
    checks = []
    for row in raw_checks:
        if not isinstance(row, Mapping):
            raise ValueError("invalid check row in gate artifact")
        for key in ("name", "passed", "observed", "operator", "threshold"):
            if key not in row:
                raise ValueError(f"gate check is missing '{key}'")
        checks.append(GateCheck(
            name=str(row["name"]),
            passed=bool(row["passed"]),
            observed=row["observed"],
            operator=str(row["operator"]),
            threshold=row["threshold"],
            detail=str(row.get("detail", "")),
        ))
    computed_passed = all(check.passed for check in checks)
    if bool(payload["passed"]) != computed_passed:
        raise ValueError("gate artifact pass flag disagrees with its checks")
    measurements = payload["measurements"]
    if not isinstance(measurements, Mapping):
        raise ValueError("gate measurements must be a mapping")
    claims = payload.get("claims", {})
    if not isinstance(claims, Mapping):
        raise ValueError("gate claims must be a mapping")
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendations, list):
        raise ValueError("gate recommendations must be a list")
    return GateResult(
        gate=gate,
        passed=computed_passed,
        checks=tuple(checks),
        measurements=dict(measurements),
        claims={str(key): bool(value) for key, value in claims.items()},
        recommendations=tuple(str(value) for value in recommendations),
    )


def save_gate_result(path: str | Path, result: GateResult) -> Path:
    """Persist a stable JSON prerequisite artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def load_gate_result(
    path: str | Path,
    *,
    expected_gate: str | None = None,
    require_passed: bool = False,
) -> GateResult:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = gate_result_from_dict(payload)
    if expected_gate is not None and result.gate != expected_gate.lower():
        raise ValueError(
            f"expected {expected_gate.lower()} artifact, found {result.gate}")
    if require_passed and not result.passed:
        raise ValueError(f"prerequisite {result.gate} did not pass")
    return result


def _coerce_artifact(value: Any) -> GateResult:
    if isinstance(value, GateResult):
        # Revalidate even in-memory results so callers cannot bypass the
        # persisted-artifact invariants with an inconsistent dataclass.
        return gate_result_from_dict(value.to_dict())
    if isinstance(value, (str, Path)):
        return load_gate_result(value)
    if isinstance(value, Mapping):
        return gate_result_from_dict(value)
    raise TypeError(f"unsupported prerequisite artifact type: {type(value).__name__}")


def _prerequisite_checks(
    gate: str,
    prerequisites: Mapping[str, Any] | Sequence[Any] | GateResult | str | Path | None,
) -> list[GateCheck]:
    required = GATE_ORDER[:GATE_ORDER.index(gate)]
    if not required:
        return []
    discovered: dict[str, GateResult] = {}
    errors: list[str] = []
    if prerequisites is None:
        errors.append("no prerequisite artifacts supplied")
    else:
        if isinstance(prerequisites, (GateResult, str, Path)):
            values = (prerequisites,)
        elif isinstance(prerequisites, Mapping):
            # A serialized GateResult is itself a mapping; otherwise values are
            # interpreted as a gate-name -> artifact registry.
            if "gate" in prerequisites and "checks" in prerequisites:
                values: Iterable[Any] = (prerequisites,)
            else:
                values = prerequisites.values()
        else:
            values = prerequisites
        for value in values:
            try:
                artifact = _coerce_artifact(value)
                if artifact.gate in discovered:
                    errors.append(f"duplicate {artifact.gate} artifact")
                discovered[artifact.gate] = artifact
            except Exception as exc:  # fail closed on any malformed artifact
                errors.append(str(exc))
    checks = []
    for name in required:
        artifact = discovered.get(name)
        passed = artifact is not None and artifact.passed
        detail = ""
        if artifact is None:
            detail = f"missing persisted {name} prerequisite"
        elif not artifact.passed:
            detail = f"persisted {name} prerequisite did not pass"
        checks.append(GateCheck(
            name=f"prerequisite_{name}", passed=passed,
            observed=None if artifact is None else artifact.passed,
            operator="is", threshold=True, detail=detail,
        ))
    if errors:
        checks.append(GateCheck(
            name="prerequisite_artifacts_well_formed", passed=False,
            observed=errors, operator="is", threshold="valid",
            detail="; ".join(errors),
        ))
    return checks


def require_prerequisite_artifacts(
    gate: str,
    prerequisites: Mapping[str, Any] | Sequence[Any] | GateResult | str | Path | None,
) -> None:
    """Raise before an expensive phase when prerequisite evidence is absent."""

    gate = gate.lower()
    if gate not in GATE_ORDER:
        raise ValueError(f"unknown gate '{gate}'")
    failed = [row for row in _prerequisite_checks(gate, prerequisites) if not row.passed]
    if failed:
        raise RuntimeError("; ".join(row.detail or row.name for row in failed))


def _array(
    metrics: Mapping[str, Any],
    key: str,
    checks: list[GateCheck],
    *,
    length: int | None = None,
) -> np.ndarray | None:
    if key not in metrics:
        checks.append(GateCheck(
            f"input_{key}", False, None, "present", True,
            f"missing metric '{key}'"))
        return None
    try:
        value = np.asarray(metrics[key], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        checks.append(GateCheck(
            f"input_{key}", False, repr(metrics[key]), "numeric", True,
            f"metric '{key}' is not numeric: {exc}"))
        return None
    valid = bool(len(value)) and bool(np.isfinite(value).all())
    if length is not None:
        valid = valid and len(value) == length
    checks.append(GateCheck(
        f"input_{key}", valid,
        {"count": int(len(value)), "finite": bool(np.isfinite(value).all())},
        "valid_vector", {"length": length},
        "" if valid else f"metric '{key}' must be finite with length {length}",
    ))
    return value if valid else None


def _role_arrays(
    metrics: Mapping[str, Any],
    key: str,
    checks: list[GateCheck],
    *,
    length: int,
) -> dict[str, np.ndarray] | None:
    value = metrics.get(key)
    if not isinstance(value, Mapping):
        checks.append(GateCheck(
            f"input_{key}", False, None, "mapping", list(LOCAL_ROLES),
            f"metric '{key}' must map both local roles to fold vectors"))
        return None
    result = {}
    for role in LOCAL_ROLES:
        nested_checks: list[GateCheck] = []
        nested = _array(value, role, nested_checks, length=length)
        checks.extend(GateCheck(
            name=f"input_{key}_{role}", passed=row.passed,
            observed=row.observed, operator=row.operator,
            threshold=row.threshold, detail=row.detail,
        ) for row in nested_checks)
        if nested is not None:
            result[role] = nested
    return result if set(result) == set(LOCAL_ROLES) else None


def _boolean(
    metrics: Mapping[str, Any], key: str, checks: list[GateCheck]
) -> bool | None:
    value = metrics.get(key)
    valid = isinstance(value, (bool, np.bool_))
    checks.append(GateCheck(
        f"input_{key}", valid, value, "boolean", True,
        "" if valid else f"missing or non-boolean protocol flag '{key}'"))
    return bool(value) if valid else None


def _condition(
    checks: list[GateCheck], name: str, passed: bool,
    observed: Any, operator: str, threshold: Any, detail: str,
) -> None:
    checks.append(GateCheck(
        name, bool(passed), _jsonable(observed), operator,
        _jsonable(threshold), "" if passed else detail,
    ))


def _finish(
    gate: str,
    checks: list[GateCheck],
    measurements: Mapping[str, Any],
    *,
    claims: Mapping[str, bool] | None = None,
    recommendations: Sequence[str] = (),
) -> GateResult:
    return GateResult(
        gate=gate,
        passed=bool(checks) and all(check.passed for check in checks),
        checks=tuple(checks),
        measurements=dict(measurements),
        claims=dict(claims or {}),
        recommendations=tuple(recommendations),
    )


def evaluate_g0(metrics: Mapping[str, Any]) -> GateResult:
    """Evaluate five-fold local capability and exploitable complementarity."""

    threshold = GATE_THRESHOLDS["g0"]
    folds = int(threshold["outer_folds"])
    checks: list[GateCheck] = []
    measurements: dict[str, Any] = {}
    nested_oof = _boolean(metrics, "nested_inner_oof_selector", checks)
    outer_selection = _boolean(metrics, "outer_test_used_for_selection", checks)
    public_transfer = _boolean(metrics, "public_only_b2_transfer", checks)
    formal_unknown = _boolean(metrics, "formal_unknown_used", checks)
    if nested_oof is not None:
        _condition(
            checks, "nested_inner_oof_selector", nested_oof, nested_oof,
            "is", True,
            "outer-test row cross-fitting is diagnostic and cannot register G0")
    if outer_selection is not None:
        _condition(
            checks, "outer_test_not_used_for_selection", not outer_selection,
            outer_selection, "is", False,
            "outer test labels entered G0 fitting or selection")
    if public_transfer is not None:
        _condition(
            checks, "public_only_b2_transfer", public_transfer,
            public_transfer, "is", True,
            "inner-to-outer transfer was not restricted to the public B2")
    if formal_unknown is not None:
        _condition(
            checks, "formal_unknown_not_used", not formal_unknown,
            formal_unknown, "is", False,
            "formal Unknown entered G0")
    local_acc = _role_arrays(metrics, "local_known_accuracy", checks, length=folds)
    local_auroc = _role_arrays(metrics, "local_auroc", checks, length=folds)
    single_h = _role_arrays(metrics, "single_h_score", checks, length=folds)
    rescue = _role_arrays(metrics, "unique_rescue", checks, length=folds)
    b2_h = _array(metrics, "b2_h_score", checks, length=folds)
    b2_oscr = _array(metrics, "b2_oscr", checks, length=folds)
    oracle_h = _array(metrics, "oracle_h_score", checks, length=folds)
    oracle_oscr = _array(metrics, "oracle_oscr", checks, length=folds)
    selector_h = _array(metrics, "public_selector_h_score", checks, length=folds)

    if local_acc is not None:
        for role, values in local_acc.items():
            observed = float(values.mean())
            measurements[f"{role}_known_accuracy_mean"] = observed
            _condition(
                checks, f"{role}_known_accuracy_ge_0_85",
                observed >= threshold["local_known_accuracy"], observed, ">=",
                threshold["local_known_accuracy"],
                f"{role} mean Known Accuracy is below 0.85")
    if local_auroc is not None:
        for role, values in local_auroc.items():
            observed = float(values.mean())
            measurements[f"{role}_auroc_mean"] = observed
            _condition(
                checks, f"{role}_auroc_ge_0_75",
                observed >= threshold["local_auroc"], observed, ">=",
                threshold["local_auroc"], f"{role} mean AUROC is below 0.75")
    if single_h is not None:
        best_single = np.maximum(single_h["waveform"], single_h["prototype"])
        measurements["best_single_h_mean"] = float(best_single.mean())
    else:
        best_single = None
    if b2_h is not None:
        measurements["b2_h_mean"] = float(b2_h.mean())
    if b2_oscr is not None:
        measurements["b2_oscr_mean"] = float(b2_oscr.mean())
    if best_single is not None and b2_h is not None:
        delta = float((b2_h - best_single).mean())
        measurements["b2_delta_h_vs_best_single"] = delta
        floor = -float(threshold["b2_max_deficit_vs_best_single_h"])
        _condition(
            checks, "b2_not_worse_than_best_single_by_0_005",
            delta >= floor, delta, ">=", floor,
            "fair B2 loses more than 0.005 H-score to the best local Agent")
    if oracle_h is not None and b2_h is not None:
        delta = float((oracle_h - b2_h).mean())
        measurements["oracle_delta_h_vs_b2"] = delta
        _condition(
            checks, "oracle_delta_h_ge_0_03",
            delta >= threshold["oracle_delta_h"], delta, ">=",
            threshold["oracle_delta_h"],
            "per-sample Agent oracle has insufficient H-score headroom")
    if oracle_oscr is not None and b2_oscr is not None:
        delta = float((oracle_oscr - b2_oscr).mean())
        measurements["oracle_delta_oscr_vs_b2"] = delta
        _condition(
            checks, "oracle_delta_oscr_ge_0_015",
            delta >= threshold["oracle_delta_oscr"], delta, ">=",
            threshold["oracle_delta_oscr"],
            "per-sample Agent oracle has insufficient OSCR headroom")
    if rescue is not None:
        for role, values in rescue.items():
            observed = float(values.mean())
            measurements[f"{role}_unique_rescue_mean"] = observed
            _condition(
                checks, f"{role}_unique_rescue_ge_0_05",
                observed >= threshold["unique_rescue"], observed, ">=",
                threshold["unique_rescue"],
                f"{role} does not uniquely rescue at least 5% of samples")
    if selector_h is not None and best_single is not None:
        delta = float((selector_h - best_single).mean())
        measurements["public_selector_delta_h_vs_best_single"] = delta
        _condition(
            checks, "public_selector_delta_h_ge_0_01",
            delta >= threshold["public_selector_delta_h"], delta, ">=",
            threshold["public_selector_delta_h"],
            "cross-fitted public selector does not expose usable complementarity")
    return _finish("g0", checks, measurements, claims={
        "nested_inner_oof_selector": bool(nested_oof),
        "outer_test_unseen_during_selection": outer_selection is False,
        "public_only_b2_transfer": bool(public_transfer),
        "formal_unknown_excluded": formal_unknown is False,
    })


def evaluate_g1(
    metrics: Mapping[str, Any],
    *,
    prerequisites: Mapping[str, Any] | Sequence[Any] | GateResult | str | Path | None,
) -> GateResult:
    """Evaluate whether frozen semantic messages themselves add value."""

    threshold = GATE_THRESHOLDS["g1"]
    folds = int(threshold["outer_folds"])
    checks = _prerequisite_checks("g1", prerequisites)
    measurements: dict[str, Any] = {}
    same_agents = _boolean(metrics, "frozen_agents_identical", checks)
    equal_bandwidth = _boolean(metrics, "parallel_equal_bandwidth", checks)
    if same_agents is not None:
        _condition(checks, "frozen_agents_identical", same_agents, same_agents,
                   "is", True, "G1 systems did not reuse identical frozen Agents")
    if equal_bandwidth is not None:
        _condition(checks, "parallel_equal_bandwidth", equal_bandwidth,
                   equal_bandwidth, "is", True,
                   "parallel and sequential controls are not bandwidth matched")

    b2_h = _array(metrics, "b2_h_score", checks, length=folds)
    seq_h = _array(metrics, "sequential_h_score", checks, length=folds)
    parallel_h = _array(metrics, "parallel_h_score", checks, length=folds)
    b2_oscr = _array(metrics, "b2_oscr", checks, length=folds)
    seq_oscr = _array(metrics, "sequential_oscr", checks, length=folds)
    parallel_oscr = _array(metrics, "parallel_oscr", checks, length=folds)
    b2_known = _array(metrics, "b2_known_accuracy", checks, length=folds)
    seq_known = _array(metrics, "sequential_known_accuracy", checks, length=folds)
    ratio = _array(metrics, "rescue_harm_ratio", checks, length=folds)
    receiver_ci = _array(metrics, "receiver_gain_ci_low", checks, length=1)
    ablations = metrics.get("ablation_h_drop")
    if not isinstance(ablations, Mapping) or not ablations:
        checks.append(GateCheck(
            "input_ablation_h_drop", False, None, "non_empty_mapping", True,
            "causal message-ablation H-score drops are missing"))
        ablation_means: dict[str, float] = {}
    else:
        ablation_means = {}
        for name, value in sorted(ablations.items()):
            nested_checks: list[GateCheck] = []
            values = _array({str(name): value}, str(name), nested_checks, length=folds)
            checks.extend(GateCheck(
                name=f"input_ablation_{row.name.removeprefix('input_')}",
                passed=row.passed, observed=row.observed, operator=row.operator,
                threshold=row.threshold, detail=row.detail,
            ) for row in nested_checks)
            if values is not None:
                ablation_means[str(name)] = float(values.mean())

    if b2_h is not None and seq_h is not None:
        delta_h = seq_h - b2_h
        measurements["delta_h_mean"] = float(delta_h.mean())
        measurements["positive_h_folds"] = int((delta_h > 0).sum())
        _condition(
            checks, "delta_h_ge_0_01", delta_h.mean() >= threshold["delta_h"],
            float(delta_h.mean()), ">=", threshold["delta_h"],
            "sequential communication does not improve mean H-score by 0.01")
        _condition(
            checks, "positive_h_folds_ge_4",
            int((delta_h > 0).sum()) >= threshold["positive_h_folds"],
            int((delta_h > 0).sum()), ">=", threshold["positive_h_folds"],
            "fewer than four of five folds have positive H-score gain")
    if b2_oscr is not None and seq_oscr is not None:
        delta = float((seq_oscr - b2_oscr).mean())
        measurements["delta_oscr_mean"] = delta
        _condition(
            checks, "delta_oscr_ge_0_005", delta >= threshold["delta_oscr"],
            delta, ">=", threshold["delta_oscr"],
            "sequential communication does not improve mean OSCR by 0.005")
    if b2_known is not None and seq_known is not None:
        delta = float((seq_known - b2_known).mean())
        measurements["known_accuracy_delta"] = delta
        floor = -float(threshold["max_known_accuracy_drop"])
        _condition(
            checks, "known_accuracy_drop_le_0_005", delta >= floor,
            delta, ">=", floor,
            "sequential communication reduces Known Accuracy by more than 0.005")
    if ratio is not None:
        observed = float(ratio.mean())
        measurements["rescue_harm_ratio_mean"] = observed
        _condition(
            checks, "rescue_harm_ratio_ge_2", observed >= threshold["rescue_harm_ratio"],
            observed, ">=", threshold["rescue_harm_ratio"],
            "queried hard-subset rescue/harm ratio is below 2")
    if receiver_ci is not None:
        observed = float(receiver_ci[0])
        measurements["receiver_gain_ci_low"] = observed
        _condition(
            checks, "receiver_gain_ci_low_gt_0", observed > 0.0, observed,
            ">", 0.0, "receiver post-message loss is not significantly lower")
    if ablation_means:
        measurements["ablation_h_drop"] = ablation_means
        best_name, best_drop = max(ablation_means.items(), key=lambda row: (row[1], row[0]))
        measurements["max_ablation_h_drop"] = best_drop
        measurements["max_ablation"] = best_name
        _condition(
            checks, "one_ablation_h_drop_ge_0_01",
            best_drop >= threshold["ablation_h_drop"], best_drop, ">=",
            threshold["ablation_h_drop"],
            "no message intervention causes a 0.01 H-score drop")

    seqcomm_supported = False
    if all(value is not None
           for value in (parallel_h, seq_h, parallel_oscr, seq_oscr)):
        seq_delta_h = float((seq_h - parallel_h).mean())
        seq_delta_oscr = float((seq_oscr - parallel_oscr).mean())
        measurements["sequential_delta_h_vs_parallel"] = seq_delta_h
        measurements["sequential_delta_oscr_vs_parallel"] = seq_delta_oscr
        seqcomm_supported = (
            seq_delta_h >= threshold["seqcomm_delta_h"]
            or seq_delta_oscr >= threshold["seqcomm_delta_oscr"]
        )
    recommendations = () if seqcomm_supported else (
        "Do not claim a SeqComm-style ordering contribution; retain only semantic consultation.",
    )
    return _finish(
        "g1", checks, measurements,
        claims={"sequential_ordering_supported": seqcomm_supported},
        recommendations=recommendations,
    )


def evaluate_g2(
    metrics: Mapping[str, Any],
    *,
    prerequisites: Mapping[str, Any] | Sequence[Any] | GateResult | str | Path | None,
) -> GateResult:
    """Evaluate learned value-of-information routing after G1 has passed."""

    threshold = GATE_THRESHOLDS["g2"]
    folds = int(threshold["outer_folds"])
    checks = _prerequisite_checks("g2", prerequisites)
    measurements: dict[str, Any] = {}
    exact = _boolean(metrics, "exact_action_enumeration", checks)
    if exact is not None:
        _condition(checks, "exact_action_enumeration", exact, exact, "is", True,
                   "router targets were not produced by exact action enumeration")
    query = _array(metrics, "query_rate", checks, length=folds)
    recovery = _array(metrics, "oracle_gain_recovery", checks, length=folds)
    correlation = _array(metrics, "gain_spearman", checks, length=folds)
    ratio = _array(metrics, "rescue_harm_ratio", checks, length=folds)
    if query is not None:
        observed = float(query.mean())
        stop = 1.0 - observed
        measurements.update(query_rate_mean=observed, stop_rate_mean=stop)
        _condition(
            checks, "query_rate_ge_0_10",
            observed >= threshold["minimum_query_rate"], observed, ">=",
            threshold["minimum_query_rate"], "query action coverage is below 10%")
        _condition(
            checks, "stop_rate_ge_0_10",
            stop >= threshold["minimum_stop_rate"], stop, ">=",
            threshold["minimum_stop_rate"], "STOP action coverage is below 10%")
    if recovery is not None:
        observed = float(recovery.mean())
        measurements["oracle_gain_recovery_mean"] = observed
        _condition(
            checks, "oracle_gain_recovery_ge_0_50",
            observed >= threshold["oracle_gain_recovery"], observed, ">=",
            threshold["oracle_gain_recovery"],
            "learned routing recovers less than 50% of exact-oracle gain")
    if correlation is not None:
        observed = float(correlation.mean())
        measurements["gain_spearman_mean"] = observed
        _condition(
            checks, "gain_spearman_ge_0_30",
            observed >= threshold["gain_spearman"], observed, ">=",
            threshold["gain_spearman"],
            "predicted and counterfactual communication gains correlate below 0.30")
    if ratio is not None:
        observed = float(ratio.mean())
        measurements["rescue_harm_ratio_mean"] = observed
        _condition(
            checks, "rescue_harm_ratio_ge_2",
            observed >= threshold["rescue_harm_ratio"], observed, ">=",
            threshold["rescue_harm_ratio"],
            "routed hard-subset rescue/harm ratio is below 2")

    recommendations: list[str] = []
    direction_rates = metrics.get("direction_rates_by_dataset")
    if direction_rates is not None:
        if not isinstance(direction_rates, Mapping) or len(direction_rates) < 2:
            checks.append(GateCheck(
                "input_direction_rates_by_dataset", False, direction_rates,
                "mapping", "at least two datasets",
                "direction pruning requires both WiSig and ORACLE evidence"))
        else:
            clean: dict[str, dict[str, float]] = {}
            valid = True
            for dataset, rates in sorted(direction_rates.items()):
                if not isinstance(rates, Mapping):
                    valid = False
                    continue
                clean[str(dataset)] = {}
                for action in ("w_first", "p_first"):
                    try:
                        value = float(rates[action])
                    except (KeyError, TypeError, ValueError):
                        valid = False
                        continue
                    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                        valid = False
                    clean[str(dataset)][action] = value
            checks.append(GateCheck(
                "input_direction_rates_by_dataset", valid, clean,
                "valid_rates", True,
                "" if valid else "invalid direction rates for cross-dataset pruning"))
            measurements["direction_rates_by_dataset"] = clean
            if valid:
                for action in ("w_first", "p_first"):
                    if all(row[action] < threshold["direction_prune_rate"]
                           for row in clean.values()):
                        recommendations.append(
                            f"Prune {action.upper()}: usage is below 5% on every dataset.")
    return _finish(
        "g2", checks, measurements,
        recommendations=recommendations,
    )


def evaluate_g3(
    metrics: Mapping[str, Any],
    *,
    prerequisites: Mapping[str, Any] | Sequence[Any] | GateResult | str | Path | None,
) -> GateResult:
    """Evaluate 15-fold replication and the frozen three-seed formal test."""

    threshold = GATE_THRESHOLDS["g3"]
    checks = _prerequisite_checks("g3", prerequisites)
    measurements: dict[str, Any] = {}
    lco = _array(metrics, "lco_h_delta", checks, length=int(threshold["lco_folds"]))
    seeds = metrics.get("formal_seeds")
    seed_valid = False
    try:
        seed_values = [int(value) for value in seeds]
        seed_valid = seed_values == [42, 43, 44]
    except (TypeError, ValueError):
        seed_values = []
    checks.append(GateCheck(
        "formal_seeds_are_42_43_44", seed_valid, seed_values, "equals",
        [42, 43, 44], "formal evaluation must use frozen seeds 42/43/44"))

    count = int(threshold["formal_seeds"])
    full_h = _array(metrics, "formal_h_score", checks, length=count)
    full_oscr = _array(metrics, "formal_oscr", checks, length=count)
    b2_h = _array(metrics, "b2_h_score", checks, length=count)
    b2_oscr = _array(metrics, "b2_oscr", checks, length=count)
    single_h = _array(metrics, "best_single_h_score", checks, length=count)
    messages_off_h = _array(metrics, "messages_off_h_score", checks, length=count)
    ratio = _array(metrics, "rescue_harm_ratio", checks, length=count)
    h_ci = _array(metrics, "bootstrap_h_ci_low", checks, length=1)
    oscr_ci = _array(metrics, "bootstrap_oscr_ci_low", checks, length=1)

    if lco is not None:
        positive = int((lco > 0).sum())
        measurements["positive_lco_h_folds"] = positive
        measurements["lco_h_delta_mean"] = float(lco.mean())
        _condition(
            checks, "positive_lco_h_folds_ge_10",
            positive >= threshold["positive_lco_h_folds"], positive, ">=",
            threshold["positive_lco_h_folds"],
            "fewer than 10 of 15 LCO folds have positive H-score gain")
    if full_h is not None and b2_h is not None:
        delta = full_h - b2_h
        measurements["formal_delta_h_mean"] = float(delta.mean())
        measurements["positive_h_seeds"] = int((delta > 0).sum())
        _condition(
            checks, "formal_delta_h_ge_0_01",
            delta.mean() >= threshold["delta_h"], float(delta.mean()), ">=",
            threshold["delta_h"],
            "formal collaboration mean H-score gain is below 0.01")
        _condition(
            checks, "positive_h_seeds_ge_2",
            int((delta > 0).sum()) >= threshold["positive_h_seeds"],
            int((delta > 0).sum()), ">=", threshold["positive_h_seeds"],
            "fewer than two formal seeds have positive H-score gain")
    if full_oscr is not None and b2_oscr is not None:
        delta = float((full_oscr - b2_oscr).mean())
        measurements["formal_delta_oscr_mean"] = delta
        _condition(
            checks, "formal_delta_oscr_ge_0_005",
            delta >= threshold["delta_oscr"], delta, ">=",
            threshold["delta_oscr"],
            "formal collaboration mean OSCR gain is below 0.005")
    if full_h is not None and single_h is not None:
        delta = float((full_h - single_h).mean())
        measurements["formal_delta_h_vs_best_single"] = delta
        _condition(
            checks, "formal_beats_best_single_h_by_0_005",
            delta >= threshold["delta_h_vs_best_single"], delta, ">=",
            threshold["delta_h_vs_best_single"],
            "formal collaboration does not beat the best local Agent by 0.005 H-score")
    if full_h is not None and messages_off_h is not None:
        delta = float((full_h - messages_off_h).mean())
        measurements["messages_off_h_drop_mean"] = delta
        _condition(
            checks, "messages_off_h_drop_ge_0_01",
            delta >= threshold["messages_off_h_drop"], delta, ">=",
            threshold["messages_off_h_drop"],
            "turning messages off does not reduce H-score by 0.01")
    if ratio is not None:
        observed = float(ratio.mean())
        measurements["rescue_harm_ratio_mean"] = observed
        _condition(
            checks, "rescue_harm_ratio_ge_2",
            observed >= threshold["rescue_harm_ratio"], observed, ">=",
            threshold["rescue_harm_ratio"],
            "formal hard-subset rescue/harm ratio is below 2")
    if h_ci is not None and oscr_ci is not None:
        h_value, oscr_value = float(h_ci[0]), float(oscr_ci[0])
        measurements["bootstrap_h_ci_low"] = h_value
        measurements["bootstrap_oscr_ci_low"] = oscr_value
        _condition(
            checks, "paired_h_or_oscr_ci_low_gt_0",
            h_value > 0.0 or oscr_value > 0.0,
            {"h": h_value, "oscr": oscr_value}, "any >", 0.0,
            "neither paired H nor OSCR bootstrap interval excludes zero")
    return _finish("g3", checks, measurements)


def select_passing_candidate(
    candidates: Mapping[str, GateResult],
    *,
    ranking: Sequence[tuple[str, str]] = (
        ("delta_h_mean", "max"), ("delta_oscr_mean", "max")),
) -> dict[str, Any]:
    """Deterministically select among passing candidates.

    Missing or non-finite ranking fields disqualify a candidate.  Lexicographic
    candidate name is the final tie-break, independent of mapping insertion
    order.
    """

    for field, direction in ranking:
        if direction not in {"max", "min"}:
            raise ValueError(f"invalid ranking direction for {field}: {direction}")
    eligible: dict[str, tuple[float, ...]] = {}
    rejected: dict[str, str] = {}
    for name in sorted(candidates):
        result = candidates[name]
        if not isinstance(result, GateResult):
            rejected[name] = "not a GateResult"
            continue
        if not result.passed:
            rejected[name] = "gate failed"
            continue
        score = []
        for field, direction in ranking:
            try:
                value = float(result.measurements[field])
            except (KeyError, TypeError, ValueError):
                rejected[name] = f"missing numeric ranking field '{field}'"
                break
            if not np.isfinite(value):
                rejected[name] = f"non-finite ranking field '{field}'"
                break
            score.append(value if direction == "max" else -value)
        else:
            eligible[name] = tuple(score)
    # max score, then lexicographically smallest name.
    selected = None
    if eligible:
        best_score = max(eligible.values())
        selected = min(name for name, score in eligible.items() if score == best_score)
    return {
        "selected": selected,
        "ranking": [[field, direction] for field, direction in ranking],
        "eligible": {name: list(eligible[name]) for name in sorted(eligible)},
        "rejected": rejected,
    }


def render_gate_report(result: GateResult) -> str:
    """Render a compact human-auditable Markdown gate report."""

    lines = [
        f"# Stage-7 {result.gate.upper()} gate",
        "",
        f"Overall: **{'PASS' if result.passed else 'FAIL'}**",
        "",
        "| Check | Result | Observed | Requirement |",
        "|---|---:|---:|---:|",
    ]
    for check in result.checks:
        observed = json.dumps(_jsonable(check.observed), ensure_ascii=False, sort_keys=True)
        required = f"{check.operator} " + json.dumps(
            _jsonable(check.threshold), ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} | "
            f"{observed} | {required} |")
    if result.recommendations:
        lines += ["", "## Recommendations", ""]
        lines.extend(f"- {value}" for value in result.recommendations)
    return "\n".join(lines) + "\n"


def save_gate_report(directory: str | Path, result: GateResult) -> dict[str, str]:
    """Write paired JSON/Markdown artifacts using stable filenames."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = save_gate_result(directory / f"{result.gate}_gate.json", result)
    markdown_path = directory / f"{result.gate}_gate.md"
    markdown_path.write_text(render_gate_report(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "GATE_ORDER", "GATE_THRESHOLDS", "GateCheck", "GateResult",
    "evaluate_g0", "evaluate_g1", "evaluate_g2", "evaluate_g3",
    "gate_result_from_dict", "load_gate_result", "render_gate_report",
    "require_prerequisite_artifacts", "save_gate_report", "save_gate_result",
    "select_passing_candidate",
]
