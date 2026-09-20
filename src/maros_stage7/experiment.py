"""Guarded Stage-7 experiment orchestration.

The runner deliberately spends computation in increasing order of risk:

``sanity`` -> one outer fold per dataset, local Agents and fair B2 only;
``g0``     -> the registered five-fold complementarity audit;
``g1``     -> frozen-Agent message-value audit;
``g2``     -> exact-counterfactual route imitation;
``g3``     -> multi-partition replication;
``formal`` -> the only phase allowed to read the dataset's formal Unknown.

Later phases fail closed when their persisted prerequisite gate did not pass.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from maros_staged.datasets import load_oracle_npz, load_wisig_subset

from .contracts import RouteAction
from .gates import (evaluate_g0, load_gate_result, save_gate_report)
from .metrics import (ClasswiseCalibrationService, complementarity_report,
                      evaluate_open_set)
from .nested_b2 import (PublicDecisionTable, collect_public_decision_table,
                        evaluate_public_b2_table, fit_nested_public_b2,
                        public_local_action_losses)
from .nested_g0 import (SelectorRowBatch, build_inner_registration_partitions,
                        inner_oof_selector_evidence)
from .model import Stage7System
from .splits import (ProvenanceSubset, build_nested_lco_protocol,
                     protocol_manifest)
from .training import (CompetitionPUGDataset, assert_stop_matches_b2,
                       collect_predictions, cross_fitted_public_selector,
                       import_b2_transfer_bundle, make_competition_pug,
                       refresh_enrollment, set_seed, train_fair_b2,
                       train_local_agents)


LOCAL_ROLES = ("waveform", "prototype")
METRIC_NAMES = ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")
CHECKPOINT_SCHEMA_VERSION = 4
_CHECKPOINT_CONFIG_METADATA = {
    "name", "output_dir", "paired_config", "comparison_target", "config_path",
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _checkpoint_config_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration fields that must remain fixed when resuming a fold."""

    return {
        str(key): _jsonable(value) for key, value in cfg.items()
        if key not in _CHECKPOINT_CONFIG_METADATA
    }


def _write_or_validate_frozen_config(path: Path, cfg: Mapping[str, Any]) -> None:
    """Create the run contract once and reject accidental in-place mutation."""

    expected = _jsonable(dict(cfg))
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            keys = sorted(
                key for key in set(observed) | set(expected)
                if observed.get(key) != expected.get(key))
            raise ValueError(
                "Stage-7 output directory already has a different frozen "
                f"configuration (changed fields: {', '.join(keys)})")
        return
    _write_json(path, expected)


def load_splits(cfg: Mapping[str, Any]):
    dataset = str(cfg.get("dataset", "")).lower()
    augment = bool(cfg.get("augment_train", False))
    if dataset == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=augment)
    if dataset == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=augment)
    raise ValueError(f"unsupported Stage-7 dataset: {dataset}")


def _system(cfg: Mapping[str, Any], num_classes: int) -> Stage7System:
    return Stage7System(
        num_classes,
        state_dim=int(cfg.get("state_dim", 128)),
        stem_channels=int(cfg.get("stem_channels", 128)),
        class_embed_dim=int(cfg.get("class_embed_dim", 24)),
        complex_hidden=int(cfg.get("complex_hidden", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 192)),
        prototype_temperature=float(cfg.get("prototype_temperature", 0.15)),
        b2_max_adaptive_mass=float(cfg.get("b2_max_adaptive_mass", 0.35)),
        b2_initial_adaptive_fraction=float(
            cfg.get("b2_initial_adaptive_fraction", 0.2)),
    )


def _balanced_limit(dataset: ProvenanceSubset, max_per_class: int,
                    seed: int) -> ProvenanceSubset:
    """Cap a provenance subset without losing leakage metadata."""
    if max_per_class <= 0:
        return dataset
    labels = np.asarray(dataset.y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    positions: list[int] = []
    for label in np.unique(labels):
        choices = np.flatnonzero(labels == label)
        if len(choices) > max_per_class:
            choices = np.sort(rng.choice(choices, max_per_class, replace=False))
        positions.extend(int(value) for value in choices)
    positions = sorted(positions)
    absolute = dataset.indices[np.asarray(positions, dtype=np.int64)]
    return ProvenanceSubset(
        dataset.dataset, absolute, source_split=dataset.source_split,
        purpose=dataset.purpose, class_mapping=dataset.class_mapping,
        force_unknown=dataset.force_unknown,
        formal_unknown=dataset.formal_unknown,
    )


def _limit_pug(dataset: CompetitionPUGDataset, maximum: int,
               seed: int) -> CompetitionPUGDataset:
    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), maximum, replace=False))
    return CompetitionPUGDataset(
        dataset.x[indices], dataset.source_indices[indices],
        dataset.source_classes[indices])


def _set_agent_trainability(system: Stage7System, mode: str) -> None:
    for parameter in system.agents.parameters():
        parameter.requires_grad_(False)
    if mode == "all":
        for parameter in system.agents.parameters():
            parameter.requires_grad_(True)
    elif mode == "open_heads":
        for module in (system.agents.waveform.open_head,
                       system.agents.prototype.open_head,
                       system.agents.waveform.competence_head,
                       system.agents.prototype.competence_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif mode != "frozen":
        raise ValueError(f"unsupported Agent trainability mode: {mode}")


def _train_g0_model(cfg: Mapping[str, Any], episode, device: torch.device,
                    seed: int, run_dir: Path, *, sanity: bool,
                    train_b2: bool = True,
                    checkpoint_name: str = "g0_model.pt") -> Stage7System:
    train_known = episode.train_known
    if sanity:
        train_known = _balanced_limit(
            train_known, int(cfg.get("sanity_max_per_class", 0)), seed)
    system = _system(cfg, len(episode.known_classes)).to(device)
    _set_agent_trainability(system, "all")
    started = time.time()
    # Closed-set identity/metric representations are learned before any
    # synthetic open-space objective is allowed to shape them.  Rejection is
    # fitted only in the subsequent frozen-backbone PUG phase.  This avoids a
    # Known-only BCE shortcut competing with the two Agents' class losses.
    local_cfg = dict(cfg)
    local_cfg["open_loss_weight"] = float(
        cfg.get("local_pretrain_open_loss_weight", 0.0))
    local_cfg["competence_weight"] = float(
        cfg.get("local_pretrain_competence_weight", 0.0))
    local_history, local_epoch = train_local_agents(
        system, train_known, None, local_cfg, device, seed)
    pretrain_predictions = collect_predictions(
        system, train_known, device, route_action=RouteAction.STOP,
        batch_size=int(cfg.get("evaluation_batch_size", 1024)))
    pretrain_accuracy = {
        role: float(np.mean(
            pretrain_predictions[f"pred_{role}"] == pretrain_predictions["y"]))
        for role in LOCAL_ROLES
    }
    pretrain_floor = float(cfg.get("pretrain_min_train_accuracy", 0.85))
    pretrain_audit = {
        "eval_mode_train_accuracy": pretrain_accuracy,
        "required_minimum": pretrain_floor,
        "passed": all(value >= pretrain_floor
                      for value in pretrain_accuracy.values()),
        "num_samples": len(train_known),
        "formal_unknown_used": False,
    }
    _write_json(run_dir / "pretrain_capability.json", pretrain_audit)
    if not pretrain_audit["passed"]:
        # Commit an explicitly incomplete diagnostic rather than spending
        # more time on enrollment/PUG/B2 with an invalid local foundation.
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "failed_local_capability_gate",
            "model": {
                key: value.detach().cpu()
                for key, value in system.state_dict().items()},
            "known_classes": list(episode.known_classes),
            "proxy_unknown_classes": list(episode.proxy_unknown_classes),
            "config_contract": _checkpoint_config_contract(cfg),
            "seed": int(seed),
            "local_epoch": int(local_epoch),
            "local_history": local_history,
            "pretrain_audit": pretrain_audit,
        }, run_dir / "local_pretrain_failed.pt")
        raise RuntimeError(
            "Stage-7 local capability gate failed before enrollment/PUG/B2: "
            + ", ".join(
                f"{role}={value:.4f}" for role, value in pretrain_accuracy.items()))
    enrollment = refresh_enrollment(
        system, train_known, device, batch_size=int(cfg.get("batch_size", 256)))
    pug = make_competition_pug(
        system, train_known, device,
        eta=float(cfg.get("pug_eta", 1.5)),
        max_per_sample=int(cfg.get("pug_variants_per_source", 1)),
        batch_size=int(cfg.get("batch_size", 256)), seed=seed + 17)
    pug = _limit_pug(pug, int(cfg.get("pug_max_samples", 8192)), seed + 19)

    # PUG is auxiliary: after the full Known pretrain only the two rejection
    # and private competence heads are allowed to move.  This prevents synthetic samples from
    # rewriting either Agent's independent class decision.
    _set_agent_trainability(system, "open_heads")
    competence_mode = str(
        cfg.get("competence_mode", "shared_unknown_logit"))
    if competence_mode not in {"shared_unknown_logit", "independent_head"}:
        raise ValueError(f"unsupported competence_mode: {competence_mode}")
    system.agents.set_competence_enabled(
        competence_mode == "independent_head"
        and float(cfg.get("competence_weight", 0.0)) > 0.0)
    open_cfg = dict(cfg)
    open_cfg["local_pretrain_epochs"] = int(cfg.get("open_head_epochs", 5))
    open_history, _ = train_local_agents(
        system, train_known, None, open_cfg, device, seed + 31,
        pug_dataset=pug)
    refresh_enrollment(
        system, train_known, device, batch_size=int(cfg.get("batch_size", 256)))
    _set_agent_trainability(system, "frozen")
    b2_history = []
    if train_b2:
        b2_history = train_fair_b2(
            system, train_known, None, cfg, device, seed + 47,
            pug_dataset=pug)
    system.eval()
    if train_b2:
        sample_x, _ = next(iter(torch.utils.data.DataLoader(
            train_known, batch_size=min(8, len(train_known)), shuffle=False)))
        assert_stop_matches_b2(system, system.local(sample_x.to(device)))
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_contract": _checkpoint_config_contract(cfg),
        "model": {key: value.detach().cpu() for key, value in system.state_dict().items()},
        "known_classes": list(episode.known_classes),
        "proxy_unknown_classes": list(episode.proxy_unknown_classes),
        "seed": int(seed), "local_epoch": int(local_epoch),
        "local_history": local_history, "open_history": open_history,
        "b2_history": b2_history, "enrollment": enrollment,
        "pretrain_audit": pretrain_audit,
        "b2_trained": bool(train_b2),
        "num_train_known": len(train_known), "num_pug": len(pug),
        "elapsed_seconds": time.time() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, run_dir / checkpoint_name)
    _write_json(run_dir / "training_history.json", {
        key: value for key, value in checkpoint.items() if key != "model"})
    return system


def _calibrated_rule(calibration: Mapping[str, np.ndarray],
                     known: Mapping[str, np.ndarray],
                     proxy: Mapping[str, np.ndarray], role: str,
                     calibration_dataset: ProvenanceSubset,
                     known_acceptance: float) -> tuple[dict, dict]:
    if role == "b2":
        pred_key, raw_key = "pred", "raw_unknown"
    else:
        pred_key, raw_key = f"pred_{role}", f"raw_{role}"
    service = ClasswiseCalibrationService(known_acceptance)
    service.fit(
        calibration[raw_key], calibration[pred_key],
        labels=calibration["y"], dataset=calibration_dataset)
    y = np.r_[known["y"], proxy["y"]]
    prediction = np.r_[known[pred_key], proxy[pred_key]]
    raw = np.r_[known[raw_key], proxy[raw_key]]
    decision = service.predict(raw, prediction)
    metrics = evaluate_open_set(
        y, prediction, decision.unknown_score, decision.threshold)
    arrays = {
        "y": y, "closed_pred": prediction,
        "unknown_score": decision.unknown_score,
        "threshold": decision.threshold,
        "reject": decision.reject,
    }
    return metrics, {"arrays": arrays, "calibration": service.state_dict()}


def _correct(row: Mapping[str, np.ndarray]) -> np.ndarray:
    y = np.asarray(row["y"])
    pred = np.asarray(row["closed_pred"])
    reject = np.asarray(row["unknown_score"]) >= np.asarray(row["threshold"])
    return np.where(y < 0, reject, (~reject) & (pred == y))


def _pairwise_agent_rescue(
    systems: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, float | int]]:
    """Measure each local Agent's rescue of the *other* local Agent.

    G0 asks whether both private views contribute complementary decisions.
    Measuring rescue against B2 makes an anchored Agent mathematically unable
    to rescue itself and therefore confounds complementarity with the current
    fusion choice.  Pairwise rescue avoids that artefact and remains an
    audit-only statistic.
    """

    success = {role: _correct(systems[role]) for role in LOCAL_ROLES}
    result: dict[str, dict[str, float | int]] = {}
    for role, other in (("waveform", "prototype"),
                        ("prototype", "waveform")):
        rescue = success[role] & ~success[other]
        failures = max(int((~success[other]).sum()), 1)
        result[role] = {
            "absolute_rate": float(rescue.mean()),
            "other_failure_rate": float(rescue.sum() / failures),
            "count": int(rescue.sum()),
        }
    return result


def _public_selector_metrics(known: Mapping[str, np.ndarray],
                             proxy: Mapping[str, np.ndarray],
                             systems: Mapping[str, Mapping[str, np.ndarray]],
                             seed: int) -> tuple[dict, dict]:
    features = np.concatenate([
        known["public_features"], proxy["public_features"]], axis=0)
    roles = LOCAL_ROLES
    losses = np.stack([(~_correct(systems[role])).astype(np.float64)
                       for role in roles], axis=1)
    selection = cross_fitted_public_selector(
        features, losses, num_folds=5, seed=seed)
    chosen = np.asarray(selection["prediction"], dtype=np.int64)
    rows = np.arange(len(chosen))
    y = np.asarray(systems[roles[0]]["y"])
    predictions = np.stack([systems[role]["closed_pred"] for role in roles], axis=1)
    scores = np.stack([systems[role]["unknown_score"] for role in roles], axis=1)
    thresholds = np.stack([systems[role]["threshold"] for role in roles], axis=1)
    pred = predictions[rows, chosen]
    score = scores[rows, chosen]
    threshold = thresholds[rows, chosen]
    metrics = evaluate_open_set(y, pred, score, threshold)
    return metrics, {**selection, "closed_pred": pred,
                     "unknown_score": score, "threshold": threshold}


def _evaluate_g0_fold(system: Stage7System, episode, cfg: Mapping[str, Any],
                      device: torch.device, seed: int, run_dir: Path,
                      *, include_outer_diagnostic: bool = True) -> dict:
    batch_size = int(cfg.get("evaluation_batch_size", 1024))
    calibration = collect_predictions(
        system, episode.calibration_known, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    known = collect_predictions(
        system, episode.test_known, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    proxy = collect_predictions(
        system, episode.test_proxy_unknown, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    metrics: dict[str, dict] = {}
    details: dict[str, dict] = {}
    systems: dict[str, Mapping[str, np.ndarray]] = {}
    for role in (*LOCAL_ROLES, "b2"):
        metrics[role], details[role] = _calibrated_rule(
            calibration, known, proxy, role, episode.calibration_known,
            float(cfg.get("known_acceptance", 0.95)))
        systems[role] = details[role]["arrays"]
    complementarity = complementarity_report(
        systems["b2"]["y"],
        {role: systems[role] for role in LOCAL_ROLES},
        baseline="waveform")
    pairwise_rescue = _pairwise_agent_rescue(systems)
    selector_metrics = None
    selector: dict[str, Any] = {}
    if include_outer_diagnostic:
        selector_metrics, selector = _public_selector_metrics(
            known, proxy, systems, seed)

    arrays: dict[str, np.ndarray] = {
        "y": systems["b2"]["y"],
        "public_features": np.concatenate(
            [known["public_features"], proxy["public_features"]]),
    }
    if include_outer_diagnostic:
        arrays["selector_action"] = np.asarray(selector["prediction"])
    for role, row in systems.items():
        for key in ("closed_pred", "unknown_score", "threshold", "reject"):
            arrays[f"{key}_{role}"] = np.asarray(row[key])
    np.savez_compressed(run_dir / "g0_scores.npz", **arrays)
    report = {
        "fold": int(episode.index),
        "known_classes": list(episode.known_classes),
        "proxy_unknown_classes": list(episode.proxy_unknown_classes),
        "metrics": metrics, "public_selector": selector_metrics,
        "public_selector_diagnostics": {
            key: value for key, value in selector.items()
            if key not in {"prediction", "target", "closed_pred",
                           "unknown_score", "threshold"}},
        "complementarity": complementarity,
        "pairwise_agent_rescue": pairwise_rescue,
        "protocol": {
            "formal_unknown_used": False,
            "calibration_and_test_known_are_disjoint": True,
            "oscr_uses_pre_rejection_prediction": True,
            "stop_exactly_equals_b2": True,
            "outer_selector_is_diagnostic_only": bool(
                include_outer_diagnostic),
        },
    }
    _write_json(run_dir / "g0_fold_metrics.json", report)
    return report


def _sanity_checks(report: Mapping[str, Any]) -> dict:
    metric = report["metrics"]
    best_local = max(metric[role]["h_score"] for role in LOCAL_ROLES)
    comp = report["complementarity"]
    unique = report.get("pairwise_agent_rescue", comp["unique_rescue"])
    checks = {
        **{f"{role}_known_accuracy_ge_0_85":
           metric[role]["known_accuracy"] >= 0.85 for role in LOCAL_ROLES},
        **{f"{role}_auroc_ge_0_75": metric[role]["auroc"] >= 0.75
           for role in LOCAL_ROLES},
        "b2_not_worse_than_best_local_by_0_005":
            metric["b2"]["h_score"] - best_local >= -0.005,
        "oracle_delta_h_ge_0_03":
            comp["oracle"]["h_score"] - metric["b2"]["h_score"] >= 0.03,
        "oracle_delta_oscr_ge_0_015":
            comp["oracle"]["oscr"] - metric["b2"]["oscr"] >= 0.015,
        **{f"{role}_unique_rescue_ge_0_05":
           unique[role].get(
               "other_failure_rate", unique[role].get(
                   "baseline_failure_rate", 0.0)) >= 0.05
           for role in LOCAL_ROLES},
        "public_selector_delta_h_ge_0_01":
            report["public_selector"]["h_score"] - best_local >= 0.01,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "note": "single-fold feasibility only; this cannot promote G1"}


def _g0a_checks(report: Mapping[str, Any]) -> dict:
    """Necessary local-capability/headroom gate before any inner training."""

    metric = report["metrics"]
    pairwise = report.get("pairwise_agent_rescue")
    if not isinstance(pairwise, Mapping):
        return {
            "passed": False,
            "checks": {"pairwise_agent_rescue_present": False},
            "note": "re-evaluate G0a with pairwise Agent rescue enabled",
        }
    best_h = max(float(metric[role]["h_score"]) for role in LOCAL_ROLES)
    best_oscr = max(float(metric[role]["oscr"]) for role in LOCAL_ROLES)
    oracle = report["complementarity"]["oracle"]
    checks = {
        **{f"{role}_known_accuracy_ge_0_85":
           float(metric[role]["known_accuracy"]) >= 0.85
           for role in LOCAL_ROLES},
        **{f"{role}_auroc_ge_0_75":
           float(metric[role]["auroc"]) >= 0.75
           for role in LOCAL_ROLES},
        "agent_oracle_delta_h_vs_best_local_ge_0_03":
            float(oracle["h_score"]) - best_h >= 0.03,
        "agent_oracle_delta_oscr_vs_best_local_ge_0_015":
            float(oracle["oscr"]) - best_oscr >= 0.015,
        **{f"{role}_pairwise_rescue_ge_0_05":
           float(pairwise[role]["other_failure_rate"]) >= 0.05
           for role in LOCAL_ROLES},
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "best_local_h": best_h, "best_local_oscr": best_oscr,
        "oracle_delta_h": float(oracle["h_score"]) - best_h,
        "oracle_delta_oscr": float(oracle["oscr"]) - best_oscr,
        "note": "necessary G0a only; passing authorises inner models, not G1",
    }


def _evaluate_public_table_fold(
    system: Stage7System,
    partition,
    tables: Mapping[str, PublicDecisionTable],
    cfg: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    """Calibrate/evaluate one held-out inner producer from public tables."""

    required = {
        "threshold_calibration_known", "meta_known", "meta_proxy_unknown"}
    if set(tables) != required:
        raise ValueError("inner public evaluation needs calibration/known/proxy tables")
    batch_size = int(cfg.get("evaluation_batch_size", 1024))
    rows = {
        name: evaluate_public_b2_table(
            system, table, device, batch_size=batch_size)
        for name, table in tables.items()
    }
    metrics: dict[str, dict[str, float]] = {}
    systems: dict[str, dict[str, np.ndarray]] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    for role in (*LOCAL_ROLES, "b2"):
        pred_key = "pred" if role == "b2" else f"pred_{role}"
        raw_key = "raw_unknown" if role == "b2" else f"raw_{role}"
        calibration = rows["threshold_calibration_known"]
        known = rows["meta_known"]
        proxy = rows["meta_proxy_unknown"]
        service = ClasswiseCalibrationService(
            float(cfg.get("known_acceptance", 0.95)))
        service.fit(
            calibration[raw_key], calibration[pred_key],
            labels=calibration["y"],
            dataset=partition.threshold_calibration_known)
        y = np.r_[known["y"], proxy["y"]]
        prediction = np.r_[known[pred_key], proxy[pred_key]]
        raw = np.r_[known[raw_key], proxy[raw_key]]
        decision = service.predict(raw, prediction)
        metrics[role] = evaluate_open_set(
            y, prediction, decision.unknown_score, decision.threshold)
        systems[role] = {
            "y": y, "closed_pred": prediction,
            "unknown_score": decision.unknown_score,
            "threshold": decision.threshold,
            "reject": decision.reject,
        }
        calibrators[role] = service.state_dict()
    pairwise = _pairwise_agent_rescue(systems)
    complementarity = complementarity_report(
        systems["b2"]["y"],
        {role: systems[role] for role in LOCAL_ROLES},
        baseline="waveform")
    sample_keys = np.r_[
        rows["meta_known"]["sample_keys"],
        rows["meta_proxy_unknown"]["sample_keys"]]
    arrays: dict[str, np.ndarray] = {"sample_keys": sample_keys}
    for role, values in systems.items():
        for name, value in values.items():
            arrays[f"{name}_{role}"] = np.asarray(value)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "inner_meta_scores.npz", **arrays)
    serialisable = {
        "inner_index": int(partition.inner_index),
        "known_classes": list(partition.known_classes),
        "proxy_unknown_classes": list(partition.proxy_unknown_classes),
        "metrics": metrics, "pairwise_agent_rescue": pairwise,
        "complementarity": complementarity,
        "calibrators": calibrators,
        "protocol": {
            "selector_evidence_source": "inner_episode_oof",
            "fit_and_evaluation_samples_disjoint": True,
            "outer_test_used": False,
            "formal_unknown_used": False,
            "public_only": True,
        },
    }
    _write_json(run_dir / "inner_meta_metrics.json", serialisable)
    return {**serialisable, "_systems": systems}


def _pool_inner_oof_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    evidence,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Pool four leave-one-inner-episode-out B2 evaluations."""

    if len(reports) != 4 or not evidence.registrable:
        raise RuntimeError("registered nested G0 requires four inner OOF reports")
    systems: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for role in (*LOCAL_ROLES, "b2"):
        systems[role] = {
            key: np.concatenate([
                np.asarray(report["_systems"][role][key])
                for report in reports])
            for key in ("y", "closed_pred", "unknown_score", "threshold", "reject")
        }
        values = systems[role]
        metrics[role] = evaluate_open_set(
            values["y"], values["closed_pred"], values["unknown_score"],
            values["threshold"])
    best_local_h = max(float(metrics[role]["h_score"])
                       for role in LOCAL_ROLES)
    delta_h = float(metrics["b2"]["h_score"]) - best_local_h
    fold_deltas = [
        float(report["metrics"]["b2"]["h_score"])
        - max(float(report["metrics"][role]["h_score"])
              for role in LOCAL_ROLES)
        for report in reports
    ]
    minimum_gain = float(cfg.get("g0_min_public_selector_delta_h", 0.01))
    minimum_positive = int(cfg.get("nested_min_positive_inner_folds", 3))
    checks = {
        "structural_inner_oof_evidence": bool(evidence.registrable),
        "four_inner_folds": len(evidence.folds) == 4,
        "nested_b2_delta_h_vs_best_local_ge_0_01": delta_h >= minimum_gain,
        "positive_inner_folds_ge_3":
            sum(value > 0 for value in fold_deltas) >= minimum_positive,
    }
    pairwise = _pairwise_agent_rescue(systems)
    complementarity = complementarity_report(
        systems["b2"]["y"],
        {role: systems[role] for role in LOCAL_ROLES},
        baseline="waveform")
    return {
        "passed": all(checks.values()), "checks": checks,
        "metrics": metrics, "delta_h_vs_best_local": delta_h,
        "fold_h_deltas": fold_deltas,
        "positive_inner_folds": sum(value > 0 for value in fold_deltas),
        "pairwise_agent_rescue": pairwise,
        "complementarity": complementarity,
        "evidence": {
            "source": evidence.source.value,
            "registrable": evidence.registrable,
            "folds": len(evidence.folds),
            "sample_count": len(evidence.sample_keys),
            "sample_keys_unique": len(evidence.sample_keys) == len(
                set(evidence.sample_keys)),
            "outer_test_used": False,
            "formal_unknown_used": False,
        },
        "_systems": systems,
    }


def _save_oof_summary(run_dir: Path, report: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "nested_oof_summary.json", {
        key: value for key, value in report.items() if key != "_systems"})
    arrays: dict[str, np.ndarray] = {}
    for role, values in report["_systems"].items():
        for name, value in values.items():
            arrays[f"{name}_{role}"] = np.asarray(value)
    np.savez_compressed(run_dir / "nested_oof_scores.npz", **arrays)


def _evaluate_nested_outer_fold(
    system: Stage7System,
    episode,
    cfg: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    *,
    nested_oof: Mapping[str, Any],
    transfer_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen inner-trained B2 once on the outer hold-out."""

    batch_size = int(cfg.get("evaluation_batch_size", 1024))
    calibration = collect_predictions(
        system, episode.calibration_known, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    known = collect_predictions(
        system, episode.test_known, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    proxy = collect_predictions(
        system, episode.test_proxy_unknown, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    metrics: dict[str, dict[str, float]] = {}
    details: dict[str, dict[str, Any]] = {}
    systems: dict[str, Mapping[str, np.ndarray]] = {}
    for role in (*LOCAL_ROLES, "b2"):
        metrics[role], details[role] = _calibrated_rule(
            calibration, known, proxy, role, episode.calibration_known,
            float(cfg.get("known_acceptance", 0.95)))
        systems[role] = details[role]["arrays"]
    complementarity = complementarity_report(
        systems["b2"]["y"],
        {role: systems[role] for role in LOCAL_ROLES},
        baseline="waveform")
    pairwise = _pairwise_agent_rescue(systems)
    arrays: dict[str, np.ndarray] = {}
    for role, row in systems.items():
        for key in ("y", "closed_pred", "unknown_score", "threshold", "reject"):
            arrays[f"{key}_{role}"] = np.asarray(row[key])
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "nested_outer_scores.npz", **arrays)
    report = {
        "fold": int(episode.index),
        "known_classes": list(episode.known_classes),
        "proxy_unknown_classes": list(episode.proxy_unknown_classes),
        "metrics": metrics,
        # The registered selector metric comes only from inner OOF evidence;
        # no selector is fitted to the outer labels.
        "public_selector": nested_oof["metrics"]["b2"],
        "nested_oof": {
            key: value for key, value in nested_oof.items()
            if key != "_systems"},
        "pairwise_agent_rescue": pairwise,
        "complementarity": complementarity,
        "transfer_audit": dict(transfer_audit),
        "protocol": {
            "formal_unknown_used": False,
            "outer_test_used_for_training_or_selection": False,
            "nested_inner_oof_selector": True,
            "public_only_b2_transfer": True,
            "calibration_and_test_known_are_disjoint": True,
            "oscr_uses_pre_rejection_prediction": True,
            "stop_exactly_equals_b2": True,
        },
    }
    _write_json(run_dir / "nested_outer_metrics.json", report)
    return report


def _load_or_collect_inner_tables(
    system: Stage7System,
    partition,
    device: torch.device,
    run_dir: Path,
    cfg: Mapping[str, Any],
) -> dict[str, PublicDecisionTable]:
    datasets = {
        "train_known": partition.train_known,
        "train_proxy_unknown": partition.train_proxy_unknown,
        "threshold_calibration_known": partition.threshold_calibration_known,
        "meta_known": partition.meta_known,
        "meta_proxy_unknown": partition.meta_proxy_unknown,
    }
    tables: dict[str, PublicDecisionTable] = {}
    for role, dataset in datasets.items():
        path = run_dir / f"public_{role}.npz"
        if path.exists():
            table = PublicDecisionTable.load(path)
            if (table.outer_index != partition.outer_index
                    or table.inner_index != partition.inner_index
                    or table.split_role != role
                    or tuple(table.sample_keys) != tuple(dataset.sample_keys)):
                raise ValueError("cached inner public table does not match protocol")
        else:
            table = collect_public_decision_table(
                system, dataset, device,
                outer_index=partition.outer_index,
                inner_index=partition.inner_index, split_role=role,
                batch_size=int(cfg.get("evaluation_batch_size", 1024)))
            table.save(path)
        tables[role] = table
    return tables


def _run_nested_outer(
    cfg: Mapping[str, Any],
    episode,
    outer_model: Stage7System,
    device: torch.device,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Train four inner systems, cross-fit B2, then evaluate outer once."""

    partitions = build_inner_registration_partitions(
        episode, seed=int(cfg.get("partition_seeds", [2026])[0]),
        calibration_fraction=float(cfg.get("nested_calibration_fraction", 0.5)))
    all_tables: dict[int, dict[str, PublicDecisionTable]] = {}
    row_batches: list[SelectorRowBatch] = []
    for partition in partitions:
        inner_dir = run_dir / f"inner{partition.inner_index}"
        checkpoint = inner_dir / "inner_local_model.pt"
        inner_seed = int(seed) + 10000 + 101 * int(partition.inner_index)
        if checkpoint.exists():
            model = _load_fold_model(
                checkpoint, cfg, partition, device)
        else:
            set_seed(inner_seed)
            model = _train_g0_model(
                cfg, partition, device, inner_seed, inner_dir,
                sanity=False, train_b2=False,
                checkpoint_name="inner_local_model.pt")
        tables = _load_or_collect_inner_tables(
            model, partition, device, inner_dir, cfg)
        all_tables[int(partition.inner_index)] = tables
        feature_rows, loss_rows = [], []
        for role in ("meta_known", "meta_proxy_unknown"):
            features, losses = public_local_action_losses(
                tables[role], device,
                batch_size=int(cfg.get("evaluation_batch_size", 1024)),
                class_weight=float(cfg.get("class_loss_weight", 1.0)),
                open_weight=float(cfg.get("open_loss_weight", 1.0)))
            feature_rows.append(features)
            loss_rows.append(losses)
        row_batches.append(SelectorRowBatch(
            producer_inner_index=int(partition.inner_index),
            sample_keys=tuple(partition.selector_sample_keys),
            public_features=np.concatenate(feature_rows),
            action_losses=np.concatenate(loss_rows)))
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evidence = inner_oof_selector_evidence(row_batches)

    crossfit_reports: list[dict[str, Any]] = []
    for partition in partitions:
        heldout = int(partition.inner_index)
        fit_tables = [
            all_tables[index][role]
            for index in sorted(all_tables) if index != heldout
            for role in ("train_known", "train_proxy_unknown")]
        carrier = _system(cfg, int(fit_tables[0].num_classes)).to(device)
        fit = fit_nested_public_b2(
            carrier, fit_tables, cfg, device,
            int(seed) + 20000 + 211 * heldout)
        candidate_dir = run_dir / "crossfit" / f"heldout_inner{heldout}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(fit.bundle), candidate_dir / "b2_transfer.pt")
        _write_json(candidate_dir / "fit_audit.json", {
            "audit": fit.audit, "history": fit.history})
        report = _evaluate_public_table_fold(
            carrier, partition, {
                role: all_tables[heldout][role]
                for role in ("threshold_calibration_known", "meta_known",
                             "meta_proxy_unknown")},
            cfg, device, candidate_dir)
        crossfit_reports.append(report)
        carrier.to("cpu")
        del carrier
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pooled = _pool_inner_oof_reports(
        crossfit_reports, evidence=evidence, cfg=cfg)
    _save_oof_summary(run_dir / "crossfit", pooled)
    if not pooled["passed"]:
        result = {
            "outer_index": int(episode.index),
            "passed": False,
            "stopped_before_outer_evaluation": True,
            "nested_oof": {
                key: value for key, value in pooled.items()
                if key != "_systems"},
            "formal_unknown_used": False,
            "outer_test_used_after_inner_failure": False,
        }
        _write_json(run_dir / "nested_g0_result.json", result)
        return result

    final_tables = [
        all_tables[index][role] for index in sorted(all_tables)
        for role in ("train_known", "train_proxy_unknown")]
    carrier = _system(cfg, int(final_tables[0].num_classes)).to(device)
    final_fit = fit_nested_public_b2(
        carrier, final_tables, cfg, device, int(seed) + 30001)
    torch.save(dict(final_fit.bundle), run_dir / "final_b2_transfer.pt")
    _write_json(run_dir / "final_b2_fit.json", {
        "audit": final_fit.audit, "history": final_fit.history})
    transfer = import_b2_transfer_bundle(outer_model, final_fit.bundle)
    outer_model.to(device).eval()
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_contract": _checkpoint_config_contract(cfg),
        "model": {key: value.detach().cpu()
                  for key, value in outer_model.state_dict().items()},
        "known_classes": list(episode.known_classes),
        "proxy_unknown_classes": list(episode.proxy_unknown_classes),
        "seed": int(seed), "nested_oof_passed": True,
        "b2_transfer_audit": transfer,
    }
    torch.save(checkpoint, run_dir / "nested_g0_model.pt")
    outer_report = _evaluate_nested_outer_fold(
        outer_model, episode, cfg, device, run_dir,
        nested_oof=pooled, transfer_audit=transfer)
    result = {
        "outer_index": int(episode.index), "passed": True,
        "stopped_before_outer_evaluation": False,
        "nested_oof": {
            key: value for key, value in pooled.items()
            if key != "_systems"},
        "outer": outer_report,
        "formal_unknown_used": False,
    }
    _write_json(run_dir / "nested_g0_result.json", result)
    return result


def _aggregate_g0(reports: list[Mapping[str, Any]]) -> dict:
    return {
        "nested_inner_oof_selector": all(
            bool(row.get("protocol", {}).get("nested_inner_oof_selector", False))
            for row in reports),
        "outer_test_used_for_selection": any(
            bool(row.get("protocol", {}).get(
                "outer_test_used_for_training_or_selection", True))
            for row in reports),
        "public_only_b2_transfer": all(
            bool(row.get("protocol", {}).get("public_only_b2_transfer", False))
            for row in reports),
        "formal_unknown_used": any(
            bool(row.get("protocol", {}).get("formal_unknown_used", True))
            for row in reports),
        "local_known_accuracy": {
            role: [row["metrics"][role]["known_accuracy"] for row in reports]
            for role in LOCAL_ROLES},
        "local_auroc": {
            role: [row["metrics"][role]["auroc"] for row in reports]
            for role in LOCAL_ROLES},
        "single_h_score": {
            role: [row["metrics"][role]["h_score"] for row in reports]
            for role in LOCAL_ROLES},
        "unique_rescue": {
            role: [row.get("pairwise_agent_rescue",
                           row["complementarity"]["unique_rescue"])[role].get(
                               "other_failure_rate",
                               row["complementarity"]["unique_rescue"][role].get(
                                   "baseline_failure_rate", 0.0))
                   for row in reports]
            for role in LOCAL_ROLES},
        "b2_h_score": [row["metrics"]["b2"]["h_score"] for row in reports],
        "b2_oscr": [row["metrics"]["b2"]["oscr"] for row in reports],
        "oracle_h_score": [row["complementarity"]["oracle"]["h_score"]
                           for row in reports],
        "oracle_oscr": [row["complementarity"]["oracle"]["oscr"]
                        for row in reports],
        "public_selector_h_score": [row["public_selector"]["h_score"]
                                    for row in reports],
    }


def _load_fold_model(path: Path, cfg: Mapping[str, Any], episode,
                     device: torch.device) -> Stage7System:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("checkpoint_schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint schema is stale; use a fresh Stage-7 output directory")
    expected_contract = _checkpoint_config_contract(cfg)
    observed_contract = payload.get("config_contract")
    if observed_contract != expected_contract:
        observed_contract = (observed_contract
                             if isinstance(observed_contract, Mapping) else {})
        keys = sorted(
            key for key in set(observed_contract) | set(expected_contract)
            if observed_contract.get(key) != expected_contract.get(key))
        raise ValueError(
            "checkpoint and current Stage-7 method/data configuration differ "
            f"(changed fields: {', '.join(keys)})")
    if tuple(payload["known_classes"]) != tuple(episode.known_classes):
        raise ValueError("checkpoint and nested-LCO class partition differ")
    if tuple(payload.get("proxy_unknown_classes", ())) != tuple(
            episode.proxy_unknown_classes):
        raise ValueError("checkpoint and nested-LCO proxy partition differ")
    model = _system(cfg, len(episode.known_classes)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def run_stage7(cfg: Mapping[str, Any], *, phase: str = "sanity",
               outer_fold: int | None = None) -> dict:
    """Execute one registered phase without bypassing a failed gate."""
    if phase not in {
            "sanity", "nested_sanity", "g0", "g1", "g2", "g3", "formal"}:
        raise ValueError(f"unsupported Stage-7 phase: {phase}")
    started = time.time()
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    splits = load_splits(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg.get("outer_folds", 5)),
        inner_folds=int(cfg.get("inner_folds", 4)),
        seed=int(cfg.get("partition_seeds", [2026])[0]))
    _write_or_validate_frozen_config(out_root / "frozen_config.json", cfg)
    _write_json(out_root / "nested_protocol_manifest.json",
                protocol_manifest(protocol))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if phase == "nested_sanity":
        fold_index = 0 if outer_fold is None else int(outer_fold)
        if not 0 <= fold_index < len(protocol.outer_folds):
            raise IndexError(f"outer fold {fold_index} is out of range")
        episode = protocol.outer_folds[fold_index]
        seed = int(cfg.get("seed", 42)) + 1000 * fold_index
        phase_dir = out_root / phase / f"fold{fold_index}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        # Reuse a completed, configuration-identical G0a checkpoint when one
        # exists.  Evaluation never mutates it; nested B2 is saved separately.
        prior_checkpoint = out_root / "sanity" / f"fold{fold_index}" / "g0_model.pt"
        local_checkpoint = phase_dir / "g0a" / "g0_model.pt"
        if prior_checkpoint.exists():
            outer_model = _load_fold_model(
                prior_checkpoint, cfg, episode, device)
            checkpoint_source = str(prior_checkpoint)
        elif local_checkpoint.exists():
            outer_model = _load_fold_model(
                local_checkpoint, cfg, episode, device)
            checkpoint_source = str(local_checkpoint)
        else:
            set_seed(seed)
            outer_model = _train_g0_model(
                cfg, episode, device, seed, phase_dir / "g0a",
                sanity=True, train_b2=False)
            checkpoint_source = str(local_checkpoint)
        g0a_dir = phase_dir / "g0a"
        g0a_dir.mkdir(parents=True, exist_ok=True)
        g0a_report = _evaluate_g0_fold(
            outer_model, episode, cfg, device, seed, g0a_dir,
            include_outer_diagnostic=False)
        g0a = _g0a_checks(g0a_report)
        _write_json(g0a_dir / "g0a_decision.json", {
            **g0a, "checkpoint_source": checkpoint_source,
            "formal_unknown_used": False})
        if not g0a["passed"]:
            result = {
                "phase": phase, "dataset": cfg["dataset"],
                "fold": fold_index, "g0a": g0a,
                "nested_started": False,
                "formal_unknown_evaluated": False,
                "elapsed_seconds": time.time() - started,
            }
            _write_json(phase_dir / "nested_sanity_report.json", result)
            return result
        nested = _run_nested_outer(
            cfg, episode, outer_model, device, seed, phase_dir)
        result = {
            "phase": phase, "dataset": cfg["dataset"],
            "fold": fold_index, "g0a": g0a, "nested": nested,
            "formal_unknown_evaluated": False,
            "elapsed_seconds": time.time() - started,
        }
        _write_json(phase_dir / "nested_sanity_report.json", result)
        return result

    if phase in {"sanity", "g0"}:
        if phase == "sanity":
            fold_indices = [0 if outer_fold is None else int(outer_fold)]
        else:
            # The previous outer-only implementation could promote a selector
            # fitted to outer test labels.  Registered G0 is fail-closed until
            # the nested single-fold screen has passed and the five-fold loop
            # is migrated to the same inner-OOF path.
            raise NotImplementedError(
                "registered G0 is disabled: run nested_sanity first; the old "
                "outer-test selector is diagnostic-only and cannot unlock G1")
        reports = []
        for fold_index in fold_indices:
            if not 0 <= fold_index < len(protocol.outer_folds):
                raise IndexError(f"outer fold {fold_index} is out of range")
            episode = protocol.outer_folds[fold_index]
            phase_dir = out_root / phase / f"fold{fold_index}"
            seed = int(cfg.get("seed", 42)) + 1000 * fold_index
            print(f"[stage7:{phase}] dataset={cfg['dataset']} fold={fold_index} "
                  f"known={len(episode.known_classes)} proxy={len(episode.proxy_unknown_classes)}")
            checkpoint_path = phase_dir / "g0_model.pt"
            metrics_path = phase_dir / "g0_fold_metrics.json"
            if checkpoint_path.exists():
                model = _load_fold_model(
                    checkpoint_path, cfg, episode, device)
                if metrics_path.exists():
                    report = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print(f"[stage7:{phase}] reused completed fold {fold_index}")
                else:
                    # Training checkpoints are committed before evaluation so
                    # an interrupted desktop turn can resume without spending
                    # another 30+5+5 epochs.  Evaluation remains deterministic
                    # and uses only the dedicated calibration/test partitions.
                    report = _evaluate_g0_fold(
                        model, episode, cfg, device, seed, phase_dir)
                    print(f"[stage7:{phase}] resumed evaluation for fold {fold_index}")
            else:
                set_seed(seed)
                model = _train_g0_model(
                    cfg, episode, device, seed, phase_dir,
                    sanity=(phase == "sanity"))
                report = _evaluate_g0_fold(
                    model, episode, cfg, device, seed, phase_dir)
            reports.append(report)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if phase == "sanity":
            decision = _sanity_checks(reports[0])
            result = {"phase": phase, "dataset": cfg["dataset"],
                      "folds": reports, "sanity": decision,
                      "elapsed_seconds": time.time() - started,
                      "formal_unknown_evaluated": False}
            _write_json(out_root / "sanity" / "sanity_report.json", result)
            return result
        gate_input = _aggregate_g0(reports)
        gate = evaluate_g0(gate_input)
        save_gate_report(out_root / "gates", gate)
        result = {"phase": phase, "dataset": cfg["dataset"],
                  "folds": reports, "gate": gate.to_dict(),
                  "elapsed_seconds": time.time() - started,
                  "formal_unknown_evaluated": False}
        _write_json(out_root / "g0" / "g0_summary.json", result)
        return result

    # Every later phase is intentionally blocked until its predecessor has a
    # persisted passing artifact.  Their numerical implementation is invoked
    # only after the cheaper causal capability audit succeeds.
    prerequisite = out_root / "gates" / f"{phase[:-1] if phase.startswith('g') else 'g3'}_gate.json"
    if phase == "g1":
        prerequisite = out_root / "gates" / "g0_gate.json"
    elif phase == "g2":
        prerequisite = out_root / "gates" / "g1_gate.json"
    elif phase == "g3":
        prerequisite = out_root / "gates" / "g2_gate.json"
    elif phase == "formal":
        prerequisite = out_root / "gates" / "g3_gate.json"
    gate = load_gate_result(prerequisite, expected_gate=prerequisite.stem[:2])
    if not gate.passed:
        raise RuntimeError(
            f"Stage-7 {phase} is blocked because {gate.gate} did not pass")
    raise NotImplementedError(
        f"Stage-7 {phase} is registered and prerequisite-guarded, but no dataset "
        "has passed the preceding gate yet; implement the phase only after the "
        "cheaper evidence justifies that computation")
