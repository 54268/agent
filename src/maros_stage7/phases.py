"""Executable, fail-closed G1/G2 phases for Stage-7.

This module deliberately sits above the model/training primitives and below
the experiment runner.  It owns the protocol details which are easy to get
wrong in an ad-hoc script:

* G1 freezes every local-decision parameter, B2 and the router.  Only the
  Waveform conditional response head and the semantic adjudicator may learn.
* G1 calibrates B2, forced directions, equal-bandwidth parallel exchange,
  sequential reliability routing and every intervention independently.
* G2 cannot start without persisted, passing G0 and G1 artifacts.  It freezes
  the complete consultation system and learns only the public-summary router
  from exact STOP/W_FIRST/P_FIRST counterfactual losses.
* Formal unknowns and outer/test proxy unknowns are never accepted by a
  fitting path.  A real proxy-unknown training set must belong to the same
  inner episode as its Known training set.

The functions return plain mappings/NumPy arrays so an experiment runner can
persist every per-sample output without granting this layer filesystem access.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import re
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .contracts import (ChallengePacket, IndependentSummaryPacket,
                        ProposalPacket, RouteAction)
from .gates import (GateResult, evaluate_g1, evaluate_g2,
                    require_prerequisite_artifacts)
from .metrics import (ClasswiseCalibrationService, communication_effect_report,
                      evaluate_open_set)
from .splits import (ProvenanceSubset, assert_no_formal_unknown,
                     assert_sample_disjoint)
from .training import (
    ExactCounterfactualSupervisor,
    _assert_training_dataset,
    balanced_action_weights,
    collect_counterfactual_supervision,
    collect_predictions,
    forced_response_loss,
    router_diagnostics,
    set_seed,
)


_INNER_KNOWN = re.compile(r"^(outer\d+_inner\d+)_train_known$")
_INNER_PROXY = re.compile(r"^(outer\d+_inner\d+)_train_proxy_unknown$")
_G1_RESPONSE_MODULES = (
    "agents.waveform.class_descriptor",
    "agents.waveform.pair_query",
    "agents.waveform.pair_residual",
    "agents.waveform.tail_head",
    "dialogue.adjudicator",
)
_WAVEFORM_RESPONSE_STATE_PREFIXES = (
    "waveform.class_descriptor.",
    "waveform.pair_query.",
    "waveform.pair_residual.",
    "waveform.tail_head.",
)
_INTERVENTIONS = (
    "messages_off",
    "messages_shuffled",
    "cross_sample_messages",
    "wrong_class_pair",
    "drop_challenger",
)


def _dataset(value):
    return value.dataset if isinstance(value, DataLoader) else value


def _purpose(value) -> str:
    return str(getattr(_dataset(value), "purpose", "")).lower()


def _validate_inner_pair(train_known, train_proxy_unknown) -> None:
    """Require a proxy and Known set to come from one registered inner fold."""

    if train_proxy_unknown is None:
        return
    known_match = _INNER_KNOWN.match(_purpose(train_known))
    proxy_match = _INNER_PROXY.match(_purpose(train_proxy_unknown))
    if known_match is None or proxy_match is None:
        raise RuntimeError(
            "proxy-unknown fitting requires matching inner train_known and "
            "train_proxy_unknown partitions")
    if known_match.group(1) != proxy_match.group(1):
        raise RuntimeError("Known and proxy-unknown data belong to different inner episodes")


def _validate_pug_origin(train_known, pug_dataset) -> None:
    if pug_dataset is None:
        return
    _assert_training_dataset(pug_dataset, "pug")
    source_classes = getattr(_dataset(pug_dataset), "source_classes", None)
    known_labels = getattr(_dataset(train_known), "y", None)
    if source_classes is not None and known_labels is not None:
        source = set(int(value) for value in np.unique(source_classes))
        known = set(int(value) for value in np.unique(known_labels))
        if not source.issubset(known):
            raise RuntimeError("PUG source classes are not contained in train_known")


def _validate_training_inputs(train_known, train_proxy_unknown=None,
                              pug_dataset=None) -> None:
    assert_no_formal_unknown(train_known, train_proxy_unknown, pug_dataset)
    _assert_training_dataset(train_known, "known")
    if train_proxy_unknown is not None:
        _assert_training_dataset(train_proxy_unknown, "proxy")
    _validate_inner_pair(train_known, train_proxy_unknown)
    _validate_pug_origin(train_known, pug_dataset)


def _weighted_training_loader(
    train_known,
    train_proxy_unknown,
    pug_dataset,
    *,
    batch_size: int,
    seed: int,
    proxy_weight: float,
    pug_weight: float,
) -> DataLoader:
    """Sample registered roles with explicit total (not per-row) weights."""

    if batch_size < 1 or proxy_weight < 0 or pug_weight < 0:
        raise ValueError("invalid Stage-7 phase loader configuration")
    datasets = [_dataset(train_known)]
    role_weights = [1.0]
    if train_proxy_unknown is not None and proxy_weight > 0:
        datasets.append(_dataset(train_proxy_unknown))
        role_weights.append(float(proxy_weight))
    if pug_dataset is not None and pug_weight > 0:
        datasets.append(_dataset(pug_dataset))
        role_weights.append(float(pug_weight))
    if any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Stage-7 phase training partitions must be non-empty")
    combined = ConcatDataset(datasets)
    weights = torch.cat([
        torch.full((len(dataset),), weight / len(dataset), dtype=torch.double)
        for dataset, weight in zip(datasets, role_weights)
    ])
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights, num_samples=len(combined), replacement=True,
        generator=generator)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler,
                      num_workers=0)


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    value: nn.Module = root
    for name in path.split("."):
        value = getattr(value, name)
    return value


def _state_snapshot(module: nn.Module, *, exclude: Sequence[str] = ()) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if not any(name.startswith(prefix) for prefix in exclude)
    }


def _state_identical(before: Mapping[str, torch.Tensor], module: nn.Module,
                     *, exclude: Sequence[str] = ()) -> bool:
    after = _state_snapshot(module, exclude=exclude)
    return before.keys() == after.keys() and all(
        torch.equal(before[name], after[name]) for name in before)


def _configure_g1_trainability(system: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    modules = [_module_by_path(system, path) for path in _G1_RESPONSE_MODULES]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    names = [name for name, parameter in system.named_parameters()
             if parameter.requires_grad]
    parameters = [parameter for parameter in system.parameters()
                  if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("G1 has no trainable conditional response parameters")
    return parameters, names


def _set_g1_modes(system: nn.Module) -> None:
    # Never let frozen encoder BatchNorm/dropout state drift during G1.
    system.eval()
    for path in _G1_RESPONSE_MODULES:
        _module_by_path(system, path).train()


def train_g1_responses(
    system: nn.Module,
    train_known,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    prerequisites,
    train_proxy_unknown=None,
    pug_dataset=None,
) -> dict[str, Any]:
    """Fit one-round responders/adjudicator while local decisions stay fixed."""

    require_prerequisite_artifacts("g1", prerequisites)
    _validate_training_inputs(train_known, train_proxy_unknown, pug_dataset)
    set_seed(seed)
    system.to(device)
    parameters, names = _configure_g1_trainability(system)
    local_before = _state_snapshot(
        system.agents, exclude=_WAVEFORM_RESPONSE_STATE_PREFIXES)
    b2_before = _state_snapshot(system.dialogue.b2_fusion)
    router_before = _state_snapshot(system.router)
    epochs = int(cfg.get("response_epochs", 5))
    batch_size = int(cfg.get("batch_size", 256))
    if epochs < 1:
        raise ValueError("response_epochs must be positive")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(cfg.get("response_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    loader = _weighted_training_loader(
        train_known, train_proxy_unknown, pug_dataset,
        batch_size=batch_size, seed=seed + 1701,
        proxy_weight=float(cfg.get("proxy_unknown_weight", 1.0)),
        pug_weight=float(cfg.get("pug_auxiliary_weight", 0.25)))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        _set_g1_modes(system)
        totals = {"loss": 0.0, "w_first": 0.0, "p_first": 0.0}
        seen = 0
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            local = system.local(x)
            loss, parts = forced_response_loss(
                system, local, labels,
                class_weight=float(cfg.get("class_loss_weight", 1.0)),
                open_weight=float(cfg.get("open_loss_weight", 1.0)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg.get("clip_grad_norm", 5.0)))
            optimizer.step()
            batch = len(labels)
            totals["loss"] += float(loss.detach()) * batch
            totals["w_first"] += parts["w_first"] * batch
            totals["p_first"] += parts["p_first"] * batch
            seen += batch
        history.append({
            "epoch": float(epoch),
            **{key: value / max(seen, 1) for key, value in totals.items()},
        })
    audit = {
        "local_decision_core_identical": _state_identical(
            local_before, system.agents,
            exclude=_WAVEFORM_RESPONSE_STATE_PREFIXES),
        "b2_identical": _state_identical(b2_before, system.dialogue.b2_fusion),
        "router_identical": _state_identical(router_before, system.router),
        "trainable_parameters": names,
        "formal_unknown_used": False,
    }
    if not all(audit[key] for key in (
            "local_decision_core_identical", "b2_identical", "router_identical")):
        raise AssertionError("G1 modified a frozen local/B2/router state")
    return {"history": history, "audit": audit}


def _validate_eval_inputs(calibration_known, test_known, test_proxy_unknown) -> None:
    assert_no_formal_unknown(calibration_known, test_known, test_proxy_unknown)
    if "calibration_known" not in _purpose(calibration_known):
        raise ValueError("G1/G2 calibration requires a dedicated calibration_known set")
    calibration_labels = np.asarray(getattr(_dataset(calibration_known), "y"), dtype=np.int64)
    known_labels = np.asarray(getattr(_dataset(test_known), "y"), dtype=np.int64)
    proxy_labels = np.asarray(getattr(_dataset(test_proxy_unknown), "y"), dtype=np.int64)
    if np.any(calibration_labels < 0) or np.any(known_labels < 0):
        raise RuntimeError("unknown rows entered Known calibration/test")
    if np.any(proxy_labels >= 0):
        raise RuntimeError("proxy-unknown evaluation rows must expose label -1")
    if all(isinstance(_dataset(value), ProvenanceSubset)
           for value in (calibration_known, test_known, test_proxy_unknown)):
        assert_sample_disjoint(
            _dataset(calibration_known), _dataset(test_known),
            _dataset(test_proxy_unknown))


def _merge_prediction_rows(*rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = set.intersection(*(set(row) for row in rows))
    return {key: np.concatenate([np.asarray(row[key]) for row in rows], axis=0)
            for key in sorted(keys)}


@torch.no_grad()
def _collect_transcript(
    system: nn.Module,
    dataset,
    device: torch.device,
    *,
    route_action: RouteAction | int | str | None,
    protocol: str,
    intervention: str | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Normalise sparse packet objects into one auditable row per sample."""

    names = (
        "proposal_sender", "proposal_top1", "proposal_top2",
        "proposal_log_odds", "proposal_local_unknown",
        "proposal_reliability", "proposal_bits", "response_sender",
        "response_kind", "response_stance", "response_alternative_class",
        "response_signed_pair_evidence", "response_open_tail_evidence",
        "response_certainty_bin", "response_reliability", "response_bits",
        "message_active",
    )
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    sender_code = {"waveform": 1, "prototype": 2}
    loader = DataLoader(_dataset(dataset), batch_size=batch_size, shuffle=False,
                        num_workers=0)
    system.eval()
    for x, _ in loader:
        x = x.to(device)
        local = system.local(x)
        action = route_action
        if isinstance(action, str):
            if action != "reliability":
                raise ValueError(f"unsupported route policy: {action}")
            action = torch.where(
                local["waveform"].reliability >= local["prototype"].reliability,
                torch.full_like(local["waveform"].top1, int(RouteAction.W_FIRST)),
                torch.full_like(local["waveform"].top1, int(RouteAction.P_FIRST)))
        output = system.deliberate(
            local, route_action=action, intervention=intervention,
            protocol=protocol)
        batch = len(x)
        integer_default = np.full(batch, -1, dtype=np.int64)
        current: dict[str, np.ndarray] = {
            "proposal_sender": np.zeros(batch, dtype=np.int64),
            "proposal_top1": integer_default.copy(),
            "proposal_top2": integer_default.copy(),
            "proposal_log_odds": np.zeros(batch, dtype=np.float32),
            "proposal_local_unknown": np.zeros(batch, dtype=np.float32),
            "proposal_reliability": np.zeros(batch, dtype=np.float32),
            "proposal_bits": np.zeros(batch, dtype=np.float32),
            "response_sender": np.zeros(batch, dtype=np.int64),
            # 0=no response, 1=conditional ChallengePacket,
            # 2=proposal-independent equal-bandwidth packet.
            "response_kind": np.zeros(batch, dtype=np.int64),
            "response_stance": integer_default.copy(),
            "response_alternative_class": integer_default.copy(),
            "response_signed_pair_evidence": np.zeros(batch, dtype=np.float32),
            "response_open_tail_evidence": np.zeros(batch, dtype=np.float32),
            "response_certainty_bin": integer_default.copy(),
            "response_reliability": np.zeros(batch, dtype=np.float32),
            "response_bits": np.zeros(batch, dtype=np.float32),
            "message_active": np.zeros(batch, dtype=bool),
        }
        for packet in output.transcript:
            active = packet.active_mask.detach().cpu().numpy().astype(bool)
            if isinstance(packet, ProposalPacket):
                current["proposal_sender"][active] = sender_code[packet.sender]
                current["proposal_top1"][active] = packet.top1.detach().cpu().numpy()[active]
                current["proposal_top2"][active] = packet.top2.detach().cpu().numpy()[active]
                current["proposal_log_odds"][active] = packet.log_odds.detach().cpu().numpy()[active]
                current["proposal_local_unknown"][active] = packet.local_unknown.detach().cpu().numpy()[active]
                current["proposal_reliability"][active] = packet.reliability.detach().cpu().numpy()[active]
                current["proposal_bits"][active] = packet.bit_cost.detach().cpu().numpy()[active]
                current["message_active"] |= active
            elif isinstance(packet, ChallengePacket):
                current["response_sender"][active] = sender_code[packet.sender]
                current["response_kind"][active] = 1
                current["response_stance"][active] = packet.stance.detach().cpu().numpy()[active]
                current["response_alternative_class"][active] = packet.alternative_class.detach().cpu().numpy()[active]
                current["response_signed_pair_evidence"][active] = packet.signed_pair_evidence.detach().cpu().numpy()[active]
                current["response_open_tail_evidence"][active] = packet.open_tail_evidence.detach().cpu().numpy()[active]
                current["response_reliability"][active] = packet.reliability.detach().cpu().numpy()[active]
                current["response_bits"][active] = packet.bit_cost.detach().cpu().numpy()[active]
            elif isinstance(packet, IndependentSummaryPacket):
                current["response_sender"][active] = sender_code[packet.sender]
                current["response_kind"][active] = 2
                current["response_alternative_class"][active] = packet.predicted_class.detach().cpu().numpy()[active]
                current["response_signed_pair_evidence"][active] = packet.confidence_margin.detach().cpu().numpy()[active]
                current["response_open_tail_evidence"][active] = packet.local_unknown.detach().cpu().numpy()[active]
                current["response_certainty_bin"][active] = packet.certainty_bin.detach().cpu().numpy()[active]
                current["response_reliability"][active] = packet.reliability.detach().cpu().numpy()[active]
                current["response_bits"][active] = packet.bit_cost.detach().cpu().numpy()[active]
            else:  # pragma: no cover - contract union is closed
                raise TypeError(f"unsupported transcript packet: {type(packet).__name__}")
        for name in names:
            rows[name].append(current[name])
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def _calibrated_rule(
    system: nn.Module,
    calibration_known,
    test_known,
    test_proxy_unknown,
    device: torch.device,
    *,
    route_action: RouteAction | int | str | None,
    protocol: str = "sequential",
    intervention: str | None = None,
    batch_size: int = 1024,
) -> dict[str, Any]:
    calibration = collect_predictions(
        system, calibration_known, device, route_action=route_action,
        protocol=protocol, intervention=intervention, batch_size=batch_size)
    known = collect_predictions(
        system, test_known, device, route_action=route_action,
        protocol=protocol, intervention=intervention, batch_size=batch_size)
    proxy = collect_predictions(
        system, test_proxy_unknown, device, route_action=route_action,
        protocol=protocol, intervention=intervention, batch_size=batch_size)
    combined = _merge_prediction_rows(known, proxy)
    transcript = _merge_prediction_rows(
        _collect_transcript(
            system, test_known, device, route_action=route_action,
            protocol=protocol, intervention=intervention,
            batch_size=batch_size),
        _collect_transcript(
            system, test_proxy_unknown, device, route_action=route_action,
            protocol=protocol, intervention=intervention,
            batch_size=batch_size),
    )
    calibrator = ClasswiseCalibrationService().fit(
        calibration["raw_unknown"], calibration["pred"],
        labels=calibration["y"], dataset=_dataset(calibration_known))
    calibrated = calibrator.predict(combined["raw_unknown"], combined["pred"])
    metrics = evaluate_open_set(
        combined["y"], combined["pred"], calibrated.unknown_score,
        calibrated.threshold)
    reject = calibrated.reject
    correct = np.where(
        combined["y"] < 0, reject,
        (~reject) & (combined["pred"] == combined["y"]))
    return {
        "metrics": metrics,
        "calibrator": calibrator.state_dict(),
        "transcript": transcript,
        "per_sample": {
            **combined,
            "calibrated_unknown": calibrated.unknown_score,
            "threshold": calibrated.threshold,
            "reject": reject,
            "correct": correct,
        },
    }


def evaluate_g1_fold(
    system: nn.Module,
    calibration_known,
    test_known,
    test_proxy_unknown,
    device: torch.device,
    *,
    fold_index: int,
    training_audit: Mapping[str, Any],
    prerequisites,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Evaluate frozen-message value with independent rule calibration."""

    require_prerequisite_artifacts("g1", prerequisites)
    _validate_eval_inputs(calibration_known, test_known, test_proxy_unknown)
    system.to(device).eval()
    rules: dict[str, dict[str, Any]] = {}
    specifications = {
        "b2": (RouteAction.STOP, "sequential", None),
        "w_first": (RouteAction.W_FIRST, "sequential", None),
        "p_first": (RouteAction.P_FIRST, "sequential", None),
        "sequential": ("reliability", "sequential", None),
        "parallel": ("reliability", "parallel", None),
    }
    for name, (action, protocol, intervention) in specifications.items():
        rules[name] = _calibrated_rule(
            system, calibration_known, test_known, test_proxy_unknown, device,
            route_action=action, protocol=protocol,
            intervention=intervention, batch_size=batch_size)
    for intervention in _INTERVENTIONS:
        rules[f"ablation_{intervention}"] = _calibrated_rule(
            system, calibration_known, test_known, test_proxy_unknown, device,
            route_action="reliability", protocol="sequential",
            intervention=intervention, batch_size=batch_size)

    baseline = rules["b2"]["per_sample"]
    sequential = rules["sequential"]["per_sample"]
    queried = sequential["route"] != int(RouteAction.STOP)
    before_error = (~baseline["correct"]).astype(np.float64)
    after_error = (~sequential["correct"]).astype(np.float64)
    communication = communication_effect_report(
        baseline["correct"], sequential["correct"], queried,
        receiver_before_loss=before_error, receiver_after_loss=after_error)
    receiver_gain_samples = (before_error - after_error)[queried]
    parallel_equal_bandwidth = np.array_equal(
        rules["sequential"]["per_sample"]["bits"],
        rules["parallel"]["per_sample"]["bits"])
    frozen_agents_identical = all(bool(training_audit.get(key, False)) for key in (
        "local_decision_core_identical", "b2_identical", "router_identical"))
    seq_h = float(rules["sequential"]["metrics"]["h_score"])
    ablation_drop = {
        name.removeprefix("ablation_"): seq_h - float(row["metrics"]["h_score"])
        for name, row in rules.items() if name.startswith("ablation_")
    }
    return {
        "fold_index": int(fold_index),
        "frozen_agents_identical": frozen_agents_identical,
        "parallel_equal_bandwidth": bool(parallel_equal_bandwidth),
        "rules": rules,
        "communication": communication,
        "receiver_gain_samples": receiver_gain_samples,
        "ablation_h_drop": ablation_drop,
        "receiver_loss_definition": "independently-calibrated open-set 0/1 error",
        "formal_unknown_used": False,
    }


def build_g1_gate_metrics(fold_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(fold_reports) != 5:
        raise ValueError("G1 requires exactly five outer-fold reports")
    ordered = sorted(fold_reports, key=lambda row: int(row["fold_index"]))
    if [int(row["fold_index"]) for row in ordered] != list(range(5)):
        raise ValueError("G1 fold indices must be exactly 0..4")

    def metric(rule: str, key: str) -> np.ndarray:
        return np.asarray([
            float(row["rules"][rule]["metrics"][key]) for row in ordered
        ], dtype=np.float64)

    receiver = np.concatenate([
        np.asarray(row["receiver_gain_samples"], dtype=np.float64).reshape(-1)
        for row in ordered
    ])
    if len(receiver) > 1:
        receiver_ci_low = float(
            receiver.mean() - 1.96 * receiver.std(ddof=1) / np.sqrt(len(receiver)))
    else:
        receiver_ci_low = 0.0
    ablation_names = sorted(set.intersection(*[
        set(row["ablation_h_drop"]) for row in ordered
    ]))
    return {
        "frozen_agents_identical": all(
            bool(row["frozen_agents_identical"]) for row in ordered),
        "parallel_equal_bandwidth": all(
            bool(row["parallel_equal_bandwidth"]) for row in ordered),
        "b2_h_score": metric("b2", "h_score"),
        "sequential_h_score": metric("sequential", "h_score"),
        "parallel_h_score": metric("parallel", "h_score"),
        "b2_oscr": metric("b2", "oscr"),
        "sequential_oscr": metric("sequential", "oscr"),
        "parallel_oscr": metric("parallel", "oscr"),
        "b2_known_accuracy": metric("b2", "known_accuracy"),
        "sequential_known_accuracy": metric("sequential", "known_accuracy"),
        "rescue_harm_ratio": np.asarray([
            float(row["communication"]["rescue_harm_ratio"]) for row in ordered
        ], dtype=np.float64),
        "receiver_gain_ci_low": np.asarray([receiver_ci_low], dtype=np.float64),
        "ablation_h_drop": {
            name: np.asarray([
                float(row["ablation_h_drop"][name]) for row in ordered
            ], dtype=np.float64)
            for name in ablation_names
        },
    }


def evaluate_g1_gate(
    fold_reports: Sequence[Mapping[str, Any]],
    *,
    prerequisites,
) -> GateResult:
    require_prerequisite_artifacts("g1", prerequisites)
    return evaluate_g1(build_g1_gate_metrics(fold_reports),
                       prerequisites=prerequisites)


def _configure_g2_trainability(system: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for parameter in system.router.parameters():
        parameter.requires_grad_(True)
    names = [name for name, parameter in system.named_parameters()
             if parameter.requires_grad]
    return list(system.router.parameters()), names


def _non_router_snapshot(system: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
        if not name.startswith("router.")
    }


def _non_router_identical(before: Mapping[str, torch.Tensor], system: nn.Module) -> bool:
    after = _non_router_snapshot(system)
    return before.keys() == after.keys() and all(
        torch.equal(before[name], after[name]) for name in before)


def train_g2_router(
    system: nn.Module,
    train_known,
    train_proxy_unknown,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    prerequisites,
    pug_dataset=None,
) -> dict[str, Any]:
    """Train only the router from exact counterfactual action labels."""

    # Check persisted evidence before allocating a loader/optimizer.
    require_prerequisite_artifacts("g2", prerequisites)
    _validate_training_inputs(train_known, train_proxy_unknown, pug_dataset)
    if train_proxy_unknown is None:
        raise RuntimeError("G2 routing requires a registered inner proxy-unknown set")
    set_seed(seed)
    system.to(device)
    parameters, names = _configure_g2_trainability(system)
    fixed_before = _non_router_snapshot(system)
    epochs = int(cfg.get("router_epochs", 5))
    batch_size = int(cfg.get("batch_size", 256))
    if epochs < 1:
        raise ValueError("router_epochs must be positive")
    optimizer = torch.optim.AdamW(
        parameters, lr=float(cfg.get("router_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    supervisor = ExactCounterfactualSupervisor(
        communication_cost=float(cfg.get("communication_cost", 0.01)),
        reference_bits=float(cfg.get("reference_bits", 64.0)),
        minimum_gain=float(cfg.get("minimum_gain", 0.0)),
        class_weight=float(cfg.get("class_loss_weight", 1.0)),
        open_weight=float(cfg.get("open_loss_weight", 1.0)))
    loader = _weighted_training_loader(
        train_known, train_proxy_unknown, pug_dataset,
        batch_size=batch_size, seed=seed + 2903,
        proxy_weight=float(cfg.get("proxy_unknown_weight", 1.0)),
        pug_weight=float(cfg.get("pug_auxiliary_weight", 0.25)))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        # Exact action targets must be computed with a deterministic, frozen
        # consultation system.  Only the public-summary router is in train mode.
        system.eval()
        system.router.train()
        totals = {"loss": 0.0, "accuracy": 0.0}
        target_counts = np.zeros(3, dtype=np.int64)
        prediction_counts = np.zeros(3, dtype=np.int64)
        seen = 0
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                local = system.local(x)
                supervision, _ = supervisor.enumerate(system, local, labels)
            logits = system.route_logits(local)
            weights = balanced_action_weights(supervision.targets).to(device)
            loss = F.cross_entropy(logits, supervision.targets, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg.get("clip_grad_norm", 5.0)))
            optimizer.step()
            prediction = logits.argmax(1)
            batch = len(labels)
            totals["loss"] += float(loss.detach()) * batch
            totals["accuracy"] += int((prediction == supervision.targets).sum())
            target_counts += np.bincount(
                supervision.targets.cpu().numpy(), minlength=3)
            prediction_counts += np.bincount(
                prediction.detach().cpu().numpy(), minlength=3)
            seen += batch
        row = {
            "epoch": float(epoch),
            "loss": totals["loss"] / max(seen, 1),
            "accuracy": totals["accuracy"] / max(seen, 1),
        }
        for index, action in enumerate((
                RouteAction.STOP, RouteAction.W_FIRST, RouteAction.P_FIRST)):
            row[f"target_{action.name.lower()}_rate"] = (
                target_counts[index] / max(seen, 1))
            row[f"predicted_{action.name.lower()}_rate"] = (
                prediction_counts[index] / max(seen, 1))
        history.append(row)
    fixed_identical = _non_router_identical(fixed_before, system)
    if not fixed_identical:
        raise AssertionError("G2 modified the frozen Agents/B2/responders")
    return {
        "history": history,
        "audit": {
            "exact_action_enumeration": True,
            "non_router_state_identical": True,
            "trainable_parameters": names,
            "formal_unknown_used": False,
        },
        "supervisor": {
            "communication_cost": supervisor.communication_cost,
            "reference_bits": supervisor.reference_bits,
            "minimum_gain": supervisor.minimum_gain,
        },
    }


def evaluate_g2_fold(
    system: nn.Module,
    calibration_known,
    test_known,
    test_proxy_unknown,
    device: torch.device,
    *,
    fold_index: int,
    supervisor: ExactCounterfactualSupervisor,
    prerequisites,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Evaluate routed decisions and their exact counterfactual headroom."""

    require_prerequisite_artifacts("g2", prerequisites)
    _validate_eval_inputs(calibration_known, test_known, test_proxy_unknown)
    system.to(device).eval()
    b2 = _calibrated_rule(
        system, calibration_known, test_known, test_proxy_unknown, device,
        route_action=RouteAction.STOP, batch_size=batch_size)
    routed = _calibrated_rule(
        system, calibration_known, test_known, test_proxy_unknown, device,
        route_action=None, batch_size=batch_size)
    combined_dataset = ConcatDataset([
        _dataset(test_known), _dataset(test_proxy_unknown)])
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    counterfactual = collect_counterfactual_supervision(
        system, loader, supervisor, device, dataset=(
            _dataset(test_known), _dataset(test_proxy_unknown)))
    diagnostics = router_diagnostics(
        counterfactual["action_losses"], counterfactual["route_logits"])
    queried = routed["per_sample"]["route"] != int(RouteAction.STOP)
    communication = communication_effect_report(
        b2["per_sample"]["correct"], routed["per_sample"]["correct"], queried)
    return {
        "fold_index": int(fold_index),
        "b2": b2,
        "routed": routed,
        "counterfactual": counterfactual,
        "diagnostics": diagnostics,
        "communication": communication,
        "formal_unknown_used": False,
    }


def build_g2_gate_metrics(
    fold_reports: Sequence[Mapping[str, Any]],
    *,
    direction_rates_by_dataset: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if len(fold_reports) != 5:
        raise ValueError("G2 requires exactly five outer-fold reports")
    ordered = sorted(fold_reports, key=lambda row: int(row["fold_index"]))
    if [int(row["fold_index"]) for row in ordered] != list(range(5)):
        raise ValueError("G2 fold indices must be exactly 0..4")
    metrics: dict[str, Any] = {
        "exact_action_enumeration": True,
        "query_rate": np.asarray([
            float(row["diagnostics"]["query_rate"]) for row in ordered]),
        "oracle_gain_recovery": np.asarray([
            float(row["diagnostics"]["oracle_gain_recovery"]) for row in ordered]),
        "gain_spearman": np.asarray([
            float(row["diagnostics"]["gain_spearman"]) for row in ordered]),
        "rescue_harm_ratio": np.asarray([
            float(row["communication"]["rescue_harm_ratio"]) for row in ordered]),
    }
    if direction_rates_by_dataset is not None:
        metrics["direction_rates_by_dataset"] = copy.deepcopy(
            dict(direction_rates_by_dataset))
    return metrics


def dataset_direction_rates(fold_reports: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not fold_reports:
        raise ValueError("direction-rate aggregation requires fold reports")
    return {
        "w_first": float(np.mean([
            row["diagnostics"]["w_first_rate"] for row in fold_reports])),
        "p_first": float(np.mean([
            row["diagnostics"]["p_first_rate"] for row in fold_reports])),
    }


def evaluate_g2_gate(
    fold_reports: Sequence[Mapping[str, Any]],
    *,
    prerequisites,
    direction_rates_by_dataset: Mapping[str, Mapping[str, float]] | None = None,
) -> GateResult:
    require_prerequisite_artifacts("g2", prerequisites)
    metrics = build_g2_gate_metrics(
        fold_reports, direction_rates_by_dataset=direction_rates_by_dataset)
    return evaluate_g2(metrics, prerequisites=prerequisites)


__all__ = [
    "build_g1_gate_metrics", "build_g2_gate_metrics",
    "dataset_direction_rates", "evaluate_g1_fold", "evaluate_g1_gate",
    "evaluate_g2_fold", "evaluate_g2_gate", "train_g1_responses",
    "train_g2_router",
]
