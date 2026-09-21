"""Small, leakage-safe G0/G1 probe for Temporal/Spectral specialization."""
from __future__ import annotations

import json
import random
import copy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from maros_staged.datasets import load_oracle_npz, load_wisig_subset

from .audit import capability_audit
from .evaluation import (open_set_rescue_metrics, pair_verification_metrics,
                         rescue_decomposition)
from .gates import (evaluate_ab_g1_gate, evaluate_directional_g15_gate)
from .observations import (build_coarse_spectral_observation,
                           build_robust_spectral_observation_v3,
                           build_spectral_observation,
                           build_temporal_observation,
                           build_temporal_patch_observation)
from .splits import (assert_no_formal_unknown, build_nested_lco_protocol,
                     protocol_manifest)
from .system import Stage8System
from .training import (local_specialization_objective,
                       refresh_candidate_prototypes)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _cap(dataset: Dataset, per_class: int, seed: int) -> Dataset:
    if per_class <= 0:
        return dataset
    if hasattr(dataset, "original_labels"):
        labels = np.asarray(dataset.original_labels, dtype=np.int64)
    elif hasattr(dataset, "y"):
        labels = np.asarray(dataset.y, dtype=np.int64)
    else:
        raise TypeError("sample capping requires dataset labels")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        chosen.extend(indices[:per_class].tolist())
    return Subset(dataset, sorted(chosen))


def _train(system: Stage8System, dataset: Dataset, cfg: Mapping,
           device: torch.device, seed: int) -> list[dict[str, float]]:
    assert_no_formal_unknown(dataset)
    training_dataset = _cap(
        dataset, int(cfg["train_samples_per_class"]), seed)
    loader = DataLoader(
        training_dataset,
        batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(
        system.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]))
    history = []
    for epoch in range(int(cfg["epochs"])):
        verification_active = epoch >= int(cfg.get("prototype_warmup_epochs", 0))
        refresh_interval = max(1, int(cfg.get("prototype_refresh_interval", 1)))
        warmup = int(cfg.get("prototype_warmup_epochs", 0))
        if verification_active and (epoch == warmup
                                    or (epoch - warmup) % refresh_interval == 0):
            refresh_candidate_prototypes(
                system, training_dataset, device,
                batch_size=int(cfg["batch_size"]))
        system.train()
        total_loss = temporal_correct = spectral_correct = seen = 0
        pieces_total = {
            name: {key: 0.0 for key in (
                "classification_loss", "positive_loss",
                "hard_negative_loss", "pair_margin_loss", "pair_accuracy")}
            for name in ("temporal", "spectral")}
        for iq, labels in loader:
            iq, labels = iq.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            temporal_observation, spectral_observation = (
                system.build_observations(iq))
            loss, pieces = local_specialization_objective(
                system, temporal_observation, spectral_observation, labels,
                verification_weight=(float(cfg["verification_weight"])
                                     if verification_active else 0.0),
                pair_margin=float(cfg["pair_margin"]))
            loss.backward()
            optimizer.step()
            batch = len(labels)
            total_loss += float(loss.detach()) * batch
            temporal_correct += int(
                pieces["decisions"]["temporal"].top1.eq(labels).sum())
            spectral_correct += int(
                pieces["decisions"]["spectral"].top1.eq(labels).sum())
            for name in ("temporal", "spectral"):
                for key in pieces_total[name]:
                    pieces_total[name][key] += float(
                        getattr(pieces[name], key).detach()) * batch
            seen += batch
        row = {
            "epoch": float(epoch + 1), "loss": total_loss / max(seen, 1),
            "verification_active": bool(verification_active),
            "temporal_train_accuracy": temporal_correct / max(seen, 1),
            "spectral_train_accuracy": spectral_correct / max(seen, 1),
        }
        for name in ("temporal", "spectral"):
            for key, value in pieces_total[name].items():
                row[f"{name}_{key}"] = value / max(seen, 1)
        history.append(row)
    refresh_candidate_prototypes(
        system, training_dataset, device, batch_size=int(cfg["batch_size"]))
    return history


@torch.no_grad()
def _collect(system: Stage8System, dataset: Dataset, cfg: Mapping,
             device: torch.device, seed: int) -> dict[str, np.ndarray]:
    assert_no_formal_unknown(dataset)
    loader = DataLoader(
        _cap(dataset, int(cfg["eval_samples_per_class"]), seed),
        batch_size=int(cfg["batch_size"]), shuffle=False)
    result: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "labels", "temporal_prediction", "spectral_prediction",
            "temporal_unknown", "spectral_unknown", "features",
            "temporal_support", "spectral_support",
            "temporal_hard_negative", "spectral_hard_negative")}
    system.eval()
    for iq, labels in loader:
        context = system.observe(iq.to(device))
        temporal = context["temporal"]
        spectral = context["spectral"]
        labels_device = labels.long().to(device)
        safe_labels = labels_device.clamp_min(0)
        temporal_masked = temporal.class_logits.detach().clone()
        spectral_masked = spectral.class_logits.detach().clone()
        temporal_masked.scatter_(1, safe_labels[:, None], float("-inf"))
        spectral_masked.scatter_(1, safe_labels[:, None], float("-inf"))
        temporal_hard = temporal_masked.argmax(1)
        spectral_hard = spectral_masked.argmax(1)
        temporal_support = system.all_candidate_support(context, "temporal")
        spectral_support = system.all_candidate_support(context, "spectral")
        common = torch.abs(
            temporal.public_diagnostics[:, :5]
            - spectral.public_diagnostics[:, :5])
        features = torch.cat([
            temporal.public_diagnostics, spectral.public_diagnostics, common,
        ], dim=1)
        values = {
            "labels": labels,
            "temporal_prediction": temporal.top1,
            "spectral_prediction": spectral.top1,
            "temporal_unknown": temporal.unknown_score,
            "spectral_unknown": spectral.unknown_score,
            "features": features,
            "temporal_support": temporal_support,
            "spectral_support": spectral_support,
            "temporal_hard_negative": temporal_hard,
            "spectral_hard_negative": spectral_hard,
        }
        for key, value in values.items():
            result[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def _threshold(calibration: dict[str, np.ndarray], name: str,
               acceptance: float) -> float:
    return float(np.quantile(calibration[f"{name}_unknown"], acceptance))


def _final_predictions(rows: dict[str, np.ndarray], name: str,
                       threshold: float) -> np.ndarray:
    prediction = rows[f"{name}_prediction"].copy()
    prediction[rows[f"{name}_unknown"] >= threshold] = -1
    return prediction


def _macro_auroc(train_x, train_y, test_x, test_y) -> float:
    classes = np.unique(train_y)
    if len(classes) < 2:
        return 0.5
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=0))
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)
    fitted_classes = model[-1].classes_
    values = []
    for column, label in enumerate(fitted_classes):
        target = test_y == label
        if target.any() and (~target).any():
            values.append(roc_auc_score(target, probabilities[:, column]))
    return float(np.mean(values)) if values else 0.5


def _binary_predictability(train_x, train_y, test_x, test_y) -> dict[str, float]:
    train_y = np.asarray(train_y, dtype=np.int64)
    test_y = np.asarray(test_y, dtype=np.int64)
    support = int(test_y.sum())
    positive_rate = float(test_y.mean()) if len(test_y) else 0.0
    if len(np.unique(train_y)) < 2:
        probability = np.full(len(test_y), float(train_y[0]) if len(train_y) else 0.0)
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=0))
        model.fit(train_x, train_y)
        class_index = list(model[-1].classes_).index(1)
        probability = model.predict_proba(test_x)[:, class_index]
    auroc = (float(roc_auc_score(test_y, probability))
             if len(np.unique(test_y)) == 2 else 0.5)
    pr_auc = (float(average_precision_score(test_y, probability))
              if support else 0.0)
    return {
        "auroc": auroc, "pr_auc": pr_auc,
        "support": float(support), "positive_rate": positive_rate,
    }


def _load(cfg: Mapping):
    if cfg["dataset"] == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=False)
    if cfg["dataset"] == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=False)
    raise ValueError("dataset must be wisig or oracle")


@torch.no_grad()
def _observation_probe_matrices(
    dataset: Dataset, profile: str, cfg: Mapping,
    device: torch.device, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lossy-view inputs and cross-domain reconstruction targets."""

    assert_no_formal_unknown(dataset)
    capped = _cap(
        dataset, int(cfg.get("leakage_samples_per_class", 32)), seed)
    loader = DataLoader(
        capped, batch_size=int(cfg["batch_size"]), shuffle=False)
    temporal_inputs = []
    spectral_inputs = []
    spectral_targets = []
    temporal_targets = []
    for iq, _ in loader:
        iq = iq.to(device)
        temporal_target = build_temporal_patch_observation(iq)
        spectral_target = build_coarse_spectral_observation(iq)
        if profile == "legacy_v1":
            temporal = build_temporal_observation(iq)
            spectral = build_spectral_observation(iq)
            temporal_values = temporal.iq
            spectral_values = F.adaptive_avg_pool1d(
                spectral.features, 64)
        elif profile == "isolated_v2":
            temporal = temporal_target
            spectral = spectral_target
            temporal_values = temporal.features
            spectral_values = spectral.features
        else:
            raise ValueError("unsupported leakage observation profile")
        temporal_inputs.append(temporal_values.flatten(1).cpu().numpy())
        spectral_inputs.append(torch.cat([
            spectral_values.flatten(1), spectral.quality,
        ], dim=1).cpu().numpy())
        spectral_targets.append(torch.cat([
            F.adaptive_avg_pool1d(spectral_target.features, 8).flatten(1),
            spectral_target.quality,
        ], dim=1).cpu().numpy())
        temporal_targets.append(torch.cat([
            F.adaptive_avg_pool1d(temporal_target.features, 8).flatten(1),
            temporal_target.quality,
        ], dim=1).cpu().numpy())
    return tuple(np.concatenate(values) for values in (
        temporal_inputs, spectral_inputs, spectral_targets, temporal_targets))


def _regression_probe(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray, seed: int,
) -> dict[str, float]:
    model = ExtraTreesRegressor(
        n_estimators=48, max_depth=12, min_samples_leaf=2,
        max_features=0.75, random_state=seed, n_jobs=-1)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    mse = float(mean_squared_error(test_y, prediction))
    target_variance = float(np.var(test_y))
    return {
        "r2": float(r2_score(test_y, prediction, multioutput="variance_weighted")),
        "normalized_rmse": float(np.sqrt(mse / max(target_variance, 1e-12))),
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "input_dim": int(train_x.shape[1]),
        "target_dim": int(train_y.shape[1]),
    }


def observation_leakage_audit(
    train_dataset: Dataset, test_dataset: Dataset, cfg: Mapping,
    device: torch.device, seed: int,
) -> dict:
    """Quantify cross-view predictability; this is an audit, not a gate."""

    requested = cfg.get("observation_profile", "legacy_v1")
    profiles = ["legacy_v1"]
    if requested == "isolated_v2":
        profiles.append("isolated_v2")
    rows = {}
    for index, profile in enumerate(profiles):
        train_t, train_s, train_st, train_tt = _observation_probe_matrices(
            train_dataset, profile, cfg, device, seed + index)
        test_t, test_s, test_st, test_tt = _observation_probe_matrices(
            test_dataset, profile, cfg, device, seed + 101 + index)
        rows[profile] = {
            "temporal_to_spectral_descriptor": _regression_probe(
                train_t, train_st, test_t, test_st, seed + index),
            "spectral_to_temporal_order_descriptor": _regression_probe(
                train_s, train_tt, test_s, test_tt, seed + 17 + index),
        }
    if "isolated_v2" in rows:
        for direction in (
                "temporal_to_spectral_descriptor",
                "spectral_to_temporal_order_descriptor"):
            rows["isolated_v2"][direction]["r2_change_vs_legacy_v1"] = float(
                rows["isolated_v2"][direction]["r2"]
                - rows["legacy_v1"][direction]["r2"])
    return {
        "formal_unknown_used": False,
        "interpretation": (
            "Lower held-out R2 indicates stronger information isolation; "
            "the audit is diagnostic and has no pass threshold."),
        "profiles": rows,
    }


@torch.no_grad()
def _collect_agent_b_diagnostics(
    system: Stage8System, dataset: Dataset, cfg: Mapping,
    device: torch.device, seed: int, *, training_cap: bool,
) -> dict[str, np.ndarray]:
    assert_no_formal_unknown(dataset)
    cap = (int(cfg["train_samples_per_class"]) if training_cap
           else int(cfg["eval_samples_per_class"]))
    loader = DataLoader(
        _cap(dataset, cap, seed), batch_size=int(cfg["batch_size"]),
        shuffle=False)
    values = {key: [] for key in (
        "labels", "temporal_prediction", "spectral_prediction",
        "temporal_support", "spectral_support",
        "temporal_hard_negative", "spectral_hard_negative",
        "spectral_state")}
    system.eval()
    for iq, labels in loader:
        iq = iq.to(device)
        labels_device = labels.long().to(device)
        temporal_observation, spectral_observation = (
            system.build_observations(iq))
        temporal, temporal_private = system.temporal.forward_private(
            temporal_observation)
        spectral, spectral_private = system.spectral.forward_private(
            spectral_observation)
        temporal_logits = temporal.class_logits.detach().clone()
        spectral_logits = spectral.class_logits.detach().clone()
        temporal_logits.scatter_(1, labels_device[:, None], float("-inf"))
        spectral_logits.scatter_(1, labels_device[:, None], float("-inf"))
        batch = {
            "labels": labels_device,
            "temporal_prediction": temporal.top1,
            "spectral_prediction": spectral.top1,
            "temporal_support": system.temporal.all_verification_scores(
                temporal_private),
            "spectral_support": system.spectral.all_verification_scores(
                spectral_private),
            "temporal_hard_negative": temporal_logits.argmax(1),
            "spectral_hard_negative": spectral_logits.argmax(1),
            "spectral_state": F.normalize(spectral_private.state, dim=1),
        }
        for key, value in batch.items():
            values[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(rows) for key, rows in values.items()}


def _state_geometry(
    states: np.ndarray, labels: np.ndarray, prototypes: np.ndarray,
) -> dict[str, float]:
    states_t = F.normalize(torch.as_tensor(states), dim=1)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    prototypes_t = F.normalize(torch.as_tensor(prototypes), dim=1)
    centers = []
    intra = []
    for label in range(len(prototypes_t)):
        mask = labels_t == label
        if not bool(mask.any()):
            raise ValueError("state geometry requires every support class")
        center = F.normalize(states_t[mask].mean(0), dim=0)
        centers.append(center)
        intra.append(torch.linalg.vector_norm(states_t[mask] - center, dim=1))
    centers_t = torch.stack(centers)
    pairwise = torch.cdist(centers_t, centers_t)
    off_diagonal = ~torch.eye(
        len(centers_t), dtype=torch.bool, device=pairwise.device)
    inter = pairwise[off_diagonal].mean()
    intra_mean = torch.cat(intra).mean()
    prototype_distances = torch.cdist(states_t, prototypes_t)
    rows = torch.arange(len(labels_t))
    true_distance = prototype_distances[rows, labels_t]
    wrong = prototype_distances.clone()
    wrong[rows, labels_t] = float("inf")
    nearest_wrong = wrong.amin(1)
    centroid_drift = torch.linalg.vector_norm(
        centers_t - prototypes_t, dim=1).mean()
    return {
        "intra_class_distance": float(intra_mean),
        "inter_class_distance": float(inter),
        "prototype_generalization_ratio": float(
            inter / intra_mean.clamp_min(1e-8)),
        "prototype_to_sample_distance": float(true_distance.mean()),
        "nearest_class_margin": float((nearest_wrong - true_distance).mean()),
        "prototype_to_eval_centroid_drift": float(centroid_drift),
    }


@torch.no_grad()
def _agent_b_nuisance_stability(
    system: Stage8System, dataset: Dataset, cfg: Mapping,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=min(int(cfg["batch_size"]), 64))
    iq, _ = next(iter(loader))
    iq = iq.to(device)
    sample_index = torch.arange(iq.shape[-1], device=device, dtype=iq.dtype)
    cfo_rotation = torch.exp(
        1j * 2.0 * torch.pi * sample_index / 64.0)
    z = torch.complex(iq[:, 0], iq[:, 1]) * cfo_rotation
    cfo_iq = torch.stack([z.real, z.imag], dim=1)

    def observation(profile: str, values: torch.Tensor):
        if profile == "legacy_v1":
            return build_spectral_observation(values)
        return build_robust_spectral_observation_v3(
            values, align=profile == "robust_v3")

    def state(agent, profile: str, values: torch.Tensor) -> torch.Tensor:
        agent.observation_profile = profile
        return F.normalize(
            agent.forward_private(observation(profile, values))[1].state,
            dim=1)

    rows = {}
    for profile in ("legacy_v1", "robust_v3_noalign", "robust_v3"):
        agent = copy.deepcopy(system.spectral).eval()
        base = state(agent, profile, iq)
        time_drifts = []
        for shift in (5, 11):
            changed = state(agent, profile, torch.roll(iq, shift, dims=-1))
            time_drifts.append(float((
                1.0 - F.cosine_similarity(base, changed, dim=1)).mean()))
        cfo_state = state(agent, profile, cfo_iq)
        rows[profile] = {
            "time_shift_state_drift": float(np.mean(time_drifts)),
            "small_cfo_state_drift": float((
                1.0 - F.cosine_similarity(base, cfo_state, dim=1)).mean()),
        }
    return rows


def run_agent_b_v3_quick(cfg: Mapping) -> dict:
    """Run only ORACLE inner fold 0 and fail fast on the registered gate."""

    if cfg["dataset"] != "oracle":
        raise ValueError("Agent-B v3 phase 1 is ORACLE fold-0 only")
    if cfg.get("temporal_observation_profile") != "legacy_v1":
        raise ValueError("Agent-B v3 must freeze Temporal at legacy_v1")
    if cfg.get("spectral_observation_profile") != "robust_v3":
        raise ValueError("Agent-B v3 quick validation requires robust_v3")
    seed = int(cfg["seed"])
    _seed(seed)
    splits = _load(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg["outer_folds"]), inner_folds=4,
        seed=int(cfg["partition_seed"]))
    episode = protocol.outer_folds[int(cfg["outer_fold"])].inner_folds[0]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
        else "cpu")
    fold_seed = seed + 1009
    _seed(fold_seed)
    system = Stage8System(
        len(episode.known_classes), state_dim=int(cfg["state_dim"]),
        hidden_channels=int(cfg["hidden_channels"]),
        observation_profile="legacy_v1",
        temporal_observation_profile="legacy_v1",
        spectral_observation_profile="robust_v3").to(device)
    history = _train(system, episode.train_known, cfg, device, fold_seed)
    train = _collect_agent_b_diagnostics(
        system, episode.train_known, cfg, device, fold_seed,
        training_cap=True)
    evaluation = _collect_agent_b_diagnostics(
        system, episode.test_known, cfg, device, fold_seed + 2,
        training_cap=False)
    spectral_pair = pair_verification_metrics(
        evaluation["spectral_support"], evaluation["labels"],
        evaluation["spectral_hard_negative"])
    temporal_pair = pair_verification_metrics(
        evaluation["temporal_support"], evaluation["labels"],
        evaluation["temporal_hard_negative"])
    temporal_ok = evaluation["temporal_prediction"] == evaluation["labels"]
    spectral_ok = evaluation["spectral_prediction"] == evaluation["labels"]
    rows = np.arange(len(evaluation["labels"]))
    labels = evaluation["labels"]
    temporal_true = evaluation["temporal_support"][rows, labels]
    spectral_true = evaluation["spectral_support"][rows, labels]
    temporal_hard = evaluation["temporal_hard_negative"]
    spectral_on_temporal = spectral_true > evaluation[
        "spectral_support"][rows, temporal_hard]
    temporal_own = temporal_true > evaluation[
        "temporal_support"][rows, temporal_hard]
    spectral_accuracy = float(spectral_ok.mean())
    pair_auroc = float(spectral_pair["pair_auroc"])
    worth_continue = spectral_accuracy >= 0.35 and pair_auroc >= 0.65
    early_stop = spectral_accuracy < 0.25 or pair_auroc < 0.60
    conclusion = (
        "AGENT-B-V3-FOLD0-PROMISING" if worth_continue
        else "SPECTRAL-PRIVATE-TASK-REDESIGN-REQUIRED")
    prototypes = system.spectral.candidate_prototypes.detach().cpu().numpy()
    report = {
        "stage": "stage8_agent_b_v3_quick_validation",
        "dataset": "oracle", "inner_fold": 0, "device": str(device),
        "formal_unknown_used": False,
        "temporal_frozen_contract": {
            "observation_profile": "legacy_v1",
            "state_dim": int(cfg["state_dim"]),
            "hidden_channels": int(cfg["hidden_channels"]),
            "epochs": int(cfg["epochs"]),
            "config_matches_v1_training_budget": True,
        },
        "spectral_profile": "robust_v3",
        "spectral_train_accuracy": float(
            (train["spectral_prediction"] == train["labels"]).mean()),
        "spectral_eval_accuracy": spectral_accuracy,
        "spectral_train_eval_gap": float(
            (train["spectral_prediction"] == train["labels"]).mean()
            - spectral_accuracy),
        "temporal_eval_accuracy": float(temporal_ok.mean()),
        "spectral_pair_metrics": spectral_pair,
        "temporal_pair_metrics": temporal_pair,
        "spectral_to_temporal_identity_rescue": float(
            (spectral_ok & ~temporal_ok).mean()),
        "spectral_to_temporal_pair_rescue": float(
            (spectral_on_temporal & ~temporal_own).mean()),
        "prototype_generalization": {
            "train": _state_geometry(
                train["spectral_state"], train["labels"], prototypes),
            "eval": _state_geometry(
                evaluation["spectral_state"], evaluation["labels"], prototypes),
        },
        "nuisance_stability": _agent_b_nuisance_stability(
            system, episode.test_known, cfg, device),
        "class_reindex_invariance": _reindex_invariance_audit(
            system, episode.test_known, cfg, device),
        "fold0_gate": {
            "early_stop": bool(early_stop),
            "worth_continuing": bool(worth_continue),
            "checks": {
                "spectral_eval_accuracy_ge_0_35": spectral_accuracy >= 0.35,
                "spectral_pair_auroc_ge_0_65": pair_auroc >= 0.65,
            },
        },
        "training": history,
        "conclusion": conclusion,
        "next_action": (
            "run_oracle_four_folds" if worth_continue
            else "stop_without_oracle_four_folds_or_wisig"),
    }
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_b_v3_fold0_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


@torch.no_grad()
def _reindex_invariance_audit(
    system: Stage8System, dataset: Dataset, cfg: Mapping,
    device: torch.device,
) -> dict[str, float | bool]:
    loader = DataLoader(dataset, batch_size=min(int(cfg["batch_size"]), 64))
    iq, _ = next(iter(loader))
    iq = iq.to(device)
    system.eval()
    original = system.observe(iq)
    changed_system = copy.deepcopy(system).eval()
    permutation = torch.randperm(system.num_classes, device=device)
    changed_system.reindex_classes(permutation)
    changed = changed_system.observe(iq)
    maximum_error = 0.0
    prediction_equal = True
    support_equal = True
    for name in ("temporal", "spectral"):
        expected_logits = torch.empty_like(original[name].class_logits)
        expected_logits[:, permutation] = original[name].class_logits
        maximum_error = max(
            maximum_error,
            float((changed[name].class_logits - expected_logits).abs().max()))
        prediction_equal = prediction_equal and bool(torch.equal(
            changed[name].top1, permutation[original[name].top1]))
        expected_support = torch.empty_like(
            system.all_candidate_support(original, name))
        expected_support[:, permutation] = system.all_candidate_support(
            original, name)
        actual_support = changed_system.all_candidate_support(changed, name)
        maximum_error = max(
            maximum_error, float((actual_support - expected_support).abs().max()))
        support_equal = support_equal and bool(torch.allclose(
            actual_support, expected_support, atol=1e-6))
    passed = prediction_equal and support_equal and maximum_error <= 1e-6
    return {
        "passed": bool(passed), "prediction_equal": prediction_equal,
        "support_equal": support_equal, "maximum_absolute_error": maximum_error,
    }


def run_local_learnability(cfg: Mapping) -> dict:
    """Overfit ORACLE/WiSig small Known sets before interpreting rescue."""

    seed = int(cfg["seed"])
    _seed(seed)
    splits = _load(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg["outer_folds"]), inner_folds=4,
        seed=int(cfg["partition_seed"]))
    episode = protocol.outer_folds[int(cfg["outer_fold"])].inner_folds[0]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
        else "cpu")
    candidates = []
    selected = None
    for candidate_index, profile in enumerate(cfg["overfit_candidates"]):
        runs = []
        for size_index, size in enumerate(cfg["small_set_sizes"]):
            run_seed = seed + 1009 * (candidate_index + 1) + 97 * (size_index + 1)
            _seed(run_seed)
            local_cfg = dict(cfg)
            local_cfg.update({
                "state_dim": int(profile["state_dim"]),
                "hidden_channels": int(profile["hidden_channels"]),
                "epochs": int(cfg["overfit_epochs"]),
                "prototype_warmup_epochs": int(
                    cfg["overfit_prototype_warmup_epochs"]),
                "learning_rate": float(cfg["overfit_learning_rate"]),
                "weight_decay": 0.0,
                "train_samples_per_class": int(size),
                "eval_samples_per_class": 0,
            })
            system = Stage8System(
                len(episode.known_classes),
                state_dim=local_cfg["state_dim"],
                hidden_channels=local_cfg["hidden_channels"],
                observation_profile=local_cfg.get(
                    "observation_profile", "legacy_v1"),
                temporal_observation_profile=local_cfg.get(
                    "temporal_observation_profile"),
                spectral_observation_profile=local_cfg.get(
                    "spectral_observation_profile")).to(device)
            history = _train(
                system, episode.train_known, local_cfg, device, run_seed)
            subset = _cap(episode.train_known, int(size), run_seed)
            rows = _collect(system, subset, local_cfg, device, run_seed)
            temporal_accuracy = float(
                (rows["temporal_prediction"] == rows["labels"]).mean())
            spectral_accuracy = float(
                (rows["spectral_prediction"] == rows["labels"]).mean())
            temporal_pair = pair_verification_metrics(
                rows["temporal_support"], rows["labels"],
                rows["temporal_hard_negative"])
            spectral_pair = pair_verification_metrics(
                rows["spectral_support"], rows["labels"],
                rows["spectral_hard_negative"])
            target = float(cfg["overfit_target_accuracy"])
            runs.append({
                "samples_per_class": int(size),
                "num_samples": int(len(rows["labels"])),
                "temporal_train_accuracy": temporal_accuracy,
                "spectral_train_accuracy": spectral_accuracy,
                "temporal_pair_metrics": temporal_pair,
                "spectral_pair_metrics": spectral_pair,
                "passed": temporal_accuracy >= target and spectral_accuracy >= target,
                "history": history,
            })
            print(
                f"[stage8:learnability:{profile['name']}] {size}/class "
                f"T/S={temporal_accuracy:.4f}/{spectral_accuracy:.4f}",
                flush=True)
        candidate_passed = all(run["passed"] for run in runs)
        candidates.append({
            "name": profile["name"],
            "state_dim": int(profile["state_dim"]),
            "hidden_channels": int(profile["hidden_channels"]),
            "passed": candidate_passed, "runs": runs,
        })
        if candidate_passed:
            selected = profile["name"]
            break
    report = {
        "stage": "stage8_local_learnability",
        "dataset": cfg["dataset"], "device": str(device),
        "formal_unknown_used": False,
        "formal_unknown_sample_count": protocol.formal_unknown_sample_count,
        "target_accuracy": float(cfg["overfit_target_accuracy"]),
        "candidates": candidates, "selected_candidate": selected,
        "passed": selected is not None,
        "outcome": "PASS" if selected is not None else "STOP-AND-REDESIGN",
        "decision": "local_learnability_pass" if selected is not None
        else "stop_and_fix_architecture_or_optimizer",
    }
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "local_learnability_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_g1_probe(cfg: Mapping) -> dict:
    """Run four tiny inner LCO episodes without touching formal Unknown."""

    seed = int(cfg["seed"])
    _seed(seed)
    splits = _load(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg["outer_folds"]),
        inner_folds=4, seed=int(cfg["partition_seed"]))
    outer = protocol.outer_folds[int(cfg["outer_fold"])]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu"
        else "cpu")
    leakage_audit = observation_leakage_audit(
        outer.inner_folds[0].train_known,
        outer.inner_folds[0].test_known,
        cfg, device, seed + 7001)
    fold_results = []
    predictor_rows = []
    for episode in outer.inner_folds:
        fold_seed = seed + 1009 * (episode.inner_index + 1)
        print(
            f"[stage8:{cfg['dataset']}:{cfg.get('observation_profile', 'legacy_v1')}] "
            f"inner fold {episode.inner_index + 1}/{len(outer.inner_folds)}",
            flush=True)
        _seed(fold_seed)
        system = Stage8System(
            len(episode.known_classes), state_dim=int(cfg["state_dim"]),
            hidden_channels=int(cfg["hidden_channels"]),
            observation_profile=cfg.get(
                "observation_profile", "legacy_v1"),
            temporal_observation_profile=cfg.get(
                "temporal_observation_profile"),
            spectral_observation_profile=cfg.get(
                "spectral_observation_profile")).to(device)
        history = _train(system, episode.train_known, cfg, device, fold_seed)
        reindex_audit = _reindex_invariance_audit(
            system, episode.test_known, cfg, device)
        calibration = _collect(
            system, episode.calibration_known, cfg, device, fold_seed + 1)
        known = _collect(system, episode.test_known, cfg, device, fold_seed + 2)
        proxy = _collect(
            system, episode.test_proxy_unknown, cfg, device, fold_seed + 3)
        combined = {
            key: np.concatenate([known[key], proxy[key]])
            for key in known}
        temporal_threshold = _threshold(
            calibration, "temporal", float(cfg["known_acceptance"]))
        spectral_threshold = _threshold(
            calibration, "spectral", float(cfg["known_acceptance"]))
        temporal_prediction = _final_predictions(
            combined, "temporal", temporal_threshold)
        spectral_prediction = _final_predictions(
            combined, "spectral", spectral_threshold)
        rescue = open_set_rescue_metrics(
            combined["labels"], temporal_prediction, spectral_prediction)
        known_mask = combined["labels"] >= 0
        known_labels = combined["labels"][known_mask]
        known_rows = np.arange(int(known_mask.sum()))
        temporal_support = combined["temporal_support"][known_mask]
        spectral_support = combined["spectral_support"][known_mask]
        temporal_hard = combined["temporal_hard_negative"][known_mask]
        spectral_hard = combined["spectral_hard_negative"][known_mask]
        temporal_pair_metrics = pair_verification_metrics(
            temporal_support, known_labels, temporal_hard)
        spectral_pair_metrics = pair_verification_metrics(
            spectral_support, known_labels, spectral_hard)
        temporal_pair_correct = temporal_support.argmax(1) == known_labels
        spectral_pair_correct = spectral_support.argmax(1) == known_labels
        temporal_true = temporal_support[known_rows, known_labels]
        spectral_true = spectral_support[known_rows, known_labels]
        temporal_on_spectral_pair = (
            temporal_true > temporal_support[known_rows, spectral_hard])
        spectral_on_temporal_pair = (
            spectral_true > spectral_support[known_rows, temporal_hard])
        temporal_own_hard = (
            temporal_true > temporal_support[known_rows, temporal_hard])
        spectral_own_hard = (
            spectral_true > spectral_support[known_rows, spectral_hard])
        temporal_pair_full = np.zeros(len(combined["labels"]), dtype=bool)
        spectral_pair_full = np.zeros_like(temporal_pair_full)
        temporal_pair_full[known_mask] = temporal_pair_correct
        spectral_pair_full[known_mask] = spectral_pair_correct
        temporal_reject = combined["temporal_unknown"] >= temporal_threshold
        spectral_reject = combined["spectral_unknown"] >= spectral_threshold
        decomposition = rescue_decomposition(
            combined["labels"], combined["temporal_prediction"],
            combined["spectral_prediction"], temporal_reject, spectral_reject,
            temporal_pair_full, spectral_pair_full)
        decomposition.update({
            "temporal_to_spectral_hard_pair_rescue": float(
                (temporal_on_spectral_pair & ~spectral_own_hard).mean()),
            "spectral_to_temporal_hard_pair_rescue": float(
                (spectral_on_temporal_pair & ~temporal_own_hard).mean()),
        })
        temporal_identity = (
            combined["temporal_prediction"][known_mask] == known_labels)
        spectral_identity = (
            combined["spectral_prediction"][known_mask] == known_labels)
        predictor_rows.append({
            "features": combined["features"][known_mask],
            "targets": {
                "temporal_rescues_spectral": (
                    temporal_identity & ~spectral_identity).astype(np.int64),
                "spectral_rescues_temporal": (
                    spectral_identity & ~temporal_identity).astype(np.int64),
                "both_fail": (
                    ~temporal_identity & ~spectral_identity).astype(np.int64),
                "no_query_needed": (
                    temporal_identity & spectral_identity).astype(np.int64),
            },
        })
        fold_results.append({
            "inner_fold": episode.inner_index,
            "known_classes": list(episode.known_classes),
            "proxy_unknown_classes": list(episode.proxy_unknown_classes),
            "training": history,
            "class_reindex_invariance": reindex_audit,
            "temporal_threshold": temporal_threshold,
            "spectral_threshold": spectral_threshold,
            "temporal_identity_accuracy": float(temporal_identity.mean()),
            "spectral_identity_accuracy": float(spectral_identity.mean()),
            "temporal_pair_metrics": temporal_pair_metrics,
            "spectral_pair_metrics": spectral_pair_metrics,
            "rescue_decomposition": decomposition,
            **rescue.__dict__,
        })
        print(
            f"[stage8:{cfg['dataset']}] fold {episode.inner_index} "
            f"T/S acc={temporal_identity.mean():.4f}/{spectral_identity.mean():.4f} "
            f"T/S pair={temporal_pair_metrics['pair_auroc']:.4f}/"
            f"{spectral_pair_metrics['pair_auroc']:.4f}",
            flush=True)

    directional = []
    for heldout in range(len(predictor_rows)):
        train_x = np.concatenate([
            row["features"] for index, row in enumerate(predictor_rows)
            if index != heldout])
        test_x = predictor_rows[heldout]["features"]
        row = {"inner_fold": heldout}
        for target_name in predictor_rows[heldout]["targets"]:
            train_y = np.concatenate([
                value["targets"][target_name]
                for index, value in enumerate(predictor_rows)
                if index != heldout])
            test_y = predictor_rows[heldout]["targets"][target_name]
            row[target_name] = _binary_predictability(
                train_x, train_y, test_x, test_y)
        directional.append(row)

    failure_diagnosis = []
    local_threshold = float(cfg["g1_min_local_known_accuracy"])
    pair_threshold = float(cfg["g1_min_pair_auroc"])
    rescue_threshold = float(cfg["g1_min_bidirectional_rescue"])
    for fold, predictability in zip(fold_results, directional):
        causes = []
        final_training = fold["training"][-1]
        for name in ("temporal", "spectral"):
            identity = float(fold[f"{name}_identity_accuracy"])
            train_accuracy = float(
                final_training[f"{name}_train_accuracy"])
            pair_auroc = float(
                fold[f"{name}_pair_metrics"]["pair_auroc"])
            if identity < local_threshold:
                causes.append({
                    "category": "local_generalization_failure",
                    "agent": name,
                    "train_accuracy": train_accuracy,
                    "heldout_identity_accuracy": identity,
                    "interpretation": (
                        "training learned but held-out evaluation failed"
                        if train_accuracy >= 0.90
                        else "local model did not learn the support task"),
                })
            if pair_auroc < pair_threshold:
                causes.append({
                    "category": "verifier_generalization_failure",
                    "agent": name, "pair_auroc": pair_auroc,
                })
        decomposition = fold["rescue_decomposition"]
        t_rescue = float(
            decomposition["temporal_to_spectral_identity_rescue"])
        s_rescue = float(
            decomposition["spectral_to_temporal_identity_rescue"])
        if t_rescue <= rescue_threshold and s_rescue <= rescue_threshold:
            causes.append({
                "category": "repeated_or_noncomplementary_capability",
                "temporal_to_spectral_rescue": t_rescue,
                "spectral_to_temporal_rescue": s_rescue,
            })
        elif min(t_rescue, s_rescue) <= rescue_threshold:
            causes.append({
                "category": "one_way_rescue",
                "temporal_to_spectral_rescue": t_rescue,
                "spectral_to_temporal_rescue": s_rescue,
            })
        for direction in (
                "temporal_rescues_spectral",
                "spectral_rescues_temporal"):
            if float(predictability[direction]["auroc"]) <= 0.5:
                causes.append({
                    "category": "rescue_not_predictable",
                    "direction": direction,
                    "auroc": float(predictability[direction]["auroc"]),
                })
        if (cfg.get("observation_profile") == "isolated_v2"
                and fold["temporal_identity_accuracy"] < local_threshold
                and fold["spectral_identity_accuracy"] < local_threshold):
            causes.append({
                "category": "possible_over_isolation",
                "interpretation": (
                    "both v2 observations fell below the local capability gate"),
            })
        failure_diagnosis.append({
            "inner_fold": fold["inner_fold"], "causes": causes,
            "passed_all_fold_level_diagnostics": not causes,
        })

    g1 = evaluate_ab_g1_gate(
        fold_results,
        min_local_known_accuracy=float(cfg["g1_min_local_known_accuracy"]),
        min_pair_auroc=float(cfg["g1_min_pair_auroc"]),
        min_hard_pair_accuracy=float(cfg["g1_min_hard_pair_accuracy"]),
        min_bidirectional_rescue=float(cfg["g1_min_bidirectional_rescue"]))
    directional_aurocs = {
        name: [row[name]["auroc"] for row in directional]
        for name in ("temporal_rescues_spectral", "spectral_rescues_temporal")}
    g15 = evaluate_directional_g15_gate(
        directional_aurocs,
        min_directional_auroc=float(cfg["g15_min_directional_auroc"]),
        min_positive_folds=int(cfg["g15_min_positive_folds"]))
    reindex_passed = all(
        row["class_reindex_invariance"]["passed"] for row in fold_results)
    class_id_embedding_present = any(
        "embedding" in name for name, _ in system.named_parameters())
    # G1.5 cannot unlock downstream work if any preceding invariant failed.
    unlock = bool(
        g1.passed and g15.passed and reindex_passed
        and not class_id_embedding_present)
    report = {
        "stage": "stage8_g0_g1_probe",
        "dataset": cfg["dataset"], "device": str(device),
        "method_profile": cfg.get("method_profile"),
        "observation_profile": cfg.get("observation_profile", "legacy_v1"),
        "formal_unknown_used": False,
        "formal_unknown_sample_count": protocol.formal_unknown_sample_count,
        "capability_audit": capability_audit().as_dict(),
        "class_id_embedding_present": class_id_embedding_present,
        "class_reindex_invariance_passed": reindex_passed,
        "observation_leakage_audit": leakage_audit,
        "folds": fold_results,
        "failure_diagnosis": failure_diagnosis,
        "directional_rescue_predictability": directional,
        "g1_gate": g1.as_dict(), "g15_gate": g15.as_dict(),
        "outcome": "PASS" if unlock else "STOP-AND-REDESIGN",
        "memory_auditor_unlocked": unlock,
        "forced_consultation_unlocked": unlock,
        "decision": ("proceed_to_memory_auditor_and_forced_consultation"
                     if unlock else "stop_and_redesign_ab"),
        "protocol": protocol_manifest(protocol),
    }
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "g1_probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
