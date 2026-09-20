"""Leakage-safe candidate screening and formal Stage-6 execution."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Subset

from maros_staged.datasets import load_oracle_npz, load_wisig_subset
from maros_stage5.protocol import (assert_no_real_unknown, build_lco_folds,
                                   class_subset)

from .model import Stage6System
from .training import (ROLES, evaluate_open_set, lco_proxy_metrics,
                       predict_system, pretrain_local_agents, set_seed,
                       train_dialogue_candidate)


CANDIDATES = {
    "b1_early_fusion": {"mode": "early_fusion", "adaptive_explorer": False},
    "b2_no_communication": {"mode": "no_comm", "adaptive_explorer": False},
    "c1_parallel_request": {"mode": "parallel_request", "adaptive_explorer": False},
    "c2_sequential": {"mode": "sequential", "adaptive_explorer": False},
    "c3_sequential_cf": {"mode": "sequential_cf", "adaptive_explorer": False},
    "c4_sequential_cf_adversarial": {"mode": "sequential_cf", "adaptive_explorer": True},
}
COLLABORATION_CANDIDATES = (
    "c2_sequential", "c3_sequential_cf", "c4_sequential_cf_adversarial")
METRIC_NAMES = ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score")


def load_splits(cfg: Dict):
    dataset = str(cfg.get("dataset", "wisig")).lower()
    if dataset == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], bool(cfg.get("augment_train", False)))
    if dataset == "oracle":
        return load_oracle_npz(cfg["oracle_root"], bool(cfg.get("augment_train", False)))
    raise ValueError(f"unsupported Stage-6 dataset: {dataset}")


def _system(cfg: Dict, num_classes: int, mode: str) -> Stage6System:
    return Stage6System(
        num_classes,
        state_dim=int(cfg.get("state_dim", 128)),
        stem_channels=int(cfg.get("stem_channels", 128)),
        intent_dim=int(cfg.get("intent_dim", 16)),
        message_dim=int(cfg.get("message_dim", 32)),
        query_dim=int(cfg.get("query_dim", 32)),
        hidden_dim=int(cfg.get("dialogue_hidden_dim", 192)),
        mode=mode,
        prototype_temperature=float(cfg.get("prototype_temperature", 0.15)),
        gate_temperature=float(cfg.get("gate_temperature", 0.5)),
        query_cost=float(cfg.get("query_cost", 0.05)),
    )


def _dataset_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "y"):
        return np.asarray(dataset.y, dtype=np.int64)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = np.asarray(dataset.dataset.y, dtype=np.int64)[dataset.indices]
        if getattr(dataset, "mapping", None) is not None:
            base = np.asarray([dataset.mapping[int(x)] for x in base], dtype=np.int64)
        return base
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))], dtype=np.int64)


def _fallback_parent(initial_agents: dict, train_set, val_set, cfg: Dict,
                     device: torch.device, seed: int, num_classes: int,
                     checkpoint_path: Path | None = None) -> dict:
    """Learn a common silent fusion parent before any protocol diverges."""
    warm_cfg = dict(cfg)
    warm_cfg["dialogue_frozen_epochs"] = int(cfg.get("fallback_warmup_epochs", 5))
    warm_cfg["joint_finetune_epochs"] = 0
    set_seed(seed)
    parent = _system(warm_cfg, int(num_classes), "no_comm").to(device)
    parent.agents.load_state_dict(initial_agents)
    history, _ = train_dialogue_candidate(
        parent, train_set, val_set, warm_cfg, device, seed,
        adaptive_explorer=False)
    state = {key: value.detach().cpu().clone()
             for key, value in parent.state_dict().items()}
    if checkpoint_path is not None:
        torch.save({"model": state, "history": history}, checkpoint_path)
    del parent
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


def balanced_limit(dataset, max_per_class: int | None, seed: int):
    if not max_per_class or max_per_class <= 0:
        return dataset
    labels = _dataset_labels(dataset)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        if len(indices) > max_per_class:
            indices = rng.choice(indices, max_per_class, replace=False)
        keep.extend(int(x) for x in indices)
    return Subset(dataset, sorted(keep))


def _aggregate_candidate_rows(rows: list[dict]) -> dict:
    result = {}
    for candidate in CANDIDATES:
        chosen = [row for row in rows if row["candidate"] == candidate]
        if not chosen:
            continue
        metrics = {}
        for metric in METRIC_NAMES:
            values = np.asarray([row["metrics"]["team"][metric] for row in chosen])
            metrics[metric] = {
                "mean": float(values.mean()), "std": float(values.std()),
                "values": values.tolist(),
            }
        role_metrics = {}
        for role in ROLES:
            role_metrics[role] = {
                metric: float(np.mean([row["metrics"][role][metric] for row in chosen]))
                for metric in METRIC_NAMES
            }
        result[candidate] = {
            "metrics": metrics,
            "role_metrics": role_metrics,
            "query_rate": float(np.mean([row["behaviour"]["query_rate"] for row in chosen])),
            "mean_edges": float(np.mean([row["behaviour"]["mean_edges"] for row in chosen])),
            "mean_bits": float(np.mean([row["behaviour"]["mean_bits"] for row in chosen])),
            "folds": len(chosen),
        }
    return result


def _select_candidate(rows: list[dict], summary: dict, cfg: Dict) -> dict:
    baseline = summary["b2_no_communication"]
    required_fraction = float(cfg.get("positive_fold_fraction", 2.0 / 3.0))
    checks = {}
    for candidate in COLLABORATION_CANDIDATES:
        if candidate not in summary:
            continue
        current = summary[candidate]
        paired = []
        for row in rows:
            if row["candidate"] != candidate:
                continue
            baseline_row = next(other for other in rows
                                if other["candidate"] == "b2_no_communication"
                                and other["partition_seed"] == row["partition_seed"]
                                and other["fold"] == row["fold"])
            paired.append(row["metrics"]["team"]["h_score"]
                          - baseline_row["metrics"]["team"]["h_score"])
        required_positive = int(math.ceil(required_fraction * len(paired)))
        best_single_h = max(row["h_score"] for row in current["role_metrics"].values())
        delta_h = (current["metrics"]["h_score"]["mean"]
                   - baseline["metrics"]["h_score"]["mean"])
        delta_oscr = (current["metrics"]["oscr"]["mean"]
                      - baseline["metrics"]["oscr"]["mean"])
        query_rate = current["query_rate"]
        checks[candidate] = {
            "known_accuracy": current["metrics"]["known_accuracy"]["mean"],
            "delta_h": delta_h, "delta_oscr": delta_oscr,
            "positive_h_folds": int(sum(value > 0 for value in paired)),
            "required_positive_h_folds": required_positive,
            "best_single_h": best_single_h,
            "query_rate": query_rate,
            "passes": bool(
                current["metrics"]["known_accuracy"]["mean"]
                >= float(cfg.get("min_known_accuracy", 0.85))
                and delta_h >= float(cfg.get("min_delta_h", 0.01))
                and delta_oscr >= float(cfg.get("min_delta_oscr", 0.005))
                and sum(value > 0 for value in paired) >= required_positive
                and current["metrics"]["h_score"]["mean"] >= best_single_h
                and 0.10 <= query_rate <= 0.90
            ),
        }
    feasible = [name for name, row in checks.items() if row["passes"]]
    if feasible:
        selected = max(feasible, key=lambda name: (
            summary[name]["metrics"]["h_score"]["mean"],
            summary[name]["metrics"]["oscr"]["mean"],
            -summary[name]["mean_bits"],
        ))
        promoted = True
    else:
        selected = "b2_no_communication"
        promoted = False
    return {"selected": selected, "collaboration_promoted": promoted,
            "checks": checks}


def run_lco_screen(splits, cfg: Dict, device: torch.device, out_root: Path) -> dict:
    rows = []
    partition_seeds = [int(x) for x in cfg.get("lco_partition_seeds", [2026])]
    candidates = list(cfg.get("candidates", CANDIDATES.keys()))
    unknown_seen = False
    for partition_seed in partition_seeds:
        folds = build_lco_folds(
            splits.num_known, int(cfg.get("lco_folds", 5)), partition_seed)
        for fold in folds:
            fold_key = f"partition{partition_seed}_fold{fold.index}"
            fold_dir = out_root / "lco" / fold_key
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_set = class_subset(splits.train, fold.support_classes, remap=True)
            val_set = class_subset(splits.val, fold.support_classes, remap=True)
            heldout_set = class_subset(splits.val, fold.heldout_classes, remap=False)
            assert_no_real_unknown(_dataset_labels(train_set))
            assert_no_real_unknown(_dataset_labels(val_set))
            assert_no_real_unknown(_dataset_labels(heldout_set))
            train_use = balanced_limit(
                train_set, int(cfg.get("lco_max_per_class", 0)),
                int(cfg["seed"]) + partition_seed + fold.index)
            local_cfg = dict(cfg)
            local_cfg["local_pretrain_epochs"] = int(
                cfg.get("lco_local_pretrain_epochs", cfg.get("local_pretrain_epochs", 30)))
            local_cfg["dialogue_frozen_epochs"] = int(
                cfg.get("lco_dialogue_frozen_epochs", cfg.get("dialogue_frozen_epochs", 10)))
            local_cfg["joint_finetune_epochs"] = int(
                cfg.get("lco_joint_finetune_epochs", cfg.get("joint_finetune_epochs", 20)))
            seed = int(cfg["seed"]) + partition_seed * 10 + fold.index * 100
            set_seed(seed)
            base = _system(local_cfg, len(fold.support_classes), "no_comm").to(device)
            print(f"[stage6:lco] {fold_key} support={len(fold.support_classes)} "
                  f"heldout={len(fold.heldout_classes)}")
            local_history, local_epoch = pretrain_local_agents(
                base, train_use, val_set, local_cfg, device, seed)
            torch.save({"agents": base.agents.state_dict(), "history": local_history,
                        "best_epoch": local_epoch, "support_classes": fold.support_classes,
                        "heldout_classes": fold.heldout_classes},
                       fold_dir / "local_agents.pt")
            initial_agents = {key: value.detach().cpu().clone()
                              for key, value in base.agents.state_dict().items()}
            del base
            if device.type == "cuda": torch.cuda.empty_cache()
            candidate_seed = seed + 10000
            parent_state = _fallback_parent(
                initial_agents, train_use, val_set, local_cfg, device,
                candidate_seed, len(fold.support_classes),
                fold_dir / "fallback_parent.pt")
            for candidate in candidates:
                spec = CANDIDATES[candidate]
                print(f"[stage6:lco] {fold_key} candidate={candidate}")
                # All candidates share not only local Agent weights but also
                # identical coordinator initialisation and batch order.
                set_seed(candidate_seed)
                system = _system(local_cfg, len(fold.support_classes), spec["mode"]).to(device)
                system.load_state_dict(parent_state)
                if candidate == "b2_no_communication" and bool(
                        local_cfg.get("use_fallback_parent_as_b2", False)):
                    history, explorer = [], None
                else:
                    history, explorer = train_dialogue_candidate(
                        system, train_use, val_set, local_cfg, device,
                        candidate_seed,
                        adaptive_explorer=bool(spec["adaptive_explorer"]))
                metrics, behaviour = lco_proxy_metrics(
                    system, val_set, heldout_set, device, local_cfg)
                row = {"partition_seed": partition_seed, "fold": fold.index,
                       "candidate": candidate, "mode": spec["mode"],
                       "adaptive_explorer": spec["adaptive_explorer"],
                       "support_classes": list(fold.support_classes),
                       "heldout_classes": list(fold.heldout_classes),
                       "metrics": metrics, "behaviour": behaviour}
                rows.append(row)
                candidate_dir = fold_dir / candidate
                candidate_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": system.state_dict(), "history": history,
                            "explorer": (None if explorer is None else explorer.state_dict())},
                           candidate_dir / "checkpoint.pt")
                (candidate_dir / "metrics.json").write_text(
                    json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
                del system, explorer
                if device.type == "cuda": torch.cuda.empty_cache()
    if unknown_seen:
        raise RuntimeError("real Unknown entered Stage-6 LCO")
    summary = _aggregate_candidate_rows(rows)
    selection = _select_candidate(rows, summary, cfg)
    report = {
        "partition_seeds": partition_seeds,
        "folds_per_partition": int(cfg.get("lco_folds", 5)),
        "rows": rows, "summary": summary, "selection": selection,
        "protocol": {"real_unknown_used": False,
                     "no_communication_allowed_to_win": True,
                     "heldout_classes_per_fold": splits.num_known // int(cfg.get("lco_folds", 5))},
    }
    lco_dir = out_root / "lco"; lco_dir.mkdir(parents=True, exist_ok=True)
    (lco_dir / "stage6_lco_selection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _train_from_common_start(parent_state: dict, candidate: str, splits, cfg: Dict,
                             device: torch.device, seed: int, out_dir: Path):
    spec = CANDIDATES[candidate]
    set_seed(seed)
    system = _system(cfg, splits.num_known, spec["mode"]).to(device)
    system.load_state_dict(parent_state)
    if candidate == "b2_no_communication" and bool(
            cfg.get("use_fallback_parent_as_b2", False)):
        history, explorer = [], None
    else:
        history, explorer = train_dialogue_candidate(
            system, splits.train, splits.val, cfg, device, seed,
            adaptive_explorer=bool(spec["adaptive_explorer"]))
    torch.save({"model": system.state_dict(), "history": history,
                "explorer": None if explorer is None else explorer.state_dict()},
               out_dir / f"{candidate}.pt")
    return system


def _formal_metrics(system: Stage6System, splits, device: torch.device, cfg: Dict,
                    intervention: str | None = None):
    val = predict_system(system, splits.val, device, intervention=intervention)
    known = predict_system(system, splits.test_known, device, intervention=intervention)
    unknown = predict_system(system, splits.test_unknown, device, intervention=intervention)
    metrics, score, threshold = evaluate_open_set(
        val, known, unknown, method="team",
        known_acceptance=float(cfg.get("known_acceptance", 0.90)),
        n_prior=float(cfg.get("calibration_prior", 25.0)))
    return metrics, score, threshold, val, known, unknown


def run_formal_seed(splits, cfg: Dict, selection: dict, device: torch.device,
                    out_root: Path, seed: int) -> dict:
    if not selection["collaboration_promoted"]:
        raise RuntimeError("LCO did not promote collaboration; formal Unknown evaluation is blocked")
    candidate = selection["selected"]
    seed_dir = out_root / "multiseed" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    base = _system(cfg, splits.num_known, "no_comm").to(device)
    local_history, local_epoch = pretrain_local_agents(
        base, splits.train, splits.val, cfg, device, seed)
    base_agents = {key: value.detach().cpu().clone()
                   for key, value in base.agents.state_dict().items()}
    torch.save({"agents": base_agents, "history": local_history,
                "best_epoch": local_epoch}, seed_dir / "local_agents.pt")
    del base
    common_candidate_seed = seed + 1000
    parent_state = _fallback_parent(
        base_agents, splits.train, splits.val, cfg, device,
        common_candidate_seed, splits.num_known,
        seed_dir / "fallback_parent.pt")
    systems = {}
    for name in (candidate, "b2_no_communication", "b1_early_fusion"):
        if name in systems:
            continue
        systems[name] = _train_from_common_start(
            parent_state, name, splits, cfg, device, common_candidate_seed, seed_dir)
    full = systems[candidate]
    results = {}
    arrays = {}
    full_result = _formal_metrics(full, splits, device, cfg)
    results["communication"] = full_result[0]
    arrays["communication"] = full_result
    for label, name in (("no_communication", "b2_no_communication"),
                        ("early_fusion", "b1_early_fusion")):
        value = _formal_metrics(systems[name], splits, device, cfg)
        results[label] = value[0]; arrays[label] = value
    interventions = {
        "messages_off": "messages_off",
        "messages_shuffled": "messages_shuffled",
        "drop_waveform_sender": "drop_waveform_sender",
        "drop_prototype_sender": "drop_prototype_sender",
        "reverse_leader": "reverse_leader",
        "random_leader": "random_leader",
    }
    for label, intervention in interventions.items():
        value = _formal_metrics(full, splits, device, cfg, intervention)
        results[label] = value[0]; arrays[label] = value
    # Local agents are evaluated from the complete system without messages.
    val, known, unknown = full_result[3], full_result[4], full_result[5]
    for role in ROLES:
        metrics, score, threshold = evaluate_open_set(
            val, known, unknown, method=role,
            known_acceptance=float(cfg.get("known_acceptance", 0.90)),
            n_prior=float(cfg.get("calibration_prior", 25.0)))
        results[role] = metrics
        arrays[role] = (metrics, score, threshold, val, known, unknown)

    y = np.r_[known["y"], np.full(len(unknown["y"]), -1, dtype=np.int64)]
    group_id = np.asarray(splits.open_test.group_ids, dtype=np.int64)
    full_score, full_tau = arrays["communication"][1:3]
    no_score, no_tau = arrays["no_communication"][1:3]
    full_pred = np.r_[full_result[4]["pred"], full_result[5]["pred"]]
    no_pred = np.r_[arrays["no_communication"][4]["pred"],
                    arrays["no_communication"][5]["pred"]]
    full_final = np.where(full_score >= full_tau, -1, full_pred)
    no_final = np.where(no_score >= no_tau, -1, no_pred)
    correct_full = full_final == y; correct_no = no_final == y
    query = np.r_[full_result[4]["gates"].max(axis=1),
                  full_result[5]["gates"].max(axis=1)] > 0
    rescue = int((query & correct_full & ~correct_no).sum())
    harm = int((query & ~correct_full & correct_no).sum())
    collaboration = {
        "query_rate": float(query.mean()),
        "rescue_count": rescue, "harm_count": harm,
        "rescue_harm_ratio": float(rescue / max(harm, 1)),
        "mean_bits": float(np.r_[full_result[4]["bits"],
                                  full_result[5]["bits"]].mean()),
        "leader_waveform_rate": float((np.r_[full_result[4]["leader"],
                                                   full_result[5]["leader"]] == 0).mean()),
    }
    npz = {"y": y, "group_id": group_id, "query": query.astype(np.int8)}
    for name, value in arrays.items():
        metric, score, threshold, _, known_row, unknown_row = value
        npz[f"u_{name}"] = score
        npz[f"tau_{name}"] = np.full(len(score), threshold)
        if name in ROLES:
            pred_key = f"pred_{name}"
        else:
            pred_key = "pred"
        npz[f"pred_{name}"] = np.r_[known_row[pred_key], unknown_row[pred_key]]
    # Full transcript and actions are saved once for auditability.
    for key in ("gates", "leader", "bits", "known_weights", "expected_gain",
                "transcript", "message_waveform", "message_prototype",
                "mask_waveform", "mask_prototype"):
        npz[key] = np.concatenate([full_result[4][key], full_result[5][key]])
    np.savez_compressed(seed_dir / "scores_and_transcript.npz", **npz)
    report = {
        "seed": seed, "candidate": candidate, "metrics": results,
        "collaboration": collaboration,
        "protocol": {"real_unknown_used_for_training": False,
                     "real_unknown_used_for_selection": False,
                     "real_unknown_used_for_threshold": False},
    }
    (seed_dir / "stage6_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_stage6(cfg: Dict, *, run_lco: bool = True,
               seeds: list[int] | None = None) -> tuple[list[dict], dict, float]:
    started = time.time()
    out_root = Path(cfg["output_dir"]); out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(cfg)
    manifest = {
        "dataset": splits.meta.get("dataset"), "num_known": splits.num_known,
        "signal_length": splits.signal_length, "train": len(splits.train),
        "validation": len(splits.val), "test_known": len(splits.test_known),
        "test_unknown": len(splits.test_unknown),
        "real_unknown_used_for_training_or_selection": False,
    }
    (out_root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_root / "frozen_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    lco_path = out_root / "lco" / "stage6_lco_selection.json"
    lco = run_lco_screen(splits, cfg, device, out_root) if run_lco else json.loads(
        lco_path.read_text(encoding="utf-8"))
    selection = lco["selection"]
    results = []
    if selection["collaboration_promoted"]:
        for seed in (seeds or [42, 43, 44]):
            results.append(run_formal_seed(
                splits, cfg, selection, device, out_root, int(seed)))
    else:
        blocked = {
            "reason": "no communication candidate passed the pre-registered LCO gain gates",
            "real_unknown_evaluated": False, "selection": selection,
        }
        (out_root / "FORMAL_EVALUATION_BLOCKED.json").write_text(
            json.dumps(blocked, indent=2, ensure_ascii=False), encoding="utf-8")
    return results, selection, time.time() - started
