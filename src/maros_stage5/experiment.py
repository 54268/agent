"""Stage-5 leave-class-out selection and formal WiSig execution."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from maros_staged.datasets import load_oracle_npz, load_wisig_subset
from maros_staged.evidence import ClassConditionalCdf, EmpiricalCdfCalibrator
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision

from .calibration_agent import (ClassConditionalThresholdAgent,
                                DecisionArbitratorAgent)
from .coordinator import EDGE_NAMES
from .agents import RoleStructuredExperts, UNIVERSAL_VIEW_NAMES
from .openmax import OpenMaxEVT
from .protocol import assert_no_real_unknown, build_lco_folds, class_subset
from .pseudo_unknown import BoundaryExplorerAgent, PUGConfig, PseudoUnknownBatch
from .training import (EvidenceStandardizer, ExpertRecords, evaluate_rule, extract_records,
                       extract_geometry_view_weights,
                       evaluate_arbitrated_rule, fit_openmax, predict_coordinator,
                       pseudo_to_records, refresh_empirical_prototypes, set_seed,
                       train_coordinator, train_experts)


@dataclass
class FoldAssets:
    index: int
    num_classes: int
    model: object
    openmax: object
    train: ExpertRecords
    val: ExpertRecords
    heldout: ExpertRecords
    pseudo: dict[str, tuple[ExpertRecords, np.ndarray, dict]]


def _concat(*records: ExpertRecords) -> ExpertRecords:
    return ExpertRecords(**{
        key: (None if getattr(records[0], key) is None else
              np.concatenate([getattr(record, key) for record in records]))
        for key in records[0].__dataclass_fields__
    })


def _pug_key(kind: str, eta: float) -> str:
    return f"{kind}_eta{eta:g}"


def _make_pseudo(model, openmax, train_records, kind, eta, seed, device):
    config = PUGConfig(kind=kind, eta=float(eta),
                       seed_ratio=0.15, variations=4, seed=int(seed))
    explorer = BoundaryExplorerAgent(config)
    decision = explorer.act(
        train_records.identity_state, train_records.geometry_state, train_records.labels,
        train_records.identity_logits.argmax(1), train_records.geometry_logits.argmax(1),
        train_records.evidence,
    )
    pseudo = decision.proposals
    records = pseudo_to_records(model, pseudo, openmax, device)
    meta = {"agent": explorer.name, "config": asdict(config),
            "num_pseudo": int(len(records.labels)), "reliability": decision.reliability,
            **decision.evidence}
    return records, pseudo.source_indices, meta


def _generator_proxy_metrics(train_known, val_known, heldout, pseudo, seed):
    standardizer = EvidenceStandardizer().fit(train_known.evidence)
    rng = np.random.default_rng(seed)
    n = min(len(train_known.labels), len(pseudo.labels), 6000)
    ik = rng.choice(len(train_known.labels), n, replace=False)
    ip = rng.choice(len(pseudo.labels), n, replace=False)
    x = np.concatenate([standardizer.transform(train_known.evidence[ik]),
                        standardizer.transform(pseudo.evidence[ip])])
    target = np.r_[np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    classifier.fit(x, target)
    val_score = classifier.predict_proba(standardizer.transform(val_known.evidence))[:, 1]
    held_score = classifier.predict_proba(standardizer.transform(heldout.evidence))[:, 1]
    open_score = np.r_[val_score, held_score]
    calibrated = EmpiricalCdfCalibrator(val_score).score(open_score)
    y = np.r_[val_known.labels, np.full(len(heldout.labels), -1, dtype=np.int64)]
    pred_val = (0.5 * val_known.identity_logits + 0.5 * val_known.geometry_logits).argmax(1)
    pred_held = (0.5 * heldout.identity_logits + 0.5 * heldout.geometry_logits).argmax(1)
    pred = np.r_[pred_val, pred_held]
    is_unknown = (y == -1).astype(np.int32)
    metrics = detection_metrics(is_unknown, calibrated)
    metrics["oscr"] = oscr(is_unknown, pred, y, 1.0 - calibrated)
    metrics.update(threshold_decision(is_unknown, y, pred, calibrated, 0.95))
    return metrics


def _selection_summary(rows, key_field, known_floor=0.85):
    keys = sorted({row[key_field] for row in rows})
    summaries = []
    for key in keys:
        chosen = [row for row in rows if row[key_field] == key]
        metrics = {}
        for metric in ("auroc", "oscr", "known_accuracy", "unknown_recall", "h_score"):
            values = [row["metrics"][metric] for row in chosen]
            metrics[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                               "values": [float(v) for v in values]}
        summary = {key_field: key, "metrics": metrics,
                   "feasible": metrics["known_accuracy"]["mean"] >= known_floor}
        if all("mean_edges" in row for row in chosen):
            values = [float(row["mean_edges"]) for row in chosen]
            summary["mean_edges"] = {"mean": float(np.mean(values)),
                                     "std": float(np.std(values)), "values": values}
        summaries.append(summary)
    pool = [row for row in summaries if row["feasible"]] or summaries
    selected = max(pool, key=lambda row: (
        row["metrics"]["h_score"]["mean"], row["metrics"]["oscr"]["mean"],
        row["metrics"]["unknown_recall"]["mean"], row["metrics"]["auroc"]["mean"],
    ))
    return summaries, selected


def _prepare_lco_assets(splits, cfg, device, out_dir):
    folds = build_lco_folds(splits.num_known, int(cfg.get("lco_folds", 5)),
                            int(cfg.get("lco_partition_seed", 2026)))
    assets = []
    lco_cfg = copy.deepcopy(cfg); lco_cfg["expert_epochs"] = int(cfg.get("lco_expert_epochs", 12))
    reuse_root = Path(cfg["reuse_lco_experts_from"]) if cfg.get("reuse_lco_experts_from") else None
    for fold in folds:
        print(f"[stage5:lco] fold={fold.index} support={len(fold.support_classes)} heldout={len(fold.heldout_classes)}")
        train_set = class_subset(splits.train, fold.support_classes, remap=True)
        val_set = class_subset(splits.val, fold.support_classes, remap=True)
        heldout_set = class_subset(splits.val, fold.heldout_classes, remap=False)
        assert_no_real_unknown(splits.train.y[train_set.indices])
        assert_no_real_unknown(splits.val.y[heldout_set.indices])
        if reuse_root is not None:
            checkpoint_path = reuse_root / "folds" / f"fold{fold.index}" / "experts.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (checkpoint["support_classes"] != fold.support_classes
                    or checkpoint["heldout_classes"] != fold.heldout_classes):
                raise ValueError(f"reused LCO fold {fold.index} has a different class partition")
            model = RoleStructuredExperts(
                len(fold.support_classes), state_dim=int(cfg.get("state_dim", 64)),
                stem_channels=int(cfg.get("stem_channels", 64)),
                message_dim=int(cfg.get("message_dim", 32)),
                prototype_temperature=float(cfg.get("prototype_temperature", 0.15)),
                geometry_view=str(cfg.get("geometry_view", "spectral")),
            ).to(device)
            incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
            unexpected = [key for key in incompatible.unexpected_keys
                          if not key.endswith(("prototypes_ready", "view_prototypes_ready"))]
            if unexpected:
                raise ValueError(f"unexpected keys in reused LCO checkpoint: {unexpected}")
            refresh_empirical_prototypes(model, train_set, device,
                                         int(cfg.get("batch_size", 256)))
            history, best_epoch = checkpoint["history"], checkpoint["best_epoch"]
            evt = OpenMaxEVT.from_state_dict(checkpoint["openmax"])
            print(f"[stage5:lco] reused deterministic experts from {checkpoint_path}")
        else:
            model, history, best_epoch = train_experts(
                train_set, val_set, len(fold.support_classes), lco_cfg, device,
                int(cfg["seed"]) + fold.index * 100,
            )
            evt = fit_openmax(model, train_set, device,
                              int(cfg.get("openmax_alpha_rank", 3)),
                              int(cfg.get("openmax_tail_size", 25)))
        train_records = extract_records(model, train_set, device, evt)
        val_records = extract_records(model, val_set, device, evt)
        held_records = extract_records(model, heldout_set, device, evt)
        held_records.labels[:] = -1
        pseudo_map = {}
        for kind in cfg.get("pug_kinds", ["competition", "disagreement", "mixed"]):
            for eta in cfg.get("pug_eta_grid", [1.0, 1.5, 2.0]):
                key = _pug_key(kind, float(eta))
                pseudo_map[key] = _make_pseudo(model, evt, train_records, kind, float(eta),
                                               int(cfg["seed"]) + fold.index * 1000, device)
        fold_dir = out_dir / "folds" / f"fold{fold.index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "support_classes": fold.support_classes,
                    "heldout_classes": fold.heldout_classes, "best_epoch": best_epoch,
                    "history": history, "openmax": evt.state_dict()}, fold_dir / "experts.pt")
        assets.append(FoldAssets(fold.index, len(fold.support_classes), model.cpu(), evt,
                                 train_records, val_records, held_records, pseudo_map))
        if device.type == "cuda": torch.cuda.empty_cache()
    return folds, assets


def run_lco_selection(splits, cfg, device, out_dir):
    folds, assets = _prepare_lco_assets(splits, cfg, device, out_dir)
    generator_rows = []
    for asset in assets:
        for key, (pseudo, _, _) in asset.pseudo.items():
            metrics = _generator_proxy_metrics(asset.train, asset.val, asset.heldout, pseudo,
                                                int(cfg["seed"]) + asset.index)
            generator_rows.append({"fold": asset.index, "pug": key, "metrics": metrics})
    generator_summary, chosen_pug = _selection_summary(generator_rows, "pug",
                                                        float(cfg.get("min_known_accuracy", 0.85)))
    selected_pug = chosen_pug["pug"]
    print(f"[stage5:lco] selected PUG={selected_pug}")

    configured_candidates = cfg.get("communication_candidates")
    if configured_candidates:
        comm_candidates = [(row["mode"], float(row["budget"]),
                            float(row.get("gate_threshold", 0.5)))
                           for row in configured_candidates]
    else:
        comm_candidates = [("none", 0.5, 0.5), ("full", 0.5, 0.5),
                           ("sparse", 0.5, 0.5), ("sparse", 0.75, 0.5),
                           ("sparse_cf", 0.5, 0.5), ("sparse_cf", 0.75, 0.5)]
    communication_rows = []
    lco_coord_cfg = copy.deepcopy(cfg)
    lco_coord_cfg["coordinator_epochs"] = int(cfg.get("lco_coordinator_epochs", 10))
    max_samples = int(cfg.get("lco_max_samples", 4096))
    for asset in assets:
        asset.model.to(device)
        pseudo, sources, _ = asset.pseudo[selected_pug]
        if len(asset.train.labels) > max_samples:
            rng = np.random.default_rng(int(cfg["seed"]) + asset.index)
            keep = rng.choice(len(asset.train.labels), max_samples, replace=False)
            train_records = asset.train.subset(keep)
        else:
            train_records = asset.train
        if len(pseudo.labels) > max_samples:
            rng = np.random.default_rng(int(cfg["seed"]) + 500 + asset.index)
            keep = rng.choice(len(pseudo.labels), max_samples, replace=False)
            pseudo_use, sources_use = pseudo.subset(keep), sources[keep]
        else:
            pseudo_use, sources_use = pseudo, sources
        open_records = _concat(asset.val, asset.heldout)
        trained_coordinators = {}
        for mode, budget, gate_threshold in comm_candidates:
            print(f"[stage5:lco] fold={asset.index} comm={mode} budget={budget} "
                  f"threshold={gate_threshold}")
            training_key = (mode, budget)
            if training_key not in trained_coordinators:
                trained_coordinators[training_key] = train_coordinator(
                    train_records, asset.val, pseudo_use, sources_use, asset.num_classes,
                    lco_coord_cfg, device,
                    int(cfg["seed"]) + asset.index * 100 + int(budget * 10),
                    mode, budget, 0.5,
                )
            else:
                print("      reused paired coordinator weights; evaluating a new hard gate threshold")
            coordinator, standardizer, history, best_epoch = trained_coordinators[training_key]
            coordinator.gate_threshold = float(gate_threshold)
            val_pred = predict_coordinator(coordinator, asset.val, standardizer, device)
            open_pred = predict_coordinator(coordinator, open_records, standardizer, device)
            val_local = predict_coordinator(
                coordinator, asset.val, standardizer, device, intervention="messages_off")
            open_local = predict_coordinator(
                coordinator, open_records, standardizer, device, intervention="messages_off")
            for score_rule in cfg.get("unknown_score_rules", [
                    "boundary", "mean3", "mean4", "geo_open_boundary", "max4"]):
                for arbitration_rule in cfg.get("arbitration_rules", ["communication"]):
                    arbitrator = DecisionArbitratorAgent(score_rule, arbitration_rule).fit(
                        asset.val, val_pred, val_local)
                    val_calibrated = arbitrator.transform(
                        asset.val, val_pred, val_local).unknown_score
                    calibrated = arbitrator.transform(
                        open_records, open_pred, open_local).unknown_score
                    is_unknown = (open_records.labels == -1).astype(np.int32)
                    base_metrics = detection_metrics(is_unknown, calibrated)
                    base_metrics["oscr"] = oscr(
                        is_unknown, open_pred["pred"], open_records.labels,
                        1.0 - calibrated)
                    for known_acceptance in cfg.get("known_acceptance_grid", [0.95]):
                        for threshold_prior in cfg.get("adaptive_threshold_priors", [None]):
                            if threshold_prior is None:
                                thresholds = np.full(
                                    len(calibrated), float(known_acceptance), dtype=np.float64)
                            else:
                                threshold_agent = ClassConditionalThresholdAgent(
                                    float(known_acceptance), float(threshold_prior)).fit(
                                        val_calibrated, val_pred["pred"])
                                thresholds = threshold_agent.thresholds(open_pred["pred"])
                            metrics = dict(base_metrics)
                            metrics.update(threshold_decision(
                                is_unknown, open_records.labels, open_pred["pred"],
                                calibrated, thresholds))
                            metrics["threshold_min"] = float(thresholds.min())
                            metrics["threshold_mean"] = float(thresholds.mean())
                            metrics["threshold_max"] = float(thresholds.max())
                            prior_name = "global" if threshold_prior is None else f"{float(threshold_prior):g}"
                            candidate = (f"{mode}_budget{budget:g}_threshold{gate_threshold:g}_"
                                         f"{score_rule}_arb{arbitration_rule}_"
                                         f"tau{float(known_acceptance):g}_prior{prior_name}")
                            communication_rows.append({"fold": asset.index,
                                                       "communication": candidate,
                                                       "mode": mode, "budget": budget,
                                                       "gate_threshold": gate_threshold,
                                                       "score_rule": score_rule,
                                                       "arbitration_rule": arbitration_rule,
                                                       "known_acceptance": float(known_acceptance),
                                                       "adaptive_threshold_prior": threshold_prior,
                                                       "metrics": metrics,
                                                       "best_epoch": best_epoch,
                                                       "mean_edges": float(open_pred["gates"].sum(1).mean())})
        asset.model.cpu()
        if device.type == "cuda": torch.cuda.empty_cache()
    communication_summary, selected_comm = _selection_summary(
        communication_rows, "communication", float(cfg.get("min_known_accuracy", 0.85)))
    # No-communication is a baseline, not a candidate final communication method.
    comm_pool = [row for row in communication_summary if not row["communication"].startswith("none_")]
    feasible = [row for row in comm_pool if row["feasible"]] or comm_pool
    selected_comm = max(feasible, key=lambda row: (
        row["metrics"]["h_score"]["mean"], row["metrics"]["oscr"]["mean"],
        row["metrics"]["unknown_recall"]["mean"],
        -row.get("mean_edges", {"mean": 4.0})["mean"],
    ))
    selected_row = next(row for row in communication_rows
                        if row["communication"] == selected_comm["communication"])
    selection = {"pug": selected_pug, "communication": selected_comm["communication"],
                 "mode": selected_row["mode"], "budget": selected_row["budget"],
                 "gate_threshold": selected_row["gate_threshold"],
                 "score_rule": selected_row["score_rule"],
                 "arbitration_rule": selected_row.get("arbitration_rule", "communication"),
                 "known_acceptance": selected_row.get("known_acceptance", 0.95),
                 "adaptive_threshold_prior": selected_row.get("adaptive_threshold_prior")}
    report = {"folds": [{"index": fold.index, "support_classes": fold.support_classes,
                          "heldout_classes": fold.heldout_classes} for fold in folds],
              "generator_rows": generator_rows, "generator_summary": generator_summary,
              "communication_rows": communication_rows,
              "communication_summary": communication_summary, "selected": selection,
              "protocol": {"real_unknown_used": False, "heldout_classes_per_fold": 8}}
    (out_dir / "lco_selection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"[stage5:lco] selected communication={selection['communication']}")
    return report


def _baseline_rule(val_records, open_records, name, known_acceptance=0.95,
                   adaptive_threshold_prior=None):
    if name == "identity":
        val_score, open_score = val_records.evidence[:, 0], open_records.evidence[:, 0]
        val_pred = val_records.identity_logits.argmax(1); open_pred = open_records.identity_logits.argmax(1)
    elif name == "prototype":
        val_score, open_score = val_records.evidence[:, 3], open_records.evidence[:, 3]
        val_pred = val_records.geometry_logits.argmax(1); open_pred = open_records.geometry_logits.argmax(1)
    elif name == "openmax":
        val_score, open_score = val_records.evidence[:, 6], open_records.evidence[:, 6]
        val_pred = val_records.geometry_logits.argmax(1); open_pred = open_records.geometry_logits.argmax(1)
    else:
        raise ValueError(name)
    if name == "identity":
        val_class = val_records.identity_logits.argmax(1)
        open_class = open_records.identity_logits.argmax(1)
    else:
        val_class = val_records.geometry_logits.argmax(1)
        open_class = open_records.geometry_logits.argmax(1)
    component_calibrator = ClassConditionalCdf(val_score, val_class)
    val_calibrated = component_calibrator.score(val_score, val_class)
    calibrated = component_calibrator.score(open_score, open_class)
    if adaptive_threshold_prior is None:
        thresholds = np.full(len(calibrated), known_acceptance, dtype=np.float64)
    else:
        threshold_agent = ClassConditionalThresholdAgent(
            known_acceptance, adaptive_threshold_prior).fit(val_calibrated, val_pred)
        thresholds = threshold_agent.thresholds(open_pred)
    y = open_records.labels; is_unknown = (y == -1).astype(np.int32)
    metrics = detection_metrics(is_unknown, calibrated)
    metrics["oscr"] = oscr(is_unknown, open_pred, y, 1.0 - calibrated)
    metrics.update(threshold_decision(is_unknown, y, open_pred, calibrated, thresholds))
    metrics["threshold_min"] = float(thresholds.min())
    metrics["threshold_mean"] = float(thresholds.mean())
    metrics["threshold_max"] = float(thresholds.max())
    return metrics, calibrated, thresholds


def run_final_seed(splits, cfg, selection, device, out_dir, seed):
    print(f"[stage5:final] seed={seed}")
    set_seed(seed); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_config.json").write_text(
        json.dumps({"config": cfg, "selection": selection}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    full_train, full_val, full_open = splits.train, splits.val, splits.open_test
    reuse_final = cfg.get("reuse_final_experts_from")
    if reuse_final:
        checkpoint = torch.load(Path(reuse_final), map_location="cpu", weights_only=False)
        model = RoleStructuredExperts(
            splits.num_known, state_dim=int(cfg.get("state_dim", 64)),
            stem_channels=int(cfg.get("stem_channels", 64)),
            message_dim=int(cfg.get("message_dim", 32)),
            prototype_temperature=float(cfg.get("prototype_temperature", 0.15)),
            geometry_view=str(cfg.get("geometry_view", "spectral")),
        ).to(device)
        state = checkpoint.get("experts", checkpoint.get("state_dict", checkpoint))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"unexpected keys in reused final checkpoint: {incompatible.unexpected_keys}")
        refresh_empirical_prototypes(model, full_train, device,
                                     int(cfg.get("batch_size", 256)))
        expert_history, expert_best = [], int(checkpoint.get("expert_best_epoch", -1))
        print(f"[stage5:final] reused frozen expert encoders from {reuse_final}")
    else:
        model, expert_history, expert_best = train_experts(
            full_train, full_val, splits.num_known, cfg, device, seed)
    evt = fit_openmax(model, full_train, device, int(cfg.get("openmax_alpha_rank", 3)),
                      int(cfg.get("openmax_tail_size", 25)))
    train_records = extract_records(model, full_train, device, evt)
    val_records = extract_records(model, full_val, device, evt)
    open_records = extract_records(model, full_open, device, evt)
    geometry_view_weights = extract_geometry_view_weights(model, full_open, device)
    kind, eta_text = selection["pug"].split("_eta")
    pseudo_records, sources, pseudo_meta = _make_pseudo(model, evt, train_records, kind,
                                                        float(eta_text), seed, device)
    coordinators = {}
    for label, mode, budget, gate_threshold in (
            ("no_communication", "none", 0.5, 0.5),
            ("communication", selection["mode"], float(selection["budget"]),
             float(selection.get("gate_threshold", 0.5)))):
        coord, scaler, history, best_epoch = train_coordinator(
            train_records, val_records, pseudo_records, sources, splits.num_known,
            cfg, device, seed, mode, budget, gate_threshold,
        )
        coordinators[label] = (coord, scaler, history, best_epoch)

    metrics, scores, predictions, decision_thresholds = {}, {}, {}, {}
    known_acceptance = float(selection.get("known_acceptance", 0.95))
    adaptive_threshold_prior = selection.get("adaptive_threshold_prior")
    for name in ("identity", "prototype", "openmax"):
        metrics[name], scores[name], decision_thresholds[name] = _baseline_rule(
            val_records, open_records, name, known_acceptance,
            adaptive_threshold_prior)
    interventions = {
        "communication": None,
        "messages_off": "messages_off",
        "messages_shuffled": "messages_shuffled",
        "drop_identity_sender": "drop_identity_sender",
        "drop_geometry_sender": "drop_geometry_sender",
        "uniform_weights": "uniform_weights",
    }
    for name, (coord, scaler, _, _) in coordinators.items():
        val_pred = predict_coordinator(coord, val_records, scaler, device)
        open_pred = predict_coordinator(coord, open_records, scaler, device)
        if name == "communication":
            val_local = predict_coordinator(
                coord, val_records, scaler, device, intervention="messages_off")
            open_local = predict_coordinator(
                coord, open_records, scaler, device, intervention="messages_off")
            metrics[name], scores[name], decision_thresholds[name] = evaluate_arbitrated_rule(
                val_records, open_records, val_pred, open_pred, val_local, open_local,
                score_rule=selection.get("score_rule", "boundary"),
                arbitration_rule=selection.get("arbitration_rule", "communication"),
                known_acceptance=known_acceptance,
                adaptive_threshold_prior=adaptive_threshold_prior,
                return_thresholds=True)
        else:
            metrics[name], scores[name], decision_thresholds[name] = evaluate_rule(
                val_records, open_records, val_pred, open_pred,
                score_rule=selection.get("score_rule", "boundary"),
                known_acceptance=known_acceptance,
                adaptive_threshold_prior=adaptive_threshold_prior,
                return_thresholds=True)
        predictions[name] = open_pred
        if name == "communication":
            for rule, intervention in interventions.items():
                if rule == "communication": continue
                v = predict_coordinator(coord, val_records, scaler, device, intervention=intervention)
                o = predict_coordinator(coord, open_records, scaler, device, intervention=intervention)
                metrics[rule], scores[rule], decision_thresholds[rule] = evaluate_arbitrated_rule(
                    val_records, open_records, v, o, val_local, open_local,
                    score_rule=selection.get("score_rule", "boundary"),
                    arbitration_rule=selection.get("arbitration_rule", "communication"),
                    known_acceptance=known_acceptance,
                    adaptive_threshold_prior=adaptive_threshold_prior,
                    return_thresholds=True)
                predictions[rule] = o

    normal = predictions["communication"]
    causal = {}
    for rule in interventions:
        if rule == "communication": continue
        causal[rule] = {
            "class_flip_rate": float((predictions[rule]["pred"] != normal["pred"]).mean()),
            "unknown_score_mean_abs_change": float(np.abs(scores[rule] - scores["communication"]).mean()),
            "auroc_delta": metrics[rule]["auroc"] - metrics["communication"]["auroc"],
            "oscr_delta": metrics[rule]["oscr"] - metrics["communication"]["oscr"],
        }
    comm_model, comm_scaler, comm_history, comm_best = coordinators["communication"]
    no_model, no_scaler, no_history, no_best = coordinators["no_communication"]
    torch.save({"experts": model.state_dict(), "expert_best_epoch": expert_best,
                "openmax": evt.state_dict(), "config": cfg}, out_dir / "experts.pt")
    for name, coord, scaler, history, best in (
        ("communication", comm_model, comm_scaler, comm_history, comm_best),
        ("no_communication", no_model, no_scaler, no_history, no_best)):
        torch.save({"state_dict": coord.state_dict(), "mode": coord.mode,
                    "budget": coord.budget_target, "evidence_mean": scaler.mean,
                    "gate_threshold": coord.gate_threshold,
                    "evidence_std": scaler.std, "best_epoch": best, "history": history,
                    "config": cfg}, out_dir / f"{name}.pt")
    if not hasattr(full_open, "group_ids"):
        raise RuntimeError("open_test is missing transmitter group IDs required by Stage-5")
    np.savez_compressed(out_dir / "scores.npz", y=open_records.labels,
                        group_id=np.asarray(full_open.group_ids, dtype=np.int64),
                        **{f"u_{name}": value for name, value in scores.items()},
                        **{f"tau_{name}": value for name, value in decision_thresholds.items()},
                        **{f"pred_{name}": value["pred"] for name, value in predictions.items()},
                        communication_gates=normal["gates"],
                        communication_known_weights=normal["known_weights"],
                        geometry_view_weights=geometry_view_weights)
    known_mask = open_records.labels != -1
    view_names = (list(UNIVERSAL_VIEW_NAMES) if geometry_view_weights.shape[1] > 1
                  else [str(cfg.get("geometry_view", "spectral"))])
    summary = {"seed": seed, "selection": selection, "metrics": metrics,
               "causal_interventions": causal, "pseudo": pseudo_meta,
               "edge_names": EDGE_NAMES,
               "mean_edge_gates": normal["gates"].mean(axis=0).tolist(),
               "mean_active_edges": float(normal["gates"].sum(axis=1).mean()),
               "geometry_sensor_actions": {
                   "view_names": view_names,
                   "known_mean_weights": geometry_view_weights[known_mask].mean(axis=0).tolist(),
                   "unknown_mean_weights": geometry_view_weights[~known_mask].mean(axis=0).tolist(),
               },
               "protocol": {"real_unknown_used_for_training": False,
                            "real_unknown_used_for_selection": False,
                            "real_unknown_used_for_threshold": False}}
    (out_dir / "stage5_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    _write_seed_report(summary, out_dir / "report_stage5.md")
    return summary


def _write_seed_report(summary, path):
    names = {"identity": "Identity", "prototype": "Prototype", "openmax": "OpenMax",
             "no_communication": "等容量无通信", "communication": "完整协作",
             "messages_off": "推理关闭消息", "messages_shuffled": "打乱消息",
             "drop_identity_sender": "删除 Identity 发送边",
             "drop_geometry_sender": "删除 Geometry 发送边", "uniform_weights": "均匀类别融合"}
    lines = [f"# Stage-5 seed {summary['seed']}", "",
             "真实 Unknown 不参与训练、选模或阈值。", "",
             "| 规则 | AUROC | OSCR | Known Acc | Unknown Recall | H-score |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in names.items():
        m = summary["metrics"][key]
        lines.append(f"| {label} | {m['auroc']:.4f} | {m['oscr']:.4f} | "
                     f"{m['known_accuracy']:.4f} | {m['unknown_recall']:.4f} | {m['h_score']:.4f} |")
    lines += ["", f"平均激活边数：{summary['mean_active_edges']:.3f} / 4。"]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage5(config: Dict, *, run_lco: bool = True, seeds: list[int] | None = None):
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = str(config.get("dataset", "wisig")).lower()
    if dataset_name == "wisig":
        source = config["wisig_pkl"]
        splits = load_wisig_subset(source, augment_train=bool(config.get("augment_train", True)))
    elif dataset_name == "oracle":
        source = config["oracle_root"]
        splits = load_oracle_npz(source, augment_train=bool(config.get("augment_train", False)))
    else:
        raise ValueError(f"unsupported Stage-5 dataset {dataset_name}")
    out_root = Path(config["output_dir"]); out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "frozen_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    group_ids = np.asarray(splits.open_test.group_ids)
    manifest = {
        "dataset": splits.meta.get("dataset"), "source": str(source),
        "signal_length": int(splits.signal_length), "num_known": int(splits.num_known),
        "counts": {"train": len(splits.train), "val": len(splits.val),
                   "known_test": len(splits.test_known), "unknown_test": len(splits.test_unknown),
                   "open_test": len(splits.open_test)},
        "evaluation_groups": {
            "known_tx": int(len(np.unique(group_ids[splits.open_test.y != -1]))),
            "unknown_tx": int(len(np.unique(group_ids[splits.open_test.y == -1]))),
        },
        "protocol": {"real_unknown_used_for_training": False,
                     "real_unknown_used_for_selection": False,
                     "real_unknown_used_for_threshold": False},
    }
    (out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if run_lco:
        lco = run_lco_selection(splits, config, device, out_root / "lco")
        selection = lco["selected"]
    else:
        lco_path = out_root / "lco" / "lco_selection.json"
        selection = json.loads(lco_path.read_text(encoding="utf-8"))["selected"]
    results = []
    for seed in (seeds or [42, 43, 44]):
        seed_cfg = copy.deepcopy(config); seed_cfg["seed"] = int(seed)
        results.append(run_final_seed(splits, seed_cfg, selection, device,
                                      out_root / "multiseed" / f"seed{seed}", int(seed)))
    elapsed = time.time() - t0
    return results, selection, elapsed
