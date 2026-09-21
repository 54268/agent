"""Paired LCO selection, then locked one-shot formal Unknown evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from maros_stage5.agents import RoleStructuredExperts
from maros_stage7.splits import (ProvenanceSubset, assert_no_formal_unknown,
                                 build_nested_lco_protocol)
from maros_stage10.experiment import _load_dataset, _seed
from maros_stage10.mother import cap_dataset, load_mother
from maros_staged.metrics_osr import detection_metrics, oscr

from .reviewer import CReviewer, probabilities, reports, route, signal_descriptor


ROUTES = ("all_full", "disagreement_only", "disagreement_low",
          "disagreement_low_lite")
# Disagreement-only and disagreement+low-confidence are diagnostic ablations.
# A deployable ABC route must review high-confidence agreement via C-lite.
ELIGIBLE_ROUTES = ("all_full", "disagreement_low_lite")
VIEW_NAMES = ("raw", "spectral", "envelope_phase", "difference_iq", "complex_iq")


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def collect(model, dataset, device, batch_size, *, formal_allowed=False):
    if not formal_allowed:
        assert_no_formal_unknown(dataset)
    elif not isinstance(dataset, ProvenanceSubset) or not dataset.formal_unknown:
        raise ValueError("formal extraction requires explicit formal provenance")
    parts = {key: [] for key in ("labels", "a_logits", "b_logits", "weights",
                                "distances", "descriptor")}
    model.eval()
    for iq, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        a = model.identity(iq.to(device))
        b = model.geometry(iq.to(device))
        values = {"labels": y.numpy(),
                  "a_logits": a.class_logits.cpu().numpy(),
                  "b_logits": b.class_logits.cpu().numpy(),
                  "weights": b.evidence["view_weights"].cpu().numpy(),
                  "distances": b.evidence["view_distances"].cpu().numpy(),
                  "descriptor": signal_descriptor(iq.numpy())}
        for key, value in values.items():
            parts[key].append(value)
    return {key: np.concatenate(value) for key, value in parts.items()}


def enrich(data):
    p = reports(data["a_logits"], data["b_logits"], data["weights"],
                data["distances"])
    fusion = 0.5 * (data["a_logits"] + data["b_logits"])
    ap = probabilities(data["a_logits"])
    bp = probabilities(data["b_logits"])
    fp = probabilities(fusion)
    confidence_choice = np.where(ap.max(1) >= bp.max(1), ap.argmax(1), bp.argmax(1))
    data.update({"report": p, "fusion_pred": fp.argmax(1),
                 "fusion_conf": fp.max(1), "a_pred": ap.argmax(1),
                 "a_confidence": ap.max(1), "b_pred": bp.argmax(1),
                 "b_confidence": bp.max(1),
                 "confidence_pred": confidence_choice,
                 "confidence_conf": np.maximum(ap.max(1), bp.max(1))})
    return data


def evaluate(cal_risk, known_risk, unknown_risk, known_pred, known_y,
             *, acceptance=0.95, unknown_pred=None, known_full=None,
             unknown_full=None):
    threshold = float(np.quantile(cal_risk, acceptance))
    accept = known_risk < threshold
    reject = unknown_risk >= threshold
    known_acc = float(np.mean((known_pred == known_y) & accept))
    unknown_recall = float(np.mean(reject))
    h = 2 * known_acc * unknown_recall / (known_acc + unknown_recall + 1e-12)
    score = np.r_[known_risk, unknown_risk]
    is_unknown = np.r_[np.zeros(len(known_risk), dtype=bool),
                       np.ones(len(unknown_risk), dtype=bool)]
    y = np.r_[known_y, np.full(len(unknown_risk), -1)]
    pred = np.r_[known_pred, np.zeros(len(unknown_risk), dtype=int)
                 if unknown_pred is None else unknown_pred]
    result = {"threshold": threshold, "known_accuracy": known_acc,
              "unknown_recall": unknown_recall, "h_score": float(h),
              "known_acceptance": float(np.mean(accept)),
              "oscr": oscr(is_unknown, pred, y, 1 - score),
              **detection_metrics(is_unknown, score)}
    if known_full is not None:
        result["c_full_fraction_known"] = float(np.mean(known_full))
    if unknown_full is not None:
        result["c_full_fraction_unknown"] = float(np.mean(unknown_full))
    return result


def _baseline(cal, known, unknown, pred_name, conf_name, acceptance):
    return evaluate(1 - cal[conf_name], 1 - known[conf_name],
                    1 - unknown[conf_name], known[pred_name], known["labels"],
                    acceptance=acceptance)


def _scores(reviewer, data):
    return reviewer.scores(data["descriptor"], data["fusion_pred"], data["report"])


def _method(cal, known, unknown, name, acceptance, lite_gate):
    if name in ROUTES or name == "lite_only":
        c, cf = route(cal["report"], cal["scores"], name, lite_gate)
        k, kf = route(known["report"], known["scores"], name, lite_gate)
        u, uf = route(unknown["report"], unknown["scores"], name, lite_gate)
    else:
        key = name.replace("all_", "")
        c, k, u = (data["scores"][key] for data in (cal, known, unknown))
        cf = np.ones(len(c), dtype=bool)
        kf = np.ones(len(k), dtype=bool)
        uf = np.ones(len(u), dtype=bool)
    result = evaluate(c, k, u, known["fusion_pred"], known["labels"],
                      acceptance=acceptance, known_full=kf, unknown_full=uf)
    return result


def _subset_val_proxy(splits, heldout, fold_index):
    indices = np.flatnonzero(np.isin(np.asarray(splits.val.y), heldout))
    return ProvenanceSubset(splits.val, indices, source_split="calibration",
                            purpose=f"outer{fold_index}_selection_proxy_unknown",
                            force_unknown=True)


def _fold(cfg, splits, fold, device, mother_cfg):
    path = Path(cfg["fold_checkpoint_pattern"].format(fold=fold.index))
    model, checkpoint = load_mother(path, mother_cfg, fold.known_classes,
                                    fold.proxy_unknown_classes, device)
    def capped(ds, field, salt, proxy=False):
        return cap_dataset(ds, int(cfg[field]), int(cfg["seed"]) + salt + 17*fold.index,
                           proxy_unknown=proxy)
    sets = {
        "train": capped(fold.train_known, "train_samples_per_class", 11),
        "cal": capped(fold.calibration_known, "cal_samples_per_class", 12),
        "val_proxy": capped(_subset_val_proxy(splits, fold.proxy_unknown_classes,
                                               fold.index),
                            "cal_samples_per_class", 15, True),
        "test": capped(fold.test_known, "test_samples_per_class", 13),
        "test_proxy": capped(fold.test_proxy_unknown,
                             "test_samples_per_class", 14, True)}
    data = {key: enrich(collect(model, ds, device, int(cfg["batch_size"])))
            for key, ds in sets.items()}
    reviewer = CReviewer(seed=int(cfg["seed"]),
                         pug_eta=float(cfg["pug_eta"])).fit(
        data["train"]["descriptor"], data["train"]["labels"],
        data["train"]["report"])
    for item in data.values():
        item["scores"] = _scores(reviewer, item)
    acceptance = float(cfg["known_acceptance"])
    gate = float(np.quantile(data["cal"]["scores"]["lite"],
                             float(cfg["lite_escalation_quantile"])))
    selection = {name: _method(data["cal"], data["cal"], data["val_proxy"],
                               name, acceptance, gate) for name in ROUTES}
    test = {name: _method(data["cal"], data["test"], data["test_proxy"],
                          name, acceptance, gate) for name in
            (*ROUTES, "lite_only", "all_no_pug", "all_no_proto",
             "all_no_reconstruction")}
    baseline = {name: _baseline(data["cal"], data["test"], data["test_proxy"],
                                *fields, acceptance) for name, fields in {
        "a_alone": ("a_pred", "a_confidence"),
        "b_alone": ("b_pred", "b_confidence"),
        "static_mean": ("fusion_pred", "fusion_conf"),
        "confidence_fusion": ("confidence_pred", "confidence_conf")}.items()}
    known = data["test"]
    a_good = known["a_pred"] == known["labels"]
    b_good = known["b_pred"] == known["labels"]
    proxy = data["test_proxy"]
    agreed_confident = (~proxy["report"]["disagree"] &
                        (proxy["report"]["a_conf"] >= 0.55) &
                        (proxy["report"]["b_conf"] >= 0.55) &
                        (proxy["report"]["a_margin"] >= 0.15) &
                        (proxy["report"]["b_margin"] >= 0.15))
    intervention = {}
    rng = np.random.default_rng(int(cfg["seed"]) + 991 + fold.index)
    for kind in ("report_shuffle", "wrong_candidate_pair"):
        altered = dict(known)
        if kind == "report_shuffle":
            order = rng.permutation(len(known["labels"]))
            altered["report"] = {key: np.asarray(v)[order]
                                 for key, v in known["report"].items()}
        else:
            altered["fusion_pred"] = np.where(
                known["fusion_pred"] == known["report"]["a_top2"],
                known["report"]["b_top2"], known["report"]["a_top2"])
        altered["scores"] = _scores(reviewer, altered)
        intervention[kind] = {
            "mean_abs_full_risk_change": float(np.mean(np.abs(
                altered["scores"]["full"] - known["scores"]["full"]))),
            "candidate_accuracy": float(np.mean(
                altered["fusion_pred"] == known["labels"]))}
    return {
        "fold": fold.index, "support_classes": list(fold.known_classes),
        "proxy_classes": list(fold.proxy_unknown_classes),
        "checkpoint_sha256": _digest(path), "best_epoch": int(checkpoint["best_epoch"]),
        "counts": {key: len(value["labels"]) for key, value in data.items()},
        "a_known_accuracy": float(a_good.mean()), "b_known_accuracy": float(b_good.mean()),
        "fusion_known_accuracy": float(np.mean(known["fusion_pred"] == known["labels"])),
        "disagreement_rate": float(known["report"]["disagree"].mean()),
        "a_only_correct": int(np.sum(a_good & ~b_good)),
        "b_only_correct": int(np.sum(b_good & ~a_good)),
        "both_wrong": int(np.sum(~a_good & ~b_good)),
        "view_usage_known": dict(zip(VIEW_NAMES,
            [float(v) for v in known["weights"].mean(axis=0)])),
        "pug_training": reviewer.pug_training,
        "lite_gate": gate, "selection_proxy": selection,
        "test_proxy": test, "baselines": baseline,
        "high_conf_agree_proxy_count": int(agreed_confident.sum()),
        "high_conf_agree_proxy_lite_escalation": int(np.sum(
            agreed_confident & (proxy["scores"]["lite"] >= gate))),
        "interventions": intervention,
        "formal_unknown_used": False,
    }


def select_dataset(cfg, output_root):
    _seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and
                          torch.cuda.is_available() else "cpu")
    splits = _load_dataset(cfg)
    protocol = build_nested_lco_protocol(splits, outer_folds=5, inner_folds=4,
                                         seed=int(cfg["partition_seed"]))
    mother_cfg = json.loads(Path(cfg["mother_config"]).read_text(encoding="utf-8"))
    rows = []
    for fold in protocol.outer_folds:
        print(f"[stage11] {cfg['dataset']} LCO fold {fold.index}/4", flush=True)
        rows.append(_fold(cfg, splits, fold, device, mother_cfg))
    route_mean = {name: float(np.mean([row["selection_proxy"][name]["h_score"]
                                     for row in rows])) for name in ROUTES}
    chosen = max(ELIGIBLE_ROUTES, key=lambda name: (
        route_mean[name], -ELIGIBLE_ROUTES.index(name)))
    result = {"dataset": cfg["dataset"], "folds": rows,
              "selection_route_h_mean": route_mean, "chosen_route": chosen,
              "formal_unknown_used": False,
              "selection_source": "LCO calibration Known + validation proxy Unknown only",
              "eligible_routes": list(ELIGIBLE_ROUTES),
              "test_proxy_not_used_for_selection": True,
              "mother_config_sha256": _digest(cfg["mother_config"]),
              "method_config_sha256": _digest(cfg["config_path"]),
              "formal_unknown_count_not_accessed": protocol.formal_unknown_sample_count}
    out = Path(output_root) / cfg["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "lco_selection_and_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _final_model(cfg, splits, device):
    mother_cfg = json.loads(Path(cfg["mother_config"]).read_text(encoding="utf-8"))
    if mother_cfg["geometry_view"] != "universal":
        raise ValueError("final mother is not universal multi-view")
    checkpoint = torch.load(cfg["final_checkpoint"], map_location="cpu",
                            weights_only=False)
    model = RoleStructuredExperts(
        int(splits.num_known), state_dim=int(mother_cfg["state_dim"]),
        stem_channels=int(mother_cfg["stem_channels"]),
        message_dim=int(mother_cfg["message_dim"]),
        prototype_temperature=float(mother_cfg["prototype_temperature"]),
        geometry_view="universal").to(device)
    model.load_state_dict(checkpoint["experts"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def formal_dataset(cfg, selection_path, output_root):
    if (Path(output_root) / cfg["dataset"] / "formal_one_shot.json").exists():
        raise FileExistsError("formal one-shot result already exists; preserve the untouched holdout")
    frozen = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if frozen.get("state") != "frozen_before_formal" or cfg["dataset"] not in frozen["routes"]:
        raise RuntimeError("formal Unknown access requires frozen paired selection")
    route_name = frozen["routes"][cfg["dataset"]]
    _seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and
                          torch.cuda.is_available() else "cpu")
    splits = _load_dataset(cfg)
    model = _final_model(cfg, splits, device)
    mapping = {k: k for k in range(int(splits.num_known))}
    train = ProvenanceSubset(splits.train, np.arange(len(splits.train)),
                             source_split="train", purpose="final_train_known",
                             class_mapping=mapping)
    cal = ProvenanceSubset(splits.val, np.arange(len(splits.val)),
                           source_split="calibration", purpose="final_calibration_known",
                           class_mapping=mapping)
    test = ProvenanceSubset(splits.test_known, np.arange(len(splits.test_known)),
                            source_split="test_known", purpose="final_test_known",
                            class_mapping=mapping)
    assert_no_formal_unknown(train, cal, test)
    datasets = {"train": train, "cal": cal, "test": test}
    data = {key: enrich(collect(model, ds, device, int(cfg["batch_size"])))
            for key, ds in datasets.items()}
    reviewer = CReviewer(seed=int(cfg["seed"]),
                         pug_eta=float(cfg["pug_eta"])).fit(
        data["train"]["descriptor"], data["train"]["labels"],
        data["train"]["report"])
    for item in data.values():
        item["scores"] = _scores(reviewer, item)
    gate = float(np.quantile(data["cal"]["scores"]["lite"],
                             float(cfg["lite_escalation_quantile"])))
    # This is the sole formal-Unknown read in Stage-11, after frozen selection.
    formal = ProvenanceSubset(splits.test_unknown,
                              np.arange(len(splits.test_unknown)),
                              source_split="formal_unknown",
                              purpose="formal_unknown", force_unknown=True,
                              formal_unknown=True)
    data["unknown"] = enrich(collect(model, formal, device,
                                      int(cfg["batch_size"]), formal_allowed=True))
    data["unknown"]["scores"] = _scores(reviewer, data["unknown"])
    acceptance = float(cfg["known_acceptance"])
    methods = {name: _method(data["cal"], data["test"], data["unknown"],
                             name, acceptance, gate) for name in
               (*ROUTES, "lite_only", "all_no_pug", "all_no_proto",
                "all_no_reconstruction")}
    baselines = {name: _baseline(data["cal"], data["test"], data["unknown"],
                                  *fields, acceptance) for name, fields in {
        "a_alone": ("a_pred", "a_confidence"),
        "b_alone": ("b_pred", "b_confidence"),
        "static_mean": ("fusion_pred", "fusion_conf"),
        "confidence_fusion": ("confidence_pred", "confidence_conf")}.items()}
    u = data["unknown"]
    agreed = (~u["report"]["disagree"] &
              (u["report"]["a_conf"] >= 0.55) &
              (u["report"]["b_conf"] >= 0.55) &
              (u["report"]["a_margin"] >= 0.15) &
              (u["report"]["b_margin"] >= 0.15))
    chosen_risk, chosen_full = route(u["report"], u["scores"], route_name, gate)
    chosen_cal, _ = route(data["cal"]["report"], data["cal"]["scores"],
                          route_name, gate)
    threshold = float(np.quantile(chosen_cal, acceptance))
    result = {"dataset": cfg["dataset"], "chosen_route": route_name,
              "selection_sha256": _digest(selection_path),
              "mother_checkpoint_sha256": _digest(cfg["final_checkpoint"]),
              "counts": {key: len(v["labels"]) for key, v in data.items()},
              "methods": methods, "baselines": baselines,
              "selected": methods[route_name],
              "high_conf_agree_formal_unknown_count": int(agreed.sum()),
              "high_conf_agree_lite_escalated": int(np.sum(
                  agreed & (u["scores"]["lite"] >= gate))),
              "high_conf_agree_rejected": int(np.sum(
                  agreed & (chosen_risk >= threshold))),
              "high_conf_agree_full_reviewed": int(np.sum(agreed & chosen_full)),
              "formal_unknown_used_for_selection_or_training": False,
              "formal_unknown_one_shot_evaluation": True,
              "pug_training": reviewer.pug_training}
    out = Path(output_root) / cfg["dataset"]
    (out / "formal_one_shot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_pair(oracle, wisig, output_root):
    metadata = {"base_config", "config_path", "dataset", "name", "oracle_root",
                "wisig_pkl", "mother_config", "fold_checkpoint_pattern",
                "final_checkpoint", "output_dir", "paired_config"}
    a = {k: v for k, v in oracle.items() if k not in metadata}
    b = {k: v for k, v in wisig.items() if k not in metadata}
    if a != b:
        raise ValueError("paired ABC method fields differ")
    out = Path(output_root)
    if any((out / name / "formal_one_shot.json").exists()
           for name in ("oracle", "wisig")):
        raise FileExistsError("Stage-11 formal Unknown has already been evaluated")
    out.mkdir(parents=True, exist_ok=True)
    selection = {"oracle": select_dataset(oracle, out),
                 "wisig": select_dataset(wisig, out)}
    freeze = {"state": "frozen_before_formal",
              "routes": {name: result["chosen_route"]
                         for name, result in selection.items()},
              "method_config": a,
              "formal_unknown_used_at_freeze": False}
    selection_path = out / "frozen_selection.json"
    selection_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    formal = {"oracle": formal_dataset(oracle, selection_path, out),
              "wisig": formal_dataset(wisig, selection_path, out)}
    summary = {"stage": "stage11_abc_open_set_rejection",
               "selection": {name: {"route": v["chosen_route"],
                    "selection_route_h_mean": v["selection_route_h_mean"],
                    "test_proxy_selected_h_mean": float(np.mean([
                        row["test_proxy"][v["chosen_route"]]["h_score"]
                        for row in v["folds"]]))}
                    for name, v in selection.items()},
               "formal": {name: {"selected": v["selected"],
                       "baselines": v["baselines"],
                       "no_pug": v["methods"]["all_no_pug"],
                       "all_full": v["methods"]["all_full"],
                       "high_conf_agree_rejected": v["high_conf_agree_rejected"],
                       "high_conf_agree_count": v["high_conf_agree_formal_unknown_count"]}
                       for name, v in formal.items()},
               "formal_unknown_used_before_freeze": False}
    (out / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
