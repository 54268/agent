"""Five-fold proxy-only Certified PUG evaluation; formal Unknown stays untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from maros_stage10.experiment import _load_dataset, _seed
from maros_stage10.mother import cap_dataset, load_mother
from maros_stage11.experiment import _subset_val_proxy, evaluate
from maros_stage11.reviewer import probabilities, reports
from maros_stage5.openmax import OpenMaxEVT
from maros_stage7.splits import assert_no_formal_unknown, build_nested_lco_protocol
from maros_staged.evidence import ClassConditionalCdf

from .certified_pug import (PUGTool, SupportEvidence, apply_increment,
                            certificate, iq_candidates)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


@torch.no_grad()
def collect(model, evt, dataset, device, batch_size, *, pseudo=False):
    if not pseudo:
        source = dataset
        while isinstance(source, Subset):
            source = source.dataset
        assert_no_formal_unknown(source)
    elif not isinstance(dataset, TensorDataset):
        raise ValueError("pseudo collector requires generated I/Q TensorDataset")
    fields = ("labels", "iq", "a_logits", "b_logits", "a_d1", "b_d1",
              "weights", "distances")
    parts = {key: [] for key in fields}
    model.eval()
    for iq, label in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        inputs = iq.to(device)
        a = model.identity(inputs)
        b = model.geometry(inputs)
        batch = {"labels": label.numpy(), "iq": iq.numpy(),
                 "a_logits": a.class_logits.cpu().numpy(),
                 "b_logits": b.class_logits.cpu().numpy(),
                 "a_d1": a.evidence["d1"].cpu().numpy(),
                 "b_d1": b.evidence["d1"].cpu().numpy(),
                 "weights": b.evidence["view_weights"].cpu().numpy(),
                 "distances": b.evidence["view_distances"].cpu().numpy()}
        for key, value in batch.items():
            parts[key].append(value)
    data = {key: np.concatenate(value) for key, value in parts.items()}
    data["a_pred"] = data["a_logits"].argmax(1)
    data["b_pred"] = data["b_logits"].argmax(1)
    data["openmax_raw"] = evt.score(probabilities(data["b_logits"]))
    data["report"] = reports(data["a_logits"], data["b_logits"],
                             data["weights"], data["distances"])
    data["fresh_forward"] = bool(pseudo)
    return data


def _pseudo_dataset(iq):
    x = np.asarray(iq, dtype=np.float32)
    return TensorDataset(torch.from_numpy(x),
                         torch.full((len(x),), -1, dtype=torch.long))


def _base_scores(data, tool=None, calibrator=None):
    base = np.asarray(data["u_open"], dtype=float)
    lite = apply_increment(base, data["u_proto"], 0.12)
    full = apply_increment(lite, data["u_idproto"], 0.08)
    if tool is None or not tool.enabled:
        on = full
    else:
        if calibrator is None:
            raise ValueError("PUG tool needs Known-only calibration")
        pug = calibrator.score(tool.raw_score(data), data["b_pred"])
        on = apply_increment(full, pug, 0.20)
    return {"stage5_openmax_component": base,
            "stage5_plus_c_lite": lite,
            "stage5_plus_c_full": full,
            "stage5_plus_c_pug": on}


def _review_mask(data, lite_gate):
    p = data["report"]
    direct = (p["disagree"] | (p["a_margin"] < 0.15) |
              (p["b_margin"] < 0.15) | (p["a_conf"] < 0.55) |
              (p["b_conf"] < 0.55))
    return direct | (data["base_scores"]["stage5_plus_c_lite"] >= lite_gate)


def _risk(data, method, lite_gate):
    base = data["base_scores"]
    if method == "stage5_openmax_component":
        return base[method], np.zeros(len(data["labels"]), dtype=bool)
    if method == "stage5_plus_c_lite":
        return base[method], np.zeros(len(data["labels"]), dtype=bool)
    if method == "stage5_plus_c_full_all":
        return base["stage5_plus_c_full"], np.ones(len(data["labels"]), dtype=bool)
    mask = _review_mask(data, lite_gate)
    if method == "stage5_plus_c_full_routed":
        risk = base["stage5_plus_c_full"]
    elif method == "stage5_plus_c_raw_pug":
        risk = data["raw_pug_scores"]["stage5_plus_c_pug"]
    elif method == "stage5_plus_c_certified_pug":
        risk = data["certified_pug_scores"]["stage5_plus_c_pug"]
    else:
        raise ValueError(method)
    return np.where(mask, risk, base["stage5_plus_c_lite"]), mask


METHODS = ("stage5_openmax_component", "stage5_plus_c_lite",
           "stage5_plus_c_full_all", "stage5_plus_c_full_routed",
           "stage5_plus_c_raw_pug", "stage5_plus_c_certified_pug")


def _metrics(cal, known, proxy, method, acceptance, lite_gate):
    c, _ = _risk(cal, method, lite_gate)
    k, km = _risk(known, method, lite_gate)
    u, um = _risk(proxy, method, lite_gate)
    return evaluate(c, k, u, known["b_pred"], known["labels"],
                    acceptance=acceptance, known_full=km, unknown_full=um)


def _fold(cfg, splits, fold, device, mother_cfg):
    checkpoint_path = Path(cfg["fold_checkpoint_pattern"].format(fold=fold.index))
    model, checkpoint = load_mother(checkpoint_path, mother_cfg,
                                    fold.known_classes, fold.proxy_unknown_classes,
                                    device)
    evt = OpenMaxEVT.from_state_dict(checkpoint["openmax"])
    def cap(ds, field, salt, proxy=False):
        return cap_dataset(ds, int(cfg[field]), int(cfg["seed"]) + salt +
                           fold.index * 17, proxy_unknown=proxy)
    sets = {
        "train": cap(fold.train_known, "train_samples_per_class", 11),
        "cal": cap(fold.calibration_known, "cal_samples_per_class", 12),
        "val_proxy": cap(_subset_val_proxy(splits, fold.proxy_unknown_classes,
                                           fold.index),
                         "cal_samples_per_class", 15, True),
        "test": cap(fold.test_known, "test_samples_per_class", 13),
        "test_proxy": cap(fold.test_proxy_unknown, "test_samples_per_class", 14, True)}
    data = {name: collect(model, evt, ds, device, int(cfg["batch_size"]))
            for name, ds in sets.items()}
    support = SupportEvidence().fit(data["cal"])
    for item in data.values():
        support.transform(item)
    generated = iq_candidates(data["train"],
                              int(cfg["max_seeds_per_class"]),
                              tuple(float(x) for x in cfg["etas"]),
                              int(cfg["seed"]) + fold.index)
    pseudo = collect(model, evt, _pseudo_dataset(generated["iq"]), device,
                     int(cfg["batch_size"]), pseudo=True)
    support.transform(pseudo)
    cert = certificate(generated, pseudo, data["train"], data["cal"])
    raw_tool = PUGTool(seed=int(cfg["seed"]) + fold.index).fit(
        data["train"], pseudo, np.ones(len(pseudo["labels"]), dtype=bool))
    certified_tool = PUGTool(seed=int(cfg["seed"]) + fold.index).fit(
        data["train"], pseudo, cert["accepted"])
    cal_raw = ClassConditionalCdf(raw_tool.raw_score(data["cal"]),
                                   data["cal"]["b_pred"])
    cal_cert = (ClassConditionalCdf(certified_tool.raw_score(data["cal"]),
                                    data["cal"]["b_pred"])
                if certified_tool.enabled else None)
    for item in data.values():
        item["base_scores"] = _base_scores(item)
        item["raw_pug_scores"] = _base_scores(item, raw_tool, cal_raw)
        item["certified_pug_scores"] = _base_scores(item, certified_tool, cal_cert)
    gate = float(np.quantile(
        data["cal"]["base_scores"]["stage5_plus_c_lite"],
        float(cfg["lite_escalation_quantile"])))
    acceptance = float(cfg["known_acceptance"])
    selection = {name: _metrics(data["cal"], data["cal"], data["val_proxy"],
                                name, acceptance, gate) for name in METHODS}
    test = {name: _metrics(data["cal"], data["test"], data["test_proxy"],
                           name, acceptance, gate) for name in METHODS}
    off = selection["stage5_plus_c_full_routed"]
    on = selection["stage5_plus_c_certified_pug"]
    utility = {"delta_h": on["h_score"] - off["h_score"],
               "delta_auroc": on["auroc"] - off["auroc"]}
    pug_action = ("USE" if certified_tool.enabled and utility["delta_h"] > 0
                  and utility["delta_auroc"] >= -0.01 else "IGNORE")
    selected_method = ("stage5_plus_c_certified_pug" if pug_action == "USE"
                       else "stage5_plus_c_full_routed")
    p = pseudo["report"]
    src = generated["source_index"]
    original_a = data["train"]["report"]["a_conf"][src]
    original_b = data["train"]["report"]["b_conf"][src]
    other_core = ((pseudo["b_pred"] != data["train"]["labels"][src]) &
                  (pseudo["u_proto"] < 0.5) & (p["b_conf"] >= 0.6))
    accepted = cert["accepted"]
    diagnostics = {
        "generated": int(len(accepted)), "certified": int(accepted.sum()),
        "certification_rate": float(accepted.mean()),
        "boundary_certified": int(cert["boundary"].sum()),
        "hard_certified": int(cert["hard"].sum()),
        "hard_fraction_among_certified": float(cert["hard"].sum() / max(accepted.sum(), 1)),
        "energy_failed": int((~cert["plausible_energy"]).sum()),
        "too_far_failed": int((~cert["not_too_far"]).sum()),
        "c_support_failed": int((~cert["c_outside_support"]).sum()),
        "fell_into_other_known_core": int(other_core.sum()),
        "distance_cap": cert["distance_cap"],
        "nearest_distance_median_all": float(np.median(cert["nearest_known_distance"])),
        "nearest_distance_median_certified": (float(np.median(
            cert["nearest_known_distance"][accepted])) if accepted.any() else None),
        "mean_abs_a_confidence_change_after_fresh_forward": float(np.mean(
            np.abs(p["a_conf"] - original_a))),
        "mean_abs_b_confidence_change_after_fresh_forward": float(np.mean(
            np.abs(p["b_conf"] - original_b))),
        "mean_a_confidence_generated": float(np.mean(p["a_conf"])),
        "mean_b_confidence_generated": float(np.mean(p["b_conf"])),
        "by_eta": {str(eta): {"generated": int(np.sum(generated["eta"] == eta)),
                              "certified": int(np.sum(accepted &
                                  (generated["eta"] == eta)))}
                   for eta in sorted(set(generated["eta"]))},
    }
    test_proxy = data["test_proxy"]
    agree = (~test_proxy["report"]["disagree"] &
             (test_proxy["report"]["a_conf"] >= 0.6) &
             (test_proxy["report"]["b_conf"] >= 0.6))
    chosen_cal, _ = _risk(data["cal"], selected_method, gate)
    chosen_proxy, chosen_full = _risk(test_proxy, selected_method, gate)
    baseline_cal, _ = _risk(data["cal"], "stage5_openmax_component", gate)
    baseline_proxy, _ = _risk(test_proxy, "stage5_openmax_component", gate)
    threshold = float(np.quantile(chosen_cal, acceptance))
    baseline_threshold = float(np.quantile(baseline_cal, acceptance))
    chosen_reject = chosen_proxy >= threshold
    baseline_reject = baseline_proxy >= baseline_threshold
    a_good = data["test"]["a_pred"] == data["test"]["labels"]
    b_good = data["test"]["b_pred"] == data["test"]["labels"]
    return {
        "fold": fold.index, "checkpoint_sha256": _sha(checkpoint_path),
        "support_classes": list(fold.known_classes),
        "proxy_unknown_classes": list(fold.proxy_unknown_classes),
        "counts": {name: int(len(item["labels"])) for name, item in data.items()},
        "a_known_accuracy": float(a_good.mean()),
        "b_known_accuracy": float(b_good.mean()),
        "a_only_correct": int(np.sum(a_good & ~b_good)),
        "b_only_correct": int(np.sum(b_good & ~a_good)),
        "disagreement_rate": float(data["test"]["report"]["disagree"].mean()),
        "pug_certificate": diagnostics,
        "pug_tool_enabled": certified_tool.enabled,
        "pug_validation_utility": utility,
        "pug_action": pug_action,
        "selected_method": selected_method,
        "lite_gate": gate,
        "selection_proxy": selection,
        "test_proxy": test,
        "test_selected": test[selected_method],
        "high_conf_agree_proxy_count": int(agree.sum()),
        "high_conf_agree_proxy_full_reviewed": int(np.sum(agree & chosen_full)),
        "high_conf_agree_proxy_rejected": int(np.sum(
            agree & chosen_reject)),
        "high_conf_agree_proxy_newly_rejected_vs_stage5_openmax": int(np.sum(
            agree & chosen_reject & ~baseline_reject)),
        "high_conf_agree_proxy_lost_rejections_vs_stage5_openmax": int(np.sum(
            agree & ~chosen_reject & baseline_reject)),
        "c_action_counts_proxy": {
            "REJECT": int(chosen_reject.sum()),
            "CHALLENGE_then_KEEP": int(np.sum(chosen_full & ~chosen_reject)),
            "KEEP_without_full": int(np.sum(~chosen_full & ~chosen_reject)),
        },
        "formal_unknown_used": False,
    }


def run_dataset(cfg, output_root):
    _seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and
                          torch.cuda.is_available() else "cpu")
    splits = _load_dataset(cfg)
    protocol = build_nested_lco_protocol(splits, outer_folds=5, inner_folds=4,
                                         seed=int(cfg["partition_seed"]))
    mother_cfg = json.loads(Path(cfg["mother_config"]).read_text(encoding="utf-8"))
    rows = []
    for fold in protocol.outer_folds:
        print(f"[certified-pug] {cfg['dataset']} fold {fold.index}/4", flush=True)
        rows.append(_fold(cfg, splits, fold, device, mother_cfg))
    means = {name: {metric: float(np.mean([
        row["test_proxy"][name][metric] for row in rows]))
        for metric in ("h_score", "auroc", "unknown_recall", "known_accuracy", "oscr", "fpr95")}
        for name in METHODS}
    selected = {metric: float(np.mean([row["test_selected"][metric]
                                     for row in rows]))
                for metric in ("h_score", "auroc", "unknown_recall", "known_accuracy",
                               "oscr", "fpr95")}
    result = {"stage": "stage12_certified_iq_pug_proxy_probe",
              "dataset": cfg["dataset"], "seed": cfg["seed"],
              "method_config_sha256": _sha(cfg["config_path"]),
              "mother_config_sha256": _sha(cfg["mother_config"]),
              "formal_unknown_used": False,
              "formal_unknown_count_not_accessed": protocol.formal_unknown_sample_count,
              "baseline_scope": "Stage-5 frozen expert OpenMax component; historical full Stage-5 communication only contextual",
              "folds": rows, "test_proxy_means": means,
              "utility_selected_test_proxy_mean": selected,
              "pug_use_fold_count": int(sum(row["pug_action"] == "USE" for row in rows)),
              "total_generated": int(sum(row["pug_certificate"]["generated"] for row in rows)),
              "total_certified": int(sum(row["pug_certificate"]["certified"] for row in rows)),
              "total_hard_certified": int(sum(row["pug_certificate"]["hard_certified"] for row in rows))}
    out = Path(output_root) / cfg["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "proxy_fivefold.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_pair(oracle, wisig, output_root):
    metadata = {"base_config", "config_path", "dataset", "name", "oracle_root",
                "wisig_pkl", "mother_config", "fold_checkpoint_pattern",
                "paired_config"}
    a = {key: value for key, value in oracle.items() if key not in metadata}
    b = {key: value for key, value in wisig.items() if key not in metadata}
    if a != b:
        raise ValueError("paired Certified PUG algorithm settings must match")
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {"oracle": run_dataset(oracle, out),
                "wisig": run_dataset(wisig, out)}
    summary = {"stage": "stage12_certified_iq_pug_proxy_probe",
               "formal_unknown_used": False,
               "datasets": {name: {"test_proxy_means": row["test_proxy_means"],
                                   "utility_selected_test_proxy_mean": row[
                                       "utility_selected_test_proxy_mean"],
                                   "pug_use_fold_count": row["pug_use_fold_count"],
                                   "total_generated": row["total_generated"],
                                   "total_certified": row["total_certified"],
                                   "total_hard_certified": row["total_hard_certified"]}
                            for name, row in datasets.items()}}
    (out / "paired_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
