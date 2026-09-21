"""Strict paired Stage-5-full + C + feature Certified PUG proxy test."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from maros_stage5.calibration_agent import (ClassConditionalThresholdAgent,
                                             DecisionArbitratorAgent)
from maros_stage5.openmax import OpenMaxEVT
from maros_stage5.training import (ExpertRecords, extract_records,
                                   predict_coordinator, pseudo_to_records,
                                   refresh_empirical_prototypes,
                                   train_coordinator)
from maros_stage7.splits import assert_no_formal_unknown, build_nested_lco_protocol
from maros_stage10.experiment import _load_dataset, _seed
from maros_stage10.mother import cap_dataset, load_mother
from maros_stage11.experiment import _subset_val_proxy
from maros_stage12.certified_pug import iq_candidates
from maros_staged.evidence import ClassConditionalCdf, EmpiricalCdfCalibrator
from maros_staged.metrics_osr import detection_metrics, oscr, threshold_decision

from .feature_pug import (CSupport, FeaturePUGTool, certify,
                          generate_candidates, incremental)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_selection_tail(path):
    """Read the immutable selected block without loading the 70+ MB audit JSON."""
    path = Path(path)
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        text = f.read().decode("utf-8")
    start = text.rfind('"selected":')
    if start < 0:
        raise ValueError("cannot locate frozen Stage-5 selection")
    try:
        return json.JSONDecoder().raw_decode(
            text[start + len('"selected":'):].lstrip())[0]
    except json.JSONDecodeError as error:
        raise ValueError("cannot decode frozen Stage-5 selection") from error


def _cap_train(dataset, cfg, fold_index):
    capped = cap_dataset(dataset, int(cfg["train_samples_per_class"]),
                         int(cfg["seed"]) + 101 + fold_index)
    maximum = int(cfg["lco_max_samples"])
    if len(capped) <= maximum:
        return capped
    rng = np.random.default_rng(int(cfg["seed"]) + 701 + fold_index)
    keep = np.sort(rng.choice(len(capped), maximum, replace=False))
    return Subset(capped, keep.tolist())


def _concat(a: ExpertRecords, b: ExpertRecords):
    return ExpertRecords(**{
        key: (None if getattr(a, key) is None else
              np.concatenate([getattr(a, key), getattr(b, key)]))
        for key in a.__dataclass_fields__})


def _summary(logits):
    z = np.asarray(logits, dtype=float)
    z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    top = np.sort(p, axis=1)[:, -2:]
    return p.argmax(1), top[:, 1], top[:, 1] - top[:, 0]


def _iq_array(dataset):
    assert_no_formal_unknown(dataset)
    return np.stack([dataset[index][0].numpy() for index in range(len(dataset))])


def _base_system(model, evt, train, calibration, selection, mother_cfg,
                 device, seed):
    kind, eta = selection["pug"].split("_eta")
    pseudo_batch, _ = generate_candidates(train, kind, float(eta), seed)
    pseudo = pseudo_to_records(model, pseudo_batch, evt, device)
    cfg = copy.deepcopy(mother_cfg)
    cfg["coordinator_epochs"] = int(mother_cfg.get("lco_coordinator_epochs", 10))
    coordinator, scaler, history, best = train_coordinator(
        train, calibration, pseudo, pseudo_batch.source_indices,
        train.identity_logits.shape[1], cfg, device, seed,
        selection["mode"], float(selection["budget"]),
        float(selection["gate_threshold"]))
    val_comm = predict_coordinator(coordinator, calibration, scaler, device)
    val_local = predict_coordinator(coordinator, calibration, scaler, device,
                                    intervention="messages_off")
    arbitrator = DecisionArbitratorAgent(
        selection["score_rule"], selection["arbitration_rule"]).fit(
            calibration, val_comm, val_local)
    return coordinator, scaler, arbitrator, history, best, len(pseudo.labels)


def _base_outputs(records, coordinator, scaler, arbitrator, device):
    comm = predict_coordinator(coordinator, records, scaler, device)
    local = predict_coordinator(coordinator, records, scaler, device,
                                intervention="messages_off")
    risk = arbitrator.transform(records, comm, local).unknown_score
    return {"risk": risk, "pred": comm["pred"],
            "gates": comm["gates"], "weights": comm["known_weights"]}


def _evaluate(cal_risk, known_risk, proxy_risk, cal_pred, known_pred,
              known_y, proxy_pred, acceptance, prior):
    calibrator = EmpiricalCdfCalibrator(cal_risk)
    c = calibrator.score(cal_risk)
    k = calibrator.score(known_risk)
    u = calibrator.score(proxy_risk)
    if prior is None:
        kt = np.full(len(k), acceptance)
        ut = np.full(len(u), acceptance)
    else:
        service = ClassConditionalThresholdAgent(acceptance, float(prior)).fit(c, cal_pred)
        kt = service.thresholds(known_pred)
        ut = service.thresholds(proxy_pred)
    score = np.r_[k, u]
    y = np.r_[known_y, np.full(len(u), -1)]
    pred = np.r_[known_pred, proxy_pred]
    is_unknown = (y == -1).astype(int)
    thresholds = np.r_[kt, ut]
    metrics = detection_metrics(is_unknown, score)
    metrics["oscr"] = oscr(is_unknown, pred, y, 1-score)
    metrics.update(threshold_decision(is_unknown, y, pred, score, thresholds))
    metrics["known_reject_count"] = int(np.sum(k >= kt))
    metrics["proxy_reject_count"] = int(np.sum(u >= ut))
    return metrics, {"cal": c, "known": k, "proxy": u,
                     "known_threshold": kt, "proxy_threshold": ut}


def _route(records, support, base_risk, c_weight, lite_gate, pug_score=None,
           pug_weight=0.0):
    c = support.transform(records)
    lite = np.maximum.reduce([c["openmax"], c["geometry"], c["identity"]])
    ap, ac, am = _summary(records.identity_logits)
    bp, bc, bm = _summary(records.geometry_logits)
    direct = (ap != bp) | (ac < .55) | (bc < .55) | (am < .15) | (bm < .15)
    review = direct | (lite >= lite_gate)
    risk = incremental(base_risk, lite, c_weight)
    risk = np.where(review, risk, base_risk)
    if pug_score is not None and pug_weight > 0:
        risk = np.where(review, incremental(risk, pug_score, pug_weight), risk)
    return risk, review


def _candidate_tool(train, calibration, pseudo, mask, support, seed):
    tool = FeaturePUGTool(seed).fit(train, pseudo, support, mask)
    if not tool.enabled:
        return tool, None
    raw = tool.raw_score(calibration, support)
    return tool, ClassConditionalCdf(raw, calibration.geometry_logits.argmax(1))


def _tool_score(tool, calibrator, records, support):
    if not tool.enabled or calibrator is None:
        return np.zeros(len(records.labels))
    return calibrator.score(tool.raw_score(records, support),
                            records.geometry_logits.argmax(1))


def _select_weight(data, base_outputs, support, c_weight, lite_gate,
                   pug_scores, weights, acceptance, prior):
    rows = []
    for weight in weights:
        risk = {}
        for name in ("cal", "cal_proxy"):
            risk[name], _ = _route(data[name], support, base_outputs[name]["risk"],
                                   c_weight, lite_gate, pug_scores[name], weight)
        metrics, _ = _evaluate(
            risk["cal"], risk["cal"], risk["cal_proxy"],
            base_outputs["cal"]["pred"], base_outputs["cal"]["pred"],
            data["cal"].labels, base_outputs["cal_proxy"]["pred"],
            acceptance, prior)
        rows.append({"weight": float(weight), "metrics": metrics})
    baseline = rows[0]["metrics"]["known_accuracy"]
    feasible = [row for row in rows
                if row["metrics"]["known_accuracy"] >= baseline - 0.005]
    chosen = max(feasible or rows, key=lambda row: (
        row["metrics"]["h_score"], row["metrics"]["auroc"],
        row["metrics"]["oscr"], -row["weight"]))
    return rows, chosen


def _old_iq_tool(model, evt, train_dataset, train, calibration, support,
                 cfg, device, seed):
    iq = _iq_array(train_dataset)
    ap, ac, am = _summary(train.identity_logits)
    bp, bc, bm = _summary(train.geometry_logits)
    train_support = support.transform(train)
    packet = {"iq": iq, "labels": train.labels,
              "report": {"a_margin": am, "b_margin": bm,
                         "disagree": ap != bp},
              "u_proto": train_support["geometry"]}
    generated = iq_candidates(packet, int(cfg["iq_max_seeds_per_class"]),
                              tuple(float(x) for x in cfg["iq_etas"]), seed)
    dataset = TensorDataset(torch.from_numpy(generated["iq"]),
                            torch.full((len(generated["iq"]),), -1,
                                       dtype=torch.long))
    pseudo = extract_records(model, dataset, device, evt,
                             batch_size=int(cfg["batch_size"]))
    tool, calibrator = _candidate_tool(
        train, calibration, pseudo, np.ones(len(pseudo.labels), dtype=bool),
        support, seed)
    return tool, calibrator, int(len(pseudo.labels))


def _fold(cfg, splits, fold, mother_cfg, selection, device):
    checkpoint_path = Path(cfg["fold_checkpoint_pattern"].format(fold=fold.index))
    model, checkpoint = load_mother(checkpoint_path, mother_cfg,
                                    fold.known_classes, fold.proxy_unknown_classes,
                                    device)
    train_dataset = _cap_train(fold.train_known, cfg, fold.index)
    refresh_empirical_prototypes(model, train_dataset, device,
                                 int(cfg["batch_size"]))
    evt = OpenMaxEVT.from_state_dict(checkpoint["openmax"])
    datasets = {"train": train_dataset, "cal": fold.calibration_known,
                "cal_proxy": _subset_val_proxy(splits, fold.proxy_unknown_classes,
                                                 fold.index),
                "test": fold.test_known, "test_proxy": fold.test_proxy_unknown}
    for dataset in datasets.values():
        assert_no_formal_unknown(dataset)
    data = {name: extract_records(model, dataset, device, evt,
                                  batch_size=int(cfg["batch_size"]))
            for name, dataset in datasets.items()}
    coordinator, scaler, arbitrator, history, best, base_pseudo_count = _base_system(
        model, evt, data["train"], data["cal"], selection, mother_cfg,
        device, int(cfg["seed"]) + fold.index * 100)
    base = {name: _base_outputs(records, coordinator, scaler, arbitrator, device)
            for name, records in data.items() if name != "train"}
    support = CSupport().fit(data["cal"])
    acceptance = float(selection["known_acceptance"])
    prior = selection.get("adaptive_threshold_prior")
    base_metrics, base_detail = _evaluate(
        base["cal"]["risk"], base["test"]["risk"], base["test_proxy"]["risk"],
        base["cal"]["pred"], base["test"]["pred"], data["test"].labels,
        base["test_proxy"]["pred"], acceptance, prior)
    cal_c = support.transform(data["cal"])
    cal_lite = np.maximum.reduce([cal_c["openmax"], cal_c["geometry"],
                                  cal_c["identity"]])
    lite_gate = float(np.quantile(cal_lite,
                                  float(cfg["lite_escalation_quantile"])))
    # C is selected as a non-negative increment over the complete Stage-5 risk.
    c_rows = []
    for weight in cfg["c_weight_grid"]:
        risks = {name: _route(data[name], support, base[name]["risk"],
                              float(weight), lite_gate)[0]
                 for name in ("cal", "cal_proxy")}
        metrics, _ = _evaluate(risks["cal"], risks["cal"], risks["cal_proxy"],
                               base["cal"]["pred"], base["cal"]["pred"],
                               data["cal"].labels, base["cal_proxy"]["pred"],
                               acceptance, prior)
        c_rows.append({"weight": float(weight), "metrics": metrics})
    base_val_known = c_rows[0]["metrics"]["known_accuracy"]
    c_pool = [r for r in c_rows if r["metrics"]["known_accuracy"] >=
              base_val_known - 0.005]
    c_choice = max(c_pool or c_rows, key=lambda r: (
        r["metrics"]["h_score"], r["metrics"]["auroc"], -r["weight"]))
    c_weight = c_choice["weight"]
    c_risk = {name: _route(data[name], support, base[name]["risk"],
                           c_weight, lite_gate)[0]
              for name in ("cal", "test", "test_proxy")}
    c_metrics, c_detail = _evaluate(
        c_risk["cal"], c_risk["test"], c_risk["test_proxy"],
        base["cal"]["pred"], base["test"]["pred"], data["test"].labels,
        base["test_proxy"]["pred"], acceptance, prior)

    options_raw, options_cert, diagnostics = [], [], []
    objects = {}
    for kind in cfg["feature_kinds"]:
        for eta in cfg["feature_etas"]:
            key = f"{kind}_eta{float(eta):g}"
            batch, generator = generate_candidates(
                data["train"], kind, float(eta),
                int(cfg["seed"]) + fold.index * 1000)
            pseudo = pseudo_to_records(model, batch, evt, device)
            mask, diagnostic = certify(data["train"], batch, pseudo, support)
            raw_tool, raw_cal = _candidate_tool(
                data["train"], data["cal"], pseudo,
                np.ones(len(pseudo.labels), dtype=bool), support,
                int(cfg["seed"]) + fold.index)
            cert_tool, cert_cal = _candidate_tool(
                data["train"], data["cal"], pseudo, mask, support,
                int(cfg["seed"]) + fold.index)
            diagnostic.update({"kind": kind, "eta": float(eta),
                               "generator": generator,
                               "fresh_state_forward": True})
            diagnostics.append(diagnostic)
            for label, tool, calibrator, rows in (
                    ("raw", raw_tool, raw_cal, options_raw),
                    ("certified", cert_tool, cert_cal, options_cert)):
                scores = {name: _tool_score(tool, calibrator, data[name], support)
                          for name in ("cal", "cal_proxy", "test", "test_proxy")}
                weight_rows, chosen = _select_weight(
                    data, base, support, c_weight, lite_gate, scores,
                    [float(x) for x in cfg["pug_weight_grid"]],
                    acceptance, prior)
                row = {"key": key, "kind": kind, "eta": float(eta),
                       "pool": label, "tool_enabled": tool.enabled,
                       "weight_rows": weight_rows, "chosen": chosen,
                       "certified_count": int(mask.sum())}
                rows.append(row)
                objects[(label, key)] = (tool, calibrator, scores)
    def best_option(rows):
        enabled = [r for r in rows if r["tool_enabled"]]
        return max(enabled or rows, key=lambda r: (
            r["chosen"]["metrics"]["h_score"],
            r["chosen"]["metrics"]["auroc"],
            r["chosen"]["metrics"]["oscr"],
            -r["chosen"]["weight"]))
    raw_choice, cert_choice = best_option(options_raw), best_option(options_cert)

    def evaluate_choice(label, choice):
        _, _, scores = objects[(label, choice["key"])]
        weight = float(choice["chosen"]["weight"])
        risk, review = {}, {}
        for name in ("cal", "test", "test_proxy"):
            risk[name], review[name] = _route(
                data[name], support, base[name]["risk"], c_weight, lite_gate,
                scores[name], weight)
        metrics, detail = _evaluate(
            risk["cal"], risk["test"], risk["test_proxy"],
            base["cal"]["pred"], base["test"]["pred"], data["test"].labels,
            base["test_proxy"]["pred"], acceptance, prior)
        return metrics, detail, risk, review
    raw_metrics, raw_detail, raw_risk, raw_review = evaluate_choice("raw", raw_choice)
    cert_metrics, cert_detail, cert_risk, cert_review = evaluate_choice(
        "certified", cert_choice)
    val_off = c_choice["metrics"]
    val_on = cert_choice["chosen"]["metrics"]
    utility = {"delta_h": val_on["h_score"] - val_off["h_score"],
               "delta_auroc": val_on["auroc"] - val_off["auroc"],
               "delta_unknown_recall": val_on["unknown_recall"] -
                                       val_off["unknown_recall"],
               "delta_known_accuracy": val_on["known_accuracy"] -
                                       val_off["known_accuracy"]}
    action = ("USE" if utility["delta_h"] > 0 and
              utility["delta_auroc"] >= -0.01 and
              utility["delta_known_accuracy"] >= -0.005 else
              "DOWNWEIGHT" if utility["delta_h"] > 0 else "IGNORE")
    utility_metrics = cert_metrics if action in {"USE", "DOWNWEIGHT"} else c_metrics
    utility_detail = cert_detail if action in {"USE", "DOWNWEIGHT"} else c_detail

    # Strict paired Stage-12 I/Q interpolation failure-scheme control.
    iq_tool, iq_cal, iq_count = _old_iq_tool(
        model, evt, train_dataset, data["train"], data["cal"], support,
        cfg, device, int(cfg["seed"]) + fold.index)
    iq_scores = {name: _tool_score(iq_tool, iq_cal, data[name], support)
                 for name in ("cal", "cal_proxy", "test", "test_proxy")}
    iq_rows, iq_choice = _select_weight(
        data, base, support, c_weight, lite_gate, iq_scores,
        [float(x) for x in cfg["pug_weight_grid"]], acceptance, prior)
    iq_risk = {name: _route(data[name], support, base[name]["risk"], c_weight,
                            lite_gate, iq_scores[name], iq_choice["weight"])[0]
               for name in ("cal", "test", "test_proxy")}
    iq_metrics, _ = _evaluate(
        iq_risk["cal"], iq_risk["test"], iq_risk["test_proxy"],
        base["cal"]["pred"], base["test"]["pred"], data["test"].labels,
        base["test_proxy"]["pred"], acceptance, prior)

    ap, ac, am = _summary(data["test_proxy"].identity_logits)
    bp, bc, bm = _summary(data["test_proxy"].geometry_logits)
    high = (ap == bp) & (ac >= .6) & (bc >= .6) & (am >= .15) & (bm >= .15)
    base_reject = base_detail["proxy"] >= base_detail["proxy_threshold"]
    final_reject = utility_detail["proxy"] >= utility_detail["proxy_threshold"]
    known_base_reject = base_detail["known"] >= base_detail["known_threshold"]
    known_final_reject = utility_detail["known"] >= utility_detail["known_threshold"]
    a_good = data["test"].identity_logits.argmax(1) == data["test"].labels
    b_good = data["test"].geometry_logits.argmax(1) == data["test"].labels
    return {
        "fold": fold.index, "checkpoint_sha256": _sha(checkpoint_path),
        "frozen_stage5_selection": selection,
        "stage5_coordinator": {"epochs": best, "last_history": history[-1],
                               "training_pseudo_count": base_pseudo_count},
        "counts": {name: int(len(records.labels)) for name, records in data.items()},
        "known": {"a_accuracy": float(a_good.mean()),
                  "b_accuracy": float(b_good.mean()),
                  "a_only_correct": int(np.sum(a_good & ~b_good)),
                  "b_only_correct": int(np.sum(b_good & ~a_good)),
                  "disagreement_rate": float(np.mean(
                      data["test"].identity_logits.argmax(1) !=
                      data["test"].geometry_logits.argmax(1)))},
        "B0_stage5_full": base_metrics,
        "B1_stage5_plus_c": c_metrics,
        "B2_raw_feature_pug": raw_metrics,
        "B3_certified_feature_pug": cert_metrics,
        "B4_utility_gate": utility_metrics,
        "Old_stage12_iq_pug": iq_metrics,
        "c_selection": {"rows": c_rows, "chosen": c_choice,
                        "lite_gate": lite_gate},
        "raw_selection": raw_choice, "certified_selection": cert_choice,
        "utility": utility, "pug_action": action,
        "iq_selection": {"generated": iq_count, "rows": iq_rows,
                         "chosen": iq_choice},
        "feature_diagnostics": diagnostics,
        "high_conf_agree_proxy": {
            "count": int(high.sum()),
            "stage5_rejected": int(np.sum(high & base_reject)),
            "c_or_certified_new_rescue": int(np.sum(high & final_reject & ~base_reject)),
            "lost_stage5_rejection": int(np.sum(high & ~final_reject & base_reject)),
            "stage5_plus_c_new_rescue": int(np.sum(high &
                (c_detail["proxy"] >= c_detail["proxy_threshold"]) & ~base_reject)),
            "certified_incremental_rescue_over_c": int(np.sum(high & final_reject &
                ~(c_detail["proxy"] >= c_detail["proxy_threshold"])))},
        "known_incremental_false_reject": int(np.sum(
            ~known_base_reject & known_final_reject)),
        "review_fraction": {"known": float(np.mean(
            _route(data["test"], support, base["test"]["risk"], c_weight,
                   lite_gate)[1])),
                            "proxy_unknown": float(np.mean(cert_review["test_proxy"]))},
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
    selection = load_selection_tail(cfg["stage5_selection"])
    rows = []
    for fold in protocol.outer_folds:
        print(f"[stage13] {cfg['dataset']} fold {fold.index}/4", flush=True)
        rows.append(_fold(cfg, splits, fold, mother_cfg, selection, device))
    methods = ("B0_stage5_full", "B1_stage5_plus_c", "B2_raw_feature_pug",
               "B3_certified_feature_pug", "B4_utility_gate",
               "Old_stage12_iq_pug")
    metrics = ("h_score", "unknown_recall", "known_accuracy", "auroc",
               "oscr", "fpr95")
    means = {method: {metric: float(np.mean([row[method][metric] for row in rows]))
                      for metric in metrics} for method in methods}
    result = {"stage": "stage13_inherited_abc_feature_certified_pug",
              "dataset": cfg["dataset"], "seed": cfg["seed"],
              "formal_unknown_used": False,
              "formal_unknown_count_not_accessed": protocol.formal_unknown_sample_count,
              "stage5_selection_sha256": _sha(cfg["stage5_selection"]),
              "stage5_selection": selection,
              "method_config_sha256": _sha(cfg["config_path"]),
              "folds": rows, "test_proxy_means": means,
              "pug_actions": {name: sum(r["pug_action"] == name for r in rows)
                              for name in ("USE", "DOWNWEIGHT", "IGNORE")},
              "high_conf_agree_proxy": {
                  key: int(sum(r["high_conf_agree_proxy"][key] for r in rows))
                  for key in rows[0]["high_conf_agree_proxy"]},
              "known_incremental_false_reject": int(sum(
                  r["known_incremental_false_reject"] for r in rows))}
    out = Path(output_root) / cfg["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "proxy_fivefold.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_pair(oracle, wisig, output_root):
    metadata = {"base_config", "config_path", "dataset", "name", "oracle_root",
                "wisig_pkl", "mother_config", "fold_checkpoint_pattern",
                "stage5_selection", "paired_config"}
    a = {k: v for k, v in oracle.items() if k not in metadata}
    b = {k: v for k, v in wisig.items() if k not in metadata}
    if a != b:
        raise ValueError("paired Stage-13 method settings differ")
    out = Path(output_root); out.mkdir(parents=True, exist_ok=True)
    datasets = {"oracle": run_dataset(oracle, out),
                "wisig": run_dataset(wisig, out)}
    summary = {"stage": "stage13_inherited_abc_feature_certified_pug",
               "formal_unknown_used": False,
               "datasets": {name: {"test_proxy_means": value["test_proxy_means"],
                                   "pug_actions": value["pug_actions"],
                                   "high_conf_agree_proxy": value[
                                       "high_conf_agree_proxy"],
                                   "known_incremental_false_reject": value[
                                       "known_incremental_false_reject"]}
                            for name, value in datasets.items()}}
    (out / "paired_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
