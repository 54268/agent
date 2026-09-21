"""Stage-5-mother A/B consultation feasibility, before building Open-Set C."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from maros_staged.datasets import load_oracle_npz, load_wisig_subset
from maros_stage7.splits import (assert_no_formal_unknown,
                                 build_nested_lco_protocol)
from maros_stage8.evidence import shuffle_certificate

from .consultation import (IdentityQueryPolicy, IndependentEvidenceInvestigator,
                           apply_certificate, per_sample_nll,
                           publish_candidate_support, top_pair)
from .mother import cap_dataset, collect_evidence, load_mother
from .open_set import OpenSetExaminer, identity_public_after_scores
from .services import KnownQuantileThresholdService
from .system import Stage10System


VIEW_NAMES = ("raw", "spectral", "envelope_phase", "difference_iq", "complex_iq")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _load_dataset(cfg: dict):
    if cfg["dataset"] == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=False)
    if cfg["dataset"] == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=False)
    raise ValueError("unsupported dataset")


def _metrics(logits: np.ndarray, labels: np.ndarray,
             query_mask: np.ndarray) -> dict:
    prediction = logits.argmax(1)
    return {
        "known_accuracy": float(np.mean(prediction == labels)),
        "known_nll": float(np.mean(per_sample_nll(logits, labels))),
        "query_fraction": float(np.mean(query_mask)),
        "query_count": int(np.sum(query_mask)),
    }


def _changed(base: np.ndarray, update: np.ndarray, labels: np.ndarray,
             mask: np.ndarray) -> dict:
    before = base.argmax(1)
    after = update.argmax(1)
    return {
        "candidate_flips": int(np.sum(mask & (before != after))),
        "rescues": int(np.sum(mask & (before != labels) & (after == labels))),
        "harms": int(np.sum(mask & (before == labels) & (after != labels))),
        "changed_score_samples": int(np.sum(mask &
            (np.abs(base - update).max(1) > 1e-8))),
    }


def _view_usage(data) -> dict:
    return {name: float(data.view_weights[:, index].mean())
            for index, name in enumerate(VIEW_NAMES)}


def _open_metrics(calibration_risk: np.ndarray, known_risk: np.ndarray,
                  proxy_risk: np.ndarray, known_scores: np.ndarray,
                  known_labels: np.ndarray, acceptance: float) -> dict:
    threshold = KnownQuantileThresholdService(acceptance).fit(calibration_risk)
    known_reject = threshold.reject(known_risk)
    proxy_reject = threshold.reject(proxy_risk)
    known_accuracy = float(np.mean(
        (known_scores.argmax(1) == known_labels) & ~known_reject))
    unknown_recall = float(np.mean(proxy_reject))
    h_score = (2 * known_accuracy * unknown_recall /
               (known_accuracy + unknown_recall)
               if known_accuracy + unknown_recall else 0.0)
    auroc = float(roc_auc_score(
        np.r_[np.zeros(len(known_risk)), np.ones(len(proxy_risk))],
        np.r_[known_risk, proxy_risk]))
    return {
        "threshold": threshold.threshold,
        "known_accept_rate": float(np.mean(~known_reject)),
        "known_accuracy": known_accuracy,
        "unknown_recall": unknown_recall,
        "h_score": h_score,
        "auroc": auroc,
    }


def run_dataset(cfg: dict) -> dict:
    _seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto"
                          and torch.cuda.is_available() else
                          cfg["device"] if cfg["device"] != "auto" else "cpu")
    splits = _load_dataset(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=5, inner_folds=4,
        seed=int(cfg["partition_seed"]))
    fold = protocol.outer_folds[int(cfg["outer_fold"])]
    for dataset in (fold.train_known, fold.calibration_known,
                    fold.test_known, fold.test_proxy_unknown):
        assert_no_formal_unknown(dataset)
    mother_cfg = json.loads(Path(cfg["mother_config"]).read_text(encoding="utf-8"))
    checkpoint_path = Path(cfg["mother_checkpoint"])
    mother, checkpoint = load_mother(
        checkpoint_path, mother_cfg, fold.known_classes,
        fold.proxy_unknown_classes, device)
    def capped(dataset, field, offset, *, proxy=False):
        return cap_dataset(dataset, int(cfg[field]), int(cfg["seed"]) + offset,
                           proxy_unknown=proxy)
    train = collect_evidence(mother, capped(fold.train_known,
        "train_samples_per_class", 11), device, int(cfg["batch_size"]))
    calibration = collect_evidence(mother, capped(fold.calibration_known,
        "calibration_samples_per_class", 12), device, int(cfg["batch_size"]))
    test_dataset = capped(fold.test_known, "test_samples_per_class", 13)
    test = collect_evidence(mother, test_dataset, device, int(cfg["batch_size"]))
    proxy = collect_evidence(mother, capped(fold.test_proxy_unknown,
        "test_samples_per_class", 14, proxy=True), device, int(cfg["batch_size"]))

    train_top1, train_top2 = top_pair(train.identity_logits)
    responder = IndependentEvidenceInvestigator().fit(
        train.geometry_private(), train.labels, train_top1, train_top2)
    scale = float(cfg["certificate_scale"])
    val_a, val_b = top_pair(calibration.identity_logits)
    val_certificate = responder.answer(calibration.geometry_private(), val_a, val_b,
                                       np.ones(len(calibration), dtype=bool))
    val_updated = apply_certificate(calibration.identity_logits,
                                    val_certificate, scale)
    gain = (per_sample_nll(calibration.identity_logits, calibration.labels)
            - per_sample_nll(val_updated, calibration.labels))
    policy = IdentityQueryPolicy(
        budget=float(cfg["query_budget"]), cost=float(cfg["query_cost"]),
        seed=int(cfg["seed"])).fit(calibration.identity_public, gain)

    candidate_a, candidate_b = top_pair(test.identity_logits)
    active, expected_gain = policy.ask(test.identity_public)
    count = int(active.sum())
    rng = np.random.default_rng(int(cfg["seed"]) + 101)
    random_mask = np.zeros(len(test), dtype=bool)
    random_mask[rng.choice(len(test), size=count, replace=False)] = True
    confidence_mask = np.zeros(len(test), dtype=bool)
    confidence_mask[np.argsort(test.identity_public[:, 1])[:count]] = True
    masks = {
        "no_communication": np.zeros(len(test), dtype=bool),
        "random_query": random_mask,
        "confidence_central_query": confidence_mask,
        "agent_initiated_query": active,
        "always_query": np.ones(len(test), dtype=bool),
    }
    conditions, interventions, updated_by_condition = {}, {}, {}
    for name, mask in masks.items():
        if not mask.any():
            updated = test.identity_logits.copy()
        else:
            cert = responder.answer(test.geometry_private(), candidate_a, candidate_b, mask)
            updated = apply_certificate(test.identity_logits, cert, scale)
        conditions[name] = {
            **_metrics(updated, test.labels, mask),
            **_changed(test.identity_logits, updated, test.labels, mask),
        }
        if name in {"no_communication", "agent_initiated_query"}:
            updated_by_condition[name] = updated
    if count:
        correct = responder.answer(test.geometry_private(), candidate_a, candidate_b, active)
        order = rng.permutation(len(test)).astype(np.int64)
        shuffled = shuffle_certificate(correct, torch.from_numpy(order))
        shuffled_updated = apply_certificate(test.identity_logits, shuffled, scale)
        interventions["message_shuffle"] = {
            **_metrics(shuffled_updated, test.labels, active),
            **_changed(test.identity_logits, shuffled_updated, test.labels, active),
        }
        third = np.argsort(test.identity_logits, axis=1)[:, -3].astype(np.int64)
        wrong = responder.answer(test.geometry_private(), candidate_a, third, active)
        wrong = replace(wrong, candidate_b=correct.candidate_b,
                        alternative_class=correct.alternative_class)
        wrong_updated = apply_certificate(test.identity_logits, wrong, scale)
        interventions["wrong_candidate_pair"] = {
            **_metrics(wrong_updated, test.labels, active),
            **_changed(test.identity_logits, wrong_updated, test.labels, active),
        }
    a_pred = test.identity_logits.argmax(1)
    b_pred = test.geometry_logits.argmax(1)
    pair_includes_truth = (candidate_a == test.labels) | (candidate_b == test.labels)
    signed = responder.signed_support(test.geometry_private(), candidate_a, candidate_b)
    verifier_choice = np.where(signed >= 0, candidate_a, candidate_b)

    # C is a separate examiner. Its only inputs are A's public scalar summary
    # and B's candidate-specific public support packet. Calibration/threshold
    # remain separate services and no formal Unknown is used.
    examiner = OpenSetExaminer(test.identity_logits.shape[1])
    train_a_public = identity_public_after_scores(
        train.identity_logits, train.identity_public)
    train_b_public = publish_candidate_support(
        train.geometry_private(), train.identity_logits.argmax(1))
    examiner.fit_memory(train_a_public, train_b_public, train.labels)

    def c_opinion(data, scores):
        a_public = identity_public_after_scores(scores, data.identity_public)
        b_public = publish_candidate_support(data.geometry_private(), scores.argmax(1))
        return examiner.examine(a_public, b_public)

    val_active_mask, _ = policy.ask(calibration.identity_public)
    val_active_certificate = responder.answer(
        calibration.geometry_private(), val_a, val_b, val_active_mask)
    val_active_scores = apply_certificate(
        calibration.identity_logits, val_active_certificate, scale)
    proxy_a, proxy_b = top_pair(proxy.identity_logits)
    proxy_active_mask, _ = policy.ask(proxy.identity_public)
    proxy_certificate = responder.answer(proxy.geometry_private(), proxy_a, proxy_b,
                                         proxy_active_mask)
    proxy_active_scores = apply_certificate(
        proxy.identity_logits, proxy_certificate, scale)
    no_scores = updated_by_condition["no_communication"]
    active_scores = updated_by_condition["agent_initiated_query"]
    c_no_val = c_opinion(calibration, calibration.identity_logits)
    c_active_val = c_opinion(calibration, val_active_scores)
    c_no_known = c_opinion(test, no_scores)
    c_active_known = c_opinion(test, active_scores)
    c_no_proxy = c_opinion(proxy, proxy.identity_logits)
    c_active_proxy = c_opinion(proxy, proxy_active_scores)
    val_fusion_scores = 0.5 * (
        calibration.identity_logits + calibration.geometry_logits)
    test_fusion_scores = 0.5 * (
        test.identity_logits + test.geometry_logits)
    proxy_fusion_scores = 0.5 * (
        proxy.identity_logits + proxy.geometry_logits)
    c_fusion_val = c_opinion(calibration, val_fusion_scores)
    c_fusion_known = c_opinion(test, test_fusion_scores)
    c_fusion_proxy = c_opinion(proxy, proxy_fusion_scores)
    acceptance = float(cfg["known_acceptance"])
    open_set = {
        "identity_msp_no_communication": _open_metrics(
            1.0 - calibration.identity_public[:, 0],
            1.0 - test.identity_public[:, 0],
            1.0 - proxy.identity_public[:, 0],
            no_scores, test.labels, acceptance),
        "examiner_no_communication": _open_metrics(
            c_no_val.risk, c_no_known.risk, c_no_proxy.risk,
            no_scores, test.labels, acceptance),
        "examiner_active_query": _open_metrics(
            c_active_val.risk, c_active_known.risk, c_active_proxy.risk,
            active_scores, test.labels, acceptance),
        "examiner_static_mean_fusion": _open_metrics(
            c_fusion_val.risk, c_fusion_known.risk, c_fusion_proxy.risk,
            test_fusion_scores, test.labels, acceptance),
        "examiner_reason_counts_known": {
            name: int(np.sum(c_active_known.dominant_reason == index))
            for index, name in enumerate(examiner.feature_names)},
        "examiner_reason_counts_proxy_unknown": {
            name: int(np.sum(c_active_proxy.dominant_reason == index))
            for index, name in enumerate(examiner.feature_names)},
        "active_query_changed_open_risk_known": int(np.sum(
            np.abs(c_active_known.risk - c_no_known.risk) > 1e-8)),
        "active_query_changed_open_risk_proxy_unknown": int(np.sum(
            np.abs(c_active_proxy.risk - c_no_proxy.risk) > 1e-8)),
    }
    threshold_service = KnownQuantileThresholdService(acceptance).fit(
        c_active_val.risk)
    runtime = Stage10System(
        mother, responder, policy, examiner, threshold_service,
        certificate_scale=scale, device=device, batch_size=int(cfg["batch_size"]))
    # Match the extraction batch shape so CUDA convolution kernels take the
    # same deterministic path as the counterfactual cache.
    replay_count = min(int(cfg["batch_size"]), len(test_dataset))
    replay_iq = torch.stack([test_dataset[index][0]
                             for index in range(replay_count)])
    live = runtime.infer(replay_iq)
    live_matches_cache = bool(
        np.allclose(live.candidate_scores, active_scores[:replay_count], atol=1e-4)
        and np.array_equal(live.blackboard.query_mask, active[:replay_count])
        and np.allclose(live.blackboard.open_opinion.risk,
                        c_active_known.risk[:replay_count], atol=1e-4))
    if not live_matches_cache:
        raise RuntimeError(
            "live one-turn path diverged from counterfactual cache: "
            f"score_max={np.abs(live.candidate_scores - active_scores[:replay_count]).max():.6g}, "
            f"query_mismatch={np.sum(live.blackboard.query_mask != active[:replay_count])}, "
            f"risk_max={np.abs(live.blackboard.open_opinion.risk - c_active_known.risk[:replay_count]).max():.6g}")
    report = {
        "stage": "stage10_stage5_mother_consultation_feasibility",
        "dataset": cfg["dataset"], "seed": cfg["seed"],
        "outer_fold": cfg["outer_fold"], "device": str(device),
        "support_classes": list(fold.known_classes),
        "proxy_unknown_classes": list(fold.proxy_unknown_classes),
        "formal_unknown_used": False,
        "formal_unknown_sample_count_not_accessed": protocol.formal_unknown_sample_count,
        "sample_counts": {"train": len(train), "calibration": len(calibration),
                          "test_known": len(test), "test_proxy_unknown": len(proxy)},
        "mother": {"checkpoint": str(checkpoint_path),
                   "sha256": _sha256(checkpoint_path),
                   "best_epoch": int(checkpoint["best_epoch"])},
        "provenance": {"config_path": cfg["config_path"],
                       "config_sha256": _sha256(cfg["config_path"]),
                       "mother_config_sha256": _sha256(cfg["mother_config"]),
                       "git_head": _git_head(),
                       "partition_seed": cfg["partition_seed"]},
        "view_usage_known": _view_usage(test),
        "view_usage_proxy_unknown": _view_usage(proxy),
        "identity_local_accuracy": float(np.mean(a_pred == test.labels)),
        "geometry_local_accuracy": float(np.mean(b_pred == test.labels)),
        "identity_only_correct": int(np.sum((a_pred == test.labels)
                                             & (b_pred != test.labels))),
        "geometry_only_correct": int(np.sum((b_pred == test.labels)
                                             & (a_pred != test.labels))),
        "pair_truth_coverage": float(np.mean(pair_includes_truth)),
        "responder_pair_accuracy_when_answerable": float(np.mean(
            verifier_choice[pair_includes_truth] == test.labels[pair_includes_truth]))
            if pair_includes_truth.any() else None,
        "calibration_counterfactual_gain": {
            "mean": float(gain.mean()), "positive_fraction": float(np.mean(gain > 0)),
            "query_policy_threshold": float(policy.threshold)},
        "conditions": conditions,
        "interventions": interventions,
        "open_set_proxy_evaluation": open_set,
        "live_runtime_replay": {
            "samples": replay_count,
            "matches_counterfactual_cache": live_matches_cache,
            "public_query_count": int(live.blackboard.query_mask.sum()),
        },
        "static_mean_fusion": _metrics(
            0.5 * (test.identity_logits + test.geometry_logits), test.labels,
            np.zeros(len(test), dtype=bool)),
        "proxy_unknown_query_fraction": float(np.mean(proxy_active_mask)),
        "query_policy_predicted_gain_mean": float(np.mean(expected_gain)),
        "limitations": [
            "Stage-5 frozen fold experts; only a new pair verifier and A-owned query policy are fitted",
            "B counterfactual outputs are cached for fair interventions; deployment must execute B only on queried rows",
            "C uses public scalar tails only; no reconstruction/consistency tools yet",
            "Unknown results are outer-fold proxy Unknown, never formal Unknown",
            "One seed and one outer fold; not a formal multi-seed result",
        ],
    }
    baseline = conditions["no_communication"]
    active_result = conditions["agent_initiated_query"]
    report["consultation_gate"] = {
        "passed": bool(
            count > 0
            and report["geometry_only_correct"] > 0
            and active_result["known_accuracy"] > baseline["known_accuracy"]
            and active_result["known_nll"] < baseline["known_nll"]
            and active_result["known_accuracy"] >
                conditions["confidence_central_query"]["known_accuracy"]
            and active_result["known_accuracy"] >
                conditions["random_query"]["known_accuracy"]
            and (not interventions or active_result["known_accuracy"] >
                 interventions["message_shuffle"]["known_accuracy"])),
        "required": ["positive unique B rescue", "active accuracy > no communication",
                     "active NLL < no communication", "active accuracy > random",
                     "active accuracy > confidence selector", "correct message > shuffle"],
    }
    report["feasibility_checks"] = {
        "correct_message_beats_shuffle_accuracy": bool(
            count and active_result["known_accuracy"] >
            interventions["message_shuffle"]["known_accuracy"]),
        "correct_message_beats_wrong_pair_accuracy": bool(
            count and active_result["known_accuracy"] >
            interventions["wrong_candidate_pair"]["known_accuracy"]),
        "open_examiner_beats_identity_msp_auroc": bool(
            open_set["examiner_active_query"]["auroc"] >
            open_set["identity_msp_no_communication"]["auroc"]),
        "active_open_h_beats_static_fusion": bool(
            open_set["examiner_active_query"]["h_score"] >
            open_set["examiner_static_mean_fusion"]["h_score"]),
        "active_open_auroc_beats_static_fusion": bool(
            open_set["examiner_active_query"]["auroc"] >
            open_set["examiner_static_mean_fusion"]["auroc"]),
        "live_runtime_matches_cache": live_matches_cache,
    }
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "consultation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_pair(oracle: dict, wisig: dict, output_dir: str | Path) -> dict:
    metadata = {"base_config", "config_path", "dataset", "name", "oracle_root",
                "wisig_pkl", "output_dir", "mother_config", "mother_checkpoint",
                "paired_config"}
    a = {key: value for key, value in oracle.items() if key not in metadata}
    b = {key: value for key, value in wisig.items() if key not in metadata}
    if a != b:
        raise ValueError("paired Stage-10 method fields must be identical")
    reports = {"oracle": run_dataset(oracle), "wisig": run_dataset(wisig)}
    summary = {
        "stage": "stage10_stage5_mother_consultation_feasibility",
        "method_config": a, "datasets": reports,
        "paired_gate_passed": all(
            report["consultation_gate"]["passed"] for report in reports.values()),
        "conclusion": "components_feasible_multiagent_necessity_unproven",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_consultation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
