"""Leakage-safe, same-method paired WiSig/ORACLE G0 experiment."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from maros_staged.datasets import load_oracle_npz, load_wisig_subset

from .audit import audit_g0
from .gates import evaluate_g0
from .policies.tool_policy import counterfactual_utility
from .splits import (assert_no_formal_unknown, build_nested_lco_protocol,
                     protocol_manifest)
from .system import Stage9G0System
from .training import cap_per_class, train_experts, train_tool_policies


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load(cfg: Mapping):
    # Dataset branching belongs only in the data loader, never in a policy.
    if cfg["dataset"] == "wisig":
        return load_wisig_subset(cfg["wisig_pkl"], augment_train=False)
    if cfg["dataset"] == "oracle":
        return load_oracle_npz(cfg["oracle_root"], augment_train=False)
    raise ValueError("dataset must be wisig or oracle")


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


@torch.no_grad()
def _evaluate_agent(agent, dataset: Dataset, cfg: Mapping,
                    device: torch.device, fixed_action: int,
                    *, known: bool) -> dict:
    assert_no_formal_unknown(dataset)
    if known:
        dataset = cap_per_class(dataset, int(cfg["eval_samples_per_class"]),
                                int(cfg["seed"]) + 303)
    else:
        # Proxy Unknown labels are -1. Sampling is based only on their retained
        # original class identities and never on formal Unknown examples.
        from torch.utils.data import Subset
        labels = np.asarray(dataset.original_labels, dtype=np.int64)
        rng = np.random.default_rng(int(cfg["seed"]) + 404)
        indices = []
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            rng.shuffle(candidates)
            indices.extend(candidates[:int(cfg["eval_samples_per_class"])].tolist())
        dataset = Subset(dataset, sorted(indices))
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False)
    agent.eval()
    action_rows, entropy_rows = [], []
    correct = accepted_correct = total = abstentions = 0
    tool_correct = {name: 0 for name in agent.tool_names}
    tool_nll = {name: 0.0 for name in agent.tool_names}
    selected_utility, fixed_utility, random_utility, oracle_utility = [], [], [], []
    per_tool_gain: dict[str, list[float]] = {name: [] for name in agent.tool_names}
    for iq, labels in loader:
        iq, labels = iq.to(device), labels.long().to(device)
        packet, private = agent.decide(iq)
        actions = private.action_indices.detach().cpu().numpy()
        action_rows.extend(int(value) for value in actions)
        entropy_rows.extend(private.policy_entropy.detach().cpu().tolist())
        abstentions += int(packet.abstain.sum())
        total += len(labels)
        if known:
            correct += int(packet.candidate_a.eq(labels).sum())
            accepted_correct += int((packet.candidate_a.eq(labels)
                                     & ~packet.abstain).sum())
            all_logits = agent.all_tool_logits(iq)
            for index, name in enumerate(agent.tool_names):
                tool_correct[name] += int(all_logits[:, index].argmax(1).eq(labels).sum())
                tool_nll[name] += float(torch.nn.functional.cross_entropy(
                    all_logits[:, index], labels, reduction="sum"))
            utility = counterfactual_utility(
                all_logits, labels, agent.tool_policy.actions,
                float(cfg["pair_cost"]), float(cfg["stop_cost"]))
            rows = torch.arange(len(labels), device=device)
            selected = utility[rows, private.action_indices]
            fixed = utility[:, fixed_action]
            selected_utility.extend(selected.cpu().tolist())
            fixed_utility.extend(fixed.cpu().tolist())
            random_utility.extend(utility.mean(1).cpu().tolist())
            oracle_utility.extend(utility.max(1).values.cpu().tolist())
            for row, action in enumerate(actions):
                for index in agent.tool_policy.actions[int(action)]:
                    per_tool_gain[agent.tool_names[index]].append(
                        float((selected[row] - fixed[row]).cpu()))
    action_count = np.bincount(action_rows, minlength=len(agent.tool_policy.actions))
    usage = {name: float(sum(count for action, count in
                             zip(agent.tool_policy.actions, action_count)
                             if index in action) / max(total, 1))
             for index, name in enumerate(agent.tool_names)}
    result = {
        "samples": total, "tool_usage": usage,
        "action_counts": {
            "+".join(agent.tool_names[index] for index in action) or "STOP": int(count)
            for action, count in zip(agent.tool_policy.actions, action_count)},
        "distinct_actions": int(np.count_nonzero(action_count)),
        "majority_action_fraction": float(action_count.max() / max(total, 1)),
        "mean_policy_entropy": float(np.mean(entropy_rows)),
        "abstain_rate": abstentions / max(total, 1),
    }
    if known:
        selected_mean = float(np.mean(selected_utility))
        result.update({
            "proposal_top1_accuracy": correct / max(total, 1),
            "accepted_correct_rate": accepted_correct / max(total, 1),
            "coverage": 1.0 - abstentions / max(total, 1),
            "selected_utility": selected_mean,
            "fixed_utility": float(np.mean(fixed_utility)),
            "random_utility": float(np.mean(random_utility)),
            "oracle_utility": float(np.mean(oracle_utility)),
            "gain_vs_fixed": selected_mean - float(np.mean(fixed_utility)),
            "gain_vs_random": selected_mean - float(np.mean(random_utility)),
            "per_tool_conditional_gain_vs_fixed": {
                name: (float(np.mean(values)) if values else None)
                for name, values in per_tool_gain.items()},
            "per_tool_accuracy": {
                name: tool_correct[name] / max(total, 1)
                for name in agent.tool_names},
            "per_tool_nll": {
                name: tool_nll[name] / max(total, 1)
                for name in agent.tool_names},
        })
    return result


@torch.no_grad()
def _preliminary_rescue(system, test_known: Dataset, cfg: Mapping,
                        device: torch.device) -> dict:
    """One-fold A/B failure overlap, not a G1 complementarity claim."""
    known = cap_per_class(test_known, int(cfg["eval_samples_per_class"]),
                          int(cfg["seed"]) + 303)
    loader = DataLoader(known, batch_size=int(cfg["batch_size"]), shuffle=False)
    counts = {"identity_only_correct": 0, "impairment_only_correct": 0,
              "both_correct": 0, "both_wrong": 0, "samples": 0}
    system.eval()
    for iq, labels in loader:
        iq, labels = iq.to(device), labels.long().to(device)
        identity, _ = system.identity.decide(iq)
        impairment, _ = system.impairment.decide(iq)
        a = identity.candidate_a.eq(labels) & ~identity.abstain
        b = impairment.candidate_a.eq(labels) & ~impairment.abstain
        counts["identity_only_correct"] += int((a & ~b).sum())
        counts["impairment_only_correct"] += int((b & ~a).sum())
        counts["both_correct"] += int((a & b).sum())
        counts["both_wrong"] += int((~a & ~b).sum())
        counts["samples"] += len(labels)
    return counts


def run_dataset(cfg: Mapping) -> dict:
    seed_all(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto"
                          and torch.cuda.is_available() else
                          cfg["device"] if cfg["device"] != "auto" else "cpu")
    splits = _load(cfg)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=int(cfg["outer_folds"]),
        inner_folds=int(cfg["inner_folds"]), seed=int(cfg["partition_seed"]))
    fold = protocol.outer_folds[int(cfg["outer_fold"])]
    assert_no_formal_unknown(fold.train_known, fold.calibration_known,
                             fold.test_known, fold.test_proxy_unknown)
    system = Stage9G0System(
        len(fold.known_classes), int(cfg["state_dim"]), int(cfg["width"])).to(device)
    expert_history = train_experts(system, fold.train_known, cfg, device)
    policy_history, fixed_actions = train_tool_policies(
        system, fold.calibration_known, cfg, device)
    agents = {}
    for agent in system.agents():
        fixed = fixed_actions[agent.role]
        agents[agent.role] = {
            "tool_names": list(agent.tool_names),
            "fixed_action_calibration": (
                "+".join(agent.tool_names[index] for index in
                         agent.tool_policy.actions[fixed]) or "STOP"),
            "known": _evaluate_agent(agent, fold.test_known, cfg, device,
                                     fixed, known=True),
            "proxy_unknown": _evaluate_agent(
                agent, fold.test_proxy_unknown, cfg, device, fixed, known=False),
        }
    report = {
        "stage": "stage9_g0", "dataset": cfg["dataset"], "seed": cfg["seed"],
        "device": str(device), "method_profile": cfg["method_profile"],
        "support_classes": list(fold.known_classes),
        "proxy_unknown_classes": list(fold.proxy_unknown_classes),
        "formal_unknown_used": False,
        "capability_audit": audit_g0().as_dict(),
        "condition_breakdown": "not_available_in_current_dataset_contract",
        "protocol": protocol_manifest(protocol),
        "agents": agents, "expert_history": expert_history,
        "policy_history": policy_history,
        "preliminary_rescue": _preliminary_rescue(
            system, fold.test_known, cfg, device),
        "provenance": {
            "git_head": _git_head(), "config_path": cfg["config_path"],
            "config_sha256": _sha256(cfg["config_path"]),
            "seed": cfg["seed"], "partition_seed": cfg["partition_seed"],
        },
    }
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "g0_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": system.state_dict(), "config": dict(cfg),
                "formal_unknown_used": False}, output / "g0_model.pt")
    return report


def run_pair(oracle: Mapping, wisig: Mapping, pair_output: str | Path) -> dict:
    if oracle["dataset"] != "oracle" or wisig["dataset"] != "wisig":
        raise ValueError("paired G0 requires ORACLE and WiSig")
    metadata = {"base_config", "config_path", "dataset", "name", "oracle_root",
                "wisig_pkl", "output_dir", "paired_config"}
    method_oracle = {key: value for key, value in oracle.items() if key not in metadata}
    method_wisig = {key: value for key, value in wisig.items() if key not in metadata}
    if method_oracle != method_wisig:
        different = sorted(key for key in set(method_oracle) | set(method_wisig)
                           if method_oracle.get(key) != method_wisig.get(key))
        raise ValueError("paired method config differs: " + ", ".join(different))
    reports = {"oracle": run_dataset(oracle), "wisig": run_dataset(wisig)}
    summary = {"stage": "stage9_g0", "datasets": reports,
               "method_config": method_oracle}
    summary["g0_gate"] = evaluate_g0(summary, dict(oracle))
    output = Path(pair_output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "g0_pair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
