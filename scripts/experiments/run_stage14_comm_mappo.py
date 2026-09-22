"""Train/evaluate Stage-14 Comm-MAPPO on one leakage-safe LCO fold."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

# Stage-5 enables deterministic PyTorch algorithms during coordinator fitting.
# CuBLAS requires this to be set before the first CUDA context is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage5.openmax import OpenMaxEVT  # noqa: E402
from maros_stage5.training import (extract_records, pseudo_to_records,
                                   refresh_empirical_prototypes)  # noqa: E402
from maros_stage7.splits import (ProvenanceSubset, assert_no_formal_unknown,
                                 build_nested_lco_protocol)  # noqa: E402
from maros_stage10.experiment import _load_dataset  # noqa: E402
from maros_stage10.mother import cap_dataset, load_mother  # noqa: E402
from maros_stage13.experiment import (_base_outputs, _base_system,
                                      load_selection_tail)  # noqa: E402
from maros_stage13.feature_pug import (CSupport, certify,
                                       generate_candidates)  # noqa: E402
from maros_stage14.data import (EvaluationEpisodeBank, TrainingEpisodeBank,
                                snapshots_from_records)  # noqa: E402
from maros_stage14.actions import BAction  # noqa: E402
from maros_stage14.env import EnvConfig  # noqa: E402
from maros_stage14.evaluation import (calibrate_known_accuracy_threshold,
                                      evaluate,
                                      metrics_at_reject_threshold)  # noqa: E402
from maros_stage14.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from maros_stage14.runner import (infer_dimensions, load_inference_actors,
                                  save_inference_actors,
                                  train_mappo)  # noqa: E402


def resolve_config(path: str | Path, stack=()) -> dict:
    path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if path in stack:
        raise ValueError("cyclic base_config inheritance")
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = resolve_config(raw["base_config"], (*stack, path)) \
        if "base_config" in raw else {}
    base.update(raw)
    base["config_path"] = str(path)
    return base


def capped(dataset, count: int, seed: int, *, proxy=False):
    return cap_dataset(dataset, int(count), int(seed), proxy_unknown=proxy)


def proxy_train_subset(splits, fold) -> ProvenanceSubset:
    labels = np.asarray(splits.train.y, dtype=np.int64)
    indices = np.flatnonzero(np.isin(labels, fold.proxy_unknown_classes))
    return ProvenanceSubset(
        splits.train, indices, source_split="train",
        purpose=f"stage14_outer{fold.index}_train_proxy_unknown",
        force_unknown=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/stage14_comm_mappo_oracle_ac.json")
    args = parser.parse_args()
    cfg = resolve_config(args.config)
    stage13 = resolve_config(cfg["stage13_config"])
    mother_cfg = json.loads(
        (ROOT / stage13["mother_config"]).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    np.random.seed(seed); torch.manual_seed(seed)
    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and
                          torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)

    splits = _load_dataset(stage13)
    protocol = build_nested_lco_protocol(
        splits, outer_folds=5, inner_folds=4,
        seed=int(stage13["partition_seed"]))
    fold = protocol.outer_folds[int(cfg["outer_fold"])]
    proxy_source = proxy_train_subset(splits, fold)
    datasets = {
        "train": capped(fold.train_known, cfg["train_samples_per_class"], seed+11),
        "train_proxy": capped(proxy_source, cfg["proxy_samples_per_class"],
                              seed+12, proxy=True),
        "cal": capped(fold.calibration_known,
                      cfg["calibration_samples_per_class"], seed+13),
        "test": capped(fold.test_known, cfg["test_samples_per_class"], seed+14),
        "test_proxy": capped(fold.test_proxy_unknown,
                             cfg["test_samples_per_class"], seed+15, proxy=True),
    }
    pug_source = capped(
        fold.train_known,
        cfg.get("pug_source_samples_per_class", cfg["train_samples_per_class"]),
        seed+16)
    assert_no_formal_unknown(*datasets.values())
    assert_no_formal_unknown(pug_source)
    checkpoint = ROOT / stage13["fold_checkpoint_pattern"].format(fold=fold.index)
    model, checkpoint_payload = load_mother(
        checkpoint, mother_cfg, fold.known_classes,
        fold.proxy_unknown_classes, device)
    refresh_empirical_prototypes(
        model, datasets["train"], device, int(stage13["batch_size"]))
    evt = OpenMaxEVT.from_state_dict(checkpoint_payload["openmax"])
    records = {name: extract_records(
        model, dataset, device, evt, batch_size=int(stage13["batch_size"]))
               for name, dataset in datasets.items()}
    pug_source_records = extract_records(
        model, pug_source, device, evt, batch_size=int(stage13["batch_size"]))
    support = CSupport().fit(records["cal"])

    # Preserve the complete frozen Stage-5 boundary implementation as C's
    # fourth tool. It is fitted only from Known/LCO/PUG training material.
    selection = load_selection_tail(ROOT / stage13["stage5_selection"])
    boundary_cfg = copy.deepcopy(mother_cfg)
    boundary_cfg["lco_coordinator_epochs"] = int(
        cfg.get("boundary_epochs", mother_cfg.get("lco_coordinator_epochs", 10)))
    coordinator, scaler, arbitrator, _, _, _ = _base_system(
        model, evt, records["train"], records["cal"], selection,
        boundary_cfg, device, seed + fold.index * 100)
    boundary = {name: _base_outputs(
        value, coordinator, scaler, arbitrator, device)["risk"]
                for name, value in records.items()}

    pseudo_batch, _ = generate_candidates(
        pug_source_records, str(cfg["pug_kind"]),
        float(cfg["pug_eta"]), seed+101)
    pseudo_records = pseudo_to_records(model, pseudo_batch, evt, device)
    certified, certification = certify(
        pug_source_records, pseudo_batch, pseudo_records, support)
    pseudo_boundary = _base_outputs(
        pseudo_records, coordinator, scaler, arbitrator, device)["risk"]

    training_rows = []
    training_rows.extend(snapshots_from_records(
        records["train"], source_kind="known", prefix="train-known",
        support=support, boundary_risk=boundary["train"]))
    training_rows.extend(snapshots_from_records(
        records["train_proxy"], source_kind="lco_proxy", prefix="train-lco",
        support=support, boundary_risk=boundary["train_proxy"]))
    training_rows.extend(snapshots_from_records(
        pseudo_records, source_kind="certified_pug", prefix="train-pug",
        support=support, boundary_risk=pseudo_boundary,
        certified_mask=certified))
    training_bank = TrainingEpisodeBank(training_rows)
    evaluation_rows = snapshots_from_records(
        records["test"], source_kind="known", prefix="test-known",
        support=support, boundary_risk=boundary["test"])
    evaluation_rows.extend(snapshots_from_records(
        records["test_proxy"], source_kind="lco_proxy", prefix="test-lco",
        support=support, boundary_risk=boundary["test_proxy"]))
    evaluation_bank = EvaluationEpisodeBank(evaluation_rows)
    calibration_bank = EvaluationEpisodeBank(snapshots_from_records(
        records["cal"], source_kind="known", prefix="cal-known",
        support=support, boundary_risk=boundary["cal"]))

    env_config = EnvConfig(
        max_steps=int(cfg["max_steps"]), message_budget=int(cfg["message_budget"]),
        tool_budget=int(cfg["tool_budget"]),
        step_cost=float(cfg.get("step_cost", 0.01)),
        message_cost=float(cfg.get("message_cost", 0.01)),
        tool_cost=float(cfg.get("tool_cost", 0.02)),
        min_tools_before_decision=int(cfg.get("min_tools_before_decision", 1)),
        use_geometry_actor=bool(cfg.get("use_geometry_actor", True)))
    actor_checkpoint = cfg.get("actors_checkpoint")
    loaded_actors = None
    actor_hidden_dim = int(cfg.get("actor_hidden_dim", 64))
    if actor_checkpoint:
        loaded_actors, _ = load_inference_actors(ROOT / actor_checkpoint, device)
        actor_hidden_dim = next(iter(loaded_actors.actors.values())).hidden_dim
    mappo_config = MAPPOConfig(
        ppo_epochs=int(cfg.get("ppo_epochs", 4)),
        minibatch_episodes=int(cfg.get("minibatch_episodes", 16)),
        actor_hidden_dim=actor_hidden_dim)
    if loaded_actors is None:
        trainer, history = train_mappo(
            training_bank, updates=int(cfg["updates"]),
            episodes_per_update=int(cfg["episodes_per_update"]), seed=seed,
            env_config=env_config, mappo_config=mappo_config, device=device)
    else:
        dimensions, state_dim = infer_dimensions(training_bank[0], env_config)
        trainer = MAPPOTrainer(dimensions, state_dim, mappo_config, device)
        if tuple(trainer.agent_ids) != tuple(loaded_actors.agent_ids):
            raise ValueError("checkpoint agents do not match configured topology")
        trainer.actors.load_state_dict(loaded_actors.state_dict())
        history = []
    metrics, episodes = evaluate(trainer, evaluation_bank, env_config)
    calibration_metrics, calibration_episodes = evaluate(
        trainer, calibration_bank, env_config)
    accuracy_targets = cfg.get(
        "known_accuracy_targets", [cfg.get("target_known_accuracy", 0.95)])
    operating_points = []
    for target in accuracy_targets:
        threshold_calibration = calibrate_known_accuracy_threshold(
            calibration_episodes, float(target))
        operating_points.append({
            "calibration": threshold_calibration,
            "test_metrics": metrics_at_reject_threshold(
                episodes, threshold_calibration["threshold"]),
        })
    matched_target = float(cfg.get("diagnostic_test_known_accuracy", 0.95))
    matched_test_threshold = calibrate_known_accuracy_threshold(
        episodes, matched_target)
    matched_test_point = {
        "warning": "diagnostic curve point only; never use this test-selected threshold for deployment",
        "selection_data": "test_known_labels_only",
        "calibration": matched_test_threshold,
        "test_metrics": metrics_at_reject_threshold(
            episodes, matched_test_threshold["threshold"]),
    }
    b_removed = None
    if env_config.use_geometry_actor:
        b_removed, _ = evaluate(
            trainer, evaluation_bank, env_config,
            forced_actions={"geometry": int(BAction.WAIT)})
    permutation = np.random.default_rng(seed + 909).permutation(len(evaluation_bank))
    tool_overrides = {}
    for index, snapshot in enumerate(evaluation_bank):
        donor = evaluation_bank[int(permutation[index])]
        tool_overrides[snapshot.sample_id] = {
            "id_prototype": donor.identity_prototype_risk,
            "geo_prototype": donor.geometry_prototype_risk,
            "openmax": donor.openmax_risk,
            "boundary": donor.boundary_risk,
        }
    tool_shuffled, _ = evaluate(
        trainer, evaluation_bank, env_config, tool_overrides=tool_overrides)
    causal = {
        "b_agent_removal": b_removed,
        "all_c_tools_shuffle": tool_shuffled,
        "delta_h_remove_b": (metrics["h_score"] - b_removed["h_score"]
                             if b_removed is not None else None),
        "delta_h_shuffle_c_tools": metrics["h_score"] - tool_shuffled["h_score"],
    }

    output = ROOT / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "stage14_heterogeneous_comm_mappo",
        "dataset": cfg["dataset"], "outer_fold": fold.index,
        "formal_unknown_used": False,
        "formal_unknown_count_not_accessed": protocol.formal_unknown_sample_count,
        "actors": list(trainer.agent_ids),
        "actor_parameter_sharing": False,
        "critic": "single_centralized_V_state_only",
        "perception_checkpoint": str(checkpoint),
        "training_episode_count": len(training_bank),
        "dataset_sizes": {name: len(dataset)
                          for name, dataset in datasets.items()},
        "pug_source_count": len(pug_source),
        "loaded_actor_checkpoint": str(ROOT / actor_checkpoint)
        if actor_checkpoint else None,
        "certification": certification,
        "history": history, "metrics": metrics, "causal_interventions": causal,
        "known_accuracy_operating_point": {
            "selection_data": "calibration_known_only",
            "calibration_default_metrics": calibration_metrics,
            "operating_points": operating_points,
            "matched_test_point_diagnostic_only": matched_test_point,
        },
    }
    (output / "stage14_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps({
                "sample_id": episode.sample_id, "label": episode.label,
                "prediction": episode.prediction, "reward": episode.total_reward,
                "trajectory": episode.trajectory}, ensure_ascii=False) + "\n")
    save_inference_actors(trainer, output / "actors_inference.pt", {
        "dataset": cfg["dataset"], "fold": fold.index,
        "observation_protocol": "stage14-v1", "formal_unknown_used": False})
    print(json.dumps({"output": str(output), "metrics": metrics,
                      "training_episodes": len(training_bank),
                      "certified_pug": int(certified.sum())},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
