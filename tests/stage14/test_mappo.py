import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage14.actions import AGENT_IDS, CAction, action_count  # noqa: E402
from maros_stage14.buffer import RolloutBuffer  # noqa: E402
from maros_stage14.env import EnvConfig, OSSEIMultiAgentEnv  # noqa: E402
from maros_stage14.evaluation import (calibrate_known_accuracy_threshold,
                                      metrics_at_reject_threshold)  # noqa: E402
from maros_stage14.mappo import MAPPOConfig, MAPPOTrainer  # noqa: E402
from maros_stage14.runner import (collect_episode, infer_dimensions,
                                  load_inference_actors,
                                  save_inference_actors)  # noqa: E402
from maros_stage14.synthetic import make_synthetic_snapshots  # noqa: E402


def test_three_actors_are_independent_and_critic_uses_state_only():
    sample = make_synthetic_snapshots(8)[0]
    dims, state_dim = infer_dimensions(sample)
    trainer = MAPPOTrainer(dims, state_dim)
    actors = [trainer.actors.actors[name] for name in AGENT_IDS]
    assert len({id(actor) for actor in actors}) == 3
    assert len({next(actor.parameters()).data_ptr() for actor in actors}) == 3
    critic_parameters = {parameter.data_ptr() for parameter in trainer.critic.parameters()}
    assert all(parameter.data_ptr() not in critic_parameters
               for actor in actors for parameter in actor.parameters())
    assert trainer.critic.state_dim == state_dim


def test_recurrent_mappo_update_is_finite_and_changes_parameters():
    torch.manual_seed(7)
    snapshots = make_synthetic_snapshots(16, seed=7)
    env_config = EnvConfig(max_steps=4)
    dims, state_dim = infer_dimensions(snapshots[0], env_config)
    trainer = MAPPOTrainer(dims, state_dim, MAPPOConfig(
        ppo_epochs=2, minibatch_episodes=4, actor_hidden_dim=24,
        critic_hidden_dim=32))
    buffer = RolloutBuffer()
    env = OSSEIMultiAgentEnv(env_config)
    for sample in snapshots[:8]:
        buffer.add(collect_episode(trainer, env, sample))
    before = {name: next(trainer.actors.actors[name].parameters()).detach().clone()
              for name in AGENT_IDS}
    statistics = trainer.update(buffer)
    assert all(np.isfinite(value) for value in statistics.values())
    assert all(not torch.equal(before[name],
                               next(trainer.actors.actors[name].parameters()))
               for name in AGENT_IDS)


def test_inference_state_dict_excludes_critic():
    sample = make_synthetic_snapshots(8)[0]
    dims, state_dim = infer_dimensions(sample)
    trainer = MAPPOTrainer(dims, state_dim)
    state = trainer.actor_state_dict()
    assert state and not any("critic" in key for key in state)


def test_actor_only_checkpoint_is_self_describing_and_loadable(tmp_path):
    sample = make_synthetic_snapshots(8)[0]
    config = EnvConfig(use_geometry_actor=False)
    dims, state_dim = infer_dimensions(sample, config)
    trainer = MAPPOTrainer(dims, state_dim)
    checkpoint = tmp_path / "actors.pt"
    save_inference_actors(trainer, checkpoint, {"fold": 0})
    actors, metadata = load_inference_actors(checkpoint)
    assert tuple(actors.agent_ids) == ("identity", "open_set")
    assert metadata == {"fold": 0}
    assert not actors.training


def test_two_agent_topology_removes_b_actor_but_keeps_geometry_as_c_tool():
    sample = make_synthetic_snapshots(8)[0]
    config = EnvConfig(use_geometry_actor=False, max_steps=4)
    env = OSSEIMultiAgentEnv(config)
    observations, _ = env.reset(sample)
    assert tuple(observations) == ("identity", "open_set")
    dims, state_dim = infer_dimensions(sample, config)
    trainer = MAPPOTrainer(dims, state_dim, MAPPOConfig(
        ppo_epochs=1, minibatch_episodes=2, actor_hidden_dim=16,
        critic_hidden_dim=24))
    assert tuple(trainer.agent_ids) == ("identity", "open_set")
    assert "geometry" not in trainer.actors.actors
    episode = collect_episode(trainer, env, sample)
    assert episode.steps and all("geometry" not in step.actions
                                 for step in episode.steps)


def test_known_only_threshold_calibration_hits_target_without_unknown_labels():
    def episode(label, hypothesis, reject_score):
        distribution = np.zeros(action_count("open_set"), dtype=np.float32)
        distribution[int(CAction.REJECT_UNKNOWN)] = reject_score
        return SimpleNamespace(
            label=label, final_hypothesis=hypothesis,
            steps=[SimpleNamespace(action_distributions={
                "open_set": distribution})])

    known_episodes = [
        episode(0, 0, 0.10), episode(1, 1, 0.20),
        episode(2, 2, 0.30), episode(3, 0, 0.05),
    ]
    episodes = known_episodes + [episode(-1, 0, 0.70), episode(-1, 1, 0.80)]
    calibration = calibrate_known_accuracy_threshold(known_episodes, 0.75)
    metrics = metrics_at_reject_threshold(episodes, calibration["threshold"])
    assert calibration["feasible"]
    assert calibration["calibration_known_accuracy"] == 0.75
    assert metrics["known_accuracy"] == 0.75
    assert metrics["unknown_recall"] == 1.0
    assert np.isfinite(metrics["auroc"])


def test_main_config_is_full_data_ac_training():
    import json

    config = json.loads((ROOT / "configs" / "experiments" /
                         "stage14_comm_mappo_oracle_ac.json")
                        .read_text(encoding="utf-8"))
    assert config["use_geometry_actor"] is False
    assert config["train_samples_per_class"] >= 100000
    assert config["test_samples_per_class"] >= 100000
    assert config["pug_source_samples_per_class"] == 512
    assert config["updates"] * config["episodes_per_update"] == 16384
