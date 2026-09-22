"""Rollout collection, training loop and checkpoint helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .buffer import RolloutBuffer, RolloutEpisode, RolloutStep
from .data import EpisodeSnapshot, TrainingEpisodeBank
from .env import EnvConfig, OSSEIMultiAgentEnv
from .mappo import MAPPOConfig, MAPPOTrainer
from .policies import HeterogeneousActors


def infer_dimensions(snapshot: EpisodeSnapshot, env_config: EnvConfig | None = None):
    env = OSSEIMultiAgentEnv(env_config)
    observations, _ = env.reset(snapshot)
    return ({agent: len(observations[agent]) for agent in env.agents},
            len(env.state()))


def collect_episode(trainer: MAPPOTrainer, env: OSSEIMultiAgentEnv,
                    snapshot: EpisodeSnapshot, *, deterministic: bool = False,
                    forced_actions: dict[str, int] | None = None):
    observations, _ = env.reset(snapshot)
    actor_states = trainer.initial_actor_states()
    critic_state = trainer.initial_critic_state()
    episode = RolloutEpisode(
        snapshot.sample_id, label=snapshot.label,
        source_kind=snapshot.source_kind)
    while True:
        masks = env.action_masks()
        state = env.state()
        actions, log_probs, probabilities, distributions, actor_states = trainer.select_actions(
            observations, masks, actor_states, deterministic)
        for agent, action in (forced_actions or {}).items():
            action = int(action)
            if not masks[agent][action]:
                raise ValueError(f"forced action is masked: {agent}:{action}")
            actions[agent] = action
            probabilities[agent] = float(distributions[agent][action])
            log_probs[agent] = float(np.log(max(probabilities[agent], 1e-12)))
        value, critic_state = trainer.value(state, critic_state)
        next_observations, reward, terminated, truncated, info = env.step(actions)
        info["trajectory"][-1]["action_probability"] = dict(probabilities)
        episode.append(RolloutStep(
            observations={key: value.copy() for key, value in observations.items()},
            action_masks={key: value.copy() for key, value in masks.items()},
            actions=dict(actions), old_log_probs=dict(log_probs),
            action_probabilities=dict(probabilities),
            action_distributions={key: value.copy()
                                  for key, value in distributions.items()},
            state=state.copy(),
            value=value, reward=reward, done=bool(terminated or truncated)))
        observations = next_observations
        if terminated or truncated:
            episode.prediction = info["prediction"]
            episode.final_hypothesis = info["decision_hypothesis"]
            episode.trajectory = info["trajectory"]
            return episode


def train_mappo(bank: TrainingEpisodeBank, *, updates: int,
                episodes_per_update: int, seed: int = 42,
                env_config: EnvConfig | None = None,
                mappo_config: MAPPOConfig | None = None,
                device: torch.device | str = "cpu"):
    if updates < 1 or episodes_per_update < 2:
        raise ValueError("training needs positive updates and at least two episodes")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dimensions, state_dim = infer_dimensions(bank[0], env_config)
    trainer = MAPPOTrainer(dimensions, state_dim, mappo_config, device)
    env = OSSEIMultiAgentEnv(env_config)
    rng = np.random.default_rng(seed)
    history = []
    for update in range(updates):
        buffer = RolloutBuffer()
        for index in bank.balanced_indices(rng, episodes_per_update):
            buffer.add(collect_episode(trainer, env, bank[int(index)]))
        stats = trainer.update(buffer)
        stats.update({
            "update": update,
            "mean_episode_reward": float(np.mean([
                episode.total_reward for episode in buffer.episodes])),
            "mean_episode_length": float(np.mean([
                len(episode.steps) for episode in buffer.episodes])),
        })
        history.append(stats)
    return trainer, history


def save_inference_actors(trainer: MAPPOTrainer, path: str | Path,
                          metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    actor_specs = {
        agent: {
            "observation_dim": actor.observation_dim,
            "action_dim": actor.action_dim,
            "hidden_dim": actor.hidden_dim,
        }
        for agent, actor in trainer.actors.actors.items()
    }
    torch.save({
        "format_version": 1,
        "agent_ids": list(trainer.agent_ids),
        "actor_specs": actor_specs,
        "actors": trainer.actor_state_dict(),
        "metadata": metadata or {},
        "critic_included": False,
    }, path)


def _infer_legacy_actor_specs(state_dict: dict[str, torch.Tensor]) -> dict:
    """Recover dimensions from pre-spec Stage-14 inference checkpoints."""
    specs = {}
    suffix = ".encoder.0.weight"
    for key, weight in state_dict.items():
        if key.startswith("actors.") and key.endswith(suffix):
            agent = key[len("actors."):-len(suffix)]
            action_weight = state_dict[f"actors.{agent}.action_head.weight"]
            specs[agent] = {
                "observation_dim": int(weight.shape[1]),
                "action_dim": int(action_weight.shape[0]),
                "hidden_dim": int(weight.shape[0]),
            }
    if not specs:
        raise ValueError("checkpoint does not contain recognizable Stage-14 actors")
    return specs


def load_inference_actors(path: str | Path,
                          device: torch.device | str = "cpu"):
    """Load the decentralized Actor-only inference artifact."""
    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("critic_included", False):
        raise ValueError("inference checkpoint must not include the critic")
    specs = payload.get("actor_specs") or _infer_legacy_actor_specs(payload["actors"])
    hidden_dims = {int(spec["hidden_dim"]) for spec in specs.values()}
    if len(hidden_dims) != 1:
        raise ValueError("all Stage-14 actors must use one hidden dimension")
    actors = HeterogeneousActors(
        {agent: int(spec["observation_dim"]) for agent, spec in specs.items()},
        hidden_dim=hidden_dims.pop()).to(device)
    for agent, spec in specs.items():
        if actors.actors[agent].action_dim != int(spec["action_dim"]):
            raise ValueError(f"action dimension mismatch for {agent}")
    actors.load_state_dict(payload["actors"])
    actors.eval()
    return actors, payload.get("metadata", {})
