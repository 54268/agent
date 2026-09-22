"""Episode-preserving recurrent rollout storage and GAE."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import AGENT_IDS


@dataclass
class RolloutStep:
    observations: dict[str, np.ndarray]
    action_masks: dict[str, np.ndarray]
    actions: dict[str, int]
    old_log_probs: dict[str, float]
    action_probabilities: dict[str, float]
    action_distributions: dict[str, np.ndarray]
    state: np.ndarray
    value: float
    reward: float
    done: bool


@dataclass
class RolloutEpisode:
    sample_id: str
    steps: list[RolloutStep] = field(default_factory=list)
    prediction: int | None = None
    label: int | None = None
    source_kind: str | None = None
    final_hypothesis: int | None = None
    trajectory: list[dict] = field(default_factory=list)

    def append(self, step: RolloutStep) -> None:
        if self.steps and self.steps[-1].done:
            raise RuntimeError("cannot append after terminal transition")
        self.steps.append(step)

    @property
    def total_reward(self) -> float:
        return float(sum(step.reward for step in self.steps))


class RolloutBuffer:
    def __init__(self):
        self.episodes: list[RolloutEpisode] = []

    def add(self, episode: RolloutEpisode) -> None:
        if not episode.steps or not episode.steps[-1].done:
            raise ValueError("buffer accepts only complete episodes")
        self.episodes.append(episode)

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return sum(len(episode.steps) for episode in self.episodes)

    def gae(self, gamma: float, gae_lambda: float):
        advantages, returns = [], []
        for episode in self.episodes:
            reward = np.asarray([step.reward for step in episode.steps], dtype=np.float32)
            value = np.asarray([step.value for step in episode.steps], dtype=np.float32)
            done = np.asarray([step.done for step in episode.steps], dtype=np.float32)
            episode_advantage = np.zeros_like(reward)
            running = 0.0
            for step in reversed(range(len(reward))):
                next_value = 0.0 if step == len(reward)-1 else value[step+1]
                continuation = 1.0 - done[step]
                delta = reward[step] + gamma * next_value * continuation - value[step]
                running = delta + gamma * gae_lambda * continuation * running
                episode_advantage[step] = running
            advantages.append(episode_advantage)
            returns.append(episode_advantage + value)
        return advantages, returns

    def validate(self) -> None:
        for episode in self.episodes:
            for step in episode.steps:
                for agent in step.actions:
                    action = int(step.actions[agent])
                    if not bool(step.action_masks[agent][action]):
                        raise ValueError("rollout contains an action invalid under its saved mask")
