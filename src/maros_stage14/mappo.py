"""Compact recurrent MAPPO implementation for configurable heterogeneous actors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBuffer
from .critic import CentralizedCritic
from .policies import HeterogeneousActors


@dataclass(frozen=True)
class MAPPOConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_episodes: int = 16
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 96


class MAPPOTrainer:
    def __init__(self, observation_dims: Mapping[str, int], state_dim: int,
                 config: MAPPOConfig | None = None,
                 device: torch.device | str = "cpu"):
        self.config = config or MAPPOConfig()
        self.device = torch.device(device)
        self.agent_ids = tuple(observation_dims)
        self.actors = HeterogeneousActors(
            observation_dims, self.config.actor_hidden_dim).to(self.device)
        self.critic = CentralizedCritic(
            state_dim, self.config.critic_hidden_dim).to(self.device)
        self.actor_optimizers = {
            agent: torch.optim.Adam(self.actors.actors[agent].parameters(),
                                    lr=self.config.actor_lr)
            for agent in self.agent_ids}
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr)

    def initial_actor_states(self):
        return self.actors.initial_states(self.device)

    def initial_critic_state(self):
        return self.critic.initial_state(1, self.device)

    @torch.no_grad()
    def select_actions(self, observations, action_masks, hidden_states,
                       deterministic: bool = False):
        actions, log_probs, probabilities, distributions, next_states = {}, {}, {}, {}, {}
        for agent in self.agent_ids:
            observation = torch.as_tensor(
                observations[agent], dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(
                action_masks[agent], dtype=torch.bool, device=self.device)
            action, log_prob, _, probs, hidden = self.actors.actors[agent].act(
                observation, hidden_states[agent], mask, deterministic)
            selected = int(action.item())
            actions[agent] = selected
            log_probs[agent] = float(log_prob.item())
            probabilities[agent] = float(probs[0, selected].item())
            distributions[agent] = probs[0].detach().cpu().numpy().copy()
            next_states[agent] = hidden
        return actions, log_probs, probabilities, distributions, next_states

    @torch.no_grad()
    def value(self, state, hidden_state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        value, next_hidden = self.critic(tensor, hidden_state)
        return float(value.item()), next_hidden

    def _actor_batch_loss(self, agent, episodes, episode_indices,
                          advantages):
        actor = self.actors.actors[agent]
        new_logs, old_logs, adv_rows, entropy_rows = [], [], [], []
        for index in episode_indices:
            episode = episodes[index]
            observations = torch.as_tensor(np.stack([
                step.observations[agent] for step in episode.steps]),
                dtype=torch.float32, device=self.device)
            masks = torch.as_tensor(np.stack([
                step.action_masks[agent] for step in episode.steps]),
                dtype=torch.bool, device=self.device)
            actions = torch.as_tensor([
                step.actions[agent] for step in episode.steps],
                dtype=torch.long, device=self.device)
            log_prob, entropy = actor.evaluate_sequence(observations, masks, actions)
            new_logs.append(log_prob)
            entropy_rows.append(entropy)
            old_logs.append(torch.as_tensor([
                step.old_log_probs[agent] for step in episode.steps],
                dtype=torch.float32, device=self.device))
            adv_rows.append(torch.as_tensor(
                advantages[index], dtype=torch.float32, device=self.device))
        new_log = torch.cat(new_logs)
        old_log = torch.cat(old_logs)
        advantage = torch.cat(adv_rows)
        entropy = torch.cat(entropy_rows)
        ratio = torch.exp(new_log - old_log)
        unclipped = ratio * advantage
        clipped = torch.clamp(
            ratio, 1.0-self.config.clip_ratio,
            1.0+self.config.clip_ratio) * advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        return policy_loss - self.config.entropy_coef * entropy.mean(), \
            policy_loss.detach(), entropy.mean().detach(), ratio.detach()

    def _critic_batch_loss(self, episodes, episode_indices, returns):
        new_values, old_values, return_rows = [], [], []
        for index in episode_indices:
            episode = episodes[index]
            states = torch.as_tensor(np.stack([step.state for step in episode.steps]),
                                     dtype=torch.float32, device=self.device)
            new_values.append(self.critic.evaluate_sequence(states))
            old_values.append(torch.as_tensor(
                [step.value for step in episode.steps], dtype=torch.float32,
                device=self.device))
            return_rows.append(torch.as_tensor(
                returns[index], dtype=torch.float32, device=self.device))
        value = torch.cat(new_values)
        old = torch.cat(old_values)
        target = torch.cat(return_rows)
        clipped = old + (value-old).clamp(
            -self.config.value_clip, self.config.value_clip)
        loss = torch.maximum((value-target).square(),
                             (clipped-target).square()).mean()
        return self.config.value_coef * loss, loss.detach()

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if not buffer.episodes:
            raise ValueError("cannot update MAPPO from an empty buffer")
        buffer.validate()
        advantages, returns = buffer.gae(
            self.config.gamma, self.config.gae_lambda)
        flat = np.concatenate(advantages)
        mean, std = float(flat.mean()), float(flat.std() + 1e-8)
        advantages = [(value-mean)/std for value in advantages]
        rng = np.random.default_rng()
        totals = {f"{agent}_policy_loss": [] for agent in self.agent_ids}
        totals.update({f"{agent}_entropy": [] for agent in self.agent_ids})
        totals.update({"critic_loss": [], "ratio_mean": []})
        episode_count = len(buffer.episodes)
        batch_size = max(1, min(self.config.minibatch_episodes, episode_count))
        for _ in range(self.config.ppo_epochs):
            order = rng.permutation(episode_count)
            for start in range(0, episode_count, batch_size):
                indices = order[start:start+batch_size].tolist()
                for agent in self.agent_ids:
                    loss, raw, entropy, ratio = self._actor_batch_loss(
                        agent, buffer.episodes, indices, advantages)
                    optimizer = self.actor_optimizers[agent]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors.actors[agent].parameters(),
                        self.config.max_grad_norm)
                    optimizer.step()
                    totals[f"{agent}_policy_loss"].append(float(raw))
                    totals[f"{agent}_entropy"].append(float(entropy))
                    totals["ratio_mean"].append(float(ratio.mean()))
                value_loss, raw_value = self._critic_batch_loss(
                    buffer.episodes, indices, returns)
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                totals["critic_loss"].append(float(raw_value))
        return {name: float(np.mean(values)) for name, values in totals.items()}

    def actor_state_dict(self):
        """Inference checkpoint deliberately excludes the centralized critic."""
        return self.actors.state_dict()
