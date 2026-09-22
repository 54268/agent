"""Independent recurrent actors for heterogeneous decentralized execution."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from .actions import action_count


class RecurrentActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def distribution(self, observation: torch.Tensor, hidden: torch.Tensor,
                     action_mask: torch.Tensor):
        if observation.ndim == 1:
            observation = observation[None]
        if action_mask.ndim == 1:
            action_mask = action_mask[None]
        encoded = self.encoder(observation)
        next_hidden = self.memory(encoded, hidden)
        logits = self.action_head(next_hidden)
        if not bool(action_mask.any(dim=-1).all()):
            raise ValueError("every actor row needs at least one valid action")
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return Categorical(logits=logits), next_hidden

    def act(self, observation: torch.Tensor, hidden: torch.Tensor,
            action_mask: torch.Tensor, deterministic: bool = False):
        distribution, next_hidden = self.distribution(
            observation, hidden, action_mask)
        action = (distribution.probs.argmax(dim=-1) if deterministic
                  else distribution.sample())
        return (action, distribution.log_prob(action), distribution.entropy(),
                distribution.probs, next_hidden)

    def evaluate_sequence(self, observations: torch.Tensor,
                          action_masks: torch.Tensor,
                          actions: torch.Tensor):
        hidden = self.initial_state(1, observations.device)
        log_probs, entropies = [], []
        for step in range(len(observations)):
            distribution, hidden = self.distribution(
                observations[step:step+1], hidden,
                action_masks[step:step+1])
            action = actions[step:step+1]
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
        return torch.cat(log_probs), torch.cat(entropies)


class IdentityPolicy(RecurrentActor):
    pass


class GeometryPolicy(RecurrentActor):
    pass


class OpenSetPolicy(RecurrentActor):
    pass


class HeterogeneousActors(nn.Module):
    def __init__(self, observation_dims: Mapping[str, int], hidden_dim: int = 64):
        super().__init__()
        classes = {"identity": IdentityPolicy, "geometry": GeometryPolicy,
                   "open_set": OpenSetPolicy}
        self.agent_ids = tuple(observation_dims)
        self.actors = nn.ModuleDict({
            agent: classes[agent](observation_dims[agent], action_count(agent), hidden_dim)
            for agent in self.agent_ids})
        if len({id(self.actors[name]) for name in self.agent_ids}) != len(self.agent_ids):
            raise RuntimeError("Stage-14 actors must not share module instances")

    def initial_states(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {agent: self.actors[agent].initial_state(1, device)
                for agent in self.agent_ids}
