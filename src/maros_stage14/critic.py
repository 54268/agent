"""Training-only centralized recurrent value function."""
from __future__ import annotations

import torch
from torch import nn


class CentralizedCritic(nn.Module):
    """Estimates V(s_t); the current sampled joint action is never an input."""

    def __init__(self, state_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, state: torch.Tensor, hidden: torch.Tensor):
        if state.ndim == 1:
            state = state[None]
        next_hidden = self.memory(self.encoder(state), hidden)
        return self.value_head(next_hidden).squeeze(-1), next_hidden

    def evaluate_sequence(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_state(1, states.device)
        values = []
        for step in range(len(states)):
            value, hidden = self(states[step:step+1], hidden)
            values.append(value)
        return torch.cat(values)
