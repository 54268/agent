"""Explicit directed communication between the three Stage-1 expert Agents."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .pseudo_unknown import AGENTS


class MultiAgentCommunicationModel(nn.Module):
    """Private adapters -> directed gated messages -> contextualized states.

    This is intentionally more than concatenation: each receiver has a query,
    each sender emits a message, every directed edge has a sample-dependent
    gate, and the receiver state is updated by its incoming aggregate.
    """

    def __init__(self, input_dims: Dict[str, int], num_classes: int,
                 evidence_dim: int = 4, message_dim: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.10,
                 update_mode: str = "gru"):
        super().__init__()
        self.message_dim = message_dim
        self.num_classes = num_classes
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dims[name] + evidence_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, message_dim),
            ) for name in AGENTS
        })
        self.type_embedding = nn.Parameter(torch.randn(len(AGENTS), message_dim) * 0.02)
        self.query = nn.Linear(message_dim, message_dim, bias=False)
        self.key = nn.Linear(message_dim, message_dim, bias=False)
        self.value = nn.Linear(message_dim, message_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(2 * message_dim, message_dim), nn.GELU(),
            nn.Linear(message_dim, 1),
        )
        if update_mode not in {"gru", "residual"}:
            raise ValueError(f"unknown communication update_mode={update_mode}")
        self.update_mode = update_mode
        if update_mode == "gru":
            # Formal Stage-3 variant: direct receiver update.
            self.token_norm = nn.ModuleList([nn.Identity() for _ in AGENTS])
            self.update = nn.ModuleList([
                nn.GRUCell(message_dim, message_dim) for _ in AGENTS
            ])
            self.register_parameter("communication_scale", None)
        else:
            # Warm-up ablation: preserve a trained no-message state and add a
            # small residual message correction.
            self.token_norm = nn.ModuleList([nn.LayerNorm(message_dim) for _ in AGENTS])
            self.update = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * message_dim, message_dim), nn.GELU(),
                    nn.Linear(message_dim, message_dim),
                ) for _ in AGENTS
            ])
            self.communication_scale = nn.Parameter(torch.full((len(AGENTS),), -2.0))
        self.norm = nn.ModuleList([nn.LayerNorm(message_dim) for _ in AGENTS])
        self.class_heads = nn.ModuleList([
            nn.Linear(message_dim, num_classes) for _ in AGENTS
        ])
        boundary_in = len(AGENTS) * message_dim
        self.boundary_head = nn.Sequential(
            nn.Linear(boundary_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _tokens(self, embeddings: Dict[str, torch.Tensor],
                evidence: torch.Tensor) -> torch.Tensor:
        tokens = []
        for index, name in enumerate(AGENTS):
            local = torch.cat([embeddings[name], evidence[:, index]], dim=-1)
            token = self.adapters[name](local) + self.type_embedding[index]
            tokens.append(self.token_norm[index](token))
        return torch.stack(tokens, dim=1)

    def forward(self, embeddings: Dict[str, torch.Tensor], evidence: torch.Tensor,
                communicate: bool = True, shuffle_messages: bool = False,
                shuffle_index: Optional[torch.Tensor] = None,
                drop_sender: Optional[int] = None) -> Dict[str, torch.Tensor]:
        tokens = self._tokens(embeddings, evidence)
        batch, n_agents, dim = tokens.shape
        adjacency = tokens.new_zeros(batch, n_agents, n_agents)
        updated = tokens

        if communicate:
            q, k, messages = self.query(tokens), self.key(tokens), self.value(tokens)
            if shuffle_messages:
                if shuffle_index is None:
                    shuffle_index = torch.arange(batch - 1, -1, -1, device=tokens.device)
                messages = messages[shuffle_index]
            logits = torch.einsum("bid,bjd->bij", q, k) / (dim ** 0.5)
            diagonal = torch.eye(n_agents, dtype=torch.bool, device=tokens.device)[None]
            logits = logits.masked_fill(diagonal, float("-inf"))
            attention = F.softmax(logits, dim=-1)
            receiver = tokens[:, :, None, :].expand(-1, -1, n_agents, -1)
            sender = tokens[:, None, :, :].expand(-1, n_agents, -1, -1)
            gates = torch.sigmoid(self.edge_gate(torch.cat([receiver, sender], dim=-1)).squeeze(-1))
            gates = gates.masked_fill(diagonal, 0.0)
            adjacency = attention * gates
            if drop_sender is not None:
                adjacency = adjacency.clone()
                adjacency[:, :, int(drop_sender)] = 0.0
            incoming = torch.einsum("bij,bjd->bid", adjacency, messages)
            states = []
            for receiver_index in range(n_agents):
                if self.update_mode == "gru":
                    state = self.update[receiver_index](incoming[:, receiver_index],
                                                        tokens[:, receiver_index])
                else:
                    delta = self.update[receiver_index](torch.cat([
                        tokens[:, receiver_index], incoming[:, receiver_index]
                    ], dim=-1))
                    scale = torch.sigmoid(self.communication_scale[receiver_index])
                    state = tokens[:, receiver_index] + scale * delta
                states.append(self.norm[receiver_index](state))
            updated = torch.stack(states, dim=1)

        agent_logits = torch.stack(
            [self.class_heads[i](updated[:, i]) for i in range(n_agents)], dim=1
        )
        fused_logits = agent_logits.mean(dim=1)
        unknown_logit = self.boundary_head(updated.flatten(1)).squeeze(-1)
        return {
            "tokens_pre": tokens,
            "tokens_post": updated,
            "adjacency": adjacency,
            "agent_logits": agent_logits,
            "fused_logits": fused_logits,
            "unknown_logit": unknown_logit,
        }
