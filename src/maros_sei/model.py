from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(x, scale)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        channels = [in_channels, 32, 64, 96]
        blocks = []
        for index in range(3):
            blocks.extend(
                [
                    nn.Conv1d(channels[index], channels[index + 1], 7 if index == 0 else 5, stride=2, padding=3 if index == 0 else 2, bias=False),
                    nn.GroupNorm(8, channels[index + 1]),
                    nn.SiLU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(192, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return F.normalize(self.projection(pooled), dim=-1)


class StatsEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, embedding_dim),
        )

    @staticmethod
    def statistics(x: torch.Tensor) -> torch.Tensor:
        i, q = x[:, 0], x[:, 1]
        amp = torch.sqrt(i.square() + q.square() + 1e-8)
        phase_delta = torch.atan2(
            q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1],
            i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1] + 1e-8,
        )
        lag_real = (i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]).mean(dim=-1)
        lag_imag = (q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]).mean(dim=-1)
        return torch.stack(
            [
                i.mean(-1),
                q.mean(-1),
                i.std(-1),
                q.std(-1),
                amp.mean(-1),
                amp.std(-1),
                amp.amax(-1),
                phase_delta.mean(-1),
                phase_delta.std(-1),
                lag_real,
                lag_imag,
                (amp[:, 1:] - amp[:, :-1]).abs().mean(-1),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.statistics(x)), dim=-1)


class CosineHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 12.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.scale = float(scale)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(h, dim=-1), F.normalize(self.weight, dim=-1))

    def target_distance(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.weight, dim=-1)[target]
        return 1.0 - (F.normalize(h, dim=-1) * prototypes).sum(dim=-1)


@dataclass
class ModelConfig:
    num_classes: int
    num_receivers: int
    num_dates: int
    embedding_dim: int = 64
    top_k: int = 2


class MAROSSEI(nn.Module):
    agent_names = ("time", "frequency_phase", "domain", "open_set")

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.time_encoder = ConvEncoder(2, dim)
        self.frequency_encoder = ConvEncoder(2, dim)
        self.domain_encoder = StatsEncoder(dim)
        self.open_encoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.LayerNorm(dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
        self.heads = nn.ModuleList([CosineHead(dim, config.num_classes) for _ in range(4)])

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.reliability = nn.Sequential(nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
        self.updaters = nn.ModuleList([nn.GRUCell(dim, dim) for _ in range(4)])
        self.fusion_router = nn.Sequential(
            nn.Linear(dim + 4, 48), nn.SiLU(), nn.Linear(48, 1)
        )

        self.rx_domain_head = nn.Linear(dim, config.num_receivers)
        self.date_domain_head = nn.Linear(dim, config.num_dates)
        self.rx_adversaries = nn.ModuleList([nn.Linear(dim, config.num_receivers) for _ in range(2)])
        self.date_adversaries = nn.ModuleList([nn.Linear(dim, config.num_dates) for _ in range(2)])
        self.private_open_head = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.unknown_head = nn.Sequential(nn.Linear(7, 24), nn.SiLU(), nn.Dropout(0.1), nn.Linear(24, 1))

    @staticmethod
    def frequency_view(x: torch.Tensor) -> torch.Tensor:
        complex_signal = torch.complex(x[:, 0].float(), x[:, 1].float())
        spectrum = torch.fft.fftshift(torch.fft.fft(complex_signal, dim=-1), dim=-1)
        magnitude = torch.log1p(spectrum.abs())
        magnitude = (magnitude - magnitude.mean(-1, keepdim=True)) / (magnitude.std(-1, keepdim=True) + 1e-5)
        phase_delta = torch.angle(spectrum[:, 1:] * spectrum[:, :-1].conj()) / math.pi
        phase_delta = F.pad(phase_delta, (1, 0))
        return torch.stack([magnitude, phase_delta], dim=1)

    def _communication(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.query(states)
        k = self.key(states)
        v = self.value(states)
        reliability = torch.sigmoid(self.reliability(states).squeeze(-1))
        scores = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(states.shape[-1])
        scores = scores + torch.log(reliability[:, None, :] + 1e-6)
        diagonal = torch.eye(states.shape[1], device=states.device, dtype=torch.bool)[None]
        scores = scores.masked_fill(diagonal, -1e4)
        top_k = min(self.config.top_k, states.shape[1] - 1)
        top_indices = scores.topk(top_k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top_indices, True)
        adjacency = torch.softmax(scores.masked_fill(~mask, -1e4), dim=-1)
        messages = torch.einsum("bij,bjd->bid", adjacency, v)
        updated = torch.stack(
            [self.updaters[index](messages[:, index], states[:, index]) for index in range(4)],
            dim=1,
        )
        return F.normalize(updated, dim=-1), adjacency, reliability

    def forward(self, x: torch.Tensor, grl_scale: float = 1.0) -> dict[str, torch.Tensor]:
        time_h = self.time_encoder(x)
        freq_h = self.frequency_encoder(self.frequency_view(x))
        domain_h = self.domain_encoder(x)
        open_h = F.normalize(self.open_encoder(torch.cat([time_h, freq_h, domain_h], dim=-1)), dim=-1)
        pre_states = torch.stack([time_h, freq_h, domain_h, open_h], dim=1)
        pre_logits = torch.stack([head(pre_states[:, i]) for i, head in enumerate(self.heads)], dim=1)

        post_states, adjacency, reliability = self._communication(pre_states)
        post_logits = torch.stack([head(post_states[:, i]) for i, head in enumerate(self.heads)], dim=1)
        agent_probs = post_logits.softmax(dim=-1)
        agent_max = agent_probs.amax(dim=-1)
        agent_entropy = -(agent_probs * torch.log(agent_probs + 1e-8)).sum(-1) / math.log(self.config.num_classes)
        agent_proto_distance = 1.0 - agent_max
        router_input = torch.cat(
            [post_states, agent_max[..., None], agent_entropy[..., None], agent_proto_distance[..., None], reliability[..., None]],
            dim=-1,
        )
        fusion_weights = torch.softmax(self.fusion_router(router_input).squeeze(-1), dim=-1)
        fused_logits = (fusion_weights[..., None] * post_logits).sum(dim=1)
        fused_embedding = F.normalize((fusion_weights[..., None] * post_states).sum(dim=1), dim=-1)

        rx_logits = self.rx_domain_head(domain_h)
        date_logits = self.date_domain_head(domain_h)
        rx_prob = rx_logits.softmax(-1)
        date_prob = date_logits.softmax(-1)
        domain_shift = 0.5 * (1.0 - rx_prob.amax(-1) + 1.0 - date_prob.amax(-1))
        mean_prob = agent_probs.mean(dim=1)
        js_divergence = (
            agent_probs * (torch.log(agent_probs + 1e-8) - torch.log(mean_prob[:, None] + 1e-8))
        ).sum(-1).mean(-1)
        fused_entropy = -(fused_logits.softmax(-1) * fused_logits.log_softmax(-1)).sum(-1) / math.log(self.config.num_classes)
        router_entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(-1) / math.log(4.0)
        private_open_logit = self.private_open_head(post_states[:, 3]).squeeze(-1)
        fused_max = fused_logits.softmax(-1).amax(-1)
        energy = -torch.logsumexp(fused_logits, dim=-1) / 12.0
        open_features = torch.stack(
            [
                private_open_logit,
                1.0 - fused_max,
                fused_entropy,
                js_divergence,
                domain_shift,
                router_entropy,
                energy,
            ],
            dim=-1,
        )
        unknown_logit = self.unknown_head(open_features).squeeze(-1)

        reversed_time = gradient_reverse(time_h, grl_scale)
        reversed_freq = gradient_reverse(freq_h, grl_scale)
        return {
            "pre_states": pre_states,
            "post_states": post_states,
            "pre_logits": pre_logits,
            "post_logits": post_logits,
            "fused_logits": fused_logits,
            "fused_embedding": fused_embedding,
            "adjacency": adjacency,
            "reliability": reliability,
            "fusion_weights": fusion_weights,
            "rx_logits": rx_logits,
            "date_logits": date_logits,
            "adv_rx_logits": torch.stack([head(state) for head, state in zip(self.rx_adversaries, [reversed_time, reversed_freq])], dim=1),
            "adv_date_logits": torch.stack([head(state) for head, state in zip(self.date_adversaries, [reversed_time, reversed_freq])], dim=1),
            "open_features": open_features,
            "private_open_logit": private_open_logit,
            "unknown_logit": unknown_logit,
            "domain_shift": domain_shift,
            "js_divergence": js_divergence,
        }

    def prototype_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [head.target_distance(states[:, index], targets) for index, head in enumerate(self.heads)],
            dim=1,
        ).mean()

