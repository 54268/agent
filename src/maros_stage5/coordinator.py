"""Two-round supervised communication and constrained open-set coordination."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import AgentOutput, MultiAgentOutput


EDGE_NAMES = ("geometry_to_identity", "identity_to_geometry",
              "identity_to_boundary", "geometry_to_boundary")


def _js_from_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pa, pb = F.softmax(a, dim=-1), F.softmax(b, dim=-1)
    mean = 0.5 * (pa + pb)
    return 0.5 * (
        (pa * (pa.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
        + (pb * (pb.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
    )


class BoundaryAuditorAgent(nn.Module):
    name = "boundary_auditor"

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


class CommunicationDecisionCoordinator(nn.Module):
    """Coordinator whose class action is restricted to a convex two-agent fusion."""

    def __init__(self, num_classes: int, state_dim: int, evidence_dim: int,
                 message_dim: int = 32, hidden_dim: int = 128,
                 mode: str = "sparse_cf", budget_target: float = 0.50,
                 gate_threshold: float = 0.50):
        super().__init__()
        if mode not in {"none", "full", "sparse", "sparse_cf",
                        "audit_sparse", "audit_sparse_cf",
                        "audit_optional", "audit_optional_cf",
                        "audit_residual", "audit_residual_cf"}:
            raise ValueError(f"unsupported communication mode {mode}")
        self.num_classes = int(num_classes)
        self.state_dim = int(state_dim)
        self.evidence_dim = int(evidence_dim)
        self.message_dim = int(message_dim)
        self.mode = mode
        self.budget_target = float(budget_target)
        self.gate_threshold = float(gate_threshold)
        if not 0.0 < self.gate_threshold < 1.0:
            raise ValueError("gate_threshold must lie in (0, 1)")
        self.id_message = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, message_dim), nn.Tanh())
        self.geo_message = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, message_dim), nn.Tanh())
        self.id_reliability = nn.Sequential(nn.Linear(state_dim + evidence_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.geo_reliability = nn.Sequential(nn.Linear(state_dim + evidence_dim, 32), nn.GELU(), nn.Linear(32, 1))
        gate_dim = 2 * message_dim + 2
        self.gate_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(gate_dim, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in EDGE_NAMES
        ])
        self.id_update = nn.Sequential(nn.Linear(message_dim, state_dim), nn.Tanh())
        self.geo_update = nn.Sequential(nn.Linear(message_dim, state_dim), nn.Tanh())
        self.id_delta_head = nn.Linear(state_dim, num_classes, bias=False)
        self.geo_delta_head = nn.Linear(state_dim, num_classes, bias=False)
        nn.init.zeros_(self.id_delta_head.weight)
        nn.init.zeros_(self.geo_delta_head.weight)
        self.weight_head = nn.Sequential(
            nn.Linear(evidence_dim + 2, 32), nn.GELU(), nn.Linear(32, 2)
        )
        disagreement_dim = 4
        boundary_dim = 2 * message_dim + evidence_dim + disagreement_dim
        self.boundary = BoundaryAuditorAgent(boundary_dim, hidden_dim)
        self.boundary_residual = BoundaryAuditorAgent(boundary_dim, hidden_dim)

    def _soft_gates(self, id_msg, geo_msg, id_rel, geo_rel) -> torch.Tensor:
        if self.mode == "none":
            return id_msg.new_zeros((len(id_msg), len(EDGE_NAMES)))
        if self.mode == "full":
            return id_msg.new_ones((len(id_msg), len(EDGE_NAMES)))
        pairs = (
            (geo_msg, id_msg, geo_rel, 1.0 - id_rel),
            (id_msg, geo_msg, id_rel, 1.0 - geo_rel),
            (id_msg, geo_msg, id_rel, 0.5 * (1.0 - id_rel + 1.0 - geo_rel)),
            (geo_msg, id_msg, geo_rel, 0.5 * (1.0 - id_rel + 1.0 - geo_rel)),
        )
        gates = []
        for head, (sender, receiver, supply, demand) in zip(self.gate_heads, pairs):
            feature = torch.cat([sender, receiver, supply[:, None], demand[:, None]], dim=1)
            gates.append(torch.sigmoid(head(feature).squeeze(-1)))
        gates = torch.stack(gates, dim=1)
        if self.mode in {"audit_sparse", "audit_sparse_cf",
                         "audit_optional", "audit_optional_cf",
                         "audit_residual", "audit_residual_cf"}:
            # Directed consultation: specialists keep their independent class
            # states and only send evidence to the Boundary Auditor.
            gates = torch.cat([torch.zeros_like(gates[:, :2]), gates[:, 2:]], dim=1)
        return gates

    def _hard_gates(self, gates: torch.Tensor, force_auditor: bool = True) -> torch.Tensor:
        hard = (gates >= self.gate_threshold).to(gates.dtype)
        # At least one specialist must reach the auditor.
        audit = hard[:, 2:4]
        missing = audit.sum(dim=1) == 0
        if force_auditor and bool(missing.any()):
            chosen = gates[missing, 2:4].argmax(dim=1)
            audit[missing] = F.one_hot(chosen, num_classes=2).to(gates.dtype)
            hard[:, 2:4] = audit
        return hard

    def forward(self, identity_state: torch.Tensor, geometry_state: torch.Tensor,
                identity_logits: torch.Tensor, geometry_logits: torch.Tensor,
                evidence: torch.Tensor, *, hard: bool = False,
                edge_override: torch.Tensor | None = None,
                shuffle_messages: bool = False,
                uniform_weights: bool = False) -> MultiAgentOutput:
        id_msg, geo_msg = self.id_message(identity_state), self.geo_message(geometry_state)
        id_rel = torch.sigmoid(self.id_reliability(torch.cat([identity_state, evidence], dim=1))).squeeze(-1)
        geo_rel = torch.sigmoid(self.geo_reliability(torch.cat([geometry_state, evidence], dim=1))).squeeze(-1)
        gates = self._soft_gates(id_msg, geo_msg, id_rel, geo_rel)
        if hard and self.mode in {"sparse", "sparse_cf", "audit_sparse", "audit_sparse_cf",
                                  "audit_optional", "audit_optional_cf",
                                  "audit_residual", "audit_residual_cf"}:
            force = self.mode not in {"audit_optional", "audit_optional_cf",
                                      "audit_residual", "audit_residual_cf"}
            gates = self._hard_gates(gates, force_auditor=force)
        if edge_override is not None:
            if edge_override.ndim == 1:
                edge_override = edge_override[None].expand(len(gates), -1)
            gates = edge_override.to(gates)
        if shuffle_messages and len(id_msg) > 1:
            # Deterministic under the caller's torch seed.
            order = torch.randperm(len(id_msg), device=id_msg.device)
            id_msg, geo_msg = id_msg[order], geo_msg[order]

        id_delta = gates[:, 0:1] * self.id_update(geo_msg)
        geo_delta = gates[:, 1:2] * self.geo_update(id_msg)
        updated_id_state = identity_state + id_delta
        updated_geo_state = geometry_state + geo_delta
        updated_id_logits = identity_logits + self.id_delta_head(id_delta)
        updated_geo_logits = geometry_logits + self.geo_delta_head(geo_delta)
        if uniform_weights:
            known_weights = evidence.new_full((len(evidence), 2), 0.5)
        else:
            known_weights = F.softmax(self.weight_head(torch.cat(
                [evidence, id_rel[:, None], geo_rel[:, None]], dim=1)), dim=1)
        fused_logits = (known_weights[:, 0:1] * updated_id_logits
                        + known_weights[:, 1:2] * updated_geo_logits)

        js = _js_from_logits(updated_id_logits, updated_geo_logits)
        pred_disagree = (updated_id_logits.argmax(1) != updated_geo_logits.argmax(1)).float()
        state_cos = F.cosine_similarity(updated_id_state, updated_geo_state, dim=1)
        reliability_gap = (id_rel - geo_rel).abs()
        disagreement = torch.stack([js, pred_disagree, state_cos, reliability_gap], dim=1)
        boundary_input = torch.cat([
            gates[:, 2:3] * id_msg,
            gates[:, 3:4] * geo_msg,
            evidence,
            disagreement,
        ], dim=1)
        if self.mode in {"audit_residual", "audit_residual_cf"}:
            baseline_input = torch.cat([
                torch.zeros_like(id_msg), torch.zeros_like(geo_msg), evidence, disagreement,
            ], dim=1)
            baseline_logit = self.boundary(baseline_input)
            audit_strength = gates[:, 2:4].amax(dim=1)
            residual = F.softplus(self.boundary_residual(boundary_input))
            unknown_logit = baseline_logit + audit_strength * residual
        else:
            baseline_logit = self.boundary(boundary_input)
            residual = torch.zeros_like(baseline_logit)
            unknown_logit = baseline_logit
        identity = AgentOutput(updated_id_state, updated_id_logits, evidence,
                               id_rel, id_msg, {"reliability": id_rel})
        geometry = AgentOutput(updated_geo_state, updated_geo_logits, evidence,
                               geo_rel, geo_msg, {"reliability": geo_rel})
        boundary = AgentOutput(boundary_input, fused_logits, unknown_logit[:, None],
                               1.0 - (torch.sigmoid(unknown_logit) - 0.5).abs() * 2.0,
                               boundary_input[:, :self.message_dim],
                               {"unknown_logit": unknown_logit,
                                "baseline_logit": baseline_logit,
                                "message_residual": residual})
        return MultiAgentOutput(fused_logits, unknown_logit, gates,
                                {"identity": identity, "geometry": geometry,
                                 "boundary": boundary}, known_weights)

    def budget_loss(self, gates: torch.Tensor) -> torch.Tensor:
        if self.mode not in {"sparse", "sparse_cf", "audit_sparse", "audit_sparse_cf",
                             "audit_optional", "audit_optional_cf",
                             "audit_residual", "audit_residual_cf"}:
            return gates.new_zeros(())
        return (gates.mean() - self.budget_target).square()


def per_sample_task_loss(output: MultiAgentOutput, unknown_target: torch.Tensor,
                         labels: torch.Tensor, class_weight: float = 0.5) -> torch.Tensor:
    binary = F.binary_cross_entropy_with_logits(output.unknown_logit, unknown_target,
                                                 reduction="none")
    known = labels >= 0
    result = binary
    if bool(known.any()):
        ce = F.cross_entropy(output.fused_class_logits[known], labels[known], reduction="none")
        result = result.clone()
        result[known] += class_weight * ce
    return result


def counterfactual_gate_loss(model: CommunicationDecisionCoordinator,
                             inputs: Dict[str, torch.Tensor], output: MultiAgentOutput,
                             unknown_target: torch.Tensor, labels: torch.Tensor,
                             temperature: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    full_loss = per_sample_task_loss(output, unknown_target, labels)
    deltas = []
    for edge in range(len(EDGE_NAMES)):
        override = output.edge_gates.detach().clone()
        override[:, edge] = 0.0
        dropped = model(**inputs, edge_override=override)
        dropped_loss = per_sample_task_loss(dropped, unknown_target, labels)
        deltas.append((dropped_loss - full_loss).detach())
    delta = torch.stack(deltas, dim=1)
    target = torch.sigmoid(delta / max(temperature, 1e-6))
    gate_loss = F.binary_cross_entropy(output.edge_gates.clamp(1e-5, 1 - 1e-5), target)
    return gate_loss, delta
