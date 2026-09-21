"""Capability-bounded Agent with a private toolbox and local belief loop."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import (BeliefPacket, CapabilityManifest, ReasonCode,
                         RequestSuggestion)
from ..policies.tool_policy import ToolPolicy
from ..tool_registry import ToolRegistry
from ..tools.encoders import EncoderTool


@dataclass(frozen=True)
class PrivateLocalState:
    logits: torch.Tensor
    action_indices: torch.Tensor
    policy_entropy: torch.Tensor


class Investigator(nn.Module):
    """Selects local tools, updates a candidate belief, and may abstain.

    The complete class logits remain private. Only a top-pair semantic packet
    is exportable. In G0 a request is a *suggestion*, not a sent message.
    """

    def __init__(self, role: str, tool_names: tuple[str, ...],
                 num_classes: int, state_dim: int = 48,
                 width: int = 24) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("at least two support classes required")
        self.role = role
        self.num_classes = num_classes
        self.manifest = CapabilityManifest(role, frozenset(tool_names))
        self.registry = ToolRegistry(self.manifest)
        for name in tool_names:
            self.registry.register(name, EncoderTool(
                name, role, num_classes, state_dim, width))
        self.tool_names = tool_names
        self.tool_policy = ToolPolicy(role, tool_names, width)

    @torch.no_grad()
    def reindex_classes(self, old_to_new: torch.Tensor) -> None:
        permutation = old_to_new.long()
        if (permutation.shape != (self.num_classes,)
                or sorted(permutation.cpu().tolist()) != list(range(self.num_classes))):
            raise ValueError("old_to_new must be a class permutation")
        for tool in self.registry.tools.values():
            local = permutation.to(tool.classifier.weight.device)
            weight = tool.classifier.weight.detach().clone()
            bias = tool.classifier.bias.detach().clone()
            tool.classifier.weight[local] = weight
            tool.classifier.bias[local] = bias

    def all_tool_logits(self, iq: torch.Tensor) -> torch.Tensor:
        """Training/diagnostic counterfactual only; never used in inference."""
        return torch.stack([
            self.registry.execute(name, iq) for name in self.tool_names], dim=1)

    def decide(self, iq: torch.Tensor) -> tuple[BeliefPacket, PrivateLocalState]:
        """Run scout, execute only selected tools, then form local belief."""
        policy_logits = self.tool_policy(iq)
        probabilities = policy_logits.softmax(1)
        action_indices = policy_logits.argmax(1)
        batch = len(iq)
        combined = iq.new_zeros((batch, self.num_classes))
        selected_logits: dict[int, torch.Tensor] = {}
        for tool_index, name in enumerate(self.tool_names):
            needed = torch.tensor([
                tool_index in self.tool_policy.actions[int(action)]
                for action in action_indices.detach().cpu().tolist()],
                device=iq.device, dtype=torch.bool)
            if bool(needed.any()):
                rows = needed.nonzero(as_tuple=False).flatten()
                proposal = self.registry.execute(name, iq.index_select(0, rows))
                contribution = iq.new_zeros((batch, self.num_classes))
                contribution.index_copy_(0, rows, proposal)
                selected_logits[tool_index] = contribution
        for action_index, action in enumerate(self.tool_policy.actions):
            rows = action_indices.eq(action_index).nonzero(as_tuple=False).flatten()
            if len(rows) and action:
                proposal = torch.stack([
                    selected_logits[index].index_select(0, rows)
                    for index in action], dim=0).mean(0)
                combined.index_copy_(0, rows, proposal)

        top = combined.softmax(1).topk(2, dim=1)
        margin = top.values[:, 0] - top.values[:, 1]
        selected_count = torch.tensor([
            len(self.tool_policy.actions[int(action)])
            for action in action_indices.detach().cpu().tolist()], device=iq.device)
        stopped = selected_count.eq(0)
        abstain = stopped | margin.lt(0.04)
        reason = torch.full_like(action_indices, int(ReasonCode.CLEAR_LOCAL_EVIDENCE))
        reason = torch.where(margin.lt(0.10),
                             torch.full_like(reason, int(ReasonCode.LOW_MARGIN)), reason)
        reason = torch.where(stopped,
                             torch.full_like(reason, int(ReasonCode.OUT_OF_SUPPORT)), reason)
        suggestion = torch.full_like(action_indices, int(RequestSuggestion.REPORT))
        ask = (RequestSuggestion.ASK_IMPAIRMENT if self.role == "identity"
               else RequestSuggestion.ASK_IDENTITY)
        suggestion = torch.where(abstain, torch.full_like(suggestion, int(ask)), suggestion)
        suggestion = torch.where(stopped,
                                 torch.full_like(suggestion, int(RequestSuggestion.ABSTAIN)),
                                 suggestion)
        used = tuple(tuple(self.tool_names[index] for index in
                           self.tool_policy.actions[int(action)])
                     for action in action_indices.detach().cpu().tolist())
        packet = BeliefPacket(
            sender=self.role, candidate_a=top.indices[:, 0],
            candidate_b=top.indices[:, 1], evidence_a=top.values[:, 0],
            evidence_b=top.values[:, 1], uncertainty=1.0 - margin,
            open_risk_opinion=None, used_tools=used, reason_code=reason,
            request_suggestion=suggestion, abstain=abstain)
        private = PrivateLocalState(
            logits=combined, action_indices=action_indices,
            policy_entropy=-(probabilities * probabilities.clamp_min(1e-9).log()).sum(1))
        return packet, private
