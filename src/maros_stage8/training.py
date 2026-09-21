"""Local specialization and candidate-conditioned verifier objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .contracts import LocalDecision
from .splits import assert_no_formal_unknown


@dataclass(frozen=True)
class VerifierObjective:
    loss: torch.Tensor
    classification_loss: torch.Tensor
    positive_loss: torch.Tensor
    hard_negative_loss: torch.Tensor
    pair_margin_loss: torch.Tensor
    pair_accuracy: torch.Tensor
    hard_negative: torch.Tensor


def candidate_conditioned_verifier_objective(
    agent: Any,
    private_state: object,
    decision: LocalDecision,
    labels: torch.Tensor,
    *,
    verification_weight: float = 1.0,
    pair_margin: float = 0.5,
) -> VerifierObjective:
    """Train true-candidate support against the closest current competitor."""

    labels = labels.long().to(decision.class_logits.device)
    if labels.shape != (len(decision.class_logits),) or bool((labels < 0).any()):
        raise ValueError("candidate verifier training requires Known labels [batch]")
    if verification_weight < 0 or pair_margin < 0:
        raise ValueError("verification_weight and pair_margin must be non-negative")
    masked = decision.class_logits.detach().clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    hard_negative = masked.argmax(1)
    positive_score = agent.verification_scores(private_state, labels)
    negative_score = agent.verification_scores(private_state, hard_negative)
    classification = F.cross_entropy(decision.class_logits, labels)
    positive = F.binary_cross_entropy_with_logits(
        positive_score, torch.ones_like(positive_score))
    negative = F.binary_cross_entropy_with_logits(
        negative_score, torch.zeros_like(negative_score))
    pair = F.relu(float(pair_margin) - positive_score + negative_score).mean()
    pair_accuracy = positive_score.gt(negative_score).float().mean()
    loss = classification + float(verification_weight) * (positive + negative + pair)
    return VerifierObjective(
        loss=loss, classification_loss=classification,
        positive_loss=positive, hard_negative_loss=negative,
        pair_margin_loss=pair, pair_accuracy=pair_accuracy,
        hard_negative=hard_negative)


def _classification_only_objective(
    decision: LocalDecision, labels: torch.Tensor,
) -> VerifierObjective:
    labels = labels.long().to(decision.class_logits.device)
    classification = F.cross_entropy(decision.class_logits, labels)
    masked = decision.class_logits.detach().clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    hard_negative = masked.argmax(1)
    zero = classification.new_zeros(())
    return VerifierObjective(
        loss=classification, classification_loss=classification,
        positive_loss=zero, hard_negative_loss=zero,
        pair_margin_loss=zero, pair_accuracy=zero,
        hard_negative=hard_negative)


def local_specialization_objective(system, temporal_observation,
                                   spectral_observation,
                                   labels: torch.Tensor, *,
                                   verification_weight: float = 1.0,
                                   pair_margin: float = 0.5):
    """Compute independent unit-scale A/B objectives without cross-view state."""

    temporal_decision, temporal_private = system.temporal.forward_private(
        temporal_observation)
    spectral_decision, spectral_private = system.spectral.forward_private(
        spectral_observation)
    if verification_weight == 0:
        temporal = _classification_only_objective(temporal_decision, labels)
        spectral = _classification_only_objective(spectral_decision, labels)
    else:
        temporal = candidate_conditioned_verifier_objective(
            system.temporal, temporal_private, temporal_decision, labels,
            verification_weight=verification_weight, pair_margin=pair_margin)
        spectral = candidate_conditioned_verifier_objective(
            system.spectral, spectral_private, spectral_decision, labels,
            verification_weight=verification_weight, pair_margin=pair_margin)
    return temporal.loss + spectral.loss, {
        "temporal": temporal, "spectral": spectral,
        "decisions": {"temporal": temporal_decision,
                      "spectral": spectral_decision},
    }


@torch.no_grad()
def refresh_candidate_prototypes(
    system,
    dataset: Dataset,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> dict[str, list[int]]:
    """Build fold-local A/B prototypes from support-Known samples only."""

    assert_no_formal_unknown(dataset)
    labels = getattr(dataset, "y", None)
    if labels is not None and bool((torch.as_tensor(labels) < 0).any()):
        raise RuntimeError("candidate prototypes require support-Known samples")
    modes = [(module, bool(module.training)) for module in system.modules()]
    system.eval()
    temporal_sum = torch.zeros(
        system.num_classes, system.temporal.candidate_prototypes.shape[1],
        device=device)
    spectral_sum = torch.zeros_like(temporal_sum)
    counts = torch.zeros(system.num_classes, dtype=torch.long, device=device)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    try:
        for iq, batch_labels in loader:
            iq = iq.to(device)
            batch_labels = batch_labels.long().to(device)
            if bool(((batch_labels < 0) | (batch_labels >= system.num_classes)).any()):
                raise ValueError("prototype labels are outside the support class range")
            temporal_observation, spectral_observation = (
                system.build_observations(iq))
            _, temporal_private = system.temporal.forward_private(
                temporal_observation)
            _, spectral_private = system.spectral.forward_private(
                spectral_observation)
            temporal_sum.index_add_(0, batch_labels, temporal_private.state)
            spectral_sum.index_add_(0, batch_labels, spectral_private.state)
            counts.index_add_(
                0, batch_labels, torch.ones_like(batch_labels, dtype=torch.long))
        if bool((counts == 0).any()):
            raise ValueError("every support class must contribute a prototype")
        denominator = counts.to(temporal_sum)[:, None]
        system.temporal.set_candidate_prototypes(
            temporal_sum / denominator, counts)
        system.spectral.set_candidate_prototypes(
            spectral_sum / denominator, counts)
    finally:
        for module, training in modes:
            module.training = training
    return {
        "temporal_counts": counts.cpu().tolist(),
        "spectral_counts": counts.cpu().tolist(),
    }
