"""Candidate-level evidence aggregation; never whole-expert reweighting."""
from __future__ import annotations

from dataclasses import replace

import torch

from .contracts import CandidateDecision, EvidenceCertificate, EvidenceStance


class CandidateEvidenceTable:
    """Mutable audit table whose updates target explicit candidate IDs."""

    def __init__(self, candidate_scores: torch.Tensor,
                 open_risk: torch.Tensor | None = None):
        if candidate_scores.ndim != 2 or candidate_scores.shape[1] < 2:
            raise ValueError("candidate_scores must have shape [batch, classes>=2]")
        self.scores = candidate_scores.clone()
        batch = candidate_scores.shape[0]
        self.open_risk = (torch.zeros(batch, device=candidate_scores.device,
                                     dtype=candidate_scores.dtype)
                          if open_risk is None else open_risk.clone())
        if self.open_risk.shape != (batch,):
            raise ValueError("open_risk must have shape [batch]")
        self._contributions: list[torch.Tensor] = []
        self._certificates: list[EvidenceCertificate] = []

    def apply(self, certificate: EvidenceCertificate,
              scale: float = 1.0) -> torch.Tensor:
        """Apply one certificate to named candidates and return its delta."""

        batch, classes = self.scores.shape
        fields = (
            certificate.candidate_a, certificate.candidate_b,
            certificate.alternative_class, certificate.stance,
            certificate.signed_candidate_evidence, certificate.domain_quality,
            certificate.open_risk, certificate.reliability,
            certificate.abstain, certificate.active_mask,
        )
        if any(not torch.is_tensor(value) or value.shape != (batch,)
               for value in fields):
            raise ValueError("certificate fields must all have shape [batch]")
        active = certificate.active_mask.bool() & ~certificate.abstain.bool()
        for candidates in (
                certificate.candidate_a, certificate.candidate_b,
                certificate.alternative_class):
            selected = candidates[active]
            if bool(((selected < 0) | (selected >= classes)).any()):
                raise ValueError("active certificate candidate is out of range")

        strength = (certificate.signed_candidate_evidence.abs()
                    * certificate.reliability.clamp(0.0, 1.0)
                    * float(scale))
        strength = torch.where(active, strength, torch.zeros_like(strength))
        delta = torch.zeros_like(self.scores)
        support_a = active & certificate.stance.eq(int(EvidenceStance.SUPPORT_A))
        support_b = active & certificate.stance.eq(int(EvidenceStance.SUPPORT_B))
        unknown = active & certificate.stance.eq(int(EvidenceStance.UNKNOWN_SUSPECT))

        # Boolean advanced indexing returns a copy, so use flat index_put with
        # accumulate=True to make every candidate-targeted update explicit.
        rows = torch.arange(batch, device=self.scores.device)
        delta.index_put_((rows[support_a], certificate.candidate_a[support_a]),
                         strength[support_a], accumulate=True)
        delta.index_put_((rows[support_a], certificate.candidate_b[support_a]),
                         -strength[support_a], accumulate=True)
        delta.index_put_((rows[support_b], certificate.candidate_a[support_b]),
                         -strength[support_b], accumulate=True)
        delta.index_put_((rows[support_b], certificate.alternative_class[support_b]),
                         strength[support_b], accumulate=True)
        self.scores = self.scores + delta
        self.open_risk = self.open_risk + torch.where(
            unknown, certificate.open_risk.abs() + strength,
            torch.zeros_like(strength))
        self._contributions.append(delta)
        self._certificates.append(certificate)
        return delta

    def decision(self) -> CandidateDecision:
        return CandidateDecision(
            candidate_scores=self.scores.clone(),
            open_risk=self.open_risk.clone(), prediction=self.scores.argmax(1),
            contributions=tuple(value.clone() for value in self._contributions),
            certificates=tuple(self._certificates))


def shuffle_certificate(
    certificate: EvidenceCertificate, order: torch.Tensor,
) -> EvidenceCertificate:
    """Counterfactual: reassign semantic payload while preserving edge/cost.

    Candidate IDs, stance, evidence, quality, risk, reliability and abstention
    travel together.  ``active_mask`` and ``bit_cost`` remain attached to the
    original communication action, matching the Stage-7 intervention audit.
    """

    batch = len(certificate.active_mask)
    if order.shape != (batch,) or order.dtype != torch.long:
        raise ValueError("shuffle order must be a long tensor of shape [batch]")
    if sorted(order.detach().cpu().tolist()) != list(range(batch)):
        raise ValueError("shuffle order must be a permutation")
    return replace(
        certificate,
        candidate_a=certificate.candidate_a[order],
        candidate_b=certificate.candidate_b[order],
        stance=certificate.stance[order],
        signed_candidate_evidence=certificate.signed_candidate_evidence[order],
        alternative_class=certificate.alternative_class[order],
        domain_quality=certificate.domain_quality[order],
        open_risk=certificate.open_risk[order],
        reliability=certificate.reliability[order],
        abstain=certificate.abstain[order],
    )
