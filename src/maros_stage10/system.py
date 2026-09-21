"""One-turn live Stage-5-mother multi-Agent inference path."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .blackboard import SharedBlackboard
from .consultation import (IdentityQueryPolicy, IndependentEvidenceInvestigator,
                           apply_certificate, publish_candidate_support,
                           top_pair)
from .mother import collect_evidence
from .open_set import OpenSetExaminer, identity_public_after_scores
from .services import KnownQuantileThresholdService


@dataclass(frozen=True)
class RuntimeDecision:
    prediction: np.ndarray
    candidate_scores: np.ndarray  # private to caller/system, never blackboard
    blackboard: SharedBlackboard


class Stage10System:
    """A owns the query; B answers the pair; C examines Known support.

    B's frozen Stage-5 geometry encoder runs for every sample to publish a
    small open-support packet for C. Conditional querying saves *message*
    bandwidth, not B encoder compute. This distinction is intentional.
    """

    def __init__(self, mother, responder: IndependentEvidenceInvestigator,
                 query_policy: IdentityQueryPolicy, examiner: OpenSetExaminer,
                 threshold: KnownQuantileThresholdService,
                 *, certificate_scale: float, device: torch.device,
                 batch_size: int = 128):
        self.mother = mother.eval()
        self.responder = responder
        self.query_policy = query_policy
        self.examiner = examiner
        self.threshold = threshold
        self.certificate_scale = float(certificate_scale)
        self.device = device
        self.batch_size = int(batch_size)

    def infer(self, iq: torch.Tensor) -> RuntimeDecision:
        if iq.ndim != 3 or iq.shape[1] != 2:
            raise ValueError("runtime expects I/Q [batch,2,length]")
        samples = TensorDataset(
            iq.detach().cpu(), torch.zeros(len(iq), dtype=torch.long))
        private = collect_evidence(
            self.mother, samples, self.device, self.batch_size)
        a, b = top_pair(private.identity_logits)
        query_mask, _ = self.query_policy.ask(private.identity_public)
        # B sees its own frozen Geometry evidence and candidate IDs only.
        response = self.responder.answer(
            private.geometry_private(), a, b, query_mask)
        updated = apply_certificate(
            private.identity_logits, response, self.certificate_scale)
        support = publish_candidate_support(
            private.geometry_private(), updated.argmax(1))
        a_public = identity_public_after_scores(updated, private.identity_public)
        opinion = self.examiner.examine(a_public, support)
        rejected = self.threshold.reject(opinion.risk)
        prediction = np.where(rejected, -1, updated.argmax(1))
        blackboard = SharedBlackboard(
            candidate_a=a, candidate_b=b, query_mask=query_mask,
            response=response, open_support=support, open_opinion=opinion)
        return RuntimeDecision(prediction=prediction,
                               candidate_scores=updated, blackboard=blackboard)

