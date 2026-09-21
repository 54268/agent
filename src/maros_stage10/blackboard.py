"""Public conversation ledger. No model states or full class logits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maros_stage8.contracts import EvidenceCertificate

from .consultation import PublicSupportPacket
from .open_set import OpenSetOpinion


@dataclass(frozen=True)
class SharedBlackboard:
    candidate_a: np.ndarray
    candidate_b: np.ndarray
    query_mask: np.ndarray
    response: EvidenceCertificate
    open_support: PublicSupportPacket
    open_opinion: OpenSetOpinion

    def __post_init__(self) -> None:
        count = len(self.candidate_a)
        if (self.candidate_b.shape != (count,)
                or self.query_mask.shape != (count,)
                or self.open_support.candidate.shape != (count,)
                or self.open_opinion.risk.shape != (count,)):
            raise ValueError("blackboard public fields must align by sample")
        if int(self.response.active_mask.sum()) != int(np.sum(self.query_mask)):
            raise ValueError("blackboard response does not match active queries")

