import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.contracts import EvidenceCertificate, EvidenceStance  # noqa: E402
from maros_stage8.evidence import (CandidateEvidenceTable,  # noqa: E402
                                   shuffle_certificate)


def test_message_shuffle_preserves_cost_but_changes_candidate_effect():
    certificate = EvidenceCertificate(
        sender="spectral", receiver="temporal",
        candidate_a=torch.tensor([0, 1, 2]),
        candidate_b=torch.tensor([1, 2, 0]),
        stance=torch.tensor([int(EvidenceStance.SUPPORT_B)] * 3),
        signed_candidate_evidence=torch.tensor([-1.0, -2.0, -4.0]),
        alternative_class=torch.tensor([1, 2, 0]),
        domain_quality=torch.tensor([0.1, 0.2, 0.3]),
        open_risk=torch.zeros(3), reliability=torch.ones(3),
        abstain=torch.zeros(3, dtype=torch.bool),
        active_mask=torch.ones(3, dtype=torch.bool),
        bit_cost=torch.full((3,), 52.0))
    shuffled = shuffle_certificate(certificate, torch.tensor([1, 2, 0]))
    clean_table = CandidateEvidenceTable(torch.zeros(3, 3))
    shuffled_table = CandidateEvidenceTable(torch.zeros(3, 3))
    clean_table.apply(certificate)
    shuffled_table.apply(shuffled)

    assert torch.equal(shuffled.bit_cost, certificate.bit_cost)
    assert torch.equal(shuffled.active_mask, certificate.active_mask)
    assert not torch.equal(
        clean_table.decision().candidate_scores,
        shuffled_table.decision().candidate_scores)

