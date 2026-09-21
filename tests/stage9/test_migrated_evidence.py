import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage9.evidence import CandidateEvidenceTable, EvidenceCertificate, EvidenceStance  # noqa: E402


def test_response_changes_candidate_belief():
    table = CandidateEvidenceTable(torch.tensor([[1.2, 1.0, -0.4]]))
    certificate = EvidenceCertificate(
        sender="impairment", receiver="identity",
        candidate_a=torch.tensor([0]), candidate_b=torch.tensor([1]),
        stance=torch.tensor([int(EvidenceStance.SUPPORT_B)]),
        signed_candidate_evidence=torch.tensor([-1.5]),
        alternative_class=torch.tensor([1]),
        domain_quality=torch.tensor([0.8]), open_risk=torch.tensor([0.0]),
        reliability=torch.tensor([1.0]), abstain=torch.tensor([False]),
        active_mask=torch.tensor([True]), bit_cost=torch.tensor([32.0]))
    delta = table.apply(certificate)
    assert torch.allclose(delta, torch.tensor([[-1.5, 1.5, 0.0]]))
    assert table.decision().prediction.tolist() == [1]

