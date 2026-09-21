import sys
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.contracts import EvidenceCertificate, EvidenceStance  # noqa: E402
from maros_stage8.evidence import CandidateEvidenceTable  # noqa: E402


def _certificate(alternative):
    batch = 2
    return EvidenceCertificate(
        sender="spectral", receiver="temporal",
        candidate_a=torch.tensor([0, 0]), candidate_b=torch.tensor([1, 1]),
        stance=torch.full((batch,), int(EvidenceStance.SUPPORT_B), dtype=torch.long),
        signed_candidate_evidence=torch.tensor([-3.0, -3.0]),
        alternative_class=torch.tensor(alternative, dtype=torch.long),
        domain_quality=torch.ones(batch), open_risk=torch.zeros(batch),
        reliability=torch.ones(batch), abstain=torch.zeros(batch, dtype=torch.bool),
        active_mask=torch.ones(batch, dtype=torch.bool), bit_cost=torch.full((batch,), 52.0))


def test_certificate_updates_only_named_candidates_not_expert_weights():
    table = CandidateEvidenceTable(torch.zeros(2, 4))
    delta = table.apply(_certificate([1, 1]))
    assert torch.equal(delta[:, 2:], torch.zeros(2, 2))
    assert torch.all(delta[:, 0] < 0)
    assert torch.all(delta[:, 1] > 0)
    assert not hasattr(table.decision(), "known_weights")


def test_alternative_class_intervention_changes_candidate_and_prediction():
    base = torch.tensor([[0.0, 0.0, 0.0, -5.0],
                         [0.0, 0.0, 0.0, -5.0]])
    first = CandidateEvidenceTable(base)
    first.apply(_certificate([1, 1]))
    second = CandidateEvidenceTable(base)
    second.apply(_certificate([2, 2]))

    assert torch.equal(first.decision().prediction, torch.tensor([1, 1]))
    assert torch.equal(second.decision().prediction, torch.tensor([2, 2]))
    assert not torch.equal(
        first.decision().candidate_scores, second.decision().candidate_scores)


def test_abstaining_certificate_is_exact_identity():
    base = torch.randn(2, 4)
    packet = replace(_certificate([1, 1]),
                     abstain=torch.ones(2, dtype=torch.bool))
    table = CandidateEvidenceTable(base)
    delta = table.apply(packet)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(table.decision().candidate_scores, base)

