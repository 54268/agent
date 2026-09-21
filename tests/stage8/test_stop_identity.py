import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.system import Stage8System  # noqa: E402


def test_no_communication_is_bit_exact_anchor_identity():
    torch.manual_seed(5)
    system = Stage8System(5, state_dim=8, hidden_channels=4).eval()
    context = system.observe(torch.randn(4, 2, 64))
    for anchor in ("temporal", "spectral"):
        stopped = system.no_communication(context, anchor=anchor)
        assert torch.equal(stopped.candidate_scores, context[anchor].class_logits)
        assert torch.equal(stopped.open_risk, context[anchor].unknown_score)
        assert torch.equal(stopped.prediction, context[anchor].top1)
        assert stopped.contributions == ()
        assert stopped.certificates == ()

