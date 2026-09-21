import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.contracts import LocalDecision  # noqa: E402
from maros_stage8.system import Stage8System  # noqa: E402


def test_public_contract_has_no_hidden_state_or_embedding_fields():
    names = {field.name for field in fields(LocalDecision)}
    forbidden = {"state", "embedding", "tokens", "prototype", "memory"}
    assert names.isdisjoint(forbidden)


def test_private_state_requires_owner_capability():
    system = Stage8System(4, state_dim=8, hidden_channels=4).eval()
    context = system.observe(torch.randn(3, 2, 64))
    public = context.public_dict()
    with pytest.raises(PermissionError, match="capability"):
        context._private_for("temporal", object())
    query = system.make_query(
        public, sender="temporal", receiver="spectral")
    with pytest.raises(PermissionError, match="capability-bearing"):
        system.answer(public, query)
    assert set(public) == {"temporal", "spectral"}

