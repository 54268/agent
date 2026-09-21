import json

import numpy as np
import torch

from maros_stage5.agents import RoleStructuredExperts
from maros_stage5.training import ExpertRecords
from maros_stage13.experiment import load_selection_tail
from maros_stage13.feature_pug import generate_candidates, incremental


def _records(n=60, classes=3, dim=8):
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(classes), n // classes)
    zi = rng.normal(size=(n, dim)).astype(np.float32)
    zg = rng.normal(size=(n, dim)).astype(np.float32)
    il = rng.normal(size=(n, classes)).astype(np.float32)
    gl = rng.normal(size=(n, classes)).astype(np.float32)
    evidence = rng.normal(size=(n, 20)).astype(np.float32)
    return ExpertRecords(zi, zg, il, gl, evidence, labels,
                         np.zeros((n, 5), dtype=int))


def test_generated_states_are_normalized_and_not_source_copies():
    records = _records()
    pseudo, _ = generate_candidates(records, "competition", 1.5, 42)
    assert np.allclose(np.linalg.norm(pseudo.identity_state, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(pseudo.geometry_state, axis=1), 1, atol=1e-5)
    assert np.any(np.abs(pseudo.identity_state -
                         records.identity_state[pseudo.source_indices]) > 1e-4)


def test_state_level_interfaces_recompute_both_agent_reports():
    torch.manual_seed(2)
    model = RoleStructuredExperts(3, state_dim=8, stem_channels=8,
                                  message_dim=4, geometry_view="raw").eval()
    x = torch.randn(5, 2, 256)
    identity = model.identity(x)
    geometry = model.geometry(x)
    identity_again = model.identity.forward_from_state(identity.state)
    geometry_again = model.geometry.forward_from_state(geometry.state)
    assert torch.allclose(identity.class_logits, identity_again.class_logits)
    assert torch.allclose(geometry.class_logits, geometry_again.class_logits)
    assert "distances" in identity_again.evidence
    assert "distance_margin" in geometry_again.evidence
    assert "view_weights" not in geometry_again.evidence


def test_c_delta_never_replaces_stronger_stage5_risk():
    base = np.array([0.9, 0.2, 0.5])
    evidence = np.array([0.1, 0.8, 0.5])
    result = incremental(base, evidence, 0.2)
    assert np.all(result >= base)
    assert result[0] == base[0]


def test_large_audit_file_tail_restores_only_frozen_selection(tmp_path):
    path = tmp_path / "selection.json"
    payload = {"noise": "x" * 9000,
               "selected": {"pug": "competition_eta1.5",
                            "score_rule": "idproto_boundary"},
               "protocol": {"real_unknown_used": False}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    selected = load_selection_tail(path)
    assert selected == payload["selected"]
