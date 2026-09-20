from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_staged.diagnose import _channel_view
from maros_staged.boundary_agent import OpenSpaceBoundaryAgent, boundary_features, make_prototypes
from maros_staged.communication import MultiAgentCommunicationModel
from maros_staged.evidence import ClassConditionalCdf
from maros_staged.metrics_osr import oscr
from maros_staged.model import Stage1ModelConfig, ThreeAgentModel
from maros_staged.pseudo_unknown import (
    DisagreementGuidedPseudoUnknownAgent,
    PseudoUnknownConfig,
)
from maros_staged.router import ReliabilityRouter


def _initialised_model() -> ThreeAgentModel:
    cfg = Stage1ModelConfig(
        num_classes=4,
        embedding_dim=16,
        stem_channels=32,
        latent_channels=8,
        signal_length=256,
    )
    model = ThreeAgentModel(cfg)
    prototypes = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    model.prototype.set_prototypes(prototypes)
    return model


def test_three_agent_loss_reaches_each_load_bearing_component() -> None:
    model = _initialised_model()
    x = torch.randn(8, 2, 256)
    y = torch.arange(8) % 4
    out = model(x, recon_classes=y)
    losses = model.losses(out, y)
    losses["total"].backward()

    assert out["identity"]["embedding"].shape == (8, 16)
    assert out["prototype"]["embedding"].shape == (8, 16)
    # Reconstruction representation is the trained decoder bottleneck, whose
    # width may differ from the two discriminative embeddings.
    assert out["reconstruction"]["embedding"].shape == (8, 8)
    assert model.identity.classifier.weight.grad is not None
    assert model.prototype.encoder.proj[-1].weight.grad is not None
    assert model.reconstruction.decoder.bottleneck.weight.grad is not None
    assert model.reconstruction.decoder.up[-1].weight.grad is not None


def test_class_conditional_cdf_is_monotone_and_bounded() -> None:
    ref = np.asarray([0.1, 0.2, 0.3, 1.0, 1.1, 1.2])
    cls = np.asarray([0, 0, 0, 1, 1, 1])
    cal = ClassConditionalCdf(ref, cls, n_prior=1.0)
    score = cal.score(np.asarray([0.0, 0.25, 2.0]), np.asarray([0, 0, 0]))
    assert np.all((0.0 <= score) & (score <= 1.0))
    assert np.all(np.diff(score) >= 0.0)


def test_all_channel_oscr_views_use_the_calibrated_unknown_score() -> None:
    data = {
        "pred_id": np.asarray([0, 1]),
        "pred_proto": np.asarray([1, 0]),
        "closed_pred": np.asarray([0, 0]),
        "u_identity": np.asarray([0.2, 0.9]),
        "u_prototype": np.asarray([0.4, 0.7]),
        "u_reconstruction": np.asarray([0.3, 0.8]),
        "u_fused_mean": np.asarray([0.3, 0.8]),
        "u_fused_max": np.asarray([0.4, 0.9]),
    }
    for channel, key in (
        ("identity", "u_identity"),
        ("prototype", "u_prototype"),
        ("reconstruction", "u_reconstruction"),
        ("fused_mean", "u_fused_mean"),
        ("fused_max", "u_fused_max"),
    ):
        _, known_confidence = _channel_view(data, channel)
        np.testing.assert_allclose(known_confidence, 1.0 - data[key])


def test_oscr_is_one_for_perfect_separation_and_classification() -> None:
    is_unknown = np.asarray([0, 0, 1, 1])
    y = np.asarray([0, 1, -1, -1])
    pred = np.asarray([0, 1, 0, 1])
    known_confidence = np.asarray([0.95, 0.9, 0.1, 0.05])
    assert oscr(is_unknown, pred, y, known_confidence) == 1.0


def test_pseudo_unknown_agent_is_deterministic_and_preserves_agent_norms() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 12)
    embeddings = {
        "identity": rng.normal(size=(36, 8)).astype(np.float32),
        "prototype": rng.normal(size=(36, 6)).astype(np.float32),
        "reconstruction": rng.normal(size=(36, 4)).astype(np.float32),
    }
    # Give every class a different centre so outward directions are defined.
    for cls in range(3):
        mask = labels == cls
        for name in embeddings:
            embeddings[name][mask, cls % embeddings[name].shape[1]] += 4.0
    evidence = rng.random((36, 3), dtype=np.float32)
    cfg = PseudoUnknownConfig(seed_ratio=0.25, variations=3, knn_k=3, seed=11)
    first = DisagreementGuidedPseudoUnknownAgent(cfg).generate(embeddings, labels, evidence)
    second = DisagreementGuidedPseudoUnknownAgent(cfg).generate(embeddings, labels, evidence)
    np.testing.assert_allclose(first["fused"], second["fused"])
    assert len(first["fused"]) == 3 * 3 * 3  # classes * seeds/class * variations
    for name in ("identity", "prototype", "reconstruction"):
        norms = np.linalg.norm(first[f"z_{name}"], axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_boundary_agent_accepts_multi_agent_geometry_features() -> None:
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(4), 5)
    embeddings = {
        "identity": rng.normal(size=(20, 8)).astype(np.float32),
        "prototype": rng.normal(size=(20, 6)).astype(np.float32),
        "reconstruction": rng.normal(size=(20, 4)).astype(np.float32),
    }
    prototypes = make_prototypes(embeddings, labels, num_classes=4)
    features, opinions = boundary_features(embeddings, prototypes)
    assert features.shape == (20, 8 + 6 + 4 + 15)
    assert opinions.shape == (20, 3)
    agent = OpenSpaceBoundaryAgent(features.shape[1], num_classes=4, hidden_dim=32)
    out = agent(torch.from_numpy(features))
    assert out["unknown_logit"].shape == (20,)
    assert out["class_logits"].shape == (20, 4)


def test_explicit_communication_has_directed_edges_and_changes_states() -> None:
    torch.manual_seed(5)
    model = MultiAgentCommunicationModel(
        {"identity": 8, "prototype": 6, "reconstruction": 4},
        num_classes=4, evidence_dim=4, message_dim=16, hidden_dim=32, dropout=0.0,
    )
    embeddings = {
        "identity": torch.randn(7, 8),
        "prototype": torch.randn(7, 6),
        "reconstruction": torch.randn(7, 4),
    }
    evidence = torch.randn(7, 3, 4)
    communicated = model(embeddings, evidence, communicate=True)
    isolated = model(embeddings, evidence, communicate=False)
    assert communicated["adjacency"].shape == (7, 3, 3)
    assert communicated["fused_logits"].shape == (7, 4)
    diagonal = communicated["adjacency"].diagonal(dim1=1, dim2=2)
    assert torch.allclose(diagonal, torch.zeros_like(diagonal))
    assert torch.all(communicated["adjacency"].sum(dim=2) > 0)
    assert not torch.allclose(communicated["tokens_post"], isolated["tokens_post"])


def test_reliability_router_only_convex_combines_agent_system_outputs() -> None:
    router = ReliabilityRouter(observation_dim=9, mode="dual", hidden_dim=16)
    observation = torch.randn(6, 9)
    no_comm = {"fused_logits": torch.randn(6, 4), "unknown_logit": torch.randn(6)}
    communication = {"fused_logits": torch.randn(6, 4), "unknown_logit": torch.randn(6)}
    out = router(observation, no_comm, communication)
    assert torch.allclose(out["known_weights"].sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.allclose(out["unknown_weights"].sum(dim=1), torch.ones(6), atol=1e-6)
    lower = torch.minimum(no_comm["fused_logits"], communication["fused_logits"])
    upper = torch.maximum(no_comm["fused_logits"], communication["fused_logits"])
    assert torch.all(out["known_logits"] >= lower - 1e-6)
    assert torch.all(out["known_logits"] <= upper + 1e-6)
