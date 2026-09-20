import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.agents import WaveformIdentityAgent  # noqa: E402
from maros_stage7.model import Stage7System  # noqa: E402
from maros_stage7.training import (  # noqa: E402
    local_metric_objective,
    refresh_enrollment,
)


class _KnownDataset(Dataset):
    purpose = "outer0_train_known"
    source_split = "train"
    formal_unknown = False

    def __init__(self):
        generator = torch.Generator().manual_seed(401)
        self.x = torch.randn(12, 2, 32, generator=generator)
        self.y = np.repeat(np.arange(3, dtype=np.int64), 4)
        for index, label in enumerate(self.y):
            self.x[index, 0] += float(label) * 0.4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], torch.tensor(int(self.y[index]))


def _system():
    torch.manual_seed(91)
    return Stage7System(
        3, state_dim=8, stem_channels=8, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)


def test_g0_v2_keeps_multiview_evidence_private():
    system = _system().eval()
    x = _KnownDataset().x[:4]
    context = system.local(x)
    assert set(context["prototype"].__dataclass_fields__) == {
        "class_logits", "unknown_score", "reliability", "top1", "top2",
        "public_summary",
    }
    private = system.agents.training_private(context)["prototype"]
    assert system.agents.prototype.view_names == (
        "raw", "spectral", "envelope_phase", "difference_iq", "complex_iq")
    assert system.agents.prototype.real_view_names == (
        "raw", "spectral", "envelope_phase", "difference_iq")
    assert private.view_states.shape == (4, 5, 8)
    assert private.view_logits.shape == (4, 5, 3)
    assert private.view_distances.shape == (4, 5, 3)
    assert torch.allclose(private.view_weights.sum(1), torch.ones(4))


def test_metric_objective_reaches_both_private_backbones():
    system = _system().train()
    dataset = _KnownDataset()
    indices = torch.tensor([0, 1, 4, 5, 8, 9])
    x = dataset.x[indices]
    y = torch.as_tensor(dataset.y[indices])
    context = system.local(x)
    loss, pieces = local_metric_objective(system, context, y)
    assert torch.isfinite(loss)
    assert set(pieces) == {
        "waveform_supcon", "prototype_supcon", "prototype_view_aux",
        "prototype_compact", "prototype_margin",
    }
    loss.backward()
    waveform_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in system.agents.waveform.encoder.parameters()
        if parameter.grad is not None)
    view_grads = [sum(
        float(parameter.grad.abs().sum())
        for parameter in encoder.parameters() if parameter.grad is not None)
        for encoder in system.agents.prototype.view_encoders]
    complex_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in system.agents.prototype.complex_encoder.parameters()
        if parameter.grad is not None)
    assert waveform_grad > 0.0
    assert all(value > 0.0 for value in view_grads)
    assert complex_grad > 0.0


def test_waveform_classifier_has_stable_raw_state_and_bounded_complex_residual():
    system = _system().train()
    dataset = _KnownDataset()
    x = dataset.x[:6]
    y = torch.as_tensor(dataset.y[:6])
    encoder = system.agents.waveform.encoder
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    raw_hook = encoder.raw_encoder.register_forward_hook(save("raw"))
    residual_hook = encoder.complex_residual.register_forward_hook(
        save("complex_delta"))
    try:
        context = system.local(x)
    finally:
        raw_hook.remove()
        residual_hook.remove()
    state = system.agents.training_private(context)["waveform"].state
    expected = (
        captured["raw"]
        + encoder.COMPLEX_RESIDUAL_BOUND * captured["complex_delta"])
    assert torch.allclose(state, expected, atol=1e-6)
    assert torch.all(
        (state - captured["raw"]).abs()
        <= encoder.COMPLEX_RESIDUAL_BOUND + 1e-6)
    # The classifier state keeps the useful magnitude learned by the raw
    # PrivateEncoder; unit normalisation is reserved for SupCon/enrollment.
    assert not torch.allclose(
        state.norm(dim=1), torch.ones(len(state)), atol=1e-4)

    loss = torch.nn.functional.cross_entropy(
        context["waveform"].class_logits, y)
    loss.backward()
    assert encoder.raw_stem.stem[0].weight.grad.abs().sum() > 0
    assert encoder.blocks[0].conv.real.weight.grad.abs().sum() > 0
    assert encoder.token_project.weight.grad.abs().sum() > 0


def test_waveform_response_head_transfers_across_class_counts():
    torch.manual_seed(52)
    inner = WaveformIdentityAgent(
        3, state_dim=8, class_embed_dim=4, complex_hidden=4)
    outer = WaveformIdentityAgent(
        7, state_dim=8, class_embed_dim=4, complex_hidden=4)
    response_modules = (
        "class_descriptor", "pair_query", "pair_residual", "tail_head")
    for name in response_modules:
        source = getattr(inner, name).state_dict()
        target = getattr(outer, name)
        target.load_state_dict(source, strict=True)
        for key, value in source.items():
            assert torch.equal(target.state_dict()[key], value)
    assert not hasattr(inner, "class_embedding")
    assert not hasattr(outer, "class_embedding")


def test_waveform_raw_capacity_uses_shared_stem_config_not_complex_width():
    system = Stage7System(
        3, state_dim=8, stem_channels=16, class_embed_dim=4,
        complex_hidden=4, hidden_dim=8)
    encoder = system.agents.waveform.encoder
    assert encoder.raw_channels == 16
    assert encoder.raw_stem.out_channels == 16
    assert encoder.blocks[0].conv.real.out_channels == 4

    # Direct construction keeps the old public API behaviour: omitting the
    # new option derives a valid raw width from complex_hidden.
    compatible = WaveformIdentityAgent(
        3, state_dim=8, class_embed_dim=4, complex_hidden=4)
    assert compatible.encoder.raw_channels == 8


def test_known_only_refresh_populates_both_agent_memories():
    system = _system().eval()
    dataset = _KnownDataset()
    learned_prototypes = system.agents.prototype.prototypes.detach().clone()
    logits_before = system.local(dataset.x[:5])["prototype"].class_logits.detach()
    report = refresh_enrollment(
        system, dataset, torch.device("cpu"), batch_size=6)
    waveform = system.agents.waveform
    prototype = system.agents.prototype
    assert bool(waveform.enrollment_ready)
    assert bool(prototype.enrollment_ready)
    assert report["num_prototype_views"] == 5.0
    assert torch.allclose(
        waveform.enrollment_prototypes.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(
        prototype.view_prototypes.norm(dim=2), torch.ones(5, 3), atol=1e-5)
    assert torch.allclose(
        prototype.enrollment_prototypes.norm(dim=1), torch.ones(3), atol=1e-5)
    # Enrollment changes only private distance/tail memory.  It must not swap
    # the trained local classifier for a different empirical classifier.
    assert torch.equal(prototype.prototypes.detach(), learned_prototypes)
    logits_after = system.local(dataset.x[:5])["prototype"].class_logits.detach()
    assert torch.allclose(logits_after, logits_before, atol=1e-6)
    assert torch.all(waveform.tail_scale >= 1e-3)
    assert torch.all(prototype.tail_scale >= 1e-3)
    assert torch.all(prototype.view_tail_scale >= 1e-3)


def test_open_heads_are_bounded_residuals_not_replacement_scores():
    system = _system().eval()
    dataset = _KnownDataset()
    refresh_enrollment(system, dataset, torch.device("cpu"), batch_size=6)
    x = dataset.x[:5]
    before = system.local(x)
    with torch.no_grad():
        system.agents.waveform.open_head[-1].bias.fill_(1000.0)
        system.agents.prototype.open_head[-1].bias.fill_(-1000.0)
    after = system.local(x)
    # Each head can contribute at most +/-0.5 around the intrinsic evidence.
    for role in ("waveform", "prototype"):
        change = (after[role].unknown_score - before[role].unknown_score).abs()
        assert torch.all(change <= 0.500001)
