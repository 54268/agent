import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage9.agents.identity_investigator import SignalIdentityInvestigator  # noqa: E402
from maros_stage9.agents.impairment_investigator import HardwareImpairmentInvestigator  # noqa: E402
from maros_stage9.contracts import CapabilityManifest, RequestSuggestion  # noqa: E402
from maros_stage9.policies.tool_policy import ToolPolicy, counterfactual_utility  # noqa: E402
from maros_stage9.splits import ProvenanceSubset, assert_no_formal_unknown  # noqa: E402
from maros_stage9.tool_registry import EvidenceTool, ToolRegistry  # noqa: E402
from maros_stage9.tools.encoders import EncoderTool  # noqa: E402


def test_agent_cannot_call_unregistered_tool():
    registry = ToolRegistry(CapabilityManifest("identity", frozenset({"raw"})))
    registry.register("raw", EncoderTool("raw", "identity", 3, 8, 4))
    with pytest.raises(PermissionError):
        registry.execute("robust_spectrum", torch.randn(2, 2, 256))
    with pytest.raises(PermissionError):
        registry.register("complex", EncoderTool("complex", "identity", 3, 8, 4))
    with pytest.raises(ValueError):
        EncoderTool("raw", "impairment", 3, 8, 4)


def test_tool_is_not_agent():
    tool = EncoderTool("raw", "identity", 3, 8, 4)
    assert isinstance(tool, EvidenceTool)
    assert not hasattr(tool, "decide")
    assert not hasattr(tool, "tool_policy")


def test_no_dataset_name_in_policy():
    source = inspect.getsource(ToolPolicy).lower()
    assert "oracle" not in source and "wisig" not in source


def test_tool_policy_changes_across_samples():
    torch.manual_seed(11)
    policy = ToolPolicy("identity", ("raw", "complex"), width=4).eval()
    iq = torch.stack([torch.zeros(2, 256), torch.randn(2, 256)])
    with torch.no_grad():
        output = policy(iq)
    assert not torch.allclose(output[0], output[1])


def test_only_selected_tool_runs_and_agent_can_abstain(monkeypatch):
    agent = SignalIdentityInvestigator(3, 8, 4).eval()
    calls = {name: 0 for name in agent.tool_names}
    for name, tool in agent.registry.tools.items():
        original = tool.forward

        def counted(iq, *, name=name, original=original):
            calls[name] += 1
            return original(iq)

        monkeypatch.setattr(tool, "forward", counted)
    with torch.no_grad():
        final = agent.tool_policy.scout[-1]
        final.weight.zero_()
        final.bias.fill_(-20)
        final.bias[0] = 20  # raw only
        packet, _ = agent.decide(torch.randn(2, 2, 256))
    assert packet.used_tools == (("raw",), ("raw",))
    assert calls == {"raw": 1, "complex": 0, "envelope": 0, "difference": 0}
    with torch.no_grad():
        final.bias.fill_(-20)
        final.bias[-1] = 20  # STOP_LOCAL_ANALYSIS
        packet, _ = agent.decide(torch.randn(2, 2, 256))
    assert bool(packet.abstain.all())
    assert bool(packet.request_suggestion.eq(int(RequestSuggestion.ABSTAIN)).all())


def test_agent_can_suggest_query_without_sending_one():
    agent = HardwareImpairmentInvestigator(3, 8, 4).eval()
    with torch.no_grad():
        final = agent.tool_policy.scout[-1]
        final.weight.zero_()
        final.bias.fill_(-20)
        final.bias[0] = 20
        for tool in agent.registry.tools.values():
            tool.classifier.weight.zero_()
            tool.classifier.bias.zero_()
        packet, _ = agent.decide(torch.randn(2, 2, 256))
    assert bool(packet.abstain.all())
    assert bool(packet.request_suggestion.eq(int(RequestSuggestion.ASK_IDENTITY)).all())


@pytest.mark.parametrize("agent_type", [SignalIdentityInvestigator,
                                         HardwareImpairmentInvestigator])
def test_class_reindex_invariance(agent_type):
    torch.manual_seed(17)
    agent = agent_type(4, 8, 4).eval()
    changed = copy.deepcopy(agent)
    permutation = torch.tensor([2, 0, 3, 1])
    changed.reindex_classes(permutation)
    iq = torch.randn(3, 2, 256)
    with torch.no_grad():
        original = agent.all_tool_logits(iq)
        reindexed = changed.all_tool_logits(iq)
        original_packet, original_private = agent.decide(iq)
        new_packet, new_private = changed.decide(iq)
    expected = torch.empty_like(original)
    expected[:, :, permutation] = original
    assert torch.allclose(reindexed, expected, atol=1e-5)
    expected_private = torch.empty_like(original_private.logits)
    expected_private[:, permutation] = original_private.logits
    assert torch.allclose(new_private.logits, expected_private, atol=1e-5)
    assert torch.equal(new_packet.candidate_a,
                       permutation[original_packet.candidate_a])


def test_counterfactual_utility_has_per_sample_choice():
    logits = torch.tensor([
        [[5.0, -1.0], [-1.0, 5.0]],
        [[-1.0, 5.0], [5.0, -1.0]],
    ])
    labels = torch.tensor([0, 0])
    actions = ((0,), (1,), (0, 1), ())
    utility = counterfactual_utility(logits, labels, actions, 0.03)
    assert utility.argmax(1).tolist() == [0, 1]


class TinyFormalDataset(Dataset):
    y = np.asarray([-1, -1], dtype=np.int64)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.randn(2, 256), torch.tensor(-1)


def test_formal_unknown_lock():
    formal = ProvenanceSubset(
        TinyFormalDataset(), [0, 1], source_split="test_unknown",
        purpose="formal_unknown", force_unknown=True, formal_unknown=True)
    with pytest.raises(RuntimeError, match="formal unknown"):
        assert_no_formal_unknown(formal)

