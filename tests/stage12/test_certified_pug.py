import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from maros_stage12.certified_pug import (PUGTool, apply_increment,
                                         certificate, iq_candidates)
from maros_stage12.experiment import collect
from maros_stage7.splits import ProvenanceSubset


def test_incremental_review_never_replaces_stronger_stage5_risk():
    base = np.array([0.2, 0.8, 0.95])
    tool = np.array([0.9, 0.1, 1.0])
    changed = apply_increment(base, tool, 0.2)
    assert np.all(changed >= base)
    assert changed[1] == base[1]
    assert np.all(changed <= 1)


def test_iq_pug_is_new_consumable_signal_not_copied_report():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(18, 2, 256)).astype(np.float32)
    y = np.repeat(np.arange(3), 6)
    public = {"a_margin": np.full(18, 0.4),
              "b_margin": np.full(18, 0.3),
              "disagree": np.zeros(18, dtype=bool)}
    train = {"iq": x, "labels": y, "report": public,
             "u_proto": np.linspace(0, 1, 18)}
    generated = iq_candidates(train, 2, (0.25, 0.45), 42)
    assert generated["iq"].shape == (12, 2, 256)
    assert np.isfinite(generated["iq"]).all()
    assert np.any(np.abs(generated["iq"] - x[generated["source_index"]]) > 0)
    assert np.all(y[generated["source_index"]] != y[generated["rival_index"]])


def test_certificate_rejects_report_copy_and_known_label():
    generated = {"iq": np.zeros((2, 2, 256)),
                 "source_index": np.array([0, 1])}
    pseudo = {"labels": np.array([-1, -1]), "fresh_forward": False}
    with pytest.raises(ValueError, match="copied"):
        certificate(generated, pseudo, {}, {})
    pseudo["fresh_forward"] = True
    pseudo["labels"][0] = 0
    with pytest.raises(ValueError, match="Unknown"):
        certificate(generated, pseudo, {}, {})


def test_pug_tool_abstains_when_certified_pool_too_small():
    tool = PUGTool().fit({"labels": np.zeros(10)},
                         {"labels": np.full(10, -1)},
                         np.ones(10, dtype=bool))
    assert not tool.enabled


def test_pseudo_iq_really_reenters_both_agents():
    class Agent:
        def __init__(self, geometry=False):
            self.calls = 0
            self.geometry = geometry

        def __call__(self, x):
            self.calls += 1
            value = x[:, 0].mean(dim=1)
            logits = torch.stack([value, -value], dim=1)
            evidence = {"d1": value.abs()}
            if self.geometry:
                evidence["view_weights"] = torch.ones(len(x), 1)
                evidence["view_distances"] = torch.ones(len(x), 1, 2)
            return type("AgentOutput", (), {"class_logits": logits,
                                             "evidence": evidence})()

    class Model:
        def __init__(self):
            self.identity = Agent()
            self.geometry = Agent(geometry=True)

        def eval(self):
            return self

    class EVT:
        def score(self, p):
            return np.zeros(len(p))

    model = Model()
    x = torch.stack([torch.ones(2, 256), -torch.ones(2, 256)])
    dataset = TensorDataset(x, torch.full((2,), -1))
    out = collect(model, EVT(), dataset, torch.device("cpu"), 2, pseudo=True)
    assert out["fresh_forward"]
    assert model.identity.calls == 1 and model.geometry.calls == 1
    assert out["a_pred"].tolist() == [0, 1]
    assert out["b_pred"].tolist() == [0, 1]


def test_formal_unknown_cannot_enter_through_subset_wrapper():
    class Source(Dataset):
        y = np.array([-1, -1])

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.zeros(2, 256), torch.tensor(-1)

    formal = ProvenanceSubset(Source(), [0, 1], source_split="formal_unknown",
                              purpose="formal_unknown", formal_unknown=True,
                              force_unknown=True)
    with pytest.raises(RuntimeError, match="formal unknown"):
        collect(None, None, Subset(formal, [0]), torch.device("cpu"), 1)
