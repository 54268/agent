"""G0 pair of autonomous local investigators; no central winner router."""
from __future__ import annotations

from torch import nn

from .agents.identity_investigator import SignalIdentityInvestigator
from .agents.impairment_investigator import HardwareImpairmentInvestigator


class Stage9G0System(nn.Module):
    def __init__(self, num_classes: int, state_dim: int = 48,
                 width: int = 24) -> None:
        super().__init__()
        self.identity = SignalIdentityInvestigator(num_classes, state_dim, width)
        self.impairment = HardwareImpairmentInvestigator(num_classes, state_dim, width)

    def agents(self):
        return (self.identity, self.impairment)

    def reindex_classes(self, old_to_new):
        for agent in self.agents():
            agent.reindex_classes(old_to_new)
