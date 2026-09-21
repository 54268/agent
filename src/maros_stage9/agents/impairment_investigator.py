"""Hardware-impairment goal without a raw temporal encoder."""
from .base import Investigator
from ..tools.observations import IMPAIRMENT_TOOLS


class HardwareImpairmentInvestigator(Investigator):
    def __init__(self, num_classes: int, state_dim: int = 48,
                 width: int = 24) -> None:
        super().__init__("impairment", IMPAIRMENT_TOOLS,
                         num_classes, state_dim, width)

