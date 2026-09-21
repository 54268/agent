"""Signal-identity goal with raw/complex/envelope/difference capabilities."""
from .base import Investigator
from ..tools.observations import IDENTITY_TOOLS


class SignalIdentityInvestigator(Investigator):
    def __init__(self, num_classes: int, state_dim: int = 48,
                 width: int = 24) -> None:
        super().__init__("identity", IDENTITY_TOOLS, num_classes, state_dim, width)

