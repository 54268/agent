"""Gate-locked placeholder for the future Value-of-Information service."""


class ValueOfInformationCoordinator:
    """Unavailable until forced consultation passes G2 on both datasets."""

    def __init__(self, *, g2_passed: bool = False):
        if not g2_passed:
            raise RuntimeError(
                "VOI Coordinator is locked until Stage-8 forced-consultation G2 passes")
        raise NotImplementedError(
            "G2 has not yet registered a Coordinator architecture")

