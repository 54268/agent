"""Gate-locked boundary for the future Open-Space Consistency Auditor."""


class OpenSpaceConsistencyAuditor:
    def __init__(self, *, known_safety_passed: bool = False):
        if not known_safety_passed:
            raise RuntimeError(
                "Consistency Auditor is locked until Known perturbation safety passes")
        raise NotImplementedError(
            "Known-safe perturbations have not yet been registered")

