"""Simple leakage-safe knowledge and case memory for Stage 15."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_GLOBAL_KNOWLEDGE = (
    "RF open-set decisions should not trust a single confidence score. "
    "High classifier confidence can coexist with receiver/domain shift. "
    "Prototype, geometry, OpenMax, boundary and multi-view consistency are "
    "complementary evidence sources. Conflicting evidence should trigger "
    "additional verification before accepting a known identity."
)


@dataclass
class Case:
    public_summary: dict[str, Any]
    revealed_tools: dict[str, Any]
    decision: str
    candidate_class: int | None
    note: str = ""


@dataclass
class CaseMemory:
    max_cases: int = 256
    cases: list[Case] = field(default_factory=list)

    def add(self, case: Case) -> None:
        self.cases.append(case)
        if len(self.cases) > self.max_cases:
            del self.cases[:-self.max_cases]

    def retrieve(self, public_summary: dict[str, Any], k: int = 3) -> list[dict]:
        """Retrieve by class agreement and confidence proximity.

        This intentionally avoids labels, source provenance and formal-unknown flags.
        """
        if not self.cases or k <= 0:
            return []
        target_class = public_summary.get("identity_top1")
        target_prob = float(public_summary.get("identity_top1_probability", 0.0))
        scored = []
        for case in self.cases:
            c_class = case.public_summary.get("identity_top1")
            c_prob = float(case.public_summary.get(
                "identity_top1_probability", 0.0))
            score = abs(target_prob - c_prob)
            if c_class != target_class:
                score += 1.0
            scored.append((score, case))
        scored.sort(key=lambda row: row[0])
        return [{
            "public_summary": case.public_summary,
            "revealed_tools": case.revealed_tools,
            "decision": case.decision,
            "candidate_class": case.candidate_class,
            "note": case.note,
        } for _, case in scored[:k]]
