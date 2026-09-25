"""Role-private memory and shared domain knowledge for Stage 15."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_GLOBAL_KNOWLEDGE = (
    "Open-set RF identification should not trust one confidence score. "
    "Receiver/domain shift can preserve high softmax confidence while changing "
    "class-conditional geometry. Identity evidence, geometry evidence, OpenMax, "
    "boundary risk and multi-view consistency are complementary. Agents should "
    "communicate only evidence they actually observed and should seek additional "
    "evidence when their local views conflict."
)


@dataclass
class Case:
    """One role's own past experience; never contains ground truth."""

    local_summary: dict[str, Any]
    private_tools: dict[str, Any]
    observed_messages: list[dict[str, Any]]
    final_decision: str
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

    @staticmethod
    def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        keys = set(left) & set(right)
        if not keys:
            return 10.0
        distances = []
        for key in keys:
            a, b = left[key], right[key]
            if isinstance(a, (int, float, np.integer, np.floating)) and                     isinstance(b, (int, float, np.integer, np.floating)):
                distances.append(abs(float(a) - float(b)))
            else:
                distances.append(0.0 if a == b else 1.0)
        return float(np.mean(distances)) if distances else 10.0

    def retrieve(self, local_summary: dict[str, Any],
                 k: int = 3) -> list[dict[str, Any]]:
        if not self.cases or k <= 0:
            return []
        ranked = sorted(
            self.cases,
            key=lambda case: self._distance(local_summary, case.local_summary))
        return [{
            "local_summary": case.local_summary,
            "private_tools": case.private_tools,
            "observed_messages": case.observed_messages[-6:],
            "final_decision": case.final_decision,
            "candidate_class": case.candidate_class,
            "note": case.note,
        } for case in ranked[:k]]
