"""Fixed-size, class-rank based action spaces for the three Stage-14 roles."""
from __future__ import annotations

from enum import IntEnum
from typing import Type


AGENT_IDS = ("identity", "geometry", "open_set")


class AAction(IntEnum):
    WAIT = 0
    PROPOSE_TOP1 = 1
    PROPOSE_TOP2 = 2
    ASK_B = 3
    ASK_C = 4
    SEND_EVIDENCE_B = 5
    SEND_EVIDENCE_C = 6
    RECHECK = 7
    ABSTAIN = 8


class BAction(IntEnum):
    WAIT = 0
    SUPPORT = 1
    CHALLENGE_TOP1 = 2
    CHALLENGE_TOP2 = 3
    ASK_A = 4
    ASK_C = 5
    SEND_EVIDENCE_A = 6
    SEND_EVIDENCE_C = 7
    RECHECK_VIEW = 8
    ABSTAIN = 9


class CAction(IntEnum):
    WAIT = 0
    QUERY_A = 1
    QUERY_B = 2
    CHECK_ID_PROTOTYPE = 3
    CHECK_GEO_PROTOTYPE = 4
    CHECK_OPENMAX = 5
    CHECK_BOUNDARY = 6
    ACCEPT_CURRENT_KNOWN = 7
    REJECT_UNKNOWN = 8


ACTION_ENUMS: dict[str, Type[IntEnum]] = {
    "identity": AAction,
    "geometry": BAction,
    "open_set": CAction,
}


def action_count(agent: str) -> int:
    return len(ACTION_ENUMS[agent])


def action_name(agent: str, action: int) -> str:
    return ACTION_ENUMS[agent](int(action)).name
