import sys
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage10.blackboard import SharedBlackboard  # noqa: E402
from maros_stage10.system import Stage10System  # noqa: E402


def test_blackboard_has_no_private_state():
    names = {field.name.lower() for field in fields(SharedBlackboard)}
    assert not any(token in name for name in names
                   for token in ("embedding", "hidden", "logits", "prototype_table",
                                 "private", "memory", "raw_iq"))


def test_runtime_has_no_central_winner_router():
    assert not hasattr(Stage10System, "router")
    assert not hasattr(Stage10System, "winner")

