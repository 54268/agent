import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.splits import build_inner_folds  # noqa: E402


def test_inner_lco_holds_every_class_exactly_once():
    classes = tuple(range(12))
    folds = build_inner_folds(classes, num_folds=4, seed=2026)
    held = [value for fold in folds for value in fold.heldout_classes]
    assert sorted(held) == list(classes)
    for fold in folds:
        assert set(fold.support_classes).isdisjoint(fold.heldout_classes)
        assert set(fold.support_classes) | set(fold.heldout_classes) == set(classes)
