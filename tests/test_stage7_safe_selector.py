import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage7.nested_g0 import SelectorRowBatch  # noqa: E402
from maros_stage7.safe_selector import fit_safe_public_selector  # noqa: E402


def _batch(index: int, shift: float = 0.0) -> SelectorRowBatch:
    rng = np.random.default_rng(100 + index)
    x = rng.normal(size=(80, 4))
    gain = x[:, 0] + shift
    losses = np.c_[np.ones(80), np.ones(80) - 0.3 * gain]
    return SelectorRowBatch(
        producer_inner_index=index,
        sample_keys=tuple(f"p{index}:{row}" for row in range(80)),
        public_features=x.astype(np.float32),
        action_losses=losses.astype(np.float32))


def test_safe_selector_uses_public_oof_threshold_and_anchor_fallback():
    selector = fit_safe_public_selector([_batch(0), _batch(1), _batch(2)])
    audit = selector.audit
    assert audit.public_only and not audit.outer_test_used
    assert audit.fit_oof_gain > 0
    assert 0.05 <= audit.fit_oof_switch_rate <= 0.5
    assert set(selector.choose(np.array([[-3, 0, 0, 0], [3, 0, 0, 0]]))) <= {0, 1}


def test_safe_selector_rejects_duplicate_producers_and_bad_grid():
    with pytest.raises(ValueError, match="unique"):
        fit_safe_public_selector([_batch(0), _batch(0)])
    with pytest.raises(ValueError, match="probability_grid"):
        fit_safe_public_selector([_batch(0), _batch(1)],
                                 probability_grid=(0.8, 0.7))
