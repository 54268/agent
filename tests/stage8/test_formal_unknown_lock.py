import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maros_stage8.splits import ProvenanceSubset, assert_no_formal_unknown  # noqa: E402


class TinyDataset(Dataset):
    def __init__(self):
        self.y = np.asarray([-1, -1], dtype=np.int64)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.randn(2, 32), torch.tensor(-1)


def test_formal_unknown_remains_locked_from_stage8_training_paths():
    formal = ProvenanceSubset(
        TinyDataset(), [0, 1], source_split="test_unknown",
        purpose="formal_unknown", force_unknown=True, formal_unknown=True)
    with pytest.raises(RuntimeError, match="formal unknown"):
        assert_no_formal_unknown(formal)

