import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_probe import geometric_lengths


def test_geometric_lengths_match_survival_probability():
    x=geometric_lengths(300000,.9,torch.Generator().manual_seed(17))
    assert int(x.min())>=1
    assert float((x>4).float().mean())==pytest.approx(.9**4,abs=.004)
    assert float(x.float().mean())==pytest.approx(10.,abs=.08)


def test_invalid_survival_probability_rejected():
    with pytest.raises(ValueError):geometric_lengths(1,1.,torch.Generator())
