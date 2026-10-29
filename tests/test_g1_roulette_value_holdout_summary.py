import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_holdout_summary import assess


def test_cross_error_and_paired_difference_allow_negative_values():
    candidates={'original':{'a':1.,'b':0.},'offset':{'a':1.,'b':-.5},
                'affine':{'a':2.,'b':-1.},'constant':{'a':0.,'b':0.}}
    result=assess([{'clip':'a','frame':1,'value':1.,'targets':[0.,2.]}],candidates)
    assert result['cross_error_means']['original']==-1.
    assert result['differences_vs_original']['offset']['mean']==pytest.approx(.25)
    assert result['cross_error_means']['affine']==-1.


def test_invalid_candidate_is_rejected():
    with pytest.raises(ValueError):assess([{'value':1.,'targets':[0.,2.]}],{})
