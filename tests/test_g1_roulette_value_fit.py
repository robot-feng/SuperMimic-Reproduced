import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_roulette_value_fit import fit


def test_offset_and_affine_regularized_toward_original_slope():
    pairs=[{'value':float(x),'targets':[2.*x+1,2.*x+1]} for x in (0,1,2)]
    result=fit(pairs,.1)
    assert result['offset']['b']==2.
    expected=(4/3+.1)/(2/3+.1)
    assert result['affine']['a']==pytest.approx(expected)
    assert result['affine']['b']==pytest.approx(3-expected)
    assert result['constant']=={'a':0.,'b':3.}


def test_fit_rejects_nonfinite():
    with pytest.raises(ValueError):fit([{'value':0.,'targets':[1.,1.]},{'value':float('nan'),'targets':[0.,0.]}])
