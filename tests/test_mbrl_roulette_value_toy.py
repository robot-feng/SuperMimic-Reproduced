import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from mbrl_roulette_value_toy import exact_value,fixed_bootstrap,rr_sample,run


def test_roulette_telescope_handles_terminal_before_cutoff():
    f=np.array([1,3,5]);t=np.array([5,2,5]);r=.01;g=.99;c=.8
    target=rr_sample(f,t,r,g,g,c)
    direct=c+np.minimum(f,t)*(r-(1-g)*c)-(f<=t)*g*c
    assert np.allclose(target,direct)
    assert target[0]==pytest.approx(r)
    assert fixed_bootstrap(np.array([1]),80,r,g,c)[0]==pytest.approx(r)


def test_exact_geometric_value_and_double_sample():
    result=run(128,samples=400000,p=.03,critic=.8)
    r=result['results'];truth=exact_value(.01,.99,.03)
    assert abs(r['roulette']['mean']-truth)<5*r['roulette']['mean_se']
    cross=r['cross_error_0.8'];assert abs(cross['mean']-cross['truth'])<5*cross['se']
    assert abs(r['fixed_bootstrap']['mean']-truth)>10*r['fixed_bootstrap']['mean_se']
