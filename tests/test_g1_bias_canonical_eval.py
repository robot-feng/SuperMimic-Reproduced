import sys
from pathlib import Path

import torch
import pytest
import copy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_bias_canonical_eval import canonicalize
from g1_bias_canonical_summary import summarize


def test_canonicalization_preserves_joint_target_and_gauge():
    g=torch.Generator().manual_seed(3)
    action=torch.randn(5,29,generator=g);xi=torch.randn(5,33,generator=g)*.01
    scale=torch.rand(29,generator=g)+.1;delta=torch.randn(5,29,generator=g)*.005
    shifted=xi.clone();shifted[:,1:30]+=delta
    shifted_action=action+delta/scale
    orig,ctx=canonicalize(action,xi,scale,1.)
    new,new_ctx=canonicalize(shifted_action,shifted,scale,1.)
    assert torch.allclose(orig,new,atol=1e-6)
    assert torch.equal(ctx,new_ctx)
    assert torch.allclose(scale*orig-ctx[:,1:30],scale*action-xi[:,1:30],atol=1e-6)


def test_summary_checks_pair_and_identity():
    results={h:{'0.0':{'per_window_nse':[1.,2.]},
                '1.0':{'per_window_nse':[.5,3.], 'paired_nse_delta_mean':.25,'lower_nse_count':1}}
             for h in ('1','8','16','32')}
    e={'seed':1,'window_sha256':'win','checkpoint_sha256':'ck','count':2,'script_sha256':'script',
       'scale_probe_sha256':'scale','results':results}
    r={'seed':1,'windows_sha256':'win','checkpoint_sha256':'ck','batch_with_replacement':2,'counted_simulator_transitions':3}
    e2=copy.deepcopy(e);r2=copy.deepcopy(r);e2['seed']=r2['seed']=2
    assert summarize([e,e2],[r,r2])['summary']['8']['mean_delta']==pytest.approx(.25)
    e2['results']['8']['1.0']['lower_nse_count']=2
    with pytest.raises(ValueError,match='count mismatch'):summarize([e,e2],[r,r2])
