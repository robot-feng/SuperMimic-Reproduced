import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'analysis'))
from g1_bias_retrain import replace_bias
from g1_bias_retrain_summary import summarize


def test_replacement_preserves_per_step_physical_target():
    g=torch.Generator().manual_seed(19)
    actions=torch.randn(8,4,29,generator=g,requires_grad=True)
    xi=torch.randn(4,33,generator=g)*.01;new=torch.randn(4,29,generator=g)*.01
    scale=torch.rand(29,generator=g)+.1
    shifted,context=replace_bias(actions,xi,new,scale)
    before=actions*scale-xi[None,:,1:30]
    after=shifted*scale-context[None,:,1:30]
    assert torch.allclose(before,after,atol=3e-7)
    assert torch.equal(context[:,:1],xi[:,:1]) and torch.equal(context[:,30:],xi[:,30:])
    assert torch.equal(torch.autograd.grad(shifted.sum(),actions)[0],torch.ones_like(actions))


def test_invalid_zero_scale_is_rejected():
    with pytest.raises(ValueError):replace_bias(torch.zeros(8,4,29),torch.zeros(4,33),torch.zeros(4,29),torch.zeros(29))


def test_summary_rejects_missing_and_mismatched_models():
    docs=[]
    for seed in (1,2):
        for mode in ('raw','canonical','augment'):
            vals={str(e):{k:{h:{'mean':float(seed),'per_window_nse':[float(seed)]*128} for h in ('1','8','16','32')}
                          for k in ('raw','canonical')} for e in (3,4)}
            docs.append({'seed':seed,'mode':mode,'before':vals,'after':vals,'checkpoint_sha256':'c','scale_probe_sha256':'s',
                         'script_sha256':'p','source_files':{},'train_seeds':[5,6],'eval_seeds':[3,4],
                         'updates':60,'batch_size':16,'horizon':8,'optimizer':'Adam','lr':2e-5,'training_model_steps':7680})
    assert summarize(docs)['summary']['8']['canonical_vs_raw']['mean_delta']==0
    with pytest.raises(ValueError,match='incomplete'):summarize(docs[:-1])
    docs[-1]['lr']=1e-3
    with pytest.raises(ValueError,match='protocol'):summarize(docs)
