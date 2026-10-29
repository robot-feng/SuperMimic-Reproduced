"""Inference-only action/bias canonicalization on recorded G1 windows."""
import argparse
import gc
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
import torch
from g1_directional_probe import config_from_checkpoint
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_tail_policy_switch_probe import sha
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)
ALPHAS=(0.,.5,1.)


def canonicalize(action,xi,scale,alpha):
    """Move alpha of encoder bias into raw action, preserving joint targets."""
    if action.shape[-1]!=29 or xi.shape[-1]!=33 or scale.shape!=(29,) or (scale<=0).any():
        raise ValueError('invalid action, context, or action scales')
    context=xi.clone();bias=xi[...,1:30]
    context[...,1:30]=(1-alpha)*bias
    return action-alpha*bias/scale,context


@torch.no_grad()
def evaluate(wm,data,scale):
    s,a,xi=data['states'],data['a_exec'],data['xi'];n=len(xi)
    if len(a)!=32 or len(s)!=33 or a.shape[1]!=n or s.shape[1]!=n:raise ValueError('invalid window')
    result={str(t):[] for t in (1,8,16,32)}
    for idx in torch.arange(n).split(16):
        size=len(idx);st=s[0,idx].expand(len(ALPHAS),size,-1).clone()
        for t in range(32):
            ac=[];cx=[]
            for alpha in ALPHAS:
                aa,xx=canonicalize(a[t,idx],xi[idx],scale,alpha);ac.append(aa);cx.append(xx)
            states=wm.step(st.reshape(-1,st.shape[-1]),torch.stack(ac).reshape(-1,29),torch.stack(cx).reshape(-1,33))
            st=states.reshape(len(ALPHAS),size,-1)
            if str(t+1) in result:
                target=s[t+1,idx].expand_as(st)
                error=wm.spec.normalized_sq_error(st.reshape(-1,st.shape[-1]),target.reshape(-1,target.shape[-1])).reshape(len(ALPHAS),size)
                if not torch.isfinite(error).all():raise ValueError('nonfinite error')
                result[str(t+1)].append(error)
    out={}
    for h,parts in result.items():
        err=torch.cat(parts,dim=1)
        out[h]={str(alpha):{'mean_nse':float(err[i].mean()),'median_nse':float(err[i].median()),
             'paired_nse_delta_mean':float((err[i]-err[0]).mean()),'lower_nse_count':int((err[i]<err[0]).sum()),
             'per_window_nse':err[i].tolist()} for i,alpha in enumerate(ALPHAS)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--scale-probe',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing to overwrite')
    report=json.loads(a.report.read_text(encoding='utf8'));probe=json.loads(a.scale_probe.read_text(encoding='utf8'))
    if sha(a.windows)!=report['windows_sha256'] or sha(a.checkpoint)!=report['checkpoint_sha256'] or report['horizon']!=32 or probe['checkpoint_sha256']!=report['checkpoint_sha256']:
        raise ValueError('input identity mismatch')
    scale=torch.as_tensor(probe['action_scale'])
    if scale.ndim!=2 or scale.shape[1]!=29 or not torch.allclose(scale,scale[0].expand_as(scale),atol=0,rtol=0):
        raise ValueError('nonuniform action scale')
    scale=scale[0]
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    _,cfg=config_from_checkpoint(ck);wm=WorldModel(cfg,AnalyticHeads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
    del ck;gc.collect();data=torch.load(a.windows,map_location='cpu',weights_only=True)
    out={'seed':report['seed'],'window_sha256':report['windows_sha256'],'checkpoint_sha256':report['checkpoint_sha256'],
         'scale_probe_sha256':sha(a.scale_probe),'script_sha256':sha(Path(__file__)),'count':len(data['xi']),
         'results':evaluate(wm,data,scale),
         'notes':'Fixed recorded executed actions and physical context, replace ONLY world-model input with alpha canonicalization. No action clipping; action scale and target order were checked in paired physical probe. Recorded windows sampled with replacement and correlated, with survival selection; no retraining or policy evaluation.'}
    if sha(a.windows)!=report['windows_sha256'] or sha(a.checkpoint)!=report['checkpoint_sha256']:raise ValueError('input changed')
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'seed':out['seed'],'delta':{h:out['results'][h]['1.0']['paired_nse_delta_mean'] for h in out['results']}}))


if __name__=='__main__':main()
