"""Equal-update offline G1 model refit with physical-target symmetry controls."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
import torch
from g1_directional_probe import config_from_checkpoint
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_mature_wm_fit import select_crop
from g1_tail_policy_switch_probe import sha
from g1_wm_horizon_fit import state_window_loss
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def replace_bias(actions,xi,new_bias,scale):
    """Keep position target invariant; actions shape (T,N,29), xi (N,33)."""
    if actions.ndim!=3 or actions.shape[-1]!=29 or xi.shape!=(actions.shape[1],33) or new_bias.shape!=(actions.shape[1],29) or scale.shape!=(29,) or (scale<=0).any():
        raise ValueError('incompatible physical action representation')
    out=xi.clone();out[:,1:30]=new_bias
    return actions+(new_bias-xi[:,1:30])[None]/scale, out


@torch.no_grad()
def evaluate(wm,data,scale,canonical):
    states,actions,xi=data['states'],data['a_exec'],data['xi']
    if len(actions)!=32 or len(states)!=33 or len(xi)!=states.shape[1]:raise ValueError('expected H32 windows')
    if canonical:actions,xi=replace_bias(actions,xi,torch.zeros_like(xi[:,1:30]),scale)
    errors={str(t):[] for t in (1,8,16,32)}
    for idx in torch.arange(len(xi)).split(16):
        pred=states[0,idx]
        for t in range(32):
            pred=wm.step(pred,actions[t,idx],xi[idx])
            if str(t+1) in errors:
                err=wm.spec.normalized_sq_error(pred,states[t+1,idx])
                if not torch.isfinite(err).all():raise ValueError('nonfinite eval')
                errors[str(t+1)].append(err)
    return {t:{'mean':float(e.mean()),'median':float(e.median()),'p90':float(torch.quantile(e,.9)),
               'per_window_nse':e.tolist()} for t,parts in errors.items() for e in [torch.cat(parts)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('raw','canonical','augment'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--weights-output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--scale-probe',type=Path,default=Path('docs/analysis/g1_encoder_bias_symmetry_8600.json'))
    p.add_argument('--eval-seeds',nargs='+',type=int,default=(8620,8621,8622))
    a=p.parse_args()
    if a.output.resolve()==a.weights_output.resolve() or a.output.exists() or a.weights_output.exists():raise FileExistsError('refusing to overwrite')
    train_seeds=(5801,5802);eval_seeds=tuple(a.eval_seeds);reports={};data={};sources={}
    if len(set(eval_seeds))!=len(eval_seeds) or not eval_seeds or set(eval_seeds)&set(train_seeds):
        raise ValueError('evaluation seeds must be unique and disjoint from fitting seeds')
    for seed in train_seeds+eval_seeds:
        stem=f'g1_paired_h32_seed{seed}' if seed in train_seeds else f'g1_bias_holdout_windows_{seed}'
        report_path=Path('docs/analysis')/(stem+'.json');window_path=Path('docs/analysis')/(stem+'.pt')
        report=json.loads(report_path.read_text(encoding='utf8'))
        if report['seed']!=seed or report['horizon']!=32 or sha(window_path)!=report['windows_sha256']:
            raise ValueError('replay identity mismatch')
        reports[seed]=report;data[seed]=torch.load(window_path,map_location='cpu',weights_only=True)
        sources[seed]={'report':str(report_path),'report_sha256':sha(report_path),'windows':str(window_path),'window_sha256':sha(window_path)}
    checkpoint_hash=sha(a.checkpoint);probe=json.loads(a.scale_probe.read_text(encoding='utf8'))
    if any(r['checkpoint_sha256']!=checkpoint_hash for r in reports.values()) or probe['checkpoint_sha256']!=checkpoint_hash:
        raise ValueError('checkpoint mismatch')
    scales=torch.as_tensor(probe['action_scale'])
    if scales.ndim!=2 or scales.shape[1]!=29 or not torch.allclose(scales,scales[0].expand_as(scales),atol=0,rtol=0):
        raise ValueError('nonuniform action scales')
    scale=scales[0];ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    if ck['iteration']!=2000:raise ValueError('unexpected checkpoint')
    _,cfg=config_from_checkpoint(ck);wm=WorldModel(cfg,AnalyticHeads,'cpu');wm.load_state_dict(ck['wm']);wm.eval()
    del ck;gc.collect()
    train={k:torch.cat([data[s][k] for s in train_seeds],dim=1 if k in ('states','a_exec') else 0)
           for k in ('states','a_exec','xi')}
    generator=torch.Generator().manual_seed(a.seed)
    indices=torch.randint(len(train['xi']),(60,16),generator=generator)
    offsets=torch.randint(25,(60,16),generator=generator)
    augmentation=torch.Generator().manual_seed(a.seed+100000)
    opt=torch.optim.Adam(wm.dynamics.parameters(),lr=2e-5)
    def eval_all():
        return {str(s):{kind:evaluate(wm,data[s],scale,kind=='canonical') for kind in ('raw','canonical')} for s in eval_seeds}
    before=eval_all();started=time.perf_counter();losses=[]
    for i,ix in enumerate(indices):
        states,actions=select_crop(train['states'],train['a_exec'],ix,offsets[i],8,torch.arange)
        xi=train['xi'][ix]
        if a.mode=='canonical':actions,xi=replace_bias(actions,xi,torch.zeros_like(xi[:,1:30]),scale)
        elif a.mode=='augment':
            new_bias=.02*torch.rand(xi.shape[0],29,generator=augmentation)-.01
            actions,xi=replace_bias(actions,xi,new_bias,scale)
        loss=state_window_loss(wm,states,actions,xi,8)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(wm.dynamics.parameters(),1.);opt.step()
        losses.append(float(loss.detach()))
    seconds=time.perf_counter()-started;after=eval_all()
    for source in sources.values():
        if sha(Path(source['report']))!=source['report_sha256'] or sha(Path(source['windows']))!=source['window_sha256']:
            raise ValueError('training or evaluation input changed')
    if sha(a.checkpoint)!=checkpoint_hash:raise ValueError('checkpoint changed')
    with a.weights_output.open('xb') as f:torch.save(wm.dynamics.state_dict(),f)
    out={'mode':a.mode,'seed':a.seed,'checkpoint_sha256':checkpoint_hash,'scale_probe_sha256':sha(a.scale_probe),
         'script_sha256':sha(Path(__file__)),'source_files':sources,'train_seeds':train_seeds,'eval_seeds':eval_seeds,
         'updates':60,'batch_size':16,'horizon':8,'optimizer':'Adam cold start','lr':2e-5,
         'training_model_steps':60*16*8,'training_seconds':seconds,'losses':losses,'weights_sha256':sha(a.weights_output),
         'before':before,'after':after,
         'notes':'Same frozen checkpoint, replay windows, crop/minibatch seed, model steps and optimizer; only network input representation/augmentation differs. Canonical-trained model must use canonical inputs at inference. Augmentation samples a physically equivalent bias within training bounds, fixed over each window. State-only loss; no terminal head, actor or policy update. Fixed-action open-loop evaluation.'}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'mode':a.mode,'seed':a.seed,'after_raw_h8':{s:after[s]['raw']['8']['mean'] for s in after},'after_canonical_h8':{s:after[s]['canonical']['8']['mean'] for s in after},'training_seconds':seconds}))


if __name__=='__main__':main()
