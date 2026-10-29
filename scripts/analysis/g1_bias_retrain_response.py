"""True G1 action responses versus symmetry-trained model refits, one step."""
import argparse
import gc
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
import torch
from g1_bias_canonical_eval import canonicalize
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_noisy_imagination_direction import frame_samples
from g1_tail_policy_switch_probe import sha
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-seed',type=int,default=8710);p.add_argument('--checkpoint',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    ckhash=sha(a.checkpoint);sources={};weights={}
    for mode in ('raw','canonical','augment'):
        base=Path(f'docs/analysis/g1_bias_retrain_confirm_{mode}_{a.fit_seed}')
        report=base.with_suffix('.json');weight=base.with_suffix('.pt')
        info=json.loads(report.read_text(encoding='utf8'))
        if info['mode']!=mode or info['seed']!=a.fit_seed or info['checkpoint_sha256']!=ckhash or info['weights_sha256']!=sha(weight):
            raise ValueError('trained weight identity mismatch')
        sources[mode]={'report':str(report),'report_sha256':sha(report),'weight':str(weight),'weight_sha256':sha(weight)}
        weights[mode]=torch.load(weight,map_location='cpu',weights_only=True)
    probe_path=Path('docs/analysis/g1_encoder_bias_symmetry_8600.json')
    probe=json.loads(probe_path.read_text(encoding='utf8'));scales=torch.tensor(probe['action_scale']);scale=scales[0]
    if probe['checkpoint_sha256']!=ckhash or scales.ndim!=2 or scales.shape[1]!=29 or not torch.equal(scales,scale.expand_as(scales)):
        raise ValueError('scale identity mismatch')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    pcfg,wcfg=config_from_checkpoint(ck);actor=Policy(pcfg);actor.load_state_dict(ck['policy']);actor.eval()
    wm=WorldModel(wcfg,AnalyticHeads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
    a.clips=['walk2_subject1','dance2_subject1','jumps1_subject2'];a.eval_clips=['walk1_subject1']
    a.frames_per_clip=4;a.eval_frames_per_clip=1;a.warmup=10;a.horizon=a.eval_horizon=1
    a.future=int(ck['config']['env']['future_frames']);a.history=int(ck['config']['policy_trainer']['history'])
    del ck;gc.collect();files=discover_npz(Path('data/npz/lafan1'),names=a.clips+a.eval_clips)
    if len(files)!=4:raise ValueError('missing clips')
    frames,_,labels,_=frame_samples(files,a,a.future);n=len(frames);env=None
    try:
        env,obs,bad,_,io,ctx,cmd,_,_=make_setup(files,frames,4,a.seed,a.warmup,a.history,a.future,actor,True)
        term=env.action_manager.get_term('joint_pos');robot=env.scene['robot'];ids=term.target_ids
        if term.noise_std!=0 or term.cfg.clip is not None or not torch.equal(ids,torch.arange(29)):
            raise ValueError('action assumptions violated')
        actual_scale=torch.as_tensor(term.scale)
        if actual_scale.ndim!=2 or not torch.allclose(actual_scale,scale.expand_as(actual_scale),atol=0,rtol=0):raise ValueError('scale changed')
        with torch.no_grad():
            state=io.read();obs0=obs['policy'].clone();goal=obs['goal'].clone();frame=cmd.frame.clone()
            bias=.02*torch.rand(n,29,generator=torch.Generator().manual_seed(a.seed+9800000))-.01
            robot.data.encoder_bias[:]=bias.repeat(4,1);xi=ctx.read()
            base_action=actor(obs0[:n],goal[:n]);gen=torch.Generator().manual_seed(a.seed+9900000)
            direction=torch.randn(n,29,generator=gen);direction=direction/direction.norm(dim=-1,keepdim=True)
            action=base_action.repeat(4,1);amplitudes=(.05,.2)
            for k,eps in enumerate(amplitudes):
                action[2*k*n:(2*k+1)*n]+=eps*direction;action[(2*k+1)*n:(2*k+2)*n]-=eps*direction
            original=wm.step(state,action,xi);predictions={'original':original}
            for mode in ('raw','canonical','augment'):
                wm.dynamics.load_state_dict(weights[mode],strict=True)
                if mode=='canonical':ac,xx=canonicalize(action,xi,scale,1.)
                else:ac,xx=action,xi
                predictions[mode]=wm.step(state,ac,xx)
                if mode=='raw':
                    ac,xx=canonicalize(action,xi,scale,1.)
                    predictions['raw_canonical']=wm.step(state,ac,xx)
            _,_,terminals,timeouts,_=env.step(action);after=io.read();executed=term.executed_raw
            if (executed-action).abs().max()>1e-6:raise ValueError('executed action mismatch')
            groups=[]
            for k,eps in enumerate(amplitudes):
                plus=slice(2*k*n,(2*k+1)*n);minus=slice((2*k+1)*n,(2*k+2)*n);base=slice(0,n)
                initial_error=torch.maximum(wm.spec.normalized_sq_error(state[plus],state[base]),wm.spec.normalized_sq_error(state[minus],state[base]))
                xi_error=torch.maximum((xi[plus]-xi[base]).abs().amax(-1),(xi[minus]-xi[base]).abs().amax(-1))
                obs_error=torch.maximum((obs0[plus]-obs0[base]).abs().amax(-1),(obs0[minus]-obs0[base]).abs().amax(-1))
                frame_error=(frame[plus]!=frame[base])|(frame[minus]!=frame[base])
                invalid=(bad[plus]|bad[minus]|terminals[plus]|terminals[minus]|timeouts[plus]|timeouts[minus]|frame_error|
                         (initial_error>1e-8)|(xi_error>1e-6)|(obs_error>1e-3))
                truth=wm.spec.error_vector(after[plus],after[minus])/(2*eps)
                entries=[]
                for i in range(n):
                    row={'clip':labels[i],'frame':int(frames[i]),'valid':not bool(invalid[i]),'initial_nse':float(initial_error[i]),
                         'context_error':float(xi_error[i]),'observation_error':float(obs_error[i]),'frame_mismatch':bool(frame_error[i]),
                         'true_response_norm':float(truth[i].norm()),'models':{}}
                    for mode,pred in predictions.items():
                        response=wm.spec.error_vector(pred[plus],pred[minus])/(2*eps)
                        fwd=(wm.spec.normalized_sq_error(pred[plus],after[plus])+wm.spec.normalized_sq_error(pred[minus],after[minus]))/2
                        row['models'][mode]={'response_mse':float((response[i]-truth[i]).square().mean()),'forward_nse':float(fwd[i])}
                    entries.append(row)
                valid=[x for x in entries if x['valid']]
                groups.append({'epsilon':eps,'valid_count':len(valid),'records':entries,
                    'summary':{mode:{metric:sum(x['models'][mode][metric] for x in valid)/len(valid) for metric in ('response_mse','forward_nse')}
                               for mode in predictions} if valid else {}})
        if sha(a.checkpoint)!=ckhash or any(sha(Path(s['weight']))!=s['weight_sha256'] or sha(Path(s['report']))!=s['report_sha256'] for s in sources.values()):
            raise ValueError('input changed')
        out={'seed':a.seed,'fit_seed':a.fit_seed,'checkpoint_sha256':ckhash,'source_models':sources,'script_sha256':sha(Path(__file__)),
             'scale_probe_sha256':sha(probe_path),'counted_simulator_transitions':4*n*(a.warmup+1),'groups':groups,
             'notes':'Same three offline fitted models and checkpoint for all collection seeds. Four matched one-step action branches after no-noise zero-bias warmup; same newly assigned bias in every branch. Direction is random in raw action coordinates, not actor policy gradient. Fixed friction, excludes resets and mismatched states. Two scales share starts; not independent observations or policy training.'}
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps({'seed':a.seed,'groups':[{'epsilon':g['epsilon'],'valid':g['valid_count'],'summary':g['summary']} for g in groups]}))
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
