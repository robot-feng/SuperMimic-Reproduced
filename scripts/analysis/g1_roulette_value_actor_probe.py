"""Fixed-checkpoint model actor directions from real-return value calibration."""
import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import torch
from torch.nn.utils import parameters_to_vector
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_noisy_imagination_direction import evaluate,frame_samples,select_and_repeat
from g1_policy_direction_probe import model_objective
from g1_real_score_probe_crossfit import tensor_hash
from g1_tail_policy_switch_probe import sha
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY,ROOT_BODY,TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic,CriticCfg,Policy
from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


class AffineCritic(torch.nn.Module):
    def __init__(self,critic,a,b):
        super().__init__();self.critic=critic;self.a=a;self.b=b
    def forward(self,*inputs):
        return self.a*self.critic(*inputs)+self.b


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();wp=a.output.with_suffix('.pt');fp=a.output.with_suffix('.failed.json')
    if any(x.exists() for x in (a.output,wp,fp)):raise FileExistsError('refusing overwrite')
    fitpath=Path('docs/analysis/g1_roulette_value_fit_9850_9852.json');fit=json.loads(fitpath.read_text(encoding='utf8'))
    ckpath=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt')
    if sha(ckpath)!='daa70a13e451e15676f306a32e23d9b0c48f3bea3f504a088d966dc31dd18163':raise ValueError('checkpoint mismatch')
    source_paths=[Path(__file__),fitpath,ckpath,Path('scripts/analysis/g1_noisy_imagination_direction.py'),
                  Path('scripts/analysis/g1_policy_direction_probe.py'),Path('scripts/analysis/g1_grounded_gradient_probe.py')]
    source={str(x):sha(x) for x in source_paths};ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    pcfg,wcfg=config_from_checkpoint(ck);ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg);actor.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic'])
    actor.eval();critic.eval().requires_grad_(False)
    a.clips=['walk1_subject1','dance1_subject1','jumps1_subject1'];a.eval_clips=['dance2_subject2','jumps1_subject5','walk2_subject3']
    a.frames_per_clip=a.eval_frames_per_clip=4;a.eval_replicates=1;a.warmup=10;a.horizon=8;a.eval_horizon=80;a.eps=.005
    a.history=int(ck['config']['policy_trainer']['history']);a.future=int(ck['config']['env']['future_frames']);a.gamma=float(ck['config']['policy_trainer']['gamma'])
    files=discover_npz(Path('data/npz/lafan1'),names=a.clips+a.eval_clips)
    if len(files)!=6:raise ValueError('missing reference clips')
    source.update({str(x):sha(x) for x in files});train_frames,test_frames,train_labels,test_labels=frame_samples(files,a,a.future)
    env=None;phase='setup';completed_steps=0
    try:
        env,obs,bad,starts,_,_,cmd,_,_=make_setup(files,train_frames,1,a.seed,a.warmup,a.history,a.future,actor,True)
        completed_steps+=len(train_frames)*a.warmup
        if bad.any():raise ValueError('gradient warmup failed; no outcome filtering')
        starts=select_and_repeat(starts,torch.arange(len(train_frames)),1);lib=cmd.library
        rec=flatten_history_term_major(torch.stack([analytic_observation(starts.hist_states[t],starts.hist_actions[t],starts.xi,lib.default_joint_pos) for t in range(a.history)]))
        obs_error=float((rec-obs['policy']).abs().max());goal_error=float((lib.goal_window(starts.frame,a.future)-obs['goal']).abs().max())
        if obs_error>1e-3 or goal_error>1e-4:raise ValueError('gradient start mismatch')
        identity=tensor_hash(starts.state,starts.xi,starts.frame,starts.prev_action,starts.hist_states,starts.hist_actions)
        robot=env.scene['robot'];heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        wm=WorldModel(wcfg,heads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
        gradients={};objectives={}
        for name in ('original','offset','affine'):
            c=fit['candidates'][name];proxy=AffineCritic(critic,c['a'],c['b'])
            obj,surv,reward,bootstrap,_=model_objective(wm,actor,proxy,lib,starts,a.horizon,a.future,a.gamma)
            grad=torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(obj,tuple(actor.parameters()))])
            if not torch.isfinite(grad).all() or grad.norm()<1e-12:raise ValueError('invalid gradient')
            gradients[name]=grad;objectives[name]={'objective':float(obj.detach()),'survival':surv,
                'reward':float(reward.detach().mean()),'bootstrap':float(bootstrap.detach().mean()),'grad_norm':float(grad.norm()),'gradient_sha256':tensor_hash(grad)}
        directions={k:v/v.norm() for k,v in gradients.items()}
        cosines={name:float(torch.dot(gradients['original'].double(),g.double())/(gradients['original'].double().norm()*g.double().norm())) for name,g in gradients.items()}
        base=parameters_to_vector(actor.parameters()).detach().clone()
        env.close();env=None;del obs,starts,ck;gc.collect()
        results={};inits={}
        for name in ('affine','offset'):
            phase='evaluate_'+name
            result,initial=evaluate(files,test_frames,test_labels,a.seed,actor,critic,wm,[directions['original'],directions[name]],base,a,sigma=.1)
            completed_steps+=6*len(test_frames)*(a.warmup+a.eval_horizon)
            if any(result[x]['base_control_max_abs_reward_difference']>1e-4 or result[x]['base_control_max_abs_bootstrap_difference']>1e-4 or result[x]['base_control_failure_disagreements'] for x in ('short','long')):
                raise ValueError('duplicate baseline mismatch')
            results[name]=result;inits[name]=tensor_hash(initial['state'],initial['obs'],initial['goal'],initial['eligible'])
        if inits['affine']!=inits['offset']:raise ValueError('cross-evaluation start mismatch')
        for section in ('short','long'):
            r1=results['affine'][section]['pairs'];r2=results['offset'][section]['pairs']
            if [x['index'] for x in r1]!=[x['index'] for x in r2]:raise ValueError('different eligible pairs')
            for x,y in zip(r1,r2):
                for key in ('base_reward','base_control_reward','base_bootstrap','base_failed','clean_gradient'):
                    if x[key]!=y[key]:raise ValueError(f'cross-evaluation baseline mismatch {key}')
        if completed_steps!=13080:raise ValueError('unexpected transition budget')
        if any(sha(Path(name))!=digest for name,digest in source.items()):raise ValueError('source changed')
        torch.save(gradients,wp)
        out={'seed':a.seed,'source_sha256':source,'fit_sha256':sha(fitpath),'gradient_vector_sha256':sha(wp),
             'gradient_frames':train_frames.tolist(),'gradient_labels':train_labels,'gradient_start_sha256':identity,
             'gradient_observation_error':obs_error,'gradient_goal_error':goal_error,'evaluation_frames':test_frames.tolist(),
             'evaluation_labels':test_labels,'evaluation_initial_sha256':inits,'horizon':a.horizon,'eval_horizon':a.eval_horizon,
             'eps':a.eps,'objectives':objectives,'gradient_cosines':cosines,'results':results,
             'simulator_transitions':completed_steps,'notes':'Exploratory value candidates selected after examining earlier holdout; frozen checkpoint, one normalized parameter perturbation, two paired evaluations. 80-step actual reward truncated; not multi-round training.'}
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({'seed':a.seed,'cosines':cosines,'steps':completed_steps}))
    except Exception as e:
        fp.write_text(json.dumps({'seed':a.seed,'phase':phase,'error':repr(e),'completed_counted_steps':completed_steps,
                                  'note':'A failed in-progress evaluation may have consumed additional steps.'},indent=2),encoding='utf8')
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
