"""Compare frozen-actor gradients using two refits on matched G1 starts."""
import argparse
import gc
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
import torch
from torch.nn.utils import parameters_to_vector
import g1_noisy_imagination_direction as real_eval
from g1_bias_canonical_eval import canonicalize
from g1_directional_probe import config_from_checkpoint
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_policy_direction_probe import model_objective
from g1_tail_policy_switch_probe import sha
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY,FOOT_GEOM_REGEX,ROOT_BODY,TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic,CriticCfg,Policy
from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def bias_setup(files,frames,groups,seed,warmup,history,future,policy,shared_origin=False):
    n=len(frames)
    cfg=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=n*groups,future_frames=future,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',
        record_transitions=False,seed=seed,anchor_obs=False)
    env=make_env(cfg,device='cpu')
    if shared_origin:env.scene.env_origins.zero_()
    cmd=env.command_manager.get_term('motion');io=StateIO(env);ctx=SimContext(env,FOOT_GEOM_REGEX,'torso_link')
    cmd.forced_frame[:]=frames.repeat(groups);obs,_=env.reset()
    bias=.02*torch.rand(n,29,generator=torch.Generator().manual_seed(seed+10300000))-.01
    env.scene['robot'].data.encoder_bias[:]=bias.repeat(groups,1)
    obs=env.observation_manager.compute(update_history=True)
    bad=torch.zeros(groups*n,dtype=torch.bool);states=[];actions=[]
    with torch.no_grad():
        for t in range(warmup+1):
            states.append(io.read().clone());actions.append(env.action_manager.action.clone())
            if t<warmup:
                shared=policy(obs['policy'][:n],obs['goal'][:n]).repeat(groups,1)
                obs,_,term,tout,_=env.step(shared);bad|=term|tout
        hs=torch.stack(states[-history:]);ha=torch.stack(actions[-history:]);xi=ctx.read()
        start=StartBatch(state=hs[-1,:n],xi=xi[:n],frame=cmd.frame[:n].clone(),prev_action=ha[-1,:n],
                         hist_states=hs[:,:n],hist_actions=ha[:,:n])
    return env,obs,bad,start,io,ctx,cmd,hs,ha


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-seed',type=int,default=8710);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    ckpath=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt');ckhash=sha(ckpath)
    sources={};weights={}
    for mode in ('raw','canonical'):
        stem=Path(f'docs/analysis/g1_bias_retrain_confirm_{mode}_{a.fit_seed}');rp=stem.with_suffix('.json');wp=stem.with_suffix('.pt')
        r=json.loads(rp.read_text(encoding='utf8'))
        if r['mode']!=mode or r['seed']!=a.fit_seed or r['checkpoint_sha256']!=ckhash or r['weights_sha256']!=sha(wp):raise ValueError('model identity mismatch')
        sources[mode]={'report':str(rp),'report_sha256':sha(rp),'weight':str(wp),'weight_sha256':sha(wp)}
        weights[mode]=torch.load(wp,map_location='cpu',weights_only=True)
    probe=Path('docs/analysis/g1_encoder_bias_symmetry_8600.json');pr=json.loads(probe.read_text(encoding='utf8'))
    scales=torch.tensor(pr['action_scale']);scale=scales[0]
    if pr['checkpoint_sha256']!=ckhash or scales.ndim!=2 or scales.shape[1]!=29 or not torch.equal(scales,scale.expand_as(scales)):
        raise ValueError('invalid scale')
    ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    pcfg,wcfg=config_from_checkpoint(ck);ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg);actor.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic'])
    actor.eval();critic.eval().requires_grad_(False)
    a.clips=['walk1_subject1','dance1_subject1','jumps1_subject1'];a.eval_clips=['walk2_subject1','dance2_subject1','jumps1_subject2']
    a.frames_per_clip=a.eval_frames_per_clip=a.eval_replicates=1;a.warmup=10;a.horizon=8;a.eval_horizon=80;a.eps=.005
    a.history=int(ck['config']['policy_trainer']['history']);a.future=int(ck['config']['env']['future_frames']);a.gamma=float(ck['config']['policy_trainer']['gamma'])
    if ck['iteration']!=2000 or ck['config']['env'].get('anchor_obs',False):raise ValueError('unsupported checkpoint')
    files=discover_npz(Path('data/npz/lafan1'),names=a.clips+a.eval_clips)
    if len(files)!=6:raise ValueError('missing clips')
    train_frames,test_frames,train_clips,test_clips=real_eval.frame_samples(files,a,a.future);env=None
    try:
        env,obs,bad,starts,_,_,cmd,_,_=bias_setup(files,train_frames,1,a.seed,a.warmup,a.history,a.future,actor,True)
        eligible=(~bad).nonzero().flatten()
        if len(eligible)<2:raise ValueError('insufficient gradient starts')
        starts=real_eval.select_and_repeat(starts,eligible,1);lib=cmd.library
        reconstructed=flatten_history_term_major(torch.stack([analytic_observation(starts.hist_states[t],starts.hist_actions[t],starts.xi,lib.default_joint_pos) for t in range(a.history)]))
        obs_error=float((reconstructed-obs['policy'][eligible]).abs().max());goal_error=float((lib.goal_window(starts.frame,a.future)-obs['goal'][eligible]).abs().max())
        if obs_error>1e-3 or goal_error>1e-4:raise ValueError(f'history mismatch {obs_error} {goal_error}')
        robot=env.scene['robot'];term=env.action_manager.get_term('joint_pos')
        if term.cfg.clip is not None or not torch.equal(term.target_ids,torch.arange(29)) or not torch.allclose(term.scale,scale.expand_as(term.scale),atol=0,rtol=0):
            raise ValueError('invalid target scale or order')
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        wm=WorldModel(wcfg,heads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
        original_step=wm.step
        def canonical_step(s,action,xi):
            aa,xx=canonicalize(action,xi,scale,1.)
            return original_step(s,aa,xx)
        wm.step=canonical_step
        grads=[];objectives=[]
        for mode in ('raw','canonical'):
            wm.dynamics.load_state_dict(weights[mode],strict=True)
            obj,survival,reward,boot,_=model_objective(wm,actor,critic,lib,starts,a.horizon,a.future,a.gamma)
            grad=torch.cat([x.detach().reshape(-1) for x in torch.autograd.grad(obj,tuple(actor.parameters()))])
            if not torch.isfinite(grad).all() or grad.norm()<1e-12:raise ValueError('invalid gradient')
            grads.append(grad);objectives.append({'model':mode,'objective':float(obj.detach()),'gradient_norm':float(grad.norm()),
                'survival':survival,'reward_mean':float(reward.detach().mean()),'bootstrap_mean':float(boot.detach().mean())})
        cosine=float(torch.dot(grads[0].double(),grads[1].double())/(grads[0].double().norm()*grads[1].double().norm()))
        directions=[g/g.norm() for g in grads];base=parameters_to_vector(actor.parameters()).detach().clone()
        train_bias=starts.xi[:,1:30].clone();start_frames=starts.frame.tolist()
    finally:
        if env is not None:env.close()
    del env,obs,starts,ck;gc.collect()
    real_eval.make_setup=bias_setup
    result,_=real_eval.evaluate(files,test_frames,test_clips,a.seed,actor,critic,wm,directions,base,a,sigma=.1)
    if any(result[x]['base_control_max_abs_reward_difference']>1e-4 or result[x]['base_control_max_abs_bootstrap_difference']>1e-4 or result[x]['base_control_failure_disagreements'] for x in ('short','long')):
        raise ValueError('base repeat mismatch')
    if sha(ckpath)!=ckhash or any(sha(Path(v['report']))!=v['report_sha256'] or sha(Path(v['weight']))!=v['weight_sha256'] for v in sources.values()):raise ValueError('source changed')
    out={'seed':a.seed,'fit_seed':a.fit_seed,'checkpoint_sha256':ckhash,'source_models':sources,'script_sha256':sha(Path(__file__)),
         'scale_probe_sha256':sha(probe),'gradient_clips':train_clips,'gradient_frames':train_frames.tolist(),
         'gradient_eligible_indices':eligible.tolist(),'gradient_start_frames':start_frames,'gradient_bias_rms':float(train_bias.square().mean().sqrt()),
         'gradient_input_observation_error':obs_error,'gradient_input_goal_error':goal_error,'evaluation_clips':test_clips,'evaluation_frames':test_frames.tolist(),
         'horizon':a.horizon,'eval_horizon':a.eval_horizon,'eps':a.eps,'gradient_cosine':cosine,'objectives':objectives,'results':result,
         'direction_mapping':{'clean_gradient':'raw_refit_canonical_input','noise_gradient':'canonical_refit_canonical_input'},
         'counted_simulator_transitions':len(train_frames)*a.warmup+6*len(test_frames)*(a.warmup+a.eval_horizon),
         'notes':'One fixed pair of refitted dynamics and same checkpoint; 3 gradient and 3 different clip evaluation starts, fixed friction, shared nonzero encoder bias, no observation noise, shared action noise. Normalized gradient directions, single finite perturbation, not Adam or multi-round policy training. Both dynamics use canonical input; analytic observation retains true bias. 80-step real reward is truncated.'}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'seed':a.seed,'cosine':cosine,'long':result['long']}))


if __name__=='__main__':main()
