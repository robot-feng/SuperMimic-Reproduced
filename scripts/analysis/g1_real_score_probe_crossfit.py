"""Small G1 real/model score and model-pathwise control-variate diagnostic.

Same fixed reference start and Gaussian noise tape; real env resets between
samples. This is not a training run. A frozen critic is used for bootstrap.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('WARP_CACHE_PATH',r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import torch

from clean_command_noise import replace_execution
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_model_score_probe import flattened_grad, model_rollout, policy_observations
from g1_noisy_imagination_direction import frame_samples
from g1_score_estimator import score_surrogate
from g1_score_blend_moments import BlendMoments,paired_risk_difference
from g1_score_vector_moments import VectorMoments
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def clone_start(start):
    return StartBatch(**{k: getattr(start,k).detach().clone() for k in
                         ('state','xi','frame','prev_action','hist_states','hist_actions')})


def tensor_hash(*values):
    digest=hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def reset_and_warmup(env, cmd, frames, actor, warmup, injection):
    cmd.forced_frame[:] = frames
    obs, _ = env.reset()
    bad = torch.zeros(len(frames),dtype=torch.bool)
    # Do not mutate a noise tape view retained by a caller.
    injection['noise'] = torch.zeros_like(injection['noise'])
    for _ in range(warmup):
        action = actor(obs['policy'],obs['goal'])
        obs, _, term, timeout, _ = env.step(action)
        bad |= term|timeout
    if bad.any():
        raise ValueError('reset/warmup failed')
    return obs


@torch.no_grad()
def collect_real(env, obs, actor, critic, wm, io, ctx, cmd, tape, injection, gamma, action_rate_scale):
    horizon,n,_ = tape.shape
    alive = torch.ones(n,dtype=torch.bool)
    obss, goals, actions, rewards, active, baselines = [],[],[],[],[],[]
    failures = torch.zeros(n,dtype=torch.bool)
    failure_step=torch.full((n,),-1,dtype=torch.long)
    rate_error = 0.
    terminal_audits=[]
    for t in range(horizon):
        state = io.read()
        if alive.any() and (not torch.isfinite(state[alive]).all() or not torch.isfinite(obs['policy'][alive]).all() or not torch.isfinite(obs['goal'][alive]).all()):
            raise ValueError('nonfinite real state or observations before reset')
        ref = reference_at(cmd.library,cmd.frame)
        value = critic(wm.spec.canonical(state),obs['goal'],wm.spec.normalize_xi(ctx.read()),wm.heads.root_error_features(state,ref))
        obss.append(obs['policy'].clone()); goals.append(obs['goal'].clone())
        active.append(alive.clone()); baselines.append(gamma**t*alive.float()*value)
        action = actor(obs['policy'],obs['goal'])
        actions.append(action.clone())
        expected_rate = action_rate_scale*(action-obs['policy'][:,-action.shape[-1]:]).square().sum(-1)
        injection['noise'] = tape[t]
        injection['pre_reset']=None
        injection['capture_terminal']=True
        obs, reward, term, timeout, _ = env.step(action)
        injection['capture_terminal']=False
        if (alive & timeout).any():
            raise ValueError('timeout encountered; no post-selection allowed')
        terminated_ids=(alive&term).nonzero().flatten()
        if terminated_ids.numel():
            terminal=injection['pre_reset']
            if terminal is None:
                raise ValueError('terminal pre-reset capture mismatch')
            keep=alive[terminal['ids']]
            if not torch.equal(terminal['ids'][keep],terminated_ids):raise ValueError('terminal ids mismatch')
            terminal_state=terminal['state'][keep];terminal_reward=terminal['reward'][keep]
            if not torch.isfinite(terminal_state).all() or not torch.isfinite(terminal_reward).all():
                raise ValueError('nonfinite terminal state/reward before reset')
            reward_error=float((terminal_reward-reward[terminated_ids]).abs().max())
            if reward_error>1e-6:raise ValueError('terminal reward changed across reset')
            terminal_audits.append({'step':t+1,'count':int(len(terminated_ids)),
                                    'reward_reset_error':reward_error,'state_abs_max':float(terminal_state.abs().max())})
        if alive.any() and not torch.isfinite(reward[alive]).all():
            raise ValueError('nonfinite real reward')
        actual_rate = env.reward_manager.get_term('action_rate_l2')
        if alive.any():
            rate_error = max(rate_error,float((actual_rate[alive]-expected_rate[alive]).abs().max()))
        rewards.append(reward.clone())
        failure_step=torch.where(alive&term,torch.full_like(failure_step,t+1),failure_step)
        failures |= alive & term
        alive &= ~term
    state = io.read()
    ref = reference_at(cmd.library,cmd.frame)
    value = critic(wm.spec.canonical(state),obs['goal'],wm.spec.normalize_xi(ctx.read()),wm.heads.root_error_features(state,ref))
    if alive.any() and (not torch.isfinite(state[alive]).all() or not torch.isfinite(value[alive]).all()):
        raise ValueError('nonfinite final survivor state or critic')
    return {'observations':torch.stack(obss),'goals':torch.stack(goals),'actions':torch.stack(actions),
            'rewards':torch.stack(rewards),'active':torch.stack(active),'baselines':torch.stack(baselines),
            'bootstrap':gamma**horizon*alive.float()*value,'failed':failures,'failure_step':failure_step,'terminal_audits':terminal_audits,
            'action_rate_max_error':rate_error}


def describe(values):
    return {'mean':mean(values),'std':stdev(values),'se':stdev(values)/(len(values)**.5),'values':values}


def fit_time_baseline(episodes,gamma,per_env=False):
    """Independent time baseline, pooled or per fixed start, conditional on survival."""
    if not episodes:raise ValueError('no baseline pilot episodes')
    horizon=len(episodes[0]['rewards'])
    n=episodes[0]['rewards'].shape[1]
    shape=(horizon,n) if per_env else (horizon,)
    total=torch.zeros(shape);counts=torch.zeros(shape)
    for ep in episodes:
        if ep['rewards'].shape!=(horizon,n) or ep['rewards'].shape!=ep['active'].shape:
            raise ValueError('misaligned pilot trajectory')
        discounted=torch.stack([gamma**t*ep['active'][t].float()*ep['rewards'][t] for t in range(horizon)])
        future=discounted.flip(0).cumsum(0).flip(0)+ep['bootstrap'][None]
        active=ep['active'].float()
        total+=(future*active) if per_env else (future*active).sum(1)
        counts+=active if per_env else active.sum(1)
    return torch.where(counts>0,total/counts.clamp_min(1),torch.zeros_like(total)),counts


def fit_leave_one_start_critic_offsets(episodes,gamma):
    """Fit additive real-return corrections from other fixed starts only."""
    if not episodes:raise ValueError('no baseline pilot episodes')
    h,n=episodes[0]['rewards'].shape
    if n<2:raise ValueError('leave-one-start-out needs at least two starts')
    sums=torch.zeros(h,n);counts=torch.zeros(h,n)
    for ep in episodes:
        if ep['rewards'].shape!=(h,n) or ep['active'].shape!=(h,n) or ep['baselines'].shape!=(h,n):
            raise ValueError('misaligned critic pilot trajectory')
        active=ep['active'].float()
        discounted=torch.stack([gamma**t*active[t]*ep['rewards'][t] for t in range(h)])
        future=discounted.flip(0).cumsum(0).flip(0)+ep['bootstrap'][None]
        sums+=(future-ep['baselines'])*active
        counts+=active
    peer_sums=sums.sum(1,keepdim=True)-sums
    peer_counts=counts.sum(1,keepdim=True)-counts
    offsets=torch.where(peer_counts>0,peer_sums/peer_counts.clamp_min(1),torch.zeros_like(peer_sums))
    return offsets,peer_counts,counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--npz-dir',type=Path,default=Path('data/npz/lafan1'))
    p.add_argument('--clips',nargs='+',default=['walk1_subject1'])
    p.add_argument('--eval-clips',nargs='+',default=['dance1_subject1'])
    p.add_argument('--seed',type=int,default=3420)
    p.add_argument('--frames-per-clip',type=int,default=1)
    p.add_argument('--forced-frame-global',type=int,help='diagnostic only: explicit global frame in selected clip library')
    p.add_argument('--eval-frames-per-clip',type=int,default=1)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--antithetic',action='store_true',help='opposite noise tapes form independent pairs')
    p.add_argument('--vector-stats',action='store_true',help='stream full-gradient trace, IID tapes only')
    p.add_argument('--vector-split',type=int,help='fit vector coefficient on first N samples, validate on remaining samples')
    p.add_argument('--vector-output',type=Path,help='save the independent validation mean gradients for held-out directional evaluation')
    p.add_argument('--optimizer-jacobian',type=Path,help='fixed Adam Jacobian metadata for weighted vector covariance')
    p.add_argument('--noise-std',type=float,default=.1)
    p.add_argument('--noise-seed-offset',type=int,default=3500000)
    p.add_argument('--empirical-baseline-pilot',type=int,default=0,help='independent real trajectories to fit a time/survival baseline')
    p.add_argument('--empirical-baseline-per-env',action='store_true',help='fit separate time baseline for each fixed diagnostic start')
    p.add_argument('--empirical-baseline-loo-critic',action='store_true',help='fit time-dependent critic offsets from all other starts')
    p.add_argument('--baseline-seed-offset',type=int,default=4300000)
    p.add_argument('--baseline-vector-stats',action='store_true',help='stream full-gradient variance for empirical versus critic baselines')
    p.add_argument('--baseline-vector-split',type=int,help='fit baseline interpolation on first N tapes and validate on the rest')
    p.add_argument('--blend-vector-pilot',type=int,help='fit full-vector model/real blend on N IID tapes, then evaluate independent paired tapes')
    p.add_argument('--blend-vector-output',type=Path,help='save validation mean gradients for independent held-out policy updates')
    p.add_argument('--blend-validation-vector-stats',action='store_true',help='stream full real/model gradient variances on validation tapes')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identity-output',type=Path,help='optional diagnostic tensors for cross-layout alignment')
    a=p.parse_args()
    if a.output.exists() or (a.vector_output is not None and a.vector_output.exists()):
        raise FileExistsError('refusing to overwrite an experiment output')
    if a.optimizer_jacobian is not None and (not a.vector_stats or a.vector_split is None or a.vector_output is not None):
        p.error('optimizer Jacobian requires vector-stats and vector-split, without vector-output')
    if a.baseline_vector_split is not None and (not a.baseline_vector_stats or min(a.baseline_vector_split,a.repeats-a.baseline_vector_split)<2):
        p.error('baseline vector split requires baseline vector stats and at least two tapes per split')
    if set(a.clips)&set(a.eval_clips) or min(a.frames_per_clip,a.eval_frames_per_clip,a.horizon)<1 or a.warmup<10 or a.repeats<2 or a.noise_std<=0 or a.empirical_baseline_pilot<0 or (a.empirical_baseline_pilot and a.baseline_seed_offset==a.noise_seed_offset) or (a.antithetic and (a.repeats<4 or a.repeats%2)) or (a.antithetic and (a.vector_stats or a.baseline_vector_stats)) or (a.baseline_vector_stats and (not a.empirical_baseline_pilot or a.vector_stats)) or (a.vector_split is not None and (not a.vector_stats or min(a.vector_split,a.repeats-a.vector_split)<2)) or (a.vector_output is not None and a.vector_split is None):
        p.error('invalid configuration')
    if a.blend_vector_pilot is not None and (a.blend_vector_pilot<2 or a.repeats-a.blend_vector_pilot<4 or (a.repeats-a.blend_vector_pilot)%2 or not a.empirical_baseline_pilot or a.antithetic or a.vector_stats or a.baseline_vector_stats):
        p.error('blend requires independent empirical baseline, >=2 pilot, >=2 heldout pairs and no other vector mode')
    if a.blend_vector_output is not None and a.blend_vector_pilot is None:
        p.error('blend vector output requires blend vector pilot')
    if a.blend_validation_vector_stats and (a.blend_vector_pilot is None or a.blend_vector_output is not None):
        p.error('validation moments require blend mode and are exclusive with saved gradient means')
    if (a.empirical_baseline_per_env or a.empirical_baseline_loo_critic) and not a.empirical_baseline_pilot:
        p.error('per-start or cross-start baseline requires pilot trajectories')
    if a.empirical_baseline_per_env and a.empirical_baseline_loo_critic:
        p.error('per-start and leave-one-start-out baselines are mutually exclusive')
    from g1_mature_fit_actor_probe import checkpoint_pickle_module
    ck=torch.load(a.ckpt,map_location='cpu',weights_only=False,
                  pickle_module=checkpoint_pickle_module(os.name != 'nt'))
    pcfg,wcfg=config_from_checkpoint(ck)
    if ck['config']['env'].get('anchor_obs',False):raise ValueError('anchor observation unsupported')
    a.future=int(ck['config']['env']['future_frames']);a.history=int(ck['config']['policy_trainer']['history'])
    a.gamma=float(ck['config']['policy_trainer']['gamma']);a.eval_horizon=a.horizon
    ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg)
    actor.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic'])
    actor.eval();critic.eval().requires_grad_(False)
    jacobian=None
    jacobian_info=None
    if a.optimizer_jacobian is not None:
        import numpy as np
        from g1_tail_policy_switch_probe import sha
        info=json.loads(a.optimizer_jacobian.read_text(encoding='utf8'))
        path=Path(info['jacobian_path'].replace('\\','/'))
        if sha(a.ckpt)!=info['checkpoint_sha256'] or sha(path)!=info['jacobian_sha256']:
            raise ValueError('optimizer Jacobian provenance mismatch')
        named=list(actor.named_parameters()); layers=info['result']['layers']
        if list(layers)!=[name for name,_ in named] or any(layers[name]['count']!=p.numel() for name,p in named):
            raise ValueError('optimizer Jacobian parameter order mismatch')
        array=np.load(path,allow_pickle=False,mmap_mode='c')
        if array.shape!=(sum(p.numel() for _,p in named),) or array.dtype!=np.dtype('<f4'):
            raise ValueError('optimizer Jacobian shape mismatch')
        jacobian=torch.from_numpy(array)
        jacobian_info={'metadata':str(a.optimizer_jacobian),'metadata_sha256':sha(a.optimizer_jacobian),
                       'jacobian_sha256':sha(path),'reference_gradient_seed':info['gradient_seed'],
                       'notes':'Fixed unclipped local Adam Jacobian; metric on raw gradient covariance, not exact Adam update variance.'}
    files=discover_npz(a.npz_dir,names=a.clips+a.eval_clips)
    if len(files)!=len(a.clips)+len(a.eval_clips):raise ValueError('missing clips')
    frames,_,labels,_=frame_samples(files,a,a.future);gc.collect()
    if a.forced_frame_global is not None:
        if len(a.clips)!=1 or a.frames_per_clip!=1 or a.forced_frame_global<0:
            raise ValueError('forced frame requires one clip and one frame')
        frames=torch.tensor([a.forced_frame_global],dtype=torch.long);labels=[a.clips[0]]
    env=None
    try:
        env,obs,bad,start,io,ctx,cmd,_,_=make_setup(files,frames,1,a.seed,a.warmup,a.history,a.future,actor,True)
        if a.forced_frame_global is not None:
            selected=[clip for clip in cmd.library.clips if clip.name==a.clips[0]]
            if len(selected)!=1 or not selected[0].start<=a.forced_frame_global<selected[0].start+selected[0].length-a.warmup-a.horizon-2:
                raise ValueError('forced frame does not lie in the selected clip with sufficient horizon')
        if bad.any():raise ValueError('warmup failed')
        start=clone_start(start);initial_obs=obs['policy'].clone();initial_goal=obs['goal'].clone()
        lib=cmd.library;robot=env.scene['robot']
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        wm=WorldModel(wcfg,heads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
        initial_reconstructed=flatten_history_term_major(torch.stack([analytic_observation(start.hist_states[t],start.hist_actions[t],start.xi,lib.default_joint_pos) for t in range(a.history)]))
        init_err=float((initial_reconstructed-initial_obs).abs().max())
        if init_err>1e-3:raise ValueError('initial history mismatch')
        n=len(frames)
        if a.empirical_baseline_loo_critic and n<2:raise ValueError('leave-one-start-out requires at least two starts')
        _,_,j=model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,torch.zeros(a.horizon,n,29))
        grad=flattened_grad(j.mean(),actor);direction=grad/grad.norm();del grad,j
        hashes={'start':tensor_hash(start.state,start.xi,start.frame,start.prev_action,start.hist_states,start.hist_actions),
                'initial_obs_goal':tensor_hash(initial_obs,initial_goal),'direction':tensor_hash(direction)}
        if a.identity_output is not None:
            a.identity_output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state':start.state,'xi':start.xi,'frame':start.frame,'prev_action':start.prev_action,
                        'hist_states':start.hist_states,'hist_actions':start.hist_actions,
                        'obs':initial_obs,'goal':initial_goal,'direction':direction},a.identity_output)
        term=env.action_manager.get_term('joint_pos')
        if term.noise_std!=0:raise ValueError('automatic action noise must be disabled')
        orig=term.process_actions
        injection={'noise':torch.zeros(n,29),'clean_error':0.,'executed_error':0.,'calls':0,'capture_terminal':False,'pre_reset':None}
        def process(actions):
            orig(actions);replace_execution(term,actions,injection['noise']);injection['calls']+=1
            injection['clean_error']=max(injection['clean_error'],float((term.raw_action-actions).abs().max()))
            injection['executed_error']=max(injection['executed_error'],float((term.executed_raw-actions-injection['noise']).abs().max()))
        term.process_actions=process
        original_reset_idx=env._reset_idx
        def observed_reset_idx(ids):
            if injection['capture_terminal']:
                injection['pre_reset']={'ids':ids.detach().clone(),'state':io.read()[ids].detach().clone(),
                                        'reward':env.reward_buf[ids].detach().clone()}
            return original_reset_idx(ids)
        env._reset_idx=observed_reset_idx
        rng=torch.Generator().manual_seed(a.noise_seed_offset+a.seed)
        empirical=None;pilot_info=None
        if a.empirical_baseline_pilot:
            pilot_rng=torch.Generator().manual_seed(a.baseline_seed_offset+a.seed)
            episodes=[];pilot_failures=0
            for _ in range(a.empirical_baseline_pilot):
                pilot_obs=reset_and_warmup(env,cmd,frames,actor,a.warmup,injection)
                if (pilot_obs['policy']-initial_obs).abs().max()>1e-3 or (pilot_obs['goal']-initial_goal).abs().max()>1e-4 or (io.read()-start.state).abs().max()>1e-4 or (ctx.read()-start.xi).abs().max()>1e-5 or not torch.equal(cmd.frame,start.frame):
                    raise ValueError('baseline pilot reset mismatch')
                tape_pilot=a.noise_std*torch.randn(a.horizon,n,29,generator=pilot_rng)
                ep=collect_real(env,pilot_obs,actor,critic,wm,io,ctx,cmd,tape_pilot,injection,a.gamma,heads.reward_cfg.dt*heads.reward_cfg.w_action_rate)
                if ep['action_rate_max_error']>1e-5:raise ValueError('baseline pilot action-rate mismatch')
                episodes.append(ep);pilot_failures+=int(ep['failed'].sum())
            if a.empirical_baseline_loo_critic:
                empirical,counts,critic_counts=fit_leave_one_start_critic_offsets(episodes,a.gamma)
            else:
                empirical,counts=fit_time_baseline(episodes,a.gamma,per_env=a.empirical_baseline_per_env)
                critic_counts=counts
            per_axis=a.empirical_baseline_per_env or a.empirical_baseline_loo_critic
            critic_sum=(torch.stack([ep['baselines'] for ep in episodes]).sum(0) if per_axis
                        else torch.stack([ep['baselines'].sum(1) for ep in episodes]).sum(0))
            critic_mean=torch.where(critic_counts>0,critic_sum/critic_counts.clamp_min(1),torch.zeros_like(critic_counts))
            pilot_info={'trajectories':a.empirical_baseline_pilot*n,'seed_offset':a.baseline_seed_offset,
                        'baseline':empirical.tolist(),'critic_baseline_mean':critic_mean.tolist(),
                        'survivor_counts':counts.tolist(),'failures':pilot_failures,
                        'empty_time_steps':int((counts==0).sum())}
            if a.empirical_baseline_per_env:pilot_info['per_env']=True
            if a.empirical_baseline_loo_critic:
                pilot_info.update({'loo_critic':True,'baseline_is_critic_offset':True,'own_survivor_counts':critic_counts.tolist()})
            del episodes
        if jacobian is not None:
            from g1_weighted_vector_moments import WeightedVectorMoments,validate as validate_weighted
        def new_moments():
            return WeightedVectorMoments(jacobian) if jacobian is not None else VectorMoments()
        rows=[];moments=new_moments() if a.vector_stats else None
        baseline_moments=VectorMoments() if a.baseline_vector_stats else None
        blend_moments=BlendMoments() if a.blend_vector_pilot is not None else None
        blend_validation_moments=VectorMoments() if a.blend_validation_vector_stats else None
        blend_fit=None;blend_pending=None;blend_pairs=[];blend_mean_real=None;blend_mean_model=None;blend_mean_count=0
        split_fit=None;baseline_split_fit=None;fit_mean_real=None;fit_mean_zero=None
        for i in range(a.repeats):
            if not a.antithetic or i%2==0:
                tape=a.noise_std*torch.randn(a.horizon,n,29,generator=rng)
            else:
                tape=-tape
            obs=reset_and_warmup(env,cmd,frames,actor,a.warmup,injection)
            check_obs=float((obs['policy']-initial_obs).abs().max())
            check_goal=float((obs['goal']-initial_goal).abs().max())
            check_state=float((io.read()-start.state).abs().max())
            check_xi=float((ctx.read()-start.xi).abs().max())
            if check_obs>1e-3 or check_goal>1e-4 or check_state>1e-4 or check_xi>1e-5 or not torch.equal(cmd.frame,start.frame):
                raise ValueError(f'reset mismatch obs={check_obs} goal={check_goal} state={check_state} xi={check_xi}')
            scale=heads.reward_cfg.dt*heads.reward_cfg.w_action_rate
            real=collect_real(env,obs,actor,critic,wm,io,ctx,cmd,tape,injection,a.gamma,scale)
            if real['action_rate_max_error']>1e-5:
                raise ValueError(f"actual action rate mismatch: {real['action_rate_max_error']}")
            active=real['active']
            rs,real_check=score_surrogate(actor,real['observations'],real['goals'],real['actions'],tape,
                real['rewards'],active.float(),real['bootstrap'],a.gamma,a.noise_std,a.history,scale,
                active=active,baseline=real['baselines'])
            real_grad=flattened_grad(rs.mean(),actor)
            real_score=float(torch.dot(real_grad,direction))
            empirical_score=None
            if empirical is not None:
                if a.empirical_baseline_loo_critic:
                    empirical_baseline=real['baselines']+empirical*active.float()
                else:
                    empirical_baseline=(empirical if a.empirical_baseline_per_env else empirical[:,None])*active.float()
                er,emp_check=score_surrogate(actor,real['observations'],real['goals'],real['actions'],tape,
                    real['rewards'],active.float(),real['bootstrap'],a.gamma,a.noise_std,a.history,scale,
                    active=active,baseline=empirical_baseline)
                if max(emp_check.values())>1e-4:raise ValueError('empirical-baseline action replay mismatch')
                empirical_grad=flattened_grad(er.mean(),actor)
                empirical_score=float(torch.dot(empirical_grad,direction))
                if baseline_moments is not None:
                    # Track the empirical gradient directly: subtracting huge,
                    # nearly equal covariance terms is numerically unstable.
                    baseline_moments.add(real_grad,empirical_grad)
                    if a.baseline_vector_split is not None and i+1==a.baseline_vector_split:
                        baseline_split_fit=baseline_moments.summary()
                        baseline_moments=VectorMoments()
                if a.blend_vector_pilot is None:del empirical_grad
                del er,empirical_baseline
            if not a.vector_stats:del real_grad
            ro,vals,j=model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,tape)
            path_grad=flattened_grad(j.mean(),actor)
            model_path=float(torch.dot(path_grad,direction));del ro,vals,j
            if a.blend_vector_pilot is not None:
                if i<a.blend_vector_pilot:
                    blend_moments.add(empirical_grad,path_grad)
                    if i+1==a.blend_vector_pilot:
                        blend_fit=blend_moments.fit();blend_moments=None;gc.collect()
                else:
                    if blend_validation_moments is not None:
                        blend_validation_moments.add(empirical_grad,path_grad)
                    if a.blend_vector_output is not None:
                        blend_mean_count+=1
                        if blend_mean_real is None:
                            blend_mean_real=empirical_grad.clone();blend_mean_model=path_grad.clone()
                        else:
                            blend_mean_real.add_(empirical_grad-blend_mean_real,alpha=1/blend_mean_count)
                            blend_mean_model.add_(path_grad-blend_mean_model,alpha=1/blend_mean_count)
                    if blend_pending is None:
                        blend_pending=(empirical_grad-path_grad,path_grad)
                    else:
                        blend_pairs.append(paired_risk_difference(blend_pending[0],blend_pending[1],empirical_grad,blend_fit['coefficient']))
                        blend_pending=None
                del empirical_grad
            if not a.vector_stats:del path_grad
            with torch.no_grad():
                ro,vals,j=model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,tape)
                model_obs=policy_observations(wm,lib,start,ro)
                model_boot=a.gamma**a.horizon*ro.survival[-1]*vals[-1]
                model_base=torch.stack([a.gamma**t*ro.survival[t]*vals[t] for t in range(a.horizon)])
            ms,model_check=score_surrogate(actor,model_obs,ro.goals[:-1],ro.actions,tape,ro.rewards,ro.survival[:-1],
                model_boot,a.gamma,a.noise_std,a.history,scale,baseline=model_base)
            score_grad=flattened_grad(ms.mean(),actor)
            model_score=float(torch.dot(score_grad,direction))
            if a.vector_stats:
                zero_grad=path_grad-score_grad
                moments.add(real_grad,zero_grad)
                if a.vector_split is not None and i+1==a.vector_split:
                    split_fit=moments.summary()
                    if a.vector_output is not None:
                        fit_mean_real=moments.mean_r.detach().clone()
                        fit_mean_zero=moments.mean_z.detach().clone()
                    # Keep only scalar fit statistics while validating later samples.
                    moments=new_moments()
                del zero_grad
                del real_grad,path_grad
            del score_grad
            if max(*real_check.values(),*model_check.values())>1e-4:
                raise ValueError(f'policy replay mismatch {real_check} {model_check}')
            rows.append({'real_score':real_score,'model_score':model_score,'model_pathwise':model_path,
                         'control_variate':real_score+model_path-model_score,'failed':int(real['failed'].sum()),
                         'failure_step':real['failure_step'].tolist(),'bootstrap':real['bootstrap'].tolist(),'terminal_audits':real['terminal_audits'],
                         'reset_obs_error':check_obs,'reset_goal_error':check_goal,'reset_state_error':check_state,'reset_xi_error':check_xi,
                         'action_rate_max_error':real['action_rate_max_error'],'real_replay':real_check,'model_replay':model_check})
            if empirical_score is not None:
                rows[-1]['real_score_empirical_baseline']=empirical_score
                rows[-1]['control_variate_empirical_baseline']=empirical_score+model_path-model_score
            print(f"{i+1}/{a.repeats} real={real_score:.3f} modelscore={model_score:.3f} path={model_path:.3f} cv={rows[-1]['control_variate']:.3f}",flush=True)
            del real,rs,ro,vals,j,model_obs,ms
            gc.collect()
        if injection['clean_error']>1e-6 or injection['executed_error']>1e-5:
            raise ValueError(f'action injection mismatch {injection}')
        keys=('real_score','model_score','model_pathwise','control_variate')
        if empirical is not None:keys+=('real_score_empirical_baseline','control_variate_empirical_baseline')
        if a.antithetic:
            summary={k:describe([(rows[2*i][k]+rows[2*i+1][k])/2 for i in range(a.repeats//2)]) for k in keys}
        else:
            summary={k:describe([r[k] for r in rows]) for k in keys}
        split_result=None
        if a.vector_split is not None:
            fit,validation=split_fit,moments.summary()
            if jacobian is not None:
                split_result={'fit':fit,'validation':validation,**validate_weighted(fit,validation),
                              'notes':'Independent pilot/validation noise with fixed actor/model/Jacobian; weighted metric is a local proxy, not exact clipped Adam update variance.'}
                coefficient=None
            else:
                coefficient=max(-1.,min(1.,fit['fitted_coefficient']))
                vr,vz,cov=validation['real_variance_trace'],validation['zero_variance_trace'],validation['cross_covariance_trace']
                split_result={'fit':fit,'validation':validation,'pilot_clipped_coefficient':coefficient,
                              'validation_fitted_ratio':(vr+2*coefficient*cov+coefficient*coefficient*vz)/vr,
                              'notes':'Fit and validation use different IID noise samples from the same generator, fixed start, actor and model. Estimated coefficient minimizes empirical trace, not every direction.'}
            if a.vector_output is not None:
                a.vector_output.parent.mkdir(parents=True,exist_ok=True)
                if fit_mean_real is None or fit_mean_zero is None:
                    raise ValueError('missing pilot mean for full-budget baseline')
                from g1_score_paired_math import combined_mean, crossfit_mean
                if a.vector_split != a.repeats-a.vector_split:
                    raise ValueError('crossfit requires equal folds')
                all_real=combined_mean(fit_mean_real,moments.mean_r,a.vector_split,a.repeats-a.vector_split)
                reverse_coefficient=max(-1.,min(1.,validation['fitted_coefficient']))
                crossfit=crossfit_mean(fit_mean_real,moments.mean_r,fit_mean_zero,moments.mean_z,coefficient,reverse_coefficient)
                with a.vector_output.open('xb') as vector_file:
                    torch.save({'mean_real':moments.mean_r,'mean_all_real':all_real,'mean_zero':moments.mean_z,'mean_crossfit':crossfit,
                            'coefficient':coefficient,'reverse_coefficient':reverse_coefficient,
                            'model_direction':direction,'identity_sha256':hashes,'seed':a.seed,'frames':frames,
                            'checkpoint':str(a.ckpt),'horizon':a.horizon,'noise_std':a.noise_std,
                            'noise_seed_offset':a.noise_seed_offset,'fit_count':a.vector_split,
                            'validation_count':a.repeats-a.vector_split},vector_file)
        baseline_vector_result=None
        baseline_vector_split_result=None
        if baseline_moments is not None and a.baseline_vector_split is None:
            v=baseline_moments.summary()
            baseline_vector_result={'count':v['count'],'critic_variance_trace':v['real_variance_trace'],
                'empirical_variance_trace':v['zero_variance_trace'],
                'empirical_to_critic_variance_ratio':v['zero_variance_trace']/v['real_variance_trace'],
                'cross_covariance_trace':v['cross_covariance_trace'],
                'critic_mean_norm':v['real_mean_norm'],'empirical_mean_norm':v['zero_mean_norm'],
                'notes':'Directly estimated two gradient variances, rather than subtracting nearly equal variance terms.'}
        elif baseline_moments is not None:
            from g1_baseline_blend import fit_and_validate
            baseline_vector_split_result=fit_and_validate(baseline_split_fit,baseline_moments.summary())
        blend_vector_result=None
        if a.blend_vector_pilot is not None:
            if blend_pending is not None or len(blend_pairs)!=(a.repeats-a.blend_vector_pilot)//2:raise ValueError('incomplete heldout pairs')
            blend_vector_result={'pilot':blend_fit,'heldout_pairs':blend_pairs,
                'summary':{k:describe([r[k] for r in blend_pairs]) for k in ('blend_minus_model','score_minus_model')},
                'notes':'Independent time-baseline pilot. Fit coefficient on first IID tapes; subsequent alternating candidate and target trajectories are independent. Squared-distance differences to target estimate gradient-MSE differences for this fixed start and noisy finite-horizon objective. Empirical squared bias is clipped, target size equals one trajectory; only a few samples. No online model fitting or training claim.'}
            if blend_validation_moments is not None:
                b=blend_validation_moments.summary()
                blend_vector_result['validation_moments']={'count':b['count'],
                    'real_variance_trace':b['real_variance_trace'],'model_variance_trace':b['zero_variance_trace'],
                    'real_model_covariance_trace':b['cross_covariance_trace'],
                    'real_mean_norm':b['real_mean_norm'],'model_mean_norm':b['zero_mean_norm']}
            if a.blend_vector_output is not None:
                if blend_mean_count!=a.repeats-a.blend_vector_pilot:raise ValueError('validation mean count mismatch')
                a.blend_vector_output.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'mean_real':blend_mean_real,'mean_model':blend_mean_model,'model_direction':direction,'alpha':blend_fit['coefficient'],
                            'identity_sha256':hashes,'seed':a.seed,'frames':frames,'checkpoint':str(a.ckpt),
                            'horizon':a.horizon,'noise_std':a.noise_std,'noise_seed_offset':a.noise_seed_offset,
                            'baseline_seed_offset':a.baseline_seed_offset,'baseline_per_env':a.empirical_baseline_per_env,
                            'baseline_loo_critic':a.empirical_baseline_loo_critic,'pilot_count':a.blend_vector_pilot,
                            'validation_count':blend_mean_count},a.blend_vector_output)
                blend_vector_result['vector_output']=str(a.blend_vector_output)
        out={'checkpoint':str(a.ckpt),'seed':a.seed,'frames':frames.tolist(),'labels':labels,'warmup':a.warmup,'horizon':a.horizon,
             'repeats':a.repeats,'antithetic':a.antithetic,'noise_std':a.noise_std,'noise_seed_offset':a.noise_seed_offset,'initial_obs_error':init_err,'identity_sha256':hashes,'summary':summary,'results':rows,
             'vector_moments':moments.summary() if moments is not None and a.vector_split is None else None,
             'baseline_vector_moments':baseline_vector_result,
             'baseline_vector_split':baseline_vector_split_result,
             'blend_vector_moments':blend_vector_result,
             'vector_split':split_result,
             'optimizer_jacobian':jacobian_info,
             'vector_output':str(a.vector_output) if a.vector_output is not None else None,
             'empirical_baseline_pilot':pilot_info,
             'action_injection_checks':{k:injection[k] for k in ('calls','clean_error','executed_error')},
             'simulator_transition_counts':{'initial_warmup':n*a.warmup,'repeated_warmup':n*a.warmup*a.repeats,'measurement':n*a.horizon*a.repeats,
                                            'baseline_pilot_warmup':n*a.warmup*a.empirical_baseline_pilot,
                                            'baseline_pilot_measurement':n*a.horizon*a.empirical_baseline_pilot},
             'notes':'Fixed checkpoint and reference starts; one small repeated-sample diagnostic, not an independent seed study. When enabled, standard errors use independent antithetic pairs, not individual correlated trajectories. Real reward uses actual simulator termination; any timeout aborts. Default real and model scores use a pre-action frozen critic baseline; optional empirical real-score baseline is fit on separate trajectories. Model uses soft survival and frozen critic bootstrap, real bootstrap only on survival. Sigma changes the objective; observation noise and domain randomization disabled. Model and real share same noise tape. Frozen critic is not calibrated; no training or performance improvement claim.'}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf8') as output_file:
            output_file.write(json.dumps(out,indent=2,allow_nan=False))
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
