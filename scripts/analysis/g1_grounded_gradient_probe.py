"""Matched G1 pure-model vs true-forward/model-backward policy directions.

Reference resets, fixed friction and no noise. A small diagnostic, not an
actor update or equal-budget training comparison. Real-forward simulation
uses additional simulator transitions. Reset/censor handling is documented.
"""
from __future__ import annotations
import argparse,gc,json,os,time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1');os.environ.setdefault('WARP_CACHE_PATH',r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from g1_directional_probe import config_from_checkpoint
from g1_policy_direction_probe import model_objective
from policy_direction_math import advance_true_rollout,replace_forward
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
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)

def make_setup(files,frames,groups,seed,warmup,history,future,policy,shared_origin=False):
 n=len(frames);cfg=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=n*groups,future_frames=future,
  action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',record_transitions=False,seed=seed,anchor_obs=False)
 env=make_env(cfg,device='cpu')
 if shared_origin:env.scene.env_origins.zero_()
 cmd=env.command_manager.get_term('motion');io=StateIO(env);ctx=SimContext(env,FOOT_GEOM_REGEX,'torso_link')
 cmd.forced_frame[:]=frames.repeat(groups);obs,_=env.reset();bad=torch.zeros(groups*n,dtype=torch.bool);states=[];actions=[]
 with torch.no_grad():
  for t in range(warmup+1):
   states.append(io.read().clone());actions.append(env.action_manager.action.clone())
   if t<warmup:
    shared=policy(obs['policy'][:n],obs['goal'][:n]).repeat(groups,1);obs,_,term,tout,_=env.step(shared);bad|=term|tout
  hs=torch.stack(states[-history:]);ha=torch.stack(actions[-history:]);xi=ctx.read()
  start=StartBatch(state=hs[-1,:n],xi=xi[:n],frame=cmd.frame[:n].clone(),prev_action=ha[-1,:n],hist_states=hs[:,:n],hist_actions=ha[:,:n])
 return env,obs,bad,start,io,ctx,cmd,hs,ha

def true_forward_objective(env,obs,start,wm,policy,critic,lib,io,ctx,cmd,horizon,future,gamma,initial_valid):
 """Return one surrogate objective per actual trajectory and survival mask."""
 s=start.state;xi=start.xi;prev=start.prev_action;n=len(s)
 history=[analytic_observation(start.hist_states[i],start.hist_actions[i],xi,lib.default_joint_pos) for i in range(start.hist_states.shape[0])]
 survival=torch.ones(n);discounted=torch.zeros(n);real_sum=torch.zeros(n);alive=initial_valid.clone()
 max_state_error=0.;max_obs_error=0.;reward_errors=[]
 for t in range(horizon):
  policy_obs=flatten_history_term_major(torch.stack(history));mask=alive.clone()
  if mask.any():max_obs_error=max(max_obs_error,float((policy_obs.detach()[mask]-obs['policy'][mask]).abs().max()))
  frame=cmd.frame.clone();ref=reference_at(lib,frame);goal=lib.goal_window(frame,future)
  action=policy(policy_obs,goal);pred=wm.step(s,action,xi)
  obs,real_reward,term,tout,_=env.step(action.detach());real_state=io.read()
  s=replace_forward(pred,real_state);max_state_error=max(max_state_error,float((s.detach()-real_state).abs().max()))
  reward=wm.heads.reward(s,action,prev,ref)['total'];logit=wm.termination_logits(s,ref)
  discounted=discounted+gamma**t*survival*reward
  with torch.no_grad():
   real_sum+=gamma**t*torch.where(alive,real_reward,torch.zeros_like(real_reward))
   if mask.any():reward_errors.extend((reward.detach()[mask]-real_reward[mask]).tolist())
   alive&=~(term|tout)
  survival=survival*(1-torch.sigmoid(logit));prev=action
  history=history[1:]+[analytic_observation(s,action,xi,lib.default_joint_pos)]
 frame=cmd.frame.clone();ref=reference_at(lib,frame);goal=lib.goal_window(frame,future)
 val=critic(wm.spec.canonical(s),goal,wm.spec.normalize_xi(xi),wm.heads.root_error_features(s,ref))
 objective=discounted+gamma**horizon*survival*val
 return objective,alive,{'max_state_forward_error':max_state_error,'max_observation_error':max_obs_error,
  'mean_absolute_analytic_vs_sim_reward':sum(abs(x) for x in reward_errors)/len(reward_errors) if reward_errors else None,
  'sim_reward_mean':float(real_sum[alive].mean()) if alive.any() else None,'soft_survival_mean':float(survival.detach()[alive].mean()) if alive.any() else None}

def main():
 p=argparse.ArgumentParser();p.add_argument('--ckpt',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
 p.add_argument('--npz-dir',type=Path,default=Path('data/npz/lafan1'));p.add_argument('--clips',nargs='+',default=['walk1_subject1','dance1_subject1','jumps1_subject1'])
 p.add_argument('--eval-clips',nargs='+',default=None,help='distinct clips for evaluation only; their data may have appeared in pretraining')
 p.add_argument('--seed',type=int,default=2500);p.add_argument('--frames-per-clip',type=int,default=2);p.add_argument('--warmup',type=int,default=10)
 p.add_argument('--heldout-frames-per-clip',type=int,default=0,help='evaluate on separately sampled, disjoint reference frames')
 p.add_argument('--horizon',type=int,default=8);p.add_argument('--eps',nargs='+',type=float,default=[.02]);p.add_argument('--output',type=Path,default=Path('docs/analysis/g1_grounded_gradient.json'))
 p.add_argument('--long-eval-horizon',type=int,default=None,help='continue all simulator branches and compare truncated real reward without bootstrap')
 p.add_argument('--include-base',action='store_true',help='add an unperturbed-policy branch for paired update evaluation')
 p.add_argument('--shared-origin',action='store_true',help='set plane-environment origins to zero before reset, for diagnostic comparison only')
 a=p.parse_args()
 if a.frames_per_clip<1 or a.heldout_frames_per_clip<0 or a.warmup<10 or a.horizon<1 or any(e<=0 for e in a.eps) or (a.long_eval_horizon is not None and a.long_eval_horizon<a.horizon):raise ValueError('invalid configuration')
 if a.eval_clips and (a.heldout_frames_per_clip<1 or set(a.clips)&set(a.eval_clips)):raise ValueError('evaluation clips must be disjoint and heldout-frames-per-clip positive')
 if a.include_base and a.long_eval_horizon is None:raise ValueError('base comparison requires a long evaluation horizon')
 started=time.perf_counter();ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);pcfg,wcfg=config_from_checkpoint(ck)
 if ck['config']['env'].get('anchor_obs',False):raise NotImplementedError('anchor observations not supported')
 ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
 policy,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg);policy.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic']);policy.eval();critic.eval().requires_grad_(False)
 requested_clips=a.clips+(a.eval_clips or []);files=discover_npz(a.npz_dir,names=requested_clips)
 if len(files)!=len(requested_clips):raise ValueError('incomplete or repeated clip list')
 # Load clip boundaries in an initial real environment and force common start frames.
 # Library metadata is accessed via the environment; deterministic random frames are generated there.
 future=int(ck['config']['env']['future_frames']);hist=int(ck['config']['policy_trainer']['history']);gamma=float(ck['config']['policy_trainer']['gamma'])
 bootstrap_cfg=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=1,future_frames=future,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',record_transitions=False,seed=a.seed,anchor_obs=False)
 meta_env=make_env(bootstrap_cfg,device='cpu')
 try:
  clips=meta_env.command_manager.get_term('motion').library.clips;rng=torch.Generator().manual_seed(a.seed);frame_parts=[];eval_parts=[];gradient_labels=[];eval_labels=[]
  for clip in clips:
   avail=clip.length-a.warmup-max(a.horizon,a.long_eval_horizon or a.horizon)-2
   if avail<=0:raise ValueError('clip too short')
   selected=torch.empty(0,dtype=torch.long)
   if clip.name in a.clips:
    selected=torch.randint(0,avail,(a.frames_per_clip,),generator=rng);frame_parts.append(clip.start+selected)
    gradient_labels.extend([clip.name]*a.frames_per_clip)
   if a.heldout_frames_per_clip and (a.eval_clips is None or clip.name in a.eval_clips):
    candidates=torch.ones(avail,dtype=torch.bool);candidates[selected]=False
    remaining=candidates.nonzero().squeeze(-1)
    if len(remaining)<a.heldout_frames_per_clip:raise ValueError('not enough held-out frames')
    eval_parts.append(clip.start+remaining[torch.randperm(len(remaining),generator=rng)[:a.heldout_frames_per_clip]])
    eval_labels.extend([clip.name]*a.heldout_frames_per_clip)
  frames=torch.cat(frame_parts)
  eval_frames=torch.cat(eval_parts) if eval_parts else frames
  if not eval_parts:eval_labels=gradient_labels
 finally:meta_env.close();del meta_env;gc.collect()
 n=len(frames);env=None
 try:
  env,obs,bad,start,io,ctx,cmd,hs,ha=make_setup(files,frames,1,a.seed,a.warmup,hist,future,policy,a.shared_origin)
  lib=cmd.library;robot=env.scene['robot'];heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
   robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
  wm=WorldModel(wcfg,heads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
  _,_,model_reward,model_boot,_=model_objective(wm,policy,critic,lib,start,a.horizon,future,gamma)
  hybrid_objective,hybrid_alive,hybrid_metrics=true_forward_objective(env,obs,start,wm,policy,critic,lib,io,ctx,cmd,a.horizon,future,gamma,~bad)
  if hybrid_metrics['max_state_forward_error']>1e-5 or hybrid_metrics['max_observation_error']>1e-3:
   raise ValueError(f'grounded forward or history does not match simulator: {hybrid_metrics}')
  mask=~bad&hybrid_alive
  gradient_exclusions={'warmup_reset':int(bad.sum()),'post_warmup_failure_or_timeout':int((~bad&~hybrid_alive).sum())}
  if not mask.any():raise ValueError('no warmup and grounded-rollout survivors')
  params=tuple(policy.parameters())
  gm=torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad((model_reward+model_boot)[mask].mean(),params)])
  gh=torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(hybrid_objective[mask].mean(),params)])
  if not torch.isfinite(gm).all() or not torch.isfinite(gh).all() or min(float(gm.norm()),float(gh.norm()))<1e-12:raise ValueError('invalid gradient')
  directions=[gm/gm.norm(),gh/gh.norm()];cos=float(torch.nn.functional.cosine_similarity(gm.double(),gh.double(),dim=0))
  initial_state=start.state.detach().clone()
  initial_frame=start.frame.clone();initial_xi=start.xi.clone();initial_mask=mask.clone();base=parameters_to_vector(policy.parameters()).detach().clone()
 finally:
  if env is not None:env.close()
 del env,obs,io,ctx,cmd,hs,ha,model_reward,model_boot,hybrid_objective
 gc.collect()
 env=None
 try:
  groups=4*len(a.eps)+2*int(a.include_base);m=len(eval_frames);env,obs,bad,start2,io,ctx,cmd,hs,ha=make_setup(files,eval_frames,groups,a.seed,a.warmup,hist,future,policy,a.shared_origin);lib=cmd.library
  origin_max=float(env.scene.env_origins.abs().max())
  if a.shared_origin and origin_max!=0:raise ValueError('shared origin was not applied')
  mismatch=torch.zeros(m,dtype=torch.bool);xi2=ctx.read()
  for k in range(groups):
   sl=slice(k*m,(k+1)*m)
   mismatch|=wm.spec.normalized_sq_error(hs[-1,:m],hs[-1,sl])>1e-8
   mismatch|=(cmd.frame[sl]!=cmd.frame[:m])|((xi2[sl]-xi2[:m]).abs().amax(-1)>1e-5)
   mismatch|=((obs['policy'][sl]-obs['policy'][:m]).abs().amax(-1)>1e-3)|((obs['goal'][sl]-obs['goal'][:m]).abs().amax(-1)>1e-4)
  eligible=~bad.reshape(groups,m).any(0)&~mismatch
  evaluation_exclusions={'warmup_reset':int(bad.reshape(groups,m).any(0).sum()),'initial_mismatch':int(mismatch.sum())}
  if not a.heldout_frames_per_clip:
   mismatch=wm.spec.normalized_sq_error(initial_state,hs[-1,:m])>1e-8
   eligible&=initial_mask&~mismatch
   if eligible.any() and (not torch.equal(cmd.frame[:m][eligible],initial_frame[eligible]) or (xi2[:m][eligible]-initial_xi[eligible]).abs().max()>1e-5):raise ValueError('second environment differs at start')
  reconstructed=flatten_history_term_major(torch.stack([analytic_observation(start2.hist_states[i],start2.hist_actions[i],start2.xi,lib.default_joint_pos) for i in range(hist)]))
  initial_obs_err=float((obs['policy'][:m][eligible]-reconstructed[eligible]).abs().max()) if eligible.any() else float('nan')
  if initial_obs_err>1e-3:raise ValueError('evaluation observations differ from reconstructed history')
  if not eligible.any():raise ValueError('no matched evaluation starts')
  with torch.no_grad():
   rewards=torch.zeros(groups*m);alive=torch.ones(groups*m,dtype=torch.bool);failed=torch.zeros_like(alive);tout_seen=torch.zeros_like(alive);failure_step=torch.full((groups*m,),-1,dtype=torch.long)
   for t in range(a.horizon):
    acts=[]
    for k in range(groups):
     if a.include_base and k>=groups-2:params_vector=base
     else:
      method=k//(2*len(a.eps));j=(k//2)%len(a.eps);sign=1. if k%2==0 else -1.;params_vector=base+sign*a.eps[j]*directions[method]
     vector_to_parameters(params_vector,policy.parameters());sl=slice(k*m,(k+1)*m);acts.append(policy(obs['policy'][sl],obs['goal'][sl]))
    obs,reward,term,tout,_=env.step(torch.cat(acts));failure_step=torch.where(alive&term,torch.full_like(failure_step,t+1),failure_step);rewards,alive,failed,tout_seen=advance_true_rollout(rewards,alive,failed,tout_seen,reward,term,tout,gamma**t)
   state=io.read();ref=reference_at(lib,cmd.frame);v=critic(wm.spec.canonical(state),obs['goal'],wm.spec.normalize_xi(ctx.read()),wm.heads.root_error_features(state,ref))
   boot=torch.where(alive,gamma**a.horizon*v,torch.zeros_like(v));returns=rewards+boot;rows=[]
   common_good=eligible&~tout_seen.reshape(groups,m).any(0)
   for method,name in enumerate(('model','real_forward_model_backward')):
    for i,eps in enumerate(a.eps):
     k=method*2*len(a.eps)+2*i;plus=slice(k*m,(k+1)*m);minus=slice((k+1)*m,(k+2)*m)
     good=common_good;fd=(returns[plus]-returns[minus])/(2*eps);reward_fd=(rewards[plus]-rewards[minus])/(2*eps)
     rows.append({'method':name,'eps':eps,'valid_pairs':int(good.sum()),'failure_pairs':int((good&(failed[plus]|failed[minus])).sum()),
      'excluded_timeouts':int((eligible&~good).sum()),'real_fd_mean':float(fd[good].mean()) if good.any() else None,'real_fd_values':fd[good].tolist(),
      'real_reward_fd_mean':float(reward_fd[good].mean()) if good.any() else None,'real_reward_fd_values':reward_fd[good].tolist(),
      'indices':good.nonzero().squeeze(-1).tolist()})
   long_rows=[]
   if a.long_eval_horizon is not None:
    for t in range(a.horizon,a.long_eval_horizon):
     acts=[]
     for k in range(groups):
      if a.include_base and k>=groups-2:params_vector=base
      else:
       method=k//(2*len(a.eps));j=(k//2)%len(a.eps);sign=1. if k%2==0 else -1.;params_vector=base+sign*a.eps[j]*directions[method]
      vector_to_parameters(params_vector,policy.parameters());sl=slice(k*m,(k+1)*m);acts.append(policy(obs['policy'][sl],obs['goal'][sl]))
     obs,reward,term,tout,_=env.step(torch.cat(acts));failure_step=torch.where(alive&term,torch.full_like(failure_step,t+1),failure_step);rewards,alive,failed,tout_seen=advance_true_rollout(rewards,alive,failed,tout_seen,reward,term,tout,gamma**t)
    good=eligible&~tout_seen.reshape(groups,m).any(0)
    for method,name in enumerate(('model','real_forward_model_backward')):
     for i,eps in enumerate(a.eps):
      k=method*2*len(a.eps)+2*i;plus=slice(k*m,(k+1)*m);minus=slice((k+1)*m,(k+2)*m)
      fd=(rewards[plus]-rewards[minus])/(2*eps)
      long_rows.append({'method':name,'eps':eps,'valid_pairs':int(good.sum()),'excluded_timeouts':int((eligible&~good).sum()),
       'failure_pairs':int((good&(failed[plus]|failed[minus])).sum()),'real_truncated_reward_fd_mean':float(fd[good].mean()) if good.any() else None,
       'real_truncated_reward_fd_values':fd[good].tolist(),'indices':good.nonzero().squeeze(-1).tolist(),
       'pairs':[{'index':j,'clip':eval_labels[j],'frame':int(eval_frames[j]),'reward_plus':float(rewards[plus][j]),'reward_minus':float(rewards[minus][j]),
        'failure_step_plus':int(failure_step[plus][j]),'failure_step_minus':int(failure_step[minus][j]),'survived_plus':bool(alive[plus][j]),'survived_minus':bool(alive[minus][j])} for j in good.nonzero().squeeze(-1).tolist()]})
      if a.include_base:
       b=slice((groups-2)*m,(groups-1)*m);delta=rewards[plus]-rewards[b]
       long_rows[-1]['vs_base']={'mean_reward_delta':float(delta[good].mean()) if good.any() else None,'reward_delta_values':delta[good].tolist(),
        'higher_reward_count':int((good&(delta>0)).sum()),'base_failure_count':int((good&failed[b]).sum()),'updated_failure_count':int((good&failed[plus]).sum()),
        'pairs':[{'index':j,'reward_base':float(rewards[b][j]),'failure_step_base':int(failure_step[b][j]),'survived_base':bool(alive[b][j])} for j in good.nonzero().squeeze(-1).tolist()]}
   base_control=None
   if a.include_base:
    b=slice((groups-2)*m,(groups-1)*m);c=slice((groups-1)*m,groups*m);control_good=eligible&~tout_seen.reshape(groups,m).any(0);err=(rewards[b]-rewards[c]).abs()
    base_control={'valid_pairs':int(control_good.sum()),'mean_abs_reward_difference':float(err[control_good].mean()) if control_good.any() else None,
     'max_abs_reward_difference':float(err[control_good].max()) if control_good.any() else None,'failure_disagreement_count':int((control_good&(failed[b]!=failed[c])).sum()),
     'failure_step_differences':(failure_step[b][control_good]-failure_step[c][control_good]).tolist(),
     'pairs':[{'index':j,'clip':eval_labels[j],'frame':int(eval_frames[j]),'reward_first':float(rewards[b][j]),'reward_second':float(rewards[c][j]),
      'failure_step_first':int(failure_step[b][j]),'failure_step_second':int(failure_step[c][j])} for j in control_good.nonzero().squeeze(-1).tolist()]}
   vector_to_parameters(base,policy.parameters())
  out={'checkpoint':str(a.ckpt),'seed':a.seed,'horizon':a.horizon,'shared_origin':a.shared_origin,'max_absolute_origin':origin_max,'candidate_starts':n,'gradient_starts':int(initial_mask.sum()),'gradient_exclusions':gradient_exclusions,'evaluation_candidate_starts':m,'evaluation_starts':int(eligible.sum()),'evaluation_exclusions':evaluation_exclusions,'heldout_reference_frames':bool(a.heldout_frames_per_clip),'gradient_clips':gradient_labels,'evaluation_clips':eval_labels,'evaluation_clip_disjoint':bool(a.eval_clips),'gradient_frames':frames.tolist(),'evaluation_frames':eval_frames.tolist(),
   'gradient_cosine':cos,'model_gradient_norm':float(gm.norm()),'grounded_gradient_norm':float(gh.norm()),'initial_obs_max_error':initial_obs_err,
   'simulator_transition_counts':{'grounded_gradient':n*a.horizon,'finite_difference_evaluation':groups*m*(a.long_eval_horizon or a.horizon),'warmup':(n+groups*m)*a.warmup},
   'true_forward_checks':hybrid_metrics,'rows':rows,'long_eval_horizon':a.long_eval_horizon,'long_rows':long_rows,'base_control':base_control,'wall_seconds':time.perf_counter()-started,
   'notes':'Fixed policy and model; true-forward uses detached simulator states and learned model Jacobians, analytic rewards and learned soft termination. Gradient starts condition on baseline real survival; branches include real failures and exclude timeouts. Real targets include same possibly biased critic bootstrap. No gradient through simulator; no training or equal-sample claim.'}
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2,allow_nan=False))
 finally:
  vector_to_parameters(base,policy.parameters())
  if env is not None:env.close()
if __name__=='__main__':main()
