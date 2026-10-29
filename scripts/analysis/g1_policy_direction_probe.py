"""G1 matched policy-parameter directional probe, model vs simulator (CPU).

Pairs share reference initialization and warm-up actions, then use policy
parameters +/- eps * normalized model policy gradient. Retains failures with
terminal reward and zero bootstrap; excludes timeouts. Not a policy update.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1');os.environ.setdefault('WARP_CACHE_PATH',r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from g1_directional_probe import config_from_checkpoint
from policy_direction_math import objective_components,advance_true_rollout
from g1_foot_geometry import FootClearance
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY,FOOT_GEOM_REGEX,ROOT_BODY,TRACKED_BODIES
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic,CriticCfg,Policy
from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)

def model_objective(wm,policy,critic,lib,starts,horizon,future,gamma):
 ro=imagine(wm,policy,lib,starts,horizon,future,obs_noise=False,anchor_obs=False)
 xi=wm.spec.normalize_xi(starts.xi)
 vh=critic(wm.spec.canonical(ro.states[-1]),ro.goals[-1],xi,ro.e_root[-1])
 reward,bootstrap=objective_components(ro,vh,gamma)
 return policy_objective(ro,vh,gamma),float(ro.survival[-1].mean().detach()),reward,bootstrap,ro.states[-1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--ckpt',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
 p.add_argument('--npz-dir',type=Path,default=Path('data/npz/lafan1'));p.add_argument('--clips',nargs='+',default=['walk1_subject1','dance1_subject1','jumps1_subject1'])
 p.add_argument('--seed',type=int,default=91);p.add_argument('--frames-per-clip',type=int,default=3);p.add_argument('--warmup',type=int,default=10)
 p.add_argument('--horizon',type=int,default=8);p.add_argument('--eps',nargs='+',type=float,default=[.02,.1]);p.add_argument('--max-initial-nse',type=float,default=1e-8)
 p.add_argument('--foot-clearance',action='store_true',help='record paired endpoint capsule-plane gaps for surviving simulator branches')
 p.add_argument('--direction-objective',choices=['full','reward'],default='full',help='choose model objective differentiated to obtain the shared parameter direction')
 p.add_argument('--compare-gradient-components',action='store_true',help='measure reward and critic-bootstrap gradient norms and alignment')
 p.add_argument('--true-eval-horizon',type=int,default=None,help='optionally continue real paired branches to this many steps without value bootstrap')
 p.add_argument('--output',type=Path,default=Path('docs/analysis/g1_policy_direction_probe.json'));a=p.parse_args()
 if a.warmup<10 or any(x<=0 for x in a.eps) or (a.true_eval_horizon is not None and a.true_eval_horizon<a.horizon):p.error('warmup >=10, positive eps, true-eval-horizon >= horizon required')
 t0=time.perf_counter();ck=torch.load(a.ckpt,map_location='cpu',weights_only=False);pcfg,wcfg=config_from_checkpoint(ck)
 if ck['config']['env'].get('anchor_obs',False):raise NotImplementedError('probe supports anchor_obs=False only')
 ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
 policy,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg);policy.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic']);policy.eval();critic.eval().requires_grad_(False)
 files=discover_npz(a.npz_dir,names=a.clips)
 if len(files)!=len(a.clips):raise ValueError('incomplete clips')
 n=len(files)*a.frames_per_clip;groups=2*len(a.eps);future=int(ck['config']['env']['future_frames']);gamma=float(ck['config']['policy_trainer']['gamma'])
 cfg=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=n*groups,future_frames=future,action_noise_std=0.,obs_noise=False,
   reset_noise=False,domain_randomization=False,sampling_mode='start',record_transitions=False,seed=a.seed,anchor_obs=False)
 env=make_env(cfg,device='cpu')
 try:
  cmd=env.command_manager.get_term('motion');lib=cmd.library;io=StateIO(env);ctx=SimContext(env,FOOT_GEOM_REGEX,'torso_link');robot=env.scene['robot']
  mj=load_g1_model();geometry=FootClearance(mj) if a.foot_clearance else None
  if geometry is not None and io.origins[:,2].abs().max()>1e-6:raise ValueError('expected flat plane at origin-relative height zero')
  heads=AnalyticHeads(geometry.fk if geometry is not None else G1Kinematics(mj,device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
  wm=WorldModel(wcfg,heads,'cpu');wm.load_state_dict(ck['wm']);wm.eval().requires_grad_(False)
  rng=torch.Generator().manual_seed(a.seed);frames=[];labels=[]
  for clip in lib.clips:
   avail=clip.length-a.warmup-max(a.horizon,a.true_eval_horizon or a.horizon)-2
   if avail<=0:raise ValueError('clip too short')
   frames.append(clip.start+torch.randint(0,avail,(a.frames_per_clip,),generator=rng));labels.extend([clip.name]*a.frames_per_clip)
  frames=torch.cat(frames);cmd.forced_frame[:]=frames.repeat(groups);obs,_=env.reset();bad=torch.zeros(env.num_envs,dtype=torch.bool)
  history=[];actions=[]
  with torch.no_grad():
   for t in range(a.warmup+1):
    history.append(io.read().clone());actions.append(env.action_manager.action.clone())
    if t<a.warmup:
     shared=policy(obs['policy'][:n],obs['goal'][:n]).repeat(groups,1)
     obs,_,term,tout,_=env.step(shared);bad|=term|tout
   state=history[-1];xi=ctx.read();mismatch=torch.zeros(n,dtype=torch.bool)
   for k in range(1,groups):
    mismatch|=wm.spec.normalized_sq_error(state[:n],state[k*n:(k+1)*n])>a.max_initial_nse
   eligible=~(bad.reshape(groups,n).any(0)|mismatch);idx=eligible.nonzero().squeeze(-1)
   if len(idx)==0:raise RuntimeError('no matched non-reset initial states')
   h=int(ck['config']['policy_trainer']['history']);hist_states=torch.stack(history[-h:])[:,:n][:,idx]
   hist_actions=torch.stack(actions[-h:])[:,:n][:,idx]
   starts=StartBatch(state=state[:n][idx],xi=xi[:n][idx],frame=cmd.frame[:n][idx],prev_action=actions[-1][:n][idx],hist_states=hist_states,hist_actions=hist_actions)
   hist_obs=torch.stack([analytic_observation(starts.hist_states[t],starts.hist_actions[t],starts.xi,lib.default_joint_pos) for t in range(h)])
   obs_err=float((flatten_history_term_major(hist_obs)-obs['policy'][:n][idx]).abs().max())
   goal_err=float((lib.goal_window(starts.frame,future)-obs['goal'][:n][idx]).abs().max())
   if obs_err > 1e-3 or goal_err > 1e-4:
    raise RuntimeError(f'initial policy inputs do not match: obs={obs_err}, goal={goal_err}')
  base=parameters_to_vector(policy.parameters()).detach().clone()
  j,survival,reward_part,bootstrap_part,last_state=model_objective(wm,policy,critic,lib,starts,a.horizon,future,gamma)
  components=None;params=tuple(policy.parameters())
  if a.compare_gradient_components:
   full_grad=torch.cat([z.detach().reshape(-1) for z in torch.autograd.grad(j,params,retain_graph=True)])
   reward_grad=torch.cat([z.detach().reshape(-1) for z in torch.autograd.grad(reward_part.mean(),params)])
   boot_grad=full_grad-reward_grad
   def cosine(x,y):
    denom=x.norm()*y.norm()
    return float(torch.dot(x,y)/denom) if denom>1e-12 else None
   components={'full_norm':float(full_grad.norm()),'reward_norm':float(reward_grad.norm()),'bootstrap_norm':float(boot_grad.norm()),
    'reward_vs_bootstrap_cosine':cosine(reward_grad,boot_grad),'reward_vs_full_cosine':cosine(reward_grad,full_grad)}
   g=reward_grad if a.direction_objective=='reward' else full_grad
  else:
   if a.direction_objective=='reward':j=reward_part.mean()
   grad=torch.autograd.grad(j,params);g=torch.cat([z.detach().reshape(-1) for z in grad])
  gn=float(g.norm())
  if not torch.isfinite(g).all() or gn<1e-12:raise RuntimeError('nonfinite or vanishing model gradient')
  direction=g/g.norm();del j,reward_part,bootstrap_part,last_state
  model_diffs=[];model_gaps=[]
  with torch.no_grad():
   for eps in a.eps:
    vector_to_parameters(base+eps*direction,policy.parameters());jp,_,rp,bp,sp=model_objective(wm,policy,critic,lib,starts,a.horizon,future,gamma)
    vector_to_parameters(base-eps*direction,policy.parameters());jm,_,rm,bm,sm=model_objective(wm,policy,critic,lib,starts,a.horizon,future,gamma)
    model_diffs.append(((rp-rm)/(2*eps),(bp-bm)/(2*eps)))
    if geometry is not None:model_gaps.append((geometry(sp).min(-1).values,geometry(sm).min(-1).values))
   vector_to_parameters(base,policy.parameters())
   rewards=torch.zeros(env.num_envs);alive=torch.ones(env.num_envs,dtype=torch.bool);failed=torch.zeros_like(alive);timed_out=torch.zeros_like(alive)
   failure_step=torch.full((env.num_envs,),-1,dtype=torch.long)
   for t in range(a.horizon):
    joint=[]
    for k in range(groups):
     eps=a.eps[k//2];sign=1. if k%2==0 else -1.
     vector_to_parameters(base+sign*eps*direction,policy.parameters())
     sl=slice(k*n,(k+1)*n);joint.append(policy(obs['policy'][sl],obs['goal'][sl]))
    obs,reward,term,tout,_=env.step(torch.cat(joint))
    failure_step=torch.where(alive&term,torch.full_like(failure_step,t+1),failure_step)
    rewards,alive,failed,timed_out=advance_true_rollout(rewards,alive,failed,timed_out,reward,term,tout,gamma**t)
   final=io.read();ref=reference_at(lib,cmd.frame);er=heads.root_error_features(final,ref)
   v=critic(wm.spec.canonical(final),obs['goal'],wm.spec.normalize_xi(ctx.read()),er)
   bootstrap=torch.where(alive,gamma**a.horizon*v,torch.zeros_like(v));returns=rewards+bootstrap
   real_gaps=geometry(final).min(-1).values if geometry is not None else None
   result_groups=[]
   for i,eps in enumerate(a.eps):
    plus=slice(2*i*n,(2*i+1)*n);minus=slice((2*i+1)*n,(2*i+2)*n)
    good=eligible&~(timed_out[plus]|timed_out[minus]);survivors=good&~(failed[plus]|failed[minus]);fd=(returns[plus]-returns[minus])/(2*eps)
    mr=model_diffs[i][0][good[idx]];mb=model_diffs[i][1][good[idx]];matched_model=mr+mb
    real_reward=((rewards[plus]-rewards[minus])/(2*eps))[good]
    real_boot=((bootstrap[plus]-bootstrap[minus])/(2*eps))[good]
    informative=(mr.abs()>1e-5)&(real_reward.abs()>1e-5)
    row={'eps':eps,'valid_pairs':int(good.sum()),'excluded_timeouts':int((eligible&~good).sum()),
      'failure_pairs':int((good&(failed[plus]|failed[minus])).sum()),'survivor_pairs':int(survivors.sum()),
      'survivor_model_fd':float((model_diffs[i][0]+model_diffs[i][1])[survivors[idx]].mean()) if survivors.any() else None,
      'survivor_real_fd':float(fd[survivors].mean()) if survivors.any() else None,
      'model_fd_all_initial':float((model_diffs[i][0]+model_diffs[i][1]).mean()),'model_fd_matched':float(matched_model.mean()) if good.any() else None,
      'model_fd_values':matched_model.tolist(),'sign_agreement':float(((matched_model*fd[good])>0).float().mean()) if good.any() else None,
      'model_reward_fd_values':mr.tolist(),'real_reward_fd_values':real_reward.tolist(),
      'model_reward_fd_mean':float(mr.mean()) if good.any() else None,
      'real_reward_fd_mean':float(real_reward.mean()) if good.any() else None,
      'model_bootstrap_fd_values':mb.tolist(),'real_bootstrap_fd_values':real_boot.tolist(),
      'reward_informative_pairs':int(informative.sum()),'reward_sign_agreement':float(((mr[informative]*real_reward[informative])>0).float().mean()) if informative.any() else None,
      'real_fd_mean':float(fd[good].mean()) if good.any() else None,
      'real_fd_median':float(torch.quantile(fd[good],.5)) if good.any() else None,
      'real_fd_values':fd[good].tolist(),'clips':[labels[j] for j in range(n) if good[j]]}
    if geometry is not None:
     paired=[]
     for j in survivors.nonzero().squeeze(-1).tolist():
      q=int((idx==j).nonzero()[0,0]);mp,mm=model_gaps[i][0][q],model_gaps[i][1][q]
      gp,gm=real_gaps[plus][j],real_gaps[minus][j]
      paired.append({'clip':labels[j],'model_gap_plus_m':float(mp),'model_gap_minus_m':float(mm),
       'real_gap_plus_m':float(gp),'real_gap_minus_m':float(gm),'mean_gap_error_m':float(((mp-gp)+(mm-gm))/2),
       'model_fd':float((model_diffs[i][0]+model_diffs[i][1])[q]),'real_fd':float(fd[j]),
       'model_reward_fd':float(model_diffs[i][0][q]),'real_reward_fd':float(((rewards[plus]-rewards[minus])/(2*eps))[j])})
     row['survivor_clearance_records']=paired
    result_groups.append(row)
   long_rows=[]
   if a.true_eval_horizon is not None:
    for t in range(a.horizon,a.true_eval_horizon):
     joint=[]
     for k in range(groups):
      eps=a.eps[k//2];sign=1. if k%2==0 else -1.
      vector_to_parameters(base+sign*eps*direction,policy.parameters());sl=slice(k*n,(k+1)*n)
      joint.append(policy(obs['policy'][sl],obs['goal'][sl]))
     obs,reward,term,tout,_=env.step(torch.cat(joint))
     failure_step=torch.where(alive&term,torch.full_like(failure_step,t+1),failure_step)
     rewards,alive,failed,timed_out=advance_true_rollout(rewards,alive,failed,timed_out,reward,term,tout,gamma**t)
    for i,eps in enumerate(a.eps):
     plus=slice(2*i*n,(2*i+1)*n);minus=slice((2*i+1)*n,(2*i+2)*n)
     good=eligible&~(timed_out[plus]|timed_out[minus]);fd=(rewards[plus]-rewards[minus])/(2*eps)
     long_rows.append({'eps':eps,'valid_pairs':int(good.sum()),'excluded_timeouts':int((eligible&~good).sum()),
      'failure_pairs':int((good&(failed[plus]|failed[minus])).sum()),'survivor_pairs':int((good&alive[plus]&alive[minus]).sum()),
      'failure_step_plus':failure_step[plus][good].tolist(),'failure_step_minus':failure_step[minus][good].tolist(),
      'real_long_reward_fd_mean':float(fd[good].mean()) if good.any() else None,'real_long_reward_fd_values':fd[good].tolist(),
      'clips':[labels[j] for j in range(n) if good[j]]})
   vector_to_parameters(base,policy.parameters())
  result={'checkpoint':str(a.ckpt),'seed':a.seed,'horizon':a.horizon,'warmup':a.warmup,'direction_objective':a.direction_objective,'candidate_pairs':n,
   'eligible_initial_pairs':int(eligible.sum()),'excluded_initial_mismatch':int(mismatch.sum()),'model_grad_norm':gn,'gradient_components':components,
   'reference_initial_frames':frames.tolist(),'eligible_initial_indices':idx.tolist(),'frame_after_warmup':starts.frame.tolist(),
   'initial_mean_model_survival':survival,'initial_obs_max_abs_error':obs_err,'initial_goal_max_abs_error':goal_err,
   'true_eval_horizon':a.true_eval_horizon,'long_reward_groups':long_rows,
   'groups':result_groups,'wall_seconds':time.perf_counter()-t0,
   'notes':'Real failures are retained, terminal reward counted and bootstrap zero; timeouts are excluded. Model full objective and short real total use fixed critic bootstrap; long_reward_groups are finite-horizon true reward only, without bootstrap. Model uses soft termination. Small correlated sample from deterministic reference resets/fixed friction; simulator body kinematics lag final state one substep.'}
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
 finally:env.close()
if __name__=='__main__':main()
