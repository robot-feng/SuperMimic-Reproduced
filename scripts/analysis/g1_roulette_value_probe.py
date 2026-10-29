"""Frozen-policy G1 value targets with independent geometric truncation."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import torch
from clean_command_noise import replace_execution
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_noisy_imagination_direction import frame_samples
from g1_online_cv_pilot import warmup
from g1_real_score_probe_crossfit import tensor_hash
from g1_tail_policy_switch_probe import sha
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY,ROOT_BODY,TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic,CriticCfg,Policy
from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
from supermimic.world_model.state_spec import StateSpec
from supermimic.world_model.wm_trainer import reference_at

torch.set_num_threads(1)


def geometric_lengths(n,q,generator):
    if n<1 or not 0<q<1:raise ValueError('invalid geometric parameters')
    u=torch.rand(n,generator=generator,dtype=torch.float64)
    return torch.ceil(torch.log1p(-u)/math.log(q)).clamp_min(1).long()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();failed_path=a.output.with_suffix('.failed.json')
    if a.output.exists() or failed_path.exists():raise FileExistsError('refusing overwrite')
    ckpt=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt')
    source_paths=[Path(__file__),Path('scripts/analysis/g1_online_cv_pilot.py'),Path('scripts/analysis/g1_noisy_imagination_direction.py'),
                  Path('scripts/analysis/g1_grounded_gradient_probe.py'),Path('scripts/analysis/clean_command_noise.py'),ckpt]
    sources={str(p):sha(p) for p in source_paths}
    ck=torch.load(ckpt,map_location='cpu',weights_only=False,pickle_module=checkpoint_pickle_module(os.name!='nt'))
    pcfg,wcfg=config_from_checkpoint(ck);ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg);actor.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic'])
    actor.eval().requires_grad_(False);critic.eval().requires_grad_(False)
    history=int(ck['config']['policy_trainer']['history']);future=int(ck['config']['env']['future_frames'])
    gamma=float(ck['config']['policy_trainer']['gamma']);q=gamma
    if abs(gamma-.99)>1e-9:raise ValueError('unexpected gamma')
    a.clips=['dance2_subject1','jumps1_subject2','walk2_subject1'];a.eval_clips=['walk1_subject1']
    a.frames_per_clip=4;a.eval_frames_per_clip=1;a.horizon=a.eval_horizon=200;a.warmup=10
    files=discover_npz(Path('data/npz/lafan1'),names=a.clips+a.eval_clips)
    if len(files)!=4:raise ValueError('missing clips')
    frames,_,labels,_=frame_samples(files,a,future);frames=frames.repeat(2);n=len(frames)
    if n!=24:raise ValueError('unexpected start count')
    env=None;steps=0;loop=0
    try:
        env,_,_,_,io,ctx,cmd,_,_=make_setup(files,frames,1,a.seed,0,history,future,actor,True)
        lib=cmd.library;robot=env.scene['robot'];spec=StateSpec(wcfg.state_spec,'cpu')
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        term=env.action_manager.get_term('joint_pos');original=term.process_actions
        if term.noise_std!=0:raise ValueError('automatic noise enabled')
        injection={'noise':torch.zeros(n,29),'capture_terminal':False,'pre_reset':None,'max_clean':0.,'max_exec':0.}
        def inject(clean):
            original(clean);replace_execution(term,clean,injection['noise'])
            injection['max_clean']=max(injection['max_clean'],float((term.raw_action-clean).abs().max()))
            injection['max_exec']=max(injection['max_exec'],float((term.executed_raw-clean-injection['noise']).abs().max()))
        term.process_actions=inject;reset=env._reset_idx
        def capture(ids):
            if injection['capture_terminal']:
                injection['pre_reset']={'ids':ids.clone(),'state':io.read()[ids].clone(),'reward':env.reward_buf[ids].clone()}
            return reset(ids)
        env._reset_idx=capture
        obs,start,warm_fail,warm_envs=warmup(env,actor,io,ctx,cmd,frames,history,a.warmup,injection,True);steps+=n*a.warmup
        recon=flatten_history_term_major(torch.stack([analytic_observation(start.hist_states[t],start.hist_actions[t],start.xi,lib.default_joint_pos) for t in range(history)]))
        obs_error=float((recon-obs['policy']).abs().max())
        pair_error=max(float((start.state[:12]-start.state[12:]).abs().max()),float((start.xi[:12]-start.xi[12:]).abs().max()),
            float((start.hist_states[:,:12]-start.hist_states[:,12:]).abs().max()),float((start.hist_actions[:,:12]-start.hist_actions[:,12:]).abs().max()),
            float((obs['policy'][:12]-obs['policy'][12:]).abs().max()),float((obs['goal'][:12]-obs['goal'][12:]).abs().max()))
        if obs_error>1e-3 or pair_error>1e-4:raise ValueError(f'unpaired initial states {obs_error} {pair_error}')
        initial_hash=tensor_hash(start.state,start.xi,start.frame,start.prev_action,start.hist_states,start.hist_actions,obs['policy'],obs['goal'])
        def values(obs):
            state=io.read();ref=reference_at(lib,cmd.frame)
            return critic(spec.canonical(state),obs['goal'],spec.normalize_xi(ctx.read()),heads.root_error_features(state,ref))
        with torch.no_grad():
            initial_v=values(obs).clone();current=initial_v.clone()
            if float((initial_v[:12]-initial_v[12:]).abs().max())>1e-5:raise ValueError('initial value mismatch')
            rr=initial_v.clone();fixed=torch.zeros(n);boot200=torch.zeros(n)
            alive=torch.ones(n,dtype=torch.bool);rr_fail=torch.zeros(n,dtype=torch.bool);fixed_fail=torch.zeros(n,dtype=torch.bool)
            rr_steps=torch.zeros(n,dtype=torch.long);fixed_steps=torch.zeros(n,dtype=torch.long)
            cut=geometric_lengths(n,q,torch.Generator().manual_seed(19000000+a.seed))
            cap=2000;maxloop=min(cap,max(200,int(cut.max())))
            tape=.1*torch.randn(maxloop,n,29,generator=torch.Generator().manual_seed(20000000+a.seed))
            rate_scale=heads.reward_cfg.dt*heads.reward_cfg.w_action_rate;rate_error=0.;terminal_error=0.
            for t in range(maxloop):
                ractive=alive & (t<cut);hactive=alive & (t<200);needed=ractive|hactive
                if not needed.any():break
                action=actor(obs['policy'],obs['goal'])
                expected_rate=rate_scale*(action-obs['policy'][:,-29:]).square().sum(-1)
                injection['noise']=tape[t];injection['pre_reset']=None;injection['capture_terminal']=True
                next_obs,reward,done,timeout,_=env.step(action);injection['capture_terminal']=False
                steps+=n;loop+=1
                if (needed&timeout).any():raise ValueError('timeout in evaluated trajectory')
                ids=(needed&done).nonzero().flatten()
                if ids.numel():
                    capture_data=injection['pre_reset']
                    if capture_data is None:raise ValueError('missing terminal capture')
                    matched=torch.isin(capture_data['ids'],ids)
                    if not torch.equal(capture_data['ids'][matched],ids):raise ValueError('terminal capture mismatch')
                    terminal_error=max(terminal_error,float((capture_data['reward'][matched]-reward[ids]).abs().max()))
                    if not torch.isfinite(capture_data['state'][matched]).all():raise ValueError('nonfinite terminal state')
                actual_rate=env.reward_manager.get_term('action_rate_l2')
                rate_error=max(rate_error,float((expected_rate[needed]-actual_rate[needed]).abs().max()))
                next_v=values(next_obs);surviving=~(done|timeout)
                if not torch.isfinite(reward[needed]).all() or not torch.isfinite(current[needed]).all() or not torch.isfinite(next_v[needed&surviving]).all():
                    raise ValueError('nonfinite value or reward')
                delta=reward+gamma*torch.where(surviving,next_v,torch.zeros_like(next_v))-current
                rr+=torch.where(ractive,(gamma/q)**t*delta,torch.zeros_like(delta))
                fixed+=torch.where(hactive,gamma**t*reward,torch.zeros_like(reward))
                rr_steps+=ractive.long();fixed_steps+=hactive.long()
                rr_fail|=ractive&done;fixed_fail|=hactive&done
                alive &= surviving
                if t==199:boot200=torch.where(alive,gamma**200*next_v,torch.zeros_like(next_v))
                obs=next_obs;current=next_v
            if (alive & (loop<cut)).any():raise ValueError('roulette safety cap reached while alive')
            if rate_error>1e-5 or terminal_error>1e-6 or injection['max_exec']>1e-5 or injection['max_clean']>1e-6:raise ValueError('action/reward audit mismatch')
            records=[]
            for i in range(12):
                branches=[]
                for j in (i,i+12):
                    branches.append({'initial_value':float(initial_v[j]),'roulette_target':float(rr[j]),'roulette_length':int(cut[j]),
                        'roulette_effective_steps':int(rr_steps[j]),'roulette_failed':bool(rr_fail[j]),
                        'fixed_reward':float(fixed[j]),'fixed_bootstrap':float(boot200[j]),'fixed_target':float(fixed[j]+boot200[j]),
                        'fixed_effective_steps':int(fixed_steps[j]),'fixed_failed':bool(fixed_fail[j])})
                records.append({'index':i,'clip':labels[i],'forced_frame':int(frames[i]),'branches':branches})
        if any(sha(Path(path))!=digest for path,digest in sources.items()):raise ValueError('source changed')
        output={'seed':a.seed,'gamma':gamma,'q':q,'warmup':a.warmup,'horizon_fixed':200,'cutoff_cap':2000,
                'initial_sha256':initial_hash,'noise_tape_sha256':tensor_hash(tape),'cutoff_sha256':tensor_hash(cut),
                'paired_start_error':pair_error,'obs_reconstruction_error':obs_error,'warmup_failures':warm_fail,'warmup_failed_envs':warm_envs,
                'action_checks':{'max_clean':injection['max_clean'],'max_exec':injection['max_exec'],'max_action_rate':rate_error,'max_terminal_reward':terminal_error},
                'simulator_steps_executed':steps,'vectorized_rollout_iterations':loop,'roulette_effective_steps':int(rr_steps.sum()),
                'fixed_effective_steps':int(fixed_steps.sum()),'source_sha256':sources,'records':records,
                'notes':'Two independent action-noise and roulette streams per forced reference frame, frozen actor and critic. Rolling continues until both targets end or failure; full vectorized environment calls include masked transitions. No learned dynamics rollout.'}
        a.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps({'seed':a.seed,'simulator_steps':steps,'rr_effective_steps':int(rr_steps.sum()),'rr_failed':int(rr_fail.sum()),'maxloop':loop}))
    except Exception as exc:
        failed_path.write_text(json.dumps({'seed':a.seed,'error':repr(exc),'simulator_steps_executed':steps,'rollout_iterations':loop},indent=2),encoding='utf8')
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
