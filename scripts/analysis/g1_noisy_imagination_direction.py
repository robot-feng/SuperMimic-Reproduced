"""G1 clean-vs-noisy imagined gradients with held-out real paired rollouts.

Fixed checkpoint diagnostic only. It evaluates theta +/- a normalized direction
and theta itself; no actor optimizer step, no world-model retraining, and no claim
of a training or sample-efficiency improvement. CPU flat-ground worlds share an
origin. Within each real noise replicate all policy branches share noise.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WARP_CACHE_PATH', r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from clean_command_noise import replace_execution
from g1_tail_switch_schedule import shared_prefix_noise, update_sign
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_policy_direction_probe import model_objective
from noisy_imag_step import NoisyImagStep
from policy_direction_math import advance_true_rollout, objective_components
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def select_and_repeat(start, indices, copies):
    return StartBatch(state=start.state[indices].repeat(copies, 1), xi=start.xi[indices].repeat(copies, 1),
                      frame=start.frame[indices].repeat(copies), prev_action=start.prev_action[indices].repeat(copies, 1),
                      hist_states=start.hist_states[:, indices].repeat(1, copies, 1),
                      hist_actions=start.hist_actions[:, indices].repeat(1, copies, 1))


def imagined_parts(wm, actor, critic, lib, start, horizon, future, gamma, tape):
    step = NoisyImagStep(wm, lib.default_joint_pos, tape)
    ro = imagine(wm, actor, lib, start, horizon, future, obs_noise=False, step_fn=step, anchor_obs=False)
    if step.index != horizon:
        raise ValueError('noise tape did not cover the rollout')
    xi = wm.spec.normalize_xi(start.xi)
    vh = critic(wm.spec.canonical(ro.states[-1]), ro.goals[-1], xi, ro.e_root[-1])
    return objective_components(ro, vh, gamma)


def objective_gradients(wm, actor, critic, lib, start, horizon, future, gamma, tape):
    reward, boot = imagined_parts(wm, actor, critic, lib, start, horizon, future, gamma, tape)
    params = tuple(actor.parameters())
    full = torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad((reward + boot).mean(), params, retain_graph=True)])
    immediate = torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(reward.mean(), params)])
    if not torch.isfinite(full).all() or float(full.norm()) < 1e-12:
        raise ValueError('invalid gradient')
    return full, immediate, {'reward_mean': float(reward.detach().mean()), 'bootstrap_mean': float(boot.detach().mean())}


def frame_samples(files, args, future):
    cfg = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=1, future_frames=future,
                               action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
                               sampling_mode='start', record_transitions=False, seed=args.seed, anchor_obs=False)
    env = make_env(cfg, device='cpu')
    try:
        gen = torch.Generator().manual_seed(args.seed)
        gradient, evaluation, gradient_clips, evaluation_clips = [], [], [], []
        for clip in env.command_manager.get_term('motion').library.clips:
            available = clip.length - args.warmup - max(args.horizon, args.eval_horizon) - 2
            if available < 1:
                raise ValueError('clip too short')
            if clip.name in args.clips:
                gradient.append(clip.start + torch.randint(available, (args.frames_per_clip,), generator=gen))
                gradient_clips += [clip.name] * args.frames_per_clip
            elif clip.name in args.eval_clips:
                evaluation.append(clip.start + torch.randint(available, (args.eval_frames_per_clip,), generator=gen))
                evaluation_clips += [clip.name] * args.eval_frames_per_clip
        if not gradient or not evaluation:
            raise ValueError('no gradient or evaluation frames')
        return torch.cat(gradient), torch.cat(evaluation), gradient_clips, evaluation_clips
    finally:
        env.close()


def evaluate(files, frames, labels, seed, actor, critic, wm, directions, base, args, sigma):
    """Return paired actual simulator rewards for both directions and baseline."""
    n_base = len(frames)
    frames = frames.repeat(args.eval_replicates)
    n = len(frames)
    groups = 6  # model +/-, noise-aware +/-, base and duplicate base
    switching = bool(getattr(args, 'tail_switch', False))
    shared_prefix = bool(getattr(args, 'shared_prefix_noise', False))
    long_bootstrap = bool(getattr(args, 'long_bootstrap', False))
    if shared_prefix and (not switching or args.eval_replicates < 2):
        raise ValueError('shared prefix requires switching and multiple replicates')
    if switching and (directions[0].shape != directions[1].shape or not torch.equal(directions[0], directions[1])):
        raise ValueError('switching pairs must share exactly the same update')
    env = None
    try:
        env, obs, bad, start, io, ctx, cmd, hs, _ = make_setup(files, frames, groups, seed, args.warmup,
                                                                args.history, args.future, actor, True)
        lib = cmd.library
        xi = ctx.read()
        mismatch = torch.zeros(n, dtype=torch.bool)
        for k in range(1, groups):
            sl = slice(k*n, (k+1)*n)
            mismatch |= wm.spec.normalized_sq_error(hs[-1, :n], hs[-1, sl]) > 1e-8
            mismatch |= (obs['policy'][:n] - obs['policy'][sl]).abs().amax(-1) > 1e-3
            mismatch |= (obs['goal'][:n] - obs['goal'][sl]).abs().amax(-1) > 1e-4
            mismatch |= (xi[:n] - xi[sl]).abs().amax(-1) > 1e-5
            mismatch |= cmd.frame[:n] != cmd.frame[sl]
        warmup_bad = bad.reshape(groups, n).any(0)
        eligible = ~(warmup_bad | mismatch)
        if not eligible.any():
            raise ValueError('no matched evaluation starts')
        obs_recon = flatten_history_term_major(torch.stack([
            analytic_observation(start.hist_states[t], start.hist_actions[t], start.xi, lib.default_joint_pos)
            for t in range(args.history)]))
        obs_error = float((obs_recon[eligible] - obs['policy'][:n][eligible]).abs().max())
        goal_error = float((lib.goal_window(start.frame, args.future)[eligible] - obs['goal'][:n][eligible]).abs().max())
        if obs_error > 1e-3 or goal_error > 1e-4:
            raise ValueError(f'initial history or goal mismatch {obs_error}, {goal_error}')
        initial = {'state': hs[-1, :n].clone(), 'obs': obs['policy'][:n].clone(),
                   'goal': obs['goal'][:n].clone(), 'eligible': eligible.clone()}

        term = env.action_manager.get_term('joint_pos')
        if term.noise_std != 0:
            raise ValueError('automatic noise must be disabled for paired noise')
        original_process = term.process_actions
        injected = {'noise': torch.zeros(groups*n, term.action_dim), 'calls': 0, 'clean_error': 0., 'executed_error': 0.}

        def process(actions):
            original_process(actions)
            replace_execution(term, actions, injected['noise'])
            injected['calls'] += 1
            injected['clean_error'] = max(injected['clean_error'], float((term.raw_action - actions).abs().max()))
            injected['executed_error'] = max(injected['executed_error'], float((term.executed_raw - actions - injected['noise']).abs().max()))

        term.process_actions = process
        original_reset_idx=env._reset_idx
        terminal_capture={'value':None}
        def observed_reset_idx(ids):
            terminal_capture['value']={'ids':ids.detach().clone(),'state':io.read()[ids].detach().clone(),
                                       'reward':env.reward_buf[ids].detach().clone()}
            return original_reset_idx(ids)
        env._reset_idx=observed_reset_idx
        noise_gen = torch.Generator().manual_seed(160000 + seed)
        tail_seed=getattr(args,'tail_noise_seed',None)
        if tail_seed is not None and (not switching or args.eval_replicates != 1):
            raise ValueError('separate tail seed requires one switching replicate')
        tail_gen = torch.Generator().manual_seed(tail_seed) if tail_seed is not None else None
        returns = torch.zeros(groups*n)
        alive = torch.ones(groups*n, dtype=torch.bool)
        failed = torch.zeros_like(alive)
        timeout = torch.zeros_like(alive)
        fail_step = torch.full((groups*n,), -1, dtype=torch.long)
        short = None;terminal_audits=[];switch_audit=None;replicate_audit=None;prefix_hashes=None
        with torch.no_grad():
            for t in range(args.eval_horizon):
                if alive.any() and (not torch.isfinite(io.read()[alive]).all() or
                    not torch.isfinite(obs['policy'][alive]).all() or not torch.isfinite(obs['goal'][alive]).all()):
                    raise ValueError('nonfinite active state or observation before step')
                acts = []
                for k in range(groups):
                    if switching:
                        sign=update_sign(k,t,args.horizon)
                        weights=base if sign==0 else base+sign*args.eps*directions[0]
                    elif k >= 4:
                        weights = base
                    else:
                        sign = 1. if k % 2 == 0 else -1.
                        weights = base + sign * args.eps * directions[k // 2]
                    vector_to_parameters(weights, actor.parameters())
                    sl = slice(k*n, (k+1)*n)
                    acts.append(actor(obs['policy'][sl], obs['goal'][sl]))
                generator=tail_gen if tail_gen is not None and t>=args.horizon else noise_gen
                noise = sigma * torch.randn((n, term.action_dim), generator=generator)
                # Draw the whole tensor even when the prefix is shared so later
                # random draws do not depend on this intervention.
                noise = shared_prefix_noise(noise,n_base,args.eval_replicates,t,args.horizon,shared_prefix)
                injected['noise'] = noise.repeat(groups, 1)
                terminal_capture['value']=None
                obs, r, ended, tout, _ = env.step(torch.cat(acts))
                if alive.any() and not torch.isfinite(r[alive]).all():raise ValueError('nonfinite active reward')
                terminated=(alive & (ended | tout)).nonzero().flatten()
                if terminated.numel():
                    captured=terminal_capture['value']
                    if captured is None:raise ValueError('missing pre-reset terminal capture')
                    keep=alive[captured['ids']]
                    if not torch.equal(captured['ids'][keep],terminated):raise ValueError('terminal id mismatch')
                    state=captured['state'][keep];reward=captured['reward'][keep]
                    if not torch.isfinite(state).all() or not torch.isfinite(reward).all():raise ValueError('nonfinite terminal before reset')
                    error=float((reward-r[terminated]).abs().max())
                    if error>1e-6:raise ValueError('terminal reward changed across reset')
                    terminal_audits.append({'step':t+1,'count':int(len(terminated)),'reward_reset_error':error,
                                            'state_abs_max':float(state.abs().max())})
                fail_step = torch.where(alive & ended, torch.full_like(fail_step, t+1), fail_step)
                returns, alive, failed, timeout = advance_true_rollout(returns, alive, failed, timeout, r, ended, tout, args.gamma**t)
                if t + 1 == args.horizon:
                    state = io.read()
                    if switching:
                        items=[]
                        for first,second in ((0,2),(1,3)):
                            i=slice(first*n,(first+1)*n);j=slice(second*n,(second+1)*n)
                            valid=eligible & alive[i] & alive[j]
                            if ((failed[i]!=failed[j]) & eligible).any() or ((timeout[i]!=timeout[j]) & eligible).any():
                                raise ValueError('matching switching branches terminated differently before switch')
                            def maxdiff(x,y,mask):
                                return float((x[mask]-y[mask]).abs().max()) if mask.any() else 0.
                            metrics={'surviving_pairs':int(valid.sum()),
                                     'state_max_abs':maxdiff(state[i],state[j],valid),
                                     'observation_max_abs':maxdiff(obs['policy'][i],obs['policy'][j],valid),
                                     'goal_max_abs':maxdiff(obs['goal'][i],obs['goal'][j],valid),
                                     'context_max_abs':maxdiff(ctx.read()[i],ctx.read()[j],valid),
                                     'reward_max_abs':maxdiff(returns[i],returns[j],eligible),
                                     'frame_mismatches':int(((cmd.frame[i]!=cmd.frame[j]) & valid).sum()),
                                     'survivor_indices':valid.nonzero().flatten().tolist()}
                            if (metrics['state_max_abs']>1e-5 or metrics['observation_max_abs']>1e-3 or
                                metrics['goal_max_abs']>1e-4 or metrics['context_max_abs']>1e-5 or
                                metrics['reward_max_abs']>1e-4 or metrics['frame_mismatches']):
                                raise ValueError(f'switch-point mismatch: {metrics}')
                            items.append(metrics)
                        switch_audit={'plus':items[0],'minus':items[1]}
                    if shared_prefix:
                        # Independent environments must actually reach the same
                        # extended state before interpreting later noise draws as
                        # conditional repetitions. Check all groups and copies.
                        maxima={'state_max_abs':0., 'observation_max_abs':0., 'goal_max_abs':0.,
                                'context_max_abs':0., 'reward_max_abs':0., 'frame_mismatches':0,
                                'termination_mismatches':0, 'checked_survivors':0}
                        current_xi=ctx.read()
                        for k in range(groups):
                            first=k*n
                            for rep in range(1,args.eval_replicates):
                                i=slice(first,first+n_base);j=slice(first+rep*n_base,first+(rep+1)*n_base)
                                if not torch.equal(eligible[:n_base],eligible[rep*n_base:(rep+1)*n_base]):
                                    raise ValueError('replicate warmup eligibility differs')
                                valid=eligible[:n_base] & alive[i] & alive[j]
                                maxima['termination_mismatches'] += int(((failed[i]!=failed[j]) | (timeout[i]!=timeout[j])).sum())
                                maxima['checked_survivors'] += int(valid.sum())
                                for key, x in (('state_max_abs',state),('observation_max_abs',obs['policy']),
                                               ('goal_max_abs',obs['goal']),('context_max_abs',current_xi),
                                               ('reward_max_abs',returns)):
                                    if valid.any():
                                        maxima[key]=max(maxima[key],float((x[i][valid]-x[j][valid]).abs().max()))
                                maxima['frame_mismatches'] += int(((cmd.frame[i]!=cmd.frame[j]) & valid).sum())
                        if (maxima['state_max_abs']>1e-5 or maxima['observation_max_abs']>1e-3 or
                            maxima['goal_max_abs']>1e-4 or maxima['context_max_abs']>1e-5 or
                            maxima['reward_max_abs']>1e-4 or maxima['frame_mismatches'] or
                            maxima['termination_mismatches']):
                            raise ValueError(f'replicate prefix mismatch: {maxima}')
                        replicate_audit=maxima
                    ref = reference_at(lib, cmd.frame)
                    val = critic(wm.spec.canonical(state), obs['goal'], wm.spec.normalize_xi(ctx.read()), wm.heads.root_error_features(state, ref))
                    boot = torch.where(alive, args.gamma**args.horizon * val, torch.zeros_like(val))
                    if tail_seed is not None:
                        values={'state':state,'observation':obs['policy'],'goal':obs['goal'],
                                'context':ctx.read(),'frame':cmd.frame,'reward':returns,'alive':alive,
                                'failed':failed,'timeout':timeout,'bootstrap':boot}
                        prefix_hashes={name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                       for name,value in values.items()}
                    short = {'reward': returns.clone(), 'boot': boot.clone(), 'failed': failed.clone(), 'timeout': timeout.clone()}
            if short is None or injected['calls'] != args.eval_horizon or injected['clean_error'] != 0 or injected['executed_error'] > 1e-5:
                raise ValueError('invalid paired noise injection or horizon')
            if not all(torch.isfinite(x).all() for x in (returns, short['reward'], short['boot'], obs['policy'], obs['goal'])):
                raise ValueError('nonfinite evaluation result')
            def records(out_reward, out_boot, out_failed, out_timeout, is_long):
                good = eligible & ~out_timeout.reshape(groups, n).any(0)
                rows = []
                for j in good.nonzero().squeeze(-1).tolist():
                    item = {'index': j, 'frame': int(frames[j]), 'clip': labels[j % n_base], 'replicate': j // n_base,
                            'base_reward': float(out_reward[4*n+j]), 'base_control_reward': float(out_reward[5*n+j]),
                            'base_bootstrap': float(out_boot[4*n+j]), 'base_control_bootstrap': float(out_boot[5*n+j]),
                            'base_failed': bool(out_failed[4*n+j]), 'control_failed': bool(out_failed[5*n+j])}
                    if is_long:
                        item['base_failure_step'] = int(fail_step[4*n+j])
                        item['control_failure_step'] = int(fail_step[5*n+j])
                    for k, name in ((0, 'clean_gradient'), (2, 'noise_gradient')):
                        reward_plus, reward_minus = out_reward[k*n+j], out_reward[(k+1)*n+j]
                        boot_plus, boot_minus = out_boot[k*n+j], out_boot[(k+1)*n+j]
                        item[name] = {'reward_plus': float(reward_plus), 'reward_minus': float(reward_minus),
                                      'bootstrap_plus': float(boot_plus), 'bootstrap_minus': float(boot_minus),
                                      'failed_plus': bool(out_failed[k*n+j]), 'failed_minus': bool(out_failed[(k+1)*n+j]),
                                      'reward_delta_vs_base': float(reward_plus - out_reward[4*n+j]),
                                      'total_delta_vs_base': float(reward_plus + boot_plus - out_reward[4*n+j] - out_boot[4*n+j]),
                                      'reward_center_difference': float((reward_plus - reward_minus) / (2*args.eps)),
                                      'total_center_difference': float((reward_plus + boot_plus - reward_minus - boot_minus) / (2*args.eps))}
                        if is_long:
                            item[name]['failure_step_plus'] = int(fail_step[k*n+j])
                            item[name]['failure_step_minus'] = int(fail_step[(k+1)*n+j])
                    rows.append(item)
                if not rows:
                    raise ValueError('no valid evaluation pairs')
                return {'valid_pairs': len(rows), 'excluded_timeouts': int(eligible.sum()) - len(rows),
                        'base_control_max_abs_reward_difference': max(abs(x['base_reward']-x['base_control_reward']) for x in rows),
                        'base_control_max_abs_bootstrap_difference': max(abs(x['base_bootstrap']-x['base_control_bootstrap']) for x in rows),
                        'base_control_failure_disagreements': sum(x['base_failed'] != x['control_failed'] for x in rows), 'pairs': rows}
            short_result = records(short['reward'], short['boot'], short['failed'], short['timeout'], False)
            final_boot = torch.zeros_like(returns)
            if long_bootstrap:
                state=io.read()
                ref=reference_at(lib,cmd.frame)
                val=critic(wm.spec.canonical(state),obs['goal'],wm.spec.normalize_xi(ctx.read()),
                           wm.heads.root_error_features(state,ref))
                final_boot=torch.where(alive,args.gamma**args.eval_horizon*val,torch.zeros_like(val))
                if not torch.isfinite(final_boot).all():raise ValueError('nonfinite final bootstrap')
            long_result = records(returns, final_boot, failed, timeout, True)
        return {'sigma': sigma, 'candidate_pairs': n, 'eligible_pairs': int(eligible.sum()),
                'warmup_reset_pairs': int(warmup_bad.sum()), 'initial_mismatch_pairs': int(mismatch.sum()),
                'initial_obs_error': obs_error, 'initial_goal_error': goal_error,
                'terminal_audits':terminal_audits, **({'switch_audit':switch_audit} if switching else {}),
                **({'replicate_prefix_audit':replicate_audit} if shared_prefix else {}),
                **({'serial_prefix_hashes':prefix_hashes} if tail_seed is not None else {}),
                'action_checks': {'calls': injected['calls'], 'max_clean_error': injected['clean_error'], 'max_executed_error': injected['executed_error']},
                'short': short_result, 'long': long_result}, initial
    finally:
        vector_to_parameters(base, actor.parameters())
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--npz-dir', type=Path, default=Path('data/npz/lafan1'))
    p.add_argument('--clips', nargs='+', default=['walk1_subject1', 'dance1_subject1', 'jumps1_subject1'])
    p.add_argument('--eval-clips', nargs='+', default=['walk2_subject1', 'dance2_subject1', 'jumps1_subject2'])
    p.add_argument('--seed', type=int, default=3100)
    p.add_argument('--frames-per-clip', type=int, default=2)
    p.add_argument('--eval-frames-per-clip', type=int, default=2)
    p.add_argument('--gradient-replicates', type=int, default=4)
    p.add_argument('--eval-replicates', type=int, default=2)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--eval-horizon', type=int, default=80)
    p.add_argument('--noise-std', type=float, default=.1)
    p.add_argument('--eps', type=float, default=.005)
    p.add_argument('--verify-model-fd', action='store_true', help='finite-difference check model gradients using the same fixed noise tapes')
    p.add_argument('--model-fd-eps', type=float, default=.001)
    p.add_argument('--output', type=Path, default=Path('docs/analysis/g1_noisy_imagination_direction.json'))
    args = p.parse_args()
    if set(args.clips) & set(args.eval_clips) or min(args.frames_per_clip, args.eval_frames_per_clip, args.eval_replicates) < 1 or args.gradient_replicates < 2 or args.gradient_replicates % 2 or args.warmup < 10 or args.horizon < 1 or args.eval_horizon < args.horizon or args.noise_std < 0 or args.eps <= 0 or args.model_fd_eps <= 0:
        p.error('invalid configurations or overlapping gradient/evaluation clips')
    started = time.perf_counter()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    pcfg, wcfg = config_from_checkpoint(ck)
    if ck['config']['env'].get('anchor_obs', False):
        raise NotImplementedError('anchor observations not supported')
    args.future = int(ck['config']['env']['future_frames'])
    args.history = int(ck['config']['policy_trainer']['history'])
    args.gamma = float(ck['config']['policy_trainer']['gamma'])
    ccfg = CriticCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck['config']['critic'].items()})
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor.load_state_dict(ck['policy'])
    critic.load_state_dict(ck['critic'])
    actor.eval()
    critic.eval().requires_grad_(False)
    files = discover_npz(args.npz_dir, names=args.clips + args.eval_clips)
    if len(files) != len(args.clips) + len(args.eval_clips):
        raise ValueError('missing or duplicated clips')
    gradient_frames, eval_frames, gradient_labels, eval_labels = frame_samples(files, args, args.future)
    gc.collect()
    env = None
    try:
        env, obs, bad, start, io, ctx, cmd, _, _ = make_setup(files, gradient_frames, 1, args.seed, args.warmup,
                                                               args.history, args.future, actor, True)
        lib = cmd.library
        eligible = ~bad
        if not eligible.any():
            raise ValueError('no gradient starts survive warmup')
        robot = env.scene['robot']
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device='cpu'), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(wcfg, heads, 'cpu')
        wm.load_state_dict(ck['wm'])
        wm.eval().requires_grad_(False)
        starts = select_and_repeat(start, eligible, 1)
        reconstructed = flatten_history_term_major(torch.stack([analytic_observation(starts.hist_states[t], starts.hist_actions[t], starts.xi, lib.default_joint_pos) for t in range(args.history)]))
        obs_err = float((reconstructed - obs['policy'][eligible]).abs().max())
        goal_err = float((lib.goal_window(starts.frame, args.future) - obs['goal'][eligible]).abs().max())
        if obs_err > 1e-3 or goal_err > 1e-4:
            raise ValueError(f'gradient history or goal mismatch: {obs_err}, {goal_err}')
        n = len(starts.state)
        zero = torch.zeros(args.horizon, n, 29)
        with torch.no_grad():
            _, _, r_ref, b_ref, _ = model_objective(wm, actor, critic, lib, starts, args.horizon, args.future, args.gamma)
            r_zero, b_zero = imagined_parts(wm, actor, critic, lib, starts, args.horizon, args.future, args.gamma, zero)
            parity = float(torch.cat(((r_ref-r_zero).abs(), (b_ref-b_zero).abs())).max())
        if parity > 1e-6:
            raise ValueError(f'zero-noise objective differs from production: {parity}')
        clean_grad, clean_reward_grad, clean_obj = objective_gradients(wm, actor, critic, lib, starts, args.horizon, args.future, args.gamma, zero)
        repeated = select_and_repeat(starts, torch.ones(n, dtype=torch.bool), 2)
        rng = torch.Generator().manual_seed(150000 + args.seed)
        half = args.gradient_replicates // 2
        noise_grads, noise_reward_grads, noise_objs, noise_tapes = [], [], [], []
        for _ in range(half):
            samples = args.noise_std * torch.randn(args.horizon, n, 29, generator=rng)
            tape = torch.cat((samples, -samples), dim=1)
            noise_tapes.append(tape)
            g, gr, obj = objective_gradients(wm, actor, critic, lib, repeated, args.horizon, args.future, args.gamma, tape)
            noise_grads.append(g); noise_reward_grads.append(gr); noise_objs.append(obj)
        stacked = torch.stack(noise_grads)
        noisy_grad = stacked.mean(0)
        noisy_reward_grad = torch.stack(noise_reward_grads).mean(0)
        noisy_obj = {k: sum(x[k] for x in noise_objs) / half for k in ('reward_mean', 'bootstrap_mean')}
        mc = {'antithetic_pairs': half, 'sample_norms': [float(g.norm()) for g in noise_grads],
              'mean_vector_se_norm': float(stacked.double().var(0, unbiased=True).sum().sqrt() / half**.5) if half > 1 else None,
              'pair_gradient_cosine': float(torch.nn.functional.cosine_similarity(noise_grads[0].double(), noise_grads[1].double(), dim=0)) if half > 1 else None}
        cos = float(torch.nn.functional.cosine_similarity(clean_grad.double(), noisy_grad.double(), dim=0))
        directions = [clean_grad / clean_grad.norm(), noisy_grad / noisy_grad.norm()]
        def components(full, reward):
            bootstrap = full - reward
            return {'full_norm': float(full.norm()), 'reward_norm': float(reward.norm()), 'bootstrap_norm': float(bootstrap.norm())}
        stats = {'clean': components(clean_grad, clean_reward_grad), 'noise': components(noisy_grad, noisy_reward_grad),
                 'gradient_cosine': cos, 'clean_objective': clean_obj, 'noisy_objective': noisy_obj,
                 'gradient_difference_norm': float((noisy_grad-clean_grad).norm()),
                 'reward_gradient_cosine': float(torch.nn.functional.cosine_similarity(clean_reward_grad.double(), noisy_reward_grad.double(), dim=0)),
                 'reward_gradient_difference_norm': float((noisy_reward_grad-clean_reward_grad).norm()),
                 'noise_gradient_mc': mc,
                 'zero_noise_objective_parity_max': parity, 'gradient_obs_error': obs_err, 'gradient_goal_error': goal_err}
        base = parameters_to_vector(actor.parameters()).detach().clone()
        if args.verify_model_fd:
            verification = {}
            with torch.no_grad():
                try:
                    for name, gradient, direction, fd_starts, tapes in (
                        ('clean', clean_grad, directions[0], starts, [zero]),
                        ('noise', noisy_grad, directions[1], repeated, noise_tapes)):
                        values = []
                        for sign in (1., -1.):
                            vector_to_parameters(base + sign*args.model_fd_eps*direction, actor.parameters())
                            objectives = []
                            for fd_tape in tapes:
                                fr, fb = imagined_parts(wm, actor, critic, lib, fd_starts, args.horizon, args.future, args.gamma, fd_tape)
                                objectives.append(float((fr + fb).mean()))
                            values.append(sum(objectives) / len(objectives))
                        fd = (values[0] - values[1]) / (2*args.model_fd_eps)
                        expected = float(torch.dot(gradient, direction))
                        verification[name] = {'autograd_projection': expected, 'finite_difference': fd,
                                              'relative_error': abs(fd-expected) / max(abs(expected), 1e-12)}
                finally:
                    vector_to_parameters(base, actor.parameters())
            stats['fixed_noise_model_finite_difference'] = verification
        gradient_count = n
    finally:
        if env is not None:
            env.close()
    del env, obs, io, ctx, cmd, ck, starts, repeated, clean_grad, clean_reward_grad, noisy_grad, noisy_reward_grad
    gc.collect()
    modes = {}
    starts_by_mode = {}
    for mode, sigma in (('deterministic', 0.), ('noisy', args.noise_std)):
        result, start_values = evaluate(files, eval_frames, eval_labels, args.seed, actor, critic, wm, directions, base, args, sigma)
        modes[mode], starts_by_mode[mode] = result, start_values
        gc.collect()
    x, y = starts_by_mode['deterministic'], starts_by_mode['noisy']
    common = x['eligible'] & y['eligible']
    if not common.any():
        raise ValueError('no common starts across evaluation modes')
    mode_state_error = float((x['state'][common] - y['state'][common]).abs().max())
    mode_obs_error = float((x['obs'][common] - y['obs'][common]).abs().max())
    mode_goal_error = float((x['goal'][common] - y['goal'][common]).abs().max())
    if mode_state_error > 1e-5 or mode_obs_error > 1e-4 or mode_goal_error > 1e-5:
        raise ValueError('deterministic and noisy evaluations have mismatched start states')
    out = {'checkpoint': str(args.ckpt), 'seed': args.seed, 'horizon': args.horizon, 'eval_horizon': args.eval_horizon,
           'noise_std': args.noise_std, 'eps': args.eps, 'gradient_replicates': args.gradient_replicates,
           'eval_replicates': args.eval_replicates, 'gradient_candidates': len(gradient_frames), 'gradient_starts': gradient_count,
           'gradient_frames': gradient_frames.tolist(), 'gradient_clips': gradient_labels, 'eval_frames': eval_frames.tolist(), 'eval_clips': eval_labels,
           'gradient_stats': stats, 'evaluation_modes': modes,
           'cross_mode_start_checks': {'common_pairs': int(common.sum()), 'max_state_error': mode_state_error, 'max_obs_error': mode_obs_error, 'max_goal_error': mode_goal_error},
           'simulator_transition_counts': {'gradient_warmup': len(gradient_frames)*args.warmup,
               'evaluation_warmup': 2*6*len(eval_frames)*args.eval_replicates*args.warmup,
               'evaluation': 2*6*len(eval_frames)*args.eval_replicates*args.eval_horizon},
           'wall_seconds': time.perf_counter()-started,
           'notes': 'Noisy model gradient uses finite antithetic noise samples; reward and history use clean commands. Real eval uses disjoint gradient/eval clips but checkpoint may have seen them. Fixed checkpoint, paired 80-step truncated reward, no training. Fixed friction, no observation or warmup action noise. Short objective includes potentially biased frozen critic; long objective does not.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
