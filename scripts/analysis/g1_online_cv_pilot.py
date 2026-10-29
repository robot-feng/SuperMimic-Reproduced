"""Research pilot: fresh real score and model control variate in online G1.

CPU only. Actor and world model update each round; the checkpoint critic is
frozen. A fits the model; B is withheld until after the actor gradient. The
model baseline may fit A+B. All arms receive equal real environment steps.
This narrow experiment is not the original SuperMimic training protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch

from clean_command_noise import replace_execution
from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_mature_fit_actor_probe import checkpoint_pickle_module
from g1_model_score_probe import model_rollout, policy_observations
from g1_noisy_imagination_direction import frame_samples
from g1_online_window_data import FreshTransitions, capture_real
from g1_empirical_policy_guard import guard, reference_actions
from g1_adam_zero_momentum import zero_gradient_delta, compare_updates
from g1_adam_moment_intervention import reset_first_moment
from g1_tail_policy_switch_probe import sha
from g1_real_score_probe_crossfit import collect_real, tensor_hash
from g1_score_estimator import score_surrogate
from g1_terminal_training_pilot import evaluate
from policy_direction_math import per_trajectory_objective
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.replay_buffer import StartBatch, WindowBatch
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def subset_start(start, sl):
    return StartBatch(start.state[sl].clone(), start.xi[sl].clone(), start.frame[sl].clone(),
                      start.prev_action[sl].clone(), start.hist_states[:, sl].clone(),
                      start.hist_actions[:, sl].clone())


def subset_transitions(transitions, sl):
    return FreshTransitions(**{key: getattr(transitions, key)[:, sl].clone()
                               for key in FreshTransitions.__dataclass_fields__})


def sample_windows(w, count, rng):
    n = w.states.shape[1]
    indices = torch.randint(n, (count,), generator=rng)
    return WindowBatch(w.states[:, indices], w.a_exec[:, indices], w.a_clean[:, indices],
                       w.xi[indices], w.reward[:, indices], w.done[:, indices], w.frame[:, indices])


def episode_history(states, actions, generations, episode_starts, history):
    """Backfill history from the first state after the latest actual reset."""
    hs = torch.stack(states[-history:]); ha = torch.stack(actions[-history:])
    eg = torch.stack(generations[-history:]); first = torch.stack(episode_starts[-history:])
    same = eg == generations[-1][None, :]
    index = same.float().argmax(0)
    cols = torch.arange(hs.shape[1], device=hs.device)
    return (torch.where(same[..., None], hs, hs[index, cols][None]),
            torch.where((same & ~first)[..., None], ha, torch.zeros_like(ha)))


@torch.no_grad()
def warmup(env, actor, io, ctx, cmd, frames, history, steps, injection, allow_reset=False):
    cmd.forced_frame[:] = frames
    obs, _ = env.reset()
    injection['noise'].zero_()
    states, actions, generations, episode_starts = [], [], [], []
    bad = torch.zeros(len(frames), dtype=torch.bool)
    generation = torch.zeros(len(frames), dtype=torch.long)
    is_start = torch.ones(len(frames), dtype=torch.bool)
    failure_count = 0
    for t in range(steps + 1):
        states.append(io.read().clone())
        actions.append(env.action_manager.action.clone())
        generations.append(generation.clone()); episode_starts.append(is_start.clone())
        if t < steps:
            obs, _, done, timeout, _ = env.step(actor(obs['policy'], obs['goal']))
            if timeout.any(): raise ValueError('timeout in warmup')
            bad |= done; failure_count += int(done.sum())
            generation += done.long(); is_start = done.clone()
    if bad.any() and not allow_reset:
        raise ValueError(f'warmup failed: {bad.nonzero().flatten().tolist()}')
    hs, ha = episode_history(states, actions, generations, episode_starts, history)
    start = StartBatch(hs[-1].clone(), ctx.read().clone(), cmd.frame.clone(), ha[-1].clone(), hs, ha)
    return obs, start, failure_count, int(bad.sum())


def gradient(output, actor):
    return [x.detach() for x in torch.autograd.grad(output, tuple(actor.parameters()))]


def actor_objective(rollout, values, gamma, bootstrap_weight):
    """Change only actor's end-value coefficient; preserve state gradients."""
    return per_trajectory_objective(rollout, values[-1]*bootstrap_weight, gamma)


def component_stats(full,reward):
    """Euclidean gradient components before clipping or Adam preconditioning."""
    if len(full)!=len(reward):raise ValueError('gradient length mismatch')
    rr=bb=ff=rb=0.
    for f,r in zip(full,reward):
        f=f.double();r=r.double();b=f-r
        if not torch.isfinite(f).all() or not torch.isfinite(r).all():raise ValueError('nonfinite gradient')
        rr+=float((r*r).sum());bb+=float((b*b).sum());ff+=float((f*f).sum());rb+=float((r*b).sum())
    return {'reward_norm':rr**.5,'bootstrap_norm':bb**.5,'full_norm':ff**.5,'reward_dot_bootstrap':rb,
            'reward_bootstrap_cosine':rb/(rr*bb)**.5 if rr>0 and bb>0 else None,
            'norm_identity_error':abs(ff-(rr+bb+2*rb))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('model', 'score', 'cv'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--window', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--frames-per-clip', type=int, default=1)
    p.add_argument('--noise-std', type=float, default=.1)
    p.add_argument('--actor-lr', type=float, default=2e-4)
    p.add_argument('--wm-lr', type=float, default=2e-4)
    p.add_argument('--resume-optimizer', action='store_true',
                   help='restore checkpoint actor and world-model Adam moments; use explicit LR arguments')
    p.add_argument('--reset-actor-first-moment', action='store_true',
                   help='once after restoring, zero actor exp_avg only; keep exp_avg_sq and step')
    p.add_argument('--allow-warmup-reset', action='store_true',
                   help='retain reset starts with episode-aware history instead of aborting on physical failure')
    p.add_argument('--model-path-ab', action='store_true',
                   help='model arm averages its actor path gradient across both A and B noise tapes')
    p.add_argument('--actor-kl-limit',type=float,default=None,
                   help='backtrack actor Adam update using empirical action KL on live on-policy states')
    p.add_argument('--actor-bootstrap-weight',type=float,default=1.,choices=(0.,1.),
                   help='ablate frozen critic contribution to the model actor objective')
    p.add_argument('--actor-decompose-first',action='store_true',
                   help='audit actor reward and bootstrap gradients in first model update')
    a = p.parse_args()
    weights_path = a.output.with_suffix('.pt')
    failure_path = a.output.with_suffix('.failed.json')
    failure_weights_path = a.output.with_suffix('.failed.pt')
    if any(path.exists() for path in (a.output, weights_path, failure_path, failure_weights_path)):
        raise FileExistsError('refusing to overwrite an experiment')
    if min(a.rounds, a.horizon, a.window, a.batch_size, a.frames_per_clip) < 1 or a.window > a.horizon or a.noise_std <= 0:
        p.error('invalid settings')
    if a.model_path_ab and a.arm != 'model': p.error('model-path-ab applies only to the model arm')
    if a.reset_actor_first_moment and not a.resume_optimizer: p.error('moment intervention requires optimizer restore')
    if a.actor_bootstrap_weight != 1. and a.arm != 'model':p.error('bootstrap ablation is only implemented for model path')
    if a.actor_decompose_first and (a.arm != 'model' or a.actor_bootstrap_weight != 1.):
        p.error('gradient decomposition requires full model-path objective')
    torch.manual_seed(a.seed)
    started = time.perf_counter()
    audit_paths = [Path(__file__), Path('scripts/analysis/g1_empirical_policy_guard.py'),
                   Path('scripts/analysis/g1_online_window_data.py'), Path('scripts/analysis/g1_real_score_probe_crossfit.py'),
                   Path('scripts/analysis/g1_model_score_probe.py'), Path('scripts/analysis/policy_direction_math.py'),
                   Path('scripts/analysis/g1_adam_zero_momentum.py'),
                   Path('scripts/analysis/g1_adam_moment_intervention.py'),
                   Path('scripts/analysis/g1_terminal_training_pilot.py'), a.ckpt]
    audit_hashes = {str(path): sha(path) for path in audit_paths}
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False,
                    pickle_module=checkpoint_pickle_module(os.name != 'nt'))
    pcfg, wcfg = config_from_checkpoint(ck)
    ccfg = CriticCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck['config']['critic'].items()})
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor.load_state_dict(ck['policy']); critic.load_state_dict(ck['critic'])
    actor.eval(); critic.eval().requires_grad_(False)
    history = int(ck['config']['policy_trainer']['history'])
    future = int(ck['config']['env']['future_frames'])
    gamma = float(ck['config']['policy_trainer']['gamma'])
    if ck['config']['env'].get('anchor_obs', False):
        raise ValueError('anchor observations unsupported')
    a.clips = ['walk1_subject1', 'dance1_subject1', 'jumps1_subject1']
    a.eval_clips = ['dance2_subject1', 'jumps1_subject2', 'walk2_subject1']
    a.eval_frames_per_clip = 1; a.warmup = max(10, history); a.eval_horizon = a.horizon
    files = discover_npz(Path('data/npz/lafan1'), names=a.clips + a.eval_clips)
    if len(files) != 6: raise ValueError('missing clips')
    frames, _, labels, _ = frame_samples(files, a, future)
    frames = frames.repeat(2)
    evaluation_before = evaluate(actor, seed=9400+a.seed)
    env = None
    sim_steps = [0]
    try:
        env, _, _, _, io, ctx, cmd, _, _ = make_setup(files, frames, 1, a.seed, 0, history, future, actor, True)
        environment_step = env.step
        def counted_step(*args, **kwargs):
            result = environment_step(*args, **kwargs)
            sim_steps[0] += len(frames)
            return result
        env.step = counted_step
        lib = cmd.library; robot = env.scene['robot']
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device='cpu'), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(wcfg, heads, 'cpu'); wm.load_state_dict(ck['wm']); wm.eval()
        trainer = WorldModelTrainer(wm, lib, WMTrainerCfg(window=a.window, batch_size=a.batch_size,
                                    batches_per_update=1, lr=a.wm_lr, compile=None))
        actor_opt = torch.optim.Adam(actor.parameters(), lr=a.actor_lr, betas=(.9, .95))
        moment_intervention = None
        if a.resume_optimizer:
            if 'policy_opt' not in ck or 'wm_opt' not in ck:
                raise ValueError('checkpoint does not contain actor and model optimizer states')
            actor_opt.load_state_dict(ck['policy_opt']); trainer.opt.load_state_dict(ck['wm_opt'])
            for group in actor_opt.param_groups: group['lr'] = a.actor_lr
            for group in trainer.opt.param_groups: group['lr'] = a.wm_lr
            if a.reset_actor_first_moment:
                moment_intervention = reset_first_moment(actor_opt)
        term = env.action_manager.get_term('joint_pos')
        if term.noise_std != 0: raise ValueError('automatic action noise enabled')
        injection = {'noise': torch.zeros(len(frames), 29), 'capture_terminal': False, 'pre_reset': None}
        process = term.process_actions
        def inject(clean):
            process(clean); replace_execution(term, clean, injection['noise'])
        term.process_actions = inject
        reset_idx = env._reset_idx
        def capture_reset(ids):
            if injection['capture_terminal']:
                injection['pre_reset'] = {'ids': ids.clone(), 'state': io.read()[ids].clone(),
                                          'reward': env.reward_buf[ids].clone()}
            return reset_idx(ids)
        env._reset_idx = capture_reset
        action_rng = torch.Generator().manual_seed(7000000 + a.seed)
        fit_rng = torch.Generator().manual_seed(8000000 + a.seed)
        n = len(frames)//2; records = []
        initial_hash = tensor_hash(*actor.parameters())
        for rnd in range(a.rounds):
            obs, start, warmup_failures, warmup_failed_envs = warmup(
                env, actor, io, ctx, cmd, frames, history, a.warmup, injection, a.allow_warmup_reset)
            reconstructed = flatten_history_term_major(torch.stack([
                analytic_observation(start.hist_states[t], start.hist_actions[t], start.xi, lib.default_joint_pos)
                for t in range(history)]))
            obs_error = float((reconstructed-obs['policy']).abs().max())
            if obs_error > 1e-3: raise ValueError(f'initial history mismatch: {obs_error}')
            start_pair_error = max(float((start.state[:n]-start.state[n:]).abs().max()),
                                   float((start.xi[:n]-start.xi[n:]).abs().max()),
                                   float((obs['policy'][:n]-obs['policy'][n:]).abs().max()),
                                   float((obs['goal'][:n]-obs['goal'][n:]).abs().max()))
            if start_pair_error > 1e-3 or not torch.equal(start.frame[:n], start.frame[n:]):
                raise ValueError(f'paired initial states differ: {start_pair_error}')
            # A and B share starting frames; their independently sampled execution
            # noise is never used to fit the CV/score model in this round.
            tape = a.noise_std * torch.randn(a.horizon, 2*n, 29, generator=action_rng)
            wm.requires_grad_(False)
            scale = heads.reward_cfg.dt * heads.reward_cfg.w_action_rate
            real, transitions = capture_real(collect_real, env, io, ctx, cmd, term, injection,
                 env, obs, actor, critic, wm, io, ctx, cmd, tape, injection, gamma, scale)
            if real['action_rate_max_error'] > 1e-5: raise ValueError('action rate mismatch')
            b_hash = tensor_hash(*(getattr(transitions, k)[:, n:] for k in FreshTransitions.__dataclass_fields__))
            fit_data = transitions if a.arm == 'model' else subset_transitions(transitions, slice(0,n))
            available = fit_data.windows(a.window)
            w = sample_windows(available, a.batch_size, fit_rng)
            wm.requires_grad_(True); loss, wm_metrics = trainer.loss_on(w)
            trainer.opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); trainer.opt.step(); wm.requires_grad_(False)
            if b_hash != tensor_hash(*(getattr(transitions, k)[:, n:] for k in FreshTransitions.__dataclass_fields__)):
                raise ValueError('held-out B data changed')
            bstart = subset_start(start, slice(n, 2*n)); btape = tape[:, n:]
            path_start, path_tape = (start, tape) if a.model_path_ab else (bstart, btape)
            ro, vals, objective = model_rollout(wm, actor, critic, lib, path_start, a.horizon, future, gamma, path_tape)
            if a.actor_bootstrap_weight != 1.:
                objective=actor_objective(ro,vals,gamma,a.actor_bootstrap_weight)
            model_obj = float(objective.detach().mean())
            model_survival = float(ro.survival[-1].detach().mean())
            real_reward = sum(gamma**t * (real['active'][t, n:] * real['rewards'][t, n:]).mean()
                              for t in range(a.horizon))
            reward_grad=None
            if a.actor_decompose_first and rnd==0:
                reward_obj=actor_objective(ro,vals,gamma,0.)
                reward_grad=[x.detach() for x in torch.autograd.grad(reward_obj.mean(),tuple(actor.parameters()),retain_graph=True)]
            path = gradient(objective.mean(), actor)
            decomp=component_stats(path,reward_grad) if reward_grad is not None else None
            del ro, vals, objective
            checks = {}; component_norms = {'model_path': float(torch.sqrt(sum(x.double().square().sum() for x in path)))}
            if a.arm != 'model':
                b = {k: (v[:, n:] if k in ('observations','goals','actions','rewards','active','baselines') else v[n:])
                     for k, v in real.items() if k in ('observations','goals','actions','rewards','active','baselines','bootstrap')}
                rs, real_check = score_surrogate(actor, b['observations'], b['goals'], b['actions'], btape,
                    b['rewards'], b['active'].float(), b['bootstrap'], gamma, a.noise_std, history, scale,
                    active=b['active'], baseline=b['baselines'])
                rg = gradient(rs.mean(), actor); del rs
                component_norms['real_score'] = float(torch.sqrt(sum(x.double().square().sum() for x in rg)))
                checks['real_replay'] = real_check
                if a.arm == 'score':
                    chosen = rg
                else:
                    with torch.no_grad():
                        ro, vals, _ = model_rollout(wm, actor, critic, lib, bstart, a.horizon, future, gamma, btape)
                        m_obs = policy_observations(wm, lib, bstart, ro)
                        boot = gamma**a.horizon * ro.survival[-1] * vals[-1]
                        baseline = torch.stack([gamma**t * ro.survival[t] * vals[t] for t in range(a.horizon)])
                    ms, model_check = score_surrogate(actor, m_obs, ro.goals[:-1], ro.actions, btape,
                        ro.rewards, ro.survival[:-1], boot, gamma, a.noise_std, history, scale, baseline=baseline)
                    mg = gradient(ms.mean(), actor); del ms, ro, vals
                    component_norms['model_score'] = float(torch.sqrt(sum(x.double().square().sum() for x in mg)))
                    chosen = [r + p - m for r, p, m in zip(rg, path, mg)]
                    checks['model_replay'] = model_check
                if any(max(v.values()) > 1e-4 for v in checks.values()): raise ValueError(f'score parity failure {checks}')
            else:
                chosen = path
            norm = float(torch.sqrt(sum(x.double().square().sum() for x in chosen)))
            if not all(torch.isfinite(x).all() for x in chosen): raise ValueError('nonfinite gradient')
            actor_opt.zero_grad(set_to_none=True)
            actor_before = [p.detach().clone() for p in actor.parameters()]
            zero_delta=zero_gradient_delta(actor_opt,tuple(actor.parameters())) if decomp is not None else None
            active=real['active']
            guard_obs=real['observations'][active].detach();guard_goals=real['goals'][active].detach()
            old_actions=reference_actions(actor,guard_obs,guard_goals)
            for parameter, g in zip(actor.parameters(), chosen): parameter.grad = -g / a.horizon
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(actor.parameters(), .1)
            actor_opt.step()
            if decomp is not None:
                proposed_delta=[p.detach()-old for p,old in zip(actor.parameters(),actor_before)]
                decomp['adam_zero_counterfactual']=compare_updates(proposed_delta,zero_delta,path,reward_grad)
            guard_diagnostics=guard(actor,actor_before,guard_obs,guard_goals,old_actions,a.noise_std,a.actor_kl_limit)
            update_norm = float(torch.sqrt(sum((p.detach().double()-old.double()).square().sum()
                                               for p, old in zip(actor.parameters(), actor_before))))
            row = {'round': rnd+1, 'wm_loss': float(loss.detach()), 'model_objective': model_obj,
                   'model_survival_h': model_survival, 'real_discounted_reward_B': float(real_reward),
                   'real_bootstrap_B': float(real['bootstrap'][n:].mean()),
                   'actor_ascent_grad_norm': norm, 'actor_update_norm': update_norm,
                   'warmup_failures': warmup_failures, 'warmup_failed_envs': warmup_failed_envs,
                   'A_failures': int(real['failed'][:n].sum()),
                   'B_failures': int(real['failed'][n:].sum()), 'fit_available_windows': int(available.states.shape[1]),
                   'fit_terminal_labels': int(w.done.sum()), 'active_A': int(real['active'][:, :n].sum()),
                   'active_B': int(real['active'][:, n:].sum()), 'initial_obs_error': obs_error,
                   'A_B_start_max_error': start_pair_error, 'B_data_sha256': b_hash,
                   'noise_sha256': tensor_hash(tape), 'gradient_component_norms': component_norms, 'parity': checks,
                   'actor_guard':guard_diagnostics,'actor_component_diagnostic':decomp}
            records.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
        env.close(); env = None
        assessment = evaluate(actor, seed=9400+a.seed)
        if sim_steps[0] != a.rounds * len(frames) * (a.warmup + a.horizon):
            raise ValueError('unexpected number of simulator transitions')
        if any(sha(path) != audit_hashes[str(path)] for path in audit_paths):
            raise ValueError('audited source or checkpoint changed')
        output = {'arm': a.arm, 'seed': a.seed, 'source_checkpoint': str(a.ckpt), 'initial_actor_sha256': initial_hash,
                  'settings': vars(a) | {'ckpt': str(a.ckpt), 'output': str(a.output)}, 'training_frames': frames.tolist(),
                  'training_clip_labels': labels, 'rounds': records, 'evaluation_before': evaluation_before,
                  'evaluation': assessment,
                  'train_simulator_transitions': sim_steps[0],
                  'source_sha256': audit_hashes,
                  'initial_setup_env_transitions': 0, 'model_updates': a.rounds, 'actor_updates': a.rounds,
                  'moment_intervention': moment_intervention,
                  'wall_seconds': time.perf_counter()-started,
                  'limitations': 'Fixed reference frames, deterministic reset/warmup and observations; action noise only; frozen source critic; no replay across rounds; model baseline uses A+B, score/CV use A; unequal compute; small pilot, not original training. Optimizer moments restored only if resume_optimizer is true. Actor guard, if enabled, bounds empirical Gaussian action KL only on current real active observations; its Adam moments follow the full proposed gradient, not the accepted scale.'}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'actor': actor.state_dict(), 'wm': wm.state_dict(), 'actor_opt': actor_opt.state_dict(),
                    'wm_opt': trainer.opt.state_dict()}, weights_path)
        output['saved_weights_sha256'] = sha(weights_path)
        a.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    except Exception as exc:
        if 'records' in locals():
            a.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'actor': actor.state_dict(), 'wm': wm.state_dict(), 'actor_opt': actor_opt.state_dict(),
                        'wm_opt': trainer.opt.state_dict()}, failure_weights_path)
            failure = {'arm': a.arm, 'seed': a.seed, 'source_checkpoint': str(a.ckpt),
                       'settings': vars(a) | {'ckpt': str(a.ckpt), 'output': str(a.output)},
                       'completed_rounds': len(records), 'rounds': records,
                       'train_simulator_transitions': sim_steps[0], 'evaluation_before': evaluation_before,
                       'error': str(exc), 'traceback': traceback.format_exc(),
                       'partial_checkpoint': str(failure_weights_path)}
            failure_path.write_text(json.dumps(failure, indent=2, allow_nan=False), encoding='utf8')
        raise
    finally:
        if env is not None: env.close()


if __name__ == '__main__': main()
