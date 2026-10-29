"""G1 model-only score/pathwise identity check at a fixed checkpoint.

Fixed reference starts and noisy imagination, no real-score estimate or policy
update. Actor history includes direct gradients through prior clean commands.
"""
import argparse
import gc
import json
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WARP_CACHE_PATH', r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from g1_directional_probe import config_from_checkpoint
from g1_grounded_gradient_probe import make_setup
from g1_noisy_imagination_direction import frame_samples
from g1_score_estimator import score_surrogate
from noisy_imag_step import NoisyImagStep
from policy_direction_math import per_trajectory_objective
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, CriticCfg, Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

torch.set_num_threads(1)


def model_rollout(wm, actor, critic, lib, start, horizon, future, gamma, noise):
    step = NoisyImagStep(wm, lib.default_joint_pos, noise)
    ro = imagine(wm, actor, lib, start, horizon, future, obs_noise=False, step_fn=step, anchor_obs=False)
    if step.index != horizon:
        raise ValueError('noise tape length mismatch')
    n = len(start.state)
    xi = wm.spec.normalize_xi(start.xi)
    vals = critic(wm.spec.canonical(ro.states.reshape((horizon+1)*n, -1)),
                  ro.goals.reshape((horizon+1)*n, -1), xi[None].expand(horizon+1,n,-1).reshape((horizon+1)*n,-1),
                  ro.e_root.reshape((horizon+1)*n,-1)).reshape(horizon+1,n)
    obj = per_trajectory_objective(ro, vals[-1], gamma)
    return ro, vals, obj


@torch.no_grad()
def policy_observations(wm, lib, start, ro):
    hist = [analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, lib.default_joint_pos)
            for i in range(start.hist_states.shape[0])]
    result = []
    for t in range(len(ro.actions)):
        result.append(flatten_history_term_major(torch.stack(hist)))
        obs = analytic_observation(ro.states[t+1], ro.actions[t], start.xi, lib.default_joint_pos)
        hist = hist[1:]+[obs]
    return torch.stack(result)


def flattened_grad(output, actor):
    return torch.cat([g.detach().flatten() for g in torch.autograd.grad(output, tuple(actor.parameters()))])


def describe(values):
    return {'mean': mean(values), 'std': stdev(values), 'se': stdev(values)/(len(values)**.5), 'values': values}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--npz-dir', type=Path, default=Path('data/npz/lafan1'))
    p.add_argument('--clips', nargs='+', default=['walk1_subject1'])
    p.add_argument('--eval-clips', nargs='+', default=['dance1_subject1'])
    p.add_argument('--seed', type=int, default=3400)
    p.add_argument('--frames-per-clip', type=int, default=1)
    p.add_argument('--eval-frames-per-clip', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--repeats', type=int, default=16)
    p.add_argument('--antithetic', action='store_true', help='paired opposite noise tapes; repeats counts total rollouts')
    p.add_argument('--noise-std', type=float, default=.1)
    p.add_argument('--fd', type=float, default=.001)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if set(a.clips)&set(a.eval_clips) or min(a.frames_per_clip, a.eval_frames_per_clip, a.horizon) < 1 or a.warmup < 10 or a.repeats < 2 or a.noise_std <= 0 or a.fd <= 0 or (a.antithetic and (a.repeats < 4 or a.repeats % 2)):
        p.error('invalid configuration')
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    pcfg, wcfg = config_from_checkpoint(ck)
    if ck['config']['env'].get('anchor_obs', False):
        raise ValueError('anchor observations unsupported')
    a.future = int(ck['config']['env']['future_frames'])
    a.history = int(ck['config']['policy_trainer']['history'])
    a.gamma = float(ck['config']['policy_trainer']['gamma'])
    a.eval_horizon = a.horizon
    ccfg = CriticCfg(**{k: tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor.load_state_dict(ck['policy'])
    critic.load_state_dict(ck['critic'])
    actor.eval(); critic.eval().requires_grad_(False)
    files = discover_npz(a.npz_dir, names=a.clips+a.eval_clips)
    if len(files) != len(a.clips)+len(a.eval_clips):
        raise ValueError('missing clips')
    frames, _, labels, _ = frame_samples(files, a, a.future)
    gc.collect()
    env = None
    try:
        env, obs, bad, start, io, ctx, cmd, _, _ = make_setup(files, frames, 1, a.seed, a.warmup, a.history, a.future, actor, True)
        if bad.any():
            raise ValueError('warmup reset: choose a fresh seed')
        lib = cmd.library
        reconstructed = flatten_history_term_major(torch.stack([analytic_observation(start.hist_states[t],start.hist_actions[t],start.xi,lib.default_joint_pos) for t in range(a.history)]))
        obs_error = float((reconstructed-obs['policy']).abs().max())
        goal_error = float((lib.goal_window(start.frame,a.future)-obs['goal']).abs().max())
        if obs_error > 1e-3 or goal_error > 1e-4:
            raise ValueError('initial observation or goal mismatch')
        robot = env.scene['robot']
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device='cpu'), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(wcfg,heads,'cpu'); wm.load_state_dict(ck['wm']); wm.eval().requires_grad_(False)
        n = len(start.state)
        _,_,objective = model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,torch.zeros(a.horizon,n,29))
        baseline_grad = flattened_grad(objective.mean(), actor)
        if not torch.isfinite(baseline_grad).all() or baseline_grad.norm() < 1e-12:
            raise ValueError('invalid model baseline gradient')
        direction = baseline_grad/baseline_grad.norm()
        base_weights = parameters_to_vector(actor.parameters()).detach().clone()
        del objective, baseline_grad
        rng = torch.Generator().manual_seed(3400000+a.seed)
        results = []
        for i in range(a.repeats):
            if not a.antithetic or i % 2 == 0:
                tape = a.noise_std*torch.randn(a.horizon,n,29,generator=rng)
            else:
                tape = -tape
            ro, vals, obj = model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,tape)
            path = float(torch.dot(flattened_grad(obj.mean(),actor),direction))
            del ro, vals, obj
            with torch.no_grad():
                ro, vals, obj = model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,tape)
                observations = policy_observations(wm,lib,start,ro)
                boot = a.gamma**a.horizon*ro.survival[-1]*vals[-1]
                baseline = torch.stack([a.gamma**t*ro.survival[t]*vals[t] for t in range(a.horizon)])
            common = (actor,observations,ro.goals[:-1],ro.actions,tape,ro.rewards,ro.survival[:-1],boot,
                      a.gamma,a.noise_std,a.history,heads.reward_cfg.dt*heads.reward_cfg.w_action_rate)
            plain, check = score_surrogate(*common)
            score = float(torch.dot(flattened_grad(plain.mean(),actor),direction))
            centered, check2 = score_surrogate(*common,baseline=baseline)
            score_baseline = float(torch.dot(flattened_grad(centered.mean(),actor),direction))
            if max(check['action_replay_max_error'],check['history_replay_max_error'],check2['action_replay_max_error']) > 1e-4:
                raise ValueError(f'score replay mismatch: {check} {check2}')
            with torch.no_grad():
                objective_fd = []
                try:
                    for sign in (1.,-1.):
                        vector_to_parameters(base_weights+sign*a.fd*direction,actor.parameters())
                        _,_,j = model_rollout(wm,actor,critic,lib,start,a.horizon,a.future,a.gamma,tape)
                        objective_fd.append(float(j.mean()))
                finally:
                    vector_to_parameters(base_weights,actor.parameters())
            fd = (objective_fd[0]-objective_fd[1])/(2*a.fd)
            results.append({'pathwise': path, 'score': score, 'score_critic_baseline': score_baseline,
                            'fixed_tape_fd': fd, **check})
            print(f'{i+1}/{a.repeats} path={path:.4f} score={score:.4f} base={score_baseline:.4f} fd={fd:.4f}',flush=True)
            del ro, vals, obj, observations, plain, centered, common
            gc.collect()
        keys = ('pathwise','score','score_critic_baseline','fixed_tape_fd')
        if a.antithetic:
            summary = {k:describe([(results[2*i][k]+results[2*i+1][k])/2 for i in range(a.repeats//2)]) for k in keys}
            comparison = {}
            for k in keys:
                positives = torch.tensor([results[2*i][k] for i in range(a.repeats//2)],dtype=torch.float64)
                negatives = torch.tensor([results[2*i+1][k] for i in range(a.repeats//2)],dtype=torch.float64)
                pooled = torch.cat((positives,negatives))
                # Estimated iid variance for averaging two independent draws
                # with the same marginal distribution; noisy for few pairs.
                iid_est = float(pooled.var(unbiased=True)/2)
                pair_est = float(((positives+negatives)/2).var(unbiased=True))
                comparison[k] = {'pair_variance':pair_est,'iid_two_draw_variance_est':iid_est,
                                 'ratio':pair_est/iid_est if iid_est > 0 else None}
        else:
            summary = {k:describe([r[k] for r in results]) for k in keys}
            comparison = None
        out = {'checkpoint':str(a.ckpt),'seed':a.seed,'frames':frames.tolist(),'labels':labels,'horizon':a.horizon,
               'repeats':a.repeats,'antithetic':a.antithetic,'noise_std':a.noise_std,'fd':a.fd,'initial_obs_error':obs_error,'initial_goal_error':goal_error,
               'direction':'unit clean imagined gradient at fixed starts','results':results,'summary':summary,'antithetic_comparison':comparison,
               'real_warmup_transitions':len(frames)*a.warmup,
               'notes':'Model-only, fixed G1 checkpoint and same reference starts, no observation noise/domain randomization. Includes soft survival and frozen critic bootstrap. Gaussian score retains direct actor-dependence through clean-action history and action-rate reward. The baseline is pre-action critic times discounted survival. Fixed-tape pathwise FD checks model, not real simulator. Noise repeats share reference starts; no real score or policy training.'}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        if env is not None: env.close()


if __name__ == '__main__':
    main()
