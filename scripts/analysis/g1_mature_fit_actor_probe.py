"""Compare mature-checkpoint actor directions with a replay-refit G1 dynamics model."""
import argparse
import json
import os
import pathlib
import pickle
import types
from pathlib import Path

from g1_tail_policy_switch_probe import available_memory, sha


def checkpoint_pickle_module(windows_as_pure):
    """Read Windows path metadata without constructing a WindowsPath on Linux."""
    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if windows_as_pure and module == 'pathlib' and name == 'WindowsPath':
                return pathlib.PureWindowsPath
            return super().find_class(module, name)
    compatible = types.ModuleType('checkpoint_pickle')
    compatible.Unpickler = CompatibleUnpickler
    compatible.load = pickle.load
    compatible.loads = pickle.loads
    compatible.dump = pickle.dump
    compatible.dumps = pickle.dumps
    return compatible


def validate_candidate(prior, checkpoint_path, weights_path):
    if (prior['source'] != 'direct8' or prior['position_alpha'] != 1.0 or
        prior['updates'] != 60 or prior['horizon'] != 8 or
        sha(checkpoint_path) != prior['checkpoint_sha256'] or
        sha(weights_path) != prior['dynamics_sha256']):
        raise ValueError('checkpoint or candidate changed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prior_path = Path('docs/analysis/g1_paired_wm_direct8.json')
    weights_path = Path('docs/analysis/g1_paired_wm_direct8.pt')
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    # Reports generated on Windows store backslashes; accept them on Linux too.
    ckpath = Path(prior['checkpoint'].replace('\\', '/'))
    prior_sha, script_sha = sha(prior_path), sha(Path(__file__))
    validate_candidate(prior, ckpath, weights_path)
    free = available_memory()
    if free is not None and free < 2200 * 1024 * 1024:
        raise RuntimeError(f'insufficient available physical memory: {free} bytes; simulator not started')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('WARP_CACHE_PATH', r'D:\MinicondaEnv\warp-cache')
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    import gc
    import torch
    from torch.nn.utils import parameters_to_vector
    from g1_directional_probe import config_from_checkpoint
    from g1_grounded_gradient_probe import make_setup
    from g1_noisy_imagination_direction import evaluate, frame_samples, select_and_repeat
    from g1_policy_direction_probe import model_objective
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Critic, CriticCfg, Policy
    from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
    from supermimic.world_model.world_model import WorldModel
    torch.set_num_threads(1)
    ck = torch.load(ckpath, map_location='cpu', weights_only=False, mmap=True,
                    pickle_module=checkpoint_pickle_module(os.name != 'nt'))
    if ck['iteration'] != 2000:
        raise ValueError('unexpected checkpoint iteration')
    pcfg, wcfg = config_from_checkpoint(ck)
    ccfg = CriticCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck['config']['critic'].items()})
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor.load_state_dict(ck['policy']); critic.load_state_dict(ck['critic'])
    actor.eval(); critic.eval().requires_grad_(False)
    args.clips = ['walk1_subject1', 'dance1_subject1', 'jumps1_subject1']
    args.eval_clips = ['walk2_subject1', 'dance2_subject1', 'jumps1_subject2']
    args.frames_per_clip = args.eval_frames_per_clip = args.eval_replicates = 1
    args.warmup, args.horizon, args.eval_horizon, args.eps = 10, 8, 80, .005
    args.history = int(ck['config']['policy_trainer']['history'])
    args.future = int(ck['config']['env']['future_frames'])
    args.gamma = float(ck['config']['policy_trainer']['gamma'])
    if ck['config']['env'].get('anchor_obs', False):
        raise ValueError('anchor observations unsupported')
    files = discover_npz(Path('data/npz/lafan1'), names=args.clips + args.eval_clips)
    if len(files) != 6:
        raise ValueError('missing clips')
    train_frames, test_frames, train_clips, test_clips = frame_samples(files, args, args.future)
    env = None
    try:
        env, obs, bad, starts, _, _, cmd, _, _ = make_setup(files, train_frames, 1, args.seed, args.warmup,
                                                              args.history, args.future, actor, True)
        eligible = (~bad).nonzero().flatten()
        if len(eligible) < 2:
            raise ValueError('fewer than two valid gradient starts')
        starts = select_and_repeat(starts, eligible, 1)
        lib = cmd.library
        reconstructed = flatten_history_term_major(torch.stack([
            analytic_observation(starts.hist_states[t], starts.hist_actions[t], starts.xi, lib.default_joint_pos)
            for t in range(args.history)]))
        obs_error = float((reconstructed - obs['policy'][eligible]).abs().max())
        goal_error = float((lib.goal_window(starts.frame, args.future) - obs['goal'][eligible]).abs().max())
        if obs_error > 1e-3 or goal_error > 1e-4:
            raise ValueError('gradient state does not match actor inputs')
        robot = env.scene['robot']
        heads = AnalyticHeads(G1Kinematics(load_g1_model(), device='cpu'), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(wcfg, heads, 'cpu')
        fitted = torch.load(weights_path, map_location='cpu', weights_only=True)
        wm.load_state_dict(ck['wm'])
        if set(fitted) != set(wm.dynamics.state_dict()):
            raise ValueError('candidate must contain exactly dynamics parameters')
        grads, objectives = [], []
        for name in ('original', 'refit'):
            if name == 'refit':
                wm.dynamics.load_state_dict(fitted, strict=True)
            wm.eval().requires_grad_(False)
            objective, survival, reward, bootstrap, _ = model_objective(wm, actor, critic, lib, starts,
                                                                          args.horizon, args.future, args.gamma)
            g = torch.cat([x.detach().reshape(-1) for x in torch.autograd.grad(objective, tuple(actor.parameters()))])
            if not torch.isfinite(g).all() or g.norm() < 1e-12:
                raise ValueError('invalid actor gradient')
            grads.append(g)
            objectives.append({'name': name, 'objective': float(objective.detach()), 'grad_norm': float(g.norm()),
                               'survival': survival, 'reward_mean': float(reward.detach().mean()),
                               'bootstrap_mean': float(bootstrap.detach().mean())})
        directions = [g / g.norm() for g in grads]
        cosine = float(torch.dot(grads[0].double(), grads[1].double()) / (grads[0].double().norm() * grads[1].double().norm()))
        base = parameters_to_vector(actor.parameters()).detach().clone()
        frame_after = starts.frame.tolist()
    finally:
        if env is not None:
            env.close()
    del env, obs, starts, ck, fitted
    gc.collect()
    result, _ = evaluate(files, test_frames, test_clips, args.seed, actor, critic, wm,
                         directions, base, args, sigma=.1)
    if any(result[section]['base_control_max_abs_reward_difference'] > 1e-4 or
           result[section]['base_control_max_abs_bootstrap_difference'] > 1e-4 or
           result[section]['base_control_failure_disagreements'] for section in ('short', 'long')):
        raise ValueError('base repeat audit failed')
    if (sha(ckpath) != prior['checkpoint_sha256'] or sha(weights_path) != prior['dynamics_sha256'] or
        sha(prior_path) != prior_sha or sha(Path(__file__)) != script_sha):
        raise ValueError('inputs changed during evaluation')
    output = {'seed': args.seed, 'checkpoint_sha256': prior['checkpoint_sha256'], 'fit_sha256': prior_sha,
              'weights_sha256': prior['dynamics_sha256'], 'script_sha256': script_sha, 'candidate': 'mature_direct8_60',
              'gradient_clips': train_clips, 'gradient_frames': train_frames.tolist(),
              'gradient_eligible_indices': eligible.tolist(), 'gradient_frames_after_warmup': frame_after,
              'evaluation_clips': test_clips, 'evaluation_frames': test_frames.tolist(),
              'horizon': args.horizon, 'eval_horizon': args.eval_horizon, 'eps': args.eps,
              'direction_mapping': {'clean_gradient': 'original_model', 'noise_gradient': 'refit_model'},
              'gradient_cosine': cosine, 'gradient_input_observation_error': obs_error,
              'gradient_input_goal_error': goal_error, 'objectives': objectives, 'results': result,
              'counted_simulator_transitions': len(train_frames) * args.warmup + 6 * len(test_frames) * (args.warmup + args.eval_horizon),
              'notes': 'Frozen checkpoint, same actor/critic/head and reference starts; changed only dynamics; unit-norm directions, not Adam steps. True 80-step reward truncated; evaluation includes action noise but gradient imagination is noise-free.'}
    with args.output.open('x', encoding='utf8') as f:
        f.write(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in output.items() if k != 'results'} | {'short': result['short'], 'long': result['long']}, indent=2))


if __name__ == '__main__':
    main()
