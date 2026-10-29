"""Compare checkpoint critic values with simulator H-step bootstrapped targets (CPU).

This probes a small, noise-free reference-reset distribution. It does not estimate
full-episode Monte Carlo value or the training-distribution critic MSE.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("WARP_CACHE_PATH", r"D:\MinicondaEnv\warp-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from g1_directional_probe import config_from_checkpoint
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX
from supermimic.policy.nets import Critic, CriticCfg, Policy
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec

torch.set_num_threads(1)


def probe(path: Path, clips: list[Path], frames_per_clip: int, seed: int, warmup: int, horizon: int,
          corrections: dict[str, list[float]] | None = None, record_features: bool = False) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pcfg, wcfg = config_from_checkpoint(ck)
    ccfg = CriticCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["config"]["critic"].items()})
    policy, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    policy.load_state_dict(ck["policy"])
    critic.load_state_dict(ck["critic"])
    policy.eval()
    critic.eval()
    n = frames_per_clip * len(clips)
    cfg = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in clips], num_envs=n,
        future_frames=int(ck["config"]["env"]["future_frames"]), action_noise_std=0.0,
        obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=seed, anchor_obs=bool(ck["config"]["env"].get("anchor_obs", False)))
    env = make_env(cfg, device="cpu")
    try:
        cmd = env.command_manager.get_term("motion")
        io, context = StateIO(env), SimContext(env, FOOT_GEOM_REGEX, "torso_link")
        spec = StateSpec(wcfg.state_spec, "cpu")
        gen = torch.Generator().manual_seed(seed)
        frames, labels = [], []
        for clip in cmd.library.clips:
            remaining = clip.length - warmup - horizon - 2
            if remaining <= 0:
                raise ValueError(f"clip {clip.name} too short for selected probe")
            frames.append(clip.start + torch.randint(0, remaining, (frames_per_clip,), generator=gen))
            labels.extend([clip.name] * frames_per_clip)
        frames = torch.cat(frames)
        cmd.forced_frame[:] = frames
        obs, _ = env.reset()
        eligible = torch.ones(n, dtype=torch.bool)
        with torch.no_grad():
            for _ in range(warmup):
                obs, _, term, timeout, _ = env.step(policy(obs["policy"], obs["goal"]))
                eligible &= ~(term | timeout)

            def value(current_obs):
                s = io.read()
                ref = cmd.library.ref(cmd.frame)
                ref_frame = ReferenceFrame(None, None, None, None, ref.root_pos_w, ref.root_quat_w)
                e_root = AnalyticHeads.root_error_features(s, ref_frame)
                feat,goal,xi=spec.canonical(s),current_obs['goal'],spec.normalize_xi(context.read())
                original=critic(feat,goal,xi,e_root)
                if corrections is None and not record_features:
                    return {'original':original},None
                x=torch.cat((feat,goal*critic.goal_scale,xi,e_root),dim=-1)
                hidden=critic.mlp.net[:-1](x)
                if record_features and not torch.allclose(critic.mlp.net[-1](hidden).squeeze(-1),original,atol=1e-6,rtol=1e-6):
                    raise ValueError('saved hidden features do not reconstruct critic')
                out={'original':original}
                for name,coefficients in (corrections or {}).items():
                    b=torch.as_tensor(coefficients,dtype=hidden.dtype,device=hidden.device)
                    if len(b) not in (1,hidden.shape[-1]+1):raise ValueError('invalid correction shape')
                    out[name]=original+b[0]+(hidden@b[1:] if len(b)>1 else 0.)
                return out,hidden if record_features else None

            initial_values,initial_hidden = value(obs)
            initial_value = initial_values['original']
            gamma = float(ck["config"]["policy_trainer"]["gamma"])
            alive = eligible.clone()
            valid = eligible.clone()
            terminated_count = torch.zeros(n, dtype=torch.bool)
            returns = torch.zeros(n)
            for t in range(horizon):
                obs, rew, term, timeout, _ = env.step(policy(obs["policy"], obs["goal"]))
                returns += (gamma**t) * rew * alive
                valid &= ~(timeout & alive)
                terminated_count |= term & alive
                alive &= ~(term | timeout)
            final_values,final_hidden = value(obs)
            final_value = final_values['original']
            target = returns + (gamma**horizon) * final_value * alive
            residual = initial_value - target
            good = valid
            data = {"checkpoint": str(path), "iteration": int(ck["iteration"]), "seed": seed,
                    "warmup_steps": warmup, "horizon": horizon, "gamma": gamma, "total_starts": n,
                    "eligible_after_warmup": int(eligible.sum()), "valid_after_timeout": int(good.sum()),
                    "terminated_within_horizon": int((terminated_count & good).sum()),
                    "reference_start_frames": frames.tolist(), "clips": labels,
                    "value_mean": float(initial_value[good].mean()) if good.any() else None,
                    "target_mean": float(target[good].mean()) if good.any() else None,
                    "residual_mean": float(residual[good].mean()) if good.any() else None,
                    "residual_rmse": float(residual[good].square().mean().sqrt()) if good.any() else None,
                    "discounted_reward_mean": float(returns[good].mean()) if good.any() else None,
                    "bootstrap_survivors": int((alive & good).sum()),
                    "records": [{"clip": labels[i], "frame": int(frames[i]), "valid": bool(good[i]),
                                 "terminated": bool(terminated_count[i]), "value": float(initial_value[i]),
                                 "target": float(target[i]), "discounted_rewards": float(returns[i])}
                                for i in range(n)]}
            if corrections is not None:
                variants={}
                for name in initial_values:
                    v0,vend=initial_values[name],final_values[name]
                    residuals=v0-returns-(gamma**horizon)*vend*alive
                    if not torch.isfinite(residuals[good]).all():raise ValueError('nonfinite corrected critic residual')
                    def stats(mask):
                        if not mask.any():return {'n':0}
                        err=residuals[mask]
                        return {'n':int(mask.sum()),'residual_mean':float(err.mean()),
                                'residual_rmse':float(err.square().mean().sqrt())}
                    variants[name]={'overall':stats(good),'failed':stats(good & terminated_count),
                                    'survived':stats(good & alive),
                                    'records':[{'index':i,'value':float(v0[i]),'terminal_value':float(vend[i]),
                                                'residual':float(residuals[i])} for i in range(n)]}
                data['critic_variants']=variants
            if record_features:
                if not torch.isfinite(initial_hidden).all() or not torch.isfinite(final_hidden).all():
                    raise ValueError('nonfinite saved critic features')
                data['critic_hidden_features']={'initial':initial_hidden.tolist(),'final':final_hidden.tolist(),
                    'notes':'Final hidden features on failed environments are post-reset and must be masked; on survivors they are the 200-step bootstrap inputs.'}
        return data
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", type=Path, default=[
        Path("logs/lafan1-rtx5060-eager/ckpt_0001000.pt"),
        Path("logs/lafan1-rtx5060-eager/ckpt_0001500.pt"),
        Path("logs/lafan1-rtx5060-eager/ckpt_0002000.pt")])
    p.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    p.add_argument("--clips", nargs="+", default=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    p.add_argument("--frames-per-clip", type=int, default=4)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path("docs/analysis/critic_real_probe.json"))
    args = p.parse_args()
    files = discover_npz(args.npz_dir, names=args.clips)
    if len(files) != len(args.clips):
        raise ValueError(f"clip list incomplete: {args.clips}")
    results = [probe(c, files, args.frames_per_clip, args.seed, args.warmup, args.horizon) for c in args.checkpoints]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"definition": f"{args.horizon}-step simulator reward plus same checkpoint critic bootstrap; noise-free fixed-friction, reference reset", "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": [{k: v for k, v in r.items() if k != "records"} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
