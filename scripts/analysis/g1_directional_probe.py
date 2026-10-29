"""CPU-only paired action-response diagnostic for an existing SuperMimic checkpoint.

Reference-frame resets and deterministic domain settings are deliberate. This is
not an on-policy replay distribution or a training-performance evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("WARP_CACHE_PATH", r"D:\MinicondaEnv\warp-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, FOOT_GEOM_REGEX, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.wm_trainer import reference_at

torch.set_num_threads(1)


def config_from_checkpoint(ck):
    p = {k: tuple(v) if isinstance(v, list) else v for k, v in ck["config"]["policy"].items()}
    w = dict(ck["config"]["world_model"])
    ss = {k: tuple(v) if isinstance(v, list) else v for k, v in w.pop("state_spec").items()}
    for key, value in list(w.items()):
        if isinstance(value, list):
            w[key] = tuple(value)
    return PolicyCfg(**p), WorldModelCfg(state_spec=StateSpecCfg(**ss), **w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("logs/lafan1-rtx5060-eager/ckpt_0001500.pt"))
    parser.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    parser.add_argument("--clips", nargs="+", default=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.05, 0.2, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Run the checkpoint policy on duplicated worlds before branching.")
    parser.add_argument("--max-initial-nse", type=float, default=1e-8,
                        help="Exclude pairs whose normalized initial state difference exceeds this value.")
    parser.add_argument("--output", type=Path, default=Path("docs/analysis/g1_directional_probe.json"))
    args = parser.parse_args()
    if args.frames_per_clip < 1 or args.warmup_steps < 0 or args.max_initial_nse < 0 or any(d <= 0 for d in args.deltas):
        parser.error("frames-per-clip and deltas must be positive; warmup-steps must be nonnegative")
    t0 = time.perf_counter()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pcfg, wcfg = config_from_checkpoint(ck)
    policy = Policy(pcfg)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    files = discover_npz(args.npz_dir, names=args.clips)
    if len(files) != len(args.clips):
        raise ValueError(f"requested clips {args.clips}, found {[str(f) for f in files]}")
    n = len(files) * args.frames_per_clip
    env_args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files], num_envs=2 * len(args.deltas) * n,
        future_frames=int(ck["config"]["env"]["future_frames"]), action_noise_std=0.0,
        obs_noise=False, reset_noise=False, domain_randomization=False, sampling_mode="start",
        record_transitions=False, seed=args.seed, anchor_obs=bool(ck["config"]["env"].get("anchor_obs", False)),
    )
    env = make_env(env_args, device="cpu")
    try:
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        io = StateIO(env)
        context = SimContext(env, FOOT_GEOM_REGEX, "torso_link")
        fk = G1Kinematics(load_g1_model(), device="cpu")
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        wm = WorldModel(wcfg, heads, "cpu")
        wm.load_state_dict(ck["wm"])
        wm.eval()
        gen = torch.Generator().manual_seed(args.seed)
        frames, clip_labels = [], []
        for clip in command.library.clips:
            offsets = torch.randint(0, max(clip.length - 2, 1), (args.frames_per_clip,), generator=gen)
            frames.append(offsets + clip.start)
            clip_labels.extend([clip.name] * args.frames_per_clip)
        frames = torch.cat(frames)
        command.forced_frame[:] = frames.repeat(2 * len(args.deltas))
        obs, _ = env.reset()
        with torch.no_grad():
            warmup_done = torch.zeros(env.num_envs, dtype=torch.bool)
            for _ in range(args.warmup_steps):
                raw_action = policy(obs["policy"], obs["goal"])
                shared_action = raw_action[:n].repeat(2 * len(args.deltas), 1)
                obs, _, term, tout, _ = env.step(shared_action)
                warmup_done |= term | tout
            start = io.read()
            xi = context.read()
            ref = reference_at(command.library, command.frame)
            prev_action = env.action_manager.action.clone()
            raw_action = policy(obs["policy"], obs["goal"])
            action = raw_action[:n].repeat(2 * len(args.deltas), 1)
            direction = torch.randn(n, action.shape[-1], generator=gen)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            perturbed = action.clone()
            for i, delta in enumerate(args.deltas):
                perturbed[(2 * i) * n : (2 * i + 1) * n] += delta * direction
                perturbed[(2 * i + 1) * n : (2 * i + 2) * n] -= delta * direction
            pred = wm.step(start, perturbed, xi)
            model_terms = wm.heads.reward(pred, perturbed, prev_action, ref)
            model_reward = model_terms["total"]
            _, sim_reward, terminated, timeout, _ = env.step(perturbed)
            tracking_terms = ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori",
                              "motion_body_lin_vel", "motion_body_ang_vel")
            sim_tracking = sum(env.reward_manager.get_term(name) for name in tracking_terms)
            model_tracking = sum(model_terms[name] for name in tracking_terms)
            observed = io.read()
            groups = []
            for i, delta in enumerate(args.deltas):
                plus = slice(2 * i * n, (2 * i + 1) * n)
                minus = slice((2 * i + 1) * n, (2 * i + 2) * n)
                sim = wm.spec.error_vector(observed[plus], observed[minus]) / (2 * delta)
                model = wm.spec.error_vector(pred[plus], pred[minus]) / (2 * delta)
                initial_difference = wm.spec.normalized_sq_error(start[plus], start[minus])
                reset = (terminated[plus] | terminated[minus] | timeout[plus] | timeout[minus] |
                         warmup_done[plus] | warmup_done[minus])
                mismatch = initial_difference > args.max_initial_nse
                good = ~(reset | mismatch)
                action_center_difference = (raw_action[plus] - raw_action[minus]).abs().amax(-1)
                norm_sim = sim.norm(dim=-1)
                norm_model = model.norm(dim=-1)
                cosine = torch.nn.functional.cosine_similarity(sim, model, dim=-1, eps=1e-8)
                informative = good & (norm_sim > 1e-4) & (norm_model > 1e-4)
                err_plus = wm.spec.normalized_sq_error(pred[plus], observed[plus])
                err_minus = wm.spec.normalized_sq_error(pred[minus], observed[minus])
                sim_reward_direction = (sim_reward[plus] - sim_reward[minus]) / (2 * delta)
                model_reward_direction = (model_reward[plus] - model_reward[minus]) / (2 * delta)
                reward_informative = good & (sim_reward_direction.abs() > 1e-6)
                sim_tracking_direction = (sim_tracking[plus] - sim_tracking[minus]) / (2 * delta)
                model_tracking_direction = (model_tracking[plus] - model_tracking[minus]) / (2 * delta)
                tracking_informative = good & (sim_tracking_direction.abs() > 1e-6)
                parts = {}
                for key, first, last in (("root_position", 0, 3), ("root_orientation", 3, 6),
                                         ("root_linear_velocity", 6, 9), ("root_angular_velocity", 9, 12),
                                         ("joint_position", 12, 41), ("joint_velocity", 41, 70)):
                    s_part, m_part = sim[:, first:last], model[:, first:last]
                    valid_part = good & (s_part.norm(dim=-1) > 1e-4) & (m_part.norm(dim=-1) > 1e-4)
                    c_part = torch.nn.functional.cosine_similarity(s_part, m_part, dim=-1, eps=1e-8)
                    parts[key] = {"informative_pairs": int(valid_part.sum()),
                                  "mean_cosine": float(c_part[valid_part].mean()) if valid_part.any() else None,
                                  "fraction_negative_cosine": float((c_part[valid_part] < 0).float().mean()) if valid_part.any() else None}
                per_clip = {}
                for clip in command.library.clips:
                    m = torch.tensor([label == clip.name for label in clip_labels]) & good
                    per_clip[clip.name] = {"valid_pairs": int(m.sum()),
                        "response_mse": float(((sim[m] - model[m]) ** 2).mean()) if m.any() else None}
                groups.append({"delta": delta, "valid_pairs": int(good.sum()), "excluded_auto_resets": int(reset.sum()),
                    "excluded_initial_mismatch": int((mismatch & ~reset).sum()),
                    "informative_pairs": int(informative.sum()),
                    "initial_state_max_nse": float(initial_difference.max()),
                    "valid_initial_state_max_nse": float(initial_difference[good].max()) if good.any() else None,
                    "base_action_max_abs_difference": float(action_center_difference.max()),
                    "mean_response_mse": float(((sim[good] - model[good]) ** 2).mean()) if good.any() else None,
                    "mean_response_cosine": float(cosine[informative].mean()) if informative.any() else None,
                    "fraction_negative_cosine": float((cosine[informative] < 0).float().mean()) if informative.any() else None,
                    "mean_sim_response_norm": float(norm_sim[good].mean()) if good.any() else None,
                    "mean_model_response_norm": float(norm_model[good].mean()) if good.any() else None,
                    "mean_one_step_prediction_nse": float(((err_plus[good] + err_minus[good]) / 2).mean()) if good.any() else None,
                    "reward_direction_informative_pairs": int(reward_informative.sum()),
                    "reward_direction_sign_agreement": float(((sim_reward_direction[reward_informative] * model_reward_direction[reward_informative]) > 0).float().mean()) if reward_informative.any() else None,
                    "reward_direction_sim_abs_mean": float(sim_reward_direction[reward_informative].abs().mean()) if reward_informative.any() else None,
                    "reward_direction_model_abs_mean": float(model_reward_direction[reward_informative].abs().mean()) if reward_informative.any() else None,
                    "tracking_direction_informative_pairs": int(tracking_informative.sum()),
                    "tracking_direction_sign_agreement": float(((sim_tracking_direction[tracking_informative] * model_tracking_direction[tracking_informative]) > 0).float().mean()) if tracking_informative.any() else None,
                    "response_components": parts, "per_clip": per_clip,
                    "pairs": [{"clip": clip_labels[j], "frame_global": int(frames[j]), "valid": bool(good[j]),
                               "initial_state_nse": float(initial_difference[j]),
                               "response_cosine": float(cosine[j]) if informative[j] else None,
                               "response_mse": float(((sim[j] - model[j]) ** 2).mean()) if good[j] else None,
                               "sim_reward_direction": float(sim_reward_direction[j]) if good[j] else None,
                               "model_reward_direction": float(model_reward_direction[j]) if good[j] else None,
                               "sim_tracking_direction": float(sim_tracking_direction[j]) if good[j] else None,
                               "model_tracking_direction": float(model_tracking_direction[j]) if good[j] else None}
                              for j in range(n)]})
        result = {"checkpoint": str(args.ckpt), "checkpoint_iteration": int(ck["iteration"]), "seed": args.seed,
                  "device": "cpu", "clip_names": clip_labels, "frames_global": frames.tolist(),
                  "reset_distribution": "reference frames, noise-free, friction=1, no domain randomization",
                  "warmup_steps": args.warmup_steps, "num_pairs_per_delta": n, "groups": groups,
                  "max_initial_nse": args.max_initial_nse,
                  "reward_note": "Simulator body kinematics used by reward lag final qpos by one 5ms substep; analytic reward uses predicted final state.",
                  "wall_seconds": time.perf_counter() - t0}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "checkpoint_iteration": result["checkpoint_iteration"],
                          "groups": [{k: v for k, v in group.items() if k != "pairs"} for group in groups],
                          "wall_seconds": result["wall_seconds"]}, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
