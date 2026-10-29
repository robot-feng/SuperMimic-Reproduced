"""CPU G1 replay experiment: corrected vs shifted terminal labels, same samples."""
from __future__ import annotations

import argparse
import copy
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
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.world_model import TerminationHead

torch.set_num_threads(1)


def collect(ck, files, seed, worlds, steps):
    pcfg, wcfg = config_from_checkpoint(ck)
    policy = Policy(pcfg)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    cfg = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=worlds,
        future_frames=int(ck["config"]["env"]["future_frames"]), action_noise_std=0.1,
        obs_noise=True, reset_noise=True, domain_randomization=True, sampling_mode="uniform",
        replay_capacity=steps * 2 + 8, record_transitions=True, seed=seed,
        anchor_obs=bool(ck["config"]["env"].get("anchor_obs", False)))
    env = make_env(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        with torch.no_grad():
            for _ in range(steps):
                obs, _, _, _, _ = env.step(policy(obs["policy"], obs["goal"]))
            buf = env.recorder_manager.get_term("replay").buffer
            indexes = buf.window_mask(1).nonzero()
            e, t = indexes.unbind(-1)
            tn = (t + 1) % buf.T
            state_next = buf.s[e, tn]
            ref = env.command_manager.get_term("motion").library.ref(buf.frame[e, t])
            ref_frame = ReferenceFrame(None, None, None, None, ref.root_pos_w, ref.root_quat_w)
            features = AnalyticHeads.termination_features(state_next[:, :3], state_next[:, 3:7], ref_frame)
            return {"features": features.clone(), "correct": buf.done[e, t].clone(),
                    "shifted": buf.done[e, tn].clone(), "env_id": e.clone()}, wcfg
    finally:
        env.close()


@torch.no_grad()
def metrics(head, x, y):
    logit = head(x)
    p = torch.sigmoid(logit)
    positive, negative = y.bool(), ~y.bool()
    pred = p >= 0.5
    return {"count": len(y), "positives": int(positive.sum()),
            "bce": float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y)),
            "brier": float((p-y).square().mean()), "accuracy": float((pred == positive).float().mean()),
            "recall": float(pred[positive].float().mean()) if positive.any() else None,
            "false_positive_rate": float(pred[negative].float().mean()) if negative.any() else None,
            "mean_positive_prob": float(p[positive].mean()) if positive.any() else None,
            "mean_negative_prob": float(p[negative].mean()) if negative.any() else None}


def finetune(init, train, target, seed, updates, batch, lr):
    head = copy.deepcopy(init)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(10000 + seed)
    for _ in range(updates):
        idx = torch.randint(len(target), (batch,), generator=gen)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(train[idx]), target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("logs/lafan1-rtx5060-eager/ckpt_0002000.pt"))
    p.add_argument("--npz-dir", type=Path, default=Path("data/npz/lafan1"))
    p.add_argument("--clips", nargs="+", default=["walk1_subject1", "dance1_subject1", "jumps1_subject1"])
    p.add_argument("--worlds", type=int, default=36)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--output", type=Path, default=Path("docs/analysis/g1_terminal_label_ablation.json"))
    args = p.parse_args()
    if args.worlds < 3 or args.steps < 8:
        p.error("worlds must be >= 3 and steps >= 8")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    files = discover_npz(args.npz_dir, names=args.clips)
    if len(files) != len(args.clips):
        raise ValueError("incomplete clip list")
    data, cfg = collect(ck, files, args.seed, args.worlds, args.steps)
    train = data["env_id"] < (args.worlds * 2 // 3)
    test = ~train
    initial = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
    initial.load_state_dict({k.removeprefix("term_head."): v for k, v in ck["wm"].items() if k.startswith("term_head.")})
    original = metrics(initial, data["features"][test], data["correct"][test])
    updated = {}
    for label in ("correct", "shifted"):
        head = finetune(initial, data["features"][train], data[label][train], args.seed, args.updates, 256, 0.0002)
        updated[label] = metrics(head, data["features"][test], data["correct"][test])
    result = {"checkpoint": str(args.ckpt), "iteration": int(ck["iteration"]), "seed": args.seed,
              "worlds": args.worlds, "steps_per_world": args.steps, "head_updates": args.updates,
              "train_count": int(train.sum()), "test_count": int(test.sum()),
              "train_correct_positives": int(data["correct"][train].sum()),
              "train_shifted_positives": int(data["shifted"][train].sum()),
              "train_disagree": int((data["correct"][train] != data["shifted"][train]).sum()),
              "test_correct_positives": int(data["correct"][test].sum()),
              "baseline": original, "finetuned": updated,
              "notes": "labels sampled as 1-step windows; features use actual next states; head only; held-out envs, not unseen clips; no actor/critic update"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
