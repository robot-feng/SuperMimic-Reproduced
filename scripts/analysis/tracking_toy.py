"""Small, reproducible counterexamples for the SuperMimic algorithm review.

Run from the repository root: .venv/Scripts/python.exe scripts/analysis/tracking_toy.py
This does not load a checkpoint or run a robot training job.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from supermimic.world_model.heads import analytic_observation, anchor_relative_obs


def main() -> None:
    rng = np.random.default_rng(7)
    # One-step tracking x' = x + a. Hidden x cannot be inferred from a constant
    # observation; the full-observation linear policy can recover a = -x.
    train = rng.uniform(-1.0, 1.0, 40000)
    test = rng.uniform(-1.0, 1.0, 100000)
    target = -train
    hidden = np.linalg.lstsq(np.ones((len(train), 1)), target, rcond=None)[0]
    observed = np.linalg.lstsq(np.column_stack((np.ones(len(train)), train)), target, rcond=None)[0]
    hidden_error = np.mean((test + hidden[0]) ** 2)
    observed_error = np.mean((test + observed[0] + observed[1] * test) ** 2)

    # Actual implementation: translate the robot while fixing proprioception and
    # the reference. Proprioceptive history cannot reveal that translation.
    states = torch.zeros(2, 71)
    states[:, 2] = 0.8
    states[:, 3] = 1.0
    states[1, 0] = 0.5
    xi = torch.zeros(2, 33)
    obs = analytic_observation(states, torch.zeros(2, 29), xi, torch.zeros(29))
    ref_pos = torch.tensor([[0.0, 0.0, 0.8]]).expand(2, -1)
    ref_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
    anchor = anchor_relative_obs(states[:, :3], states[:, 3:7], ref_pos, ref_quat)

    # True dynamics x'=x+a; model has a small constant drift +eps. The model
    # optimizes one constant action for terminal x_H=0, regularized by lambda.
    eps, lam = 0.02, 0.1
    model_bias = {}
    for horizon in (8, 32):
        action = -horizon * eps / (horizon + lam)
        model_x = horizon * (action + eps)
        true_x = horizon * action
        model_bias[str(horizon)] = {
            "action": action,
            "predicted_terminal_x": model_x,
            "true_terminal_x": true_x,
            "true_terminal_squared_error": true_x**2,
        }

    # Noise discrepancy depends on nonlinear action response; linear dynamics
    # have no mean discrepancy in this zero-mean example.
    noise = rng.normal(0.0, 0.1, 1000000)
    action = 0.5
    nonlinear = {"clean": action**2, "noisy_mean": float(np.mean((action + noise) ** 2)),
                 "theory_noisy_mean": action**2 + 0.1**2}
    linear = {"clean": action, "noisy_mean": float(np.mean(action + noise))}

    # A frame-uniform reference-only proxy, not the replay sampler distribution.
    lengths = []
    for path in Path("data/npz/lafan1").glob("*.npz"):
        with np.load(path) as clip:
            lengths.append(len(clip["joint_pos"]))
    endpoint_proxy = {"clips": len(lengths), "total_frames": sum(lengths),
                      "frames_with_fewer_than_32_advances": sum(min(32, n) for n in lengths),
                      "fraction_if_frames_uniform": sum(min(32, n) for n in lengths) / sum(lengths) if lengths else None}
    result = {
        "seed": 7,
        "device": "cpu",
        "one_step_tracking": {"hidden_mse": float(hidden_error), "observed_mse": float(observed_error),
                              "hidden_theoretical_mse": 1 / 3},
        "actual_observation_translation": {"proprio_max_difference": float((obs[1] - obs[0]).abs().max()),
                                           "relative_anchor_position_difference": (anchor[1, :3] - anchor[0, :3]).tolist()},
        "model_bias": model_bias,
        "action_noise_nonlinear": nonlinear,
        "action_noise_linear": linear,
        "clip_endpoint_frame_uniform_proxy": endpoint_proxy,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
