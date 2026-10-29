"""Conditional score estimator with SuperMimic clean-action observation history.

Research-only helper. Observations use term-major 10-frame layout, with actions
in the last 29*history entries. State observations and realized rewards are
held fixed, while previously issued clean commands retain their direct parameter
dependence. This is required for likelihood-ratio gradients with policy memory.
"""
from __future__ import annotations

import torch


def score_surrogate(policy, observations, goals, recorded_actions, noise, rewards,
                    survival, bootstrap, gamma, sigma, history, action_rate_scale,
                    active=None, detach_history=False, baseline=None):
    """Return a per-trajectory score surrogate and replay integrity diagnostics.

    rewards[t] is the unweighted r_{t+1}; survival[t] gates that reward and is
    fixed in score differentiation; bootstrap is already discounted and gated.
    ``active[t]`` checks parity only on still-valid real trajectories.
    """
    if sigma <= 0 or history < 1 or observations.ndim != 3:
        raise ValueError('invalid score input')
    horizon, n, obs_dim = observations.shape
    if recorded_actions.shape[:2] != (horizon, n) or noise.shape != recorded_actions.shape:
        raise ValueError('action/noise mismatch')
    action_dim = recorded_actions.shape[-1]
    if goals.shape[:2] != (horizon, n) or rewards.shape != (horizon, n) or survival.shape != (horizon, n) or bootstrap.shape != (n,):
        raise ValueError('trajectory shape mismatch')
    action_size = history*action_dim
    if obs_dim < action_size:
        raise ValueError('observation history too short')
    if active is None:
        active = torch.ones(horizon, n, dtype=torch.bool, device=observations.device)
    if active.shape != (horizon,n):
        raise ValueError('active mask mismatch')
    if baseline is not None and baseline.shape != (horizon,n):
        raise ValueError('baseline shape mismatch')
    action_history = observations[0, :, -action_size:].detach().reshape(n, history, action_dim)
    previous = action_history[:, -1]
    commands, scores, direct = [], [], []
    parity, history_error = 0., 0.
    for t in range(horizon):
        expected = observations[t, :, -action_size:].detach().reshape(n, history, action_dim)
        if active[t].any():
            history_error = max(history_error, float((action_history.detach()[active[t]]-expected[active[t]]).abs().max()))
        obs = torch.cat((observations[t, :, :-action_size].detach(), action_history.flatten(1)), dim=-1)
        command = policy(obs, goals[t].detach())
        if active[t].any():
            parity = max(parity, float((command.detach()[active[t]]-recorded_actions[t][active[t]]).abs().max()))
        # Noise is the realized executed action minus the recorded clean mean.
        # The mean remains differentiable, including via clean-action history.
        scores.append((noise[t].detach()*command).sum(-1)/(sigma*sigma))
        direct.append(gamma**t*survival[t].detach()*action_rate_scale*(command-previous).square().sum(-1))
        commands.append(command)
        previous = command
        next_command = command.detach() if detach_history else command
        action_history = torch.cat((action_history[:, 1:], next_command[:, None]), dim=1)
    contrib = torch.stack([gamma**t * survival[t].detach() * rewards[t].detach() for t in range(horizon)])
    future = contrib.flip(0).cumsum(0).flip(0) + bootstrap.detach()[None]
    if baseline is not None:
        future = future-baseline.detach()
    surrogate = (torch.stack(scores)*future).sum(0)+torch.stack(direct).sum(0)
    return surrogate, {'action_replay_max_error': parity, 'history_replay_max_error': history_error}
