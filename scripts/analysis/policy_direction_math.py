"""Per-trajectory equivalent of the SuperMimic policy objective."""
import torch


def replace_forward(model_state, real_state):
    """Use the real value in forward, and only the learned Jacobian in backward."""
    return model_state + (real_state - model_state).detach()


def objective_components(ro, terminal_values, gamma):
    horizon = ro.rewards.shape[0]
    discounts = gamma ** torch.arange(horizon, device=ro.rewards.device, dtype=ro.rewards.dtype)
    reward = (discounts[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    bootstrap = torch.zeros_like(reward) if terminal_values is None else gamma**horizon * ro.survival[-1] * terminal_values
    return reward, bootstrap


def per_trajectory_objective(ro, terminal_values, gamma):
    reward, bootstrap = objective_components(ro, terminal_values, gamma)
    return reward + bootstrap


def advance_true_rollout(total, alive, failed, timed_out, reward, terminated, timeout, discount):
    """Count the terminal reward once and stop accumulating after reset."""
    total = total + discount * torch.where(alive, reward, torch.zeros_like(reward))
    failed = failed | (alive & terminated)
    timed_out = timed_out | (alive & timeout)
    alive = alive & ~(terminated | timeout)
    return total, alive, failed, timed_out
