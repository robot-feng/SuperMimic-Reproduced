"""Controlled, one-time actor Adam first-moment intervention for experiments."""
import torch


@torch.no_grad()
def reset_first_moment(optimizer):
    """Zero only existing exp_avg tensors; retain step and second moments exactly."""
    states = [optimizer.state[p] for group in optimizer.param_groups for p in group['params']]
    if not states or any(not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys() for state in states):
        raise ValueError('optimizer must have fully initialized Adam state')
    if any(not torch.isfinite(s['exp_avg']).all() for s in states):
        raise ValueError('nonfinite first moment')
    norm = sum(float(s['exp_avg'].double().square().sum()) for s in states) ** .5
    for state in states:
        state['exp_avg'].zero_()
    return {'first_moment_norm_before': norm, 'first_moment_norm_after': 0.,
            'parameter_tensors': len(states)}
