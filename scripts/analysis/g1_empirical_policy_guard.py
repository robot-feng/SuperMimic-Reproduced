"""Backtrack a proposed policy update on empirical Gaussian action KL."""
import math
import torch


@torch.no_grad()
def reference_actions(policy,observations,goals):
    if len(observations)<1 or len(goals)!=len(observations):raise ValueError('empty or mismatched policy states')
    out=policy(observations,goals).detach()
    if not torch.isfinite(out).all():raise ValueError('nonfinite reference actions')
    return out


@torch.no_grad()
def guard(policy,old_params,observations,goals,old_actions,sigma,limit=None,max_backtracks=10):
    """Interpolate old and proposed weights; preserve optimizer moments.

    The guarantee holds only on the supplied finite, fixed observation set.
    """
    if not math.isfinite(sigma) or sigma<=0 or (limit is not None and (not math.isfinite(limit) or limit<0)):
        raise ValueError('invalid KL configuration')
    params=list(policy.parameters())
    if len(params)!=len(old_params) or max_backtracks<0:raise ValueError('invalid actor parameters')
    proposal=[p.detach().clone() for p in params]
    if any(p.shape!=old.shape for p,old in zip(params,old_params)):raise ValueError('parameter shape mismatch')
    baseline=None;accepted=None;actual=None;trial=0;proposed_distribution=None;accepted_distribution=None
    for i in range(max_backtracks+1):
        alpha=2.**(-i) if limit is not None else 1.
        if i:
            for p,old,new in zip(params,old_params,proposal):p.copy_(old+alpha*(new-old))
        current=policy(observations,goals)
        if current.shape!=old_actions.shape or not torch.isfinite(current).all():raise ValueError('invalid candidate actions')
        per_state=(current-old_actions).square().sum(-1)/(2*sigma*sigma)
        kl=float(per_state.mean());trial+=1
        distribution={'p95':float(torch.quantile(per_state,.95)),'max':float(per_state.max())}
        if baseline is None:baseline=kl;proposed_distribution=distribution
        if limit is None or kl<=limit:
            accepted=alpha;actual=kl;accepted_distribution=distribution;break
    if accepted is None:
        for p,old in zip(params,old_params):p.copy_(old)
        restored=policy(observations,goals)
        per_state=(restored-old_actions).square().sum(-1)/(2*sigma*sigma)
        actual=float(per_state.mean());trial+=1;accepted=0.
        accepted_distribution={'p95':float(torch.quantile(per_state,.95)),'max':float(per_state.max())}
        if actual>1e-10:raise ValueError('failed to restore old policy')
    return {'proposed_empirical_kl':baseline,'accepted_empirical_kl':actual,'accepted_alpha':accepted,
            'proposed_per_state_kl':proposed_distribution,'accepted_per_state_kl':accepted_distribution,
            'policy_forward_evaluations':trial,'num_states':len(observations),'kl_limit':limit}
