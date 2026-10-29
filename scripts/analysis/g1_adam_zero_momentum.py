"""Compute Adam's counterfactual next update for an all-zero gradient."""
import math
import torch


@torch.no_grad()
def zero_gradient_delta(optimizer,params):
    if len(optimizer.param_groups)!=1 or list(optimizer.param_groups[0]['params'])!=list(params):
        raise ValueError('expected a single ordered optimizer group')
    g=optimizer.param_groups[0]
    if g['weight_decay']!=0 or g['amsgrad'] or g.get('maximize',False):raise ValueError('unsupported Adam options')
    b1,b2=g['betas'];lr=g['lr'];eps=g['eps'];out=[]
    for p in params:
        state=optimizer.state[p]
        if not {'step','exp_avg','exp_avg_sq'}<=state.keys():raise ValueError('missing Adam moments')
        step=float(state['step'])+1
        if step!=int(step) or step<1:raise ValueError('invalid Adam step')
        denom=(state['exp_avg_sq']*b2).sqrt()/math.sqrt(1-b2**step)+eps
        out.append((-lr/(1-b1**step))*(state['exp_avg']*b1)/denom)
    return out


def compare_updates(proposed,zero,full_gradient,reward_gradient):
    if len({len(proposed),len(zero),len(full_gradient),len(reward_gradient)})!=1:raise ValueError('length mismatch')
    pp=zz=dd=ff=rr=pf=zf=pr=zr=pz=0.
    for p,z,f,r in zip(proposed,zero,full_gradient,reward_gradient):
        p,z,f,r=(x.double() for x in (p,z,f,r));d=p-z
        if not all(torch.isfinite(x).all() for x in (p,z,f,r)):raise ValueError('nonfinite data')
        pp+=float((p*p).sum());zz+=float((z*z).sum());dd+=float((d*d).sum())
        ff+=float((f*f).sum());rr+=float((r*r).sum());pf+=float((p*f).sum());zf+=float((z*f).sum())
        pr+=float((p*r).sum());zr+=float((z*r).sum());pz+=float((p*z).sum())
    cos=lambda dot,a,b:dot/(a*b)**.5 if a>0 and b>0 else None
    return {'proposed_norm':pp**.5,'zero_gradient_adam_norm':zz**.5,'difference_norm':dd**.5,
            'proposed_vs_zero_cosine':cos(pz,pp,zz),'proposed_vs_full_gradient_cosine':cos(pf,pp,ff),
            'zero_vs_full_gradient_cosine':cos(zf,zz,ff),'proposed_vs_reward_gradient_cosine':cos(pr,pp,rr),
            'zero_vs_reward_gradient_cosine':cos(zr,zz,rr)}
