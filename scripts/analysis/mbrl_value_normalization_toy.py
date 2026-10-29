"""Check survival-aware value normalization and actor gradients in a scalar toy."""
import argparse
import json
import math
import random
from pathlib import Path


def objective(rewards, deaths, terminal, gamma):
    survival, total = 1., 0.
    for t, (r, d) in enumerate(zip(rewards, deaths, strict=True)):
        total += gamma**t * survival * r
        survival *= 1. - d
    return total + gamma**len(rewards)*survival*terminal


def gae(rewards, deaths, values, gamma, lam):
    ret = values[-1]; output = []
    for t in range(len(rewards)-1, -1, -1):
        ret = rewards[t] + gamma*(1.-deaths[t])*((1.-lam)*values[t+1] + lam*ret)
        output.append(ret)
    return list(reversed(output))


def transform(rewards, deaths, values, gamma, mu, sigma):
    if sigma <= 0.: raise ValueError('scale must be positive')
    return ([(r-mu*(1.-gamma*(1.-d)))/sigma for r, d in zip(rewards, deaths, strict=True)],
            [(v-mu)/sigma for v in values])


def toy_rollout(theta, horizon=8):
    state, rewards, deaths, values = 0., [], [], []
    for _ in range(horizon):
        values.append(2. - (state-.8)**2)
        action = theta - .4*state
        state = .85*state + .25*action
        rewards.append(1. - (state-.8)**2 - .08*action**2)
        deaths.append(1./(1.+math.exp(-5.*(state-.75))))
    values.append(2. - (state-.8)**2)
    return rewards, deaths, values


def scalar(theta, variant, gamma, mu, sigma):
    rewards, deaths, values = toy_rollout(theta)
    if variant == 'raw': return objective(rewards, deaths, values[-1], gamma)
    vnorm = (values[-1]-mu)/sigma
    if variant == 'consistent':
        transformed, _ = transform(rewards, deaths, values, gamma, mu, sigma)
    elif variant == 'terminal_only':
        transformed = [r/sigma for r in rewards]
    else: raise ValueError('unknown variant')
    # Rescaling the normalized objective preserves gradient magnitude as well.
    return sigma*objective(transformed,deaths,vnorm,gamma) + mu


def grad(theta, variant, gamma, mu, sigma, eps=1e-5):
    return (scalar(theta+eps,variant,gamma,mu,sigma)-scalar(theta-eps,variant,gamma,mu,sigma))/(2*eps)


def run(seeds, steps, lr, gamma, mu, sigma):
    max_obj_error = max_grad_error = max_gae_error = 0.
    records = []
    for seed in seeds:
        rng = random.Random(seed); theta0 = rng.uniform(-.4,.4)
        theta = {key:theta0 for key in ('raw','consistent','terminal_only')}
        for step in range(steps):
            for key in theta:
                theta[key] = max(-2.,min(2.,theta[key]+lr*grad(theta[key],key,gamma,mu,sigma)))
        for x in (theta0, *theta.values()):
            rewards, deaths, values = toy_rollout(x)
            rn,vn=transform(rewards,deaths,values,gamma,mu,sigma)
            max_obj_error=max(max_obj_error,abs(scalar(x,'raw',gamma,mu,sigma)-scalar(x,'consistent',gamma,mu,sigma)))
            max_grad_error=max(max_grad_error,abs(grad(x,'raw',gamma,mu,sigma)-grad(x,'consistent',gamma,mu,sigma)))
            raw=gae(rewards,deaths,values,gamma,.95);norm=gae(rn,deaths,vn,gamma,.95)
            max_gae_error=max(max_gae_error,max(abs((a-mu)/sigma-b) for a,b in zip(raw,norm,strict=True)))
        records.append({'seed':seed,'initial_theta':theta0,'final_theta':theta,
            'raw_objective_at_final':{key:scalar(val,'raw',gamma,mu,sigma) for key,val in theta.items()}})
    return {'config':{'seeds':list(seeds),'steps':steps,'lr':lr,'gamma':gamma,'mu':mu,'sigma':sigma,'horizon':8},
            'max_objective_identity_error':max_obj_error,'max_finite_difference_gradient_error':max_grad_error,
            'max_gae_identity_error':max_gae_error,'records':records,
            'notes':'Scalar differentiable soft-survival toy; true and modeled dynamics/hazard coincide. No G1, optimization improvement, or novelty claim. Consistent transformation should reproduce raw trajectory; terminal-only changes objective.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=run(range(12),60,.04,.99,2.,2.5)
    a.output.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(data,indent=2,allow_nan=False))


if __name__=='__main__':main()
