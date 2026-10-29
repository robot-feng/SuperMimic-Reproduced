"""Exact geometric-failure toy for randomized TD-telescope targets."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def exact_value(reward,gamma,p):
    return reward/(1-gamma*(1-p))


def rr_sample(failure,cutoff,reward,gamma,q,critic):
    """Geometric cut/failure lifetimes start at one; supports q==gamma."""
    if q!=gamma:raise ValueError('closed form here requires q == gamma')
    n=np.minimum(failure,cutoff)
    return critic+n*(reward-(1-gamma)*critic)-(failure<=cutoff)*gamma*critic


def fixed_bootstrap(failure,horizon,reward,gamma,critic):
    n=np.minimum(failure,horizon)
    return reward*(1-gamma**n)/(1-gamma)+(failure>horizon)*gamma**horizon*critic


def run(seed,samples=300000,gamma=.99,p=.01,reward=.01,critic=.8,horizon=80):
    rng=np.random.default_rng(seed);q=gamma
    f1=rng.geometric(p,size=samples);f2=rng.geometric(p,size=samples)
    t1=rng.geometric(1-q,size=samples);t2=rng.geometric(1-q,size=samples)
    target1=rr_sample(f1,t1,reward,gamma,q,critic);target2=rr_sample(f2,t2,reward,gamma,q,critic)
    fixed=fixed_bootstrap(f1,horizon,reward,gamma,critic);truth=exact_value(reward,gamma,p)
    out={}
    for name,x in [('roulette',target1),('fixed_bootstrap',fixed)]:
        out[name]={'mean':float(x.mean()),'bias':float(x.mean()-truth),'variance':float(x.var(ddof=1)),
                   'mse':float(np.mean((x-truth)**2)),'mean_se':float(x.std(ddof=1)/math.sqrt(samples))}
    for value in (critic,0.):
        cross=(value-target1)*(value-target2)
        out['cross_error_'+str(value)]={'mean':float(cross.mean()),'truth':float((value-truth)**2),
                                        'se':float(cross.std(ddof=1)/math.sqrt(samples)),'negative_fraction':float((cross<0).mean())}
    return {'seed':seed,'samples':samples,'gamma':gamma,'q':q,'failure_probability':p,'reward':reward,'critic':critic,
            'horizon':horizon,'true_value':truth,'results':out,
            'mean_effective_steps':{'roulette':float(np.minimum(f1,t1).mean()),'fixed':float(np.minimum(f1,horizon).mean())},
            'notes':'Scalar i.i.d. failure toy, exact geometric episode and independent roulette; no dynamics, robot or model training. Fixed and roulette share failure sample; cross error uses a second independent trajectory and roulette.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=300000);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    result=[run(a.seed,a.samples,p=prob,critic=c) for prob in (.001,.03) for c in (.1,.8)]
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
