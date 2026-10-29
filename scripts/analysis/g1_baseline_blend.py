"""Fit a scalar interpolation of paired score baselines on separate samples."""
import argparse
import json
import math
from pathlib import Path


def variance(moments, weight):
    a=moments['real_variance_trace'];b=moments['zero_variance_trace']
    cov=moments['cross_covariance_trace']
    return (1-weight)**2*a+weight**2*b+2*weight*(1-weight)*cov


def fit_and_validate(fit, validation):
    if min(fit['count'],validation['count'])<2:
        raise ValueError('need independent fit and validation batches')
    a=fit['real_variance_trace'];b=fit['zero_variance_trace'];c=fit['cross_covariance_trace']
    difference=a+b-2*c
    if difference<=0 or validation['real_variance_trace']<=0:
        raise ValueError('degenerate baseline differences')
    weight=max(0.,min(1.,(a-c)/difference))
    return {'fit':fit,'validation':validation,'candidate_weight':weight,
            'validation_mixture_trace':variance(validation,weight),
            'validation_mixture_to_critic_ratio':variance(validation,weight)/validation['real_variance_trace'],
            'notes':'Same finite-horizon score estimator; baseline interpolation weight fit on separate IID tapes. This is not a policy-learning comparison.'}


def audit(data):
    from g1_empirical_baseline_summary import summarize
    checked=summarize(data)
    split=data['baseline_vector_split']
    if not data['empirical_baseline_pilot'].get('loo_critic'):
        raise ValueError('expected leave-one-start-out critic baseline')
    if split['fit']['count']+split['validation']['count']!=data['repeats']:
        raise ValueError('split count mismatch')
    recomputed=fit_and_validate(split['fit'],split['validation'])
    for key in ('candidate_weight','validation_mixture_trace','validation_mixture_to_critic_ratio'):
        if not math.isclose(split[key],recomputed[key],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('mismatched mixture result: '+key)
    return {'seed':data['seed'],'frames':data['frames'],'labels':data['labels'],
            'pilot_trajectories':data['empirical_baseline_pilot']['trajectories'],
            'fit_tapes':split['fit']['count'],'validation_tapes':split['validation']['count'],
            'candidate_weight':split['candidate_weight'],'validation_critic_trace':split['validation']['real_variance_trace'],
            'validation_full_candidate_trace':split['validation']['zero_variance_trace'],
            'validation_mixture_trace':split['validation_mixture_trace'],
            'validation_mixture_to_critic_ratio':split['validation_mixture_to_critic_ratio'],
            'failures':checked['failures'],'budget':checked['budget'],
            'notes':'Preselected fit/validation split of IID tapes for fixed states and frozen critic, with a separately fitted baseline. Bootstrap and finite-sample limitations remain.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(json.loads(args.input.read_text(encoding='utf8')))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2))
