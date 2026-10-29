"""Audit paired G1 empirical-vs-critic score baseline experiment."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean,stdev


def validate_failure_bootstrap(row,n_envs):
    """A survivor may bootstrap even when another member of the batch fails."""
    steps=row['failure_step'];bootstrap=row['bootstrap']
    if len(steps)!=n_envs or len(bootstrap)!=n_envs or sum(step>0 for step in steps)!=row['failed']:
        raise ValueError('misaligned failure mask')
    if not all(math.isfinite(v) for v in bootstrap):raise ValueError('nonfinite bootstrap')
    if any(step>0 and abs(value)>1e-8 for step,value in zip(steps,bootstrap)):
        raise ValueError('failed trajectory must not bootstrap')
    audits=row.get('terminal_audits',[])
    if sum(item['count'] for item in audits)!=row['failed']:
        raise ValueError('failure reward audit missing or mismatched')
    if any(item['reward_reset_error']>1e-6 or not math.isfinite(item['state_abs_max']) for item in audits):
        raise ValueError('invalid terminal audit')


def summarize(data,fd=None):
    pilot=data.get('empirical_baseline_pilot')
    if not pilot or pilot['seed_offset']==data['noise_seed_offset'] or data.get('antithetic',False):
        raise ValueError('need independent IID pilot and measurement')
    n=data['repeats'];rows=data['results']
    if len(rows)!=n or n<2 or len(pilot['baseline'])!=data['horizon']:
        raise ValueError('invalid sample or horizon')
    names=('real_score','real_score_empirical_baseline','control_variate','control_variate_empirical_baseline')
    result={'seed':data['seed'],'frames':data['frames'],'horizon':data['horizon'],'pilot':pilot,'samples':n,
            'failures':sum(r['failed'] for r in rows),'failure_steps':[r['failure_step'] for r in rows],
            'budget':data['simulator_transition_counts'],'measurements':{}}
    for row in rows:
        if any(max(row[k].values())>1e-4 for k in ('real_replay','model_replay')) or row['action_rate_max_error']>1e-5:
            raise ValueError('trajectory replay mismatch')
        validate_failure_bootstrap(row,len(data['frames']))
        if not math.isclose(row['control_variate_empirical_baseline'],row['real_score_empirical_baseline']+row['model_pathwise']-row['model_score'],abs_tol=1e-6):
            raise ValueError('control variate mismatch')
    reference=stdev([r['real_score'] for r in rows])**2
    if reference<=0:raise ValueError('degenerate reference variance')
    for name in names:
        values=[r[name] for r in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite estimate')
        sd=stdev(values)
        result['measurements'][name]={'mean':mean(values),'std':sd,'se':sd/n**.5,'variance_ratio_vs_critic_score':sd**2/reference,'values':values}
        if not math.isclose(data['summary'][name]['mean'],mean(values),abs_tol=1e-8):raise ValueError('summary mismatch')
    vector=data.get('baseline_vector_moments')
    if vector is not None:
        critic=vector.get('critic_variance_trace',vector.get('real_variance_trace'))
        empirical=vector.get('empirical_variance_trace',vector.get('zero_variance_trace'))
        if vector['count']!=n or critic is None or empirical is None or min(critic,empirical)<=0:
            raise ValueError('invalid vector variance')
        ratio=empirical/critic
        if not math.isclose(ratio,vector['empirical_to_critic_variance_ratio'],rel_tol=1e-7):raise ValueError('vector ratio mismatch')
        result['vector_variance']={'critic_trace':critic,'empirical_trace':empirical,'empirical_to_critic_ratio':ratio,'samples':n}
    if fd is not None:
        if not fd.get('sequential') or data['identity_sha256']!=fd.get('identity_sha256') or any(data[k]!=fd[k] for k in ('seed','frames','labels','horizon','noise_std')):
            raise ValueError('finite difference does not match score start')
        if data['noise_seed_offset']==3600000 or pilot['seed_offset']==3600000:
            raise ValueError('finite difference noise not independent')
        result['finite_difference']={'mean':fd['finite_difference_mean'],'se':fd['finite_difference_se'],'eps':fd['eps'],
                                     'failures':sum(sum(r['failures']) for r in fd['results']),
                                     'base_duplicate_max_error':max(r['base_duplicate_max_error'] for r in fd['results'])}
    result['notes']='Fixed G1 starts; selection and failure rates depend on the input experiment. Scalar projection and, when supplied, full-gradient variance trace use independent baseline pilot and IID measurement noises. Pilot cost is additional. Finite differences have sampling and finite-epsilon error. Neither estimator demonstrates policy learning or training improvement.'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--fd',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.input.read_text(encoding='utf8'));fd=json.loads(a.fd.read_text(encoding='utf8')) if a.fd else None
    out=summarize(data,fd);a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='measurements'},indent=2));print({k:{m:v[m] for m in ('mean','se','variance_ratio_vs_critic_score')} for k,v in out['measurements'].items()})


if __name__=='__main__':main()
