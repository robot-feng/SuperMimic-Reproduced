"""Reconstruct randomized TD targets from saved step ledgers."""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from g1_tail_policy_switch_probe import sha


def reconstruct(data,gamma,q,horizon=200):
    v=data['initial_value'];cut=data['cutoff'];rows=data['rows'];n=len(v)
    if v.ndim!=1 or cut.shape!=(n,) or not rows or not 0<q<1 or not 0<gamma<1:
        raise ValueError('invalid ledger')
    rr=v.clone();fixed=torch.zeros_like(v);boot=torch.zeros_like(v)
    alive=torch.ones(n,dtype=torch.bool);rfail=torch.zeros(n,dtype=torch.bool);hfail=torch.zeros(n,dtype=torch.bool)
    rs=torch.zeros(n,dtype=torch.long);hs=torch.zeros(n,dtype=torch.long);prior=None
    for t,row in enumerate(rows):
        expected_r=alive & (t<cut);expected_h=alive & (t<horizon)
        if (not torch.equal(row['alive_before'],alive) or not torch.equal(row['roulette_active'],expected_r)
            or not torch.equal(row['fixed_active'],expected_h) or not (expected_r|expected_h).any()):
            raise ValueError('active mask mismatch')
        if t==0 and not torch.equal(row['current'],v):raise ValueError('initial critic mismatch')
        if prior is not None and not torch.equal(row['current'],prior):raise ValueError('value sequence discontinuity')
        reward=row['reward'];done=row['done'];timeout=row['timeout'];next_v=row['next_value']
        if (timeout & (expected_r|expected_h)).any():raise ValueError('timeout encountered')
        surv=~(done|timeout);delta=reward+gamma*torch.where(surv,next_v,torch.zeros_like(next_v))-row['current']
        rr+=torch.where(expected_r,(gamma/q)**t*delta,torch.zeros_like(delta))
        fixed+=torch.where(expected_h,gamma**t*reward,torch.zeros_like(reward))
        rs+=expected_r.long();hs+=expected_h.long();rfail|=expected_r&done;hfail|=expected_h&done
        alive &= surv
        if t==horizon-1:boot=torch.where(alive,gamma**horizon*next_v,torch.zeros_like(next_v))
        prior=next_v
    if (alive & (len(rows)<cut)).any() or (alive & (len(rows)<horizon)).any():raise ValueError('incomplete rollout')
    return {'roulette_target':rr,'fixed_reward':fixed,'fixed_bootstrap':boot,
            'roulette_effective_steps':rs,'fixed_effective_steps':hs,'roulette_failed':rfail,'fixed_failed':hfail}


def summarize(report,data):
    if len(report['records'])!=12 or len(data['initial_value'])!=24:raise ValueError('sample count mismatch')
    n=24;rows=data['rows'];gamma=report['gamma'];q=report['q']
    if report['simulator_steps_executed']!=n*(report['warmup']+len(rows)) or report['vectorized_rollout_iterations']!=len(rows):
        raise ValueError('step count mismatch')
    if report['paired_start_error']>1e-4 or report['obs_reconstruction_error']>1e-3:raise ValueError('unpaired starts')
    checks=report['action_checks']
    if checks['max_clean']>1e-6 or checks['max_exec']>1e-5 or checks['max_action_rate']>1e-5 or checks['max_terminal_reward']>1e-6:
        raise ValueError('action or terminal audit failed')
    reconstructed=reconstruct(data,gamma,q,report['horizon_fixed'])
    if int(reconstructed['roulette_effective_steps'].sum())!=report['roulette_effective_steps'] or int(reconstructed['fixed_effective_steps'].sum())!=report['fixed_effective_steps']:
        raise ValueError('effective step count mismatch')
    pairs=[]
    for i,record in enumerate(report['records']):
        if record['index']!=i or len(record['branches'])!=2:raise ValueError('invalid record')
        for branch,j in zip(record['branches'],(i,i+12)):
            if branch['roulette_length']!=int(data['cutoff'][j]) or abs(branch['initial_value']-float(data['initial_value'][j]))>1e-6:
                raise ValueError('branch initial mismatch')
            for key,arr in reconstructed.items():
                expected=branch[key];actual=arr[j].item()
                if isinstance(expected,bool):
                    if expected!=bool(actual):raise ValueError('termination mismatch')
                elif abs(expected-actual)>1e-6:raise ValueError(f'reconstruction mismatch {key}')
            if abs(branch['fixed_target']-(branch['fixed_reward']+branch['fixed_bootstrap']))>1e-5:raise ValueError('fixed target mismatch')
        x,y=record['branches'];v=x['initial_value']
        if abs(v-y['initial_value'])>1e-5:raise ValueError('different paired values')
        g1=x['roulette_target'];g2=y['roulette_target']
        if not all(math.isfinite(z) for z in (v,g1,g2)):raise ValueError('nonfinite value')
        candidates={'original':v,'minus_half':v-.5,'plus_half':v+.5,'zero':0.}
        pairs.append({'clip':record['clip'],'frame':record['forced_frame'],'value':v,'targets':[g1,g2],
                      'cross_error':{k:(val-g1)*(val-g2) for k,val in candidates.items()},
                      'fixed_residuals':[v-x['fixed_target'],v-y['fixed_target']]})
    aggregate={k:statistics.mean(pair['cross_error'][k] for pair in pairs) for k in pairs[0]['cross_error']}
    return {'seed':report['seed'],'pairs':pairs,'cross_error_mean':aggregate,
            'negative_cross_error_count':{k:sum(pair['cross_error'][k]<0 for pair in pairs) for k in aggregate},
            'roulette_target_mean':statistics.mean(g for pair in pairs for g in pair['targets']),
            'initial_value_mean':statistics.mean(pair['value'] for pair in pairs),
            'fixed_residual_mean':statistics.mean(z for pair in pairs for z in pair['fixed_residuals']),
            'roulette_failures':sum(b['roulette_failed'] for r in report['records'] for b in r['branches']),
            'fixed_failures':sum(b['fixed_failed'] for r in report['records'] for b in r['branches']),
            'simulator_steps_executed':report['simulator_steps_executed'],'roulette_effective_steps':report['roulette_effective_steps'],
            'fixed_effective_steps':report['fixed_effective_steps'],'iterations':len(rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    output=[]
    for path in a.inputs:
        report=json.loads(path.read_text(encoding='utf8'));ledger=path.with_suffix('.pt')
        if sha(ledger)!=report['ledger_sha256'] or any(sha(Path(name))!=digest for name,digest in report['source_sha256'].items()):
            raise ValueError('source/ledger mismatch')
        data=torch.load(ledger,map_location='cpu',weights_only=True)
        r=summarize(report,data);r['report_sha256']=sha(path);r['ledger_sha256']=sha(ledger);output.append(r)
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
