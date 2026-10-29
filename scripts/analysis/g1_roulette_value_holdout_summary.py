"""Audit independent RR targets and evaluate frozen affine calibrations."""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from g1_roulette_value_long_audit import summarize as audit_ledger
from g1_roulette_value_fit import fit,sha

ARMS=('original','offset','affine','constant')


def assess(pairs,candidates):
    if set(candidates)!=set(ARMS) or not pairs:raise ValueError('candidate mismatch')
    rows=[]
    for p in pairs:
        v=p['value'];g1,g2=p['targets'];loss={}
        for name in ARMS:
            c=candidates[name];pred=c['a']*v+c['b']
            loss[name]=(pred-g1)*(pred-g2)
            if not math.isfinite(loss[name]):raise ValueError('nonfinite loss')
        rows.append({'clip':p['clip'],'frame':p['frame'],'cross_errors':loss,
                     'differences_vs_original':{name:loss[name]-loss['original'] for name in ARMS[1:]}})
    means={name:statistics.mean(r['cross_errors'][name] for r in rows) for name in ARMS}
    differences={name:{'mean':statistics.mean(r['differences_vs_original'][name] for r in rows),
                       'improved_count':sum(r['differences_vs_original'][name]<0 for r in rows)} for name in ARMS[1:]}
    return {'cross_error_means':means,'differences_vs_original':differences,'rows':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--training-audit',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    m=json.loads(a.fit.read_text(encoding='utf8'));train=json.loads(a.training_audit.read_text(encoding='utf8'))
    if sha(a.training_audit)!=m['audit_sha256'] or sha(Path('scripts/analysis/g1_roulette_value_fit.py'))!=m['script_sha256']:
        raise ValueError('fit source mismatch')
    training=[r for r in train if r['seed'] in (9850,9851,9852)]
    if {r['seed'] for r in training}!={9850,9851,9852} or len(training)!=3:raise ValueError('training seeds mismatch')
    for report,checksum in m['source_reports'].items():
        if sha(Path(report))!=checksum:raise ValueError('training report changed')
    training_pairs=[p for r in training for p in r['pairs']]
    if fit(training_pairs,m['regularization'])!=m['candidates']:raise ValueError('fit mismatch')
    if len(a.inputs)!=3:raise ValueError('expected three holdout seeds')
    seen=set();outputs=[];allpairs=[];trainframes={(x['clip'],x['frame']) for x in m['training_frames']}
    for path in a.inputs:
        d=json.loads(path.read_text(encoding='utf8'));seed=d['seed'];seen.add(seed);ledger=path.with_suffix('.pt')
        if sha(ledger)!=d['ledger_sha256']:raise ValueError('ledger mismatch')
        for name,digest in d['source_sha256'].items():
            if name.endswith('/g1_roulette_value_long_probe.py'):
                local=Path('scripts/analysis/g1_roulette_value_long_probe.py')
            elif name.endswith('/g1_roulette_value_ledger_probe.py'):
                local=Path('scripts/analysis/g1_roulette_value_ledger_probe.py')
            else:
                local=Path(name)
            if sha(local)!=digest:raise ValueError(f'source mismatch: {name}')
        result=audit_ledger(d,torch.load(ledger,map_location='cpu',weights_only=True))
        if any((pair['clip'],pair['frame']) in trainframes for pair in result['pairs']):raise ValueError('training/holdout frame overlap')
        compared=assess(result['pairs'],m['candidates']);allpairs.extend(result['pairs'])
        outputs.append({'seed':seed,'assessment':compared,'roulette_failures':result['roulette_failures'],
            'fixed_failures':result['fixed_failures'],'simulator_steps_executed':result['simulator_steps_executed'],
            'roulette_effective_steps':result['roulette_effective_steps'],'fixed_effective_steps':result['fixed_effective_steps'],
            'report_sha256':sha(path),'ledger_sha256':sha(ledger)})
    if seen!={9860,9861,9862}:raise ValueError('holdout seeds mismatch')
    out={'training_seeds':m['training_seeds'],'candidates':m['candidates'],'fit_sha256':sha(a.fit),
         'results':outputs,'pooled':assess(allpairs,m['candidates']),
         'notes':'Randomized TD estimators and two independent branches per starting state. Cross errors can be negative; 36 states share checkpoint and three clip identities; no actor training.'}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
