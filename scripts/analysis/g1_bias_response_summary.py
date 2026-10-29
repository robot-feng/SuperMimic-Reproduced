"""Audit and aggregate matched G1 action response errors."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(docs):
    if len(docs)<2 or len({d['seed'] for d in docs})!=len(docs):raise ValueError('missing or duplicate seeds')
    first=docs[0]
    for d in docs:
        if any(d[k]!=first[k] for k in ('fit_seed','checkpoint_sha256','source_models','script_sha256','scale_probe_sha256')):
            raise ValueError('source mismatch')
        if [g['epsilon'] for g in d['groups']]!=[.05,.2]:raise ValueError('amplitude mismatch')
    result={}
    for j,eps in enumerate((.05,.2)):
        rows=[];seed_stats={}
        for d in docs:
            g=d['groups'][j];valid=[x for x in g['records'] if x['valid']]
            if len(valid)!=g['valid_count'] or not valid:raise ValueError('invalid pair counts')
            for mode in ('original','raw','raw_canonical','canonical','augment'):
                for metric in ('response_mse','forward_nse'):
                    vals=[x['models'][mode][metric] for x in valid]
                    if not all(math.isfinite(x) for x in vals):raise ValueError('nonfinite response')
                    if abs(statistics.mean(vals)-g['summary'][mode][metric])>1e-8:raise ValueError('summary mismatch')
            for x in valid:
                if x['initial_nse']>1e-8 or x['context_error']>1e-6 or x['observation_error']>1e-3 or x['frame_mismatch']:
                    raise ValueError('invalid pair accepted')
                rows.append(x)
            seed_stats[str(d['seed'])]={mode:{metric:statistics.mean(x['models'][mode][metric] for x in valid)
               for metric in ('response_mse','forward_nse')} for mode in ('original','raw','raw_canonical','canonical','augment')}
        counts={}
        for name,baseline in (('canonical','raw'),('canonical','raw_canonical'),('augment','raw')):
            key=name+'_vs_'+baseline
            counts[key]={'response_better_count':sum(x['models'][name]['response_mse']<x['models'][baseline]['response_mse'] for x in rows),
                         'forward_better_count':sum(x['models'][name]['forward_nse']<x['models'][baseline]['forward_nse'] for x in rows)}
        result[str(eps)]={'valid_count':len(rows),'seed_means':seed_stats,'comparisons':counts,
                          'means':{mode:{metric:statistics.mean(x['models'][mode][metric] for x in rows)
                            for metric in ('response_mse','forward_nse')} for mode in ('original','raw','raw_canonical','canonical','augment')}}
    return {'seeds':[d['seed'] for d in docs],'fit_seed':first['fit_seed'],'checkpoint_sha256':first['checkpoint_sha256'],
            'source_models':first['source_models'],'simulator_transitions':sum(d['counted_simulator_transitions'] for d in docs),
            'results':result,'notes':'Paired one-step finite action response. Same underlying model fit and same checkpoint across collection seeds; amplitudes share starts and no complete training comparison. Per-start observations correlated; excluded reset pairs.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing to overwrite')
    out=summarize([json.loads(f.read_text(encoding='utf8')) for f in a.inputs]);out['inputs']=[str(x) for x in a.inputs]
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({'seeds':out['seeds'],'simulator_transitions':out['simulator_transitions']}))


if __name__=='__main__':main()
