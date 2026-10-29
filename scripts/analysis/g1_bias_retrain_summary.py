"""Paired summary of model refits across optimization and collection seeds."""
import argparse
import json
import math
import statistics
from pathlib import Path
from g1_tail_policy_switch_probe import sha


def summarize(docs):
    modes=('raw','canonical','augment');lookup={}
    for d in docs:
        k=(d['seed'],d['mode'])
        if k in lookup or d['mode'] not in modes:raise ValueError('duplicate or unknown model')
        lookup[k]=d
    seeds=sorted({s for s,_ in lookup})
    if len(seeds)<2 or set(lookup)!={(s,m) for s in seeds for m in modes}:raise ValueError('incomplete grid')
    first=docs[0];keys=('checkpoint_sha256','scale_probe_sha256','script_sha256','source_files','train_seeds','eval_seeds',
                        'updates','batch_size','horizon','optimizer','lr','training_model_steps')
    if any(any(d[k]!=first[k] for k in keys) for d in docs):raise ValueError('protocol mismatch')
    eval_seeds=first['eval_seeds']
    if len(set(eval_seeds))!=len(eval_seeds) or len(eval_seeds)<2:raise ValueError('invalid evaluation seeds')
    for d in docs:
        for stage in ('before','after'):
            for e in eval_seeds:
                for kind in ('raw','canonical'):
                    for h in ('1','8','16','32'):
                        row=d[stage][str(e)][kind][h];values=row['per_window_nse']
                        if len(values)!=128 or not all(math.isfinite(v) for v in values) or abs(statistics.mean(values)-row['mean'])>1e-5:
                            raise ValueError('evaluation data mismatch')
    comparisons={'canonical_vs_raw':(('canonical','canonical'),('raw','raw')),
                 'augment_vs_raw':(('augment','raw'),('raw','raw')),
                 'canonical_vs_raw_canonical_inference':(('canonical','canonical'),('raw','canonical')),
                 'augment_vs_raw_canonical_inference':(('augment','raw'),('raw','canonical'))}
    summary={}
    for h in ('1','8','16','32'):
        summary[h]={}
        for name,((m1,k1),(m0,k0)) in comparisons.items():
            cells=[]
            for s in seeds:
                for e in eval_seeds:
                    x=lookup[s,m1]['after'][str(e)][k1][h]['mean'];y=lookup[s,m0]['after'][str(e)][k0][h]['mean']
                    if not math.isfinite(x) or not math.isfinite(y):raise ValueError('nonfinite error')
                    cells.append({'optimization_seed':s,'eval_seed':e,'candidate':x,'baseline':y,'difference':x-y})
            summary[h][name]={'candidate_mean':statistics.mean(c['candidate'] for c in cells),
               'baseline_mean':statistics.mean(c['baseline'] for c in cells),
               'mean_delta':statistics.mean(c['difference'] for c in cells),'lower_cost_cells':sum(c['difference']<0 for c in cells),
               'cells':cells,'per_optimization_seed_mean_delta':{str(s):statistics.mean(c['difference'] for c in cells if c['optimization_seed']==s) for s in seeds},
               'per_collection_seed_mean_delta':{str(e):statistics.mean(c['difference'] for c in cells if c['eval_seed']==e) for e in eval_seeds}}
    return {'optimization_seeds':seeds,'collection_seeds':eval_seeds,'training_model_steps_per_arm':first['training_model_steps'],
            'checkpoint_sha256':first['checkpoint_sha256'],'script_sha256':first['script_sha256'],
            'scale_probe_sha256':first['scale_probe_sha256'],'summary':summary,
            'notes':'Crossed optimizer and collection seeds: cells share training data, pretrained checkpoint, and evaluation windows. Nine cells are not nine independent experiments. Equal model steps, not equal complete CPU time. Frozen actor/terminal head; open-loop state prediction only.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing to overwrite')
    docs=[json.loads(f.read_text(encoding='utf8')) for f in a.inputs]
    result=summarize(docs);result['inputs']=[str(x) for x in a.inputs]
    for source in docs[0]['source_files'].values():
        if sha(Path(source['report']))!=source['report_sha256'] or sha(Path(source['windows']))!=source['window_sha256']:
            raise ValueError('source file hash mismatch')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps({'optimization_seeds':result['optimization_seeds'],'collection_seeds':result['collection_seeds']}))


if __name__=='__main__':main()
