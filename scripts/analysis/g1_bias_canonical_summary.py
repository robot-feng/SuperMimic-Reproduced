"""Validate source identity and aggregate fixed-checkpoint paired G1 errors."""
import argparse
import json
import math
import statistics
from pathlib import Path

from g1_tail_policy_switch_probe import sha


def summarize(evals,reports):
    if len(evals)!=len(reports) or len(evals)<2:raise ValueError('need paired seed results')
    seeds=set();checkpoints=set();scripts=set();scales=set();summary={}
    for e,r in zip(evals,reports):
        if e['seed']!=r['seed'] or e['window_sha256']!=r['windows_sha256'] or e['checkpoint_sha256']!=r['checkpoint_sha256']:
            raise ValueError('source identity mismatch')
        if e['seed'] in seeds or e['count']!=r['batch_with_replacement']:raise ValueError('duplicate seed or wrong count')
        seeds.add(e['seed']);checkpoints.add(e['checkpoint_sha256']);scripts.add(e['script_sha256']);scales.add(e['scale_probe_sha256'])
        for h in ('1','8','16','32'):
            base=e['results'][h]['0.0']['per_window_nse'];v=e['results'][h]['1.0']['per_window_nse']
            if len(base)!=e['count'] or len(v)!=e['count'] or not all(math.isfinite(x) for x in base+v):raise ValueError('invalid window errors')
            diff=[y-x for x,y in zip(base,v)]
            if abs(statistics.mean(diff)-e['results'][h]['1.0']['paired_nse_delta_mean'])>1e-5:
                raise ValueError('paired delta mismatch')
            if sum(x<0 for x in diff)!=e['results'][h]['1.0']['lower_nse_count']:raise ValueError('count mismatch')
            summary.setdefault(h,[]).append({'seed':e['seed'],'base_mean':statistics.mean(base),
                   'canonical_mean':statistics.mean(v),'delta':statistics.mean(diff),'lower_windows':sum(x<0 for x in diff)})
    if len(checkpoints)!=1 or len(scripts)!=1 or len(scales)!=1:raise ValueError('protocol mismatch')
    for h,rows in summary.items():
        delta=[r['delta'] for r in rows]
        summary[h]={'base_mean':statistics.mean(r['base_mean'] for r in rows),
                    'canonical_mean':statistics.mean(r['canonical_mean'] for r in rows),
                    'mean_delta':statistics.mean(delta),'seed_standard_error':statistics.stdev(delta)/math.sqrt(len(delta)),
                    'better_seed_count':sum(x<0 for x in delta),'seeds':rows}
    return {'seeds':sorted(seeds),'checkpoint_sha256':next(iter(checkpoints)),
            'model_script_sha256':next(iter(scripts)),'scale_probe_sha256':next(iter(scales)),
            'summary':summary,'new_sim_transitions':sum(r['counted_simulator_transitions'] for r in reports),
            'notes':'Same pretrained checkpoint; three correlated window sets sampled with replacement. Seed SE is descriptive, not a generalization confidence bound. Open-loop fixed actions only.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--evals',nargs='+',type=Path,required=True);p.add_argument('--reports',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing to overwrite')
    reports=[json.loads(x.read_text(encoding='utf8')) for x in a.reports]
    for r in reports:
        pth=Path(r['window_path'])
        if sha(pth)!=r['windows_sha256']:raise ValueError('window file hash mismatch')
    out=summarize([json.loads(x.read_text(encoding='utf8')) for x in a.evals],reports)
    out['eval_paths']=[str(x) for x in a.evals];out['report_paths']=[str(x) for x in a.reports]
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
