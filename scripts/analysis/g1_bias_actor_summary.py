"""Audit fixed-direction true-rollout comparisons of bias model refits."""
import argparse
import json
import math
import statistics
from pathlib import Path
from g1_tail_policy_switch_probe import sha


def summarize(docs):
    if len(docs)<2 or len({d['seed'] for d in docs})!=len(docs):raise ValueError('missing or duplicate seeds')
    for d in docs:
        if any(d[k]!=docs[0][k] for k in ('fit_seed','checkpoint_sha256','source_models','script_sha256','scale_probe_sha256','horizon','eval_horizon','eps')):
            raise ValueError('protocol mismatch')
        if d['direction_mapping']!={'clean_gradient':'raw_refit_canonical_input','noise_gradient':'canonical_refit_canonical_input'}:
            raise ValueError('direction mapping mismatch')
        if d['gradient_bias_rms']<=0 or d['gradient_input_observation_error']>1e-3 or d['gradient_input_goal_error']>1e-4:
            raise ValueError('invalid gradient inputs')
        if len(d['gradient_eligible_indices'])<2 or d['counted_simulator_transitions']!=1650:raise ValueError('budget mismatch')
        if set(d['gradient_clips'])&set(d['evaluation_clips']):raise ValueError('clip overlap')
        v=d['results'];checks=v['action_checks']
        if v['sigma']!=.1 or checks['calls']!=80 or checks['max_clean_error']!=0 or checks['max_executed_error']>1e-5:
            raise ValueError('invalid action noise audit')
        for name in ('short','long'):
            s=v[name]
            if s['valid_pairs']!=len(s['pairs']) or s['valid_pairs']<1 or s['excluded_timeouts']+s['valid_pairs']!=v['eligible_pairs'] or s['base_control_max_abs_reward_difference']>1e-4 or s['base_control_max_abs_bootstrap_difference']>1e-4 or s['base_control_failure_disagreements']:
                raise ValueError('invalid pairing or timeout accounting')
            for row in s['pairs']:
                i=row['index']
                if not 0<=i<len(d['evaluation_frames']) or row['frame']!=d['evaluation_frames'][i] or row['clip']!=d['evaluation_clips'][i]:raise ValueError('evaluation identity mismatch')
                if row['base_failed']!=row['control_failed'] or abs(row['base_reward']-row['base_control_reward'])>1e-4:raise ValueError('base mismatch')
                for key in ('clean_gradient','noise_gradient'):
                    c=row[key];delta=c['reward_plus']-row['base_reward']
                    if not math.isfinite(delta) or abs(delta-c['reward_delta_vs_base'])>1e-6:raise ValueError('reward mismatch')
    results={}
    for name in ('short','long'):
        rows=[];by_seed={}
        for d in docs:
            cases=[]
            for row in d['results'][name]['pairs']:
                raw=row['clean_gradient'];canon=row['noise_gradient'];delta=canon['reward_plus']-raw['reward_plus']
                v={'seed':d['seed'],'clip':row['clip'],'frame':row['frame'],'canonical_minus_raw_reward':delta,
                   'raw_delta_vs_base':raw['reward_delta_vs_base'],'canonical_delta_vs_base':canon['reward_delta_vs_base'],
                   'base_failed':row['base_failed'],'raw_failed':raw['failed_plus'],'canonical_failed':canon['failed_plus']}
                cases.append(v);rows.append(v)
            by_seed[str(d['seed'])]={'count':len(cases),'mean_reward_difference':statistics.mean(c['canonical_minus_raw_reward'] for c in cases),
                                     'gradient_cosine':d['gradient_cosine']}
        results[name]={'count':len(rows),'mean_reward_difference':statistics.mean(c['canonical_minus_raw_reward'] for c in rows),
          'canonical_better_count':sum(c['canonical_minus_raw_reward']>0 for c in rows),
          'base_failures':sum(c['base_failed'] for c in rows),'raw_failures':sum(c['raw_failed'] for c in rows),
          'canonical_failures':sum(c['canonical_failed'] for c in rows),'by_seed':by_seed,'rows':rows}
    return {'seeds':[d['seed'] for d in docs],'fit_seed':docs[0]['fit_seed'],'checkpoint_sha256':docs[0]['checkpoint_sha256'],
            'counted_simulator_transitions':sum(d['counted_simulator_transitions'] for d in docs),'results':results,
            'notes':'Same pretrained checkpoint and single pair of model fits. Three seeds change starting frames and action noise, not model training. Finite unit-norm direction perturbation; actual 80-step reward truncated; no multiround policy training.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    docs=[json.loads(p.read_text(encoding='utf8')) for p in a.inputs]
    out=summarize(docs);out['inputs']=[str(p) for p in a.inputs]
    checks={'script_sha256':Path('scripts/analysis/g1_bias_actor_direction.py'),
            'scale_probe_sha256':Path('docs/analysis/g1_encoder_bias_symmetry_8600.json'),
            'checkpoint_sha256':Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt')}
    for key,path in checks.items():
        if sha(path)!=docs[0][key]:raise ValueError('source hash mismatch')
    for source in docs[0]['source_models'].values():
        if sha(Path(source['report']))!=source['report_sha256']:raise ValueError('model report hash mismatch')
    out['weights_note']='Weight hashes were checked on the server before and after rollout; weight files remain on server.'
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
