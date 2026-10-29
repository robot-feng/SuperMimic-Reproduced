"""Fit fixed global value calibration to independent roulette targets."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def fit(pairs,regularization=.1):
    if len(pairs)<2 or regularization<=0:raise ValueError('invalid fit settings')
    x=[p['value'] for p in pairs];y=[statistics.mean(p['targets']) for p in pairs]
    if not all(math.isfinite(t) for t in x+y):raise ValueError('nonfinite data')
    xm=statistics.mean(x);ym=statistics.mean(y)
    var=statistics.mean((v-xm)**2 for v in x)
    cov=statistics.mean((v-xm)*(t-ym) for v,t in zip(x,y))
    slope=(cov+regularization)/(var+regularization)
    return {'original':{'a':1.,'b':0.},'offset':{'a':1.,'b':ym-xm},
            'affine':{'a':slope,'b':ym-slope*xm},'constant':{'a':0.,'b':ym}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refusing overwrite')
    rows=json.loads(a.audit.read_text(encoding='utf8'));group=[x for x in rows if x['seed'] in (9850,9851,9852)]
    if {x['seed'] for x in group}!={9850,9851,9852} or len(group)!=3:raise ValueError('training seeds mismatch')
    source={}
    for x in group:
        report=Path(f"docs/analysis/g1_roulette_value_ledger_{x['seed']}.json")
        if sha(report)!=x['report_sha256']:raise ValueError('training report hash mismatch')
        if len(x['pairs'])!=12:raise ValueError('missing initial states')
        source[str(report)]=sha(report)
    pairs=[p for x in group for p in x['pairs']]
    result={'training_seeds':[9850,9851,9852],'training_pairs':len(pairs),'regularization':.1,
            'candidates':fit(pairs),'training_frames':[{'clip':p['clip'],'frame':p['frame']} for p in pairs],
            'source_reports':source,'audit_sha256':sha(a.audit),'script_sha256':sha(Path(__file__)),
            'notes':'Fixed candidates fit after inspecting earlier results, using paired unbiased random-truncation targets; no actor update. Evaluate on newly sampled frames.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
