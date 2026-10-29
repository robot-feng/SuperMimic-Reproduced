"""Audit saved float32 update vectors against streamed gradient moments."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from g1_gradient_layer_analysis_lite import open_metadata


def audit(case):
    vectors=case['saved_updates']
    if set(vectors)!={'model','score'}:raise ValueError('missing saved update vectors')
    arrays={name:np.load(v['path'],mmap_mode='r',allow_pickle=False) for name,v in vectors.items()}
    size=vectors['model']['shape'][0]
    if any(a.shape!=(size,) or a.dtype!=np.dtype('<f4') for a in arrays.values()):raise ValueError('unexpected saved vector layout')
    z,prefix,meta=open_metadata(Path(case['vectors']))
    hashes={key:hashlib.sha256() for key in arrays}
    sums={key:{'norm2':0.,'dot_model':0.,'dot_score':0.} for key in arrays}
    try:
        if meta['seed']!=case['seed'] or meta['checkpoint']!=case['checkpoint']:raise ValueError('source identity mismatch')
        with z.open(prefix+'data/'+meta['mean_model'].storage_key) as gm,z.open(prefix+'data/'+meta['mean_real'].storage_key) as gs:
            for start in range(0,size,65536):
                stop=min(size,start+65536);count=stop-start
                rawm,raws=gm.read(count*4),gs.read(count*4)
                if len(rawm)!=count*4 or len(raws)!=count*4:raise ValueError('truncated gradient')
                m=np.frombuffer(rawm,dtype='<f4').astype(np.float64)
                s=np.frombuffer(raws,dtype='<f4').astype(np.float64)
                for name,arr in arrays.items():
                    chunk=np.asarray(arr[start:stop]);hashes[name].update(chunk.tobytes())
                    x=chunk.astype(np.float64)
                    if not np.isfinite(x).all():raise ValueError('nonfinite update')
                    sums[name]['norm2']+=float(x@x);sums[name]['dot_model']+=float(x@m);sums[name]['dot_score']+=float(x@s)
            if gm.read(1) or gs.read(1):raise ValueError('gradient data too long')
    finally:z.close()
    out={}
    for name in arrays:
        if hashes[name].hexdigest()!=vectors[name]['sha256_raw_float32']:raise ValueError('update hash mismatch')
        values={'norm':math.sqrt(sums[name]['norm2']),'dot_model':sums[name]['dot_model'],'dot_score':sums[name]['dot_score']}
        g=case['global']
        expected={'norm':g['update_'+name+'_norm'],
                  'dot_model':g['model_update_dot_ascent_gradient' if name=='model' else 'score_update_dot_model_ascent_gradient'],
                  'dot_score':g['model_update_dot_score_ascent_gradient' if name=='model' else 'score_update_dot_ascent_gradient']}
        errors={key:abs(values[key]-expected[key]) for key in values}
        if any(errors[k]>max(1e-7,abs(expected[k])*2e-6) for k in errors):raise ValueError('vector moments mismatch')
        out[name]={'path':vectors[name]['path'],'sha256_raw_float32':hashes[name].hexdigest(),
                   'recomputed':values,'absolute_errors':errors}
    return {'seed':case['seed'],'checkpoint':case['checkpoint'],'source':case['vectors'],'updates':out,
            'notes':'Audits saved float32 vector bytes and moments against source gradients and float64 step summary. This does not validate the physical simulator or prove policy improvement.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('metadata',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    obj=json.loads(a.metadata.read_text(encoding='utf8'));out={'cases':[audit(x) for x in obj['cases']]}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
