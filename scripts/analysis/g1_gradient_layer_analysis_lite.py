"""Stream stored full G1 gradient vectors into per-layer summaries without torch."""
import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from checkpoint_sampler_lite import RestrictedUnpickler, TensorStub


def open_metadata(path):
    archive=zipfile.ZipFile(path)
    files=[n for n in archive.namelist() if n.endswith('/data.pkl')]
    if len(files)!=1:raise ValueError('unexpected archive format')
    prefix=files[0][:-len('data.pkl')]
    if archive.read(prefix+'byteorder')!=b'little':raise ValueError('unsupported byte order')
    obj=RestrictedUnpickler(io.BytesIO(archive.read(files[0]))).load()
    return archive,prefix,obj


def layers(checkpoint):
    z,_,ck=open_metadata(checkpoint)
    try:
        params=[]
        for name,stub in ck['policy'].items():
            if name.endswith('.scale'):continue
            if not name.endswith(('.weight','.bias')) or not isinstance(stub,TensorStub):
                raise ValueError('unexpected policy state key')
            params.append((name,math.prod(stub.shape)))
        if not params or sum(n for _,n in params)<1:raise ValueError('empty actor')
        return params,ck['iteration']
    finally:z.close()


def summarize(real_sq,model_sq,dot,count):
    rn=math.sqrt(real_sq);mn=math.sqrt(model_sq)
    return {'parameters':count,'real_norm':rn,'model_norm':mn,'cosine':dot/(rn*mn) if rn*mn else None,
            'difference_norm':math.sqrt(max(0.,real_sq+model_sq-2*dot)),
            'real_dot_model':dot}


def chunk_moments(raw_real,raw_model,count):
    if len(raw_real)!=4*count or len(raw_model)!=4*count:raise ValueError('truncated gradient')
    x=np.frombuffer(raw_real,dtype='<f4').astype(np.float64)
    y=np.frombuffer(raw_model,dtype='<f4').astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite gradient')
    return float(x@x),float(y@y),float(x@y),count


def analyze(checkpoint,vectors):
    params,iteration=layers(checkpoint);n=sum(size for _,size in params)
    archive,prefix,data=open_metadata(vectors)
    try:
        if data['checkpoint']!=str(checkpoint):raise ValueError('checkpoint mismatch')
        a,b=data['mean_real'],data['mean_model']
        for t in (a,b):
            if not isinstance(t,TensorStub) or t.dtype!='float32' or t.shape!=(n,) or t.stride!=(1,) or t.offset!=0 or t.storage_size!=n:
                raise ValueError('expected contiguous full gradient storage')
        grouped={};overall=[0.,0.,0.,0]
        with archive.open(prefix+'data/'+a.storage_key) as ra,archive.open(prefix+'data/'+b.storage_key) as rb:
            for name,size in params:
                group=name.rsplit('.',1)[0];acc=grouped.setdefault(group,[0.,0.,0.,0])
                remain=size
                while remain:
                    take=min(remain,65536);ar=ra.read(4*take);br=rb.read(4*take)
                    inc=chunk_moments(ar,br,take)
                    for i,value in enumerate(inc):acc[i]+=value;overall[i]+=value
                    remain-=take
            if ra.read(1) or rb.read(1):raise ValueError('unexpected storage suffix')
        groups={key:summarize(*value) for key,value in grouped.items()}
        sections={}
        for label,prefix_name in [('encoder','encoders.'),('decoder','decoder.')]:
            v=[sum(values[i] for key,values in grouped.items() if key.startswith(prefix_name)) for i in range(4)]
            sections[label]=summarize(*v)
        return {'checkpoint':str(checkpoint),'iteration':iteration,'vectors':str(vectors),'seed':data['seed'],
                'horizon':data['horizon'],'noise_std':data['noise_std'],'pilot_count':data['pilot_count'],
                'validation_count':data['validation_count'],'parameter_order':params,'global':summarize(*overall),
                'sections':sections,'linear_layers':groups,
                'notes':'Stored validation average score gradient versus average model path gradient; both noisy finite-horizon estimates, same selected checkpoint, no inferred layerwise variance or training improvement. State-dict weight/bias order and total length assumed to match policy.parameters; excludes scale buffers; parameterization and layer sizes affect norm.'}
    finally:archive.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('vectors',nargs='+',type=Path)
    p.add_argument('--checkpoint',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'cases':[analyze(a.checkpoint,path) for path in a.vectors]}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
