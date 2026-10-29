"""Read saved sampler metadata without materializing model/optimizer tensors.

Only supports the narrowly whitelisted pickle globals in the local checkpoint.
Forensics/analysis, not a general PyTorch checkpoint loader.
"""
from __future__ import annotations
import argparse
import ast
import collections
import io
import json
import math
import pathlib
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path


class FloatStorage:
    pass


class LongStorage:
    pass


@dataclass
class TensorStub:
    storage_key: str
    storage_size: int
    offset: int
    shape: tuple
    stride: tuple
    dtype: str


def rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks, *rest):
    return TensorStub(storage[0],storage[1],offset,tuple(shape),tuple(stride),storage[2])


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):collections.OrderedDict,
                 ('pathlib','WindowsPath'):pathlib.WindowsPath,
                 ('torch','FloatStorage'):FloatStorage,
                 ('torch','LongStorage'):LongStorage,
                 ('torch._utils','_rebuild_tensor_v2'):rebuild_tensor}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'unsupported global: {module}.{name}')
        return allowed[(module,name)]

    def persistent_load(self,identifier):
        if len(identifier)!=5 or identifier[0]!='storage' or identifier[1] not in (FloatStorage,LongStorage):
            raise pickle.UnpicklingError('unsupported persistent storage')
        return str(identifier[2]),int(identifier[4]),'float32' if identifier[1] is FloatStorage else 'int64'


def load_sampler(path):
    with zipfile.ZipFile(path) as archive:
        pkl=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(pkl)!=1:raise ValueError('unexpected checkpoint archive')
        root=pkl[0][:-len('data.pkl')]
        data=RestrictedUnpickler(io.BytesIO(archive.read(pkl[0]))).load()
        endian=archive.read(root+'byteorder')
        if endian not in (b'little',b'big'):raise ValueError('invalid byte order')
        def read_vector(key):
            tensor=data[key]
            if not isinstance(tensor,TensorStub) or tensor.dtype!='float32' or len(tensor.shape)!=1 or tensor.stride!=(1,) or tensor.offset!=0 or tensor.shape[0]!=tensor.storage_size:
                raise ValueError('sampler tensor must be a contiguous full float storage')
            raw=archive.read(root+'data/'+tensor.storage_key)
            if len(raw)!=4*tensor.storage_size:raise ValueError('storage size mismatch')
            result=struct.unpack(('<' if endian==b'little' else '>')+str(tensor.storage_size)+'f',raw)
            if not all(math.isfinite(x) for x in result):raise ValueError('nonfinite sampler vector')
            return result
        return {'iteration':data['iteration'],'config':data['config'],
                'bin_return':read_vector('bin_return'),'bin_memory':read_vector('bin_memory')}


def npz_frames(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open('joint_pos.npy') as f:
            magic=f.read(8)
            if magic[:6]!=b'\x93NUMPY' or magic[6] not in (1,2,3):raise ValueError('invalid npy header')
            size_len=2 if magic[6]==1 else 4
            size_raw=f.read(size_len)
            if len(size_raw)!=size_len:raise ValueError('truncated npy header')
            size=int.from_bytes(size_raw,'little')
            if size>1_000_000:raise ValueError('oversized npy header')
            raw=f.read(size)
            if len(raw)!=size:raise ValueError('truncated npy header')
            header=ast.literal_eval(raw.decode('latin1' if magic[6]<3 else 'utf8'))
            shape=header['shape']
            if len(shape)!=2 or shape[1]!=29 or shape[0]<3:raise ValueError('unexpected motion shape')
            return shape[0]


def distribution(lengths,returns,memory,horizon,bin_frames=50,temperature=25.,gmax=10.,memory_alpha=.01):
    """Expected frame-bin probabilities for the next return-sampler reset call."""
    bins=[]
    for name,length in lengths:
        for start in range(0,length,bin_frames):
            end=min(start+bin_frames,length);bins.append((name,length,start,end))
    if len(bins)!=len(returns) or len(bins)!=len(memory):raise ValueError('sampler/bin count mismatch')
    if horizon<3 or min(memory)<0 or sum(memory)<=0:raise ValueError('invalid sampler input')
    logits=[-temperature*r/gmax for r in returns];top=max(logits)
    weights=[math.exp(v-top) for v in logits];normalizer=sum(weights);hard=[v/normalizer for v in weights]
    updated=[(1-memory_alpha)*m+memory_alpha*h for m,h in zip(memory,hard,strict=True)]
    mem_total=sum(updated);count=len(bins)
    components={'uniform_bins':[1/count]*count,'hard':hard,'memory':[x/mem_total for x in updated]}
    components['mixture']=[.5*u+.4*h+.1*m for u,h,m in zip(components['uniform_bins'],hard,components['memory'],strict=True)]
    # sample f from bin, clamp to last-2, then reset advances command once.
    # For H>=3, resulting distance to last <= H-2 iff sampled local f >= last-H+1.
    fractions=[max(0,end-max(start,length-1-horizon+1))/(end-start) for _,length,start,end in bins]
    results={key:sum(p*q for p,q in zip(ps,fractions,strict=True)) for key,ps in components.items()}
    concentration={}
    for key,ps in components.items():
        ordered=sorted(range(count),key=ps.__getitem__,reverse=True)
        entropy=-sum(x*math.log(x) for x in ps if x>0)
        concentration[key]={'top_1_mass':ps[ordered[0]],'top_10_mass':sum(ps[i] for i in ordered[:10]),
            'entropy':entropy,'entropy_effective_bins':math.exp(entropy),
            'inverse_simpson_bins':1/sum(x*x for x in ps),
            'top_bins':[{'index':i,'clip':bins[i][0],'clip_frame_start':bins[i][2],
                         'clip_frame_end_exclusive':bins[i][3],'probability':ps[i],
                         'bin_return':returns[i]} for i in ordered[:10]]}
    per_clip=[];names=[name for name,_ in lengths]
    for name in names:
        ids=[i for i,b in enumerate(bins) if b[0]==name]
        mass=sum(components['mixture'][i] for i in ids)
        near=sum(components['mixture'][i]*fractions[i] for i in ids)
        per_clip.append({'clip':name,'selection_probability':mass,'near_boundary_probability':near})
    total_frames=sum(length for _,length in lengths)
    uniform_frames=sum(min(length,horizon) for _,length in lengths)/total_frames
    return {'clips':len(lengths),'frames':total_frames,'bins':count,'horizon':horizon,
            'reset_near_boundary_by_component':results,'reference_uniform_frames':uniform_frames,
            'per_clip':per_clip,'concentration':concentration,
            'notes':'Probability for one reset call from the saved sampler state under fixed bin values and continuous-uniform frame approximation. Not the replay occupancy, and ignores subsequent sampling updates, failure and episode length.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt',type=Path,default=Path('logs/lafan1-rtx5060-eager/ckpt_0002000.pt'))
    p.add_argument('--npz-dir',type=Path,default=Path('data/npz/lafan1'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ck=load_sampler(a.ckpt);config=ck['config']
    if config['data'].get('clips') is not None or config['data'].get('limit') is not None or config['data'].get('manifest') is not None or config['data'].get('library') is not None:
        raise ValueError('this audit requires the full unfiltered npz library')
    if config['env']['sampling_mode']!='return':raise ValueError('checkpoint does not use return sampling')
    files=sorted(a.npz_dir.glob('*.npz'))
    lengths=[(f.stem,npz_frames(f)) for f in files]
    out=distribution(lengths,ck['bin_return'],ck['bin_memory'],int(config['policy_trainer']['horizon']))
    out.update({'checkpoint':str(a.ckpt),'iteration':ck['iteration'],'assumed_sampler_defaults':{'bin_frames':50,
        'uniform_ratio':.5,'hard_ratio':.4,'memory_ratio':.1,'softmax_temp':25.,'g_max':10.,'memory_alpha':.01}})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('per_clip','concentration')},indent=2))


if __name__=='__main__':main()
