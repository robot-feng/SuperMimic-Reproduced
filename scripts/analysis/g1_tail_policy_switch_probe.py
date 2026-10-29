"""Six-branch fixed-update G1 counterfactual with identical pre-switch actions."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def available_memory():
    if os.name!='nt':return None
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total_phys',ctypes.c_ulonglong),
                  ('avail_phys',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),
                  ('avail_page',ctypes.c_ulonglong),('total_virt',ctypes.c_ulonglong),
                  ('avail_virt',ctypes.c_ulonglong),('avail_extended',ctypes.c_ulonglong)]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('cannot determine available memory')
    return int(status.avail_phys)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=Path('docs/analysis/g1_adam_update_vectors_seed3500.json'))
    p.add_argument('--npz-dir',type=Path,default=Path('data/npz/lafan1'))
    p.add_argument('--seed',type=int,default=5110)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--replicates',type=int,default=1)
    p.add_argument('--eval-horizon',type=int,default=80)
    p.add_argument('--shared-prefix-noise',action='store_true')
    p.add_argument('--long-bootstrap',action='store_true')
    p.add_argument('--tail-noise-seed',type=int)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.replicates<1 or a.eval_horizon<8 or (a.shared_prefix_noise and a.replicates<2):
        raise ValueError('invalid repetition or horizon')
    if a.tail_noise_seed is not None and (a.replicates!=1 or a.shared_prefix_noise):
        raise ValueError('serial tail seeds require a single replicate')
    if a.output.exists():raise FileExistsError(a.output)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from g1_adam_update_vector_audit import audit
    from g1_gradient_layer_analysis_lite import open_metadata
    cases=json.loads(a.manifest.read_text(encoding='utf8'))['cases']
    if len(cases)!=1:raise ValueError('unexpected manifest')
    case=cases[0]; checked=audit(case)
    if case['seed']!=3500 or case['vector_horizon']!=8:raise ValueError('unexpected source')
    ckpt=Path(case['checkpoint']);ck_hash=sha(ckpt)
    source,_,meta=open_metadata(Path(case['vectors']));source.close()
    if meta['horizon']!=8 or float(meta['noise_std'])!=.1:raise ValueError('unexpected gradient protocol')
    if a.verify_only:
        out={'verified_only':True,'manifest_sha256':sha(a.manifest),'checkpoint_sha256':ck_hash,'audit':checked,
             'available_memory_bytes':available_memory(),'notes':'No G1 simulator was started.'}
        a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2));return
    free=available_memory()
    required_mib=2200 if a.replicates>1 else 1300
    if free is not None and free<required_mib*1024*1024:
        raise RuntimeError(f'insufficient available physical memory: {free} bytes; simulator not started')
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ.setdefault('WARP_CACHE_PATH',r'D:\MinicondaEnv\warp-cache')
    import gc
    import numpy as np
    import torch
    from torch.nn.utils import parameters_to_vector
    from types import SimpleNamespace
    from g1_directional_probe import config_from_checkpoint
    from g1_grounded_gradient_probe import make_setup
    from g1_noisy_imagination_direction import evaluate,frame_samples
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY,ROOT_BODY,TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.policy.nets import Critic,CriticCfg,Policy
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.state_spec import StateSpec
    torch.set_num_threads(1)
    ck=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    if int(ck['iteration'])!=case['iteration'] or ck['config']['env'].get('anchor_obs',False):
        raise ValueError('checkpoint mismatch or unsupported observations')
    pcfg,wcfg=config_from_checkpoint(ck)
    ccfg=CriticCfg(**{k:tuple(v) if isinstance(v,list) else v for k,v in ck['config']['critic'].items()})
    actor,critic=Policy(pcfg),Critic(ccfg,pcfg=pcfg)
    actor.load_state_dict(ck['policy']);critic.load_state_dict(ck['critic'])
    actor.eval();critic.eval().requires_grad_(False)
    base=parameters_to_vector(actor.parameters()).detach().clone()
    values=np.load(case['saved_updates']['model']['path'],allow_pickle=False,mmap_mode='c')
    if values.shape!=tuple(base.shape):raise ValueError('update dimension mismatch')
    direction=torch.from_numpy(values)
    a.clips=['dance1_subject1','jumps1_subject1','walk1_subject1']
    a.eval_clips=['dance2_subject1','jumps1_subject2','walk2_subject1']
    a.frames_per_clip=1;a.eval_frames_per_clip=1;a.eval_replicates=a.replicates;a.warmup=10;a.horizon=8
    a.future=int(ck['config']['env']['future_frames']);a.history=int(ck['config']['policy_trainer']['history'])
    a.gamma=float(ck['config']['policy_trainer']['gamma']);a.eps=1.;a.tail_switch=True
    del ck;gc.collect()
    files=discover_npz(a.npz_dir,names=a.clips+a.eval_clips)
    if len(files)!=6:raise ValueError('reference clips missing')
    _,frames,_,labels=frame_samples(files,a,a.future)
    env=None
    try:
        env,_,bad,_,_,_,_,_,_=make_setup(files,frames[:1],1,a.seed,a.warmup,a.history,a.future,actor,True)
        if bad.any():raise ValueError('head setup warmup failed')
        robot=env.scene['robot']
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cpu'),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,
                            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        wm=SimpleNamespace(spec=StateSpec(wcfg.state_spec,'cpu'),heads=heads)
    finally:
        if env is not None:env.close()
    del env;gc.collect()
    result,initial=evaluate(files,frames,labels,a.seed,actor,critic,wm,[direction,direction],base,a,.1)
    if result['eligible_pairs']<2:raise ValueError('too few valid initial pairs')
    if min(result['short']['valid_pairs'],result['long']['valid_pairs'])<2:raise ValueError('too few valid evaluation pairs')
    if any(result[k]['base_control_failure_disagreements'] or result[k]['base_control_max_abs_reward_difference']>1e-4 for k in ('short','long')):
        raise ValueError('baseline repeat mismatch')
    if sha(ckpt)!=ck_hash:raise ValueError('checkpoint changed')
    hashes={key:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for key,value in initial.items()}
    out={'verified_only':False,'seed':a.seed,'manifest':str(a.manifest),'manifest_sha256':sha(a.manifest),
         'checkpoint':str(ckpt),'checkpoint_sha256':ck_hash,'vector_audit':checked,'initial_hashes':hashes,
         'reference_frames':frames.tolist(),'eval_clips':labels,'noise_std':.1,'warmup':10,'switch_step':8,'horizon':a.eval_horizon,
         'replicates':a.replicates,'shared_prefix_noise':a.shared_prefix_noise,'long_bootstrap':a.long_bootstrap,
         'tail_noise_seed':a.tail_noise_seed,
         'branch_mapping':{'clean_gradient':'full_model_update','noise_gradient':'model_update_then_baseline','base':'baseline'},
         'results':result,'simulator_transition_counts':{'head_setup_warmup':10,'evaluation_warmup':6*len(frames)*a.replicates*a.warmup,
                                                         'evaluation':6*len(frames)*a.replicates*a.eval_horizon},
         'notes':'Finite policy switches at t=8, same initial update and noise, no training. Switch-point matching checks do not verify all internal physics state. Short bootstrap was recorded for audit only.'}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'seed':a.seed,'switch_audit':result['switch_audit'],'counts':out['simulator_transition_counts']},indent=2))


if __name__=='__main__':main()
