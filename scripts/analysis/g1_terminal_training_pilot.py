"""Small G1 training pilot: current versus legacy shifted terminal labels."""
import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('WARP_CACHE_PATH',r'D:\MinicondaEnv\warp-cache')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import torch
from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.training.loop import Trainer


def install_label_audit(trainer,legacy):
    """Capture actual sampled row indices, optionally restore prior wrong labels."""
    buf=trainer.buffer;pick=buf._sample_from_mask;sample=buf.sample_windows;loss=trainer.wm_trainer.loss_on
    active={};counts={'training_windows':0,'training_labels':0,'correct_positives':0,'legacy_positives':0,'different_labels':0}
    def capture(mask,n,*,check_nonempty=True):
        e,t=pick(mask,n,check_nonempty=check_nonempty);active['indices']=(e,t);return e,t
    def sampled(n,k,*,check_nonempty=True,window_mask=None):
        w=sample(n,k,check_nonempty=check_nonempty,window_mask=window_mask)
        e,t=active['indices'];offs=torch.arange(1,k+1,device=buf.device)[:,None]
        shifted=buf.done[e[None,:].expand(k,-1),(t[None,:]+offs)%buf.T]
        active['current_stats']={'labels':w.done.numel(),'correct':int(w.done.sum().item()),
                                 'legacy':int(shifted.sum().item()),'different':int((w.done!=shifted).sum().item())}
        if legacy:w.done=shifted
        return w
    def measured(w):
        d=active['current_stats'];counts['training_windows']+=w.done.shape[1];counts['training_labels']+=d['labels']
        counts['correct_positives']+=d['correct'];counts['legacy_positives']+=d['legacy'];counts['different_labels']+=d['different']
        return loss(w)
    buf._sample_from_mask=capture;buf.sample_windows=sampled;trainer.wm_trainer.loss_on=measured
    return counts


def digest(*tensors):
    h=hashlib.sha256()
    for tensor in tensors:
        t=tensor.detach().cpu().contiguous();h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()


def evaluate(policy,seed):
    clips=['dance2_subject1','jumps1_subject2','walk2_subject1'];files=discover_npz(Path('data/npz/lafan1'),names=clips)
    if len(files)!=3:raise ValueError('missing evaluation clips')
    args=SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files],num_envs=3,future_frames=10,
        episode_length_s=1e5,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,seed=seed)
    env=make_env(args,device='cpu')
    try:
        cmd=env.command_manager.get_term('motion');lib=cmd.library
        cmd.forced_frame[:]=lib.clip_start[torch.arange(3)];obs,_=env.reset();io=StateIO(env)
        initial_digest=digest(io.read(),obs['policy'],obs['goal'],cmd.frame)
        alive=torch.ones(3,dtype=torch.bool);failed=torch.zeros(3,dtype=torch.bool);timeout=torch.zeros(3,dtype=torch.bool)
        reward=torch.zeros(3);steps=torch.zeros(3,dtype=torch.long)
        with torch.no_grad():
            for t in range(80):
                obs,r,term,tout,_=env.step(policy(obs['policy'],obs['goal']))
                if not torch.isfinite(r[alive]).all():raise ValueError('nonfinite evaluation reward')
                reward+=.99**t*torch.where(alive,r,torch.zeros_like(r));steps+=alive.long()
                failed|=alive&term;timeout|=alive&tout;alive&=~(term|tout)
        return {'initial_state_obs_hash':initial_digest,'seed':seed,'simulator_transitions':240,
                'clips':[{'name':lib.clips[i].name,'reward':float(reward[i]),'steps':int(steps[i]),
                          'failed':bool(failed[i]),'timeout':bool(timeout[i])} for i in range(3)]}
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('fixed','legacy'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run_name=f'research-terminal-pilot-{a.mode}-{a.seed}';root=Path('logs')/run_name
    if root.exists() or a.output.exists():raise FileExistsError('refusing to overwrite experiment')
    torch.set_num_threads(1);cfg=TrainConfig();cfg.run_name=run_name;cfg.seed=a.seed;cfg.device='cpu'
    cfg.data.clips=['walk1_subject1','dance1_subject1','jumps1_subject1'];cfg.env.num_envs=8;cfg.env.replay_capacity=256
    cfg.world_model.hidden=(128,128);cfg.world_model.term_hidden=(16,16)
    cfg.policy.encoder_hidden=(128,64);cfg.policy.decoder_hidden=(128,64);cfg.critic.hidden=(64,32)
    cfg.wm_trainer.batch_size=16;cfg.wm_trainer.batches_per_update=1;cfg.wm_trainer.compile=None
    cfg.policy_trainer.batch_size=16;cfg.policy_trainer.horizon=8;cfg.policy_trainer.critic_epochs=1
    cfg.policy_trainer.critic_minibatches=2;cfg.policy_trainer.compile=None
    cfg.schedule.total_iterations=200;cfg.schedule.warmup_iterations=40;cfg.schedule.updates_per_env_step=.4
    cfg.schedule.log_interval=25;cfg.schedule.checkpoint_interval=1000;cfg.schedule.lr_decay_fraction=5.
    started=time.perf_counter();trainer=Trainer(cfg);counts=install_label_audit(trainer,a.mode=='legacy')
    try:
        trainer.run()
        if trainer.iteration!=200 or trainer.num_updates!=64:raise RuntimeError('unexpected training steps or updates')
        if counts['training_windows']!=64*16 or counts['training_labels']!=64*16*8:raise RuntimeError('label audit count mismatch')
        if not all(torch.isfinite(p).all() for model in (trainer.wm,trainer.policy,trainer.critic) for p in model.parameters()):
            raise RuntimeError('nonfinite parameters')
        actor=trainer.policy.eval();trainer.writer.close();trainer.env.close()
        del trainer;gc.collect()
        assessment=evaluate(actor,seed=9400+a.seed)
        report={'mode':a.mode,'seed':a.seed,'run_name':run_name,'train_env_transitions':1600,'model_updates':64,
                'training_label_audit':counts,'evaluation':assessment,'wall_seconds':time.perf_counter()-started,
                'notes':'Small CPU G1 pilot, random initialization, three training clips and three different fixed evaluation clips. Fixed and shifted labels are sampled on each independently evolving replay. Two variants share initialization and evaluation settings but data distributions may diverge. No same-budget original-sized training or independent-seed performance conclusion.'}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps(report,indent=2))
    finally:
        if 'trainer' in locals():
            if trainer.writer is not None:trainer.writer.close()
            trainer.env.close()


if __name__=='__main__':main()
