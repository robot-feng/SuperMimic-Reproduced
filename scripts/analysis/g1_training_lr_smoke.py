"""Run a tiny CPU G1 training smoke with distinct actor/critic learning rates.

This uses reduced networks and budgets. It checks the production training
pipeline, not learning quality or a comparison with a full-scale model.
"""
import argparse
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
from supermimic.training.loop import Trainer
from supermimic.training.schedule import training_learning_rates
from tfevents_lite import read_scalars
from g1_gradient_layer_analysis_lite import open_metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or not a.run_name.startswith('research-lr-smoke-'):
        p.error('run name must be a simple research-lr-smoke-* name')
    root=Path('logs')/a.run_name
    if root.exists() or a.output.exists():raise FileExistsError('refusing to overwrite existing experiment')
    torch.set_num_threads(1)
    cfg=TrainConfig();cfg.run_name=a.run_name;cfg.seed=4711;cfg.device='cpu';cfg.data.clips=['walk1_subject1']
    cfg.env.num_envs=8;cfg.env.replay_capacity=64
    cfg.world_model.hidden=(64,64);cfg.world_model.term_hidden=(16,16)
    cfg.policy.encoder_hidden=(64,32);cfg.policy.decoder_hidden=(64,32);cfg.critic.hidden=(32,16)
    cfg.wm_trainer.batch_size=8;cfg.wm_trainer.batches_per_update=1;cfg.wm_trainer.compile=None
    cfg.policy_trainer.batch_size=8;cfg.policy_trainer.horizon=4;cfg.policy_trainer.critic_epochs=1
    cfg.policy_trainer.critic_minibatches=2;cfg.policy_trainer.compile=None
    cfg.wm_trainer.lr=.0002;cfg.policy_trainer.lr=.0002;cfg.policy_trainer.critic_lr=.00005
    cfg.schedule.total_iterations=16;cfg.schedule.warmup_iterations=12;cfg.schedule.updates_per_env_step=.5
    cfg.schedule.log_interval=2;cfg.schedule.checkpoint_interval=100
    start=time.perf_counter();trainer=Trainer(cfg);trainer.run()
    expected=training_learning_rates(trainer.iteration,cfg.schedule.total_iterations,cfg.wm_trainer.lr,
        cfg.policy_trainer.lr,cfg.policy_trainer.critic_lr,cfg.schedule.lr_final_ratio,cfg.schedule.lr_decay_fraction)
    actual=(trainer.wm_trainer.opt.param_groups[0]['lr'],trainer.policy_trainer.opt.param_groups[0]['lr'],
            trainer.policy_trainer.critic_opt.param_groups[0]['lr'])
    if trainer.iteration!=16 or trainer.num_updates!=2 or not all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(expected,actual,strict=True)):
        raise RuntimeError('training or optimizer learning-rate check failed')
    if not all(torch.isfinite(p).all() for model in (trainer.wm,trainer.policy,trainer.critic) for p in model.parameters()):
        raise RuntimeError('nonfinite trained parameters')
    checkpoint=root/'ckpt_latest.pt';events=list((root/'tb').glob('events.out.tfevents.*'))
    if not checkpoint.is_file() or len(events)!=1:raise RuntimeError('missing checkpoint or event file')
    tags={'learning_rate/actor','learning_rate/critic','policy/critic_loss','wm/loss'}
    records=read_scalars(events[0],tags)
    by_tag={tag:[r for r in records if r['tag']==tag] for tag in tags}
    if any(not by_tag[tag] or any(not math.isfinite(r['value']) for r in by_tag[tag]) for tag in tags):
        raise RuntimeError('missing or nonfinite TensorBoard learning rate/loss')
    actor_last=by_tag['learning_rate/actor'][-1];critic_last=by_tag['learning_rate/critic'][-1]
    if actor_last['step']!=critic_last['step'] or not math.isclose(critic_last['value'],actor_last['value']*.25,rel_tol=1e-6):
        raise RuntimeError('TensorBoard actor/critic rate mismatch')
    archive,_,metadata=open_metadata(checkpoint)
    try:
        checkpoint_rates=tuple(metadata[k]['param_groups'][0]['lr'] for k in ('wm_opt','policy_opt','critic_opt'))
        if metadata['iteration']!=trainer.iteration or metadata['num_updates']!=trainer.num_updates or not all(
            math.isclose(x,y,rel_tol=1e-12) for x,y in zip(actual,checkpoint_rates,strict=True)
        ):raise RuntimeError('checkpoint training state mismatch')
    finally:archive.close()
    report={'run_name':a.run_name,'device':'cpu','seed':cfg.seed,'clips':cfg.data.clips,'envs':cfg.env.num_envs,
            'iterations':trainer.iteration,'updates':trainer.num_updates,'world_hidden':cfg.world_model.hidden,
            'actor_hidden':cfg.policy.encoder_hidden,'critic_hidden':cfg.critic.hidden,
            'initial_lrs':{'world_model':cfg.wm_trainer.lr,'actor':cfg.policy_trainer.lr,'critic':cfg.policy_trainer.critic_lr},
            'actual_final_lrs':dict(zip(('world_model','actor','critic'),actual,strict=True)),
            'expected_final_lrs':dict(zip(('world_model','actor','critic'),expected,strict=True)),
            'checkpoint_lrs':dict(zip(('world_model','actor','critic'),checkpoint_rates,strict=True)),
            'environment_transitions':cfg.env.num_envs*trainer.iteration,'checkpoint':str(checkpoint),'event':str(events[0]),
            'event_records':records,
            'wall_seconds':time.perf_counter()-start,
            'notes':'Tiny networks, 8 CPU G1 envs, H=4, two optimizer updates, no resume and one clip. Pipeline smoke only; not meaningful G1 performance or algorithm comparison.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
