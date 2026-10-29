import unittest
from types import SimpleNamespace

import torch

from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.training.schedule import cosine_lr, training_learning_rates
from supermimic.training.loop import Trainer


class LearningRateTests(unittest.TestCase):
    def test_independent_rates_reach_optimizers_on_schedule(self):
        actor_a=torch.nn.Parameter(torch.zeros(1));actor_b=torch.nn.Parameter(torch.zeros(1))
        critic=torch.nn.Parameter(torch.zeros(1))
        trainer=PolicyTrainer.__new__(PolicyTrainer)
        trainer.opt=torch.optim.Adam([{'params':[actor_a]},{'params':[actor_b]}],lr=.1)
        trainer.critic_opt=torch.optim.Adam([critic],lr=.1)
        for step in (0,20,80,100):
            wm,actor,critic_rate=training_learning_rates(step,100,.002,.003,.0007,.03,.8)
            trainer.set_lr(actor,critic_rate)
            self.assertAlmostEqual(wm,cosine_lr(step,100,.002,.03,.8))
            self.assertAlmostEqual(actor,cosine_lr(step,100,.003,.03,.8))
            self.assertAlmostEqual(critic_rate,cosine_lr(step,100,.0007,.03,.8))
            self.assertTrue(all(g['lr']==actor for g in trainer.opt.param_groups))
            self.assertEqual(trainer.critic_opt.param_groups[0]['lr'],critic_rate)
        trainer.set_lr(.004)
        self.assertTrue(all(g['lr']==.004 for g in trainer.opt.param_groups))
        self.assertEqual(trainer.critic_opt.param_groups[0]['lr'],.004)

    def test_zero_model_or_actor_rate_does_not_disable_critic(self):
        wm,actor,critic=training_learning_rates(50,100,0.,0.,.0007)
        self.assertEqual(wm,0.)
        self.assertEqual(actor,0.)
        self.assertGreater(critic,0.)

    def test_training_loop_passes_and_logs_last_independent_rates(self):
        class Env:
            num_envs=1
            def reset(self):return {},{}
            def step(self,action):
                return {},torch.ones(1),torch.zeros(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool),{}
        class ModelTrainer:
            def __init__(self):self.rates=[]
            def set_lr(self,lr):self.rates.append(lr)
            def update(self,buffer):return {'loss':torch.tensor(1.)}
        class ActorTrainer:
            def __init__(self):self.rates=[]
            def set_lr(self,actor_lr,critic_lr):self.rates.append((actor_lr,critic_lr))
            def update(self,buffer):return {'critic_loss':torch.tensor(2.)}
        class Writer:
            def __init__(self):self.values={}
            def add_scalar(self,k,v,it):self.values[k]=(v,it)
            def flush(self):pass
        schedule=SimpleNamespace(total_iterations=2,updates_per_env_step=1.,policy_updates_per_wm_update=1,
            warmup_iterations=0,lr_final_ratio=.03,lr_decay_fraction=.8,log_interval=2,checkpoint_interval=100,max_hours=None)
        config=SimpleNamespace(schedule=schedule,wm_trainer=SimpleNamespace(lr=.002),
            policy_trainer=SimpleNamespace(lr=.003,critic_lr=.0007,profile_timing=False))
        trainer=Trainer.__new__(Trainer);trainer.cfg=config;trainer.env=Env();trainer.device='cpu'
        trainer.is_main=True;trainer.world=1;trainer.iteration=trainer.start_iteration=trainer.num_updates=0
        trainer._utd_acc=0.;trainer.ep_reward_sum=torch.zeros(1);trainer.ep_len=torch.zeros(1);trainer.max_step_reward=1.
        trainer.wm_trainer=ModelTrainer();trainer.policy_trainer=ActorTrainer();trainer.buffer=SimpleNamespace(size=0)
        trainer.command=SimpleNamespace(metrics={'sampling_entropy':torch.tensor([0.])},bin_return=torch.tensor([1.]))
        trainer.writer=Writer();trainer.act=lambda obs:torch.zeros(1,1);trainer.save=lambda *args:None
        trainer.run()
        wm,actor,critic=training_learning_rates(2,2,.002,.003,.0007,.03,.8)
        self.assertEqual(trainer.wm_trainer.rates[-1],wm)
        self.assertEqual(trainer.policy_trainer.rates[-1],(actor,critic))
        for key,expected in (('lr',wm),('learning_rate/actor',actor),('learning_rate/critic',critic)):
            self.assertEqual(trainer.writer.values[key][1],2)
            self.assertAlmostEqual(trainer.writer.values[key][0],expected,places=10)
        self.assertEqual(trainer.writer.values['policy/critic_loss'],(2.,2))


if __name__=='__main__':unittest.main()
